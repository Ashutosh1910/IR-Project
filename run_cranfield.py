import argparse
import csv
import json
import sys
from pathlib import Path

from irengine import InvertedIndex, search
from irengine.cranfield import (
    DATASET,
    DEV_FRACTION,
    FOLDS,
    RELEVANCE_THRESHOLD,
    SLOPES,
    SPLIT_SEED,
    binarize,
    kfold_split,
    load_cranfield,
    select_slope,
    split_queries,
)
from irengine.metrics import evaluate
from irengine.significance import RESAMPLES, paired_randomization_test

ROOT = Path(__file__).resolve().parent
FROZEN_PATH = ROOT / "frozen_slope.json"
K = 10


def system_name(s):
    return "cosine" if s is None else f"pivoted_s{s:g}"


def summarize(ev):
    return {
        "queries": len(ev.per_query),
        f"P@{K}": ev.mean_precision,
        f"R@{K}": ev.mean_recall,
        f"F1@{K}": ev.mean_f1,
        "MAP": ev.map,
    }


def print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'system':<16}{'P@10':>8}{'R@10':>8}{'F1@10':>8}{'MAP':>8}   queries")
    for name, m in rows:
        print(f"  {name:<16}{m['P@10']:>8.4f}{m['R@10']:>8.4f}{m['F1@10']:>8.4f}"
              f"{m['MAP']:>8.4f}   {m['queries']}")


def compare_ap(pivoted, cosine):
    ap_piv = [q.average_precision for q in pivoted.per_query]
    ap_cos = [q.average_precision for q in cosine.per_query]
    assert [q.query_id for q in pivoted.per_query] == [q.query_id for q in cosine.per_query]
    better = sum(p > c for p, c in zip(ap_piv, ap_cos))
    worse = sum(p < c for p, c in zip(ap_piv, ap_cos))
    return better, worse, len(ap_cos) - better - worse, paired_randomization_test(ap_piv, ap_cos)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=ROOT / "results" / "cranfield")
    parser.add_argument("--force", action="store_true",
                        help="overwrite frozen_slope.json even if it holds a different slope")
    args = parser.parse_args(argv)

    documents, queries, graded = load_cranfield()
    # Same engine and preprocessing as EPC
    index = InvertedIndex(documents)
    print(f"Cranfield: {index.N} documents, {len(queries)} queries, "
          f"{sum(len(g) for g in graded.values())} graded judgments")
    print(f"  vocabulary {index.vocabulary_size}, zero-norm documents "
          f"{index.N - index.num_nonzero_docs}, pivot p = {index.pivot:.6f}")

    grade_counts: dict = {}
    for grades in graded.values():
        for g in grades.values():
            grade_counts[g] = grade_counts.get(g, 0) + 1
    print("  grade counts: " + ", ".join(f"{g}: {grade_counts[g]}" for g in sorted(grade_counts)))
    qrels = binarize(graded, RELEVANCE_THRESHOLD)
    no_rel = [q for q, rel in qrels.items() if not rel]
    print(f"  relevant = grade >= {RELEVANCE_THRESHOLD}: "
          f"{sum(len(r) for r in qrels.values())} relevant pairs, "
          f"{len(queries) - len(no_rel)} queries with >= 1 relevant, {len(no_rel)} without")

    systems = [None] + SLOPES
    # Rank every query once per system; splits only choose which to score
    rankings = {
        s: {qid: [d for d, _ in search(index, text, s=s)] for qid, text in queries.items()}
        for s in systems
    }

    def ev(s, qids, rel=qrels):
        return evaluate({q: rankings[s][q] for q in qids}, rel, k=K)

    # First attempt: one dev/held-out split
    dev, held = split_queries(queries, SPLIT_SEED, DEV_FRACTION)
    print(f"\n=== Single split (seed {SPLIT_SEED}), dev {len(dev)} / "
          f"held-out {len(held)} queries ===")
    dev_evals = {s: ev(s, dev) for s in systems}
    held_evals = {s: ev(s, held) for s in systems}
    print_table("Dev sweep (selection by MAP)", [(system_name(s), summarize(dev_evals[s])) for s in systems])
    split_s = select_slope({s: dev_evals[s].map for s in SLOPES})
    split_cmp = compare_ap(held_evals[split_s], held_evals[None])
    split_confirmed = held_evals[split_s].map > held_evals[None].map
    print_table(f"Held-out check (s = {split_s:g} chosen on dev)",
                [("cosine", summarize(held_evals[None])),
                 (system_name(split_s), summarize(held_evals[split_s]))])
    print(f"  AP per query, pivoted vs cosine: {split_cmp[0]} better, {split_cmp[1]} worse, "
          f"{split_cmp[2]} equal; paired randomization p = {split_cmp[3]:.4f}")
    print(f"  confirmed (held-out MAP above cosine): {'yes' if split_confirmed else 'NO'}")
    print_table("Held-out sweep (context only, not used for selection)",
                [(system_name(s), summarize(held_evals[s])) for s in systems])

    # Final procedure: 5-fold cross-validation
    folds = kfold_split(queries, FOLDS, SPLIT_SEED)
    print(f"\n=== Final procedure: {FOLDS}-fold cross-validation (seed {SPLIT_SEED}), "
          f"folds of {', '.join(str(len(f)) for f in folds)} queries ===")
    out_of_fold_s, fold_rows = {}, []
    print(f"  {'fold':<6}{'chosen s':>9}{'train MAP':>11}{'test MAP cos':>14}{'test MAP piv':>14}   test queries")
    for i, fold in enumerate(folds, 1):
        train = [q for other in folds if other is not fold for q in other]
        train_maps = {s: ev(s, train).map for s in SLOPES}
        s_i = select_slope(train_maps)
        test_cos, test_piv = ev(None, fold), ev(s_i, fold)
        out_of_fold_s.update(dict.fromkeys(fold, s_i))
        fold_rows.append({
            "fold": i, "chosen_s": s_i, "train_queries": len(ev(None, train).per_query),
            "train_map_chosen": train_maps[s_i], "train_map_cosine": ev(None, train).map,
            "test_queries": len(test_cos.per_query),
            "test_map_cosine": test_cos.map, "test_map_pivoted": test_piv.map,
        })
        print(f"  {i:<6}{s_i:>9g}{train_maps[s_i]:>11.4f}{test_cos.map:>14.4f}"
              f"{test_piv.map:>14.4f}   {len(test_cos.per_query)}")

    all_ids = list(queries)
    cv_cos = ev(None, all_ids)
    cv_piv = evaluate({q: rankings[out_of_fold_s[q]][q] for q in all_ids}, qrels, k=K)
    cv_cmp = compare_ap(cv_piv, cv_cos)
    cv_confirmed = cv_piv.map > cv_cos.map
    print_table("Out-of-fold result (each query scored with the s chosen without it)",
                [("cosine", summarize(cv_cos)), ("pivoted (CV)", summarize(cv_piv))])
    print(f"  AP per query, pivoted vs cosine: {cv_cmp[0]} better, {cv_cmp[1]} worse, "
          f"{cv_cmp[2]} equal; paired randomization p = {cv_cmp[3]:.4f} ({RESAMPLES} resamples)")
    print(f"  confirmed (out-of-fold MAP above cosine): {'yes' if cv_confirmed else 'NO'}")

    all_evals = {s: ev(s, all_ids) for s in systems}
    # Frozen slope: best MAP over all queries
    chosen = select_slope({s: all_evals[s].map for s in SLOPES})
    print_table("All-query sweep (selects the frozen slope)",
                [(system_name(s), summarize(all_evals[s])) for s in systems])
    print(f"\nFrozen slope: s = {chosen:g} (MAP {all_evals[chosen].map:.4f} "
          f"vs cosine {all_evals[None].map:.4f} over {len(cv_cos.per_query)} queries)")

    sensitivity = {}
    for t in sorted(g for g in grade_counts if g > 0):
        rel_t = binarize(graded, t)
        maps = {s: ev(s, all_ids, rel_t).map for s in SLOPES}
        best = select_slope(maps)
        sensitivity[t] = (best, maps[best], ev(None, all_ids, rel_t).map,
                          len(ev(None, all_ids, rel_t).per_query))
    print("\nThreshold sensitivity over all queries (context only):")
    for t, (best, best_map, cos_map, n) in sensitivity.items():
        print(f"  grade >= {t}: best s = {best:g}  MAP {best_map:.4f} vs cosine {cos_map:.4f}  ({n} queries)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "system", "s", "queries", "P@10", "R@10", "F1@10", "MAP"])
        for split, evals in (("dev", dev_evals), ("heldout", held_evals), ("all", all_evals)):
            for s in systems:
                m = summarize(evals[s])
                w.writerow([split, system_name(s), "" if s is None else s, m["queries"],
                            m["P@10"], m["R@10"], m["F1@10"], m["MAP"]])
    with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "s", "query_id", "num_relevant", "P@10", "R@10", "F1@10", "AP"])
        for s in systems:
            for q in all_evals[s].per_query:
                w.writerow([system_name(s), "" if s is None else s, q.query_id,
                            q.num_relevant, q.precision, q.recall, q.f1, q.average_precision])
    with open(out / "cv_folds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
        w.writeheader()
        w.writerows(fold_rows)
    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SPLIT_SEED, "dev": dev, "heldout": held, "cv_folds": folds}, f, indent=2)

    frozen = {
        "s": chosen,
        "selected_on": f"{DATASET}, all queries with >= 1 relevant document, MAP, "
                       f"s in {SLOPES[0]:g}..{SLOPES[-1]:g} step 0.1",
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "seed": SPLIT_SEED,
        "pipeline": "Porter stemming; lnc.ltc weights (documents 1+ln tf, queries (1+ln tf)*idf)",
        "all_queries_evaluated": len(cv_cos.per_query),
        "all_queries_map": {"cosine": all_evals[None].map, "pivoted": all_evals[chosen].map},
        "cv": {
            "folds": FOLDS,
            "fold_slopes": [r["chosen_s"] for r in fold_rows],
            "out_of_fold_map": {"cosine": cv_cos.map, "pivoted": cv_piv.map},
            "ap_better_worse_equal": list(cv_cmp[:3]),
            "randomization_p": cv_cmp[3],
            "confirmed": cv_confirmed,
        },
        "single_split": {
            "s": split_s,
            "dev_map": {"cosine": dev_evals[None].map, "pivoted": dev_evals[split_s].map},
            "heldout_map": {"cosine": held_evals[None].map, "pivoted": held_evals[split_s].map},
            "heldout_ap_better_worse_equal": list(split_cmp[:3]),
            "heldout_randomization_p": split_cmp[3],
            "heldout_confirmed": split_confirmed,
        },
        "cranfield_pivot": index.pivot,
    }
    # Keep an existing different slope unless --force
    if FROZEN_PATH.exists():
        old = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        if old.get("s") != chosen and not args.force:
            print(f"\n{FROZEN_PATH.name} already freezes s = {old.get('s')}, this run chose "
                  f"s = {chosen:g}; not overwriting (use --force).", file=sys.stderr)
            return 1
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}/sweep.csv, per_query.csv, cv_folds.csv, split.json and {FROZEN_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
