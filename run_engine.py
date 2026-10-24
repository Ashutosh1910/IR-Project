import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from irengine import InvertedIndex, search
from irengine.data import index_documents, load_articles, load_qrels, load_queries
from irengine.metrics import evaluate

ROOT = Path(__file__).resolve().parent
DEFAULT_SLOPES = [round(0.1 * i, 1) for i in range(11)]
FROZEN_SLOPE_PATH = ROOT / "frozen_slope.json"


# Slope chosen on Cranfield, once run_cranfield.py has run
def frozen_slope():
    if not FROZEN_SLOPE_PATH.exists():
        return None
    with open(FROZEN_SLOPE_PATH, encoding="utf-8") as f:
        return float(json.load(f)["s"])


def build_index(args) -> tuple[InvertedIndex, dict]:
    articles = load_articles(args.articles)
    index = InvertedIndex(index_documents(articles, include_title=not args.no_title))
    return index, {a["document_id"]: a for a in articles}


def cmd_stats(args):
    index, _ = build_index(args)
    nonzero = [n for n in index.norms if n > 0]
    print(f"documents            {index.N}")
    print(f"vocabulary           {index.vocabulary_size}")
    print(f"tokens/doc (after preprocessing)  mean {statistics.mean(index.doc_lengths):.1f}  "
          f"min {min(index.doc_lengths)}  max {max(index.doc_lengths)}")
    print(f"zero-norm documents  {index.N - index.num_nonzero_docs}")
    print(f"norm n_d             min {min(nonzero):.4f}  max {max(nonzero):.4f}")
    print(f"pivot p              {index.pivot:.6f}  (mean norm over {index.num_nonzero_docs} nonzero-norm docs)")
    zero_idf = sum(1 for v in index.idf.values() if v == 0)
    print(f"terms in every doc   {zero_idf} (IDF = 0)")


def cmd_search(args):
    index, by_id = build_index(args)
    systems = [("cosine", None)] + [(f"pivoted s={s:g}", s) for s in args.s]
    for name, s in systems:
        ranking = search(index, args.query, s=s, k=args.k)
        print(f"\n== {name} ({len(ranking)} shown) ==")
        if not ranking:
            print("  no document shares a weighted term with the query")
        for i, (doc_id, score) in enumerate(ranking, 1):
            a = by_id[doc_id]
            print(f"  {i:>2}. {score:.6f}  [{doc_id}] {a['title'][:70]}  ({a['word_count']} words)")


def _fmt(x):
    return "   -  " if x is None else f"{x:.4f}"


def cmd_evaluate(args):
    index, _ = build_index(args)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels, known_doc_ids=set(index.doc_ids))
    unknown = set(qrels) - set(queries)
    if unknown:
        print(f"warning: qrels mention queries not in the query file: {sorted(unknown)}", file=sys.stderr)

    systems = [("cosine", None)] + [(f"pivoted_s{s:g}", s) for s in args.s]
    summary, rows = [], []
    for name, s in systems:
        rankings = {qid: [d for d, _ in search(index, text, s=s)] for qid, text in queries.items()}
        ev = evaluate(rankings, qrels, k=args.k)
        summary.append({
            "system": name, "s": s, "evaluated_queries": len(ev.per_query),
            f"P@{args.k}": ev.mean_precision, f"R@{args.k}": ev.mean_recall,
            f"F1@{args.k}": ev.mean_f1, "MAP": ev.map,
        })
        for q in ev.per_query:
            rows.append({"system": name, "s": s, **q.__dict__})
        no_rel = ev.no_relevant

    print(f"{'system':<16}{'P@'+str(args.k):>8}{'R@'+str(args.k):>8}{'F1@'+str(args.k):>8}{'MAP':>8}   queries")
    for r in summary:
        print(f"{r['system']:<16}{_fmt(r[f'P@{args.k}']):>8}{_fmt(r[f'R@{args.k}']):>8}"
              f"{_fmt(r[f'F1@{args.k}']):>8}{_fmt(r['MAP']):>8}   {r['evaluated_queries']}")

    # Queries without relevant documents are listed, not averaged
    if no_rel:
        print("\nQueries with no relevant documents (excluded from all averages):")
        for q in no_rel:
            why = "all labels 0" if q.labelled else "no labels found"
            print(f"  {q.query_id}: {why}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["system"])
            w.writeheader()
            w.writerows(rows)
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "pivot": index.pivot,
                "systems": summary,
                "no_relevant_queries": [q.__dict__ for q in no_rel],
            }, f, indent=2)
        print(f"\nwrote {out / 'per_query.csv'} and {out / 'summary.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--articles", default=ROOT / "articles.json", help="articles.json or articles.csv")
    parser.add_argument("--no-title", action="store_true", help="index body text only")
    sub = parser.add_subparsers(dest="command", required=True)

    frozen = frozen_slope()
    sub.add_parser("stats", help="collection and index statistics").set_defaults(func=cmd_stats)

    p = sub.add_parser("search", help="rank articles for one query")
    p.add_argument("query")
    p.add_argument("--s", type=float, nargs="*", default=[0.5 if frozen is None else frozen],
                   help="pivoted slopes to show next to cosine (default: the slope in "
                        "frozen_slope.json, or 0.5 if it does not exist yet)")
    p.add_argument("--k", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("evaluate", help="P@k, R@k, F1@k and MAP for cosine and pivoted")
    p.add_argument("--queries", required=True)
    p.add_argument("--qrels", required=True)
    p.add_argument("--s", type=float, nargs="*",
                   default=DEFAULT_SLOPES if frozen is None else [frozen],
                   help="pivoted slopes to evaluate (default: the slope in frozen_slope.json, "
                        "or 0.0, 0.1, ..., 1.0 if it does not exist yet)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", help="directory for per_query.csv and summary.json")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
