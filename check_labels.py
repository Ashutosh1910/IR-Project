"""Check the EPC relevance labels and write qrels.csv (step 27, "Checks after the run").

Reads results/llm_labels.jsonl (one record per (query, article) pair) and checks:
  1. every pair is present once, with all four fields and a label of 0 or 1;
  2. every excerpt other than "none" appears word for word in its article's title
     or body text (curly vs. straight quotes and whitespace are normalised on both
     sides);
  3. lists the uncertain pairs for hand review;
  4. reports queries with no relevant article.

Manual review decisions (steps 30-32) are read from manual_labels.csv and override
the LLM label for their pair. Each row must name a labelled pair, carry that pair's
LLM label unchanged, and give a manual label of 0 or 1. The LLM output in
results/llm_labels.jsonl is never modified.

Then writes qrels.csv (query_id,document_id,relevance,source), where source is
"manual" for a hand-reviewed pair and "llm" otherwise, and results/llm_labels.csv,
both with "\n" line endings so a rerun leaves the committed files unchanged.

    python check_labels.py [--labels results/llm_labels.jsonl] [--manual manual_labels.csv] [--qrels qrels.csv]
"""

import argparse
import csv
import json
import re
import sys

QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "′": "'",
                        "“": '"', "”": '"', "„": '"', "″": '"', "⁠": ""})


def norm(s):
    return re.sub(r"\s+", " ", s.translate(QUOTES)).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="results/llm_labels.jsonl")
    p.add_argument("--articles", default="articles.json")
    p.add_argument("--queries", default="queries.json")
    p.add_argument("--manual", default="manual_labels.csv",
                   help="hand-review decisions that override LLM labels (skipped if missing)")
    p.add_argument("--qrels", default="qrels.csv")
    p.add_argument("--csv", default="results/llm_labels.csv")
    args = p.parse_args()

    articles = {str(d["document_id"]): d for d in json.load(open(args.articles, encoding="utf-8"))}
    queries = [q["query_id"] for q in json.load(open(args.queries, encoding="utf-8"))]
    # The labeller sees the title and the body, so an excerpt may come from either.
    texts = {k: norm(f"{d['title']}\n{d['text']}") for k, d in articles.items()}

    records, problems = {}, []
    with open(args.labels, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r.get("query_id"), str(r.get("document_id")))
            if key in records:
                problems.append(f"line {n}: duplicate pair {key}")
            records[key] = r
            if key[0] not in queries or key[1] not in articles:
                problems.append(f"line {n}: unknown query or document {key}")
                continue
            if r.get("label") not in (0, 1):
                problems.append(f"{key}: label must be 0 or 1")
            if not isinstance(r.get("uncertain"), bool):
                problems.append(f"{key}: uncertain must be true or false")
            if not str(r.get("reason", "")).strip():
                problems.append(f"{key}: empty reason")
            ex = r.get("excerpt")
            if not isinstance(ex, str) or not ex.strip():
                problems.append(f"{key}: missing excerpt")
            elif ex != "none":
                if norm(ex) not in texts[key[1]]:
                    problems.append(f"{key}: excerpt not found in article: {ex[:80]!r}")
                if len(ex.split()) > 50:
                    problems.append(f"{key}: excerpt longer than 50 words")
            elif r.get("label") == 1:
                problems.append(f"{key}: relevant label with excerpt 'none'")

    manual = {}
    try:
        with open(args.manual, encoding="utf-8", newline="") as f:
            for n, row in enumerate(csv.DictReader(f), 2):
                key = (row["query_id"], str(row["document_id"]))
                if key in manual:
                    problems.append(f"{args.manual} line {n}: duplicate pair {key}")
                elif key not in records:
                    problems.append(f"{args.manual} line {n}: pair {key} has no LLM label")
                elif row["manual_label"] not in ("0", "1"):
                    problems.append(f"{args.manual} line {n}: manual_label must be 0 or 1")
                elif row["llm_label"] != str(records[key].get("label")):
                    problems.append(f"{args.manual} line {n}: llm_label {row['llm_label']} does not "
                                    f"match the LLM output ({records[key].get('label')})")
                else:
                    manual[key] = int(row["manual_label"])
    except FileNotFoundError:
        pass

    def final(key):
        return manual.get(key, records[key].get("label"))

    missing = [(q, d) for q in queries for d in articles if (q, d) not in records]
    for key in missing[:20]:
        problems.append(f"missing pair {key}")
    if len(missing) > 20:
        problems.append(f"... and {len(missing) - 20} more missing pairs")

    print(f"{len(records)} records, {len(missing)} missing, {len(problems)} problems")
    for msg in problems:
        print("  PROBLEM", msg)

    print("\nRelevant articles per query (after manual review):")
    for q in queries:
        rel = [d for (qq, d) in records if qq == q and final((qq, d)) == 1]
        unc = sum(1 for (qq, d), r in records.items() if qq == q and r.get("uncertain"))
        man = sum(1 for (qq, d) in manual if qq == q)
        flag = "   <-- no relevant article (report separately, step 17)" if not rel else ""
        print(f"  {q:4} relevant={len(rel):3}  uncertain={unc:3}  manual={man:3}{flag}")
    uncertain = [k for k, r in records.items() if r.get("uncertain")]
    changed = [k for k in manual if manual[k] != records[k].get("label")]
    print(f"\nUncertain pairs: {len(uncertain)}, of which hand-reviewed: "
          f"{sum(1 for k in uncertain if k in manual)}")
    print(f"Manual decisions: {len(manual)}, changed from the LLM label: {len(changed)} "
          f"({sum(1 for k in changed if manual[k] == 1)} to relevant, "
          f"{sum(1 for k in changed if manual[k] == 0)} to not relevant)")

    if problems:
        sys.exit("Fix the problems above before writing qrels.")

    order = {q: i for i, q in enumerate(queries)}
    rows = sorted(records.values(), key=lambda r: (order[r["query_id"]], int(r["document_id"])))
    with open(args.qrels, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["query_id", "document_id", "relevance", "source"])
        for r in rows:
            key = (r["query_id"], str(r["document_id"]))
            w.writerow([r["query_id"], r["document_id"], final(key), "manual" if key in manual else "llm"])
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["query_id", "document_id", "title", "label", "uncertain", "reason", "excerpt",
                    "final_label", "label_source"])
        for r in rows:
            key = (r["query_id"], str(r["document_id"]))
            w.writerow([r["query_id"], r["document_id"], articles[str(r["document_id"])]["title"],
                        r["label"], str(r["uncertain"]).lower(), r["reason"], r["excerpt"],
                        final(key), "manual" if key in manual else "llm"])
    print(f"\nWrote {args.qrels} and {args.csv} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
