"""Check the EPC relevance labels and write qrels.csv (step 27, "Checks after the run").

Reads results/llm_labels.jsonl (one record per (query, article) pair) and checks:
  1. every pair is present once, with all four fields and a label of 0 or 1;
  2. every excerpt other than "none" appears word for word in its article
     (curly vs. straight quotes and whitespace are normalised on both sides);
  3. lists the uncertain pairs for hand review;
  4. reports queries with no relevant article.

Then writes qrels.csv (query_id,document_id,relevance) and results/llm_labels.csv.

    python check_labels.py [--labels results/llm_labels.jsonl] [--qrels qrels.csv]
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
    p.add_argument("--qrels", default="qrels.csv")
    p.add_argument("--csv", default="results/llm_labels.csv")
    args = p.parse_args()

    articles = {str(d["document_id"]): d for d in json.load(open(args.articles, encoding="utf-8"))}
    queries = [q["query_id"] for q in json.load(open(args.queries, encoding="utf-8"))]
    texts = {k: norm(d["text"]) for k, d in articles.items()}

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

    missing = [(q, d) for q in queries for d in articles if (q, d) not in records]
    for key in missing[:20]:
        problems.append(f"missing pair {key}")
    if len(missing) > 20:
        problems.append(f"... and {len(missing) - 20} more missing pairs")

    print(f"{len(records)} records, {len(missing)} missing, {len(problems)} problems")
    for msg in problems:
        print("  PROBLEM", msg)

    print("\nRelevant articles per query:")
    for q in queries:
        rel = [d for (qq, d), r in records.items() if qq == q and r.get("label") == 1]
        unc = sum(1 for (qq, d), r in records.items() if qq == q and r.get("uncertain"))
        flag = "   <-- no relevant article (report separately, step 17)" if not rel else ""
        print(f"  {q:4} relevant={len(rel):3}  uncertain={unc:3}{flag}")
    print(f"\nUncertain pairs for hand review: {sum(1 for r in records.values() if r.get('uncertain'))}")

    if problems:
        sys.exit("Fix the problems above before writing qrels.")

    order = {q: i for i, q in enumerate(queries)}
    rows = sorted(records.values(), key=lambda r: (order[r["query_id"]], int(r["document_id"])))
    with open(args.qrels, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "document_id", "relevance"])
        for r in rows:
            w.writerow([r["query_id"], r["document_id"], r["label"]])
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "document_id", "title", "label", "uncertain", "reason", "excerpt"])
        for r in rows:
            w.writerow([r["query_id"], r["document_id"], articles[str(r["document_id"])]["title"],
                        r["label"], str(r["uncertain"]).lower(), r["reason"], r["excerpt"]])
    print(f"\nWrote {args.qrels} and {args.csv} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
