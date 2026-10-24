# Manual review of the EPC labels (steps 30–33)

The LLM labels in `results/llm_labels.jsonl` are never edited. Every hand-review
decision is recorded in `manual_labels.csv`, and `check_labels.py` applies those
decisions on top of the LLM labels when it writes `qrels.csv`. Each row of `qrels.csv`
says where its label came from: `source` is `manual` for a hand-reviewed pair and
`llm` for every other pair.

```
python check_labels.py
```

## `manual_labels.csv`

| Column | Meaning |
|---|---|
| `query_id`, `document_id` | the pair |
| `llm_label` | the LLM's label, copied from the LLM output; the checker rejects a row where it no longer matches |
| `manual_label` | the reviewer's decision: 1 relevant, 0 not relevant |
| `changed` | 1 if the reviewer changed the LLM label |
| `stage` | which review the row comes from: `uncertain` (step 30) for now; positives and the negative sample (steps 30–31) will add their own stages |
| `reviewer`, `reviewed_at` | who decided and when (UTC) |
| `note` | the reviewer's reason, if one was given |

## Stage 1: the uncertain pairs (step 30, part 1)

- **What was reviewed:** all 36 pairs the LLM flagged `uncertain`. Of these, the LLM
  had labelled 9 relevant and 27 not relevant.
- **How:** one reviewer (Ashu) judged each pair against
  `docs/relevance_guideline.md`, in a review page that showed the query, its
  description, the LLM's label, reason and excerpt, and the full article with the
  excerpt highlighted. The reviewer could see the LLM's label, but no retrieval
  scores or rankings. No notes were recorded.
- **When:** 2026-10-24.

**Result: 9 of the 36 labels changed (25%). 27 were kept.**

| Query | Document | LLM | Manual |
|---|---:|---:|---:|
| Q3 | 7784 | 1 | 0 |
| Q3 | 10490 | 1 | 0 |
| Q4 | 6977 | 0 | 1 |
| Q5 | 9930 | 0 | 1 |
| Q6 | 5170 | 0 | 1 |
| Q11 | 7429 | 0 | 1 |
| Q11 | 7434 | 0 | 1 |
| Q11 | 7988 | 0 | 1 |
| Q11 | 8016 | 0 | 1 |

Seven labels went from not relevant to relevant, and two the other way. As a result
the number of relevant pairs rose from 38 to 43. Q11 accounts for four of the changes:
it went from 4 relevant articles to 8. Every query still has at least two relevant
articles, so none is excluded under step 17.

## Freeze (step 34)

The labels were frozen on **2026-10-24**, before any method comparison with the
current engine. The frozen file is **`qrels_final.csv`**: 1,200 pairs and 43 relevant,
of which 36 labels are `manual` and 1,164 are `llm`. `labels_frozen.json` records its
SHA-256 checksum, and `run_epc.py` refuses to run if the file no longer matches.
`qrels.csv` remains the working copy that `check_labels.py` rewrites.

**The review was only partly done when the labels were frozen.** The group decided to
freeze rather than wait. Two parts of the planned review were not done:

1. **The remaining positives (step 30).** The LLM labelled 38 pairs relevant, and 9 of
   those were in the uncertain set above, so 29 positives were never reviewed by hand.
2. **The random sample of negatives (step 31).**

So in the step 33 count, **36 of 1,200 judgments were reviewed and 9 were corrected**.
The other 1,164 labels are unreviewed LLM output. The report must say so, and list it
under limitations (residual LLM errors).

One EPC evaluation had already been run with the earlier engine, before this freeze
and before the review (see `docs/changes.md`). The labels for these 36 pairs were
decided without looking at any retrieval scores or rankings.
