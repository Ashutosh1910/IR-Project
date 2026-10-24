# Main EPC evaluation (steps 35–40)

```
python run_epc.py
```

The script runs in about a second. It prints every table on this page and writes
`results/epc/{summary.json, per_query.csv, top10.csv, length_groups.csv,
article_groups.csv}`.

**Setup**
- **Engine:** Porter stemming, lnc.ltc weights, title + body indexed (see
  `docs/changes.md`). Pivot p = 33.128197.
- **Slope:** **s = 0.8**, read from `frozen_slope.json` (chosen on Cranfield, step 24).
- **Labels:** **`qrels_final.csv`**, frozen 2026-10-24. The script checks its SHA-256
  against `labels_frozen.json` before running. 43 of the 1,200 pairs are relevant;
  36 labels were reviewed by hand and 1,164 are unreviewed LLM labels (see
  `docs/label_review.md`).
- **Queries:** 12. Every query has at least two relevant articles, so none is excluded
  under step 17.

## Steps 35–37: retrieval quality

| System | P@10 | R@10 | F1@10 | MAP |
|---|---:|---:|---:|---:|
| cosine | 0.2667 | 0.8542 | 0.3967 | 0.7109 |
| pivoted, s = 0.8 | 0.2667 | 0.8542 | 0.3967 | **0.7289** |

- **P@10, R@10 and F1@10 are identical.** Both methods put exactly the same relevant
  articles in the top 10 for every query.
- **MAP is 0.018 higher with pivoting.** It is better on 4 queries, worse on 2 and
  equal on 6. Paired randomization test (10,000 sign flips): p = 0.132 for AP and
  p = 1.0 for P@10. With 12 queries this difference is not statistically significant.
- **The two methods mostly agree.** 7 of the 12 queries keep exactly the same set of
  articles in the top 10, and the other 5 swap one article each. The MAP difference
  comes from reordering inside the top 10 (step 40).

| Query | Relevant | P@10 | R@10 | AP cosine | AP pivoted | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
| Q1 placements | 2 | 0.20 | 1.00 | 0.5833 | 0.5833 | 0 |
| Q2 HuEl | 4 | 0.40 | 1.00 | 1.0000 | 1.0000 | 0 |
| Q3 fest budget | 4 | 0.40 | 1.00 | 0.5429 | 0.6361 | +0.0933 |
| Q4 online semester | 6 | 0.50 | 0.83 | 0.5413 | 0.5956 | +0.0543 |
| Q5 students union election process | 3 | 0.30 | 1.00 | 0.8095 | 0.8333 | +0.0238 |
| Q6 mental health counselling support | 3 | 0.20 | 0.67 | 0.7436 | 0.7436 | 0 |
| Q7 packing list for freshers | 4 | 0.30 | 0.75 | 0.6756 | 0.6708 | −0.0048 |
| Q8 dual degree minor options | 2 | 0.20 | 1.00 | 0.7000 | 0.7500 | +0.0500 |
| Q9 sports secretary promises | 2 | 0.20 | 1.00 | 1.0000 | 1.0000 | 0 |
| Q10 mess food quality | 3 | 0.30 | 1.00 | 0.7667 | 0.7667 | 0 |
| Q11 SU manifesto promises | 8 | 0.00 | 0.00 | 0.1679 | 0.1670 | −0.0009 |
| Q12 student protests | 2 | 0.20 | 1.00 | 1.0000 | 1.0000 | 0 |

P@10 and R@10 are the same for both methods on every query, so each is shown once.

- **P@10 is capped by the number of relevant articles.** A query with 2 relevant
  articles can reach at most 0.2. R@10 is the better guide here: it is 1.0 for 9 of
  the 12 queries.
- **Q2 "HuEl" retrieves only 5 articles,** because only 5 contain the word. All 4
  relevant ones rank first under both methods.
- **Q11 fails for both methods.** None of its 8 relevant articles reaches the top 10
  (the best is rank 13). The query asks for an evaluation of promises kept, but
  the relevant articles are report cards, interviews and statements that use other
  words. Four of the eight became relevant in the hand review (`docs/label_review.md`).

## Step 38: length groups

Groups are tertiles of body word count, the measure in the article manifest (before
stopword removal, title excluded). Articles are sorted by (word count, document ID)
and cut at ranks 33 and 67.

| Group | Articles | Words |
|---|---:|---|
| short | 33 | 1,014–1,293 |
| medium | 34 | 1,297–1,801 |
| long | 33 | 1,893–4,343 |

Every article has at least 1,000 words, so "short" here means short for this
collection.

## Step 39: lengths of retrieved and relevant articles

| Group | Relevant pairs | Cosine top-10 slots | Pivoted top-10 slots | Relevant found in top 10 (both methods) |
|---|---:|---:|---:|---:|
| short | 10 (23%) | 26 (23%) | 22 (19%) | 5 of 10 |
| medium | 14 (33%) | 43 (37%) | 43 (37%) | 11 of 14 |
| long | 19 (44%) | 46 (40%) | 50 (43%) | 16 of 19 |
| total | 43 | 115 | 115 | 32 of 43 |

There are 115 top-10 slots, not 120, because Q2 retrieves only 5 articles. Mean words:
collection 1,709, relevant articles 1,930, cosine top 10 1,805, pivoted top 10 1,841.

- **Relevant articles skew long.** Long articles make up a third of the collection
  but 44% of relevant pairs.
- **Pivoting moves the top 10 towards that.** Cosine gives long articles 40% of its
  top-10 slots; pivoting raises that to 43%, closer to the 44% share of relevant
  articles. The extra four slots come from short articles, which fall from 23% to
  19%. This is the shift Singhal et al. describe.
- **The shift never changes which relevant articles are found.** In every group, both
  methods find exactly the same relevant articles in their top 10. Short relevant
  articles are found least often (5 of 10) under both methods, so this is not
  something pivoting caused.

## Step 40: where the methods disagree most

Queries are ranked by |ΔAP|, then by how many articles were swapped into or out of the
top 10, then by how many were reordered within it.

| Query | ΔAP | Top-10 swaps | Reordered in top 10 |
|---|---:|---:|---:|
| Q3 | +0.0933 | 0 | 4 |
| Q4 | +0.0543 | 0 | 6 |
| Q8 | +0.0500 | 0 | 7 |
| Q5 | +0.0238 | 1 | 4 |
| Q7 | −0.0048 | 0 | 5 |
| Q11 | −0.0009 | 1 | 5 |
| Q9, Q1, Q12 | 0 | 1 each | 4, 2, 0 |
| Q6, Q10, Q2 | 0 | 0 | 3, 0, 0 |

**The three largest AP changes all come from reordering inside the top 10:**

- **Q3 "fest budget":** a short non-relevant article (10490, 1,070 words) ranked 1st
  under cosine. Pivoting drops it to 2nd and lifts the relevant 9377 (medium) to 1st.
  The relevant 8008 (medium) falls from 7th to 9th. AP 0.543 → 0.636.
- **Q4 "online semester":** three long relevant articles move up one place each:
  5307 from 6th to 5th, 7251 from 7th to 6th, and 5305 from 10th to 8th. They pass a
  medium non-relevant article and two short ones. AP 0.541 → 0.596.
- **Q8 "dual degree minor options":** the long relevant 7763 moves from 5th to 4th,
  passing a medium non-relevant article. AP 0.700 → 0.750.

**Every top-10 swap brings in a longer article and removes a shorter one:**

| Query | Entered (words, group) | Left (words, group) | Relevant? |
|---|---|---|---|
| Q1 | 7557 (2,454, long) | 10685 (1,571, medium) | neither |
| Q5 | 8341 (1,904, long) | 8016 (1,181, short) | neither |
| Q9 | 9930 (1,722, medium) | 9136 (1,014, short) | neither |
| Q11 | 8341 (1,904, long) | 7282 (1,124, short) | neither |
| Q12 | 9179 (2,198, long) | 7452 (1,108, short) | neither |

In all five swaps the article entering and the article leaving are both non-relevant,
at ranks 10–13. That is why P@10 and R@10 do not change at all.

**The two small losses:**
- **Q7:** the short relevant 9136 (1,014 words, the shortest article in the
  collection) slips from 14th to 15th.
- **Q11:** short relevant articles slip one or two places, outside the top 10.

Both are the cost Singhal et al. predict for pivoting: it penalises short documents.
