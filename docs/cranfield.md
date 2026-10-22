# Cranfield development: choosing the slope s (steps 19–24)

Run everything with:

```
pip install -r requirements.txt
python run_cranfield.py
```

After the first download the script takes about 2 seconds. It prints every table on
this page and writes:

- `results/cranfield/sweep.csv`, `per_query.csv`, `cv_folds.csv` and `split.json`
- `frozen_slope.json`

Code: `irengine/cranfield.py`, `irengine/significance.py` and `run_cranfield.py`.
Tests: `tests/test_cranfield.py`.

**Outcome: s = 0.8**, chosen by MAP over all Cranfield queries and checked with 5-fold
cross-validation. A first attempt with a single dev/held-out split chose s = 0.4,
which then failed its held-out check. Both are reported below.

## Step 19: loading

`ir_datasets.load("cranfield")` (ir_datasets 0.6.3) gives 1,400 documents, 225 queries
and 1,837 graded judgments. The engine, preprocessing and stopword list are the same
ones used for EPC. Nothing is tuned for Cranfield.

- Each document's `text` field already starts with its title (1,398 of 1,400), so
  only `text` is indexed. Adding the title again would count title words twice.
- Documents 471 and 995 are empty. They stay in the collection (N = 1,400) as
  zero-norm documents, which always score 0 and are left out of the pivot (step 13).
- The index has 7,362 terms and pivot **p = 35.099377**. This pivot is used only on
  Cranfield. EPC uses its own pivot (77.752347). The slope s carries over to EPC; the
  pivot does not.
- Every query ID in the qrels exists, every judged document exists, and no
  (query, document) pair appears twice. The loader checks all three.

## Step 20: relevance grades and binarization

| Grade | Meaning (from `qrels_defs()`) | Judgments |
|---:|---|---:|
| 4 | complete answer to the question | 363 |
| 3 | high degree of relevance | 734 |
| 2 | useful as background, or suggests methods | 387 |
| 1 | minimum interest, e.g. included for historical reasons | 128 |
| −1 | no interest | 225 |

Higher grades mean more relevant. Every query has exactly one −1 judgment. Unjudged
documents count as non-relevant.

**Decision: relevant ⇔ grade ≥ 2.** Our EPC guideline counts a document as relevant
only if it *substantively addresses* the need. Grades 2–4 match that. Grade 1 (kept
"for historical reasons") is closer to merely mentioning the topic, which the EPC
guideline counts as non-relevant.

This gives 1,484 relevant pairs. 215 queries have at least one relevant document. The
other 10 have only grade 1 or −1 judgments. As in step 17, they are reported
separately and left out of every average.

## Why there is a dev part and a held-out part

- **Dev MAP is used to choose s.** Because we take the best of 11 dev values, the
  winner's dev MAP is biased upwards: part of its lead is luck in those particular
  queries. So dev MAP shows which slope looks best. It is not an honest measure of
  how well that slope works.
- **Held-out MAP measures how the chosen s does on queries that played no part in
  choosing it.** This is the unbiased estimate, and the comparison with cosine on
  these queries is what shows whether the gain is real. It also catches a choice
  that looks good only because of noise in the dev queries, which is exactly what
  happened in the first attempt below.
- **The held-out queries are never used to choose s.** Picking the best held-out value
  would turn them into a second dev set, leaving nothing independent to check the
  choice.

## Steps 21–23, first attempt: one dev/held-out split

We sorted the 225 query IDs, shuffled them with `random.Random(15)` and split them in
half: **112 dev and 113 held-out queries**. After removing queries with no relevant
document, 107 dev and 108 held-out queries were evaluated. Selection rule: the highest
dev MAP, with ties going to the larger s (closer to plain cosine). Check: the chosen s
must have a higher held-out MAP than cosine.

| s | cosine | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev MAP | 0.2312 | 0.2211 | 0.2265 | 0.2335 | 0.2358 | **0.2396** | 0.2339 | 0.2346 | 0.2371 | 0.2346 | 0.2327 | 0.2312 |
| held-out MAP | 0.2730 | 0.2438 | 0.2499 | 0.2497 | 0.2491 | 0.2547 | 0.2614 | 0.2716 | 0.2773 | 0.2834 | 0.2790 | 0.2730 |

Dev chose **s = 0.4**. On held-out queries it scored **MAP 0.2547, below cosine's
0.2730**: better on 54 queries, worse on 48 and equal on 6 (paired randomization
p = 0.149). **The check failed.** Most of the loss came from a few queries with a single
relevant document, where one document dropping a few ranks costs a lot of AP. Query
121 lost 0.833 AP, and queries 119 and 146 lost 0.5 each.

**Diagnostic** (a one-off analysis, not part of `run_cranfield.py`): we repeated the
same split-and-select procedure with 1,000 different seeds. It picked s = 0.8 in 659
splits, s = 0.7 in 292 and s = 0.4 in only 10. The chosen s beat cosine on held-out in
862 of the 1,000. So seed 15 produced an unusual dev half. The procedure itself is
sound, but one split is too noisy at this size. That is why the selection was redone
with cross-validation, which uses every query for both choosing and checking, never
both at once.

## Steps 21–23, final procedure: 5-fold cross-validation

The 225 query IDs are shuffled with seed 15 and dealt into 5 folds of 45 (in
`split.json`). For each fold, s is chosen by MAP on the other four folds (same
selection rule) and then scored on the held-out fold. Every query is therefore scored
with a slope chosen without it.

| Fold | s chosen on the other 4 folds | Test MAP, cosine | Test MAP, pivoted | Test queries |
|---:|---:|---:|---:|---:|
| 1 | 0.7 | 0.2604 | 0.2508 | 44 |
| 2 | 0.8 | 0.2521 | 0.2591 | 42 |
| 3 | 0.8 | 0.2896 | 0.2977 | 41 |
| 4 | 0.8 | 0.2487 | 0.2509 | 44 |
| 5 | 0.8 | 0.2126 | 0.2255 | 44 |

Out-of-fold results over all 215 evaluated queries:

| System | P@10 | R@10 | F1@10 | MAP |
|---|---:|---:|---:|---:|
| cosine | 0.1972 | 0.3491 | 0.2228 | 0.2522 |
| pivoted, s chosen per fold | **0.1995** | **0.3551** | **0.2260** | **0.2562** |

Per-query AP: pivoted is better on 115 queries, worse on 83 and equal on 17. Paired
randomization test (10,000 sign flips, seed 15): p = 0.283.

**Check passed:** out-of-fold MAP is above cosine, and four of the five folds chose
s = 0.8. The gain is small (+0.004 MAP) and not statistically significant, so the
report should present it as a slight improvement on Cranfield, not a clear one.

## Step 24: the frozen slope

The final slope is the one with the highest MAP over all 215 queries, using the same
rule. The cross-validation above estimates how well this procedure does on unseen
queries.

| s | cosine | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | **0.8** | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAP (215 queries) | 0.2522 | 0.2325 | 0.2382 | 0.2416 | 0.2425 | 0.2472 | 0.2477 | 0.2532 | 0.2573 | **0.2591** | 0.2560 | 0.2522 |

**s = 0.8.** s = 1.0 gives exactly the cosine rankings, as step 14 requires. The choice
holds at other binarization thresholds too: the best s is 0.8 for grade ≥ 1, 2 and 4,
and 0.7 for grade ≥ 3.

`frozen_slope.json` records s = 0.8, the threshold, the seed, the cross-validation
results and the failed first attempt. `run_engine.py search` and `run_engine.py
evaluate` use this slope by default for EPC. `run_cranfield.py` will not overwrite a
frozen file that holds a different slope unless `--force` is given. It was used once,
to replace the uncommitted s = 0.4 file when the method changed. No EPC evaluation had
been run at any point.

## Why the gain is small

Cranfield documents are short abstracts with a narrow range of lengths: 94.8 tokens
on average after preprocessing, median 84. Relevant documents are only slightly
longer than average (96.8 tokens; mean norm 35.07 against the pivot 35.10). Cosine's
top 10 average 81.6 tokens, so cosine does favour shorter documents. s = 0.8 brings
that to 89.2, while s = 0.4 overcorrects to 109.9. With so little length variation
there is little bias for pivoting to remove. EPC articles are about ten times longer
and vary more in length, so the slope chosen here may not be ideal for EPC. The EPC
results should say so (step 52).

## What the report should state

- The method changed after the first held-out check failed, and why (the
  1,000-split diagnostic).
- Cranfield served only to choose s. EPC remained untouched until s was frozen.
- On Cranfield, the gain from pivoting is small and not statistically significant.
