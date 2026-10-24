# Cranfield development: choosing the slope s (steps 19–24)

Run everything with:

```
pip install -r requirements.txt
python run_cranfield.py
```

After the first download the script takes a few seconds. It prints every table on
this page and writes:

- `results/cranfield/sweep.csv`, `per_query.csv`, `cv_folds.csv` and `split.json`
- `frozen_slope.json`

Code: `irengine/cranfield.py`, `irengine/significance.py` and `run_cranfield.py`.
Tests: `tests/test_cranfield.py`.

**Outcome: s = 0.8**, the slope with the highest MAP over all Cranfield queries. The
5-fold cross-validation check **did not confirm a gain**: scored on queries not used
to choose it, pivoted MAP was 0.2811 against cosine's 0.2828. On Cranfield, pivoting
gives no reliable improvement with the current pipeline. Details and the history of
the pipeline change are below.

## Pipeline change (read this first)

The numbers on this page use the current engine: **Porter stemming and the paper's
lnc.ltc weights** (documents 1 + ln tf with no IDF, queries (1 + ln tf) · IDF; see
`docs/engine_design.md`).

The first freeze used an earlier engine: TF · IDF on both documents and queries, and
no stemming. It also chose s = 0.8, and an EPC evaluation was run with it. A later
check against the paper found two differences: the paper puts IDF on query weights
only (section 3), and its SMART indexing stems words. Both were changed to match the
paper, and this script was rerun with `--force`. The change was decided from the
paper and the Cranfield results below, not from EPC results. The EPC relevance labels
do not depend on the engine and are unchanged; EPC must be re-evaluated with the
current engine.

| Pipeline | Cosine MAP | Best s | MAP at best s | Pivoted MAP, 5-fold out-of-fold |
|---|---:|---:|---:|---:|
| TF · IDF both sides, no stemming (first freeze) | 0.2522 | 0.8 | 0.2591 | 0.2562 |
| TF · IDF both sides, Porter | 0.2692 | 0.6 | 0.2757 | 0.2699 |
| lnc.ltc, no stemming | 0.2643 | 0.9 | 0.2695 | 0.2695 |
| **lnc.ltc, Porter (current)** | **0.2828** | **0.8** | **0.2894** | **0.2811** |

MAP over the 215 evaluated queries. The first row comes from this script at commit
a861d6a. The middle two rows come from one-off comparison scripts (not in the repo)
using the same preprocessing, splits and selection rule.

- Each change helps both methods. For cosine, lnc.ltc adds 0.012–0.014 MAP and
  stemming 0.017–0.019.
- Stemming removes pivoting's out-of-fold advantage. Without stemming, pivoted beats
  cosine out of fold by 0.004–0.005 MAP; with stemming the difference is +0.0007
  (TF · IDF) or −0.0017 (lnc.ltc).

## Step 19: loading

`ir_datasets.load("cranfield")` (ir_datasets 0.6.3) gives 1,400 documents, 225 queries
and 1,837 graded judgments. The engine, preprocessing and stopword list are the same
ones used for EPC. Nothing is tuned for Cranfield.

- Each document's `text` field already starts with its title (1,398 of 1,400), so
  only `text` is indexed. Adding the title again would count title words twice.
- Documents 471 and 995 are empty. They stay in the collection (N = 1,400) as
  zero-norm documents, which always score 0 and are left out of the pivot (step 13).
- The index has 4,735 terms (stems) and pivot **p = 10.344049**. This pivot is used
  only on Cranfield. EPC uses its own pivot (33.128197). The slope s carries over to
  EPC; the pivot does not.
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
  these queries is what shows whether the gain is real.
- **The held-out queries are never used to choose s.** Picking the best held-out value
  would turn them into a second dev set, leaving nothing independent to check the
  choice.

## Steps 21–23, single split (the first method tried)

We sorted the 225 query IDs, shuffled them with `random.Random(15)` and split them in
half: **112 dev and 113 held-out queries**. After removing queries with no relevant
document, 107 dev and 108 held-out queries were evaluated. Selection rule: the highest
dev MAP, with ties going to the larger s (closer to plain cosine). Check: the chosen s
must have a higher held-out MAP than cosine.

| s | cosine | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev MAP | 0.2779 | 0.2626 | 0.2710 | 0.2749 | 0.2774 | 0.2826 | 0.2846 | 0.2823 | **0.2855** | 0.2848 | 0.2760 | 0.2779 |
| held-out MAP | 0.2877 | 0.2590 | 0.2676 | 0.2736 | 0.2777 | 0.2860 | 0.2848 | 0.2860 | 0.2920 | 0.2939 | 0.2915 | 0.2877 |

Dev chose **s = 0.7**. On held-out queries it scored **MAP 0.2920 against cosine's
0.2877**: better on 56 queries, worse on 43 and equal on 9 (paired randomization
p = 0.732). The check passes, but the gain is small and well within noise.

With the first-freeze engine, the same split chose s = 0.4, which then *failed* its
held-out check. That failure is why the method was switched to cross-validation.

**Diagnostic** (a one-off analysis, not part of `run_cranfield.py`): we repeated the
split-and-select procedure with 1,000 different seeds.

| Engine | Most-picked slopes (of 1,000 splits) | Chosen s beat cosine on held-out |
|---|---|---:|
| First freeze (TF · IDF both sides, no stemming) | 0.8: 659, 0.7: 292, 0.4: 10 | 862 |
| Current (lnc.ltc, Porter) | 0.8: 480, 0.7: 330, 0.4: 92, 1.0: 54, 0.5: 32 | 666 |

With the current engine the choice is much less stable, and in a third of the splits
the chosen slope does no better than cosine on held-out queries. (When s = 1.0 is
chosen, held-out MAP equals cosine and counts as not beating it.)

## Steps 21–23, final procedure: 5-fold cross-validation

The 225 query IDs are shuffled with seed 15 and dealt into 5 folds of 45 (in
`split.json`). For each fold, s is chosen by MAP on the other four folds (same
selection rule) and then scored on the held-out fold. Every query is therefore scored
with a slope chosen without it.

| Fold | s chosen on the other 4 folds | Test MAP, cosine | Test MAP, pivoted | Test queries |
|---:|---:|---:|---:|---:|
| 1 | 0.4 | 0.2842 | 0.2512 | 44 |
| 2 | 0.7 | 0.2760 | 0.2709 | 42 |
| 3 | 0.8 | 0.3289 | 0.3451 | 41 |
| 4 | 0.8 | 0.2882 | 0.2948 | 44 |
| 5 | 0.8 | 0.2396 | 0.2472 | 44 |

Out-of-fold results over all 215 evaluated queries:

| System | P@10 | R@10 | F1@10 | MAP |
|---|---:|---:|---:|---:|
| cosine | 0.2205 | 0.3871 | 0.2492 | 0.2828 |
| pivoted, s chosen per fold | 0.2205 | 0.3840 | 0.2484 | 0.2811 |

Per-query AP: pivoted is better on 118 queries, worse on 86 and equal on 11. Paired
randomization test (10,000 sign flips, seed 15): p = 0.811.

**Check failed:** out-of-fold MAP is slightly below cosine. The three folds that chose
s = 0.8 gained; fold 1 chose s = 0.4 and lost 0.033 MAP, and fold 2 chose s = 0.7 and
lost 0.005, which together outweigh the gains. The
MAP curve has two nearby peaks (around 0.4–0.5 and 0.7–0.8), so which one wins depends
on the queries, and s = 0.4 hurts on unseen queries.

## Step 24: the frozen slope

The final slope is the one with the highest MAP over all 215 queries, using the same
rule. This rule was fixed before any results were seen and is kept even though the
cross-validation check failed. Switching to s = 1 (plain cosine) because of the
failed check would be another choice made after seeing results.

| s | cosine | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | **0.8** | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAP (215 queries) | 0.2828 | 0.2608 | 0.2693 | 0.2742 | 0.2776 | 0.2843 | 0.2847 | 0.2842 | 0.2888 | **0.2894** | 0.2838 | 0.2828 |

**s = 0.8.** s = 1.0 gives exactly the cosine rankings, as step 14 requires. At other
binarization thresholds the best s is 0.7 (grade ≥ 1, 3 and 4), next to 0.8.

`frozen_slope.json` records s = 0.8, the pipeline, the threshold, the seed, the
cross-validation results (including `"confirmed": false`) and the single-split
result. `run_engine.py search` and `run_engine.py evaluate` use this slope by default
for EPC. `run_cranfield.py` will not overwrite a frozen file that holds a different
slope unless `--force` is given.

## Why the gain is small

Cranfield documents are short abstracts with a narrow range of lengths: 94.9 tokens
on average after preprocessing (the 1,398 nonempty documents), median 84. Relevant
documents are only slightly longer (96.8 tokens on average; mean norm 10.44 against
the pivot 10.34).

Cosine still favours shorter documents: averaged over the 215 evaluated queries, its
top 10 have 84.2 tokens. At s = 0.8 that rises to 94.4, close to the relevant
documents' 96.8, which is the correction the paper describes. s = 0.4 overcorrects to
117.3. So pivoting does fix the length bias in what is retrieved, but with so little
length variation the fix barely changes which relevant documents are found.

EPC articles are about ten times longer and vary more in length, so the slope chosen
here may not be ideal for EPC. The EPC results should say so (step 52).

## What the report should state

- The pipeline was changed after the first freeze to match the paper (lnc.ltc weights,
  stemming), and the slope was refrozen. The first freeze had already been used for
  one EPC evaluation.
- The single-split method was replaced by cross-validation after it failed with the
  first-freeze engine.
- With the current engine, pivoting gives no reliable gain on Cranfield: the
  cross-validation check failed (0.2811 vs 0.2828, p = 0.81), though s = 0.8 moves the
  length of retrieved documents close to that of relevant ones.
