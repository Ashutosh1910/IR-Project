# Changes to the engine after the first slope freeze

This page records one change to the retrieval pipeline made after the slope was first
frozen (commit a861d6a). Details are in `docs/engine_design.md` (steps 10–11),
`docs/validation.md` (step 18) and `docs/cranfield.md` (steps 19–24).

## What changed

1. **Porter stemming.** Every token left after stopword removal is reduced to its
   Porter stem (`elections` → `elect`, `HuEls` → `huel`). The stemmer is written into
   the engine (`irengine/porter.py`) from Porter (1980), using the original rules. On
   all 16,009 distinct words in Cranfield and EPC it matches the Snowball "porter"
   stemmer, apart from one- and two-letter words, which are left unchanged as in
   Porter's own code. Stopwords are removed before stemming.
2. **lnc.ltc weights, as in the paper.** Documents are weighted 1 + ln tf with **no IDF**;
   queries are weighted (1 + ln tf) · IDF. Document norms and the pivot are computed
   from the document weights. Before, both sides used TF · IDF, which counted IDF twice
   in every score.
3. **Slope refrozen.** `run_cranfield.py --force` was rerun with the new engine. The
   slope is again **s = 0.8**. `frozen_slope.json` now also records the pipeline it was
   chosen under, and its single-split entry is renamed `single_split` (it was
   `first_attempt_single_split`).

## Why

A check of the engine against Singhal, Buckley & Mitra (1996) found two differences:

- Section 3 of the paper says "the idf factor is only used in the query term weights
  and not in the document term weights" (SMART lnc.ltc). The proposal had described
  TF · IDF on both sides as an adaptation; it was changed to match the paper.
- The paper's experiments ran on SMART, whose standard indexing stems words. The paper
  itself does not mention stemming. Without it, word forms such as `election` and
  `elections` are unrelated terms, which hurts both methods.

Both changes apply identically to cosine and pivoted, so neither method is favoured.
The decision was based on the paper and on Cranfield, not on EPC results.

## Effect on Cranfield (MAP, 215 queries)

| Engine | Cosine | Best pivoted (s) | Pivoted, 5-fold out-of-fold |
|---|---:|---:|---:|
| Before: TF · IDF both sides, no stemming | 0.2522 | 0.2591 (0.8) | 0.2562 |
| After: lnc.ltc, Porter | 0.2828 | 0.2894 (0.8) | 0.2811 |

- Both methods score higher.
- The 5-fold check now **fails**: on queries not used to choose s, pivoted MAP is
  0.2811 against cosine's 0.2828 (p = 0.81). Stemming is what removes the advantage.
  s = 0.8 is kept because the selection rule was fixed in advance.
- On the single dev/held-out split, the dev half chooses s = 0.7, which beats cosine
  on the held-out half (0.2920 vs 0.2877, p = 0.73).
- s = 0.8 still corrects the length of what is retrieved: cosine's top 10 average 84.2
  tokens, s = 0.8's average 94.4, and relevant documents average 96.8.

Full sweeps are in `docs/cranfield.md`.

## Effect on EPC

- Index: 7,744 terms (was 11,848), norms 24.15–53.87, pivot p = 33.128197 (was
  77.752347). Token counts are unchanged.
- The relevance labels (`qrels.csv`, `results/llm_labels.*`) do not depend on the
  engine and are unchanged.
- One EPC evaluation had been run with the old engine. **EPC results must be
  regenerated** with `python run_engine.py evaluate`, after the human review of the
  labels.

## Tests

- `tests/test_preprocess.py`: normalisation tests now check tokens before stemming;
  new tests cover stemming in the pipeline and Porter's examples for each step.
- `tests/test_toy_example.py` and `docs/validation.md`: the hand-calculated example was
  redone for stems and lnc.ltc weights.
- `tests/test_corpus.py`: the NumPy reference model uses document weights without IDF.
- All 70 tests pass. Deliberately broken copies of the engine (IDF on document
  weights; stemming off) both fail the tests.
