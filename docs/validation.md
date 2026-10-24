# Step 18: hand-calculated validation example

Every number below was worked out by hand from the formulas in `docs/engine_design.md`
(lnc.ltc weights: documents 1 + ln tf with no IDF, queries (1 + ln tf) · IDF).
`tests/test_toy_example.py` compares the engine to these closed forms, and also to the
6-decimal values printed here, so this page and the code cannot drift apart.

Notation: L = ln 2 = 0.693147, so ln 4 = 2L, and 1 + ln 2 = 1 + L = 1.693147.

## Collection (N = 4)

| Doc | Text | Terms after preprocessing (Porter stems) |
|---|---|---|
| 1 | "Apple apple banana." | appl ×2, banana ×1 |
| 2 | "Apple, cherry!" | appl, cherri |
| 3 | "Banana cherry date" | banana, cherri, date |
| 4 | "The and of it." | *(all stopwords → zero vector)* |

Document 4 still counts toward N = 4.

## Weights

| Term | df | IDF = ln(4/df) (queries only) |
|---|---|---|
| appl, banana, cherri | 2 | L = 0.693147 |
| date | 1 | 2L = 1.386294 |

TF: count 1 → 1, count 2 → 1 + L. Document weights are TF alone, so the rare term
`date` gets the same document weight (1) as the common ones.

| Doc | Weights | Norm n_d |
|---|---|---|
| 1 | appl 1+L = 1.693147, banana 1 | √((1+L)² + 1) = **1.966405** |
| 2 | appl 1, cherri 1 | √2 = **1.414214** |
| 3 | banana 1, cherri 1, date 1 | √3 = **1.732051** |
| 4 | none | **0** |

**Pivot** (nonzero-norm documents only): p = (1.966405 + 1.414214 + 1.732051) / 3 =
**1.704223**. Wrongly including document 4 would give 1.278167.

## Query q1 = "apple banana"

w_q = (appl L, banana L), n_q = L√2 = 0.980258.

Dot products: d1 = L·(1+L) + L·1 = L(2+L) = 1.866747; d2 = L = 0.693147;
d3 = L = 0.693147; d4 = 0.

| Doc | Cosine | s = 0.5: factor → score | s = 0: factor → score |
|---|---|---|---|
| 1 | L(2+L) / (L√2·1.966405) = **0.968439** | 1.835314 → **1.037611** | p → **1.117426** |
| 2 | L / (L√2·√2) = **0.5** | 1.559218 → **0.453501** | p → **0.414914** |
| 3 | L / (L√2·√3) = 1/√6 = **0.408248** | 1.718137 → **0.411554** | p → **0.414914** |
| 4 | 0 (n_d = 0) | 0 | 0 |

Here factor = (1 − s)p + s·n_d, and score = dot / (n_q · factor). Pivoted scores are
not bounded by 1.

Rankings: cosine 1, 2, 3. s = 0.5: 1, 2, 3. s = 0: 1, then **2 and 3 tie**, so the
lower ID wins: 1, 2, 3. Document 4 is never retrieved.

## Query q2 = "banana cherry"

n_q = L√2. Dot products: d1 = L, d2 = L, d3 = 2L = 1.386294.

| Doc | Cosine | s = 0.5 | s = 0 |
|---|---|---|---|
| 1 (n₁ > p) | **0.359594** | **0.385278** | **0.414914** |
| 2 (n₂ < p) | **0.5** | **0.453501** | **0.414914** |
| 3 | **0.816497** | **0.823109** | **0.829829** |

As s falls, document 1 (norm above the pivot) **gains** and document 2 (norm below
the pivot) **loses**, which is how the pivot is meant to work. Rankings: cosine 3, 2, 1.
s = 0.5: 3, 2, 1. s = 0: 3, then **1 and 2 tie**, so the lower ID wins: **3, 1, 2**. At
s = 0 every nonzero document has the same factor p, so scores are just
dot / (n_q·p), proportional to the dot product.

## Special cases

- **s = 1 equals cosine**: the factor is 0·p + 1·n_d = n_d, so the scores are identical
  (the test checks exact equality on the toy collection and on all 100 EPC articles).
- **s = 0 gives equal factors**: every nonzero document has the factor p = 1.704223.
- **Zero-vector document**: document 4 has n_d = 0. It scores 0 under every s and every
  query, is never retrieved, and is left out of p.
- **Zero-vector query**: "the zebra" has a stopword and a term not in the vocabulary,
  so the query vector is empty and n_q = 0. Every score is 0 and nothing is retrieved.
- **Unseen terms ignored**: "apple banana zebra quokka" gives exactly the q1 scores.
- **Term in every document**: in a collection "common alpha", "common beta", "common",
  `common` has IDF 0, so its query weight is 0 and "common" retrieves nothing. It still
  counts toward the document norms (√2, √2, 1), because documents carry no IDF.

## Metrics

Relevance labels: q1 → {1, 3}; q2 → {2}; q3 = "the zebra" → {1}; q4 = "apple" →
labelled, none relevant; q5 = "cherry" → no labels.

**Cosine**

| Query | Ranking | P@10 | R@10 | F1@10 | AP |
|---|---|---|---|---|---|
| q1 | 1, 2, 3 | 2/10 = 0.2 | 2/2 = 1 | 2·0.2·1/1.2 = 0.333333 | (1/1 + 2/3)/2 = 0.833333 |
| q2 | 3, 2, 1 | 0.1 | 1 | 0.2/1.1 = 0.181818 | (1/2)/1 = 0.5 |
| q3 | (empty) | 0 | 0 | 0 | 0 |
| **Mean (3 queries)** | | **0.1** | **0.666667** | **17/99 = 0.171717** | **MAP = 4/9 = 0.444444** |

q4 retrieves documents 1 and 2 but has no relevant documents. q5 has no labels.
Both are **listed separately and left out of every mean**.

**Pivoted**: at s = 0.5 the rankings match cosine, so MAP = 4/9. At s = 0, q2 becomes
3, 1, 2, which puts document 2 at rank 3 and gives AP = 1/3. MAP = (5/6 + 1/3 + 0)/3 =
**7/18 = 0.388889**.

`tests/test_metrics.py` adds a ranking longer than 10: ranking 1–15 with relevant
{1, 3, 12, 20} gives P@10 = 0.2, R@10 = 0.5, F1@10 = 0.285714, and
AP = (1 + 2/3 + 3/12 + 0)/4 = 0.479167. This checks that AP counts ranks past 10 and
scores an unretrieved relevant document as 0.

## Further checks on the real collection

`tests/test_corpus.py` rebuilds the same model as a dense NumPy matrix, written
separately from the engine (it reuses only the preprocessing). It checks that IDF, every norm, the pivot, and all 100
cosine and pivoted scores (s ∈ {0, 0.2, 0.5, 0.8, 1}) match to a relative tolerance
of 1e-10 for several queries. The queries include a stopword-only query and an
unseen-terms-only query. The test also checks that the CSV and JSON files build the
same index.

**Bug-catching check.** The test suite was run against deliberately broken copies of
the engine: pivot averaged over all documents, log10 IDF, raw TF, reversed tie-break,
swapped s and 1−s, AP divided by retrieved hits, and zero-score documents retrieved.
Every one was caught. After the switch to stemming and lnc.ltc, two more were added:
IDF applied to document weights, and stemming switched off. Both are caught.

`tests/test_preprocess.py` checks the Porter stemmer on the examples given for each
step in Porter (1980).
