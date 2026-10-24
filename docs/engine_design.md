# Retrieval engine: design decisions (steps 10–17)

All retrieval code lives in `irengine/` and uses only the Python standard library.
NumPy is used only in `tests/test_corpus.py`, as an independent reference
implementation. No prebuilt TF–IDF vectoriser or retrieval library is used.

| Step | What | Where |
|---|---|---|
| 10 | Preprocessing, stemming, inverted index | `irengine/preprocess.py`, `irengine/stopwords.py`, `irengine/porter.py`, `irengine/index.py` |
| 11 | Weights (lnc.ltc) | `irengine/index.py` (`log_tf`, `InvertedIndex.idf`, `vectorize`) |
| 12 | Cosine scoring | `irengine/scoring.py` (`cosine_scores`) |
| 13 | Pivot | `irengine/index.py` (`InvertedIndex.pivot`) |
| 14 | Pivoted scoring | `irengine/scoring.py` (`pivoted_scores`), `InvertedIndex.pivoted_factor(s)` |
| 15 | Tie-breaking | `irengine/scoring.py` (`rank`) |
| 16 | P@10, R@10, F1@10, MAP | `irengine/metrics.py` |
| 17 | Queries with no relevant documents | `irengine/metrics.py` (`evaluate`), `run_engine.py evaluate` |
| 18 | Validation | `docs/validation.md`, `tests/` |
| 19–24 | Cranfield slope selection and freeze | `docs/cranfield.md`, `run_cranfield.py` |

## Step 10: preprocessing and index

Articles and queries go through the same function, `irengine.preprocess.preprocess`:

1. Unicode NFKD, accents dropped (`café` → `cafe`), invisible characters removed
   (soft hyphen, zero-width spaces, word joiner U+2060, BOM). The EPC text contains a
   few word joiners.
2. Curly and modifier apostrophes (`’ ‘ ʼ`) become `'`, then the text is lowercased.
3. A possessive `'s` is dropped (`EC’s` → `ec`). Remaining apostrophes are deleted, so
   contractions stay one token (`don’t` → `dont`) and plural possessives lose the
   apostrophe (`Students’` → `students`).
4. Tokens are maximal runs of `[a-z0-9]`. Hyphens, dashes and other punctuation separate
   tokens (`co-curricular` → `co`, `curricular`). Numbers are kept (`2026`, `1970s`).
5. Stopwords come from a fixed list: the NLTK English list, written into the code
   (`stopwords.py`), plus `i’m, i’ve, we’ve, they’re, they’ve, they’ll, they’d`. The list
   goes through steps 1–4 itself, so it matches the normalised text (`it's` → `it`).
   Contractions that collapse into ordinary words (`we'll` → `well`, `I'd` → `id`,
   `she'll` → `shell`) are left out on purpose.
6. Every remaining token is reduced to its **Porter stem** (`elections` → `elect`,
   `HuEls` → `huel`), using `irengine/porter.py`, written from Porter (1980). It follows
   the original rules, not later variants. On all 16,009 distinct words in Cranfield
   and EPC it matches the Snowball "porter" stemmer, except that one- and two-letter
   words are left unchanged, as in Porter's own code. Stopwords are removed before
   stemming.

**Why stem.** The paper does not mention stemming, but its experiments ran on SMART,
whose standard indexing stems words. Without it, `election` and `elections` are
unrelated terms. Stemming is applied
identically for both methods, so it does not favour either. On Cranfield it raised
cosine MAP from 0.2643 to 0.2828 (with lnc.ltc weights); see `docs/cranfield.md`.

**What is indexed.** By default the title is indexed together with the body
(`title + "\n" + text`), because the relevance judges see both. `--no-title` indexes
the body only. The proposal's word counts (1,014–4,343 words) are body-only and come
before stopword removal. They are a different measure from the vector norm the
engine uses.

**Index structure.** `InvertedIndex.postings[term]` holds a list of
`(document position, raw count)`, sorted by position. `weighted_postings[term]` holds
the same entries with the document weight w(t,d). Document norms and the pivot are
computed once, when the index is built, and frozen after that.

Current EPC collection (title + body): 100 documents, 7,744 terms (stems), 967.8 tokens
per document after preprocessing (range 496–2,664), norms 24.15–53.87, pivot
**p = 33.128197**, no zero-norm documents, and no term in every document.

## Step 11: weights (the paper's lnc.ltc)

The paper (section 3) weights documents and queries differently: "the idf factor is
only used in the query term weights and not in the document term weights". In SMART
notation this is lnc.ltc, and the engine follows it.

- TF(t,x) = 1 + ln f(t,x) if f(t,x) > 0, else 0. IDF(t) = ln(N / df(t)).
- **Documents:** w(t,d) = TF(t,d), with no IDF. The norm n_d and the pivot are
  computed from these weights, as in the paper (its pivot is "the average cosine
  factor for 1+log(tf) weighted documents").
- **Queries:** w(t,q) = TF(t,q) · IDF(t). `vectorize` drops query terms that are not
  in the vocabulary before the query norm is computed, so an unseen term changes
  neither the dot product nor n_q (step 12).
- **N counts every document in the collection, including zero-vector ones.** A
  document that is all stopwords still belongs to the collection.
- A term that appears in every document has IDF 0, so its query weight is 0 and it
  never adds to a score. It still counts toward the document norms, because
  documents carry no IDF.

An earlier version used TF · IDF on both sides, which counts IDF twice in every score
and builds the norms from IDF-weighted vectors. The proposal called that an
adaptation; it was replaced to match the paper. On Cranfield, lnc.ltc gives higher
MAP for both cosine and pivoted (see `docs/cranfield.md`).

## Step 12: cosine

S_cos(q,d) = Σ_t w(t,q) w(t,d) / (n_q n_d). Dot products are added up term by term
over the postings lists, with query terms visited in sorted order, so the floating-point
result never depends on dict order. If n_q = 0 (empty query, all stopwords, or all
unseen terms), every score is 0. If n_d = 0, that document scores 0.

## Step 13: pivot

p = mean of n_d over D⁺ = {d : n_d > 0}. Zero-norm documents are excluded. If D⁺ is
empty, p is set to 0.0, but that value is never used because every score is 0 anyway.
p uses the same scale as n_d (vector norm, not token count). It is computed from the
clean collection, and `pivoted_scores(..., pivot=...)` lets later experiments keep it
fixed while edited documents are rescored.

## Step 14: pivoted cosine

S_piv(q,d) = Σ_t w(t,q) w(t,d) / (n_q · ((1 − s) p + s n_d)), with s ∈ [0, 1]. Values of
s outside that range are rejected. Everything else (weights, zero-vector handling,
ignoring unseen terms) is the same as cosine. At s = 1 the factor is exactly n_d, so
the scores are bit-for-bit identical to cosine. At s = 0 every nonzero document has
the factor p. The cosine path is still kept as separate code, so that "s = 1 equals
cosine" is a real check of the pivoted code and not true by construction.

## Step 15: ranking and ties

1. **Only documents with score > 0 are retrieved.** A score of 0 means the document
   shares no weighted term with the query (or has a zero vector). Listing such
   documents in an arbitrary order would give the metrics credit for nothing.
2. Documents are sorted by **decreasing score**, and **ties go to the lower document
   ID**.
3. Scores are compared **after rounding to 12 decimal places** (`TIE_DECIMALS`). Two
   scores that are mathematically equal but differ by floating-point noise
   (e.g. `0.1 + 0.2` vs `0.3`) still count as a tie. Real score differences in this
   collection are far larger than 1e-12. The reported score is never rounded.

Why document ID: the order is fully deterministic and does not depend on the order
of the input file (this is tested), and it does not use length or score, so it gives
neither method an advantage. One side effect: EPC IDs are WordPress post IDs, which
roughly follow publication order, so ties go slightly to older articles. Exact ties
are rare except at s = 0, where documents with equal dot products also get equal
factors.

## Step 16: metrics

For a ranking r₁, r₂, … and relevant set R (|R| > 0):

- P@10 = |top 10 ∩ R| / 10. **The divisor is always 10**, even when fewer than 10
  documents are retrieved.
- R@10 = |top 10 ∩ R| / |R|.
- F1@10 = 2·P·R / (P + R), or 0 when both are 0.
- AP = (1/|R|) Σ P@i over the ranks i that hold a relevant document, using the
  **full** ranking (not cut at 10). A relevant document that is never retrieved
  adds 0. This matches trec_eval.
- MAP = mean AP over the evaluated queries.

All averages are macro averages over queries. **Mean F1@10 is the mean of
per-query F1@10**, not the F1 of mean P and mean R. Rankings with duplicate IDs are
rejected.

## Step 17: queries with no relevant documents

R@10, F1@10 and AP are undefined when |R| = 0. Such queries are listed in
`Evaluation.no_relevant`, printed separately by `run_engine.py evaluate`, and written
to `summary.json`. They are excluded from **every** average, including P@10, so all
four metrics are averaged over the same set of queries. The tool tells apart a query
whose labels are all 0 from a query with no labels at all. Qrels that name an
unknown document ID cause an error.

## File formats for evaluation

- `queries.json`: `[{"query_id": "Q1", "text": "...", "description": "..."}]` (or a CSV
  with `query_id,text`).
- `qrels.csv`: `query_id,document_id,relevance`. Relevance > 0 means relevant.
  Rows with relevance 0 are allowed.
