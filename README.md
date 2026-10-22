# Correcting Length Bias: Pivoted Document Length Normalization (Group 15)

A TF–IDF retrieval engine written from scratch that compares cosine normalization with
pivoted cosine normalization (Singhal, Buckley & Mitra, 1996) on 100 English Press
Club articles (`articles.json` / `articles.csv`).

## Usage

Requires Python 3.10+. The engine uses only the standard library. `ir_datasets` is
needed for the Cranfield run and NumPy for the tests (`pip install -r requirements.txt`).

```
python run_engine.py stats
python run_cranfield.py
python run_engine.py search "students union election commission" --k 10
python run_engine.py evaluate --queries queries.json --qrels qrels.csv --out results/
python -m unittest -v
```

`run_cranfield.py` picks the pivoted slope on Cranfield by MAP, checks it with 5-fold
cross-validation and freezes it in `frozen_slope.json` (currently **s = 0.8**; see
`docs/cranfield.md`). `search` and `evaluate` run cosine plus pivoted at the frozen
slope by default (`--s` changes the list). It reports P@10, R@10, F1@10 and MAP, and lists queries with no relevant
documents separately, leaving them out of the averages. File formats are described in
`docs/engine_design.md`.

## Layout

```
irengine/
  preprocess.py, stopwords.py   tokenisation and fixed stopword list   (step 10)
  index.py                      inverted index, log-TF, IDF, norms, pivot (steps 10, 11, 13)
  scoring.py                    cosine, pivoted, ranking and tie-break (steps 12, 14, 15)
  metrics.py                    P@k, R@k, F1@k, AP/MAP, no-relevant handling (steps 16, 17)
  data.py                       loaders for articles, queries and qrels
  cranfield.py                  Cranfield loading, binarization, split, CV folds, slope choice (steps 19-23)
  significance.py               paired randomization test
run_engine.py                   command-line runner
run_cranfield.py                Cranfield slope sweep and freeze (steps 19-24)
frozen_slope.json               the frozen slope s used for all EPC runs (step 24)
docs/engine_design.md           design decisions for steps 10–17
docs/validation.md              hand-calculated example (step 18)
docs/cranfield.md               Cranfield decisions and results (steps 19-24)
tests/                          unit tests: toy example, metrics, preprocessing, Cranfield helpers,
                                NumPy cross-check
```
