# Correcting Length Bias: Pivoted Document Length Normalization (Group 15)

A TF–IDF retrieval engine written from scratch that compares cosine normalization with
pivoted cosine normalization (Singhal, Buckley & Mitra, 1996) on 100 English Press
Club articles (`articles.json` / `articles.csv`).

## Usage

Requires Python 3.10+. NumPy is needed only for the tests.

```
python run_engine.py stats
python run_engine.py search "students union election commission" --s 0 0.5 --k 10
python run_engine.py evaluate --queries queries.json --qrels qrels.csv --out results/
python -m unittest -v
```

`evaluate` runs cosine plus pivoted at s = 0.0, 0.1, …, 1.0 by default (`--s` changes
the list). It reports P@10, R@10, F1@10 and MAP, and lists queries with no relevant
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
run_engine.py                   command-line runner
docs/engine_design.md           design decisions for steps 10–17
docs/validation.md              hand-calculated example (step 18)
tests/                          unit tests: toy example, metrics, preprocessing, NumPy cross-check
```
