"""Cross-checks on the real EPC collection against an independent dense NumPy
implementation of the same formulas (written separately from the engine).
Documents are weighted 1 + ln tf with no IDF; queries (1 + ln tf) * IDF (lnc.ltc)."""

import math
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from irengine import InvertedIndex, cosine_scores, pivoted_scores, preprocess, search
from irengine.data import index_documents, load_articles

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "articles.json"
CSV = ROOT / "articles.csv"

QUERIES = [
    "students union election commission",
    "mess food quality hostel",
    "placement internship season companies",
    "cultural fest oasis",
    "the of and",                  # stopwords only -> zero query
    "quokka xylophone zebra",      # unseen terms only -> zero query
    "election election election",  # repeated term
]


@unittest.skipUnless(JSON.exists(), "articles.json not present")
class Corpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.articles = load_articles(JSON)
        cls.ix = InvertedIndex(index_documents(cls.articles))

        # dense reference model
        docs = [Counter(preprocess(t)) for _, t in index_documents(cls.articles)]
        vocab = sorted(set().union(*docs))
        cls.col = {t: j for j, t in enumerate(vocab)}
        tf = np.zeros((len(docs), len(vocab)))
        for i, c in enumerate(docs):
            for t, n in c.items():
                tf[i, cls.col[t]] = n
        df = (tf > 0).sum(axis=0)
        cls.ref_idf = np.log(len(docs) / df)
        with np.errstate(divide="ignore"):
            logtf = np.where(tf > 0, 1 + np.log(np.where(tf > 0, tf, 1)), 0.0)
        cls.W = logtf  # lnc: no IDF on documents
        cls.ref_norms = np.sqrt((cls.W ** 2).sum(axis=1))
        cls.ref_pivot = cls.ref_norms[cls.ref_norms > 0].mean()

    def ref_query(self, text):
        q = np.zeros(len(self.col))
        for t, n in Counter(preprocess(text)).items():
            if t in self.col:
                q[self.col[t]] = (1 + math.log(n)) * self.ref_idf[self.col[t]]
        return q

    def ref_scores(self, text, s=None):
        q = self.ref_query(text)
        nq = np.linalg.norm(q)
        if nq == 0:
            return np.zeros(len(self.W))
        nd = self.ref_norms if s is None else (1 - s) * self.ref_pivot + s * self.ref_norms
        return np.where(self.ref_norms > 0, self.W @ q / (nq * nd), 0.0)

    def test_collection_shape(self):
        self.assertEqual(self.ix.N, 100)
        self.assertEqual(len(set(self.ix.doc_ids)), 100)
        self.assertEqual(self.ix.vocabulary_size, len(self.col))

    def test_csv_and_json_build_identical_index(self):
        ix_csv = InvertedIndex(index_documents(load_articles(CSV)))
        self.assertEqual(ix_csv.doc_ids, self.ix.doc_ids)
        self.assertEqual(ix_csv.postings, self.ix.postings)

    def test_idf_norms_pivot_match_reference(self):
        for t, j in self.col.items():
            self.assertAlmostEqual(self.ix.idf[t], self.ref_idf[j], places=12)
        np.testing.assert_allclose(self.ix.norms, self.ref_norms, rtol=1e-12)
        self.assertAlmostEqual(self.ix.pivot, self.ref_pivot, places=10)

    def test_scores_match_reference(self):
        for q in QUERIES:
            qv = self.ix.vectorize(q)
            np.testing.assert_allclose(cosine_scores(self.ix, qv), self.ref_scores(q), rtol=1e-10, atol=1e-15)
            for s in (0.0, 0.2, 0.5, 0.8, 1.0):
                np.testing.assert_allclose(
                    pivoted_scores(self.ix, qv, s), self.ref_scores(q, s), rtol=1e-10, atol=1e-15
                )

    def test_s1_equals_cosine_on_corpus(self):
        for q in QUERIES:
            self.assertEqual(search(self.ix, q, s=1.0), search(self.ix, q))

    def test_s0_equal_factors_on_corpus(self):
        factors = [f for f, n in zip(self.ix.pivoted_factors(0.0), self.ix.norms) if n > 0]
        self.assertEqual(set(factors), {self.ix.pivot})

    def test_rankings_sorted_and_positive(self):
        for q in QUERIES:
            for s in (None, 0.0, 0.5):
                ranking = search(self.ix, q, s=s)
                scores = [sc for _, sc in ranking]
                self.assertTrue(all(sc > 0 for sc in scores))
                self.assertEqual(scores, sorted(scores, reverse=True))

    def test_zero_queries_retrieve_nothing(self):
        self.assertEqual(search(self.ix, "the of and"), [])
        self.assertEqual(search(self.ix, "quokka xylophone zebra", s=0.3), [])


if __name__ == "__main__":
    unittest.main()
