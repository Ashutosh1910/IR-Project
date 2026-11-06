"""Checks for the attack module (irengine/attacks.py).

A hand-calculated toy edit, then checks on the real EPC collection: an empty edit
reproduces the clean scores, collection statistics stay frozen, padding alone never
raises a score, s = 1 still equals cosine, padding never contains a query stem, and
target selection follows its rules.
"""

import copy
import math
import unittest
from collections import Counter
from pathlib import Path

from irengine import InvertedIndex, cosine_scores, pivoted_scores, preprocess, rank
from irengine.attacks import (
    GROUPS,
    Attack,
    added_text,
    attack_grid,
    attacked_ranking,
    edited_document,
    edited_score,
    padding_stream,
    padding_text,
    padding_words,
    query_terms,
    rank_of,
    repetition_text,
    select_targets,
    split_sentences,
    top_k,
)
from irengine.data import index_documents, load_articles, load_qrels, load_queries

ROOT = Path(__file__).resolve().parent.parent
L = math.log(2)
S = 0.8

TOY = [
    (1, "Apple apple banana."),
    (2, "Apple, cherry!"),
    (3, "Banana cherry date"),
    (4, "The and of it."),
]


class ToyEdit(unittest.TestCase):
    """Doc 3 "Banana cherry date" gets "apple banana" appended twice (query "apple banana")."""

    def setUp(self):
        self.ix = InvertedIndex(TOY)
        self.qvec = self.ix.vectorize("apple banana")  # appl: L, banana: L (df 2 of N 4)
        extra = repetition_text("apple banana", 2)
        self.assertEqual(extra, "apple banana apple banana")
        self.doc = edited_document(Counter(preprocess(TOY[2][1])), extra)

    def test_weights_and_norm(self):
        # counts after the edit: appl 2, banana 3, cherri 1, date 1
        self.assertEqual(self.doc.weights, {"appl": 1 + L, "banana": 1 + math.log(3),
                                            "cherri": 1.0, "date": 1.0})
        norm = math.sqrt((1 + L) ** 2 + (1 + math.log(3)) ** 2 + 2)
        self.assertAlmostEqual(self.doc.norm, norm, places=12)
        self.assertEqual((self.doc.added_words, self.doc.added_tokens), (4, 4))

    def test_scores_use_frozen_pivot(self):
        dot = L * (1 + L) + L * (1 + math.log(3))
        nq = L * math.sqrt(2)
        clean_pivot = (math.sqrt((1 + L) ** 2 + 1) + math.sqrt(2) + math.sqrt(3)) / 3
        self.assertAlmostEqual(edited_score(self.ix, self.qvec, self.doc),
                               dot / (nq * self.doc.norm), places=12)
        self.assertAlmostEqual(edited_score(self.ix, self.qvec, self.doc, S),
                               dot / (nq * ((1 - S) * clean_pivot + S * self.doc.norm)), places=12)

    def test_ranking_changes_only_the_target(self):
        # Clean cosine, times n_q: doc 1 L(2+L)/N1 = 0.949, doc 2 L/sqrt2 = 0.490, doc 3 L/sqrt3 = 0.400.
        # Edited doc 3: L(2+L+ln3)/norm = 0.863, so it passes doc 2 but not doc 1.
        clean = cosine_scores(self.ix, self.qvec)
        self.assertEqual(rank_of(rank(self.ix, clean), 3, self.ix.N + 1), 3)
        new = edited_score(self.ix, self.qvec, self.doc)
        ranking = attacked_ranking(self.ix, clean, 3, new)
        self.assertEqual(rank_of(ranking, 3, self.ix.N + 1), 2)
        self.assertEqual([d for d, _ in ranking if d != 3],
                         [d for d, _ in rank(self.ix, clean) if d != 3])
        self.assertEqual(rank_of(ranking, 4, self.ix.N + 1), 5)  # zero vector: unretrieved


class Helpers(unittest.TestCase):
    def test_grid(self):
        grid = attack_grid()
        self.assertEqual(len(grid), 24)
        self.assertEqual(len(set(grid)), 24)
        self.assertEqual(sum(a.kind == "combined" for a in grid), 16)

    def test_query_terms_drop_stopwords_keep_order_and_repeats(self):
        q9 = "what new sports facilities and tournaments did the sports secretary promise to students"
        self.assertEqual(query_terms(q9), ["new", "sports", "facilities", "tournaments", "sports",
                                           "secretary", "promise", "students"])
        self.assertEqual(query_terms("HuEl"), ["huel"])
        self.assertEqual(len(repetition_text(q9, 5).split()), 40)

    def test_padding_words_round_half_up(self):
        self.assertEqual(padding_words(1014, 25), 254)  # 253.5
        self.assertEqual(padding_words(1018, 25), 255)  # 254.5 (banker's rounding would give 254)
        self.assertEqual(padding_words(1709, 200), 3418)

    def test_padding_text_rejects_query_stems(self):
        with self.assertRaises(AssertionError):
            padding_text(["the", "Elections", "were", "held"], 4, "election process")
        self.assertEqual(padding_text(["a", "b", "c"], 2, "election"), "a b")
        with self.assertRaises(ValueError):
            padding_text(["a"], 2, "election")


@unittest.skipUnless((ROOT / "articles.json").exists(), "articles.json not present")
class OnEPC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from run_epc import length_groups

        cls.articles = load_articles(ROOT / "articles.json")
        cls.by_id = {a["document_id"]: a for a in cls.articles}
        cls.texts = dict(index_documents(cls.articles))
        cls.ix = InvertedIndex(index_documents(cls.articles))
        cls.queries = load_queries(ROOT / "queries.json")
        cls.qrels = load_qrels(ROOT / "qrels_final.csv", known_doc_ids=set(cls.ix.doc_ids))
        cls.groups, _ = length_groups(cls.articles)
        cls.targets = select_targets(cls.ix, cls.queries, cls.qrels, cls.groups, S)
        cls.sentences = split_sentences(cls.articles)

    def edits(self, qid, doc, attacks):
        text = self.queries[qid]
        stream = padding_stream(self.sentences, text, self.qrels[qid], doc, qid=qid)
        base = Counter(preprocess(self.texts[doc]))
        wc = int(self.by_id[doc]["word_count"])
        return text, [(a, added_text(a, text, stream, wc)) for a in attacks], base

    def test_empty_edit_reproduces_clean_scores(self):
        empty = {d: edited_document(Counter(preprocess(self.texts[d])), "") for d in self.ix.doc_ids}
        for qid, text in self.queries.items():
            qvec = self.ix.vectorize(text)
            cos, piv = cosine_scores(self.ix, qvec), pivoted_scores(self.ix, qvec, S)
            for d in self.ix.doc_ids:
                doc = empty[d]
                self.assertEqual(doc.norm, self.ix.norms[self.ix.position[d]])
                self.assertEqual(edited_score(self.ix, qvec, doc), cos[self.ix.position[d]])
                self.assertEqual(edited_score(self.ix, qvec, doc, S), piv[self.ix.position[d]])

    def test_collection_statistics_stay_frozen(self):
        before = (copy.deepcopy(self.ix.idf), self.ix.pivot, self.ix.N, list(self.ix.norms))
        qid = "Q5"
        doc = self.targets[qid][0]
        text, edits, base = self.edits(qid, doc, attack_grid())
        qvec = self.ix.vectorize(text)
        clean = pivoted_scores(self.ix, qvec, S)
        for _, extra in edits:
            ed = edited_document(base, extra)
            attacked_ranking(self.ix, clean, doc, edited_score(self.ix, qvec, ed, S))
        self.assertEqual((self.ix.idf, self.ix.pivot, self.ix.N, self.ix.norms), before)

    def test_padding_never_raises_a_score_and_has_no_query_stems(self):
        pads = [a for a in attack_grid() if a.kind == "pad"]
        for qid, docs in self.targets.items():
            q_stems = set(preprocess(self.queries[qid]))
            for doc in docs:
                text, edits, base = self.edits(qid, doc, pads)
                qvec = self.ix.vectorize(text)
                clean = edited_document(base, "")
                for _, extra in edits:
                    self.assertFalse(q_stems & set(preprocess(extra)))
                    ed = edited_document(base, extra)
                    self.assertGreater(ed.norm, clean.norm)
                    for s in (None, S):
                        self.assertLessEqual(edited_score(self.ix, qvec, ed, s),
                                             edited_score(self.ix, qvec, clean, s))

    def test_padding_levels_are_prefixes_and_match_word_counts(self):
        qid = "Q3"
        doc = self.targets[qid][0]
        wc = int(self.by_id[doc]["word_count"])
        text, edits, _ = self.edits(qid, doc, [Attack("pad", pad_pct=p) for p in (25, 200)]
                                    + [Attack("combined", repeats=10, pad_pct=25)])
        small, large, combined = (extra for _, extra in edits)
        self.assertEqual(len(small.split()), padding_words(wc, 25))
        self.assertEqual(len(large.split()), padding_words(wc, 200))
        self.assertTrue(large.startswith(small))
        self.assertTrue(combined.endswith(small))
        self.assertEqual(len(combined.split()), 10 * len(query_terms(text)) + padding_words(wc, 25))

    def test_slope_one_still_equals_cosine(self):
        qid = "Q9"
        doc = self.targets[qid][3]
        text, edits, base = self.edits(qid, doc, attack_grid())
        qvec = self.ix.vectorize(text)
        for _, extra in edits:
            ed = edited_document(base, extra)
            self.assertEqual(edited_score(self.ix, qvec, ed, 1.0), edited_score(self.ix, qvec, ed))

    def test_edit_on_counts_equals_preprocessing_joined_text(self):
        qid = "Q12"
        doc = self.targets[qid][5]
        _, edits, base = self.edits(qid, doc, [Attack("combined", repeats=5, pad_pct=50)])
        extra = edits[0][1]
        joined = Counter(preprocess(self.texts[doc] + "\n" + extra))
        self.assertEqual(edited_document(base, extra).weights,
                         edited_document(joined, "").weights)

    def test_target_selection(self):
        self.assertEqual(self.targets, select_targets(self.ix, self.queries, self.qrels, self.groups, S))
        for qid, docs in self.targets.items():
            qvec = self.ix.vectorize(self.queries[qid])
            top = top_k(rank(self.ix, cosine_scores(self.ix, qvec))) | top_k(
                rank(self.ix, pivoted_scores(self.ix, qvec, S)))
            self.assertEqual(len(docs), 6)
            self.assertEqual(len(set(docs)), 6)
            self.assertEqual([self.groups[d] for d in docs], [g for g in GROUPS for _ in range(2)])
            for d in docs:
                self.assertNotIn(d, self.qrels[qid])
                self.assertNotIn(d, top)


if __name__ == "__main__":
    unittest.main()
