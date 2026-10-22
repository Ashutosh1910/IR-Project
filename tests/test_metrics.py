"""Step 16/17: metric definitions on hand-worked rankings."""

import unittest

from irengine.metrics import (
    average_precision,
    evaluate,
    f1_at_k,
    precision_at_k,
    recall_at_k,
)


class Metrics(unittest.TestCase):
    # ranking 1..15; relevant = {1, 3, 12, 20}; doc 20 is never retrieved
    RANKING = list(range(1, 16))
    REL = {1, 3, 12, 20}

    def test_cutoff_at_10(self):
        self.assertAlmostEqual(precision_at_k(self.RANKING, self.REL), 2 / 10)
        self.assertAlmostEqual(recall_at_k(self.RANKING, self.REL), 2 / 4)
        self.assertAlmostEqual(f1_at_k(self.RANKING, self.REL), 2 * 0.2 * 0.5 / 0.7)

    def test_ap_counts_ranks_beyond_10_and_unretrieved_as_zero(self):
        self.assertAlmostEqual(
            average_precision(self.RANKING, self.REL), (1 / 1 + 2 / 3 + 3 / 12 + 0) / 4
        )

    def test_fewer_than_k_retrieved_still_divides_by_k(self):
        self.assertAlmostEqual(precision_at_k([5, 6], {5}), 1 / 10)
        self.assertAlmostEqual(recall_at_k([5, 6], {5}), 1.0)

    def test_perfect_and_empty(self):
        self.assertEqual(average_precision([1, 2], {1, 2}), 1.0)
        self.assertEqual(average_precision([], {1}), 0.0)
        self.assertEqual(f1_at_k([], {1}), 0.0)
        self.assertEqual(f1_at_k([9, 8], {1}), 0.0)

    def test_undefined_without_relevant_documents(self):
        for fn in (recall_at_k, f1_at_k):
            with self.assertRaises(ValueError):
                fn([1, 2], set())
        with self.assertRaises(ValueError):
            average_precision([1, 2], set())

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            precision_at_k([1, 1, 2], {1})

    def test_evaluate_excludes_no_relevant_queries_from_means(self):
        rankings = {"a": [1, 2, 3], "b": [4, 5], "c": [1]}
        qrels = {"a": {2}, "b": {5, 9}, "c": set()}
        ev = evaluate(rankings, qrels, k=2)
        self.assertEqual([q.query_id for q in ev.per_query], ["a", "b"])
        self.assertEqual([q.query_id for q in ev.no_relevant], ["c"])
        # a: P@2=1/2 R@2=1 AP=1/2 ; b: P@2=1/2 R@2=1/2 AP=(1/2)/2
        self.assertAlmostEqual(ev.mean_precision, 0.5)
        self.assertAlmostEqual(ev.mean_recall, 0.75)
        self.assertAlmostEqual(ev.mean_f1, (2 / 3 + 0.5) / 2)
        self.assertAlmostEqual(ev.map, (0.5 + 0.25) / 2)

    def test_evaluate_with_nothing_evaluable(self):
        ev = evaluate({"x": [1]}, {})
        self.assertIsNone(ev.map)
        self.assertIsNone(ev.mean_precision)


if __name__ == "__main__":
    unittest.main()
