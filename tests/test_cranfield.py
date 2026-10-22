"""Steps 20-23: binarization, query split, slope selection and the significance test.

These tests use small hand-made inputs, so they run without ir_datasets or a download."""

import unittest

from irengine.cranfield import binarize, kfold_split, select_slope, split_queries
from irengine.significance import paired_randomization_test

GRADED = {
    "1": {10: 4, 11: 2, 12: 1, 13: -1},
    "2": {20: 1, 21: -1},
    "3": {},
}


class Binarize(unittest.TestCase):
    def test_default_threshold_is_grade_2(self):
        self.assertEqual(binarize(GRADED), {"1": {10, 11}, "2": set(), "3": set()})

    def test_other_thresholds(self):
        self.assertEqual(binarize(GRADED, 1)["1"], {10, 11, 12})
        self.assertEqual(binarize(GRADED, 1)["2"], {20})
        self.assertEqual(binarize(GRADED, 4)["1"], {10})

    def test_no_interest_grade_is_never_relevant(self):
        for t in (1, 2, 3, 4):
            rel = binarize(GRADED, t)
            self.assertNotIn(13, rel["1"])
            self.assertNotIn(21, rel["2"])

    def test_every_query_keeps_an_entry(self):
        self.assertEqual(set(binarize(GRADED, 4)), {"1", "2", "3"})


class Split(unittest.TestCase):
    IDS = [str(i) for i in range(1, 226)]

    def test_partition(self):
        dev, held = split_queries(self.IDS, seed=15)
        self.assertEqual(len(dev), 112)
        self.assertEqual(len(held), 113)
        self.assertFalse(set(dev) & set(held))
        self.assertEqual(set(dev) | set(held), set(self.IDS))

    def test_fixed_by_seed_and_independent_of_input_order(self):
        a = split_queries(self.IDS, seed=15)
        b = split_queries(list(reversed(self.IDS)), seed=15)
        self.assertEqual(a, b)
        self.assertNotEqual(a, split_queries(self.IDS, seed=16))

    def test_parts_in_numeric_id_order(self):
        dev, held = split_queries(self.IDS, seed=15)
        self.assertEqual(dev, sorted(dev, key=int))
        self.assertEqual(held, sorted(held, key=int))


class KFold(unittest.TestCase):
    IDS = [str(i) for i in range(1, 226)]

    def test_five_disjoint_folds_of_45_cover_all_queries(self):
        folds = kfold_split(self.IDS, k=5, seed=15)
        self.assertEqual([len(f) for f in folds], [45] * 5)
        self.assertEqual(sorted(q for f in folds for q in f), sorted(self.IDS))

    def test_uneven_sizes_differ_by_at_most_one(self):
        sizes = [len(f) for f in kfold_split([str(i) for i in range(1, 13)], k=5, seed=15)]
        self.assertEqual(sorted(sizes), [2, 2, 2, 3, 3])

    def test_fixed_by_seed_and_independent_of_input_order(self):
        a = kfold_split(self.IDS, k=5, seed=15)
        self.assertEqual(a, kfold_split(list(reversed(self.IDS)), k=5, seed=15))
        self.assertNotEqual(a, kfold_split(self.IDS, k=5, seed=16))
        for fold in a:
            self.assertEqual(fold, sorted(fold, key=int))


class SelectSlope(unittest.TestCase):
    def test_highest_map_wins(self):
        self.assertEqual(select_slope({0.0: 0.20, 0.3: 0.25, 1.0: 0.22}), 0.3)

    def test_tie_goes_to_larger_slope(self):
        self.assertEqual(select_slope({0.2: 0.25, 0.6: 0.25, 1.0: 0.10}), 0.6)
        # equal up to float noise is still a tie
        self.assertEqual(select_slope({0.2: 0.1 + 0.2, 0.6: 0.3}), 0.6)


class RandomizationTest(unittest.TestCase):
    def test_identical_systems_give_p_1(self):
        self.assertEqual(paired_randomization_test([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]), 1.0)

    def test_consistent_improvement_is_significant(self):
        # exact two-sided p for 10 equal positive differences is 2 / 2**10
        p = paired_randomization_test([1.0] * 10, [0.0] * 10)
        self.assertLess(p, 0.01)

    def test_deterministic_and_symmetric(self):
        a, b = [0.3, 0.1, 0.8, 0.4], [0.2, 0.3, 0.5, 0.4]
        self.assertEqual(paired_randomization_test(a, b), paired_randomization_test(a, b))
        self.assertEqual(paired_randomization_test(a, b), paired_randomization_test(b, a))

    def test_rejects_unpaired_input(self):
        with self.assertRaises(ValueError):
            paired_randomization_test([0.1], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
