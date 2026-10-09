"""Statistical logic tests on invented counts; no performance evidence."""

import unittest
import numpy as np
from advanced_stats import balanced_accuracy, bootstrap


class TrialBootstrapTests(unittest.TestCase):
    def test_balanced_accuracy_and_missing_class(self):
        self.assertEqual(float(balanced_accuracy(np.array([3, 2, 2, 1]))), .625)
        with self.assertRaises(ValueError):
            balanced_accuracy(np.array([3, 0, 0, 1]))

    def test_paired_identical_models_have_zero_difference(self):
        counts = np.zeros((3, 2, 5, 4), dtype=int)
        counts[:] = [2, 3, 1, 1]
        a, omitted = bootstrap({'object': counts}, 100)
        b, _ = bootstrap({'object': counts}, 100)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:, 0] - a[:, 1], np.zeros(len(a)))
        self.assertEqual(omitted, 0)

    def test_all_single_class_resamples_are_rejected(self):
        counts = np.zeros((2, 2, 5, 4), dtype=int)
        counts[..., 0] = 4
        with self.assertRaises(ValueError):
            bootstrap({'object': counts}, 10)


if __name__ == '__main__':
    unittest.main()
