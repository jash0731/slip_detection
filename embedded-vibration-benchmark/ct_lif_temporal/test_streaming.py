import unittest

import numpy as np

from .core import add_only_decisions, bypass_decisions
from .run import stream_decisions


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.records = ([dict(group=0, end_sample=i) for i in [63, 79, 95, 127]]
                        + [dict(group=1, end_sample=i) for i in [63, 79, 95]])
        self.scores = np.array([.9, .2, .8, .1, .3, .95, .1])

    def test_causality_resets_and_evaluation_equivalence(self):
        for alpha in [.4, .7]:
            for gap in [0., .05]:
                for margin in [None, 0., .15]:
                    choice = dict(kind='bypass', alpha=alpha, gap=gap, margin=margin, threshold=.55)
                    decisions = stream_decisions(self.scores, self.records, choice, .5)
                    reference = bypass_decisions(self.scores, self.records, alpha, gap, margin, [.55])[:, 0]
                    np.testing.assert_array_equal(decisions, reference)
                    for n in range(1, len(self.scores) + 1):
                        np.testing.assert_array_equal(
                            decisions[:n], stream_decisions(self.scores[:n], self.records[:n], choice, .5))
                    self.assertEqual(decisions[3], stream_decisions(
                        self.scores[3:4], self.records[3:4], choice, .5)[0])
                    np.testing.assert_array_equal(decisions[4:], stream_decisions(
                        self.scores[4:], self.records[4:], choice, .5))

    def test_batch_order_does_not_change_chronological_results(self):
        choice = dict(kind='bypass', alpha=.4, gap=.05, margin=.15, threshold=.55)
        permutation = np.array([6, 3, 0, 2, 4, 1, 5])
        original = stream_decisions(self.scores, self.records, choice, .5)
        shuffled = stream_decisions(self.scores[permutation],
                                    [self.records[i] for i in permutation], choice, .5)
        np.testing.assert_array_equal(original, shuffled[np.argsort(permutation)])

    def test_add_only_never_removes_an_existing_alarm(self):
        for alpha in [.4, .7]:
            for guard in [.05, .1, .2]:
                choice = dict(kind='add_only', alpha=alpha, guard=guard, threshold=.65)
                decisions = stream_decisions(self.scores, self.records, choice, .5)
                self.assertTrue(np.all(decisions[self.scores >= .5]))
                reference = add_only_decisions(self.scores, self.records, alpha, guard, [.65], .5)[:, 0]
                np.testing.assert_array_equal(decisions, reference)


if __name__ == '__main__':
    unittest.main()
