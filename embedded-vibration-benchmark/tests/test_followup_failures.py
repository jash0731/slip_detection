import unittest
import numpy as np

from followup.failures import label_context, contiguous_error_runs


class FailureAuditTests(unittest.TestCase):
    def test_context_and_annotated_age_are_separate(self):
        labels = np.array([-1] * 10 + [0] * 40 + [1] * 40)
        self.assertEqual(label_context(labels, 31)[0], 'ignored_context')
        self.assertEqual(label_context(labels, 45)[0], 'stable_non_slip')
        context, age, band, fraction = label_context(labels, 55)
        self.assertEqual((context, age, band), ('mixed_held_states', 5, '0_to_15_samples'))
        self.assertAlmostEqual(fraction, 6 / 32)
        self.assertEqual(label_context(labels, 89)[0], 'stable_slip')

    def test_error_runs_break_on_gaps_labels_and_correct_decisions(self):
        rows = [dict(end_sample=e, label=l, decision=d) for e, l, d in
                ((31, 1, 0), (47, 1, 0), (79, 1, 0), (95, 0, 1), (111, 0, 0), (127, 0, 1))]
        runs = contiguous_error_runs(rows)
        self.assertEqual([[r['end_sample'] for r in block] for block in runs], [[31, 47], [79], [95], [127]])


if __name__ == '__main__':
    unittest.main()
