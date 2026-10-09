import unittest

from followup.analyze import paired_counts, intervals


def fixtures():
    return [dict(held_object='object', seed='7', group=str(group), end_sample=str(end),
                 model=model, label=str(label), decision=str(decision), draw=str(draw))
            for group in (0, 1) for end, label in ((63, 0), (79, 1))
            for model, ndraws in (('ordered', 1), ('shuffled', 3)) for draw in range(ndraws)
            for decision in [label if model == 'ordered' or draw == 0 else 1 - label]]


class FollowupAnalysisTests(unittest.TestCase):
    def test_draws_average_counts_and_preserve_pairing(self):
        arrays, methods, seeds, windows = paired_counts(fixtures(), lambda r: r['model'], {'shuffled': 3})
        self.assertEqual(windows, 4)
        result = intervals(arrays, methods, [('ordered', 'shuffled')], draws=20)
        self.assertEqual(result['means']['ordered'], 1.)
        self.assertAlmostEqual(result['means']['shuffled'], 1 / 3)
        self.assertAlmostEqual(result['comparisons'][0]['difference_percentage_points'], 200 / 3)
        self.assertEqual(result['replicates_retained'], 20)

    def test_missing_or_relabelled_pair_rejected(self):
        rows = fixtures()
        with self.assertRaisesRegex(ValueError, 'Missing paired'):
            paired_counts(rows[:-1], lambda r: r['model'], {'shuffled': 3})
        rows[-1]['label'] = '0'
        with self.assertRaisesRegex(ValueError, 'different labels'):
            paired_counts(rows, lambda r: r['model'], {'shuffled': 3})

    def test_duplicate_draw_rejected(self):
        rows = fixtures()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            paired_counts(rows + [rows[0]], lambda r: r['model'], {'shuffled': 3})


if __name__ == '__main__':
    unittest.main()
