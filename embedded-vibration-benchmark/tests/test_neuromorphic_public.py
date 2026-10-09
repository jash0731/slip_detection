import unittest
import numpy as np

from neuromorphic.public_experiment import prepare


class PublicPipelineTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(7)
        return {'x': rng.normal(size=(128, 54)), 'group': np.repeat([0, 1], 64),
                'pose': np.repeat([1, 3], 64), 'original_label': np.tile(np.repeat([4, 5], 32), 2),
                'trial_names': np.array(['train', 'test']), 't_ms': np.tile(np.arange(64) * 5.55, 2)}

    def test_heldout_values_do_not_change_training_calibration(self):
        data = self.data()
        windows, scale, _ = prepare(data)
        data['x'][64:] *= 1000
        changed, changed_scale, _ = prepare(data)
        self.assertEqual(scale, changed_scale)
        self.assertEqual(windows[:2], changed[:2])

    def test_ignore_gaps_are_not_concatenated(self):
        data = self.data()
        data['original_label'][10] = 3
        windows, _, excluded = prepare(data)
        self.assertEqual(excluded, 1)
        self.assertFalse(any(w['group'] == 0 and w['start_sample'] == 0 for w in windows))
        self.assertTrue(any(w['group'] == 0 and w['start_sample'] == 32 for w in windows))

    def test_trial_boundaries_reset_first_difference(self):
        data = self.data()
        data['x'][64:] += 1e6
        windows, _, _ = prepare(data)
        heldout = next(w for w in windows if w['group'] == 1)
        self.assertEqual(heldout['raw'][0], 0)


if __name__ == '__main__':
    unittest.main()
