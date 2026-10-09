import unittest
import numpy as np

from advanced.data import causal_derivatives, split_trials, prepare_fold
from advanced.train import trial_bootstrap, select_threshold


class AdvancedDataTests(unittest.TestCase):
    def sample(self):
        rng = np.random.default_rng(3)
        return {'x': rng.normal(size=(48, 54)).astype(np.float32),
                'group': np.repeat(np.arange(6), 8), 'object': np.repeat([0, 0, 1, 1, 2, 2], 8),
                'y_held': np.tile([-1, -1, -1, 0, -1, -1, -1, 1], 6),
                'trial_names': np.array([f'trial{i}' for i in range(6)]),
                't_ms': np.tile(np.arange(8) * 5.5, 6)}

    def test_outer_and_inner_groups_disjoint(self):
        data = self.sample()
        splits = split_trials(data, 2, 7)
        self.assertEqual(set(splits['test']), {4, 5})
        self.assertFalse(set(splits['train']) & set(splits['validation']))
        self.assertFalse(set(splits['train']) & set(splits['test']))

    def test_train_only_scaling_and_prior_ignored_context_retained(self):
        data = self.sample()
        splits = {'train': [0, 1], 'validation': [2, 3], 'test': [4, 5]}
        original, scale = prepare_fold(data, causal_derivatives(data), splits, window=4, stride=4)
        self.assertEqual(len(original['test'][1]), 4)
        # Three ignored history labels do not remove a last-tick held target.
        self.assertTrue(np.any(original['test'][0][0, 1:] != 0))
        data['x'][32:] *= 1000
        changed, changed_scale = prepare_fold(data, causal_derivatives(data), splits, window=4, stride=4)
        np.testing.assert_array_equal(scale, changed_scale)
        np.testing.assert_array_equal(original['train'][0], changed['train'][0])

    def test_future_samples_cannot_change_earlier_window(self):
        data = self.sample()
        splits = {'train': [0, 1], 'validation': [2, 3], 'test': [4, 5]}
        before, _ = prepare_fold(data, causal_derivatives(data), splits, window=4, stride=4)
        data['x'][36:40] += 100
        after, _ = prepare_fold(data, causal_derivatives(data), splits, window=4, stride=4)
        np.testing.assert_array_equal(before['test'][0][0], after['test'][0][0])

    def test_paired_bootstrap_identical_models_difference_zero(self):
        labels = np.array([0, 1, 0, 1])
        result = trial_bootstrap(labels, labels, labels, np.array([0, 0, 1, 1]), repetitions=30)
        self.assertEqual(result['lower_95'], 0)
        self.assertEqual(result['upper_95'], 0)

    def test_validation_threshold_selection(self):
        threshold = select_threshold(np.array([0, 0, 1, 1]), np.array([-1., -.3, .2, 1.]))
        self.assertGreater(threshold, -.3)
        self.assertLessEqual(threshold, .2)
