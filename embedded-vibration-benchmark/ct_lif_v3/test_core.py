import unittest
import numpy as np
from .core import raw_features, compress, pooling_weights, balanced_weights, ResidualLIF, operating_points, fit_linear, linear_forward


class RevisionTests(unittest.TestCase):
    def test_soft_compression_preserves_large_amplitudes(self):
        x = compress(np.array([0., .1, 1., 2., 10.]))
        self.assertTrue(np.all(np.diff(x) > 0))
        self.assertTrue(np.all((x >= 0) & (x < 1)))

    def test_invariant_representation_and_gain(self):
        raw = np.random.default_rng(4).normal(size=(2, 34, 18, 3))
        for variant in ['mean', 'amplitude', 'invariant']:
            np.testing.assert_allclose(raw_features(raw * .8, np.ones(3), variant), .8 * raw_features(raw, np.ones(3), variant))
        np.testing.assert_allclose(raw_features(raw, np.ones(3), 'invariant'), raw_features(-raw, np.ones(3), 'invariant'))

    def test_last_33_frames_suffice(self):
        raw = np.random.default_rng(4).normal(size=(2, 34, 18, 3))
        for variant in ['mean', 'amplitude', 'invariant']:
            np.testing.assert_array_equal(raw_features(raw, np.ones(3), variant), raw_features(raw[:, 1:], np.ones(3), variant))

    def test_recent_readout_weights(self):
        weights = pooling_weights('recent')
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertAlmostEqual(weights[-1] / weights[0], 5)
        self.assertAlmostEqual(weights[-8:].sum(), .625)

    def test_training_cells_have_equal_total_weight(self):
        y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1])
        obj = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        weights = balanced_weights(y, obj)
        totals = [weights[(obj == o) & (y == label)].sum() for o in [0, 1] for label in [0, 1]]
        np.testing.assert_allclose(totals, totals[0])
        self.assertAlmostEqual(weights.mean(), 1)

    def test_subthreshold_path_retains_evidence(self):
        x = np.full((2, 32, 3), .1)
        x[1] *= 2
        model = ResidualLIF(3, 2, residual=True)
        model.params['W'][:] = 0
        model.params['q'][:] = 1
        np.testing.assert_allclose(model.forward(x), [.3, .6])

    def test_exact_readout_gradients(self):
        x = np.random.default_rng(2).uniform(size=(4, 32, 3))
        model = ResidualLIF(3, 2, residual=True, pooling='recent')
        y, weights = np.array([0., 1., 1., 0.]), np.array([.5, 1.5, 1.5, .5])
        _, grads = model.gradients(x, y, weights)
        for key in ['q', 'w', 'c']:
            for i in range(model.params[key].size):
                original = model.params[key][i]
                model.params[key][i] = original + 1e-6
                plus = model.gradients(x, y, weights)[0]
                model.params[key][i] = original - 1e-6
                minus = model.gradients(x, y, weights)[0]
                model.params[key][i] = original
                self.assertAlmostEqual((plus - minus) / 2e-6, grads[key][i], places=7)

    def test_joint_constraint_and_fallback_are_explicit(self):
        y = np.tile([0., 1.], 4)
        objects = np.repeat([0, 1], 4)
        reference = {'slip_recall': .8, 'false_alarm_rate': .1}
        good = operating_points(y, np.where(y == 1, .8, .2), objects, reference)
        self.assertTrue(good['joint']['validation_joint_improvement'])
        bad = operating_points(y, np.full(8, .5), objects, reference)
        self.assertFalse(bad['joint']['validation_joint_improvement'])

    def test_linear_control_fits_separable_data(self):
        x = np.repeat(np.array([.1, .2, 2., 3.])[:, None, None], 32, axis=1)
        y, objects = np.array([0., 0., 1., 1.]), np.zeros(4)
        params = fit_linear(x, y, objects, 'uniform', .001)
        np.testing.assert_array_equal(linear_forward(compress(x), params, 'uniform') > 0, y)


if __name__ == '__main__':
    unittest.main(verbosity=2)
