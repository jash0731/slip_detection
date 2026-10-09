import unittest
import numpy as np
from ct_lif import encode, fit_scale, TinyModel, sigmoid, loss_value, fit, audit_splits


def raw_from_d(d):
    return np.concatenate([np.zeros((len(d), 1, 18, 3)), np.cumsum(d, axis=1)], axis=1)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(31)
        self.raw = raw_from_d(self.rng.normal(size=(5, 10, 18, 3)))
        self.scale = fit_scale(self.raw)

    def test_bounds_and_silence(self):
        f = encode(self.raw, self.scale)
        self.assertEqual(f.shape, (5, 9, 12))
        self.assertTrue(np.isfinite(f).all() and (f >= 0).all() and (f <= 1 + 1e-12).all())
        np.testing.assert_array_equal(encode(np.zeros_like(self.raw), self.scale), 0)

    def test_cancellation_survives_mean_blindness(self):
        d = np.zeros((1, 3, 18, 3))
        d[:, :, :9, 0] = 1
        d[:, :, 9:, 0] = -1
        raw = raw_from_d(d)
        np.testing.assert_array_equal(encode(raw, np.ones(3), variant='mean'), 0)
        f = encode(raw, np.ones(3))
        np.testing.assert_allclose(f[:, :, 6], 1)
        np.testing.assert_allclose(f[:, :, 9], 0)

    def test_turnover_distinguishes_equal_histograms(self):
        d = np.zeros((1, 3, 18, 3))
        for t in range(3):
            d[0, t, 2*t, 0] = 1
            d[0, t, 2*t+1, 0] = -1
        f = encode(raw_from_d(d), np.ones(3))
        entropy = encode(raw_from_d(d), np.ones(3), variant='entropy')
        np.testing.assert_allclose(f[:, :, 9], 1)
        np.testing.assert_allclose(entropy[:, :, 9], 0)

    def test_positive_gain_invariance_above_gate(self):
        np.testing.assert_allclose(encode(self.raw, self.scale), encode(self.raw * np.array([.2, 2, 4]), self.scale), atol=1e-12)

    def test_gate_suppresses_tiny_signal(self):
        self.assertLess(np.max(encode(self.raw * 1e-8, self.scale)), 1e-5)

    def test_fixed_pin_permutation_and_offsets(self):
        permutation = self.rng.permutation(18)
        offset = self.rng.normal(size=(1, 1, 18, 3))
        np.testing.assert_allclose(encode(self.raw, self.scale), encode(self.raw[:, :, permutation] + offset, self.scale), atol=1e-12)

    def test_causal_prefix(self):
        changed = self.raw.copy()
        changed[:, 7:] += 1e4
        np.testing.assert_allclose(encode(self.raw, self.scale)[:, :5], encode(changed, self.scale)[:, :5])

    def test_invalid_inputs(self):
        for raw, scale in [(self.raw[:, :, :1], self.scale), (self.raw * np.nan, self.scale), (self.raw, [1, 0, 1])]:
            with self.assertRaises(ValueError):
                encode(raw, scale)

    def test_parameter_budgets_and_reset(self):
        for inputs, hidden in [(12, 16), (6, 28)]:
            model = TinyModel(inputs, hidden)
            self.assertEqual(sum(p.size for p in model.params.values()), 225)
            x = self.rng.uniform(size=(3, 8, inputs))
            np.testing.assert_allclose(model.forward(x), np.concatenate([model.forward(x[i:i+1]) for i in range(3)]))

    def test_ann_and_temporal_gradients(self):
        x = self.rng.uniform(.1, .5, size=(4, 5, 3))
        y = np.array([0, 1, 1, 0])
        for kind in ['ann', 'temporal']:
            model = TinyModel(3, 4, kind)
            _, grads = model.gradients(x, y, 1.3)
            for key, index in [('W', (1, 2)), ('b', (2,)), ('w', (1,)), ('c', (0,))]:
                original = model.params[key][index]
                model.params[key][index] = original + 1e-6
                plus = loss_value(model.forward(x), y, 1.3)
                model.params[key][index] = original - 1e-6
                minus = loss_value(model.forward(x), y, 1.3)
                model.params[key][index] = original
                self.assertAlmostEqual(grads[key][index], (plus - minus) / 2e-6, places=7)

    def test_lif_gradient_against_forward_sensitivity(self):
        x = self.rng.uniform(0, 2, size=(4, 6, 3))
        y = np.array([0, 1, 0, 1])
        model = TinyModel(3, 4)
        _, grads = model.gradients(x, y)
        i, j = 1, 2
        u = np.zeros((4, 4))
        tangent = np.zeros(4)
        dzdw = np.zeros(4)
        spike_count = 0
        for t in range(6):
            v = .875 * u + x[:, t] @ model.params['W'] + model.params['b']
            s = (v >= 1).astype(float)
            spike_count += s.sum()
            tangent = .875 * tangent + x[:, t, i]
            dzdw += model.params['w'][j] * tangent / (1 + np.abs(v[:, j] - 1)) ** 2 / 6
            u = v - s
        self.assertGreater(spike_count, 0)
        expected = np.mean((sigmoid(model.forward(x)) - y) * dzdw)
        self.assertAlmostEqual(grads['W'][i, j], expected, places=12)

    def test_fit_smoke(self):
        x = self.rng.uniform(size=(24, 6, 12))
        y = np.tile([0., 1.], 12)
        model = TinyModel()
        before = model.params['W'].copy()
        threshold, history = fit(model, x[:16], y[:16], x[16:], y[16:], epochs=3)
        self.assertTrue(.05 <= threshold <= .95 and len(history) == 3)
        self.assertTrue(np.isfinite(model.forward(x)).all())
        self.assertFalse(np.array_equal(before, model.params['W']))

    def test_split_leakage_rejected(self):
        def split(obj, trial):
            return (self.raw[:2], np.array([0., 1.]), np.array([obj, obj]), np.array([trial, trial]))
        with self.assertRaises(ValueError):
            audit_splits([split('a', '1'), split('a', '1')])
        with self.assertRaises(ValueError):
            audit_splits([split('a', '1'), split('a', '2'), split('a', '3')])
        audit_splits([split('a', '1'), split('a', '2'), split('b', '1')])


if __name__ == '__main__':
    unittest.main(verbosity=2)
