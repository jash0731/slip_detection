import unittest
import numpy as np
from .core import features
from ct_lif_study.reference.ct_lif import encode
from ct_lif_v3.core import raw_features, compress


class FusionTests(unittest.TestCase):
    def test_exact_feature_branches(self):
        raw = np.random.default_rng(1).normal(size=(3, 34, 18, 3))
        scale = np.ones(3)
        fused = features(raw, scale)
        self.assertEqual(fused.shape, (3, 32, 18))
        np.testing.assert_allclose(fused[:, :, :12], compress(encode(raw, scale)))
        np.testing.assert_allclose(fused[:, :, 12:], compress(raw_features(raw, scale, 'mean')))

    def test_ct_information_remains_for_cancelled_motion(self):
        d = np.zeros((1, 33, 18, 3))
        for t in range(33):
            d[:, t, t % 9, :] = 1
            d[:, t, 9 + t % 9, :] = -1
        raw = np.concatenate([np.zeros((1,1,18,3)), np.cumsum(d, axis=1)], axis=1)
        fused = features(raw, np.ones(3))
        np.testing.assert_allclose(fused[:, :, 12:], 0)
        self.assertTrue(np.all(fused[:, :, 6:12] > 0))

    def test_common_motion_retains_amplitude(self):
        raw = np.arange(34)[None,:,None,None] * np.full((1,1,18,3), .05)
        small = features(raw, np.ones(3))
        large = features(raw * 2, np.ones(3))
        self.assertTrue(np.all(large[:, :, 12:15] > small[:, :, 12:15]))
        np.testing.assert_allclose(small[:, :, 6:12], 0, atol=1e-14)

    def test_history_and_batch_independence(self):
        raw = np.random.default_rng(3).normal(size=(2, 40, 18, 3))
        x = features(raw, np.ones(3))
        np.testing.assert_array_equal(x, features(raw[:, -34:], np.ones(3)))
        np.testing.assert_array_equal(x[:1], features(raw[:1], np.ones(3)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
