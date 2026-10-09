import tempfile
from pathlib import Path
import unittest

from neuromorphic.experiment import (Network, encode, traces, split_recordings,
                                    fit_readout, score, export_header)


class NeuromorphicTests(unittest.TestCase):
    def test_dense_sparse_temporal_parity_and_cost(self):
        dense, sparse = Network(), Network(True)
        for sample in [0, .2, -.4, .7, 1.2, 0] * 31:
            self.assertEqual(dense.step(sample), sparse.step(sample))
            self.assertEqual(dense.voltage, sparse.voltage)
        self.assertEqual(dense.counts, sparse.counts)
        self.assertLess(sparse.operations, dense.operations)

    def test_sparse_silence_still_leaks(self):
        network = Network(True)
        network.voltage[0] = 0.5
        network.step(0)
        self.assertEqual(network.voltage[0], 0.4375)
        self.assertEqual(network.operations, 0)
        self.assertEqual(network.samples, 1)

    def test_encoder_boundaries_and_sign(self):
        self.assertEqual(encode(.12), (1, 0, 0, 0))
        self.assertEqual(encode(-.6), (1, 1, 1, 0))
        self.assertEqual(encode(1.2), (1, 1, 1, 1))

    def test_recording_split_disjoint_and_deterministic(self):
        recordings = traces()
        self.assertEqual(recordings, traces())
        train, test = split_recordings(recordings)
        self.assertFalse(set(r['recording_id'] for r in train) & set(r['recording_id'] for r in test))
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 40)

    def test_trained_readout_learns_separable_features(self):
        rows = [(0,), (.1,), (.8,), (1,)]
        weights, bias = fit_readout(rows, [0, 0, 1, 1])
        self.assertLess(score((.1,), weights, bias), 0)
        self.assertGreater(score((.8,), weights, bias), 0)
        self.assertEqual((weights, bias), fit_readout(rows, [0, 0, 1, 1]))

    def test_header_export_reproducible_and_dynamic_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.h'
            fixture = {'raw': [0, 1.2, .4], 'recording_id': 'synthetic_test'}
            export_header(path, fixture, [1] * 12, 0, sample_interval_us=5556)
            first = path.read_bytes()
            export_header(path, fixture, [1] * 12, 0, sample_interval_us=5556)
            self.assertEqual(first, path.read_bytes())
            self.assertIn(b'TRACE_LENGTH=3', first)
            self.assertIn(b'SAMPLE_INTERVAL_US=5556', first)


if __name__ == '__main__':
    unittest.main()
