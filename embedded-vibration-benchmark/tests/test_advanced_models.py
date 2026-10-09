import unittest
import numpy as np
import torch

from advanced.models import TemporalModel, SurrogateSpike
from advanced.export import quantize, reference


class AdvancedModelTests(unittest.TestCase):
    def test_hard_spike_surrogate_gradient(self):
        values = torch.tensor([-.5, 0., .5], requires_grad=True)
        outputs = SurrogateSpike.apply(values)
        self.assertEqual(outputs.tolist(), [0, 1, 1])
        outputs.sum().backward()
        self.assertTrue(torch.all(values.grad > 0))

    def test_hidden_weights_receive_training_gradient(self):
        torch.manual_seed(7)
        model = TemporalModel('lif')
        inputs = torch.rand(4, 32, 6)
        logits = model(inputs)
        logits.sum().backward()
        self.assertGreater(float(model.input_layer.weight.grad.abs().sum()), 0)

    def test_alif_adaptation_changes_dynamics(self):
        lif, alif = TemporalModel('lif'), TemporalModel('alif')
        with torch.no_grad():
            lif.input_layer.weight.zero_()
            lif.input_layer.bias.fill_(.6)
            alif.load_state_dict(lif.state_dict())
            _, _, adaptation, alif_counts = alif(torch.zeros(1, 32, 6), True)
            _, _, _, lif_counts = lif(torch.zeros(1, 32, 6), True)
        self.assertTrue(torch.all(adaptation > 0))
        self.assertTrue(torch.all(alif_counts < lif_counts))

    def test_capacity_matched_and_window_reset(self):
        for kind in ('lif', 'alif', 'ann'):
            model = TemporalModel(kind)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 129)
            trace = torch.rand(1, 32, 6)
            self.assertTrue(torch.equal(model(trace), model(trace)))

    def test_ordered_float_reference_matches_torch_spikes(self):
        torch.manual_seed(3)
        trace = np.random.default_rng(2).random((32, 6)).astype(np.float32)
        for kind in ('lif', 'alif', 'ann'):
            model = TemporalModel(kind)
            with torch.no_grad():
                logits, _, _, counts = model(torch.from_numpy(trace[None]), True)
            ordered = reference(model, trace)
            self.assertAlmostEqual(float(logits[0]), ordered['score'], places=5)
            np.testing.assert_allclose(counts[0].numpy(), ordered['hidden_sum'], atol=1e-5)

    def test_weight_quantization_range_error_and_zero_tensor(self):
        values = np.array([-2., -.1, .1, 2.], np.float32)
        codes, scale = quantize(values)
        self.assertEqual(codes.dtype, np.int8)
        self.assertLessEqual(float(np.max(np.abs(codes.astype(np.float32) * scale - values))), float(scale) / 2 + 1e-6)
        zero, zero_scale = quantize(np.zeros(4))
        self.assertTrue(np.all(zero == 0))
        self.assertGreater(zero_scale, 0)
