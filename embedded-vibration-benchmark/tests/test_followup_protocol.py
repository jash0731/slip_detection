import unittest

import numpy as np
import torch

from followup.common import (LeakyRelu, make_model, perturb, select_candidate,
                             common_endpoints, detailed_quality)


class FollowupProtocolTests(unittest.TestCase):
    def test_capacity_initialization_and_conventional_memory(self):
        models = []
        for kind in ('ann', 'lif', 'leaky_relu'):
            torch.manual_seed(7)
            models.append(make_model(kind))
        self.assertEqual([sum(p.numel() for p in m.parameters()) for m in models], [129] * 3)
        for model in models[1:]:
            torch.testing.assert_close(model.input_layer.weight, models[0].input_layer.weight)
        network = LeakyRelu()
        with torch.no_grad():
            network.input_layer.weight.zero_()
            network.input_layer.weight[0, 0] = 1
            network.input_layer.bias.zero_()
            network.output_layer.weight.zero_()
            network.output_layer.weight[0, 0] = 1
            network.output_layer.bias.zero_()
        early = torch.zeros(1, 3, 6)
        early[0, 0, 0] = 1
        late = torch.zeros_like(early)
        late[0, 1, 0] = 1
        self.assertGreater(float(network(early).detach()), float(network(late).detach()))
        torch.testing.assert_close(network(torch.zeros_like(early)), torch.tensor([0.]))

    def test_joint_shuffle_preserves_endpoint_and_multiset(self):
        x = np.arange(2 * 8 * 6, dtype=np.float32).reshape(2, 8, 6)
        records = [{'group': 9, 'end_sample': 31}, {'group': 10, 'end_sample': 47}]
        for condition in ('shuffled_history', 'reversed_history'):
            changed = perturb(x, records, condition)
            np.testing.assert_array_equal(changed[:, -1], x[:, -1])
            for old, new in zip(x, changed):
                self.assertEqual(sorted(map(tuple, old)), sorted(map(tuple, new)))
            np.testing.assert_array_equal(changed, perturb(x, records, condition))
        model = make_model('ann')
        with torch.no_grad():
            torch.testing.assert_close(model(torch.from_numpy(x)), model(torch.from_numpy(perturb(x, records, 'shuffled_history'))))

    def test_validation_selection_and_ties(self):
        candidates = [dict(window=64, rate=.01, info={'validation_balanced_accuracy': .7}, test_score=.99),
                      dict(window=16, rate=.005, info={'validation_balanced_accuracy': .7}, test_score=.1),
                      dict(window=16, rate=.01, info={'validation_balanced_accuracy': .7}, test_score=.2)]
        self.assertIs(select_candidate(candidates), candidates[2])
        candidates[0]['info']['validation_balanced_accuracy'] = .71
        self.assertIs(select_candidate(candidates), candidates[0])

    def test_common_endpoint_alignment(self):
        prepared = {'test': (np.zeros((4, 32, 6), dtype=np.float32), np.array([0, 1, 0, 1]),
                             [{'end_sample': end} for end in (31, 47, 63, 79)])}
        aligned = common_endpoints(prepared)['test']
        self.assertEqual([r['end_sample'] for r in aligned[2]], [63, 79])
        np.testing.assert_array_equal(aligned[1], [0, 1])

    def test_single_class_subgroup_is_undefined(self):
        scores = detailed_quality(np.array([1, 1]), np.array([1, 0]))
        self.assertIsNone(scores['balanced_accuracy'])
        self.assertIsNone(scores['false_alarm_rate'])
        self.assertEqual(scores['slip_recall'], .5)


if __name__ == '__main__':
    unittest.main()
