"""Post-training int8 weight-only drift audit, no retraining or threshold retuning."""
import csv
import json
from pathlib import Path
import numpy as np
import torch

from .data import causal_derivatives, prepare_fold
from .models import TemporalModel
from .export import quantize
from .train import predict, quality


def run():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    output = root / 'advanced' / 'results'
    metadata = json.loads((output / 'metadata.json').read_text())
    with np.load(root / 'data' / 'arq_all_trials.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    derivatives = causal_derivatives(data)
    results = []
    for fold in metadata['folds']:
        prepared, _ = prepare_fold(data, derivatives, fold['split_groups'], window=metadata['protocol']['window'])
        inputs, labels, _ = prepared['test']
        for kind in ('lif', 'alif', 'ann'):
            model = TemporalModel(kind)
            model.load_state_dict(torch.load(output / f'{fold["name"]}_{kind}.pt', weights_only=True))
            floating = predict(model, inputs)
            with torch.no_grad():
                activities = torch.cat([model(torch.from_numpy(inputs[start:start + 256]), True)[3]
                                        for start in range(0, len(inputs), 256)])
                mean_activity = float(activities.mean() / inputs.shape[1])
                for parameter in (model.input_layer.weight, model.output_layer.weight):
                    codes, scale = quantize(parameter.numpy())
                    parameter.copy_(torch.from_numpy(codes.astype(np.float32) * scale))
            quantized = predict(model, inputs)
            threshold = fold['thresholds'][kind]
            f_decision, q_decision = floating >= threshold, quantized >= threshold
            results.append({'source': 'public_recorded_host_numeric_audit', 'fold': fold['name'], 'model': kind,
                            'test_windows': len(labels), 'float_balanced_accuracy': quality(labels, f_decision)['balanced_accuracy'],
                            'q8_weight_balanced_accuracy': quality(labels, q_decision)['balanced_accuracy'],
                            'float_f1': quality(labels, f_decision)['f1'], 'q8_weight_f1': quality(labels, q_decision)['f1'],
                            'decision_disagreements': int(np.sum(f_decision != q_decision)),
                            'max_absolute_logit_drift': float(np.max(np.abs(floating - quantized))),
                            'mean_hidden_activity_per_tick': mean_activity,
                            'activity_semantics': 'spike rate' if kind != 'ann' else 'ReLU activation magnitude',
                            'float_parameter_bytes': 516, 'q8_weight_parameter_bytes_with_scales': 188,
                            'numeric_state': 'float32 in both; dequantized weights; no integer-MAC benchmark'})
    with (output / 'numeric_audit.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print('Post-training numeric audit complete, thresholds unchanged.')


if __name__ == '__main__':
    run()
