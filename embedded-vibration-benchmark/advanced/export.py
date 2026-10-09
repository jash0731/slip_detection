"""Deterministic float32/int8-weight export; states and biases remain float32."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .models import BETA, RHO, BASE_THRESHOLD, ADAPT_STRENGTH


def quantize(values):
    array = np.asarray(values, dtype=np.float32)
    scale = np.float32(max(float(np.abs(array).max()) / 127, 1e-12))
    codes = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    return codes, scale


def reference(model, trace, quantized=False):
    """Explicit ordered float32 reference matching straightforward C loops."""
    iw = model.input_layer.weight.detach().numpy().copy()
    ow = model.output_layer.weight.detach().numpy()[0].copy()
    bias = model.input_layer.bias.detach().numpy()
    output_bias = np.float32(model.output_layer.bias.detach().numpy()[0])
    if quantized:
        q, scale = quantize(iw)
        iw = q.astype(np.float32) * scale
        q, scale = quantize(ow)
        ow = q.astype(np.float32) * scale
    voltage, adaptation, hidden_sum = (np.zeros(16, np.float32) for _ in range(3))
    for inputs in trace:
        for h in range(16):
            current = np.float32(0)
            for i in range(6):
                current = np.float32(current + np.float32(iw[h, i] * inputs[i]))
            current = np.float32(current + bias[h])
            if model.kind == 'ann':
                activity = np.float32(max(current, 0))
            else:
                voltage[h] = np.float32(np.float32(BETA * voltage[h]) + current)
                threshold = np.float32(BASE_THRESHOLD + np.float32(ADAPT_STRENGTH * adaptation[h])) if model.kind == 'alif' else np.float32(BASE_THRESHOLD)
                activity = np.float32(voltage[h] >= threshold)
                voltage[h] = np.float32(voltage[h] - np.float32(activity * threshold))
                if model.kind == 'alif':
                    adaptation[h] = np.float32(np.float32(RHO * adaptation[h]) + activity)
            hidden_sum[h] = np.float32(hidden_sum[h] + activity)
    value = np.float32(0)
    for h in range(16):
        value = np.float32(value + np.float32(ow[h] * np.float32(hidden_sum[h] / len(trace))))
    value = np.float32(value + output_bias)
    return {'voltage': voltage.tolist(), 'adaptation': adaptation.tolist(),
            'hidden_sum': hidden_sum.tolist(), 'counts': hidden_sum.tolist() if model.kind != 'ann' else [0] * 16,
            'score': float(value)}


def export_models(path, models, thresholds, trace, fixture_record, scales, identity):
    entries = []
    for kind in ('lif', 'alif', 'ann'):
        model = models[kind]
        iw = model.input_layer.weight.detach().numpy()
        ow = model.output_layer.weight.detach().numpy()[0]
        iq, inscale = quantize(iw)
        oq, outscale = quantize(ow)
        item = {'kind': kind, 'model_id': f'advanced_{identity}_{kind}',
                'input_weights': iw.tolist(), 'input_bias': model.input_layer.bias.detach().numpy().tolist(),
                'output_weights': ow.tolist(), 'output_bias': float(model.output_layer.bias.detach()[0]),
                'input_q': iq.tolist(), 'input_scale': float(inscale), 'output_q': oq.tolist(),
                'output_scale': float(outscale), 'decision_logit_threshold': thresholds[kind],
                'expected_float': reference(model, trace), 'expected_quant': reference(model, trace, True)}
        for mode in ('float', 'quant'):
            item[f'expected_{mode}']['decision'] = int(item[f'expected_{mode}']['score'] >= thresholds[kind])
        with torch.no_grad():
            score, v, a, hidden = model(torch.tensor(trace[None]), return_state=True)
        item['torch_reference'] = {'score': float(score[0]), 'hidden_sum': hidden[0].tolist(),
                                   'max_voltage_difference': float(np.max(np.abs(v[0].numpy() - np.array(item['expected_float']['voltage']))))}
        if model.kind != 'ann' and not np.array_equal(hidden[0].numpy(), np.array(item['expected_float']['hidden_sum'])):
            raise ValueError('Torch and ordered float32 spike parity failed')
        item['sha256'] = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        entries.append(item)
    result = {'schema': 'advanced_tactile_models_v1', 'data_origin': 'public_recorded_gripper_tactile',
              'inputs': 6, 'hidden': 16, 'trace_length': len(trace), 'sample_interval_us': 5556,
              'beta': BETA, 'rho': RHO, 'base_threshold': BASE_THRESHOLD, 'adapt_strength': ADAPT_STRENGTH,
              'trace': trace.tolist(), 'fixture': fixture_record, 'axis_scales_train_only': scales.tolist(),
              'input_description': 'mean pin x/y/z first differences, train p95 scale, clip +/-4, positive xyz then negative xyz',
              'state_reset': 'zero voltage/adaptation/hidden sums each window',
              'fixture_selection': 'first held-out positive window; no prediction screening; parity demo only',
              'quantization': 'int8 input/output weights per tensor; biases and states float32; no integer-MAC claim',
              'limitations': 'recorded existing-slip classification; no incipient onset, live sensing or ESP32 measurement',
              'models': entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    return result
