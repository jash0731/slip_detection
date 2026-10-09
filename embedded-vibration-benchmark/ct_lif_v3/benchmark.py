import csv
import json
import platform
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from .run import ROOT, HERE, OUT, write_csv, write_json
from .core import ResidualLIF, compress, raw_features, linear_forward
from ct_lif_v2.core import encode as encode_v2
from ct_lif_study.run import load_data, windows
from ct_lif_study.reference.ct_lif import TinyModel, encode as encode_v1, sigmoid


def literal(value):
    result = format(float(np.float32(value)), '.9g')
    return result + ('.0' if '.' not in result and 'e' not in result else '') + 'f'


def main():
    data, _ = load_data()
    raw, _, records = windows(data, list(range(90)))
    objects = np.array([r['object'] for r in records], dtype='<i4')
    v3 = json.loads((OUT / 'all_choices_before_test.json').read_text())
    v2 = json.loads((ROOT / 'ct_lif_v2/results/all_choices_before_test.json').read_text())
    uses = []
    for fold in v3['folds']:
        for choice in fold['selected']:
            if choice['model'] in ['ct_v3', 'mean_lif', 'ct_rate', 'linear_control']:
                uses.append({'model': choice['model'], 'policy': choice['policy'], 'held': fold['held'], 'seed': fold['seed'],
                             'version': 3, 'checkpoint': str((OUT / (choice['candidate']['name'] + '.npz')).relative_to(ROOT)),
                             'threshold': choice['threshold']})
        for model, stem in [('ct_v1', 'ct_lif'), ('mean_lif28', 'mean_lif28')]:
            path = ROOT / 'ct_lif_study/results' / f'object{fold["held"]}_seed{fold["seed"]}_{stem}.npz'
            with np.load(path, allow_pickle=False) as z:
                threshold = float(z['threshold'])
            uses.append({'model': model, 'policy': 'original', 'held': fold['held'], 'seed': fold['seed'], 'version': 1,
                         'checkpoint': str(path.relative_to(ROOT)), 'threshold': threshold})
    for fold in v2['folds']:
        choice = next(c for c in fold['selected'] if c['model'] == 'ct_v2' and c['policy'] == 'balanced')
        uses.append({'model': 'ct_v2', 'policy': 'balanced', 'held': fold['held'], 'seed': fold['seed'], 'version': 2,
                     'checkpoint': 'ct_lif_v2/results/' + choice['candidate']['name'] + '.npz', 'threshold': choice['threshold']})
    unique = {use['checkpoint']: use for use in uses}
    header = ['#pragma once', '#include "runtime.hpp"', '#include "legacy_runtime.hpp"',
              'using Predictor=float(*)(const float*);', 'struct ModelSpec {const char* name; int held; Predictor predict;};']
    specs, references, dimensions = [], {}, {}
    for index, (name, use) in enumerate(unique.items()):
        mask = objects == use['held']
        with np.load(ROOT / name, allow_pickle=False) as z:
            variant, kind = str(z['variant']), str(z['kind'])
            if use['version'] == 3:
                pooling = str(z['pooling'])
                residual = bool(z['residual'])
                if kind == 'linear':
                    C, H = len(z['q']), 0
                    params = {k: z[k].copy() for k in ['q', 'c']}
                    logits = linear_forward(compress(raw_features(raw[mask], z['scale'], variant)), params, pooling)
                else:
                    C, H = z['W'].shape
                    model = ResidualLIF(C, H, residual, pooling)
                    model.params = {k: z[k].copy() for k in model.params}
                    params = model.params
                    logits = model.forward(compress(raw_features(raw[mask], z['scale'], variant)))
                T, raw_intervals = 32, 32
                variant_id = {'mean': 0, 'amplitude': 1, 'invariant': 2}[variant]
                expression = f'infer_v3<{C},{H},{str(residual).lower()},{str(pooling=="recent").lower()},{variant_id}>(raw,W{index},b{index},w{index},q{index},{literal(z["c"][0])},scale{index})'
                payload = 4 * (2 * H + C + (C if residual or kind == 'linear' else 0))
            else:
                C, H = z['W'].shape
                T = 32 if use['version'] == 1 else int(z['window'])
                model = TinyModel(C, H, kind)
                model.params = {k: z[k].copy() for k in model.params}
                params = model.params
                encoded = encode_v1(raw[mask], z['scale'], float(z['gate']), variant) if use['version'] == 1 else encode_v2(raw[mask], z['scale'], variant, T)
                logits = model.forward(encoded)
                variant_id = {'ct': 0, 'c_only': 1, 'mean': 2, 'amplitude': 3, 'shrink': 4}[variant]
                raw_intervals = T + (1 if variant == 'ct' else 0)
                expression = f'infer_window<{C},{H},{T},0,{variant_id}>(raw,W{index},b{index},w{index},{literal(z["c"][0])},scale{index})'
                payload = 4 * (2 * H + C + (57 if variant == 'ct' else 0))
            for key in ['W', 'b', 'w', 'q', 'scale']:
                values = z[key] if key in z.files else np.zeros(1)
                header.append(f'static const float {key}{index}[]={{' + ','.join(literal(v) for v in values.ravel()) + '};')
            header.append(f'static float predict{index}(const float* raw) {{return {expression};}}')
            specs.append(f'{{"{name}",{use["held"]},predict{index}}}')
            parameter_count = sum(v.size for v in params.values())
            references[name] = (np.flatnonzero(mask), logits)
            dimensions[name] = {'inputs': C, 'hidden': H, 'window': T, 'variant': variant,
                                'parameters': parameter_count, 'parameter_bytes_float32': 4 * parameter_count,
                                'dense_input_macs_per_window': T * C * H,
                                'input_history_bytes_float32': 4 * 54 * (raw_intervals + 1),
                                'state_feature_payload_bytes': payload, 'effective_raw_support_intervals': raw_intervals}
    header.append('static const ModelSpec models[]={' + ','.join(specs) + '};')
    (HERE / 'models_generated.hpp').write_text('\n'.join(header) + '\n')
    compiler = subprocess.run(['g++', '--version'], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        with (folder / 'input.bin').open('wb') as f:
            np.array([len(raw)], dtype='<i4').tofile(f)
            objects.tofile(f)
            raw.astype('<f4').tofile(f)
        build = subprocess.run(['g++', '-O3', '-std=c++17', '-ffp-contract=off', str(HERE / 'benchmark.cpp'), '-o', str(folder / 'bench')], capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError(build.stderr)
        completed = subprocess.run([str(folder / 'bench'), str(folder / 'input.bin'), str(OUT / 'cpp_predictions.csv'), str(OUT / 'host_timings.csv')], check=True, capture_output=True, text=True)
        print(completed.stdout)
    cpp = {name: [] for name in unique}
    with (OUT / 'cpp_predictions.csv').open() as f:
        for row in csv.DictReader(f):
            cpp[row['checkpoint']].append((int(row['index']), float(row['logit'])))
    parity = {}
    for name, (indices, logits) in references.items():
        observed = cpp[name]
        np.testing.assert_array_equal(indices, [i for i, _ in observed])
        values = np.array([v for _, v in observed])
        assert np.isfinite(values).all()
        parity[name] = {'max_absolute_logit_difference': float(np.max(np.abs(values - logits))), 'windows': len(logits)}
    timings = {}
    with (OUT / 'host_timings.csv').open() as f:
        for row in csv.DictReader(f):
            timings[row['checkpoint']] = {key: float(row[key]) for key in ['median_us', 'p95_us', 'samples']}
    table = []
    for use in uses:
        name = use['checkpoint']
        values = np.array([v for _, v in cpp[name]])
        changed = int(np.sum((sigmoid(references[name][1]) >= use['threshold']) != (sigmoid(values) >= use['threshold'])))
        table.append({**use, **dimensions[name], **timings[name], **parity[name], 'changed_decisions_float32': changed})
    write_csv(OUT / 'resources_by_fold.csv', table)
    aggregate = []
    for model, policy in sorted({(r['model'], r['policy']) for r in table}):
        selected = [r for r in table if (r['model'], r['policy']) == (model, policy)]
        aggregate.append({'model': model, 'policy': policy,
                          **{key: float(np.mean([r[key] for r in selected])) for key in ['parameters', 'parameter_bytes_float32', 'dense_input_macs_per_window', 'median_us', 'p95_us', 'input_history_bytes_float32', 'state_feature_payload_bytes']},
                          'minimum_parameters': min(r['parameters'] for r in selected), 'maximum_parameters': max(r['parameters'] for r in selected),
                          'changed_decisions_float32': sum(r['changed_decisions_float32'] for r in selected),
                          'max_absolute_logit_difference': max(r['max_absolute_logit_difference'] for r in selected)})
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), cpu)
    write_csv(OUT / 'resources.csv', aggregate)
    write_json(OUT / 'benchmark.json', {'cpu': cpu, 'platform': platform.platform(), 'compiler': compiler,
                                       'kernel': 'input-major contiguous hidden-neuron accumulation in both new and legacy C++ kernels; original summation order within each neuron preserved',
                                       'flags': '-O3 -std=c++17 -ffp-contract=off', 'distinct_checkpoints': len(unique),
                                       'timing': '5 shuffled model-order rounds; 32 warmups and1024 measured calls each round; encoding and reset-window logit inference, excluding acquisition, window assembly, IO, sigmoid, radio',
                                       'memory': 'calculated logical float32 payload, not measured peak RAM; excludes stack temporaries, code, acquisition and firmware',
                                       'not_target_device_or_energy_measurement': True, 'resources': aggregate})
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
