import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from advanced.data import split_trials
from .reference.ct_lif import TinyModel, encode, fit_scale, fit, sigmoid, metrics


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
METHODS = {
    'ct_lif': ('ct', 'lif', 16),
    'ct_ann': ('ct', 'ann', 16),
    'ct_temporal': ('ct', 'temporal', 16),
    'mean_lif16': ('mean', 'lif', 16),
    'mean_lif28': ('mean', 'lif', 28),
    'c_only_lif': ('c_only', 'lif', 16),
    'turnover_lif': ('turnover', 'lif', 16),
    'entropy_lif': ('entropy', 'lif', 16),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_data():
    metadata = json.loads((ROOT / 'advanced/results/metadata.json').read_text())
    manifest = json.loads((ROOT / 'data/arq_all_manifest.json').read_text())
    data_path = ROOT / 'data/arq_all_trials.npz'
    if sha(data_path) != metadata['data_sha256'] or sha(data_path) != manifest['prepared']['sha256']:
        raise ValueError('Prepared dataset checksum mismatch')
    with np.load(data_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if data['x'].shape != (228521, 54) or not np.isfinite(data['x']).all():
        raise ValueError('Unexpected prepared raw data')
    expected = np.where(data['original_label'] == 4, 0, np.where(data['original_label'] == 5, 1, -1))
    np.testing.assert_array_equal(expected, data['y_held'])
    return data, metadata


def windows(data, groups):
    xs, ys, records = [], [], []
    for group in groups:
        rows = np.flatnonzero(data['group'] == group)
        if not np.all(np.diff(data['t_ms'][rows]) > 0):
            raise ValueError('Invalid trial timestamps')
        for end in range(63, len(rows), 16):
            target = int(data['y_held'][rows[end]])
            if target < 0:
                continue
            indices = rows[end - 33:end + 1]
            xs.append(data['x'][indices].reshape(34, 18, 3))
            ys.append(target)
            records.append({'group': int(group), 'trial': str(data['trial_names'][group]),
                            'object': int(data['object'][rows[end]]), 'end_sample': end,
                            't_ms': float(data['t_ms'][rows[end]])})
    if len(set(ys)) != 2:
        raise ValueError('Both classes required in partition')
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), records


def verify_splits(data, fold):
    held = int(fold['name'].split('_')[0][6:])
    seed = int(fold['name'].split('_')[1][4:])
    splits = fold['split_groups']
    if splits != split_trials(data, held, seed):
        raise ValueError('Frozen split differs from the original splitter')
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if set(splits[a]) & set(splits[b]):
            raise ValueError('Trial overlap')
    if set(np.unique(data['object'][np.isin(data['group'], splits['test'])])) != {held}:
        raise ValueError('Incorrect held-out object')
    for part in ['train', 'validation']:
        if held in data['object'][np.isin(data['group'], splits[part])]:
            raise ValueError('Held-out object in development')
    return held, seed, splits


def protocol():
    return {'status': 'frozen_before_new_training_and_test_inference',
            'candidate': 'unchanged CT-LIF v0.1 NumPy reference',
            'primary': 'ct_lif fixed gate .02 and learning rate .005',
            'methods': METHODS, 'window': 32, 'raw_samples': 34,
            'endpoint_grid': 'trial index 63,79,95,... with final label4/5',
            'seeds': [7, 19], 'folds': 3, 'fits': 48,
            'training': 'reference fit: Adam .005; batch64; max18 epochs; patience4; weighted BCE; clip5',
            'threshold': 'reference validation BA selection on probability .05:.01:.95',
            'scale': 'training raw windows only; p95 mean absolute pin derivative, including overlap repetition',
            'data_sha256': sha(ROOT / 'data/arq_all_trials.npz'),
            'reference_sha256': sha(HERE / 'reference/ct_lif.py'),
            'runner_sha256': sha(Path(__file__)),
            'original_metadata_sha256': sha(ROOT / 'advanced/results/metadata.json'),
            'metrics': 'object-macro then seed-mean BA, accuracy, recall, FPR and positive F1',
            'bootstrap': '2000 whole-trial draws within each object, shared across seeds and methods, seed20261008',
            'evidence': 'exploratory recorded-data classification; historical failures motivated the candidate',
            'no_hyperparameter_search': True,
            'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()}}


def train_all():
    data, original = load_data()
    OUT.mkdir(exist_ok=False)
    write_json(OUT / 'protocol.json', protocol())
    all_choices = []
    started = time.perf_counter()
    for fold in original['folds']:
        held, seed, splits = verify_splits(data, fold)
        tr = windows(data, splits['train'])
        va = windows(data, splits['validation'])
        scale = fit_scale(tr[0])
        encoded = {variant: (encode(tr[0], scale, .02, variant), encode(va[0], scale, .02, variant))
                   for variant in sorted({spec[0] for spec in METHODS.values()})}
        for name, (variant, kind, hidden) in METHODS.items():
            tag = f'object{held}_seed{seed}_{name}'
            x, xv = encoded[variant]
            model = TinyModel(x.shape[-1], hidden, kind, seed)
            before = time.perf_counter()
            threshold, history = fit(model, x, tr[1], xv, va[1], lr=.005, seed=seed)
            probs = sigmoid(model.forward(xv))
            choice = {'tag': tag, 'held_object': str(data['object_names'][held]), 'held': held, 'seed': seed,
                      'model': name, 'variant': variant, 'kind': kind, 'hidden': hidden,
                      'threshold': threshold, 'gate': .02, 'rate': .005,
                      'parameters': int(sum(p.size for p in model.params.values())),
                      'scale': scale.tolist(), 'split_groups': splits,
                      'train_windows': len(tr[1]), 'validation_windows': len(va[1]),
                      'history': history, 'validation': metrics(va[1], probs, threshold),
                      'training_seconds': time.perf_counter() - before}
            np.savez(OUT / f'{tag}.npz', **model.params, scale=scale, threshold=threshold,
                     kind=kind, variant=variant, gate=.02, beta=model.beta)
            write_json(OUT / f'{tag}_selection.json', choice)
            all_choices.append(choice)
            print(f'TRAIN {len(all_choices)}/48 {tag}: validation_BA={choice["validation"]["balanced_accuracy"]:.4f}; epochs={len(history)}; seconds={choice["training_seconds"]:.2f}', flush=True)
        del tr, va, encoded
    write_json(OUT / 'all_choices_before_test.json', {'choices': all_choices, 'training_seconds': time.perf_counter() - started})
    print('TRAIN COMPLETE; every model and threshold frozen; test inference not yet run', flush=True)


def detailed(y, probability, threshold):
    result = metrics(y, probability, threshold)
    tp, tn, fp, fn = (result[k] for k in ['tp', 'tn', 'fp', 'fn'])
    result.update(accuracy=(tp + tn) / len(y), f1=2*tp / max(2*tp + fp + fn, 1),
                  precision=tp / (tp+fp) if tp+fp else None)
    return result


def evaluate_all():
    if (OUT / 'summary.csv').exists():
        raise ValueError('Evaluation already exists; inspect saved results')
    data, _ = load_data()
    frozen = json.loads((OUT / 'all_choices_before_test.json').read_text())
    checks = json.loads((OUT / 'protocol.json').read_text())
    if checks['reference_sha256'] != sha(HERE / 'reference/ct_lif.py') or checks['runner_sha256'] != sha(Path(__file__)):
        raise ValueError('Code changed after protocol freeze')
    summaries, predictions = [], []
    cache = {}
    for choice in frozen['choices']:
        held = choice['held']
        if held not in cache:
            cache[held] = windows(data, choice['split_groups']['test'])
        raw, y, records = cache[held]
        with np.load(OUT / (choice['tag'] + '.npz'), allow_pickle=False) as z:
            model = TinyModel(*z['W'].shape, kind=str(z['kind']))
            model.params = {k: z[k].copy() for k in model.params}
            model.beta = float(z['beta'])
            x = encode(raw, z['scale'], float(z['gate']), str(z['variant']))
        probability = sigmoid(model.forward(x))
        base = {k: choice[k] for k in ['tag', 'held_object', 'held', 'seed', 'model', 'threshold', 'parameters']}
        result = {**base, 'test_windows': len(y), 'test_trials': len({r['group'] for r in records}),
                  **detailed(y, probability, choice['threshold'])}
        summaries.append(result)
        predictions.extend({**base, **r, 'label': int(label), 'probability': float(p), 'decision': int(p >= choice['threshold'])}
                           for r, label, p in zip(records, y, probability))
        print(f'TEST {choice["tag"]}: BA={result["balanced_accuracy"]:.4f}; recall={result["slip_recall"]:.4f}; FPR={result["false_alarm_rate"]:.4f}', flush=True)
    write_csv(OUT / 'summary.csv', summaries)
    write_csv(OUT / 'predictions.csv', predictions)
    write_json(OUT / 'complete.json', {'fits': len(summaries), 'prediction_rows': len(predictions),
                                     'unique_endpoints': len({(p['group'], p['end_sample']) for p in predictions}),
                                     'protocol_sha256': sha(OUT / 'protocol.json'),
                                     'selections_sha256': sha(OUT / 'all_choices_before_test.json')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'evaluate'])
    args = parser.parse_args()
    train_all() if args.stage == 'train' else evaluate_all()


if __name__ == '__main__':
    main()
