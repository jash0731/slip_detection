import argparse
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from ct_lif_study.run import ROOT, load_data, windows, verify_splits, sha, write_json, write_csv, detailed
from ct_lif_study.reference.ct_lif import sigmoid
from ct_lif_v2.run import fit_scale
from ct_lif_v2.inference import load_detector as load_v2
from .core import raw_features, compress, ResidualLIF, fit, fit_linear, linear_forward, validation_metrics, operating_points


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
FAMILIES = {'ct_amplitude': ('amplitude', True), 'ct_invariant': ('invariant', True),
            'mean_lif': ('mean', False), 'ct_rate': ('amplitude', False)}
POLICIES = ['balanced', 'joint']


def choose(candidates, policy):
    def key(candidate):
        point = candidate['operating_points'][policy]
        feasible = point['validation_joint_improvement'] if policy == 'joint' else True
        return (not feasible, -point['balanced_accuracy'], point['false_alarm_rate'], candidate['parameters'], candidate['name'])
    return min(candidates, key=key)


def source_hashes():
    paths = [HERE / 'core.py', HERE / 'run.py', ROOT / 'ct_lif_study/run.py',
             ROOT / 'ct_lif_study/reference/ct_lif.py', ROOT / 'ct_lif_v2/core.py',
             ROOT / 'ct_lif_v2/run.py', ROOT / 'ct_lif_v2/inference.py',
             ROOT / 'ct_lif_v2/results/all_choices_before_test.json', ROOT / 'data/arq_all_trials.npz']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def train():
    data, original = load_data()
    OUT.mkdir(exist_ok=False)
    protocol = {'revision': 'CT-LIF v3 residual evidence and recent pooling',
                'exploratory_reused_benchmark': True, 'families': FAMILIES,
                'neurons': [8, 12], 'rates': [.005, .01], 'pooling': ['uniform', 'recent'],
                'window': 32, 'leak': .875, 'neural_fits': 192, 'linear_fits': 96,
                'training': 'Adam; 32 max epochs; patience6; batch64; clip5; equal total weight per observed object/class cell',
                'augmentation': 'neural families share per-window loguniform gain[.8,1.25] before rational compression; linear controls have no augmentation',
                'linear': '9/6 pooled features, ridge logistic Newton solver; ridge[.0001,.001,.01,.1], uniform/recent; no spiking or hidden layer',
                'scale': 'training-only p95 mean absolute pin differences over unique raw rows',
                'recent_pool': 'half uniform plus half last8 uniform; final8 gets62.5% total weight',
                'policies': POLICIES, 'primary_policy': 'joint',
                'joint_rule': 'validation macro recall >= v2 balanced and FPR <= v2 balanced; maximize BA among feasible candidates, otherwise best BA and explicit infeasibility flag',
                'balanced_rule': 'maximize validation macro class-balanced accuracy; then lower FPR, parameter count, name',
                'composite_budget': 'ct_v3 chooses between16 candidates per fold; each neural control8; linear control16',
                'missing_validation_classes': 'undefined rates excluded from the corresponding class macro; disclosed, no per-object false-alarm guarantee',
                'source_hashes': source_hashes(), 'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()}}
    write_json(OUT / 'protocol.json', protocol)
    all_folds, count = [], 0
    started = time.perf_counter()
    for fold in original['folds']:
        held, seed, splits = verify_splits(data, fold)
        tr, va = windows(data, splits['train']), windows(data, splits['validation'])
        objects = np.array([r['object'] for r in tr[2]])
        val_objects = np.array([r['object'] for r in va[2]])
        v2 = load_v2(held, seed, 'balanced')
        reference = validation_metrics(va[1], v2.predict(va[0])['probability'], val_objects, v2.threshold)
        scale = fit_scale(data, splits['train'])
        encoded = {variant: (raw_features(tr[0], scale, variant), raw_features(va[0], scale, variant))
                   for variant in ['amplitude', 'invariant', 'mean']}
        candidates = []
        for family, (variant, residual) in FAMILIES.items():
            x, xv = encoded[variant]
            for pooling in ['uniform', 'recent']:
                for hidden in [8, 12]:
                    for rate in [.005, .01]:
                        name = f'object{held}_seed{seed}_{family}_{pooling}_h{hidden}_lr{rate}'
                        model = ResidualLIF(x.shape[-1], hidden, residual, pooling, seed)
                        history = fit(model, x, tr[1], objects, xv, va[1], val_objects, rate, seed)
                        probability = sigmoid(model.forward(compress(xv)))
                        info = {'name': name, 'family': family, 'kind': 'lif', 'variant': variant, 'pooling': pooling,
                                'hidden': hidden, 'inputs': x.shape[-1], 'residual': residual, 'rate': rate,
                                'parameters': sum(v.size for v in model.params.values()), 'history': history,
                                'operating_points': operating_points(va[1], probability, val_objects, reference)}
                        np.savez(OUT / (name + '.npz'), **model.params, kind='lif', variant=variant,
                                 pooling=pooling, residual=residual, scale=scale)
                        write_json(OUT / (name + '.json'), info)
                        candidates.append(info)
                        count += 1
            print(f'TRAIN {count}/288 object{held} seed{seed} {family}; test untouched', flush=True)
        for variant in ['amplitude', 'invariant']:
            x, xv = encoded[variant]
            for pooling in ['uniform', 'recent']:
                for ridge in [.0001, .001, .01, .1]:
                    name = f'object{held}_seed{seed}_linear_{variant}_{pooling}_ridge{ridge}'
                    params = fit_linear(x, tr[1], objects, pooling, ridge)
                    probability = sigmoid(linear_forward(compress(xv), params, pooling))
                    info = {'name': name, 'family': 'linear_control', 'kind': 'linear', 'variant': variant, 'pooling': pooling,
                            'hidden': 0, 'inputs': x.shape[-1], 'residual': False, 'ridge': ridge,
                            'parameters': sum(v.size for v in params.values()),
                            'operating_points': operating_points(va[1], probability, val_objects, reference)}
                    np.savez(OUT / (name + '.npz'), **params, kind='linear', variant=variant,
                             pooling=pooling, residual=False, scale=scale)
                    write_json(OUT / (name + '.json'), info)
                    candidates.append(info)
                    count += 1
        selected = []
        for policy in POLICIES:
            for family in [*FAMILIES, 'ct_v3', 'linear_control']:
                options = [c for c in candidates if c['family'] == family] if family != 'ct_v3' else [c for c in candidates if c['family'] in ['ct_amplitude', 'ct_invariant']]
                winner = choose(options, policy)
                point = winner['operating_points'][policy]
                selected.append({'model': family, 'policy': policy, 'candidate': winner, 'threshold': point['threshold'],
                                 'validation_joint_improvement': point['validation_joint_improvement']})
                if family == 'ct_v3':
                    print(f'CHOOSE {held}/{seed} {policy}: {winner["family"]} BA={point["balanced_accuracy"]:.4f} recall={point["slip_recall"]:.4f} FPR={point["false_alarm_rate"]:.4f} feasible={point["validation_joint_improvement"]}', flush=True)
        all_folds.append({'held': held, 'held_object': str(data['object_names'][held]), 'seed': seed,
                          'splits': splits, 'v2_balanced_validation_reference': reference, 'selected': selected})
        write_json(OUT / f'object{held}_seed{seed}_choices.json', all_folds[-1])
    checkpoint_hashes = {p.name: sha(p) for p in sorted(OUT.glob('*.npz'))}
    write_json(OUT / 'all_choices_before_test.json', {'folds': all_folds, 'fits': count,
                                                    'checkpoint_hashes': checkpoint_hashes, 'training_seconds': time.perf_counter() - started})
    print('ALL MODELS AND OPERATING POINTS FROZEN; no v3 test inference performed.', flush=True)


def checkpoint_score(path, raw):
    with np.load(path, allow_pickle=False) as z:
        x = compress(raw_features(raw, z['scale'], str(z['variant'])))
        if str(z['kind']) == 'linear':
            logits = linear_forward(x, {k: z[k] for k in ['q', 'c']}, str(z['pooling']))
        else:
            model = ResidualLIF(*z['W'].shape, residual=bool(z['residual']), pooling=str(z['pooling']))
            model.params = {k: z[k].copy() for k in model.params}
            logits = model.forward(x)
    return sigmoid(logits)


def evaluate():
    if (OUT / 'summary.csv').exists():
        raise ValueError('Evaluation already exists')
    data, _ = load_data()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for relative, digest in protocol['source_hashes'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Frozen source changed: ' + relative)
    frozen = json.loads((OUT / 'all_choices_before_test.json').read_text())
    for name, digest in frozen['checkpoint_hashes'].items():
        if sha(OUT / name) != digest:
            raise ValueError('Checkpoint changed: ' + name)
    summaries, predictions = [], []
    for fold in frozen['folds']:
        raw, y, records = windows(data, fold['splits']['test'])
        cache = {}
        for choice in fold['selected']:
            candidate = choice['candidate']
            path = OUT / (candidate['name'] + '.npz')
            if path not in cache:
                cache[path] = checkpoint_score(path, raw)
            probability = cache[path]
            base = {'model': choice['model'], 'policy': choice['policy'], 'held': fold['held'], 'held_object': fold['held_object'],
                    'seed': fold['seed'], 'checkpoint': str(path.relative_to(ROOT)), 'threshold': choice['threshold'],
                    'selected_family': candidate['family'], 'parameters': candidate['parameters']}
            row = {**base, **detailed(y, probability, choice['threshold'])}
            summaries.append(row)
            predictions.extend({**base, **record, 'label': int(label), 'probability': float(p), 'decision': int(p >= choice['threshold'])}
                               for record, label, p in zip(records, y, probability))
            if choice['model'] == 'ct_v3':
                print(f'TEST {fold["held_object"]}/{fold["seed"]} {choice["policy"]}: BA={row["balanced_accuracy"]:.4f} recall={row["slip_recall"]:.4f} FPR={row["false_alarm_rate"]:.4f}', flush=True)
    write_csv(OUT / 'summary.csv', summaries)
    write_csv(OUT / 'predictions.csv', predictions)
    write_json(OUT / 'complete.json', {'summary_rows': len(summaries), 'prediction_rows': len(predictions),
                                     'protocol_sha256': sha(OUT / 'protocol.json'), 'choices_sha256': sha(OUT / 'all_choices_before_test.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'evaluate'])
    arguments = parser.parse_args()
    train() if arguments.stage == 'train' else evaluate()
