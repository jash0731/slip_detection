import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ct_lif_study.reference.ct_lif import sigmoid
from ct_lif_study.run import ROOT, detailed, load_data, verify_splits, windows
from ct_lif_v3.core import balanced_weights, operating_points, validation_metrics
from ct_lif_v4.core import features, load_model


def summary_inputs(model, x):
    return np.column_stack([
        model.forward(x), x.mean(axis=1), x.std(axis=1), x[:, -8:].mean(axis=1)
    ])


def predict(bundle, raw):
    x = features(raw, bundle['scale'], 'ct_only')
    if bundle['tree'] is None:
        scores = sigmoid(bundle['backbone'].forward(x))
    else:
        scores = bundle['tree'].predict_proba(summary_inputs(bundle['backbone'], x))[:, 1]
    return scores, scores >= bundle['threshold']


def selection_key(entry):
    point = entry['point']
    return (-point['balanced_accuracy'], point['false_alarm_rate'], entry['nodes'], entry['name'])


def train(output=None):
    if output is not None:
        output = Path(output)
        if output.exists():
            raise FileExistsError('Choose a new output directory to preserve existing results')
    data, metadata = load_data()
    results = ROOT / 'ct_lif_v4/results'
    saved = json.loads((results / 'all_choices_before_test.json').read_text())['folds']
    frozen, choices = [], []
    count = 0
    for fold in metadata['folds']:
        held, seed, splits = verify_splits(data, fold)
        old = next(f for f in saved if f['held'] == held and f['seed'] == seed)
        original = next(c for c in old['selected'] if c['variant'] == 'ct_only')
        backbone, scale, _ = load_model(results / (original['name'] + '.npz'))
        tr, va = windows(data, splits['train']), windows(data, splits['validation'])
        objects = np.array([r['object'] for r in tr[2]])
        val_objects = np.array([r['object'] for r in va[2]])
        z = summary_inputs(backbone, features(tr[0], scale, 'ct_only'))
        zv = summary_inputs(backbone, features(va[0], scale, 'ct_only'))
        ref = validation_metrics(va[1], sigmoid(zv[:, 0]), val_objects, original['threshold'])
        base = dict(name='baseline', tree=None, point=ref, nodes=0, depth_sum=0)
        joint, balanced, candidates = [base], [base], []
        for leaves in [3, 7]:
            for iterations in [40, 80]:
                for rate in [.05, .1]:
                    tree = HistGradientBoostingClassifier(
                        max_leaf_nodes=leaves, max_iter=iterations, learning_rate=rate,
                        l2_regularization=1., min_samples_leaf=20,
                        early_stopping=False, random_state=seed)
                    tree.fit(z, tr[1], sample_weight=balanced_weights(tr[1], objects))
                    points = operating_points(va[1], tree.predict_proba(zv)[:, 1], val_objects, ref)
                    entry = dict(
                        name=f'leaves{leaves}_iterations{iterations}_rate{rate}', tree=tree,
                        nodes=sum(len(t[0].nodes) for t in tree._predictors),
                        depth_sum=sum(int(t[0].nodes['depth'].max()) for t in tree._predictors))
                    candidate = dict(entry, point=points['balanced'])
                    candidates.append(candidate)
                    balanced.append(candidate)
                    if points['joint']['validation_joint_improvement']:
                        joint.append(dict(entry, point=points['joint']))
                    count += 1
        selected = dict(baseline=base, joint=min(joint, key=selection_key),
                        balanced=min(balanced, key=selection_key),
                        nonlinear=min(candidates, key=selection_key))
        frozen.append((held, seed, splits, scale, backbone, selected))
        choices.append(dict(held=held, seed=seed, selections={
            method: {k: v for k, v in entry.items() if k != 'tree'}
            for method, entry in selected.items()}))
        print('FIT', count, '/48', held, seed,
              {k: v['name'] for k, v in selected.items()}, flush=True)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'choices_before_test.json').write_text(json.dumps(choices, indent=2) + '\n')
    print('ALL CHOICES FROZEN BEFORE TEST INFERENCE', flush=True)
    rows, summaries = {}, {}
    for held, seed, splits, scale, backbone, selected in frozen:
        raw, y, _ = windows(data, splits['test'])
        for method, entry in selected.items():
            bundle = dict(backbone=backbone, scale=scale, tree=entry['tree'],
                          threshold=entry['point']['threshold'], held=held, seed=seed)
            scores, _ = predict(bundle, raw)
            row = detailed(y, scores, bundle['threshold'])
            row.update(held=held, seed=seed, name=entry['name'], tree_nodes=entry['nodes'],
                       tree_comparison_bound=entry['depth_sum'])
            rows.setdefault(method, []).append(row)
            if output is not None and method in ['joint', 'balanced']:
                joblib.dump(bundle, output / f'object{held}_seed{seed}_{method}.joblib')
    for method, entries in rows.items():
        result = {k: float(np.mean([r[k] for r in entries])) for k in [
            'balanced_accuracy', 'slip_recall', 'false_alarm_rate',
            'tree_nodes', 'tree_comparison_bound']}
        result['accuracy'] = (sum(r['tp'] + r['tn'] for r in entries) /
                              sum(r['tp'] + r['tn'] + r['fp'] + r['fn'] for r in entries))
        summaries[method] = result
        print('RESULT', method, json.dumps(result), flush=True)
    report = dict(fits=count, summaries=summaries, folds=rows,
                  exploratory_reused_benchmark=True)
    if output is not None:
        (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='Optional new directory for results and local model bundles')
    train(parser.parse_args().output)
