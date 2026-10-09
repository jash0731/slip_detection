"""Reproducible, bounded end-to-end surrogate-gradient comparison.

python3 -m advanced.train --epochs 18 --seeds 7 19
Public recorded classification only. No host/device timing or onset prediction.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch import nn

from .data import causal_derivatives, split_trials, prepare_fold
from .models import TemporalModel, LogisticControl
from .export import export_models


def quality(labels, decisions):
    labels = np.asarray(labels, dtype=int)
    decisions = np.asarray(decisions, dtype=int)
    tp = int(np.sum((labels == 1) & (decisions == 1)))
    tn = int(np.sum((labels == 0) & (decisions == 0)))
    fp = int(np.sum((labels == 0) & (decisions == 1)))
    fn = int(np.sum((labels == 1) & (decisions == 0)))
    return {'balanced_accuracy': (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2,
            'f1': 2 * tp / max(2 * tp + fp + fn, 1), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def select_threshold(labels, logits):
    candidates = np.linspace(-2.5, 2.5, 51)
    return float(max(candidates, key=lambda t: (quality(labels, logits >= t)['balanced_accuracy'], -abs(t))))


def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(inputs[start:start + 256]))
                          for start in range(0, len(inputs), 256)]).numpy()


def train_model(kind, training, validation, epochs, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = LogisticControl() if kind == 'logistic' else TemporalModel('lif' if kind == 'lif_frozen' else kind)
    if kind == 'lif_frozen':
        for parameter in model.input_layer.parameters():
            parameter.requires_grad_(False)
    original_input = None if kind == 'logistic' else model.input_layer.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-4)
    x, y, _ = training
    vx, vy, _ = validation
    positive_weight = float(np.sum(y == 0) / np.sum(y == 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    tensor_x, tensor_y = torch.from_numpy(x), torch.from_numpy(y)
    best_loss, best_state, best_epoch, stale = float('inf'), None, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(order), 256):
            indices = order[start:start + 256]
            optimizer.zero_grad()
            logits = model(tensor_x[indices])
            loss = criterion(logits, tensor_y[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        validation_logits = predict(model, vx)
        validation_loss = float(criterion(torch.from_numpy(validation_logits), torch.from_numpy(vy)))
        history.append({'epoch': epoch + 1, 'train_loss': total / len(x), 'validation_loss': validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(best_state)
    validation_logits = predict(model, vx)
    threshold = select_threshold(vy, validation_logits)
    changed = None if original_input is None else float((model.input_layer.weight - original_input).abs().max().detach())
    if changed is not None and changed == 0 and kind != 'lif_frozen':
        raise ValueError('Hidden input weights did not learn')
    return model, threshold, history, {'selected_epoch': best_epoch, 'validation_loss': best_loss,
                                      'positive_loss_weight': positive_weight,
                                      'max_input_weight_change': changed,
                                      'parameters': sum(p.numel() for p in model.parameters()),
                                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def trial_bootstrap(labels, a, b, groups, repetitions=1000, seed=42):
    """Paired difference in balanced accuracy, sampling whole trials only."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    differences = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if len(np.unique(labels[indices])) < 2:
            continue
        differences.append(quality(labels[indices], a[indices])['balanced_accuracy'] -
                           quality(labels[indices], b[indices])['balanced_accuracy'])
    return {'comparison': 'paired LIF minus ANN balanced accuracy', 'resampling_unit': 'whole held-out trial',
            'repetitions_requested': repetitions, 'repetitions_valid': len(differences), 'excluded_single_class': repetitions - len(differences), 'lower_95': float(np.percentile(differences, 2.5)) if differences else None,
            'upper_95': float(np.percentile(differences, 97.5)) if differences else None,
            'limitations': 'descriptive conditional on trained models and this object, not across-object population inference'}


def run(data_path, output, epochs, seeds, window=32):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    derivatives = causal_derivatives(data)
    output.mkdir(parents=True, exist_ok=True)
    predictions, summary, fold_metadata = [], [], []
    for held in range(len(data['object_names'])):
        for seed in seeds:
            name = f'object{held}_seed{seed}_window{window}'
            splits = split_trials(data, held, seed)
            prepared, scales = prepare_fold(data, derivatives, splits, window=window)
            models, thresholds, details, fold_decisions = {}, {}, {}, {}
            # Test inputs/labels are not evaluated until every training/validation
            # choice for this fold is frozen.
            for kind in ('logistic', 'ann', 'lif', 'alif'):
                model, threshold, history, info = train_model(kind, prepared['train'], prepared['validation'], epochs, seed)
                models[kind], thresholds[kind], details[kind] = model, threshold, info
                torch.save(model.state_dict(), output / f'{name}_{kind}.pt')
                (output / f'{name}_{kind}_history.json').write_text(json.dumps(history, indent=2) + '\n')
            test_x, test_y, records = prepared['test']
            for kind, model in models.items():
                logits = predict(model, test_x)
                decisions = (logits >= thresholds[kind]).astype(int)
                fold_decisions[kind] = decisions
                result = quality(test_y, decisions)
                summary.append({'source': 'public_recorded_host_trained', 'held_object': str(data['object_names'][held]),
                                'seed': seed, 'window': window, 'model': kind,
                                'test_windows': len(test_y), 'test_trials': len(set(r['group'] for r in records)), 'assigned_test_trials': len(splits['test']),
                                'decision_logit_threshold': thresholds[kind], **result})
                for record, label, logit, decision in zip(records, test_y, logits, decisions):
                    predictions.append({'source': 'public_recorded_host_trained', 'held_object': str(data['object_names'][held]),
                                        'seed': seed, 'window': window, 'model': kind, **record,
                                        'label': int(label), 'logit': float(logit), 'decision': int(decision)})
                print(f'{name} {kind}: BA={result["balanced_accuracy"]:.3f} F1={result["f1"]:.3f}', flush=True)
            groups = np.array([r['group'] for r in records])
            fold_metadata.append({'name': name, 'split_groups': splits, 'axis_scales_train_only': scales.tolist(),
                                  'class_counts': {split: {'0': int(np.sum(values[1] == 0)), '1': int(np.sum(values[1] == 1))}
                                                   for split, values in prepared.items()},
                                  'models': details, 'thresholds': thresholds,
                                  'paired_trial_bootstrap': trial_bootstrap(test_y, fold_decisions['lif'], fold_decisions['ann'], groups)})
            if held == 0 and seed == seeds[0]:
                fixture_index = int(np.flatnonzero(test_y == 1)[0])
                export_models(output.parent / 'export' / 'models.json', models, thresholds,
                              test_x[fixture_index], records[fixture_index], scales, name)
                print('Advanced models.json export ready', flush=True)
    for filename, rows in (('summary.csv', summary), ('predictions.csv', predictions)):
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {'evidence': 'public recorded gripper tactile classification, computer only',
                'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                                'platform': platform.platform(), 'threads': 2, 'deterministic_algorithms': True},
                'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
                'source_sha256': data['source_sha256'].tolist(), 'object_names': data['object_names'].tolist(),
                'protocol': {'outer_test': 'leave-one-object-out', 'inner_validation': '20% whole trials per remaining object',
                             'window': window, 'stride': 16, 'target': 'held-state label4/5 at finaltick only',
                             'context': 'all past samples, including ignored/nonheld and transitions',
                             'epoch_budget': epochs, 'seeds': seeds, 'early_stop': 'validation weighted loss patience4',
                             'decision_threshold': 'validation balanced accuracy grid -2.5..2.5 step0.1',
                             'optimization': 'Adam0.01 weightdecay0.0001 batch256 clipgradient5',
                             'surrogate': 'hard threshold forward; fast sigmoid derivative (1+5abs(v-threshold))^-2',
                             'model_capacity': 'LIF/ALIF/ANN each129 learned parameters; logistic7',
                             'state': 'reset each window; current-driven inputs, not binary input spikes'},
                'folds': fold_metadata, 'wall_seconds': time.perf_counter() - started,
                'limitations': ['Three source objects, one recorded sensor setup; not population generalization',
                                'Finaltick existing-slip labels, no independently measured incipient onset',
                                '32samples approximately178ms; inference speed is not observation delay',
                                'Original timestamps irregular; neuron decay per sample not time-corrected',
                                'No ESP32 measurements or energy claim',
                                'Group bootstrap conditional on trained models; two seeds not independent datasets']}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--data', type=Path, default=root / 'data' / 'arq_all_trials.npz')
    parser.add_argument('--output', type=Path, default=root / 'advanced' / 'results')
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19])
    parser.add_argument('--window', type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.window < 2:
        parser.error('epochs must be positive and window at least2')
    run(args.data, args.output, args.epochs, args.seeds, args.window)


if __name__ == '__main__':
    main()
