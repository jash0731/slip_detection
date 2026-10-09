"""Append a same-architecture fixed-hidden/readout-only control to trained results.

Uses identical initial hidden weights, input representation, group splits, epoch
budget and validation rules as LIF. Freeze hidden weights AND biases; only train
the output readout. Recomputes features instead of changing the optimization budget.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import causal_derivatives, split_trials, prepare_fold
from .train import train_model, predict, quality


def run(data_path, output):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    metadata = json.loads((output / 'metadata.json').read_text())
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    derivatives = causal_derivatives(data)
    additions, summary, details = [], [], []
    epochs = metadata['protocol']['epoch_budget']
    window = metadata['protocol']['window']
    for held in range(len(data['object_names'])):
        for seed in metadata['protocol']['seeds']:
            splits = split_trials(data, held, seed)
            prepared, _ = prepare_fold(data, derivatives, splits, window=window)
            model, threshold, history, info = train_model('lif_frozen', prepared['train'], prepared['validation'], epochs, seed)
            if info['max_input_weight_change'] != 0:
                raise ValueError('Frozen hidden weights changed')
            x, y, records = prepared['test']
            logits = predict(model, x)
            decisions = (logits >= threshold).astype(int)
            result = quality(y, decisions)
            summary.append({'source': 'public_recorded_host_trained', 'held_object': str(data['object_names'][held]),
                            'seed': seed, 'window': window, 'model': 'lif_frozen', 'test_windows': len(y),
                            'test_trials': len(set(r['group'] for r in records)), 'assigned_test_trials': len(splits['test']),
                            'decision_logit_threshold': threshold, **result})
            for record, label, logit, decision in zip(records, y, logits, decisions):
                additions.append({'source': 'public_recorded_host_trained', 'held_object': str(data['object_names'][held]),
                                  'seed': seed, 'window': window, 'model': 'lif_frozen', **record,
                                  'label': int(label), 'logit': float(logit), 'decision': int(decision)})
            name = f'object{held}_seed{seed}_window{window}_lif_frozen'
            torch.save(model.state_dict(), output / f'{name}.pt')
            (output / f'{name}_history.json').write_text(json.dumps(history, indent=2) + '\n')
            details.append({'name': name, 'split_groups': splits, 'threshold': threshold, **info})
            print(f'{name}: BA={result["balanced_accuracy"]:.3f} F1={result["f1"]:.3f}', flush=True)
    for filename, rows in [('predictions.csv', additions), ('summary.csv', summary)]:
        path = output / filename
        with path.open(newline='') as stream:
            existing = [row for row in csv.DictReader(stream) if row['model'] != 'lif_frozen']
        fieldnames = list(rows[0])
        # Correct v2 assigned-vs-scored trial counts if this run follows an earlier
        # training process whose code predated that metadata clarification.
        if filename == 'summary.csv':
            for row in existing:
                row['assigned_test_trials'] = 30
                groups = {r['group'] for r in additions if r['held_object'] == row['held_object'] and r['seed'] == int(row['seed'])}
                row['test_trials'] = len(groups)
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing + rows)
    (output / 'frozen_ablation.json').write_text(json.dumps({'protocol': 'same initialized LIF hidden weights and biases frozen; only17 readout parameters trainable; same budget/splits', 'folds': details}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--data', type=Path, default=root / 'data' / 'arq_all_trials.npz')
    parser.add_argument('--output', type=Path, default=root / 'advanced' / 'results')
    args = parser.parse_args()
    run(args.data, args.output)
