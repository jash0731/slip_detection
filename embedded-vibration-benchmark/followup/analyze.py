"""Validate paired artifacts and estimate trial-level uncertainty, conditional on fits."""
import csv
import hashlib
import json
from collections import defaultdict

import numpy as np

from .common import ROOT, select_candidate
from .experiments import OUT, write_csv


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def paired_counts(rows, method_key, repetitions=None):
    """Keep seed/model/window pairings; average declared shuffle draws, not labels."""
    repetitions = repetitions or {}
    methods = sorted({method_key(row) for row in rows})
    seeds = sorted({int(row['seed']) for row in rows})
    objects = sorted({row['held_object'] for row in rows})
    groups = {obj: sorted({int(row['group']) for row in rows if row['held_object'] == obj}) for obj in objects}
    arrays = {obj: np.zeros((len(groups[obj]), len(seeds), len(methods), 4)) for obj in objects}
    locations = {obj: {group: index for index, group in enumerate(groups[obj])} for obj in objects}
    labels, seen, multiplicities, owners = {}, set(), defaultdict(int), {}
    draw_ids = defaultdict(set)
    for row in rows:
        obj, seed, group, end, method = row['held_object'], int(row['seed']), int(row['group']), int(row['end_sample']), method_key(row)
        label, decision = int(row['label']), int(row['decision'])
        if label not in (0, 1) or decision not in (0, 1):
            raise ValueError('Invalid binary prediction.')
        if owners.setdefault(group, obj) != obj:
            raise ValueError('Trial occurs under multiple held-out objects.')
        window_key = obj, group, end
        if labels.setdefault(window_key, label) != label:
            raise ValueError('Paired windows have different labels.')
        replicate = int(row.get('draw', 0))
        key = (*window_key, seed, method, replicate)
        if key in seen:
            raise ValueError('Duplicate paired prediction.')
        seen.add(key)
        multiplicities[(*window_key, seed, method)] += 1
        draw_ids[(*window_key, seed, method)].add(replicate)
        count = np.array([label == 1 and decision == 1, label == 0 and decision == 0,
                          label == 0 and decision == 1, label == 1 and decision == 0], dtype=float)
        arrays[obj][locations[obj][group], seeds.index(seed), methods.index(method)] += count / repetitions.get(method, 1)
    expected = {(*window, seed, method) for window in labels for seed in seeds for method in methods}
    if set(multiplicities) != expected or any(multiplicities[key] != repetitions.get(key[-1], 1) for key in expected):
        raise ValueError('Missing paired endpoints/seeds/methods or shuffle draws.')
    for key in expected:
        if draw_ids[key] != set(range(repetitions.get(key[-1], 1))):
            raise ValueError('Shuffle draw IDs do not match the declared fixed draws.')
    return arrays, methods, seeds, len(labels)


def balanced(counts):
    tp, tn, fp, fn = np.moveaxis(counts, -1, 0)
    if np.any(tp + fn == 0) or np.any(tn + fp == 0):
        raise ValueError('Single-class resample.')
    return .5 * (tp / (tp + fn) + tn / (tn + fp))


def intervals(arrays, methods, comparisons, draws=2000):
    point = np.mean([balanced(counts.sum(axis=0)).mean(axis=0) for counts in arrays.values()], axis=0)
    rng, retained, omitted = np.random.default_rng(20261007), [], 0
    for _ in range(draws):
        try:
            values = [balanced(counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)).mean(axis=0)
                      for obj, counts in sorted(arrays.items())]
        except ValueError:
            omitted += 1
            continue
        retained.append(np.mean(values, axis=0))
    if not retained:
        raise ValueError('No valid trial-level resamples.')
    matrix = np.array(retained)
    return {'means': {method: float(point[index]) for index, method in enumerate(methods)},
            'comparisons': [{'first': a, 'second': b, 'difference_percentage_points': float(100 * (point[methods.index(a)] - point[methods.index(b)])),
                             'conditional_95_percentage_points': (100 * np.quantile(matrix[:, methods.index(a)] - matrix[:, methods.index(b)], [.025, .975])).tolist()}
                            for a, b in comparisons],
            'replicates_requested': draws, 'replicates_retained': len(retained), 'excluded_single_class': omitted,
            'unit': 'whole grasp trial within object, paired across methods and the two fixed seeds',
            'limits': 'Conditional on observed objects/fitted models; excludes training/model-selection uncertainty; exploratory comparisons'}


def summarize():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    if metadata['status'] != 'complete':
        raise ValueError('Training/diagnostics did not complete.')
    summary = read_csv(OUT / 'robustness_summary.csv')
    if len(summary) != 120 or len({row['configuration'] for row in summary}) != 120:
        raise ValueError('Expected exactly108 candidate and12 shuffled-training fits.')
    for fold in metadata['folds']:
        for kind, chosen in fold['validation_choices'].items():
            candidates = [c for c in fold['candidates'] if c['kind'] == kind and c['condition'] == 'ordered']
            if select_candidate(candidates)['name'] != chosen:
                raise ValueError('Stored choice differs from prespecified validation selection.')
    selected_rows = [row for row in summary if row['validation_selected'] == '1']
    selected_names = {name for fold in metadata['folds'] for name in fold['validation_choices'].values()}
    if len(selected_rows) != 18 or {row['configuration'] for row in selected_rows} != selected_names:
        raise ValueError('Selected test rows differ from frozen validation choices.')
    write_csv(OUT / 'selected_error_rates.csv', selected_rows)
    selected = read_csv(OUT / 'selected_predictions.csv')
    if {row['configuration'] for row in selected} != selected_names or any(row['source'] != 'public_recorded_host_followup' for row in selected):
        raise ValueError('Wrong selected configuration or evidence source.')
    arrays, methods, seeds, windows = paired_counts(selected, lambda row: row['model'])
    chosen_stats = intervals(arrays, methods, [('lif', 'ann'), ('lif', 'leaky_relu')])
    chosen_stats.update(unique_windows=windows, seeds=seeds)
    for obj, counts in arrays.items():
        chosen_stats.setdefault('per_object', {})[obj] = {model: float(balanced(counts.sum(axis=0)).mean(axis=0)[index])
                                                        for index, model in enumerate(methods)}
    diagnostic = read_csv(OUT / 'temporal_diagnostics_predictions.csv')
    expected_draws = {f'{model}:{condition}': (3 if condition == 'shuffled_history' else 1)
                      for model in ('ann', 'lif', 'alif') for condition in ('ordered', 'reversed_history', 'shuffled_history')}
    da, dm, _, _ = paired_counts(diagnostic, lambda row: f'{row["model"]}:{row["condition"]}', expected_draws)
    temporal_stats = intervals(da, dm, [('lif:ordered', 'lif:shuffled_history'), ('lif:ordered', 'lif:reversed_history'),
                                      ('alif:ordered', 'alif:shuffled_history')])
    robust_predictions = read_csv(OUT / 'robustness_predictions.csv')
    matched = [row for row in robust_predictions if row['window'] == '32' and float(row['learning_rate']) == .01
               and row['model'] in ('lif', 'leaky_relu')]
    sa, sm, _, _ = paired_counts(matched, lambda row: f'{row["model"]}:{row["input_condition"]}')
    retrain_stats = intervals(sa, sm, [('lif:ordered', 'lif:shuffled_history'),
                                     ('leaky_relu:ordered', 'leaky_relu:shuffled_history'), ('lif:ordered', 'leaky_relu:ordered')])
    grid = defaultdict(list)
    for row in summary:
        if row['input_condition'] == 'ordered':
            grid[row['model'], int(row['window']), float(row['learning_rate'])].append(float(row['balanced_accuracy']))
    grid_rows = [{'source': 'public_recorded_host_followup', 'model': model, 'window': window, 'learning_rate': rate,
                  'macro_object_seed_balanced_accuracy': float(np.mean(values)), 'fixed_fold_seed_runs': len(values)}
                 for (model, window, rate), values in sorted(grid.items())]
    write_csv(OUT / 'macro_grid.csv', grid_rows)
    diagnostics_summary = read_csv(OUT / 'temporal_diagnostics_summary.csv')
    changes = {}
    for kind in ('ann', 'lif', 'alif'):
        for condition in ('reversed_history', 'shuffled_history'):
            rows = [r for r in diagnostics_summary if r['model'] == kind and r['condition'] == condition]
            changes[f'{kind}:{condition}'] = dict(changed_decisions=sum(int(r['changed_decisions']) for r in rows),
                                                 repeated_evaluations=sum(int(r['test_windows']) for r in rows),
                                                 max_absolute_logit_change=max(float(r['max_absolute_logit_change']) for r in rows))
    result = {'source': 'public_recorded_host_followup', 'selected_models': chosen_stats,
              'frozen_order_diagnostics': temporal_stats, 'matched_shuffled_retraining': retrain_stats,
              'frozen_decision_changes': changes, 'fixed_grid': grid_rows,
              'selection_warning': 'Fixed-grid test maximum is sensitivity evidence, never model selection or a new deployment export',
              'threshold_grid_boundary_fits': sum(abs(float(row['threshold'])) >= 2.49 for row in summary),
              'artifacts_sha256': {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                                  for name in ('selected_predictions.csv', 'robustness_predictions.csv', 'temporal_diagnostics_predictions.csv')}}
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('selected_models', 'frozen_order_diagnostics', 'matched_shuffled_retraining')}, indent=2))
    return result


if __name__ == '__main__':
    summarize()
