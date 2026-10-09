"""Post-hoc error audit of frozen recorded-host predictions, not physical onset."""
import csv
import hashlib
import json
from collections import defaultdict

import numpy as np

from .common import ROOT, read_data, detailed_quality
from .experiments import write_csv

OUT = ROOT / 'followup/failures'
SOURCE = 'public_recorded_host_error_audit'


def label_context(labels, end, window=32):
    context = labels[end - window + 1:end + 1]
    target = int(labels[end])
    if len(context) != window or target not in (0, 1):
        raise ValueError('Expected a full eligible endpoint window.')
    category = ('ignored_context' if np.any(context < 0) else
                'stable_non_slip' if np.all(context == 0) else
                'stable_slip' if np.all(context == 1) else 'mixed_held_states')
    beginning = end
    while beginning > 0 and labels[beginning - 1] == target:
        beginning -= 1
    age = end - beginning
    age_band = '0_to_15_samples' if age < 16 else '16_to_31_samples' if age < 32 else '32_plus_samples'
    return category, age, age_band, float(np.mean(context == target))


def make_metric(rows, base):
    labels = np.array([r['label'] for r in rows])
    decisions = np.array([r['decision'] for r in rows])
    return {**base, 'windows': len(rows), 'eligible_trials': len({r['group'] for r in rows}),
            'non_slip_windows': int(np.sum(labels == 0)), 'slip_windows': int(np.sum(labels == 1)),
            **detailed_quality(labels, decisions)}


def contiguous_error_runs(rows):
    """Only consecutive scored endpoints, not bridging ignored labels or gaps."""
    runs, current = [], []
    for row in sorted(rows, key=lambda r: r['end_sample']):
        error = row['decision'] != row['label']
        if not error:
            if current:
                runs.append(current)
                current = []
            continue
        if current and (row['end_sample'] != current[-1]['end_sample'] + 16 or row['label'] != current[-1]['label']):
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)
    return runs


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    data, derivatives, original = read_data()
    path = ROOT / 'advanced/results/predictions.csv'
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    group_rows = {int(group): np.flatnonzero(data['group'] == group) for group in np.unique(data['group'])}
    with path.open() as stream:
        raw = list(csv.DictReader(stream))
    records, seen = [], set()
    for row in raw:
        record = dict(row)
        for key in ('seed', 'group', 'object', 'end_sample', 'label', 'decision'):
            record[key] = int(row[key])
        record['logit'] = float(row['logit'])
        record['t_ms'] = float(row['t_ms'])
        key = record['seed'], record['model'], record['group'], record['end_sample']
        if key in seen:
            raise ValueError('Duplicate saved window prediction.')
        seen.add(key)
        rows = group_rows[record['group']]
        labels = data['y_held'][rows]
        if labels[record['end_sample']] != record['label']:
            raise ValueError('Saved label differs from original endpoint.')
        category, age, band, fraction = label_context(labels, record['end_sample'])
        record.update(context=category, label_run_age_samples=age, label_run_age_band=band,
                      current_state_fraction=fraction, source=SOURCE)
        records.append(record)
    by_fold, by_trial, by_context, by_age = (defaultdict(list) for _ in range(4))
    for row in records:
        fold = row['held_object'], row['seed'], row['model']
        by_fold[fold].append(row)
        by_trial[(*fold, row['group'])].append(row)
        by_context[(*fold, row['context'])].append(row)
        by_age[(*fold, row['label_run_age_band'])].append(row)
    tables = []
    for filename, groups, extra in (
        ('per_object.csv', by_fold, None), ('per_trial.csv', by_trial, 'group'),
        ('by_context.csv', by_context, 'context'), ('by_label_age.csv', by_age, 'label_run_age_band')):
        table = []
        for key, rows in sorted(groups.items()):
            base = dict(source=SOURCE, held_object=key[0], seed=key[1], model=key[2])
            if extra:
                base[extra] = key[3]
            if extra == 'group':
                base['trial'] = rows[0]['trial']
            table.append(make_metric(rows, base))
        write_csv(OUT / filename, table)
        tables.append(table)
    errors, examples = [], []
    thresholds = {(int(f['name'].split('_')[0][6:]), int(f['name'].split('_')[1][4:]), kind): threshold
                  for f in original['folds'] for kind, threshold in f['thresholds'].items()}
    frozen_meta = json.loads((ROOT / 'advanced/results/frozen_ablation.json').read_text())
    thresholds.update({(int(f['name'].split('_')[0][6:]), int(f['name'].split('_')[1][4:]), 'lif_frozen'): f['threshold']
                       for f in frozen_meta['folds']})
    for key, rows in sorted(by_trial.items()):
        for block in contiguous_error_runs(rows):
            errors.append(dict(source=SOURCE, held_object=key[0], seed=key[1], model=key[2],
                               group=key[3], trial=block[0]['trial'],
                               error_type='miss' if block[0]['label'] else 'false_alarm',
                               consecutive_endpoints=len(block), start_end_sample=block[0]['end_sample'],
                               last_end_sample=block[-1]['end_sample'],
                               endpoint_span_ms=block[-1]['t_ms'] - block[0]['t_ms']))
    for key, rows in sorted(by_fold.items()):
        threshold = thresholds[rows[0]['object'], key[1], key[2]]
        incorrect = [r for r in rows if r['label'] != r['decision']]
        # Fixed illustrative rule: largest wrong margin of each error type in each fold.
        for label, error_type in ((0, 'false_alarm'), (1, 'miss')):
            candidates = [r for r in incorrect if r['label'] == label]
            if not candidates:
                continue
            chosen = max(candidates, key=lambda r: (abs(r['logit'] - threshold), -r['group'], -r['end_sample']))
            examples.append({**chosen, 'error_type': error_type, 'threshold': threshold,
                             'absolute_wrong_margin': abs(chosen['logit'] - threshold)})
    write_csv(OUT / 'consecutive_errors.csv', errors)
    write_csv(OUT / 'illustrative_errors.csv', examples)
    shift = []
    for fold in original['folds']:
        held = int(fold['name'].split('_')[0][6:])
        seed = int(fold['name'].split('_')[1][4:])
        scales = np.array(fold['axis_scales_train_only'])
        for split in ('train', 'test'):
            mask = np.isin(data['group'], fold['split_groups'][split])
            values = derivatives[mask]
            ratio = np.abs(values) / scales
            for axis, axis_name in enumerate(('x', 'y', 'z')):
                shift.append(dict(source=SOURCE, held_object=str(data['object_names'][held]), seed=seed,
                                  partition=split, axis=axis_name, raw_samples=len(values),
                                  absolute_derivative_p95=float(np.percentile(np.abs(values[:, axis]), 95)),
                                  train_axis_scale=float(scales[axis]),
                                  derivative_p95_over_train_scale=float(np.percentile(ratio[:, axis], 95)),
                                  fraction_beyond_input_clip=float(np.mean(ratio[:, axis] > 4))))
    write_csv(OUT / 'feature_shift.csv', shift)
    metadata = {'source': SOURCE, 'original_predictions_sha256': original_hash,
                'source_data_sha256': original['data_sha256'], 'prediction_rows': len(records),
                'unique_windows': len({(r['group'], r['end_sample']) for r in records}),
                'seeds': [7, 19], 'models': sorted({r['model'] for r in records}),
                'limits': ['Descriptive subgroups defined after first results, not hypothesis-confirming independent tests',
                           'Label-run age is annotation history, not physical slip onset or detector latency',
                           'Overlapping windows and repeated seeds are dependent; no subgroup significance test',
                           'Single-class subgroup balanced accuracy is undefined and stored blank',
                           'Error-run span measures scored endpoints and may be shorter than full error duration',
                           'Illustrative errors selected by largest wrong margin are not typical-case accuracy',
                           'Feature shift uses all original train/test rows including ignored context; not a causal explanation']}
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if hashlib.sha256(path.read_bytes()).hexdigest() != original_hash:
        raise ValueError('Original predictions changed during audit.')
    print(f'COMPLETE error audit: {len(records)} prediction rows, {metadata["unique_windows"]} unique windows; source remains frozen.')


if __name__ == '__main__':
    run()
