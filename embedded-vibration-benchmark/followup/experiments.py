"""Run the fixed temporal/robustness protocol, with validation-only selection."""
import csv
import hashlib
import json
import platform
import sys
import time

import numpy as np
import torch

from advanced.data import prepare_fold, split_trials
from advanced.train import predict
from .common import (ROOT, MODELS, WINDOWS, RATES, SEEDS, read_data, prepare_aligned,
                     shuffled_partitions, train_candidate, select_candidate,
                     perturb, detailed_quality, make_model)

OUT = ROOT / 'followup/results'
SOURCE = 'public_recorded_host_followup'


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty artifact: {path}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity(held, seed, window, rate, kind, condition):
    rate_name = str(rate).replace('.', 'p')
    return f'object{held}_seed{seed}_w{window}_lr{rate_name}_{kind}_{condition}'


def evaluate(candidate, test, base):
    x, y, records = test
    logits = predict(candidate['model'], x)
    decisions = (logits >= candidate['threshold']).astype(int)
    info = {**base, 'configuration': candidate['name'], 'model': candidate['kind'],
            'window': candidate['window'], 'learning_rate': candidate['rate'],
            'input_condition': candidate['condition'], 'validation_selected': int(candidate.get('selected', False)),
            'threshold': candidate['threshold']}
    summary = {**info, 'test_windows': len(y), 'test_trials': len({r['group'] for r in records}),
               'validation_balanced_accuracy': candidate['info']['validation_balanced_accuracy'],
               'selected_epoch': candidate['info']['selected_epoch'],
               'epochs_run': candidate['info']['epochs_run'], **detailed_quality(y, decisions)}
    rows = [{**info, **record, 'label': int(label), 'logit': float(logit), 'decision': int(decision)}
            for record, label, logit, decision in zip(records, y, logits, decisions)]
    return summary, rows


def run_grid(data, derivatives, original):
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    predictions, summary, metadata = [], [], []
    for held in range(3):
        for seed in SEEDS:
            splits = split_trials(data, held, seed)
            frozen_fold = next(f for f in original['folds'] if f['name'] == f'object{held}_seed{seed}_window32')
            if splits != frozen_fold['split_groups']:
                raise ValueError('Trial split differs from the frozen study.')
            prepared = {window: prepare_aligned(data, derivatives, splits, window)[0] for window in WINDOWS}
            for partition in ('train', 'validation', 'test'):
                reference = [(r['group'], r['end_sample'], int(y))
                             for r, y in zip(prepared[32][partition][2], prepared[32][partition][1])]
                for window in WINDOWS:
                    current = [(r['group'], r['end_sample'], int(y))
                               for r, y in zip(prepared[window][partition][2], prepared[window][partition][1])]
                    if current != reference:
                        raise ValueError('Observation lengths changed endpoint labels or schedules.')
            shuffled = shuffled_partitions(prepared[32], seed)
            candidates, choices = [], {}
            # Freeze all validation-based choices in this fold before test inference.
            for window in WINDOWS:
                for rate in RATES:
                    for kind in MODELS:
                        name = identity(held, seed, window, rate, kind, 'ordered')
                        model, threshold, history, info = train_candidate(
                            kind, prepared[window]['train'], prepared[window]['validation'], rate, seed)
                        torch.save(model.state_dict(), OUT / f'{name}.pt')
                        (OUT / f'{name}_history.json').write_text(json.dumps(history, indent=2) + '\n')
                        candidates.append(dict(name=name, kind=kind, window=window, rate=rate,
                                               condition='ordered', model=model, threshold=threshold, info=info))
                        print(f'TRAIN {name}: validation BA={info["validation_balanced_accuracy"]:.3f}; epoch={info["selected_epoch"]}', flush=True)
            for kind in MODELS:
                chosen = select_candidate([c for c in candidates if c['kind'] == kind])
                chosen['selected'] = True
                choices[kind] = chosen['name']
            for kind in ('lif', 'leaky_relu'):
                name = identity(held, seed, 32, .01, kind, 'shuffled_history')
                model, threshold, history, info = train_candidate(kind, shuffled['train'], shuffled['validation'], .01, seed)
                torch.save(model.state_dict(), OUT / f'{name}.pt')
                (OUT / f'{name}_history.json').write_text(json.dumps(history, indent=2) + '\n')
                candidates.append(dict(name=name, kind=kind, window=32, rate=.01,
                                       condition='shuffled_history', model=model, threshold=threshold, info=info))
                print(f'TRAIN {name}: validation BA={info["validation_balanced_accuracy"]:.3f}', flush=True)
            freeze = {'held_object': str(data['object_names'][held]), 'seed': seed,
                      'split_groups': splits, 'validation_choices': choices,
                      'candidates': [{key: c[key] for key in ('name', 'kind', 'window', 'rate', 'condition', 'threshold', 'info')}
                                     for c in candidates]}
            (OUT / f'object{held}_seed{seed}_choices_before_test.json').write_text(json.dumps(freeze, indent=2) + '\n')
            for candidate in candidates:
                test = (shuffled if candidate['condition'] == 'shuffled_history' else prepared[candidate['window']])['test']
                base = dict(source=SOURCE, held_object=str(data['object_names'][held]), seed=seed)
                record, rows = evaluate(candidate, test, base)
                summary.append(record)
                predictions.extend(rows)
                print(f'TEST {candidate["name"]}: BA={record["balanced_accuracy"]:.3f}; selected={record["validation_selected"]}', flush=True)
            metadata.append(freeze)
            # Persist progress after each complete fold; completion metadata comes last.
            write_csv(OUT / 'robustness_summary.csv', summary)
            write_csv(OUT / 'robustness_predictions.csv', predictions)
    write_csv(OUT / 'selected_predictions.csv', [r for r in predictions if r['validation_selected']])
    return metadata, time.perf_counter() - started


def frozen_diagnostics(data, derivatives, original):
    original_rows = {}
    with (ROOT / 'advanced/results/predictions.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['model'] in ('ann', 'lif', 'alif'):
                original_rows[int(row['seed']), row['model'], int(row['group']), int(row['end_sample'])] = row
    summaries, predictions = [], []
    conditions = [('ordered', 0), ('reversed_history', 0)] + [('shuffled_history', draw) for draw in range(3)]
    for fold in original['folds']:
        held = int(fold['name'].split('_')[0].replace('object', ''))
        seed = int(fold['name'].split('_')[1].replace('seed', ''))
        prepared, _ = prepare_fold(data, derivatives, fold['split_groups'], window=32)
        x, y, records = prepared['test']
        perturbed = {(condition, draw): perturb(x, records, condition, draw=draw)
                     for condition, draw in conditions}
        for kind in ('ann', 'lif', 'alif'):
            model = make_model(kind)
            model.load_state_dict(torch.load(ROOT / 'advanced/results' / f'{fold["name"]}_{kind}.pt', weights_only=True))
            threshold = fold['thresholds'][kind]
            original_logits = predict(model, x)
            for record, logit in zip(records, original_logits):
                saved = original_rows[seed, kind, record['group'], record['end_sample']]
                if abs(float(saved['logit']) - float(logit)) > 1e-5:
                    raise ValueError('Frozen checkpoint failed to reproduce saved predictions.')
            for condition, draw in conditions:
                logits = predict(model, perturbed[condition, draw])
                decisions = (logits >= threshold).astype(int)
                base = dict(source=SOURCE, held_object=str(data['object_names'][held]), seed=seed,
                            model=kind, condition=condition, draw=draw, window=32, threshold=threshold)
                summaries.append({**base, 'test_windows': len(y), 'test_trials': len({r['group'] for r in records}),
                                  'changed_decisions': int(np.sum(decisions != (original_logits >= threshold))),
                                  'max_absolute_logit_change': float(np.max(np.abs(logits - original_logits))),
                                  **detailed_quality(y, decisions)})
                predictions.extend([{**base, **record, 'label': int(label), 'logit': float(logit), 'decision': int(decision)}
                                    for record, label, logit, decision in zip(records, y, logits, decisions)])
                print(f'DIAGNOSTIC {fold["name"]} {kind} {condition} draw{draw}: BA={summaries[-1]["balanced_accuracy"]:.3f}', flush=True)
    write_csv(OUT / 'temporal_diagnostics_summary.csv', summaries)
    write_csv(OUT / 'temporal_diagnostics_predictions.csv', predictions)


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data, derivatives, original = read_data()
    OUT.mkdir(parents=True, exist_ok=True)
    before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (ROOT / 'advanced/export/models.json', ROOT / 'advanced/results/predictions.csv', ROOT / 'advanced/results/metadata.json')}
    manifest = {'source': SOURCE, 'status': 'running', 'protocol_sha256': hashlib.sha256((ROOT / 'FOLLOWUP_PROTOCOL.md').read_bytes()).hexdigest(),
                'original_frozen_hashes': before, 'environment': dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform()),
                'candidate_fits': 108, 'shuffled_training_fits': 12, 'epochs_max': 18,
                'windows': WINDOWS, 'learning_rates': RATES, 'seeds': SEEDS,
                'selection': 'validation BA; ties shorter window then learning rate0.01; freeze before test',
                'test_schedule': 'same label endpoints for all windows; end>=63; stride16'}
    (OUT / 'metadata.json').write_text(json.dumps(manifest, indent=2) + '\n')
    folds, seconds = run_grid(data, derivatives, original)
    frozen_diagnostics(data, derivatives, original)
    for name, expected in before.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('An original frozen artifact changed.')
    manifest.update(status='complete', folds=folds, grid_wall_seconds=seconds)
    (OUT / 'metadata.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('COMPLETE public_recorded_host_followup; original results/export unchanged.', flush=True)


if __name__ == '__main__':
    main()
