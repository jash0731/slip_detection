import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ct_lif_nonlinear.run import summary_inputs
from ct_lif_study.run import ROOT, detailed, load_data, verify_splits, windows
from ct_lif_v3.core import balanced_weights, validation_metrics
from ct_lif_v4.core import features, load_model
from .core import (TemporalAlarm, add_only_decisions, bypass_decisions,
                   chronological_order, positive_runs)


THRESHOLDS = np.r_[np.linspace(0, 1, 201), np.nextafter(1., 2.)]


def validation_arrays(y, objects, records, decisions):
    recall = np.mean([decisions[(objects == obj) & (y == 1)].mean(axis=0)
                      for obj in np.unique(objects) if ((objects == obj) & (y == 1)).any()], axis=0)
    fpr = np.mean([decisions[(objects == obj) & (y == 0)].mean(axis=0)
                   for obj in np.unique(objects) if ((objects == obj) & (y == 0)).any()], axis=0)
    runs = positive_runs(y, records)
    event = np.mean([np.mean([decisions[run].any(axis=0) for run in runs
                             if objects[run[0]] == obj], axis=0)
                     for obj in np.unique(objects)
                     if any(objects[run[0]] == obj for run in runs)], axis=0)
    return recall, fpr, (recall + 1 - fpr) / 2, event


def recall_key(e):
    return (-e['recall'], -e['ba'], e['fpr'], -e['event'], e['name'])


def ba_key(e):
    return (-e['ba'], e['fpr'], -e['event'], e['name'])


def entries(decisions, y, objects, records, spec):
    recall, fpr, ba, event = validation_arrays(y, objects, records, decisions)
    return [dict(spec, threshold=float(t), recall=float(recall[j]), fpr=float(fpr[j]),
                 ba=float(ba[j]), event=float(event[j]), name=spec['name'] + f'_t{t}')
            for j, t in enumerate(THRESHOLDS)]


def select(scores, y, objects, records, current_threshold):
    recall, fpr, ba, event = validation_arrays(
        y, objects, records, (scores >= current_threshold)[:, None])
    base = dict(name='current_nonlinear', kind='bypass', alpha=1., gap=0., margin=None,
                threshold=current_threshold, recall=float(recall[0]), fpr=float(fpr[0]),
                ba=float(ba[0]), event=float(event[0]))
    eligible, protected, balanced, smooth = [base], [base], [base], []
    for alpha in [.4, .7]:
        decisions = bypass_decisions(scores, records, alpha, 0., None, THRESHOLDS)
        smooth.extend(entries(decisions, y, objects, records,
                              dict(kind='bypass', alpha=alpha, gap=0., margin=None,
                                   name=f'a{alpha}_g0.0_minf')))
        for gap in [0., .05]:
            for margin in [0., .05, .15, .25]:
                decisions = bypass_decisions(scores, records, alpha, gap, margin, THRESHOLDS)
                candidates = entries(decisions, y, objects, records,
                                     dict(kind='bypass', alpha=alpha, gap=gap, margin=margin,
                                          name=f'a{alpha}_g{gap}_m{margin}'))
                balanced.extend(candidates)
                for candidate in candidates:
                    if (candidate['recall'] >= base['recall'] - 1e-12
                            and candidate['fpr'] <= base['fpr'] + 1e-12):
                        eligible.append(candidate)
                        if candidate['event'] >= base['event'] - 1e-12:
                            protected.append(candidate)
    add = [dict(base, kind='add_only', threshold=None, guard=0.)]
    for alpha in [.4, .7]:
        for guard in [.05, .1, .2]:
            decisions = add_only_decisions(scores, records, alpha, guard, THRESHOLDS, current_threshold)
            candidates = entries(decisions, y, objects, records,
                                 dict(kind='add_only', alpha=alpha, guard=guard,
                                      name=f'a{alpha}_slack{guard}_extra'))
            add.extend(c for c in candidates if c['fpr'] <= base['fpr'] + 1e-12)
    return dict(current_nonlinear=base, previous_smoothing=min(smooth, key=ba_key),
                recall_constrained=min(eligible, key=recall_key),
                event_protected=min(protected, key=recall_key),
                balanced_secondary=min(balanced, key=ba_key), add_only=min(add, key=recall_key))


def stream_decisions(scores, records, choice, current_threshold):
    alarm = TemporalAlarm(choice, current_threshold)
    decisions = np.zeros(len(scores), dtype=bool)
    for i in chronological_order(records):
        r = records[i]
        decisions[i] = alarm.step(scores[i], r['group'], r['end_sample'])
    return decisions


def run(output=None):
    if output is not None:
        output = Path(output)
        if output.exists():
            raise FileExistsError('Choose a new output directory')
    data, metadata = load_data()
    saved = json.loads((ROOT / 'ct_lif_v4/results/all_choices_before_test.json').read_text())['folds']
    prior = json.loads((ROOT / 'ct_lif_nonlinear/choices_before_test.json').read_text())
    frozen, choices = [], []
    for fold in metadata['folds']:
        held, seed, splits = verify_splits(data, fold)
        old = next(f for f in saved if f['held'] == held and f['seed'] == seed)
        original = next(c for c in old['selected'] if c['variant'] == 'ct_only')
        backbone, scale, _ = load_model(ROOT / 'ct_lif_v4/results' / (original['name'] + '.npz'))
        choice = next(f for f in prior if f['held'] == held and f['seed'] == seed)['selections']['balanced']
        match = re.fullmatch(r'leaves(\d+)_iterations(\d+)_rate([\d.]+)', choice['name'])
        if match is None:
            raise ValueError('Unexpected nonlinear baseline choice')
        tr, va = windows(data, splits['train']), windows(data, splits['validation'])
        objects = np.array([r['object'] for r in tr[2]])
        val_objects = np.array([r['object'] for r in va[2]])
        model = HistGradientBoostingClassifier(
            max_leaf_nodes=int(match[1]), max_iter=int(match[2]), learning_rate=float(match[3]),
            l2_regularization=1., min_samples_leaf=20, early_stopping=False, random_state=seed)
        model.fit(summary_inputs(backbone, features(tr[0], scale, 'ct_only')), tr[1],
                  sample_weight=balanced_weights(tr[1], objects))
        scores = model.predict_proba(summary_inputs(backbone, features(va[0], scale, 'ct_only')))[:, 1]
        current_threshold = choice['point']['threshold']
        ref = validation_metrics(va[1], scores, val_objects, current_threshold)
        if not np.isclose(ref['balanced_accuracy'], choice['point']['balanced_accuracy'], atol=1e-12, rtol=0):
            raise ValueError('Baseline did not reproduce')
        selected = select(scores, va[1], val_objects, va[2], current_threshold)
        choices.append(dict(held=held, seed=seed, current_threshold=current_threshold, selections=selected))
        frozen.append((held, seed, splits, backbone, scale, model, current_threshold, selected))
        print('VALIDATED', held, seed, {k: v['name'] for k, v in selected.items()}, flush=True)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'choices_before_test.json').write_text(json.dumps(choices, indent=2, allow_nan=False) + '\n')
    print('ALL CHOICES FROZEN BEFORE TEST INFERENCE', flush=True)
    rows, events = {}, {}
    for held, seed, splits, backbone, scale, model, current_threshold, selected in frozen:
        raw, y, records = windows(data, splits['test'])
        scores = model.predict_proba(summary_inputs(backbone, features(raw, scale, 'ct_only')))[:, 1]
        runs = positive_runs(y, records)
        for method, choice in selected.items():
            decisions = stream_decisions(scores, records, choice, current_threshold)
            if choice['kind'] == 'add_only':
                thresholds = [np.inf if choice['threshold'] is None else choice['threshold']]
                reference = add_only_decisions(scores, records, choice['alpha'], choice['guard'],
                                               thresholds, current_threshold)[:, 0]
                assert np.all(decisions[scores >= current_threshold])
            else:
                reference = bypass_decisions(scores, records, choice['alpha'], choice['gap'],
                                              choice['margin'], [choice['threshold']])[:, 0]
            assert np.array_equal(decisions, reference)
            row = detailed(y, decisions.astype(float), .5)
            row.update(held=held, seed=seed, name=choice['name'])
            rows.setdefault(method, []).append(row)
            for segment in runs:
                hits = [i for i in segment if decisions[i]]
                events.setdefault(method, []).append(dict(
                    held=held, seed=seed, group=records[segment[0]]['group'],
                    first=records[segment[0]]['end_sample'], detected=bool(hits),
                    delay_ms=None if not hits else records[hits[0]]['t_ms'] - records[segment[0]]['t_ms']))
    summaries = {}
    for method, rs in rows.items():
        result = {k: float(np.mean([r[k] for r in rs])) for k in [
            'balanced_accuracy', 'slip_recall', 'false_alarm_rate']}
        result['accuracy'] = (sum(r['tp'] + r['tn'] for r in rs) /
                              sum(r['tp'] + r['tn'] + r['fp'] + r['fn'] for r in rs))
        result['evaluated_positive_segments'] = len(events[method])
        result['detected_segments'] = sum(e['detected'] for e in events[method])
        result['tp'] = sum(r['tp'] for r in rs)
        result['fp'] = sum(r['fp'] for r in rs)
        if method != 'current_nonlinear':
            paired = list(zip(events['current_nonlinear'], events[method]))
            common = [v['delay_ms'] - u['delay_ms'] for u, v in paired if u['detected'] and v['detected']]
            result.update(
                newly_detected_segments=sum(not u['detected'] and v['detected'] for u, v in paired),
                newly_missed_segments=sum(u['detected'] and not v['detected'] for u, v in paired),
                mean_first_hit_delay_change_ms=float(np.mean(common)))
        summaries[method] = result
        print('RESULT', method, json.dumps(result), flush=True)
    report = dict(summaries=summaries, folds=rows, exploratory_reused_benchmark=True,
                  temporal_rules=dict(bypass=16, smoothing_controls=2, add_only=6),
                  thresholds_per_rule=len(THRESHOLDS), retrained_reference_models=6)
    if output is not None:
        (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='New directory for JSON results and selections')
    run(parser.parse_args().output)
