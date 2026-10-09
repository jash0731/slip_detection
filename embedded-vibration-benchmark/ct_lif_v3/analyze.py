import csv
import json
import numpy as np
from .run import ROOT, OUT, write_json, write_csv


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def measures(counts):
    tp, tn, fp, fn = np.moveaxis(counts, -1, 0)
    recall = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan, dtype=float), where=tp + fn > 0)
    fpr = np.divide(fp, fp + tn, out=np.full_like(fp, np.nan, dtype=float), where=fp + tn > 0)
    return np.stack([(recall + 1 - fpr) / 2, recall, fpr], axis=-1)


def main():
    rows = read(OUT / 'predictions.csv')
    for row in read(ROOT / 'ct_lif_v2/results/predictions.csv'):
        if row['model'] in ['ct_v2', 'mean_lif']:
            rows.append({**row, 'model': row['model'] if row['model'] == 'ct_v2' else 'v2_mean_lif'})
    for row in read(ROOT / 'ct_lif_study/results/predictions.csv'):
        if row['model'] in ['ct_lif', 'mean_lif28', 'c_only_lif']:
            names = {'ct_lif': 'ct_v1', 'mean_lif28': 'mean_lif28', 'c_only_lif': 'v1_cancellation_lif'}
            rows.append({**row, 'model': names[row['model']], 'policy': 'original'})
    keys = list(dict.fromkeys((r['model'], r['policy']) for r in rows))
    mapping = {key: i for i, key in enumerate(keys)}
    groups = {o: sorted({int(r['group']) for r in rows if int(r['object']) == o}) for o in range(3)}
    group_indices = {o: {g: i for i, g in enumerate(groups[o])} for o in groups}
    matrices = {o: np.zeros((len(keys), 2, len(groups[o]), 4), dtype=int) for o in groups}
    identities = {}
    names = {}
    for row in rows:
        key = row['model'], row['policy']
        obj, seed, group, endpoint = (int(row[k]) for k in ['object', 'seed', 'group', 'end_sample'])
        y, p = int(row['label']), int(row['decision'])
        fold_key = (*key, seed, obj)
        identity = (group, endpoint, y)
        bucket = identities.setdefault(fold_key, set())
        if identity in bucket:
            raise ValueError('Duplicate test endpoint')
        bucket.add(identity)
        index = 0 if y == 1 and p == 1 else 1 if y == 0 and p == 0 else 2 if y == 0 else 3
        matrices[obj][mapping[key], [7, 19].index(seed), group_indices[obj][group], index] += 1
        names[obj] = row['held_object']
    for key in keys:
        for obj in range(3):
            for seed in [7, 19]:
                assert identities[(*key, seed, obj)] == identities[(*keys[0], seed, obj)]
    aggregate, per_object = [], []
    for index, (model, policy) in enumerate(keys):
        scores = []
        total = np.zeros(4, dtype=int)
        for obj in groups:
            counts = matrices[obj][index].sum(axis=1)
            score = measures(counts)
            scores.append(score)
            total += counts.sum(axis=0)
            ba, recall, fpr = score.mean(axis=0)
            per_object.append({'model': model, 'policy': policy, 'held_object': names[obj],
                               'balanced_accuracy': float(ba), 'slip_recall': float(recall), 'false_alarm_rate': float(fpr)})
        ba, recall, fpr = np.stack(scores).mean(axis=(0, 1))
        aggregate.append({'model': model, 'policy': policy, 'balanced_accuracy': float(ba), 'slip_recall': float(recall),
                          'false_alarm_rate': float(fpr), 'pooled_accuracy': float((total[0] + total[1]) / total.sum()),
                          **{key: int(value) for key, value in zip(['tp', 'tn', 'fp', 'fn'], total)}})
    random = np.random.default_rng(20261009)
    bootstraps = []
    for _ in range(2000):
        scores = []
        for obj in groups:
            multiplicities = np.bincount(random.integers(len(groups[obj]), size=len(groups[obj])), minlength=len(groups[obj]))
            counts = (matrices[obj] * multiplicities[None, None, :, None]).sum(axis=2)
            scores.append(measures(counts))
        result = np.stack(scores).mean(axis=(0, 2))
        if np.isfinite(result).all():
            bootstraps.append(result)
    bootstraps = np.stack(bootstraps)
    intervals = []
    for policy in ['balanced', 'joint']:
        candidate_index = mapping['ct_v3', policy]
        for opponent in [('ct_v2', 'balanced'), ('mean_lif', policy), ('linear_control', policy),
                         ('mean_lif28', 'original'), ('ct_v1', 'original'), ('ct_rate', policy)]:
            difference = bootstraps[:, candidate_index] - bootstraps[:, mapping[opponent]]
            point = [aggregate[candidate_index][k] - aggregate[mapping[opponent]][k]
                     for k in ['balanced_accuracy', 'slip_recall', 'false_alarm_rate']]
            intervals.append({'candidate_model': 'ct_v3', 'candidate_policy': policy, 'analysis_role': 'planned_comparison', 'opponent': opponent[0], 'opponent_policy': opponent[1],
                              'difference': dict(zip(['balanced_accuracy', 'slip_recall', 'false_alarm_rate'], point)),
                              'interval_95': {key: np.quantile(difference[:, i], [.025, .975]).tolist()
                                              for i, key in enumerate(['balanced_accuracy', 'slip_recall', 'false_alarm_rate'])}})
    candidate_index = mapping['mean_lif', 'balanced']
    for opponent in [('ct_v2', 'balanced'), ('v2_mean_lif', 'balanced'), ('mean_lif28', 'original')]:
        difference = bootstraps[:, candidate_index] - bootstraps[:, mapping[opponent]]
        point = [aggregate[candidate_index][k] - aggregate[mapping[opponent]][k]
                 for k in ['balanced_accuracy', 'slip_recall', 'false_alarm_rate']]
        intervals.append({'candidate_model': 'mean_lif', 'candidate_policy': 'balanced',
                          'analysis_role': 'posthoc_control_comparison', 'opponent': opponent[0], 'opponent_policy': opponent[1],
                          'difference': dict(zip(['balanced_accuracy', 'slip_recall', 'false_alarm_rate'], point)),
                          'interval_95': {key: np.quantile(difference[:, i], [.025, .975]).tolist()
                                          for i, key in enumerate(['balanced_accuracy', 'slip_recall', 'false_alarm_rate'])}})
    frozen = json.loads((OUT / 'all_choices_before_test.json').read_text())
    selections = []
    for fold in frozen['folds']:
        for choice in fold['selected']:
            if choice['model'] == 'ct_v3':
                candidate = choice['candidate']
                selections.append({'held_object': fold['held_object'], 'seed': fold['seed'], 'policy': choice['policy'],
                                   'name': candidate['name'], 'family': candidate['family'], 'pooling': candidate['pooling'],
                                   'parameters': candidate['parameters'], 'threshold': choice['threshold'],
                                   'validation_joint_improvement': choice['validation_joint_improvement'],
                                   'validation': candidate['operating_points'][choice['policy']],
                                   'reference_validation': fold['v2_balanced_validation_reference']})
    write_csv(OUT / 'aggregate.csv', aggregate)
    write_csv(OUT / 'per_object.csv', per_object)
    write_json(OUT / 'analysis.json', {'aggregate': aggregate, 'per_object': per_object, 'intervals': intervals,
                                      'selections': selections, 'valid_bootstraps': len(bootstraps),
                                      'training_seconds': frozen['training_seconds'],
                                      'limitations': 'Exploratory repeated benchmark; conditional trial bootstrap omits new-object, retraining, selection and multiple-comparison uncertainty.'})
    focus = [r for r in aggregate if r['model'] in ['ct_v3', 'mean_lif', 'linear_control', 'ct_v2', 'ct_v1', 'mean_lif28']]
    print(json.dumps({'aggregate': focus, 'ct_objects': [r for r in per_object if r['model'] == 'ct_v3']}, indent=2))
    plot(aggregate)


def plot(aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entries = [('CT-LIF v2 balanced', 'ct_v2', 'balanced'), ('Optimized mean LIF', 'mean_lif', 'balanced'),
               ('CT residual experiment', 'ct_v3', 'joint'), ('Earlier mean LIF (28)', 'mean_lif28', 'original')]
    figures, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    colors = ['#9cabb5', '#087e8b', '#d17a43', '#745ca7']
    for axis, metric, title in zip(axes, ['balanced_accuracy', 'slip_recall', 'false_alarm_rate'],
                                   ['Balanced accuracy', 'Slips detected', 'False alarms']):
        values = [100 * next(r[metric] for r in aggregate if (r['model'], r['policy']) == (model, policy)) for _, model, policy in entries]
        axis.barh(range(len(entries)), values, color=colors)
        axis.set(yticks=range(len(entries)), yticklabels=[r[0] for r in entries], xlim=(0, 100), xlabel='Percent', title=title)
        axis.invert_yaxis()
        axis.spines[['top', 'right']].set_visible(False)
        for i, value in enumerate(values):
            axis.text(value + 1, i, f'{value:.2f}', va='center', fontsize=9)
    figures.suptitle('Optimized mean LIF improved all three averages over V2', fontsize=14)
    figures.text(.02, .015, 'Exploratory reused benchmark; family recommendation follows test comparison. Higher accuracy/recall and lower false alarms are better.', fontsize=8)
    figures.tight_layout(rect=[0, .05, 1, .94])
    figures.savefig(OUT / 'comparison.png', dpi=180)
    plt.close(figures)


if __name__ == '__main__':
    main()
