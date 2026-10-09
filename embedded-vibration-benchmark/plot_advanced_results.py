"""Publication-style figures from saved trained-model results, never board estimates."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ('logistic', 'ann', 'lif_frozen', 'lif', 'alif')
LABELS = ('Logistic', 'ANN', 'Frozen LIF', 'Trained LIF', 'Trained ALIF')
COLORS = ('#8297a9', '#416789', '#989898', '#ae5728', '#717b31')


def save(fig, name):
    target = ROOT / 'advanced/figures'
    target.mkdir(exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(target / f'{name}.{extension}', dpi=180)
    plt.close(fig)


def main():
    stats = json.loads((ROOT / 'advanced/results/statistical_summary.json').read_text())
    objects = ('brush', 'screwDriver', 'SpoolSolder')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    for index, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
        means = [100 * np.mean([seed[model] for seed in stats['per_object_seed_balanced_accuracy'][obj].values()]) for obj in objects]
        centers = x + (index - 2) * .145
        ax.bar(centers, means, .135, label=label, color=color)
        for obj, center in zip(objects, centers):
            points = [100 * seed[model] for seed in stats['per_object_seed_balanced_accuracy'][obj].values()]
            ax.scatter([center - .022, center + .022], points, color='#202020', s=12, zorder=3)
    ax.axhline(50, color='#888888', linestyle='--', linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_xticks(x, ('Brush', 'Screwdriver', 'Solder spool'))
    ax.set_ylabel('Held-object balanced accuracy (%)')
    ax.set_title('Leave-one-object-out tactile classification', pad=45)
    ax.legend(frameon=False, ncol=3, loc='lower left', bbox_to_anchor=(0, 1.01), fontsize=10)
    fig.text(.11, .04, 'Public recorded host results. Bars: mean of two fixed training/split seeds; dots: each seed.\n'
             '87 eligible test trials across three objects; overlapping windows; no incipient-onset or ESP32 timing claim.', fontsize=9)
    fig.subplots_adjust(bottom=.20, top=.76)
    save(fig, 'cross_object_quality')

    pairs = stats['paired_comparisons']
    names = {'lif': 'Trained LIF', 'ann': 'ANN', 'alif': 'Trained ALIF', 'lif_frozen': 'Frozen LIF', 'logistic': 'Logistic'}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, row in enumerate(pairs):
        mean = row['difference_percentage_points']
        low, high = row['conditional_paired_trial_95_percentage_points']
        ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt='o', color='#416789', capsize=5)
        ax.annotate(f'{mean:+.2f}', (mean, index), xytext=(0, 11), textcoords='offset points', ha='center', fontsize=10)
    ax.set_yticks(range(len(pairs)), [f"{names[row['first']]} minus {names[row['second']]}" for row in pairs])
    ax.invert_yaxis()
    ax.axvline(0, color='#888888', linestyle='--', linewidth=1)
    ax.set_xlabel('Difference in macro-object balanced accuracy (percentage points)')
    ax.set_title('Paired trial uncertainty, conditional on this study', pad=18)
    ax.margins(y=.18)
    fig.text(.12, .04, '2,000 object-stratified whole-trial resamples; fixed fitted models and two seeds.\n'
             'Intervals are conditional on three observed objects; no population or adjusted significance claim.', fontsize=9)
    fig.subplots_adjust(left=.30, bottom=.23, top=.87)
    save(fig, 'paired_trial_intervals')

    with (ROOT / 'advanced/results/numeric_audit.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row['source'] != 'public_recorded_host_numeric_audit' for row in rows):
        raise ValueError('Expected numerical audit source, not board performance data.')
    models = ('lif', 'alif', 'ann')
    disagreement = [100 * sum(int(row['decision_disagreements']) for row in rows if row['model'] == model) /
                    sum(int(row['test_windows']) for row in rows if row['model'] == model) for model in models]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bars = axes[0].bar(('Float32', 'Int8 weights\n+ float bias/scales'), (516, 188), color=('#416789', '#ae5728'), width=.55)
    axes[0].bar_label(bars, padding=4)
    axes[0].set_ylim(0, 620)
    axes[0].set_ylabel('Logical parameter storage (bytes)')
    bars = axes[1].bar(('LIF', 'ALIF', 'ANN'), disagreement, color=('#ae5728', '#717b31', '#416789'), width=.55)
    axes[1].bar_label(bars, labels=[f'{value:.3f}%' for value in disagreement], padding=5)
    axes[1].set_ylim(0, .55)
    axes[1].set_ylabel('Changed decisions versus float (%)')
    fig.suptitle('Weight storage and numerical trade-offs', y=.97)
    fig.text(.11, .025, '12,018 held-out window evaluations per model; thresholds frozen. States/arithmetic remain float32.\n'
             'Logical parameter bytes exclude state/buffers; this is not measured firmware savings, speed or energy.', fontsize=9)
    fig.subplots_adjust(left=.09, right=.97, wspace=.42, bottom=.24, top=.87)
    save(fig, 'weight_quantization')
    print('Saved cross-object, paired-trial and weight-quantization PNG/SVG figures.')


if __name__ == '__main__':
    main()
