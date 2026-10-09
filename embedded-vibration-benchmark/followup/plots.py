"""Scientific plots of recorded-host follow-up results with explicit evidence labels."""
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT

RESULTS = ROOT / 'followup/results'
FIGURES = ROOT / 'followup/figures'
OBJECTS = ('brush', 'screwDriver', 'SpoolSolder')
MODEL_NAMES = {'ann': 'Mean-ReLU ANN', 'lif': 'LIF', 'leaky_relu': 'Leaky-ReLU', 'alif': 'ALIF'}
COLORS = {'ann': '#416789', 'lif': '#ae5728', 'leaky_relu': '#65834c', 'alif': '#717b31'}


def read(name):
    with name.open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(FIGURES / f'{name}.{extension}', dpi=180)
    plt.close(fig)


def main():
    analysis = json.loads((RESULTS / 'analysis.json').read_text())
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    diagnostic = analysis['frozen_order_diagnostics']['means']
    conditions = ('ordered', 'reversed_history', 'shuffled_history')
    for index, model in enumerate(('ann', 'lif', 'alif')):
        axes[0].bar(np.arange(3) + (index - 1) * .24,
                    [100 * diagnostic[f'{model}:{condition}'] for condition in conditions],
                    .23, color=COLORS[model], label=MODEL_NAMES[model])
    axes[0].set_xticks(range(3), ('Original', 'Past reversed', 'Past shuffled'))
    axes[0].set_title('Frozen models: input-order diagnostics')
    retrained = analysis['matched_shuffled_retraining']['means']
    for index, model in enumerate(('lif', 'leaky_relu')):
        axes[1].bar(np.arange(2) + (index - .5) * .28,
                    [100 * retrained[f'{model}:{condition}'] for condition in ('ordered', 'shuffled_history')],
                    .27, color=COLORS[model], label=MODEL_NAMES[model])
    axes[1].set_xticks(range(2), ('Original order', 'Shuffled train/val/test'))
    axes[1].set_title('Matched retraining: window32 / lr0.01')
    for axis in axes:
        axis.set_ylim(0, 100)
        axis.set_ylabel('Mean object balanced accuracy (%)')
        axis.axhline(50, color='#888', linestyle='--', linewidth=1)
        axis.legend(frameon=False, fontsize=10)
    fig.suptitle('Temporal order contributes under this representation', fontsize=15)
    fig.text(.08, .035, 'Public recorded host results; mean of three objects and two fixed seeds. Last frame and input multiset preserved.\n'
             'Left: three shuffle draws averaged, frozen thresholds. Right: shuffled models trained and calibrated separately.', fontsize=10)
    fig.subplots_adjust(top=.82, bottom=.22, wspace=.27)
    save(fig, 'temporal_controls')

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.8))
    for axis, model in zip(axes, ('ann', 'lif', 'leaky_relu')):
        values = np.array([[100 * next(r['macro_object_seed_balanced_accuracy'] for r in analysis['fixed_grid']
                                     if r['model'] == model and r['window'] == window and r['learning_rate'] == rate)
                            for rate in (.005, .01)] for window in (16, 32, 64)])
        im = axis.imshow(values, cmap='YlGnBu', vmin=50, vmax=65, aspect='auto')
        for y in range(3):
            for x in range(2):
                axis.text(x, y, f'{values[y, x]:.1f}', ha='center', va='center', color='white' if values[y, x] > 60 else '#172632')
        axis.set_xticks((0, 1), ('0.005', '0.010'))
        axis.set_yticks((0, 1, 2), ('16 (~89 ms)', '32 (~178 ms)', '64 (~356 ms)'))
        axis.set_xlabel('Adam learning rate')
        axis.set_title(MODEL_NAMES[model])
    axes[0].set_ylabel('Window samples (nominal duration)')
    fig.suptitle('Sensitivity: identical held-out endpoints, fixed candidate grid', fontsize=14)
    fig.subplots_adjust(left=.14, right=.88, bottom=.24, top=.82, wspace=.55)
    cax = fig.add_axes([.91, .26, .016, .55])
    fig.colorbar(im, cax=cax, label='Balanced accuracy (%)')
    fig.text(.08, .04, 'Public recorded host results; each cell averages three objects × two seeds. Test-grid maxima are not used for selection.\n'
             'All partitions share endpoint labels; only available past context changes. Observation times are nominal, not detection latency.', fontsize=10)
    save(fig, 'robustness_grid')

    rows = read(RESULTS / 'selected_error_rates.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for index, model in enumerate(('ann', 'lif', 'leaky_relu')):
        for axis, metric in zip(axes, ('balanced_accuracy', 'miss_rate')):
            values = [[100 * float(r[metric]) for r in rows if r['held_object'] == obj and r['model'] == model]
                      for obj in OBJECTS]
            centers = np.arange(3) + (index - 1) * .24
            axis.bar(centers, [np.mean(value) for value in values], .23, color=COLORS[model], label=MODEL_NAMES[model])
            for center, points in zip(centers, values):
                axis.scatter([center - .025, center + .025], points, s=13, color='#222', zorder=3)
    for axis in axes:
        axis.set_xticks(range(3), ('Brush', 'Screwdriver', 'Solder spool'))
        axis.set_ylim(0, 100)
        axis.legend(frameon=False, fontsize=10)
    axes[0].set_title('Held-object classification quality')
    axes[0].set_ylabel('Balanced accuracy (%)')
    axes[0].axhline(50, color='#888', linestyle='--', linewidth=1)
    axes[1].set_title('Missed slip: lower is better')
    axes[1].set_ylabel('False-negative rate (%)')
    fig.suptitle('Validation-selected models retain important transfer failures', fontsize=14)
    fig.text(.08, .035, 'Public recorded host results; each model chooses window/lr from validation only. Bars average two seeds; dots show each seed.\n'
             'Different selected windows have different observation costs. These are existing-slip labels, not physical early-warning performance.', fontsize=10)
    fig.subplots_adjust(top=.82, bottom=.22, wspace=.25)
    save(fig, 'selected_quality_and_misses')

    contexts = ('stable_non_slip', 'stable_slip', 'mixed_held_states', 'ignored_context')
    context_rows = read(ROOT / 'followup/failures/by_context.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    cmap = plt.get_cmap('OrRd').copy()
    cmap.set_bad('#dddddd')
    for axis, metric, title in zip(axes, ('false_alarm_rate', 'miss_rate'), ('False alarms on non-slip endpoints', 'Misses on slip endpoints')):
        matrix = np.full((3, 4), np.nan)
        for i, obj in enumerate(OBJECTS):
            for j, context in enumerate(contexts):
                values = [100 * float(r[metric]) for r in context_rows
                          if r['held_object'] == obj and r['model'] == 'lif' and r['context'] == context and r[metric] != '']
                if values:
                    matrix[i, j] = np.mean(values)
        im = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')
        for i in range(3):
            for j in range(4):
                value = matrix[i, j]
                axis.text(j, i, 'N/A' if np.isnan(value) else f'{value:.1f}%', ha='center', va='center',
                          color='white' if np.isfinite(value) and value > 60 else '#222')
        axis.set_xticks(range(4), ('Stable\nnon-slip', 'Stable\nslip', 'Mixed\nheld states', 'Ignored\ncontext'))
        axis.set_yticks(range(3), ('Brush', 'Screwdriver', 'Solder spool'))
        axis.set_title(title, fontsize=11)
    fig.suptitle('Original frozen window32 LIF: failure context', fontsize=14)
    fig.subplots_adjust(left=.12, right=.88, top=.82, bottom=.26, wspace=.35)
    cax = fig.add_axes([.91, .29, .016, .52])
    fig.colorbar(im, cax=cax, label='Error rate (%)')
    fig.text(.08, .04, 'Public recorded host post-hoc audit; cells average two fixed seeds. Grey: no relevant class, not zero errors.\n'
             'Context is based on the last 32 source labels. Subgroups/windows are dependent; no physical onset or significance claim.', fontsize=10)
    save(fig, 'failure_context')


if __name__ == '__main__':
    main()
