"""Paired trial bootstrap of saved cross-object predictions; no hardware evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ("logistic", "ann", "lif_frozen", "lif", "alif")
PAIRS = (("lif", "ann"), ("alif", "lif"), ("lif", "lif_frozen"), ("lif", "logistic"))


def confusion(label, decision):
    # tp, tn, fp, fn
    return np.array([label == 1 and decision == 1, label == 0 and decision == 0,
                     label == 0 and decision == 1, label == 1 and decision == 0], dtype=np.int64)


def balanced_accuracy(counts):
    tp, tn, fp, fn = np.moveaxis(counts, -1, 0)
    if np.any(tp + fn == 0) or np.any(tn + fp == 0):
        raise ValueError("Balanced accuracy undefined for a single-class resample.")
    return .5 * (tp / (tp + fn) + tn / (tn + fp))


def load_counts(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["source"] != "public_recorded_host_trained" for row in rows):
        raise ValueError("Expected trained-model public-recorded host predictions.")
    objects = sorted({row["held_object"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    windows = {int(row["window"]) for row in rows}
    if len(windows) != 1 or {row["model"] for row in rows} != set(MODELS):
        raise ValueError("Expected one fixed horizon and all five controlled models.")
    groups = {obj: sorted({int(row["group"]) for row in rows if row["held_object"] == obj}) for obj in objects}
    arrays = {obj: np.zeros((len(groups[obj]), len(seeds), len(MODELS), 4), dtype=np.int64) for obj in objects}
    locations = {obj: {group: index for index, group in enumerate(groups[obj])} for obj in objects}
    observed, labels = set(), {}
    object_ids, group_owners, trial_names = {}, {}, {}
    for row in rows:
        obj, seed, model, group = row["held_object"], int(row["seed"]), row["model"], int(row["group"])
        label, decision = int(row["label"]), int(row["decision"])
        if label not in (0, 1) or decision not in (0, 1):
            raise ValueError("Labels/decisions must be binary.")
        if object_ids.setdefault(obj, int(row["object"])) != int(row["object"]) or group_owners.setdefault(group, obj) != obj or trial_names.setdefault(group, row["trial"]) != row["trial"]:
            raise ValueError("Object/group/trial identity changed across predictions.")
        window_key = obj, group, int(row["end_sample"])
        previous_label = labels.setdefault(window_key, label)
        if previous_label != label:
            raise ValueError("Paired models disagree on test labels.")
        key = (*window_key, seed, model)
        if key in observed:
            raise ValueError("Duplicate test prediction.")
        observed.add(key)
        arrays[obj][locations[obj][group], seeds.index(seed), MODELS.index(model)] += confusion(label, decision)
    expected = {(obj, group, end, seed, model) for obj, group, end in labels for seed in seeds for model in MODELS}
    if observed != expected:
        raise ValueError("Missing paired windows/models/seeds; do not bootstrap mismatched inputs.")
    return arrays, objects, seeds, groups, next(iter(windows)), len(labels)


def bootstrap(arrays, replicates=2000, seed=20261007):
    if replicates <= 0:
        raise ValueError("Positive bootstrap replicate count required.")
    values, omitted = [], 0
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        scores = []
        try:
            for obj in sorted(arrays):
                counts = arrays[obj]
                selection = rng.integers(0, len(counts), size=len(counts))
                scores.append(balanced_accuracy(counts[selection].sum(axis=0)).mean(axis=0))
        except ValueError:
            omitted += 1
            continue
        values.append(np.mean(scores, axis=0))
    if not values:
        raise ValueError("No two-class bootstrap replicates; more independent trials required.")
    return np.asarray(values), omitted


def analyze(path, output, replicates=2000):
    arrays, objects, seeds, groups, window, window_count = load_counts(path)
    per_object = {obj: balanced_accuracy(counts.sum(axis=0)) for obj, counts in arrays.items()}
    point = np.mean([scores.mean(axis=0) for scores in per_object.values()], axis=0)
    draws, omitted = bootstrap(arrays, replicates)
    model_stats = {model: {"macro_object_balanced_accuracy": float(point[index]),
                           "conditional_trial_bootstrap_95": np.quantile(draws[:, index], [.025, .975]).tolist()}
                   for index, model in enumerate(MODELS)}
    comparisons = []
    for first, second in PAIRS:
        a, b = MODELS.index(first), MODELS.index(second)
        comparisons.append({"first": first, "second": second,
                            "difference_percentage_points": float(100 * (point[a] - point[b])),
                            "conditional_paired_trial_95_percentage_points": (100 * np.quantile(draws[:, a] - draws[:, b], [.025, .975])).tolist()})
    result = {"source": "public_recorded_host_trained", "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "window_samples": window, "unique_test_windows": window_count,
              "objects": objects, "seeds": seeds, "evaluated_trials": {obj: len(groups[obj]) for obj in objects},
              "estimand": "Mean per-object window balanced accuracy, averaged over two fixed training/split seeds",
              "bootstrap": {"requested": replicates, "retained": len(draws), "omitted_single_class": omitted,
                            "seed": 20261007, "unit": "whole grasp trial, sampled within each object with identical multiplicities for all models/seeds"},
              "model_results": model_stats, "paired_comparisons": comparisons,
              "per_object_seed_balanced_accuracy": {obj: {str(seed): {model: float(scores[index, j]) for j, model in enumerate(MODELS)} for index, seed in enumerate(seeds)} for obj, scores in per_object.items()},
              "limits": ["Conditional on these three objects and fixed fitted models; not population object generalization",
                         "Does not propagate training/model-selection uncertainty; seeds also vary inner validation splits",
                         "Windows overlap and are dependent; resampling preserves entire trials",
                         "Window-weighted quality within object; longer trials still contribute more windows",
                         "No physical incipient onset, hardware timing or measured energy",
                         "Comparisons are exploratory; no multiple-testing-adjusted significance or noninferiority claim"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "statistical_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "paired_bootstrap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "advanced/results/predictions.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "advanced/results")
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    result = analyze(args.predictions, args.output, args.replicates)
    for model, values in result["model_results"].items():
        print(f"{model}: macro-object balanced accuracy {100 * values['macro_object_balanced_accuracy']:.2f}%")
    for row in result["paired_comparisons"]:
        print(row)
    print("Conditional paired-trial intervals only; no general architecture superiority claim.")


if __name__ == "__main__":
    main()
