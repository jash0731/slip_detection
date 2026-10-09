"""Exploratory held-object slip classification on public ARQ-CRISP brush trials.

Not incipient prediction: compare homogeneous labelled 32-sample windows only.
Train on whole pose-1/2 trials; hold out every pose-3 trial. No overlapping windows.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import platform
import sys
import time

import numpy as np

from .experiment import features, fit_readout, score, metrics, export_header, WEIGHTS, THRESHOLDS, LEAK, SEED

WINDOW = 32
MODELS = ("threshold", "ema", "scalar_lif", "ann", "snn_dense", "snn_sparse")


def prepare(data):
    """Keep each original trial intact; calibration uses training poses only."""
    amplitudes = np.zeros(len(data["x"]), dtype=np.float64)
    for group in np.unique(data["group"]):
        rows = np.flatnonzero(data["group"] == group)
        values = data["x"][rows].astype(np.float64)
        # RMS first differences across all 54 axes; first trial sample has no past.
        differences = np.diff(values, axis=0, prepend=values[:1])
        amplitudes[rows] = np.sqrt(np.mean(differences ** 2, axis=1))
    training_mask = data["pose"] != 3
    scale = max(float(np.percentile(amplitudes[training_mask], 95)), 1e-9)
    amplitudes /= scale
    windows, excluded = [], 0
    for group in np.unique(data["group"]):
        rows = np.flatnonzero(data["group"] == group)
        for start in range(0, len(rows) - WINDOW + 1, WINDOW):
            indices = rows[start:start + WINDOW]
            labels = data["original_label"][indices]
            if not (np.all(labels == 4) or np.all(labels == 5)):
                excluded += 1
                continue
            windows.append({"recording_id": f"arq_brush_trial{int(group):02d}_sample{start:04d}",
                            "trial": str(data["trial_names"][int(group)]),
                            "group": int(group), "pose": int(data["pose"][indices[0]]),
                            "label": int(labels[0] == 5),
                            "raw": tuple(float(x) for x in amplitudes[indices]),
                            "sample_interval_us": int(round(float(np.median(np.diff(data["t_ms"][indices]))) * 1000)),
                            "start_sample": start})
    return windows, scale, excluded


def run(data_path, output, header):
    data = np.load(data_path, allow_pickle=False)
    windows, scale, excluded = prepare(data)
    train = [w for w in windows if w["pose"] != 3]
    test = [w for w in windows if w["pose"] == 3]
    if not train or not test or set(w["label"] for w in test) != {0, 1}:
        raise ValueError("Train/test data must exist and test must contain both classes")
    output.mkdir(parents=True, exist_ok=True)
    summaries, predictions, fitted = [], [], {}
    for model in MODELS:
        rows = [features(w["raw"], model)[0] for w in train]
        weights, bias = fit_readout(rows, [w["label"] for w in train])
        fitted[model] = (weights, bias)
        labels, decisions, elapsed = [], [], []
        for window in test:
            started = time.perf_counter_ns()
            row, _ = features(window["raw"], model)
            logit = score(row, weights, bias)
            elapsed.append((time.perf_counter_ns() - started) / 1000)
            decision = int(logit >= 0)
            labels.append(window["label"])
            decisions.append(decision)
            predictions.append({"source": "public_recorded_host", "model": model,
                                "trial": window["trial"], "pose": window["pose"],
                                "start_sample": window["start_sample"],
                                "label": window["label"], "prediction": decision, "logit": logit})
        summaries.append({"source": "public_recorded_host", "model": model,
                          "test_windows": len(test), "test_trials": len(set(w["group"] for w in test)),
                          **metrics(labels, decisions),
                          "host_median_us_per_window": statistics.median(elapsed)})
    assert fitted["snn_dense"] == fitted["snn_sparse"]
    for name, rows in (("summary.csv", summaries), ("predictions.csv", predictions)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    fixture = next(w for w in test if w["label"] == 1)
    export_header(header, fixture, *fitted["snn_sparse"],
                  model_id="arq_brush_fixed_lif_trained_readout_v1", sample_interval_us=fixture["sample_interval_us"],
                  data_origin="public_recorded_tactile", input_description="54-axis RMS first difference / train p95")
    metadata = {"source": "public_recorded_host",
                "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
                "model_configuration": {"hidden_weights_seed": SEED + 1, "snn_hidden": 12, "ann_hidden": 8, "leak": LEAK, "thresholds": THRESHOLDS, "snn_weights": WEIGHTS, "state_reset": "zero each32samplewindow", "readout_steps": 400},
                "fixture_selection": {"rule": "first held-out positive window in deterministic order; no prediction screening; parity only", "trial": fixture["trial"], "label": fixture["label"], "start_sample": fixture["start_sample"], "end_sample_exclusive": fixture["start_sample"] + WINDOW},
                "timing_protocol": "one pass per window; fixed model order; descriptive only, not a reliable host cost comparison", "dataset": "ARQ-CRISP 2021 brush only",
                "source_sha256": str(data["source_sha256"]),
                "converted_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "window_samples": WINDOW, "split": "poses 1/2 training; pose 3 untouched test",
                "train_trials": sorted(set(w["trial"] for w in train)),
                "test_trials": sorted(set(w["trial"] for w in test)),
                "train_windows": len(train), "test_windows": len(test),
                "train_class_counts": {str(y): sum(w["label"] == y for w in train) for y in (0, 1)},
                "test_class_counts": {str(y): sum(w["label"] == y for w in test) for y in (0, 1)},
                "excluded_nonhomogeneous_or_nonheld_windows": excluded,
                "test_trial_class_counts": {trial: {str(y): sum(w["trial"] == trial and w["label"] == y for w in test) for y in (0, 1)} for trial in sorted(set(w["trial"] for w in test))},
                "input_normalization_train_p95": scale,
                "fitted_readouts": fitted, "fixture": fixture["recording_id"],
                "training": "fixed hidden weights; logistic readouts, 400 full-batch steps, train-only normalization",
                "limitations": ["Only one brush object; small pose-held-out sample",
                                "Windows within each trial are dependent; no independent-window confidence intervals",
                                "Original label4 held no-slip vs label5 held slip; labels0/1/2/3/6 excluded",
                                "Only completely homogeneous windows retained using labels: conditional selected-subset score, not general online accuracy",
                                "Zero-state reset per window, not continuous streaming",
                                "ANN8 and SNN12 are not capacity matched; no general architecture comparison",
                                "Approximately 178ms observation horizon; not incipient-slip lead time",
                                "Input preprocessing not included in model timing; exported scalar fixture precomputed",
                                "Host timing is Python; no ESP32 results",
                                "Not piezo sensor validation; original sensor has 54 tactile axes",
                                "No hyperparameter search and no validation set; fixed demonstration configuration"]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Public exploratory study: {len(train)} training / {len(test)} held-out windows.")
    for row in summaries:
        print(f"{row['model']}: balanced accuracy {row['balanced_accuracy']:.3f}, F1 {row['f1']:.3f}")
    print("Not physical incipient-slip prediction or hardware timing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data" / "arq_brush_trials.npz")
    parser.add_argument("--output", type=Path, default=root / "neuromorphic" / "public_results")
    parser.add_argument("--header", type=Path, default=root / "firmware" / "esp32_snn_replay" / "generated_model.h")
    args = parser.parse_args()
    run(args.data, args.output, args.header)


if __name__ == "__main__":
    main()
