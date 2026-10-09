"""Reproducible SYNTHETIC inference scaffold, implemented with the standard library.

Run from the project directory: python3 -m neuromorphic.experiment
All labels are generator labels, never ground truth for physical gripper slip.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import platform
import sys
import time

SEED = 20261006
INPUTS, HIDDEN, TRACE_LENGTH = 4, 12, 256
THRESHOLDS = (0.12, 0.30, 0.60, 1.00)
LEAK = 0.875
MODEL_ID = "synthetic_demo_feedforward_lif_readout_v1"


def encode(sample):
    """A causal, memoryless four-band binary amplitude encoder."""
    return tuple(int(abs(sample) >= level) for level in THRESHOLDS)


def reservoir_weights():
    rng = random.Random(SEED + 1)
    return tuple(tuple(rng.uniform(0.06, 0.65) for _ in range(HIDDEN))
                 for _ in range(INPUTS))


WEIGHTS = reservoir_weights()


class Network:
    """Connected 4-input/12-LIF network with persistent state and a linear readout.

    Neurons update on EVERY sample, even if all inputs are zero. Sparse mode only
    skips zero-valued input contributions; it does not skip leak or neuron updates.
    One subtractive-reset spike per neuron per tick, no refractory period.
    """

    def __init__(self, sparse=False):
        self.sparse = sparse
        self.voltage = [0.0] * HIDDEN
        self.counts = [0] * HIDDEN
        self.operations = 0
        self.samples = 0
        self.input_spikes = 0

    def step(self, sample):
        inputs = encode(sample)
        self.input_spikes += sum(inputs)
        spikes = []
        for h in range(HIDDEN):
            # Keep dense and sparse summation order identical.
            contribution = 0.0
            for i, active in enumerate(inputs):
                if not self.sparse or active:
                    contribution += active * WEIGHTS[i][h]
                    self.operations += 1
            value = LEAK * self.voltage[h] + contribution
            fired = int(value >= 1.0)
            self.voltage[h] = value - fired
            self.counts[h] += fired
            spikes.append(fired)
        self.samples += 1
        return tuple(spikes)

    def features(self):
        return tuple(count / self.samples for count in self.counts)


def traces(count=120, seed=SEED):
    """Independent synthetic recordings. Two designed waveform families only.

    Positive labels denote an artificial burst-rich family inspired by vibration;
    they are not annotated incipient-slip events. No synthetic onset is used as
    a physical slip onset or as an end-to-end latency reference.
    """
    result = []
    for index in range(count):
        rng = random.Random(seed + 1000 + index)
        label = index % 2
        onset = rng.randrange(32, 112)
        phase = rng.uniform(0, 2 * math.pi)
        frequency = rng.uniform(0.10, 0.50)
        amplitude = rng.uniform(0.35, 1.10)
        raw = []
        for tick in range(TRACE_LENGTH):
            sample = rng.gauss(0.0, rng.uniform(0.035, 0.12))
            # Both classes contain background vibration and isolated impacts.
            sample += 0.07 * math.sin(0.07 * tick + phase)
            if rng.random() < 0.012:
                sample += rng.uniform(-0.9, 0.9)
            if label and tick >= onset:
                envelope = 0.4 + 0.6 * abs(math.sin(0.031 * (tick - onset)))
                sample += amplitude * envelope * math.sin(frequency * tick + phase)
            raw.append(sample)
        result.append({"recording_id": f"synthetic_{index:04d}",
                       "label": label, "raw": tuple(raw)})
    return result


def split_recordings(recordings):
    """Split whole recordings, balanced within each synthetic class."""
    train, test = [], []
    for label in (0, 1):
        group = [r for r in recordings if r["label"] == label]
        random.Random(SEED + label).shuffle(group)
        boundary = len(group) * 2 // 3
        train.extend(group[:boundary])
        test.extend(group[boundary:])
    return train, test


ANN_WEIGHTS = tuple(tuple(random.Random(SEED + 50 + h * INPUTS + i).uniform(-1, 1)
                         for i in range(INPUTS)) for h in range(8))


def features(raw, model):
    if model.startswith("snn_"):
        network = Network(sparse=model.endswith("sparse"))
        for sample in raw:
            network.step(sample)
        return network.features(), network
    if model == "threshold":
        return (sum(abs(x) >= 0.30 for x in raw) / len(raw),), None
    if model == "ema":
        value, total = 0.0, 0.0
        for sample in raw:
            value = 0.875 * value + 0.125 * abs(sample)
            total += value
        return (total / len(raw),), None
    if model == "scalar_lif":
        voltage, count = 0.0, 0
        for sample in raw:
            voltage = 0.875 * voltage + abs(sample)
            fired = int(voltage >= 1.0)
            voltage -= fired
            count += fired
        return (count / len(raw),), None
    if model == "ann":
        # A small 4->8 fixed tanh feature layer plus a trained logistic readout.
        # It consumes exactly the same encoded sequence and horizon as the SNN.
        totals = [0.0] * 8
        for sample in raw:
            inputs = encode(sample)
            for h, row in enumerate(ANN_WEIGHTS):
                totals[h] += math.tanh(sum(w * x for w, x in zip(row, inputs)))
        return tuple(total / len(raw) for total in totals), None
    raise ValueError(model)


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def fit_readout(rows, labels, steps=400):
    """Train only a logistic readout by full-batch gradient descent.

    Feature normalization is fitted on TRAIN data only. Convert fitted coefficients
    back to original feature units for deployment. No hidden weights are learned.
    """
    columns = list(zip(*rows))
    mean = [statistics.mean(column) for column in columns]
    scale = [max(statistics.pstdev(column), 1e-6) for column in columns]
    normalized = [[(v - m) / s for v, m, s in zip(row, mean, scale)] for row in rows]
    weights, bias = [0.0] * len(mean), 0.0
    for _ in range(steps):
        grad, bias_grad = [0.0] * len(weights), 0.0
        for row, label in zip(normalized, labels):
            error = sigmoid(bias + sum(w * x for w, x in zip(weights, row))) - label
            bias_grad += error
            for i, value in enumerate(row):
                grad[i] += error * value
        for i in range(len(weights)):
            weights[i] -= 0.15 * (grad[i] / len(rows) + 0.01 * weights[i])
        bias -= 0.15 * bias_grad / len(rows)
    raw_weights = tuple(w / s for w, s in zip(weights, scale))
    raw_bias = bias - sum(w * m for w, m in zip(raw_weights, mean))
    return raw_weights, raw_bias


def score(row, weights, bias):
    return bias + sum(value * weight for value, weight in zip(row, weights))


def metrics(labels, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    return {"accuracy": (tp + tn) / len(labels),
            "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def c_float(value):
    text = format(value, ".9g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"


def export_header(path, fixture, weights, bias, model_id=MODEL_ID, sample_interval_us=1000,
                  data_origin="synthetic", input_description="synthetic scalar amplitude"):
    network = Network(sparse=True)
    for sample in fixture["raw"]:
        network.step(sample)
    expected_score = score(network.features(), weights, bias)
    rows = ["// Fixed hidden LIF weights; trained logistic readout. No live gripper validation.",
            "// Python uses binary64; C uses binary32. Compare with tolerances in replay sketch.",
            "#pragma once", "namespace gripper_model {",
            f'constexpr char MODEL_ID[] = "{model_id}";',
            f'constexpr char DATA_ORIGIN[] = "{data_origin}";',
            f'constexpr char INPUT_DESCRIPTION[] = "{input_description}";',
            f'constexpr char INPUT_ID[] = "{fixture["recording_id"]}";',
            f"constexpr int INPUTS=4, HIDDEN=12, TRACE_LENGTH={len(fixture['raw'])};",
            f"constexpr unsigned SAMPLE_INTERVAL_US={sample_interval_us};",
            "constexpr float LEAK=0.875f, THRESHOLD=1.0f, RESET=1.0f;",
            "constexpr float INPUT_THRESHOLDS[INPUTS] = {" + ",".join(map(c_float, THRESHOLDS)) + "};",
            "constexpr float WEIGHTS[INPUTS][HIDDEN] = {"]
    rows += ["  {" + ",".join(map(c_float, row)) + "}," for row in WEIGHTS]
    rows += ["};", "constexpr float READOUT[HIDDEN] = {" + ",".join(map(c_float, weights)) + "};",
             f"constexpr float BIAS = {c_float(bias)};",
             "constexpr float TRACE[TRACE_LENGTH] = {"]
    for start in range(0, len(fixture["raw"]), 8):
        rows.append("  " + ",".join(map(c_float, fixture["raw"][start:start + 8])) + ",")
    rows += ["};", "constexpr int EXPECTED_COUNTS[HIDDEN] = {" + ",".join(map(str, network.counts)) + "};",
             "constexpr float EXPECTED_VOLTAGE[HIDDEN] = {" + ",".join(map(c_float, network.voltage)) + "};",
             f"constexpr float EXPECTED_SCORE = {c_float(expected_score)};", "}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows))


def run(output, header, repeats=5):
    recordings = traces()
    train, test = split_recordings(recordings)
    output.mkdir(parents=True, exist_ok=True)
    models = ("threshold", "ema", "scalar_lif", "ann", "snn_dense", "snn_sparse")
    summary, predictions, fitted = [], [], {}
    for model in models:
        train_features = [features(r["raw"], model)[0] for r in train]
        weights, bias = fit_readout(train_features, [r["label"] for r in train])
        fitted[model] = (weights, bias)
        labels, predicted, durations = [], [], []
        operations, input_spikes = 0, 0
        for recording in test:
            elapsed = []
            for _ in range(repeats):
                started = time.perf_counter_ns()
                row, network = features(recording["raw"], model)
                logit = score(row, weights, bias)
                elapsed.append((time.perf_counter_ns() - started) / 1000)
            durations.extend(elapsed)
            label = int(logit >= 0)
            labels.append(recording["label"])
            predicted.append(label)
            if network:
                operations += network.operations
                input_spikes += network.input_spikes
            predictions.append({"source": "synthetic_host", "model": model,
                                "recording_id": recording["recording_id"],
                                "generator_label": recording["label"], "prediction": label,
                                "logit": logit})
        ordered = sorted(durations)
        summary.append({"source": "synthetic_host", "model": model,
                        "test_recordings": len(test), **metrics(labels, predicted),
                        "host_median_us_per_recording": statistics.median(durations),
                        "host_p95_us_per_recording": ordered[math.ceil(0.95 * len(ordered)) - 1],
                        "input_contributions_evaluated": operations if model.startswith("snn_") else "",
                        "input_activity_fraction": input_spikes / (len(test) * TRACE_LENGTH * INPUTS)
                        if model.startswith("snn_") else ""})
    for name, rows in (("summary.csv", summary), ("predictions.csv", predictions)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {"source": "synthetic_host", "seed": SEED,
                "environment": {"python": sys.version, "platform": platform.platform()}, "sample_period_ms": 1,
                "decision_horizon_samples": TRACE_LENGTH, "synthetic_labels_only": True,
                "train_recordings": [r["recording_id"] for r in train],
                "test_recordings": [r["recording_id"] for r in test],
                "timing_repeats": repeats, "timing_scope": "host Python encoding + features + readout",
                "training": "fixed hidden layers; train-only normalized logistic readouts, 400 batch steps",
                "model_id": MODEL_ID, "fitted_readouts": fitted,
                "dataset_sha256": hashlib.sha256(json.dumps(recordings, sort_keys=True).encode()).hexdigest(),
                "limitations": ["No hardware measurement", "No physical slip data",
                                "Synthetic family discrimination is not slip accuracy",
                                "Host timing is not ESP32 timing or energy",
                                "Whole-recording decision is not incipient-slip detection latency"]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    weights, bias = fitted["snn_sparse"]
    fixture = next(r for r in test if r["label"] == 1)
    export_header(header, fixture, weights, bias)
    # Same logical model, evaluated over all held-out traces in both schedules.
    assert fitted["snn_dense"] == fitted["snn_sparse"]
    print(f"Synthetic host scaffold: {len(train)} train / {len(test)} test recordings.")
    print(f"Results: {output}; Arduino fixture: {header}")
    print("Generator-label scores do NOT establish physical gripper slip detection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, default=base / "neuromorphic" / "synthetic_results")
    parser.add_argument("--header", type=Path, default=base / "neuromorphic" / "synthetic_model.h")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    run(args.output, args.header, args.repeats)


if __name__ == "__main__":
    main()
