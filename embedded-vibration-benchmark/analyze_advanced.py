"""Validate actual ESP32 trained-model replay captures; reject host result files."""

import argparse
import csv
import io
import math
from pathlib import Path
import re
import statistics

FIELDS = "source run_id model_id model_hash input_id input_hash trial precision step input_prepare_us inference_us duration_us hidden_spikes final_score decision_threshold final_decision reference_decision decision_drift".split()
INT_FIELDS = "trial step input_prepare_us inference_us duration_us hidden_spikes final_decision reference_decision decision_drift".split()
PRECISIONS = ("float32", "int8_weights_float_state")


def validate_capture(text):
    lines = text.splitlines()
    if not any(line.startswith("# COMPLETE hardware_advanced_replay") for line in lines):
        raise ValueError("Missing advanced COMPLETE marker; preserve the full capture.")
    if not any(line.startswith("# fixture_parity=pass") for line in lines):
        raise ValueError("Missing fixture parity confirmation.")
    metadata = {}
    for line in lines:
        if line.startswith("#"):
            metadata.update(re.findall(r"\b(trials|samples_per_trace|models|precisions|hidden)=(\d+)", line))
    keys = ("trials", "samples_per_trace", "models", "precisions", "hidden")
    if any(key not in metadata or int(metadata[key]) <= 0 for key in keys):
        raise ValueError("Missing positive replay dimensions.")
    dimensions = {key: int(metadata[key]) for key in keys}
    if dimensions["precisions"] != 2:
        raise ValueError("Expected float and weight-quantized pairs.")
    reader = csv.DictReader(io.StringIO("\n".join(line for line in lines if line and not line.startswith("#"))))
    if reader.fieldnames != FIELDS:
        raise ValueError("Expected advanced replay schema, not other hardware/host CSV.")
    rows, groups, hashes, identity = [], {}, {}, None
    for raw in reader:
        if None in raw or any(value is None or value == "" for value in raw.values()):
            raise ValueError("Malformed/incomplete CSV row.")
        if raw["source"] != "hardware_advanced_replay":
            raise ValueError("Only hardware_advanced_replay rows are accepted.")
        row = dict(raw)
        try:
            for key in INT_FIELDS:
                row[key] = int(raw[key])
            for key in ("final_score", "decision_threshold"):
                row[key] = float(raw[key])
        except ValueError as error:
            raise ValueError("Invalid numeric field.") from error
        if any(row[key] < 0 for key in INT_FIELDS) or any(not math.isfinite(row[key]) for key in ("final_score", "decision_threshold")):
            raise ValueError("Negative count/duration or nonfinite score/threshold.")
        if row["precision"] not in PRECISIONS:
            raise ValueError("Unknown precision; int8 mode retains float states.")
        if any(not re.fullmatch(r"[0-9a-f]{64}", row[key]) for key in ("model_hash", "input_hash")):
            raise ValueError("Expected SHA256 model/input hashes.")
        current = row["run_id"], row["input_id"], row["input_hash"]
        if identity is None:
            identity = current
        if current != identity:
            raise ValueError("Mixed run/input identities.")
        previous_hash = hashes.setdefault(row["model_id"], row["model_hash"])
        if previous_hash != row["model_hash"]:
            raise ValueError("Model hash changed within one capture.")
        if not 1 <= row["trial"] <= dimensions["trials"] or row["step"] >= dimensions["samples_per_trace"] or row["hidden_spikes"] > dimensions["hidden"]:
            raise ValueError("Trial, step or spikes outside declared dimensions.")
        if row["duration_us"] != row["input_prepare_us"] + row["inference_us"]:
            raise ValueError("Duration must equal preparation plus inference.")
        if any(row[key] not in (0, 1) for key in ("final_decision", "reference_decision", "decision_drift")):
            raise ValueError("Decision flags must be binary.")
        if row["final_decision"] != int(row["final_score"] >= row["decision_threshold"]) or row["decision_drift"] != int(row["final_decision"] != row["reference_decision"]):
            raise ValueError("Decision/threshold/drift inconsistency.")
        if row["precision"] == "float32" and row["decision_drift"]:
            raise ValueError("Float decision failed the float reference.")
        key = row["model_id"], row["trial"], row["precision"]
        group = groups.setdefault(key, {})
        if row["step"] in group:
            raise ValueError("Duplicate replay step.")
        group[row["step"]] = row
        rows.append(row)
    if len(hashes) != dimensions["models"]:
        raise ValueError("Wrong number of model identities.")
    expected = {(model, trial, precision) for model in hashes for trial in range(1, dimensions["trials"] + 1) for precision in PRECISIONS}
    if set(groups) != expected:
        raise ValueError("Missing model/trial/precision combinations.")
    for group in groups.values():
        if set(group) != set(range(dimensions["samples_per_trace"])):
            raise ValueError("Missing steps.")
        for key in ("final_score", "decision_threshold", "final_decision", "reference_decision", "decision_drift"):
            if len({row[key] for row in group.values()}) != 1:
                raise ValueError("Final recording output changed within trace.")
    for model in hashes:
        references = set()
        thresholds = set()
        for trial in range(1, dimensions["trials"] + 1):
            for precision in PRECISIONS:
                row = groups[model, trial, precision][0]
                references.add(row["reference_decision"])
                thresholds.add(row["decision_threshold"])
        if len(references) != 1 or len(thresholds) != 1:
            raise ValueError("Reference/threshold changed across paired replay.")
    return rows, dimensions


def summarize(rows):
    result = []
    for model in sorted({row["model_id"] for row in rows}):
        for precision in PRECISIONS:
            selected = [row for row in rows if row["model_id"] == model and row["precision"] == precision]
            durations = sorted(row["duration_us"] for row in selected)
            trace_rows = [row for row in selected if row["step"] == 0]
            result.append({"model": model, "precision": precision,
                           "median_step_us": statistics.median(durations),
                           "p95_step_us": durations[math.ceil(.95 * len(durations)) - 1],
                           "max_step_us": max(durations), "traces": len(trace_rows),
                           "drifted_traces": sum(row["decision_drift"] for row in trace_rows)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        rows, dimensions = validate_capture(args.capture.read_text())
    except (ValueError, OSError) as error:
        parser.exit(2, f"Capture rejected: {error}\n")
    print(f"Validated {len(rows)} board rows for {dimensions['models']} models.")
    for row in summarize(rows):
        print(f"{row['model']} {row['precision']}: median/p95/max step {row['median_step_us']}/{row['p95_step_us']}/{row['max_step_us']} us; drifted repeats {row['drifted_traces']}/{row['traces']}.")
    print("Prepared-feature replay only; repeated fixture drift is not population accuracy. Readout timing is in trace metadata; no physical slip latency or energy measurement.")


if __name__ == "__main__":
    main()
