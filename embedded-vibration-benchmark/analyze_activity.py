"""Validate ESP32 execution captures for controlled synthetic activity workloads."""

import argparse
import csv
import io
import math
from pathlib import Path
import re
import statistics

FIELDS = "source run_id model_id trial mode active_inputs input_channels steps duration_us input_contributions hidden_updates checksum".split()
INT_FIELDS = "trial active_inputs input_channels steps duration_us input_contributions hidden_updates".split()


def validate_capture(text):
    lines = text.splitlines()
    if not any(line.startswith("# COMPLETE hardware_activity_sweep") for line in lines):
        raise ValueError("Missing activity COMPLETE marker; preserve the full capture.")
    if not any(line.startswith("# activity_state_parity=pass") for line in lines):
        raise ValueError("Missing board state-parity confirmation.")
    metadata = {}
    for line in lines:
        if line.startswith("#"):
            metadata.update(re.findall(r"\b(trials|steps|conditions|inputs|hidden)=(\d+)", line))
    keys = ("trials", "steps", "conditions", "inputs", "hidden")
    if any(key not in metadata or int(metadata[key]) <= 0 for key in keys):
        raise ValueError("Missing positive activity dimensions.")
    dimensions = {key: int(metadata[key]) for key in keys}
    if dimensions["conditions"] != dimensions["inputs"] + 1:
        raise ValueError("Expected every active-channel level, including zero.")
    reader = csv.DictReader(io.StringIO("\n".join(line for line in lines if line and not line.startswith("#"))))
    if reader.fieldnames != FIELDS:
        raise ValueError("Expected the activity schema, not replay/GPIO/host results.")
    pairs, rows, identity = {}, [], None
    for raw in reader:
        if None in raw or any(value is None or value == "" for value in raw.values()):
            raise ValueError("Malformed/incomplete activity row.")
        if raw["source"] != "hardware_activity_sweep":
            raise ValueError("Expected actual hardware_activity_sweep rows, not host timings.")
        row = dict(raw)
        try:
            for key in INT_FIELDS:
                row[key] = int(raw[key])
            row["checksum"] = float(raw["checksum"])
        except ValueError as error:
            raise ValueError("Invalid numeric activity field.") from error
        if not math.isfinite(row["checksum"]) or any(row[key] < 0 for key in INT_FIELDS) or row["duration_us"] <= 0:
            raise ValueError("Nonfinite checksum or invalid count/duration.")
        if row["mode"] not in ("dense", "sparse"):
            raise ValueError("Unknown accumulation mode.")
        current = row["run_id"], row["model_id"]
        if identity is None:
            identity = current
        if current != identity:
            raise ValueError("Mixed run/model identities.")
        if not 1 <= row["trial"] <= dimensions["trials"] or row["active_inputs"] >= dimensions["conditions"]:
            raise ValueError("Trial/activity outside declared range.")
        if row["steps"] != dimensions["steps"] or row["input_channels"] != dimensions["inputs"]:
            raise ValueError("Workload dimensions do not match metadata.")
        expected_updates = row["steps"] * dimensions["hidden"]
        expected_contributions = expected_updates * (row["input_channels"] if row["mode"] == "dense" else row["active_inputs"])
        if row["hidden_updates"] != expected_updates or row["input_contributions"] != expected_contributions:
            raise ValueError("Operation counts do not match the declared algorithm.")
        key = row["active_inputs"], row["trial"], row["mode"]
        if key in pairs:
            raise ValueError("Duplicate activity row.")
        pairs[key] = row
        rows.append(row)
    expected = {(active, trial, mode) for active in range(dimensions["conditions"]) for trial in range(1, dimensions["trials"] + 1) for mode in ("dense", "sparse")}
    if set(pairs) != expected:
        raise ValueError("Incomplete activity sweep or dense/sparse pairs.")
    for active in range(dimensions["conditions"]):
        for trial in range(1, dimensions["trials"] + 1):
            if abs(pairs[active, trial, "dense"]["checksum"] - pairs[active, trial, "sparse"]["checksum"]) > 1e-5:
                raise ValueError("Paired output checksum mismatch.")
    return rows, dimensions


def summarize(rows):
    summaries = []
    for active in sorted({row["active_inputs"] for row in rows}):
        result = {"source": "hardware_activity_sweep", "active_inputs": active}
        for mode in ("dense", "sparse"):
            group = [row for row in rows if row["active_inputs"] == active and row["mode"] == mode]
            means = [row["duration_us"] / row["steps"] for row in group]
            result[f"{mode}_median_batch_mean_us_per_tick"] = statistics.median(means)
        result["sparse_dense_ratio"] = result["sparse_median_batch_mean_us_per_tick"] / result["dense_median_batch_mean_us_per_tick"]
        summaries.append(result)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        rows, dimensions = validate_capture(args.capture.read_text())
    except (ValueError, OSError) as error:
        parser.exit(2, f"Capture rejected: {error}\n")
    print(f"Validated {len(rows)} board rows; {dimensions['trials']} paired repeats/activity.")
    for row in summarize(rows):
        print(f"active {row['active_inputs']}/{dimensions['inputs']}: dense/sparse median batch mean {row['dense_median_batch_mean_us_per_tick']:.4f}/{row['sparse_median_batch_mean_us_per_tick']:.4f} us/tick; ratio {row['sparse_dense_ratio']:.3f}.")
    print("Measured board execution on synthetic constant activity; hidden updates only. No physical slip accuracy or energy measurement.")


if __name__ == "__main__":
    main()
