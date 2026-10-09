"""Validate and summarize actual ESP32 replay captures, never host results."""

import argparse
import csv
import io
import math
from pathlib import Path
import re
import statistics

FIELDS = "source run_id model_id input_id trial mode pacing step active_inputs hidden_spikes encode_us inference_us duration_us start_lateness_us deadline_miss final_score final_decision".split()
INT_FIELDS = "trial step active_inputs hidden_spikes encode_us inference_us duration_us start_lateness_us deadline_miss final_decision".split()


def validate_capture(text):
    lines = text.splitlines()
    if not any(line.startswith("# COMPLETE.") for line in lines):
        raise ValueError("Missing COMPLETE marker: preserve the full serial capture.")
    metadata = {}
    for line in lines:
        if line.startswith("#"):
            metadata.update(re.findall(r"\b(trials|samples_per_trace|inputs|hidden|nominal_interval_us)=(\d+)", line))
    required = ("trials", "samples_per_trace", "inputs", "hidden", "nominal_interval_us")
    if any(key not in metadata or int(metadata[key]) <= 0 for key in required):
        raise ValueError("Missing positive replay dimensions/pacing metadata.")
    dimensions = {key: int(metadata[key]) for key in required}
    reader = csv.DictReader(io.StringIO("\n".join(line for line in lines if line and not line.startswith("#"))))
    if reader.fieldnames != FIELDS:
        raise ValueError("Expected ESP32 replay schema; host and GPIO result files are different.")
    rows, groups, identity = [], {}, None
    for raw in reader:
        if None in raw or any(value is None or value == "" for value in raw.values()):
            raise ValueError("Malformed or incomplete CSV row.")
        if raw["source"] != "hardware_replay":
            raise ValueError("Only hardware_replay captures are accepted; no host/synthetic timing.")
        row = dict(raw)
        try:
            for key in INT_FIELDS:
                row[key] = int(raw[key])
            row["final_score"] = float(raw["final_score"])
        except ValueError as error:
            raise ValueError("Invalid numeric replay field.") from error
        if not math.isfinite(row["final_score"]) or any(row[key] < 0 for key in INT_FIELDS):
            raise ValueError("Negative count/duration or nonfinite score.")
        if row["mode"] not in ("dense", "sparse") or row["pacing"] not in ("throughput", "paced"):
            raise ValueError("Unknown mode/pacing.")
        current = tuple(row[key] for key in ("run_id", "model_id", "input_id", "pacing"))
        if identity is None:
            identity = current
        if current != identity:
            raise ValueError("Mixed runs, models, inputs or pacing in one capture.")
        if not 1 <= row["trial"] <= dimensions["trials"] or row["step"] >= dimensions["samples_per_trace"]:
            raise ValueError("Trial/step outside declared dimensions.")
        if row["active_inputs"] > dimensions["inputs"] or row["hidden_spikes"] > dimensions["hidden"]:
            raise ValueError("Spike count outside declared dimensions.")
        if row["duration_us"] != row["encode_us"] + row["inference_us"]:
            raise ValueError("duration_us must equal encoding plus inference duration.")
        if row["deadline_miss"] not in (0, 1) or row["final_decision"] != int(row["final_score"] >= 0):
            raise ValueError("Invalid deadline flag or score/decision mismatch.")
        expected_miss = int(row["pacing"] == "paced" and row["start_lateness_us"] + row["duration_us"] > dimensions["nominal_interval_us"])
        if row["deadline_miss"] != expected_miss:
            raise ValueError("Deadline flag inconsistent with declared interval and measured duration.")
        if row["pacing"] == "throughput" and row["start_lateness_us"] != 0:
            raise ValueError("Throughput rows cannot report scheduled-start lateness.")
        key = row["trial"], row["mode"]
        group = groups.setdefault(key, {})
        if row["step"] in group:
            raise ValueError("Duplicate replay step.")
        group[row["step"]] = row
        rows.append(row)
    expected_keys = {(trial, mode) for trial in range(1, dimensions["trials"] + 1) for mode in ("dense", "sparse")}
    if set(groups) != expected_keys:
        raise ValueError("Incomplete dense/sparse trial pairs.")
    for group in groups.values():
        if set(group) != set(range(dimensions["samples_per_trace"])):
            raise ValueError("Missing replay steps.")
        scores = {row["final_score"] for row in group.values()}
        if len(scores) != 1:
            raise ValueError("Final recording score changed within a trace.")
    for trial in range(1, dimensions["trials"] + 1):
        for step in range(dimensions["samples_per_trace"]):
            dense, sparse = groups[trial, "dense"][step], groups[trial, "sparse"][step]
            if any(dense[key] != sparse[key] for key in ("active_inputs", "hidden_spikes", "final_decision")) or abs(dense["final_score"] - sparse["final_score"]) > 1e-5:
                raise ValueError("Dense/sparse recorded output parity failed.")
    return rows, dimensions


def percentile95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def summarize(rows):
    summaries = []
    for mode in ("dense", "sparse"):
        selected = [row for row in rows if row["mode"] == mode]
        durations = [row["duration_us"] for row in selected]
        summaries.append({"source": "hardware_replay", "mode": mode,
                          "steps": len(selected), "median_step_us": statistics.median(durations),
                          "p95_step_us": percentile95(durations), "max_step_us": max(durations),
                          "median_encoding_us": statistics.median(row["encode_us"] for row in selected),
                          "median_inference_us": statistics.median(row["inference_us"] for row in selected),
                          "deadline_misses": sum(row["deadline_miss"] for row in selected),
                          "pacing": selected[0]["pacing"]})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        rows, dimensions = validate_capture(args.capture.read_text())
    except (ValueError, OSError) as error:
        parser.exit(2, f"Capture rejected: {error}\n")
    print(f"Validated {len(rows)} hardware_replay rows; {dimensions['trials']} paired trials.")
    for summary in summarize(rows):
        print(f"{summary['mode']}: median/p95/max step {summary['median_step_us']}/{summary['p95_step_us']}/{summary['max_step_us']} us; deadline misses {summary['deadline_misses']} ({summary['pacing']}).")
    print("Replay compute only. This is not sensor acquisition or physical slip delay; no energy measurement.")


if __name__ == "__main__":
    main()
