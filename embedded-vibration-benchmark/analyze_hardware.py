"""Validate/summarize Experiment A serial captures; never generate measurements."""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

COLUMNS = (
    "run_id,trial_id,trial_kind,acquisition,detector,poll_interval_ms,core_config,"
    "load,parameters,pulse_schedule_us,first_pulse_us,decision_us,latency_us,"
    "outcome,notes"
).split(",")
METHODS = [("polling", str(ms)) for ms in (1, 5, 10, 20)] + [("interrupt", "")]
WIDTHS = (250, 1000, 3000, 8000, 15000)
PHASES = tuple(i * 1000 + (i * 347 + 137) % 1000 for i in range(20))


def key_values(value):
    return dict(part.split("=", 1) for part in value.split(";") if "=" in part)


def read_capture(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(line for line in handle if line.strip() and not line.startswith("#"))
        if reader.fieldnames != COLUMNS:
            raise ValueError("expected EXPERIMENT.md hardware CSV header; simulation CSV is not accepted")
        rows = list(reader)
    if not rows:
        raise ValueError("capture contains no trial rows")
    groups = defaultdict(list)
    run_ids = set()
    trial_ids = set()
    for row_number, row in enumerate(rows, start=2):
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("row does not have exactly 15 fields")
            notes = key_values(row["notes"])
            if not row["run_id"].startswith("hardware-") or notes.get("source") != "hardware":
                raise ValueError("row is not labeled as a hardware capture")
            if notes.get("stimulus") != "gpio_loopback":
                raise ValueError("expected gpio_loopback stimulus")
            run_ids.add(row["run_id"])
            trial_id = int(row["trial_id"])
            if trial_id in trial_ids:
                raise ValueError("duplicate trial_id")
            trial_ids.add(trial_id)
            method = (row["acquisition"], row["poll_interval_ms"])
            if method not in METHODS or row["detector"] != "immediate" or row["trial_kind"] != "target":
                raise ValueError("expected Experiment A target/immediate acquisition settings")
            params = key_values(row["parameters"])
            width = int(params["width_us"])
            phase = int(params["phase_us"])
            deadline = int(params["deadline_us"])
            epoch = int(params["epoch_us"])
            requested_rise = int(params["requested_rise_us"])
            if width not in WIDTHS or phase not in PHASES:
                raise ValueError("unexpected pulse width/phase")
            if requested_rise != epoch + 40000 + phase or deadline != requested_rise + width + 40000:
                raise ValueError("inconsistent planned schedule/deadline")
            first = int(row["first_pulse_us"])
            schedule = json.loads(row["pulse_schedule_us"])
            if (not isinstance(schedule, list) or len(schedule) != 1 or
                    not isinstance(schedule[0], list) or len(schedule[0]) != 2 or
                    any(type(t) is not int for t in schedule[0])):
                raise ValueError("pulse_schedule_us must be [[integer rise, integer fall]]")
            if schedule[0][0] != first or first < requested_rise or schedule[0][1] <= first:
                raise ValueError("invalid actual pulse timestamps")
            if row["outcome"] == "miss":
                if row["decision_us"] or row["latency_us"]:
                    raise ValueError("miss must retain empty decision and latency")
            elif row["outcome"] in ("detected", "false_alarm"):
                decision = int(row["decision_us"])
                latency = int(row["latency_us"])
                if latency != decision - first or not epoch <= decision < deadline:
                    raise ValueError("decision/latency is inconsistent or outside the deadline")
                if (row["outcome"] == "detected") != (decision >= first):
                    raise ValueError("pre-stimulus decisions must be false_alarm")
            else:
                raise ValueError("unexpected outcome for a target trial")
            groups[method].append(row)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise ValueError(f"CSV row {row_number}: {error}") from error
    if len(run_ids) != 1:
        raise ValueError("capture must contain one run_id; summarize runs separately")
    if trial_ids != set(range(1, 501)):
        raise ValueError(f"incomplete run: expected unique trial IDs 1..500, got {len(trial_ids)}")
    for method in METHODS:
        trials = groups[method]
        combinations = {(int(key_values(row["parameters"])["width_us"]),
                         int(key_values(row["parameters"])["phase_us"])) for row in trials}
        if len(trials) != 100 or combinations != {(w, p) for w in WIDTHS for p in PHASES}:
            raise ValueError(f"{method}: expected each of 100 width/phase combinations exactly once")
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["acquisition"], row["poll_interval_ms"])].append(row)
    summaries = []
    for acquisition, interval in METHODS:
        trials = groups[(acquisition, interval)]
        latencies = sorted(int(row["latency_us"]) for row in trials if row["outcome"] == "detected")
        misses = sum(row["outcome"] == "miss" for row in trials)
        alarms = sum(row["outcome"] == "false_alarm" for row in trials)
        timing_flags = sum(
            key_values(row["notes"]).get("stimulus_after_deadline") == "1" or
            int(key_values(row["notes"]).get("skipped_polls", "0")) > 0
            for row in trials
        )
        summaries.append({
            "method": f"poll_{interval}ms" if interval else acquisition,
            "trials": len(trials), "detected": len(latencies), "misses": misses,
            "miss_rate": misses / len(trials), "pre_stimulus_alarms": alarms,
            "timing_flagged_trials": timing_flags,
            "median_us": statistics.median(latencies) if latencies else "",
            "p95_us": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else "",
            "max_us": max(latencies) if latencies else "",
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="complete ESP32 hardware serial CSV (500 rows)")
    args = parser.parse_args()
    try:
        rows = read_capture(args.capture)
        summaries = summarize(rows)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"HARDWARE electrical loopback: {rows[0]['run_id']}")
    writer = csv.DictWriter(sys.stdout, fieldnames=summaries[0].keys())
    writer.writeheader()
    writer.writerows(summaries)
    print("Latencies use detected trials only; p95 uses nearest rank. No trials were dropped.")
    print("Inspect raw timing flags/lateness. No noise-only trials: no general false-alarm rate.")


if __name__ == "__main__":
    main()
