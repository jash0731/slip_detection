"""Logic-only pre-hardware exercise. Output is never a measured ESP32 result."""

import csv
import math
import random
from pathlib import Path

OUT = Path(__file__).with_name("simulation_results.csv")
RNG = random.Random(20261002)


def polling_first_detection(pulses, interval_ms, phase_ms):
    """Return first sample falling inside a [start, end) digital pulse."""
    for start, end in pulses:
        n = max(0, math.ceil((start - phase_ms) / interval_ms))
        sample = phase_ms + n * interval_ms
        if start <= sample < end:
            return sample
    return None


def counter_first_detection(events, count=3, window_ms=30):
    for idx in range(count - 1, len(events)):
        if events[idx] - events[idx - count + 1] <= window_ms:
            return events[idx]
    return None


def lif_first_detection(events, tau_ms=20, threshold=2.2):
    voltage = 0.0
    previous = None
    for event in events:
        if previous is not None:
            voltage *= math.exp(-(event - previous) / tau_ms)
        voltage += 1.0
        previous = event
        if voltage >= threshold:
            return event
    return None


def main():
    rows = []
    for trial in range(100):
        first = 100 + RNG.uniform(0, 20)
        width = RNG.choice((1, 3, 8, 15))
        pulses = [(first, first + width)]
        phase = RNG.uniform(0, 20)
        for interval in (1, 5, 10, 20):
            decision = polling_first_detection(pulses, interval, phase % interval)
            rows.append({"experiment": "acquisition", "trial": trial,
                         "kind": "single_pulse", "method": f"poll_{interval}ms",
                         "first_event_ms": round(first, 4),
                         "decision_ms": "" if decision is None else round(decision, 4),
                         "latency_ms": "" if decision is None else round(decision - first, 4),
                         "outcome": "miss" if decision is None else "detected"})
        # Zero models ideal event observation; real ISR and GPIO latency must be measured.
        rows.append({"experiment": "acquisition", "trial": trial,
                     "kind": "single_pulse", "method": "ideal_event_observation",
                     "first_event_ms": round(first, 4), "decision_ms": round(first, 4),
                     "latency_ms": 0, "outcome": "detected"})

    patterns = {
        "target_burst": [100, 106, 112, 118],
        "noise_isolated": [100],
        "noise_sparse": [100, 160, 220],
    }
    detectors = {
        "immediate": lambda events: events[0] if events else None,
        "counter_3_in_30ms": counter_first_detection,
        "lif_tau20_threshold2.2": lif_first_detection,
    }
    for kind, events in patterns.items():
        for name, detector in detectors.items():
            decision = detector(events)
            outcome = ("detected" if decision is not None else "miss") if kind == "target_burst" else ("false_alarm" if decision is not None else "correct_reject")
            rows.append({"experiment": "detector_logic", "trial": 0,
                         "kind": kind, "method": name,
                         "first_event_ms": events[0],
                         "decision_ms": "" if decision is None else decision,
                         "latency_ms": "" if decision is None else decision - events[0],
                         "outcome": outcome})

    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} SIMULATED rows to {OUT}")
    for row in rows[-9:]:
        print(row["kind"], row["method"], row["outcome"], row["latency_ms"])


if __name__ == "__main__":
    main()
