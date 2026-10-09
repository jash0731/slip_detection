"""Synthetic H1 scheduling sweep. Host timings cannot establish ESP32 speedups."""
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
import random
import statistics
import time

from .experiment import Network, SEED, WEIGHTS


def run(output=None):
    output = output or Path(__file__).parent / "synthetic_results" / "activity_sweep.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for activity in (0.0, 0.25, 0.5, 0.75, 1.0):
        # Exact number of fully active ticks, shuffled reproducibly. Thus average
        # activity is controlled without pretending to be tactile sensor data.
        raw = [1.2] * int(2048 * activity) + [0.0] * int(2048 * (1 - activity))
        random.Random(SEED).shuffle(raw)
        results = {}
        for sparse in (False, True):
            elapsed = []
            for _ in range(7):
                network = Network(sparse=sparse)
                started = time.perf_counter_ns()
                for sample in raw:
                    network.step(sample)
                elapsed.append((time.perf_counter_ns() - started) / 1000)
            results[sparse] = network
            rows.append({"source": "synthetic_host_activity_sweep", "input_activity_fraction": activity,
                         "schedule": "sparse" if sparse else "dense", "samples": len(raw),
                         "repeats": 7, "input_contributions_evaluated": network.operations,
                         "hidden_neuron_updates": len(raw) * 12,
                         "host_median_us": statistics.median(elapsed),
                         "host_min_us": min(elapsed), "host_max_us": max(elapsed),
                         "parity": "PASS"})
        assert results[False].counts == results[True].counts
        assert results[False].voltage == results[True].voltage
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output.parent / "activity_metadata.json").write_text(json.dumps({"source": "synthetic_host_activity_sweep", "hidden_kernel_id": "fixed_lif_hidden_seed20261007", "weights_sha256": hashlib.sha256(json.dumps(WEIGHTS).encode()).hexdigest(), "shared_hidden_weights": "same hidden kernel as both public-trained and synthetic-trained readout exports; readout not timed", "environment": {"python": sys.version, "platform": platform.platform()}, "timing_scope": "encoding and hidden update only; no readout; 7 repeats fixed dense-then-sparse order", "limitations": "synthetic activity and host Python timing; descriptive only, no board or energy claim"}, indent=2) + "\n")
    print(f"Synthetic activity sweep: exact dense/sparse parity; {output}")


if __name__ == "__main__":
    run()
