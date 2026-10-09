"""Plot existing software measurements without creating hardware evidence."""

import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "software_results"


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith("#")))


def main():
    OUT.mkdir(exist_ok=True)
    public = read_csv(ROOT / "neuromorphic/public_results/summary.csv")
    if not public or any(row["source"] != "public_recorded_host" for row in public):
        raise ValueError("Expected public-recorded host results.")
    lookup = {row["model"]: row for row in public}
    models = ("threshold", "ema", "scalar_lif", "ann", "snn_sparse")
    labels = ("Threshold", "EMA", "Scalar LIF", "Fixed ANN\n+ readout", "Fixed SNN\n+ readout")
    values = [100 * float(lookup[model]["balanced_accuracy"]) for model in models]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=["#416789"] * 4 + ["#ae5728"], width=0.65)
    ax.axhline(50, color="#888888", linewidth=1, linestyle="--")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Recorded gripper tactile data: exploratory host classification", pad=16)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=5)
    fig.text(0.11, 0.025, "Brush only; pose 3 held out; 55 homogeneous windows / 9 trials.\n"
             "Dependent windows; onset transitions excluded; no incipient-slip or ESP32 accuracy claim.", fontsize=9)
    fig.subplots_adjust(bottom=0.23, top=0.87)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"public_quality.{suffix}", dpi=180)
    plt.close(fig)

    host_path = OUT / "host_cpp_activity.csv"
    rows = read_csv(host_path)
    if len(rows) != 100 or any(row["source"] != "host_reference" for row in rows):
        raise ValueError("Expected 100 host-reference activity rows; never board captures.")
    pairs = {}
    for row in rows:
        key = int(row["active_inputs"]), int(row["trial"])
        pair = pairs.setdefault(key, {})
        if row["mode"] in pair or int(row["steps"]) != 200000 or int(row["input_channels"]) != 4:
            raise ValueError("Unexpected duplicate/dimensions in host benchmark.")
        pair[row["mode"]] = row
    if set(pairs) != {(active, trial) for active in range(5) for trial in range(1, 11)}:
        raise ValueError("Incomplete activity pairs.")
    for pair in pairs.values():
        if set(pair) != {"dense", "sparse"} or abs(float(pair["dense"]["checksum"]) - float(pair["sparse"]["checksum"])) > 1e-5:
            raise ValueError("Host activity checksum parity failed.")
    medians = {mode: [statistics.median(int(pairs[active, trial][mode]["duration_ns"]) / 200000 for trial in range(1, 11)) for active in range(5)] for mode in ("dense", "sparse")}
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, color in (("dense", "#416789"), ("sparse", "#ae5728")):
        ax.plot([0, 25, 50, 75, 100], medians[mode], marker="o", label=mode.capitalize(), color=color)
    ax.set_xlabel("Active encoder channels (%)")
    ax.set_ylabel("Median host time per hidden-layer tick (ns)")
    ax.set_title("Host C++ activity sweep — not ESP32 measurements", pad=16)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.text(0.11, 0.025, "Apple Silicon / Clang -O2; 10 paired repetitions × 200,000 ticks per condition.\n"
             "Constant synthetic input; encoding/readout excluded. Curves are host-specific; no energy claim.", fontsize=9)
    fig.subplots_adjust(bottom=0.23, top=0.87)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"host_activity.{suffix}", dpi=180)
    plt.close(fig)
    metadata = {"source": "host_reference", "platform": platform.platform(),
                "compiler": subprocess.check_output(["c++", "--version"], text=True).strip(),
                "flags": "-std=c++11 -O2 -Wall -Wextra -pedantic",
                "repetitions": 10, "ticks_per_condition": 200000,
                "model_id": rows[0]["model_id"],
                "raw_sha256": hashlib.sha256(host_path.read_bytes()).hexdigest(),
                "benchmark_source_sha256": hashlib.sha256((ROOT / "cpp/activity_benchmark.cpp").read_bytes()).hexdigest(),
                "model_header_sha256": hashlib.sha256((ROOT / "firmware/esp32_snn_replay/generated_model.h").read_bytes()).hexdigest(),
                "median_ns_per_tick": medians,
                "timing_scope": "hidden updates only; excludes input encoding and readout",
                "evidence_limit": "Host/compiler-specific controlled synthetic workload; no ESP32 or slip-onset measurement"}
    (OUT / "host_cpp_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Plotted public host quality and host C++ activity; hardware measurements remain pending.")


if __name__ == "__main__":
    main()
