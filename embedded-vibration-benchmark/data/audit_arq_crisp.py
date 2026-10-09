"""Audit public uSkin recordings; this is not an ESP32 or incipient-slip result."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def audit(path):
    trials = []
    with h5py.File(path, "r") as hf:
        def visit(name, item):
            if not isinstance(item, h5py.Group) or "tactile_data_raw" not in item:
                return
            raw = np.asarray(item["tactile_data_raw"])
            labels = np.asarray(item["tactile_slips_label"])
            ts = np.asarray(item["tactile_timestamps"])
            if raw.shape != (18, 3, len(labels)) or ts.shape != labels.shape:
                raise ValueError(f"Unexpected shape in {name}")
            dt = np.diff(ts)
            if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(ts)):
                raise ValueError(f"Non-finite data in {name}")
            values, counts = np.unique(labels, return_counts=True)
            slip = np.isin(labels, [1, 5])
            onset = np.flatnonzero(slip & ~np.r_[False, slip[:-1]])
            trials.append({"trial_id": name, "shape": list(raw.shape),
                "samples": len(ts), "duration_ms": float(ts[-1]-ts[0]),
                "median_dt_ms": float(np.median(dt)), "min_dt_ms": float(dt.min()),
                "max_dt_ms": float(dt.max()), "nonpositive_dt_count": int((dt<=0).sum()),
                "label_counts": {str(int(v)): int(c) for v,c in zip(values,counts)},
                "slip_run_count": len(onset), "slip_run_start_indices": onset.tolist()})
        hf.visititems(visit)
    return {"evidence_type": "public_hardware_recording_not_our_hardware",
        "source_file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trial_count": len(trials), "samples": sum(t["samples"] for t in trials),
        "trials": trials}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hdf5", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = json.dumps(audit(args.hdf5), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
