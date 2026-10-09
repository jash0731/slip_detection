# Public robotic-gripper recordings

This directory contains source audits and preparation tools. The downloaded HDF5
and derived NPZ are local, ignored files: reproduce them from the pinned source.
The recordings were collected by the cited authors on real robotic hardware;
they are not our ESP32 measurements, simulations, or piezo recordings.

## Current three-object experiment

```sh
python3 data/fetch_arq_all.py
python3 data/prepare_arq_all.py
```

These verify all three pinned HDF5 payloads and prepare `arq_all_trials.npz`.
See [ADVANCED_DATA_REVIEW.md](../ADVANCED_DATA_REVIEW.md) for its distinct NPZ
interface, source counts and independent split audit. The commands and table
below retain the earlier brush-only preparation.

## Earlier brush-only experiment

From the project root, with Python, NumPy and h5py available:

```sh
python3 data/fetch_arq_crisp.py
python3 data/audit_arq_crisp.py data/public/ARQ_CRISP_2021/slipDataset_brush_tactile.h5 --output data/public/ARQ_CRISP_2021/audit.json
python3 data/prepare_arq_crisp.py
```

The fetcher downloads 41,105,856 bytes and verifies the upstream Git-LFS SHA256.
Source notices and README files are retained under `public/ARQ_CRISP_2021`.
See [DATA_REVIEW.md](../DATA_REVIEW.md) for label limitations and citation.

## NPZ interface

Load `arq_brush_trials.npz` with `numpy.load(..., allow_pickle=False)`:

| Key | Meaning |
| --- | --- |
| `x` | float32 `(41707,54)` raw values: pin0 x/y/z, pin1 x/y/z, ... |
| `y` | int8 per sample: 0 no-slip, 1 perceived slip, -1 excluded manipulation/other |
| `y_held` | Preferred primary labels: 0 for upstream4, 1 for upstream5, -1 otherwise |
| `original_label` | Original upstream integer label retained for audit |
| `t_ms` | Original timestamps shifted to zero separately for each trial |
| `group` | Integer recording identifier; do not split adjacent samples across partitions |
| `pose` | Grasp pose 1, 2, or 3; useful for a held-out-pose test |
| `trial_names` | Lookup table for the 30 recording paths |
| `source_sha256` | SHA256 of original HDF5 |
| `evidence_type` | `public_hardware_recording_not_our_hardware` |

Values are not normalized. Fit any scaling, encoder thresholds and model
parameters on training recordings only. Reset all causal filters/neuron state
at recording boundaries; retain ignored samples as temporal context if needed,
but exclude their labels from fitting and scoring. In particular, do not remove
ignored samples and pretend the remaining rows form an uninterrupted timeline.

A tiny firmware trace can be selected from an explicitly held-out recording
later. Declare any subsampling, feature reduction or time compression: each
changes the demonstrated workload. Grasp trial replay is a suitable software
hardware demo without a physical gripper; it does not demonstrate closed-loop
grip adjustment or validate sensor transfer.
