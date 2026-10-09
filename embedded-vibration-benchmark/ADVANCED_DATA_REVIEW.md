# All-object data and evaluation audit

This extension keeps robotic-gripper slip detection as the application, using
public uSkin/EZGripper recordings replayed in software. It supports a more
credible cross-object classification experiment than the brush-only scaffold.
It does not establish incipient-slip prediction, sensor transfer to piezo, or
closed-loop grip adjustment. No physical gripper or new component is needed for
the planned ESP32 replay demo.

## Reproducible source and preparation

Use the authors' [ARQ-CRISP 2021 dataset](https://github.com/ARQ-CRISP/slip_detection_dataset_2021)
at commit `a536ee3301af0566218782b00de334b6a6630979`. The repository's
BSD-3-Clause license and original documentation are retained in
`data/public/ARQ_CRISP_2021/`. Cite Zenha et al., *Tactile Slip Detection in the
Wild Leveraging Distributed Sensing of both Normal and Shear Forces*,
[IROS 2021](https://doi.org/10.1109/IROS51168.2021.9636602).

Run from the project root with Python, NumPy and h5py:

```sh
python3 data/fetch_arq_all.py
python3 data/prepare_arq_all.py
```

The fetcher verifies each actual Git-LFS payload against its pinned pointer's
SHA256 and byte count; merely downloading the raw pointer is insufficient.
It downloads about 266 MB total, reusing already verified files. Preparation
independently rechecks hashes, shapes, finite data, recognized labels and strictly
increasing timestamps, then writes `data/arq_all_trials.npz` and an audited
`data/arq_all_manifest.json`. Binary HDF5/NPZ files are ignored by Git and
reproducible. The manifest retains source URLs, hashes and trial-level counts.

Load NPZ with `allow_pickle=False`. Keys:

| Key | Meaning |
| --- | --- |
| `x` | Raw float32 rows with 54 channels, pin0 x/y/z through pin17 x/y/z |
| `original_label` | Original integer labels 0–6 |
| `y_held` | 0 for label4; 1 for label5; -1 otherwise |
| `t_ms` | Original milliseconds, shifted separately to zero for every trial |
| `group` | Globally unique trial identifier |
| `object` | 0 brush, 1 screwDriver, 2 SpoolSolder |
| `pose` | Grasp pose 1, 2 or 3 |
| `trial_names` / `object_names` | String lookup tables |
| `source_sha256` | Three hashes in object order |
| `source_commit` / `evidence_type` | Pinned provenance; public hardware recordings, not our board results |

## Verified all-object audit

All three actual HDF5 files were downloaded and checksum-verified. Preparation
succeeded with 90 trial groups and 228,521 finite 54-channel samples. All original
samples, including ignored labels, remain in order; no trial has nonpositive time
increments. The audited manifest records every trial separately.

| Object | Samples | Label4 held non-slip | Label5 held slip | Trials with label4 / label5 | Frame interval range, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| brush | 41,707 | 8,618 | 5,588 | 10 / 28 | 0.46–15.33 |
| screwDriver | 92,660 | 22,304 | 19,125 | 26 / 27 | 0.66–19.80 |
| SpoolSolder | 94,154 | 30,443 | 10,122 | 24 / 25 | 0.46–20.63 |

The asymmetry in negative-trial availability is especially relevant when brush
is the held-out object. Different object distributions and sequence lengths make
pooled frame accuracy insufficient. Retain per-object folds and paired grasp-trial
uncertainty. Nominal sampling frequency does not eliminate the observed jitter.

## Evaluation rules

The [authors' label documentation](https://github.com/ARQ-CRISP/slip_detection_dataset_2021/blob/main/data/README.md)
identifies labels4/5 as no-slip/slip while a held object moves with the robot arm.
Use this primary task to reduce free-space versus contact shortcuts. Labels0/1
involve a static arm, and label0 also includes not-held samples; labels2/3/6
represent other manipulation events. Retain all samples as temporal context,
but do not score or train supervised losses on those other labels.

A causal window ending at sample t uses inputs t-W+1 through t and its label at
t, never a majority label or a requirement that the entire window has one class.
Do not remove ignored rows and concatenate the remaining time points. Report
whether neuron state resets for each window or persists across a whole recording:
both are valid protocols, but their scores and computational costs differ.
Observation horizons and repeated classifications of overlapping windows are not
independent samples or physical response latency.

For each of three outer folds, keep one entire object for test. Divide the other
two objects into disjoint whole-trial training and validation partitions.
Fit scaling/encoder thresholds only on the training partition; select epochs,
decision thresholds, model type and hyperparameters only using validation.
Do not select a checkpoint, quantizer, demo trace or winning method using outer
test accuracy. Document split identifiers and their disjointness in metadata.
Comparing all three outer folds is exploratory evidence on these three objects,
not an estimate for arbitrary unseen materials or grippers.

Use the same causal input summaries, horizon and split for conventional and
spiking classifiers. Training a SNN with surrogate gradients is a stronger model
experiment than the earlier fixed hidden LIF bank, but provides no automatic
novelty or energy advantage. Compare matched-capacity ANN/LIF/ALIF candidates and
include simple baselines. Any quantization experiment must distinguish trained
weights, neuron state arithmetic and deployment parity.

Frame/window metrics overrepresent long recordings. Report per-object outcomes
and uncertainty obtained by resampling whole grasp trials, preferably paired
model differences on the same trials. A single-class resampled trial set makes
balanced accuracy undefined; record such omissions rather than assigning perfect
scores. Three objects limit object-level statistical claims even with 90 grasps.

Manual perceived-slip annotations provide no independently calibrated first
incipient-slip instant. Sampling jitter and preprocessing observation time also
matter. ESP32 kernel timing on exported inputs cannot establish sub-5 ms physical
detection, full-pipeline energy efficiency or closed-loop grip success. Prefix
all evidence as public-recorded host, synthetic host, or measured board replay.

## Hardware demo selection

Export a deterministic segment from an explicitly held-out object's recording,
with original label/trace indices and the model checkpoint's split recorded.
Do not screen segments by whether a model predicts correctly. One fixture is a
numerical parity and timing demonstration, not a test-set accuracy result.
Include both low- and high-activity controlled replay when comparing dense and
sparse schedules, and preserve equivalent state updates. Precomputed derivative
features in flash mean timing excludes sensor acquisition and feature extraction;
measure or state those omissions plainly.

## Independent advanced-pipeline review

Executable checks of `advanced/data.py` passed for all three held-out objects
and both seeds 7/19: training, validation and test trial sets are pairwise
disjoint, and every test trial belongs to its held-out object. The preprocessing
code fits axis scales on training rows only. Each window uses prior inputs and
its final tick's label; transitions and otherwise ignored prior samples remain
inside its input context. State resets per window. Checkpoint selection uses
validation weighted loss and decision threshold selection uses validation balanced
accuracy before test inference. No test calibration leakage was found.

For window32/stride16, actual eligible test sets are:

| Held-out object | Windows | Evaluated trials | No-slip windows | Slip windows |
| --- | ---: | ---: | ---: | ---: |
| brush | 884 | 29 | 535 | 349 |
| screwDriver | 2,586 | 30 | 1,396 | 1,190 |
| SpoolSolder | 2,539 | 28 | 1,907 | 632 |

Each fold assigns 30 test trials, but some contain no eligible endpoint at the
chosen stride. Report eligible-trial counts separately from assigned counts.
Changing seed changes both initialization and inner train/validation partitions,
so the variability is not exclusively optimization randomness. Using the same
seed for different windows preserves the trial partition; final tick schedules
and number of windows can still differ.

The reviewed bootstrap resamples whole eligible test trials with replacement,
using the same resample for paired LIF/ANN differences. It skips single-class
resamples. Its interval is descriptive conditional on the trained models and
held-out object, not a significance test for a population of objects. Record
requested, retained and single-class-excluded replicate counts.

The advanced export chooses the first positive held-out brush window in fixed
order for the first declared seed, without screening model correctness. It uses
precomputed six-axis derivative summaries. The export records a nominal 5,556 µs
sample interval, but the advanced board sketch runs unpaced throughput replay,
not live sensor signals or faithful irregular original acquisition timing.
This is a model parity and compute-cost fixture; it cannot estimate accuracy.
