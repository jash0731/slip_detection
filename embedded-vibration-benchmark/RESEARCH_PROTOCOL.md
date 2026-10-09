# Research protocol: neuromorphic slip inference for robotic grippers

The current trained-model extension is specified in
[RESEARCH_UPGRADE.md](RESEARCH_UPGRADE.md) and [ADVANCED_SOFTWARE.md](ADVANCED_SOFTWARE.md).
This first-stage protocol retains the separate dense/sparse activity question.

Protocol draft, 6 October 2026. The physical application remains detecting slip
in a robotic grasp so that a later controller can adjust grip force. The present
software study evaluates the computation needed for that detector. It does not
establish the onset of physical slip or demonstrate closed-loop grasp control.

## Primary questions

1. For an identical small SNN, when does sparse accumulation reduce work relative
   to dense accumulation, and when does its bookkeeping erase that benefit?
2. Does the SNN offer a useful task-quality/computation trade-off relative to
   conventional controls that receive the same causal input?

An advantage is a hypothesis. Equivalent or worse SNN results are valid outcomes.
The first question can be studied on controlled traces. The second requires
defensible task labels before making a claim about robotic slip.

## Initial comparison

Include an amplitude/threshold control, a causal EMA/envelope control, a scalar
LIF accumulator, one complete small connected SNN, and a small ANN where feasible.
The full SNN may use fixed spiking features and a learned output/readout; identify
this as readout training rather than end-to-end surrogate-gradient training.
Adaptive recurrent models and a liquid-state machine remain justified follow-ups,
not prerequisites for completing this first study.

Give all methods the same raw trace, observation horizon, reset boundary, and
development/test recording split. Include input encoding and feature construction
in full-pipeline costs. A streaming decision and an end-of-window classification
must not be compared as if their latency boundaries were identical.

## Evidence stages

| Stage | What it can establish | What remains untested |
| --- | --- | --- |
| Synthetic host reference | Repeatability, causal logic, training/export path, stress patterns, dense/sparse numerical parity | Real tactile task accuracy, ESP32 latency, physical slip |
| Real public tactile data on host | Quality for the dataset's labels and held-out recordings | Transfer to our sensor/gripper, on-board execution |
| ESP32 flash-trace replay | Actual board computation, state/buffer memory, throughput and paced replay behavior | ADC acquisition, physical onset delay, grip control |
| Labelled physical grasping trials | Detection quality and delay for the tested contact/sensor/gripper conditions | Generality beyond tested objects, materials and devices |

Synthetic classes must be called stable-like and slip-like activity, with their
generator described. Labels created by our generator are not independent physical
ground truth. Public stable-versus-rotational-slip labels also do not automatically
give incipient-slip onset timestamps.

## Data and tuning

Keep entire source recordings or sessions in one split. Fit normalization,
readout weights and thresholds only on training/development splits; freeze all
choices before evaluating the held-out test split. Report class balance, sample
rate, channel mapping, duration, licensing and exclusions. Preserve raw data;
transformed/exported fixtures require their own provenance and hashes.

Start with one compact network and a modest development budget. Record every
selected hyperparameter. Do not tune on the final test set or select the final
model solely because its displayed test score is best. For any pilot already
inspected, call it exploratory and obtain an independent confirmation split.

## Measurements

- Quality: confusion counts, balanced accuracy and F1; for true onset-labelled
  streams, false alarms per idle duration, detection rate and onset delay.
- Work: attempted versus active synaptic accumulations and neuron updates.
  Operation counts describe the implementation and are not measured energy.
- Host timing: full-trace elapsed time with warm-up and repeated measurements;
  record interpreter, compiler, optimization flags and host. Treat timing as
  exploratory, especially for Python sparse/dense comparisons.
- Board timing: per-step and per-trace median, p95 and maximum measured duration;
  state deadlines and count overruns. Exclude serial formatting/transmission
  from inference timing, and describe that exclusion. Throughput replay and
  paced streaming have different meanings.
- Memory: distinguish static model/state/buffers, firmware flash/RAM totals,
  stacks, and dynamic allocation. Host object sizes are not ESP32 RAM estimates.

Randomize or counterbalance method order and repeat runs/boots for a hardware
comparison. Use paired inputs. Estimate uncertainty across independent recordings
or repeated runs, not overlapping samples treated as independent replications.
Do not attach physical-performance confidence intervals to synthetic data.

## Required numerical checks

Dense and sparse modes must preserve neuron state, spike sequence and output
within a declared tolerance at every step. Check quiet inputs, isolated events,
bursts, resets and consecutive recordings. Host and exported C++ must use the
same decay, threshold, reset, weights, input encoding and readout semantics.
Zero-weight connections may be skipped; skipping arbitrary clock ticks can
change decay or firing and needs a different equivalence argument.

## Existing electrical acquisition experiment

`EXPERIMENT.md` and `HARDWARE.md` retain their GPIO loopback protocol and exact
CSV schema. This supporting experiment compares electrical observation methods.
Its rows must never be mixed with SNN replay rows or called physical slip latency.
Core-placement experiments require confirmation that the user's chip has the
relevant cores. GPIO loopback requires a wire; flash-trace replay does not.

## Reporting rule

Every results table carries a source label: `synthetic_host`,
`synthetic_host_activity_sweep`, `public_recorded_host`, `host_reference`,
`hardware_replay`, `hardware_activity_sweep`, or `physical_gripper`.
A table may appear only after the corresponding run exists.
Record null and negative findings. Until actual measurements exist, board and
gripper sections remain pending rather than being filled with estimates.
