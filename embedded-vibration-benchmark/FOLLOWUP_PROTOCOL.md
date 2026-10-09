# Follow-up research protocol: memory, robustness and failures

Specified 7 October 2026, before evaluating the new configurations. This is an
exploratory extension informed by the first study, not an independent preregistration.
Original results, checkpoints and the Arduino deployment export stay frozen.

## 1. Temporal mechanism

- Add a conventional leaky-ReLU network with the same 6→16→1 learned parameters,
  initialization, inputs and training procedure as LIF/ANN. Its hidden activation
  is ReLU(0.875 × previous activation + affine input); the readout averages hidden
  activity. The diagonal leak is fixed, with no learned recurrent matrix. This
  supplies temporal state without discrete spikes or reset.
- Diagnose the original fixed checkpoints using three deterministic permutations
  of each window's past frames and reversed past frames. Preserve the last frame,
  all six channels jointly, window labels and input multiset. Original thresholds
  remain fixed. Mean-ReLU ANN should be invariant apart from numerical roundoff.
- Also train LIF and leaky-ReLU on deterministically shuffled train/validation
  windows, then evaluate similarly shuffled test windows. This controls the
  training-distribution mismatch of test-only shuffling. Use window32/lr0.01,
  both declared seeds and all object folds, without selecting a shuffle draw.
- Changes indicate sensitivity to order in this representation/window protocol.
  They do not prove biological mechanisms, physical incipient onset, or isolate
  temporal memory from every difference between spikes, reset and optimization.

## 2. Robustness and model selection

Fixed candidate grid: windows16/32/64; Adam learning rates0.005/0.01;
ANN/LIF/leaky-ReLU; seeds7/19; three held-out objects. At most18 epochs, patience4,
class-weighted BCE, weight decay0.0001, batch256, gradient clipping5.
This is108 candidate fits plus12 shuffled-training fits, not an open-ended search.

Use the original whole-trial splits and training-only normalization. Score common
endpoint schedules in every partition: end_sample≥63, then stride16. Thus all
window lengths see identical endpoint labels and trial IDs; longer windows get
more past context. Observation horizons are approximately89/178/356ms at the
nominal rate; actual timestamps are irregular. All models reset per window.

Threshold selection uses the existing validation-balanced-accuracy grid. Choose
window/lr independently per model using validation balanced accuracy; deterministic
 ties favor shorter windows then lr0.01. Complete and record all candidate choices
for a fold before any test predictions. Evaluate the predetermined grid as a
sensitivity analysis, but never select a winner by test score. Save validation
choices, histories, trained parameters, raw predictions and split/provenance.
Resample whole trials for paired model differences on common test endpoints.
Uncertainty remains conditional on these objects and fitted models.

## 3. Failure analysis

Audit the original frozen predictions with per-object/per-trial confusion counts,
slip recall, non-slip recall, precision, false alarms and misses. Stratify windows
by original label context (stable state, mixed held states, ignored context) and
by distance in samples from the start of the current annotated label run.
These are post-hoc descriptive subgroups, not independent physical onset labels
or training/selection inputs. Report both seeds and eligible trial/class counts.
Single-class groups have undefined balanced accuracy, never artificial perfect
scores. Examine train/test feature distribution shifts and retain illustrative
incorrect windows selected by explicit rules rather than claiming typical cases.

## Evidence and stopping

Finish after this fixed grid, three frozen-checkpoint shuffle draws, reversed-history
diagnostics, failure tables/figures and independent logical checks. Training and
classification are computer-side public-recorded results. No ESP32 upload, raw
sensor acquisition, power measurement or physical gripper result is generated.
Interpret findings whether the original advantage survives, disappears or reverses.
