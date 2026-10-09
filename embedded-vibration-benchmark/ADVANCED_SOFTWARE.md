# End-to-end trained tactile models on recorded gripper data

This extension keeps **robotic gripper slip detection** as the application and
asks whether trainable spiking dynamics offer useful classification and deployment
trade-offs on a low-cost controller. It preserves the original scaffold unchanged.
All results here are computed from public recorded tactile signals on a computer.
The exported C++ model is intended for ESP32 replay; no board result is fabricated.

## What is substantively different

- Hidden input weights and biases **and** output weights and bias are trained.
  LIF and adaptive LIF use surrogate-gradient backpropagation through time.
- The ANN uses the same six inputs, sixteen hidden units and 129 learned parameters.
- A same-initialization frozen-LIF control isolates hidden-layer training from
  merely using a different encoding or dataset split.
- Three objects and ninety original trials support leave-one-object-out tests.
  Whole-trial internal validation controls stopping and decision thresholds.
- The target label is taken at the final sample of each causal window. Past ignored
  states and transitions remain in the context; there is no homogeneous-window
  oracle selection.
- Float32/int8-weight export includes deterministic input fixtures and numerical
  state references, plus a held-out numerical drift audit.

This remains a compact current-driven SNN with one spiking hidden layer and a
nonspiking linear output readout. It is not a deep recurrent network, biologically
complete neuron model, on-device training system, or guaranteed novelty claim.

## Reproduce

Python 3, NumPy and CPU PyTorch are required. The dataset fetch/conversion scripts
and licence/provenance are in `data/`. Use its prepared `arq_all_trials.npz`.
From this project directory:

```sh
python3 -m unittest discover -s tests -p 'test_advanced*.py'
python3 -m advanced.train --epochs 18 --seeds 7 19
python3 -m advanced.frozen_ablation
python3 -m advanced.numeric_audit
```

Training is CPU-only, capped at two threads with deterministic PyTorch algorithms.
The main comparison completed in approximately 225 seconds on the development
computer; this is training wall time, not inference performance. Different machines
and library versions can produce numerical differences. Dataset hashes, versions,
seeds, splits, per-epoch histories, trained checkpoints and selected thresholds are
saved in `advanced/results/`. Raw predictions contain object, trial, final sample
index/time, target label, logit and decision.

`advanced/export/models.json` exports the **first predetermined seed/fold**, which
holds out the brush object, for LIF, ALIF and ANN. The export is not selected by
best test performance. Its fixture is the first held-out positive window in fixed
order, selected without screening its prediction. The deployment agent's converter
generates the self-contained Arduino header and verifies float32 C++ parity.

## Input and evaluation protocol

The three ARQ-CRISP objects are brush, screwDriver and SpoolSolder. Each has thirty
grasp trials with eighteen tactile pins and x/y/z readings. For each original trial:

1. Average the eighteen readings for each axis and compute first differences.
   The first trial sample has zero derivative; trials are never concatenated.
2. Estimate the 95th percentile of absolute differences independently for each
   axis using **training trials only**. Divide by these fixed scales and clip to
   the range -4 to 4.
3. Split each signed axis into positive and negative components. Input order is
   positive x/y/z, then negative x/y/z: six continuous nonnegative channels.

These are continuous input currents rather than binary input spikes. Mean pin
summaries reduce spatial information; their adequacy is an assumption to test.
The source sensor is not a low-cost piezo sensor, so this study does not establish
transfer to a piezo-equipped gripper.

Use 32-sample windows at stride 16. Windows overlap within a trial, but entire
trials belong to exactly one split. A window is eligible when its **last** label is
4 (held, no slip) or 5 (held, slipping). Its earlier samples can contain other
labels or transitions; no future input beyond that last sample is included.

For each of three outer folds, hold out every trial of one object. Of each remaining
object's thirty trials, assign six whole trials to internal validation and the rest
to training using a fixed seed. Seeds 7 and 19 are predetermined repetitions, not
independent datasets or opportunities to choose the best test seed. Test prediction
occurs only after all models' choices for that fold are frozen.

Class-balanced binary cross entropy uses the negative/positive training-count ratio
as the positive loss weight. Adam uses learning rate 0.01, weight decay 0.0001,
batch size 256, gradient clipping at 5, and an 18-epoch maximum. Stop after four
epochs without validation-loss improvement and restore the best validation-loss
checkpoint. Select the final logit threshold on validation balanced accuracy over
the fixed grid -2.5 to 2.5 in steps of 0.1. There is no test-set hyperparameter tuning.

## Exact models

| Model | Representation and learnable parameters |
|---|---|
| Logistic control | Window mean of six input channels into a trained logit; 7 parameters |
| ANN | Per-sample 6→16 affine/ReLU, window-mean hidden activation, linear logit; 129 parameters |
| LIF | Per-sample 6→16 affine current, leaky voltage/spikes, window spike-rate readout; 129 parameters |
| ALIF | Same LIF with spike-driven adaptive thresholds; 129 parameters plus sixteen adaptation states |
| Frozen LIF | Same initialized LIF input weights/biases frozen; only 17 output parameters trained |

LIF, ALIF and ANN use identical seeded initial input weights/biases and output
initialization. Input weights start normally distributed with mean 0.20 and standard
deviation 0.20, and input biases start at 0.12. Initializations are deliberately
simple and not claimed to be optimal.

Every window resets voltage, adaptation and accumulated activity to zero. At each
tick, LIF/ALIF use:

```text
current[h] = sum_i(W[h][i] * input[i]) + input_bias[h]
v[h] = 0.875 * v[h] + current[h]
threshold[h] = 1.0                       # LIF
threshold[h] = 1.0 + 0.4 * a[h]          # ALIF, using old adaptation
spike[h] = 1 if v[h] >= threshold[h], else 0
v[h] -= spike[h] * threshold[h]
a[h] = 0.95 * a[h] + spike[h]            # ALIF only
count[h] += spike[h]
logit = output_bias + sum_h(output_weight[h] * count[h] / 32)
```

There is one spike per tick and subtractive reset, with no refractory interval or
learned recurrent connections. ANN replaces these state dynamics with ReLU and
accumulates its activations. ALIF's threshold constants are fixed, not trained.

Forward spikes are hard thresholds. Backward uses the surrogate derivative
`1 / (1 + 5 * abs(v - threshold))**2`. Reset spikes are detached during training;
the adaptive threshold can still receive gradients through adaptation. Hidden input
weight changes are recorded and asserted nonzero for trained LIF/ALIF. Frozen-LIF
changes are asserted zero. This is actual hidden-layer learning, not merely an
export of random weights.

## Initial quality findings

Scores below average the two predetermined seeds within each object, then average
the three objects equally. They are descriptive, conditional on these objects and
this sensor setup; paired trial-bootstrap results are separate artifacts.

| Model | Object/seed-macro balanced accuracy | Object/seed-macro F1 |
|---|---:|---:|
| Logistic control | 0.573 | 0.407 |
| Capacity-matched ANN | 0.572 | 0.408 |
| Trained LIF | **0.608** | **0.495** |
| Trained ALIF | 0.575 | 0.430 |
| Frozen LIF/readout-only | 0.574 | 0.444 |

The trained LIF result is modestly higher in this fixed comparison. Improvement is
uneven: brush is easier, while the other held-out objects remain much closer to
chance balanced accuracy. Adaptive thresholds do not improve the aggregate here.
This supports further investigation of a specific LIF configuration; it does not
establish that SNNs generally outperform ANNs or that ALIF is intrinsically worse.

`summary.csv` records each object's/seed's confusion counts and selected threshold.
Its `test_trials` counts trials that actually contributed scored windows;
`assigned_test_trials` counts all thirty assigned outer-test trials. These differ
when a trial has no eligible final labels. Group bootstrap resamples whole trials,
never overlapping windows as independent observations. Confidence intervals are
conditional on the trained models and observed objects, not population estimates
for every robotic gripper.

## Numerical precision and deployment

The JSON export contains float32 parameters, int8 input/output weight tensors with
one symmetric scale per tensor, float32 biases, the 32×6 fixture, decision threshold,
expected final voltage/adaptation/activity sums and logit. An explicitly ordered
float32 reference mirrors simple C++ accumulation; hidden spike counts are compared
against the trained PyTorch forward pass before export.

Quantized weights are dequantized for arithmetic with float32 states/biases. This
reduces parameter storage, **not** a claim of integer-MAC execution or faster energy
use. The 129-parameter models use 516 bytes for float32 parameters. Int8 weights,
two scales and unchanged float32 biases use 188 bytes; neuron state, traces and
buffers require additional memory in both variants.

The post-training numeric audit keeps original checkpoints and validation-selected
thresholds fixed. Across 12,018 held-out window evaluations (six folds/seeds), int8
weights change 48 LIF decisions, 15 ALIF decisions and 11 ANN decisions. Maximum
absolute logit drift is approximately 0.090, 0.087 and 0.034 respectively. These
results illustrate why a storage optimization needs an output-quality check;
`numeric_audit.csv` records per-fold accuracy/F1 and activity as well.

The export records a nominal sample interval of 5,556 µs. Original timestamps are
irregular; the advanced firmware runs unpaced throughput replay on precomputed
six-channel inputs. It does not wait for this interval or faithfully reproduce
sensor timing. GPIO acquisition, ADC conditioning and live gripper control remain
separate experiments.

## Limits that remain central

Thirty-two samples require approximately 178 ms of observation at the nominal
sample rate. An inference kernel finishing quickly cannot erase that observation
delay. These labels describe existing held-object slip, not an independently
measured incipient-slip onset; no lead-time or sub-5-ms physical detection claim is
supported. State resets per window also differ from a fully continuous detector.

Three objects from one source setup provide a useful controlled extension, not
enough evidence for universal material or sensor generalization. Two seeds do not
replace independent datasets. Model sizes and training budgets are small, and
hyperparameters are not comprehensively optimized. Board timing/memory must be
measured on the actual ESP32; energy requires power instrumentation. The report
should retain these limits while presenting the reproducible methods and results.
