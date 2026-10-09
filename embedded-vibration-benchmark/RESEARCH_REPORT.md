# Spiking inference on an ESP32 for robotic-gripper slip detection

**First-stage exploratory report — 6 October 2026**

The current three-object trained-model extension is in [ADVANCED_REPORT.md](ADVANCED_REPORT.md).
This earlier report retains its original protocol and results for traceability.

## Abstract

This project investigates how a low-cost microcontroller can execute spiking
models for tactile slip detection in robotic grippers. The first implementation
compares ordinary temporal features, a scalar leaky integrate-and-fire (LIF)
accumulator, a small conventional neural feature layer, and a connected LIF
network with a learned readout. On a public brush-grasping dataset, the selected
SNN obtains 0.599 balanced accuracy on a held-out grasp pose, compared with
0.664 for EMA and scalar-LIF controls. This configuration does not establish a
spiking accuracy advantage. Controlled host activity sweeps verify equivalent
dense/sparse computation and illustrate activity-dependent execution cost.
An Arduino-compatible ESP32 replay implementation passes numerical checks and
compiles for a classic ESP32 target. Actual board timings are pending. Results
concern homogeneous slip-state windows and software execution; they do not
establish incipient-slip prediction, physical sensing, energy savings, or grip
control. A physical gripper is not required for the planned course demonstration.

## Motivation and prior work

A gripper should recognize an unstable grasp early enough for an appropriate
control response. That application motivates studying causal tactile inference
on inexpensive hardware. An ESP32 implements neuron dynamics in ordinary
software; savings associated with dedicated neuromorphic hardware cannot simply
be assumed. Encoding, state updates and observation windows can dominate cost.

Prior work already demonstrates [SNN/ANN inference on an ESP32-C6](https://pubmed.ncbi.nlm.nih.gov/42657619/),
[adaptive recurrent SNN vibration processing on an STM32](https://link.springer.com/article/10.1007/s00521-024-10191-5),
and [neuromorphic slip classification with specialized tactile hardware](https://arxiv.org/html/2509.09546v1).
The first source was assessed from its abstract; it does not establish a
quality-matched benefit for our streaming task or board. The current project
therefore makes no first-ever implementation claim. Its proposed contribution
is a reproducible characterization of model quality and execution trade-offs
on an identified low-cost board, with the conditions and failures documented.
See [RESEARCH_SCOPING.md](RESEARCH_SCOPING.md) for the broader model-space review.

## Questions and hypotheses

**H1:** Sparse input accumulation reduces work at low activity, with an
activity-dependent cost advantage that can disappear at high activity.

**H3:** A compact SNN offers a useful quality/cost operating point relative to
simple controls and a conventional network under the same input horizon.

These are testable propositions, not assumed benefits. H1 first requires
numerical equivalence. H3 requires held-out labelled recordings and fair cost
boundaries. Quantization, adaptive/recurrent neurons and core placement are
possible extensions, not implemented findings in this report.

## Data and task

The [ARQ-CRISP authors' robotic pick-and-place dataset](https://github.com/ARQ-CRISP/slip_detection_dataset_2021)
provides tactile recordings from an EZGripper/UR5 setup. We downloaded and
SHA256-verified the brush file at a pinned commit, preserving its license and
documentation. Our audit found 30 trials, 41,707 samples and 54 tactile axes,
with finite values and increasing timestamps. This is public recorded hardware
data; it was not collected by our ESP32.

The primary task distinguishes annotation 4, held without slip during arm
motion, from annotation 5, held with perceived slip during arm motion. Other
labels are excluded from scoring. This avoids making an absent object the main
negative class. Labels are manual perceived states; they are not an independent
measurement of the first incipient-slip instant. Dataset details and source
citations appear in [DATA_REVIEW.md](DATA_REVIEW.md).

For each original trial, we compute the RMS first difference across all axes
and divide it by a training-only 95th-percentile scale. The transform uses the
current and preceding sample. Trials are never concatenated, and ignored gaps
are not removed from the source timeline. The resulting scalar reduces the
sensor's information; it is not a piezo voltage or proof of sensor transfer.

Poses 1/2 supply training data; pose 3 is held out. Consecutive nonoverlapping
32-sample windows are retained only when all annotations are 4 or all are 5.
There are 306 training windows and 55 test windows from nine test trials. Test
balance is 13 negative and 42 positive windows. Only three test trials supply
negatives. Selecting homogeneous windows using annotations excludes onset
transitions, so the scores describe this conditional subset rather than general
online detection. State resets at each window. The observation horizon is
approximately 178 ms, not a sub-5 ms physical decision.

## Models and training

All methods receive the same prepared scalar window and decide after its end.
Conventional controls use threshold occupancy, mean EMA envelope, or scalar-LIF
spike rate. The ANN has four binary amplitude inputs and eight fixed tanh
units. The connected SNN has the same four encoded inputs and twelve fixed LIF
neurons, with per-sample voltage decay, weighted input, threshold firing and
subtractive reset. Its readout uses the hidden neurons' window spike rates.

Only the logistic readouts are trained, using 400 full-batch gradient steps and
training-only feature normalization. Hidden weights are seeded and fixed. The
output is a nonspiking linear score with a zero decision threshold. This is a
complete connected spiking inference pipeline with a trained readout, not an
end-to-end trained or recurrent SNN. ANN/SNN hidden capacities differ; this is
a comparison of specific configurations, not a general architecture ranking.
No test-set hyperparameter search was performed. Exact parameters and equations
are in [SOFTWARE_EXPERIMENT.md](SOFTWARE_EXPERIMENT.md).

Dense mode evaluates all input contributions each tick. Sparse mode skips only
zero-valued inputs, preserving summation order, decay and threshold updates.
It does not skip silent time steps. Python tests verify temporal parity and
the C++ implementation checks the exported fixture, reset, decay and encoding.

## Exploratory findings

### Public recorded-data classification

Source: `public_recorded_host`. The table is descriptive; windows within trials
are dependent, so no independent-window confidence intervals are attached.

| Method | Balanced accuracy | F1 | True positives / 42 | False positives / 13 |
| --- | ---: | ---: | ---: | ---: |
| Threshold occupancy + readout | 0.575 | 0.525 | 16 | 3 |
| EMA + readout | 0.664 | 0.567 | 17 | 1 |
| Scalar LIF + readout | 0.664 | 0.567 | 17 | 1 |
| Fixed ANN + readout | 0.587 | 0.548 | 17 | 3 |
| Fixed SNN + readout, either schedule | 0.599 | 0.571 | 18 | 3 |

![Exploratory public-recording quality](software_results/public_quality.png)

The SNN has slightly higher recall/F1 at this fixed decision threshold but lower
balanced accuracy than the simpler controls. This does not establish a useful
overall advantage. It motivates studying encoding and decision thresholds using
validation data, not assuming spiking models are better. The observed result
does not reject all SNN designs: hidden weights, representation and dynamics
have not been optimized. Raw predictions, confusion counts, environment,
configuration and split metadata are in `neuromorphic/public_results/`.

Public Python timing columns are debugging observations from one pass with
fixed method order. They exclude raw 54-axis preprocessing and are not used
here as a reliable performance comparison.

### Controlled activity and execution

Source: `synthetic_host_activity_sweep`. A 2,048-tick synthetic sequence contains
0%, 25%, 50%, 75% or 100% all-channel-active ticks. Dense processing evaluates
98,304 input contributions in every condition. Sparse processing evaluates
0, 24,576, 49,152, 73,728 or 98,304. Both schedules perform 24,576 neuron
updates. Exact state/spike parity passes. This is an operation-count result;
the counts are not CPU instructions or energy measurements.

A separate source, `host_reference`, times the Arduino C++ hidden-update kernel
on this Apple Silicon computer using Clang and `-O2`. Ten paired repetitions
process 200,000 ticks per condition, with constant inputs activating zero to
four encoder channels. This channel-activity workload differs from the Python
temporal on/off sweep. Encoder/readout costs are excluded.

![Host C++ activity timing](software_results/host_activity.png)

On the saved host run, sparse median timing is lower at low channel activity and
approaches dense timing at full activity. This supports investigating a cost
crossover on the board; it does not establish an ESP32 benefit. Tiny host timings
depend on compiler optimization and the controlled workload. All 50 paired
checksums match. Raw rows, compiler/flags, hashes and medians are saved under
`software_results/`. Python activity timing is also explicitly host-only.

### Deployment preparation

The Arduino sketch replays one predetermined held-out public positive window
through dense and sparse modes, with a state reset for each repetition. The
fixture was selected by label/order without screening predictions; it is a
numerical check, not a representative accuracy sample. Ten paired runs alternate
method order. Throughput and nominally paced modes log raw durations and pacing
deadline misses. Model/trace identity, chip/clock, memory fields and fixture
checks are metadata. Logging happens after the timed trace.

The default scalar fixture is preprocessed offline; board timing excludes the
54-axis reduction and normalization. Paced replay uses a typical fixed interval,
not every irregular original timestamp. The final-window score is not a per-tick
slip onset. These exclusions are explicit in [SNN_HARDWARE.md](SNN_HARDWARE.md).

The `a` command implements the board's controlled activity experiment: zero
through four active channels, ten paired trials per level, 10,000 ticks per
batch. Warm-up and exact paired final-state checks precede acceptance; serial,
resets, encoding and readout are excluded from batch timing. It emits a separate
100-row schema accepted by `analyze_activity.py`. This experiment is runnable;
its board findings remain pending until an actual capture exists.

Compilation with the Espressif Arduino core 3.3.12 for classic `esp32:esp32:esp32`
passed with warnings enabled: 275,324 bytes flash and 23,836 bytes static RAM.
Those totals describe that build, not model-only RAM or remaining heap on every
ESP32 variant. The user's exact module is still unconfirmed. No board was
flashed and no ESP32 timing capture exists yet.

## Limitations and next validation

The initial study uses one object, a small number of negative grasp trials, a
single pose split and fixed configurations. It loses spatial/shear information
through scalar reduction and excludes transition windows. This prevents strong
generalization or early-warning conclusions. No sensor cost, battery life,
energy, deterministic scheduling or closed-loop grip response was measured.

The next required step is to identify the board, upload the replay sketch and
save actual captures. Running its controlled activity sweep on the same board
is needed before an ESP32 crossover claim. Follow-up quality work should use
additional objects, validation partitions and causal transition evaluation.
Adaptive/recurrent networks or surrogate-gradient training can then be compared
when they answer a specific failure of the present representation/model.

The course demo needs only an ESP32 and USB data cable. A physical gripper is
outside the requested demonstration scope; the application remains robotic
grasping. The existing GPIO polling/interrupt experiment is a supporting
acquisition benchmark and retains its separate schema.

## Reproduction and evidence record

Use the commands in [README.md](README.md), [SOFTWARE_EXPERIMENT.md](SOFTWARE_EXPERIMENT.md)
and [cpp/README.md](cpp/README.md). Run `python3 plot_software_results.py` after
generating the public results and copying the host activity CSV to
`software_results/host_cpp_activity.csv`. The report tables correspond to these
saved exploratory runs. Rerunning host timing can change timing values.

Integration checks: 22 Python tests passed; host C++ parity tests passed;
public-data rerun reproduced the reported quality scores; classic ESP32 compile
passed. [DELIVERY_CHECKLIST.md](DELIVERY_CHECKLIST.md) distinguishes completed
software from pending board measurements. No synthetic row is presented as
measured hardware or real robotic-slip accuracy.
