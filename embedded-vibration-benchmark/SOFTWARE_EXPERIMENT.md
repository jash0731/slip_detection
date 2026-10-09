# Robotic-gripper tactile inference: executable research scaffold

This is the retained first-stage fixed-hidden experiment. For the current
three-object end-to-end trained comparison, use
[ADVANCED_SOFTWARE.md](ADVANCED_SOFTWARE.md) and the project [README](README.md).

This package keeps robotic grasping and slip detection as the application. It
implements an initial hypothesis-testing comparison, not a finished incipient-slip
detector. Three evidence types are kept distinct: synthetic host validation,
classification of public recorded gripper tactile data on a computer, and ESP32
replay measurements that must be collected separately. No board result is included.

## Reproduce

Run from this project directory with Python 3. NumPy is needed only for the public
data conversion and experiment; the synthetic model and activity sweep use the
standard library. The public-data audit/conversion instructions and source licence
are in `data/`. The downloaded HDF5 and derived NPZ are ignored local files;
run the fetch/audit/preparation steps before the public experiment on a new checkout.

```sh
python3 -m unittest discover -s tests -p 'test_neuromorphic*.py'
python3 -m neuromorphic.experiment
python3 -m neuromorphic.activity_sweep
python3 -m neuromorphic.public_experiment
```

The first experiment produces `neuromorphic/synthetic_results/` and a separate
`neuromorphic/synthetic_model.h`. The activity sweep adds its CSV to that synthetic
directory. The public experiment produces `neuromorphic/public_results/` and
`firmware/esp32_snn_replay/generated_model.h`, the model used by the Arduino replay
sketch. The public header is regenerated deterministically; host timing columns
vary between runs. Seed, split, dataset hash, fitted coefficients, and limitations
are recorded in JSON metadata. CSV `source` values identify the evidence type.

## What is implemented and trained

Every model receives the same scalar sequence over the same window, emits a
decision only after that window, and resets its model state at each window.

| Model | Causal features over the window | What training changes |
|---|---|---|
| Threshold | Fraction of samples whose magnitude is at least 0.30 | Logistic decision coefficient/bias |
| EMA | Mean of the magnitude envelope, decay 0.875 | Logistic decision coefficient/bias |
| Scalar LIF | Mean spike count from one leaky accumulator | Logistic decision coefficient/bias |
| ANN | Four amplitude bands into eight fixed tanh units; window-mean activations | Eight-unit logistic readout |
| Connected SNN | Four amplitude bands into twelve persistent LIF neurons; window spike rates | Twelve-unit logistic readout |

The amplitude-band thresholds are 0.12, 0.30, 0.60 and 1.00. The small ANN and SNN
have seeded **fixed hidden weights**; only their readouts are learned. This is a
controlled fixed-feature/readout comparison, not a claim that either architecture
is fully trained or optimally tuned. The SNN has connected inputs, a full hidden
neuron layer and an output readout; its output is logistic, not a spiking neuron.
There is no recurrent connection, adaptation, online learning, or deep network.

For each input sample the SNN applies:

```text
input[i] = 1 if abs(sample) >= band[i], else 0
v[h] = 0.875 * v[h] + sum_i(input[i] * W[i][h])
spike[h] = 1 if v[h] >= 1, else 0
v[h] = v[h] - spike[h]
count[h] += spike[h]
final_logit = bias + sum_h(readout[h] * count[h] / window_samples)
prediction = final_logit >= 0
```

One spike is allowed per neuron per tick with subtractive reset and no refractory
period. Voltage and counts persist within the window. Decay is per sample, rather
than corrected for irregular timestamps. Python uses double precision; exported
C++ constants and device arithmetic are single precision. The fixture includes
expected counts, final voltages, and logit for cross-language verification.

Training fits feature means/standard deviations on training examples only, then
runs 400 full-batch logistic-gradient steps with learning rate 0.15 and coefficient
regularization 0.01 in normalized feature units. Export converts coefficients back
to original feature units. No test-set tuning is performed. This first configuration
was fixed as a demonstration; it has no validation search or model-selection claim.

## Public recorded gripper experiment

The ARQ-CRISP brush recording contains 30 grasp trials with 18 tactile pins and
three axes per pin. Its provenance, licence and label meanings are preserved in
`data/public/ARQ_CRISP_2021/`. This is data from an existing gripper experiment,
not measurements taken with our ESP32 or with a low-cost piezo sensor.

For each trial, compute the RMS first difference across all 54 raw tactile axes.
The first sample of a trial has difference zero; samples are never concatenated
across trials. Divide this scalar by its 95th percentile over the training trials.
This calibration is frozen before processing held-out data. No interpolation,
future samples, or test-fitted scaling are used.

Use poses 1 and 2 for training and hold out **all pose-3 trials** for testing. Divide
each original trial into consecutive, nonoverlapping 32-sample windows. Retain only
windows entirely labelled 4 (held, no slip) or entirely labelled 5 (held, slipping).
Exclude mixed windows and labels 0/1/2/3/6. This avoids classifying a not-held object
as the principal negative class and preserves ignored gaps. Per-trial class counts
are recorded in metadata.

The resulting exploratory comparison has 306 training windows and 55 test windows
(13 no-slip, 42 slip). Some trials contribute only one class, and there are very few
independent negative trials in the test set. Windows from one trial are dependent;
the following scores are descriptive, without independent-window confidence
intervals or significance claims.

| Model | Held-out balanced accuracy | Held-out F1 |
|---|---:|---:|
| Threshold | 0.575 | 0.525 |
| EMA | 0.664 | 0.567 |
| Scalar LIF | 0.664 | 0.567 |
| Fixed-layer ANN + learned readout | 0.587 | 0.548 |
| Fixed-layer SNN + learned readout | 0.599 | 0.571 |

**The SNN does not establish an accuracy advantage in this initial configuration.**
Identical dense and sparse schedules produce identical predictions. A negative
result is useful: it motivates testing encoding and neuron dynamics with validation
data, rather than claiming that spiking alone improves slip detection.

Thirty-two samples are approximately 178 ms at the typical acquisition rate. These
are homogeneous *existing-slip* classification windows. They deliberately exclude
onset transitions and cannot establish early warning, sub-5-ms slip detection, or
incipient-slip lead time. There is only one object, one sensor arrangement, and one
held-out pose. The public source's annotated state is not independent physical
ground truth collected by this project.

Selection of homogeneous windows uses their annotations, including labels later
in that window. This defines a conditional evaluation subset; it is not an online
window-selection algorithm. Model inference itself uses causal input only. Models
reset to zero at every window, so this first classifier is not a continuously
running stateful detector. The eight-unit ANN and twelve-unit SNN are not capacity
matched; conclusions apply only to these configurations. The replay fixture is
the first held-out positive window in deterministic order, chosen without screening
predictions, for parity verification rather than representative accuracy.

Host timings include amplitude encoding, model feature generation and readout,
but exclude conversion of the 54-channel source into the normalized scalar. They
measure Python implementations on the current computer. The replay fixture is
also already preprocessed; any ESP32 timing concerns that scalar-input inference,
not a complete 54-channel sensor pipeline. Replay uses a typical fixed interval;
the original recording intervals are not perfectly regular.

Public host timing is a single pass per window with fixed model order. It is saved
for debugging and is not a reliable comparative execution-cost result. Environment
versions and configuration are captured in metadata; use C++ board measurements
for the performance conclusion.

## H1: sparse activity and scheduling cost

`activity_sweep.py` generates 2,048-sample synthetic sequences with exact mean input
activity fractions 0, 0.25, 0.5, 0.75 and 1.0. At each active tick all four input
channels fire; placement is shuffled with a fixed seed. This is a controlled cost
sweep, not a realistic tactile spike distribution.

Dense accumulation evaluates all four input contributions for each neuron on every
tick (98,304 contributions per sequence). Sparse accumulation skips zero inputs,
so its counts are 0, 24,576, 49,152, 73,728 and 98,304 respectively. **Both perform
24,576 hidden neuron updates**, including decay and threshold checks at silent
ticks. Tests check exact output/state parity. This optimization preserves the model
and does not claim event-driven decay skipping.

Seven host repetitions per schedule record median, minimum and maximum time. The
initial computer run showed sparse timing lower through 75% activity and slightly
higher at 100%; repeat runs and compiler/device results can differ. This suggests
an activity-dependent crossover worth measuring on the ESP32, not an established
hardware advantage. Contributions are an algorithmic count, not CPU instructions,
clock cycles, joules, or battery life. The board sketch must supply hardware timing.

## Synthetic validation

The seeded generator creates 120 independent waveform recordings, with background
vibration and isolated impacts in both classes and constructed oscillatory bursts
in the positive family. Eighty whole recordings train the readouts; forty test them.
High synthetic classification scores only establish that the implementation can
distinguish those designed families. They are not reported as gripper slip accuracy.

The tests cover dense/sparse temporal parity, decay during silent inputs, threshold
boundaries, deterministic training/export, disjoint recording splits, train-only
public calibration, ignored gaps, and trial-boundary resets. Device parity tests
and compile checks are separate from these algorithm tests.

## What remains to establish the research claims

1. Run the Arduino replay and record actual board identity, timing and memory.
2. Measure the activity sweep in the same C++ implementation on that board; use
   repeated runs and record any interruptions or deadline misses.
3. Expand public-data coverage to more objects and independent trials, freeze a
   validation protocol, and test adaptive/recurrent neurons if justified by results.
4. If early warning is retained as a claim, define a physical slip-onset reference
   and evaluate transition sequences causally with end-to-end observation delay.
5. Measure power with instrumentation before making an energy-efficiency claim.

The existing GPIO polling/interrupt experiment remains an acquisition-timing
component. It is not the robotic-gripper inference experiment itself.

## Recreate report plots

The C++ sweep is a separate host workload with constant inputs activating zero
through four encoder channels, unlike the Python temporal on/off sweep.

```sh
mkdir -p software_results
c++ -std=c++11 -O2 -Wall -Wextra -pedantic cpp/activity_benchmark.cpp -o /tmp/gripper-activity
/tmp/gripper-activity > software_results/host_cpp_activity.csv
python3 plot_software_results.py
```

Matplotlib is required for this step. PNG/SVG plots carry evidence labels.
The plotting tool validates all 50 paired checksums and records host/compiler
metadata. These figures must not be relabelled as ESP32 or energy results.
