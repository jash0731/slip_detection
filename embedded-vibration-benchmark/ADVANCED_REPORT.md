# Trained spiking inference on ESP32 for robotic-gripper slip detection

Exploratory research report, 7 October 2026. Course review target: 9 October.

This fixed32-sample report is retained for traceability. The completed temporal,
robustness and error investigations appear in [FOLLOWUP_REPORT.md](FOLLOWUP_REPORT.md).
Its validation-selected software models are distinct from this Arduino export.

## Abstract

We investigate whether compact spiking models offer useful classification and
storage trade-offs for tactile slip recognition on an inexpensive ESP32. Public
robotic-gripper recordings provide three objects, ninety grasp trials and 228,521
raw samples. We compare a logistic control, a conventional neural network (ANN),
trained leaky integrate-and-fire (LIF), adaptive LIF (ALIF), and a frozen-hidden
LIF ablation. Leave-one-object-out testing evaluates transfer to an unseen object.
Across two predetermined training/validation seeds, trained LIF achieves 60.8%
mean object balanced accuracy, versus 57.2% for the matched ANN and 57.5% for ALIF.
Int8 weight storage reduces logical parameters from 516 to 188 bytes while
changing some held-out decisions. Exported Arduino inference passes host numerical
checks and classic-ESP32 compilation. Board execution, physical incipient-slip
prediction and energy savings remain unmeasured.

## Motivation and contribution

A robotic gripper needs to recognize slipping contact so that a later controller
can adjust its grip. This project studies the detector and its computing platform;
the course demonstration replays recorded contact signals without building a
mechanical gripper. The experimental contribution is a reproducible comparison of
learning, adaptation, object transfer and parameter precision under small model
budgets. It is not a claim to invent spiking slip detection.

[Neftci et al.](https://arxiv.org/abs/1901.09948) motivate surrogate-gradient
training through discrete spikes. [Bellec et al.](https://papers.neurips.cc/paper_files/paper/2018/hash/c203d8a151612acf12457e4d67635a95-Abstract.html)
motivate adaptive thresholds; our feed-forward ALIF is substantially simpler than
their recurrent networks. [Taunyazov et al.](https://arxiv.org/abs/2009.07083)
establish event-driven tactile SNN perception as prior work, while
[Grover et al.](https://arxiv.org/abs/2202.09549) demonstrate the relevance of
conventional temporal slip models. The matched ANN here is a small baseline,
not a reproduction of their temporal convolutional network. See
[the targeted literature review](RESEARCH_UPGRADE.md) for design decisions.

## Hypotheses and methods

The exploratory questions are whether hidden-weight learning improves over a
readout-only LIF, whether adaptation improves over LIF, and whether reduced weight
precision saves storage without substantial decision changes. Actual ESP32 timing
is an additional deployment question awaiting board measurements.

The [ARQ-CRISP source dataset](https://github.com/ARQ-CRISP/slip_detection_dataset_2021)
contains brush, screwdriver and solder-spool grasps, thirty trials per object,
with eighteen tactile pins reporting three axes each. Downloads are pinned to
commit `a536ee3301af0566218782b00de334b6a6630979`, checked against upstream
SHA256/byte counts, and accompanied by the retained licence. Source labels 4/5
identify held non-slip/slip during arm movement. They describe perceived slip,
not independently calibrated physical onset. The recordings use a multi-axis
tactile sensor; transfer to our inexpensive vibration sensor is unvalidated.

For each trial, average pins separately on x/y/z, calculate causal first
differences, normalize using training-only axis scales, clip and split signs into
six nonnegative input channels. Use 32-sample windows at stride 16 and label each
window by its final sample. Earlier transitions remain context. There is no
future input or homogeneous-label window selection. State resets at each window.
At the source nominal rate, a window spans approximately 178 ms; actual intervals
are irregular and the dynamics use sample steps rather than timestamp-dependent
leaks. These six inputs are continuous currents, not binary input spikes.

Hold out all trials of one object, cycling through all three objects. In each
fold reserve six complete trials per remaining object for validation and use the
other twenty-four for training. Seeds 7/19 vary both initialization and these
internal splits. Fit preprocessing on training trials, select epochs and
thresholds on validation, and freeze every choice before evaluating the test
object. The evaluation contains 6,009 unique windows across 87 eligible trials;
three of the ninety assigned trials contain no eligible endpoint windows.

ANN/LIF/ALIF share a 6→16→1 architecture and 129 learned parameters. ANN averages
ReLU activity; LIF averages hard hidden spikes; ALIF additionally raises thresholds
after firing with a decaying adaptation state. LIF/ALIF train input weights,
biases and readout through surrogate gradients. Frozen LIF starts with identical
hidden parameters but learns only the readout. Logistic regression uses the
window mean of six channels. All methods use the same causal representation.

Training uses class-weighted binary cross entropy, Adam, a maximum of eighteen
epochs, validation-loss early stopping and validation-only decision thresholds.
Saved configurations, histories, checkpoints and raw predictions support audit.
Exact equations and settings appear in [ADVANCED_SOFTWARE.md](ADVANCED_SOFTWARE.md).
The [independent data review](ADVANCED_DATA_REVIEW.md) found no object/trial split
overlap or test-derived normalization in these six runs.

## Classification results

Balanced accuracy averages slip recall and non-slip recall, reducing misleading
effects of class imbalance. Report means over two seeds within each object, then
weight the three objects equally. F1 measures the positive slip class.

| Model | Mean balanced accuracy | Mean slip F1 |
| --- | ---: | ---: |
| Logistic control | 57.3% | 0.407 |
| Matched ANN | 57.2% | 0.408 |
| Frozen LIF, trained readout | 57.4% | 0.444 |
| Trained LIF | **60.8%** | **0.495** |
| Trained ALIF | 57.5% | 0.430 |

| Held-out object | ANN | Trained LIF | Trained ALIF |
| --- | ---: | ---: | ---: |
| Brush | 63.7% | 71.3% | 63.6% |
| Screwdriver | 55.3% | 56.2% | 55.4% |
| Solder spool | 52.7% | 54.8% | 53.6% |

![Held-object quality, public recorded host evaluation](advanced/figures/cross_object_quality.png)

LIF has a modest aggregate advantage here, with substantially weaker transfer to
the screwdriver and spool. Adaptation adds state without improving this aggregate.
These outcomes support a specific bounded comparison, not general superiority of
SNNs or inferiority of adaptive neurons. Scores from the earlier one-object
experiment cannot isolate improvement: its representation and evaluation task
were different. The frozen-LIF comparison is the controlled learning ablation.

We calculate 2,000 paired bootstrap replicates by resampling **whole grasp trials
within each object**, preserving overlapping windows and pairing all models/seeds.
All requested replicates have both classes. LIF minus ANN is **+3.58 percentage
points**, with a conditional 95% interval of **+1.51 to +5.70**. LIF minus frozen
LIF is +3.38 points (+1.11 to +6.52); ALIF minus LIF is −3.28 (−5.07 to −1.65).
These intervals condition on the three observed objects and fixed fitted models;
they exclude training uncertainty and population-level object generalization.
Longer trials still contribute more windows within an object. Comparisons are
exploratory, without multiple-testing correction or a noninferiority claim.

![Paired whole-trial uncertainty intervals](advanced/figures/paired_trial_intervals.png)

## Precision and deployment

Post-training quantization stores input/output weights as symmetric int8 tensors
with separate scales. Biases, neuron states and arithmetic remain float32. Each
129-parameter model uses **516 bytes** for float32 parameters, or **188 bytes**
including int8 weights, two scales and unchanged biases: a **63.6% reduction** in
logical parameter storage. Neuron state, buffers, runtime and input traces are
additional. The comparison firmware includes all three models and both precision
banks, so its full flash size does not demonstrate this saving.

Across 12,018 window evaluations per model (6,009 windows × two seeds), int8 weights
change 48 LIF decisions (0.40%), 15 ALIF decisions (0.12%) and 11 ANN decisions
(0.09%). Maximum absolute logit drift is 0.090, 0.087 and 0.034, respectively.
Quality changes can be positive or negative; thresholds stay frozen. This is
weight-storage quantization, not fully integer inference or evidence of speed
or energy savings.

![Parameter storage and decision changes](advanced/figures/weight_quantization.png)

The exported first predetermined fold/seed contains all three models plus float
and quantized reference states. Host C++ checks validate spike counts, state
updates and readouts. The advanced Arduino sketch compiles with Espressif core
3.3.12 for classic ESP32: **278,136 bytes flash, 23,252 bytes static RAM**, no
warnings. Fifty Python checks pass across the project. The connected ESP32-D0WD-V3
was then uploaded and produced a validated 1,920-row capture. Measured median/p95/
maximum prepared-input step times in microseconds were LIF float20/21/36, LIF
int8-weight19/21/34, ALIF float22/23/25, ALIF int8-weight21/22/26, ANN float18/19/25
and ANN int8-weight17/19/24. Every ten-repeat pair had zero decision drift.

Sending `s` at 115200 baud runs ten repetitions of each model and
precision, producing 1,920 CSV rows. This measures unpaced computation on prepared
inputs, excluding raw tactile preprocessing. A single fixed positive fixture is
used for numerical/timing verification; all three exported models classify that
fixture as non-slip. Its failure is retained rather than selecting a successful
example. Repeated fixture predictions are not an accuracy dataset. Follow
[ADVANCED_HARDWARE.md](ADVANCED_HARDWARE.md) and retain the completion metadata.

## Conclusion and next evidence

The completed software study and first board replay support a small trained LIF network as the best
aggregate classifier among these matched configurations, while documenting modest
quality, poor transfer on some objects and no benefit from this ALIF variant.
Weight quantization provides a concrete storage/error trade-off. The next required
course-demo evidence is now the validated USB replay using only the board and
USB data cable. A separate sensor/LED demonstration can show acquisition once the
actual sensor is identified; it does not validate these trained tactile models.

Physical incipient onset, continuous operation, cheap-sensor transfer, complete
sensor-to-decision latency, energy and closed-loop gripping remain open. Public
recorded host results, synthetic tests and future hardware captures retain distinct
source fields and folders. See [VERIFICATION.md](VERIFICATION.md) for commands and
[DELIVERY_CHECKLIST.md](DELIVERY_CHECKLIST.md) for outstanding work.
