# Research scoping: spiking inference on a low-cost ESP32

Exploratory review, 6 October 2026. This is a research proposal and literature
assessment, not an implementation or a claim of hardware results. It records the
user's clarified objective: hypothesis testing about how low-cost microcontrollers
behave under neuromorphic workloads for tactile slip detection in robotic
grippers. LIF is one candidate, not a required final model. The user subsequently
clarified that the course demonstration does not require a physical gripper;
an ESP32 inference/replay demonstration is the planned hardware component.

This document preserves the initial scoping stage. Implementation has since
progressed; see `RESEARCH_REPORT.md` and `DELIVERY_CHECKLIST.md` for current
software results and outstanding board measurements.

## Recommended question

**Under what input activity, model size, numerical precision, and CPU workload
does a compact spiking model offer a useful accuracy/latency/memory trade-off on
our ESP32, relative to conventional alternatives?**

Provisional title: **Spiking Neural Inference on a Low-Cost ESP32 for Robotic
Gripper Slip Detection: Accuracy and Execution Trade-offs**.

Use one-board language until multiple MCU models are actually evaluated. The
application can motivate the study without assuming that vibrations alone
establish incipient slip. Validating physical slip requires independent labels.
Positive, null, and negative findings all answer the research question.

## Deadline and proposed deliverable

The user reports that approximately 80% should be ready by Friday evening,
9 October 2026, with today being Tuesday. The reported module text, “ISM 2.4G
802.11 b/g/n”, identifies wireless capabilities rather than the exact SoC.
Board identity remains unconfirmed; do not assume two cores or a particular
numeric acceleration path.

For this deadline, define completion by deliverables rather than claiming that
80% of physical validation is finished. A defensible Friday package is a clear
research question, closest-work comparison, frozen experiment protocol,
reproducible offline reference and tests, one compact trained SNN if the chosen
data are usable, conventional baselines, analysis scripts, and an Arduino-ready
trace-replay sketch. Report software results as software results, and mark
ESP32 measurements as pending until actually run. Completing this package
does not itself validate incipient slip or guarantee the course's 80% criterion.

Suggested work sequence, subject to dataset access and the course rubric:

- Tuesday: inspect candidate data and labels; choose the task and split;
  finish the closest-work comparison and freeze primary outcomes.
- Wednesday: build a host reference, simple controls and one small trained
  feed-forward SNN. Check exported numerical behavior on identical traces.
- Thursday: implement the static C/C++ runtime and Arduino replay wrapper;
  compile for the confirmed board; run reproducible host comparisons and
  prepare the reporting/plotting pipeline. Board execution is a separate check.
- Friday: reproduce runs, complete the report and presentation, and document
  unresolved hardware tests. Add measured replay timings only if actual board
  runs have succeeded before the deadline.

Prioritize H1 and H3 below. Adaptive/recurrent networks remain an extension;
quantization, multi-core scheduling and energy measurement are not required
for the initial package. If no defensible labelled dataset can be prepared in
time, narrow the report to SNN runtime characterization on controlled traces
and remove task-accuracy and incipient-slip claims rather than inventing labels.

## What the literature already establishes

These are targeted primary sources, not an exhaustive systematic review. A
candidate research gap is not an established novelty claim.

| Source | Relevant evidence | What it does not establish for us |
| --- | --- | --- |
| [Hansert, Gechter and Offermann, 2026: A Fair Energy Comparison of Spiking and Artificial Neural Networks on a General-Purpose Microcontroller](https://pubmed.ncbi.nlm.nih.gov/42657619/) | The published abstract describes int8 C/C++ SNN and ANN inference on an ESP32-C6, using MNIST and matched MLP architectures. It reports inference energy savings in its conditions. | We inspected the abstract, not the full experimental detail. It does not establish our board's behavior, continuous tactile latency, or a quality-matched advantage for our task. |
| [Barchi et al., 2024: Energy efficient and low-latency spiking neural networks on embedded microcontrollers through spiking activity tuning](https://link.springer.com/article/10.1007/s00521-024-10191-5) | Full text describes an embedded recurrent network with 20 adaptive LIF neurons and two outputs for vibration-based structural monitoring on STM32F4. It studies firing activity, processing cycles, and measured energy. | This is a different MCU and application. Its signal pipeline includes preprocessing and FFT, so it does not support a universal claim that SNNs eliminate windows or preprocessing. |
| [Jurzec, Szydlo and Wielgosz, 2025 preprint: Compression and Inference of Spiking Neural Networks on Resource-Constrained Hardware](https://arxiv.org/html/2511.12136v1) | Describes SNNTorch model export to a C runtime and reports Arduino Portenta H7 deployment. Execution maintains neuron state across time steps and uses pruning. | Its reported results are not ESP32 measurements. Portability and numerical equivalence still need checking. |
| [Lu et al., 2025 preprint: A Neuromorphic Incipient Slip Detection System using Papillae Morphology](https://arxiv.org/html/2509.09546v1) | Uses a spiking convolutional network with NeuroTac and a purpose-built tactile skin for slip-state classification. | That sensor and mechanical setup differ substantially from a simple piezo sensor. Its result cannot validate our hardware or be reproduced by substituting an SW-420. |
| [Van Wyk and Falco, 2018: Slip Detection: Analysis and Calibration of Univariate Tactile Signals](https://arxiv.org/abs/1806.10451) | Studies window size, sampling rate, material, slip speed, and sensor variability using univariate tactile signals. | It does not establish an SNN advantage. It is useful guidance for application-level controls. |

The broad ideas “SNN on a microcontroller”, “SNN on an ESP32”, “SNN processes
vibrations”, and “SNN detects slip” therefore have prior art. Our contribution
could be a reproducible characterization of an exact board and streaming task,
including where an advantage disappears. Publication-level novelty needs a more
focused follow-up review after choosing the experiment.

## Model space and selection

The following sizes are proposed starting budgets, not measured capabilities.
An entire SNN can be small: input encoding, a connected hidden spiking layer,
weighted output/readout, and persistent state all execute on the board. Training
on the PC does not make on-board inference incomplete.

| Candidate | Role | Selection judgment |
| --- | --- | --- |
| Threshold with hysteresis, EMA/envelope threshold, and event count in a window | Non-neural controls | Required: they may solve the problem more cheaply. Tune fairly on development data. |
| Single LIF temporal accumulator | Minimal spiking control | Keep, but distinguish it from a learned multi-neuron SNN. |
| Small feed-forward LIF SNN, initially 16–64 hidden neurons | Full trained SNN | Preferred first network. Simple state and an auditable C/C++ implementation. |
| Adaptive LIF / small recurrent spiking network | Longer temporal memory | Strong second candidate if the feed-forward model lacks useful history. Extra state and recurrence must justify their cost. |
| Liquid state machine / spiking reservoir with a trained readout | Alternative temporal network | Promising optional route: freeze recurrent weights and train the readout offline. Try this instead of adding many more architectures. |
| Izhikevich / quadratic neuron | Richer spike and burst dynamics | Useful optional neuron-cost ablation if those dynamics answer a hypothesis. Biological detail alone is not a reason to choose it. See the [original model paper](https://www.izhikevich.org/publications/spikes.pdf). |
| Convolutional SNN or large recurrent model | Larger spatial/temporal model | Defer unless data and baselines justify it. Spatial convolution is not the default for one piezo channel. |

A manageable main study has simple controls, one full SNN, and one small ANN
control. Select at most one recurrent/adaptive extension after the first
experiments. Evaluate operating points across an accuracy/cost frontier rather
than declaring a model superior because it won one chosen setting.

## Coherent deployment method

1. Define a causal input representation and decision rule. Retain timestamps and
   specify resets, windows, and any warm-up. Evaluate change/level-crossing
   encoding or direct current input as candidates. Rate encoding is another
   option, but its observation time belongs in latency accounting.
2. Train offline using an established framework such as
   [snnTorch](https://snntorch.readthedocs.io/en/latest/). Its
   [surrogate-gradient tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)
   provides a training route for non-differentiable spikes. A reservoir instead
   permits training only a readout.
3. Export weights and neuron parameters as fixed C/C++ arrays. Allocate state
   and scratch buffers once. Keep the Arduino sketch wrapper thin. This avoids
   requiring Python, model-file parsing, or a training framework on the ESP32.
4. Establish a floating-point reference first. Explore int8 weights with wider
   integer accumulators/states only after validating scaling, saturation,
   thresholds, and reset behavior. Weight quantization and state quantization
   are separate choices. Measure numerical drift and changed decisions.
5. Compare host Python, host C/C++, and later ESP32 outputs on identical traces.
   Check hidden states/spikes where possible, not just final labels. Freeze the
   model/export version for measurement.
6. On the ESP32, distinguish neuron update time, inference/readout time,
   encoding/preprocessing time, queue delay, and end-to-end decision time.
   Retain all deadline misses and dropped inputs.

The [SpikeEdge authors' repository](https://github.com/karol-jurzec/snn-generator/)
is useful deployment evidence, but its README lists JSON-C, MATIO, and POSIX
threads and describes dynamic model loading. It is not a verified drop-in
Arduino ESP32 library. Reusing a minimal inference kernel or building a small
static runtime is a porting proposal, not something already validated here.

An ordinary ESP32 executes spiking dynamics in software. Do not assume the
advantages of dedicated neuromorphic hardware automatically transfer. Likewise,
“event-driven” is not guaranteed by binary activations: dense clocked updates
can still visit every neuron on every time step. Sparse synaptic accumulation
and fully asynchronous execution are different optimizations. Skipping updates
must preserve decay, synaptic currents, recurrent/delayed events, and any
autonomous firing; numerical equivalence needs testing.

## Hypotheses worth testing

| Hypothesis | Controlled comparison | Evidence that could reject it |
| --- | --- | --- |
| H1: exploiting sparse spike activity lowers execution cost at low activity, with a crossover at higher activity | Run the same trained SNN and input traces with dense accumulation versus sparse spike accumulation. Sweep activity and model size; verify equivalent decisions first. | Sparse execution never wins, or queue/index overhead overwhelms the savings. |
| H2: reduced numerical precision lowers memory/cost while retaining acceptable task quality | Compare floating-point and quantized versions of the same model on a frozen test set. | Accuracy/false alarms deteriorate or rounding changes temporal behavior excessively. |
| H3: a compact trained SNN improves the task-quality/cost frontier over a single accumulator and conventional controls | Use the same recording splits, causal information, decision deadline, and validation/tuning budget. Include a small ANN. | An ordinary filter or ANN matches or beats the SNN across useful operating points. |
| H4: separating acquisition from inference changes tail latency under CPU load | On an actual dual-core board, compare same-core and separate-core task placement with identical replay/input, clocks, priorities, and workload. | Queue overhead cancels the improvement or deadline misses remain unchanged. |

These are hypotheses, not expected results. Freeze primary outcomes and an
acceptable task-quality margin before evaluating the final test set. Do not
expand every factor into a huge factorial experiment: begin with H1 and H3;
add H2 or H4 if time and hardware support them.

## Measurement and fairness

- Quality: balanced accuracy/F1 for labelled classification, and detection rate,
  noise false-alarm rate, and decision delay for event detection. Report the
  full trade-off when models deliberately wait for more evidence.
- Execution: median/p95/maximum runtime, deadline-miss rate, queue growth and
  dropped inputs. A model executing faster than real time in a tight replay
  loop has not yet proven that it sustains the live input rate.
- Memory: weights, neuron state, input/scratch buffers, task stacks, and runtime
  allocations. Measure total firmware resource usage separately from model
  state; do not claim the whole pipeline fits in three variables.
- Energy: only claim measured energy if suitable current measurement is added.
  Runtime, CPU cycles, and spike counts are not energy measurements. Include
  acquisition and idle time when discussing overall device energy.
- Comparisons: give every method the same raw recordings and causal time
  horizon. If SNN input is encoded and ANN input is analog, separately measure
  the information and processing cost introduced by encoding. Include an
  encoded-input ANN control when useful.
- Runtime optimization: use a matched straightforward C implementation to
  isolate model effects, and distinguish a practical optimized comparison.
  [Espressif ESP-NN](https://github.com/espressif/esp-nn) supplies optimized NN
  kernels; availability and benefit depend on the chip and chosen operations.
  Avoid giving one method optimized kernels while presenting the result as a
  pure algorithmic comparison.
- Reproducibility: record board/SoC, toolchain, clock, radio state, priorities,
  actual core IDs, input/model hashes, and seeds. Randomize model run order and
  repeat across boots to expose order/thermal/run effects.
- Statistics: keep whole recordings/sessions/objects out of training, rather
  than mixing overlapping windows across splits. Use paired comparisons on
  the same recordings and confidence intervals with recording/run as the
  resampling unit. Thousands of windows from one recording are not thousands
  of independent experiments. One board does not establish all ESP32 variants.

The [classic ESP32 datasheet](https://documentation.espressif.com/esp32_datasheet_en.html)
and [ESP32-C6 datasheet](https://documentation.espressif.com/esp32-c6_datasheet_en.html)
describe different CPU/memory arrangements. Confirm the user's exact chip before
choosing core-placement experiments or assuming results from the C6 transfer.

## Data and the no-sensor path

Three evidence categories must stay distinct:

1. **Synthetic host tests:** validate logic/export and stress activity patterns.
   They provide no measured ESP32 timings or physical slip accuracy.
2. **ESP32 replay benchmarks:** store traces in flash and run the actual
   inference code on the board. With a USB data cable, this can measure compute
   and memory without a sensor or GPIO loopback wire. It establishes hardware
   execution costs, not sensor acquisition performance or actual slip detection.
   Run both untimed throughput replay and paced streaming replay.
3. **Physical acquisition/application tests:** evaluate the sensor chain and
   labelled contact/slip trials. Incipient-slip claims need an independent
   reference and a documented onset definition; simply tapping a sensor is
   insufficient.

Candidate public data, inspected at their source pages but not downloaded or
validated in this review:

- [VibroTact, University of Bath](https://researchdata.bath.ac.uk/1351/): four
  piezo vibration channels, textures and sliding procedures, CSV data and
  CC-BY-4.0 licensing. Relevant for vibration representation/replay, but the
  listed labels do not establish incipient-slip onset.
- [NeuTouch slip classification data, NUS authors' project](https://clear-nus.github.io/visuotactile/download.html):
  stable/rotational-slip recordings with tactile spikes and other modalities.
  Useful for a public-data inference study after checking documentation,
  licensing, labels and channel mapping. Its sensing modality differs from a
  single inexpensive piezo sensor; do not imply transfer has been validated.

For the intended analog experiment, sensor selection and conditioning should
follow the data question. An SW-420's digital event output discards analog
information; it would define a different task from the original piezo/ADC
proposal. No components are selected or purchased by this review.

## Next sequence, before committing to implementation

1. Confirm exact ESP32 model and available equipment. Friday evening,
   9 October 2026 is the review target. The user has confirmed that a physical
   gripper is not required; the hardware demonstration will be recorded-input
   inference on the ESP32.
2. Choose one main hypothesis and one dataset/task with defensible labels.
   Inspect a small data sample, metadata, license and splits before training.
3. Audit the closest prior work against that specific experiment. Scope the
   contribution as replication/characterization or a candidate extension.
4. Freeze baseline methods, primary metrics, tuning budget, and candidate SNN.
5. Implement the offline reference and static export, then compile an Arduino
   replay sketch. Only after that proceed to board profiling and acquisition.

The existing GPIO polling/interrupt sketch remains an acquisition sanity check.
It does not test a neuron model or full SNN and is not the central experiment
for the clarified research question. This review leaves existing firmware,
simulation results and protocol untouched. No new model is trained, no board
is flashed, and no hardware measurements are claimed.
