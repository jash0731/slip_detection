# ESP32 USB-only SNN replay

This is the robotic-gripper project's **compute demonstration**. It replays the
exported input trace through a connected 4-input, 12-hidden-neuron LIF network
with persistent voltage states and a trained linear readout. Hidden connection
weights are fixed; the readout is learned offline. It is not an end-to-end
trained SNN, live slip detector, or grip controller. Read the exported header
and run metadata for the actual input and training provenance.

## Wiring and upload

1. Connect the ESP32 to the computer using a **USB data cable**. No sensor,
   jumper wire, breadboard, GPIO loopback, or external gripper is needed.
2. In Arduino IDE install Espressif's **esp32** board package. Select the board
   matching the module marking and its serial port. The Wi-Fi `802.11 b/g/n`
   marking does not identify the ESP32 model.
3. Open `firmware/esp32_snn_replay/esp32_snn_replay.ino`. Keep
   `generated_model.h` and `snn_runtime.h` in the same sketch folder. Upload.
4. Open Serial Monitor at **115200 baud**, reset the board, and send `s` plus
   Enter for unpaced throughput replay, `p` plus Enter for paced replay, or
   `a` plus Enter for the controlled input-activity experiment.
5. Save all output, including `#` metadata lines and the CSV header, through
   `# COMPLETE`. Reset to perform another independent run.

Analyze a saved capture from the project folder:

```sh
python3 analyze_replay.py hardware_results/replay-run.csv
```

The analyzer rejects incomplete/mixed captures and checks paired outputs,
dimensions and deadline flags before reporting step median/p95/maximum and
miss counts. It accepts this replay schema only, never host result files.

The sketch uses the ordinary Arduino loop with no task pinning or dual-core
assumption. It must be compiled/uploaded for the actual board; classic ESP32
compile verification does not prove compatibility with every ESP32 variant.
On boards using native USB, enable the board's USB CDC-on-boot setting if its
Serial Monitor otherwise remains empty.

Build verified with Arduino ESP32 **3.3.12**, target `esp32:esp32:esp32`, all
warnings enabled: **275,324 bytes flash; 23,836 bytes static RAM**, no warnings.
This is compile evidence only. The
actual user board has not been identified, uploaded, or measured.

## What is measured

Each command produces ten repetitions per dense/sparse mode, alternating mode
order and resetting state between traces. A suppressed warm-up in each mode
checks the exported Python fixture (exact spike counts and 1e-5 voltage/score
tolerance) before emitting benchmark rows. Dense evaluates all input weights;
sparse skips only zero input events. Both decay and test every neuron each
tick. This is sparse input accumulation, not a completely event-driven solver.

`source=hardware_replay` is written **only by a running board**. Building the
sketch or running C++ correctness checks creates no hardware results. Recorded
tactile input and synthetic input remain different evidence sources; the
header and `input_origin` metadata identify which is compiled in.

Raw CSV columns are:

```text
source,run_id,model_id,input_id,trial,mode,pacing,step,active_inputs,hidden_spikes,encode_us,inference_us,duration_us,start_lateness_us,deadline_miss,final_score,final_decision
```

These rows have their own schema; they are **not** the GPIO experiment's
`EXPERIMENT.md` acquisition rows and must not be sent to `analyze_hardware.py`.
`step` is zero-based; `trial` starts at one. `duration_us` sums the encoder and
hidden-layer intervals. The final recording's raw logistic score/decision is
repeated on each row; it is not a per-step prediction or slip-onset timestamp.
`# trace` records aggregate elapsed and readout time. Timing excludes serial
output but includes timestamp overhead; aggregate elapsed also includes buffer
writes and, in paced mode, waits. Short microsecond-clock intervals can be zero.

Paced replay schedules ticks from the trace start using the exported nominal
sample interval. Lateness is actual start minus scheduled start; a deadline
miss means hidden-layer completion passed the next tick's scheduled start.
Overdue ticks are processed in order, with no simulated input drops. This
measures servicing a memory trace, not an ADC/interrupt queue. Throughput mode
has no deadlines and reports zero for those fields.

Static model state and replay buffer sizes, chip, CPU frequency, SDK, and free
heap are metadata. These partial byte counts are **not total firmware RAM**;
use the build report and include stacks/runtime allocations when reporting
total footprint. No power or energy measurements are made.

## Controlled activity sweep (`a`)

Save its separate full capture and validate with:

```sh
python3 analyze_activity.py hardware_results/activity-run.csv
```

The summary reports median batch-mean time per tick and a sparse/dense ratio
for each activity level. It checks all declared pairs, operation counts and
checksum parity. These batch means are not per-tick tail latency.

This directly runs the sparse-computation hypothesis on the ESP32 when uploaded.
Five constant **synthetic** amplitudes produce 0, 1, 2, 3, or 4 active channels.
Each condition gets dense/sparse warm-up and ten paired 10,000-tick batches,
alternating order. Timing covers hidden-layer updates only: reset, encoding,
readout, checksum, serial output, and between-batch yielding are excluded.
Timestamp overhead remains included. A volatile checksum keeps the computed
state observable. Every pair must have identical final voltages and spike
counts before its rows are emitted; errors leave an incomplete capture.

```text
source,run_id,model_id,trial,mode,active_inputs,input_channels,steps,duration_us,input_contributions,hidden_updates,checksum
```

Source is `hardware_activity_sweep`; it requires an actual board run. There are
100 rows in a completed sweep. `input_contributions` counts source-level
weight terms: dense = inputs × hidden × ticks, sparse = active × hidden × ticks.
It does not measure executed CPU instructions. Both modes perform every hidden
neuron's decay and threshold test on every tick. These rows use a separate
schema from both recorded replay and GPIO acquisition. This workload does not
evaluate public-data classification accuracy, incipient slip, or energy.

## Evidence boundary

An actual board run demonstrates that ESP32 executes the model and measures
its replay cost. It does not demonstrate piezo sensor transfer, independent
incipient-slip ground truth, grip adjustment, or generalization to a new
physical gripper. Public-data offline classification and board timing should
be reported separately. No hardware capture is supplied with this package.

Host parity check instructions are in `cpp/README.md`.

Current export uses one public recorded ARQ brush tactile window, reduced
offline from 54 channels to a normalized scalar first-difference signal. That
preprocessing is **not performed or timed on the board**; `encode_us` measures
only the four threshold channels applied to that prepared scalar. The compiled
trace is 32 samples at its exported median nominal interval; replaying this one
window repeatedly is a timing demonstration, not a hardware accuracy dataset.
