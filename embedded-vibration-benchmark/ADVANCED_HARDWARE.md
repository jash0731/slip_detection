# Advanced USB-only ESP32 model comparison

The robotic-gripper application remains slip recognition. This sketch compares
**surrogate-gradient-trained LIF**, **surrogate-gradient-trained ALIF**, and a
capacity-matched **conventional ReLU ANN** on identical prepared recorded tactile
input. Each network has six continuous input channels, sixteen hidden units,
and one learned output logit. Input/output weights and biases are learned;
this is different from the earlier fixed-hidden-weight demonstration.

## Upload and run

1. Connect the ESP32 using a USB **data** cable. No sensor, wire, or gripper is
   required for recorded-input computation.
2. Install Espressif's ESP32 board package in Arduino IDE. Select the board
   matching the actual module marking and the correct serial port.
3. Open `firmware/esp32_advanced_replay/esp32_advanced_replay.ino` with
   `advanced_runtime.h` and `generated_advanced_model.h` in the same folder.
4. Upload, open Serial Monitor at **115200 baud**, reset, then send **`s`** and
   Enter. Save every line from the metadata to `# COMPLETE`.
5. Reset to repeat an independent board run. Native-USB boards may require
   USB CDC-on-boot enabled in their board settings.

The code uses the ordinary Arduino loop and does not assume a dual-core chip.
The connected board identified itself as **ESP32-D0WD-V3**, dual-core, 240 MHz.

Compile verified on classic `esp32:esp32:esp32` with Arduino ESP32 **3.3.12**,
all warnings enabled: **278,136 bytes flash; 23,252 bytes static RAM**, no
warnings. These are build measurements; the capture below contains executed-board timings.

Save the complete Serial Monitor output as `hardware_results/advanced-run.csv`,
including comment metadata and the completion marker, then validate it:

```sh
python3 analyze_advanced.py hardware_results/advanced-run.csv
```

The analyzer rejects incomplete runs, inconsistent model/input identifiers,
missing model/precision pairs and invalid decisions. It reports median, p95 and
maximum step times, plus changed decisions on this repeated fixture. Those
changes are not an estimate of population slip accuracy.

## Models and precision

Every 32-sample window resets state. LIF updates voltage with leak plus input
current, emits at most one spike per tick, and subtracts threshold on firing.
ALIF computes threshold using the **previous** adaptation state, then updates
adaptation after firing. ANN applies ReLU per tick. All three accumulate hidden
activity and use its mean in the learned output readout. The discrete model
parameters are exported from training; unpaced replay does not change their
logical sample interval or simulate elapsed wall-clock neuron dynamics.

Every model runs with both float32 weights and **int8 weight storage with
float32 arithmetic/state**. Input/output matrices use separate tensor scales;
weights are dequantized on demand. Biases, activations, states, accumulation,
and readout remain float32. This is neither a fully integer network nor a claim
that int8 storage makes inference faster. Quantized output may differ from the
float model; the original validation-selected decision threshold stays fixed.

The sketch runs a fixture preflight against independent ordered Python
float32 and quantized references. Hidden spike counts must match exactly;
voltage/adaptation/hidden sums/logit use 1e-4 absolute tolerance. Every measured
trace is checked again. Float versus quantized equality is **not required**.
Failure stops the run without a completion marker.

## Timing and raw capture

Ten repetitions run for each of three models and two precisions. Model order
rotates and precision order alternates between trials. State and buffers are
preallocated; serial output is outside the trace timing. There is no task
pinning, binary spike encoder, or sparse accumulation optimization.

```text
source,run_id,model_id,model_hash,input_id,input_hash,trial,precision,step,input_prepare_us,inference_us,duration_us,hidden_spikes,final_score,decision_threshold,final_decision,reference_decision,decision_drift
```

`source=hardware_advanced_replay` is emitted only when the sketch actually runs
on a board. Precision is `float32` or `int8_weights_float_state`. The first
validated capture is `hardware_results/advanced-run-clean.csv`; the original
`advanced-run.csv` retains reset boot bytes before its text header. A completed
capture has 1,920 rows (10 × 3 × 2 × 32). Step numbers start at zero. The final
window score/decision repeats on its step rows; it is not a per-step slip-onset
decision. `reference_decision` uses the float reference and the model's frozen
validation threshold; `decision_drift` indicates a changed binary decision.
`# trace` metadata records whole-trace and separate readout duration.

`input_prepare_us` copies **already prepared** six-channel features into a
small input array. It excludes recorded raw 54-channel derivative extraction,
normalization, ADC, interrupt queue, and sensor acquisition. `inference_us`
includes current accumulation, neuron/activation updates, and hidden sums.
`duration_us` sums these two intervals; readout is separate. Microsecond timer
resolution and timestamp overhead limit very short step timings. Aggregate
trace time also includes timestamp calls and buffer writes.

This is unpaced throughput replay. No queue losses, real-time sensor deadlines,
end-to-end physical detection delay, energy, or power are measured. A single
held-out positive fixture is exported without selecting for a successful
prediction; it can be misclassified. Full classification results belong to the
offline test dataset, not this repeated timing fixture.

## Provenance and memory

Each row includes model/input SHA256 identifiers. Input hash is SHA256 over the
prepared trace's row-major little-endian float32 bytes. Model hash comes from
the frozen JSON model entry; the whole JSON export hash is also metadata.
These identifiers support artifact matching; they do not authenticate a
capture or make fabricated measurements impossible.

## First measured capture

The connected board produced 1,920 rows (10 repetitions × 3 models × 2
precisions × 32 steps). The validator passed all pairs, fixture parity checks and
decision consistency. Median/p95/max step time in microseconds was:

| Model | Float32 | Int8 weights, float state |
| --- | --- | --- |
| LIF | 20 / 21 / 36 | 19 / 21 / 34 |
| ALIF | 22 / 23 / 25 | 21 / 22 / 26 |
| ANN | 18 / 19 / 25 | 17 / 19 / 24 |

These timings cover prepared six-channel inputs and inference. They exclude raw
sensor acquisition, feature extraction, physical detection delay and energy.
Every repeated fixture decision matched the float reference; this remains a
timing/parity fixture, not an accuracy estimate.

Each model has 129 learned parameters: 112 weights and 17 biases. Logical
parameter storage is **516 bytes** for float32, or **188 bytes** for int8 weights,
two float32 scales, and float32 biases. Architecture/dynamics/decision constants
and metadata are additional. This comparison sketch compiles **all models and
both banks simultaneously**, so its build footprint does not demonstrate the
flash saved by a separately compiled int8-only deployment.

The common reusable state object is **260 bytes**. It contains voltage,
adaptation, hidden sums, counts, and a sample counter, including fields unused
by some architectures. It is not a minimal ANN allocation or total firmware
RAM. Input trace, record buffer, export references, runtime/stack, and all model
banks must also be accounted for. Printed byte buckets and Arduino's build
report distinguish these from theoretical parameter sizes.

## Reproduce the export and host checks

From the project folder:

```sh
python3 cpp/advanced_export_header.py
c++ -std=c++11 -O2 -ffp-contract=off -Wall -Wextra -pedantic cpp/advanced_test.cpp -o /tmp/gripper-advanced-test
/tmp/gripper-advanced-test
```

The generator reads `advanced/export/models.json`, validates dimensions, and
creates the self-contained sketch header. Host tests check adaptation/reset
ordering and all three models in both precision modes against Python fixtures.
`-ffp-contract=off` preserves the ordered float32 reference; do not enable
`-ffast-math`. Host correctness results are `host_reference`, never ESP32
measurements. Arduino compilation likewise creates no hardware capture.

The original GPIO and fixed-hidden SNN sketches are preserved as separate
experiments with different schemas.
