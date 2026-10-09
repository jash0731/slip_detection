# Verification record

Updated 7 October 2026. Software checks and first ESP32 capture completed.

## Follow-up temporal, robustness and error investigations

**50 Python tests passed** after adding 10 checks covering matched initialization/
capacity, conventional temporal state, joint frame shuffling/endpoint preservation,
ANN invariance, common endpoint schedules, validation-only tie rules, single-class
subgroups, annotated-history semantics, error-run gaps and paired shuffle counts.
The complete suite command remains `python3 -m unittest discover -s tests -v`.

The declared 108 grid fits plus 12 shuffled-training fits completed. All models use
the original six whole-trial partitions and training-only scaling. Saved histories,
choices-before-test records, raw predictions and completion metadata are in
`followup/results/`. Selected-model and order-control intervals retain all 2,000
paired trial resamples. The original three checkpoint types reproduce their saved
logits within 1e-5 before perturbation. ANN changes no decisions across reversal
or three shuffled draws. All 120 candidate thresholds are inside the original grid.

The original 60,090 prediction rows were audited with source-label checks, per-trial
and per-context confusion tables. Original export, prediction and metadata hashes
remain unchanged. Four new PNG/SVG figures were visually inspected; their captions
separate host-recorded quality, post-hoc error subgroups and observation support.
No follow-up checkpoint has been exported/compiled for Arduino; the previous
classic-ESP32 build verification applies to the frozen 32-sample sketch only.
Reproduction commands are in [FOLLOWUP_SOFTWARE.md](FOLLOWUP_SOFTWARE.md).

## Current advanced extension

```sh
python3 -m unittest discover -s tests -v
python3 advanced_stats.py
c++ -std=c++11 -O2 -ffp-contract=off -Wall -Wextra -pedantic cpp/advanced_test.cpp -o /tmp/gripper-advanced-test
/tmp/gripper-advanced-test
```

**40 Python tests passed** across the original and advanced pipelines. Additional
checks cover trial/object separation, causal endpoint windows, normalization,
surrogate-gradient learning, adaptation/reset behavior, quantization/export,
whole-trial bootstrap and complete advanced-capture integrity. Capture tests use
fabricated timing fixtures, not measurements. C++ checks passed for all three
models in float32 and int8-weight/float-state precision against the ordered Python
reference, with explicit adaptation and readout checks.

Actual source data checksums passed for all 90 trials. The bounded main training
comparison ran on CPU with PyTorch 2.9.1, NumPy 2.2.6 and Python 3.13.5;
`advanced/results/metadata.json` retains environment and split identifiers.
The separate frozen-LIF run completes the five-model controlled comparison.
Statistics regenerated from the saved raw predictions reproduce the reported
60.784% LIF and 57.209% ANN means. All 2,000 whole-trial bootstrap draws were
retained; no single-class omission occurred. The numeric audit covers 12,018
held-out evaluations per model. Three new PNG/SVG figures were visually inspected.

Advanced Arduino build:

```sh
arduino-cli compile --fqbn esp32:esp32:esp32 --warnings all firmware/esp32_advanced_replay
```

Espressif core **3.3.12**: **278,136 bytes flash**, **23,252 bytes static RAM**,
no warnings. This confirms classic-target compilation, not actual user's target
or execution. The frozen JSON export SHA256 is
`ab2c1a818ef17c181f240bdeb8d7bb37a5c2e134da4d210cc72fb5c06945245f`.
The comparison includes all models/banks, so these build bytes cannot demonstrate
the logical 516→188 byte per-model parameter saving.

The connected board identified as ESP32-D0WD-V3 (dual-core, 240 MHz) and uploaded
successfully on `/dev/cu.usbserial-0001`. The validated clean capture is
`hardware_results/advanced-run-clean.csv`; `advanced-run.csv` retains the original
few binary reset bytes before the text header. `analyze_advanced.py` accepted all
1,920 rows, all model/precision pairs and the completion marker. Median/p95/max
step timings in microseconds: LIF float20/21/36, LIF int8-weight19/21/34,
ALIF float22/23/25, ALIF int8-weight21/22/26, ANN float18/19/25, ANN int8-weight
17/19/24. All ten repeats per pair had zero decision drift. These are prepared
feature/inference timings only; they exclude raw preprocessing, acquisition,
physical onset and energy. Metadata and hashes support provenance but do not
independently authenticate a file as board-generated.

## First-stage checks, retained from 6 October

## Python integration

```sh
python3 -m unittest discover -s tests -v
```

**22 passed.** Covers existing GPIO capture validation, replay/activity capture
integrity, dense/sparse temporal parity, encoding, decay, training/export,
recording partitions, train-only calibration, ignored gaps and trial resets.
Capture-test timestamps are fabricated fixtures and are not results.

## Public-data reproduction

```sh
python3 -m neuromorphic.public_experiment
```

Rerun reproduced 306 training / 55 held-out windows and the report's confusion
counts/quality scores. SNN balanced accuracy 0.598901; EMA/scalar LIF 0.663919.
The exported public fixture passes host C++ checks. Timing columns are variable
descriptive host observations, not reliable public-method cost comparisons.

## Shared C++ kernel

```sh
c++ -std=c++11 -O2 -Wall -Wextra -pedantic cpp/tests/test_snn_runtime.cpp -o /tmp/gripper-snn-test
/tmp/gripper-snn-test
```

**Passed**, including per-tick dense/sparse equality, exported Python fixture,
encoding boundaries, decay/reset and 10,000 ticks at each of five activity levels.
The executable is host-only and generates no hardware evidence.

The saved host activity sweep has 100 rows, ten paired repetitions per level.
The plotting validator verifies all 50 checksums, and produces PNG/SVG figures
with explicit source labels. Both rendered plots were inspected.

## Arduino build

The temporary Arduino CLI toolchain compiled the replay sketch with:

```sh
arduino-cli compile --fqbn esp32:esp32:esp32 --warnings all firmware/esp32_snn_replay
```

Espressif Arduino core 3.3.12, classic ESP32 default target: **275,324 bytes flash,
23,836 bytes static RAM, no warnings** after adding all three commands. These
figures describe that target/build. The actual user's board still needs
identification, matching-target compilation, upload and capture.

The pre-existing loopback sketch remains separately compile-verified for a
classic dual-core ESP32. Its target restriction does not apply to the new
ordinary-loop replay sketch.

## Pending evidence

- Actual `s`, `p`, and `a` board captures and repeated boots.
- ESP32 crossover/quality-cost conclusion; no host timing is substituted.
- Physical onset, cheap-sensor transfer, energy and grip control, if later pursued.

When saving board captures, retain this exact sketch folder and generated header,
record board/port/toolchain/settings, and keep metadata/COMPLETE lines. Model and
code identifiers refer to the frozen source; changing weights or firmware requires
an updated identifier and a new result series.
