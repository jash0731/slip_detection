# ESP32 spiking inference for robotic-gripper slip detection

**Research question:** How do training, adaptive neurons and weight precision affect
classification quality and deployment cost on a low-cost ESP32?

Robotic grippers remain the application. The course hardware demo runs recorded
contact inputs on the ESP32 using only a USB data cable. A physical gripper,
motor or extra component is not required. The current main experiment trains
6-input/16-hidden-unit LIF, adaptive LIF and matched ANN models on public tactile
recordings. It is a slip-state classification prototype; incipient onset and
closed-loop gripping are future validation questions.

## Start here

- [Latest follow-up report](FOLLOWUP_REPORT.md): temporal controls, robustness and missed-slip/false-alarm analysis.
- [Initial trained-model report](ADVANCED_REPORT.md): fixed 32-sample comparison, quantization and deployment limits.
- [Plain-language explanation and viva guide](VIVA_GUIDE.md).
- [ESP32 upload/run instructions](ADVANCED_HARDWARE.md): current USB-only comparison.
- [Delivery checklist](DELIVERY_CHECKLIST.md): completed work and board checks still pending.
- [Additional literature and experimental contract](RESEARCH_UPGRADE.md).
- [Exact model/training protocol](ADVANCED_SOFTWARE.md), [data audit](ADVANCED_DATA_REVIEW.md), and [verification record](VERIFICATION.md).

## Current software findings

The follow-up completes 108 fixed candidate and 12 shuffled-training fits. With
window/rate chosen independently for each model using validation only, mean
held-object balanced accuracy is **62.9% LIF, 59.9% conventional leaky-ReLU,
57.2% mean-ReLU ANN**. Shuffling past frames lowers LIF quality, supporting a
contribution from temporal order. The advantage does not survive every training
configuration. Selected LIF still misses **82%** of annotated slip windows on the
held-out solder spool. These are bounded exploratory results, not a ready grip alarm.
See [FOLLOWUP_REPORT.md](FOLLOWUP_REPORT.md) and
[reproduction commands](FOLLOWUP_SOFTWARE.md). The Arduino export below remains
the original frozen 32-sample comparison; follow-up models are software-only.

### Initial fixed 32-sample results, retained

Ninety original trials across three objects provide 6,009 unique scored test
windows across 87 eligible trials. Entire objects are held out. Normalization
is fit on training trials; epochs and thresholds use internal validation.
Two predetermined seeds are averaged within each object, then objects equally.

| Model | Mean held-object balanced accuracy |
| --- | ---: |
| Logistic control | 57.3% |
| Matched ANN | 57.2% |
| Frozen LIF, trained readout | 57.4% |
| Trained LIF | **60.8%** |
| Trained adaptive LIF | 57.5% |

The LIF advantage is modest and varies by object. Int8 weight storage reduces
logical parameter bytes from 516 to 188, with some changed decisions. These
are host evaluation and parameter-accounting results. No ESP32 execution-time,
energy or physical slip-onset result has been collected.

## Reproduce on a computer

Commands run from this folder. Training needs Python, NumPy, h5py and CPU PyTorch;
plotting needs Matplotlib. Saved results and the Arduino sketch can be inspected
without retraining. The Arduino sketch has no Python or PyTorch dependency.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-advanced.txt
python3 data/fetch_arq_all.py
python3 data/prepare_arq_all.py
python3 -m advanced.train --epochs 18 --seeds 7 19
python3 -m advanced.frozen_ablation
python3 -m advanced.numeric_audit
python3 advanced_stats.py
python3 plot_advanced_results.py
python3 cpp/advanced_export_header.py
python3 -m unittest discover -s tests -v
```

The fetch requires internet and approximately 266 MB of original HDF5 data.
Hashes and licences are retained; raw downloads and prepared data are ignored
by Git. Training produces raw predictions, histories, checkpoints and metadata
in `advanced/results/`, and a predetermined deployment export in `advanced/export/`.
Exact reproduction environment is recorded in the results metadata.

## Run the hardware demo

Open `firmware/esp32_advanced_replay/esp32_advanced_replay.ino` in Arduino IDE,
select the matching board and serial port, then upload. Keep both supplied
headers in the sketch folder. Serial Monitor: **115200 baud**, reset, send **s**.
Save every line through `# COMPLETE`, including metadata, then run:

```sh
python3 analyze_advanced.py hardware_results/advanced-run.csv
```

The board compares LIF/ALIF/ANN with float32 and int8-weight/float-state storage,
with numerical parity checks and ten repetitions per combination. The saved
fixture is deliberately fixed without screening for correct predictions; it is
misclassified by all three exported models. This is a computation/parity demo,
not an on-board accuracy test. Raw tactile preprocessing is offline and excluded
from board timing. Follow [ADVANCED_HARDWARE.md](ADVANCED_HARDWARE.md) for details.
The connected ESP32-D0WD-V3 was uploaded successfully. The first validated
1,920-row capture is `hardware_results/advanced-run-clean.csv`: median step
times are 17–22 µs across models/precisions, with zero decision drift. These
times cover prepared-input replay only; they are not physical sensing latency.

## Keep evidence separate

| Evidence | Location | Meaning |
| --- | --- | --- |
| Follow-up host investigations | `followup/results/`, `followup/failures/` | Temporal/robustness comparisons and error audit; no board timing |
| Current public-recording host evaluation | `advanced/results/` | Trained models on authors' real gripper recordings; our computer results |
| Earlier public-recording host evaluation | `neuromorphic/public_results/` | Different first-stage protocol; not directly comparable accuracy |
| Synthetic logic/activity tests | `neuromorphic/synthetic_results/` | Artificial labels/workloads; no real slip accuracy |
| Host C++ activity sweep | `software_results/host_cpp_activity.csv` | Host CPU timings; not ESP32 measurements |
| Actual board captures, after upload | `hardware_results/` | ESP32 computation; no live sensing or physical onset validation |
| Original GPIO simulation | `simulation_results.csv` | Simulated acquisition only |

Runtime, spikes and storage are not measured energy. Preserve source fields,
model/input hashes and complete metadata with any exported figure or table.

## Supporting first-stage experiments

The initial fixed-hidden 4-input/12-LIF pipeline and dense/sparse activity sweep
remain reproducible through [SOFTWARE_EXPERIMENT.md](SOFTWARE_EXPERIMENT.md),
[RESEARCH_REPORT.md](RESEARCH_REPORT.md), [SNN_HARDWARE.md](SNN_HARDWARE.md), and
[cpp/README.md](cpp/README.md). Its sketch supports s/p/a; those commands and
CSV schemas are separate from the current advanced comparison.

The original 1/5/10/20 ms polling versus GPIO interrupt experiment remains in
`firmware/esp32_acquisition/esp32_acquisition.ino`. [EXPERIMENT.md](EXPERIMENT.md)
retains its raw CSV schema; [HARDWARE.md](HARDWARE.md) covers classic-ESP32
GPIO25 → 1 kΩ → GPIO26 loopback wiring. Use `analyze_hardware.py` for its capture.
This measures electrical acquisition latency and does not establish slip accuracy.
An unidentified vibration sensor needs its own acquisition setup; it is not
interchangeable with the public multi-axis tactile sensor.
