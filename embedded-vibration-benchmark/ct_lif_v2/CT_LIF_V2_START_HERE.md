# Start here: CT-LIF v2

The modified algorithm, trained weights and measured results are included. Read `CT_LIF_RESULTS.md` before choosing a mode: the strict mode lowers false alarms but detects only 38.10% of slip windows. Ordinary mean LIF remains a strong, often better control.

The revision is in `ct_lif_v2/`. The previous study remains in `ct_lif_study/`, with its reports preserved as `CT_LIF_V1_RESULTS.md` and `CT_LIF_V1_START_HERE.md` inside that directory. Original uploaded source and data files are preserved.

## Environment

Run commands from the extracted project root, the directory containing this file. The recorded run used Python 3.12.14, NumPy 2.3.5 and Matplotlib 3.10.8. This NumPy study does not require PyTorch or an HDF5 reader because it uses the included prepared raw data archive.

Create a virtual environment if needed:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r ct_lif_v2/requirements.txt
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r ct_lif_v2/requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

The C++ host benchmark additionally requires `g++` with C++17 support on the command path. It was run with GCC 13.3.0 on Linux; its binary is compiled locally and is not distributed.

## Inspect and verify the delivered run

```bash
python -m unittest -v ct_lif_v2.test_core
python -m ct_lif_v2.verify
```

The audit checks the frozen training/data hashes, exact original trial splits, endpoint labels, 192 checkpoints, confusion counts, and all 36,054 v2 decisions through the inference interface.

To recompute the summary, exploratory intervals and chart from saved predictions:

```bash
python -m ct_lif_v2.analyze
```

To repeat the host timing and float32 parity study:

```bash
python -m ct_lif_v2.benchmark
```

This overwrites host benchmark outputs with measurements from your machine. Preserve the delivered results first if you need to retain the original measurements. It does not measure ESP32 latency or energy.

## Reproduce training and evaluation

The delivered results are already complete. Use a fresh extraction, then preserve them once before starting a new run:

```bash
python -c "from pathlib import Path; Path('ct_lif_v2/results').rename('ct_lif_v2/results_delivered')"
python -m ct_lif_v2.run train
python -m ct_lif_v2.run evaluate
python -m ct_lif_v2.analyze
python -m ct_lif_v2.verify
python -m ct_lif_v2.benchmark
```

The rename fails if `results_delivered` already exists. Training refuses to overwrite an existing `results` folder, and evaluation refuses to overwrite an existing summary. The v1 checkpoints in `ct_lif_study/results` must remain in place because the v2 experiment recalibrates those controls. Different numerical environments may produce different training trajectories near spike thresholds.

Training saves every checkpoint and operating-point choice before the separate test phase. Changing settings after seeing test scores is a new exploratory experiment, not independent confirmation.

## Main files

| File or directory | Purpose |
|---|---|
| `CT_LIF_RESULTS.md` | Measured results, comparisons, costs and limitations |
| `ct_lif_v2/README.md` | Algorithm interface and model-use details |
| `ct_lif_v2/core.py` | Encoders, augmentation, training and threshold selection |
| `ct_lif_v2/inference.py` | Load a saved fold model and classify raw windows |
| `ct_lif_v2/run.py` | Frozen training, split checks and held-object evaluation |
| `ct_lif_v2/runtime.hpp` | Float32 C++ encoding and inference |
| `ct_lif_v2/models_generated.hpp` | Exported checkpoints used in the C++ study |
| `ct_lif_v2/results/protocol.json` | Frozen settings, environment and source hashes |
| `ct_lif_v2/results/all_choices_before_test.json` | Validation choices made before test inference |
| `ct_lif_v2/results/*.npz` and matching JSON | 192 fitted models and validation histories |
| `ct_lif_v2/results/predictions.csv` | Every evaluated model/mode prediction |
| `ct_lif_v2/results/aggregate.csv` | All aggregate scores, including recalibrated controls |
| `ct_lif_v2/results/per_object.csv` | Object-level detection and false-alarm rates |
| `ct_lif_v2/results/analysis.json` | Metrics, trial intervals and v2 selections |
| `ct_lif_v2/results/false_alarm_tradeoff.png` | Recall versus false alarms |
| `ct_lif_v2/results/resources.csv` | Model size, calculated payloads and host timing |
| `ct_lif_v2/results/resources_by_fold.csv` | Checkpoint costs and float32 parity |
| `ct_lif_v2/results/verification.json` | Reproduction audit results |

The saved models belong to particular held-object folds. There is no all-object model calibrated for a new deployment. The 10% and 20% modes describe validation targets; they are not guarantees on new inputs.
