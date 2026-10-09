# Start here: improved compact LIF option

The latest study found a useful **mean-input LIF** option: 71.22% balanced accuracy, 60.27% slip recall and 17.83% false alarms, versus V2 balanced's 67.98%, 58.86% and 22.90%. It is slower than V2 on the host. The added residual CT candidate failed to improve the result and remains documented.

The useful option is an ordinary-LIF derivative. Its recommendation follows the held-object comparison and is exploratory; do not describe it as a proven novel CT algorithm. Read `CT_LIF_RESULTS.md` for trade-offs, earlier controls and uncertainty.

## Setup

Run commands from the extracted project root, the directory containing this file. The recorded environment was Python 3.12.14, NumPy 2.3.5 and Matplotlib 3.10.8. This study uses the included prepared raw data, so PyTorch and an HDF5 reader are not required.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r ct_lif_v3/requirements.txt
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r ct_lif_v3/requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

## Use the practical option

```python
import numpy as np
from ct_lif_v3.inference import load_optimized_detector

detector = load_optimized_detector(held=1, seed=7)
raw = np.load('my_windows.npy', allow_pickle=False)
result = detector.predict(raw)
print(result['probability'])
print(result['decision'])
```

Supply shape `(N, samples, 54)` or `(N, samples, 18, 3)`, with at least 33 consecutive raw frames per window, in the dataset's original units and pin/axis ordering. Retain the batch dimension for one window. Do not normalize before calling the detector.

`held=0/1/2` selects the brush/screwdriver/solder-spool research fold: that object was **excluded from training**. Seed is 7 or 19. There is no single all-object model calibrated for a new sensor or deployment. Scores are sigmoid outputs, not proven calibrated real-world probabilities; decision 1 means slip.

`load_optimized_detector` explicitly loads `model='mean_lif', policy='balanced'`. The generic `load_detector` defaults to the experimental residual CT family and is intended for reproducing that experiment. See `ct_lif_v3/README.md` for other controls.

## Verify or regenerate summaries

```bash
python -m unittest -v ct_lif_v3.test_core
python -m ct_lif_v3.verify
python -m ct_lif_v3.analyze
```

The audit verifies all 288 checkpoint hashes, frozen source/data hashes, original splits, endpoint labels, confusion counts and all 144,216 newly evaluated decisions through the inference interface. The analysis command recomputes metrics, conditional trial intervals and the chart from saved predictions.

For float32 C++ parity and host timing, install `g++` with C++17 support on the command path:

```bash
python -m ct_lif_v3.benchmark
```

This overwrites host-timing files with results from your machine. Preserve the delivered results first if needed. It does not benchmark ESP32 hardware or energy.

## Reproduce training

On a fresh extraction, preserve the completed results before starting another run:

```bash
python -c "from pathlib import Path; Path('ct_lif_v3/results').rename('ct_lif_v3/results_delivered')"
python -m ct_lif_v3.run train
python -m ct_lif_v3.run evaluate
python -m ct_lif_v3.analyze
python -m ct_lif_v3.verify
python -m ct_lif_v3.benchmark
```

The rename is intended once and fails if `results_delivered` already exists. Training refuses to overwrite an existing results directory; evaluation refuses to overwrite a completed summary. Keep the v1/v2 code and saved results because this experiment uses their frozen models and comparisons. Numerical-library differences can change training near spike thresholds.

## Files

| Path | Purpose |
|---|---|
| `CT_LIF_RESULTS.md` | Current results and interpretation |
| `ct_lif_v3/core.py` | Features, ordinary/residual LIF, training and selection |
| `ct_lif_v3/inference.py` | Saved-model prediction interface |
| `ct_lif_v3/results/optimized_models.json` | Six useful checkpoints with thresholds and fold identities |
| `ct_lif_v3/results/recommendation.json` | Explicit post-hoc family recommendation |
| `ct_lif_v3/results/protocol.json` | Protocol fixed before new training/test inference |
| `ct_lif_v3/results/all_choices_before_test.json` | Every pretest selection and all checkpoint hashes |
| `ct_lif_v3/results/aggregate.csv` | All model/mode scores |
| `ct_lif_v3/results/predictions.csv` | Every new held-object prediction |
| `ct_lif_v3/results/resources.csv` | Size, logical payload and host timing |
| `ct_lif_v3/results/analysis.json` | Full scores, selections and uncertainty |
| `ct_lif_v3/results/comparison.png` | Detection comparison figure |
| `ct_lif_v3/runtime.hpp` | New C++ inference kernel |
| `ct_lif_v3/models_generated.hpp` | All selected checkpoint exports used in the benchmark |
| `ct_lif_v3/README.md` | Algorithm details and C++ interface |

Earlier code and data remain preserved. Prior reports are `ct_lif_study/CT_LIF_V1_RESULTS.md` and `ct_lif_v2/CT_LIF_V2_RESULTS.md`. None of the delivered host results establishes deployed sensor or ESP32 performance.
