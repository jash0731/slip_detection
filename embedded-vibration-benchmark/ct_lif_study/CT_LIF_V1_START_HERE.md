# Start here: completed CT-LIF evaluation

Read `CT_LIF_RESULTS.md` first. **Measured CT-LIF result: 68.83% mean balanced accuracy.** This is not the best new method in the study; the 225-parameter mean-input LIF scored 71.31% and the turnover-removed ablation scored 70.96%.

The complete project includes the source recordings, prepared dataset, original project results, added CT-LIF code, 48 fitted checkpoints, predictions and analysis. The new CT-LIF module uses NumPy only for training and matplotlib for the chart; it does not require PyTorch or h5py when using the included prepared dataset. Original project commands may require their separate requirements files.

## Run on your laptop

Python 3.11+ is required for the pinned NumPy version. Tested here using Python 3.12.14 and NumPy 2.3.5.

Open a terminal in the extracted `embedded-vibration-benchmark` directory:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r ct_lif_study/requirements.txt
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r ct_lif_study/requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

To recompute the delivered summary, bootstrap and chart from saved predictions:

```bash
python -m ct_lif_study.analyze
```

To re-run the model tests:

```bash
cd ct_lif_study/reference
python -m unittest -v test_ct_lif.py
cd ../..
```

## Reproduce training and held-object evaluation

The delivered `results` folder is complete. Preserve it before training a new run:

```bash
python -c "from pathlib import Path; Path('ct_lif_study/results').rename('ct_lif_study/results_delivered')"
python -m ct_lif_study.run train
python -m ct_lif_study.run evaluate
python -m ct_lif_study.analyze
```

The rename command is intended once on a fresh extraction. It will fail if `results_delivered` already exists; use a fresh extraction for another independent run. Training refuses to overwrite an existing results folder. No model settings should be changed after examining test scores without declaring a new exploratory experiment.

Training loads the original `advanced/results/metadata.json` split groups and asserts exact agreement with `advanced.data.split_trials`. The prepared dataset checksum is verified against both the old experiment metadata and the data manifest. The study writes all checkpoint/threshold selections before its separate test phase. Results can differ across numerical-library implementations, so retain your environment and compare raw predictions if exact reproduction matters.

The recorded runtime is from this computer and does not predict your laptop's duration. Neither training runtime nor host prediction runtime is an ESP32 benchmark.

## Files to inspect

| Path | Contents |
|---|---|
| `CT_LIF_RESULTS.md` | Results, limitations and interpretation |
| `ct_lif_study/reference/CT_LIF_ALGORITHM.md` | Original candidate specification, preserved as written before evaluation; its “not evaluated” status describes that earlier stage |
| `ct_lif_study/reference/ct_lif.py` | Unchanged encoder and NumPy models/trainer |
| `ct_lif_study/run.py` | Dataset adapter, split checks, training and frozen evaluation |
| `ct_lif_study/analyze.py` | Macro metrics, paired trial intervals and chart |
| `ct_lif_study/results/protocol.json` | Settings, source hashes and environment frozen before training |
| `ct_lif_study/results/*_selection.json` | Epoch histories, validation metrics, threshold and fold data for each fit |
| `ct_lif_study/results/*.npz` | Fitted model weights and preprocessing parameters |
| `ct_lif_study/results/predictions.csv` | Every new test prediction with object, trial, endpoint and seed |
| `ct_lif_study/results/analysis.json` | Full numerical results and uncertainty |
| `ct_lif_study/results/ct_lif_results.png` | Comparison figure |

The recorded study has no hyperparameter sweep. The wider grids proposed in the original candidate document were optional future work and were not executed. There is no single trained “all-object deployment” checkpoint: each model belongs to a held-object fold and should be used with its own scale and threshold.
