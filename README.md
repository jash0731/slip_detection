# Slip detection with CT-LIF

The latest experimental approach adds a nonlinear classifier to the existing CT-LIF score and CT-feature summaries.

[Nonlinear approach: setup, training and inference](embedded-vibration-benchmark/ct_lif_nonlinear/README.md)

| Model | Balanced accuracy | Slip recall | False-positive rate |
|---|---:|---:|---:|
| Tuned CT-only baseline | 71.82% | 63.29% | 19.64% |
| Nonlinear extension selected for validation balanced accuracy | 73.27% | 61.60% | 15.06% |

The extension improves balanced accuracy and reduces false positives on this repeatedly used benchmark, with lower recall and additional tree computation. Independent confirmation is needed; publication novelty is unverified. The stricter validation-constrained selection achieved 72.51% test balanced accuracy.

From `embedded-vibration-benchmark`:

```bash
python -m pip install -r ct_lif_nonlinear/requirements.txt
python -B -m ct_lif_nonlinear.run
```

The repository contains source, protocols and measured summaries. Prepared data and saved baseline model binaries are not all included. Restore the previously delivered complete CT_LIF_Tested_Project.zip into this checkout before running the experiment. Original dataset: https://github.com/ARQ-CRISP/slip_detection_dataset_2021.

The nonlinear runner can save model bundles locally with `--output ct_lif_nonlinear/outputs/run1`. Generated model bundles and outputs are ignored by Git.

Earlier work remains available:
- [Original CT-LIF setup](embedded-vibration-benchmark/CT_LIF_START_HERE.md)
- [CT + optimized mean-input results](embedded-vibration-benchmark/CT_LIF_RESULTS.md)
- [Trial-weighting and regularization experiments](embedded-vibration-benchmark/ct_lif_followup_20261013/README.md)

Twelve unreferenced console logs were removed; structured results and baseline source were retained.
