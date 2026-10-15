# Slip detection with CT-LIF

## Pressure-conditioned LIF prototype

A separate [pressure-conditioned LIF experiment](embedded-vibration-benchmark/pressure_lif/README.md) implements the uploaded ADXL345 + FSR protocol. Its Python and host C++ kernels passed software checks, including 20,000 randomized parity updates. **Real slip accuracy and ESP32 performance are unmeasured:** no matching sensor recordings with independent video labels are available. The tests also demonstrate that limited inhibition can suppress weak evidence and pressure decrease can create an alarm after recent acceleration. These software checks do not replace the physical pilot or held-out test session.

From `embedded-vibration-benchmark`:

```bash
python -B -m pressure_lif.validate
python -B -m pressure_lif.replay recordings.csv --config pressure_lif/starting_config.json
```

## Earlier nonlinear and temporal CT-LIF experiments


The project contains a nonlinear CT-LIF classifier and experimental temporal decision rules. The latest work adds smoothing with an immediate-alarm bypass, and a separate rule that preserves all existing alarms.

[Temporal approach: results, setup and streaming inference](embedded-vibration-benchmark/ct_lif_temporal/README.md)

| Selection | Balanced accuracy | Slip recall | False-positive rate | Positive segments detected |
|---|---:|---:|---:|---:|
| Tuned CT-only baseline | 71.82% | 63.29% | 19.64% | — |
| Current nonlinear model | 73.27% | 61.60% | 15.06% | 409/470 |
| Temporal bypass selected for validation balanced accuracy | 74.51% | 62.71% | 13.68% | 379/470 |
| Preserve existing alarms and add temporal detections | 73.47% | 62.24% | 15.30% | 409/470 |

The bypass improves window-level metrics but misses more positive segments. The alarm-preserving rule retains existing detections but increases false positives. The 470 segments include repeated seed evaluations and are not independent physical slip events. These are exploratory results from a repeatedly used benchmark; independent confirmation is needed.

From `embedded-vibration-benchmark`:

```bash
python -m pip install -r ct_lif_temporal/requirements.txt
python -B -m unittest ct_lif_temporal.test_streaming
python -B -m ct_lif_temporal.run
```

The temporal runner prints results by default. Use `--output ct_lif_temporal/outputs/run1` to write JSON results and validation choices to a new local directory. It does not save trained model weights.

The repository contains source, protocols and measured summaries. Prepared data and saved baseline model binaries are not all included. Restore the previously delivered complete CT_LIF_Tested_Project.zip into this checkout before running the experiments. Original dataset: https://github.com/ARQ-CRISP/slip_detection_dataset_2021.

The [nonlinear runner](embedded-vibration-benchmark/ct_lif_nonlinear/README.md) can save nonlinear model bundles locally with `--output ct_lif_nonlinear/outputs/run1`. Generated bundles and output folders are ignored by Git.

Earlier work remains available:
- [Nonlinear approach and its baseline comparison](embedded-vibration-benchmark/ct_lif_nonlinear/README.md)
- [Original CT-LIF setup](embedded-vibration-benchmark/CT_LIF_START_HERE.md)
- [CT + optimized mean-input results](embedded-vibration-benchmark/CT_LIF_RESULTS.md)
- [Trial-weighting and regularization experiments](embedded-vibration-benchmark/ct_lif_followup_20261013/README.md)

Publication novelty and hardware speed or energy gains have not been established. Twelve unreferenced console logs were removed previously; structured results and baseline dependencies remain.
