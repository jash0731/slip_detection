# CT-LIF follow-up experiments — 13 October 2026

These experiments did not establish an overall improvement over tuned CT-only LIF. The existing model and inference implementation are unchanged.

## Latest experiment: trial weighting and regularization

72 new fits: three training regimes, two temporal pooling modes, two learning rates, and six object/seed splits. Each candidate has exactly the hidden-layer size of the saved CT-only baseline for its split (8 or 12 neurons).

- Trial balancing gives equal total training weight to each trial within each observed object/class cell, while preserving equal cell weights.
- Regularization adds 0.001 times the current weight to gradients for W and w; biases are not penalized. This corresponds to a quadratic penalty with coefficient 0.001/2.
- Combined training applies both.
- CT features, beta=0.875, 32 time steps, feature compression, augmentation, Adam, early stopping, validation loss and threshold grid remain as in the existing v3/v4 training.
- Primary selection considers only validation thresholds whose recall is no lower and false-positive rate no higher than the saved baseline. It maximizes validation balanced accuracy, then prefers lower false-positive rate and fewer parameters, with a name tie-break. The baseline is an eligible fallback.
- Secondary selection maximizes validation balanced accuracy without the joint constraint.
- Every split's choice is fixed before any test inference in this run.

| Method | Macro balanced accuracy | Macro slip recall | Macro false-positive rate | Pooled accuracy |
|---|---:|---:|---:|---:|
| Existing tuned CT-only LIF | 71.8237% | 63.2907% | 19.6432% | 73.6146% |
| Primary validation selection | 71.7058% | 62.7445% | 19.3328% | 73.5064% |
| Secondary validation selection | 71.1882% | 57.7790% | 15.4025% | 73.8226% |
| Trial-balanced family | 69.1488% | 51.4341% | 13.1364% | 72.5994% |
| Regularized family | 70.5814% | 61.4852% | 20.3223% | 73.1985% |
| Trial-balanced + regularized family | 68.9467% | 53.3878% | 15.4945% | 72.4913% |

All methods average 141 parameters and 3,840 input multiply-accumulates per window. These are operation counts, not hardware latency or energy measurements.

Primary selection retains the baseline in five splits. Only held object 1, seed 19 changes to regularized uniform pooling, 12 neurons, learning rate 0.01.

Paired trial bootstrap (1,000 replicates, seed 91) gives these conditional 95% intervals for primary minus baseline, in percentage points:

| Metric | Lower | Upper |
|---|---:|---:|
| Balanced accuracy | -0.2886 | +0.0127 |
| Recall | -0.7611 | -0.3643 |
| False-positive rate | -0.5180 | -0.1223 |

Trials are resampled within held objects, sharing each resample across seeds and methods. These intervals do not include retraining, new-object uncertainty or correction for repeated experiments.

## Previous memory-length experiment

192 new fits varied beta over 0.5, 0.75, 0.9375 and 0.96875, hidden size over 4 and 8, pooling over uniform and recent, and learning rate over 0.005 and 0.01 across the same six splits.

The primary validation selection achieved 71.1789% balanced accuracy, 61.7505% recall and 19.3928% false-positive rate. Its mean parameter count was 122.33 and mean input MAC count 3,328, a 13.33% input-MAC reduction from baseline. The lower computation came with lower recall and balanced accuracy.

## Reproduce the latest 72-fit study

The adjacent run_training_refinements.py reconstructs the executed in-memory experiment as a standalone runner. It prints aggregate and per-fold results without writing checkpoints or reports. Bootstrap intervals above are recorded from the original run; the runner does not recompute bootstrap intervals.

From embedded-vibration-benchmark:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python ct_lif_followup_20261013/run_training_refinements.py
```

Use the previously delivered complete CT_LIF_Tested_Project.zip to supply data/arq_all_trials.npz, the saved ct_lif_v4/results/*.npz baseline checkpoints, and accompanying metadata. These binary inputs are not included in this commit. The existing load_data and verify_splits functions verify the dataset and split protocol.

This script adapts the existing training function in memory with checked source substitutions to preserve the original optimizer behavior. Changes to that training function may require reviewing the adapter.

## Interpretation and artifact status

Results are macro averages over three held objects and two seeds unless labeled pooled. Seed repetitions reuse test endpoints; they are not independent new datasets. The benchmark has been repeatedly examined, so these are exploratory findings, not fresh independent confirmation. No publication novelty, universal superiority or hardware-efficiency claim is established.

The latest run checked SHA-256 contents of all 1,973 local project files before and after execution; all were unchanged. Experimental weights were not saved. This commit adds the report, numeric results and reproduction script only; it does not replace the existing CT-LIF model.
