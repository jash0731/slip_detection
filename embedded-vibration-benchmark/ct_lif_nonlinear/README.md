# Nonlinear CT-LIF extension

This experimental extension improved balanced accuracy and false-positive rate on the reused benchmark, with a small reduction in slip recall and additional computation. It is not an improvement in every metric, and publication novelty is unverified.

## Measured results

| Method | Balanced accuracy | Slip recall | False-positive rate | Pooled accuracy |
|---|---:|---:|---:|---:|
| Existing tuned CT-only LIF | 71.82% | 63.29% | 19.64% | 73.61% |
| Primary constrained validation selection | 72.51% | 61.85% | 16.84% | 74.54% |
| Secondary maximum-BA validation selection | 73.27% | 61.60% | 15.06% | 75.07% |

The first three metrics are macro averages over six held-object/seed splits. Pooled accuracy weights every endpoint equally. Seeds reuse the same held-object test endpoints. Exact values and per-fold confusion matrices are in results.json; validation selections are in choices_before_test.json.

The primary rule requires validation recall to be at least the baseline and validation false-positive rate to be no higher, then maximizes validation balanced accuracy. The baseline remains eligible. These constraints did not both transfer to test data: test recall decreased. The secondary rule maximizes validation balanced accuracy without those constraints. Both selections were frozen across all six folds before test inference.

The original primary paired-trial bootstrap interval for the balanced-accuracy change was -0.1081 to +1.5592 percentage points (1,000 replicates). It includes no improvement and is conditional on existing objects and fitted models. No correction for repeated experiments was applied. This benchmark has been examined repeatedly; independent data are needed for confirmation.

## Method

The saved CT-only LIF backbone remains fixed. Its logit plus the mean, population standard deviation, and last-eight-step mean of each of the 12 compressed CT channels form a 37-feature vector. A histogram gradient-boosted classifier learns nonlinear interactions in this vector.

Training tests 3 or 7 leaves per tree, 40 or 80 boosting iterations, and learning rate 0.05 or 0.1: 48 fits across six splits. L2 regularization is 1, minimum samples per leaf is 20, early stopping is disabled, and training samples use the existing object/class-balanced weights. Validation uses the existing threshold grid and macro metrics.

The baseline input computation remains. The primary selection adds an average of 533.33 tree nodes and an average per-model bound of 199.67 tree comparisons per window. The secondary selection adds 600 nodes and a 228.67 comparison bound on average. Computing the feature summaries is also extra work. These counts are not measured hardware timing or energy savings.

The stage-two classifier is trained on training-set backbone scores, not out-of-fold scores. The backbone itself was already selected using the existing validation splits. This is an exploratory extension of that selected baseline, not an independent end-to-end model-selection study.

## Setup and reproduction

Run from embedded-vibration-benchmark with Python 3.11 or later:

```bash
python -m pip install -r ct_lif_nonlinear/requirements.txt
python -B -m ct_lif_nonlinear.run
```

The default command prints results without saving model bundles. It requires the prepared data/arq_all_trials.npz, baseline ct_lif_v4/results/*.npz checkpoints, and existing metadata. Restore these from the previously delivered complete CT_LIF_Tested_Project.zip. A Git-only checkout does not contain all those binary inputs. Data provenance: https://github.com/ARQ-CRISP/slip_detection_dataset_2021.

To save trained models and results locally, choose a new output directory:

```bash
python -B -m ct_lif_nonlinear.run --output ct_lif_nonlinear/outputs/run1
```

The runner refuses to overwrite an existing output directory. It saves validation selections before evaluating tests, then saves 12 local model bundles (six folds, two selection rules). Outputs and bundles are ignored by Git. No trained weights or dataset binaries are added in this commit.

## Inference

```python
import joblib
from ct_lif_nonlinear.run import predict

bundle = joblib.load("ct_lif_nonlinear/outputs/run1/object0_seed7_balanced.joblib")
scores, decisions = predict(bundle, raw_windows)
```

raw_windows must have shape N x at-least-34 x 18 x 3, in the original sensor ordering. Only the last 34 frames are used. Scores are classifier outputs, not calibrated slip probabilities. Each bundle belongs to a specific evaluation fold; this experiment does not train a final deployment model on all available objects.

## Verification and cleanup

The packaged runner reproduced the original aggregate metrics. All 12 temporary saved bundles were reloaded and reproduced their test confusion matrices; inference was also checked for batch independence. Temporary verification bundles were removed after testing.

Twelve unreferenced console logs from earlier CT-LIF runs were removed from the repository. Structured experiment results, source code, baseline dependencies and historical comparisons were retained. This cleanup does not claim that every historical file is needed by the new approach.
