# Temporal extensions for nonlinear CT-LIF

These experiments add causal temporal decisions to the nonlinear CT-LIF model. They improve some window-level metrics, but do not establish higher recall, fewer false alarms and preservation of short slip segments together.

## Reproduced results

| Selection | Macro balanced accuracy | Macro slip recall | Macro false-positive rate | Pooled accuracy | Positive segments detected |
|---|---:|---:|---:|---:|---:|
| Current nonlinear model | 73.27% | 61.60% | 15.06% | 75.07% | 409/470 |
| Previous smoothing control | 74.16% | 62.26% | 13.93% | 75.82% | 372/470 |
| Bypass, maximum validation balanced accuracy | 74.51% | 62.71% | 13.68% | 76.04% | 379/470 |
| Bypass, validation recall/FPR constraints | 74.26% | 63.81% | 15.29% | 75.64% | 390/470 |
| Bypass, additional validation event constraint | 74.23% | 63.71% | 15.26% | 75.64% | 392/470 |
| Preserve existing alarms and add temporal detections | 73.47% | 62.24% | 15.30% | 75.17% | 409/470 |

The first three metrics average three held objects and two seeds. Pooled accuracy and segment counts use all six folds. The two seeds reuse test endpoints: 470 evaluated segments are not 470 independent physical slip events. A segment is a contiguous run of positive labels on the sampled evaluation grid, broken by a trial change or an endpoint gap other than 16 frames.

Exact metrics, confusion matrices and event comparisons are in results.json. Fold-specific settings and their validation metrics are in choices_before_test.json.

The bypass selection for balanced accuracy improves window-level balanced accuracy by 1.24 percentage points and reduces false-positive rate by 1.38 points relative to the current nonlinear model. It loses 30 previously detected positive segments, with an average first-hit delay increase of about 20 ms among segments detected by both methods.

The additional validation event constraint does not guarantee preservation on test data: it still loses 17 segments. Its name describes a validation selection rule, not a test-time guarantee.

The add-only rule preserves every original positive decision by construction. In this evaluation it adds 29 true-positive windows and 17 false-positive windows, retains all 409 detected segments, and detects no previously missed segment. It cannot reduce false-positive rate, because it never removes an existing alarm.

## Methods and selection

The original nonlinear model is reconstructed from its saved validation choices. CT-LIF weights, nonlinear feature extraction and tree hyperparameters remain unchanged. Only temporal rules and thresholds are selected anew.

For each trial, the smoother updates s = alpha * current_score + (1 - alpha) * previous_s. It initializes from the first available score. Memory resets on a trial change or a gap other than the expected 16-frame step. Predictions never use class labels; labels are used only for validation selection and evaluation.

The bypass combines smoothing with separate on/off thresholds and an immediate current-score trigger. The immediate trigger does not change the smoother's alarm state. There are 16 rules: alpha in {0.4, 0.7}, threshold gap in {0, 0.05}, and immediate-trigger margin in {0, 0.05, 0.15, 0.25}. Two smoothing-only rules serve as controls.

The add-only approach always keeps current_score >= original_threshold. It can add a detection when the smoothed score passes an additional threshold and the current score is no more than a chosen guard below the original threshold. There are six rules: two alpha values and guard in {0.05, 0.1, 0.2}.

Each rule tests 201 thresholds from 0 to 1 at spacing 0.005, plus an all-negative threshold just above 1. These are inexpensive operating-point evaluations, not thousands of newly trained neural models. Only six nonlinear reference classifiers are retrained.

The recall-constrained rule requires validation recall at least the current model's recall and validation false-positive rate no higher, then maximizes recall. The event-constrained rule additionally requires the macro fraction of detected positive validation segments to be no lower. Both retain the current model as a fallback. The add-only rule permits no increase in validation false-positive rate, with fallback to the current model. Constraints can fail to transfer to test data.

All fold selections are frozen before any test inference. The experiment does not select the best approach using test metrics to produce an independently validated final winner.

## Reproduction

From embedded-vibration-benchmark:

```bash
python -m pip install -r ct_lif_temporal/requirements.txt
python -B -m unittest ct_lif_temporal.test_streaming
python -B -m ct_lif_temporal.run
```

The default runner prints results without writing outputs. To save JSON results and choices to a new directory:

```bash
python -B -m ct_lif_temporal.run --output ct_lif_temporal/outputs/run1
```

The output directory must not already exist. The runner saves choices before test evaluation. It does not save trained weights. Prepared data and baseline NPZ checkpoints are required; restore them from the previously delivered complete CT_LIF_Tested_Project.zip as explained in ../ct_lif_nonlinear/README.md. A Git-only checkout lacks these binary inputs.

The packaged runner reproduced all reported metric triples and segment counts to numerical tolerance 1e-12. Streaming decisions were checked against batch evaluation for every selected test configuration. Unit tests cover causality, trial/gap resets, chronological ordering and preservation of existing alarms.

## Streaming inference

Generate the original nonlinear balanced-model bundles locally using the nonlinear runner. Choose the same held object and seed in choices_before_test.json and in the model bundle. For example:

```python
import json
import joblib
from ct_lif_nonlinear.run import predict
from ct_lif_temporal.core import TemporalAlarm

folds = json.load(open("ct_lif_temporal/choices_before_test.json"))
fold = next(f for f in folds if f["held"] == 0 and f["seed"] == 7)
choice = fold["selections"]["add_only"]
alarm = TemporalAlarm(choice, fold["current_threshold"])
bundle = joblib.load("ct_lif_nonlinear/outputs/run1/object0_seed7_balanced.joblib")
scores, _ = predict(bundle, raw_window_batch)
decision = alarm.step(float(scores[0]), group=0, end_sample=63)
```

Keep the same TemporalAlarm instance for subsequent windows and call step in chronological order. Provide a distinct group ID per trial and the actual frame index of each endpoint. Only past and current scores enter a prediction. Alarm state is caller-owned; use a separate instance for each concurrent stream.

The additional state is a smoothed score, an alarm flag and the previous endpoint identity. Tree size does not increase. Arithmetic and comparisons are added per window; embedded timing and energy have not been measured. Scores are not calibrated slip probabilities. Fold-specific models are evaluation artifacts, not a final deployment model trained on every object.

## Interpretation limits

The benchmark has been examined repeatedly. Results are exploratory and need independent confirmation. A gain in macro window recall can coexist with fewer detected segments or fewer pooled true positives because folds and segments have different weights.

Earlier conditional paired-trial bootstrap intervals for the balanced-accuracy-selected bypass versus the current nonlinear model were +0.67 to +1.92 percentage points for balanced accuracy, +0.23 to +2.16 for recall, and -2.24 to -0.55 for false-positive rate. They condition on the existing objects and fitted models, omit retraining uncertainty, and do not correct for repeated searches. This runner reproduces point estimates and event comparisons, not bootstrap intervals.

No claim of publication novelty, universal superiority, guaranteed deployment performance or measured hardware efficiency is made.
