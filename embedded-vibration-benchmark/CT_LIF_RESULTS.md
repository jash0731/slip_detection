# CT-LIF with optimized mean inputs: measured results

The optimized mean-input branch has been integrated into CT-LIF as `ct_mean`. It retains cancellation and turnover features. The measured combined result is **69.19% balanced accuracy, 51.49% slip recall and 13.12% false-positive rate**. Adding the mean branch reduced false alarms relative to optimized mean-LIF alone, but decreased recall and balanced accuracy. This experiment does not demonstrate improvement in every aspect.

## Comparison

These percentages are macro averages across the three held objects and two seeds. False-positive rate measures negative windows classified as slip; it is not an alarm-events-per-hour measurement.

| Model | Balanced accuracy | Slip recall | False-positive rate |
|---|---:|---:|---:|
| Original CT-LIF | 68.83% | 66.23% | 28.57% |
| Earlier CT-LIF balanced variant | 67.98% | 58.86% | 22.90% |
| Optimized mean-LIF alone | 71.22% | 60.27% | 17.83% |
| **CT-LIF + optimized mean inputs** | **69.19%** | **51.49%** | **13.12%** |
| Tuned CT-only control | 71.82% | 63.29% | 19.64% |

The combined model reduces false-positive rate by 4.71 percentage points compared with optimized mean-LIF, while losing 8.78 points of recall and 2.04 points of balanced accuracy. The tuned CT-only control has the highest balanced accuracy among the new experiment's three families, but its false-positive rate exceeds the mean-only control's. It is a separate control, not the result of adding mean inputs. The combined model remained the prespecified primary variant after evaluation.

The combined model's pooled confusion counts are TP 2,133, TN 6,764, FP 912 and FN 2,209. Its pooled ordinary accuracy is 74.03%; this differs from macro balanced accuracy and should not replace it when reporting performance.

## What changed

The original CT branch produces 12 nonnegative channels: six sign-split gated normalized common-motion channels, three cancellation channels and three cancellation-times-turnover channels. The new branch appends six sign-split mean-motion channels, giving 18 inputs.

For each axis, the mean branch uses the mean pin derivative divided by a scale fitted only on training trials. The scale is the training 95th percentile of mean absolute pin derivative. This retains mean amplitude that instantaneous CT normalization can discard. All 18 channels pass through smooth compression `u / (1 + u)` before the LIF layer.

The neuron equation remains ordinary LIF: leak 0.875, threshold 1 and subtractive reset, followed by a mean-spike linear readout. This is an input-feature fusion, not a newly established neuron model. The CT cancellation and turnover pathways remain active even when opposing pin motion cancels the mean.

`ct_only` uses the same compression and training with only the 12 CT inputs. `mean_lif` uses the six mean inputs alone. The new mean-only control exactly reproduces the previously delivered optimized mean-LIF predictions.

## Training and evaluation

Each family receives the same eight-candidate grid: hidden width 8 or 12, learning rate 0.005 or 0.01, and uniform or recent pooling. With three families and six object/seed folds, this produces 144 fits. Recent pooling combines half uniform weight across 32 steps and half across the last eight; the final eight steps therefore receive 62.5% of the total weight.

Training uses Adam, batches of 64, at most 32 epochs, early-stopping patience 6 and gradient norm clipping at 5. Example weights give equal total weight to each observed object/class cell. A common gain between 0.8 and 1.25 augments the uncompressed feature vector. This is feature-space augmentation; scaling normalized CT features is not equivalent to changing raw pin amplitudes.

Scaling uses unique raw rows from training trials only. Architecture, checkpoint and threshold are selected by validation macro balanced accuracy, with lower false-positive rate, fewer parameters and name as tie breakers. Thresholds use a 0.005 grid plus an all-negative option. Missing validation-class rates remain undefined, including two folds without negative brush validation examples; available-class macro selection is retained. Consequently validation is limited by sparse class coverage.

All checkpoint hashes and selections were frozen before the separate test inference phase. No test-driven threshold or primary-model switch was made in this experiment. However, this benchmark has been used repeatedly in earlier development, so these results are exploratory rather than an untouched confirmation set.

Data come from the ARQ-CRISP slip dataset, source commit `a536ee3301af0566218782b00de334b6a6630979`. The prepared archive contains 228,521 raw samples, 90 trials and 54 channels (18 pins × three axes). Six original held-object/seed partitions, seeds 7 and 19, are retained. Evaluation uses 6,009 unique endpoints from 87 eligible trials and 12,018 predictions per model across both seeds. The seeds reuse endpoints and are not independent observations. Only original phase labels 4 and 5 enter the binary endpoint evaluation; other phases are excluded. Each window uses its final 34 raw frames.

## Object-level combined results

| Held object | Balanced accuracy | Slip recall | False-positive rate |
|---|---:|---:|---:|
| Brush | 78.74% | 71.78% | 14.30% |
| Screwdriver | 66.65% | 52.56% | 19.27% |
| Solder spool | 62.17% | 30.14% | 5.79% |

The low solder-spool false-positive rate accompanies substantial missed slips. An aggregate false-positive improvement should therefore not be presented as reliable detection for every object.

A paired whole-trial bootstrap with 2,000 draws gives combined-minus-mean-only intervals of −3.21 to −0.73 percentage points for balanced accuracy, −10.87 to −6.61 for recall and −6.84 to −2.72 for false-positive rate. These conditional intervals resample trials within the existing objects with fixed fitted models. They do not quantify uncertainty for new objects, retraining or model selection, and do not include a multiple-comparison correction.

## Efficiency and verification

| New family | Mean parameter count | Parameter range | Mean dense input MACs/window |
|---|---:|---:|---:|
| Combined CT + mean | 214.33 | 161–241 | 6,144 |
| Tuned CT-only | 141.00 | 113–169 | 3,840 |
| Mean-only | 86.33 | 65–97 | 2,048 |

Counts depend on the validation-selected width. Combined float32 parameter storage is 644–964 bytes, excluding activation state and runtime overhead. Input MAC counts are calculated as `32 × input_channels × hidden_width`; they exclude feature encoding and other operations. The original CT model has 225 parameters and 6,144 input MACs. The combined model is not uniformly smaller and adds feature computation. No new target-device latency, peak RAM or energy measurement supports a hardware-efficiency claim.

All four feature tests passed: exact branch mapping, retained cancellation/turnover under mean cancellation, retained mean amplitude and history/batch independence. Public inference verification reproduced all 36,054 decisions across the three new families, checked splits, labels, frozen source/weights and confusion counts. Saved models and the audit are included in `ct_lif_v4/results`. Training took approximately 178 seconds on the evaluation host; this is not embedded inference latency.

## Use and research claims

Start with `CT_LIF_START_HERE.md`. `ct_mean` is the default combined variant; `ct_only` and `mean_lif` are explicit alternatives. Delivered models are held-object research folds, not a deployment-calibrated all-object model. Outputs are uncalibrated sigmoid scores.

Publication novelty has not been established. A paper may study this fusion and its measured trade-off, but should not claim it is unpublished everywhere, superior to every algorithm, or a new LIF equation. A literature review, stronger baselines and fresh confirmation data are needed to support stronger claims. Alarm-event rates and detection delay also remain unmeasured. Prior studies and their reports are preserved alongside the new experiment.
