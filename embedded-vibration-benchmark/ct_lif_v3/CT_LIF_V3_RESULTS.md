# CT-LIF follow-up: improved compact LIF option

9 October 2026. Completed recorded-data study; source, fitted weights, predictions and reproducible checks are included.

**The optimized mean-input LIF improves all three average detection measures over CT-LIF v2's balanced mode:** balanced accuracy increases from **67.98% to 71.22%**, slip recall increases from **58.86% to 60.27%**, and false alarms decrease from **22.90% to 17.83%**.

The proposed residual CT extension did not improve the result. The useful option is the simpler ordinary-LIF derivative tested alongside it. It is explicitly named `mean_lif`, not presented as a novel superior CT algorithm. Recommending this family follows the test comparison and is an **exploratory, post-hoc choice**, although every family's architecture, weights and thresholds were selected using training/validation data before this test phase.

The useful variant is still compact, but **it is slower than V2 on this host**. It does not improve every engineering measure. Its slip-recall gain is small and not statistically conclusive in the conditional trial bootstrap.

## 1. Measured comparisons

| Model / mode | Balanced accuracy | Slips detected | False-alarm rate |
|---|---:|---:|---:|
| Previous CT-LIF v2, balanced | 67.98% | 58.86% | 22.90% |
| Optimized mean-input LIF, balanced | 71.22% | 60.27% | 17.83% |
| Optimized mean-input LIF, joint policy | 71.08% | 58.33% | 16.16% |
| CT residual experiment, primary joint policy | 65.39% | 54.74% | 23.96% |
| CT residual experiment, balanced | 65.61% | 52.86% | 21.63% |
| Same amplitude features, spike-only readout, balanced | 66.89% | 48.11% | 14.33% |
| Linear control, balanced | 55.93% | 40.26% | 28.39% |
| Previous ordinary-LIF control, balanced | 69.37% | 60.25% | 21.50% |
| Earlier 28-neuron mean LIF, original | 71.31% | 70.54% | 27.92% |
| Original CT-LIF | 68.83% | 66.23% | 28.57% |

Scores average equally over the three held-out objects and two seeds. Recall is the proportion of actual slip windows detected. False-alarm rate is the proportion of actual non-slip windows incorrectly classified as slip. Balanced accuracy is `(recall + 1 - false-alarm rate)/2`.

Compared with V2 balanced, the useful variant gains **3.24 percentage points** of balanced accuracy and **1.41 points** of slip recall, while reducing false alarms by **5.07 points** (about 22.2% relative reduction). Its ordinary pooled accuracy is **74.40%**; that differently weighted score must not replace the 71.22% balanced-accuracy comparison.

The earlier 28-neuron mean LIF remains slightly higher in balanced accuracy at 71.31% and detects more slips, at the cost of more false alarms. The new variant is not superior to every earlier operating point. V2 strict has fewer false alarms than the new balanced variant, but also much lower slip recall; the improvement claim here is specifically versus **V2 balanced**, not V2 strict.

## 2. What the useful variant changes

For each axis and time step, take the mean of the 18 pin differences, `m`. Compute an axis scale `s` as the 95th percentile of mean absolute pin differences using unique raw rows in training trials only. Set:

`z = m / (s + abs(m))`

Then split each axis into positive and negative parts, giving six nonnegative input channels. This smooth compression preserves ordering among large amplitudes instead of clipping all of them to the same maximum. The corresponding training implementation computes the algebraically equivalent `u/(1+u)` on nonnegative scaled channels.

The detector has 8 or 12 ordinary LIF neurons: leak 0.875, threshold 1 and subtractive reset. It reads 32 differences from 33 raw frames, resets neuronal state for each window, and uses a linear readout of pooled spikes. There is no new LIF neuron equation.

The training and selection changes are:

- Equal total example weight for each observed object/class cell, so a large object's examples do not dominate the training loss.
- A narrow common-gain augmentation, sampled log-uniformly from 0.8 to 1.25 before compression.
- Either uniform pooling or a recent-weighted readout: half of the weight is uniform across all 32 steps and half is uniform across the final eight. The final eight therefore receive 62.5% of the weight. Validation selected this recent pooling for five of six useful fold models.
- Validation-selected width, learning rate, early-stopping checkpoint and decision threshold.

The useful model does **not** use turnover, cancellation shrinkage or the additional residual input readout. Those experimental additions remain available for comparison. The code retains the failed experiment rather than silently relabeling the ordinary-LIF control as a successful new CT method.

## 3. Object-specific limits

| Held-out object | V2 recall | New recall | V2 false alarms | New false alarms | New balanced accuracy |
|---|---:|---:|---:|---:|---:|
| brush | 75.07% | 78.37% | 17.38% | 14.02% | 82.17% |
| screwDriver | 68.53% | 67.65% | 42.44% | 31.66% | 67.99% |
| SpoolSolder | 32.99% | 34.81% | 8.89% | 7.81% | 63.50% |

False-alarm rates and balanced accuracy improve for all three objects in this comparison, but screwdriver recall drops slightly. Overall recall remains only 60.27%, and spool recall is 34.81%. The averages therefore do not imply reliable detection on every object or that false alarms have been solved.

These are classifications of overlapping **windows** at the original endpoint grid, not numbers of alarm events per minute. The evaluation uses final source labels 4/5 only; other task phases are outside the reported false-alarm metric. Debouncing, alarm duration and physical slip-onset delay have not been evaluated.

## 4. Size and speed

| Model | Mean parameters | Float32 weight bytes | Input-layer MACs/window | Mean of fold median host times |
|---|---:|---:|---:|---:|
| Original CT-LIF | 225.00 | 900.00 | 6144 | 6.395 microseconds |
| CT-LIF v2 balanced | 82.00 | 328.00 | 1824 | 2.355 microseconds |
| Optimized mean LIF balanced | 86.33 | 345.33 | 2048 | 3.136 microseconds |
| CT residual experiment, joint | 122.83 | 491.33 | 2944 | 3.876 microseconds |
| Earlier 28-neuron mean LIF | 225.00 | 900.00 | 5376 | 2.702 microseconds |

Each useful model has **65 or 97 trainable parameters**, requiring **260 or 388 bytes** for float32 weights; scales and the decision threshold are additional. The table averages six different fold models, so a fractional parameter count is not the size of one deployed model.

Compared with V2 balanced, the useful model has about **5.3% more parameters on average**, **12.3% more input-layer MACs**, and **33.2% longer host inference time** in this paired run. Compared with the original CT-LIF, it has about 61.6% fewer parameters and is about 2.04 times faster on this host. The earlier mean-LIF28 is faster than the useful variant in this optimized-kernel run despite having more weights, showing why parameter count alone does not establish speed.

Calculated float32 raw-history payload is 7,128 bytes for the useful variant versus 6,552 bytes on average for V2 balanced. Its calculated state/feature payload is 109.33 versus 100 bytes on average. These are logical component sizes, **not measured peak RAM**. Stack temporaries, acquisition buffers, code, metadata and firmware are excluded. The generic timing harness supplies 34 frames to every model even when fewer are needed.

The C++17 float32 benchmark ran on AMD EPYC 9V74 with GCC 13.3.0 and `-O3 -std=c++17 -ffp-contract=off`. It measures encoding, reset-window inference and linear logit output, excluding acquisition, window assembly, I/O, sigmoid, scheduling and radio. Each of 45 distinct checkpoints received 5,120 measured calls over five shuffled model-order rounds. Reported time is the mean of six fold medians, not a pooled latency percentile.

A contiguous hidden-neuron accumulation loop was applied to both new and legacy C++ kernels, preserving the summation order within each neuron. Before-change timing records remain in `ct_lif_v3/results/kernel_before_optimization/`. This kernel change did not alter model selection or learned weights. **No selected decisions changed** in the float32 C++ comparison against NumPy. For the useful family, the largest absolute logit difference was below 0.00000125 on the tested recordings.

Energy, ESP32 latency/cycles, measured peak RAM and robustness on a physical deployment are unmeasured. Dense input MAC counts omit preprocessing, neuron updates, pooling and output operations; they are not total inference costs. No event-driven or sparse-hardware energy savings are claimed.

## 5. Protocol and evidence quality

The source is the [ARQ-CRISP dataset](https://github.com/ARQ-CRISP/slip_detection_dataset_2021) supplied with the project, pinned to commit `a536ee3301af0566218782b00de334b6a6630979`. The prepared-data SHA256 is `bdc90f416e6a523e3af29816c9389b0a28ac0301aaf1e85d5c92e6bd3ec55fbd`.

The exact previous six trial partitions were retained: leave one of three objects out, with seeds 7 and 19. There are **6,009 unique eligible endpoints in 87 trials**, evaluated with both seeds for **12,018 predictions per model/mode**. The two seeds are not new independent observations. Endpoints remain at trial indices 63,79,95,... with final label 4/5; no window crosses a trial boundary.

The bounded experiment ran **192 neural fits**: four families, two widths, two learning rates, two pooling choices and six folds. Families were amplitude/residual LIF, sign-invariant/residual LIF, mean-input ordinary LIF, and amplitude/spike-only LIF. The composite residual CT selector could choose between 16 candidates per fold; each separate neural control had eight. Another **96 linear-control fits** used two encoders, two pooling choices, four ridge values and six folds. Linear controls had no gain augmentation. This is not equal total tuning budget across every composite/control.

Neural training used Adam, batch64, rates 0.005/0.01, maximum32 epochs, validation-weighted-loss patience6 and gradient norm limit5. The original detached-reset surrogate-gradient rule was retained. The complete training/selection phase took about **203 seconds** on the host.

Two policies were frozen before test inference. `balanced` maximizes validation macro balanced class recall. The experimental primary `joint` policy additionally requires macro slip recall at least V2 balanced and macro false-alarm rate no higher than V2 balanced; if no candidate meets both conditions, it falls back to maximum balanced accuracy and records that failure. These are macro constraints, not per-object guarantees. The residual selector could satisfy them in only three of six folds and then performed worse on the held-out test averages.

Two validation folds lack brush non-slip examples. Such false-alarm rates remain undefined and are excluded from the corresponding class macro; they are not set to zero. Original split definitions, endpoint identities and labels were checked. Source/data hashes and hashes of all **288 checkpoints** were frozen before the test phase. No architecture, weight or threshold was retuned after the new test results.

Nine targeted tests passed, including exact readout-gradient checks, soft-compression behavior, equal object/class weighting, recent pooling and explicit constraint failure. The delivered inference wrapper reproduced **144,216 decisions** across every newly evaluated family/mode, including 24,036 residual CT decisions and both policies for the useful mean LIF. The audit also checked confusion counts, original trial splits, 6,009 unique endpoints and frozen source/checkpoint hashes.

The table below is an explicitly **post-hoc comparison** of the useful control against V2 balanced, using 2,000 paired whole-trial bootstrap draws within each object. Seeds and models stay paired within each draw.

| Change versus V2 balanced | Observed difference | Exploratory 95% interval |
|---|---:|---:|
| Balanced accuracy | +3.24 points | +2.01 to +4.51 points |
| Slip recall | +1.41 points | -0.44 to +3.45 points |
| False-alarm rate | -5.07 points | -7.18 to -2.80 points |

The recall interval includes zero: the observed 1.41-point gain is not conclusive. These intervals condition on the fitted models and three recorded objects. They omit uncertainty from new objects, retraining, family selection and multiple comparisons. Earlier results from the same dataset informed this revision, so neither freezing the latest test phase nor these conditional intervals establish independent confirmation.

## 6. Delivered use and interpretation

Use `ct_lif_v3.inference.load_optimized_detector(...)` for the practical mean-LIF option. Use `load_detector(..., model='ct_v3')` only when reproducing the residual CT experiment. Model names in all records remain explicit. The root `CT_LIF_START_HERE.md` gives setup, inference and reproduction commands.

The project preserves the original source/data and both earlier CT studies. Their reports are `ct_lif_study/CT_LIF_V1_RESULTS.md` and `ct_lif_v2/CT_LIF_V2_RESULTS.md`. New code, 288 checkpoints, validation histories, predictions, comparisons, uncertainty estimates and C++ exports are under `ct_lif_v3/`.

Supported claim: this configured mean-input LIF improved the three average detection measures over V2 balanced on the reused benchmark, with a modest increase in compute and memory. It is a practical candidate for further testing, not an established novel method, universal winner or deployment-ready detector. Fresh objects/sessions and target-device measurements are needed before making broader or journal-level superiority claims.
