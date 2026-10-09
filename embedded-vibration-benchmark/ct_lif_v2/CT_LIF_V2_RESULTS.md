# CT-LIF v2: measured revision and false-alarm trade-offs

8 October 2026. Completed experiment on the supplied project and tactile recordings.

**The revision is smaller and faster than the previous CT-LIF. It reduces false alarms by sacrificing slip detection. It does not outperform ordinary LIF in every aspect.** The strict mode detects only 38.10% of slip windows, so its 9.91% mean false-alarm rate is not an overall detection improvement.

The previous version is called “v1” below for comparison; its original report identified the candidate as v0.1. Its code, weights and predictions are preserved in `ct_lif_study/`, with the prior report in `ct_lif_study/CT_LIF_V1_RESULTS.md`.

## 1. What changed

The original CT encoder divided the common pin motion by current total activity and added a turnover feature. The v2 experiment removes turnover, preserves signal amplitude relative to a training-only scale, and tests a compact suppression of opposing pin motion. The LIF neuron update itself remains the existing implementation.

For pin difference `d[p,a,t]` on axis `a`, compute:

- Common motion: `m[a,t] = mean_p(d[p,a,t])`.
- Total activity: `A[a,t] = mean_p(abs(d[p,a,t]))`.
- Opposing motion: `R[a,t] = max(A[a,t] - abs(m[a,t]), 0)`.
- Scale `s[a]`: the 95th percentile of `A` over unique raw differences in training trials, computed separately for each axis.

Two encoders are available. The **amplitude encoder** uses nine channels: positive and negative parts of `m/s`, plus `R/s`. The **shrink encoder** first sets `m_shrink = sign(m) * max(abs(m) - 0.5 R, 0)`, then uses its positive and negative parts divided by `s`, giving six channels. Every channel is clipped to [0,1]. Shrinkage can remove useful slip motion as well as noise; the experiment measures that cost.

Both use 8 or 10 LIF neurons and 16 or 32 encoded steps. The neuron uses leak 0.875, threshold 1, subtractive reset, and a linear readout of mean spikes. State resets for each window. Common positive gain augmentation, sampled log-uniformly from [0.5,2], is applied during training before clipping. The same augmentation and tuning grid are given to the new ordinary-LIF and temporal-ANN controls.

“CT-LIF v2” denotes the rule that selects between the two encoders using validation data for each fold and operating mode. It does not ensemble their predictions. The three modes can select different weights or window lengths, in addition to different decision thresholds.

## 2. Main results

Values below are percentages averaged equally over the three held-out objects and two seeds. Slip recall is `TP/(TP+FN)`; false-alarm rate is `FP/(FP+TN)`; balanced accuracy is `(recall + 1 - false-alarm rate)/2`. They are **window classification** measures for source labels 4/5, not false alarm events per minute. Other labels, including non-holding phases, are not evaluated.

| Method and operating mode | Balanced accuracy | Slips detected | False-alarm rate |
|---|---:|---:|---:|
| Previous CT-LIF, original | 68.83% | 66.23% | 28.57% |
| CT-LIF v2, balanced | 67.98% | 58.86% | 22.90% |
| CT-LIF v2, 20% validation target | 67.27% | 51.42% | 16.88% |
| CT-LIF v2, 10% validation target | 64.09% | 38.10% | 9.91% |

The strict mode lowers the mean false-alarm rate by **18.66 percentage points** relative to the original CT-LIF, but lowers slip recall by **28.14 percentage points**. The corresponding pooled ordinary accuracies are 69.40%, 71.04%, 71.39% and 72.19%. Those pooled scores weight objects by endpoint count and favor the more common non-slip class; they do not replace balanced accuracy.

The names “10%” and “20%” refer to **validation targets**, not guaranteed test behavior.

## 3. Comparisons that prevent misleading improvement claims

| Method and operating mode | Balanced accuracy | Slips detected | False-alarm rate |
|---|---:|---:|---:|
| Matched ordinary mean LIF, balanced | 69.37% | 60.25% | 21.50% |
| Matched ordinary mean LIF, 20% target | 68.78% | 54.79% | 17.24% |
| Matched ordinary mean LIF, 10% target | 64.78% | 40.25% | 10.69% |
| Previous CT-LIF, recalibrated balanced | 69.03% | 58.19% | 20.12% |
| Previous CT-LIF, recalibrated 20% target | 68.77% | 54.59% | 17.05% |
| Previous CT-LIF, recalibrated 10% target | 66.23% | 43.11% | 10.65% |
| Earlier mean LIF, 28 neurons, original | 71.31% | 70.54% | 27.92% |
| Earlier mean LIF, 28 neurons, 10% target | 63.93% | 37.72% | 9.86% |
| Matched temporal ReLU ANN, balanced | 65.71% | 50.27% | 18.85% |
| Matched temporal ReLU ANN, 10% target | 60.36% | 33.70% | 12.98% |

Recalibrating the previous CT-LIF without changing its weights already gives **10.65% false alarms with 43.11% slip detection**. V2's strict mode gives 9.91% and 38.10%. Much of the reduction relative to the original model is an operating-point trade-off, not evidence that the new features distinguish slip better.

The matched ordinary LIF has higher balanced accuracy and recall at every corresponding mode. Its balanced mode is also smaller, faster and has fewer false alarms than the v2 selector. The previously tested 28-neuron mean LIF retains the highest balanced accuracy among these displayed original operating points. The temporal ANN is faster on this host. Universal superiority is therefore contradicted by the tested controls.

The individual v2 families also remain visible:

| Method and operating mode | Balanced accuracy | Slips detected | False-alarm rate |
|---|---:|---:|---:|
| Amplitude-preserving LIF family, balanced | 65.68% | 53.99% | 22.62% |
| Cancellation-shrink LIF family, balanced | 69.49% | 59.51% | 20.53% |
| Validation choice of either family, balanced | 67.98% | 58.86% | 22.90% |

The shrink family slightly exceeds the matched mean LIF's balanced accuracy in this exploratory comparison, but has lower recall. The frozen v2 selector sometimes picks the amplitude family and scores lower overall. Selecting the shrink family after seeing these test scores would be a new, test-informed choice requiring fresh confirmation.

Every model/mode, including cancellation-only controls, is reported in `ct_lif_v2/results/aggregate.csv`; per-fold scores and predictions are also included.

## 4. Performance varies substantially by object

Each cell gives **slips detected / false-alarm rate**, averaged over seeds.

| Held-out object | Balanced: recall / FPR | 20% target: recall / FPR | 10% target: recall / FPR |
|---|---:|---:|---:|
| brush | 75.07% / 17.38% | 54.15% / 4.11% | 40.69% / 2.06% |
| screwDriver | 68.53% / 42.44% | 63.49% / 36.21% | 53.91% / 25.57% |
| SpoolSolder | 32.99% / 8.89% | 36.63% / 10.30% | 19.70% / 2.10% |

The strict mode still falsely labels **25.57% of screwdriver non-slip windows**. Its 9.91% overall mean must not be described as a 10% limit on every object. For the solder spool it detects only 19.70% of slip windows.

## 5. Compute, size and host timing

Numbers are averages across the six selected fold models, not the size of one fractional model. Each selected v2 model has **65–111 trainable parameters** (260–444 weight bytes in float32), versus 225 parameters (900 bytes) for the previous CT-LIF. Scale and threshold storage are additional.

| Model | Parameters | Float32 weight bytes | Input-layer MACs/window | Mean of fold median host times |
|---|---:|---:|---:|---:|
| Previous CT-LIF | 225.00 | 900.00 | 6144 | 6.451 microseconds |
| Earlier mean LIF, 28 neurons | 225.00 | 900.00 | 5376 | 3.849 microseconds |
| CT-LIF v2, balanced | 82.00 | 328.00 | 1824 | 2.507 microseconds |
| CT-LIF v2, both alarm modes | 79.33 | 317.33 | 1952 | 2.620 microseconds |
| Matched mean LIF, balanced | 73.00 | 292.00 | 1568 | 2.166 microseconds |
| Matched temporal ANN, balanced | 100.00 | 400.00 | 2400 | 1.572 microseconds |

The balanced v2 mode uses 63.6% fewer trainable parameters and 70.3% fewer input-layer multiply-accumulates than prior CT-LIF. Its measured mean of fold median inference times is about **2.57 times faster**. Both alarm modes select the same six checkpoints in this run, with different thresholds, and are about 2.46 times faster than prior CT-LIF. These are host results, not guarantees about another processor.

The C++17 float32 benchmark includes encoding, reset-window inference and linear readout. It excludes sensor acquisition, window assembly, I/O, sigmoid, scheduling and radio. It used an AMD EPYC 9V74 host, GCC 13.3.0 with `-O3 -std=c++17 -ffp-contract=off`, five shuffled model-order rounds, and 5,120 timed calls per distinct selected checkpoint. The table averages each fold's median; its “p95” counterpart in the CSV averages fold p95 values and is not an overall service-latency percentile. Timing noise and host scheduling remain relevant.

All 51 distinct exported checkpoints were checked against NumPy on their held-object windows. **No selected decision changed** under float32 C++ inference. Maximum absolute v2 logit difference was below 0.000001. This is evidence for these recordings, not bitwise equivalence for arbitrary input.

Calculated float32 raw-history payload is 6,552 bytes on average for balanced v2, 7,128 for the alarm modes and 7,344 for prior CT-LIF. Calculated feature/state payload is 100, 97.33 and 404 bytes, respectively. These are logical component sizes, **not measured peak RAM**; stack temporaries, metadata, code, firmware and acquisition buffers are excluded. The generic benchmark still passes 34 raw frames to every model. A streaming encoder could change buffering requirements and needs separate implementation and measurement.

Input-layer MAC counts exclude preprocessing, neuron updates and readout. Inference remains dense; sparse/event-driven savings are not claimed. Energy consumption, ESP32 cycles/latency, peak RAM, alarm-event counts and slip-onset delay have not been measured. Model history lengths do not by themselves measure detection delay. No quantization or hardware energy result is fabricated.

## 6. Experimental protocol and checks

- Dataset: [ARQ-CRISP slip detection dataset](https://github.com/ARQ-CRISP/slip_detection_dataset_2021), using the supplied project copy at source commit `a536ee3301af0566218782b00de334b6a6630979`.
- Prepared data: 228,521 raw samples, 90 trials, 18 pins and three axes. SHA256 of `data/arq_all_trials.npz`: `bdc90f416e6a523e3af29816c9389b0a28ac0301aaf1e85d5c92e6bd3ec55fbd`.
- Exact six existing trial partitions: three leave-one-object-out folds, seeds 7 and 19. All data scaling uses training trials only.
- Test set: 6,009 unique eligible endpoints in 87 trials; each is evaluated with two seeds, giving 12,018 predictions per model/mode. The two seeds are not independent new observations. Endpoints remain at trial indices 63,79,95,... with final label 4 or 5, even when a candidate uses a shorter history.
- Four new families × two histories × two widths × two learning rates × six folds = **192 fits**. Each separate family has eight candidates per fold. The composite v2 selector can choose among 16 candidates; its larger search budget is disclosed. This is not equal total tuning budget between the composite and a single control family.
- Training: weighted binary cross entropy, Adam, batch 64, at most 18 epochs, validation-loss early stopping with patience 4, gradient norm limit 5, rates 0.005/0.01. The original NumPy surrogate-gradient implementation is reused. The recorded training phase took about 99 seconds on this host, including candidate validation/selection work; this is not comparable to a one-model training time.
- Thresholds: a grid from 0 to 1 with step 0.005, plus an all-negative threshold. Balanced mode maximizes macro validation class recall. Alarm modes cap false-alarm rate separately for each validation object with non-slip examples, then maximize the minimum object slip recall, with documented tie-breaking. A minimum validation slip recall of 40% is recorded as a usefulness check, not enforced or guaranteed. Strict v2 fails that check in three of six folds; the 20% mode fails in two.
- Two validation folds contain brush slip examples but no brush non-slip examples. Their brush false-alarm constraint cannot be estimated. Missing-class rates remain undefined and are recorded explicitly; available class recalls contribute to the validation score. Even other folds have small non-slip validation counts. No distribution-free false-alarm guarantee is implied.
- An initial training attempt stopped at this missing-class condition before any v2 test inference. The handling was corrected, incomplete checkpoints discarded, and all 192 fits restarted. Its prior protocol/log are retained under `pretest_validation_fix/`.
- All new checkpoint and threshold choices were saved before the separate v2 test phase. Training source and data hashes are checked at evaluation. Existing CT-LIF and mean-LIF weights were separately recalibrated with the same validation rules.
- Eight mathematical/data-handling v2 tests and the 13 existing reference tests passed. The inference wrapper reproduced all **36,054 v2 model/mode decisions**, and the audit rechecked endpoint identities, original split definitions, labels, confusion counts and frozen hashes. Original uploaded source/data files remain unchanged; generated cache files are not preserved.

Paired whole-trial bootstrap intervals use 2,000 draws, resampling trials within object and keeping models/seeds paired. For strict v2 minus original CT-LIF, the exploratory 95% interval is **−23.31 to −14.88 percentage points** for false alarms and **−32.98 to −22.96 points** for slip recall. The balanced-accuracy difference is −4.74 points, interval −7.54 to −1.48. The intervals condition on these three objects and fitted models; they omit new-object uncertainty, retraining, selection and multiple-comparison adjustment.

This dataset and its earlier held-object scores already informed the new design. Freezing choices before the latest test phase improves traceability but does not turn the reused benchmark into independent confirmation.

## 7. What can be claimed

Supported: a compact CT-LIF revision, lower measured host inference cost versus the previous CT-LIF, and explicit validation-selected false-alarm/recall operating points on these recordings.

Not established: better overall discrimination than ordinary LIF, superiority to all other algorithms, lower energy, deployment readiness, performance on new sensors/objects, or unpublished novelty. This work changes the encoder, model size and training/selection procedure; it does not introduce a new LIF neuron equation. A journal claim of a novel superior detector is not supported by these results.

A useful next experiment would add representative stable-contact negatives and new objects/sessions, define an acceptable minimum recall before training, and validate once on a fresh test set. Actual target-device measurements are needed for latency, RAM and energy. The strict mode should not be selected solely because its mean false-alarm percentage looks better.

Reproduction and use instructions are in `CT_LIF_START_HERE.md` and `ct_lif_v2/README.md`. The delivered weights are fold-specific research models, not a single model trained and calibrated for a new deployment.
