# CT-LIF measured results

**8 October 2026 — actual recorded-data evaluation, CT-LIF v0.1**

**Main result: CT-LIF achieved 68.83% mean balanced accuracy.** Its ordinary accuracy over all test-window predictions was 69.40%. These are different statistics; use **68.83% balanced accuracy** when comparing with the earlier 60.78% and 62.86% results.

This is a completed host-computer experiment on the supplied public tactile recordings, not a synthetic accuracy estimate. It does not establish physical sensor transfer, ESP32 performance or publication novelty.

## 1. What was run

- Dataset: ARQ-CRISP `slip_detection_dataset_2021`, using the prepared data and original recordings in the uploaded project.
- Dataset contains 228,521 raw samples, 90 grasp trials, three objects, 18 pins and three axes per pin.
- Three leave-one-object-out folds, each run with seeds 7 and 19, using the exact six saved trial partitions.
- Final evaluation: **6,009 unique endpoints across 87 eligible trials**, evaluated for each seed; **12,018 predictions per model**. The other three assigned trials have no eligible endpoint. Two seed evaluations are not independent new observations.
- Eight methods trained, giving **48 fits**. All model checkpoints and thresholds were frozen before any new test inference.
- CT-LIF settings stayed exactly as specified in the earlier candidate: 12→16→1, gate fraction .02, Adam rate .005, at most 18 epochs, batch64, patience4, weighted BCE, detached reset and the original NumPy surrogate derivative. No hyperparameter search or change after observing results.
- Each input window contains 32 encoded steps from 34 raw frames. Endpoints are trial indices 63,79,95,… with source label4/5 at the last frame. All original labels, trial partitions and evaluated endpoints match the historical prediction files.
- Compared with the earlier 32-difference model, CT needs one additional preceding raw sample for its first turnover value. New controls receive the same 34-frame raw windows, but their effective encoder histories can differ. The earlier validation-selected model can use 16/32/64 encoded steps.

The dataset's author repository is https://github.com/ARQ-CRISP/slip_detection_dataset_2021 . This run used the uploaded copy pinned to commit `a536ee3301af0566218782b00de334b6a6630979`, not an unpinned fresh download. All three raw HDF5 file checksums were matched to the bundled manifest. The prepared file SHA256 is `bdc90f416e6a523e3af29816c9389b0a28ac0301aaf1e85d5c92e6bd3ec55fbd`, matching the original experiment metadata.

## 2. Full comparison

Scores average within object across the two seeds and then equally across the three objects. Recall refers to actual slip windows detected. False-alarm rate refers to actual non-slip windows incorrectly called slip.

| Method | Mean balanced accuracy | Mean slip recall | Mean false-alarm rate |
|---|---:|---:|---:|
| **CT-LIF, fixed v0.1** | **68.83%** | **66.23%** | **28.57%** |
| CT features + mean-ReLU ANN | 54.96% | 51.38% | 41.46% |
| CT features + temporal ReLU ANN | 68.96% | 68.63% | 30.72% |
| New mean-input LIF, 16 neurons | 69.56% | 64.03% | 24.91% |
| New mean-input LIF, 28 neurons | **71.31%** | 70.54% | 27.92% |
| Cancellation-only LIF ablation | **70.96%** | 65.21% | 23.29% |
| Separate-turnover LIF control | 69.14% | 63.63% | 25.35% |
| Entropy/change LIF control | 68.44% | 62.24% | 25.37% |
| Earlier fixed LIF, saved predictions | 60.78% | 55.03% | 33.46% |
| Earlier validation-selected LIF, saved predictions | 62.86% | 51.74% | 26.02% |

The new 28-neuron mean-input LIF and CT-LIF each have 225 parameters. The 16-neuron mean-input LIF has 129. The cancellation-only ablation retains the six common-motion channels and three cancellation channels, setting the three turnover channels to zero. Its 225 stored parameters include 48 input weights that have no effect on predictions; its effective input-to-hidden representation is smaller.

The new mean-input controls use the candidate's training-scale convention and clipping to ±1. They are **not replicas of the historical model**, which used a different normalizer, clipping, initialization, surrogate slope, batch size, weight decay, learning rate and threshold grid. The historical numbers were recomputed from supplied saved predictions on exactly aligned endpoints, not obtained by retraining the original PyTorch code here.

Therefore, the increase over 62.86% is a measured comparison of complete pipelines, **not proof that the proposed encoder or turnover mechanism caused the improvement**. The new controls are the more informative comparison for the candidate mechanism.

## 3. Results on each unseen object

| Held-out object | CT-LIF balanced accuracy | Slip recall | False-alarm rate | Unique test windows |
|---|---:|---:|---:|---:|
| Brush | **78.35%** | 74.93% | 18.22% | 884 |
| Screwdriver | **60.51%** | 72.98% | **51.97%** | 2,586 |
| Solder spool | **67.63%** | **50.79%** | **15.52%** | 2,539 |

The solder-spool result improves recall relative to the earlier selected LIF's **18.04%**, but its false-alarm rate increases from **4.12% to 15.52%**. CT-LIF still misses roughly half of the spool slip windows.

For the screwdriver, more than half of non-slip windows become false alarms. For the brush, recall decreases from the earlier selected LIF's 79.37% to 74.93%, while false alarms decrease from 28.04% to 18.22%. These operating-point trade-offs matter; an aggregate accuracy number does not make this a reliable grip-protection system.

## 4. Uncertainty and what the ablations say

We ran 2,000 paired whole-trial bootstrap draws within each object, sharing sampled trial IDs across seeds and methods. All draws were valid. The conditional 95% interval for CT-LIF's mean BA is **64.77–72.35%**.

| CT-LIF minus comparator | BA difference | Conditional paired 95% interval |
|---|---:|---:|
| Earlier selected LIF | +5.97 percentage points | +2.45 to +9.68 |
| New mean-input LIF, 28 neurons | **−2.48 points** | **−4.20 to −0.49** |
| Cancellation-only ablation | **−2.13 points** | **−3.33 to −0.98** |
| CT + temporal ANN | −0.13 points | −1.97 to +1.86 |
| Separate-turnover control | −0.31 points | −0.98 to +0.39 |

These intervals condition on the three observed objects and fixed fitted models. They do not include retraining uncertainty, the research selection process or performance variation across arbitrary new objects. Comparisons are exploratory and have no multiple-testing correction.

The observed ablations do **not support a benefit from cancellation-weighted turnover** in this fixed configuration. Removing turnover produces a higher aggregate score and lower aggregate false-alarm rate. A conventional temporal network with the same CT features performs similarly to CT-LIF. The simpler, equally sized mean-input LIF has the best observed aggregate BA among the newly tested methods.

This does not prove turnover can never help under any conditions. It means the candidate's central advantage has **not been demonstrated by this experiment**. Do not rename the 71.31% baseline as CT-LIF or report only the comparison against 62.86%.

## 5. What can be claimed

A supported statement is:

“The proposed CT-LIF candidate achieved 68.83% mean balanced accuracy on three-object leave-one-object-out recorded-data evaluation. Although it exceeded the historical selected pipeline and increased spool recall, simpler controls achieved higher aggregate performance. The results do not establish an advantage for the proposed cancellation–turnover coupling.”

The supplied reports' failures motivated the candidate, so the three objects are development-informed evaluation data at the research level. A strong generalization or publication claim needs independent objects/sessions and a more complete closest-work comparison. Novelty remains unresolved.

No CT-LIF ESP32 firmware, device timings, energy measurement, live sensor transfer, physical incipient-slip detection or closed-loop control was produced by this run. Hardware files elsewhere in the uploaded project concern earlier implementations and are not evidence for CT-LIF.

## 6. Verification and reproduction

- Original candidate encoder/trainer copied without modification; its SHA256 is retained in `ct_lif_study/results/protocol.json`.
- All six saved trial splits checked against the original splitter; no train/validation/test trial overlap.
- Every new endpoint and label matched against the original loader, for all partitions and seeds.
- 13 candidate unit tests passed, including causal encoding and gradient checks.
- All 48 checkpoints, histories and selections saved. No threshold selected at the threshold-grid boundary.
- Window predictions, confusion counts, macro summaries and paired uncertainty analysis saved.
- Full study training took approximately **46.12 seconds in this host environment**. This is training wall time, not inference latency or ESP32 timing.

Files: `ct_lif_study/results/summary.csv`, `predictions.csv`, `aggregate.csv`, `per_object.csv`, `analysis.json`, `protocol.json`, `all_choices_before_test.json`, and the 48 `.npz` checkpoints. A comparison chart is in `ct_lif_study/results/ct_lif_results.png`.

See `CT_LIF_START_HERE.md` for commands and dependencies. Historical source files and results are preserved; this study is an added module.
