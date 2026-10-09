# Third study: residual CT experiment and optimized mean LIF

The study includes two distinct outcomes. The predeclared residual CT candidate (`ct_v3`, primary policy `joint`) did not improve performance. A simpler mean-input LIF control (`mean_lif`, policy `balanced`) improved all three average detection measures over V2 balanced and is provided as the practical candidate.

The family recommendation was made after viewing test results. Its individual fold models and thresholds were already selected using validation data and frozen before test inference. This is exploratory evidence on a reused benchmark, not independent confirmation or a claim of unpublished novelty.

## Practical detector

Use `inference.load_optimized_detector(held, seed)` for the mean-input LIF candidate. It is equivalent to `load_detector(held, seed, policy='balanced', model='mean_lif')`.

For the residual CT experiment, call `load_detector(held, seed, policy='joint', model='ct_v3')`. Other recorded families are `ct_amplitude`, `ct_invariant`, `ct_rate` and `linear_control`. Each has both `balanced` and `joint` policies.

The `held` index identifies the object withheld from training (0 brush, 1 screwdriver, 2 solder spool), not an object the model is specifically trained to recognize. Seed is 7 or 19. Do not choose a fold or seed using new test labels.

The predictor accepts finite raw arrays with shape `(N, samples, 54)` or `(N, samples, 18, 3)`, where samples is at least 33. Input order is time, pin, axis, with 18 pins and 3 axes. Only the final 33 frames are used, neuronal state resets for every window, and no window should cross a trial boundary. Keep raw units and do not pre-normalize.

`Detector(checkpoint_path, recorded_threshold)` loads any new v3 checkpoint explicitly. Use its matching saved threshold and scale. This wrapper is not an interface for old v1/v2 checkpoint schemas. Its probability field is an uncalibrated sigmoid score; decision 1 denotes slip and 0 denotes non-slip.

## Features and network

`raw_features` forms per-frame pin differences, common axis motion `m = mean(d)` and opposing magnitude `R = max(mean(abs(d)) - abs(m), 0)`. Training-only axis scale is the p95 of mean absolute pin motion over unique training raw differences. Each nonnegative scaled channel `u` is smoothly compressed as `u/(1+u)`.

| Variant | Channels | Content |
|---|---:|---|
| `mean` | 6 | Positive and negative common motion; useful ordinary-LIF candidate |
| `amplitude` | 9 | Mean channels plus opposing magnitude |
| `invariant` | 6 | Absolute common motion plus opposing magnitude |

The useful variant is equivalently the positive/negative split of `m/(s+abs(m))`. C++ uses this algebraic simplification, avoiding unnecessary opposing-motion computation for that variant.

LIF neurons have leak 0.875, threshold 1 and subtractive reset. Hidden width is 8 or 12. Uniform pooling weights all 32 spike steps equally. Recent pooling allocates half the weight uniformly and half to the final 8 steps: those final 8 receive 62.5% of the total weight. There is no ongoing cross-window neuronal state.

The ordinary mean model uses only the pooled spikes and a linear output. Residual variants add a learned linear output from identically pooled input features. That branch preserves subthreshold input evidence but failed to improve the held-object averages in this study. The linear control uses only pooled input features without hidden neurons.

## Training and operating points

The four neural families share object/class-cell-balanced sample weights, Adam, up to 32 epochs, patience 6, norm limit 5, batch 64 and a per-window common positive gain drawn log-uniformly from [0.8,1.25] before compression. Each family tests widths 8/12, rates 0.005/0.01 and two pooling choices. Linear controls use ridge logistic fitting with no gain augmentation.

The `balanced` policy maximizes validation macro balanced class recall. The `joint` policy seeks validation macro recall at least V2 balanced and false alarms no higher than V2 balanced; it falls back to best balanced accuracy if infeasible. Every such failure is recorded. These are macro constraints, not per-object guarantees. Missing-class rates remain undefined rather than being treated as zero.

The composite residual family searches 16 candidates per fold; each separate neural control searches 8. The linear control searches 16. All288 checkpoint hashes and every selection were written before new held-object inference. The default experimental policy and its negative result are preserved in the protocol.

## C++ and costs

`runtime.hpp` implements the new models and `legacy_runtime.hpp` the older comparison models with the same dense-layer loop optimization. `models_generated.hpp` exports 45 distinct selected checkpoints. Each `ModelSpec` contains its checkpoint name, held-object index and predictor function.

Predictors take a pointer to 34 raw frames of 54 float32 values. They read only their required history and return a logit. Apply sigmoid and the exact threshold from the saved selection or resource CSV. The benchmark excludes sigmoid, threshold comparison, acquisition, window assembly and radio.

This is a research export, not firmware for a single deployment. The useful family has 65/97 parameters, but is slower than V2 in this host run. A small parameter count does not by itself establish fast inference, low peak RAM or low energy. Actual ESP32 and sensor measurements remain necessary.

## Reproducibility

Use the root `CT_LIF_START_HERE.md` for setup and commands. Nine targeted tests passed. `python -m ct_lif_v3.verify` checks 144,216 delivered decisions through the public inference interface, together with sources, weights, splits, labels and confusion counts. The C++ benchmark found zero changed decisions relative to NumPy on all selected test windows.

`results/recommendation.json` explicitly records the post-hoc practical choice. `results/optimized_models.json` lists its six validation-selected fold checkpoints. `results/analysis.json` labels the additional mean-LIF uncertainty comparisons as post-hoc. `SHA256SUMS` covers the delivered v3 files except itself and generated Python caches.
