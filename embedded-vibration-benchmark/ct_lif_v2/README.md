# CT-LIF v2 implementation

This revision changes input representation, network width, training augmentation and operating-point selection. It reuses the original LIF neuron and surrogate-gradient implementation in `ct_lif_study/reference/ct_lif.py`.

This is a compact-model experiment, not a demonstrated universally superior or unpublished algorithm. The delivered composite selector has 67.98% balanced accuracy. Its strict mode has 9.91% mean false alarms with 38.10% slip recall. See the root report for ordinary-LIF controls and per-object failures.

## Encoders

`core.encode(raw, scale, variant, window)` accepts a finite array of shape `(N, samples, 18, 3)` in the same pin/axis ordering and raw units as the dataset. It uses the last `window` causal differences. Supply at least `window + 1` raw frames. Do not normalize raw frames first; the saved training scale is applied inside the encoder.

| Variant | Channels | Behavior |
|---|---:|---|
| `mean` | 6 | Positive/negative common pin motion divided by training scale; ordinary-LIF control |
| `amplitude` | 9 | Mean channels plus opposed-motion magnitude divided by training scale |
| `shrink` | 6 | Subtract half the opposed-motion magnitude from absolute common motion, floor at zero, restore sign, then split and scale |

Opposed motion is `max(mean(abs(pin_difference)) - abs(mean(pin_difference)), 0)`. Channels are clipped to [0,1]. No turnover, entropy, online threshold adaptation, alarm debounce or alarm refractory period is introduced. Those behaviors have not been measured in this experiment.

Selected models have 8 or 10 hidden neurons and 16 or 32 encoded steps. State resets per window. The endpoint grid remains unchanged; no window crosses a trial boundary.

## Use a saved research model

```python
import numpy as np
from ct_lif_v2.inference import load_detector

detector = load_detector(held=1, seed=7, policy='balanced')
raw = np.load('my_windows.npy', allow_pickle=False)
result = detector.predict(raw)
print(result['probability'])
print(result['decision'])
```

Input shape is `(N, samples, 54)` or `(N, samples, 18, 3)`. The detector uses its last `detector.window + 1` raw frames and resets neuronal state. Retain the batch dimension for one window. Scores are sigmoid outputs, not established calibrated real-world probabilities. Decisions are 0 for non-slip and 1 for slip using the saved validation threshold.

`held=0/1/2` identifies the object excluded from training: brush, screwdriver or solder spool. It selects a **research fold**, not a known object that the model was trained to recognize. Seed is 7 or 19. Policies are `balanced`, `fpr20` and `fpr10`. Each uses the checkpoint and threshold frozen in `results/all_choices_before_test.json`. Do not select a seed or fold using new test labels.

For an explicit v2 checkpoint and its recorded threshold:

```python
from ct_lif_v2.inference import Detector

detector = Detector(checkpoint_path, recorded_threshold)
result = detector.predict(raw)
```

The explicit interface also supports new mean-LIF and temporal-ANN checkpoints, using their matching validation thresholds. Old v1 checkpoints have a different encoder schema and are not supported by this wrapper.

## Selection and false-alarm targets

Training selects an early-stopping checkpoint using validation loss. Balanced mode maximizes average validation class recall. A false-alarm mode requires its cap on every validation object with non-slip examples, then favors higher minimum object slip recall. Candidate architecture selection follows the same objective, breaking later ties by input-layer MAC count, parameter count and learning rate.

The composite `ct_v2` chooses either `amplitude_lif` or `shrink_lif`; individual families and ordinary-LIF controls remain visible. The composite has twice the candidate count of each individual control family. Modes may select different checkpoints. No final choice is made using test scores.

Missing validation classes are explicitly recorded. Low false-alarm rates can accompany near-zero slip recall; `useful_validation_recall` records whether minimum object recall reaches 40%. Strict v2 fails this check in three of six folds. Targets apply only to observed validation windows, not arbitrary unseen objects.

## C++ reference

`runtime.hpp` implements float32 encoders and LIF/temporal readouts. `models_generated.hpp` contains 51 distinct selected checkpoints. Each `ModelSpec` gives a checkpoint name, held-object index and predictor; this is a research export, not a production firmware image.

Each C++ predictor takes a pointer to **34 consecutive raw frames × 54 float32 values**, ordered as frame, pin, axis. It reads the model's required history and returns a logit. Apply sigmoid and the matching threshold from `all_choices_before_test.json` or `resources_by_fold.csv`. Host timing excludes sigmoid and the decision operation.

Do not flash every fold as one deployment model. A deployment model needs validation for the actual sensor environment and a separate export of its weights, scale and threshold. Target latency, peak RAM and energy must be measured there; none is established by the host benchmark.

## Reproduce and audit

Run from the project root. `python -m ct_lif_v2.verify` checks all 36,054 revised decisions. `python -m unittest -v ct_lif_v2.test_core` checks mathematical behavior and operating-point selection. Training, evaluation, analysis and host timing commands are in `CT_LIF_START_HERE.md`.

`SHA256SUMS` covers delivered v2 files except itself and generated caches. Training-source/data hashes in `results/protocol.json` were frozen before the new test phase. The old study and reports remain separate.
