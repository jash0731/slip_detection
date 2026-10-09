# CT-LIF with optimized mean inputs

The combined variant `ct_mean` retains CT-LIF's 12 common-motion, cancellation and turnover channels and appends six signed mean-motion channels. All 18 nonnegative channels are smoothly compressed with `u/(1+u)` before entering the same LIF layer. The mean branch retains amplitude that CT's instantaneous common-motion normalization can discard.

This is a feature-fusion CT-LIF variant. Its LIF neuron equation remains the ordinary leak 0.875, threshold 1, subtractive-reset equation. Publication novelty has not been established.

Two controls are included: `ct_only` uses only the CT channels, and `mean_lif` uses only the optimized mean channels. All three receive the same eight-candidate validation grid and training procedure. The primary variant is `ct_mean`; it is not replaced by a better control after looking at test results.

## Use

Run Python from the extracted project root:

```python
import numpy as np
from ct_lif_v4.inference import load_detector

detector = load_detector(held=1, seed=7, variant='ct_mean')
raw = np.load('my_windows.npy', allow_pickle=False)
result = detector.predict(raw)
print(result['probability'])
print(result['decision'])
```

Inputs have shape `(N, samples, 54)` or `(N, samples, 18, 3)`, with at least 34 consecutive raw frames. Use the original dataset's pin ordering and units without pre-normalization. One input window still needs its batch dimension. The model consumes only its final 34 frames and resets neuronal state per window.

Held-object indices 0/1/2 identify brush/screwdriver/solder spool excluded from training. Seeds are 7 or 19. These are research folds, not a single model calibrated for a new deployment. Output scores are uncalibrated sigmoid scores; decision 1 means slip.

## Reproduce

Use Python 3.12 and install the existing `ct_lif_v3/requirements.txt`. The NumPy study uses the included prepared raw archive and does not need PyTorch or an HDF5 reader.

```bash
python -m unittest -v ct_lif_v4.test_core
python -m ct_lif_v4.verify
python -m ct_lif_v4.analyze
```

For a fresh training run, preserve delivered results once on a fresh extraction:

```bash
python -c "from pathlib import Path; Path('ct_lif_v4/results').rename('ct_lif_v4/results_delivered')"
python -m ct_lif_v4.run train
python -m ct_lif_v4.run evaluate
python -m ct_lif_v4.analyze
python -m ct_lif_v4.verify
```

Set OPENBLAS_NUM_THREADS and OMP_NUM_THREADS to 1 before Python starts for efficient small-matrix training. Training refuses to overwrite an existing results folder; evaluation refuses to overwrite a completed summary. Keep the earlier studies because the analysis uses their saved predictions for comparison.

## Protocol

The three families each test widths 8/12, rates 0.005/0.01 and uniform/recent pooling over 32 steps. The recent readout allocates 62.5% of pooling weight to the final 8 steps. Training uses object/class-cell-balanced example weights, Adam, batch 64, maximum 32 epochs, patience 6 and gradient norm limit 5. Training augments the uncompressed feature vector with a common gain in [0.8,1.25]. This is feature-space augmentation: normalized CT coherence/turnover features do not scale with raw pin amplitudes in the same way.

Training scale uses unique raw rows from training trials only. Architecture, early-stopping checkpoint and threshold are selected using validation macro balanced accuracy. Undefined validation-class rates remain undefined. All 144 checkpoint hashes and every selection are frozen before separate test inference. The original six trial partitions and endpoint grid are retained.

The study checks feature-branch equivalence, retained cancellation/turnover information, mean amplitude and history/batch behavior. The public inference audit checks predictions, labels, splits, frozen weights and confusion counts. Model size and dense input MAC counts are calculated; target-device timing, peak RAM, energy, alarm-event counts and detection delay are not measured for this combined variant.

Results are exploratory on the repeatedly used three-object benchmark. They do not establish an unpublished algorithm or general performance superiority. `SHA256SUMS` covers delivered v4 files except itself and generated Python caches.
