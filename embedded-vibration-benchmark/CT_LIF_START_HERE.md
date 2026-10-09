# Start here: CT-LIF with optimized mean inputs

The combined model is `ct_mean`. Read `CT_LIF_RESULTS.md` for measured trade-offs: 69.19% balanced accuracy, 51.49% recall and 13.12% false-positive rate. The higher-accuracy `ct_only` control is a separate variant.

Use Python 3.12 from the extracted project root:

```bash
python -m pip install -r ct_lif_v3/requirements.txt
python -m unittest -v ct_lif_v4.test_core
python -m ct_lif_v4.verify
```

```python
import numpy as np
from ct_lif_v4.inference import load_detector

detector = load_detector(held=1, seed=7, variant='ct_mean')
raw = np.load('my_windows.npy', allow_pickle=False)
result = detector.predict(raw)
print(result['probability'])
print(result['decision'])
```

Input shape is `(N, samples, 54)` or `(N, samples, 18, 3)`, with at least 34 consecutive frames in the original units and pin ordering, without pre-normalization. Include a batch dimension for a single window. The model consumes the final 34 frames and resets state per window. Decision 1 means slip; scores are not calibrated probabilities.

Held indices 0/1/2 are brush/screwdriver/solder spool excluded from training. Seed must be 7 or 19. These saved research folds are not a universal deployment checkpoint. Select `variant='ct_only'` or `variant='mean_lif'` explicitly to use a control.

To reproduce summaries, run `python -m ct_lif_v4.analyze`. For fresh training on a fresh extraction, preserve delivered results first:

```bash
python -c "from pathlib import Path; Path('ct_lif_v4/results').rename('ct_lif_v4/results_delivered')"
python -m ct_lif_v4.run train
python -m ct_lif_v4.run evaluate
python -m ct_lif_v4.analyze
python -m ct_lif_v4.verify
```

Before starting Python, set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` (Linux/macOS: `export NAME=1`; Windows PowerShell: `$env:NAME='1'`). Training refuses to overwrite existing results. Keep prior studies: analysis uses their saved predictions. Details are in `ct_lif_v4/README.md`; the combined model has no new embedded timing or energy benchmark.
