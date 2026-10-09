import json
from pathlib import Path
import numpy as np
from .core import ResidualLIF, raw_features, compress, linear_forward
from ct_lif_study.reference.ct_lif import sigmoid


class Detector:
    def __init__(self, checkpoint, threshold):
        self.checkpoint = Path(checkpoint)
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= np.nextafter(1., 2.):
            raise ValueError('Invalid threshold')
        with np.load(self.checkpoint, allow_pickle=False) as z:
            self.scale = z['scale'].copy()
            self.variant = str(z['variant'])
            self.pooling = str(z['pooling'])
            self.kind = str(z['kind'])
            if self.kind == 'linear':
                self.params = {k: z[k].copy() for k in ['q', 'c']}
                self.model = None
            elif self.kind == 'lif':
                self.model = ResidualLIF(*z['W'].shape, residual=bool(z['residual']), pooling=self.pooling)
                self.model.params = {k: z[k].copy() for k in self.model.params}
            else:
                raise ValueError('Unknown checkpoint type')

    def predict(self, raw):
        raw = np.asarray(raw)
        if raw.ndim == 3 and raw.shape[-1] == 54:
            raw = raw.reshape(raw.shape[0], raw.shape[1], 18, 3)
        if raw.ndim != 4 or raw.shape[1] < 33 or raw.shape[2:] != (18, 3) or not len(raw):
            raise ValueError('Expected nonempty N x samples x 18 x 3 or N x samples x 54, with at least33 raw frames')
        features = compress(raw_features(raw[:, -33:], self.scale, self.variant))
        logits = linear_forward(features, self.params, self.pooling) if self.kind == 'linear' else self.model.forward(features)
        probability = sigmoid(logits)
        return {'probability': probability, 'decision': (probability >= self.threshold).astype(np.int8), 'threshold': self.threshold}


def load_detector(held, seed=7, policy='joint', model='ct_v3', results=None):
    results = Path(results) if results is not None else Path(__file__).resolve().parent / 'results'
    frozen = json.loads((results / 'all_choices_before_test.json').read_text())
    folds = [fold for fold in frozen['folds'] if fold['held'] == held and fold['seed'] == seed]
    if len(folds) != 1:
        raise ValueError('Expected one saved held-object fold and seed')
    choices = [choice for choice in folds[0]['selected'] if choice['model'] == model and choice['policy'] == policy]
    if len(choices) != 1:
        raise ValueError('Unknown model or policy')
    choice = choices[0]
    return Detector(results / (choice['candidate']['name'] + '.npz'), choice['threshold'])


def load_optimized_detector(held, seed=7, results=None):
    return load_detector(held=held, seed=seed, policy='balanced', model='mean_lif', results=results)
