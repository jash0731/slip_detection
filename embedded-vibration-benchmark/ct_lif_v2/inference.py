import json
from pathlib import Path
import numpy as np
from .core import encode
from ct_lif_study.reference.ct_lif import TinyModel, sigmoid


class Detector:
    def __init__(self, checkpoint, threshold):
        self.checkpoint = Path(checkpoint)
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= np.nextafter(1., 2.):
            raise ValueError('Invalid decision threshold')
        with np.load(self.checkpoint, allow_pickle=False) as z:
            self.scale = z['scale'].copy()
            self.variant = str(z['variant'])
            self.window = int(z['window'])
            self.model = TinyModel(*z['W'].shape, kind=str(z['kind']))
            self.model.params = {key: z[key].copy() for key in self.model.params}

    def predict(self, raw):
        raw = np.asarray(raw)
        if raw.ndim == 3 and raw.shape[-1] == 54:
            raw = raw.reshape(raw.shape[0], raw.shape[1], 18, 3)
        if raw.ndim != 4 or raw.shape[2:] != (18, 3) or raw.shape[1] < self.window + 1:
            raise ValueError(f'Expected N x samples x 18 x 3 or N x samples x 54; at least {self.window + 1} samples')
        features = encode(raw[:, -(self.window + 1):], self.scale, self.variant, self.window)
        probability = sigmoid(self.model.forward(features))
        return {'probability': probability, 'decision': (probability >= self.threshold).astype(np.int8),
                'threshold': self.threshold}


def load_detector(held, seed=7, policy='balanced', results=None):
    results = Path(results) if results is not None else Path(__file__).resolve().parent / 'results'
    frozen = json.loads((results / 'all_choices_before_test.json').read_text())
    matches = [fold for fold in frozen['folds'] if fold['held'] == held and fold['seed'] == seed]
    if len(matches) != 1:
        raise ValueError('Expected one saved held-object fold and seed')
    choices = [choice for choice in matches[0]['selected'] if choice['model'] == 'ct_v2' and choice['policy'] == policy]
    if len(choices) != 1:
        raise ValueError('Policy must be balanced, fpr20 or fpr10')
    choice = choices[0]
    return Detector(results / (choice['candidate']['name'] + '.npz'), choice['threshold'])
