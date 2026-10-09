import numpy as np
from ct_lif_study.reference.ct_lif import encode as ct_encode
from ct_lif_v3.core import raw_features, compress, ResidualLIF


def features(raw, scale, variant='ct_mean', compressed=True):
    raw = np.asarray(raw)
    if raw.ndim != 4 or raw.shape[1] < 34 or raw.shape[2:] != (18, 3):
        raise ValueError('Expected N x samples>=34 x 18 x 3')
    raw = raw[:, -34:]
    if variant == 'mean_lif':
        x = raw_features(raw, scale, 'mean')
    elif variant in ['ct_mean', 'ct_only']:
        ct = ct_encode(raw, scale, .02, 'ct')
        x = np.concatenate([ct, raw_features(raw, scale, 'mean')], axis=-1) if variant == 'ct_mean' else ct
    else:
        raise ValueError('Unknown variant')
    return compress(x) if compressed else x


def load_model(checkpoint):
    with np.load(checkpoint, allow_pickle=False) as z:
        model = ResidualLIF(*z['W'].shape, residual=False, pooling=str(z['pooling']))
        model.params = {key: z[key].copy() for key in model.params}
        return model, z['scale'].copy(), str(z['variant'])
