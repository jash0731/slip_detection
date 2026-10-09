import numpy as np
from ct_lif_study.reference.ct_lif import TinyModel, sigmoid, loss_value, metrics


def components(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 4 or raw.shape[2:] != (18, 3) or raw.shape[1] < 2 or not np.isfinite(raw).all():
        raise ValueError('Expected finite N x samples x 18 x 3 input')
    d = np.diff(raw, axis=1)
    m = d.mean(axis=2)
    a = np.abs(d).mean(axis=2)
    return m, np.maximum(a - np.abs(m), 0)


def encode_components(m, opposed, scale, variant, window, clip=True):
    if not 1 <= window <= m.shape[1] or m.shape != opposed.shape:
        raise ValueError('Invalid feature history')
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid training scale')
    m, opposed = m[:, -window:], opposed[:, -window:]
    if variant == 'shrink':
        m = np.sign(m) * np.maximum(np.abs(m) - .5 * opposed, 0)
    z = m / scale
    x = np.concatenate([np.maximum(z, 0), np.maximum(-z, 0)], axis=-1)
    if variant == 'amplitude':
        x = np.concatenate([x, opposed / scale], axis=-1)
    elif variant not in ('mean', 'shrink'):
        raise ValueError('Unknown variant')
    return np.minimum(x, 1) if clip else x


def encode(raw, scale, variant='amplitude', window=32, clip=True):
    return encode_components(*components(raw), scale, variant, window, clip)


def object_metrics(y, p, objects, threshold):
    return [{'object':int(obj),'positive_count':int((y[objects==obj]==1).sum()),
             'negative_count':int((y[objects==obj]==0).sum()),
             **metrics(y[objects == obj], p[objects == obj], threshold)} for obj in sorted(set(objects))]


def select_operating_points(y, p, objects):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both validation classes required')
    candidates = []
    for threshold in np.r_[np.linspace(0, 1, 201), np.nextafter(1., 2.)]:
        rows = object_metrics(y, p, objects, threshold)
        recalls=[r['slip_recall'] for r in rows if r['slip_recall'] is not None]
        fprs=[r['false_alarm_rate'] for r in rows if r['false_alarm_rate'] is not None]
        candidates.append({'threshold': float(threshold),
                           'macro_ba': float((np.mean(recalls)+1-np.mean(fprs))/2),
                           'macro_recall': float(np.mean(recalls)),
                           'minimum_recall': min(recalls),
                           'macro_fpr': float(np.mean(fprs)),
                           'maximum_fpr': max(fprs),
                           'objects_without_non_slip_validation': [r['object'] for r in rows if not r['negative_count']],
                           'per_object': rows})
    out = {'balanced': max(candidates, key=lambda r: (r['macro_ba'], -r['maximum_fpr'], -abs(r['threshold']-.5), r['threshold']))}
    for cap in [.2, .1]:
        allowed = [r for r in candidates if r['maximum_fpr'] <= cap + 1e-12]
        selected = max(allowed, key=lambda r: (r['minimum_recall'], r['macro_recall'], r['macro_ba'], -r['macro_fpr'], r['threshold']))
        out[f'fpr{int(cap*100)}'] = {**selected, 'validation_fpr_cap': cap,
                                    'useful_validation_recall': selected['minimum_recall'] >= .4}
    return out


def fit(model, x, y, xv, yv, epochs=18, lr=.005, seed=7):
    rng = np.random.default_rng(seed)
    positive_weight = float((y == 0).sum() / (y == 1).sum())
    first = {k: np.zeros_like(v) for k, v in model.params.items()}
    second = {k: np.zeros_like(v) for k, v in model.params.items()}
    best, bad, step = np.inf, 0, 0
    best_params = {k: v.copy() for k, v in model.params.items()}
    history = []
    xv = np.minimum(xv, 1)
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 64):
            ids = order[start:start + 64]
            gains = np.exp(rng.uniform(np.log(.5), np.log(2.), (len(ids), 1, 1)))
            batch = np.minimum(x[ids] * gains, 1)
            _, grads = model.gradients(batch, y[ids], positive_weight)
            norm = np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
            step += 1
            for k in model.params:
                g = grads[k] * min(1., 5./max(norm, 1e-12))
                first[k] = .9*first[k] + .1*g
                second[k] = .999*second[k] + .001*g*g
                model.params[k] -= lr*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
        val_loss = loss_value(model.forward(xv), yv, positive_weight)
        history.append({'epoch': epoch+1, 'validation_loss': val_loss})
        if val_loss < best-1e-10:
            best, bad = val_loss, 0
            best_params = {k: v.copy() for k, v in model.params.items()}
        else:
            bad += 1
        if bad >= 4:
            break
    model.params = best_params
    return history
