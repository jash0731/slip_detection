import numpy as np
from ct_lif_study.reference.ct_lif import sigmoid, metrics
from ct_lif_v2.core import components


def raw_features(raw, scale, variant):
    m, opposed = components(raw)
    if m.shape[1] < 32:
        raise ValueError('At least 33 raw frames are required')
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Expected three positive training scales')
    m, opposed = m[:, -32:] / scale, opposed[:, -32:] / scale
    if variant == 'invariant':
        return np.concatenate([np.abs(m), opposed], axis=-1)
    signed = np.concatenate([np.maximum(m, 0), np.maximum(-m, 0)], axis=-1)
    if variant == 'mean':
        return signed
    if variant != 'amplitude':
        raise ValueError('Unknown encoder')
    return np.concatenate([signed, opposed], axis=-1)


def compress(x):
    return x / (1 + x)


def pooling_weights(pooling, steps=32):
    weights = np.full(steps, 1 / steps)
    if pooling == 'recent':
        weights *= .5
        weights[-8:] += .5 / 8
    elif pooling != 'uniform':
        raise ValueError('Unknown pooling')
    return weights


def balanced_weights(y, objects):
    weights = np.zeros(len(y), dtype=float)
    for obj in np.unique(objects):
        for label in np.unique(y[objects == obj]):
            selected = (objects == obj) & (y == label)
            weights[selected] = 1 / selected.sum()
    return weights / weights.mean()


def weighted_loss(logits, y, weights):
    return float(np.mean(weights * (np.logaddexp(0, logits) - y * logits)))


class ResidualLIF:
    def __init__(self, inputs, hidden, residual=True, pooling='uniform', seed=7):
        rng = np.random.default_rng(seed)
        self.residual = bool(residual)
        self.pooling = pooling
        self.params = {'W': rng.normal(0, np.sqrt(2 / inputs), (inputs, hidden)),
                       'b': np.zeros(hidden), 'w': rng.normal(0, 1 / np.sqrt(hidden), hidden),
                       'c': np.zeros(1)}
        if self.residual:
            self.params['q'] = np.zeros(inputs)

    def forward(self, x, cache=False):
        state = np.zeros((len(x), len(self.params['b'])))
        pooled = np.zeros_like(state)
        weights = pooling_weights(self.pooling, x.shape[1])
        voltages = []
        for t in range(x.shape[1]):
            v = .875 * state + x[:, t] @ self.params['W'] + self.params['b']
            spike = (v >= 1).astype(float)
            state = v - spike
            pooled += weights[t] * spike
            if cache:
                voltages.append(v)
        evidence = np.einsum('ntc,t->nc', x, weights)
        logits = pooled @ self.params['w'] + self.params['c'][0]
        if self.residual:
            logits += evidence @ self.params['q']
        return (logits, (pooled, evidence, voltages, weights)) if cache else logits

    def gradients(self, x, y, sample_weight):
        logits, (pooled, evidence, voltages, weights) = self.forward(x, cache=True)
        dz = (sigmoid(logits) - y) * sample_weight / len(y)
        grads = {key: np.zeros_like(value) for key, value in self.params.items()}
        grads['w'] = pooled.T @ dz
        grads['c'][0] = dz.sum()
        if self.residual:
            grads['q'] = evidence.T @ dz
        direct = dz[:, None] * self.params['w'][None, :]
        carry = np.zeros_like(direct)
        for t in reversed(range(x.shape[1])):
            dv = weights[t] * direct / (1 + np.abs(voltages[t] - 1)) ** 2 + carry
            carry = .875 * dv
            grads['W'] += x[:, t].T @ dv
            grads['b'] += dv.sum(axis=0)
        return weighted_loss(logits, y, sample_weight), grads


def fit(model, x, y, objects, xv, yv, val_objects, rate, seed):
    rng = np.random.default_rng(seed)
    weights = balanced_weights(y, objects)
    val_weights = balanced_weights(yv, val_objects)
    first = {k: np.zeros_like(v) for k, v in model.params.items()}
    second = {k: np.zeros_like(v) for k, v in model.params.items()}
    best, bad, step = np.inf, 0, 0
    best_params = {k: v.copy() for k, v in model.params.items()}
    history = []
    for epoch in range(32):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 64):
            ids = order[start:start + 64]
            gain = np.exp(rng.uniform(np.log(.8), np.log(1.25), (len(ids), 1, 1)))
            batch = compress(x[ids] * gain)
            _, grads = model.gradients(batch, y[ids], weights[ids])
            norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            step += 1
            for k in model.params:
                g = grads[k] * min(1., 5. / max(norm, 1e-12))
                first[k] = .9 * first[k] + .1 * g
                second[k] = .999 * second[k] + .001 * g * g
                model.params[k] -= rate * (first[k] / (1 - .9 ** step)) / (np.sqrt(second[k] / (1 - .999 ** step)) + 1e-8)
        loss = weighted_loss(model.forward(compress(xv)), yv, val_weights)
        history.append({'epoch': epoch + 1, 'validation_weighted_loss': loss})
        if loss < best - 1e-10:
            best, bad = loss, 0
            best_params = {k: v.copy() for k, v in model.params.items()}
        else:
            bad += 1
        if bad >= 6:
            break
    model.params = best_params
    return history


def fit_linear(x, y, objects, pooling, ridge):
    evidence = np.einsum('ntc,t->nc', compress(x), pooling_weights(pooling))
    design = np.column_stack([evidence, np.ones(len(evidence))])
    weights = balanced_weights(y, objects)
    theta = np.zeros(design.shape[1])
    penalty = np.full(len(theta), ridge)
    penalty[-1] = 0
    for _ in range(40):
        logits = design @ theta
        p = sigmoid(logits)
        gradient = design.T @ (weights * (p - y)) / len(y) + penalty * theta
        hessian = design.T @ (design * (weights * p * (1 - p))[:, None]) / len(y)
        hessian += np.diag(penalty + 1e-9)
        delta = np.linalg.solve(hessian, gradient)
        loss = weighted_loss(logits, y, weights) + .5 * np.sum(penalty * theta ** 2)
        scale = 1.
        for _ in range(20):
            candidate = theta - scale * delta
            candidate_loss = weighted_loss(design @ candidate, y, weights) + .5 * np.sum(penalty * candidate ** 2)
            if candidate_loss <= loss + 1e-12:
                break
            scale *= .5
        theta = candidate
        if np.max(np.abs(scale * delta)) < 1e-8:
            break
    return {'q': theta[:-1], 'c': theta[-1:]}


def linear_forward(x, params, pooling):
    return np.einsum('ntc,t->nc', x, pooling_weights(pooling)) @ params['q'] + params['c'][0]


def validation_metrics(y, probability, objects, threshold):
    rows = [{'object': int(obj), **metrics(y[objects == obj], probability[objects == obj], threshold)}
            for obj in np.unique(objects)]
    recall = np.mean([r['slip_recall'] for r in rows if r['slip_recall'] is not None])
    fpr = np.mean([r['false_alarm_rate'] for r in rows if r['false_alarm_rate'] is not None])
    return {'threshold': float(threshold), 'balanced_accuracy': float((recall + 1 - fpr) / 2),
            'slip_recall': float(recall), 'false_alarm_rate': float(fpr), 'per_object': rows,
            'objects_without_non_slip': [r['object'] for r in rows if r['false_alarm_rate'] is None]}


def operating_points(y, probability, objects, reference):
    points = [validation_metrics(y, probability, objects, t)
              for t in np.r_[np.linspace(0, 1, 201), np.nextafter(1., 2.)]]
    def key(point):
        return (point['balanced_accuracy'], -point['false_alarm_rate'], -abs(point['threshold'] - .5))
    for point in points:
        point['validation_joint_improvement'] = bool(point['slip_recall'] >= reference['slip_recall'] - 1e-12
                                                    and point['false_alarm_rate'] <= reference['false_alarm_rate'] + 1e-12)
    balanced = max(points, key=key)
    eligible = [p for p in points if p['validation_joint_improvement']]
    joint = max(eligible, key=key) if eligible else balanced
    return {'balanced': balanced, 'joint': joint}
