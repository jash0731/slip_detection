import argparse
import json
from pathlib import Path
import numpy as np


def check_raw(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 4 or raw.shape[1] < 3 or raw.shape[2:] != (18, 3):
        raise ValueError('Expected raw shape (windows, time>=3, 18, 3)')
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Input must be nonempty and finite')
    return raw


def fit_scale(raw):
    d = np.diff(check_raw(raw), axis=1)
    scale = np.quantile(np.abs(d).mean(axis=2), .95, axis=(0, 1))
    if np.any(scale <= 0):
        raise ValueError('Each axis needs nonzero training variation')
    return scale


def encode(raw, scale, gate_fraction=.02, variant='ct'):
    raw = check_raw(raw)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('scale must contain three positive finite values')
    if not np.isfinite(gate_fraction) or gate_fraction <= 0:
        raise ValueError('gate_fraction must be positive')
    d = np.diff(raw, axis=1)
    a = np.abs(d).mean(axis=2)
    m = d.mean(axis=2)
    if variant == 'mean':
        z = np.clip(m[:, 1:] / scale, -1, 1)
        return np.concatenate([np.maximum(z, 0), np.maximum(-z, 0)], axis=-1)
    good = a > 1e-12 * scale
    q = np.divide(m, a, out=np.zeros_like(m), where=good)
    c = np.where(good, np.clip(1 - np.abs(q), 0, 1), 0)
    total = np.abs(d).sum(axis=2, keepdims=True)
    p = np.divide(np.abs(d), total, out=np.zeros_like(d), where=good[:, :, None, :])
    turnover = .5 * np.abs(p[:, 1:] - p[:, :-1]).sum(axis=2)
    turnover *= good[:, 1:] & good[:, :-1]
    r = np.minimum(1, a / (gate_fraction * scale))
    common = r[:, 1:] * q[:, 1:]
    cancellation = r[:, 1:] * c[:, 1:]
    joint = np.minimum(r[:, 1:], r[:, :-1]) * c[:, 1:] * turnover
    if variant == 'c_only':
        joint = np.zeros_like(joint)
    elif variant == 'turnover':
        joint = np.minimum(r[:, 1:], r[:, :-1]) * turnover
    elif variant == 'entropy':
        h = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=2) / np.log(18)
        cancellation = r[:, 1:] * h[:, 1:]
        joint = np.minimum(r[:, 1:], r[:, :-1]) * np.abs(h[:, 1:] - h[:, :-1])
    elif variant != 'ct':
        raise ValueError('Unknown encoder variant')
    return np.concatenate([np.maximum(common, 0), np.maximum(-common, 0), cancellation, joint], axis=-1)


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def loss_value(logits, y, positive_weight=1.):
    weights = np.where(y == 1, positive_weight, 1.)
    return float(np.mean(weights * (np.logaddexp(0, logits) - y * logits)))


def metrics(y, p, threshold):
    pred = p >= threshold
    pos = y == 1
    neg = ~pos
    recall = float(pred[pos].mean()) if pos.any() else None
    fpr = float(pred[neg].mean()) if neg.any() else None
    return {'balanced_accuracy': (recall + 1 - fpr) / 2 if recall is not None and fpr is not None else None,
            'slip_recall': recall, 'false_alarm_rate': fpr,
            'tp': int((pred & pos).sum()), 'fn': int((~pred & pos).sum()),
            'fp': int((pred & neg).sum()), 'tn': int((~pred & neg).sum())}


class TinyModel:
    def __init__(self, inputs=12, hidden=16, kind='lif', seed=7):
        if kind not in ('lif', 'temporal', 'ann'):
            raise ValueError('kind must be lif, temporal or ann')
        rng = np.random.default_rng(seed)
        self.kind = kind
        self.beta = .875
        self.params = {'W': rng.normal(0, np.sqrt(2 / inputs), (inputs, hidden)),
                       'b': np.zeros(hidden), 'w': rng.normal(0, 1 / np.sqrt(hidden), hidden),
                       'c': np.zeros(1)}

    def forward(self, x, cache=False):
        state = np.zeros((len(x), len(self.params['b'])))
        activities, voltages = [], []
        for t in range(x.shape[1]):
            v = x[:, t] @ self.params['W'] + self.params['b']
            if self.kind != 'ann':
                v = v + self.beta * state
            if self.kind == 'lif':
                h = (v >= 1).astype(float)
                state = v - h
            else:
                h = np.maximum(v, 0)
                state = h
            activities.append(h)
            voltages.append(v)
        h = np.stack(activities, axis=1)
        pooled = h.mean(axis=1)
        z = pooled @ self.params['w'] + self.params['c'][0]
        return (z, (x, pooled, voltages)) if cache else z

    def gradients(self, x, y, positive_weight=1.):
        z, (_, pooled, voltages) = self.forward(x, cache=True)
        dz = (sigmoid(z) - y) * np.where(y == 1, positive_weight, 1.) / len(y)
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        grads['w'] = pooled.T @ dz
        grads['c'][0] = dz.sum()
        direct = dz[:, None] * self.params['w'][None, :] / x.shape[1]
        carry = np.zeros_like(direct)
        for t in reversed(range(x.shape[1])):
            v = voltages[t]
            if self.kind == 'lif':
                dv = direct / (1 + np.abs(v - 1)) ** 2 + carry
                carry = self.beta * dv
            else:
                dv = (direct + carry) * (v > 0)
                carry = self.beta * dv if self.kind == 'temporal' else np.zeros_like(dv)
            grads['W'] += x[:, t].T @ dv
            grads['b'] += dv.sum(axis=0)
        return loss_value(z, y, positive_weight), grads


def fit(model, x, y, xv, yv, epochs=18, lr=.005, batch_size=64, seed=7):
    rng = np.random.default_rng(seed)
    positive_weight = float((y == 0).sum() / (y == 1).sum())
    first = {k: np.zeros_like(v) for k, v in model.params.items()}
    second = {k: np.zeros_like(v) for k, v in model.params.items()}
    best, bad, step = np.inf, 0, 0
    best_params = {k: v.copy() for k, v in model.params.items()}
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            ids = order[start:start + batch_size]
            _, grads = model.gradients(x[ids], y[ids], positive_weight)
            norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            step += 1
            for k in model.params:
                g = grads[k] * min(1., 5. / max(norm, 1e-12))
                first[k] = .9 * first[k] + .1 * g
                second[k] = .999 * second[k] + .001 * g * g
                model.params[k] -= lr * (first[k] / (1 - .9 ** step)) / (np.sqrt(second[k] / (1 - .999 ** step)) + 1e-8)
        validation_loss = loss_value(model.forward(xv), yv, positive_weight)
        history.append({'epoch': epoch + 1, 'validation_loss': validation_loss})
        if validation_loss < best - 1e-10:
            best = validation_loss
            best_params = {k: v.copy() for k, v in model.params.items()}
            bad = 0
        else:
            bad += 1
        if bad >= 4:
            break
    model.params = best_params
    pv = sigmoid(model.forward(xv))
    thresholds = np.linspace(.05, .95, 91)
    threshold = max(thresholds, key=lambda t: (metrics(yv, pv, t)['balanced_accuracy'], -abs(t - .5), -t))
    return float(threshold), history


def load_split(path):
    with np.load(path, allow_pickle=False) as data:
        raw = check_raw(data['raw'])
        y = np.asarray(data['y'], dtype=float)
        objects = np.asarray(data['object_id']).astype(str)
        trials = np.asarray(data['trial_id']).astype(str)
    if any(a.shape != (len(raw),) for a in (y, objects, trials)) or not np.isin(y, [0, 1]).all():
        raise ValueError('y, object_id and trial_id must have one valid entry per window')
    return raw, y, objects, trials


def audit_splits(splits):
    keys = [set(zip(s[2], s[3])) for s in splits]
    for i in range(len(keys)):
        for j in range(i):
            if keys[i] & keys[j]:
                raise ValueError('Whole-trial split overlap detected')
    for s in splits[:2]:
        if set(s[1]) != {0., 1.}:
            raise ValueError('Training and validation must each contain both classes')
    if len(splits) == 3:
        if len(set(splits[2][2])) != 1 or set(splits[2][2]) & (set(splits[0][2]) | set(splits[1][2])):
            raise ValueError('Test must be exactly one entirely held-out object')
    if len({s[0].shape[1:] for s in splits}) != 1:
        raise ValueError('All splits must have the same raw window shape')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--test')
    parser.add_argument('--out', required=True)
    parser.add_argument('--variant', choices=['ct', 'mean', 'c_only', 'turnover', 'entropy'], default='ct')
    parser.add_argument('--kind', choices=['lif', 'temporal', 'ann'], default='lif')
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--lr', type=float, default=.005)
    parser.add_argument('--gate', type=float, default=.02)
    args = parser.parse_args()
    if args.hidden < 1 or args.lr <= 0 or not np.isfinite(args.lr):
        raise ValueError('Invalid hidden width or learning rate')
    splits = [load_split(args.train), load_split(args.validation)]
    if args.test:
        splits.append(load_split(args.test))
    audit_splits(splits)
    scale = fit_scale(splits[0][0])
    x = [encode(s[0], scale, args.gate, args.variant) for s in splits[:2]]
    model = TinyModel(x[0].shape[-1], args.hidden, args.kind, args.seed)
    threshold, history = fit(model, x[0], splits[0][1], x[1], splits[1][1], lr=args.lr, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / 'checkpoint.npz', **model.params, scale=scale, threshold=threshold,
             gate=args.gate, variant=args.variant, kind=args.kind, beta=model.beta)
    report = {'status': 'exploratory; novelty and hardware performance unverified',
              'config': vars(args), 'threshold': threshold, 'history': history,
              'split_audit': {'development_objects': sorted(set(splits[0][2]) | set(splits[1][2])),
                              'raw_shape': list(splits[0][0].shape[1:])},
              'parameters': sum(v.size for v in model.params.values()),
              'validation': metrics(splits[1][1], sigmoid(model.forward(x[1])), threshold)}
    (out / 'selection_before_test.json').write_text(json.dumps(report, indent=2))
    if args.test:
        test = splits[2]
        probabilities = sigmoid(model.forward(encode(test[0], scale, args.gate, args.variant)))
        report['test'] = metrics(test[1], probabilities, threshold)
        np.savez(out / 'predictions.npz', probability=probabilities, y=test[1], object_id=test[2], trial_id=test[3])
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
