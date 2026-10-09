"""Shared deterministic protocol helpers; source data and original exports stay frozen."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from advanced.data import causal_derivatives, prepare_fold, split_trials
from advanced.models import TemporalModel, HIDDEN, BETA
from advanced.train import predict, quality, select_threshold

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('ann', 'lif', 'leaky_relu')
WINDOWS = (16, 32, 64)
RATES = (.005, .01)
SEEDS = (7, 19)


class LeakyRelu(TemporalModel):
    """Matched learned capacity, fixed diagonal recurrence, no spikes/reset."""
    def __init__(self):
        super().__init__('ann')

    def forward(self, inputs):
        state = inputs.new_zeros((len(inputs), HIDDEN))
        total = torch.zeros_like(state)
        for tick in range(inputs.shape[1]):
            state = torch.relu(BETA * state + self.input_layer(inputs[:, tick]))
            total = total + state
        return self.output_layer(total / inputs.shape[1]).squeeze(1)


def make_model(kind):
    return LeakyRelu() if kind == 'leaky_relu' else TemporalModel(kind)


def read_data():
    path = ROOT / 'data/arq_all_trials.npz'
    original = json.loads((ROOT / 'advanced/results/metadata.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != original['data_sha256']:
        raise ValueError('Prepared source changed since the original experiment.')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return data, causal_derivatives(data), original


def common_endpoints(prepared, minimum=63):
    aligned = {}
    for name, (x, y, records) in prepared.items():
        keep = np.array([row['end_sample'] >= minimum for row in records])
        aligned[name] = (x[keep], y[keep], [r for r, use in zip(records, keep) if use])
        if not keep.any() or len(np.unique(y[keep])) != 2:
            raise ValueError('Aligned partition must have both classes.')
    return aligned


def prepare_aligned(data, derivatives, splits, window):
    prepared, scales = prepare_fold(data, derivatives, splits, window=window, stride=16)
    return common_endpoints(prepared), scales


def perturb(x, records, condition, seed=20261007, draw=0):
    """Permute entire past frames; endpoint and within-frame channel identity remain."""
    if condition == 'ordered':
        return x.copy()
    result = x.copy()
    if condition == 'reversed_history':
        result[:, :-1] = x[:, -2::-1]
    elif condition == 'shuffled_history':
        for index, record in enumerate(records):
            rng = np.random.default_rng(np.random.SeedSequence(
                [seed, draw, int(record['group']), int(record['end_sample']), x.shape[1]]))
            order = rng.permutation(x.shape[1] - 1)
            result[index, :-1] = x[index, order]
    else:
        raise ValueError(condition)
    return result


def shuffled_partitions(prepared, seed):
    return {name: (perturb(x, records, 'shuffled_history', seed=seed + 10000), y, records)
            for name, (x, y, records) in prepared.items()}


def train_candidate(kind, training, validation, rate, seed, epochs=18):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = make_model(kind)
    initial = model.input_layer.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=1e-4)
    x, y, _ = training
    vx, vy, _ = validation
    positive_weight = float(np.sum(y == 0) / np.sum(y == 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    best_loss, best_state, best_epoch, stale = float('inf'), None, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        total = 0.
        for start in range(0, len(order), 256):
            indices = order[start:start + 256]
            optimizer.zero_grad()
            logits = model(tx[indices])
            loss = criterion(logits, ty[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        vlogits = predict(model, vx)
        vloss = float(criterion(torch.from_numpy(vlogits), torch.from_numpy(vy)))
        history.append({'epoch': epoch + 1, 'train_loss': total / len(x), 'validation_loss': vloss})
        if vloss < best_loss - 1e-5:
            best_loss, best_epoch, stale = vloss, epoch + 1, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(best_state)
    vlogits = predict(model, vx)
    threshold = select_threshold(vy, vlogits)
    changed = float((model.input_layer.weight - initial).abs().max().detach())
    if not changed:
        raise ValueError('Trainable hidden weights did not change.')
    return model, threshold, history, {
        'selected_epoch': best_epoch, 'epochs_run': len(history), 'validation_loss': best_loss,
        'validation_balanced_accuracy': quality(vy, vlogits >= threshold)['balanced_accuracy'],
        'threshold': threshold, 'positive_loss_weight': positive_weight,
        'max_input_weight_change': changed, 'parameters': sum(p.numel() for p in model.parameters())}


def select_candidate(candidates):
    """Only recorded validation quality, then prespecified deterministic tie breaks."""
    return min(candidates, key=lambda c: (-c['info']['validation_balanced_accuracy'],
                                         c['window'], abs(c['rate'] - .01)))


def detailed_quality(labels, decisions):
    result = quality(labels, decisions)
    tp, tn, fp, fn = (result[key] for key in ('tp', 'tn', 'fp', 'fn'))
    result.update(slip_recall=tp / (tp + fn) if tp + fn else None,
                  non_slip_recall=tn / (tn + fp) if tn + fp else None,
                  precision=tp / (tp + fp) if tp + fp else None,
                  false_alarm_rate=fp / (tn + fp) if tn + fp else None,
                  miss_rate=fn / (tp + fn) if tp + fn else None)
    if not tp + fn or not tn + fp:
        result['balanced_accuracy'] = None
    return result
