import numpy as np


def causal_derivatives(data):
    """Mean normal/shear axis differences, no cross-trial concatenation."""
    values = data['x'].astype(np.float32).reshape(-1, 18, 3).mean(axis=1)
    differences = np.zeros_like(values)
    for group in np.unique(data['group']):
        rows = np.flatnonzero(data['group'] == group)
        differences[rows[1:]] = values[rows[1:]] - values[rows[:-1]]
    return differences


def split_trials(data, held_object, seed):
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for obj in np.unique(data['object']):
        groups = np.unique(data['group'][data['object'] == obj])
        if obj == held_object:
            test.extend(groups.tolist())
        else:
            rng.shuffle(groups)
            nval = max(1, len(groups) // 5)
            validation.extend(groups[:nval].tolist())
            train.extend(groups[nval:].tolist())
    return {'train': train, 'validation': validation, 'test': test}


def prepare_fold(data, differences, splits, window=32, stride=16):
    train_mask = np.isin(data['group'], splits['train'])
    scales = np.maximum(np.percentile(np.abs(differences[train_mask]), 95, axis=0), 1e-6).astype(np.float32)
    normalized = np.clip(differences / scales, -4, 4)
    inputs = np.stack([np.maximum(normalized[:, i], 0) for i in range(3)] +
                      [np.maximum(-normalized[:, i], 0) for i in range(3)], axis=1)
    result = {}
    for name, groups in splits.items():
        xs, ys, records = [], [], []
        for group in groups:
            rows = np.flatnonzero(data['group'] == group)
            for end in range(window - 1, len(rows), stride):
                target = int(data['y_held'][rows[end]])
                if target < 0:
                    continue
                indices = rows[end - window + 1:end + 1]
                xs.append(inputs[indices])
                ys.append(target)
                records.append({'group': int(group), 'trial': str(data['trial_names'][group]),
                                'object': int(data['object'][rows[end]]), 'end_sample': end,
                                't_ms': float(data['t_ms'][rows[end]])})
        if not xs or len(set(ys)) < 2:
            raise ValueError(f'{name} must contain both classes')
        result[name] = (np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), records)
    return result, scales
