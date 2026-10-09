"""Export verified public uSkin data for grouped, causal host experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

EXPECTED_SHA256 = 'a2aef56bf4cfc5e2d17ae09772295cb207c733e3a73059913939ba5a7a1e2139'


def prepare(source, target):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError('Source is not the inspected pinned brush dataset')
    features, ys, held_ys, original, timestamps, groups, poses, trial_names = [], [], [], [], [], [], [], []
    with h5py.File(source, 'r') as hf:
        names = []
        hf.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Group)
                     and 'tactile_data_raw' in obj else None)
        for index, name in enumerate(sorted(names)):
            obj = hf[name]
            raw = np.asarray(obj['tactile_data_raw'], dtype=np.float32)
            labels = np.asarray(obj['tactile_slips_label'], dtype=np.int8)
            ts = np.asarray(obj['tactile_timestamps'], dtype=np.float64)
            # Axis order: pin0 x,y,z; pin1 x,y,z; ... pin17 x,y,z.
            features.append(raw.reshape(54, -1).T)
            y = np.full(labels.shape, -1, dtype=np.int8)
            y[np.isin(labels, [0, 4])] = 0
            y[np.isin(labels, [1, 5])] = 1
            held_y = np.full(labels.shape, -1, dtype=np.int8)
            held_y[labels == 4] = 0
            held_y[labels == 5] = 1
            held_ys.append(held_y)
            ys.append(y); original.append(labels); timestamps.append(ts-ts[0])
            groups.append(np.full(len(ts), index, dtype=np.int16))
            poses.append(np.full(len(ts), int(name.split('/')[2][4:]), dtype=np.int8))
            trial_names.append(name)
    np.savez_compressed(target, x=np.concatenate(features), y=np.concatenate(ys),
        y_held=np.concatenate(held_ys), original_label=np.concatenate(original), t_ms=np.concatenate(timestamps),
        group=np.concatenate(groups), pose=np.concatenate(poses),
        trial_names=np.asarray(trial_names), source_sha256=np.asarray(digest),
        evidence_type=np.asarray('public_hardware_recording_not_our_hardware'))
    print(json.dumps({'output': str(target), 'trials': len(trial_names),
                      'samples': sum(len(y) for y in ys), 'features': 54}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('data/public/ARQ_CRISP_2021/slipDataset_brush_tactile.h5'))
    ap.add_argument('--output', type=Path, default=Path('data/arq_brush_trials.npz'))
    args = ap.parse_args()
    prepare(args.source, args.output)
