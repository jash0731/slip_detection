"""Audit/prepare all-object ARQ data, without removing ignored temporal context."""
from pathlib import Path
import argparse
import hashlib
import json
import h5py
import numpy as np
try:
    from .fetch_arq_all import FILES, COMMIT, checksum
except ImportError:
    from fetch_arq_all import FILES, COMMIT, checksum


def prepare(source_dir, output, manifest_path):
    arrays = {k: [] for k in ('x','original_label','y_held','t_ms','group','object','pose')}
    names, hashes, records = [], [], []
    for object_index, (object_name, filename, expected, size) in enumerate(FILES):
        path = source_dir/filename
        if path.stat().st_size != size or checksum(path) != expected:
            raise ValueError(f'Unverified source: {filename}')
        hashes.append(expected)
        trials, dt_min, dt_max, counts = [], float('inf'), 0., {}
        with h5py.File(path, 'r') as hf:
            found = []
            hf.visititems(lambda name, obj: found.append(name) if isinstance(obj, h5py.Group)
                         and 'tactile_data_raw' in obj else None)
            for name in sorted(found):
                obj = hf[name]
                raw = np.asarray(obj['tactile_data_raw'], dtype=np.float32)
                label = np.asarray(obj['tactile_slips_label'], dtype=np.int8)
                ts = np.asarray(obj['tactile_timestamps'], dtype=np.float64)
                if raw.shape != (18,3,len(label)) or ts.shape != label.shape:
                    raise ValueError(f'Unexpected shapes: {name}')
                dt = np.diff(ts)
                if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(ts)) or np.any(dt<=0):
                    raise ValueError(f'Invalid values or timing: {name}')
                if not set(np.unique(label)).issubset(set(range(7))):
                    raise ValueError(f'Unrecognized labels: {name}')
                held = np.full(len(label), -1, dtype=np.int8)
                held[label==4] = 0; held[label==5] = 1
                group_id = len(names)
                arrays['x'].append(raw.reshape(54,-1).T)
                arrays['original_label'].append(label); arrays['y_held'].append(held)
                arrays['t_ms'].append(ts-ts[0])
                arrays['group'].append(np.full(len(ts),group_id,dtype=np.int16))
                arrays['object'].append(np.full(len(ts),object_index,dtype=np.int8))
                arrays['pose'].append(np.full(len(ts),int(name.split('/')[2][4:]),dtype=np.int8))
                names.append(name)
                tc = {str(int(k)):int(v) for k,v in zip(*np.unique(label,return_counts=True))}
                for k,v in tc.items(): counts[k] = counts.get(k,0)+v
                trials.append({'trial_id':name,'group':group_id,'samples':len(ts),
                               'label_counts':tc,'duration_ms':float(ts[-1]-ts[0])})
                dt_min=min(dt_min,float(dt.min()));dt_max=max(dt_max,float(dt.max()))
        records.append({'object':object_name,'filename':filename,'bytes':size,'sha256':expected,
             'source_url':f'https://media.githubusercontent.com/media/ARQ-CRISP/slip_detection_dataset_2021/{COMMIT}/data/{filename}',
             'verified':True,'trials':trials,'samples':sum(t['samples'] for t in trials),
             'label_counts':counts,'min_dt_ms':dt_min,'max_dt_ms':dt_max})
    packed = {key:np.concatenate(value) for key,value in arrays.items()}
    packed.update(trial_names=np.asarray(names),object_names=np.asarray([f[0] for f in FILES]),
        source_sha256=np.asarray(hashes),source_commit=np.asarray(COMMIT),
        evidence_type=np.asarray('public_hardware_recording_not_our_hardware'))
    np.savez_compressed(output,**packed)
    manifest={'dataset':'ARQ-CRISP slip_detection_dataset_2021','source_commit':COMMIT,
        'license':'BSD-3-Clause repository license; retained data/public/ARQ_CRISP_2021/LICENSE',
        'evidence_type':'public_hardware_recording_not_our_hardware','files':records,
        'prepared':{'filename':output.name,'sha256':checksum(output),
            'samples':len(packed['x']),'trials':len(names),'features':54,
            'label_mapping':'y_held:4=>0,5=>1,all other labels=>-1; all samples retained'}}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['prepared']))


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir',type=Path,default=root/'public'/'ARQ_CRISP_2021')
    ap.add_argument('--output',type=Path,default=root/'arq_all_trials.npz')
    ap.add_argument('--manifest',type=Path,default=root/'arq_all_manifest.json')
    args=ap.parse_args()
    prepare(args.source_dir,args.output,args.manifest)
