import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from ct_lif_study.run import ROOT, windows, load_data, verify_splits, sha, write_json, write_csv, detailed
from ct_lif_study.reference.ct_lif import TinyModel, sigmoid, encode as encode_v1
from .core import components, encode_components, select_operating_points, fit


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
FAMILIES = {'amplitude_lif': ('amplitude', 'lif'), 'shrink_lif': ('shrink', 'lif'),
            'mean_lif': ('mean', 'lif'), 'amplitude_temporal': ('amplitude', 'temporal')}
POLICIES = ['balanced', 'fpr20', 'fpr10']


def fit_scale(data, train_groups):
    activities = []
    for group in train_groups:
        raw = data['x'][data['group'] == group].astype(float).reshape(-1,18,3)
        activities.append(np.abs(np.diff(raw, axis=0)).mean(axis=1))
    scale = np.quantile(np.concatenate(activities), .95, axis=0)
    if np.any(scale <= 0):
        raise ValueError('Inactive axis')
    return scale


def choose(candidates, policy):
    def key(c):
        point = c['operating_points'][policy]
        cost = (c['window'] * c['inputs'] * c['hidden'], c['parameters'], c['rate'])
        if policy == 'balanced':
            return (-point['macro_ba'], point['maximum_fpr'], *cost)
        return (-point['minimum_recall'], -point['macro_recall'], -point['macro_ba'], *cost)
    return min(candidates, key=key)


def train():
    data, original = load_data()
    OUT.mkdir(exist_ok=False)
    protocol = {'candidate': 'CT-LIF v2: amplitude or cancellation-shrink compact representations',
                'frozen_before_new_test': True, 'families': FAMILIES,
                'hidden': [8,10], 'windows': [16,32], 'learning_rates': [.005,.01],
                'fits': 192, 'seeds': [7,19], 'policies': POLICIES,
                'gain_augmentation': 'common positive gain loguniform[.5,2] per training window before clipping; all new families',
                'selection': 'macro validation BA for balanced; maximize minimum validation object recall under max-object FPR cap for alarm policies',
                'primary_alarm_target': 'fpr10; every validation object with non-slip examples <=10%; minimum recall>=40% reported as usefulness check, not guaranteed',
                'single_class_validation': 'absent-class rates stay undefined; macro score averages each available class recall over its eligible objects; missing non-slip validation explicitly recorded',
                'candidate_family_choice': 'amplitude/shrink chosen using same validation rule; doubles candidate budget relative to one control family',
                'scale': 'p95 mean absolute pin differences across unique raw rows of training trials only',
                'historical_controls': 'v1 CT-LIF, v1 mean-LIF28, v1 cancellation-only recalibrated with identical operating-point rule on their validation data',
                'original_fixed_thresholds_also_reported': True,
                'evidence': 'exploratory repeated dataset; not independent confirmation',
                'no_energy_or_esp32_claim': True,
                'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [HERE/'core.py', Path(__file__), ROOT/'data/arq_all_trials.npz', ROOT/'ct_lif_study/reference/ct_lif.py']},
                'environment': {'python':sys.version,'numpy':np.__version__,'platform':platform.platform()}}
    write_json(OUT/'protocol.json', protocol)
    all_choices, count = [], 0
    started=time.perf_counter()
    for fold in original['folds']:
        held,seed,splits=verify_splits(data,fold)
        tr,va=windows(data,splits['train']),windows(data,splits['validation'])
        train_parts,val_parts=components(tr[0]),components(va[0])
        val_objects=np.array([r['object'] for r in va[2]])
        scale=fit_scale(data,splits['train'])
        candidates=[]
        for family,(variant,kind) in FAMILIES.items():
            for window in [16,32]:
                x=encode_components(*train_parts,scale,variant,window,clip=False)
                xv=encode_components(*val_parts,scale,variant,window,clip=False)
                for hidden in [8,10]:
                    for rate in [.005,.01]:
                        name=f'object{held}_seed{seed}_{family}_w{window}_h{hidden}_lr{rate}'
                        model=TinyModel(x.shape[-1],hidden,kind,seed)
                        history=fit(model,x,tr[1],xv,va[1],lr=rate,seed=seed)
                        valp=sigmoid(model.forward(np.minimum(xv,1)))
                        points=select_operating_points(va[1],valp,val_objects)
                        info={'name':name,'family':family,'variant':variant,'kind':kind,'window':window,'hidden':hidden,
                              'inputs':x.shape[-1],'rate':rate,'seed':seed,'held':held,'scale':scale.tolist(),
                              'parameters':int(sum(p.size for p in model.params.values())),
                              'operating_points':points,'history':history}
                        np.savez(OUT/(name+'.npz'),**model.params,scale=scale,kind=kind,variant=variant,window=window)
                        write_json(OUT/(name+'.json'),info)
                        candidates.append(info)
                        count+=1
            print(f'TRAIN {count}/192: object{held} seed{seed} {family} complete; test untouched',flush=True)
        selected=[]
        for policy in POLICIES:
            for family in [*FAMILIES,'ct_v2']:
                options=[c for c in candidates if c['family']==family] if family!='ct_v2' else [c for c in candidates if c['family'] in ['amplitude_lif','shrink_lif']]
                winner=choose(options,policy)
                selected.append({'model':family,'policy':policy,'candidate':winner,'threshold':winner['operating_points'][policy]['threshold']})
        for oldname in ['ct_lif','mean_lif28','c_only_lif']:
            tag=f'object{held}_seed{seed}_{oldname}'
            path=ROOT/'ct_lif_study/results'/(tag+'.npz')
            with np.load(path,allow_pickle=False) as z:
                model=TinyModel(*z['W'].shape,kind=str(z['kind']))
                model.params={k:z[k].copy() for k in model.params}
                valx=encode_v1(va[0],z['scale'],float(z['gate']),str(z['variant']))
            points=select_operating_points(va[1],sigmoid(model.forward(valx)),val_objects)
            for policy in POLICIES:
                selected.append({'model':'previous_'+oldname,'policy':policy,'old_path':str(path.relative_to(ROOT)),
                                 'threshold':points[policy]['threshold'],'operating_point':points[policy]})
        all_choices.append({'held':held,'held_object':str(data['object_names'][held]),'seed':seed,'splits':splits,'selected':selected})
        write_json(OUT/f'object{held}_seed{seed}_choices.json',all_choices[-1])
    write_json(OUT/'all_choices_before_test.json',{'folds':all_choices,'fits':count,'training_seconds':time.perf_counter()-started})
    print('ALL CHOICES FROZEN. No new test inference yet.',flush=True)


def evaluate():
    if (OUT/'summary.csv').exists():
        raise ValueError('Evaluation already exists')
    data,_=load_data()
    protocol=json.loads((OUT/'protocol.json').read_text())
    for rel,digest in protocol['source_hashes'].items():
        if sha(ROOT/rel)!=digest: raise ValueError('Frozen source changed')
    frozen=json.loads((OUT/'all_choices_before_test.json').read_text())
    summaries,predictions=[],[]
    for fold in frozen['folds']:
        raw,y,records=windows(data,fold['splits']['test'])
        parts=components(raw)
        cache={}
        for choice in fold['selected']:
            if 'old_path' in choice:
                path=ROOT/choice['old_path']
                family='previous'
            else:
                candidate=choice['candidate']
                path=OUT/(candidate['name']+'.npz')
                family=candidate['family']
            if path not in cache:
                with np.load(path,allow_pickle=False) as z:
                    model=TinyModel(*z['W'].shape,kind=str(z['kind']))
                    model.params={k:z[k].copy() for k in model.params}
                    x=encode_v1(raw,z['scale'],float(z['gate']),str(z['variant'])) if family=='previous' else encode_components(*parts,z['scale'],str(z['variant']),int(z['window']))
                cache[path]=sigmoid(model.forward(x))
            probability=cache[path]
            base={'model':choice['model'],'policy':choice['policy'],'held':fold['held'],'held_object':fold['held_object'],
                  'seed':fold['seed'],'threshold':choice['threshold'],'selected_family':family,'checkpoint':str(path.relative_to(ROOT))}
            result={**base,**detailed(y,probability,choice['threshold'])}
            summaries.append(result)
            predictions.extend({**base,**r,'label':int(label),'probability':float(p),'decision':int(p>=choice['threshold'])}
                               for r,label,p in zip(records,y,probability))
            if choice['model']=='ct_v2':
                print(f'TEST ct_v2 {choice["policy"]} {fold["held_object"]} seed{fold["seed"]}: BA={result["balanced_accuracy"]:.4f}; recall={result["slip_recall"]:.4f}; FPR={result["false_alarm_rate"]:.4f}; {family}',flush=True)
    write_csv(OUT/'summary.csv',summaries)
    write_csv(OUT/'predictions.csv',predictions)
    write_json(OUT/'complete.json',{'summary_rows':len(summaries),'prediction_rows':len(predictions),
                                 'choices_sha256':sha(OUT/'all_choices_before_test.json'),'protocol_sha256':sha(OUT/'protocol.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['train','evaluate'])
    args=parser.parse_args()
    train() if args.stage=='train' else evaluate()
