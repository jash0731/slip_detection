import argparse
import json
import time
import numpy as np
from pathlib import Path
from .core import features, load_model
from ct_lif_v3.core import ResidualLIF, fit, operating_points
from ct_lif_v2.run import fit_scale
from ct_lif_study.run import ROOT, load_data, windows, verify_splits, write_json, write_csv, sha, detailed
from ct_lif_study.reference.ct_lif import sigmoid


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'


def train():
    data, metadata = load_data()
    OUT.mkdir(exist_ok=False)
    paths = [HERE/'core.py', HERE/'run.py', ROOT/'ct_lif_v3/core.py', ROOT/'ct_lif_v2/run.py',
             ROOT/'ct_lif_study/reference/ct_lif.py', ROOT/'ct_lif_study/run.py', ROOT/'data/arq_all_trials.npz']
    protocol = {'algorithm':'CT-LIF plus optimized mean input', 'primary':'ct_mean', 'input_channels':18,
                'branches':'12 original CT common/cancellation/turnover channels + 6 optimized signed mean-motion channels, jointly compressed u/(1+u)',
                'controls':['ct_only','mean_lif'], 'hidden':[8,12], 'rates':[.005,.01], 'pooling':['uniform','recent'],
                'fits':144, 'selection':'validation macro balanced accuracy, then lower FPR and fewer parameters',
                'training':'v3 Adam,32 max epochs,patience6,batch64,object/class-balanced weights; shared feature gain loguniform[.8,1.25] before compression',
                'augmentation_note':'feature-space gain; not equivalent to multiplying raw CT pin data, because CT coherence and turnover are normalized',
                'exploratory_reused_benchmark':True, 'no_novelty_or_hardware_claim':True,
                'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    write_json(OUT/'protocol.json',protocol)
    folds, count = [], 0
    start=time.perf_counter()
    for fold in metadata['folds']:
        held,seed,splits=verify_splits(data,fold)
        tr,va=windows(data,splits['train']),windows(data,splits['validation'])
        obj=np.array([r['object'] for r in tr[2]])
        vo=np.array([r['object'] for r in va[2]])
        scale=fit_scale(data,splits['train'])
        selected=[]
        for variant in ['ct_mean','ct_only','mean_lif']:
            x,xv=features(tr[0],scale,variant,False),features(va[0],scale,variant,False)
            choices=[]
            for pooling in ['uniform','recent']:
                for hidden in [8,12]:
                    for rate in [.005,.01]:
                        name=f'object{held}_seed{seed}_{variant}_{pooling}_h{hidden}_lr{rate}'
                        model=ResidualLIF(x.shape[-1],hidden,False,pooling,seed)
                        history=fit(model,x,tr[1],obj,xv,va[1],vo,rate,seed)
                        probability=sigmoid(model.forward(xv/(1+xv)))
                        point=operating_points(va[1],probability,vo,{'slip_recall':0.,'false_alarm_rate':1.})['balanced']
                        info={'name':name,'variant':variant,'pooling':pooling,'hidden':hidden,'rate':rate,
                              'parameters':sum(v.size for v in model.params.values()),'inputs':x.shape[-1],
                              'threshold':point['threshold'],'validation':point,'history':history}
                        np.savez(OUT/(name+'.npz'),**model.params,variant=variant,pooling=pooling,scale=scale)
                        write_json(OUT/(name+'.json'),info)
                        choices.append(info);count+=1
            winner=min(choices,key=lambda c:(-c['validation']['balanced_accuracy'],c['validation']['false_alarm_rate'],c['parameters'],c['name']))
            selected.append(winner)
            print(f'TRAIN {count}/144 object{held} seed{seed} {variant}; test untouched',flush=True)
        folds.append({'held':held,'seed':seed,'held_object':str(data['object_names'][held]),'splits':splits,'selected':selected})
    write_json(OUT/'all_choices_before_test.json',{'folds':folds,'fits':count,'training_seconds':time.perf_counter()-start,
                                                'checkpoint_hashes':{p.name:sha(p) for p in sorted(OUT.glob('*.npz'))}})
    print('ALL CHOICES FROZEN; no new test inference',flush=True)


def evaluate():
    if (OUT/'summary.csv').exists(): raise ValueError('Evaluation already exists')
    data,_=load_data()
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['source_hashes'].items():
        if sha(ROOT/path)!=digest: raise ValueError('Frozen source changed')
    frozen=json.loads((OUT/'all_choices_before_test.json').read_text())
    for name,digest in frozen['checkpoint_hashes'].items():
        if sha(OUT/name)!=digest: raise ValueError('Checkpoint changed')
    summaries,predictions=[],[]
    for fold in frozen['folds']:
        raw,y,records=windows(data,fold['splits']['test'])
        for choice in fold['selected']:
            model,scale,variant=load_model(OUT/(choice['name']+'.npz'))
            p=sigmoid(model.forward(features(raw,scale,variant)))
            base={'model':variant,'held':fold['held'],'seed':fold['seed'],'held_object':fold['held_object'],
                  'threshold':choice['threshold'],'checkpoint':choice['name']+'.npz','parameters':choice['parameters']}
            row={**base,**detailed(y,p,choice['threshold'])};summaries.append(row)
            predictions.extend({**base,**r,'label':int(label),'probability':float(prob),'decision':int(prob>=choice['threshold'])}
                               for r,label,prob in zip(records,y,p))
            print(f'TEST {variant} {fold["held_object"]} seed{fold["seed"]} BA={row["balanced_accuracy"]:.4f}',flush=True)
    write_csv(OUT/'summary.csv',summaries);write_csv(OUT/'predictions.csv',predictions)
    write_json(OUT/'complete.json',{'predictions':len(predictions),'choices_sha256':sha(OUT/'all_choices_before_test.json'),
                                  'protocol_sha256':sha(OUT/'protocol.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate'])
    train() if parser.parse_args().stage=='train' else evaluate()
