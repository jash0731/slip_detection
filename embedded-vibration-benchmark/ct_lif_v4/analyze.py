import csv
import json
import numpy as np
from .run import ROOT,OUT,write_json,write_csv


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def measures(counts):
    tp,tn,fp,fn=np.moveaxis(counts,-1,0)
    recall=np.divide(tp,tp+fn,out=np.full_like(tp,np.nan,dtype=float),where=tp+fn>0)
    fpr=np.divide(fp,fp+tn,out=np.full_like(fp,np.nan,dtype=float),where=fp+tn>0)
    return np.stack([(recall+1-fpr)/2,recall,fpr],axis=-1)


def main():
    rows=read(OUT/'predictions.csv')
    for row in read(ROOT/'ct_lif_v3/results/predictions.csv'):
        if (row['model'],row['policy'])==('mean_lif','balanced'):
            rows.append({**row,'model':'previous_optimized_mean'})
    for row in read(ROOT/'ct_lif_study/results/predictions.csv'):
        if row['model']=='ct_lif':rows.append({**row,'model':'original_ct'})
    for row in read(ROOT/'ct_lif_v2/results/predictions.csv'):
        if (row['model'],row['policy'])==('ct_v2','balanced'):rows.append({**row,'model':'ct_v2_balanced'})
    keys=list(dict.fromkeys(r['model'] for r in rows));mapping={k:i for i,k in enumerate(keys)}
    groups={o:sorted({int(r['group']) for r in rows if int(r['object'])==o}) for o in range(3)}
    gi={o:{g:i for i,g in enumerate(groups[o])} for o in groups}
    matrices={o:np.zeros((len(keys),2,len(groups[o]),4),dtype=int) for o in groups}
    identities={};names={}
    for r in rows:
        o,seed,g,end,y,p=(int(r[k]) for k in ['object','seed','group','end_sample','label','decision'])
        bucket=identities.setdefault((r['model'],o,seed),set());identity=(g,end,y)
        if identity in bucket:raise ValueError('Duplicate endpoint')
        bucket.add(identity);names[o]=r['held_object']
        index=0 if y==1 and p==1 else 1 if y==0 and p==0 else 2 if y==0 else 3
        matrices[o][mapping[r['model']],[7,19].index(seed),gi[o][g],index]+=1
    for key in keys:
        for o in groups:
            for seed in [7,19]:assert identities[key,o,seed]==identities[keys[0],o,seed]
    aggregate,objects=[],[]
    for i,key in enumerate(keys):
        scores=[];total=np.zeros(4,dtype=int)
        for o in groups:
            counts=matrices[o][i].sum(axis=1);total+=counts.sum(axis=0);score=measures(counts);scores.append(score)
            ba,recall,fpr=score.mean(axis=0)
            objects.append({'model':key,'held_object':names[o],'balanced_accuracy':float(ba),'slip_recall':float(recall),'false_alarm_rate':float(fpr)})
        ba,recall,fpr=np.stack(scores).mean(axis=(0,1))
        aggregate.append({'model':key,'balanced_accuracy':float(ba),'slip_recall':float(recall),'false_alarm_rate':float(fpr),
                          'pooled_accuracy':float((total[0]+total[1])/total.sum()),**{k:int(v) for k,v in zip(['tp','tn','fp','fn'],total)}})
    rng=np.random.default_rng(20261009);boot=[]
    for _ in range(2000):
        scores=[]
        for o in groups:
            weights=np.bincount(rng.integers(len(groups[o]),size=len(groups[o])),minlength=len(groups[o]))
            scores.append(measures((matrices[o]*weights[None,None,:,None]).sum(axis=2)))
        score=np.stack(scores).mean(axis=(0,2))
        if np.isfinite(score).all():boot.append(score)
    boot=np.stack(boot);intervals=[]
    for opponent in ['ct_only','mean_lif','original_ct','previous_optimized_mean','ct_v2_balanced']:
        diff=boot[:,mapping['ct_mean']]-boot[:,mapping[opponent]]
        intervals.append({'opponent':opponent,'interval_95':{key:np.quantile(diff[:,i],[.025,.975]).tolist()
                                                          for i,key in enumerate(['balanced_accuracy','slip_recall','false_alarm_rate'])}})
    frozen=json.loads((OUT/'all_choices_before_test.json').read_text())
    resources=[]
    for family in ['ct_mean','ct_only','mean_lif']:
        choices=[next(c for c in f['selected'] if c['variant']==family) for f in frozen['folds']]
        resources.append({'model':family,'parameters_mean':float(np.mean([c['parameters'] for c in choices])),
                          'parameters_min':min(c['parameters'] for c in choices),'parameters_max':max(c['parameters'] for c in choices),
                          'input_macs_mean':float(np.mean([32*c['inputs']*c['hidden'] for c in choices]))})
    write_csv(OUT/'aggregate.csv',aggregate);write_csv(OUT/'per_object.csv',objects)
    write_json(OUT/'analysis.json',{'aggregate':aggregate,'per_object':objects,'intervals':intervals,'resources':resources,
                                  'training_seconds':frozen['training_seconds'],'valid_bootstraps':len(boot),
                                  'limitation':'Exploratory reused benchmark; conditional trial intervals omit family selection/retraining/new-object uncertainty; no hardware timing.'})
    print(json.dumps({'aggregate':aggregate,'resources':resources},indent=2))


if __name__=='__main__':main()
