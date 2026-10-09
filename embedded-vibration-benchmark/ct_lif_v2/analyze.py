import csv
import json
import numpy as np
from .run import ROOT, OUT, write_json, write_csv


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def measures(c):
    tp,tn,fp,fn=np.moveaxis(c,-1,0)
    recall=np.divide(tp,tp+fn,out=np.full_like(tp,np.nan,dtype=float),where=(tp+fn)>0)
    fpr=np.divide(fp,fp+tn,out=np.full_like(fp,np.nan,dtype=float),where=(fp+tn)>0)
    return np.stack([(recall+1-fpr)/2,recall,fpr],axis=-1)


def main():
    rows=read(OUT/'predictions.csv')
    for r in read(ROOT/'ct_lif_study/results/predictions.csv'):
        if r['model'] in ['ct_lif','mean_lif28','c_only_lif']:
            rows.append({**r,'model':'previous_'+r['model'],'policy':'original'})
    keys=list(dict.fromkeys((r['model'],r['policy']) for r in rows))
    maps={key:i for i,key in enumerate(keys)}
    groups={obj:sorted({int(r['group']) for r in rows if int(r['object'])==obj}) for obj in range(3)}
    matrices={o:np.zeros((len(keys),2,len(groups[o]),4),dtype=np.int64) for o in range(3)}
    gi={o:{g:i for i,g in enumerate(groups[o])} for o in groups}
    names={int(r['object']):r['held_object'] for r in rows}
    seen=set()
    for r in rows:
        key=(r['model'],r['policy'])
        object_id=int(r['object'])
        unique=(*key,int(r['seed']),int(r['group']),int(r['end_sample']))
        if unique in seen: raise ValueError('Duplicate evaluation')
        seen.add(unique)
        y,p=int(r['label']),int(r['decision'])
        c=0 if y==1 and p==1 else 1 if y==0 and p==0 else 2 if y==0 else 3
        matrices[object_id][maps[key],[7,19].index(int(r['seed'])),gi[object_id][int(r['group'])],c]+=1
    aggregate,per_object=[],[]
    for i,(model,policy) in enumerate(keys):
        scores=[]
        full=np.zeros(4,dtype=np.int64)
        for o in range(3):
            counts=matrices[o][i].sum(axis=1)
            if not np.all(counts.sum(axis=1)==matrices[o][0].sum(axis=(1,2))):
                raise ValueError('Endpoint totals differ')
            fold=measures(counts)
            scores.append(fold)
            full+=counts.sum(axis=0)
            b,r,f=fold.mean(axis=0)
            per_object.append({'model':model,'policy':policy,'held_object':names[o],
                               'balanced_accuracy':float(b),'slip_recall':float(r),'false_alarm_rate':float(f)})
        b,r,f=np.stack(scores).mean(axis=(0,1))
        aggregate.append({'model':model,'policy':policy,'balanced_accuracy':float(b),'slip_recall':float(r),
                          'false_alarm_rate':float(f),'pooled_accuracy':float((full[0]+full[1])/full.sum()),
                          **{k:int(v) for k,v in zip(['tp','tn','fp','fn'],full)}})
    rng=np.random.default_rng(20261009)
    boots=[]
    for _ in range(2000):
        scores=[]
        for o in range(3):
            weights=np.bincount(rng.integers(len(groups[o]),size=len(groups[o])),minlength=len(groups[o]))
            counts=(matrices[o]*weights[None,None,:,None]).sum(axis=2)
            scores.append(measures(counts))
        item=np.stack(scores).mean(axis=(0,2))
        if np.isfinite(item).all(): boots.append(item)
    boots=np.stack(boots)
    intervals=[]
    for policy in ['balanced','fpr20','fpr10']:
        ci=maps['ct_v2',policy]
        for opponent,op in [('previous_ct_lif','original'),('previous_mean_lif28','original'),
                            ('mean_lif',policy),('amplitude_temporal',policy),('previous_ct_lif',policy)]:
            oi=maps[opponent,op]
            difference=boots[:,ci]-boots[:,oi]
            intervals.append({'candidate_policy':policy,'opponent':opponent,'opponent_policy':op,
                              'ba_difference':aggregate[ci]['balanced_accuracy']-aggregate[oi]['balanced_accuracy'],
                              'ba_interval':np.quantile(difference[:,0],[.025,.975]).tolist(),
                              'recall_interval':np.quantile(difference[:,1],[.025,.975]).tolist(),
                              'fpr_interval':np.quantile(difference[:,2],[.025,.975]).tolist()})
    choices=json.loads((OUT/'all_choices_before_test.json').read_text())
    selection=[]
    for fold in choices['folds']:
        for c in fold['selected']:
            if c['model']=='ct_v2':
                spec=c['candidate']
                selection.append({'held_object':fold['held_object'],'seed':fold['seed'],'policy':c['policy'],
                                  'family':spec['family'],'hidden':spec['hidden'],'window':spec['window'],
                                  'parameters':spec['parameters'],'threshold':c['threshold'],
                                  'validation':spec['operating_points'][c['policy']]})
    write_csv(OUT/'aggregate.csv',aggregate)
    write_csv(OUT/'per_object.csv',per_object)
    write_json(OUT/'analysis.json',{'aggregate':aggregate,'per_object':per_object,'intervals':intervals,
                                  'valid_bootstraps':len(boots),'ct_v2_selections':selection,
                                  'training_seconds':choices['training_seconds'],
                                  'limitations':'Exploratory conditional intervals; no new-object, retraining, selection or multiple-comparison adjustment.'})
    focus=[r for r in aggregate if r['model'] in ['ct_v2','mean_lif','amplitude_temporal','previous_ct_lif','previous_mean_lif28']]
    print(json.dumps({'aggregate':focus,'ct_objects':[r for r in per_object if r['model']=='ct_v2']},indent=2))
    plot(aggregate)


def plot(aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,6))
    labels={'ct_v2':'CT-LIF v2','mean_lif':'Matched mean LIF','amplitude_temporal':'Matched temporal ANN',
            'previous_ct_lif':'CT-LIF v1, recalibrated','previous_mean_lif28':'Earlier mean LIF (28), recalibrated'}
    styles={'ct_v2':('#087e8b','o'),'mean_lif':('#745ca7','s'),'amplitude_temporal':('#d17a43','^'),
            'previous_ct_lif':('#65758b','D'),'previous_mean_lif28':('#9a6b77','v')}
    for model,label in labels.items():
        rows=[r for r in aggregate if r['model']==model and r['policy'] in ['fpr10','fpr20','balanced']]
        rows.sort(key=lambda r:r['false_alarm_rate'])
        color,marker=styles[model]
        ax.plot([100*r['false_alarm_rate'] for r in rows],[100*r['slip_recall'] for r in rows],marker=marker,color=color,label=label)
        if model=='ct_v2':
            for r in rows:
                ax.annotate(r['policy'],(100*r['false_alarm_rate'],100*r['slip_recall']),xytext=(5,6),textcoords='offset points',fontsize=9)
    original=next(r for r in aggregate if r['model']=='previous_ct_lif' and r['policy']=='original')
    ax.scatter([100*original['false_alarm_rate']],[100*original['slip_recall']],facecolors='none',edgecolors='#65758b',s=75,zorder=5)
    ax.annotate('Previous CT-LIF\noriginal threshold',(100*original['false_alarm_rate'],100*original['slip_recall']),
                xytext=(-8,12),textcoords='offset points',ha='right',fontsize=9)
    ax.set(xlim=(0,35),ylim=(0,80),xlabel='False-alarm rate on held-out objects (%)',ylabel='Actual slips detected (%)',
           title='Fewer false alarms can mean more missed slips')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9,loc='lower right')
    ax.spines[['top','right']].set_visible(False)
    fig.text(.02,.01,'All thresholds selected on validation data. Labels fpr10/fpr20 describe validation targets, not guaranteed test rates.',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,1])
    fig.savefig(OUT/'false_alarm_tradeoff.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
