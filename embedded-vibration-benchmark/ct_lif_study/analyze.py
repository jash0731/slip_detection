import csv
import json
from pathlib import Path
import numpy as np
from .run import ROOT, HERE, OUT, write_json, write_csv


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def quality(count):
    tp, tn, fp, fn = np.moveaxis(count, -1, 0)
    recall = np.divide(tp, tp+fn, out=np.full_like(tp, np.nan, dtype=float), where=(tp+fn)>0)
    fpr = np.divide(fp, tn+fp, out=np.full_like(fp, np.nan, dtype=float), where=(tn+fp)>0)
    return (recall + 1 - fpr) / 2


def detail(count):
    tp, tn, fp, fn = (int(i) for i in count)
    total = tp+tn+fp+fn
    return {'balanced_accuracy': float(quality(count)), 'accuracy': (tp+tn)/total,
            'slip_recall': tp/(tp+fn), 'false_alarm_rate': fp/(tn+fp),
            'f1': 2*tp/max(2*tp+fp+fn,1), 'precision': tp/(tp+fp) if tp+fp else None}


def main():
    rows = read(OUT / 'predictions.csv')
    indices = {(int(r['seed']), int(r['group']), int(r['end_sample'])): int(r['label'])
               for r in rows if r['model']=='ct_lif'}
    if len(indices) != 12018:
        raise ValueError('Unexpected evaluated endpoint count')
    sources = [('historical_fixed_lif', ROOT/'advanced/results/predictions.csv'),
               ('historical_selected_lif', ROOT/'followup/results/selected_predictions.csv')]
    for name,path in sources:
        matched = []
        for r in read(path):
            key=(int(r['seed']),int(r['group']),int(r['end_sample']))
            if r['model']!='lif' or key not in indices:
                continue
            if indices[key]!=int(r['label']):
                raise ValueError('Historical label mismatch')
            matched.append({**r,'model':name})
        if len(matched)!=len(indices):
            raise ValueError('Historical endpoints not matched exactly')
        rows.extend(matched)
    methods=list(dict.fromkeys(r['model'] for r in rows))
    groups={obj:sorted({int(r['group']) for r in rows if int(r['object'])==obj}) for obj in range(3)}
    seeds=[7,19]
    matrices={}
    for obj in range(3):
        matrix=np.zeros((len(methods),2,len(groups[obj]),4),dtype=np.int64)
        mi={name:i for i,name in enumerate(methods)}
        gi={group:i for i,group in enumerate(groups[obj])}
        seen=set()
        for r in rows:
            if int(r['object'])!=obj: continue
            key=(r['model'],int(r['seed']),int(r['group']),int(r['end_sample']))
            if key in seen: raise ValueError('Duplicate prediction')
            seen.add(key)
            y,p=int(r['label']),int(r['decision'])
            index=0 if y==1 and p==1 else 1 if y==0 and p==0 else 2 if y==0 else 3
            matrix[mi[r['model']],seeds.index(int(r['seed'])),gi[int(r['group'])],index]+=1
        matrices[obj]=matrix
    object_names={int(r['object']):r['held_object'] for r in rows}
    object_rows=[]
    aggregates=[]
    for m,name in enumerate(methods):
        fold_results=[]
        all_counts=np.zeros(4,dtype=int)
        for obj in range(3):
            counts=matrices[obj][m].sum(axis=1)
            results=[detail(c) for c in counts]
            fold_results.extend(results)
            all_counts+=counts.sum(axis=0)
            object_rows.append({'model':name,'held_object':object_names[obj],
                                **{k:float(np.mean([r[k] for r in results])) if all(r[k] is not None for r in results) else None for k in results[0]},
                                'unique_windows':int(counts[0].sum()),'eligible_trials':len(groups[obj])})
        aggregate={'model':name,**{k:float(np.mean([r[k] for r in fold_results])) if all(r[k] is not None for r in fold_results) else None for k in fold_results[0]}}
        aggregate['pooled_accuracy']=(all_counts[0]+all_counts[1])/all_counts.sum()
        aggregate.update({k:int(v) for k,v in zip(['tp','tn','fp','fn'],all_counts)})
        aggregates.append(aggregate)
    rng=np.random.default_rng(20261008)
    sampled=[]
    rejected=0
    for _ in range(2000):
        bas=[]
        for obj in range(3):
            ids=rng.integers(len(groups[obj]),size=len(groups[obj]))
            weights=np.bincount(ids,minlength=len(groups[obj]))
            counts=(matrices[obj]*weights[None,None,:,None]).sum(axis=2)
            bas.append(quality(counts))
        stacked=np.stack(bas,axis=-1)
        if not np.isfinite(stacked).all():
            rejected+=1
            continue
        sampled.append(stacked.mean(axis=(1,2)))
    samples=np.asarray(sampled)
    ct=methods.index('ct_lif')
    intervals=[]
    for m,name in enumerate(methods):
        delta=samples[:,ct]-samples[:,m]
        intervals.append({'comparison':'ct_lif minus '+name,
                          'difference':aggregates[ct]['balanced_accuracy']-aggregates[m]['balanced_accuracy'],
                          'lower_95':float(np.quantile(delta,.025)),'upper_95':float(np.quantile(delta,.975))})
    write_csv(OUT/'aggregate.csv',aggregates)
    write_csv(OUT/'per_object.csv',object_rows)
    write_json(OUT/'analysis.json',{'aggregate':aggregates,'per_object':object_rows,'paired_intervals':intervals,
                                 'bootstrap_valid':len(samples),'bootstrap_rejected':rejected,
                                 'ct_lif_ba_interval':[float(np.quantile(samples[:,ct],.025)),float(np.quantile(samples[:,ct],.975))],
                                 'limitations':'Conditional on observed objects and fitted models; not selection/retraining or new-object uncertainty; exploratory multiple comparisons.'})
    print(json.dumps({'aggregate':aggregates,'ct_lif_objects':[r for r in object_rows if r['model']=='ct_lif'],
                      'paired_intervals':intervals,'valid_bootstraps':len(samples)},indent=2))
    plot(aggregates,object_rows)


def plot(aggregates,object_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'ct_lif':'CT-LIF','ct_ann':'CT + ANN','ct_temporal':'CT + temporal ANN',
            'mean_lif28':'Mean + LIF (28)','c_only_lif':'Cancellation only','turnover_lif':'Separate turnover',
            'entropy_lif':'Entropy control','historical_fixed_lif':'Earlier fixed LIF',
            'historical_selected_lif':'Earlier selected LIF'}
    display=[r for r in aggregates if r['model'] in labels]
    fig,axs=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1.25,1]})
    values=[100*r['balanced_accuracy'] for r in display]
    colors=['#087e8b' if r['model']=='ct_lif' else '#adb7c4' if r['model'].startswith('historical') else '#547aa5' for r in display]
    axs[0].barh([labels[r['model']] for r in display],values,color=colors)
    axs[0].invert_yaxis()
    axs[0].set_xlim(0,100)
    axs[0].set_xlabel('Mean balanced accuracy (%)')
    for i,v in enumerate(values): axs[0].text(v+1,i,f'{v:.2f}%',va='center',fontsize=9)
    axs[0].set_title('Recorded-data comparisons')
    ct=[r for r in object_rows if r['model']=='ct_lif']
    x=np.arange(3)
    axs[1].bar(x-.18,[100*r['slip_recall'] for r in ct],.36,label='Slip recall',color='#087e8b')
    axs[1].bar(x+.18,[100*r['false_alarm_rate'] for r in ct],.36,label='False-alarm rate',color='#d77653')
    axs[1].set_xticks(x,['Brush','Screwdriver','Solder spool'])
    axs[1].set_ylim(0,100)
    axs[1].set_ylabel('Rate (%)')
    axs[1].set_title('CT-LIF: detection and false alarms')
    axs[1].legend(loc='upper right',fontsize=9)
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('CT-LIF v0.1 • 3 held-out objects × 2 seeds',fontweight='bold')
    fig.text(.02,.01,'Exploratory host evaluation. Earlier models use their saved predictions; training procedures differ. No ESP32 measurements.',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95])
    fig.savefig(OUT/'ct_lif_results.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
