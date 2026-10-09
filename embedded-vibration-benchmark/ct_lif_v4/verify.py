import csv
import json
import numpy as np
from .run import ROOT,OUT,sha,write_json,load_data,windows,verify_splits
from .inference import load_detector


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    frozen=json.loads((OUT/'all_choices_before_test.json').read_text())
    complete=json.loads((OUT/'complete.json').read_text())
    for p,digest in protocol['source_hashes'].items():assert sha(ROOT/p)==digest
    for p,digest in frozen['checkpoint_hashes'].items():assert sha(OUT/p)==digest
    assert complete['choices_sha256']==sha(OUT/'all_choices_before_test.json')
    assert complete['protocol_sha256']==sha(OUT/'protocol.json')
    assert frozen['fits']==len(frozen['checkpoint_hashes'])==144
    with (OUT/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
    data,metadata=load_data()
    for fold in metadata['folds']:verify_splits(data,fold)
    endpoints=set();checked=0;counts={}
    for fold in frozen['folds']:
        raw,y,records=windows(data,fold['splits']['test'])
        keys=[(r['group'],r['end_sample']) for r in records];endpoints.update(keys)
        for c in fold['selected']:
            saved=[r for r in rows if (r['model'],int(r['held']),int(r['seed']))==(c['variant'],fold['held'],fold['seed'])]
            assert [(int(r['group']),int(r['end_sample'])) for r in saved]==keys
            np.testing.assert_array_equal(y,[int(r['label']) for r in saved])
            result=load_detector(fold['held'],fold['seed'],c['variant']).predict(raw.reshape(len(raw),34,54))
            np.testing.assert_allclose(result['probability'],[float(r['probability']) for r in saved],atol=1e-12,rtol=1e-12)
            np.testing.assert_array_equal(result['decision'],[int(r['decision']) for r in saved])
            pred=result['decision']
            count=[int(((y==a)&(pred==b)).sum()) for a,b in [(1,1),(0,0),(0,1),(1,0)]]
            counts[c['variant']]=counts.get(c['variant'],np.zeros(4,dtype=int))+count;checked+=len(saved)
    analysis=json.loads((OUT/'analysis.json').read_text())
    for key,count in counts.items():
        row=next(r for r in analysis['aggregate'] if r['model']==key)
        np.testing.assert_array_equal(count,[row[k] for k in ['tp','tn','fp','fn']])
        assert count.sum()==12018
    assert len(endpoints)==6009 and len({g for g,e in endpoints})==87
    result={'checkpoint_and_source_hashes_match':True,'checkpoints':144,'inference_decisions_verified':checked,
            'unique_endpoints':len(endpoints),'eligible_trials':87,'original_splits_and_confusion_counts_match':True}
    write_json(OUT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
