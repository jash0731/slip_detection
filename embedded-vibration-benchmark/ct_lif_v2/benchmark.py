import json
import csv
import subprocess
import tempfile
import platform
from pathlib import Path
import numpy as np
from .run import ROOT,HERE,OUT,write_json,write_csv
from ct_lif_study.run import load_data,windows
from ct_lif_study.reference.ct_lif import TinyModel,encode as encode_v1,sigmoid
from .core import encode


def literal(value):
    result=format(float(np.float32(value)),'.9g')
    if '.' not in result and 'e' not in result: result+='.0'
    return result+'f'


def main():
    data,_=load_data()
    raw,y,records=windows(data,list(range(90)))
    objects=np.array([r['object'] for r in records],dtype='<i4')
    frozen=json.loads((OUT/'all_choices_before_test.json').read_text())
    checkpoints={}
    uses=[]
    for fold in frozen['folds']:
        for choice in fold['selected']:
            if choice['model'] not in ['ct_v2','mean_lif','amplitude_temporal','previous_ct_lif','previous_mean_lif28','previous_c_only_lif']:
                continue
            old='old_path' in choice
            path=ROOT/choice['old_path'] if old else OUT/(choice['candidate']['name']+'.npz')
            name=str(path.relative_to(ROOT))
            checkpoints[name]={'path':path,'held':fold['held'],'old':old}
            uses.append({'model':choice['model'],'policy':choice['policy'],'held':fold['held'],'seed':fold['seed'],
                         'checkpoint':name,'threshold':choice['threshold']})
    header=['#pragma once','#include "runtime.hpp"','using Predictor=float(*)(const float*);',
            'struct ModelSpec {const char* name; int held; Predictor predict;};']
    specs=[]
    references={}
    dimensions={}
    for index,(name,info) in enumerate(checkpoints.items()):
        with np.load(info['path'],allow_pickle=False) as z:
            C,H=z['W'].shape
            variant=str(z['variant'])
            T=32 if info['old'] else int(z['window'])
            variant_id=({'ct':0,'c_only':1,'mean':2} if info['old'] else {'mean':2,'amplitude':3,'shrink':4})[variant]
            kind=str(z['kind'])
            if kind not in ['lif','temporal']: raise ValueError('Unsupported C++ model')
            kind_id=0 if kind=='lif' else 1
            for key in ['W','b','w','scale']:
                header.append(f'static const float {key}{index}[]={{'+','.join(literal(v) for v in z[key].ravel())+'};')
            header.append(f'static float predict{index}(const float* raw) {{return infer_window<{C},{H},{T},{kind_id},{variant_id}>(raw,W{index},b{index},w{index},{literal(z["c"][0])},scale{index});}}')
            specs.append(f'{{"{name}",{info["held"]},predict{index}}}')
            mask=objects==info['held']
            x=encode_v1(raw[mask],z['scale'],float(z['gate']),variant) if info['old'] else encode(raw[mask],z['scale'],variant,T)
            model=TinyModel(C,H,kind)
            model.params={k:z[k].copy() for k in model.params}
            references[name]=(np.flatnonzero(mask),model.forward(x))
            parameters=C*H+2*H+1
            dimensions[name]={'inputs':C,'hidden':H,'window':T,'variant':variant,'parameters':parameters,
                              'parameter_bytes_float32':4*parameters,'dense_input_macs_per_window':T*C*H,
                              'input_history_bytes_float32':54*(T+(2 if variant=='ct' else 1))*4,
                              'state_and_feature_payload_bytes':4*(2*H+C+(57 if variant=='ct' else 0)),
                              'effective_raw_support_intervals':T+(1 if variant=='ct' else 0)}
    header.append('static const ModelSpec models[]={'+','.join(specs)+'};')
    (HERE/'models_generated.hpp').write_text('\n'.join(header)+'\n')
    compiler=subprocess.run(['g++','--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        with (folder/'input.bin').open('wb') as f:
            np.array([len(raw)],dtype='<i4').tofile(f)
            objects.tofile(f)
            raw.astype('<f4').tofile(f)
        build=subprocess.run(['g++','-O3','-std=c++17','-ffp-contract=off',str(HERE/'benchmark.cpp'),'-o',str(folder/'bench')],capture_output=True,text=True)
        if build.returncode: raise RuntimeError(build.stderr)
        completed=subprocess.run([str(folder/'bench'),str(folder/'input.bin'),str(OUT/'cpp_predictions.csv'),str(OUT/'host_timings.csv')],check=True,capture_output=True,text=True)
        print(completed.stdout)
    cpp={name:[] for name in checkpoints}
    with (OUT/'cpp_predictions.csv').open() as f:
        for r in csv.DictReader(f): cpp[r['checkpoint']].append((int(r['index']),float(r['logit'])))
    parity={}
    for name,(indices,logits) in references.items():
        observed=cpp[name]
        np.testing.assert_array_equal(indices,[i for i,l in observed])
        values=np.array([l for i,l in observed])
        if not np.isfinite(values).all(): raise ValueError('Nonfinite C++ output')
        parity[name]={'max_absolute_logit_difference':float(np.max(np.abs(values-logits))),'windows':len(logits)}
    timing={}
    with (OUT/'host_timings.csv').open() as f:
        for r in csv.DictReader(f): timing[r['checkpoint']]={k:float(r[k]) for k in ['median_us','p95_us','samples']}
    table=[]
    for use in uses:
        name=use['checkpoint']
        reference=references[name][1]
        cpp_logit=np.array([v for _,v in cpp[name]])
        changed=int(np.sum((sigmoid(reference)>=use['threshold'])!=(sigmoid(cpp_logit)>=use['threshold'])))
        table.append({**use,**dimensions[name],**timing[name],**parity[name],'changed_decisions_float32':changed})
    write_csv(OUT/'resources_by_fold.csv',table)
    aggregates=[]
    for model,policy in sorted({(r['model'],r['policy']) for r in table}):
        group=[r for r in table if (r['model'],r['policy'])==(model,policy)]
        aggregates.append({'model':model,'policy':policy,
                           **{key:float(np.mean([r[key] for r in group])) for key in ['parameters','parameter_bytes_float32','dense_input_macs_per_window','median_us','p95_us','input_history_bytes_float32','state_and_feature_payload_bytes','effective_raw_support_intervals']},
                           'minimum_parameters':min(r['parameters'] for r in group),'maximum_parameters':max(r['parameters'] for r in group),
                           'changed_decisions_float32':sum(r['changed_decisions_float32'] for r in group),
                           'max_absolute_logit_difference':max(r['max_absolute_logit_difference'] for r in group)})
    write_csv(OUT/'resources.csv',aggregates)
    write_json(OUT/'benchmark.json',{'compiler':compiler,'flags':'-O3 -std=c++17 -ffp-contract=off','platform':platform.platform(),
                                   'description':'CPU host float32 C++ encoding + reset-window inference + linear readout; no acquisition, IO, scheduler, sigmoid or radio',
                                   'timing':'5 shuffled-order rounds, 32 warm-ups and 1024 calls per checkpoint/round, per-call steady_clock',
                                   'memory':'logical float32 payload; raw-history reference implementation, excluding stack temporaries, code, allocator, acquisition and firmware',
                                   'not_esp32_or_energy':True,'resources':aggregates})
    print(json.dumps([r for r in aggregates if r['model'] in ['ct_v2','previous_ct_lif','previous_mean_lif28']],indent=2))


if __name__=='__main__':
    main()
