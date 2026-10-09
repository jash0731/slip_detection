import json
from pathlib import Path
import numpy as np
from .core import features, load_model
from ct_lif_study.reference.ct_lif import sigmoid


class Detector:
    def __init__(self, checkpoint, threshold):
        self.model,self.scale,self.variant=load_model(checkpoint)
        self.threshold=float(threshold)
        if not np.isfinite(self.threshold) or not 0<=self.threshold<=np.nextafter(1.,2.):
            raise ValueError('Invalid threshold')

    def predict(self, raw):
        raw=np.asarray(raw)
        if raw.ndim==3 and raw.shape[-1]==54:
            raw=raw.reshape(raw.shape[0],raw.shape[1],18,3)
        probability=sigmoid(self.model.forward(features(raw,self.scale,self.variant)))
        return {'probability':probability,'decision':(probability>=self.threshold).astype(np.int8),'threshold':self.threshold}


def load_detector(held,seed=7,variant='ct_mean',results=None):
    folder=Path(results) if results else Path(__file__).resolve().parent/'results'
    frozen=json.loads((folder/'all_choices_before_test.json').read_text())
    folds=[f for f in frozen['folds'] if f['held']==held and f['seed']==seed]
    if len(folds)!=1: raise ValueError('Unknown research fold or seed')
    choices=[c for c in folds[0]['selected'] if c['variant']==variant]
    if len(choices)!=1: raise ValueError('Unknown variant')
    choice=choices[0]
    return Detector(folder/(choice['name']+'.npz'),choice['threshold'])
