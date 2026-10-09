import unittest
import numpy as np
from .core import encode, select_operating_points, object_metrics


def raw_from_diff(d):
    return np.concatenate([np.zeros((len(d),1,18,3)),np.cumsum(d,axis=1)],axis=1)


class RevisionTests(unittest.TestCase):
    def test_amplitude_retained(self):
        raw=raw_from_diff(np.full((1,4,18,3),.1))
        a=encode(raw,np.ones(3),window=4)
        b=encode(raw*2,np.ones(3),window=4)
        np.testing.assert_allclose(b,2*a)

    def test_opposed_motion_and_suppression(self):
        d=np.ones((1,4,18,3))
        d[:,:,9:]*=-1
        raw=raw_from_diff(d)
        x=encode(raw,np.ones(3),window=4)
        np.testing.assert_allclose(x[:,:,:6],0)
        np.testing.assert_allclose(x[:,:,6:],1)
        np.testing.assert_allclose(encode(raw,np.ones(3),'shrink',4),0)

    def test_coherent_motion_matches_mean(self):
        d=np.full((1,4,18,3),.2)
        d[:,:,:,1]*=-1
        raw=raw_from_diff(d)
        a=encode(raw,np.ones(3),'amplitude',4)
        m=encode(raw,np.ones(3),'mean',4)
        s=encode(raw,np.ones(3),'shrink',4)
        np.testing.assert_allclose(a[:,:,:6],m)
        np.testing.assert_allclose(s,m)
        np.testing.assert_allclose(a[:,:,6:],0,atol=1e-15)

    def test_gain_augmentation_matches_raw_gain(self):
        rng=np.random.default_rng(1)
        raw=raw_from_diff(rng.normal(size=(2,8,18,3)))
        for variant in ['mean','amplitude','shrink']:
            rawgain=encode(raw*.8,np.ones(3),variant,8)
            featuregain=np.minimum(.8*encode(raw,np.ones(3),variant,8,clip=False),1)
            np.testing.assert_allclose(rawgain,featuregain,atol=1e-14)

    def test_causal_frame_features_and_reset(self):
        rng=np.random.default_rng(2)
        raw=raw_from_diff(rng.normal(size=(2,8,18,3)))
        a=encode(raw,np.ones(3),window=8)
        np.testing.assert_allclose(encode(raw[:,:5],np.ones(3),window=4),a[:,:4])
        np.testing.assert_allclose(encode(raw[:1],np.ones(3),window=8),a[:1])

    def test_threshold_caps_each_object(self):
        y=np.tile(np.r_[np.zeros(100),np.ones(50)],2)
        obj=np.repeat([0,1],150)
        p=np.r_[np.linspace(.01,.8,100),np.linspace(.3,.99,50),np.linspace(.02,.9,100),np.linspace(.4,.98,50)]
        points=select_operating_points(y,p,obj)
        for policy,cap in [('fpr10',.1),('fpr20',.2)]:
            rows=object_metrics(y,p,obj,points[policy]['threshold'])
            self.assertTrue(all(r['false_alarm_rate']<=cap+1e-12 for r in rows))
            self.assertGreater(points[policy]['macro_recall'],0)

    def test_all_negative_policy_identified(self):
        y=np.tile([0.,1.],4)
        obj=np.repeat([0,1],4)
        p=np.ones(8)
        points=select_operating_points(y,p,obj)
        self.assertFalse(points['fpr10']['useful_validation_recall'])
        self.assertEqual(points['fpr10']['macro_recall'],0)
        self.assertEqual(points['fpr10']['maximum_fpr'],0)

    def test_missing_validation_class_disclosed(self):
        points=select_operating_points(np.array([0.,0.,1.,1.]),np.array([.1,.2,.8,.9]),np.array([0,0,1,1]))
        self.assertEqual(points['fpr10']['objects_without_non_slip_validation'],[1])
        self.assertIsNone(points['fpr10']['per_object'][1]['false_alarm_rate'])


if __name__=='__main__':
    unittest.main(verbosity=2)
