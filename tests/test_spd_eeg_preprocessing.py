import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('builder',Path(__file__).resolve().parents[1]/'scripts/build_spd_eeg_dataset.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class EEGTests(unittest.TestCase):
    def test_subject_disjoint_and_spd(self):
        x=np.random.RandomState(4).normal(size=(40,6,30))
        labels=np.tile(['feet','right_hand'],20)
        subjects=np.repeat(np.arange(10),4)
        d,m=b.prepare(x,labels,subjects)
        sets=[set(d['subjects'][d[s+'_indices']]) for s in ('train','val','test')]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertGreater(np.linalg.eigvalsh(d['covariances']).min(),0)
        again,_=b.prepare(x,labels,subjects)
        np.testing.assert_array_equal(d['train_indices'],again['train_indices'])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            b.prepare(np.full((10,6,30),np.nan),np.tile([0,1],5),np.arange(10))
