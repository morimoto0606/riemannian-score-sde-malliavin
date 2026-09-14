import unittest
import numpy as np
from riemannian_score_sde.taxi_data import validate


class TaxiDataTests(unittest.TestCase):
    def test_partition_and_spd_validation(self):
        x = np.tile(np.eye(10), (6,1,1))
        c = np.zeros((6,13))
        idx = dict(train=np.array([0,1]),val=np.array([2,3]),test=np.array([4,5]))
        validate(x,c,idx)
        with self.assertRaises(ValueError):
            validate(x,c,dict(idx,test=np.array([3,5])))
        x[0,0,0] = -1
        with self.assertRaises(ValueError):
            validate(x,c,idx)

    def test_reject_wrong_context_and_nonfinite(self):
        x=np.tile(np.eye(10),(3,1,1))
        idx={s:np.array([i]) for i,s in enumerate(('train','val','test'))}
        with self.assertRaises(ValueError):
            validate(x,np.zeros((3,2)),idx)
        x[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            validate(x,np.zeros((3,13)),idx)
