import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('eeg_filter',Path(__file__).resolve().parents[1]/'scripts/filter_spd_eeg_dataset.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FilterTests(unittest.TestCase):
    def test_boundary_mapping_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            src,out=Path(d)/'raw.npz',Path(d)/'filtered.npz'
            x=np.repeat(np.eye(2)[None],7,axis=0);x[1]*=10000
            np.savez(src,covariances=x,labels=[0,0,1,0,1,0,1],subjects=['a']*3+['b']*2+['c']*2,train_indices=[0,1,2],val_indices=[3,4],test_indices=[5,6],metadata_json=json.dumps({}))
            before=src.read_bytes();r=m.filter_dataset(src,out)
            self.assertEqual(src.read_bytes(),before)
            with np.load(out) as z:
                np.testing.assert_array_equal(z['source_indices'],[0,2,3,4,5,6])
                np.testing.assert_array_equal(z['train_indices'],[0,1])
                np.testing.assert_array_equal(z['covariances'],x[[0,2,3,4,5,6]])
            self.assertEqual(r['outlier_filter']['removed'][0]['source_index'],1)
            with self.assertRaises(FileExistsError):m.filter_dataset(src,out)
