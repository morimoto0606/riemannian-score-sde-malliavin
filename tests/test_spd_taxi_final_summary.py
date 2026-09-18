from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import evaluate_spd_taxi_final as final


class FinalSummaryTests(unittest.TestCase):
    def test_partial_frechet_does_not_produce_three_seed_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root/'data.npz'
            np.savez(dataset, covariances=np.ones((4,1,1)), train_indices=[0],
                     val_indices=[1], test_indices=[2,3])
            manifest = dict(dataset=str(dataset),dataset_rows=[2,3],test_indices=[0,1])
            for seed in range(3):
                for method in final.METHODS:
                    dest=root/f'seed{seed}'/method
                    dest.mkdir(parents=True)
                    for idx in range(2):
                        p=dest/f'test_{idx:04d}.npy'
                        np.save(p,np.ones((2,1,1)))
                        metrics={m:1. for m in final.METRICS}
                        if seed==2 and method=='ism' and idx==1:
                            metrics['airm_squared_frechet_to_target']=None
                        final.write_json(p.with_suffix('.json'),dict(
                            sampling=dict(complete=True,attempted=2,rejected=0),
                            metrics=metrics,sample_sha256=final.digest(p)))
            pooled=dict(generated_to_validation_nn={},validation_to_generated_nn={},
                        validation_to_train_nn={})
            with patch('riemannian_score_sde.spd_validation_metrics.distribution_metrics',
                       side_effect=lambda *args:dict(pooled)):
                result=final.summarize(root,manifest)
            self.assertFalse(result['complete'])
            ism=result['across_training_seeds']['ism']['airm_squared_frechet_to_target']
            self.assertEqual(ism['seeds_complete'],2)
            self.assertIsNone(ism['mean'])
            self.assertEqual(result['across_training_seeds']['varadhan']['airm_squared_frechet_to_target']['mean'],1.)
            self.assertIn('generated_to_test_nn',result['runs']['varadhan_seed0']['pooled_distribution'])
            (root/'seed0/varadhan/test_0000.npy').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Samples changed'):
                final.summarize(root,manifest)


if __name__=='__main__':
    unittest.main()
