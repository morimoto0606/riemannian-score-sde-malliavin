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


    def test_recompute_preserves_originals_and_incomplete_generation(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dataset=root/'data.npz'
            np.savez(dataset,covariances=np.array([np.eye(2)]))
            manifest=dict(dataset=str(dataset),dataset_sha256=final.digest(dataset),
                          test_indices=[0],dataset_rows=[0])
            for seed in range(3):
                for method in final.METHODS:
                    dest=root/f'seed{seed}'/method
                    dest.mkdir(parents=True)
                    path=dest/'test_0000.npy'
                    np.save(path,np.array([np.eye(2),2*np.eye(2)]))
                    final.write_json(path.with_suffix('.json'),dict(
                        test_index=0,dataset_row=0,sample_sha256=final.digest(path),
                        sampling=dict(complete=not(seed==1 and method=='ism')),
                        metrics={'old':True}))
            before={p:final.digest(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(final,'summarize'):
                output=final.recompute_metrics(root,manifest)
            for path,checksum in before.items():
                self.assertEqual(final.digest(path),checksum)
            for seed in range(3):
                for method in final.METHODS:
                    relative=Path(f'seed{seed}')/method/'test_0000.npy'
                    self.assertEqual(final.digest(output/relative),final.digest(root/relative))
                    report=json.loads((output/relative.with_suffix('.json')).read_text())
                    if seed==1 and method=='ism':
                        self.assertIsNone(report['metrics'])
                    else:
                        self.assertEqual(report['metrics']['frechet_solver']['solver_version'],3)
                        self.assertTrue(report['metrics']['frechet_solver']['converged'])


if __name__=='__main__':
    unittest.main()
