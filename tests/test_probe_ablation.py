import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import probe_ablation as runner
import probe_ablation_metrics as metrics


def fixture_config():
    return dict(mode='train', resume=False, steps=100000, seed=0, batch_size=512,
        teacher=dict(_target_='riemannian_score_sde.teachers.MalliavinTeacher',
                     divergence_mode='hutchinson', hutchinson_probes=1,
                     hutchinson_noise='rademacher', covariance_regularization=1e-6),
        dataset=dict(seed=0), manifold=dict(n=3), flow=dict(N=100),
        loss=dict(like_w=False, time_weighting=True, time_weight_lambda=0.0),
        train_plot=True, train_val=True, test_plot=True, test_val=True, test_test=True,
        eval_batch_size=2048, ckpt_dir='ckpt', logs_dir='logs', logdir='results',
        generated_samples_path='old/generated_samples.npy',
        logger=dict(csv=dict(_target_='score_sde.utils.loggers_pl.CSVLogger', save_dir='logs')),
        architecture=dict(hidden_shapes=[512]*5, act='sin'),
        scheduler=dict(warmup=100, decay=99900), ema_rate=0.999, eps=0.0002,
        beta_schedule=dict(beta_0=0.001, beta_f=5.0))


class ProbeProtocolTests(unittest.TestCase):
    def test_prepare_saved_paper_configs_freezes_sources_and_protects_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)/'repo'
            repo.mkdir()
            for folder in ('config','score_sde','riemannian_score_sde','scripts','docs','data','geomstats/geomstats'):
                (repo/folder).mkdir(parents=True)
            for file in ('main.py','run.py','geomstats/geomstats/__init__.py','docs/probe_ablation_proposals.txt'):
                (repo/file).write_text('fixture')
            paper = repo/'results/so3_complete_100k_v1'
            paper.mkdir(parents=True)
            records=[]
            for seed in range(3):
                cfg=fixture_config()
                cfg['seed']=seed
                original=repo/f'results/original_seed{seed}'
                (original/'logs/version_0').mkdir(parents=True)
                (original/'ckpt').mkdir()
                (original/'ckpt/state').write_bytes(b'untouched')
                hparams=original/'logs/version_0/hparams.yaml'
                runner.write_json(hparams,cfg)
                name=f'so3_malliavin_lambda0_seed{seed}'
                evaluation=paper/'evaluation'/name
                (evaluation/'evaluation').mkdir(parents=True)
                (evaluation/'.hydra').mkdir()
                (evaluation/'COMPLETE').write_text('OK')
                (evaluation/'generated_samples.npy').write_bytes(b'saved generated')
                (evaluation/'evaluation/reference_samples.npy').write_bytes(b'common reference')
                cfg.update(mode='test',seed=0,eval_batch_size=512,test_val=False,test_test=False)
                runner.write_json(evaluation/'.hydra/config.yaml',cfg)
                records.append(dict(name=name,source=str(hparams),source_sha256=runner.digest(hparams),
                                    checkpoint=str(original/'ckpt')))
            runner.write_json(paper/'manifest.json',dict(runs=records))
            before={str(p):runner.digest(p) for p in (repo/'results').rglob('*') if p.is_file()}
            args=SimpleNamespace(repo=repo,output=repo/'results/new_probes',suite='so3',seeds=None,check_only=False)
            with patch.object(runner,'load_config',side_effect=lambda p:json.loads(Path(p).read_text())), \
                 patch.object(runner.subprocess,'check_output',return_value='fixture-revision\n'), \
                 patch.object(runner.subprocess,'run'):
                runner.prepare(args)
            manifest=json.loads((args.output/'manifest.json').read_text())
            self.assertEqual(len(manifest['runs']),9)
            self.assertEqual(sum(not r['reuse'] for r in manifest['runs']),6)
            self.assertTrue((args.output/'source/docs/probe_ablation_proposals.txt').is_file())
            runner.check_source(args.output)
            for r in manifest['runs']: runner.protect_original(r)
            self.assertTrue(all(runner.digest(p)==v for p,v in before.items()))
            with self.assertRaises(FileExistsError): runner.prepare(args)

    def test_clone_changes_only_probes_and_output_paths(self):
        original = fixture_config()
        before = copy.deepcopy(original)
        for probes in (4, 8):
            cfg, changes = runner.training_config(original, probes, Path('/fresh/run'))
            for key in set(original)-runner.ADMIN-{'teacher'}:
                self.assertEqual(cfg[key], original[key], key)
            self.assertTrue(cfg['test_plot'])  # Affects streaming SO3 dataset state.
            self.assertTrue(cfg['train_plot'])
            self.assertEqual(cfg['eval_batch_size'], 2048)
            self.assertEqual(cfg['teacher']['hutchinson_probes'], probes)
            self.assertEqual(original, before)
            scientific = {k:v for k,v in changes.items() if k.split('.')[0] not in runner.ADMIN}
            self.assertEqual(scientific, {'teacher.hutchinson_probes': [1, probes]})

    def test_reject_wrong_baseline(self):
        cfg = fixture_config()
        runner.validate_source(cfg, 'so3', 0)
        for key, value in [('hutchinson_probes', 8), ('covariance_regularization', 1e-8),
                           ('hutchinson_noise', 'normal')]:
            wrong = copy.deepcopy(cfg)
            wrong['teacher'][key] = value
            with self.assertRaises(ValueError): runner.validate_source(wrong, 'so3', 0)
        cfg['seed'] = 1
        with self.assertRaises(ValueError): runner.validate_source(cfg, 'so3', 0)

    def test_source_and_checkpoint_changes_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'source').mkdir()
            (out/'source/a.py').write_text('original')
            runner.write_json(out/'source_fingerprints.json', {'a.py':runner.digest(out/'source/a.py')})
            runner.check_source(out)
            (out/'source/a.py').write_text('changed')
            with self.assertRaises(ValueError): runner.check_source(out)
            (out/'ckpt').mkdir()
            (out/'ckpt/state').write_bytes(b'checkpoint')
            expected = runner.fingerprint(out/'ckpt')
            (out/'ckpt/state').write_bytes(b'changed checkpoint')
            self.assertNotEqual(expected, runner.fingerprint(out/'ckpt'))

    def test_exclusive_output_and_gpu_cpu_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            runner.write_json(path, {'R':4})
            with self.assertRaises(FileExistsError): runner.write_json(path, {'R':8})
            self.assertEqual(json.loads(path.read_text()), {'R':4})
        with patch.dict('os.environ', {'JAX_PLATFORMS':'cpu', 'JAX_PLATFORM_NAME':'cpu'}):
            train = runner.environment(Path('/frozen'), False)
            self.assertEqual(train['JAX_PLATFORM_NAME'], 'gpu')
            self.assertNotIn('JAX_PLATFORMS', train)
            evaluation = runner.environment(Path('/frozen'), True)
            self.assertEqual(evaluation['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(evaluation['JAX_PLATFORMS'], 'cpu')

    def test_training_log_ambiguity_is_not_silently_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for version, loss in [(0, 2.0), (1, 3.0)]:
                p = root/f'logs/version_{version}'
                p.mkdir(parents=True)
                (p/'metrics.csv').write_text('step,train/loss,train/wall_time\n99999,%s,100\n' % loss)
            result = runner.log_metrics(root)
            self.assertTrue(result['train/loss']['ambiguous'])
            self.assertIsNone(result['train/loss']['value'])
            self.assertNotIn('train/loss_last10_mean', result)

    def test_pending_summary_never_reports_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [dict(name=f'R{r}', probes=r, training_seed=0, config_difference={},
                            checkpoint='/existing' if r==1 else '/new', config_path='/config') for r in (1,4,8)]
            runner.write_json(root/'manifest.json', dict(runs=records, seeds=[0]))
            runner.summarize(root)
            summary = json.loads((root/'summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['paired_against_R1']['8'], [])


class MetricTests(unittest.TestCase):
    def test_so3_distance_convention_and_direction(self):
        identity = np.eye(3)
        quarter_turn = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        x, y = np.array([identity, quarter_turn]), np.array([identity])
        np.testing.assert_allclose(metrics.so3_nearest(x,y), [0,np.sqrt(2)*np.pi/2])
        np.testing.assert_allclose(metrics.so3_nearest(y,x), [0])

    def test_so3_nn_exactly_matches_paper_function(self):
        # Extract these NumPy-only functions from the actual paper evaluator,
        # avoiding its JAX imports. This catches convention/dtype regressions.
        import ast
        tree = ast.parse((ROOT/'scripts/evaluate_so3_models.py').read_text())
        names = {'pairwise_rotation_angles','pairwise_so3_geodesic_distances',
                 'nearest_neighbor_geodesic_distances'}
        module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        ns = {'np':np}
        exec(compile(module, '<paper-functions>', 'exec'), ns)
        rng = np.random.default_rng(40)
        x, y = rng.normal(size=(17,3,3)).astype('float32'), rng.normal(size=(23,3,3)).astype('float32')
        for a,b in [(x,y),(y,x)]:
            np.testing.assert_array_equal(metrics.so3_nearest(a,b,7), ns['nearest_neighbor_geodesic_distances'](a,b,7))

    def test_metrics_reject_wrong_sample_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'few.npy'
            np.save(path, np.repeat(np.eye(3)[None], 2000, axis=0))
            with self.assertRaises(ValueError): metrics.so3_metrics(path,path)


if __name__ == '__main__': unittest.main()
