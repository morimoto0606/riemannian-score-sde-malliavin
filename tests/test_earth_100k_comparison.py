"""No JAX, training, or environment installation required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('l5', ROOT/'scripts/earth_100k_comparison.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LaunchGuards(unittest.TestCase):
    def fixture(self, root):
        name = 'earthquake_spectrum_lambda5_seed0'
        dest = root/name
        (dest/'input_config').mkdir(parents=True)
        config = dest/'input_config/config.yaml'
        config.write_text('mode: train\n')
        record = dict(name=name, x64_required=True, config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
        (root/'manifest.json').write_text(json.dumps(dict(repo=str(ROOT), runs=[record]*72)))
        return dest, config

    def test_checkpoint_and_changed_config_block_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dest, config=self.fixture(root)
            with patch.object(module.subprocess, 'run') as run:
                config.write_text('modified')
                with self.assertRaises(AssertionError): module.run_one(root, 0)
                config.write_text('mode: train\n')
                (dest/'ckpt').mkdir()
                with self.assertRaises(ValueError): module.run_one(root, 0)
                run.assert_not_called()

    def test_spectrum_x64_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            with patch.object(module.subprocess, 'run') as run:
                run.return_value.returncode=0
                module.run_one(root, 0)
                self.assertEqual(run.call_args.kwargs['env']['JAX_ENABLE_X64'], 'true')
                with self.assertRaises(FileExistsError): module.run_one(root, 0)
                self.assertEqual(run.call_count, 1)

    def test_all_conditions_and_schedule(self):
        conditions = list(module.configurations())
        self.assertEqual(len(conditions), 72)
        self.assertEqual(len(set(conditions)), 72)
        for dataset, method, weight, seed in conditions:
            teacher = {'malliavin': 'MalliavinTeacher', 'varadhan': 'VaradhanTeacher',
                       'spectrum': 'SpectrumTeacher', 'ism': 'unused'}[method]
            raw = dict(dataset={'name': dataset}, seed=seed, steps=600000,
                       loss={'_target_': 'x.get_ism_loss_fn' if method == 'ism' else 'x.get_dsm_loss_fn',
                             'like_w': method == 'ism'}, teacher={'_target_': 'x.'+teacher},
                       batch_size=512, architecture={'hidden_shapes': [512,512]},
                       splits=[.8,.1,.1], train_val=True)
            cfg = module.configure(raw, dataset, method, weight, seed, Path('/tmp/new'))
            self.assertEqual(cfg['steps'], 100000)
            self.assertFalse(cfg['resume'])
            self.assertEqual(cfg['scheduler']['schedules'][1]['decay_steps'], 99000)
            self.assertEqual(cfg['loss']['time_weighting'], weight == 5)
            self.assertEqual(cfg['loss']['time_weight_lambda'], weight)
            self.assertEqual(cfg['loss']['like_w'], method == 'ism')
            self.assertEqual(cfg['architecture'], raw['architecture'])
            self.assertEqual(cfg['splits'], raw['splits'])
            self.assertEqual(raw['steps'], 600000)
            self.assertNotIn('time_weight_lambda', raw['loss'])
            if method == 'spectrum':
                self.assertTrue(cfg['enable_x64'])
            smoke = module.configure(raw, dataset, method, weight, seed, Path('/tmp/smoke'), True)
            self.assertEqual(smoke['steps'], 1)
            self.assertFalse(smoke['train_val'])
        self.assertEqual(conditions[18], ('earthquake', 'spectrum', 0, 0))
        self.assertEqual(conditions[21], ('earthquake', 'spectrum', 5, 0))
        job=(ROOT/'job/earth_100k_comparison.q').read_text()
        self.assertIn('#PBS -J 0-71', job)

    def test_bad_source_rejected(self):
        raw = dict(dataset={'name': 'flood'}, seed=0)
        with self.assertRaises(ValueError):
            module.configure(raw, 'earthquake', 'ism', 0, 0, Path('/tmp/new'))


if __name__ == '__main__': unittest.main()
