import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('launcher', Path(__file__).resolve().parents[1] / 'scripts/run_earth_malliavin_lambda0.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class TrainingConfigSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.configs = {}
        self.cfg = dict(mode='train', seed=1, steps=100000,
                        teacher={'_target_': 'riemannian_score_sde.teachers.MalliavinTeacher', 'hutchinson_probes': 1},
                        loss={'time_weight_lambda': 5., 'time_weighting': True},
                        dataset={'_target_': 'riemannian_score_sde.datasets.earth.Earthquake'})

    def put(self, name, cfg):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        self.configs[p] = copy.deepcopy(cfg)
        return p

    def select(self):
        return launcher.select_training_config(self.root, self.configs.__getitem__)

    def test_training_hparams_win_over_overwritten_test_config(self):
        p = self.put('logs/version_0/hparams.yaml', self.cfg)
        self.put('.hydra/config.yaml', dict(mode='test', seed=0))
        self.put('logs/version_1/hparams.yaml', dict(mode='test', seed=0))
        self.assertEqual(self.select()[0], p)
        launcher.validate_training_config(self.select()[1], 'earthquake', 1)

    def test_test_only_is_rejected(self):
        self.put('.hydra/config.yaml', dict(mode='test'))
        with self.assertRaisesRegex(ValueError, 'No mode=train'):
            self.select()

    def test_training_hydra_fallback(self):
        p = self.put('.hydra/config.yaml', self.cfg)
        self.assertEqual(self.select()[0], p)

    def test_conflicting_probes_rejected_but_resume_allowed(self):
        self.put('logs/version_0/hparams.yaml', self.cfg)
        other = copy.deepcopy(self.cfg)
        other.update(resume=True, now='later')
        p = self.put('logs/version_2/hparams.yaml', other)
        self.select()
        self.configs[p]['teacher']['hutchinson_probes'] = 8
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.select()

    def test_wrong_seed_teacher_dataset_or_steps_rejected(self):
        for key, value in [('seed', 0), ('teacher', {'_target_': 'VaradhanTeacher'}),
                           ('dataset', {'_target_': 'Flood'}), ('steps', 600000)]:
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg)
                cfg[key] = value
                with self.assertRaises(ValueError):
                    launcher.validate_training_config(cfg, 'earthquake', 1)
