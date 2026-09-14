"""Server-side Hydra composition checks for continuous Taxi conditions."""
from pathlib import Path
import unittest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]


class TaxiConfigTests(unittest.TestCase):
    def test_three_objectives(self):
        for method in ('varadhan', 'ism', 'malliavin_hutchinson'):
            with self.subTest(method=method), initialize_config_dir(config_dir=str(ROOT/'config')):
                cfg = compose(config_name='main', overrides=['experiment=spd_taxi_'+method])
                self.assertEqual(cfg.manifold.n, 10)
                self.assertTrue(cfg.dataset._target_.endswith('SPDTaxiDataset'))
                self.assertTrue(cfg.flow._target_.endswith('TaxiSPDBrownian'))
                self.assertEqual(cfg.beta_schedule.beta_0, .01)
                self.assertEqual(cfg.beta_schedule.beta_f, 1.)
                self.assertEqual(cfg.generation.context_split, 'test')
                self.assertFalse(cfg.loss.time_weighting)
                self.assertFalse(cfg.generation.enabled)
                if method == 'ism':
                    self.assertIsNone(cfg.teacher)
                    self.assertTrue(cfg.loss._target_.endswith('get_ism_loss_fn'))
                else:
                    self.assertTrue(cfg.teacher._target_.endswith(
                        'VaradhanTeacher' if method == 'varadhan' else 'SPDMalliavinTeacher'))
