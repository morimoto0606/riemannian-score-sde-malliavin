"""Hydra 1.1 composition checks without importing JAX or training."""
from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.types import RunMode


ROOT = Path(__file__).resolve().parents[1]


class SPDFinanceConfigTests(unittest.TestCase):
    def test_schedule_tree_and_three_objectives(self):
        for method in ("varadhan", "ism", "malliavin_hutchinson"):
            with self.subTest(method=method), initialize_config_dir(
                    config_dir=str(ROOT / "config"), job_name="spd-config-test"):
                overrides = ["experiment=spd_finance_" + method]
                cfg = compose(config_name="main", overrides=overrides)
                defaults = GlobalHydra.instance().config_loader().compute_defaults_list(
                    config_name="main", overrides=overrides, run_mode=RunMode.RUN)
                paths = [entry.config_path for entry in defaults.defaults]
                self.assertLess(paths.index("beta_schedule/linear"),
                                paths.index("beta_schedule/spd_finance"))
                self.assertEqual(sum(entry.override_key == "beta_schedule"
                                     for entry in defaults.defaults), 1)
                self.assertEqual(cfg.beta_schedule.beta_0, .01)
                self.assertEqual(cfg.beta_schedule.beta_f, 1.)
                self.assertTrue(cfg.enable_x64)
                self.assertEqual(cfg.dtype, "float64")
                self.assertEqual(cfg.dataset.dataset_seed, 0)
                self.assertEqual(cfg.flow.N, 16)
                self.assertTrue(cfg.flow._target_.endswith("SPDBrownian"))
                self.assertTrue(cfg.generator._target_.endswith("SPDGenerator"))
                if method == "ism":
                    self.assertIsNone(cfg.teacher)
                    self.assertTrue(cfg.loss._target_.endswith("get_ism_loss_fn"))
                else:
                    self.assertTrue(cfg.loss._target_.endswith("get_dsm_loss_fn"))
                    target = "VaradhanTeacher" if method == "varadhan" else "SPDMalliavinTeacher"
                    self.assertTrue(cfg.teacher._target_.endswith(target))

    def test_no_schedule_overlay_for_existing_experiments(self):
        for experiment in ("earthquake", "flood", "volcano", "so3_varadhan"):
            with self.subTest(experiment=experiment), initialize_config_dir(
                    config_dir=str(ROOT / "config"), job_name="legacy-config-test"):
                overrides = ["experiment=" + experiment]
                cfg = compose(config_name="main", overrides=overrides)
                defaults = GlobalHydra.instance().config_loader().compute_defaults_list(
                    config_name="main", overrides=overrides, run_mode=RunMode.RUN)
                self.assertNotIn("beta_schedule/spd_finance",
                                 [entry.config_path for entry in defaults.defaults])
                self.assertEqual(cfg.beta_schedule.beta_0, .001)
                self.assertEqual(cfg.beta_schedule.beta_f, 5.)


if __name__ == "__main__":
    unittest.main()
