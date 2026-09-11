"""Metadata checks without training, JAX, or map rendering."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/postprocess_s2_earth_data.py"
spec = importlib.util.spec_from_file_location("earthquake_postprocess", SCRIPT)
postprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(postprocess)


class RunMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "earthquake_malliavin_lambda5_seed0"
        (self.run_dir / ".hydra").mkdir(parents=True)

    def save_config(self, teacher=None, loss=None, **extra):
        cfg = {"experiment": "earthquake", "teacher": teacher,
               "loss": {"_target_": "riemannian_score_sde.losses.get_dsm_loss_fn", "n_max": 5}}
        if loss:
            cfg["loss"].update(loss)
        cfg.update(extra)
        OmegaConf.save(OmegaConf.create(cfg), self.run_dir / ".hydra/config.yaml")

    def test_teacher_targets(self):
        for target, options, expected in [
            ("HeatTeacher", {}, "heat"),
            ("SpectrumTeacher", {}, "spectrum"),
            ("VaradhanTeacher", {}, "varadhan"),
            ("MalliavinTeacher", {}, "malliavin"),
            ("MalliavinTeacher", {"divergence_mode": "hutchinson"}, "malliavin_hutchinson"),
        ]:
            with self.subTest(expected=expected):
                self.save_config({"_target_": "riemannian_score_sde.teachers." + target, **options})
                self.assertEqual(postprocess.load_run_metadata(self.run_dir)["teacher"], expected)

    def test_legacy_strings_and_dsm_fallback(self):
        for teacher, n_max, expected in [("varadhan", 5, "varadhan"),
                                         ("heat", -1, "heat"),
                                         (None, -1, "varadhan"), (None, 5, "heat")]:
            with self.subTest(teacher=teacher, n_max=n_max):
                self.save_config(teacher, {"n_max": n_max})
                self.assertEqual(postprocess.load_run_metadata(self.run_dir)["teacher"], expected)

    def test_ism_takes_priority_over_inherited_teacher(self):
        self.save_config({"_target_": "riemannian_score_sde.teachers.HeatTeacher"},
                         {"_target_": "riemannian_score_sde.losses.get_ism_loss_fn"})
        self.assertEqual(postprocess.load_run_metadata(self.run_dir)["teacher"], "ism")

    def test_weight_and_experiment_interpolations(self):
        self.save_config({"_target_": "riemannian_score_sde.teachers.MalliavinTeacher",
                          "divergence_mode": "hutchinson"},
                         {"time_weight_lambda": "${weight}", "time_weighting": True},
                         weight=5, experiment="${name}", name="earthquake_malliavin",
                         work_dir="${hydra:runtime.cwd}")
        self.assertEqual(postprocess.load_run_metadata(self.run_dir), {
            "dataset": "earthquake", "method": "malliavin_hutchinson",
            "teacher": "malliavin_hutchinson", "experiment": "earthquake_malliavin",
            "time_weighting": True, "time_weight_lambda": 5.0})

    def test_missing_or_unknown_config_never_claims_heat(self):
        with self.assertRaises(FileNotFoundError):
            postprocess.load_run_metadata(self.run_dir)
        self.save_config({"_target_": "custom.UnknownTeacher"})
        with self.assertRaises(ValueError):
            postprocess.load_run_metadata(self.run_dir)

    def test_metrics_json_uses_run_dir_even_with_separate_output(self):
        observed = np.array([[0., 0.], [35., 140.], [-30., -60.], [50., 10.]])
        data_path = Path(self.temp.name) / "quakes.csv"
        np.savetxt(data_path, observed, delimiter=",", header="a\nb\nc\nd", comments="")
        points = postprocess.latlon_to_upstream_s2(observed)
        np.save(self.run_dir / "generated_samples.npy", points)
        output = Path(self.temp.name) / "output"
        self.save_config({"_target_": "riemannian_score_sde.teachers.MalliavinTeacher",
                          "divergence_mode": "hutchinson"},
                         {"time_weight_lambda": 5.0, "time_weighting": True})
        with patch.object(postprocess, "save_scatter_outputs"), patch.object(postprocess, "save_density_comparison"):
            postprocess.main(["--run-dir", str(self.run_dir), "--output-dir", str(output),
                              "--data-path", str(data_path), "--metric-subsample", "4"])
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(metrics["teacher"], "malliavin_hutchinson")
        self.assertEqual(metrics["time_weight_lambda"], 5.0)
        self.assertEqual(metrics["experiment"], "earthquake")
        self.assertEqual(metrics["s2_rbf_mmd"], postprocess.s2_rbf_mmd(points, points, sigma=1.0, n_sub=4, seed=0))
        geodesic = postprocess.nearest_neighbor_geodesic(points, points, n_sub=4, seed=0)
        self.assertEqual(metrics["nearest_neighbor_geodesic_mean"], geodesic["mean"])


if __name__ == "__main__":
    unittest.main()
