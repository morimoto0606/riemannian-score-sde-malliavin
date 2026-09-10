"""Dataset routing and coordinate checks without JAX or Cartopy."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "s2_postprocess", ROOT / "scripts/postprocess_s2_earth_data.py")
post = importlib.util.module_from_spec(spec)
spec.loader.exec_module(post)
from riemannian_score_sde.earth_data import EARTH_DATA, spherical_angles


class EarthDataPostprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.rows = {}
        for i, (key, definition) in enumerate(EARTH_DATA.items()):
            rows = np.array([[10.+i, 20.], [-30., 40.+i], [50., -60.+i]])
            self.rows[key] = rows
            np.savetxt(self.data / definition['file'], rows, delimiter=',',
                       header='\n'.join(['header']*definition['skip_header']), comments='')

    def make_run(self, dataset):
        run = self.root / f'{dataset}_malliavin_lambda5_seed0'
        (run / '.hydra').mkdir(parents=True)
        cfg = OmegaConf.create({
            'dataset': {'_target_': 'riemannian_score_sde.datasets.earth.' + EARTH_DATA[dataset]['class'],
                        'name': 'volcano' if dataset == 'volcanoe' else dataset,
                        'data_dir': '${data_dir}'},
            'experiment': dataset, 'work_dir': str(self.root), 'data_dir': '${work_dir}/data',
            'teacher': {'_target_': 'riemannian_score_sde.teachers.MalliavinTeacher',
                        'divergence_mode': 'hutchinson'},
            'loss': {'_target_': 'riemannian_score_sde.losses.get_dsm_loss_fn',
                     'time_weighting': True, 'time_weight_lambda': 5.0}})
        OmegaConf.save(cfg, run / '.hydra/config.yaml')
        np.save(run / 'generated_samples.npy', post.latlon_to_upstream_s2(self.rows[dataset]))
        return run

    def test_all_datasets_use_corresponding_csv_and_metadata(self):
        for dataset in EARTH_DATA:
            with self.subTest(dataset=dataset):
                run = self.make_run(dataset)
                with patch.object(post, 'save_scatter_outputs') as scatter, patch.object(post, 'save_density_comparison'):
                    post.main(['--run-dir', str(run)])
                np.testing.assert_array_equal(scatter.call_args.args[0], self.rows[dataset])
                self.assertEqual(scatter.call_args.kwargs['dataset'], dataset)
                metrics = json.loads((run / 'metrics.json').read_text())
                self.assertEqual(metrics['dataset'], dataset)
                self.assertEqual(metrics['method'], 'malliavin_hutchinson')
                self.assertEqual(metrics['teacher'], metrics['method'])
                self.assertEqual(metrics['real_count'], 3)
                self.assertEqual(Path(metrics['reference_data_path']).name, EARTH_DATA[dataset]['file'])
                points = post.latlon_to_upstream_s2(self.rows[dataset])
                self.assertEqual(metrics['s2_rbf_mmd'], post.s2_rbf_mmd(points, points))

    def test_conflicting_config_and_directory_fail(self):
        run = self.make_run('flood')
        cfg = OmegaConf.load(run / '.hydra/config.yaml')
        cfg.dataset._target_ = 'riemannian_score_sde.datasets.earth.Earthquake'
        OmegaConf.save(cfg, run / '.hydra/config.yaml')
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            post.load_run_metadata(run)

    def test_embedding_matches_training_angles_for_every_dataset(self):
        for dataset in EARTH_DATA:
            rows = post.load_latlon(self.data / EARTH_DATA[dataset]['file'], dataset)
            angles = spherical_angles(rows, np)
            theta, phi = angles.T
            training_formula = np.stack([np.sin(theta)*np.cos(phi),
                                         np.sin(theta)*np.sin(phi), np.cos(theta)], -1)
            np.testing.assert_allclose(post.latlon_to_upstream_s2(rows), training_formula, atol=1e-15)
            np.testing.assert_allclose(post.upstream_s2_to_latlon(training_formula), rows, atol=1e-12)

    def test_scatter_filenames_and_titles(self):
        for dataset in EARTH_DATA:
            plt = MagicMock()
            fig = plt.figure.return_value
            with patch.object(post, '_plot_backend', return_value=(plt, MagicMock(), MagicMock())), \
                 patch.object(post, '_add_map_axis', return_value=(MagicMock(), None)) as axis:
                post.save_scatter_outputs(self.rows[dataset], self.rows[dataset], self.root,
                                          central_lat=0, central_lon=0, dataset=dataset, method='ism')
            names = [call.args[0].name for call in fig.savefig.call_args_list]
            self.assertEqual(names[:3], [f'{dataset}_{kind}_map.png' for kind in ['real','generated','overlay']])
            titles = [call.args[2] for call in axis.call_args_list]
            self.assertIn('Observed ' + EARTH_DATA[dataset]['plural'], titles)
            self.assertTrue(any('ism' in title for title in titles))


if __name__ == '__main__':
    unittest.main()
