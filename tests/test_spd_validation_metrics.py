import unittest
import numpy as np
from riemannian_score_sde.spd_validation_metrics import airm, frechet_mean, evaluate


class ValidationMetricsTests(unittest.TestCase):
    def test_airm_diagonal_matches_log_ratios(self):
        a = np.diag([1e-12, 2., 1e12])
        b = np.diag([1e12, 8., 1e-12])
        expected = np.linalg.norm(np.log(np.diag(a) / np.diag(b)))
        self.assertAlmostEqual(airm(a, b), expected, places=12)
        self.assertAlmostEqual(airm(b, a), expected, places=12)

    def test_airm_ill_conditioned_generalized_spectrum(self):
        # All input entries are exact binary fractions. Each input is SPD,
        # but their generalized spectrum spans about 18 decimal orders.
        # Direct generalized eigensolvers lose the small eigenvalues here.
        q = np.array([[1., 1., 1., 1.], [1., -1., 1., -1.],
                      [1., 1., -1., -1.], [1., -1., -1., 1.]]) / 2
        small = 2. ** -30
        diagonal_a = np.array([1., small, .5, 2 * small])
        diagonal_b = np.array([small, 1., 2 * small, .5])
        a = (q * diagonal_a) @ q.T
        b = (q * diagonal_b) @ q.T
        expected = np.linalg.norm(np.log(diagonal_a / diagonal_b))
        np.testing.assert_allclose([airm(a, b), airm(b, a)], expected,
                                   rtol=0, atol=2e-6)

    def test_airm_congruence_invariance(self):
        a=np.array([[2.,.4],[.4,1.]])
        b=np.array([[1.,.2],[.2,3.]])
        c=np.array([[2.,1.],[0.,.5]])
        self.assertAlmostEqual(airm(a,b),airm(c@a@c.T,c@b@c.T),places=10)
        self.assertAlmostEqual(airm(a,b),airm(b,a),places=10)

    def test_diagonal_frechet_is_geometric_mean(self):
        samples=np.array([np.diag([1.,4.]),np.diag([9.,16.])])
        mean, report=frechet_mean(samples)
        self.assertTrue(report['converged'])
        np.testing.assert_allclose(mean,np.diag([3.,8.]),rtol=1e-7)

    def test_identical_samples_have_zero_errors(self):
        x=np.array([[2.,.3],[.3,1.]])
        report=evaluate(np.repeat(x[None],4,axis=0),x)
        for key in ('airm_frechet_to_target','frobenius_arithmetic_to_target','sample_pairwise_airm_mean'):
            self.assertAlmostEqual(report[key],0.,places=10)
        self.assertEqual(report['distance_algorithm'], 'cholesky_factor_svd')
        self.assertEqual(report['frechet_solver']['max_iterations'], 128)
        self.assertEqual(report['frechet_solver']['tolerance'], 1e-6)

    def test_indefinite_is_not_repaired(self):
        for invalid in (np.diag([-1., 2.]), np.diag([0., 2.])):
            for a, b in ((invalid, np.eye(2)), (np.eye(2), invalid)):
                with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                    airm(a, b)

    def test_airm_nonfinite_inputs_are_rejected(self):
        for value in (np.nan, np.inf):
            invalid = np.diag([value, 1.])
            for a, b in ((invalid, np.eye(2)), (np.eye(2), invalid)):
                with self.subTest(value=value, a=a, b=b), self.assertRaises(ValueError):
                    airm(a, b)

    def test_evaluate_forwards_explicit_frechet_options(self):
        samples = np.array([[[2., .7], [.7, 1.]], [[1., .2], [.2, 3.]],
                            [[4., 0.], [0., .5]]])
        report = evaluate(samples, np.eye(2), frechet_max_iterations=1,
                          frechet_tolerance=1e-12)
        solver = report['frechet_solver']
        self.assertEqual(solver['max_iterations'], 1)
        self.assertEqual(solver['tolerance'], 1e-12)
        self.assertEqual(solver['iterations'], 1)
        self.assertEqual(solver['termination_reason'], 'iteration_limit')
        self.assertFalse(solver['converged'])
        self.assertIsNone(report['airm_frechet_to_target'])

    def test_nonconvergence_is_explicit(self):
        x=np.array([np.diag([1.,4.]),np.diag([9.,16.])])
        _, report=frechet_mean(x,max_iterations=1,tolerance=1e-30)
        self.assertFalse(report['converged'])

    def test_log_mmd_and_nn_identical_sets(self):
        from riemannian_score_sde.spd_validation_metrics import distribution_metrics
        x=np.array([np.eye(2),np.diag([2.,3.]),np.diag([4.,5.])])
        report=distribution_metrics(x,x,x)
        self.assertAlmostEqual(report['mmd2_biased'],0.,places=12)
        self.assertAlmostEqual(report['generated_to_validation_nn']['mean'],0.,places=12)
        self.assertGreater(report['bandwidth_squared'],0.)

    def test_solver_records_common_tolerance(self):
        _, report = frechet_mean(np.repeat(np.eye(2)[None], 2, axis=0))
        self.assertEqual(report['tolerance'], 1e-6)
        self.assertEqual(report['termination_reason'], 'gradient_tolerance')

    def test_recompute_preserves_source_and_samples(self):
        import importlib.util
        import tempfile
        import json
        import contextlib
        import io
        from pathlib import Path
        script = Path(__file__).resolve().parents[1]/'scripts/validate_spd_taxi.py'
        spec = importlib.util.spec_from_file_location('taxi_validation', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.array([np.eye(2), np.diag([2., 3.])])
            data = root/'data.npz'
            np.savez(data, covariances=x, train_indices=np.array([0]))
            manifest = dict(dataset=str(data), dataset_sha256=module.digest(data),
                            val_indices=[0], dataset_rows=[1])
            for method in module.METHODS:
                dest = root/method
                dest.mkdir()
                np.save(dest/'val_0000.npy', x)
                module.write_json(dest/'val_0000.json', dict(
                    val_index=0, dataset_row=1, metrics=None,
                    sample_sha256=module.digest(dest/'val_0000.npy'),
                    sampling=dict(complete=True, attempted=2, rejected=0)))
            before = {p:module.digest(p) for p in root.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                output = module.recompute_metrics(root, manifest)
            self.assertTrue(json.loads((output/'summary.json').read_text())['complete'])
            for path, checksum in before.items():
                self.assertEqual(module.digest(path), checksum)
            for method in module.METHODS:
                report = json.loads((output/method/'val_0000.json').read_text())
                self.assertEqual(report['metrics']['frechet_solver']['tolerance'], 1e-6)
                self.assertEqual(module.digest(output/method/'val_0000.npy'),
                                 module.digest(root/method/'val_0000.npy'))
            (root/module.METHODS[0]/'val_0000.npy').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Saved samples changed'):
                module.recompute_metrics(root, manifest)

    def test_frechet_congruence_with_ill_conditioned_samples(self):
        rng = np.random.RandomState(91)
        x = []
        for _ in range(20):
            q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
            x.append((q * np.exp(rng.uniform(-2, 2, 4))) @ q.T)
        x = np.array(x)
        c = np.diag([.01, .1, 1., 10.])
        original, r1 = frechet_mean(x)
        transformed, r2 = frechet_mean(c @ x @ c.T)
        self.assertTrue(r1['converged'])
        self.assertTrue(r2['converged'])
        self.assertLess(airm(transformed, c @ original @ c.T), 1e-6)
        self.assertEqual(r2['whitening'], 'cholesky')
