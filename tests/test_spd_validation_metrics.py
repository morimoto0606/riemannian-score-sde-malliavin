import unittest
import numpy as np
from riemannian_score_sde.spd_validation_metrics import airm, frechet_mean, evaluate


class ValidationMetricsTests(unittest.TestCase):
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

    def test_indefinite_is_not_repaired(self):
        with self.assertRaises(ValueError):
            airm(np.diag([-1.,2.]),np.eye(2))

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
