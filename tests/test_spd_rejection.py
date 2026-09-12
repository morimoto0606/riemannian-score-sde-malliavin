import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("spd_rejection", Path(__file__).resolve().parents[1] / "riemannian_score_sde/spd_rejection.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RejectionTests(unittest.TestCase):
    def test_mixed_batch_refills_only_invalid(self):
        batches = iter([np.array([np.eye(2), np.full((2, 2), np.nan)]), np.array([2*np.eye(2)])])
        x, r = m.collect_spd_samples(lambda n: next(batches), 2, 2, 4)
        np.testing.assert_array_equal(x, [np.eye(2), 2*np.eye(2)])
        self.assertEqual((r['attempted'], r['rejected']), (3, 1))
        self.assertTrue(r['complete'])

    def test_all_invalid_bounded(self):
        x, r = m.collect_spd_samples(lambda n: np.zeros((n, 2, 2)), 3, 2, 5)
        self.assertIsNone(x)
        self.assertEqual(r['attempted'], 5)
        self.assertTrue(r['limit_reached'])
        self.assertEqual(r['rejection_rate'], 1)

    def test_valid_batches_unchanged(self):
        sizes = []
        def draw(n):
            sizes.append(n)
            return np.repeat(np.eye(2)[None], n, axis=0)
        x, r = m.collect_spd_samples(draw, 5, 2, 10)
        self.assertEqual(sizes, [2, 2, 1])
        self.assertEqual(len(x), 5)
        self.assertEqual(r['rejected'], 0)

    def test_shape_errors_not_swallowed(self):
        with self.assertRaises(ValueError):
            m.collect_spd_samples(lambda n: np.ones((n, 4)), 2, 2, 4)

    def test_sampler_errors_not_swallowed(self):
        def draw(n):
            raise RuntimeError('device failure')
        with self.assertRaises(RuntimeError):
            m.collect_spd_samples(draw, 2, 2, 4)

    def test_invalid_classes(self):
        self.assertEqual(m.rejection_reason(np.array([[1., 1.], [0., 1.]])), 'asymmetric')
        self.assertEqual(m.rejection_reason(np.diag([-1., 1.])), 'not_positive_definite')
        self.assertEqual(m.rejection_reason(np.eye(5)*1e100), 'numerical_range')
