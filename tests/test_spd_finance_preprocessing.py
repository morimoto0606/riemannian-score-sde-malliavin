"""Small offline tests: covariance meaning, leakage, missingness, quality rules."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_spd_finance_dataset.py"
spec = importlib.util.spec_from_file_location("spd_finance_build", SCRIPT)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class FinancePreprocessingTests(unittest.TestCase):
    def prices(self, n=700):
        rng = np.random.default_rng(4)
        returns = rng.normal(scale=0.01, size=(n - 1, 5))
        prices = np.exp(np.vstack([np.zeros(5), returns.cumsum(axis=0)]))
        return pd.DataFrame(prices, index=pd.bdate_range("2010-01-01", periods=n))

    def test_sample_covariance_and_causality(self):
        prices = self.prices()
        cov, returns, ends = pipeline.rolling_covariances(prices, 60)
        self.assertEqual(cov.shape, (640, 5, 5))
        np.testing.assert_allclose(cov[0], np.cov(returns.iloc[:60], rowvar=False, ddof=1))
        changed = prices.copy()
        changed.iloc[400:] *= 2.0
        later, _, later_ends = pipeline.rolling_covariances(changed, 60)
        np.testing.assert_array_equal(ends, later_ends)
        np.testing.assert_array_equal(cov[ends < 399], later[ends < 399])

    def test_missing_prices_do_not_create_multiday_returns(self):
        prices = self.prices()
        prices.iloc[100, 2] = np.nan
        _, returns, ends = pipeline.rolling_covariances(prices, 60)
        self.assertTrue(np.isnan(returns.iloc[99, 2]))
        self.assertTrue(np.isnan(returns.iloc[100, 2]))
        self.assertFalse(np.isin(ends, np.arange(99, 160)).any())
        self.assertIn(160, ends)

    def test_splits_share_neither_prices_nor_returns(self):
        _, _, ends = pipeline.rolling_covariances(self.prices(), 60)
        split = pipeline.chronological_indices(ends, 60)
        supports = []
        for idx in split.values():
            used_prices = set()
            for end in ends[idx]:
                used_prices.update(range(end - 60 + 1, end + 2))
            supports.append(used_prices)
        for earlier, later in zip(supports[:-1], supports[1:]):
            self.assertTrue(earlier.isdisjoint(later))
            self.assertLess(max(earlier), min(later))
        self.assertEqual(sum(map(len, split.values())), len(ends) - 120)
        with self.assertRaises(ValueError):
            pipeline.chronological_indices(ends[:100], 60)
        with self.assertRaises(ValueError):
            pipeline.chronological_indices(ends, 60, [0.8, 0.2, 0.2])

    def test_regularization_is_explicit_and_minimal(self):
        good = np.diag([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])[None]
        fixed, eps, *rest = pipeline.validate_spd(good)
        np.testing.assert_array_equal(fixed, good)
        self.assertEqual(eps[0], 0)
        singular = good.copy()
        singular[0, 0, 0] = 0
        with self.assertRaises(ValueError):
            pipeline.validate_spd(singular)
        fixed, eps, *_ = pipeline.validate_spd(singular, allow_regularization=True)
        self.assertAlmostEqual(eps[0] / 5e-18, 1)
        self.assertGreater(np.linalg.eigvalsh(fixed).min(), 0)
        bad = good.copy()
        bad[0, 0, 1] = 1
        with self.assertRaises(ValueError):
            pipeline.validate_spd(bad)
        bad = good.copy()
        bad[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            pipeline.validate_spd(bad, allow_regularization=True)

    def test_quarantine_keeps_sessions_and_does_not_rescale(self):
        index = pd.bdate_range("2026-03-27", periods=4)
        prices = pd.DataFrame({"1306.T": [375., 37., 36., 381.]}, index=index)
        rules = json.loads((SCRIPT.parents[1] / "config/dataset/spd_finance_quality.json").read_text())
        cleaned, applied = pipeline.quarantine_prices(prices, rules)
        self.assertEqual(len(applied), 2)
        self.assertEqual(len(cleaned), 4)
        self.assertEqual(cleaned["1306.T"].isna().sum(), 2)
        repaired = prices.copy()
        repaired.iloc[1:3] *= 10
        cleaned, applied = pipeline.quarantine_prices(repaired, rules)
        self.assertFalse(applied)
        pd.testing.assert_frame_equal(cleaned, repaired)
        with self.assertRaises(ValueError):
            pipeline.quarantine_prices(prices, {"max_abs_log_return": 0.5})

    def test_exclude_live_duplicate_bars_before_date_validation(self):
        times = pd.to_datetime(["2026-09-06T23:00Z", "2026-09-07T23:00Z", "2026-09-08T08:00Z"])
        chart = {"meta": {"exchangeTimezoneName": "Europe/London", "firstTradeDate": 0},
                 "timestamp": [int(t.timestamp()) for t in times],
                 "indicators": {"quote": [{"close": [150., None, 151.]}]}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "FX.json"
            path.write_text(json.dumps({"chart": {"result": [chart]}}))
            series, audit = pipeline.read_prices(path, "2026-09-01", "2026-09-08", "Close")
            self.assertEqual(len(series), 1)
            self.assertEqual(str(series.index[0].date()), "2026-09-07")
            self.assertFalse(audit["adjusted_price_available"])
            with self.assertRaises(ValueError):
                pipeline.read_prices(path, "2026-09-01", "2026-09-09", "Close")
            with self.assertRaises(ValueError):
                pipeline.read_prices(path, "2026-09-01", "2026-09-08", "Adj Close")


if __name__ == "__main__":
    unittest.main()
