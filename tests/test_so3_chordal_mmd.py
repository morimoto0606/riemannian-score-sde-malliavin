"""Saved-sample SO(3) MMD regression checks (NumPy and stdlib only)."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "recompute_so3_chordal_mmd.py"
spec = importlib.util.spec_from_file_location("so3_chordal_mmd", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def rotations(angles):
    """Exact SO(3) test data: rotations about the third axis."""
    angles = np.asarray(angles, dtype=np.float64)
    result = np.zeros((len(angles), 3, 3))
    result[:, 0, 0] = result[:, 1, 1] = np.cos(angles)
    result[:, 1, 0] = np.sin(angles)
    result[:, 0, 1] = -np.sin(angles)
    result[:, 2, 2] = 1.0
    return result


def direct_mmd2(x, y):
    """Independent scalar-pair reference, including duplicate observations."""
    def kernel(a, b):
        return np.exp(-np.sum((a - b) ** 2) / 2.0)

    n, m = len(x), len(y)
    xx = sum(kernel(x[i], x[j]) for i in range(n) for j in range(n) if i != j)
    yy = sum(kernel(y[i], y[j]) for i in range(m) for j in range(m) if i != j)
    xy = sum(kernel(a, b) for a in x for b in y)
    return xx / (n * (n - 1)) + yy / (m * (m - 1)) - 2 * xy / (n * m)


class ChordalMMDTests(unittest.TestCase):
    def test_blocked_estimator_matches_direct_unequal_samples_and_duplicates(self):
        x = rotations([0, 0, 0.6, 1.4, 2.7])
        y = rotations([0, 0.3, 0.7, 1.3, 1.9, 2.3, 2.9])
        expected = direct_mmd2(x, y)
        for block_size in (1, 2, 4, 20):
            with self.subTest(block_size=block_size):
                self.assertAlmostEqual(mod.chordal_mmd2(x, y, block_size), expected, places=13)
                self.assertAlmostEqual(
                    mod.chordal_mmd2(x.reshape(-1, 9), y.reshape(-1, 9), block_size),
                    expected, places=13,
                )

    def test_negative_unbiased_estimate_is_not_clipped(self):
        x = rotations([0, 0.8, 2.0])
        expected = direct_mmd2(x, x)
        self.assertLess(expected, 0.0)
        self.assertAlmostEqual(mod.chordal_mmd2(x, x, 2), expected, places=13)

    def test_subsampling_matches_original_generated_then_reference_order(self):
        # Reproduce the original evaluate_so3_generation.py RNG calls, not
        # a separate RNG for each distribution or reference-first ordering.
        rng = np.random.default_rng(0)
        expected_g = rng.choice(16384, 2000, replace=False)
        expected_r = rng.choice(16384, 2000, replace=False)
        gi, ri = mod.metric_indices(16384, 16384)
        np.testing.assert_array_equal(gi, expected_g)
        np.testing.assert_array_equal(ri, expected_r)
        self.assertFalse(np.array_equal(gi, ri))

    def test_invalid_shapes_nonfinite_values_and_insufficient_samples_fail(self):
        x = rotations([0, 1, 2])
        with self.assertRaises(ValueError):
            mod.chordal_mmd2(x[:1], x)
        with self.assertRaises(ValueError):
            mod.chordal_mmd2(np.zeros((3, 8)), x)
        damaged = x.copy()
        damaged[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            mod.chordal_mmd2(damaged, x)
        with self.assertRaises(ValueError):
            mod.metric_indices(3, 5, count=4)


class SavedInputTests(unittest.TestCase):
    def make_inputs(self, root):
        paths = []
        for method, weight, seed, name in mod.conditions():
            run = root / "evaluation" / name
            (run / "evaluation").mkdir(parents=True)
            offset = {"malliavin": 0, "varadhan": 0.15, "ism": 0.35}[method]
            generated = rotations(np.linspace(0, 2.0, 7) + offset + weight * .01 + seed * .03)
            reference = rotations(np.linspace(0.1, 2.5, 8))
            for path, array in ((run / "generated_samples.npy", generated),
                                (run / "evaluation" / "reference_samples.npy", reference)):
                np.save(path, array)
                paths.append(path)
            # Legacy evaluation artifacts must remain unchanged as well.
            old = run / "evaluation" / "metrics.json"
            old.write_text('{"rbf_mmd": 0.123}\n', encoding="utf-8")
            paths.append(old)
        return paths

    def run_small(self, root, output):
        original_indices = mod.metric_indices
        # Reduce only the fixture sample count; exercise real loading, MMD,
        # CSV/JSON/NPZ writing, hashing, and all 18-condition aggregation.
        with patch.object(mod, "SUBSAMPLE", 4), patch.object(
            mod, "metric_indices", side_effect=lambda n, m: original_indices(n, m, 4)
        ), contextlib.redirect_stdout(io.StringIO()):
            mod.main(["--root", str(root), "--output", str(output), "--block-size", "3"])

    def test_all_runs_aggregate_sample_sd_and_preserve_saved_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "saved"
            output = Path(tmp) / "new-evaluation"
            paths = self.make_inputs(root)
            before = {p: p.read_bytes() for p in paths}
            self.run_small(root, output)
            self.assertEqual(before, {p: p.read_bytes() for p in paths})

            with (output / "comparison_per_seed.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            with (output / "comparison_summary.csv").open() as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 18)
            self.assertEqual(len(summary), 6)
            for aggregate in summary:
                values = [float(row["value"]) for row in rows
                          if (row["method"], row["lambda_value"]) ==
                             (aggregate["method"], aggregate["lambda_value"])]
                self.assertEqual(aggregate["n"], "3")
                self.assertAlmostEqual(float(aggregate["mean"]), np.mean(values), places=14)
                self.assertAlmostEqual(float(aggregate["std"]), np.std(values, ddof=1), places=14)

            report = json.loads((output / "provenance.json").read_text())
            self.assertEqual(len(report["runs"]), 18)
            self.assertTrue(report["identical_saved_reference_files"])
            saved_indices = np.load(output / "subsample_indices.npz", allow_pickle=False)
            self.addCleanup(saved_indices.close)
            self.assertEqual(len(saved_indices.files), 36)
            for run in report["runs"]:
                self.assertEqual(run["generated_sha256"], mod.sha256(Path(run["generated_path"])))
                self.assertEqual(run["reference_sha256"], mod.sha256(Path(run["reference_path"])))
                gi = saved_indices[run["name"] + "_generated"]
                ri = saved_indices[run["name"] + "_reference"]
                generated = np.load(run["generated_path"], allow_pickle=False)
                reference = np.load(run["reference_path"], allow_pickle=False)
                expected = direct_mmd2(generated[gi], reference[ri])
                row = next(row for row in rows if run["name"] ==
                           "so3_{}_lambda{}_seed{}".format(row["method"], row["lambda_value"], row["seed"]))
                self.assertAlmostEqual(float(row["value"]), expected, places=13)

            # A rerun never silently overwrites the successful evaluation.
            output_before = {p: p.read_bytes() for p in output.iterdir()}
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_small(root, output)
            self.assertEqual(output_before, {p: p.read_bytes() for p in output.iterdir()})

    def test_missing_last_input_fails_before_loading_or_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "saved"
            output = Path(tmp) / "new-evaluation"
            self.make_inputs(root)
            last_name = list(mod.conditions())[-1][-1]
            (root / "evaluation" / last_name / "evaluation" / "reference_samples.npy").unlink()
            with patch.object(mod, "load_rotations") as load, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mod.main(["--root", str(root), "--output", str(output)])
            load.assert_not_called()
            self.assertFalse(output.exists())

    def test_invalid_saved_array_fails_without_repair_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "saved"
            output = Path(tmp) / "new-evaluation"
            self.make_inputs(root)
            name = list(mod.conditions())[2][-1]
            path = root / "evaluation" / name / "generated_samples.npy"
            damaged = rotations([0, 1, 2, 3, 4, 5, 6])
            damaged[0, 0, 0] = np.inf
            np.save(path, damaged)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                self.run_small(root, output)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
