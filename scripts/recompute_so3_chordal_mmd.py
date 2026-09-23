#!/usr/bin/env python3
"""Recompute SO(3) MMD^2 from saved arrays; never train, sample, or overwrite.

Uses exp(-||R-Q||_F^2 / 2), a Gaussian kernel on the matrix embedding.
Matches the saved evaluator's default_rng(0), generated-first/reference-next
subsampling (2000 each) and diagonal-excluded, unbiased MMD^2 estimator.
An unbiased finite-sample estimate can be negative; no clipping is applied.

Only NumPy and the Python standard library are required. Input matrices are
not projected, normalized, filtered, or replaced. Constraint errors are reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


METRIC_SEED = 0
SUBSAMPLE = 2000
SIGMA = 1.0
METRIC = "chordal_rbf_mmd2_unbiased"
METHODS = ("malliavin", "varadhan", "ism")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_revision() -> str | None:
    """Record the checkout revision when the script runs inside the repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def load_rotations(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray):
        raise ValueError(f"Expected a .npy array: {path}")
    if values.ndim == 2 and values.shape[1] == 9:
        values = values.reshape(-1, 3, 3)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError(f"Expected shape (n,3,3) or (n,9): {path}")
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError(f"Expected finite real numeric entries: {path}")
    if len(values) < SUBSAMPLE:
        raise ValueError(f"Need at least {SUBSAMPLE} saved samples: {path}")
    # Float64 arithmetic changes precision only; values are not repaired.
    return np.asarray(values, dtype=np.float64)


def constraint_errors(rotations: np.ndarray) -> dict[str, float]:
    gram = np.swapaxes(rotations, -1, -2) @ rotations
    orthogonality = np.linalg.norm(gram - np.eye(3), axis=(-2, -1))
    determinant = np.abs(np.linalg.det(rotations) - 1.0)
    return {
        "orthogonality_frobenius_max": float(orthogonality.max()),
        "absolute_determinant_error_max": float(determinant.max()),
    }


def metric_indices(
    generated_count: int, reference_count: int, count: int = SUBSAMPLE
) -> tuple[np.ndarray, np.ndarray]:
    if count < 2 or min(generated_count, reference_count) < count:
        raise ValueError("Each distribution must contain the requested >=2 samples")
    rng = np.random.default_rng(METRIC_SEED)
    generated = rng.choice(generated_count, count, replace=False)
    reference = rng.choice(reference_count, count, replace=False)
    return generated, reference


def kernel_sum(
    left: np.ndarray,
    right: np.ndarray,
    block_size: int,
    *,
    exclude_self: bool = False,
) -> float:
    """Sum the kernel in O(block_size^2 * 9) temporary memory."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if exclude_self and left is not right:
        raise ValueError("Self-diagonal exclusion requires the same input array")
    total = 0.0
    for i in range(0, len(left), block_size):
        for j in range(0, len(right), block_size):
            difference = left[i : i + block_size, None, :] - right[
                None, j : j + block_size, :
            ]
            squared_distance = np.einsum("ijk,ijk->ij", difference, difference)
            kernel = np.exp(-squared_distance / (2.0 * SIGMA**2))
            if exclude_self and i == j:
                # Only identical sample indices are excluded; distinct duplicate
                # rotations still contribute. Cross-sample diagonals remain.
                np.fill_diagonal(kernel, 0.0)
            total += float(kernel.sum(dtype=np.float64))
    return total


def chordal_mmd2(
    generated: np.ndarray, reference: np.ndarray, block_size: int = 256
) -> float:
    if min(len(generated), len(reference)) < 2:
        raise ValueError("MMD^2 requires at least two samples per distribution")
    x = np.asarray(generated, dtype=np.float64).reshape(len(generated), -1)
    y = np.asarray(reference, dtype=np.float64).reshape(len(reference), -1)
    if x.shape[1] != 9 or y.shape[1] != 9:
        raise ValueError("Expected 3x3 matrices or flattened length-9 matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("MMD^2 inputs must be finite")
    n, m = len(x), len(y)
    result = (
        kernel_sum(x, x, block_size, exclude_self=True) / (n * (n - 1))
        + kernel_sum(y, y, block_size, exclude_self=True) / (m * (m - 1))
        - 2.0 * kernel_sum(x, y, block_size) / (n * m)
    )
    if not np.isfinite(result):
        raise ValueError("Non-finite MMD^2 result")
    return float(result)


def conditions():
    for method in METHODS:
        for weight in (0, 5):
            for seed in range(3):
                yield method, weight, seed, f"so3_{method}_lambda{weight}_seed{seed}"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Saved so3_complete_100k_v1 directory (read only).")
    parser.add_argument("--output", type=Path,
                        help="Fresh output directory; default ROOT/chordal_mmd_sigma1_seed0.")
    parser.add_argument("--block-size", type=int, default=256,
                        help="Pairwise block size; does not change sample selection (256).")
    args = parser.parse_args(argv)
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    root = args.root.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else
              root / "chordal_mmd_sigma1_seed0")
    if not root.is_dir():
        parser.error(f"Input directory not found: {root}")
    if output.exists():
        parser.error(f"Output already exists; choose a fresh directory: {output}")

    # Check all 36 inputs before doing any calculation or writing output.
    inputs = []
    for method, weight, seed, name in conditions():
        run = root / "evaluation" / name
        paths = (run / "generated_samples.npy",
                 run / "evaluation" / "reference_samples.npy")
        for path in paths:
            if not path.is_file():
                parser.error(f"Missing saved samples: {path}")
        inputs.append((method, weight, seed, name, paths))

    rows, provenance, indices = [], [], {}
    for method, weight, seed, name, paths in inputs:
        generated, reference = (load_rotations(path) for path in paths)
        gi, ri = metric_indices(len(generated), len(reference))
        value = chordal_mmd2(generated[gi], reference[ri], args.block_size)
        rows.append(dict(method=method, lambda_value=weight, seed=seed,
                         metric=METRIC, value=value))
        constraints = dict(generated=constraint_errors(generated),
                           reference=constraint_errors(reference))
        if any(error > 1e-3 for side in constraints.values() for error in side.values()):
            print(f"WARNING: {name}: rotation constraint error >1e-3; "
                  "reported as saved, without projection or filtering.", file=sys.stderr)
        provenance.append(dict(name=name,
            generated_path=str(paths[0]), generated_sha256=sha256(paths[0]),
            reference_path=str(paths[1]), reference_sha256=sha256(paths[1]),
            generated_count=len(generated), reference_count=len(reference),
            constraints=constraints))
        indices[name + "_generated"] = gi
        indices[name + "_reference"] = ri
        print(f"{name}: {METRIC}={value:.12g}", flush=True)

    summary = []
    for method in METHODS:
        for weight in (0, 5):
            values = [row["value"] for row in rows
                      if row["method"] == method and row["lambda_value"] == weight]
            summary.append(dict(method=method, lambda_value=weight, metric=METRIC,
                n=len(values), mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1))))
    report = dict(kernel="exp(-||R-Q||_F^2/(2*sigma^2))", sigma=SIGMA,
        estimator="unbiased MMD^2; self-diagonals excluded, all cross pairs included",
        metric_seed=METRIC_SEED, metric_subsample_per_distribution=SUBSAMPLE,
        subsampling_order="generated first, reference next; without replacement",
        block_size=args.block_size, numpy_version=np.__version__,
        git_commit=repository_revision(), script_sha256=sha256(Path(__file__)),
        input_root=str(root),
        identical_saved_reference_files=len({r["reference_sha256"] for r in provenance}) == 1,
        runs=provenance)

    # Existing evaluations are untouched; a rerun must choose a new directory.
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "comparison_per_seed.csv", rows)
    write_csv(output / "comparison_summary.csv", summary)
    with (output / "provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "subsample_indices.npz").open("xb") as handle:
        np.savez_compressed(handle, **indices)
    print(f"Saved 18 runs and 6 three-seed summaries to {output}")


if __name__ == "__main__":
    main()
