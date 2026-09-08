#!/usr/bin/env python3
"""Evaluate saved Poincare-H2 samples against the saved target distribution."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("GEOMSTATS_BACKEND", "jax")

import jax
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate

from evaluate_so3_models import load_run_config, resolve_generated_samples, resolve_run_seeds
from generation_metrics import (
    distribution_summary,
    geodesic_rbf_mmd,
    nearest_neighbor_distances,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-samples", type=int, default=16384)
    parser.add_argument("--reference-sample-seed", type=int, default=None)
    parser.add_argument("--metric-subsample", type=int, default=2000)
    parser.add_argument("--metric-seed", type=int, default=0)
    parser.add_argument("--mmd-sigma", type=float, default=1.0)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    return parser.parse_args()


def load_h2_samples(path: Path, label: str) -> np.ndarray:
    samples = np.asarray(np.load(path, allow_pickle=False))
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"{label} samples must have shape [n,2]")
    if not np.isfinite(samples).all() or np.any(np.sum(samples**2, axis=-1) >= 1.0):
        raise ValueError(f"{label} samples must be finite points inside the Poincare ball")
    return samples


def pairwise_poincare_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_sq = np.sum(left**2, axis=-1)[:, None]
    right_sq = np.sum(right**2, axis=-1)[None, :]
    diff_sq = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=-1)
    argument = 1.0 + 2.0 * diff_sq / ((1.0 - left_sq) * (1.0 - right_sq))
    return np.arccosh(np.maximum(argument, 1.0))


def generate_reference(cfg, sample_count: int, sample_seed: int):
    manifold = instantiate(cfg.manifold)
    if getattr(manifold, "coords_type", None) != "ball" or manifold.dim != 2:
        raise ValueError("saved run is not two-dimensional Poincare H2")
    dataset = instantiate(cfg.dataset, rng=jax.random.PRNGKey(sample_seed))
    dataset.batch_dims = [sample_count]
    samples = np.asarray(next(dataset)[0])
    target = {
        "component_centers": np.asarray(dataset.mean).tolist(),
        "component_scales": np.asarray(dataset.scale).tolist(),
        "component_weights": [1.0 / dataset.K] * dataset.K,
    }
    return samples, target


def save_comparison(reference: np.ndarray, generated: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    boundary = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    for axis, samples, title in zip(
        axes, (reference, generated), ("Reference", "Generated")
    ):
        axis.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=1)
        axis.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.25)
        axis.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal", title=title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_run_config(run_dir)
    training_seed, target_seed = resolve_run_seeds(cfg)
    reference_seed = (
        target_seed + 10000
        if args.reference_sample_seed is None
        else args.reference_sample_seed
    )
    generated_path = resolve_generated_samples(run_dir)
    generated = load_h2_samples(generated_path, "generated")
    reference, target = generate_reference(cfg, args.reference_samples, reference_seed)
    reference = load_h2_samples_array(reference, "reference")
    np.save(output_dir / "reference_samples.npy", reference)

    rng = np.random.default_rng(args.metric_seed)
    generated_metric = generated[
        rng.choice(len(generated), min(args.metric_subsample, len(generated)), replace=False)
    ]
    reference_metric = reference[
        rng.choice(len(reference), min(args.metric_subsample, len(reference)), replace=False)
    ]
    generated_to_reference = nearest_neighbor_distances(
        generated_metric,
        reference_metric,
        args.distance_chunk_size,
        pairwise_poincare_distances,
    )
    reference_to_generated = nearest_neighbor_distances(
        reference_metric,
        generated_metric,
        args.distance_chunk_size,
        pairwise_poincare_distances,
    )
    mmd = geodesic_rbf_mmd(
        reference_metric,
        generated_metric,
        args.mmd_sigma,
        args.distance_chunk_size,
        pairwise_poincare_distances,
    )
    save_comparison(reference, generated, output_dir / "generated_vs_reference.png")

    report = {
        "run_dir": str(run_dir),
        "generated_samples_path": str(generated_path),
        "reference_samples_path": str(output_dir / "reference_samples.npy"),
        "representation": "poincare_ball_h2",
        "distance_convention": "poincare_geodesic_distance_curvature_minus_one",
        "target_distribution": target,
        "seeds": {
            "training_seed": training_seed,
            "target_seed": target_seed,
            "reference_sample_seed": int(reference_seed),
            "metric_seed": int(args.metric_seed),
        },
        "number_generated_samples": int(len(generated)),
        "number_reference_samples": int(len(reference)),
        "metric_subsample_generated": int(len(generated_metric)),
        "metric_subsample_reference": int(len(reference_metric)),
        "rbf_mmd": float(mmd),
        "rbf_sigma": float(args.mmd_sigma),
        "generated_to_reference_nn": distribution_summary(generated_to_reference),
        "reference_to_generated_nn": distribution_summary(reference_to_generated),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    row = {
        "training_seed": training_seed,
        "target_seed": target_seed,
        "reference_sample_seed": reference_seed,
        "metric_seed": args.metric_seed,
        "number_generated_samples": len(generated),
        "number_reference_samples": len(reference),
        "rbf_mmd": mmd,
        "rbf_sigma": args.mmd_sigma,
    }
    for prefix, summary in (
        ("generated_to_reference_nn", report["generated_to_reference_nn"]),
        ("reference_to_generated_nn", report["reference_to_generated_nn"]),
    ):
        row.update({f"{prefix}_{key}": value for key, value in summary.items()})
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def load_h2_samples_array(samples: np.ndarray, label: str) -> np.ndarray:
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"{label} samples must have shape [n,2]")
    if not np.isfinite(samples).all() or np.any(np.sum(samples**2, axis=-1) >= 1.0):
        raise ValueError(f"{label} samples must be finite points inside the Poincare ball")
    return samples


if __name__ == "__main__":
    main()
