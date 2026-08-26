#!/usr/bin/env python3
"""Postprocess one existing SO(3) run without resampling its trained model."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("GEOMSTATS_BACKEND", "jax")

import matplotlib.pyplot as plt
import numpy as np

from evaluate_so3_models import (
    constraint_metrics,
    generate_shared_target,
    geodesic_rbf_mmd,
    load_rotations,
    load_run_config,
    nearest_neighbor_geodesic_distances,
    resolve_generated_samples,
)
from riemannian_score_sde.utils.vis import (
    SO3_TAIT_BRYAN_LABELS,
    SO3_TAIT_BRYAN_RANGES,
    compute_so3_euler_histogram_comparison,
    plot_so3,
    plot_so3_euler_density_difference,
    plot_so3_euler_overlay,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/evaluation.",
    )
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=None,
        help="Defaults to the number of saved generated samples.",
    )
    parser.add_argument("--reference-seed", type=int, default=10000)
    parser.add_argument("--metric-subsample", type=int, default=2000)
    parser.add_argument("--metric-seed", type=int, default=0)
    parser.add_argument("--mmd-sigma", type=float, default=1.0)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument("--euler-bins", type=int, default=100)
    parser.add_argument(
        "--euler-samples",
        type=int,
        default=None,
        help="Matched samples per distribution; defaults to the smaller dataset.",
    )
    parser.add_argument("--euler-seed", type=int, default=0)
    return parser.parse_args(argv)


def _finite_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if np.isfinite(number) else None


def _training_log_summary(run_dir: Path) -> dict:
    """Select final logged values by maximum step, never by version number."""

    metric_names = ("train/loss", "val/loss", "val/logp", "test/logp")
    observations = {name: [] for name in metric_names}
    for metrics_path in sorted(run_dir.glob("logs/version_*/metrics.csv")):
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            for row_index, row in enumerate(csv.DictReader(handle), start=2):
                step = _finite_float(row.get("step"))
                if step is None:
                    continue
                for name in metric_names:
                    value = _finite_float(row.get(name))
                    if value is not None:
                        observations[name].append(
                            {
                                "step": int(step),
                                "value": value,
                                "source": str(metrics_path),
                                "row": row_index,
                            }
                        )

    summary = {}
    for name, candidates in observations.items():
        if not candidates:
            continue
        max_step = max(item["step"] for item in candidates)
        finalists = [item for item in candidates if item["step"] == max_step]
        distinct_values = {item["value"] for item in finalists}
        if len(distinct_values) == 1:
            summary[name] = {
                "step": max_step,
                "value": finalists[0]["value"],
                "source": finalists[0]["source"],
                "ambiguous": False,
            }
        else:
            summary[name] = {
                "step": max_step,
                "value": None,
                "ambiguous": True,
                "candidates": finalists,
            }
    return summary


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _save_nearest_neighbor_plot(
    generated_to_reference: np.ndarray,
    reference_to_generated: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    bins = 50
    ax.hist(
        generated_to_reference,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Generated to reference",
    )
    ax.hist(
        reference_to_generated,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Reference to generated",
    )
    ax.set_xlabel(r"SO(3) geodesic distance $||Log(R_1^T R_2)||_F$")
    ax.set_ylabel("Density")
    ax.set_title("Bidirectional nearest-neighbor distances")
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_mmd_plot(mmd: float, sigma: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), tight_layout=True)
    ax.bar(["Generated vs reference"], [mmd], color="tab:purple", alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Unbiased RBF MMD")
    ax.set_title(r"SO(3) geodesic RBF MMD ($\sigma={}$)".format(sigma))
    ax.text(0, mmd, "{:.6g}".format(mmd), ha="center", va="bottom")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _flatten_for_csv(report: dict) -> dict:
    nearest = report["nearest_neighbor_geodesic_distance"]
    coverage = report["reference_to_generated_nearest_neighbor"]
    constraints = report["constraints"]
    row = {
        "number_generated_samples": report["number_generated_samples"],
        "number_reference_samples": report["number_reference_samples"],
        "metric_subsample_generated": report["metric_subsample_generated"],
        "metric_subsample_reference": report["metric_subsample_reference"],
        "rbf_mmd": report["rbf_mmd"],
        "rbf_sigma": report["rbf_sigma"],
    }
    for key, value in nearest.items():
        row["generated_to_reference_nn_" + key] = value
    for key, value in coverage.items():
        row["reference_to_generated_nn_" + key] = value
    row["orthogonality_error_mean"] = constraints["orthogonality_frobenius"][
        "mean"
    ]
    row["orthogonality_error_max"] = constraints["orthogonality_frobenius"][
        "max"
    ]
    row["determinant_error_mean"] = constraints["absolute_determinant_error"][
        "mean"
    ]
    row["determinant_error_max"] = constraints["absolute_determinant_error"][
        "max"
    ]
    for metric_name, metric in report["training_logs"].items():
        prefix = metric_name.replace("/", "_")
        row[prefix + "_step"] = metric["step"]
        row[prefix + "_value"] = metric["value"]
        row[prefix + "_ambiguous"] = metric["ambiguous"]
    return row


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_path = resolve_generated_samples(run_dir)
    generated = load_rotations(generated_path, "generated")
    reference_count = (
        len(generated)
        if args.reference_samples is None
        else args.reference_samples
    )
    if len(generated) < 2 or reference_count < 2 or args.metric_subsample < 2:
        raise ValueError(
            "generated, reference-samples, and metric-subsample must be at least two"
        )
    if args.mmd_sigma <= 0.0 or args.distance_chunk_size < 1:
        raise ValueError("mmd-sigma and distance-chunk-size must be positive")

    cfg = load_run_config(run_dir)
    reference = generate_shared_target(cfg, reference_count, args.reference_seed)
    np.save(output_dir / "reference_samples.npy", reference)

    rng = np.random.default_rng(args.metric_seed)
    generated_indices = rng.choice(
        len(generated), min(args.metric_subsample, len(generated)), replace=False
    )
    reference_indices = rng.choice(
        len(reference), min(args.metric_subsample, len(reference)), replace=False
    )
    generated_metric = generated[generated_indices]
    reference_metric = reference[reference_indices]
    generated_to_reference = nearest_neighbor_geodesic_distances(
        generated_metric, reference_metric, args.distance_chunk_size
    )
    reference_to_generated = nearest_neighbor_geodesic_distances(
        reference_metric, generated_metric, args.distance_chunk_size
    )
    mmd = geodesic_rbf_mmd(
        reference_metric,
        generated_metric,
        args.mmd_sigma,
        args.distance_chunk_size,
    )

    generated_vs_data = plot_so3(reference, generated, size=12)
    generated_vs_data.savefig(
        output_dir / "generated_vs_data.png", dpi=180, bbox_inches="tight"
    )
    plt.close(generated_vs_data)
    euler_comparison = compute_so3_euler_histogram_comparison(
        reference,
        generated,
        bins=args.euler_bins,
        max_samples=args.euler_samples,
        seed=args.euler_seed,
    )
    euler_overlay = plot_so3_euler_overlay(euler_comparison, size=12)
    euler_overlay.savefig(
        output_dir / "euler_angle_overlay.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(euler_overlay)
    euler_difference = plot_so3_euler_density_difference(
        euler_comparison,
        size=12,
    )
    euler_difference.savefig(
        output_dir / "euler_angle_density_difference.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(euler_difference)
    _save_nearest_neighbor_plot(
        generated_to_reference,
        reference_to_generated,
        output_dir / "nearest_neighbor_summary.png",
    )
    _save_mmd_plot(mmd, args.mmd_sigma, output_dir / "mmd_summary.png")

    report = {
        "run_dir": str(run_dir),
        "generated_samples_path": str(generated_path),
        "reference_samples_path": str(output_dir / "reference_samples.npy"),
        "representation": "3x3_rotation_matrix",
        "distance_convention": "frobenius_norm_of_matrix_log",
        "number_generated_samples": int(len(generated)),
        "number_reference_samples": int(len(reference)),
        "metric_subsample_generated": int(len(generated_metric)),
        "metric_subsample_reference": int(len(reference_metric)),
        "rbf_mmd": float(mmd),
        "rbf_sigma": float(args.mmd_sigma),
        "nearest_neighbor_geodesic_distance": _distribution_summary(
            generated_to_reference
        ),
        "reference_to_generated_nearest_neighbor": _distribution_summary(
            reference_to_generated
        ),
        "constraints": constraint_metrics(generated),
        "euler_angle_comparison": {
            "conversion": "_SpecialOrthogonal3Vectors.tait_bryan_angles_from_matrix",
            "angle_order": ["alpha", "beta", "gamma"],
            "axis_labels": list(SO3_TAIT_BRYAN_LABELS),
            "axis_limits_radians": [list(value) for value in SO3_TAIT_BRYAN_RANGES],
            "bins": int(euler_comparison["bins"]),
            "normalization": "density",
            "matched_samples_per_distribution": int(
                euler_comparison["sample_count"]
            ),
            "subsampling_seed": int(euler_comparison["seed"]),
        },
        "training_logs": _training_log_summary(run_dir),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")
    csv_row = _flatten_for_csv(report)
    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_row))
        writer.writeheader()
        writer.writerow(csv_row)

    print("Saved SO(3) generation evaluation to {}".format(output_dir))


if __name__ == "__main__":
    main()
