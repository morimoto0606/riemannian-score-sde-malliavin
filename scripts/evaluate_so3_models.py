#!/usr/bin/env python3
"""Compare generated SO(3) samples against one shared wrapped-mixture target.

The evaluator saves De Bortoli-style Euler-angle marginals, bidirectional
nearest-neighbour rotation-angle distances, a geodesic-kernel MMD, and SO(3)
constraint residuals.  Optional teacher diagnostics compare a Malliavin target
with Varadhan on exactly the same endpoints and report results by time bin.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Sequence

os.environ.setdefault("GEOMSTATS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from riemannian_score_sde.teachers import (
    MalliavinTeacher,
    VaradhanTeacher,
    so3_left_invariant_frame,
)
from riemannian_score_sde.utils.vis import plot_so3


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=RUN_DIR",
        help="Model label and Hydra run directory; repeat for every model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-real", type=int, default=16384)
    parser.add_argument("--metric-subsample", type=int, default=2000)
    parser.add_argument("--metric-seed", type=int, default=0)
    parser.add_argument(
        "--target-sample-seed",
        type=int,
        default=10000,
        help="Changes target draws but not the saved mixture component means.",
    )
    parser.add_argument("--mmd-sigma", type=float, default=1.0)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument(
        "--teacher-diagnostic-run-dir",
        type=Path,
        default=None,
        help="Optional SO(3) Malliavin run whose saved teacher config is checked.",
    )
    parser.add_argument("--teacher-samples", type=int, default=128)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    return parser.parse_args(argv)


def _parse_runs(values: Sequence[str]) -> list[tuple[str, Path]]:
    runs = []
    labels = set()
    for value in values:
        if "=" not in value:
            raise ValueError("--run must have the form NAME=RUN_DIR")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or label in labels:
            raise ValueError(
                "run labels must be unique and contain only letters, digits, '.', '_' or '-'"
            )
        labels.add(label)
        runs.append((label, Path(raw_path).expanduser().resolve()))
    return runs


def _saved_samples_setting(run_dir: Path) -> str | None:
    config_path = run_dir / ".hydra/config.yaml"
    if not config_path.is_file():
        return None
    match = re.search(
        r"^generated_samples_path:\s*(.+?)\s*$",
        config_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return None if match is None else match.group(1).strip().strip("'\"")


def _localize_path(value: str) -> Path:
    path = Path(value.replace("${work_dir}", str(REPOSITORY_ROOT))).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if path.is_file():
        return path.resolve()
    if "results" in path.parts:
        index = path.parts.index("results")
        rebased = REPOSITORY_ROOT.joinpath(*path.parts[index:])
        if rebased.is_file():
            return rebased.resolve()
    return path


def resolve_generated_samples(run_dir: Path) -> Path:
    candidates = [run_dir / "generated_samples.npy"]
    setting = _saved_samples_setting(run_dir)
    if setting is not None:
        candidates.append(_localize_path(setting))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n".join("  - {}".format(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find generated_samples.npy for {}. Checked:\n{}".format(
            run_dir, checked
        )
    )


def load_rotations(path: Path, label: str) -> np.ndarray:
    rotations = np.asarray(np.load(path, allow_pickle=False))
    if rotations.ndim == 2 and rotations.shape[-1] == 9:
        rotations = rotations.reshape((-1, 3, 3))
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("{} samples must have shape [n,3,3] or [n,9]".format(label))
    if not np.isfinite(rotations).all():
        raise ValueError("{} samples contain non-finite values".format(label))
    return rotations


def load_run_config(run_dir: Path):
    config_path = run_dir / ".hydra/config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError("Missing saved Hydra config: {}".format(config_path))
    return OmegaConf.load(config_path)


def _fairness_signature(cfg) -> dict:
    """Select settings that must be shared across teacher comparisons."""

    keys = (
        "manifold",
        "dataset",
        "architecture",
        "embedding",
        "generator",
        "model",
        "optim",
        "scheduler",
        "beta_schedule",
        "flow",
        "batch_size",
        "steps",
        "warmup_steps",
        "ema_rate",
        "eps",
        "seed",
    )
    return {
        key: OmegaConf.to_container(cfg[key], resolve=True)
        if OmegaConf.is_config(cfg.get(key))
        else cfg.get(key)
        for key in keys
    }


def generate_shared_target(cfg, sample_count: int, sample_seed: int) -> np.ndarray:
    manifold = instantiate(cfg.manifold)
    dataset = instantiate(cfg.dataset, rng=jax.random.PRNGKey(int(cfg.seed)))
    dataset.batch_dims = [sample_count]
    # Wrapped fixes the mixture means during construction.  Replacing only its
    # subsequent sample key gives an independent evaluation draw from exactly
    # the same saved target distribution.
    dataset.rng = jax.random.PRNGKey(sample_seed)
    samples = np.asarray(next(dataset)[0])
    if samples.shape[1:] != (3, 3) or getattr(manifold, "n", None) != 3:
        raise ValueError("saved run is not a matrix SO(3) experiment")
    return samples


def pairwise_rotation_angles(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return physical SO(3) rotation angles in radians."""

    traces = np.einsum("aij,bij->ab", left, right)
    cosines = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosines)


def nearest_neighbor_rotation_angles(
    source: np.ndarray,
    reference: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    nearest = []
    for start in range(0, len(source), chunk_size):
        angles = pairwise_rotation_angles(source[start : start + chunk_size], reference)
        nearest.append(np.min(angles, axis=1))
    return np.concatenate(nearest)


def _kernel_sum(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
    chunk_size: int,
) -> float:
    total = 0.0
    for start in range(0, len(left), chunk_size):
        angles = pairwise_rotation_angles(left[start : start + chunk_size], right)
        total += float(np.exp(-(angles**2) / (2.0 * sigma**2)).sum())
    return total


def geodesic_rbf_mmd(
    real: np.ndarray,
    generated: np.ndarray,
    sigma: float,
    chunk_size: int,
) -> float:
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("MMD requires at least two real and generated samples")
    real_sum = _kernel_sum(real, real, sigma, chunk_size) - len(real)
    generated_sum = _kernel_sum(generated, generated, sigma, chunk_size) - len(
        generated
    )
    cross_sum = _kernel_sum(real, generated, sigma, chunk_size)
    return (
        real_sum / (len(real) * (len(real) - 1))
        + generated_sum / (len(generated) * (len(generated) - 1))
        - 2.0 * cross_sum / (len(real) * len(generated))
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def constraint_metrics(rotations: np.ndarray) -> dict[str, dict[str, float]]:
    identity = np.eye(3, dtype=rotations.dtype)
    gram = np.swapaxes(rotations, -1, -2) @ rotations
    orthogonality = np.linalg.norm(gram - identity, axis=(-2, -1))
    determinant = np.abs(np.linalg.det(rotations) - 1.0)
    return {
        "orthogonality_frobenius": _summary(orthogonality),
        "absolute_determinant_error": _summary(determinant),
    }


def evaluate_model(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    subsample: int,
    seed: int,
    sigma: float,
    chunk_size: int,
) -> dict:
    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(real), min(subsample, len(real)), replace=False)
    generated_idx = rng.choice(
        len(generated), min(subsample, len(generated)), replace=False
    )
    real_metric = real[real_idx]
    generated_metric = generated[generated_idx]
    generated_to_real = nearest_neighbor_rotation_angles(
        generated_metric, real_metric, chunk_size
    )
    real_to_generated = nearest_neighbor_rotation_angles(
        real_metric, generated_metric, chunk_size
    )
    return {
        "sample_count": int(len(generated)),
        "metric_sample_count_real": int(len(real_metric)),
        "metric_sample_count_generated": int(len(generated_metric)),
        "distance_unit": "rotation_angle_radians",
        "generated_to_real_nearest_neighbor": _summary(generated_to_real),
        "real_to_generated_nearest_neighbor": _summary(real_to_generated),
        "geodesic_rbf_mmd_unbiased": float(
            geodesic_rbf_mmd(real_metric, generated_metric, sigma, chunk_size)
        ),
        "geodesic_rbf_sigma_radians": float(sigma),
        "constraints": constraint_metrics(generated),
    }


def save_metrics(output_dir: Path, metrics: dict) -> None:
    with (output_dir / "so3_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=False)
        handle.write("\n")

    rows = []
    for model, values in metrics["models"].items():
        rows.append(
            {
                "model": model,
                "generated_to_real_nn_mean": values[
                    "generated_to_real_nearest_neighbor"
                ]["mean"],
                "generated_to_real_nn_median": values[
                    "generated_to_real_nearest_neighbor"
                ]["median"],
                "generated_to_real_nn_max": values[
                    "generated_to_real_nearest_neighbor"
                ]["max"],
                "real_to_generated_nn_mean": values[
                    "real_to_generated_nearest_neighbor"
                ]["mean"],
                "real_to_generated_nn_median": values[
                    "real_to_generated_nearest_neighbor"
                ]["median"],
                "real_to_generated_nn_max": values[
                    "real_to_generated_nearest_neighbor"
                ]["max"],
                "geodesic_rbf_mmd_unbiased": values["geodesic_rbf_mmd_unbiased"],
                "orthogonality_frobenius_mean": values["constraints"][
                    "orthogonality_frobenius"
                ]["mean"],
                "orthogonality_frobenius_max": values["constraints"][
                    "orthogonality_frobenius"
                ]["max"],
                "absolute_determinant_error_mean": values["constraints"][
                    "absolute_determinant_error"
                ]["mean"],
                "absolute_determinant_error_max": values["constraints"][
                    "absolute_determinant_error"
                ]["max"],
            }
        )
    with (output_dir / "so3_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def teacher_diagnostics(
    run_dir: Path,
    sample_count: int,
    batch_size: int,
    sample_seed: int,
) -> tuple[dict, list[dict]]:
    if sample_count < 1 or batch_size < 1:
        raise ValueError("teacher sample and batch sizes must be positive")
    if sample_count % batch_size != 0:
        raise ValueError("teacher-samples must be divisible by teacher-batch-size")
    cfg = load_run_config(run_dir)
    manifold = instantiate(cfg.manifold)
    beta_schedule = instantiate(cfg.beta_schedule)
    sde = instantiate(cfg.flow, manifold=manifold, beta_schedule=beta_schedule)
    teacher = instantiate(cfg.teacher)
    if not isinstance(teacher, MalliavinTeacher):
        raise ValueError("teacher diagnostic run must use MalliavinTeacher")
    if teacher.rb_enabled:
        raise ValueError("SO(3) teacher diagnostics require rb_enabled=false")
    dataset = instantiate(cfg.dataset, rng=jax.random.PRNGKey(int(cfg.seed)))
    dataset.rng = jax.random.PRNGKey(sample_seed)
    reference = VaradhanTeacher()
    edges = np.asarray([float(cfg.eps), 0.1, 0.25, 0.5, 0.75, float(sde.tf)])
    rng = jax.random.PRNGKey(sample_seed + 1)
    differences = []
    cosines = []
    all_times = []
    remaining = sample_count

    def sample_and_compare(key, initial, times):
        endpoint, target = teacher.sample_and_score(key, sde, initial, times)
        varadhan = reference.score_at_endpoint(sde, initial, endpoint, times)
        return endpoint, target, varadhan

    sample_and_compare = jax.jit(sample_and_compare)
    while remaining:
        current = min(batch_size, remaining)
        dataset.batch_dims = [current]
        initial = next(dataset)[0]
        rng, time_rng, teacher_rng = jax.random.split(rng, 3)
        times = jax.random.uniform(
            time_rng,
            (current,),
            minval=float(cfg.eps),
            maxval=float(sde.tf),
        )
        endpoint, target, varadhan = sample_and_compare(
            teacher_rng, initial, times
        )

        def coordinates(point, vector):
            frame = so3_left_invariant_frame(manifold, point)
            return frame.T @ vector.reshape(-1)

        target_coords = jax.vmap(coordinates)(endpoint, target)
        reference_coords = jax.vmap(coordinates)(endpoint, varadhan)
        delta = target_coords - reference_coords
        differences.append(np.asarray(jnp.sum(delta**2, axis=-1)))
        denominator = jnp.linalg.norm(target_coords, axis=-1) * jnp.linalg.norm(
            reference_coords, axis=-1
        )
        cosine = jnp.sum(target_coords * reference_coords, axis=-1) / jnp.maximum(
            denominator, 1e-12
        )
        cosines.append(np.asarray(cosine))
        all_times.append(np.asarray(times))
        remaining -= current

    times = np.concatenate(all_times)
    squared_errors = np.concatenate(differences)
    cosine = np.concatenate(cosines)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (times >= lower) & (times < upper)
        if upper == edges[-1]:
            mask |= times == upper
        rows.append(
            {
                "time_lower": float(lower),
                "time_upper": float(upper),
                "count": int(mask.sum()),
                "tangent_rmse": (
                    float(np.sqrt(np.mean(squared_errors[mask])))
                    if mask.any()
                    else None
                ),
                "cosine_similarity": (
                    float(np.mean(cosine[mask])) if mask.any() else None
                ),
            }
        )
    metadata = {
        "run_dir": str(run_dir),
        "teacher": type(teacher).__name__,
        "reference_teacher": type(reference).__name__,
        "samples": int(sample_count),
        "hutchinson_probes": int(teacher.hutchinson_probes),
        "rb_enabled": bool(teacher.rb_enabled),
    }
    return metadata, rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_real < 2 or args.metric_subsample < 2:
        raise ValueError("num-real and metric-subsample must be at least two")
    if args.mmd_sigma <= 0.0 or args.distance_chunk_size < 1:
        raise ValueError("mmd-sigma and distance-chunk-size must be positive")
    runs = _parse_runs(args.run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_cfg = load_run_config(runs[0][1])
    fairness_signature = _fairness_signature(reference_cfg)
    real = generate_shared_target(
        reference_cfg, args.num_real, args.target_sample_seed
    )
    np.save(args.output_dir / "shared_target_samples.npy", real)

    model_metrics = {}
    for label, run_dir in runs:
        run_cfg = load_run_config(run_dir)
        if _fairness_signature(run_cfg) != fairness_signature:
            raise ValueError(
                "{} does not share the dataset/model/optimizer/SDE/training "
                "settings of {}".format(run_dir, runs[0][1])
            )
        samples_path = resolve_generated_samples(run_dir)
        generated = load_rotations(samples_path, label)
        model_metrics[label] = evaluate_model(
            real,
            generated,
            subsample=args.metric_subsample,
            seed=args.metric_seed,
            sigma=args.mmd_sigma,
            chunk_size=args.distance_chunk_size,
        )
        model_metrics[label]["run_dir"] = str(run_dir)
        model_metrics[label]["generated_samples_path"] = str(samples_path)
        model_metrics[label]["loss_weighting"] = {
            "like_w": run_cfg.loss.get("like_w", None),
            "time_weighting": bool(run_cfg.loss.get("time_weighting", False)),
            "time_weight_lambda": float(
                run_cfg.loss.get("time_weight_lambda", 0.0)
            ),
        }
        figure = plot_so3(real, generated, size=12)
        figure.savefig(
            args.output_dir / "euler_target_vs_{}.png".format(label),
            dpi=180,
            bbox_inches="tight",
        )

    report = {
        "target_sample_count": int(len(real)),
        "target_sample_seed": int(args.target_sample_seed),
        "metric_seed": int(args.metric_seed),
        "shared_experiment_settings": fairness_signature,
        "models": model_metrics,
    }
    if args.teacher_diagnostic_run_dir is not None:
        metadata, rows = teacher_diagnostics(
            args.teacher_diagnostic_run_dir.expanduser().resolve(),
            args.teacher_samples,
            args.teacher_batch_size,
            args.target_sample_seed + 100,
        )
        report["teacher_diagnostics"] = {"metadata": metadata, "time_bins": rows}
        with (args.output_dir / "teacher_diagnostics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    save_metrics(args.output_dir, report)
    print("Saved SO(3) evaluation to {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
