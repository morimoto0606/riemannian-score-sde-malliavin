#!/usr/bin/env python3
"""Diagnose an online Malliavin teacher against the conditional Heat score.

For one fixed ``x_0`` and ``t``, this draws many forward paths and compares
the Heat transition score with a local conditional mean of the pathwise
Malliavin weights.  Raw pathwise weights are intentionally reported
separately: Malliavin integration by parts identifies their conditional mean,
not each individual weight, with the transition score.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("GEOMSTATS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from riemannian_score_sde.teachers import HeatTeacher


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="earthquake_malliavin_hutchinson",
        choices=("earthquake_malliavin", "earthquake_malliavin_hutchinson"),
    )
    parser.add_argument("--num-paths", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=32)
    parser.add_argument("--time", type=float, default=0.2)
    parser.add_argument("--data-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hutchinson-probes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _norm_summary(vectors: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(vectors, axis=-1)
    return {"mean": float(norms.mean()), "std": float(norms.std())}


def _parallel_transport_s2(
    source_points: np.ndarray,
    target_point: np.ndarray,
    source_vectors: np.ndarray,
) -> np.ndarray:
    """Parallel transport tangent vectors along shortest S2 geodesics."""

    dot = np.sum(source_points * target_point[None, :], axis=-1)
    denominator = np.maximum(1.0 + dot, 1e-8)
    coefficient = np.sum(source_vectors * target_point[None, :], axis=-1)
    return source_vectors - (coefficient / denominator)[:, None] * (
        source_points + target_point[None, :]
    )


def _knn_conditional_mean(
    endpoints: np.ndarray,
    pathwise_weights: np.ndarray,
    k: int,
) -> np.ndarray:
    cosine = np.clip(endpoints @ endpoints.T, -1.0, 1.0)
    distances = np.arccos(cosine)
    np.fill_diagonal(distances, np.inf)  # leave-one-out conditional mean
    neighbours = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    estimates = np.empty_like(pathwise_weights)
    for index, neighbour_indices in enumerate(neighbours):
        transported = _parallel_transport_s2(
            endpoints[neighbour_indices],
            endpoints[index],
            pathwise_weights[neighbour_indices],
        )
        estimates[index] = transported.mean(axis=0)
    return estimates


def _comparison_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    difference = candidate - reference
    candidate_norm = np.linalg.norm(candidate, axis=-1)
    reference_norm = np.linalg.norm(reference, axis=-1)
    denominator = np.maximum(candidate_norm * reference_norm, 1e-12)
    cosine = np.sum(candidate * reference, axis=-1) / denominator
    return {
        "rmse": float(np.sqrt(np.mean(np.sum(difference**2, axis=-1)))),
        "mean_cosine": float(np.mean(cosine)),
        "difference_norm_mean": float(np.linalg.norm(difference, axis=-1).mean()),
    }


def _compose_config(repository_root: Path, experiment: str):
    config_dir = str((repository_root / "config").resolve())
    overrides = [
        f"experiment={experiment}",
        f"work_dir={repository_root.resolve()}",
        f"data_dir={(repository_root / 'data').resolve()}/",
    ]
    try:
        context = initialize_config_dir(
            config_dir=config_dir,
            job_name="debug_malliavin_teacher_scale",
            version_base=None,
        )
    except TypeError:  # Hydra 1.1 used by upstream.
        context = initialize_config_dir(
            config_dir=config_dir,
            job_name="debug_malliavin_teacher_scale",
        )
    with context:
        return compose(config_name="main", overrides=overrides)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_paths < 2:
        raise ValueError("num-paths must be at least two")
    if not 1 <= args.knn_k < args.num_paths:
        raise ValueError("knn-k must satisfy 1 <= k < num-paths")
    if not 0.0 < args.time <= 1.0:
        raise ValueError("time must be in (0, 1]")

    repository_root = Path(__file__).resolve().parents[1]
    cfg = _compose_config(repository_root, args.experiment)
    manifold = instantiate(cfg.manifold)
    beta_schedule = instantiate(cfg.beta_schedule)
    sde = instantiate(cfg.flow, manifold=manifold, beta_schedule=beta_schedule)
    dataset = instantiate(cfg.dataset, rng=jax.random.PRNGKey(args.seed))
    if not 0 <= args.data_index < len(dataset):
        raise IndexError("data-index is outside the Earthquake dataset")

    teacher_overrides = {}
    if args.hutchinson_probes is not None:
        if args.hutchinson_probes < 1:
            raise ValueError("hutchinson-probes must be positive")
        teacher_overrides["hutchinson_probes"] = args.hutchinson_probes
    malliavin_teacher = instantiate(cfg.teacher, **teacher_overrides)
    heat_teacher = HeatTeacher(n_max=cfg.loss.n_max, thresh=cfg.loss.thresh)
    effective_grw_duration = args.time - malliavin_teacher.sampler_eps
    if effective_grw_duration <= 0.0:
        raise ValueError(
            "time must be greater than the Malliavin teacher sampler_eps"
        )

    initial_point = dataset[args.data_index]
    initial_points = jnp.repeat(initial_point[None, :], args.num_paths, axis=0)
    terminal_times = jnp.full(
        (args.num_paths,),
        args.time,
        dtype=initial_points.dtype,
    )
    rng = jax.random.PRNGKey(args.seed)

    endpoint, pathwise_weight = jax.jit(
        lambda key, x_0, time: malliavin_teacher.sample_and_score(
            key,
            sde,
            x_0,
            time,
        )
    )(rng, initial_points, terminal_times)
    heat_endpoint, _ = heat_teacher.sample_and_score(
        rng,
        sde,
        initial_points,
        terminal_times,
    )
    heat_score = heat_teacher.score_at_endpoint(
        sde,
        initial_points,
        endpoint,
        terminal_times,
    )

    endpoint_np = np.asarray(endpoint)
    weight_np = np.asarray(pathwise_weight)
    heat_np = np.asarray(heat_score)
    conditional_mean = _knn_conditional_mean(
        endpoint_np,
        weight_np,
        args.knn_k,
    )
    sigma = float(
        np.asarray(
            sde.marginal_prob(
                jnp.zeros((1, 3), dtype=initial_points.dtype),
                jnp.asarray([args.time], dtype=initial_points.dtype),
            )[1]
        ).reshape(-1)[0]
    )

    report = {
        "experiment": args.experiment,
        "fixed_initial_data_index": args.data_index,
        "fixed_time": args.time,
        "sampler_eps": float(malliavin_teacher.sampler_eps),
        "effective_grw_duration": float(effective_grw_duration),
        "divergence_mode": malliavin_teacher.divergence_mode,
        "hutchinson_probes": int(malliavin_teacher.hutchinson_probes),
        "hutchinson_noise": malliavin_teacher.hutchinson_noise,
        "covariance_regularization": float(
            malliavin_teacher.covariance_regularization
        ),
        "num_paths": args.num_paths,
        "knn_k": args.knn_k,
        "endpoint_max_abs_error": float(
            np.max(np.abs(endpoint_np - np.asarray(heat_endpoint)))
        ),
        "raw_pathwise_malliavin_norm": _norm_summary(weight_np),
        "heat_conditional_score_norm": _norm_summary(heat_np),
        "knn_conditional_malliavin_norm": _norm_summary(conditional_mean),
        "sigma_times_raw_malliavin_norm": _norm_summary(sigma * weight_np),
        "sigma_times_heat_score_norm": _norm_summary(sigma * heat_np),
        "sigma_times_raw_malliavin_vs_heat_rmse": float(
            sigma * _comparison_metrics(weight_np, heat_np)["rmse"]
        ),
        "raw_malliavin_vs_heat": _comparison_metrics(weight_np, heat_np),
        "knn_conditional_malliavin_vs_heat": _comparison_metrics(
            conditional_mean,
            heat_np,
        ),
        "raw_malliavin_tangency_max_abs": float(
            np.max(np.abs(np.sum(endpoint_np * weight_np, axis=-1)))
        ),
        "interpretation": (
            "Correctness is assessed by knn_conditional_malliavin_vs_heat; "
            "raw_malliavin_vs_heat includes irreducible pathwise teacher variance."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
