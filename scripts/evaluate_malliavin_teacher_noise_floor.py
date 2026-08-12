#!/usr/bin/env python3
"""Estimate the validation noise floor of the Earthquake Malliavin teacher.

The script trains two diagnostic regressors on a fixed, independently sampled
teacher dataset:

* m(X_t, t), matching the information available to the upstream score model;
* m(X_0, X_t, t), isolating Malliavin path noise and providing the quantity
  that should agree with the conditional Heat transition score.

It does not load, alter, or train the upstream score model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("GEOMSTATS_BACKEND", "jax")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from riemannian_score_sde.noise_floor import (
    comparison_metrics,
    heat_comparison_rows,
    noise_floor_rows,
    residual_energy,
    uniform_time_edges,
)
from riemannian_score_sde.teachers import HeatTeacher
from score_sde.datasets import random_split


DEFAULT_OUTPUT_DIR = Path("results/earthquake_malliavin_teacher_noise_floor")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="earthquake_malliavin_hutchinson",
        choices=("earthquake_malliavin_hutchinson",),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-samples", type=int, default=16384)
    parser.add_argument("--validation-samples", type=int, default=4096)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--regression-batch-size", type=int, default=256)
    parser.add_argument("--regression-steps", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument(
        "--regression-weighting",
        choices=("sigma", "unweighted"),
        default="sigma",
        help="Training weighting only; both validation metrics are always saved.",
    )
    parser.add_argument("--time-bins", type=int, default=10)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plateau-low", type=float, default=0.88)
    parser.add_argument("--plateau-high", type=float, default=0.90)
    return parser.parse_args(argv)


def _compose_config(repository_root: Path, experiment: str):
    config_dir = str((repository_root / "config").resolve())
    overrides = [
        "experiment={}".format(experiment),
        "work_dir={}".format(repository_root.resolve()),
        "data_dir={}/".format((repository_root / "data").resolve()),
    ]
    try:
        context = initialize_config_dir(
            config_dir=config_dir,
            job_name="evaluate_malliavin_teacher_noise_floor",
            version_base=None,
        )
    except TypeError:  # Hydra 1.1 used by upstream.
        context = initialize_config_dir(
            config_dir=config_dir,
            job_name="evaluate_malliavin_teacher_noise_floor",
        )
    with context:
        return compose(config_name="main", overrides=overrides)


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "train_samples",
        "validation_samples",
        "teacher_batch_size",
        "regression_batch_size",
        "regression_steps",
        "eval_every",
        "time_bins",
    ):
        if getattr(args, name) < 1:
            raise ValueError("--{} must be positive".format(name.replace("_", "-")))
    if args.train_samples % args.teacher_batch_size != 0:
        raise ValueError("train-samples must be divisible by teacher-batch-size")
    if args.validation_samples % args.teacher_batch_size != 0:
        raise ValueError("validation-samples must be divisible by teacher-batch-size")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if not 0.0 <= args.plateau_low <= args.plateau_high:
        raise ValueError("invalid plateau interval")


def _earthquake_split(cfg, seed: int):
    """Reproduce the upstream split-key sequence without changing the split."""

    rng = jax.random.PRNGKey(seed)
    _, split_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=split_rng)
    train, validation, test = random_split(dataset, lengths=cfg.splits, rng=split_rng)
    return dataset, train, validation, test


def _subset_points(dataset, subset) -> np.ndarray:
    data = np.asarray(dataset.data)
    indices = np.asarray(subset.indices, dtype=np.int64)
    return np.asarray(data[indices], dtype=np.float32)


def _generate_teacher_dataset(
    *,
    initial_pool: np.ndarray,
    sample_count: int,
    batch_size: int,
    min_time: float,
    max_time: float,
    seed: int,
    sde,
    malliavin_teacher,
    heat_teacher,
) -> Dict[str, np.ndarray]:
    """Generate fixed diagnostic data using the production teacher methods."""

    numpy_rng = np.random.default_rng(seed)
    jax_rng = jax.random.PRNGKey(seed)
    sample_teacher = jax.jit(
        lambda key, x_0, time: malliavin_teacher.sample_and_score(
            key, sde, x_0, time
        )
    )
    score_heat = jax.jit(
        lambda x_0, endpoint, time: heat_teacher.score_at_endpoint(
            sde, x_0, endpoint, time
        )
    )

    chunks = {
        "initial_point": [],
        "endpoint": [],
        "time": [],
        "target": [],
        "heat_score": [],
        "sigma": [],
    }
    n_batches = sample_count // batch_size
    for batch_index in range(n_batches):
        selected = numpy_rng.integers(0, initial_pool.shape[0], size=batch_size)
        initial = jnp.asarray(initial_pool[selected])
        times = jnp.asarray(
            numpy_rng.uniform(min_time, max_time, size=batch_size),
            dtype=initial.dtype,
        )
        jax_rng, batch_rng = jax.random.split(jax_rng)
        endpoint, target = sample_teacher(batch_rng, initial, times)
        heat_score = score_heat(initial, endpoint, times)
        sigma = sde.marginal_prob(jnp.zeros_like(endpoint), times)[1]

        for name, value in (
            ("initial_point", initial),
            ("endpoint", endpoint),
            ("time", times),
            ("target", target),
            ("heat_score", heat_score),
            ("sigma", sigma),
        ):
            chunks[name].append(np.asarray(value))
        completed = (batch_index + 1) * batch_size
        if batch_index == 0 or completed == sample_count or completed % 1024 == 0:
            print("teacher samples: {}/{}".format(completed, sample_count))

    return {
        name: np.concatenate(values, axis=0) for name, values in chunks.items()
    }


def _make_tangent_regressor(
    manifold, sde, architecture_config, include_initial: bool
):
    """Build a tangent regressor with the upstream raw/sigma parameterisation."""

    output_shape = manifold.isom_group.dim

    def forward(endpoint, time, initial_point):
        features = (
            jnp.concatenate((endpoint, initial_point), axis=-1)
            if include_initial
            else endpoint
        )
        network = instantiate(architecture_config, output_shape=output_shape)
        weights = network(features, time)
        generators = manifold.div_free_generators(endpoint)
        raw_output = jnp.einsum("...n,...dn->...d", weights, generators)
        raw_output = manifold.to_tangent(raw_output, endpoint)
        sigma = sde.marginal_prob(jnp.zeros_like(endpoint), time)[1]
        return raw_output / sigma[..., None]

    return hk.transform(forward)


def _prediction_batches(model, params, dataset, batch_size: int) -> np.ndarray:
    apply_model = jax.jit(
        lambda endpoint, time, initial: model.apply(
            params, None, endpoint, time, initial
        )
    )
    predictions = []
    for start in range(0, dataset["time"].shape[0], batch_size):
        stop = min(start + batch_size, dataset["time"].shape[0])
        predictions.append(
            np.asarray(
                apply_model(
                    jnp.asarray(dataset["endpoint"][start:stop]),
                    jnp.asarray(dataset["time"][start:stop]),
                    jnp.asarray(dataset["initial_point"][start:stop]),
                )
            )
        )
    return np.concatenate(predictions, axis=0)


def _train_regressor(
    *,
    label: str,
    model,
    train_data: Dict[str, np.ndarray],
    validation_data: Dict[str, np.ndarray],
    steps: int,
    batch_size: int,
    learning_rate: float,
    eval_every: int,
    weighting: str,
    seed: int,
) -> Tuple[object, Dict[str, object]]:
    init_count = min(batch_size, train_data["time"].shape[0])
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    params = model.init(
        init_rng,
        jnp.asarray(train_data["endpoint"][:init_count]),
        jnp.asarray(train_data["time"][:init_count]),
        jnp.asarray(train_data["initial_point"][:init_count]),
    )
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)
    endpoint = jnp.asarray(train_data["endpoint"])
    time = jnp.asarray(train_data["time"])
    initial = jnp.asarray(train_data["initial_point"])
    target = jnp.asarray(train_data["target"])
    sigma_squared = jnp.asarray(train_data["sigma"] ** 2)

    def loss_for_batch(current_params, indices):
        prediction = model.apply(
            current_params,
            None,
            endpoint[indices],
            time[indices],
            initial[indices],
        )
        squared_error = jnp.sum((prediction - target[indices]) ** 2, axis=-1)
        if weighting == "sigma":
            squared_error = sigma_squared[indices] * squared_error
        return jnp.mean(squared_error)

    @jax.jit
    def update(current_params, current_optimizer_state, key):
        indices = jax.random.randint(
            key, (batch_size,), 0, endpoint.shape[0]
        )
        loss, gradient = jax.value_and_grad(loss_for_batch)(current_params, indices)
        updates, next_optimizer_state = optimizer.update(
            gradient, current_optimizer_state
        )
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_optimizer_state, loss

    history: List[Dict[str, float]] = []
    for step in range(1, steps + 1):
        rng, step_rng = jax.random.split(rng)
        params, optimizer_state, train_loss = update(
            params, optimizer_state, step_rng
        )
        if step == 1 or step % eval_every == 0 or step == steps:
            validation_prediction = _prediction_batches(
                model, params, validation_data, batch_size
            )
            validation_weights = (
                validation_data["sigma"] ** 2 if weighting == "sigma" else None
            )
            validation_energy = residual_energy(
                validation_data["target"],
                validation_prediction,
                validation_weights,
            )
            validation_loss = validation_energy["numerator"]
            history.append(
                {
                    "step": step,
                    "train_loss": float(np.asarray(train_loss)),
                    "validation_loss": validation_loss,
                    "validation_relative_loss": validation_energy["ratio"],
                }
            )
            print(
                "{} step={} train={:.6e} val={:.6e} relative={:.6f}".format(
                    label,
                    step,
                    history[-1]["train_loss"],
                    validation_loss,
                    validation_energy["ratio"],
                )
            )
    # Return the fixed final training iterate.  Validation is diagnostic only:
    # selecting a checkpoint on this same set would bias the reported noise
    # floor downward.
    return params, {
        "label": label,
        "conditioning": "X0,Xt,t" if "transition" in label else "Xt,t",
        "training_weighting": weighting,
        "final_validation_loss": history[-1]["validation_loss"],
        "validation_used_for_model_selection": False,
        "history": history,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_noise_plot(
    path: Path,
    rows: List[Dict[str, float]],
    plateau_low: float,
    plateau_high: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = np.asarray([row["time_center"] for row in rows])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(
        centers,
        [row["marginal_ratio"] for row in rows],
        "o-",
        label="E||T-m(Xt,t)||^2 / E||T||^2",
    )
    ax.plot(
        centers,
        [row["marginal_sigma_weighted_ratio"] for row in rows],
        "o-",
        label="sigma-weighted marginal floor",
    )
    ax.plot(
        centers,
        [row["transition_sigma_weighted_ratio"] for row in rows],
        "o--",
        label="sigma-weighted pathwise floor given X0,Xt,t",
    )
    ax.axhspan(
        plateau_low,
        plateau_high,
        color="black",
        alpha=0.10,
        label="observed train/relative_loss plateau",
    )
    ax.set_xlabel("t")
    ax.set_ylabel("validation residual / teacher energy")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _overall_heat_rows(rows: List[Dict[str, object]]) -> Dict[str, Dict]:
    return {
        row["regressor"]: row
        for row in rows
        if row["scope"] == "overall"
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    repository_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = repository_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _compose_config(repository_root, args.experiment)
    manifold = instantiate(cfg.manifold)
    beta_schedule = instantiate(cfg.beta_schedule)
    sde = instantiate(cfg.flow, manifold=manifold, beta_schedule=beta_schedule)
    dataset, train_subset, validation_subset, test_subset = _earthquake_split(
        cfg, args.seed
    )
    train_pool = _subset_points(dataset, train_subset)
    validation_pool = _subset_points(dataset, validation_subset)

    malliavin_teacher = instantiate(
        cfg.teacher,
        divergence_mode="hutchinson",
        hutchinson_probes=4,
    )
    heat_teacher = HeatTeacher(n_max=cfg.loss.n_max, thresh=cfg.loss.thresh)
    min_time = float(sde.t0 + cfg.loss.eps)
    configured_max_time = getattr(cfg.loss, "max_t", None)
    max_time = (
        float(args.max_time)
        if args.max_time is not None
        else float(sde.tf if configured_max_time is None else configured_max_time)
    )
    if max_time <= min_time or max_time > float(sde.tf):
        raise ValueError("max-time must be in ({}, {}]".format(min_time, sde.tf))

    print("generating training teacher dataset")
    train_data = _generate_teacher_dataset(
        initial_pool=train_pool,
        sample_count=args.train_samples,
        batch_size=args.teacher_batch_size,
        min_time=min_time,
        max_time=max_time,
        seed=args.seed + 1001,
        sde=sde,
        malliavin_teacher=malliavin_teacher,
        heat_teacher=heat_teacher,
    )
    print("generating independent validation teacher dataset")
    validation_data = _generate_teacher_dataset(
        initial_pool=validation_pool,
        sample_count=args.validation_samples,
        batch_size=args.teacher_batch_size,
        min_time=min_time,
        max_time=max_time,
        seed=args.seed + 2001,
        sde=sde,
        malliavin_teacher=malliavin_teacher,
        heat_teacher=heat_teacher,
    )

    marginal_model = _make_tangent_regressor(
        manifold, sde, cfg.architecture, include_initial=False
    )
    transition_model = _make_tangent_regressor(
        manifold, sde, cfg.architecture, include_initial=True
    )
    marginal_params, marginal_training = _train_regressor(
        label="marginal_regressor",
        model=marginal_model,
        train_data=train_data,
        validation_data=validation_data,
        steps=args.regression_steps,
        batch_size=args.regression_batch_size,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        weighting=args.regression_weighting,
        seed=args.seed + 3001,
    )
    transition_params, transition_training = _train_regressor(
        label="transition_regressor",
        model=transition_model,
        train_data=train_data,
        validation_data=validation_data,
        steps=args.regression_steps,
        batch_size=args.regression_batch_size,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        weighting=args.regression_weighting,
        seed=args.seed + 4001,
    )
    marginal_prediction = _prediction_batches(
        marginal_model,
        marginal_params,
        validation_data,
        args.regression_batch_size,
    )
    transition_prediction = _prediction_batches(
        transition_model,
        transition_params,
        validation_data,
        args.regression_batch_size,
    )

    target = validation_data["target"]
    heat_score = validation_data["heat_score"]
    sigma_squared = validation_data["sigma"] ** 2
    edges = uniform_time_edges(min_time, max_time, args.time_bins)
    noise_rows = noise_floor_rows(
        validation_data["time"],
        target,
        marginal_prediction,
        transition_prediction,
        sigma_squared,
        edges,
    )
    heat_rows = heat_comparison_rows(
        validation_data["time"],
        marginal_prediction,
        transition_prediction,
        heat_score,
        sigma_squared,
        edges,
    )
    marginal_raw = residual_energy(target, marginal_prediction)
    marginal_weighted = residual_energy(
        target, marginal_prediction, sigma_squared
    )
    transition_raw = residual_energy(target, transition_prediction)
    transition_weighted = residual_energy(
        target, transition_prediction, sigma_squared
    )
    heat_overall = _overall_heat_rows(heat_rows)
    plateau_midpoint = 0.5 * (args.plateau_low + args.plateau_high)
    weighted_ratio = marginal_weighted["ratio"]
    tangency = {
        "marginal_max_abs": float(
            np.max(
                np.abs(
                    np.sum(validation_data["endpoint"] * marginal_prediction, axis=-1)
                )
            )
        ),
        "transition_max_abs": float(
            np.max(
                np.abs(
                    np.sum(
                        validation_data["endpoint"] * transition_prediction,
                        axis=-1,
                    )
                )
            )
        ),
    }

    summary = {
        "experiment": args.experiment,
        "purpose": "estimate Malliavin teacher conditional variance",
        "teacher": {
            "implementation": type(malliavin_teacher).__name__,
            "divergence_mode": malliavin_teacher.divergence_mode,
            "hutchinson_probes": int(malliavin_teacher.hutchinson_probes),
            "hutchinson_noise": malliavin_teacher.hutchinson_noise,
            "covariance_regularization": float(
                malliavin_teacher.covariance_regularization
            ),
        },
        "data": {
            "split_seed": args.seed,
            "upstream_splits": [float(value) for value in cfg.splits],
            "earthquake_train_pool": int(len(train_subset)),
            "earthquake_validation_pool": int(len(validation_subset)),
            "earthquake_test_pool": int(len(test_subset)),
            "teacher_train_samples": args.train_samples,
            "teacher_validation_samples": args.validation_samples,
            "min_time": min_time,
            "max_time": max_time,
            "time_bins": args.time_bins,
        },
        "regression": {
            "architecture": "DivFreeGenerator fields with existing Concat architecture",
            "score_parameterization": (
                "tangent raw output / sigma(t), matching upstream"
            ),
            "training_weighting": args.regression_weighting,
            "batch_size": args.regression_batch_size,
            "steps": args.regression_steps,
            "learning_rate": args.learning_rate,
            "marginal": marginal_training,
            "transition": transition_training,
        },
        "validation": {
            "marginal_noise_floor_given_Xt_t": {
                "interpretation": (
                    "Upper-bound estimate of the relative loss reachable by a "
                    "model observing only (Xt,t); includes regressor "
                    "approximation error."
                ),
                "unweighted": marginal_raw,
                "sigma_weighted": marginal_weighted,
            },
            "malliavin_path_noise_given_X0_Xt_t": {
                "interpretation": (
                    "Upper-bound estimate of Malliavin-specific path variance; "
                    "includes transition-regressor approximation error."
                ),
                "unweighted": transition_raw,
                "sigma_weighted": transition_weighted,
            },
            "conditional_mean_vs_heat": {
                "marginal_xt_t": heat_overall["marginal_xt_t"],
                "transition_x0_xt_t": heat_overall["transition_x0_xt_t"],
                "note": (
                    "Only the transition regressor is theoretically expected to "
                    "match the Heat transition score. The marginal regressor targets "
                    "the Earthquake mixture score."
                ),
            },
            "pathwise_target_vs_heat": comparison_metrics(target, heat_score),
            "prediction_tangency": tangency,
        },
        "plateau_comparison": {
            "observed_interval": [args.plateau_low, args.plateau_high],
            "observed_midpoint": plateau_midpoint,
            "sigma_weighted_marginal_noise_floor": weighted_ratio,
            "sigma_weighted_pathwise_noise_floor_given_X0_Xt_t": (
                transition_weighted["ratio"]
            ),
            "pathwise_share_of_marginal_floor": float(
                transition_weighted["ratio"] / (weighted_ratio + 1e-8)
            ),
            "noise_floor_over_plateau_midpoint": float(
                weighted_ratio / (plateau_midpoint + 1e-8)
            ),
            "within_observed_interval": bool(
                args.plateau_low <= weighted_ratio <= args.plateau_high
            ),
        },
    }

    summary_path = output_dir / "noise_floor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "noise_floor_by_time.csv", noise_rows)
    _write_csv(output_dir / "conditional_mean_vs_heat.csv", heat_rows)
    _save_noise_plot(
        output_dir / "teacher_noise_by_time.png",
        noise_rows,
        args.plateau_low,
        args.plateau_high,
    )

    print(
        "\nvalidation sigma-weighted marginal noise floor: {:.6f}".format(
            weighted_ratio
        )
    )
    print(
        "validation sigma-weighted path noise given X0,Xt,t: {:.6f}".format(
            transition_weighted["ratio"]
        )
    )
    transition_heat = heat_overall["transition_x0_xt_t"]
    print("transition conditional mean vs Heat")
    print("  RMSE = {:.6f}".format(transition_heat["rmse"]))
    print("  relative RMSE = {:.6f}".format(transition_heat["relative_rmse"]))
    print("  cosine similarity = {:.6f}".format(transition_heat["cosine_similarity"]))
    print("saved diagnostics in {}".format(output_dir))


if __name__ == "__main__":
    main()
