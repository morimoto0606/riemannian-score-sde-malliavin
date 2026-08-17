#!/usr/bin/env python3
"""Evaluate a trained Earthquake score model by diffusion-time bin.

The saved Hydra snapshot determines the dataset, SDE, model, and teacher.  The
diagnostic never trains or updates the model and never applies training-time
loss weighting to its metrics.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("GEOMSTATS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from riemannian_score_sde.score_diagnostics import score_error_rows
from score_sde.datasets import TensorDataset, random_split
from score_sde.utils import restore


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MIN_TIME = 0.001
MAX_TIME = 1.0
DEFAULT_OUTPUT_NAME = "score_error_by_time.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        help="Run directory containing .hydra/config.yaml and its checkpoint.",
    )
    source.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        help=(
            "Checkpoint directory (or arrays.npy/tree.pkl within it). The run "
            "directory is inferred from an ancestor containing .hydra/config.yaml."
        ),
    )
    parser.add_argument("--validation-samples", type=int, default=4096)
    parser.add_argument("--time-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size; lower this if a pathwise teacher uses much memory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV. With one run, defaults to RUN_DIR/"
            f"{DEFAULT_OUTPUT_NAME}; with multiple runs, defaults to the current directory."
        ),
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("validation_samples", "time_bins", "batch_size"):
        if getattr(args, name) < 1:
            raise ValueError("--{} must be positive".format(name.replace("_", "-")))


def _absolute(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _find_run_dir(path: Path) -> Path:
    start = path.parent if path.is_file() else path
    for candidate in (start, *start.parents):
        if (candidate / ".hydra/config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find .hydra/config.yaml in {} or its ancestors".format(path)
    )


def _resolve_target(path: Path, *, is_checkpoint: bool) -> tuple[Path, Path]:
    resolved = _absolute(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    if is_checkpoint:
        checkpoint_dir = resolved.parent if resolved.is_file() else resolved
        run_dir = _find_run_dir(checkpoint_dir)
    else:
        run_dir = resolved
        config_path = run_dir / ".hydra/config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        cfg = OmegaConf.load(config_path)
        checkpoint_dir = run_dir / str(cfg.get("ckpt_dir", "ckpt"))

    for filename in ("arrays.npy", "tree.pkl"):
        if not (checkpoint_dir / filename).is_file():
            raise FileNotFoundError(checkpoint_dir / filename)
    return run_dir, checkpoint_dir


def _load_run_config(run_dir: Path) -> DictConfig:
    cfg = OmegaConf.load(run_dir / ".hydra/config.yaml")

    # Hydra's runtime cwd is unavailable when an immutable snapshot is loaded
    # directly. Rebase only repository-owned input paths for local evaluation.
    cfg.work_dir = str(REPOSITORY_ROOT)
    cfg.data_dir = str(REPOSITORY_ROOT / "data") + os.sep
    if isinstance(cfg.get("dataset"), DictConfig) and "data_dir" in cfg.dataset:
        cfg.dataset.data_dir = cfg.data_dir
    return cfg


def _instantiate_teacher(cfg: DictConfig):
    teacher_cfg = cfg.get("teacher")
    if isinstance(teacher_cfg, DictConfig) and "_target_" in teacher_cfg:
        return instantiate(teacher_cfg)

    # Older run snapshots stored the Hydra config-group name (for example
    # "heat") instead of the resolved mapping. Load that group generically;
    # there is no teacher-specific branch in this evaluator.
    if isinstance(teacher_cfg, str):
        if Path(teacher_cfg).name != teacher_cfg:
            raise ValueError("invalid saved teacher config name: {}".format(teacher_cfg))
        config_path = REPOSITORY_ROOT / "config/teacher" / (teacher_cfg + ".yaml")
        if not config_path.is_file():
            raise FileNotFoundError(
                "Saved teacher {!r} has no config at {}".format(
                    teacher_cfg, config_path
                )
            )
        bundle = OmegaConf.create(
            {
                "eps": cfg.get("eps", MIN_TIME),
                "loss": OmegaConf.to_container(cfg.loss, resolve=False),
                "teacher": OmegaConf.to_container(
                    OmegaConf.load(config_path), resolve=False
                ),
            }
        )
        return instantiate(bundle.teacher)

    raise ValueError("saved run config does not define an instantiable teacher")


def _build_model(cfg: DictConfig, model_manifold):
    def model(y, t, context=None):
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifold=model_manifold,
        )
        if context is not None:
            t_expanded = jnp.expand_dims(t.reshape(-1), -1)
            if context.shape[0] != y.shape[0]:
                context = jnp.repeat(jnp.expand_dims(context, 0), y.shape[0], 0)
            context = jnp.concatenate([t_expanded, context], axis=-1)
        else:
            context = t
        return score(y, context)

    return hk.transform_with_state(model)


def _restore_components(cfg: DictConfig, checkpoint_dir: Path):
    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    beta_schedule = instantiate(cfg.beta_schedule)
    sde = instantiate(
        cfg.flow,
        manifold=model_manifold,
        beta_schedule=beta_schedule,
    )
    model = _build_model(cfg, model_manifold)
    teacher = _instantiate_teacher(cfg)
    train_state = restore(str(checkpoint_dir))
    score_fn = sde.reparametrise_score_fn(
        model,
        train_state.params_ema,
        train_state.model_state,
        False,
        False,
    )
    return sde.manifold, transform, sde, teacher, score_fn, train_state


def _validation_pool(cfg: DictConfig) -> np.ndarray:
    # Reproduce the exact RNG sequence used in run.py for dataset creation and
    # train/validation/test splitting. Diagnostic sampling uses the CLI seed
    # separately and therefore does not alter the saved validation partition.
    rng = jax.random.PRNGKey(int(cfg.seed))
    _, split_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=split_rng)
    if not isinstance(dataset, TensorDataset):
        raise TypeError("Earthquake evaluation requires a finite TensorDataset")
    _, validation, _ = random_split(dataset, lengths=cfg.splits, rng=split_rng)
    indices = np.asarray(validation.indices, dtype=np.int64)
    return np.asarray(dataset.data)[indices]


def _evaluate_run(
    run_dir: Path,
    checkpoint_dir: Path,
    *,
    validation_samples: int,
    time_bins: int,
    seed: int,
    batch_size: int,
) -> list[dict[str, object]]:
    cfg = _load_run_config(run_dir)
    manifold, transform, sde, teacher, score_fn, train_state = _restore_components(
        cfg, checkpoint_dir
    )
    validation_pool = _validation_pool(cfg)
    if validation_pool.shape[0] == 0:
        raise ValueError("saved validation split is empty")

    numpy_rng = np.random.default_rng(seed)
    selected = numpy_rng.integers(
        0, validation_pool.shape[0], size=validation_samples
    )
    initial_data = validation_pool[selected]
    times = numpy_rng.uniform(MIN_TIME, MAX_TIME, size=validation_samples)

    sample_teacher = jax.jit(
        lambda key, x_0, time: teacher.sample_and_score(key, sde, x_0, time)
    )
    evaluate_score = jax.jit(
        lambda key, endpoint, time: score_fn(
            endpoint, time, None, rng=key
        )
    )

    teacher_energy_chunks = []
    model_error_chunks = []
    cosine_chunks = []
    rng = jax.random.PRNGKey(seed)
    for start in range(0, validation_samples, batch_size):
        end = min(start + batch_size, validation_samples)
        x_0 = transform.inv(jnp.asarray(initial_data[start:end]))
        time = jnp.asarray(times[start:end], dtype=x_0.dtype)
        rng, teacher_rng, model_rng = jax.random.split(rng, 3)
        endpoint, target = sample_teacher(teacher_rng, x_0, time)
        prediction = evaluate_score(model_rng, endpoint, time).reshape(endpoint.shape)

        difference = prediction - target
        teacher_energy = manifold.metric.squared_norm(target, endpoint)
        model_energy = manifold.metric.squared_norm(prediction, endpoint)
        model_error = manifold.metric.squared_norm(difference, endpoint)
        inner_product = manifold.metric.inner_product(prediction, target, endpoint)
        norm_product = jnp.sqrt(
            jnp.maximum(model_energy, 0.0) * jnp.maximum(teacher_energy, 0.0)
        )
        cosine = jnp.where(
            norm_product > 0.0,
            inner_product
            / jnp.maximum(norm_product, jnp.finfo(norm_product.dtype).tiny),
            0.0,
        )
        cosine = jnp.clip(cosine, -1.0, 1.0)

        teacher_energy_chunks.append(np.asarray(teacher_energy))
        model_error_chunks.append(np.asarray(model_error))
        cosine_chunks.append(np.asarray(cosine))
        print("{}: {}/{} samples".format(run_dir.name, end, validation_samples))

    loss_cfg = cfg.get("loss", {})
    metadata = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_step": int(np.asarray(train_state.step)),
        "teacher": type(teacher).__name__,
        "seed": seed,
        "validation_samples": validation_samples,
        "training_time_weighting": bool(loss_cfg.get("time_weighting", False)),
        "training_time_weight_lambda": float(
            loss_cfg.get("time_weight_lambda", 0.0)
        ),
        "metric_time_weighting": False,
    }
    return score_error_rows(
        times,
        np.concatenate(teacher_energy_chunks),
        np.concatenate(model_error_chunks),
        np.concatenate(cosine_chunks),
        time_bins=time_bins,
        metadata=metadata,
    )


def _output_path(args: argparse.Namespace, run_dirs: Sequence[Path]) -> Path:
    if args.output is not None:
        return _absolute(args.output)
    if len(run_dirs) == 1:
        return run_dirs[0] / DEFAULT_OUTPUT_NAME
    return Path.cwd() / DEFAULT_OUTPUT_NAME


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    _validate_args(args)
    raw_targets = args.run_dir if args.run_dir is not None else args.checkpoint
    is_checkpoint = args.checkpoint is not None
    targets = [
        _resolve_target(path, is_checkpoint=is_checkpoint) for path in raw_targets
    ]

    rows = []
    for run_dir, checkpoint_dir in targets:
        rows.extend(
            _evaluate_run(
                run_dir,
                checkpoint_dir,
                validation_samples=args.validation_samples,
                time_bins=args.time_bins,
                seed=args.seed,
                batch_size=args.batch_size,
            )
        )

    output_path = _output_path(args, [run_dir for run_dir, _ in targets])
    _write_csv(output_path, rows)
    print("saved: {}".format(output_path))
    return output_path


if __name__ == "__main__":
    main()
