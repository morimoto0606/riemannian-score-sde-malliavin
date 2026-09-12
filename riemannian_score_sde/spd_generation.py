"""Small, separate sample artifact path: no compact-manifold likelihood/plots."""

import json
import hashlib
from pathlib import Path

import jax
import numpy as np

from riemannian_score_sde.spd_rejection import collect_spd_samples
from omegaconf import OmegaConf


def spd_summary(samples):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != x.shape[2] or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite SPD matrix samples")
    symmetry = np.max(np.abs(x - x.swapaxes(-1, -2)))
    if symmetry > 1e-10 * np.max(np.abs(x)):
        raise ValueError("Non-symmetric generated samples")
    eigenvalues = np.linalg.eigvalsh(x)
    sign, logdet = np.linalg.slogdet(x)
    if eigenvalues.min() <= 0 or np.any(sign != 1) or not np.isfinite(logdet).all():
        raise ValueError("Non-SPD generated samples: investigate integration/precision; no repair applied")
    condition = eigenvalues[:, -1] / eigenvalues[:, 0]
    if not np.isfinite(condition).all():
        raise ValueError("Nonfinite SPD condition number")
    determinant_min, determinant_max = np.exp(logdet.min()), np.exp(logdet.max())
    if not np.isfinite(determinant_max):
        raise ValueError("SPD determinant overflow; inspect generated scale before reporting metrics")
    return {"count": len(x), "symmetry_max_abs": float(symmetry),
            "non_spd_count": 0,
            "minimum_eigenvalue": float(eigenvalues.min()),
            "determinant_min": float(determinant_min),
            "determinant_max": float(determinant_max),
            "logdet_min": float(logdet.min()), "logdet_max": float(logdet.max()),
            "condition_number_max": float(condition.max()), "regularization_eps": 0.0}


def save_generation(cfg, pushforward, model, train_state):
    count, batch_size, steps = (int(cfg.generation[k]) for k in ("count", "batch_size", "steps"))
    if min(count, batch_size, steps) < 1:
        raise ValueError("Positive generation count, batch_size and steps required")
    sampler = pushforward.get_sampler((model, train_state.params_ema, train_state.model_state),
                                      train=False, N=steps, eps=cfg.eps, predictor="GRW")
    # Cache the combined terminal+reverse graph across equal-sized batches.
    sampler = jax.jit(sampler, static_argnums=(1,))
    key = jax.random.PRNGKey(int(cfg.generation.seed))
    output = Path(str(cfg.generated_samples_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Use a new generation output path: " + str(output))

    label = cfg.generation.get("class_label", None)
    if hasattr(pushforward.base, "sample_conditioned") and label not in (0, 1):
        raise ValueError("EEG generation.class_label must be 0 or 1")

    def draw(size):
        nonlocal key
        key, batch_key = jax.random.split(key)
        context = None if label is None else jax.numpy.tile(jax.numpy.eye(2)[int(label)], (size, 1))
        return np.asarray(sampler(batch_key, (size,), context))

    configured_limit = cfg.generation.get("max_attempts", None)
    max_attempts = count * 2 if configured_limit is None else int(configured_limit)
    samples, rejection = collect_spd_samples(draw, count, batch_size, max_attempts)
    report = {"spd": spd_summary(samples) if samples is not None else None,
              "sampling": rejection, "class_label": label, "experiment": str(cfg.experiment),
              "terminal_law": "class_conditional_empirical_train_forward_GRW" if label is not None else "empirical_train_forward_GRW", "reverse_steps": steps,
              "forward_steps": int(pushforward.sde.N), "reverse_end_time": float(cfg.eps),
              "generation_seed": int(cfg.generation.seed), "training_seed": int(cfg.seed),
              "checkpoint_step": int(train_state.step),
              "dataset_seed": int(cfg.dataset.dataset_seed),
              "training_data_sha256": hashlib.sha256(np.ascontiguousarray(
                  pushforward.sde.limiting.data, dtype=np.float64).tobytes()).hexdigest(),
              "loss": OmegaConf.to_container(cfg.loss, resolve=True)}
    metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    if not rejection["complete"]:
        raise RuntimeError("SPD generation attempt limit reached; see " + str(metadata_path))
    np.save(output, samples)
