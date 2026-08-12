"""S2 kernel Rao--Blackwell estimators for Malliavin score targets."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


ENERGY_EPS = 1e-8


def s2_parallel_transport(
    source_points: np.ndarray,
    target_point: np.ndarray,
    source_vectors: np.ndarray,
    eps: float = ENERGY_EPS,
) -> np.ndarray:
    """Parallel transport vectors from source points to one S2 target point."""

    source_points = np.asarray(source_points, dtype=np.float64)
    target_point = np.asarray(target_point, dtype=np.float64).reshape(-1)
    source_vectors = np.asarray(source_vectors, dtype=np.float64)
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("source_points must have shape [n, 3]")
    if target_point.shape != (3,):
        raise ValueError("target_point must have shape [3]")
    if source_vectors.shape != source_points.shape:
        raise ValueError("source_vectors must match source_points shape")
    dot = np.sum(source_points * target_point[None, :], axis=-1)
    denominator = np.maximum(1.0 + dot, eps)
    coefficient = np.sum(source_vectors * target_point[None, :], axis=-1)
    return source_vectors - (coefficient / denominator)[:, None] * (
        source_points + target_point[None, :]
    )


def _numpy_vectors(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("{} must be a non-empty rank-two array".format(label))
    if not np.isfinite(values).all():
        raise ValueError("{} must contain finite values".format(label))
    return values


def rao_blackwell_estimate_s2(
    source_endpoints: np.ndarray,
    source_times: np.ndarray,
    source_targets: np.ndarray,
    query_endpoints: np.ndarray,
    query_times: np.ndarray,
    *,
    spatial_bandwidth: float,
    time_bandwidth: float,
    source_chunk_size: int,
    self_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Estimate E[T | Xt=x, t] via chunked S2 geodesic kernel regression."""

    source_endpoints = _numpy_vectors(source_endpoints, "source_endpoints")
    source_targets = _numpy_vectors(source_targets, "source_targets")
    query_endpoints = _numpy_vectors(query_endpoints, "query_endpoints")
    if source_endpoints.shape[1] != 3 or query_endpoints.shape[1] != 3:
        raise ValueError("source_endpoints and query_endpoints must be S2 vectors")
    if source_targets.shape != source_endpoints.shape:
        raise ValueError("source_targets must match source_endpoints shape")
    source_times = np.asarray(source_times, dtype=np.float64).reshape(-1)
    query_times = np.asarray(query_times, dtype=np.float64).reshape(-1)
    if source_times.shape != (source_endpoints.shape[0],):
        raise ValueError("source_times has incompatible shape")
    if query_times.shape != (query_endpoints.shape[0],):
        raise ValueError("query_times has incompatible shape")
    if spatial_bandwidth <= 0.0 or time_bandwidth <= 0.0:
        raise ValueError("bandwidths must be positive")
    if source_chunk_size < 1:
        raise ValueError("source_chunk_size must be positive")

    if self_indices is not None:
        self_indices = np.asarray(self_indices, dtype=np.int64).reshape(-1)
        if self_indices.shape != (query_endpoints.shape[0],):
            raise ValueError("self_indices must have one index per query")

    estimates = np.zeros_like(query_endpoints)
    effective_counts = np.zeros((query_endpoints.shape[0],), dtype=np.float64)
    for query_index in range(query_endpoints.shape[0]):
        query_x = query_endpoints[query_index]
        query_t = query_times[query_index]
        numerator = np.zeros((3,), dtype=np.float64)
        sum_weights = 0.0
        sum_squared_weights = 0.0
        for start in range(0, source_endpoints.shape[0], source_chunk_size):
            stop = min(start + source_chunk_size, source_endpoints.shape[0])
            source_x = source_endpoints[start:stop]
            source_t = source_times[start:stop]
            source_v = source_targets[start:stop]
            cosine = np.clip(
                np.sum(source_x * query_x[None, :], axis=-1), -1.0, 1.0
            )
            geodesic = np.arccos(cosine)
            time_delta = source_t - query_t
            log_weights = -0.5 * (geodesic / spatial_bandwidth) ** 2
            log_weights -= 0.5 * (time_delta / time_bandwidth) ** 2
            weights = np.exp(np.clip(log_weights, -745.0, 50.0))
            if self_indices is not None:
                self_index = int(self_indices[query_index])
                if start <= self_index < stop:
                    weights[self_index - start] = 0.0
            transported = s2_parallel_transport(source_x, query_x, source_v)
            numerator += np.sum(weights[:, None] * transported, axis=0)
            sum_weights += float(np.sum(weights))
            sum_squared_weights += float(np.sum(weights**2))
        if sum_weights > 0.0:
            estimates[query_index] = numerator / sum_weights
        effective_counts[query_index] = (
            (sum_weights**2) / max(sum_squared_weights, ENERGY_EPS)
            if sum_weights > 0.0
            else 0.0
        )
    return {
        "estimate": estimates,
        "effective_neighbor_count": effective_counts,
    }


def rao_blackwell_estimate_s2_batch(
    endpoints,
    times,
    targets,
    *,
    spatial_bandwidth: float,
    time_bandwidth: float,
    eps: float = ENERGY_EPS,
):
    """JAX leave-one-out RB estimate using the current training minibatch.

    Every source target is parallel transported to each query endpoint before
    kernel averaging.  Removing the diagonal prevents the path from averaging
    itself, matching the diagnostic estimator's ``self_indices`` behaviour.
    """

    import jax.numpy as jnp

    if spatial_bandwidth <= 0.0 or time_bandwidth <= 0.0:
        raise ValueError("bandwidths must be positive")
    if endpoints.ndim != 2 or endpoints.shape[-1] != 3:
        raise ValueError("endpoints must have shape [batch, 3]")
    if targets.shape != endpoints.shape:
        raise ValueError("targets must match endpoints shape")
    if times.ndim != 1 or times.shape[0] != endpoints.shape[0]:
        raise ValueError("times must have shape [batch]")

    # Rows are queries and columns are sources.
    cosine = jnp.clip(endpoints @ endpoints.T, -1.0, 1.0)
    geodesic = jnp.arccos(cosine)
    time_delta = times[None, :] - times[:, None]
    log_weights = -0.5 * (geodesic / spatial_bandwidth) ** 2
    log_weights -= 0.5 * (time_delta / time_bandwidth) ** 2
    diagonal = jnp.eye(endpoints.shape[0], dtype=bool)
    log_weights = jnp.where(diagonal, -jnp.inf, log_weights)

    # Stable row normalisation. For a singleton minibatch, fall back to the
    # raw target rather than introducing a NaN into training.
    row_max = jnp.max(log_weights, axis=1, keepdims=True)
    row_max = jnp.where(jnp.isfinite(row_max), row_max, 0.0)
    weights = jnp.exp(log_weights - row_max)
    weights = jnp.where(diagonal, 0.0, weights)
    sum_weights = jnp.sum(weights, axis=1)

    source_x = endpoints[None, :, :]
    query_x = endpoints[:, None, :]
    source_v = targets[None, :, :]
    denominator = jnp.maximum(1.0 + cosine, eps)
    coefficient = jnp.sum(source_v * query_x, axis=-1)
    transported = source_v - (coefficient / denominator)[..., None] * (
        source_x + query_x
    )
    numerator = jnp.sum(weights[..., None] * transported, axis=1)
    estimate = numerator / jnp.maximum(sum_weights, eps)[..., None]
    estimate = jnp.where((sum_weights > eps)[..., None], estimate, targets)

    sum_squared_weights = jnp.sum(weights**2, axis=1)
    effective_count = sum_weights**2 / jnp.maximum(sum_squared_weights, eps)
    return estimate, effective_count
