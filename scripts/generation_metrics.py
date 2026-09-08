"""Geometry-agnostic sample metrics driven by a pairwise distance function."""

from __future__ import annotations

from typing import Callable

import numpy as np


PairwiseDistance = Callable[[np.ndarray, np.ndarray], np.ndarray]


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def nearest_neighbor_distances(
    source: np.ndarray,
    reference: np.ndarray,
    chunk_size: int,
    pairwise_distance: PairwiseDistance,
) -> np.ndarray:
    nearest = []
    for start in range(0, len(source), chunk_size):
        distances = pairwise_distance(source[start : start + chunk_size], reference)
        nearest.append(np.min(distances, axis=1))
    return np.concatenate(nearest)


def _kernel_sum(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
    chunk_size: int,
    pairwise_distance: PairwiseDistance,
) -> float:
    total = 0.0
    for start in range(0, len(left), chunk_size):
        distances = pairwise_distance(left[start : start + chunk_size], right)
        total += float(np.exp(-(distances**2) / (2.0 * sigma**2)).sum())
    return total


def geodesic_rbf_mmd(
    real: np.ndarray,
    generated: np.ndarray,
    sigma: float,
    chunk_size: int,
    pairwise_distance: PairwiseDistance,
) -> float:
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("MMD requires at least two real and generated samples")
    real_sum = _kernel_sum(real, real, sigma, chunk_size, pairwise_distance) - len(real)
    generated_sum = _kernel_sum(
        generated, generated, sigma, chunk_size, pairwise_distance
    ) - len(generated)
    cross_sum = _kernel_sum(
        real, generated, sigma, chunk_size, pairwise_distance
    )
    return (
        real_sum / (len(real) * (len(real) - 1))
        + generated_sum / (len(generated) * (len(generated) - 1))
        - 2.0 * cross_sum / (len(real) * len(generated))
    )
