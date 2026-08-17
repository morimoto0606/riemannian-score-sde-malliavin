"""Dependency-light aggregation helpers for score diagnostics."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def uniform_time_edges(time_bins: int) -> np.ndarray:
    """Return uniform diagnostic bin edges over the full diffusion interval."""

    if time_bins < 1:
        raise ValueError("time_bins must be positive")
    return np.linspace(0.0, 1.0, time_bins + 1, dtype=np.float64)


def score_error_rows(
    times: Sequence[float],
    teacher_energy: Sequence[float],
    model_error: Sequence[float],
    cosine_similarity: Sequence[float],
    *,
    time_bins: int,
    metadata: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Aggregate unweighted per-sample score errors overall and by time."""

    arrays = tuple(
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in (times, teacher_energy, model_error, cosine_similarity)
    )
    if not arrays[0].size:
        raise ValueError("at least one sample is required")
    if len({array.size for array in arrays}) != 1:
        raise ValueError("all per-sample arrays must have the same length")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("per-sample diagnostics must be finite")

    times_array, teacher_array, error_array, cosine_array = arrays
    if np.any(teacher_array < -1e-10) or np.any(error_array < -1e-10):
        raise ValueError("squared-norm diagnostics must be non-negative")
    teacher_array = np.maximum(teacher_array, 0.0)
    error_array = np.maximum(error_array, 0.0)
    edges = uniform_time_edges(time_bins)
    if np.any(times_array < edges[0]) or np.any(times_array > edges[-1]):
        raise ValueError("times must lie in [0, 1]")

    shared = dict(metadata or {})

    def make_row(
        label: str,
        bin_index: int,
        start: float,
        end: float,
        mask: np.ndarray,
    ) -> dict[str, object]:
        count = int(np.count_nonzero(mask))
        row = {
            **shared,
            "scope": label,
            "bin_index": bin_index,
            "time_start": start,
            "time_end": end,
            "count": count,
        }
        if count == 0:
            row.update(
                {
                    "teacher_energy": np.nan,
                    "model_error": np.nan,
                    "relative_error": np.nan,
                    "RMSE": np.nan,
                    "cosine_similarity": np.nan,
                }
            )
            return row

        mean_teacher = float(np.mean(teacher_array[mask]))
        mean_error = float(np.mean(error_array[mask]))
        row.update(
            {
                "teacher_energy": mean_teacher,
                "model_error": mean_error,
                "relative_error": (
                    mean_error / mean_teacher if mean_teacher > 0.0 else np.nan
                ),
                "RMSE": float(np.sqrt(mean_error)),
                "cosine_similarity": float(np.mean(cosine_array[mask])),
            }
        )
        return row

    all_samples = np.ones(times_array.shape, dtype=bool)
    rows = [make_row("overall", -1, edges[0], edges[-1], all_samples)]
    for bin_index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_index == time_bins - 1:
            mask = (times_array >= start) & (times_array <= end)
        else:
            mask = (times_array >= start) & (times_array < end)
        rows.append(make_row("time_bin", bin_index, start, end, mask))
    return rows
