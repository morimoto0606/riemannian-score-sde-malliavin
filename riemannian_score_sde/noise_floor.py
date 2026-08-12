"""Pure numerical summaries for Malliavin teacher noise-floor diagnostics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from riemannian_score_sde.malliavin.rao_blackwell import (
    rao_blackwell_estimate_s2,
    s2_parallel_transport,
)


ENERGY_EPS = 1e-8


def _vectors(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("{} must be a non-empty rank-two array".format(label))
    if not np.isfinite(values).all():
        raise ValueError("{} must contain finite values".format(label))
    return values


def _weights(
    weights: Optional[np.ndarray], count: int, label: str = "weights"
) -> np.ndarray:
    if weights is None:
        return np.ones((count,), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.shape != (count,):
        raise ValueError("{} must have shape ({},)".format(label, count))
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("{} must be finite and non-negative".format(label))
    return weights


def residual_energy(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = ENERGY_EPS,
) -> Dict[str, float]:
    """Return E[w||T-m||^2], E[w||T||^2], and their ratio."""

    target = _vectors(target, "target")
    prediction = _vectors(prediction, "prediction")
    if prediction.shape != target.shape:
        raise ValueError("target and prediction must have the same shape")
    weights = _weights(weights, target.shape[0])
    squared_error = np.sum((target - prediction) ** 2, axis=-1)
    squared_target = np.sum(target**2, axis=-1)
    numerator = float(np.mean(weights * squared_error))
    denominator = float(np.mean(weights * squared_target))
    return {
        "numerator": numerator,
        "denominator": denominator,
        "ratio": float(numerator / (denominator + eps)),
    }


def comparison_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = ENERGY_EPS,
) -> Dict[str, float]:
    """Compare vector predictions with RMSE, relative RMSE, and cosine."""

    candidate = _vectors(candidate, "candidate")
    reference = _vectors(reference, "reference")
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have the same shape")
    weights = _weights(weights, candidate.shape[0])
    weight_mean = float(np.mean(weights))
    if weight_mean <= 0.0:
        raise ValueError("weights must have positive mean")

    squared_error = np.sum((candidate - reference) ** 2, axis=-1)
    squared_reference = np.sum(reference**2, axis=-1)
    mse = float(np.mean(weights * squared_error) / weight_mean)
    reference_energy = float(np.mean(weights * squared_reference) / weight_mean)
    rmse = float(np.sqrt(mse))

    candidate_norm = np.linalg.norm(candidate, axis=-1)
    reference_norm = np.linalg.norm(reference, axis=-1)
    cosine = np.sum(candidate * reference, axis=-1) / np.maximum(
        candidate_norm * reference_norm, eps
    )
    mean_cosine = float(np.mean(weights * cosine) / weight_mean)
    return {
        "rmse": rmse,
        "relative_rmse": float(rmse / (np.sqrt(reference_energy) + eps)),
        "cosine_similarity": mean_cosine,
    }


def uniform_time_edges(min_time: float, max_time: float, bins: int) -> np.ndarray:
    if not np.isfinite(min_time) or not np.isfinite(max_time):
        raise ValueError("time limits must be finite")
    if min_time >= max_time:
        raise ValueError("min_time must be less than max_time")
    if bins < 1:
        raise ValueError("bins must be positive")
    return np.linspace(min_time, max_time, bins + 1, dtype=np.float64)


def time_bin_masks(times: np.ndarray, edges: np.ndarray) -> Iterable[tuple]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    edges = np.asarray(edges, dtype=np.float64).reshape(-1)
    if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("time-bin edges must be strictly increasing")
    if not np.isfinite(times).all():
        raise ValueError("times must be finite")
    for index in range(edges.size - 1):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == edges.size - 2:
            mask = (times >= lower) & (times <= upper)
        else:
            mask = (times >= lower) & (times < upper)
        yield index, lower, upper, mask


def noise_floor_rows(
    times: np.ndarray,
    target: np.ndarray,
    marginal_prediction: np.ndarray,
    transition_prediction: np.ndarray,
    sigma_squared: np.ndarray,
    edges: np.ndarray,
) -> List[Dict[str, float]]:
    """Produce unweighted and DSM-weighted residual ratios by time bin."""

    rows = []
    for index, lower, upper, mask in time_bin_masks(times, edges):
        count = int(np.sum(mask))
        if count == 0:
            continue
        marginal = residual_energy(target[mask], marginal_prediction[mask])
        marginal_weighted = residual_energy(
            target[mask], marginal_prediction[mask], sigma_squared[mask]
        )
        transition = residual_energy(target[mask], transition_prediction[mask])
        transition_weighted = residual_energy(
            target[mask], transition_prediction[mask], sigma_squared[mask]
        )
        row = {
            "bin_index": index,
            "time_lower": lower,
            "time_upper": upper,
            "time_center": 0.5 * (lower + upper),
            "count": count,
        }
        for prefix, values in (
            ("marginal", marginal),
            ("marginal_sigma_weighted", marginal_weighted),
            ("transition", transition),
            ("transition_sigma_weighted", transition_weighted),
        ):
            for name, value in values.items():
                row["{}_{}".format(prefix, name)] = value
        rows.append(row)
    return rows


def heat_comparison_rows(
    times: np.ndarray,
    marginal_prediction: np.ndarray,
    transition_prediction: np.ndarray,
    heat_score: np.ndarray,
    sigma_squared: np.ndarray,
    edges: np.ndarray,
) -> List[Dict[str, float]]:
    """Compare both regressors with Heat overall and in each time bin."""

    rows = []

    def append_rows(scope, bin_index, lower, upper, mask):
        for regressor, prediction in (
            ("marginal_xt_t", marginal_prediction),
            ("transition_x0_xt_t", transition_prediction),
        ):
            values = comparison_metrics(prediction[mask], heat_score[mask])
            weighted = comparison_metrics(
                prediction[mask], heat_score[mask], sigma_squared[mask]
            )
            rows.append(
                {
                    "scope": scope,
                    "bin_index": bin_index,
                    "time_lower": lower,
                    "time_upper": upper,
                    "time_center": 0.5 * (lower + upper),
                    "count": int(np.sum(mask)),
                    "regressor": regressor,
                    "rmse": values["rmse"],
                    "relative_rmse": values["relative_rmse"],
                    "cosine_similarity": values["cosine_similarity"],
                    "sigma_weighted_rmse": weighted["rmse"],
                    "sigma_weighted_relative_rmse": weighted["relative_rmse"],
                    "sigma_weighted_cosine_similarity": weighted[
                        "cosine_similarity"
                    ],
                }
            )

    all_mask = np.ones(np.asarray(times).shape, dtype=bool)
    append_rows(
        "overall",
        -1,
        float(np.min(times)),
        float(np.max(times)),
        all_mask,
    )
    for index, lower, upper, mask in time_bin_masks(times, edges):
        if np.any(mask):
            append_rows("time_bin", index, lower, upper, mask)
    return rows


def heat_oracle_residual_rows(
    times: np.ndarray,
    target: np.ndarray,
    heat_score: np.ndarray,
    sigma_squared: np.ndarray,
    edges: np.ndarray,
) -> List[Dict[str, float]]:
    """Summarise Malliavin-target residuals against the fixed Heat oracle."""

    rows = []
    times = np.asarray(times, dtype=np.float64).reshape(-1)

    def append_row(scope, bin_index, lower, upper, mask):
        raw = residual_energy(target[mask], heat_score[mask])
        weighted = residual_energy(
            target[mask], heat_score[mask], sigma_squared[mask]
        )
        rows.append(
            {
                "scope": scope,
                "bin_index": bin_index,
                "time_lower": lower,
                "time_upper": upper,
                "time_center": 0.5 * (lower + upper),
                "count": int(np.sum(mask)),
                "heat_oracle_numerator": raw["numerator"],
                "heat_oracle_denominator": raw["denominator"],
                "heat_oracle_ratio": raw["ratio"],
                "heat_oracle_sigma_weighted_numerator": weighted[
                    "numerator"
                ],
                "heat_oracle_sigma_weighted_denominator": weighted[
                    "denominator"
                ],
                "heat_oracle_sigma_weighted_ratio": weighted["ratio"],
            }
        )

    all_mask = np.ones(times.shape, dtype=bool)
    append_row(
        "overall",
        -1,
        float(np.min(times)),
        float(np.max(times)),
        all_mask,
    )
    for index, lower, upper, mask in time_bin_masks(times, edges):
        if np.any(mask):
            append_row("time_bin", index, lower, upper, mask)
    return rows


def marginal_heat_oracle_residual_rows(
    times: np.ndarray,
    target: np.ndarray,
    marginal_heat_score: np.ndarray,
    sigma_squared: np.ndarray,
    edges: np.ndarray,
) -> List[Dict[str, float]]:
    """Summarise Malliavin-target residuals against the empirical Heat mixture."""

    rows = []
    times = np.asarray(times, dtype=np.float64).reshape(-1)

    def append_row(scope, bin_index, lower, upper, mask):
        raw = residual_energy(target[mask], marginal_heat_score[mask])
        weighted = residual_energy(
            target[mask], marginal_heat_score[mask], sigma_squared[mask]
        )
        rows.append(
            {
                "scope": scope,
                "bin_index": bin_index,
                "time_lower": lower,
                "time_upper": upper,
                "time_center": 0.5 * (lower + upper),
                "count": int(np.sum(mask)),
                "marginal_heat_oracle_numerator": raw["numerator"],
                "marginal_heat_oracle_denominator": raw["denominator"],
                "marginal_heat_oracle_ratio": raw["ratio"],
                "marginal_heat_oracle_sigma_weighted_numerator": weighted[
                    "numerator"
                ],
                "marginal_heat_oracle_sigma_weighted_denominator": weighted[
                    "denominator"
                ],
                "marginal_heat_oracle_sigma_weighted_ratio": weighted["ratio"],
            }
        )

    all_mask = np.ones(times.shape, dtype=bool)
    append_row(
        "overall",
        -1,
        float(np.min(times)),
        float(np.max(times)),
        all_mask,
    )
    for index, lower, upper, mask in time_bin_masks(times, edges):
        if np.any(mask):
            append_row("time_bin", index, lower, upper, mask)
    return rows


def rao_blackwell_heat_comparison_rows(
    *,
    times: np.ndarray,
    estimate: np.ndarray,
    raw_teacher: np.ndarray,
    marginal_heat_score: np.ndarray,
    sigma_squared: np.ndarray,
    effective_neighbor_count: np.ndarray,
    edges: np.ndarray,
    bandwidth_label: str,
) -> List[Dict[str, float]]:
    """Summarise Rao-Blackwell estimator quality vs marginal Heat oracle."""

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    effective_neighbor_count = np.asarray(
        effective_neighbor_count, dtype=np.float64
    ).reshape(-1)
    if effective_neighbor_count.shape != (times.shape[0],):
        raise ValueError("effective_neighbor_count must match times")

    rows = []

    def append_row(scope, bin_index, lower, upper, mask):
        metrics = comparison_metrics(estimate[mask], marginal_heat_score[mask])
        weighted_metrics = comparison_metrics(
            estimate[mask],
            marginal_heat_score[mask],
            sigma_squared[mask],
        )
        rb_error = residual_energy(marginal_heat_score[mask], estimate[mask])
        raw_error = residual_energy(marginal_heat_score[mask], raw_teacher[mask])
        rb_error_weighted = residual_energy(
            marginal_heat_score[mask],
            estimate[mask],
            sigma_squared[mask],
        )
        raw_error_weighted = residual_energy(
            marginal_heat_score[mask],
            raw_teacher[mask],
            sigma_squared[mask],
        )
        rows.append(
            {
                "scope": scope,
                "bin_index": bin_index,
                "time_lower": lower,
                "time_upper": upper,
                "time_center": 0.5 * (lower + upper),
                "count": int(np.sum(mask)),
                "bandwidth": bandwidth_label,
                "rmse": metrics["rmse"],
                "relative_rmse": metrics["relative_rmse"],
                "cosine_similarity": metrics["cosine_similarity"],
                "sigma_weighted_rmse": weighted_metrics["rmse"],
                "sigma_weighted_relative_rmse": weighted_metrics[
                    "relative_rmse"
                ],
                "sigma_weighted_cosine_similarity": weighted_metrics[
                    "cosine_similarity"
                ],
                "effective_neighbor_count_mean": float(
                    np.mean(effective_neighbor_count[mask])
                ),
                "effective_neighbor_count_min": float(
                    np.min(effective_neighbor_count[mask])
                ),
                "effective_neighbor_count_p10": float(
                    np.percentile(effective_neighbor_count[mask], 10.0)
                ),
                "raw_teacher_oracle_mse": raw_error["numerator"],
                "rao_blackwell_oracle_mse": rb_error["numerator"],
                "variance_reduction_fraction": float(
                    1.0 - rb_error["numerator"] / (raw_error["numerator"] + ENERGY_EPS)
                ),
                "raw_teacher_oracle_sigma_weighted_mse": raw_error_weighted[
                    "numerator"
                ],
                "rao_blackwell_oracle_sigma_weighted_mse": rb_error_weighted[
                    "numerator"
                ],
                "sigma_weighted_variance_reduction_fraction": float(
                    1.0
                    - rb_error_weighted["numerator"]
                    / (raw_error_weighted["numerator"] + ENERGY_EPS)
                ),
            }
        )

    all_mask = np.ones(times.shape, dtype=bool)
    append_row(
        "overall",
        -1,
        float(np.min(times)),
        float(np.max(times)),
        all_mask,
    )
    for index, lower, upper, mask in time_bin_masks(times, edges):
        if np.any(mask):
            append_row("time_bin", index, lower, upper, mask)
    return rows
