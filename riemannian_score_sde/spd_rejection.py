"""Bounded per-matrix rejection for SPD generation; never repairs samples."""
from collections import Counter
import numpy as np


def rejection_reason(x):
    if not np.isfinite(x).all():
        return "nonfinite"
    if np.max(np.abs(x - x.T)) > 1e-10 * np.max(np.abs(x)):
        return "asymmetric"
    try:
        w = np.linalg.eigvalsh(x)
        if not np.isfinite(w).all() or w[0] <= 0:
            return "not_positive_definite"
        logdet = np.linalg.slogdet(x)[1]
        with np.errstate(over="ignore", divide="ignore"):
            condition = w[-1] / w[0]
        if not np.isfinite(condition) or not np.isfinite(logdet) or logdet > np.log(np.finfo(float).max):
            return "numerical_range"
    except np.linalg.LinAlgError:
        return "linear_algebra"
    return None


def collect_spd_samples(draw, count, batch_size, max_attempts):
    if min(count, batch_size) < 1 or max_attempts < count:
        raise ValueError("Require positive count/batch_size and max_attempts >= count")
    accepted, attempted, reasons = [], 0, Counter()
    while len(accepted) < count and attempted < max_attempts:
        size = min(batch_size, count - len(accepted), max_attempts - attempted)
        batch = np.asarray(draw(size), dtype=np.float64)
        if batch.ndim != 3 or batch.shape[0] != size or batch.shape[1] != batch.shape[2] or batch.shape[1] == 0:
            raise ValueError("Sampler returned an unexpected matrix batch shape")
        attempted += size
        for x in batch:
            reason = rejection_reason(x)
            if reason is None:
                accepted.append(x)
            else:
                reasons[reason] += 1
    rejected = attempted - len(accepted)
    report = dict(policy="reject_invalid_and_refill", conditional_on_validity=True,
                  requested=count, attempted=attempted, accepted=len(accepted),
                  rejected=rejected, rejection_rate=rejected / attempted,
                  rejection_reasons=dict(reasons), max_attempts=max_attempts,
                  limit_reached=len(accepted) < count, complete=len(accepted) == count)
    return (np.stack(accepted) if accepted else None), report
