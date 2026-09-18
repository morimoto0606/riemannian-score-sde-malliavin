"""Float64 AIRM diagnostics; no eigenvalue clipping or matrix repair."""
import numpy as np
from scipy.linalg import eigvalsh


def sym(x):
    return (x + x.T) / 2


def spectral(x, fn):
    w, v = np.linalg.eigh(sym(x))
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Expected positive definite matrix')
    return (v * fn(w)) @ v.T


def airm(a, b):
    w = eigvalsh(a, b)
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Invalid generalized eigenvalues')
    return float(np.linalg.norm(np.log(w)))


def frechet_mean(samples, max_iterations=128, tolerance=1e-6):
    if max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Positive iteration limit and finite positive tolerance required')
    def report(converged, iterations, norm, reason):
        return dict(converged=converged, iterations=iterations, gradient_norm=norm,
                    tolerance=tolerance, max_iterations=max_iterations, termination_reason=reason,
                    solver_version=3, whitening='cholesky', spectrum='factor_svd')
    def quantities(mean):
        from scipy.linalg import solve_triangular
        root = np.linalg.cholesky(mean)
        logs = []
        costs = []
        for factor in factors:
            # If X=C C.T and M=L L.T, whitened X = A A.T with A=L^-1 C.
            # Its log eigenvalues are 2*log(svdvals(A)); do not form A A.T.
            relative = solve_triangular(root, factor, lower=True)
            v, singular, _ = np.linalg.svd(relative, full_matrices=False)
            if not np.isfinite(singular).all() or np.any(singular <= 0):
                raise ValueError('Invalid relative Cholesky singular values')
            logw = 2 * np.log(singular)
            logs.append((v * logw) @ v.T)
            costs.append(float(logw @ logw))
        tangent = sym(np.mean(logs, axis=0))
        return root, tangent, float(np.mean(costs)), float(np.linalg.norm(tangent))
    samples = np.asarray(samples, dtype=np.float64)
    factors = np.linalg.cholesky(samples)
    mean = samples.mean(axis=0)
    for iteration in range(max_iterations + 1):
        root, tangent, objective, norm = quantities(mean)
        if norm < tolerance:
            return mean, report(True, iteration, norm, 'gradient_tolerance')
        if iteration == max_iterations:
            return mean, report(False, iteration, norm, 'iteration_limit')
        w, v = np.linalg.eigh(tangent)
        for scale in 0.5 ** np.arange(24):
            candidate = sym(root @ ((v * np.exp(scale * w)) @ v.T) @ root.T)
            _, _, new_cost, new_norm = quantities(candidate)
            if new_cost < objective:
                mean = candidate
                break
        else:
            return mean, report(False, iteration, norm, 'line_search_stalled')


def describe(x):
    w = np.linalg.eigvalsh(x)
    if not np.isfinite(x).all() or np.any(w <= 0):
        raise ValueError('Invalid samples')
    return dict(min_eigenvalue=w[..., 0].tolist(), max_eigenvalue=w[..., -1].tolist(),
                condition_number=(w[..., -1]/w[..., 0]).tolist(),
                logdet=np.log(w).sum(axis=-1).tolist(), eigenvalues=w.tolist())


def evaluate(samples, target):
    samples = np.asarray(samples, dtype=np.float64)
    stats = describe(samples)
    target_stats = describe(target)
    mean, convergence = frechet_mean(samples)
    distances = np.array([airm(x, target) for x in samples])
    pairwise = [airm(samples[i], samples[j]) for i in range(len(samples)) for j in range(i)]
    energy = float(2*distances.mean() - np.mean(pairwise)) if pairwise else None
    return dict(
        airm_frechet_to_target=airm(mean, target) if convergence['converged'] else None,
        frechet_solver=convergence,
        airm_squared_frechet_to_target=airm(mean, target)**2 if convergence['converged'] else None,
        frobenius_frechet_to_target=float(np.linalg.norm(mean-target)) if convergence['converged'] else None,
        frobenius_arithmetic_to_target=float(np.linalg.norm(samples.mean(axis=0)-target)),
        sample_airm_to_target_mean=float(distances.mean()),
        sample_pairwise_airm_mean=float(np.mean(pairwise)) if pairwise else None,
        energy_airm_diagnostic=energy,
        generated=stats, target=target_stats,
    )


def log_vectors(matrices):
    # Flatten the full symmetric logarithm: Euclidean norm equals Frobenius norm.
    return np.array([spectral(x, np.log).reshape(-1) for x in matrices])


def distribution_metrics(generated, reference, train):
    """Pooled diagnostics; not a conditional accuracy test or proof of memorization."""
    from scipy.spatial.distance import cdist, pdist
    g, r, t = map(log_vectors, (generated, reference, train))
    reference_d2 = pdist(r, metric='sqeuclidean')
    positive = reference_d2[reference_d2 > 0]
    bandwidth2 = float(np.median(positive)) if len(positive) else 1.0
    def kernel_average(a, b):
        total = 0.
        for start in range(0,len(a),128):
            total += np.exp(-cdist(a[start:start+128],b,'sqeuclidean')/(2*bandwidth2)).sum()
        return float(total/(len(a)*len(b)))
    mmd2 = kernel_average(g,g)+kernel_average(r,r)-2*kernel_average(g,r)
    def nn(a,b):
        distances=np.concatenate([cdist(a[i:i+128],b).min(axis=1) for i in range(0,len(a),128)])
        return dict(mean=float(distances.mean()),median=float(np.median(distances)),
                    p95=float(np.percentile(distances,95)))
    return dict(metric='log-Euclidean', reference_count=len(r),generated_count=len(g),train_count=len(t),
                mmd2_biased=float(mmd2), kernel='exp(-||log(X)-log(Y)||_F^2/(2*sigma^2))',
                bandwidth_squared=bandwidth2,bandwidth_source='selected validation targets only',
                generated_to_validation_nn=nn(g,r),validation_to_generated_nn=nn(r,g),
                generated_to_train_nn=nn(g,t),validation_to_train_nn=nn(r,t))
