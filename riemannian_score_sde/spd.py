"""AIRM adapters for the vendored geomstats SPD geometry (not a new metric)."""

import functools

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jlinalg
import numpy as np

from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricAffine
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


def sym(x):
    return (x + jnp.swapaxes(x, -1, -2)) / 2


@functools.lru_cache(None)
def symmetric_basis(n):
    """Frobenius orthonormal basis: Eii, (Eij+Eji)/sqrt(2)."""
    basis = []
    for i in range(n):
        for j in range(i, n):
            e = np.zeros((n, n))
            e[i, j] = e[j, i] = 1 if i == j else 1 / np.sqrt(2)
            basis.append(e)
    return np.stack(basis)


def svec(x):
    return jnp.einsum("...ij,aij->...a", x, jnp.asarray(symmetric_basis(x.shape[-1]), x.dtype))


def smat(q, n):
    return jnp.einsum("...a,aij->...ij", q, jnp.asarray(symmetric_basis(n), q.dtype))


def frame(x):
    """V_a=L E_a L^T, X=L L^T; g_X(V_a,V_b)=delta_ab.

    Cholesky is smooth throughout SPD, including repeated eigenvalues. It
    differs from the symmetric-square-root frame only by an orthogonal gauge.
    """
    l = jnp.linalg.cholesky(x)
    basis = jnp.asarray(symmetric_basis(x.shape[-1]), x.dtype)
    return jnp.einsum("...ij,ajk,...lk->...ail", l, basis, l)


def from_frame(q, x):
    return sym(jnp.einsum("...a,...aij->...ij", q, frame(x)))


def to_frame(u, x):
    l = jnp.linalg.cholesky(x)
    linv = jnp.linalg.inv(l)
    return svec(linv @ u @ jnp.swapaxes(linv, -1, -2))


def _smooth_exp(u, x):
    """Equivalent AIRM exp, used only for smooth JVPs of the geomstats map.

    Cholesky congruence + Pade expm avoids eigenvector derivatives at repeated
    eigenvalues. No eigenvalue flooring or matrix regularization is applied.
    """
    l = jnp.linalg.cholesky(x)
    li = jnp.linalg.inv(l)
    return sym(l @ jlinalg.expm(sym(li @ u @ li.T)) @ l.T)


@jax.custom_jvp
def affine_exp(u, x):
    # The value is the existing SPDMetricAffine exp, with a roundoff-only sym.
    return sym(SPDMetricAffine(x.shape[-1]).exp(u, x))


@affine_exp.defjvp
def _affine_exp_jvp(primals, tangents):
    return affine_exp(*primals), jax.jvp(_smooth_exp, primals, tangents)[1]


class JitSPDMetricAffine(SPDMetricAffine):
    """Same AIRM log, omitting only the vendored Python bool positivity warning.

    SPDMatrices.logm(check_positive=True) attempts `if gs.any(...)` inside
    jit. Inputs are validated at dataset/artifact boundaries instead. The
    existing eigenvalue-function implementation and AIRM formula are retained.
    """

    @staticmethod
    def _aux_log(point, sqrt_base_point, inv_sqrt_base_point):
        near_id = sym(inv_sqrt_base_point @ point @ inv_sqrt_base_point)
        log_at_id = SymmetricMatrices.apply_func_to_eigvals(near_id, jnp.log, check_positive=False)
        return sym(sqrt_base_point @ log_at_id @ sqrt_base_point)


class AffineSPD(SPDMatrices):
    """Keep geomstats' AIRM; add the RNG and matrix-shape training contract."""

    is_spd_affine = True

    def __init__(self, n):
        super().__init__(n)
        self.metric = JitSPDMetricAffine(n)

    def exp(self, tangent_vec, base_point, **kwargs):
        tangent_vec, base_point = jnp.broadcast_arrays(tangent_vec, base_point)
        if base_point.ndim == 2:
            return affine_exp(tangent_vec, base_point)
        shape = base_point.shape
        u = jnp.broadcast_to(tangent_vec, shape).reshape((-1, self.n, self.n))
        x = base_point.reshape((-1, self.n, self.n))
        return jax.vmap(affine_exp)(u, x).reshape(shape)

    def random_normal_tangent(self, state, base_point, n_samples=1):
        state, key = random.split(state)
        z = random.normal(key, (n_samples, self.dim), dtype=base_point.dtype)
        return state, from_frame(z, base_point)


def explicit_grw_endpoint(sde, initial, terminal_time, standard_noise):
    """The common SPD predictor expressed as a differentiable function of Z."""
    times = jnp.linspace(sde.t0, terminal_time, standard_noise.shape[0] + 1)
    increments = jnp.diff(sde.beta_schedule.rescale_t(times))

    def step(x, inputs):
        z, tau = inputs
        return sde.manifold.exp(jnp.sqrt(tau) * from_frame(z, x), x), None

    return jax.lax.scan(step, initial, (standard_noise, increments))[0]


def riemannian_divergence(field, x, probe=None):
    """div_g S = tr(D_q svec(S)) -(n+1)/2 tr(X^-1 S).

    q=svec(X) is an orthonormal symmetric *coordinate* basis, not moving-frame
    coefficients. sqrt(det g(q)) = const * det(X)^(-(n+1)/2).
    probe=None computes the exact 15D trace; otherwise E[probe probe^T]=I.
    """
    n = x.shape[-1]
    q = svec(x)
    coordinate_field = lambda q_: svec(field(smat(q_, n)))
    value, linear = jax.linearize(coordinate_field, q)
    if probe is None:
        coordinate_div = jnp.trace(jax.vmap(linear)(jnp.eye(q.shape[0], dtype=q.dtype)))
    else:
        # The shared Rademacher helper returns float32 even in x64 runs.
        # JAX linearize requires tangent and primal dtypes to agree.
        probe = jnp.asarray(probe, dtype=q.dtype)
        coordinate_div = jnp.vdot(probe, linear(probe))
    volume_term = -(n + 1) / 2 * jnp.trace(jnp.linalg.solve(x, smat(value, n)))
    return coordinate_div + volume_term


def get_spd_div_fn(func, hutchinson_type, n):
    if hutchinson_type not in ("None", "Gaussian", "Rademacher"):
        raise ValueError(f"Unsupported SPD divergence estimator: {hutchinson_type}")

    def divergence(y, t, context, epsilon):
        x = y.reshape((-1, n, n))

        def single(point, time, ctx, probe):
            def field(p):
                c = None if ctx is None else ctx[None]
                return func(p[None], time[None], c).reshape((n, n))
            return riemannian_divergence(field, point, probe)

        probes = None if hutchinson_type == "None" else epsilon.reshape((len(x), -1))
        if probes is not None and probes.shape[-1] == n * n:
            # Other generic callers may supply iid ambient matrix probes.
            # Orthonormal symmetric projection still has covariance I_dim.
            probes = svec(probes.reshape((-1, n, n)))
        if probes is not None and probes.shape[-1] != n * (n + 1) // 2:
            raise ValueError("SPD divergence probes must have 15 symmetric coordinates (or 5x5 ambient shape)")
        return jax.vmap(single, in_axes=(0, 0, None if context is None else 0,
                                       None if probes is None else 0))(x, t, context, probes)

    return divergence
