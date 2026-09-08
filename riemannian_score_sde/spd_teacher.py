"""Discrete Malliavin integration by parts for the AIRM SPD GRW.

Uses the existing noise-space Jacobian, covariance, and exact/Hutchinson
divergence machinery. The state frame and its volume divergence are SPD
specific; see docs/spd_finance.md for the sign and frame derivations.
"""

import jax
import jax.numpy as jnp
import jax.random as random

from riemannian_score_sde.spd import (
    explicit_grw_endpoint, from_frame, symmetric_basis, to_frame,
)
from riemannian_score_sde.teachers import (
    MalliavinTeacher, compute_endpoint_jacobian, malliavin_covariance,
    sample_upstream_grw_standard_noise,
)


def cholesky_frame_divergence(n, dtype):
    """div_g(L E_ii L^T)=(n-1-2*i)/2; off-diagonal fields have div=0.

    L is lower Cholesky. At I, div_E V_ii=n-i in svec coordinates;
    grad log volume paired with V_ii is -(n+1)/2. Lower-triangular
    congruence is an AIRM isometry and transports this frame, so the result
    holds at every X. This is not the zero divergence of SO(3).
    """
    return jnp.asarray([sum(e[i, i] * (n - 1 - 2 * i) / 2 for i in range(n))
                        for e in symmetric_basis(n)], dtype=dtype)


class SPDMalliavinTeacher(MalliavinTeacher):
    def __init__(self, covariance_regularization=1e-8, **kwargs):
        super().__init__(covariance_regularization=covariance_regularization, **kwargs)
        if self.rb_enabled:
            raise ValueError("SPD Rao-Blackwell smoothing is not implemented")

    def _single_spd_sample(self, sde, initial, terminal_time, noise, divergence_key):
        shape = noise.shape
        z = noise.reshape(-1)
        n, dim = sde.manifold.n, sde.manifold.dim

        def endpoint_fn(z_):
            return explicit_grw_endpoint(sde, initial, terminal_time, z_.reshape(shape))

        def covering_state(z_):
            endpoint = endpoint_fn(z_)
            derivative = compute_endpoint_jacobian(endpoint_fn, z_)  # (n,n,K*dim)
            tangent_columns = jnp.moveaxis(derivative, -1, 0)
            jacobian = jax.vmap(lambda u: to_frame(u, endpoint))(tangent_columns).T
            covariance = malliavin_covariance(jacobian)
            # Relative ridge on Malliavin covariance only, NOT on SPD data.
            # Ridge>0 biases the IBP target, explicitly recorded in configs.
            ridge = self.covariance_regularization * jnp.trace(covariance) / dim
            inverse = jnp.linalg.solve(covariance + ridge * jnp.eye(dim, dtype=z_.dtype),
                                       jnp.eye(dim, dtype=z_.dtype))
            return endpoint, jacobian.T @ inverse

        endpoint, covering = covering_state(z)
        divergence = self._compute_divergence(lambda z_: covering_state(z_)[1], z, divergence_key)
        # E[V_a f(X)] = E[f(X) delta(U_a)]
        #             = -E[f(X)(<score,V_a> + div_g V_a)].
        skorokhod = covering.T @ z - divergence
        coordinates = -skorokhod - cholesky_frame_divergence(n, z.dtype)
        return endpoint, from_frame(coordinates, endpoint)

    def sample_and_score(self, rng, sde, y_0, t):
        if not getattr(sde.manifold, "is_spd_affine", False):
            raise ValueError("SPDMalliavinTeacher requires the AIRM SPD GRW")
        noise = sample_upstream_grw_standard_noise(
            rng, y_0.shape[0], sde.N, noise_dim=sde.manifold.dim, dtype=y_0.dtype)
        keys = random.split(random.fold_in(rng, 0x4D414C4C), y_0.shape[0])
        endpoints, scores = jax.vmap(lambda x, time, z, key: self._single_spd_sample(
            sde, x, time, z, key))(y_0, t, noise, keys)
        return jax.lax.stop_gradient(endpoints), jax.lax.stop_gradient(scores)
