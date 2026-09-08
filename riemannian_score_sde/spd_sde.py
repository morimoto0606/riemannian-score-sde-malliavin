"""Finite-time AIRM Brownian diffusion and an explicit empirical terminal law."""

import jax.numpy as jnp
import jax.random as random

from riemannian_score_sde.sde import Brownian
from riemannian_score_sde.sampling import get_pc_sampler


class EmpiricalForwardTerminal:
    """q_T^GRW = average over TRAIN samples of the forward GRW transition.

    This is not a uniform/stationary prior or a data-free Gaussian prior.
    A fresh training draw and fresh forward noise are used for every sample.
    """

    def __init__(self, sde):
        self.sde = sde
        self.data = None

    def sample(self, rng, shape):
        if self.data is None:
            raise ValueError("Bind only the training split before sampling the SPD terminal law")
        index_key, forward_key = random.split(rng)
        index = random.randint(index_key, (shape[0],), 0, self.data.shape[0])
        return self.sde.marginal_sample(forward_key, self.data[index], jnp.full((shape[0],), self.sde.tf))

    def log_prob(self, x):
        raise NotImplementedError("Empirical forward terminal density is not evaluated; use sample metrics")


class SPDBrownian(Brownian):
    def __init__(self, manifold, beta_schedule, N=16):
        if not getattr(manifold, "is_spd_affine", False):
            raise ValueError("SPDBrownian requires AffineSPD")
        if beta_schedule.t0 != 0 or beta_schedule.tf != 1:
            raise ValueError("This adapter requires the repository's normalized time interval [0,1]")
        if N < 1 or min(beta_schedule.beta_0, beta_schedule.beta_f) <= 0:
            raise ValueError("Positive step count and beta schedule required")
        super().__init__(manifold, beta_schedule, N)
        self.limiting = EmpiricalForwardTerminal(self)

    def bind_training_data(self, data):
        self.limiting.data = jnp.asarray(data)

    def drift(self, x, t):
        # GRW includes the Levi-Civita/connection drift geometrically. The
        # inherited scalar diffusion flags refer to intrinsic beta*Delta_g,
        # not to a spatially constant diffusion matrix in ambient coordinates.
        return jnp.zeros_like(x)

    def marginal_prob(self, x, t):
        # sqrt(tau) is a score preconditioner, not an exact SPD Gaussian std.
        return x, jnp.sqrt(self.beta_schedule.rescale_t(t))

    def marginal_sample(self, rng, x, t, return_hist=False):
        return get_pc_sampler(self, self.N, predictor="GRW", eps=0., return_hist=return_hist)(rng, x, tf=t)
