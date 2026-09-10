"""Conditional score teachers for the upstream Riemannian DSM loss.

The network, loss weighting, optimiser, EMA, and reverse sampler do not live
in this module.  A teacher only produces an online forward endpoint and a
conditional score target at that endpoint.
"""

import math
from typing import Callable, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
import jax.random as random

from riemannian_score_sde.malliavin.rao_blackwell import (
    rao_blackwell_estimate_s2_batch,
)


Array = jnp.ndarray
DivergenceFn = Callable[[Callable[[Array], Array], Array], Array]


def rb_time_alpha_schedule(
    times: Array,
    rb_alpha: float,
    rb_time_center: float,
    rb_time_scale: float,
) -> Array:
    """Return the sigmoid time-dependent Rao--Blackwell mixing weight."""

    return rb_alpha * jax.nn.sigmoid(
        rb_time_scale * (times - rb_time_center)
    )


def rb_alpha_schedule(
    times: Array,
    rb_alpha: float,
    rb_time_dependent: bool,
    rb_time_center: float,
    rb_time_scale: float,
) -> Array:
    """Return either a fixed or sigmoid time-dependent RB mixing weight."""

    if not rb_time_dependent:
        return jnp.full_like(times, rb_alpha)
    return rb_time_alpha_schedule(
        times,
        rb_alpha,
        rb_time_center,
        rb_time_scale,
    )


class ConditionalScoreTeacher(Protocol):
    """Interface consumed by ``get_dsm_loss_fn``."""

    def sample_and_score(self, rng, sde, y_0: Array, t: Array) -> Tuple[Array, Array]:
        """Return an online endpoint and its conditional score target."""


class HeatTeacher:
    """Wrap the unchanged upstream heat-kernel target path."""

    def __init__(self, n_max: int = 5, thresh: float = 0.5):
        self.n_max = n_max
        self.thresh = thresh

    def sample_and_score(self, rng, sde, y_0: Array, t: Array) -> Tuple[Array, Array]:
        y_t = sde.marginal_sample(rng, y_0, t)
        score_target = self.score_at_endpoint(sde, y_0, y_t, t)
        return y_t, score_target

    def score_at_endpoint(self, sde, y_0: Array, y_t: Array, t: Array) -> Array:
        """Evaluate the unchanged Heat target at a caller-provided endpoint."""

        return sde.grad_marginal_log_prob(
            y_0,
            y_t,
            t,
            n_max=self.n_max,
            thresh=self.thresh,
        )[1]


class SpectrumTeacher:
    """S2 truncated spectral score at every time, without a Varadhan branch.

    Uses the vendored heat kernel directly. Nonpositive truncated kernels
    remain nonfinite; no density floor or alternative target is substituted.
    """

    def __init__(self, n_max: int = 5):
        if not isinstance(n_max, int) or isinstance(n_max, bool) or n_max < 1:
            raise ValueError("SpectrumTeacher requires a positive integer n_max")
        self.n_max = n_max

    def sample_and_score(self, rng, sde, y_0: Array, t: Array) -> Tuple[Array, Array]:
        y_t = sde.marginal_sample(rng, y_0, t)
        return y_t, self.score_at_endpoint(sde, y_0, y_t, t)

    def score_at_endpoint(self, sde, y_0: Array, y_t: Array, t: Array) -> Array:
        from geomstats.geometry.hypersphere import Hypersphere

        if not isinstance(sde.manifold, Hypersphere) or sde.manifold.dim != 2:
            raise ValueError("SpectrumTeacher currently supports S2 only")
        if not jax.config.read("jax_enable_x64"):
            raise ValueError("SpectrumTeacher requires JAX_ENABLE_X64=true")
        tau = sde.beta_schedule.rescale_t(t)

        def log_kernel(x0, x, time):
            return sde.manifold._log_heat_kernel(
                x0, x, time, n_max=self.n_max
            ).reshape(())

        ambient_score = jax.vmap(jax.grad(log_kernel, argnums=1))(y_0, y_t, tau)
        return sde.manifold.to_tangent(ambient_score, y_t)


class VaradhanTeacher:
    """Preserve the existing ``n_max <= -1`` DSM target path."""

    def sample_and_score(self, rng, sde, y_0: Array, t: Array) -> Tuple[Array, Array]:
        y_t = sde.marginal_sample(rng, y_0, t)
        score_target = self.score_at_endpoint(sde, y_0, y_t, t)
        return y_t, score_target

    def score_at_endpoint(self, sde, y_0: Array, y_t: Array, t: Array) -> Array:
        """Evaluate the Varadhan target at a caller-provided endpoint."""

        return sde.varhadan_exp(
            y_0,
            y_t,
            jnp.zeros_like(t),
            t,
        )[1]


def s2_projected_coordinate_fields(endpoint: Array) -> Array:
    """Return ``V_j(x) = (I - xx^T)e_j`` as columns of a 3-by-3 matrix."""

    eye = jnp.eye(3, dtype=endpoint.dtype)
    return eye - jnp.outer(endpoint, endpoint)


def s2_projected_coordinate_field_divergence(endpoint: Array) -> Array:
    """Return the S2 volume divergences ``div(V_j) = -2 x_j``."""

    return -2.0 * endpoint


def s2_tangent_basis(endpoint: Array) -> Array:
    """Construct an orthonormal 3-by-2 basis of ``T_endpoint S2``.

    The least-aligned coordinate axis is selected using JAX operations.  The
    branch boundary has measure zero under the forward diffusion, while the
    selected branch remains differentiable with respect to the endpoint.
    """

    reference_index = jnp.argmin(jnp.abs(endpoint))
    reference = jax.nn.one_hot(reference_index, 3, dtype=endpoint.dtype)
    first = jnp.cross(endpoint, reference)
    first = first / jnp.linalg.norm(first)
    second = jnp.cross(endpoint, first)
    second = second / jnp.linalg.norm(second)
    return jnp.stack((first, second), axis=-1)


def compute_endpoint_jacobian(endpoint_fn: Callable[[Array], Array], z: Array) -> Array:
    """Compute the discrete noise derivative ``D_z endpoint``.

    This is the sole explicit ``jacrev`` site.  Despite the function name,
    the result is not the flow Jacobian ``J_{t,s} = dX_t / dX_s`` and is not
    ``dX_t / dX_0``.  It differentiates the endpoint with respect to the
    flattened standard Gaussian GRW increments.
    """

    return jax.jacrev(endpoint_fn)(z)


def compute_divergence_exact(vector_field_fn: Callable[[Array], Array], z: Array) -> Array:
    r"""Compute exact noise-space divergence using basis-direction JVPs.

    ``vector_field_fn(z)`` must have shape ``[noise_dim, n_fields]``.  The
    implementation never constructs its full Jacobian.  ``jax.linearize``
    caches the primal linearisation, and ``lax.fori_loop`` accumulates only
    the diagonal entries required by the trace.
    """

    vector_field, pushforward = jax.linearize(vector_field_fn, z)
    noise_dim = z.shape[0]
    initial = jnp.zeros((vector_field.shape[-1],), dtype=vector_field.dtype)

    def accumulate(index, divergence):
        direction = jax.nn.one_hot(index, noise_dim, dtype=z.dtype)
        directional_derivative = pushforward(direction)
        return divergence + directional_derivative[index]

    return jax.lax.fori_loop(0, noise_dim, accumulate, initial)


def compute_divergence_hutchinson(
    vector_field_fn: Callable[[Array], Array],
    z: Array,
    rng: Array,
    n_probes: int = 1,
    noise_type: str = "rademacher",
) -> Array:
    r"""Estimate noise-space divergence with probe-direction JVPs.

    For each output field this estimates ``tr(dU/dz)`` as
    ``E[e^T (dU/dz) e]``.  A single linearisation is shared across probes;
    neither a full Jacobian nor ``jacrev`` is used here.
    """

    if n_probes < 1:
        raise ValueError("n_probes must be positive")
    noise_type = noise_type.lower()
    if noise_type not in ("rademacher", "gaussian"):
        raise ValueError("noise_type must be 'rademacher' or 'gaussian'")

    _, pushforward = jax.linearize(vector_field_fn, z)
    probe_keys = random.split(rng, n_probes)

    def estimate(key):
        if noise_type == "rademacher":
            probe = random.rademacher(key, z.shape, dtype=z.dtype)
        else:
            probe = random.normal(key, z.shape, dtype=z.dtype)
        directional_derivative = pushforward(probe)
        return jnp.einsum("i,ij->j", probe, directional_derivative)

    return jnp.mean(jax.vmap(estimate)(probe_keys), axis=0)


def exact_divergence_jvp_count(batch_size: int, n_steps: int, updates: int) -> dict:
    """Return the basis-JVP count implied by an exact-divergence run."""

    if batch_size < 1 or n_steps < 1 or updates < 1:
        raise ValueError("batch_size, n_steps, and updates must be positive")
    per_sample = 3 * n_steps
    per_update = batch_size * per_sample
    return {
        "noise_dimension": per_sample,
        "jvps_per_update": per_update,
        "jvps_total": updates * per_update,
    }


def hutchinson_divergence_jvp_count(
    batch_size: int,
    n_steps: int,
    updates: int,
    n_probes: int,
) -> dict:
    """Return the probe-JVP count implied by a Hutchinson-divergence run."""

    if batch_size < 1 or n_steps < 1 or updates < 1 or n_probes < 1:
        raise ValueError(
            "batch_size, n_steps, updates, and n_probes must be positive"
        )
    return {
        "noise_dimension": 3 * n_steps,
        "probes_per_sample": n_probes,
        "jvps_per_update": batch_size * n_probes,
        "jvps_total": updates * batch_size * n_probes,
    }


def sample_upstream_grw_standard_noise(rng, batch_size: int, n_steps: int,
                                      noise_dim: int = 3, dtype=None) -> Array:
    """Reproduce the random keys consumed by the upstream GRW predictor.

    ``get_pc_sampler`` splits once for its no-op corrector and once for the
    predictor.  Geomstats then splits the predictor key once before drawing
    the ambient standard Gaussian.  Retaining this sequence lets the explicit
    endpoint map be checked against the native upstream endpoint.
    """

    def sample_step(state, _):
        state, _ = random.split(state)  # upstream no-op corrector key
        state, predictor_rng = random.split(state)
        _, gaussian_rng = random.split(predictor_rng)  # geomstats random.normal
        kwargs = {} if dtype is None else {"dtype": dtype}
        noise = random.normal(gaussian_rng, (batch_size, noise_dim), **kwargs)
        return state, noise

    _, noises = jax.lax.scan(sample_step, rng, xs=None, length=n_steps)
    return jnp.swapaxes(noises, 0, 1)


def upstream_s2_grw_endpoint(
    sde,
    initial_point: Array,
    terminal_time: Array,
    standard_noise: Array,
    sampler_eps: float = 1e-3,
) -> Array:
    """Evaluate one explicit-noise endpoint with upstream GRW discretisation."""

    n_steps = standard_noise.shape[0]
    start_time = sde.t0 + sampler_eps
    timesteps = jnp.linspace(start_time, terminal_time, num=n_steps, endpoint=True)
    dt = (terminal_time - start_time) / n_steps

    def step(endpoint, inputs):
        time, ambient_noise = inputs
        tangent_noise = sde.manifold.to_tangent(ambient_noise, endpoint)
        diffusion = jnp.sqrt(sde.beta_schedule.beta_t(time))
        tangent_increment = jnp.einsum(
            "...,...i,...->...i",
            diffusion,
            tangent_noise,
            jnp.sqrt(jnp.abs(dt)),
        )
        endpoint = sde.manifold.exp(
            tangent_vec=tangent_increment,
            base_point=endpoint,
        )
        return endpoint, None

    endpoint, _ = jax.lax.scan(step, initial_point, (timesteps, standard_noise))
    return endpoint


def so3_left_invariant_frame(manifold, endpoint: Array) -> Array:
    """Return a Frobenius-orthonormal left-invariant frame on SO(3).

    The returned array has shape ``[9, 3]``.  Each column is one flattened
    tangent matrix ``endpoint @ E_i``, where ``E_i`` is a normalized basis of
    ``so(3)``.  The normalization is the one used by Geomstats when drawing a
    standard normal tangent vector.
    """

    basis = jnp.asarray(manifold.lie_algebra.basis_normed, dtype=endpoint.dtype)
    translated = jnp.einsum("ab,ibc->iac", endpoint, basis)
    return jnp.swapaxes(translated.reshape((manifold.dim, -1)), 0, 1)


def so3_left_invariant_field_divergence(endpoint: Array) -> Array:
    """Return divergences of the SO(3) left-invariant orthonormal fields.

    Matrix SO(3) in this repository carries its bi-invariant metric.  The
    corresponding Riemannian volume is Haar and every left-invariant field is
    divergence-free.  Keep this explicit so the S2 ``-2 x`` formula cannot be
    accidentally reused for SO(3).
    """

    return jnp.zeros((3,), dtype=endpoint.dtype)


def upstream_so3_grw_endpoint(
    sde,
    initial_point: Array,
    terminal_time: Array,
    standard_noise: Array,
    sampler_eps: float = 1e-3,
) -> Array:
    """Evaluate an SO(3) endpoint with the upstream GRW discretisation."""

    n_steps = standard_noise.shape[0]
    start_time = sde.t0 + sampler_eps
    timesteps = jnp.linspace(start_time, terminal_time, num=n_steps, endpoint=True)
    dt = (terminal_time - start_time) / n_steps
    basis = jnp.asarray(
        sde.manifold.lie_algebra.basis_normed,
        dtype=initial_point.dtype,
    )

    def step(endpoint, inputs):
        time, noise_coordinates = inputs
        algebra_noise = jnp.einsum("i,ijk->jk", noise_coordinates, basis)
        tangent_noise = endpoint @ algebra_noise
        diffusion = jnp.sqrt(sde.beta_schedule.beta_t(time))
        tangent_increment = diffusion * jnp.sqrt(jnp.abs(dt)) * tangent_noise
        endpoint = sde.manifold.exp(
            tangent_vec=tangent_increment,
            base_point=endpoint,
        )
        return endpoint, None

    endpoint, _ = jax.lax.scan(step, initial_point, (timesteps, standard_noise))
    return endpoint


def so3_tangent_coordinate_jacobian(
    manifold,
    endpoint: Array,
    endpoint_jacobian: Array,
) -> Array:
    """Project an ambient SO(3) endpoint Jacobian onto tangent coordinates."""

    frame = so3_left_invariant_frame(manifold, endpoint)
    ambient_jacobian = endpoint_jacobian.reshape((9, -1))
    return frame.T @ ambient_jacobian


def malliavin_covariance(tangent_jacobian: Array) -> Array:
    """Return a symmetrized finite-dimensional Malliavin covariance."""

    covariance = tangent_jacobian @ tangent_jacobian.T
    return 0.5 * (covariance + covariance.T)


class MalliavinTeacher:
    """Pathwise transition-score teacher for upstream S2/SO(3) Brownian GRW.

    ``initial_point`` is fixed when differentiating the endpoint map with
    respect to its Gaussian noise.  The resulting ``endpoint_jacobian`` is
    the discrete Malliavin derivative ``D_Z X_t``, not the flow Jacobian
    ``J_{t,s}``.  Consequently the conditional expectation of this pathwise
    weight given ``(X_t, X_0)`` estimates
    ``grad log p_{t|0}(X_t | X_0)``.  The DSM regression over sampled ``X_0``
    then has the marginal score ``grad log p_t`` as its population minimizer.

    See ``docs/malliavin_teacher.md`` for the notation and full identity.
    """

    def __init__(
        self,
        covariance_regularization: float = 1e-6,
        sampler_eps: float = 1e-3,
        divergence_mode: str = "exact",
        hutchinson_probes: int = 1,
        hutchinson_noise: str = "rademacher",
        divergence_fn: Optional[DivergenceFn] = None,
        rb_enabled: bool = False,
        rb_spatial_bandwidth: float = 0.6,
        rb_time_bandwidth: float = 0.05,
        rb_alpha: float = 0.0,
        rb_time_center: float = 0.5,
        rb_time_scale: float = 20.0,
        rb_time_dependent: bool = True,
    ):
        if covariance_regularization < 0:
            raise ValueError("covariance_regularization must be non-negative")
        divergence_mode = divergence_mode.lower()
        if divergence_mode not in ("exact", "hutchinson"):
            raise ValueError("divergence_mode must be 'exact' or 'hutchinson'")
        if hutchinson_probes < 1:
            raise ValueError("hutchinson_probes must be positive")
        hutchinson_noise = hutchinson_noise.lower()
        if hutchinson_noise not in ("rademacher", "gaussian"):
            raise ValueError(
                "hutchinson_noise must be 'rademacher' or 'gaussian'"
            )
        if divergence_fn is not None and divergence_mode != "exact":
            raise ValueError(
                "a custom divergence_fn cannot be combined with divergence_mode"
            )
        if rb_spatial_bandwidth <= 0.0 or rb_time_bandwidth <= 0.0:
            raise ValueError("Rao-Blackwell bandwidths must be positive")
        if not math.isfinite(rb_alpha):
            raise ValueError("rb_alpha must be finite")
        if not math.isfinite(rb_time_center) or not math.isfinite(
            rb_time_scale
        ):
            raise ValueError("RB time-schedule parameters must be finite")
        self.covariance_regularization = covariance_regularization
        self.sampler_eps = sampler_eps
        self.divergence_mode = divergence_mode
        self.hutchinson_probes = hutchinson_probes
        self.hutchinson_noise = hutchinson_noise
        self.divergence_fn = divergence_fn
        self.rb_enabled = rb_enabled
        self.rb_time_dependent = rb_time_dependent
        self.rb_alpha = rb_alpha
        self.rb_time_center = rb_time_center
        self.rb_time_scale = rb_time_scale
        self.rb_spatial_bandwidth = rb_spatial_bandwidth
        self.rb_time_bandwidth = rb_time_bandwidth

    def _manifold_kind(self, sde) -> str:
        manifold = sde.manifold
        if (
            getattr(manifold, "dim", None) == 3
            and getattr(manifold, "n", None) == 3
            and hasattr(manifold, "lie_algebra")
            and getattr(getattr(manifold, "identity", None), "shape", None)
            == (3, 3)
        ):
            return "so3"
        if getattr(manifold, "dim", None) != 2:
            raise ValueError("MalliavinTeacher supports only S2 and matrix SO(3)")
        embedding_space = getattr(sde.manifold, "embedding_space", None)
        if getattr(embedding_space, "dim", None) != 3:
            raise ValueError("MalliavinTeacher requires the ambient R3 embedding")
        return "s2"

    def _compute_divergence(self, vector_field_fn, z, rng):
        if self.divergence_fn is not None:
            return self.divergence_fn(vector_field_fn, z)
        if self.divergence_mode == "exact":
            return compute_divergence_exact(vector_field_fn, z)
        return compute_divergence_hutchinson(
            vector_field_fn,
            z,
            rng,
            n_probes=self.hutchinson_probes,
            noise_type=self.hutchinson_noise,
        )

    def _single_s2_sample(
        self,
        sde,
        initial_point,
        terminal_time,
        standard_noise,
        divergence_rng,
    ):
        noise_shape = standard_noise.shape
        flat_noise = standard_noise.reshape(-1)

        def endpoint_fn(z):
            return upstream_s2_grw_endpoint(
                sde,
                initial_point,
                terminal_time,
                z.reshape(noise_shape),
                sampler_eps=self.sampler_eps,
            )

        def covering_state(z):
            endpoint = endpoint_fn(z)
            tangent_basis = s2_tangent_basis(endpoint)
            # D_Z X_t: derivative with respect to standard Gaussian path
            # increments.  This is not J_{t,s} or dX_t/dX_0.
            endpoint_jacobian = compute_endpoint_jacobian(endpoint_fn, z)
            tangent_jacobian = tangent_basis.T @ endpoint_jacobian
            fields = s2_projected_coordinate_fields(endpoint)
            tangent_fields = tangent_basis.T @ fields
            # Gamma = (B^T D_Z X_t)(B^T D_Z X_t)^T in tangent coordinates.
            covariance = tangent_jacobian @ tangent_jacobian.T
            covariance = 0.5 * (covariance + covariance.T)
            regularized_covariance = covariance + self.covariance_regularization * jnp.eye(
                2, dtype=z.dtype
            )
            coefficients = jnp.linalg.solve(regularized_covariance, tangent_fields)
            # U: regularised minimum-energy covering weights in noise space.
            covering = tangent_jacobian.T @ coefficients
            return endpoint, tangent_basis, covering

        def covering_fn(z):
            return covering_state(z)[2]

        endpoint, tangent_basis, covering = covering_state(flat_noise)
        covering_divergence = self._compute_divergence(
            covering_fn,
            flat_noise,
            divergence_rng,
        )
        # Finite-dimensional Skorokhod integral:
        # D^*U = delta(U) = U^T Z - div_Z U.
        gaussian_pairing = covering.T @ flat_noise
        skorokhod = gaussian_pairing - covering_divergence
        field_divergence = s2_projected_coordinate_field_divergence(endpoint)
        # One-path estimator of the directional transition score:
        # T_j = -D^*U_j - div(V_j).
        directional_score = -skorokhod - field_divergence

        # directional_score[j] estimates <grad log p_{t|0}, P_x e_j>.
        # For orthonormal B and V=P_x=B B^T, B^T V=B^T.  Thus the
        # pseudoinverse reconstruction used by scoremodel_ext is exactly
        # B(B^T directional_score); there is no additional scale factor.
        tangent_coordinates = tangent_basis.T @ directional_score
        score_target = tangent_basis @ tangent_coordinates
        return endpoint, score_target

    def _single_so3_sample(
        self,
        sde,
        initial_point,
        terminal_time,
        standard_noise,
        divergence_rng,
    ):
        noise_shape = standard_noise.shape
        flat_noise = standard_noise.reshape(-1)

        def endpoint_fn(z):
            return upstream_so3_grw_endpoint(
                sde,
                initial_point,
                terminal_time,
                z.reshape(noise_shape),
                sampler_eps=self.sampler_eps,
            )

        def covering_state(z):
            endpoint = endpoint_fn(z)
            frame = so3_left_invariant_frame(sde.manifold, endpoint)
            endpoint_jacobian = compute_endpoint_jacobian(endpoint_fn, z)
            tangent_jacobian = so3_tangent_coordinate_jacobian(
                sde.manifold,
                endpoint,
                endpoint_jacobian,
            )
            covariance = malliavin_covariance(tangent_jacobian)
            regularized_covariance = covariance + self.covariance_regularization * jnp.eye(
                3, dtype=z.dtype
            )
            # V_j is the j-th orthonormal left-invariant frame field, hence
            # its tangent-coordinate matrix is I_3.  Such fields have zero
            # Riemannian divergence for the bi-invariant SO(3) metric.
            coefficients = jnp.linalg.solve(
                regularized_covariance,
                jnp.eye(3, dtype=z.dtype),
            )
            covering = tangent_jacobian.T @ coefficients
            return endpoint, frame, covering

        def covering_fn(z):
            return covering_state(z)[2]

        endpoint, frame, covering = covering_state(flat_noise)
        covering_divergence = self._compute_divergence(
            covering_fn,
            flat_noise,
            divergence_rng,
        )
        gaussian_pairing = covering.T @ flat_noise
        skorokhod = gaussian_pairing - covering_divergence
        field_divergence = so3_left_invariant_field_divergence(endpoint)
        tangent_coordinates = -skorokhod - field_divergence
        score_target = (frame @ tangent_coordinates).reshape((3, 3))
        return endpoint, score_target

    def sample_and_score(self, rng, sde, y_0: Array, t: Array) -> Tuple[Array, Array]:
        manifold_kind = self._manifold_kind(sde)
        if manifold_kind == "so3" and self.rb_enabled:
            raise ValueError("Rao-Blackwellization is currently implemented only for S2")
        standard_noises = sample_upstream_grw_standard_noise(
            rng,
            y_0.shape[0],
            sde.N,
        )
        divergence_rngs = random.split(
            random.fold_in(rng, 0x4D414C4C),
            y_0.shape[0],
        )

        def sample_one(initial_point, terminal_time, standard_noise, divergence_rng):
            single_sample = (
                self._single_so3_sample
                if manifold_kind == "so3"
                else self._single_s2_sample
            )
            return single_sample(
                sde,
                initial_point,
                terminal_time,
                standard_noise,
                divergence_rng,
            )

        endpoint, score_target = jax.vmap(sample_one)(
            y_0, t, standard_noises, divergence_rngs
        )
        if not self.rb_enabled:
            return endpoint, score_target

        raw_target = score_target
        rb_target, _ = rao_blackwell_estimate_s2_batch(
            endpoint,
            t,
            raw_target,
            spatial_bandwidth=self.rb_spatial_bandwidth,
            time_bandwidth=self.rb_time_bandwidth,
            rb_alpha=1.0,
        )
        alpha_t = rb_alpha_schedule(
            t,
            self.rb_alpha,
            self.rb_time_dependent,
            self.rb_time_center,
            self.rb_time_scale,
        )
        score_target = raw_target + alpha_t[:, None] * (
            rb_target - raw_target
        )
        return endpoint, score_target
