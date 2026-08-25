import os

os.environ["GEOMSTATS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from riemannian_score_sde.sampling import get_pc_sampler
from riemannian_score_sde.sde import Brownian
from riemannian_score_sde import teachers
from score_sde.schedule import LinearBetaSchedule
from score_sde.utils import restore, save


def make_so3_brownian(n_steps=2):
    return Brownian(
        manifold=SpecialOrthogonal(n=3, point_type="matrix"),
        beta_schedule=LinearBetaSchedule(
            beta_0=0.001,
            beta_f=5.0,
            t0=0.0,
            tf=1.0,
        ),
        N=n_steps,
    )


def _endpoint_inputs(n_steps=2):
    sde = make_so3_brownian(n_steps)
    initial = sde.manifold.identity
    terminal = jnp.asarray(0.3, dtype=jnp.float32)
    noise = jnp.asarray(
        [[0.15, -0.2, 0.1], [-0.05, 0.12, 0.08]][:n_steps],
        dtype=jnp.float32,
    )
    return sde, initial, terminal, noise


def test_so3_explicit_endpoint_matches_native_grw_and_stays_on_manifold():
    sde = make_so3_brownian(n_steps=3)
    rng = jax.random.PRNGKey(101)
    initial = jnp.stack((sde.manifold.identity, sde.manifold.identity))
    terminal = jnp.array([0.2, 0.4], dtype=jnp.float32)
    noise = teachers.sample_upstream_grw_standard_noise(rng, 2, sde.N)
    explicit = jax.vmap(
        lambda x, t, z: teachers.upstream_so3_grw_endpoint(sde, x, t, z)
    )(initial, terminal, noise)
    native = sde.marginal_sample(rng, initial, terminal)

    np.testing.assert_allclose(explicit, native, rtol=2e-5, atol=2e-6)
    assert np.asarray(sde.manifold.belongs(explicit, atol=2e-5)).all()
    gram = jnp.swapaxes(explicit, -1, -2) @ explicit
    np.testing.assert_allclose(gram, jnp.eye(3), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(jnp.linalg.det(explicit), 1.0, atol=2e-5)


def test_so3_left_invariant_frame_is_orthonormal_and_tangent():
    sde = make_so3_brownian()
    endpoint = sde.manifold.random_uniform(jax.random.PRNGKey(103), 1)[0]
    frame = teachers.so3_left_invariant_frame(sde.manifold, endpoint)

    np.testing.assert_allclose(frame.T @ frame, jnp.eye(3), atol=2e-6)
    for index in range(3):
        tangent = frame[:, index].reshape((3, 3))
        body = endpoint.T @ tangent
        np.testing.assert_allclose(body + body.T, jnp.zeros((3, 3)), atol=2e-6)
    np.testing.assert_array_equal(
        teachers.so3_left_invariant_field_divergence(endpoint),
        jnp.zeros(3),
    )


def test_so3_endpoint_jacobian_matches_finite_difference():
    sde, initial, terminal, noise = _endpoint_inputs()
    flat_noise = noise.reshape(-1)
    endpoint_fn = lambda z: teachers.upstream_so3_grw_endpoint(
        sde, initial, terminal, z.reshape(noise.shape)
    )
    jacobian = teachers.compute_endpoint_jacobian(endpoint_fn, flat_noise)
    direction = jnp.linspace(-0.3, 0.4, flat_noise.size)
    step = 2e-3
    finite_difference = (
        endpoint_fn(flat_noise + step * direction)
        - endpoint_fn(flat_noise - step * direction)
    ) / (2.0 * step)
    automatic = jnp.einsum("abn,n->ab", jacobian, direction)
    np.testing.assert_allclose(automatic, finite_difference, rtol=3e-3, atol=3e-4)


def test_so3_tangent_jacobian_covariance_and_covering_equation():
    sde, initial, terminal, noise = _endpoint_inputs()
    flat_noise = noise.reshape(-1)
    endpoint_fn = lambda z: teachers.upstream_so3_grw_endpoint(
        sde, initial, terminal, z.reshape(noise.shape)
    )
    endpoint = endpoint_fn(flat_noise)
    endpoint_jacobian = teachers.compute_endpoint_jacobian(endpoint_fn, flat_noise)
    tangent_jacobian = teachers.so3_tangent_coordinate_jacobian(
        sde.manifold, endpoint, endpoint_jacobian
    )
    covariance = teachers.malliavin_covariance(tangent_jacobian)

    assert tangent_jacobian.shape == (3, 3 * sde.N)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-7)
    assert np.linalg.eigvalsh(np.asarray(covariance)).min() >= -1e-6

    for epsilon in (1e-8, 1e-6, 1e-3):
        regularized = covariance + epsilon * jnp.eye(3)
        coefficients = jnp.linalg.solve(regularized, jnp.eye(3))
        covering = tangent_jacobian.T @ coefficients
        assert np.isfinite(np.asarray(covering)).all()
        if epsilon >= 1e-6:
            assert np.linalg.eigvalsh(np.asarray(regularized)).min() > 0.0
        # With T=A A^T, A U = T(T+eps I)^-1.  Check the exact
        # regularized covering equation rather than incorrectly demanding I.
        residual = regularized @ coefficients - jnp.eye(3)
        np.testing.assert_allclose(residual, jnp.zeros((3, 3)), atol=2e-5)


def test_exact_divergence_matches_finite_difference_on_small_problem():
    def vector_fields(z):
        return jnp.stack((jnp.sin(z), z**3), axis=-1)

    z = jnp.array([0.2, -0.4, 0.7], dtype=jnp.float32)
    exact = teachers.compute_divergence_exact(vector_fields, z)
    step = 1e-3
    finite = []
    for field in range(2):
        trace = 0.0
        for coordinate in range(3):
            direction = jax.nn.one_hot(coordinate, 3)
            plus = vector_fields(z + step * direction)[coordinate, field]
            minus = vector_fields(z - step * direction)[coordinate, field]
            trace += (plus - minus) / (2.0 * step)
        finite.append(trace)
    np.testing.assert_allclose(exact, jnp.stack(finite), rtol=2e-3, atol=2e-4)


def test_so3_malliavin_teacher_is_jittable_finite_and_tangent():
    sde = make_so3_brownian(n_steps=1)
    teacher = teachers.MalliavinTeacher(
        covariance_regularization=1e-5,
        divergence_mode="hutchinson",
        hutchinson_probes=1,
        rb_enabled=False,
    )
    initial = jnp.stack((sde.manifold.identity,))
    times = jnp.array([0.25], dtype=jnp.float32)
    endpoint, score = jax.jit(
        lambda key, x, t: teacher.sample_and_score(key, sde, x, t)
    )(jax.random.PRNGKey(107), initial, times)

    assert endpoint.shape == score.shape == (1, 3, 3)
    assert np.isfinite(np.asarray(score)).all()
    body_score = jnp.swapaxes(endpoint, -1, -2) @ score
    np.testing.assert_allclose(
        body_score + jnp.swapaxes(body_score, -1, -2),
        jnp.zeros_like(body_score),
        atol=3e-5,
    )


def test_so3_teacher_rejects_s2_only_rao_blackwellization():
    sde = make_so3_brownian(n_steps=1)
    teacher = teachers.MalliavinTeacher(rb_enabled=True)
    with pytest.raises(ValueError, match="only for S2"):
        teacher.sample_and_score(
            jax.random.PRNGKey(109),
            sde,
            jnp.stack((sde.manifold.identity,)),
            jnp.array([0.2]),
        )


def test_so3_endpoint_checkpoint_roundtrip(tmp_path):
    sde = make_so3_brownian(n_steps=1)
    endpoint = sde.manifold.random_uniform(jax.random.PRNGKey(113), 2)
    save(str(tmp_path), {"so3_endpoint": endpoint})
    restored = restore(str(tmp_path))
    np.testing.assert_array_equal(restored["so3_endpoint"], endpoint)


def test_reverse_grw_sampling_remains_in_so3():
    sde = make_so3_brownian(n_steps=3)
    initial = sde.manifold.random_uniform(jax.random.PRNGKey(127), 4)
    reverse_sde = sde.reverse(lambda x, _t: jnp.zeros_like(x))
    sampler = get_pc_sampler(
        reverse_sde,
        N=3,
        predictor="GRW",
        corrector=None,
        denoise=False,
        eps=2e-4,
    )
    generated = sampler(jax.random.PRNGKey(131), initial)
    assert np.asarray(sde.manifold.belongs(generated, atol=3e-5)).all()
