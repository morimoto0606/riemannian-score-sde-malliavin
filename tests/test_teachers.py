import ast
import inspect
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np

from geomstats.geometry.hypersphere import Hypersphere

from riemannian_score_sde.sde import Brownian
from riemannian_score_sde import teachers
from score_sde.schedule import LinearBetaSchedule


def make_s2_brownian(n_steps=2):
    return Brownian(
        manifold=Hypersphere(2),
        beta_schedule=LinearBetaSchedule(
            beta_0=0.001,
            beta_f=5.0,
            t0=0.0,
            tf=1.0,
        ),
        N=n_steps,
    )


def test_heat_teacher_is_exact_wrapper_of_existing_path():
    sde = make_s2_brownian()
    rng = jax.random.PRNGKey(17)
    y_0 = jnp.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    t = jnp.array([0.2, 0.4], dtype=y_0.dtype)

    expected_endpoint = sde.marginal_sample(rng, y_0, t)
    expected_score = sde.grad_marginal_log_prob(
        y_0,
        expected_endpoint,
        t,
        n_max=5,
        thresh=0.5,
    )[1]
    endpoint, score = teachers.HeatTeacher().sample_and_score(rng, sde, y_0, t)

    np.testing.assert_array_equal(endpoint, expected_endpoint)
    np.testing.assert_array_equal(score, expected_score)


def test_explicit_noise_endpoint_matches_native_upstream_grw():
    sde = make_s2_brownian(n_steps=3)
    rng = jax.random.PRNGKey(29)
    y_0 = jnp.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    t = jnp.array([0.2, 0.4], dtype=y_0.dtype)
    noise = teachers.sample_upstream_grw_standard_noise(rng, 2, sde.N)
    explicit = jax.vmap(
        lambda initial, terminal, z: teachers.upstream_s2_grw_endpoint(
            sde,
            initial,
            terminal,
            z,
        )
    )(y_0, t, noise)
    native = sde.marginal_sample(rng, y_0, t)
    np.testing.assert_allclose(explicit, native, rtol=1e-6, atol=1e-7)


def test_s2_tangent_basis_is_orthonormal_and_tangent():
    endpoint = jnp.array([0.2, -0.3, jnp.sqrt(0.87)])
    basis = teachers.s2_tangent_basis(endpoint)
    np.testing.assert_allclose(basis.T @ basis, jnp.eye(2), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(endpoint @ basis, jnp.zeros(2), atol=1e-6)


def test_exact_divergence_uses_basis_jvps_and_is_jittable():
    def vector_field(z):
        return jnp.stack(
            (
                jnp.square(z),
                jnp.stack((z[0] * z[1], z[1] * z[2], z[2] * z[0])),
            ),
            axis=-1,
        )

    z = jnp.array([0.2, -0.4, 0.7])
    divergence = jax.jit(
        lambda value: teachers.compute_divergence_exact(vector_field, value)
    )(z)
    expected = jnp.array((2.0 * z.sum(), z.sum()))
    np.testing.assert_allclose(divergence, expected, rtol=1e-6, atol=1e-6)


def test_malliavin_teacher_is_batched_jittable_and_tangent():
    sde = make_s2_brownian(n_steps=2)
    teacher = teachers.MalliavinTeacher(covariance_regularization=1e-5)
    y_0 = jnp.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    t = jnp.array([0.2, 0.3], dtype=y_0.dtype)
    sample_and_score = jax.jit(
        lambda rng, initial, time: teacher.sample_and_score(
            rng,
            sde,
            initial,
            time,
        )
    )
    endpoint, score = sample_and_score(
        jax.random.PRNGKey(41),
        y_0,
        t,
    )
    assert endpoint.shape == score.shape == y_0.shape
    assert jnp.isfinite(endpoint).all()
    assert jnp.isfinite(score).all()
    np.testing.assert_allclose(
        jnp.sum(endpoint * score, axis=-1),
        jnp.zeros(y_0.shape[0]),
        atol=2e-5,
    )


def test_only_endpoint_jacobian_function_contains_explicit_jacrev():
    tree = ast.parse(inspect.getsource(teachers))
    jacrev_by_function = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        count = sum(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "jax"
            and child.func.attr == "jacrev"
            for child in ast.walk(node)
            if child is not node
        )
        if count:
            jacrev_by_function[node.name] = count
    assert jacrev_by_function == {"compute_endpoint_jacobian": 1}


def test_score_reconstruction_does_not_use_pseudoinverse():
    source = inspect.getsource(teachers.MalliavinTeacher._single_sample)
    assert "pinv" not in source
    assert "tangent_basis.T @ directional_score" in source
    assert "tangent_basis @ tangent_coordinates" in source


def test_exact_divergence_full_run_cost_is_explicit():
    cost = teachers.exact_divergence_jvp_count(
        batch_size=512,
        n_steps=100,
        updates=600_000,
    )
    assert cost == {
        "noise_dimension": 300,
        "jvps_per_update": 153_600,
        "jvps_total": 92_160_000_000,
    }
