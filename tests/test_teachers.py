import ast
import inspect
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np

from geomstats.geometry.hypersphere import Hypersphere

from riemannian_score_sde.sde import Brownian
from riemannian_score_sde import losses, teachers
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


def test_hutchinson_divergence_uses_probe_jvps_and_is_jittable():
    def diagonal_vector_fields(z):
        return jnp.stack((2.0 * z, -3.0 * z), axis=-1)

    divergence = jax.jit(
        lambda value, rng: teachers.compute_divergence_hutchinson(
            diagonal_vector_fields,
            value,
            rng,
            n_probes=1,
            noise_type="rademacher",
        )
    )(jnp.array([0.2, -0.4, 0.7]), jax.random.PRNGKey(31))

    # Rademacher probes have e_i**2 == 1, so diagonal Jacobians are exact
    # even with one probe.
    np.testing.assert_allclose(divergence, jnp.array([6.0, -9.0]))


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


def test_hutchinson_teacher_keeps_endpoint_and_tangent_output():
    sde = make_s2_brownian(n_steps=2)
    exact_teacher = teachers.MalliavinTeacher(covariance_regularization=1e-5)
    hutchinson_teacher = teachers.MalliavinTeacher(
        covariance_regularization=1e-5,
        divergence_mode="hutchinson",
        hutchinson_probes=2,
        hutchinson_noise="rademacher",
    )
    y_0 = jnp.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    t = jnp.array([0.2, 0.3], dtype=y_0.dtype)
    rng = jax.random.PRNGKey(43)

    exact_endpoint, _ = exact_teacher.sample_and_score(rng, sde, y_0, t)
    endpoint, score = jax.jit(
        lambda key, initial, time: hutchinson_teacher.sample_and_score(
            key,
            sde,
            initial,
            time,
        )
    )(rng, y_0, t)

    np.testing.assert_array_equal(endpoint, exact_endpoint)
    assert endpoint.shape == score.shape == y_0.shape
    assert jnp.isfinite(score).all()
    np.testing.assert_allclose(
        jnp.sum(endpoint * score, axis=-1),
        jnp.zeros(y_0.shape[0]),
        atol=2e-5,
    )


def test_heat_and_malliavin_targets_share_endpoint_and_report_scale():
    sde = make_s2_brownian(n_steps=2)
    heat_teacher = teachers.HeatTeacher()
    malliavin_teacher = teachers.MalliavinTeacher(
        covariance_regularization=1e-5,
        divergence_mode="hutchinson",
        hutchinson_probes=2,
    )
    y_0 = jnp.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    t = jnp.array([0.2, 0.3], dtype=y_0.dtype)
    rng = jax.random.PRNGKey(47)

    heat_endpoint, heat_target = heat_teacher.sample_and_score(rng, sde, y_0, t)
    endpoint, malliavin_target = malliavin_teacher.sample_and_score(
        rng,
        sde,
        y_0,
        t,
    )
    np.testing.assert_array_equal(endpoint, heat_endpoint)

    diagnostics = losses.compute_teacher_scale_diagnostics(
        sde,
        endpoint,
        t,
        jnp.zeros_like(endpoint),
        heat_target,
        malliavin_target,
        like_w=False,
        endpoint_max_abs_error=jnp.max(jnp.abs(endpoint - heat_endpoint)),
    )
    assert set(diagnostics) == {
        "endpoint_max_abs_error",
        "heat_rescore_max_abs_error",
        "heat_target_norm_mean",
        "heat_target_norm_std",
        "malliavin_target_norm_mean",
        "malliavin_target_norm_std",
        "predicted_score_norm_mean",
        "predicted_score_norm_std",
        "target_difference_norm_mean",
        "target_difference_norm_std",
        "heat_loss_contribution_mean",
        "heat_loss_contribution_std",
        "malliavin_loss_contribution_mean",
        "malliavin_loss_contribution_std",
        "malliavin_tangency_max_abs",
    }
    assert all(jnp.isfinite(value) for value in diagnostics.values())
    np.testing.assert_allclose(diagnostics["endpoint_max_abs_error"], 0.0)
    np.testing.assert_allclose(diagnostics["predicted_score_norm_mean"], 0.0)
    np.testing.assert_allclose(
        diagnostics["malliavin_tangency_max_abs"],
        0.0,
        atol=2e-5,
    )


def test_dsm_max_t_defaults_to_sde_tf_and_accepts_fixed_override():
    class Metric:
        @staticmethod
        def squared_norm(vector, _base_point):
            return jnp.sum(jnp.square(vector), axis=-1)

    class Manifold:
        metric = Metric()

    class SDE:
        t0 = 0.0
        tf = 1.0
        manifold = Manifold()

        @staticmethod
        def reparametrise_score_fn(_model, _params, _states, _train, _return_state):
            def score_fn(y, _t, _context, rng=None):
                del rng
                return jnp.zeros_like(y), {}

            return score_fn

        @staticmethod
        def marginal_prob(y, t):
            return jnp.zeros_like(y), jnp.ones_like(t)

    class Transform:
        @staticmethod
        def inv(data):
            return data

    class Pushforward:
        sde = SDE()
        transform = Transform()

    class RecordingTeacher:
        sampled_t = None

        def sample_and_score(self, _rng, _sde, y_0, t):
            self.sampled_t = t
            return y_0, jnp.zeros_like(y_0)

    eps = 1e-3
    batch = {
        "data": jnp.zeros((128, 3)),
        "context": None,
    }
    rng = jax.random.PRNGKey(53)

    default_teacher = RecordingTeacher()
    default_loss = losses.get_dsm_loss_fn(
        Pushforward(),
        model=None,
        teacher=default_teacher,
        eps=eps,
        max_t=None,
        like_w=False,
    )
    default_loss(rng, {}, {}, batch)

    fixed_teacher = RecordingTeacher()
    fixed_loss = losses.get_dsm_loss_fn(
        Pushforward(),
        model=None,
        teacher=fixed_teacher,
        eps=eps,
        max_t=0.5,
        like_w=False,
    )
    fixed_loss(rng, {}, {}, batch)

    expected_fixed_t = eps + (default_teacher.sampled_t - eps) * (
        (0.5 - eps) / (SDE.tf - eps)
    )
    np.testing.assert_allclose(fixed_teacher.sampled_t, expected_fixed_t)
    assert jnp.all(default_teacher.sampled_t < SDE.tf)
    assert jnp.all(fixed_teacher.sampled_t < 0.5)


def test_dsm_reports_teacher_norm_and_relative_loss_without_changing_loss():
    class Metric:
        @staticmethod
        def squared_norm(vector, _base_point):
            return jnp.sum(jnp.square(vector), axis=-1)

    class Manifold:
        metric = Metric()

    class SDE:
        t0 = 0.0
        tf = 1.0
        manifold = Manifold()

        @staticmethod
        def reparametrise_score_fn(_model, _params, _states, _train, _return_state):
            def score_fn(y, _t, _context, rng=None):
                del rng
                return jnp.zeros_like(y), {"state": jnp.asarray(1.0)}

            return score_fn

        @staticmethod
        def marginal_prob(y, t):
            return jnp.zeros_like(y), jnp.ones_like(t)

    class Transform:
        @staticmethod
        def inv(data):
            return data

    class Pushforward:
        sde = SDE()
        transform = Transform()

    class UnitTeacher:
        @staticmethod
        def sample_and_score(_rng, _sde, y_0, _t):
            return y_0, jnp.ones_like(y_0)

    loss_fn = losses.get_dsm_loss_fn(
        Pushforward(),
        model=None,
        teacher=UnitTeacher(),
        like_w=False,
        return_metrics=True,
    )
    loss, (new_model_state, metrics) = loss_fn(
        jax.random.PRNGKey(59),
        {},
        {},
        {"data": jnp.zeros((8, 3)), "context": None},
    )

    # With unit std, a zero prediction and a three-dimensional unit target,
    # the unchanged raw DSM loss and target squared norm are both three.
    np.testing.assert_allclose(loss, 3.0)
    np.testing.assert_allclose(metrics["teacher_norm"], 3.0)
    np.testing.assert_allclose(metrics["relative_loss"], 1.0)
    np.testing.assert_allclose(new_model_state["state"], 1.0)
    assert loss_fn.returns_metrics is True


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


def test_hutchinson_divergence_full_run_cost_is_explicit():
    cost = teachers.hutchinson_divergence_jvp_count(
        batch_size=512,
        n_steps=100,
        updates=600_000,
        n_probes=1,
    )
    assert cost == {
        "noise_dimension": 300,
        "probes_per_sample": 1,
        "jvps_per_update": 512,
        "jvps_total": 307_200_000,
    }
