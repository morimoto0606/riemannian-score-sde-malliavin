import os

os.environ["GEOMSTATS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from riemannian_score_sde import losses


class _Metric:
    @staticmethod
    def squared_norm(vector, _base_point):
        return jnp.sum(jnp.square(vector), axis=-1)


class _Manifold:
    metric = _Metric()


class _SDE:
    t0 = 0.0
    tf = 1.0
    manifold = _Manifold()

    @staticmethod
    def reparametrise_score_fn(_model, _params, _states, _train, _return_state):
        def score_fn(y, _t, _context, rng=None):
            del rng
            return jnp.zeros_like(y), {}

        return score_fn

    @staticmethod
    def marginal_prob(y, t):
        return jnp.zeros_like(y), jnp.ones_like(t)

    @staticmethod
    def coefficients(y, t):
        return jnp.zeros_like(y), jnp.ones_like(t)


class _Transform:
    @staticmethod
    def inv(data):
        return data


class _Pushforward:
    sde = _SDE()
    transform = _Transform()


class _RecordingTeacher:
    sampled_t = None

    def sample_and_score(self, _rng, _sde, y_0, t):
        self.sampled_t = t
        return y_0, jnp.ones_like(y_0)


def _evaluate_loss(like_w=False, **loss_kwargs):
    teacher = _RecordingTeacher()
    loss_fn = losses.get_dsm_loss_fn(
        _Pushforward(),
        model=None,
        teacher=teacher,
        like_w=like_w,
        **loss_kwargs,
    )
    loss, _ = loss_fn(
        jax.random.PRNGKey(73),
        {},
        {},
        {"data": jnp.zeros((8, 3)), "context": None},
    )
    return loss, teacher.sampled_t


def test_time_weighting_disabled_matches_legacy_loss():
    legacy_loss, _ = _evaluate_loss()
    disabled_loss, _ = _evaluate_loss(
        time_weighting=False,
        time_weight_lambda=2.0,
    )

    np.testing.assert_array_equal(disabled_loss, legacy_loss)


def test_zero_time_weight_lambda_matches_legacy_loss():
    legacy_loss, _ = _evaluate_loss()
    zero_lambda_loss, _ = _evaluate_loss(
        time_weighting=True,
        time_weight_lambda=0.0,
    )

    np.testing.assert_array_equal(zero_lambda_loss, legacy_loss)


@pytest.mark.parametrize("like_w", [False, True])
def test_positive_time_weight_lambda_weights_each_sample_by_exp_minus_lambda_t(
    like_w,
):
    time_weight_lambda = 2.0
    weighted_loss, sampled_t = _evaluate_loss(
        like_w=like_w,
        time_weighting=True,
        time_weight_lambda=time_weight_lambda,
    )

    # The unweighted per-sample loss is 3: a zero score versus a
    # three-dimensional unit teacher target with unit marginal std.
    expected_weights = jnp.exp(-time_weight_lambda * sampled_t)
    expected_loss = jnp.mean(3.0 * expected_weights)
    np.testing.assert_allclose(weighted_loss, expected_loss, rtol=1e-6, atol=1e-7)
