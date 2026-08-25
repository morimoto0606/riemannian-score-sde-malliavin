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


class _BetaSchedule:
    @staticmethod
    def beta_t(t):
        return jnp.full_like(t, 2.0)


class _SDE:
    t0 = 0.0
    tf = 1.0
    manifold = _Manifold()
    beta_schedule = _BetaSchedule()
    sampled_t = None

    @staticmethod
    def reparametrise_score_fn(_model, _params, _states, _train, _return_state):
        def score_fn(y, _t, _context, rng=None):
            del rng
            return jnp.zeros_like(y), {}

        return score_fn

    @classmethod
    def marginal_sample(cls, _rng, y, t):
        cls.sampled_t = t
        return y


class _Transform:
    @staticmethod
    def inv(data):
        return data


class _Pushforward:
    sde = _SDE()
    transform = _Transform()


def _evaluate(monkeypatch, *, like_w, time_weighting, time_weight_lambda):
    monkeypatch.setattr(
        losses,
        "div_noise",
        lambda _rng, shape, _kind: jnp.zeros(shape),
    )

    def fake_divergence(_drift_fn, _kind, _manifold):
        return lambda y, _t, _context, _epsilon: jnp.ones(y.shape[0])

    monkeypatch.setattr(losses, "get_riemannian_div_fn", fake_divergence)
    loss_fn = losses.get_ism_loss_fn(
        _Pushforward(),
        model=None,
        train=True,
        like_w=like_w,
        hutchinson_type=None,
        time_weighting=time_weighting,
        time_weight_lambda=time_weight_lambda,
    )
    value, _ = loss_fn(
        jax.random.PRNGKey(91),
        {},
        {},
        {"data": jnp.zeros((8, 3)), "context": None},
    )
    return value, _SDE.sampled_t


@pytest.mark.parametrize("like_w", [False, True])
def test_ism_zero_lambda_matches_unweighted_loss(monkeypatch, like_w):
    legacy, _ = _evaluate(
        monkeypatch,
        like_w=like_w,
        time_weighting=False,
        time_weight_lambda=5.0,
    )
    lambda_zero, _ = _evaluate(
        monkeypatch,
        like_w=like_w,
        time_weighting=True,
        time_weight_lambda=0.0,
    )
    np.testing.assert_array_equal(lambda_zero, legacy)


@pytest.mark.parametrize("like_w,like_w_factor", [(False, 1.0), (True, 2.0)])
def test_ism_time_weight_is_independent_of_like_w(
    monkeypatch,
    like_w,
    like_w_factor,
):
    weighted, sampled_t = _evaluate(
        monkeypatch,
        like_w=like_w,
        time_weighting=True,
        time_weight_lambda=5.0,
    )
    expected = jnp.mean(like_w_factor * jnp.exp(-5.0 * sampled_t))
    np.testing.assert_allclose(weighted, expected, rtol=1e-6, atol=1e-7)
