"""Server numerical checks: no training or dataset access required."""
import os
os.environ["GEOMSTATS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.legendre import legder, legval

from geomstats.geometry.hypersphere import Hypersphere
from riemannian_score_sde.sde import Brownian
from riemannian_score_sde.teachers import SpectrumTeacher
from score_sde.schedule import LinearBetaSchedule


@pytest.fixture
def sde():
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    flow = Brownian(Hypersphere(2), LinearBetaSchedule(
        beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0))
    # A pure spectral evaluation must never invoke either switching entry point.
    def forbidden(*args, **kwargs):
        raise AssertionError("Pure Spectrum called a switching/Varadhan path")
    flow.varhadan_exp = forbidden
    flow.grad_marginal_log_prob = forbidden
    yield flow
    jax.config.update("jax_enable_x64", previous)


def test_spectral_score_matches_independent_legendre_derivative(sde):
    t = jnp.array([0.1, 0.5, 1.0])
    angles = np.array([0.1, 0.6, 1.1])
    y0 = jnp.tile(jnp.array([0., 0., 1.]), (3, 1))
    yt = jnp.array(np.stack([np.sin(angles), np.zeros(3), np.cos(angles)], -1))
    n_max = 128
    actual = SpectrumTeacher(n_max).score_at_endpoint(sde, y0, yt, t)
    expected = []
    for x0, x, tau in zip(np.asarray(y0), np.asarray(yt), np.asarray(sde.beta_schedule.rescale_t(t))):
        n = np.arange(n_max + 1)
        c = (2*n+1) * np.exp(-n*(n+1)*tau/2) / (4*np.pi)
        z = np.dot(x0, x)
        expected.append(legval(z, legder(c)) / legval(z, c) * (x0-z*x))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.sum(np.asarray(actual)*np.asarray(yt), -1), 0, atol=1e-10)


def test_default_truncation_converges_at_minimum_training_time(sde):
    t = jnp.full((3,), 1e-3)
    tau = np.asarray(sde.beta_schedule.rescale_t(t))
    angles = np.array([0.5, 1., 3.]) * np.sqrt(tau)
    y0 = jnp.tile(jnp.array([0., 0., 1.]), (3, 1))
    yt = jnp.array(np.stack([np.sin(angles), np.zeros(3), np.cos(angles)], -1))
    actual = SpectrumTeacher().score_at_endpoint(sde, y0, yt, t)
    reference = SpectrumTeacher(6144).score_at_endpoint(sde, y0, yt, t)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)


def test_online_endpoint_is_unchanged(sde):
    endpoint = jnp.array([[0., 0.1, np.sqrt(0.99)]])
    sde.marginal_sample = lambda *args: endpoint
    y0 = jnp.array([[0., 0., 1.]])
    teacher = SpectrumTeacher(128)
    yt, score = teacher.sample_and_score(jax.random.PRNGKey(0), sde, y0, jnp.array([0.5]))
    np.testing.assert_array_equal(yt, endpoint)
    assert np.isfinite(score).all()
