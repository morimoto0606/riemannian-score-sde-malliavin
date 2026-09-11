"""Server-side AIRM checks and ONE optimizer update per objective.

Run in the repository's supported JAX/geomstats environment, never as a long
training benchmark. These tests are intentionally not executed on the Mac.
"""

import os
os.environ["GEOMSTATS_BACKEND"] = "jax"
os.environ["JAX_ENABLE_X64"] = "True"

from pathlib import Path
import unittest

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.linalg import eigh

from riemannian_score_sde.spd import (
    AffineSPD, affine_exp, frame, from_frame, to_frame, svec, smat,
    explicit_grw_endpoint, riemannian_divergence, symmetric_basis,
)
from riemannian_score_sde.spd_sde import SPDBrownian
from riemannian_score_sde.spd_teacher import SPDMalliavinTeacher, cholesky_frame_divergence
from riemannian_score_sde.spd_generation import spd_summary
from riemannian_score_sde.teachers import VaradhanTeacher, sample_upstream_grw_standard_noise
from riemannian_score_sde.models.vector_field import SPDGenerator
from score_sde.models.flow import SDEPushForward
from score_sde.models.transform import Id
from score_sde.schedule import LinearBetaSchedule
from riemannian_score_sde.losses import get_dsm_loss_fn, get_ism_loss_fn


ROOT = Path(__file__).resolve().parents[1]


class SPDGeometryTests(unittest.TestCase):
    def setUp(self):
        self.space = AffineSPD(5)
        a = np.random.default_rng(21).normal(size=(5, 5))
        self.x = jnp.asarray((a @ a.T + np.eye(5)) * 1e-5)
        self.sde = SPDBrownian(self.space, LinearBetaSchedule(beta_0=.01, beta_f=.2), N=2)

    def test_frame_and_15_coordinates(self):
        basis = jnp.asarray(symmetric_basis(5))
        np.testing.assert_allclose(jnp.einsum("aij,bij->ab", basis, basis), np.eye(15), atol=1e-14)
        f = frame(self.x)
        gram = jax.vmap(lambda a: jax.vmap(lambda b: self.space.metric.inner_product(a, b, self.x))(f))(f)
        np.testing.assert_allclose(gram, np.eye(15), atol=1e-12)
        q = jnp.arange(15.)
        np.testing.assert_allclose(to_frame(from_frame(q, self.x), self.x), q, atol=1e-12)
        np.testing.assert_allclose(smat(svec(self.x), 5), self.x, atol=1e-14)

    def test_geomstats_maps_distance_and_repeated_eigenvalue_derivatives(self):
        u = from_frame(jnp.arange(15.) * .001, self.x)
        y = jax.jit(self.space.exp)(u, self.x)
        np.testing.assert_allclose(jax.jit(self.space.log)(y, self.x), u, rtol=1e-8, atol=1e-14)
        expected = np.linalg.norm(np.log(eigh(np.asarray(y), np.asarray(self.x), eigvals_only=True)))
        self.assertAlmostEqual(float(self.space.metric.dist(self.x, y)), expected, places=9)
        eye, zero = jnp.eye(5), jnp.zeros((5, 5))
        h = smat(jnp.arange(15.) * .01, 5)
        _, derivative = jax.jvp(affine_exp, (zero, eye), (h, h))
        np.testing.assert_allclose(derivative, 2 * h, atol=1e-12)
        second = jax.hessian(lambda q: jnp.trace(affine_exp(smat(q, 5), eye)))(jnp.zeros(15))
        self.assertTrue(np.isfinite(second).all())

    def test_forward_matches_explicit_endpoint_and_reverse_stays_spd(self):
        key = jax.random.PRNGKey(42)
        x = jnp.stack([self.x, self.x * 2])
        t = jnp.array([.2, .7])
        endpoint, hist, _ = self.sde.marginal_sample(key, x, t, return_hist=True)
        z = sample_upstream_grw_standard_noise(key, 2, self.sde.N, noise_dim=15, dtype=x.dtype)
        explicit = jax.vmap(lambda p, time, noise: explicit_grw_endpoint(self.sde, p, time, noise))(x, t, z)
        np.testing.assert_allclose(endpoint, explicit, rtol=1e-10, atol=1e-14)
        spd_summary(np.asarray(hist).reshape((-1, 5, 5)))
        from riemannian_score_sde.sampling import get_pc_sampler
        reverse = self.sde.reverse(lambda y, time: jnp.zeros_like(y))
        sampled = get_pc_sampler(reverse, N=2, predictor="GRW", eps=.001)(key, endpoint)
        spd_summary(sampled)

    def test_ism_volume_and_frame_divergence(self):
        self.assertAlmostEqual(float(riemannian_divergence(lambda x: x, self.x)), 0., places=10)
        self.assertAlmostEqual(float(riemannian_divergence(
            lambda x: x, self.x, jnp.ones(15, dtype=jnp.float32))), 0., places=10)
        constant = jnp.eye(5)
        expected = -3 * np.trace(np.linalg.inv(np.asarray(self.x)))
        self.assertAlmostEqual(float(riemannian_divergence(lambda x: constant, self.x)) / expected, 1., places=10)
        divergences = jnp.stack([riemannian_divergence(lambda x, a=a: frame(x)[a], self.x) for a in range(15)])
        np.testing.assert_allclose(divergences, cholesky_frame_divergence(5, self.x.dtype), atol=1e-10)
        field = lambda x: x @ x
        exact = riemannian_divergence(field, self.x)
        probes = np.sqrt(15) * jnp.eye(15)
        estimate = jnp.mean(jax.vmap(lambda p: riemannian_divergence(field, self.x, p))(probes))
        np.testing.assert_allclose(exact, estimate, rtol=1e-10, atol=1e-12)

    def test_malliavin_sign_against_exact_spd1_log_normal(self):
        sde = SPDBrownian(AffineSPD(1), LinearBetaSchedule(beta_0=.1, beta_f=.1), N=1)
        x, t = jnp.array([[[2.]]]), jnp.array([.4])
        teacher = SPDMalliavinTeacher(covariance_regularization=0., divergence_mode="exact")
        endpoint, target = teacher.sample_and_score(jax.random.PRNGKey(1), sde, x, t)
        exact = -endpoint * jnp.log(endpoint / x) / sde.beta_schedule.rescale_t(t)[:, None, None]
        np.testing.assert_allclose(target, exact, rtol=1e-7, atol=1e-9)


class SPDTrainingTests(unittest.TestCase):
    def test_three_configs_and_one_update_each(self):
        # Required order: Varadhan -> ISM -> Malliavin. No long training.
        for method in ("varadhan", "ism", "malliavin_hutchinson"):
            with self.subTest(method=method):
                with initialize_config_dir(config_dir=str(ROOT / "config"), job_name="spd-test"):
                    cfg = compose(config_name="main", overrides=[
                        "experiment=spd_finance_" + method, "steps=1", "batch_size=2",
                        "flow.N=1", "architecture.hidden_shapes=[8]", "seed=0",
                        "loss.time_weighting=true", "loss.time_weight_lambda=5.0"])
                self.assertEqual(cfg.dataset.dataset_seed, 0)
                self.assertEqual(list(cfg.splits), [.7, .15, .15])
                self.assertEqual(cfg.beta_schedule.beta_0, .01)
                self.assertEqual(cfg.beta_schedule.beta_f, 1.)
                space = instantiate(cfg.manifold)
                sde = instantiate(cfg.flow, manifold=space, beta_schedule=instantiate(cfg.beta_schedule))
                push = SDEPushForward(sde, sde.limiting, transform=Id(space))

                def model(y, t, context=None):
                    return SPDGenerator(cfg.architecture, cfg.embedding, space.dim, space)(y, t)

                model = hk.transform_with_state(model)
                x = jnp.asarray(np.stack([np.diag(np.arange(1., 6.)) * 1e-5,
                                         np.diag(np.arange(2., 7.)) * 1e-5]))
                params, states = model.init(jax.random.PRNGKey(0), y=x, t=jnp.ones((2, 1)) * .5)
                kwargs = {} if cfg.teacher is None else {"teacher": instantiate(cfg.teacher)}
                loss = instantiate(cfg.loss, pushforward=push, model=model, train=True, **kwargs)
                (value, _), gradient = jax.jit(jax.value_and_grad(loss, argnums=1, has_aux=True))(
                    jax.random.PRNGKey(2), params, states, {"data": x, "context": None})
                self.assertTrue(np.isfinite(value))
                self.assertTrue(all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(gradient)))
                optimizer = optax.adam(1e-4)
                updates, _ = optimizer.update(gradient, optimizer.init(params))
                updated = optax.apply_updates(params, updates)
                self.assertTrue(any(np.any(np.asarray(a) != np.asarray(b)) for a, b in
                                    zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(updated))))
                sde.bind_training_data(x)
                sampler = push.get_sampler((model, updated, states), N=2, predictor="GRW", eps=.001)
                spd_summary(sampler(jax.random.PRNGKey(8), (2,), None))

    def test_finance_loader_split_is_independent_of_training_seed(self):
        from riemannian_score_sde.datasets.spd_finance import SPDFinanceDataset
        path = ROOT / "data/spd_finance/spd_finance_5asset_60d.npz"
        if not path.exists():
            self.skipTest("Copy the prepared dataset to the server first")
        with initialize_config_dir(config_dir=str(ROOT / "config"), job_name="spd-data-test"):
            cfg = compose(config_name="main", overrides=["experiment=spd_finance_varadhan"])
        cfg.work_dir = str(ROOT)
        loaded = instantiate(cfg.dataset, rng=jax.random.PRNGKey(0))
        self.assertEqual(loaded.data.shape, (1801, 5, 5))
        self.assertEqual([len(part) for part in loaded.chronological_splits(cfg.splits)],
                         [1260, 210, 211])
        a = SPDFinanceDataset(path, rng=jax.random.PRNGKey(0))
        b = SPDFinanceDataset(path, rng=jax.random.PRNGKey(123))
        for aa, bb in zip(a.chronological_splits([.7, .15, .15]), b.chronological_splits([.7, .15, .15])):
            np.testing.assert_array_equal(aa[:], bb[:])


if __name__ == "__main__":
    unittest.main()
