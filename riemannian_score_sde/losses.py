"""All functions related to loss computation and optimization.
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as random

from score_sde.utils import batch_mul
from score_sde.models import SDEPushForward, MoserFlow
from score_sde.utils import ParametrisedScoreFunction, TrainState
from score_sde.models import div_noise, get_riemannian_div_fn
from riemannian_score_sde.teachers import HeatTeacher, MalliavinTeacher, VaradhanTeacher


def _mean_and_std(values):
    return jnp.mean(values), jnp.std(values)


def compute_teacher_scale_diagnostics(
    sde,
    y_t,
    t,
    predicted_score,
    heat_target,
    malliavin_target,
    *,
    like_w,
    endpoint_max_abs_error=0.0,
    heat_rescore_max_abs_error=0.0,
):
    """Summarise Heat/Malliavin targets on one shared endpoint batch."""

    def vector_norm(vector):
        squared_norm = sde.manifold.metric.squared_norm(vector, y_t)
        return jnp.sqrt(jnp.maximum(squared_norm, 0.0))

    def loss_contribution(target):
        difference = predicted_score - target
        if like_w:
            squared_norm = sde.manifold.metric.squared_norm(difference, y_t)
            diffusion_squared = sde.coefficients(jnp.zeros_like(y_t), t)[1] ** 2
            return squared_norm * diffusion_squared
        std = sde.marginal_prob(jnp.zeros_like(y_t), t)[1]
        scaled_difference = std[..., None] * difference
        return sde.manifold.metric.squared_norm(scaled_difference, y_t)

    heat_norm_mean, heat_norm_std = _mean_and_std(vector_norm(heat_target))
    malliavin_norm_mean, malliavin_norm_std = _mean_and_std(
        vector_norm(malliavin_target)
    )
    predicted_norm_mean, predicted_norm_std = _mean_and_std(
        vector_norm(predicted_score)
    )
    target_difference_mean, target_difference_std = _mean_and_std(
        vector_norm(malliavin_target - heat_target)
    )
    heat_loss_mean, heat_loss_std = _mean_and_std(loss_contribution(heat_target))
    malliavin_loss_mean, malliavin_loss_std = _mean_and_std(
        loss_contribution(malliavin_target)
    )

    return {
        "endpoint_max_abs_error": jnp.asarray(endpoint_max_abs_error),
        "heat_rescore_max_abs_error": jnp.asarray(heat_rescore_max_abs_error),
        "heat_target_norm_mean": heat_norm_mean,
        "heat_target_norm_std": heat_norm_std,
        "malliavin_target_norm_mean": malliavin_norm_mean,
        "malliavin_target_norm_std": malliavin_norm_std,
        "predicted_score_norm_mean": predicted_norm_mean,
        "predicted_score_norm_std": predicted_norm_std,
        "target_difference_norm_mean": target_difference_mean,
        "target_difference_norm_std": target_difference_std,
        "heat_loss_contribution_mean": heat_loss_mean,
        "heat_loss_contribution_std": heat_loss_std,
        "malliavin_loss_contribution_mean": malliavin_loss_mean,
        "malliavin_loss_contribution_std": malliavin_loss_std,
        "malliavin_tangency_max_abs": jnp.max(
            jnp.abs(jnp.sum(y_t * malliavin_target, axis=-1))
        ),
    }


def print_teacher_scale_diagnostics(diagnostics):
    """Print one JIT-compatible Heat/Malliavin DSM scale report."""

    jax.debug.print(
        "[teacher-scale] endpoint_max_abs_error={endpoint:.3e}\n"
        "  heat_rescore_max_abs_error={heat_rescore:.3e}\n"
        "  target_norm heat_conditional={heat_mean:.6g} +/- {heat_std:.6g} "
        "malliavin_pathwise={mall_mean:.6g} +/- {mall_std:.6g}\n"
        "  predicted_score_norm={pred_mean:.6g} +/- {pred_std:.6g}\n"
        "  raw_target_difference_norm={diff_mean:.6g} +/- {diff_std:.6g}\n"
        "  loss_contribution heat_cross_loss={heat_loss_mean:.6g} +/- {heat_loss_std:.6g} "
        "malliavin_training_loss={mall_loss_mean:.6g} +/- {mall_loss_std:.6g}\n"
        "  malliavin_tangency_max_abs={tangent:.3e}",
        endpoint=diagnostics["endpoint_max_abs_error"],
        heat_rescore=diagnostics["heat_rescore_max_abs_error"],
        heat_mean=diagnostics["heat_target_norm_mean"],
        heat_std=diagnostics["heat_target_norm_std"],
        mall_mean=diagnostics["malliavin_target_norm_mean"],
        mall_std=diagnostics["malliavin_target_norm_std"],
        pred_mean=diagnostics["predicted_score_norm_mean"],
        pred_std=diagnostics["predicted_score_norm_std"],
        diff_mean=diagnostics["target_difference_norm_mean"],
        diff_std=diagnostics["target_difference_norm_std"],
        heat_loss_mean=diagnostics["heat_loss_contribution_mean"],
        heat_loss_std=diagnostics["heat_loss_contribution_std"],
        mall_loss_mean=diagnostics["malliavin_loss_contribution_mean"],
        mall_loss_std=diagnostics["malliavin_loss_contribution_std"],
        tangent=diagnostics["malliavin_tangency_max_abs"],
    )


def get_dsm_loss_fn(
    pushforward: SDEPushForward,
    model: ParametrisedScoreFunction,
    train: bool = True,
    like_w: bool = True,
    eps: float = 1e-3,
    s_zero=True,
    teacher=None,
    debug_teacher_comparison=False,
    **kwargs
):
    sde = pushforward.sde
    if teacher is None:
        if "n_max" in kwargs and kwargs["n_max"] <= -1:
            teacher = VaradhanTeacher()
        else:
            teacher = HeatTeacher(
                n_max=kwargs.get("n_max", 5),
                thresh=kwargs.get("thresh", 0.5),
            )
    if debug_teacher_comparison and not isinstance(teacher, MalliavinTeacher):
        raise ValueError(
            "debug_teacher_comparison requires a MalliavinTeacher target"
        )
    comparison_heat_teacher = HeatTeacher(
        n_max=kwargs.get("n_max", 5),
        thresh=kwargs.get("thresh", 0.5),
    )

    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        score_fn = sde.reparametrise_score_fn(model, params, states, train, True)
        y_0, context = pushforward.transform.inv(batch["data"]), batch["context"]

        rng, step_rng = random.split(rng)
        # uniformly sample from SDE timeframe
        t = random.uniform(step_rng, (y_0.shape[0],), minval=sde.t0 + eps, maxval=sde.tf)
        rng, step_rng = random.split(rng)

        # sample p(y_t | y_0)
        # compute $\nabla \log p(y_t | y_0)$
        if s_zero:  # l_{t|0}
            y_t, logp_grad = teacher.sample_and_score(step_rng, sde, y_0, t)
            std = jnp.expand_dims(sde.marginal_prob(jnp.zeros_like(y_t), t)[1], -1)
        else:  # l_{t|s}
            y_t, y_hist, timesteps = sde.marginal_sample(
                step_rng, y_0, t, return_hist=True
            )
            y_s = y_hist[-2]
            delta_t, logp_grad = sde.varhadan_exp(y_s, y_t, timesteps[-2], timesteps[-1])
            delta_t = t  # NOTE: works better?
            std = jnp.expand_dims(sde.marginal_prob(jnp.zeros_like(y_t), delta_t)[1], -1)

        # compute approximate score at y_t
        score, new_model_state = score_fn(y_t, t, context, rng=step_rng)
        score = score.reshape(y_t.shape)

        if debug_teacher_comparison:
            heat_y_t, sampled_heat_target = comparison_heat_teacher.sample_and_score(
                step_rng,
                sde,
                y_0,
                t,
            )
            endpoint_max_abs_error = jnp.max(jnp.abs(y_t - heat_y_t))
            heat_target = comparison_heat_teacher.score_at_endpoint(
                sde,
                y_0,
                y_t,
                t,
            )
            heat_rescore_max_abs_error = jnp.max(
                jnp.abs(heat_target - sampled_heat_target)
            )
            diagnostics = compute_teacher_scale_diagnostics(
                sde,
                y_t,
                t,
                score,
                heat_target,
                logp_grad,
                like_w=like_w,
                endpoint_max_abs_error=endpoint_max_abs_error,
                heat_rescore_max_abs_error=heat_rescore_max_abs_error,
            )
            print_teacher_scale_diagnostics(diagnostics)

        if not like_w:
            score = batch_mul(std, score)
            logp_grad = batch_mul(std, logp_grad)
            losses = sde.manifold.metric.squared_norm(score - logp_grad, y_t)
        else:
            # compute $E_{p{y_0}}[|| s_\theta(y_t, t) - \nabla \log p(y_t | y_0)||^2]$
            g2 = sde.coefficients(jnp.zeros_like(y_0), t)[1] ** 2
            losses = sde.manifold.metric.squared_norm(score - logp_grad, y_t) * g2

        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_ism_loss_fn(
    pushforward: SDEPushForward,
    model: ParametrisedScoreFunction,
    train: bool,
    like_w: bool = True,
    hutchinson_type="Rademacher",
    eps: float = 1e-3,
):
    sde = pushforward.sde

    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        score_fn = sde.reparametrise_score_fn(model, params, states, train, True)
        y_0, context = pushforward.transform.inv(batch["data"]), batch["context"]

        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (y_0.shape[0],), minval=sde.t0 + eps, maxval=sde.tf)

        rng, step_rng = random.split(rng)
        y_t = sde.marginal_sample(step_rng, y_0, t)
        score, new_model_state = score_fn(y_t, t, context, rng=step_rng)
        score = score.reshape(y_t.shape)

        # ISM loss
        rng, step_rng = random.split(rng)
        epsilon = div_noise(step_rng, y_0.shape, hutchinson_type)
        drift_fn = lambda y, t, context: score_fn(y, t, context, rng=step_rng)[0]
        div_fn = get_riemannian_div_fn(drift_fn, hutchinson_type, sde.manifold)
        div_score = div_fn(y_t, t, context, epsilon)
        sq_norm_score = sde.manifold.metric.squared_norm(score, y_t)
        losses = 0.5 * sq_norm_score + div_score

        if like_w:
            g2 = sde.beta_schedule.beta_t(t)
            losses = losses * g2

        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_moser_loss_fn(
    pushforward: MoserFlow,
    model: ParametrisedScoreFunction,
    alpha_m: float,
    alpha_p: float,
    K: int,
    hutchinson_type: str,
    eps: float,
    **kwargs
):
    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        y_0, context = pushforward.transform.inv(batch["data"]), batch["context"]
        model_w_dicts = (model, params, states)

        # log probability term
        rng, step_rng = random.split(rng)
        mu_plus = pushforward.mu_plus(
            y_0, context, model_w_dicts, hutchinson_type, step_rng
        )
        log_prob = jnp.mean(jnp.log(mu_plus))

        # regularization term
        rng, step_rng = random.split(rng)
        ys = pushforward.base.sample(step_rng, (K,))
        prior_prob = pushforward.nu(ys)

        rng, step_rng = random.split(rng)
        mu_minus = pushforward.mu_minus(
            ys, context, model_w_dicts, hutchinson_type, step_rng
        )
        volume_m = jnp.mean(batch_mul(mu_minus, 1 / prior_prob), axis=0)
        penalty = alpha_m * volume_m  # + alpha_p * volume_p

        loss = -log_prob + penalty

        return loss, states

    return loss_fn
