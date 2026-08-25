from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.mark.parametrize(
    "experiment_name,loss_suffix,lambda_value,teacher_suffix",
    [
        ("so3_varadhan", "get_dsm_loss_fn", 0.0, "VaradhanTeacher"),
        ("so3_ism", "get_ism_loss_fn", 0.0, None),
        (
            "so3_malliavin_hutchinson",
            "get_dsm_loss_fn",
            0.0,
            "MalliavinTeacher",
        ),
    ],
)
def test_formal_so3_experiment_configs(
    experiment_name,
    loss_suffix,
    lambda_value,
    teacher_suffix,
):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name="so3-config-test"):
        cfg = compose(config_name="main", overrides=["experiment=" + experiment_name])

    assert cfg.manifold.n == 3
    assert cfg.manifold.point_type == "matrix"
    assert cfg.dataset.K == 32
    assert cfg.beta_schedule.beta_0 == 0.001
    assert cfg.beta_schedule.beta_f == 6
    assert cfg.generator._target_.endswith("LieAlgebraGenerator")
    assert cfg.loss._target_.endswith(loss_suffix)
    assert cfg.loss.time_weighting is False
    assert cfg.loss.time_weight_lambda == lambda_value
    assert cfg.steps == 100000
    assert cfg.batch_size == 512

    if teacher_suffix is None:
        assert cfg.loss.like_w is True
        assert cfg.get("teacher") is None
    else:
        assert cfg.teacher._target_.endswith(teacher_suffix)
    if teacher_suffix == "MalliavinTeacher":
        assert cfg.teacher.divergence_mode == "hutchinson"
        assert cfg.teacher.hutchinson_probes == 1
        assert cfg.teacher.rb_enabled is False
        assert cfg.teacher.covariance_regularization == 1e-6
