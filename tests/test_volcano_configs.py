"""Canonical Volcano configuration composition, without training."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("suffix", [
    "", "_heat", "_ism", "_malliavin_hutchinson", "_spectrum", "_varadhan",
])
def test_volcano_composition(suffix):
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.1"):
        cfg = compose(config_name="main", overrides=["experiment=volcano" + suffix])
    cfg.work_dir = str(ROOT)
    OmegaConf.resolve(cfg)
    assert cfg.dataset.name == "volcano"
    assert cfg.dataset._target_ == "riemannian_score_sde.datasets.earth.VolcanicErruption"
    assert cfg.experiment == "volcano"
    assert Path(cfg.dataset.data_dir, "volerup.csv").is_file()
    if suffix == "_ism":
        assert cfg.loss._target_.endswith("get_ism_loss_fn")
    elif suffix:
        assert cfg.teacher._target_.endswith({
            "_heat": "HeatTeacher", "_malliavin_hutchinson": "MalliavinTeacher",
            "_spectrum": "SpectrumTeacher", "_varadhan": "VaradhanTeacher",
        }[suffix])
