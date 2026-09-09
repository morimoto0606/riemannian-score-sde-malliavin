import os
import hydra

# from score_sde.utils.cfg import *


@hydra.main(config_path="config", config_name="main")
def main(cfg):
    os.environ["GEOMSTATS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["WANDB_START_METHOD"] = "thread"
    if (cfg.get("enable_x64", False)
            and (cfg.get("manifold", {}).get("_target_") == "riemannian_score_sde.spd.AffineSPD"
                 or getattr(cfg.get("teacher"), "_target_", None)
                 == "riemannian_score_sde.teachers.SpectrumTeacher")):
        os.environ["JAX_ENABLE_X64"] = "True"

    from run import run

    return run(cfg)


if __name__ == "__main__":
    main()
