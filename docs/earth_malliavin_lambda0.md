# S2 Malliavin lambda=0, three seeds

`job/earth_malliavin_lambda0_seeds.q` is a PBS array of nine jobs:
0–2 earthquake, 3–5 flood, 6–8 volcano, seeds 0–2 respectively.
It retains the queue and CPU allocation of the existing lambda5 jobs.
SO(3) is excluded because its lambda0 seed directories already exist.

The runner reads each lambda5 `.hydra/config.yaml`, retains the saved teacher,
architecture, training steps, data settings, optimizer and evaluation flags,
and changes lambda to zero. Weighting remains enabled, giving exp(-0*t)=1.
Mode is fresh training (resume=false). CSV logs, generated samples and checkpoints
have isolated paths. Existing destinations are rejected, never resumed/deleted.
Historical misspelled volcano result directories are accepted read-only; ambiguous
old/new source pairs are rejected. New outputs use volcano.

Run the preflight on mims before submission:

```bash
python scripts/run_earth_malliavin_lambda0.py --check-all
qsub job/earth_malliavin_lambda0_seeds.q
```

Preflight does not train or write results. Source YAML, its checksum and the current
Git commit are recorded per run. Saved hyperparameters do not guarantee identity
of historical source code, dependencies or data. Keep the mims environment and
branch; do not switch to the CUDA13 branch just to obtain these launchers.

Training process completion is recorded separately from checkpoint validation.
Generation/postprocessing is not added by the launcher: saved training flags are
preserved. Compare with lambda5 using the same existing generation/postprocess
settings and reference data after all checkpoints are checked.

Local verification: Python syntax and bash syntax only; OmegaConf is unavailable
in the local NumPy environment. No environment was installed and no training ran.
