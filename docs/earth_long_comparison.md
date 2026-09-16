# Long S2 method comparison on mims

Keep feature/teacher-interface and the Python 3.9 environment. Stop only the
previous lambda0 array with `qdel '4992[].mims-smp'`; retain its files.
This comparison is fresh training, not a continuation from the 100k schedule.

Prepare 45 runs: earthquake and volcano 600000 updates, flood 300000 updates;
Malliavin lambda0, Malliavin lambda5, pure spectrum, pure Varadhan and ISM; seeds 0,1,2.
Each seed uses the corresponding verified Malliavin lambda5 training hparams as
its common base. Output config and a copy/hash of that source are retained.
Batch size 512, 80/10/10 split, 1000-step warmup and cosine decay follow
De Bortoli et al. appendix O:
https://mjhutchinson.info/files/rsgm/paper.pdf

This is NOT an exact paper reproduction. Pure teachers and Malliavin differ from
the paper's switching DSM. Our fixed lr=2e-4 and inherited forward schedule replace
the paper's hyperparameter search. Spectrum defaults to nmax=10 (a comparison
choice, not claimed numerically adequate at every small diffusion time).
Malliavin uses one probe, no RB, and both lambda=0 and lambda=5 in separate runs.
Other methods have no extra exponential time weighting. DSM like_w=false and
ISM like_w=true preserve their respective native weighting. Architecture, dtype,
data location and forward settings are inherited from the verified training base;
inspect prepared configs on mims before claiming full protocol equivalence.
Validation likelihood is enabled every 10000 steps; plotting and test evaluation
are disabled during training. Training completion does not imply valid checkpoints.

Example preparation:

```bash
python scripts/prepare_earth_long_comparison.py --output "$PWD/results/earth_long_v1" --check-only
python scripts/prepare_earth_long_comparison.py --output "$PWD/results/earth_long_v1"
export EARTH_LONG_ROOT="$PWD/results/earth_long_v1"
qsub -v EARTH_LONG_ROOT job/earth_long_comparison.q
```

Do not submit until checking the prepared configs on the server. Existing roots
and repeated subjob starts are refused. The entire 45-job array need not execute
concurrently; the scheduler controls allocation. No jobs were run locally.
Local verification is Python/bash syntax only; full Hydra integration requires
server validation. Existing long runs and checkpoints are never deleted.
