# Volcano commands

Run from the repository root in the server's existing training environment.
The dataset config is `config/dataset/volcano.yaml`. Experiments are `volcano`,
`volcano_heat`, `volcano_ism`, `volcano_malliavin_hutchinson`,
`volcano_spectrum`, and `volcano_varadhan`.

```bash
export GEOMSTATS_BACKEND=jax
export PYTHONPATH="$PWD/geomstats:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The upstream CSV `data/volerup.csv` and class `VolcanicErruption` are retained:
CSV parsing, coordinates, model architecture and checkpoint format are unchanged.

## Validation without training

```bash
python -m pytest -q tests/test_volcano_configs.py tests/test_s2_earth_postprocess.py
for EXP in volcano volcano_heat volcano_ism volcano_malliavin_hutchinson volcano_spectrum volcano_varadhan; do
  python main.py --cfg job --resolve experiment="$EXP"
done
GEOMSTATS_BACKEND=jax python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
root = Path.cwd()
with initialize_config_dir(config_dir=str(root / 'config'), version_base='1.1'):
    cfg = compose(config_name='main', overrides=['experiment=volcano'])
cfg.work_dir = str(root)
dataset = instantiate(cfg.dataset)
points = np.asarray(dataset.data)
assert points.ndim == 2 and points.shape[1] == 3
assert np.isfinite(points).all()
np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1, atol=1e-6)
print(type(dataset).__name__, points.shape)
PY
```

Optional one-update smoke test; no evaluation, plots or generation:

```bash
SMOKE="$(pwd)/results/volcano_heat_smoke_$(date +%Y%m%d_%H%M%S)"
test ! -e "$SMOKE" && python main.py experiment=volcano_heat mode=train \
  steps=1 batch_size=2 eval_batch_size=2 warmup_steps=0 \
  train_val=false train_plot=false test_val=false test_test=false test_plot=false \
  logger=csv "hydra.run.dir=$SMOKE"
```

## New training (100K updates; commands only)

This loop is for fresh runs. It stops if a destination already exists, including
a renamed trained Malliavin run. Do not rerun training over existing checkpoints.
Malliavin and Spectrum use lambda 5; Heat and ISM retain default weighting.

```bash
set -e
ROOT="$(pwd)"
for METHOD in malliavin heat ism spectrum; do
  EXP="volcano_${METHOD}"
  TAG="$EXP"
  EXTRA=()
  PRECISION=false
  if [ "$METHOD" = malliavin ]; then EXP=volcano_malliavin_hutchinson; fi
  if [ "$METHOD" = malliavin ] || [ "$METHOD" = spectrum ]; then
    TAG="volcano_${METHOD}_lambda5"
    EXTRA=(loss.time_weighting=true loss.time_weight_lambda=5.0)
  fi
  if [ "$METHOD" = spectrum ]; then PRECISION=true; fi
  for SEED in 0 1 2; do
    RUN="$ROOT/results/${TAG}_seed${SEED}"
    if [ -e "$RUN" ]; then printf 'Already exists: %s\n' "$RUN" >&2; exit 1; fi
    JAX_ENABLE_X64="$PRECISION" python -u main.py \
      experiment="$EXP" mode=train seed="$SEED" steps=100000 \
      batch_size=128 eval_batch_size=128 val_freq=10000 logger=csv \
      train_val=false train_plot=false test_val=false test_test=false test_plot=false \
      "${EXTRA[@]}" "hydra.run.dir=$RUN" \
      "generated_samples_path=$RUN/generated_samples.npy"
  done
done
```

## Generation from existing checkpoints

`main.py` composes the current experiment; `run.py` restores only
`ckpt_dir/tree.pkl` and `ckpt_dir/arrays.npy` in `mode=test`. It does not reload
the training run's saved Hydra config. An absolute `ckpt_dir` therefore allows
directory migration without editing checkpoint files or historical metadata.
The model/geometry and precision must match the actual training run; reproduce
any custom architecture, flow, split or precision overrides from that run.
The defaults below assume the standard experiments and float32 Malliavin/Heat/ISM.

Generation gets a fresh run directory with a current saved config. Postprocess
that directory, so historical metadata remains untouched. `mode=test` does not
save checkpoint state. Plotting is needed by the existing generation path;
Cartopy/plot dependencies must be installed on the server.

```bash
set -e
ROOT="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
for METHOD in malliavin heat ism spectrum; do
  EXP="volcano_${METHOD}"
  TAG="$EXP"
  EXTRA=()
  PRECISION=false
  if [ "$METHOD" = malliavin ]; then EXP=volcano_malliavin_hutchinson; fi
  if [ "$METHOD" = malliavin ] || [ "$METHOD" = spectrum ]; then
    TAG="volcano_${METHOD}_lambda5"
    EXTRA=(loss.time_weighting=true loss.time_weight_lambda=5.0)
  fi
  if [ "$METHOD" = spectrum ]; then PRECISION=true; fi
  for SEED in 0 1 2; do
    RUN="$ROOT/results/${TAG}_seed${SEED}"
    OUT="$ROOT/results/${TAG}_seed${SEED}_generation_${STAMP}"
    test -s "$RUN/ckpt/tree.pkl"
    test -s "$RUN/ckpt/arrays.npy"
    if [ -e "$OUT" ]; then printf 'Already exists: %s\n' "$OUT" >&2; exit 1; fi
    JAX_ENABLE_X64="$PRECISION" python -u main.py \
      experiment="$EXP" mode=test seed="$SEED" \
      batch_size=128 eval_batch_size=128 logger=csv \
      test_val=false test_test=false test_plot=true "${EXTRA[@]}" \
      "ckpt_dir=$RUN/ckpt" "hydra.run.dir=$OUT" \
      "generated_samples_path=$OUT/generated_samples.npy"
    python scripts/postprocess_s2_earth_data.py --run-dir "$OUT"
  done
done
```

Local validation: Python syntax and static config reference checks pass. Numerical
dataset instantiation is blocked by the installed JAX binary's AVX requirement;
Hydra and OmegaConf are absent and package installation was declined. The tests
and composition commands above still need to run in the server environment.
No training, generation, results edits or checkpoint edits were performed locally.
