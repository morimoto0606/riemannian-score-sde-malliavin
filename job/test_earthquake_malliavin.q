#!/bin/bash
#PBS -N earthquake_mallaivin
#PBS -j oe
#PBS -l ncpus=48
#PBS -q hi

set -euo pipefail

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"

cd "$HOME/riemannian-score-sde-malliavin"
source "$HOME/venvs/riemannian-score-sde-py39/bin/activate"
export GEOMSTATS_BACKEND=jax
export PYTHONPATH="$PWD/geomstats:$PWD"

python - <<'PY'
import geomstats
print(geomstats.__file__)
PY


python -u main.py \
  experiment=earthquake_malliavin_hutchinson \
  mode=train \
  steps=2000 \
  batch_size=32 \
  eval_batch_size=32 \
  logger=csv \
  train_val=false \
  train_plot=false \
  test_val=false \
  test_test=false \
  test_plot=false \
  loss.debug_teacher_comparison=true \
  hydra.run.dir=results/earthquake_malliavin_debug_2k \
  2>&1 | tee results/earthquake_malliavin_debug_2k.log

