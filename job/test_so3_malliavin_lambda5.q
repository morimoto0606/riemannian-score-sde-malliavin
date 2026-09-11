#!/bin/bash
#PBS -N so3_malliavin_seed1
#PBS -j oe
#PBS -q hi
#PBS -l ncpus=10
#PBS -J 0-2

set -euo pipefail

export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-$USER-${PBS_JOBID}"

mkdir -p "$MPLCONFIGDIR"

cd "$HOME/riemannian-score-sde-malliavin"
source "$HOME/venvs/riemannian-score-sde-py39/bin/activate"

export GEOMSTATS_BACKEND=jax
export PYTHONPATH="$PWD/geomstats:$PWD"

SEED="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}"
TARGET_SEED=0

RUN_DIR="results/so3_malliavin_lambda5_target0_seed${SEED}"
RUN_LOG="${RUN_DIR}/run.log"

mkdir -p "$RUN_DIR"

echo "PBS job ID: ${PBS_JOBID}"
echo "Training seed: ${SEED}"
echo "Target seed: ${TARGET_SEED}"
echo "Run directory: ${RUN_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

python - <<'PY'
import jax
print("JAX devices:", jax.devices())
PY

python -u main.py \
  experiment=so3_malliavin_hutchinson \
  mode=train \
  seed="${SEED}" \
  dataset.seed="${TARGET_SEED}" \
  steps=100000 \
  logger=csv \
  loss.time_weighting=true \
  loss.time_weight_lambda=5.0 \
  "hydra.run.dir=${RUN_DIR}" \
  2>&1 | tee "$RUN_LOG"