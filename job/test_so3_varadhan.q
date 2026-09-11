#!/bin/bash
#PBS -N so3_varadhan_seeds
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

if [[ -z "$SEED" ]]; then
    echo "PBS array index was not found."
    exit 1
fi

RUN_DIR="results/so3_varadhan_target0_seed${SEED}"
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
  experiment=so3_varadhan \
  mode=train \
  seed="${SEED}" \
  dataset.seed="${TARGET_SEED}" \
  steps=100000 \
  logger=csv \
  loss.time_weighting=true \
  loss.time_weight_lambda=0.0 \
  "hydra.run.dir=${RUN_DIR}" \
  2>&1 | tee "$RUN_LOG"