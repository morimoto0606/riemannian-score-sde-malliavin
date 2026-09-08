#!/bin/bash
#PBS -N h2_comparison
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
export MPLCONFIGDIR="/tmp/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"

cd "${PBS_O_WORKDIR:-$HOME/riemannian-score-sde-malliavin}"
source "$HOME/venvs/riemannian-score-sde-py39/bin/activate"
export GEOMSTATS_BACKEND=jax
export PYTHONPATH="$PWD/geomstats:$PWD"

TRAINING_SEED="${PBS_ARRAY_INDEX}"
TARGET_SEED=0
H2_METHOD="${H2_METHOD:?submit with -v H2_METHOD=varadhan|ism|malliavin_lambda0|malliavin_lambda5}"

case "$H2_METHOD" in
  varadhan)
    EXPERIMENT=h2_varadhan
    EXTRA_OVERRIDES=()
    ;;
  ism)
    EXPERIMENT=h2_ism
    EXTRA_OVERRIDES=()
    ;;
  malliavin_lambda0)
    EXPERIMENT=h2_malliavin_hutchinson
    EXTRA_OVERRIDES=(loss.time_weight_lambda=0.0)
    ;;
  malliavin_lambda5)
    EXPERIMENT=h2_malliavin_hutchinson
    EXTRA_OVERRIDES=(loss.time_weight_lambda=5.0)
    ;;
  *)
    echo "Unknown H2_METHOD: $H2_METHOD" >&2
    exit 2
    ;;
esac

OUTDIR="results/h2_${H2_METHOD}/seed_${TRAINING_SEED}"
mkdir -p "$OUTDIR"

python -u main.py \
  "experiment=${EXPERIMENT}" \
  mode=all \
  logger=csv \
  "seed=${TRAINING_SEED}" \
  "dataset.seed=${TARGET_SEED}" \
  steps=100000 \
  train_val=false \
  train_plot=false \
  test_val=false \
  test_test=false \
  test_plot=true \
  "hydra.run.dir=${OUTDIR}" \
  "${EXTRA_OVERRIDES[@]}" \
  > "${OUTDIR}.log" 2>&1
