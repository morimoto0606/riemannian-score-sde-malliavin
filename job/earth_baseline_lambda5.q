#!/bin/bash
#PBS -N earth_baseline_l5
#PBS -j oe
#PBS -q hi
#PBS -l ncpus=10
#PBS -J 0-8
set -euo pipefail
cd "$HOME/riemannian-score-sde-malliavin"
source "$HOME/venvs/riemannian-score-sde-py39/bin/activate"
export GEOMSTATS_BACKEND=jax PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10
export PYTHONPATH="$PWD/geomstats:$PWD"
export MPLCONFIGDIR
MPLCONFIGDIR=$(mktemp -d /tmp/earth-baseline-l5-XXXXXX)
python scripts/earth_baseline_lambda5.py \
  --output "$EARTH_L5_ROOT" --index "${PBS_ARRAY_INDEX:-${PBS_ARRAYID:?Missing array index}}"
