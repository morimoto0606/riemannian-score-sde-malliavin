#!/bin/bash
#PBS -N earth_malliavin_l0
#PBS -j oe
#PBS -q hi
#PBS -l ncpus=10
#PBS -J 0-8
set -euo pipefail
cd "$HOME/riemannian-score-sde-malliavin"
source "$HOME/venvs/riemannian-score-sde-py39/bin/activate"
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg GEOMSTATS_BACKEND=jax
export PYTHONPATH="$PWD/geomstats:$PWD"
export MPLCONFIGDIR="/tmp/matplotlib-$USER-${PBS_JOBID}"
mkdir -p "$MPLCONFIGDIR"
INDEX="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}"
if [[ -z "$INDEX" ]]; then
    echo 'Missing PBS array index' >&2
    exit 1
fi
python scripts/run_earth_malliavin_lambda0.py --index "$INDEX"
