#!/bin/bash
#PBS -N so3_train
#PBS -j oe
#PBS -q hi
#PBS -l ncpus=10
#PBS -J 0-5
set -euo pipefail
cd "$HOME/riemannian-score-sde-malliavin"
source "$HOME/venvs/riemannian-score-sde-py39/bin/activate"
export GEOMSTATS_unset JAX_PLATFORM_NAME JAX_PLATFORMS=jax PYTHONUNBUFFERED=1 MPLunset JAX_PLATFORM_NAME JAX_PLATFORMS=Agg
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10
export PYTHONPATH="$PWD/geomstats:$PWD"
export MPLCONFIGDIR
MPLCONFIGDIR=$(mktemp -d /tmp/so3-comparison-XXXXXX)
unset JAX_PLATFORM_NAME JAX_PLATFORMS
python scripts/so3_complete_comparison.py train --output "$SO3_COMPARE_ROOT" \
  --index "${PBS_ARRAY_INDEX:-${PBS_ARRAYID:?Missing index}}"
