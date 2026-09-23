#!/bin/bash
#PBS -N so3_evaluate
#PBS -j oe
#PBS -q hi
#PBS -l ncpus=6
#PBS -J 0-17
set -euo pipefail
cd "$HOME/riemannian-score-sde-malliavin"
source "$HOME/venvs/riemannian-score-sde-py39/bin/activate"
export GEOMSTATS_export JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""=jax PYTHONUNBUFFERED=1 MPLexport JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""=Agg
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export PYTHONPATH="$PWD/geomstats:$PWD"
export MPLCONFIGDIR
MPLCONFIGDIR=$(mktemp -d /tmp/so3-comparison-XXXXXX)
export JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
python scripts/so3_complete_comparison.py evaluate --output "$SO3_COMPARE_ROOT" \
  --index "${PBS_ARRAY_INDEX:-${PBS_ARRAYID:?Missing index}}"
