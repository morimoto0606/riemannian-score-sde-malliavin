#!/bin/bash
#PBS -N probe_ablation
#PBS -j oe
#PBS -q hi
#PBS -l ncpus=10
set -euo pipefail
set -o noclobber
cd "$HOME/riemannian-score-sde-malliavin"
source "$HOME/venvs/riemannian-score-sde-py39/bin/activate"
: "${PROBE_ROOT:?Prepare an ablation and pass PROBE_ROOT to qsub}"
test -f "$PROBE_ROOT/PREPARED"
export GEOMSTATS_BACKEND=jax PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10
export MPLCONFIGDIR
MPLCONFIGDIR=$(mktemp -d /tmp/probe-ablation-XXXXXX)
# The runner isolates CPU generation/checkpoint checks from GPU training.
# Serial training avoids contention among the six new runs on the shared V100.
# This follows the existing site's hi queue convention; it is not a GPU reservation.
python -u "$PROBE_ROOT/source/scripts/probe_ablation.py" batch --output "$PROBE_ROOT" \
  > "$PROBE_ROOT/batch.log" 2>&1
