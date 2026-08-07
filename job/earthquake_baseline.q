#!/bin/bash
#PBS -N earthquake_heat_upstream
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

OUTDIR="results/earthquake_upstream_heat_baseline"
LOGFILE="results/earthquake_upstream_heat_baseline.log"

mkdir -p "$OUTDIR"

echo "========================================" > "$LOGFILE"
echo "MIMS Upstream Earthquake Heat START" >> "$LOGFILE"
echo "DATE=$(date)" >> "$LOGFILE"
echo "HOST=$(hostname)" >> "$LOGFILE"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS" >> "$LOGFILE"
echo "========================================" >> "$LOGFILE"


python -u main.py \
  experiment=earthquake_heat \
  mode=train \
  logger=csv \
  seed=0 \
  hydra.run.dir="$OUTDIR" \
  >> "$LOGFILE" 2>&1


echo "===== POSTPROCESS START =====" >> "$LOGFILE"

python -u scripts/postprocess_earthquake_upstream.py \
  --run-dir "$OUTDIR" \
  >> "$LOGFILE" 2>&1


echo >> "$LOGFILE"
echo "========================================" >> "$LOGFILE"
echo "MIMS Upstream Earthquake Heat END" >> "$LOGFILE"
echo "DATE=$(date)" >> "$LOGFILE"
echo "========================================" >> "$LOGFILE"