#!/bin/bash
#PBS -N earth_100k
#PBS -j oe
#PBS -q hi
#PBS -l ncpus=10
#PBS -J 0-71
set -euo pipefail
cd "$HOME/riemannian-score-sde-malliavin"
source "$HOME/venvs/riemannian-score-sde-py39/bin/activate"
export GEOMSTATS_BACKEND=jax PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10
export PYTHONPATH="$PWD/geomstats:$PWD"
export MPLCONFIGDIR
MPLCONFIGDIR=$(mktemp -d /tmp/earth-100k-XXXXXX)
python - <<'CHECK'
import json, os
from pathlib import Path
manifest = json.loads((Path(os.environ['EARTH_100K_ROOT'])/'manifest.json').read_text())
if manifest.get('smoke') or any(r['steps'] != 100000 for r in manifest['runs']):
    raise SystemExit('Production job requires 100k configurations')
CHECK
python scripts/earth_100k_comparison.py \
  --output "$EARTH_100K_ROOT" --index "${PBS_ARRAY_INDEX:-${PBS_ARRAYID:?Missing array index}}"
