#!/bin/bash
#PBS -N earth_long_compare
#PBS -j oe
#PBS -q hi
#PBS -l ncpus=10
#PBS -J 0-44
set -euo pipefail
cd "$HOME/riemannian-score-sde-malliavin"
source "$HOME/venvs/riemannian-score-sde-py39/bin/activate"
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10
export GEOMSTATS_BACKEND=jax PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD/geomstats:$PWD"
export MPLCONFIGDIR="/tmp/matplotlib-$USER-${PBS_JOBID}"
mkdir -p "$MPLCONFIGDIR"
export EARTH_JOB_INDEX="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}"
python - <<'PY'
import json,os,sys,subprocess
from pathlib import Path
root=Path(os.environ['EARTH_LONG_ROOT']).resolve()
index=int(os.environ['EARTH_JOB_INDEX'])
manifest=json.loads((root/'manifest.json').read_text())
if len(manifest['runs']) != 45 or manifest.get('malliavin_lambdas') != [0.,5.]:
    raise ValueError('Expected the 45-run manifest with both Malliavin lambdas')
if not 0 <= index < 45:
    raise ValueError('Invalid array index')
dest=root/manifest['runs'][index]['name']
# Exclusive reservation prevents repeated submissions from overwriting checkpoints.
with (dest/'started.json').open('x') as f:
    json.dump({'job_id':os.environ.get('PBS_JOBID')},f)
with (dest/'launcher.log').open('x') as log:
    subprocess.run([sys.executable,'-u','main.py','--config-path',str(dest/'input_config'),
                    '--config-name','config','hydra.run.dir='+str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
(dest/'training_process_completed.json').write_text('{"returncode":0}\n')
PY
