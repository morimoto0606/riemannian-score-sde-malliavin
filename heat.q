#!/bin/sh
#PBS -N heat
#PBS -j oe
#PBS -l ncpus=48
#PBS -q hi

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48

export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"

cd "$HOME/scoremodel/upstream/riemannian-score-sde"

source "$HOME/venvs/riemannian-score-sde-py39/bin/activate"

export GEOMSTATS_BACKEND=jax

echo "python:"
which python

echo "geomstats:"
python -m pip show geomstats

echo "backend test:"
python - <<'PY'
import geomstats.backend as gs
print("backend OK")
PY

python main.py  experiment=earthquake  model=rsgm  loss=dsm0  mode=train  steps=100000  val_freq=1000  batch_size=128  eval_batch_size=128  train_val=false  train_plot=false  test_val=false  test_test=false  test_plot=false  logger=csv  seed=0
