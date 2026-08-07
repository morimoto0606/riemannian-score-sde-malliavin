#!/bin/sh
#PBS -N earthquake_varadhan
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
export PYTHONPATH="$PWD/geomstats:$PWD:$PYTHONPATH"

python main.py \
 experiment=earthquake_varadhan \
 mode=train \
 train_val=false \
 train_plot=false \
 test_val=false \
 test_test=false \
 test_plot=false \
 logger=csv \
 seed=0 \
 hydra.run.dir=$HOME/scoremodel/results/earthquake_teacher_comparison/upstream_varadhan