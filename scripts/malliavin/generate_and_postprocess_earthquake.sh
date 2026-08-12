#!/bin/bash
set -euo pipefail

RUN_DIR="results/earthquake_malliavin_p8_default_600k"

python -u main.py \
  experiment=earthquake_malliavin_hutchinson \
  mode=test \
  logger=csv \
  seed=0 \
  teacher.hutchinson_probes=8 \
  test_val=false \
  test_test=false \
  test_plot=true \
  generated_samples_path="$PWD/$RUN_DIR/generated_samples.npy" \
  hydra.run.dir="$RUN_DIR"

python -u scripts/postprocess_earthquake_upstream.py \
  --run-dir "$RUN_DIR"