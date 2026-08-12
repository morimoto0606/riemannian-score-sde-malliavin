#!/bin/bash
set -euo pipefail

python -u main.py \
  experiment=earthquake_malliavin_hutchinson \
  mode=train \
  logger=csv \
  seed=0 \
  teacher.hutchinson_probes=32 \
  hydra.run.dir="results/earthquake_malliavin_p32_default_600k"