#!/bin/bash
set -euo pipefail

python -u main.py \
  experiment=earthquake_malliavin_hutchinson \
  mode=train \
  logger=csv \
  seed=0 \
  teacher.hutchinson_probes=16 \
  hydra.run.dir="results/earthquake_malliavin_p16_default_600k"