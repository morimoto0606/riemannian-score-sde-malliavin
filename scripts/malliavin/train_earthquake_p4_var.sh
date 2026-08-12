#!/bin/bash
set -euo pipefail

python -u main.py \
  experiment=earthquake_malliavin_hutchinson \
  mode=train \
  logger=csv \
  seed=0 \
  teacher.hutchinson_probes=4 \
  hydra.run.dir="results/earthquake_malliavin_p4_default_600k"