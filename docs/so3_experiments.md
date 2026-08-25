# SO(3) Varadhan / ISM / Malliavin experiments

SO(3) experiments follow the same configuration convention as
`earthquake_malliavin_hutchinson`: an experiment YAML identifies only the
method, while time weighting, training length, seed, and run directory are CLI
overrides.

| Method | Experiment config |
|---|---|
| Varadhan DSM | `so3_varadhan` |
| ISM | `so3_ism` |
| Malliavin DSM | `so3_malliavin_hutchinson` |

`loss.time_weighting` and `loss.time_weight_lambda` are shared by the three
methods. ISM's existing `loss.like_w` remains a separate setting. Malliavin
covariance regularization is independently controlled by
`teacher.covariance_regularization`.

Example Malliavin run:

```bash
python -u main.py \
  experiment=so3_malliavin_hutchinson \
  mode=train \
  logger=csv \
  seed=0 \
  teacher.hutchinson_probes=1 \
  teacher.rb_enabled=false \
  loss.time_weighting=true \
  loss.time_weight_lambda=5.0 \
  steps=100000 \
  hydra.run.dir=results/so3_malliavin_time_weight_lambda5_100k
```

Varadhan and ISM use the same CLI pattern:

```bash
python -u main.py \
  experiment=so3_varadhan \
  mode=train \
  logger=csv \
  seed=0 \
  loss.time_weighting=true \
  loss.time_weight_lambda=5.0 \
  steps=100000 \
  hydra.run.dir=results/so3_varadhan_time_weight_lambda5_100k

python -u main.py \
  experiment=so3_ism \
  mode=train \
  logger=csv \
  seed=0 \
  loss.time_weighting=true \
  loss.time_weight_lambda=5.0 \
  steps=100000 \
  hydra.run.dir=results/so3_ism_time_weight_lambda5_100k
```

Use `loss.time_weight_lambda=0.0` with the same experiment names for the
unweighted comparison. No lambda-specific experiment YAML is required.

After generation, one run can be postprocessed without regenerating its model
samples or loading its checkpoint:

```bash
python -u scripts/evaluate_so3_generation.py \
  --run-dir results/so3_malliavin_hutchinson_lambda0
```

This writes `metrics.json`, `metrics.csv`, `generated_vs_data.png`,
`nearest_neighbor_summary.png`, and `mmd_summary.png` below the run's
`evaluation/` directory. The shared `scripts/evaluate_so3_models.py` utilities
provide the matrix validation, saved-config target reconstruction, repository
SO(3) geodesic convention, nearest-neighbour calculation, MMD, and constraint
metrics used by this command.
