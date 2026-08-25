# SO(3) Varadhan / ISM / Malliavin experiments

The six formal experiment configs keep the upstream SO(3) wrapped-normal
mixture, Lie-algebra score network, optimizer, SDE, batch size, and 100,000
training updates fixed.  They cross three objectives with the two requested
additional time-weight values:

| Objective | lambda=0 | lambda=5 |
|---|---|---|
| Varadhan DSM | `so3_varadhan_lambda0` | `so3_varadhan_lambda5` |
| ISM | `so3_ism_lambda0` | `so3_ism_lambda5` |
| Malliavin DSM | `so3_malliavin_hutchinson_lambda0` | `so3_malliavin_hutchinson_lambda5` |

All six configs set `loss.time_weighting=true`.  Thus lambda zero is an exact
no-op but remains an explicit, logged experimental condition.  ISM's
`loss.like_w=true` is its pre-existing diffusion/likelihood weighting.  The
additional `exp(-loss.time_weight_lambda * t)` factor is applied afterwards;
the two settings are stored independently in Hydra and CSV logger hyperparameters.

Example commands:

```bash
python -u main.py \
  experiment=so3_varadhan_lambda0 \
  logger=csv seed=0 \
  hydra.run.dir=results/so3_varadhan_lambda0

python -u main.py \
  experiment=so3_ism_lambda5 \
  logger=csv seed=0 \
  hydra.run.dir=results/so3_ism_lambda5

python -u main.py \
  experiment=so3_malliavin_hutchinson_lambda5 \
  logger=csv seed=0 \
  hydra.run.dir=results/so3_malliavin_hutchinson_lambda5
```

Before a full Malliavin run, use a small smoke test because endpoint Jacobians
through 100 matrix-exponential steps can consume much more memory than the
upstream batch size:

```bash
python -u main.py \
  experiment=so3_malliavin_hutchinson_lambda0 \
  mode=train logger=csv seed=0 \
  flow.N=2 batch_size=8 eval_batch_size=8 steps=10 \
  train_val=false train_plot=false test_val=false test_test=false test_plot=false \
  hydra.run.dir=results/so3_malliavin_smoke
```

Training logs `train/time_per_it` at validation boundaries, plus final
`train/mean_time_per_it`, compute-only `train/total_time`, and end-to-end
`train/wall_time`.

After all requested runs have produced `generated_samples.npy`, evaluate them
against one shared target sample.  Repeat `--run` for any subset of models:

```bash
python -u scripts/evaluate_so3_models.py \
  --run varadhan_l0=results/so3_varadhan_lambda0 \
  --run ism_l0=results/so3_ism_lambda0 \
  --run malliavin_l0=results/so3_malliavin_hutchinson_lambda0 \
  --output-dir results/so3_comparison_lambda0 \
  --teacher-diagnostic-run-dir results/so3_malliavin_hutchinson_lambda0
```

The evaluator writes:

- `shared_target_samples.npy`
- one `euler_target_vs_<model>.png` per model
- `so3_metrics.csv` and `so3_metrics.json`
- `teacher_diagnostics.csv` when teacher diagnostics are requested

Nearest-neighbour distances and the RBF kernel use the physical relative
rotation angle in radians.  MMD is the unbiased estimator and may therefore be
slightly negative at finite sample size.  Pairwise calculations are chunked;
`--metric-subsample` controls their quadratic compute cost.
