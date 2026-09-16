# NYC Taxi SPD benchmark

Use the published SPD-DDPM conditional NYC Taxi data, not raw TLC trips or EEG.
Sources: https://github.com/li-yun-chen/SPD-DDPM (data/condition and condition/utils.py),
https://arxiv.org/abs/2312.08200 and Ko–Lee https://arxiv.org/html/2605.31106v1 (§6.4, B.3).
Source revision: a6a65d13d80369baa8f2dc8eac352a14ee4a919e.

The importer uses only NumPy and Python's standard library. It preserves the
published matrix and predictor values: 10×10 matrices, 13 continuous predictors.
It skips the exported CSV index column. No SPD repair, filtering or additional
normalization is applied. Published training data: 7,600 rows. Published test:
1,159 rows, retained unchanged. Reserve 15% of published training using
RandomState(0) for validation: train 6,460 / val 1,140. This is not a verified
chronological split; upstream predictor preprocessing is inherited. File hashes,
source revision and partition information are recorded in metadata.

## Scope

This is a controlled comparison of our Varadhan, ISM and Malliavin objectives,
not an exact Ko–Lee reproduction (their SPD-Net/PINN architecture differs).
All methods use the same SPDGenerator, schedule and continuous predictor inputs.
Malliavin defaults to lambda=0; lambda=5 is a separately named ablation.
The empirical forward terminal uses training matrices only and is independent
of the requested predictors. It is an approximation to a conditional terminal;
conditional score training does not make that terminal exact. This limitation
must accompany conditional evaluation. Test target matrices never seed generation.
Each generation job fixes one validation/test predictor row and produces repeated
samples for that condition. Do not interpret those samples as the whole test set.

## Server preparation and smoke

```bash
source ~/riemannian_env.sh
cd ~/github/riemannian-score-sde-malliavin
git pull --ff-only
python scripts/build_spd_taxi_dataset.py --download
python scripts/smoke_spd_taxi.py --timeout 600
```

The default smoke composes configs, performs one update per objective, and checks
checkpoint step and finite state. It does not assert that a one-update model is a
good generator. Optional `--generation` also checks 8 samples at 64 reverse steps,
with at most 16 attempts; numerical failures remain errors and are not hidden.
Outputs are in a fresh results directory; failure prints the underlying log tail.
No production experiment runs automatically.

## Production commands (after server validation)

```bash
SPD_TAXI_ROOT=$(mktemp -d "$PWD/results/spd_taxi_production_XXXXXX")
for method in varadhan ism malliavin_hutchinson; do
  dest="$SPD_TAXI_ROOT/${method}_seed0"
  python main.py "experiment=spd_taxi_$method" mode=train seed=0 steps=100000 \
    logger=csv generation.enabled=false "ckpt_dir=$dest/ckpt" "hydra.run.dir=$dest" || break
done
```

For each method, sample one held-out condition (index 0 shown; repeat across held-out
rows for evaluation). Specify the production root from the training command:

```bash
method=varadhan
SPD_TAXI_OUT=$(mktemp -d "$SPD_TAXI_ROOT/${method}_generation_XXXXXX")
python main.py "experiment=spd_taxi_$method" mode=test seed=0 logger=csv \
  generation.enabled=true generation.context_split=test generation.context_index=0 \
  generation.count=20 generation.batch_size=20 generation.steps=64 \
  "ckpt_dir=$SPD_TAXI_ROOT/${method}_seed0/ckpt" \
  "hydra.run.dir=$SPD_TAXI_OUT" "generated_samples_path=$SPD_TAXI_OUT/generated_samples.npy"
```

Compare conditional sample means against each corresponding held-out target with
Frobenius and AIRM distances; also report rejection rates, spread and multiple seeds.
Do not select methods using test results or compare differing predictor rows.

## Final fits after choosing lambda

The default `dataset.split_protocol=development` retains 6,460 / 1,140 / 1,159.
Use `dataset.split_protocol=published_train` for final fits: 7,600 / 0 / 1,159.
This unions the saved train and val indices and restores source row order. The
NPZ, matrices, predictors, and test indices are unchanged. All methods and seeds
must use the same protocol. Select lambda on development validation first, then
freeze it; do not use final test metrics to select lambda.

Start new runs from scratch in new output directories, not by resuming development
checkpoints. Pass the SAME split override during generation: it also determines
which training matrices populate the empirical forward terminal. Generation
metadata records the protocol and split counts. Validation conditions cannot be
selected in published_train mode; train_val/test_val must remain false.
The saved Hydra config records the training protocol; old configs without the new
field continue to mean development. Existing server processes/data/checkpoints do
not need changes. Pull this code on the server after the current experiment ends.

Optional one-update server validation (no production training):

```bash
python scripts/smoke_spd_taxi.py --split-protocol published_train --timeout 600
```

Final training example, AFTER selecting lambda (substitute the chosen value):

```bash
SPD_FINAL_ROOT=$(mktemp -d "$PWD/results/spd_taxi_final_XXXXXX")
# Example only: use the lambda chosen on validation, not an assumed winner.
SPD_CHOSEN_LAMBDA=5.0
for seed in 0 1 2; do
  dest="$SPD_FINAL_ROOT/malliavin_lambda${SPD_CHOSEN_LAMBDA}_seed${seed}"
  python main.py experiment=spd_taxi_malliavin_hutchinson \
    dataset.split_protocol=published_train mode=train resume=false \
    "seed=$seed" steps=100000 logger=csv generation.enabled=false \
    loss.time_weighting=true "loss.time_weight_lambda=$SPD_CHOSEN_LAMBDA" \
    "ckpt_dir=$dest/ckpt" "hydra.run.dir=$dest" || break
done
```

For lambda=0, use `loss.time_weighting=false loss.time_weight_lambda=0.0`.
For Varadhan/ISM final fits, use their experiment name, the same split override
and seeds, and no time weighting. During final generation add
`dataset.split_protocol=published_train` to the generation command above and
point ckpt_dir to the corresponding final-fit checkpoint. This aligns the training
rows with the SPD-DDPM public release, not all architecture/optimization details.

## Fixed development validation comparison (no retraining)

```bash
python scripts/validate_spd_taxi.py \
  --run-root results/spd_taxi_comparison_uZ0en8
```

Defaults: select 100 of the 1,140 development validation conditions using
RandomState(2026), generate 20 matrices each at 64 reverse steps, generation seed
123 folded with the validation index, at most 40 attempts per condition. Four
methods use identical conditions and initial RNG keys (rejection/refill can cause
later sampling paths to differ). Test is never selected. Each model restores
once and reuses a JIT sampler across conditions. Existing saved Hydra configs
supply architecture/lambda; checks enforce the expected experiment, seed 0,
100,000 updates and development training data. Dataset/config/checkpoint hashes
and selected rows are saved in manifest.json. Checkpoints are verified unchanged.
A fresh output is allocated by default. Resume with `--output <printed-directory>`
and the same arguments. Do not run two launchers against the same output directory.
Use `--conditions 2` for a small server integration check before the full run.
`--timeout` defaults to 7,200 seconds per method. Runtime has not been measured.

Primary metrics follow the SPD-DDPM reference code's definitions:
https://github.com/li-yun-chen/SPD-DDPM/blob/a6a65d13d80369baa8f2dc8eac352a14ee4a919e/exp2/frechet_mean.R
https://github.com/li-yun-chen/SPD-DDPM/blob/a6a65d13d80369baa8f2dc8eac352a14ee4a919e/exp2/conditional_dis.py
For each condition, estimate the AIRM Frechet mean of 20 samples and compute its
squared AIRM distance and (unsquared) Frobenius norm distance to the real matrix.
Average over conditions. Report ordinary AIRM distance and the Frobenius error
of the arithmetic mean separately. The reference evaluation uses the first 1,100
test rows and removes some failed rows; ours uses development validation, reports
failures explicitly, and does not claim its numbers are directly comparable to the
published test table. Primary selection criterion, fixed before observing results:
mean squared AIRM error, checked alongside rejection rate and convergence. Do not
select a winner from incomplete conditions or unconverged means.

Additional, clearly separate diagnostics:
- Per-condition sample-to-target AIRM distances and pairwise sample spread.
- An AIRM energy expression (diagnostic only, no claim of strict propriety).
- Eigenvalues, condition numbers and log determinants for generated and real data.
- Pooled log-Euclidean Gaussian-kernel MMD squared (biased V-statistic); bandwidth
  squared is the median positive squared distance between selected validation
  targets, shared by all methods. Flatten full matrix logarithms so Euclidean
  distance equals log-matrix Frobenius distance. No AIRM-kernel positive-definiteness
  assumption is needed.
- Log-Euclidean NN distances in both directions between generated/validation,
  plus generated-to-train and validation-to-train. These are descriptive; small
  distances alone neither prove quality nor memorization. The pooled 2,000 generated
  matrices vs 100 targets ignore the condition pairing and have unequal sample
  sizes. Do not treat them as 2,000 independent validation conditions or use a
  sample-level significance test.

All generated arrays and per-condition JSON are retained. Summary has completion
counts, paired lambda5-minus-lambda0 squared AIRM errors, and rejection totals.
Finite matrices are not repaired; bounded rejection produces a distribution
conditioned on validity and this remains explicit. Frechet nonconvergence is
reported, not replaced by an arithmetic mean. Full GPU execution must be verified
on the server. Metrics can be resummarized without generating again:

```bash
python scripts/validate_spd_taxi.py --summarize-only --output <evaluation-directory>
```

The common Frechet gradient tolerance is now `1e-6` (previously `1e-7`).
The two-condition pilot had residual norms of about `1e-7` to `2.2e-7`
when strict cost-decrease line searches stopped; this motivates a uniform
numerical tolerance change, not a method-specific exception. Each solver report
records tolerance, iteration limit and termination reason. Unconverged primary
metrics remain null. This does not establish generation quality or select lambda.

Recompute **all methods** from existing arrays, without JAX, training or generation:

```bash
python scripts/validate_spd_taxi.py --recompute-metrics --output <evaluation-directory>
```

This verifies dataset/sample hashes, copies samples unchanged into a new
`metrics_recomputed_*` child directory and recalculates per-condition metrics and
summary there. Original reports and checkpoints remain untouched. The new
`metric_revision.json` records the source report hashes and tolerance. Use the
new summary consistently; do not combine reports made with different tolerances.
`--summarize-only` only aggregates existing per-condition metrics; it does not
rerun the Frechet solver.

Solver version 2 uses Cholesky whitening and triangular solves for both the
objective and the tangent residual. The update is `L exp(T) L.T`; the whitened
frame may change but the AIRM mean is unchanged mathematically. Convergence
still requires residual < 1e-6 and line search still requires strict objective
reduction. No clipping, jitter, residual-only acceptance, or tolerance increase
is used. Two supplied stalled cases (ISM val_0491 and lambda5 val_0096) converged
locally with this implementation; verify all conditions on the server using
`--recompute-metrics`. Reports record solver version and whitening method.
