# Final Taxi test evaluation

After verifying all nine 100000-update checkpoints are finite, run
`scripts/evaluate_spd_taxi_final.py --run-root results/spd_taxi_final_2zzhprra`.
The evaluator restores saved training configs and EMA parameters. It requires
published_train (7600 training rows, no validation rows) and verifies the dataset
against the training manifest, saved configs and checkpoint hashes. It never trains.

All 1159 held-out test contexts are evaluated for Varadhan, ISM and Malliavin
lambda0, each at training seeds 0,1,2. Each context gets 20 samples, 64 reverse GRW
steps, generation seed 123 folded with the test index. Sampling retries are capped
at 40 attempts per context and all rejection counts are retained. This produces
208620 accepted matrices if all conditions complete. This is a full evaluation,
not a short smoke test; the per-method/seed timeout defaults to 24 hours.

Metrics reuse the validation definitions: squared AIRM error of the sample
Frechet mean, Frobenius errors, sample distances and the AIRM energy diagnostic.
Frechet nonconvergence remains explicit. No jitter or relaxation is applied.
Three-seed mean and sample standard deviation for an error metric are produced
only when every test condition contributes in every seed.

Pooled log-Euclidean RBF MMD (biased squared estimator) and nearest-neighbor
statistics are supplementary marginal diagnostics. To bound quadratic cost,
a fixed RandomState(2026) permutation selects up to 2000 generated matrices per
run from the pooled 23180. Reference is all 1159 test targets; the bandwidth is
the median positive squared reference distance, common across all runs.
Training nearest neighbors use all 7600 training matrices. These are not a claim
of exact reproduction of another paper's evaluation protocol, and pooled scores
do not measure conditional prediction accuracy.

A fresh test_evaluation_* output directory is printed on launch. Each seed has
per-method test_XXXX.npy/json artifacts and checkpoint integrity reports; the
root has worker logs and summary.json with per-run and three-seed aggregates.
Do not select hyperparameters from these final test results.

To resume, repeat the command with --output pointing at the printed directory.
Only matching manifests can resume; complete sample/metric records are checked
by hash and skipped. Failed or missing conditions are retried. Concurrent runs
against the same output directory are unsupported; wait for the launcher to exit.
Use --summarize-only --output PATH to rebuild the summary without generation.
Local verification covers syntax and NumPy metric/aggregation tests, not GPU
integration or actual restored checkpoint generation.

## Fréchet solver version 3 and CPU-only revision

Version 3 computes the whitened matrix logarithm from the SVD of relative
Cholesky factors: if X=C C^T and M=L L^T, A=L^-1 C has singular values s,
and log(A A^T)=U diag(2 log(s)) U^T. It avoids explicitly forming the whitened
matrix before its eigendecomposition. The objective, strict descent test,
128-iteration limit and 1e-6 residual tolerance are unchanged; no clipping,
jitter, sample rejection or model-specific exception is added.
See https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
for the singular-value/eigenvalue identity.

On the supplied final seed0 Malliavin samples, test_0495 converged in 20
iterations with residual 8.299e-7; test_0899 in 95 with residual 4.623e-7.
Maximum sample condition numbers were about 1.24e5 and 8.76e10 respectively.
The older solver converged locally on 0495 but stalled on 0899; server/local
residual differences show sensitivity to floating-point implementation.
These checks do not guarantee convergence for every server sample.

Run the same new solver for all nine saved runs without training or generation:

```bash
python -u scripts/evaluate_spd_taxi_final.py --recompute-metrics \
  --output results/spd_taxi_final_2zzhprra/test_evaluation_29nley7i
```

This creates a fresh metrics_svd_* directory under the original evaluation.
All hashed sample arrays and original reports are preserved. Available reports
are copied, metrics recomputed from the same samples and targets, and the
summary rebuilt. Previously incomplete generation remains incomplete; it is
not repaired by this operation. Checkpoint integrity reports are copied as
historical evidence from generation; this CPU-only operation does not load
checkpoints. The revision records source report hashes and solver settings.
Compare methods using the revised reports consistently, rather than mixing
old and new solver outputs.

## Retry failed metrics from saved samples (2026-09-30)

The server diagnostic for `metrics_svd_zlbhvxay` identified seven ISM seed1
conditions. All seven already have 20 accepted samples saved:

- Test indices 213 and 644 failed with `Invalid generalized eigenvalues`.
- Indices 134, 504, 662, 1031 and 1096 reached the 128-iteration Fréchet limit.

`scripts/retry_spd_taxi_metrics.py` recomputes only failed or missing metrics
with complete saved sampling records. It scans all methods and seeds using
the same rule, preserves successful reports byte-for-byte, and rebuilds the
full three-seed summary. It does not train, sample, load checkpoints, or require
JAX/GPU execution. NumPy and SciPy are required.

AIRM distance now uses `2*log(svdvals(L_b^-1 L_a))` for Cholesky factors
`a=L_a L_a^T`, `b=L_b L_b^T`. This is the same mathematical distance, evaluated
without forming an ill-conditioned generalized eigenvalue problem. The
Fréchet solver remains version 3; retries keep tolerance `1e-6` and increase
only the iteration limit to 4096. No clipping, jitter or tolerance relaxation
is applied. Reaching a larger limit or a stalled line search still counts as
nonconvergence.

The retry creates a new directory. It checks dataset and sample hashes and
report identities, copies saved samples, and records source report hashes,
the code revision and numerical settings in `metric_revision.json`.
Successful source metrics retain their original numerical implementation;
only retried metrics use the new distance routine. The new reports identify
this with `distance_algorithm=cholesky_factor_svd`. The original revision is
never overwritten. `--dataset` accepts relocation to another server path
only when the file hash matches the source manifest.

On smp01, use the existing environment and the repository under `~/github`:

```bash
source ~/riemannian_env.sh
cd ~/github/riemannian-score-sde-malliavin
git pull --ff-only origin fix/cuda13-py312-jax0.7-compat
python -m unittest discover -s tests -p 'test_spd_validation_metrics.py' -v
python -m unittest discover -s tests -p 'test_spd_taxi_metric_retry.py' -v
python -m unittest discover -s tests -p 'test_spd_taxi_final_summary.py' -v
```

After the tests pass, run:

```bash
python -u scripts/retry_spd_taxi_metrics.py \
  --source results/spd_taxi_final_2zzhprra/test_evaluation_29nley7i/metrics_svd_zlbhvxay \
  --dataset data/spd_taxi/nyc_taxi.npz \
  --output results/spd_taxi_final_2zzhprra/test_evaluation_29nley7i/metrics_retry_20260930 \
  --frechet-max-iterations 4096
```

The output path must not exist. `Conditions to retry: 7` is expected for the
reported input revision. The final `retry_report.json` records recovered and
remaining conditions; `summary.json` contains the rebuilt comparison. If any
condition remains incomplete, the command exits nonzero while preserving its
outputs and diagnostics. The script also preserves incomplete or missing
sampling records as incomplete; it never regenerates them. A later retry may
use this new revision as `--source` with another fresh output path, retaining
any newly recovered conditions.

The new retry tests cover input identity/hash checks, selective recomputation,
dataset relocation, preserved source files and unresolved conditions. Local
syntax checks passed; numerical tests and the actual seven-condition retry
are to be run in the server environment.
