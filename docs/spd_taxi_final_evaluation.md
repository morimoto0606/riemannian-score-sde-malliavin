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
