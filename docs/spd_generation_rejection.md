# SPD generation with recorded rejection

All SPD finance methods use the same per-matrix validity checks. Nonfinite,
asymmetric, non-positive-definite, and numerically unreportable matrices are
rejected individually; valid members of the same batch are retained. No matrix
repair or clipping is performed. Sampling continues with fresh random keys until
the requested count is obtained, subject to generation.max_attempts (null means
twice generation.count). Sampler execution and shape errors remain fatal.

The output metadata's sampling section records requested, attempted, accepted,
rejected, rejection_rate, rejection_reasons, max_attempts, limit_reached and
complete. spd.non_spd_count describes ONLY retained matrices; it is not the
failure rate. Outputs represent a distribution conditional on passing validation.
Compare quality together with rejection rates, using the same sampler settings
for all methods. Rejection does not resolve unstable reverse trajectories.

When the cap is reached, metadata records the partial accepted summary and
complete=false, but no sample NPY is published; the process exits with an error.
Existing sample or metadata paths are refused to prevent stale/mixed artifacts.
Use a new output directory for every attempt. Checkpoints are read only in test
mode. Existing training does not need to be repeated.

Example after activating the server environment, from repository root:

```bash
root="$PWD/results/spd_finance_production_kFTQJM"
out=$(mktemp -d "$root/malliavin_rejection_XXXXXX")
python -u main.py experiment=spd_finance_malliavin_hutchinson \
  mode=test seed=0 logger=csv \
  loss.time_weighting=true loss.time_weight_lambda=5.0 \
  generation.enabled=true generation.count=1000 generation.batch_size=16 \
  generation.steps=64 generation.seed=123 generation.max_attempts=2000 \
  "ckpt_dir=$root/spd_finance_malliavin_lambda5_seed0/ckpt" \
  "hydra.run.dir=$out" "generated_samples_path=$out/generated_samples.npy"
```

Validation: six NumPy unit tests cover mixed batches/refill, attempt limits,
unchanged valid batches, invalid classes and propagation of shape/device errors.
Python syntax checked. GPU generation remains to be verified on the server.
