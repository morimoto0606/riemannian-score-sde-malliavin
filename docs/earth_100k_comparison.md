# Fresh 100k S2 comparison

Mims branch: feature/teacher-interface. Preserve all previous results/checkpoints.
72 runs: earthquake/flood/volcano × Malliavin/Varadhan/ISM/Spectrum × lambda 0/5 × seeds 0/1/2.
Source: results/earth_long_both_v1/*/input_config/config.yaml. Malliavin uses the lambda0 source for both new weights.
Fresh initialization; 1000 warmup + 99000 cosine decay. Native like_w, model, batch, splits, teacher probes and other source settings are retained. Spectrum enables x64; this is an explicit precision difference across methods. No generation during training. Old 600k/300k checkpoints are never resumed or edited.

## One-update Spectrum checks (PBS)

```bash
cd ~/riemannian-score-sde-malliavin
export EARTH_100K_ROOT="$PWD/results/earth_100k_smoke_v1"
python scripts/earth_100k_comparison.py --output "$EARTH_100K_ROOT" --smoke
qsub -v EARTH_100K_ROOT="$EARTH_100K_ROOT" job/earth_100k_smoke.q
```

Indices 18,21,42,45,66,69 are Spectrum seed0 for both weights on each dataset.
These retain full batch/model settings and only truncate training to one update (without validation).
Before production, inspect launcher.log and process_exit.json and restore smoke checkpoints on CPU to verify step=1 and finite state. A zero process return code alone is not proof of a valid checkpoint.

## Production, after successful smoke checks

```bash
export EARTH_100K_ROOT="$PWD/results/earth_100k_both_v1"
python scripts/earth_100k_comparison.py --output "$EARTH_100K_ROOT" --check-only
python scripts/earth_100k_comparison.py --output "$EARTH_100K_ROOT"
qsub -v EARTH_100K_ROOT="$EARTH_100K_ROOT" job/earth_100k_comparison.q
```

Existing output roots are refused. Do not rerun a started index or delete its guard to overwrite checkpoints.
Dataset offsets are 0/24/48. Within each dataset: Malliavin0 0–2, Malliavin5 3–5, Varadhan0 6–8, Varadhan5 9–11, ISM0 12–14, ISM5 15–17, Spectrum0 18–20, Spectrum5 21–23.
PBS requests 10 CPU cores per job, following the existing mims job convention; the shared GPU is not exclusively reserved. Current node allocation previously allowed nine concurrent jobs. Timing depends on contention and is not a controlled method-speed benchmark.
After training, verify step=100000 and finite checkpoint state before generation/evaluation. The prior process-finalization stall has not been diagnosed or fixed here; monitor for jobs whose progress reaches the target but do not exit.

Local validation: dependency-free configuration and launcher-guard unit tests, Python syntax, bash syntax. No local JAX training or dependency installation. Server runtime smoke remains required.
