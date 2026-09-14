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
