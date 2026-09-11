# S² real-data experiments: teacher / loss audit

## Scope and experiment names

| Dataset | Pure Varadhan | Varadhan + Spectrum | ISM |
|---|---|---|---|
| Earthquake | `earthquake_varadhan` | `earthquake_heat` | `earthquake_ism` |
| Flood | `flood_varadhan` | `flood_heat` | `flood_ism` |
| Volcano | `volcano_varadhan` | `volcano_heat` | `volcano_ism` |

`volcano` is the existing dataset key. All three datasets inherit
`earth_data`: S², identity transform, Brownian flow, RSGM, concat architecture,
no embedding, and rotational `DivFreeGenerator`. Dataset constructors read
`quakes_all.csv`, `flood.csv`, and `volerup.csv`, respectively, and convert angles
to three-dimensional unit vectors. Within each method, the three experiment
YAMLs differ only in dataset, identifiers, comments, and output paths.

## Pure Varadhan: explicit teacher and actual target

The former `teacher: varadhan` scalar was incompatible with the training call
`instantiate(cfg.teacher)` in `run.py`. The three `*_varadhan` YAMLs now select
`/teacher: varadhan`, which loads `config/teacher/varadhan.yaml` and instantiates
`riemannian_score_sde.teachers.VaradhanTeacher`. Existing loss/hyperparameters
are retained.

Path: `main.py` → `run.run` → `train` → instantiate teacher and `cfg.loss`
→ `get_dsm_loss_fn` (`s_zero=True`) → `VaradhanTeacher.sample_and_score`
→ forward `sde.marginal_sample` → `VaradhanTeacher.score_at_endpoint`
→ `Langevin.varhadan_exp(y_0, y_t, 0, t)`.

The target is `Log_{y_t}(y_0) / (tau(t) - tau(0))`, with
`tau(t) = beta_schedule.rescale_t(t)` and `tau(0)=0` here. No spectral method is
called in this target path. `n_max=-1` also selects Varadhan in the legacy
teacher-less DSM fallback; it is no longer needed to select the explicit teacher.
The historical `varhadan_exp` spelling is unchanged.

## Heat is a switching approximation, not Pure Spectrum

Both the original unsuffixed experiments (`earthquake`, `flood`, `volcano`)
and the explicit `*_heat` experiments use `dsm0`. Unsuffixed experiments have
no teacher config, so `get_dsm_loss_fn` constructs its default `HeatTeacher`.
Explicit heat experiments instantiate the same teacher from `teacher/heat.yaml`.
Its `n_max` and `thresh` interpolate `loss.n_max=5`, `loss.thresh=0.5`.

Path: `HeatTeacher.score_at_endpoint` → `Brownian.grad_marginal_log_prob`
→ rescale physical time to `tau` → vendored
`geomstats/geomstats/geometry/base.py:EmbeddedManifold.grad_marginal_log_prob`:

- `tau <= 0.5`: `metric.log(y_0, y_t) / tau`, the Varadhan score.
- `tau > 0.5`: tangent projection of the autodifferentiated truncated spectral
  log kernel in `Hypersphere._log_heat_kernel`.

For S² the kernel sums degrees `n=0,...,5`, with coefficients
`(2n+1)/(4*pi) * exp(-n*(n+1)*tau/2)` multiplying Legendre polynomials.
The factor `1/2` matches the Brownian random walk convention. With the default
linear schedule, `tau(t)=0.001*t+2.4995*t²`; switching occurs near `t=0.447`.
The implementation uses `where`, so both branches are computed before selection.
The log-kernel approximation includes a volume term, but the selected small-time
**score** is explicitly the pure log-map expression above.

`loss/dsm0.yaml` uses `(n_max=5, thresh=0.5)`; `loss/dsmv.yaml` uses
`(n_max=-1, thresh=1)` and the Varadhan fallback. There is no `loss/dsm.yaml`.

Recommended result-table label: **Varadhan + Spectrum (n_max=5, tau threshold=0.5)**.
`Heat` remains the compatibility code/config name. Do not label these results
Pure Spectrum or an exact heat-kernel score.

Pure Spectrum is now available as `earthquake_spectrum`, `flood_spectrum`,
and `volcano_spectrum`. These select `teacher/spectrum.yaml` and the dedicated
`SpectrumTeacher`, which directly differentiates `Hypersphere._log_heat_kernel`
and projects to the tangent space at every sampled time. It never calls the
switching wrapper or Varadhan. Existing Heat experiments are unchanged.

The default truncation is `teacher.n_max=4096`, not the switching baseline's 5.
At the default `eps=1e-3`, the smallest diffusion time is approximately 3.5e-6;
the last mode's exponential factor is approximately exp(-29.37). This motivates
more modes but does not establish relative accuracy for every endpoint.
The spectrum experiments set `enable_x64=true`; `main.py` enables JAX x64 for
this explicit teacher before importing the training code. Existing S2 teacher
behavior is unchanged. The teacher rejects execution without x64.

Recommended label: **Pure Spectrum (truncated, n_max=4096, float64)**.
There is no clipping of the kernel and no asymptotic fallback. Nonpositive sums
remain nonfinite, exposing truncation/cancellation problems. Small-time distant
endpoints can still suffer cancellation; reducing eps or changing the diffusion
schedule requires renewed convergence checks. This baseline is more expensive
than the 5-mode switching teacher. It is implemented but not numerically certified
locally. Run the server tests before production:

```bash
JAX_ENABLE_X64=true python -m pytest -q tests/test_spectrum_teacher.py
for DATASET in earthquake flood volcano; do
  python main.py --cfg job --resolve experiment="${DATASET}_spectrum"
  python -u main.py experiment="${DATASET}_spectrum" mode=train seed=0 \
    steps=1 batch_size=2 eval_batch_size=2 train_val=false train_plot=false \
    logger=csv "hydra.run.dir=results/smoke_${DATASET}_spectrum_seed0"
done
```

Sequential production commands (after finite-loss smoke/convergence checks):

```bash
set -e
for DATASET in earthquake flood volcano; do
  for SEED in 0 1 2; do
    JAX_ENABLE_X64=true python -u main.py \
      experiment="${DATASET}_spectrum" mode=train seed="${SEED}" \
      steps=100000 batch_size=128 eval_batch_size=128 val_freq=10000 logger=csv \
      "generated_samples_path=\${work_dir}/results/${DATASET}_spectrum_seed${SEED}/generated_samples.npy" \
      "hydra.run.dir=results/${DATASET}_spectrum_seed${SEED}"
  done
 done
```

For a precision-matched comparison, also run the other methods with
`JAX_ENABLE_X64=true` (the later legacy comparison loop uses float32).
Postprocessing recognizes `SpectrumTeacher` as `teacher: spectrum`.

## ISM and sphere divergence

All three `*_ism` experiments select `override /loss: ism`, replacing the model's
default DSM loss. They have no conditional teacher. `run.train` instantiates
`get_ism_loss_fn`, samples a forward endpoint, evaluates the score, then uses
`get_riemannian_div_fn` and `manifold.metric.squared_norm`:

`mean(beta(t) * (0.5 * ||score||² + div_M(score)))`.

The default `like_w=true` applies beta weighting; `time_weighting=false` applies
no extra exponential weighting. YAML `hutchinson_type: None` is the string
`"None"`, selecting exact Jacobian trace, not a stochastic estimator.

The implementation is shared across manifolds, not S²-specific. On the sphere,
the divergence helper uses the ambient Jacobian trace (`sqrt_g=1`). This is
consistent here because the score is a combination of rotational tangent
generators, followed by `Hypersphere.to_tangent`, whose projection uses
`I - xxᵀ/(xᵀx)`. The extension is tangent to every concentric sphere, so the
radial contribution `xᵀ Dscore x` vanishes on the unit sphere and the ambient
trace equals the surface divergence. An arbitrary ambient vector field would
not satisfy this argument. No general claim about arbitrary manifold adapters
is implied.

Static tracing finds compatible dataset/model/loss signatures up to the training
step. Numerical execution remains unverified locally.

## Seeds, common settings, and unchanged experiments

These datasets are fixed CSV observations; there is no synthetic target seed.
However, `cfg.seed` controls **both the random train/val/test split and training**:
`run.py` passes its derived key to `random_split`, which permutes dataset indices,
as well as to loaders/model initialization/training. Thus seeds 0,1,2 do not
mean model-only randomness on a fixed split. Equal seed and settings match the
split across methods. A separately fixed split seed would be an additional
behavior change and is not introduced here.

Legacy defaults differ across methods: Varadhan has 100K steps / batch 128,
Heat and ISM inherit 600K / batch 512. The commands below explicitly align
steps, batches, and validation cadence, while preserving method-specific loss
weighting. `enable_x64`/`dtype` in legacy Varadhan YAML do not activate JAX x64
for S² in `main.py`; use the same `JAX_ENABLE_X64` environment for all methods.
The commands below explicitly select float32 for all runs.

Unsuffixed experiments, Heat, ISM, and all Malliavin YAMLs are unchanged by this
audit. In particular, `*_malliavin_hutchinson` remains available. A lambda-5 run
must explicitly set `loss.time_weighting=true loss.time_weight_lambda=5.0`;
lambda alone is not sufficient if weighting is disabled.

## Server configuration validation (no JAX training)

Run from the repository root in the existing server environment:

```bash
set -e
for DATASET in earthquake flood volcano; do
  for METHOD in varadhan heat ism; do
    python main.py --cfg job --resolve \
      experiment="${DATASET}_${METHOD}" mode=train \
      steps=100000 batch_size=128 eval_batch_size=128 logger=csv
  done
done
```

`--cfg job --resolve` performs Hydra composition without entering `main`'s
training body. It does not test target instantiation or numerical loss execution.

Optional one-update smoke tests (nine separate processes; no evaluation/plots):

```bash
set -e
for DATASET in earthquake flood volcano; do
  for METHOD in varadhan heat ism; do
    JAX_ENABLE_X64=false python -u main.py \
      experiment="${DATASET}_${METHOD}" mode=train seed=0 steps=1 \
      batch_size=2 eval_batch_size=2 logger=csv \
      train_val=false train_plot=false val_freq=100000 \
      "hydra.run.dir=results/smoke_${DATASET}_${METHOD}_seed0"
  done
done
```

## Sequential 100K training: all nine experiments × three seeds

This directly runs each dataset/method/seed combination sequentially. It does
not submit jobs or start runs unless executed on the server.

```bash
set -e
for DATASET in earthquake flood volcano; do
  for METHOD in varadhan heat ism; do
    for SEED in 0 1 2; do
      JAX_ENABLE_X64=false python -u main.py \
        experiment="${DATASET}_${METHOD}" \
        mode=train seed="${SEED}" steps=100000 \
        batch_size=128 eval_batch_size=128 \
        val_freq=10000 logger=csv \
        "generated_samples_path=\${work_dir}/results/${DATASET}_${METHOD}_seed${SEED}/generated_samples.npy" \
        "hydra.run.dir=results/${DATASET}_${METHOD}_seed${SEED}"
    done
  done
done
```

`mode=train` trains and saves checkpoints; it does not perform final generation.
The per-seed sample path also prevents validation plots from overwriting samples
in another seed's directory. Use fresh run directories for independent runs.

## Local verification and remaining limits

Local checks: YAML parsing, teacher target/config references, equality of
method settings across datasets, Python syntax, and unchanged-config hashes.
Hydra is not installed in the available local Python environment; the existing
JAX environment cannot load its AVX-requiring jaxlib on this machine. Therefore
Hydra composition, target instantiation, and one-update numerical tests are
**not claimed as passed**. No environment installation, training, or evaluation
was performed. Run the server validation and smoke commands before the 100K loop.
