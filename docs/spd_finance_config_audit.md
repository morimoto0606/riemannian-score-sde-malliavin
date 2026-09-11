# SPD finance configuration audit

最新状況：サーバーでconfigテスト2件とCPU tiny smoke（3手法、各1 update・8件生成）が成功。
詳細は [CPU smoke検証結果](spd_finance_cpu_smoke_result.md) を参照。
以下の未実行・障害記述は診断時点の記録であり、GPUと本番shapeの検証は引き続き未完了。

## Status

This change has **not** been executed with Hydra or JAX locally. The installed
JAX binary cannot run on this Mac (AVX error); Hydra is absent. The user chose
server verification without adding an environment. Do not interpret static
inspection as a successful optimizer update or successful generation.

Locally checked: changed Python syntax, diff whitespace, the actual NPZ shape,
SPD eigenvalues/symmetry, persisted split counts and NumPy financial features.
All config YAML files parse successfully; six preprocessing unittests pass.
All 33 pre-existing non-SPD experiment files retain their original hashes.
No data, results or existing checkpoints were modified. No training or generation
was run. Server commands below perform the outstanding verification.

## Defaults tree and cause

`main.yaml`, not `model/rsgm.yaml`, selects the global beta schedule.
Its existing order is experiment, logger, optimizer, scheduler, **then linear**.
`model/rsgm` selects transform, flow, base, pushforward, loss and generator.

- `override beta_schedule` inside an experiment addresses the relative group
  `experiment/beta_schedule`, which does not exist in the defaults list.
- `/beta_schedule: spd_finance` adds another GROUP_DEFAULT, conflicting with
  the root's `beta_schedule: linear` selection.
- An absolute `override /beta_schedule` in the earlier experiment is not a
  reliable override of the later root selection; it is the wrong position in
  this tree. Moving linear ahead of experiment would make previously overwritten
  inline beta values in existing SO3/hyperbolic experiments effective.

To preserve existing experiment behavior, the root ordering is unchanged and a
late optional `experiment_schedule: ${experiment}` is added. Only the three SPD
method names have matching files. Each imports the CONFIG entry
`/beta_schedule/spd_finance@_global_.beta_schedule`. This merges its values after
linear without making a second GROUP_DEFAULT selection or using an override in
an interpolated subtree (which Hydra 1.1 prohibits).

Expected tree, to be confirmed by `--info defaults-tree` on the server:

```text
main
  _self_
  server/local
  experiment/spd_finance_<method>
    experiment/spd_finance
      dataset/spd_finance
      manifold/spd5
      architecture/concat
      embedding/none
      model/rsgm
        transform/id, flow/brownian, base/default, pushf/sde
        loss/dsm0 (ISM selects loss/ism), generator/div_free
      _self_  [SPDBrownian, SPDGenerator, x64, batch and generation settings]
    teacher/varadhan OR teacher/spd_malliavin_hutchinson (ISM has teacher=null)
    _self_
  logger/csv
  optim/adam
  scheduler/rcosine
  beta_schedule/linear
  experiment_schedule/spd_finance_<method>
    beta_schedule/spd_finance @ beta_schedule
```

Hydra's selected group choice is still `beta_schedule=linear`; the final object
is the SPD LinearBetaSchedule with beta_0=.01, beta_f=1.0, t0=0, tf=1.
This distinction is intentional. Inspect the final config, not only runtime
group-choice metadata. To tune SPD beta values use scalar overrides such as
`beta_schedule.beta_f=0.5`; switching the group alone does not remove the overlay.

The shared `experiment/spd_finance.yaml` is an inheritance base; select one of
the three public method experiments when running. Existing S2/SO3 config files
are unchanged; regression tests check that they receive no SPD schedule overlay.

Hydra 1.1 reference: [Defaults List, ordering, CONFIG vs GROUP_DEFAULT and
interpolation restrictions](https://hydra.cc/docs/1.1/advanced/defaults_list/).

## Mathematical implementation audit

**Varadhan:** `VaradhanTeacher.sample_and_score` samples SPDBrownian, then
`score_at_endpoint` calls `Langevin.varhadan_exp`: `Log_Y(X)/tau(t)`.
`AffineSPD` delegates to the AIRM log through `JitSPDMetricAffine`; the target
is a symmetric 5x5 tangent matrix. It is a short-time heat-kernel approximation,
not an exact finite-step GRW density score.

**ISM:** `get_ism_loss_fn` samples the forward process and uses the AIRM squared
norm plus `get_riemannian_div_fn` -> `get_spd_div_fn` -> `riemannian_divergence`.
In 15 Frobenius-orthonormal symmetric coordinates:

```text
sqrt(det g) = C det(X)^(-3)                   [SPD(5)]
div_g S = trace(D_svec(X) svec(S)) - 3 trace(X^-1 S)
loss = beta(t) [0.5 ||S||_g^2 + div_g S]
```

The trace uses Rademacher probes in 15 coordinates (or exact trace when selected).
The volume term is already present. This is not uncorrected Euclidean ambient
divergence. Integration-by-parts also requires the usual integrability/boundary
decay conditions; a smoke test cannot establish those globally for a trained net.

**Malliavin:** `SPDMalliavinTeacher` uses Cholesky frame `V_a=L E_a L^T`,
the noise-space endpoint derivative of the same discrete GRW, its 15-dimensional
frame Jacobian J, covariance `Gamma=J J^T`, and covering field
`U=J^T(Gamma+ridge I)^-1`. It computes `delta(U)=Z^T U-div_Z U` and frame
coefficients `-delta(U_a)-div_g V_a`, then reconstructs a symmetric tangent matrix.
The diagonal frame divergences are `(2,1,0,-1,-2)`; off-diagonal ones vanish.
The existing Hutchinson correction is used, not omitted. Relative ridge 1e-8
introduces bias; exact IBP requires zero ridge and full rank. Finite GRW steps,
finite probes and reverse discretization remain approximations. SPD Rao–Blackwell
smoothing is not implemented and is explicitly rejected. No new geometry or
loss implementation was needed for this configuration change.

**Dataset:** actual local NPZ has shape `(1801,5,5)`, 60-return covariance windows,
chronological purged splits `1260/210/211`, symmetry error 0, minimum eigenvalue
`9.388683657654598e-7`, non-SPD count 0, recorded regularized count 0 and eps_max 0.
The loader ignores the training RNG and uses persisted indices; dataset_seed=0
remains independent. Its Hydra instantiation is included in the server tests.

**Evaluation:** `scripts/evaluate_spd_finance_generation.py` already implements
AIRM distance, both NN directions, geodesic RBF discrepancy, eigenvalue plots,
determinant, condition number, asset variances and pairwise correlations. JSON
now also includes each ordered eigenvalue's summary and mean covariance and
correlation matrices, even with `--no-plots`. AIRM geodesic Gaussian kernels are
not guaranteed positive definite for every bandwidth; the reported MMD-form
quantity is a kernel discrepancy, not an unqualified RKHS distance.

## Server verification (bounded; no environment installation)

Activate the existing server environment and run from the repo root:

```bash
export GEOMSTATS_BACKEND=jax JAX_ENABLE_X64=true
export PYTHONPATH="$PWD/geomstats:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -p test_spd_finance_configs.py -v
python -m unittest discover -s tests -p test_spd_training.py -v
python scripts/smoke_spd_finance.py --timeout 600
```

The smoke script captures actual defaults trees, defaults lists and composed
configs, runs the three requested production-shaped one-update training jobs
with generation disabled, then restores each checkpoint into a separate run
and generates 8 samples with 4 reverse steps. It checks checkpoint_step=1,
SPD validity, unchanged checkpoint hashes, and evaluates at most 8 samples per
side without plots. Each subprocess has a time limit. No output root is reused.
`summary.json` records symmetry error, minimum eigenvalue and non_spd_count=0;
invalid samples fail instead of being repaired. A timeout is a failure, not a pass.

If production-shaped compilation cannot finish in this bound, `--tiny` uses
batch 2, forward N=1 and hidden=[16,16]. That checks a smaller numerical path and
must not be reported as a pass for production defaults. The geometry unittest
also checks finite loss/gradients, nonzero optimizer updates, frame divergence,
forward endpoint equivalence and the exact SPD(1) Malliavin sign.

`--cfg job` deliberately does not use `--resolve`: the repository registers the
legacy `eval` scheduler resolver during training imports, after Hydra's config
display path. Actual training resolves it before optimizer instantiation.

Outstanding results: all three optimizer smoke tests, restored generation,
generated SPD constraints and numerical geometry tests are **pending server
execution**, not passed locally. Long training and quality/convergence evaluation
are outside this change.

## Changed files

- `config/main.yaml`: late optional SPD schedule selection.
- `config/beta_schedule/spd_finance.yaml`: clarify overlay semantics (values unchanged).
- `config/experiment/spd_finance.yaml`: new common base, explicit float64, generation opt-in.
- `config/experiment/spd_finance_varadhan.yaml`, `spd_finance_ism.yaml`,
  `spd_finance_malliavin_hutchinson.yaml`: common inheritance and objective selection.
- `config/experiment_schedule/spd_finance_varadhan.yaml`, `spd_finance_ism.yaml`,
  `spd_finance_malliavin_hutchinson.yaml`: late schedule overlays.
- `riemannian_score_sde/spd_generation.py`: explicit non_spd_count in successful diagnostics.
- `scripts/evaluate_spd_finance_generation.py`: JSON covariance/correlation/eigenvalue summaries.
- `scripts/smoke_spd_finance.py`: bounded server end-to-end verification.
- `tests/test_spd_finance_configs.py`: defaults-list and non-SPD regression tests.
- `tests/test_spd_training.py`: actual Hydra dataset instantiation and expected split checks.
- `docs/spd_finance.md`, `docs/spd_finance_config_audit.md`: updated operation and audit report.

## Production 100K, seeds 0/1/2 (commands only)

Run only after server checks pass. This uses standard model/forward settings,
Malliavin lambda 5, and separate immutable run directories. Generation is an
explicit later `mode=test` call; training alone does not generate samples.

```bash
set -e
ROOT="$PWD"
for METHOD in varadhan ism malliavin_hutchinson; do
  EXTRA=()
  TAG="spd_finance_${METHOD}"
  if [ "$METHOD" = malliavin_hutchinson ]; then
    EXTRA=(loss.time_weighting=true loss.time_weight_lambda=5.0)
    TAG=spd_finance_malliavin_lambda5
  fi
  for SEED in 0 1 2; do
    RUN="$ROOT/results/${TAG}_seed${SEED}"
    if [ -e "$RUN" ]; then printf 'Already exists: %s\n' "$RUN" >&2; exit 1; fi
    python -u main.py "experiment=spd_finance_${METHOD}" mode=train logger=csv \
      "seed=$SEED" steps=100000 generation.enabled=false "${EXTRA[@]}" \
      "ckpt_dir=$RUN/ckpt" "hydra.run.dir=$RUN"
  done
done
```

Example generation after training (adjust method, seed and original overrides):

```bash
RUN="$PWD/results/spd_finance_malliavin_lambda5_seed0"
OUT="${RUN}_generation_$(date +%Y%m%d_%H%M%S)"
test ! -e "$OUT" && python -u main.py \
  experiment=spd_finance_malliavin_hutchinson mode=test logger=csv seed=0 \
  loss.time_weighting=true loss.time_weight_lambda=5.0 \
  generation.enabled=true generation.count=256 \
  "ckpt_dir=$RUN/ckpt" "hydra.run.dir=$OUT" \
  "generated_samples_path=$OUT/generated_samples.npy"
```
