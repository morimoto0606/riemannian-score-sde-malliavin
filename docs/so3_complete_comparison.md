# SO3 completed comparison

Use feature/teacher-interface on mims. Existing results are read only.
Reuse 12 runs: Malliavin lambda0/5, Varadhan lambda0, ISM lambda0, seeds0/1/2.
Train six missing runs: Varadhan lambda5 and ISM lambda5, seeds0/1/2, 100000 steps.
SpectrumTeacher supports S2 only and is not included.

Training hparams are selected by mode=train, not latest version. Conflicting records fail closed. Existing target distribution settings must match, with dataset.seed=0. Fresh baselines retain each seed's source training settings and schedule except lambda and output paths. Source configs and prepared config hashes are recorded. This does not independently establish identical optimization schedules across historical methods.

```bash
cd ~/riemannian-score-sde-malliavin
source ~/venvs/riemannian-score-sde-py39/bin/activate
git pull --ff-only
export SO3_COMPARE_ROOT="$PWD/results/so3_complete_100k_v1"
python scripts/so3_complete_comparison.py prepare --output "$SO3_COMPARE_ROOT" --check-only
python scripts/so3_complete_comparison.py prepare --output "$SO3_COMPARE_ROOT"
JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python scripts/so3_complete_comparison.py audit --output "$SO3_COMPARE_ROOT"
```

Only after audit succeeds:

```bash
qsub -v SO3_COMPARE_ROOT="$SO3_COMPARE_ROOT" job/so3_complete_train.q
```

New training array0–2 is Varadhan lambda5,3–5 ISM lambda5. No existing checkpoint is resumed or overwritten. Each successful training job verifies checkpoint step and finiteness before writing COMPLETE. Confirm six COMPLETE markers before evaluating:

```bash
find "$SO3_COMPARE_ROOT/training" -name COMPLETE | wc -l
qsub -v SO3_COMPARE_ROOT="$SO3_COMPARE_ROOT" job/so3_complete_evaluate.q
```

Evaluation array0–17 rechecks all checkpoints and uses CPU, common generation seed0, batch512 giving16384 rotations, EMA,100 GRW steps through current run.py. Each preserved checkpoint is hashed before/after generation. Separate output directories refuse repeated execution. Reference count16384, reference sample seed10000, target seed0, metric subset2000, metric seed0, RBF sigma1. The underlying postprocessor makes Euler distribution plots and computes bidirectional NN statistics. Training seed metadata in new JSON/CSV is corrected from training provenance; generation seed is separately recorded in JSON. No likelihood metrics are computed or mixed with historical90k/100k likelihood logs.

After18 COMPLETE markers:

```bash
python scripts/so3_complete_comparison.py summarize --output "$SO3_COMPARE_ROOT"
```

Outputs: comparison_per_seed.csv, comparison_summary.csv (MMD and both NN Mean/Median/Max), comparison.png, comparison.pdf. Error bars are sample SD across three training seeds, not confidence intervals. NN distance uses the SO3 matrix-log Frobenius convention, not the S2 angle convention; do not compare magnitudes directly with Earth data.

Local checks are dependency-free unit tests, Python syntax and PBS shell syntax; no environment installation or local training. Runtime configuration composition and numerical validation occur on mims. Historical source checkpoint provenance cannot be established from folder names alone; audit verifies finalstep/finite state, and source training configuration conflicts are rejected.
