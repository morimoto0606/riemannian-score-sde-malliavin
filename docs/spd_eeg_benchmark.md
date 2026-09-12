# EEG benchmark for the SPD comparison

This is a controlled adaptation of BNCI2014-002, NOT an exact reproduction of
Ko–Lee. Their SPD experiments are conditional NYC taxi SPD(10) and motor-imagery
EEG (BNCI2014-002 / BNCI2015-001). Their referenced code repository was empty
when checked on 2026-09-13. DiffeoCFM's public EEG pipeline supplies the upstream
reference for MOABB loading and per-trial OAS covariance estimation.

Our first benchmark uses 15 EEG channels, binary one-hot class conditioning,
and deterministic subject-disjoint train/validation/test splits. It uses all
runs, no outlier filtering, no correlation normalization. Test and validation
subjects are each ceil(20%) of subjects, using RandomState(0) permutation of
sorted subject strings. Label names and exact subject assignments are saved.
DiffeoCFM uses run/session-based grouping, additional filtering, and a different
model/evaluation protocol; our results must not be placed beside their published
numbers as direct reproductions. The generator architecture, noise schedule and
empirical forward terminal are also our controlled comparison choices.

## Data preparation

No environments or dependencies are installed by these commands. If an existing
EEG environment has MOABB, MNE, NumPy and scikit-learn, run from repository root:

```bash
python scripts/build_spd_eeg_dataset.py --download
```

Otherwise export an NPZ with `epochs` (trials,15,time), `labels` (binary strings or
integers) and `subjects` (one ID per trial) from an existing EEG environment:

```bash
python scripts/build_spd_eeg_dataset.py --epochs-npz /path/to/epochs.npz
```

The output is data/spd_eeg/bnci2014_002.npz and adjacent metadata. The builder
refuses overwrite, checks finite input, uses per-trial OAS without fitting on
other trials, and validates SPD and both labels in every split. Original EEG
units are preserved; the MOABB version or input hash is recorded. Dataset not
yet downloaded or committed. No synthetic fixture should substitute for EEG
in a reported experiment.

## Server validation

Activate the existing CUDA environment and set GEOMSTATS_BACKEND=jax,
JAX_ENABLE_X64=true, PYTHONPATH="$PWD/geomstats:$PWD" as before.

```bash
python -m unittest discover -s tests -p test_spd_eeg_preprocessing.py -v
python -m unittest discover -s tests -p test_spd_rejection.py -v
python scripts/smoke_spd_eeg.py --timeout 600
```

Smoke composes three configs, trains one tiny update each, restores each
checkpoint for labels 0 and 1, generates eight valid matrices per label and
checks checkpoint hashes unchanged. It creates a fresh results directory.
Logs are retained there; no production training is started.

## Experiment grid after smoke

Configs: spd_eeg_varadhan, spd_eeg_ism, spd_eeg_malliavin_hutchinson.
Default Malliavin time weighting is OFF (lambda0). Compare that to a separate
run with loss.time_weighting=true loss.time_weight_lambda=5.0. Use seeds 0,1,2,
same model, schedule, training budget and reverse steps. Generate each class
separately with generation.class_label=0 or 1 and report metrics by class.

Example (long training, do not execute until smoke succeeds):

```bash
out=$(mktemp -d "$PWD/results/spd_eeg_malliavin_lambda0_seed0_XXXXXX")
python -u main.py experiment=spd_eeg_malliavin_hutchinson mode=train \
  seed=0 steps=100000 logger=csv generation.enabled=false \
  loss.time_weighting=false loss.time_weight_lambda=0.0 \
  "ckpt_dir=$out/ckpt" "hydra.run.dir=$out"
gen=$(mktemp -d "$PWD/results/spd_eeg_class0_XXXXXX")
python -u main.py experiment=spd_eeg_malliavin_hutchinson mode=test \
  seed=0 logger=csv generation.enabled=true generation.class_label=0 \
  generation.count=1000 generation.steps=64 generation.max_attempts=2000 \
  "ckpt_dir=$out/ckpt" "hydra.run.dir=$gen" \
  "generated_samples_path=$gen/generated_samples.npy"
```

The initial terminal distribution draws only training matrices of the requested
class. Training sees one-hot labels together with time. Samples are separately
validated with the same bounded rejection policy across methods. Metadata
records class_label and class-conditional terminal law.

Financial controls should include lambda0 as well as the existing lambda5
checkpoint. Compare quality, rejection rate and runtime across seeds; do not
infer a general SPD weakness from a single new dataset. Official Ko–Lee CAS and
alpha/beta quality evaluations have not been implemented here; current support
is for controlled class-wise covariance-distribution comparisons.

## Validation performed locally

Two NumPy/scikit-learn preprocessing tests passed, including subject disjointness,
SPD, deterministic split and nonfinite rejection. Six sample rejection tests
passed. Changed Python files parse. Hydra and JAX runtime not available in the
local verification environment; composition, conditional training and generation
remain explicitly unverified until the server smoke passes. No new dependencies,
GPU jobs, checkpoint changes or long training were performed.

Sources:
- https://arxiv.org/html/2605.31106v1 (sections 6.4–6.5, B.3–B.4)
- https://github.com/kogyeonghoon/riem-diff-pinn
- https://github.com/antoinecollas/DiffeoCFM (data.py, cov_est.py, train.py)

Channel correction: BNCI2014-002 has 15 EEG electrodes (MOABB dataset documentation), not six. All EEG configs use SPD(15), intrinsic dimension 120. The earlier six-channel assumption was an implementation error. Source: https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2014_002.html
