# Earthquake baseline time weighting comparison

Prepare nine fresh runs: Varadhan, ISM and Spectrum with time weighting
`exp(-5*t)`, each at seeds 0, 1 and 2 for 600000 updates.
The default is Earthquake only. The supplied PBS array covers indices 0–8.

Read the existing `results/earth_long_both_v1/earthquake_METHOD_seedN/input_config/config.yaml`
files, including Spectrum's config even if its earlier training failed.
Preserve architecture, splits, optimizer, schedule, teacher settings and `like_w`.
Change time weighting, output paths and fresh-training mode only, except that
Spectrum also requires `enable_x64=true` and `JAX_ENABLE_X64=true`.
Spectrum lambda=0 must also use x64 for a matched ablation.

From the repository root in the existing mims Python 3.9 environment:

```bash
export EARTH_L5_ROOT="$PWD/results/earthquake_baseline_lambda5_v1"
python scripts/earth_baseline_lambda5.py --output "$EARTH_L5_ROOT" --check-only
python scripts/earth_baseline_lambda5.py --output "$EARTH_L5_ROOT" &&
qsub -v EARTH_L5_ROOT="$EARTH_L5_ROOT" job/earth_baseline_lambda5.q
```

Preparation never trains. The PBS job requests ten CPUs in queue `hi`, matching
the existing mims training jobs; actual GPU use follows the existing environment
and cluster policy. Do not use the CPU generation job's environment for training.
Indices 0–2 are Varadhan, 3–5 ISM, and 6–8 Spectrum.

Choose a new output root if one already exists. Existing checkpoints are never
used as initial states. Exclusive `started.json` creation prevents duplicate
execution; any existing checkpoint or modified prepared config blocks execution.
Source configs and their SHA256 hashes are retained for provenance.

`process_exit.json` records process exit, not checkpoint validity. Check final
checkpoint step and finiteness separately before evaluation. Earlier runs have
stalled after finishing training, so inspect logs and checkpoints before
terminating any long-lived process. Do not resubmit an entire array to recover
a failed run without inspecting its output first.

The optional `--datasets` flag can explicitly prepare other datasets later;
the PBS array range would then need to match the printed manifest indices.
