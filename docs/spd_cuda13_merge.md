# SPD finance and CUDA13 branch integration

Target: `fix/cuda13-py312-jax0.7-compat`, starting at `9ad0889`.
Merged source: `feature/teacher-interface` at `a654d19`.

The integration keeps the CUDA13 branch's JAX API compatibility changes,
`score_sde/ode.py` adaptations, `.gitmodules`, and geomstats commit
`3e6df8b0332b09948f2eda567598bdc9ae3623c8`. The existing main/run implementation
and model/loss/sampler runtime files remain byte-identical to the target branch.

SPD common configuration, the late beta schedule overlay, tracked financial
snapshot, generation diagnostics, evaluation additions and bounded smoke script
come from the teacher-interface branch. The Volcano spelling migration and its
shared postprocessor are retained; duplicate legacy experiment filenames were
removed only after comparing their normalized contents with the canonical files.

Conflict resolution preserves `config/teacher/spectrum.yaml` from the CUDA13
branch (`n_max=5`), rather than adopting the other branch's `n_max=10`. This avoids
changing existing GPU Spectrum runs as a side effect of the SPD merge.

Validation: all config YAML parses, changed Python syntax passes, six financial
preprocessing tests pass, and the dataset SHA256 matches the CPU-validated
snapshot. No conflict markers or unmerged index entries remain. CPU tiny smoke
success recorded in `spd_finance_cpu_smoke_result.md` belongs to the earlier
teacher-interface checkout; it is not a GPU or merged-checkout execution result.

No server commands, training, generation, dependency installation or existing
checkpoint modification were performed for this integration. Existing local
uncommitted documentation in the original worktree was preserved and its latest
verification notes were copied into this merge.

After this branch is pushed, update the GPU server repository on the same branch:

```bash
git pull --ff-only origin fix/cuda13-py312-jax0.7-compat
git submodule sync --recursive
git submodule update --init --recursive
```

The submodule update is required to use the pinned CUDA13-compatible geomstats
fork. This branch targets the existing modern GPU environment; the older MIMS
JAX 0.3.15 environment should continue using its teacher-interface branch.
