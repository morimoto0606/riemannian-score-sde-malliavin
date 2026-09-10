# S² earth-data postprocessing

Run from the repository root in the existing NumPy / OmegaConf / Matplotlib /
Cartopy environment. No JAX, training, or model checkpoint loading is required.

```bash
python -u scripts/postprocess_s2_earth_data.py --run-dir results/flood_malliavin_lambda5_seed0
python -u scripts/postprocess_s2_earth_data.py --run-dir results/flood_heat_seed0
python -u scripts/postprocess_s2_earth_data.py --run-dir results/flood_ism_seed0
```

The same entry point handles `earthquake_*` and `volcanoe_*` runs. The original
`scripts/postprocess_earthquake_upstream.py` is a compatibility wrapper and also
automatically selects the dataset. Existing CLI options remain available.

## Reference data and coordinates

The saved `.hydra/config.yaml` identifies dataset and objective. Dataset target,
name, experiment and recognizable run-directory prefix are cross-checked;
conflicting identities raise an error instead of evaluating against another
dataset. `volcano` (the existing dataset display name) maps to `volcanoe`.

| Dataset | CSV | Header rows | Observed plot title |
|---|---|---|---|
| earthquake | quakes_all.csv | 4 | Observed earthquakes |
| flood | flood.csv | 2 | Observed floods |
| volcanoe | volerup.csv | 2 | Observed volcanoes |

CSV definitions are shared with training in `riemannian_score_sde/earth_data.py`
and checked against `config/dataset/*.yaml`. The saved dataset's `data_dir` is
used, falling back to the saved global `data_dir`. Relative paths are based on
`work_dir`; an unresolved `${hydra:runtime.cwd}` uses the current repository root.
A missing reference file is an error. `--data-path` explicitly overrides the CSV
location for relocated data, retaining the selected dataset's header convention.

All three training datasets inherit the **same** `SphericalDataset` transform:
theta = latitude + pi/2, phi = longitude + pi. The postprocessor's shared NumPy
embedding is its mathematically equivalent antipodal form. Its original float64
operation order is retained to preserve Earthquake metrics. Training's angle
formula and CSV loading behavior are unchanged by the extraction.

## Outputs

- `{dataset}_real_map.png`, `{dataset}_generated_map.png`, `{dataset}_overlay_map.png`
- `scatter_comparison.png`, `density_comparison.png`
- `metrics.json`

Plots use the actual dataset and method. JSON includes `dataset`, `method`,
`teacher`, `experiment`, time-weight settings, `reference_data_path`,
`generated_count`, `real_count`, `s2_rbf_mmd`, the mean/median/max
`nearest_neighbor_geodesic_*`, `mmd_sigma`, `metric_subsample`, and
`evaluation_seed`. The coordinate label is `upstream-spherical-dataset-antipodal`.

Method/teacher labels are read from the saved loss and teacher config, including
`heat`, `spectrum`, `varadhan`, `ism`, and `malliavin_hutchinson`. ISM loss takes
priority over an inherited conditional teacher. `heat` retains its existing
code meaning (Varadhan + Spectrum switching); lambda is a separate field.

MMD and nearest-neighbor algorithms, subsampling, normalization, RNG order, and
default evaluation parameters are unchanged. Evaluation compares generated data
to the full corresponding real CSV, with the existing metric subsampling; it
does not switch to a held-out split.

## Verification

Ten unit tests pass, covering dataset routing, metadata, coordinate agreement,
conflict rejection and plot filenames/titles. Plot tests mock rendering.
Using `earthquake_malliavin_rb_time_025_100k/generated_samples.npy`, old and new
implementations produce exactly identical reference arrays and metrics. Compared
with its saved JSON, MMD is identical (`0.00014263296052297036`); nearest-neighbor
differences are at most approximately 5e-15. Existing result files were not edited.

The requested `results/flood_malliavin_lambda5_seed0` is absent locally. Its
automatic routing and metadata were tested with a temporary fixture, not the
server's generated samples. Cartopy is absent locally, so actual globe rendering
has not been rerun. No training was performed.
