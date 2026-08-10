#!/usr/bin/env python3
"""Compare upstream Earthquake Heat and Malliavin generated samples fairly.

All coordinate conversion, validation, metrics, map construction, and spherical
KDE calculations are imported from ``postprocess_earthquake_upstream.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from postprocess_earthquake_upstream import (
    _add_map_axis,
    _plot_backend,
    _resolve_map_center,
    _scatter,
    latlon_to_upstream_s2,
    load_earthquake_latlon,
    nearest_neighbor_geodesic,
    s2_rbf_mmd,
    spherical_kde,
    upstream_s2_to_latlon,
    validate_s2_points,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data/quakes_all.csv"
DEFAULT_HEAT_RUN_DIR = REPOSITORY_ROOT / "results/earthquake_upstream_heat_baseline"
DEFAULT_MALLIAVIN_RUN_DIR = (
    REPOSITORY_ROOT / "results/earthquake_malliavin_p8_default_600k"
)

GRID_LAT = 180
GRID_LON = 360
KAPPA = 80.0
DENSITY_ALPHA = 0.62
MMD_SIGMA = 1.0
METRIC_SUBSAMPLE = 2000
SEED = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--heat-run-dir", type=Path, default=DEFAULT_HEAT_RUN_DIR)
    parser.add_argument(
        "--malliavin-run-dir", type=Path, default=DEFAULT_MALLIAVIN_RUN_DIR
    )
    parser.add_argument(
        "--heat-samples-path",
        type=Path,
        default=None,
        help=(
            "Explicit Heat samples path. By default the script checks the run "
            "directory and then its saved Hydra generated_samples_path."
        ),
    )
    parser.add_argument("--malliavin-samples-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --malliavin-run-dir.",
    )
    parser.add_argument("--central-lat", type=float, default=None)
    parser.add_argument("--central-lon", type=float, default=None)
    return parser.parse_args(argv)


def _saved_generated_samples_setting(run_dir: Path) -> str | None:
    """Read generated_samples_path from the run's immutable Hydra snapshot."""

    config_path = run_dir / ".hydra/config.yaml"
    if not config_path.is_file():
        return None
    match = re.search(
        r"^generated_samples_path:\s*(.+?)\s*$",
        config_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    return match.group(1).strip().strip("'\"")


def _localise_saved_path(value: str) -> Path:
    """Map a saved server path onto this checkout without guessing its run name."""

    expanded = value.replace("${work_dir}", str(REPOSITORY_ROOT))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if path.is_file():
        return path.resolve()

    # Pulled Hydra snapshots may contain the server's absolute repository path.
    # Preserve the path suffix beginning at results/, which is part of the
    # recorded configuration, and rebase only the repository prefix.
    parts = path.parts
    if "results" in parts:
        results_index = parts.index("results")
        rebased = REPOSITORY_ROOT.joinpath(*parts[results_index:])
        if rebased.is_file():
            return rebased.resolve()
    return path


def resolve_samples_path(
    run_dir: Path,
    explicit_path: Path | None,
    *,
    label: str,
) -> Path:
    """Resolve a real artifact from an explicit path, run dir, or Hydra config."""

    run_dir = run_dir.expanduser().resolve()
    candidates: list[tuple[str, Path]] = []
    if explicit_path is not None:
        explicit = explicit_path.expanduser()
        if not explicit.is_absolute():
            explicit = REPOSITORY_ROOT / explicit
        candidates.append(("explicit argument", explicit))
    candidates.append(("run directory", run_dir / "generated_samples.npy"))

    saved_setting = _saved_generated_samples_setting(run_dir)
    if saved_setting is not None:
        candidates.append(("saved Hydra config", _localise_saved_path(saved_setting)))

    checked = []
    for source, path in candidates:
        resolved = path.resolve()
        checked.append(f"  - {source}: {resolved}")
        if resolved.is_file():
            print(f"{label} samples: {resolved} ({source})")
            return resolved

    raise FileNotFoundError(
        f"Could not find {label} generated_samples.npy. Checked:\n"
        + "\n".join(checked)
        + f"\nThe checkpoint, if present, is {run_dir / 'ckpt'}. Generate samples "
        "with mode=test; do not retrain."
    )


def load_generated(path: Path, label: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    points = validate_s2_points(raw, f"{label} generated samples")
    return points, upstream_s2_to_latlon(points)


def save_scatter_comparison(
    real_latlon: np.ndarray,
    heat_latlon: np.ndarray,
    malliavin_latlon: np.ndarray,
    output_dir: Path,
    *,
    central_lat: float,
    central_lon: float,
) -> Path:
    plt, ccrs, cfeature = _plot_backend()
    fig = plt.figure(figsize=(18, 6), dpi=200)
    datasets = (
        ("Observed", real_latlon, "#b2182b"),
        ("Heat", heat_latlon, "#2166ac"),
        ("Malliavin", malliavin_latlon, "#1b7837"),
    )
    for column, (title, latlon, color) in enumerate(datasets, start=1):
        # Reuse the plain Malliavin postprocess globe background (white ocean,
        # light-grey land and coastlines), rather than the old Heat stock image.
        ax, transform = _add_map_axis(
            fig,
            130 + column,
            title,
            ccrs,
            cfeature,
            central_lat=central_lat,
            central_lon=central_lon,
        )
        _scatter(
            ax,
            latlon,
            color=color,
            label=title,
            transform=transform,
            alpha=0.45,
        )
    fig.suptitle("Earthquake scatter: Observed / Heat / Malliavin")
    fig.tight_layout()
    path = output_dir / "scatter_observed_heat_malliavin.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")
    return path


def save_density_comparison(
    real_points: np.ndarray,
    heat_points: np.ndarray,
    malliavin_points: np.ndarray,
    output_dir: Path,
    *,
    central_lat: float,
    central_lon: float,
) -> Path:
    latitudes = np.linspace(-90.0, 90.0, GRID_LAT)
    longitudes = np.linspace(-180.0, 180.0, GRID_LON)
    densities = [
        spherical_kde(
            points,
            latitudes,
            longitudes,
            kappa=KAPPA,
        )
        for points in (real_points, heat_points, malliavin_points)
    ]

    # One physical KDE scale and one set of levels for all three panels.
    # No panel receives an individual normalization.
    common_minimum = 0.0
    common_maximum = max(float(density.max()) for density in densities)
    if not np.isfinite(common_maximum) or common_maximum <= common_minimum:
        raise FloatingPointError("shared density grid is invalid or identically zero")
    levels = np.linspace(common_minimum, common_maximum, 41)

    plt, ccrs, cfeature = _plot_backend()
    fig = plt.figure(figsize=(18, 6), dpi=200)
    contour = None
    for column, (title, density) in enumerate(
        zip(("Observed", "Heat", "Malliavin"), densities), start=1
    ):
        ax, transform = _add_map_axis(
            fig,
            130 + column,
            title,
            ccrs,
            cfeature,
            central_lat=central_lat,
            central_lon=central_lon,
        )
        contour = ax.contourf(
            longitudes,
            latitudes,
            density,
            levels=levels,
            vmin=common_minimum,
            vmax=common_maximum,
            cmap="magma",
            alpha=DENSITY_ALPHA,
            extend="max",
            transform=transform,
            zorder=3,
        )
    fig.colorbar(
        contour,
        ax=fig.axes,
        shrink=0.82,
        label=f"Spherical KDE (kappa={KAPPA:g}; shared scale)",
    )
    fig.suptitle("Earthquake density: Observed / Heat / Malliavin")
    path = output_dir / "density_observed_heat_malliavin.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")
    return path


def compute_metrics(
    label: str, generated_points: np.ndarray, real_points: np.ndarray
) -> dict[str, object]:
    geodesic = nearest_neighbor_geodesic(
        generated_points,
        real_points,
        n_sub=METRIC_SUBSAMPLE,
        seed=SEED,
    )
    return {
        "model": label,
        "generated_count": int(generated_points.shape[0]),
        "s2_rbf_mmd": s2_rbf_mmd(
            generated_points,
            real_points,
            sigma=MMD_SIGMA,
            n_sub=METRIC_SUBSAMPLE,
            seed=SEED,
        ),
        "nearest_neighbor_geodesic_mean": geodesic["mean"],
        "nearest_neighbor_geodesic_median": geodesic["median"],
        "nearest_neighbor_geodesic_max": geodesic["max"],
    }


def save_metric_comparison(
    real_points: np.ndarray,
    heat_points: np.ndarray,
    malliavin_points: np.ndarray,
    output_dir: Path,
    *,
    heat_samples_path: Path,
    malliavin_samples_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    heat = compute_metrics("Heat", heat_points, real_points)
    malliavin = compute_metrics("Malliavin", malliavin_points, real_points)
    metrics = {
        "source": "upstream-riemannian-score-sde",
        "coordinate_convention": "upstream-earthquake-antipodal",
        "real_count": int(real_points.shape[0]),
        "mmd_sigma": MMD_SIGMA,
        "metric_subsample": METRIC_SUBSAMPLE,
        "evaluation_seed": SEED,
        "metric_direction": "generated_to_observed",
        "metric_units": {"nearest_neighbor_geodesic": "radians"},
        "sample_paths": {
            "heat": str(heat_samples_path),
            "malliavin": str(malliavin_samples_path),
        },
        "models": {"heat": heat, "malliavin": malliavin},
    }

    json_path = output_dir / "heat_malliavin_metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    columns = (
        "model",
        "generated_count",
        "s2_rbf_mmd",
        "nearest_neighbor_geodesic_mean",
        "nearest_neighbor_geodesic_median",
        "nearest_neighbor_geodesic_max",
    )
    csv_path = output_dir / "heat_malliavin_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows((heat, malliavin))

    for result in (heat, malliavin):
        print(f"\n{result['model']}")
        print(f"  MMD = {result['s2_rbf_mmd']:.10f}")
        print(
            "  NN geo mean = "
            f"{result['nearest_neighbor_geodesic_mean']:.10f}"
        )
        print(
            "  NN geo median = "
            f"{result['nearest_neighbor_geodesic_median']:.10f}"
        )
        print(
            "  NN geo max = "
            f"{result['nearest_neighbor_geodesic_max']:.10f}"
        )
    print(f"\nsaved: {json_path}")
    print(f"saved: {csv_path}")
    return heat, malliavin


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    heat_run_dir = args.heat_run_dir.expanduser().resolve()
    malliavin_run_dir = args.malliavin_run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else malliavin_run_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    heat_samples_path = resolve_samples_path(
        heat_run_dir, args.heat_samples_path, label="Heat"
    )
    malliavin_samples_path = resolve_samples_path(
        malliavin_run_dir, args.malliavin_samples_path, label="Malliavin"
    )
    real_latlon = load_earthquake_latlon(args.data_path.expanduser().resolve())
    real_points = latlon_to_upstream_s2(real_latlon)
    heat_points, heat_latlon = load_generated(heat_samples_path, "Heat")
    malliavin_points, malliavin_latlon = load_generated(
        malliavin_samples_path, "Malliavin"
    )

    central_lat, central_lon = _resolve_map_center(
        real_latlon, args.central_lat, args.central_lon
    )
    print(f"map centre: latitude={central_lat:.3f}, longitude={central_lon:.3f}")

    save_metric_comparison(
        real_points,
        heat_points,
        malliavin_points,
        output_dir,
        heat_samples_path=heat_samples_path,
        malliavin_samples_path=malliavin_samples_path,
    )
    save_scatter_comparison(
        real_latlon,
        heat_latlon,
        malliavin_latlon,
        output_dir,
        central_lat=central_lat,
        central_lon=central_lon,
    )
    save_density_comparison(
        real_points,
        heat_points,
        malliavin_points,
        output_dir,
        central_lat=central_lat,
        central_lon=central_lon,
    )


if __name__ == "__main__":
    main()
