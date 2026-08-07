#!/usr/bin/env python3
"""Visualise and evaluate native upstream Earthquake reverse samples.

This script is deliberately independent of scoremodel_ext.  It reads the
upstream CSV and native ``generated_samples.npy`` artifact, converts the
repository's antipodal S2 embedding back to geographic latitude/longitude,
and writes comparison plots and distribution metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_RUN_DIR = Path("results/earthquake_upstream_heat_baseline")
log = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/quakes_all.csv"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=(
            "Run directory containing generated_samples.npy and receiving output "
            "artifacts (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--samples-path",
        type=Path,
        default=None,
        help="Generated samples path; defaults to RUN_DIR/generated_samples.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to RUN_DIR.",
    )
    parser.add_argument("--grid-lat", type=int, default=180)
    parser.add_argument("--grid-lon", type=int, default=360)
    parser.add_argument("--kappa", type=float, default=80.0)
    parser.add_argument("--mmd-sigma", type=float, default=1.0)
    parser.add_argument("--metric-subsample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--central-lat",
        type=float,
        default=None,
        help="Orthographic centre latitude; defaults to the observed-data mean.",
    )
    parser.add_argument(
        "--central-lon",
        type=float,
        default=None,
        help="Orthographic centre longitude; defaults to the observed circular mean.",
    )
    return parser.parse_args(argv)


def load_earthquake_latlon(path: Path) -> np.ndarray:
    """Load the same latitude/longitude rows used by upstream Earthquake."""

    values = np.genfromtxt(path, delimiter=",", skip_header=4)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError(f"expected non-empty latitude/longitude CSV at {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"earthquake CSV contains non-finite values: {path}")
    if np.any(np.abs(values[:, 0]) > 90.0) or np.any(np.abs(values[:, 1]) > 180.0):
        raise ValueError(f"earthquake CSV contains invalid latitude/longitude: {path}")
    return values


def latlon_to_upstream_s2(latlon_degrees: np.ndarray) -> np.ndarray:
    r"""Apply the exact upstream Earthquake embedding convention.

    ``SphericalDataset`` first forms ``(theta, phi)`` as

    ``theta = latitude + pi/2`` and ``phi = longitude + pi``.

    Geomstats then maps spherical coordinates to S2.  Equivalently, the
    resulting point is the antipode of the usual geographic xyz embedding.
    """

    lat = np.deg2rad(latlon_degrees[:, 0])
    lon = np.deg2rad(latlon_degrees[:, 1])
    cos_lat = np.cos(lat)
    return np.stack(
        (-cos_lat * np.cos(lon), -cos_lat * np.sin(lon), -np.sin(lat)),
        axis=1,
    )


def upstream_s2_to_latlon(points: np.ndarray) -> np.ndarray:
    """Invert the upstream antipodal S2 embedding into geographic degrees."""

    points = validate_s2_points(points, "S2 points")
    standard_earth = -points
    lat = np.rad2deg(np.arcsin(np.clip(standard_earth[:, 2], -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(standard_earth[:, 1], standard_earth[:, 0]))
    return np.stack((lat, lon), axis=1)


def validate_s2_points(points: np.ndarray, label: str) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"{label} must have shape (n, 3), got {points.shape}")
    if not np.issubdtype(points.dtype, np.number) or not np.isfinite(points).all():
        raise ValueError(f"{label} must contain finite numeric values")
    points = points.astype(np.float64, copy=False)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{label} contains a zero vector")
    maximum_error = float(np.max(np.abs(norms - 1.0)))
    if maximum_error > 1e-3:
        raise ValueError(
            f"{label} is not on S2; maximum norm error is {maximum_error:.6g}"
        )
    return points / norms


def stable_subsample(
    points: np.ndarray, maximum_count: int, rng: np.random.Generator
) -> np.ndarray:
    if maximum_count < 1:
        raise ValueError("subsample size must be positive")
    if points.shape[0] <= maximum_count:
        return points
    indices = rng.choice(points.shape[0], maximum_count, replace=False)
    return points[indices]


def s2_rbf_mmd(
    samples: np.ndarray,
    reference: np.ndarray,
    *,
    sigma: float = 1.0,
    n_sub: int = 2000,
    seed: int = 0,
) -> float:
    """Unbiased RBF MMD using ambient chordal distance on S2."""

    if sigma <= 0.0:
        raise ValueError("MMD sigma must be positive")
    samples = validate_s2_points(samples, "generated samples")
    reference = validate_s2_points(reference, "reference samples")
    rng = np.random.default_rng(seed)
    x = stable_subsample(samples, n_sub, rng)
    y = stable_subsample(reference, n_sub, rng)

    def gram(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance_squared = np.maximum(2.0 - 2.0 * (left @ right.T), 0.0)
        return np.exp(-distance_squared / (2.0 * sigma**2))

    k_xx = gram(x, x)
    k_yy = gram(y, y)
    k_xy = gram(x, y)
    xx = 0.0 if len(x) < 2 else (k_xx.sum() - np.trace(k_xx)) / (len(x) * (len(x) - 1))
    yy = 0.0 if len(y) < 2 else (k_yy.sum() - np.trace(k_yy)) / (len(y) * (len(y) - 1))
    return float(xx + yy - 2.0 * k_xy.mean())


def nearest_neighbor_geodesic(
    samples: np.ndarray,
    reference: np.ndarray,
    *,
    n_sub: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Summarise generated-to-real nearest-neighbour great-circle distance."""

    samples = validate_s2_points(samples, "generated samples")
    reference = validate_s2_points(reference, "reference samples")
    x = stable_subsample(samples, n_sub, np.random.default_rng(seed))
    y = stable_subsample(reference, n_sub, np.random.default_rng(seed + 1))
    cosine = np.clip(x @ y.T, -1.0, 1.0)
    nearest = np.min(np.arccos(cosine), axis=1)
    return {
        "mean": float(np.mean(nearest)),
        "median": float(np.median(nearest)),
        "max": float(np.max(nearest)),
    }


def spherical_kde(
    points: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    kappa: float,
    chunk_size: int = 128,
) -> np.ndarray:
    """Evaluate a von Mises--Fisher kernel density on a geographic grid."""

    if kappa <= 0.0:
        raise ValueError("KDE kappa must be positive")
    points = validate_s2_points(points, "KDE points")
    lat_mesh, lon_mesh = np.meshgrid(latitudes, longitudes, indexing="ij")
    grid_latlon = np.stack((lat_mesh.ravel(), lon_mesh.ravel()), axis=1)
    grid = latlon_to_upstream_s2(grid_latlon)
    density = np.empty(grid.shape[0], dtype=np.float64)
    for start in range(0, grid.shape[0], chunk_size):
        stop = min(start + chunk_size, grid.shape[0])
        density[start:stop] = np.exp(
            kappa * (grid[start:stop] @ points.T - 1.0)
        ).mean(axis=1)
    return density.reshape(lat_mesh.shape)


def _plot_backend():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs
    except ImportError:
        ccrs = None
        log.warning("Cartopy unavailable: using PlateCarree fallback")
    return plt, ccrs


def _resolve_map_center(
    real_latlon: np.ndarray,
    central_lat: float | None,
    central_lon: float | None,
) -> tuple[float, float]:
    """Resolve an Earthquake-centred globe view without changing point data."""

    if central_lat is None:
        central_lat = float(np.mean(real_latlon[:, 0]))
    if central_lon is None:
        longitude_radians = np.deg2rad(real_latlon[:, 1])
        central_lon = float(
            np.rad2deg(
                np.arctan2(
                    np.mean(np.sin(longitude_radians)),
                    np.mean(np.cos(longitude_radians)),
                )
            )
        )
    if not -90.0 <= central_lat <= 90.0:
        raise ValueError("central latitude must be in [-90, 90]")
    if not np.isfinite(central_lon):
        raise ValueError("central longitude must be finite")
    central_lon = ((central_lon + 180.0) % 360.0) - 180.0
    return central_lat, central_lon


def _add_map_axis(
    fig,
    position,
    title: str,
    ccrs,
    *,
    central_lat: float,
    central_lon: float,
):
    if ccrs is None:
        ax = fig.add_subplot(position)
        ax.set_xlim(-180.0, 180.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.grid(color="#cccccc", linewidth=0.4, alpha=0.7)
        transform = None
    else:
        projection = ccrs.Orthographic(
            central_longitude=central_lon,
            central_latitude=central_lat,
        )
        ax = fig.add_subplot(position, projection=projection)
        ax.set_global()
        ax.stock_img()
        ax.coastlines(linewidth=0.55, color="#333333")
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)
        transform = ccrs.PlateCarree()
    ax.set_title(title)
    return ax, transform


def _scatter(ax, latlon: np.ndarray, *, color: str, label: str, transform, alpha=0.5):
    kwargs = {} if transform is None else {"transform": transform}
    ax.scatter(
        latlon[:, 1],
        latlon[:, 0],
        s=2.0,
        alpha=alpha,
        color=color,
        label=label,
        linewidths=0,
        **kwargs,
    )


def save_scatter_outputs(
    real_latlon: np.ndarray,
    generated_latlon: np.ndarray,
    output_dir: Path,
    *,
    central_lat: float,
    central_lon: float,
) -> None:
    plt, ccrs = _plot_backend()

    for filename, title, points, color in (
        ("earthquake_real_map.png", "Observed earthquakes", real_latlon, "#b2182b"),
        (
            "earthquake_generated_map.png",
            "Upstream Heat generated samples",
            generated_latlon,
            "#2166ac",
        ),
    ):
        fig = plt.figure(figsize=(7, 7), dpi=200)
        ax, transform = _add_map_axis(
            fig,
            111,
            title,
            ccrs,
            central_lat=central_lat,
            central_lon=central_lon,
        )
        _scatter(ax, points, color=color, label=title, transform=transform, alpha=0.55)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig = plt.figure(figsize=(7, 7), dpi=200)
    ax, transform = _add_map_axis(
        fig,
        111,
        "Observed and generated earthquakes",
        ccrs,
        central_lat=central_lat,
        central_lon=central_lon,
    )
    _scatter(
        ax,
        real_latlon,
        color="#b2182b",
        label="Observed",
        transform=transform,
        alpha=0.45,
    )
    _scatter(
        ax,
        generated_latlon,
        color="#2166ac",
        label="Generated",
        transform=transform,
        alpha=0.35,
    )
    ax.legend(loc="lower left", markerscale=4.0)
    fig.tight_layout()
    fig.savefig(output_dir / "earthquake_overlay_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(14, 7), dpi=200)
    for column, (title, points, color) in enumerate(
        (
            ("Observed earthquakes", real_latlon, "#b2182b"),
            ("Upstream Heat samples", generated_latlon, "#2166ac"),
        ),
        start=1,
    ):
        ax, transform = _add_map_axis(
            fig,
            120 + column,
            title,
            ccrs,
            central_lat=central_lat,
            central_lon=central_lon,
        )
        _scatter(ax, points, color=color, label=title, transform=transform, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_density_comparison(
    real_points: np.ndarray,
    generated_points: np.ndarray,
    output_dir: Path,
    *,
    grid_lat: int,
    grid_lon: int,
    kappa: float,
    central_lat: float,
    central_lon: float,
) -> None:
    if grid_lat < 2 or grid_lon < 2:
        raise ValueError("density grid dimensions must be at least two")
    latitudes = np.linspace(-90.0, 90.0, grid_lat)
    longitudes = np.linspace(-180.0, 180.0, grid_lon)
    real_density = spherical_kde(
        real_points, latitudes, longitudes, kappa=kappa
    )
    generated_density = spherical_kde(
        generated_points, latitudes, longitudes, kappa=kappa
    )
    common_maximum = max(float(real_density.max()), float(generated_density.max()))
    if common_maximum <= 0.0:
        raise FloatingPointError("density grid is identically zero")
    real_density /= common_maximum
    generated_density /= common_maximum

    plt, ccrs = _plot_backend()
    fig = plt.figure(figsize=(14, 7), dpi=200)
    contour = None
    for column, (title, density) in enumerate(
        (
            ("Observed density", real_density),
            ("Upstream Heat density", generated_density),
        ),
        start=1,
    ):
        ax, transform = _add_map_axis(
            fig,
            120 + column,
            title,
            ccrs,
            central_lat=central_lat,
            central_lon=central_lon,
        )
        kwargs = {} if transform is None else {"transform": transform}
        contour = ax.contourf(
            longitudes,
            latitudes,
            density,
            levels=np.linspace(0.0, 1.0, 41),
            cmap="magma",
            alpha=0.88,
            extend="max",
            **kwargs,
        )
    fig.colorbar(contour, ax=fig.axes, shrink=0.82, label="Shared relative density")
    fig.savefig(output_dir / "density_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_path = args.data_path.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    samples_path = (
        args.samples_path.expanduser().resolve()
        if args.samples_path is not None
        else run_dir / "generated_samples.npy"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    real_latlon = load_earthquake_latlon(data_path)
    real_points = latlon_to_upstream_s2(real_latlon)
    generated_raw = np.load(samples_path, allow_pickle=False)
    generated_points = validate_s2_points(generated_raw, "generated samples")
    generated_latlon = upstream_s2_to_latlon(generated_points)
    central_lat, central_lon = _resolve_map_center(
        real_latlon,
        args.central_lat,
        args.central_lon,
    )
    log.info(
        "Using Orthographic centre: latitude=%.3f, longitude=%.3f",
        central_lat,
        central_lon,
    )

    save_scatter_outputs(
        real_latlon,
        generated_latlon,
        output_dir,
        central_lat=central_lat,
        central_lon=central_lon,
    )
    save_density_comparison(
        real_points,
        generated_points,
        output_dir,
        grid_lat=args.grid_lat,
        grid_lon=args.grid_lon,
        kappa=args.kappa,
        central_lat=central_lat,
        central_lon=central_lon,
    )

    geodesic = nearest_neighbor_geodesic(
        generated_points,
        real_points,
        n_sub=args.metric_subsample,
        seed=args.seed,
    )
    metrics = {
        "source": "upstream-riemannian-score-sde",
        "teacher": "heat",
        "coordinate_convention": "upstream-earthquake-antipodal",
        "generated_count": int(generated_points.shape[0]),
        "real_count": int(real_points.shape[0]),
        "s2_rbf_mmd": s2_rbf_mmd(
            generated_points,
            real_points,
            sigma=args.mmd_sigma,
            n_sub=args.metric_subsample,
            seed=args.seed,
        ),
        "nearest_neighbor_geodesic_mean": geodesic["mean"],
        "nearest_neighbor_geodesic_median": geodesic["median"],
        "nearest_neighbor_geodesic_max": geodesic["max"],
        "metric_units": {"nearest_neighbor_geodesic": "radians"},
        "metric_direction": "generated_to_real",
        "mmd_sigma": args.mmd_sigma,
        "metric_subsample": args.metric_subsample,
        "evaluation_seed": args.seed,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    print(f"saved Earthquake baseline artifacts in {output_dir}")


if __name__ == "__main__":
    main()
