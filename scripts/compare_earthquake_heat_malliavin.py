#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from postprocess_earthquake_upstream import (
    load_earthquake_latlon,
    latlon_to_upstream_s2,
    upstream_s2_to_latlon,
    validate_s2_points,
    nearest_neighbor_geodesic,
    s2_rbf_mmd,
    _resolve_map_center,
    spherical_kde,
)
import cartopy.crs as ccrs
import cartopy.feature as cfeature


DATA_PATH = Path("data/quakes_all.csv")

HEAT_DIR = Path("results/earthquake_upstream_heat_baseline")
MALLIAVIN_DIR = Path("results/earthquake_malliavin_p8_default_600k")

OUTPUT_DIR = MALLIAVIN_DIR

GRID_LAT = 180
GRID_LON = 360
KAPPA = 80.0

MMD_SIGMA = 1.0
METRIC_SUBSAMPLE = 2000
SEED = 0


def load_generated(run_dir: Path):
    path = run_dir / "generated_samples.npy"
    raw = np.load(path, allow_pickle=False)
    points = validate_s2_points(raw, f"generated samples: {run_dir}")
    latlon = upstream_s2_to_latlon(points)
    return points, latlon


def setup_geo_axis(ax):
    ax.add_feature(cfeature.LAND, facecolor="0.95")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.gridlines(linewidth=0.3, alpha=0.5)


def save_scatter_comparison(
    real_latlon,
    heat_latlon,
    malliavin_latlon,
    central_lat,
    central_lon,
):
    projection = ccrs.Orthographic(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        subplot_kw={"projection": projection},
    )

    datasets = [
        ("Observed", real_latlon),
        ("Heat", heat_latlon),
        ("Malliavin", malliavin_latlon),
    ]

    for ax, (title, latlon) in zip(axes, datasets):
        setup_geo_axis(ax)

        ax.scatter(
            latlon[:, 1],
            latlon[:, 0],
            s=4,
            alpha=0.45,
            transform=ccrs.PlateCarree(),
        )

        ax.set_title(title)

    fig.suptitle("Earthquake scatter comparison")
    fig.tight_layout()

    path = OUTPUT_DIR / "scatter_observed_heat_malliavin.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {path}")


def save_density_comparison(
    real_points,
    heat_points,
    malliavin_points,
    central_lat,
    central_lon,
):
    latitudes = np.linspace(-90.0, 90.0, GRID_LAT)
    longitudes = np.linspace(-180.0, 180.0, GRID_LON)

    densities = []

    for points in [real_points, heat_points, malliavin_points]:
        density = spherical_kde(
            points,
            latitudes,
            longitudes,
            kappa=KAPPA,
        )
        densities.append(density)

    # 3者共通のスケール
    vmax = max(float(np.max(d)) for d in densities)

    lon_mesh, lat_mesh = np.meshgrid(
        longitudes,
        latitudes,
        indexing="xy",
    )

    projection = ccrs.Orthographic(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        subplot_kw={"projection": projection},
    )

    titles = ["Observed", "Heat", "Malliavin"]

    levels = np.linspace(0.0, vmax, 31)

    contour = None

    for ax, title, density in zip(axes, titles, densities):
        setup_geo_axis(ax)

        contour = ax.contourf(
            lon_mesh,
            lat_mesh,
            density,
            levels=levels,
            vmin=0.0,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )

        ax.set_title(title)

    fig.colorbar(
        contour,
        ax=axes,
        shrink=0.82,
        label="Shared relative density",
    )

    fig.suptitle("Earthquake density comparison")
    fig.savefig(
        OUTPUT_DIR / "density_observed_heat_malliavin.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        "saved:",
        OUTPUT_DIR / "density_observed_heat_malliavin.png",
    )


def compute_metrics(label, generated_points, real_points):
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


def save_metric_comparison(real_points, heat_points, malliavin_points):
    heat = compute_metrics("Heat", heat_points, real_points)
    malliavin = compute_metrics("Malliavin", malliavin_points, real_points)

    metrics = {
        "coordinate_convention": "upstream-earthquake-antipodal",
        "real_count": int(real_points.shape[0]),
        "mmd_sigma": MMD_SIGMA,
        "metric_subsample": METRIC_SUBSAMPLE,
        "evaluation_seed": SEED,
        "metric_units": {
            "nearest_neighbor_geodesic": "radians",
        },
        "models": {
            "heat": heat,
            "malliavin": malliavin,
        },
    }

    json_path = OUTPUT_DIR / "heat_malliavin_metrics.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = OUTPUT_DIR / "heat_malliavin_metrics.csv"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "model,generated_count,s2_rbf_mmd,"
            "nearest_neighbor_geodesic_mean,"
            "nearest_neighbor_geodesic_median,"
            "nearest_neighbor_geodesic_max\n"
        )

        for x in [heat, malliavin]:
            f.write(
                f"{x['model']},"
                f"{x['generated_count']},"
                f"{x['s2_rbf_mmd']},"
                f"{x['nearest_neighbor_geodesic_mean']},"
                f"{x['nearest_neighbor_geodesic_median']},"
                f"{x['nearest_neighbor_geodesic_max']}\n"
            )

    print()
    print("========================================")
    print("Heat vs Malliavin")
    print("========================================")

    for x in [heat, malliavin]:
        print(f"\n{x['model']}")
        print(f"  generated_count = {x['generated_count']}")
        print(f"  MMD             = {x['s2_rbf_mmd']:.8f}")
        print(
            f"  NN geo mean     = "
            f"{x['nearest_neighbor_geodesic_mean']:.8f}"
        )
        print(
            f"  NN geo median   = "
            f"{x['nearest_neighbor_geodesic_median']:.8f}"
        )
        print(
            f"  NN geo max      = "
            f"{x['nearest_neighbor_geodesic_max']:.8f}"
        )

    print()
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_latlon = load_earthquake_latlon(DATA_PATH)
    real_points = latlon_to_upstream_s2(real_latlon)

    heat_points, heat_latlon = load_generated(HEAT_DIR)
    malliavin_points, malliavin_latlon = load_generated(MALLIAVIN_DIR)

    central_lat, central_lon = _resolve_map_center(
        real_latlon,
        None,
        None,
    )

    print(
        f"map centre: lat={central_lat:.3f}, "
        f"lon={central_lon:.3f}"
    )

    save_metric_comparison(
        real_points,
        heat_points,
        malliavin_points,
    )

    save_scatter_comparison(
        real_latlon,
        heat_latlon,
        malliavin_latlon,
        central_lat,
        central_lon,
    )

    save_density_comparison(
        real_points,
        heat_points,
        malliavin_points,
        central_lat,
        central_lon,
    )


if __name__ == "__main__":
    main()