"""Shared earth-data CSV definitions and SphericalDataset coordinates (no JAX)."""

import numpy as np


EARTH_DATA = {
    "earthquake": {"class": "Earthquake", "file": "quakes_all.csv", "skip_header": 4,
                   "plural": "earthquakes"},
    "flood": {"class": "Flood", "file": "flood.csv", "skip_header": 2,
              "plural": "floods"},
    "volcano": {"class": "VolcanicErruption", "file": "volerup.csv", "skip_header": 2,
                 "plural": "volcanoes"},
}


def spherical_angles(latlon, xp):
    """Training's original theta=lat+pi/2, phi=lon+pi convention."""
    return xp.pi * (latlon / 180.0) + xp.array([xp.pi / 2, xp.pi])[None, :]


def latlon_to_s2(latlon):
    """Equivalent antipodal form of spherical_angles, in NumPy float64.

    Retains the original postprocessor's operation order for identical metrics.
    All SphericalDataset subclasses share this convention.
    """
    lat = np.deg2rad(latlon[:, 0])
    lon = np.deg2rad(latlon[:, 1])
    cos_lat = np.cos(lat)
    return np.stack(
        (-cos_lat * np.cos(lon), -cos_lat * np.sin(lon), -np.sin(lat)), axis=1
    )


def load_latlon(path, dataset):
    spec = EARTH_DATA[dataset]
    values = np.asarray(np.genfromtxt(
        path, delimiter=",", skip_header=spec["skip_header"]), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError(f"expected non-empty latitude/longitude CSV at {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"{dataset} CSV contains non-finite values: {path}")
    if np.any(np.abs(values[:, 0]) > 90) or np.any(np.abs(values[:, 1]) > 180):
        raise ValueError(f"{dataset} CSV contains invalid latitude/longitude: {path}")
    return values
