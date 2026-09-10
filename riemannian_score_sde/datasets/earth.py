import os

from score_sde.utils import register_dataset
from score_sde.datasets import CSVDataset

import geomstats as gs
import jax.numpy as jnp
from riemannian_score_sde.earth_data import EARTH_DATA, spherical_angles


class SphericalDataset(CSVDataset):
    def __init__(self, file, extrinsic=False, delimiter=",", skip_header=1):
        super().__init__(file, delimiter=delimiter, skip_header=skip_header)

        self.manifold = gs.geometry.hypersphere.Hypersphere(2)
        self.intrinsic_data = spherical_angles(self.data, jnp)
        self.data = self.manifold.spherical_to_extrinsic(self.intrinsic_data)


class VolcanicErruption(SphericalDataset):
    def __init__(self, data_dir="data", **kwargs):
        spec = EARTH_DATA["volcanoe"]
        super().__init__(os.path.join(data_dir, spec["file"]), skip_header=spec["skip_header"])


class Fire(SphericalDataset):
    def __init__(self, data_dir="data", **kwargs):
        super().__init__(os.path.join(data_dir, "fire.csv"))


class Flood(SphericalDataset):
    def __init__(self, data_dir="data", **kwargs):
        spec = EARTH_DATA["flood"]
        super().__init__(os.path.join(data_dir, spec["file"]), skip_header=spec["skip_header"])


class Earthquake(SphericalDataset):
    def __init__(self, data_dir="data", **kwargs):
        spec = EARTH_DATA["earthquake"]
        super().__init__(os.path.join(data_dir, spec["file"]), skip_header=spec["skip_header"])
