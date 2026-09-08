"""Fixed financial SPD samples with persisted, purged chronological splits."""

import json

import numpy as np

from score_sde.datasets import SubDataset, TensorDataset


class SPDFinanceDataset(TensorDataset):
    def __init__(self, data_path, dimension=5, rolling_window=60,
                 metric="affine_invariant", dataset_seed=0,
                 split_fractions=(0.7, 0.15, 0.15), rng=None):
        # rng is supplied by run.py from the training seed. Do not use it for
        # preprocessing, split assignment, or construction of the target data.
        del rng
        with np.load(data_path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata_json"]))
            cov = data["covariances"].copy()
            self.dates = data["dates"].copy()
            self.split_indices = {name: data[f"{name}_indices"].copy()
                                  for name in ("train", "val", "test")}
            starts = data["window_start_return_index"].copy()
            ends = data["window_end_return_index"].copy()
        if cov.ndim != 3 or cov.shape[1:] != (dimension, dimension):
            raise ValueError("Dataset dimension does not match config")
        if self.metadata["rolling_window"] != rolling_window or self.metadata["metric"] != metric:
            raise ValueError("Window/metric differs from built dataset; regenerate data or fix config")
        if not np.allclose(self.metadata["split_fractions_before_purge"], split_fractions):
            raise ValueError("Split fractions differ from snapshot; regenerate dataset")
        if not np.isfinite(cov).all() or not np.allclose(cov, cov.swapaxes(-1, -2), rtol=1e-12, atol=0):
            raise ValueError("Dataset contains nonfinite or asymmetric matrices")
        if np.any(np.linalg.eigvalsh(cov) <= 0):
            raise ValueError("Dataset contains non-SPD matrices")
        if len(self.dates) != len(cov) or np.any(np.diff(self.dates).astype(int) <= 0):
            raise ValueError("Dataset dates must be strictly increasing")
        previous_end = None
        for idx in self.split_indices.values():
            if idx.ndim != 1 or idx.dtype.kind not in "iu" or not len(idx) or np.any(idx < 0) or np.any(idx >= len(cov)) or np.any(np.diff(idx) <= 0):
                raise ValueError("Invalid persisted split indices")
            if previous_end is not None and starts[idx[0]] <= previous_end + 1:
                raise ValueError("Split leaks underlying prices/returns across boundary")
            previous_end = ends[idx[-1]]
        self.dataset_seed = dataset_seed  # Reserved; deterministic data ignores it.
        super().__init__(cov)

    def chronological_splits(self, lengths):
        if lengths is None or not np.allclose(lengths, self.metadata["split_fractions_before_purge"]):
            raise ValueError("Use the dataset's configured chronological fractions; random split is forbidden")
        return tuple(SubDataset(self, indices) for indices in self.split_indices.values())
