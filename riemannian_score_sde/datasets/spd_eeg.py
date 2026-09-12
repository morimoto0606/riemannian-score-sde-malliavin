"""Prepared EEG covariances with fixed subject-disjoint splits and labels."""
import numpy as np
import jax.numpy as jnp
from score_sde.datasets import TensorDataset, SubDataset


class SPDEEGDataset(TensorDataset):
    def __init__(self, data_path, dimension=15, dataset_seed=0, rng=None):
        del rng
        with np.load(data_path, allow_pickle=False) as z:
            x, labels = z['covariances'], z['labels']
            self.indices = {s: z[s+'_indices'] for s in ('train','val','test')}
            subjects = z['subjects']
        if x.ndim != 3 or x.shape[1:] != (dimension, dimension) or not np.isfinite(x).all():
            raise ValueError('Invalid EEG covariance shape or values')
        if not np.allclose(x,x.swapaxes(-1,-2)) or np.linalg.eigvalsh(x).min() <= 0:
            raise ValueError('EEG covariances must be SPD')
        if labels.shape != (len(x),) or not np.isin(labels,[0,1]).all():
            raise ValueError('Expected binary EEG labels')
        seen = set()
        for idx in self.indices.values():
            if not len(idx) or np.any(idx<0) or np.any(idx>=len(x)) or len(np.unique(idx)) != len(idx):
                raise ValueError('Invalid split indices')
            group = set(subjects[idx].tolist())
            if seen & group:
                raise ValueError('Subject leakage between splits')
            if set(labels[idx].tolist()) != {0,1}:
                raise ValueError('Each split must contain both classes')
            seen |= group
        self.labels = jnp.asarray(labels,dtype=jnp.int32)
        self.dataset_seed = dataset_seed
        super().__init__(x)

    def __getitem__(self, idx):
        return self.data[idx], jnp.eye(2)[self.labels[idx]]

    def chronological_splits(self, lengths):
        # Persisted subject splits, not fractions or random trial splitting.
        return tuple(SubDataset(self,self.indices[s]) for s in ('train','val','test'))
