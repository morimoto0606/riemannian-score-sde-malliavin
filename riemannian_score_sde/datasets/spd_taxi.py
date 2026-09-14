"""Published Taxi matrices with continuous predictors and fixed splits."""
import numpy as np
import jax.numpy as jnp
from score_sde.datasets import TensorDataset, SubDataset
from riemannian_score_sde.taxi_data import validate, select_splits


class SPDTaxiDataset(TensorDataset):
    def __init__(self, data_path, dataset_seed=0, rng=None, split_protocol="development"):
        del rng
        with np.load(data_path, allow_pickle=False) as z:
            x, context = z['covariances'], z['contexts']
            self.indices = {s:z[s+'_indices'] for s in ('train','val','test')}
        validate(x, context, self.indices)
        self.split_protocol = split_protocol
        self.indices = select_splits(self.indices, split_protocol)
        self.contexts = jnp.asarray(context)
        self.dataset_seed = dataset_seed
        super().__init__(x)

    def __getitem__(self, idx):
        return self.data[idx], self.contexts[idx]

    def chronological_splits(self, lengths):
        return tuple(SubDataset(self, self.indices[s]) for s in ('train','val','test'))
