"""NumPy-only loader for the published SPD-DDPM conditional Taxi CSVs."""
import csv
import numpy as np


def read_csv(path):
    with open(path, newline='') as f:
        rows = csv.reader(f)
        next(rows)
        return np.array([[float(v) for v in row] for row in rows], dtype=np.float64)


def validate(x, context, indices):
    if x.ndim != 3 or x.shape[1:] != (10, 10) or not np.isfinite(x).all():
        raise ValueError('Expected finite Nx10x10 Taxi matrices')
    if not np.allclose(x, x.swapaxes(-1, -2), atol=1e-10, rtol=0) or np.linalg.eigvalsh(x).min() <= 0:
        raise ValueError('Taxi matrices must be symmetric positive definite; no repair applied')
    if context.shape != (len(x), 13) or not np.isfinite(context).all():
        raise ValueError('Expected finite Nx13 Taxi predictors')
    for idx in indices.values():
        if idx.ndim != 1 or not len(idx) or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError('Invalid split indices')
    joined = np.concatenate(list(indices.values()))
    if not np.array_equal(np.sort(joined), np.arange(len(x))):
        raise ValueError('Splits must partition all rows without overlap')


def prepare(source, seed=0):
    train = read_csv(source / 'train_data.csv')
    test_y = read_csv(source / 'test_y.csv')
    test_x = read_csv(source / 'test_spds.csv')
    if train.shape[1] != 114 or test_y.shape[1] != 14 or test_x.shape != (len(test_y), 101):
        raise ValueError('Unexpected upstream CSV schema')
    x = np.concatenate([train[:, 14:], test_x[:, 1:]]).reshape(-1, 10, 10)
    context = np.concatenate([train[:, 1:14], test_y[:, 1:]])
    perm = np.random.RandomState(seed).permutation(len(train))
    nv = int(np.ceil(0.15 * len(train)))
    indices = dict(train=perm[nv:], val=perm[:nv], test=np.arange(len(train), len(x)))
    validate(x, context, indices)
    return dict(covariances=x, contexts=context, **{k+'_indices':v for k,v in indices.items()})


def select_splits(indices, protocol='development'):
    """Select a runtime partition without changing the prepared NPZ.

    Published training rows are the union of the development train and val.
    Sorting restores their source order; test ordering is preserved exactly.
    """
    selected = {s: np.array(indices[s], copy=True) for s in ('train', 'val', 'test')}
    if protocol == 'development':
        return selected
    if protocol != 'published_train':
        raise ValueError('Taxi split_protocol must be development or published_train')
    selected['train'] = np.sort(np.concatenate([selected['train'], selected['val']]))
    selected['val'] = np.empty(0, dtype=selected['train'].dtype)
    return selected
