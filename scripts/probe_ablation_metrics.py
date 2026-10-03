#!/usr/bin/env python3
"""Numeric-only paper-matched metrics from saved samples (NumPy, no JAX)."""
import argparse
import json
from pathlib import Path

import numpy as np

from recompute_so3_chordal_mmd import (
    chordal_mmd2, constraint_errors, load_rotations, metric_indices, sha256,
)


def stats(values):
    return {key: float(fn(values)) for key, fn in
            [('mean', np.mean), ('median', np.median), ('max', np.max)]}


def so3_nearest(source, reference, chunk_size=256):
    # Same formula, chunking and original array precision as the paper evaluator.
    nearest = []
    for start in range(0, len(source), chunk_size):
        traces = np.einsum('aij,bij->ab', source[start:start+chunk_size], reference)
        distance = np.sqrt(2.0) * np.arccos(np.clip((traces-1.0)/2.0, -1.0, 1.0))
        nearest.append(distance.min(axis=1))
    return np.concatenate(nearest)


def so3_metrics(generated_path, reference_path):
    x64, y64 = load_rotations(generated_path), load_rotations(reference_path)
    if len(x64) != 16384 or len(y64) != 16384:
        raise ValueError('Paper SO3 protocol requires 16384 generated and reference samples')
    gi, ri = metric_indices(len(x64), len(y64))
    # The final chordal MMD uses float64; old NN values retained saved precision.
    x = np.load(generated_path, allow_pickle=False).reshape(-1, 3, 3)[gi]
    y = np.load(reference_path, allow_pickle=False).reshape(-1, 3, 3)[ri]
    return dict(mmd2=chordal_mmd2(x64[gi], y64[ri]),
        generated_to_reference=stats(so3_nearest(x, y)),
        reference_to_generated=stats(so3_nearest(y, x)),
        generated_count=len(x64), reference_count=len(y64),
        constraints=dict(generated=constraint_errors(x64), reference=constraint_errors(y64)),
        kernel='exp(-||R-Q||_F^2 / 2), sigma=1',
        nn_units='||Log(R^T Q)||_F = sqrt(2) times rotation angle',
        reference_scope='saved independent draws from target distribution, target seed 0, draw seed 10000',
        subset_rule='default_rng(0): generated first, reference next; 2000 each',
        subset_indices=dict(generated=gi.tolist(), reference=ri.tolist()))


def earthquake_metrics(generated_path, reference_path):
    from postprocess_s2_earth_data import (
        load_latlon, latlon_to_upstream_s2, validate_s2_points, stable_subsample, s2_rbf_mmd,
    )
    x = validate_s2_points(np.load(generated_path, allow_pickle=False), 'generated')
    if len(x) != 16384:
        raise ValueError('Paper Earth protocol requires 16384 generated samples')
    y = validate_s2_points(latlon_to_upstream_s2(load_latlon(reference_path, 'earthquake')), 'reference')
    gi = stable_subsample(np.arange(len(x)), 2000, np.random.default_rng(0))
    ri = stable_subsample(np.arange(len(y)), 2000, np.random.default_rng(1))
    distance = np.arccos(np.clip(x[gi] @ y[ri].T, -1.0, 1.0))
    return dict(mmd2=s2_rbf_mmd(x, y, sigma=1.0, n_sub=2000, seed=0),
        generated_to_reference=stats(distance.min(axis=1)),
        reference_to_generated=stats(distance.min(axis=0)),
        generated_count=len(x), reference_count=len(y), kernel='exp(-||x-y||^2 / 2), sigma=1',
        nn_units='great-circle radians', reference_scope='all observed data including training data',
        subset_rule='MMD: one rng(0), generated then reference. NN: rng(0) generated, rng(1) reference; at most 2000 each',
        subset_indices=dict(nn_generated=gi.tolist(), nn_reference=ri.tolist()))


def compute(suite, generated, reference):
    report = (so3_metrics if suite == 'so3' else earthquake_metrics)(generated, reference)
    values = {'mmd2_unbiased': report.pop('mmd2')}
    for direction in ('generated_to_reference', 'reference_to_generated'):
        values.update({direction+'_nn_'+k: v for k, v in report.pop(direction).items()})
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite generation metric')
    return dict(values=values, definition=report, suite=suite,
                generated_path=str(generated), generated_sha256=sha256(generated),
                reference_path=str(reference), reference_sha256=sha256(reference),
                estimator='diagonal-excluded unbiased MMD squared; negative estimates retained',
                metric_seed=0, metric_subsample=2000)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['so3', 'earthquake'], required=True)
    p.add_argument('--generated', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; existing metrics are never overwritten')
    report = compute(args.suite, args.generated, args.reference)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(report['values'], indent=2))


if __name__ == '__main__':
    main()
