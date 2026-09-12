#!/usr/bin/env python3
"""Create a separate EEG absolute-value-filtered dataset; preserve split membership."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def filter_dataset(source, output, threshold=10000.0):
    source, output = Path(source), Path(output)
    metadata_path = output.with_suffix('.metadata.json')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Threshold must be positive and finite')
    if output.exists() or metadata_path.exists():
        raise FileExistsError('Choose a new output path')
    with np.load(source, allow_pickle=False) as z:
        x, y, subjects = (z[k] for k in ('covariances', 'labels', 'subjects'))
        splits = {s: z[s+'_indices'] for s in ('train','val','test')}
        metadata = json.loads(str(z['metadata_json']))
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError('Source must contain finite covariance matrices')
    maximum = np.max(np.abs(x), axis=(1,2))
    keep = maximum < threshold
    retained = np.flatnonzero(keep)
    remap = np.full(len(x), -1, dtype=np.int64)
    remap[retained] = np.arange(len(retained))
    result = dict(covariances=x[keep], labels=y[keep], subjects=subjects[keep],
                  source_indices=retained)
    removed = []
    counts = {}
    for split, idx in splits.items():
        selected = idx[keep[idx]]
        if set(y[selected].tolist()) != {0,1}:
            raise ValueError('Filtering would empty a class in '+split)
        result[split+'_indices'] = remap[selected]
        counts[split] = dict(before=len(idx), after=len(selected), removed=len(idx)-len(selected))
        for i in idx[~keep[idx]]:
            removed.append(dict(source_index=int(i), split=split, subject=str(subjects[i]),
                                label=int(y[i]), maximum_absolute_entry=float(maximum[i])))
    metadata['outlier_filter'] = dict(rule='max(abs(cov)) < threshold', threshold=threshold,
                                     source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                     counts=counts, removed=removed,
                                     split_membership_preserved=True, mahalanobis_filter=False)
    metadata['counts'] = {s:counts[s]['after'] for s in counts}
    metadata['protocol'] = 'controlled absolute-value filtering ablation, not exact Ko-Lee reproduction'
    metadata.pop('npz_sha256', None)
    result['metadata_json'] = np.array(json.dumps(metadata))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    metadata['npz_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/spd_eeg/bnci2014_002.npz'))
    p.add_argument('--output',type=Path,default=Path('data/spd_eeg/bnci2014_002_abs10000.npz'))
    p.add_argument('--threshold',type=float,default=10000.)
    a=p.parse_args()
    print(json.dumps(filter_dataset(a.source,a.output,a.threshold),indent=2))


if __name__=='__main__':main()
