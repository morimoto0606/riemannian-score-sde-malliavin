#!/usr/bin/env python3
"""Prepare BNCI2014-002; no training, no automatic dependency installation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def prepare(epochs, labels, subjects):
    """OAS per trial; deterministic subject-disjoint split; no outlier filtering."""
    from sklearn.covariance import OAS
    x = np.asarray(epochs, dtype=np.float64)
    subjects = np.asarray(subjects).astype(str)
    labels = np.asarray(labels).astype(str)
    if x.ndim != 3 or x.shape[1] != 15 or x.shape[2] < 2:
        raise ValueError(f'Expected (trials, 15 channels, time>=2) BNCI2014-002 epochs; got {x.shape}')
    if not np.isfinite(x).all():
        raise ValueError(f'Nonfinite EEG values: {np.count_nonzero(~np.isfinite(x))}; shape={x.shape}')
    if len(subjects)!=len(x) or len(labels)!=len(x):
        raise ValueError('Mismatched trial metadata')
    classes, y = np.unique(labels,return_inverse=True)
    if len(classes)!=2:
        raise ValueError('Expected two motor imagery classes')
    people=np.unique(subjects)
    if len(people)<5:
        raise ValueError('At least five subjects required')
    people=np.random.RandomState(0).permutation(people)
    ntest=max(1,int(np.ceil(.2*len(people)))); nval=max(1,int(np.ceil(.2*len(people))))
    groups={'test':people[:ntest],'val':people[ntest:ntest+nval],'train':people[ntest+nval:]}
    cov=np.stack([OAS(store_precision=False).fit(trial.T).covariance_ for trial in x])
    if np.linalg.eigvalsh(cov).min()<=0:
        raise ValueError('OAS produced a non-SPD matrix')
    result=dict(covariances=cov,labels=y,subjects=subjects)
    for s,p in groups.items():
        idx=np.flatnonzero(np.isin(subjects,p))
        if set(y[idx])!={0,1}:
            raise ValueError('Each split must contain both classes')
        result[s+'_indices']=idx
    meta=dict(dataset='BNCI2014-002',class_names=classes.tolist(),dimension=15,
              covariance='OAS, per trial; no correlation normalization',
              split='subject-disjoint, RandomState(0) permutation; ceil(20%) val/test',
              subjects={s:p.tolist() for s,p in groups.items()},
              counts={s:len(result[s+'_indices']) for s in groups},
              protocol='controlled adaptation, not exact Ko-Lee reproduction',
              outlier_filter=False)
    return result,meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--epochs-npz',type=Path,help='epochs, labels, subjects; epochs=(trials,15,time)')
    source.add_argument('--download',action='store_true',help='Use existing MOABB/MNE installation to obtain EEG')
    parser.add_argument('--output',type=Path,default=Path('data/spd_eeg/bnci2014_002.npz'))
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.metadata.json').exists():
        parser.error('Output already exists; choose a new path')
    if args.download:
        try:
            import moabb
            from moabb.datasets import BNCI2014_002
            from moabb.paradigms import MotorImagery
        except ImportError:
            parser.error('MOABB/MNE unavailable. No packages installed. Use a prepared --epochs-npz or an existing EEG environment.')
        epochs,labels,metadata=MotorImagery().get_data(dataset=BNCI2014_002())
        subjects=metadata['subject'].to_numpy()
        provenance={'loader':'MOABB MotorImagery defaults','moabb_version':moabb.__version__}
    else:
        with np.load(args.epochs_npz,allow_pickle=False) as z:
            epochs,labels,subjects=(z[k] for k in ('epochs','labels','subjects'))
        provenance={'epochs_sha256':hashlib.sha256(args.epochs_npz.read_bytes()).hexdigest()}
    result,meta=prepare(epochs,labels,subjects)
    meta.update(provenance)
    result['metadata_json']=np.array(json.dumps(meta))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**result)
    meta['npz_sha256']=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    main()
