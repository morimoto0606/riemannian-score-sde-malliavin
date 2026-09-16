#!/usr/bin/env python3
"""Sequential final Taxi fits: three methods x seeds 0,1,2; no generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHODS = ('varadhan', 'ism', 'malliavin_hutchinson')


def commands(output, dataset):
    for seed in range(3):
        for method in METHODS:
            name = ('malliavin_lambda0' if method == 'malliavin_hutchinson' else method) + f'_seed{seed}'
            dest = output / name
            yield name, [sys.executable, '-u', 'main.py', f'experiment=spd_taxi_{method}',
                         f'seed={seed}', 'steps=100000', 'mode=train', 'resume=false', 'logger=csv',
                         'dataset.split_protocol=published_train', 'dataset.dataset_seed=0',
                         'dataset.data_path='+str(dataset), 'loss.time_weighting=false',
                         'loss.time_weight_lambda=0.0', 'generation.enabled=false',
                         'train_val=false', 'test_val=false', 'test_test=false',
                         'train_plot=false', 'test_plot=false',
                         'ckpt_dir='+str(dest/'ckpt'), 'hydra.run.dir='+str(dest),
                         'generated_samples_path='+str(dest/'generated_samples.npy')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'data/spd_taxi/nyc_taxi.npz')
    p.add_argument('--output', type=Path, help='Must not exist; default is a new results/spd_taxi_final_* directory')
    p.add_argument('--check-only', action='store_true', help='Validate data and print plan without training')
    args = p.parse_args()
    import numpy as np
    from riemannian_score_sde.taxi_data import validate, select_splits
    dataset = args.dataset.resolve()
    with np.load(dataset, allow_pickle=False) as z:
        indices = {s:z[s+'_indices'] for s in ('train','val','test')}
        validate(z['covariances'], z['contexts'], indices)
        final = select_splits(indices, 'published_train')
        counts = {s:len(final[s]) for s in final}
        if counts != dict(train=7600, val=0, test=1159):
            raise ValueError('Unexpected published split sizes: '+str(counts))
    print('Final split:', counts, flush=True)
    output = args.output.resolve() if args.output else ROOT/'results/spd_taxi_final_NEW'
    if args.output and output.exists():
        raise FileExistsError('Refusing existing output: '+str(output))
    if args.check_only:
        for name, _ in commands(output, dataset):
            print(name, 'steps=100000 lambda=0 split=published_train')
        return
    if args.output:
        output.mkdir(parents=True)
    else:
        output = Path(tempfile.mkdtemp(prefix='spd_taxi_final_', dir=ROOT/'results'))
    plan = list(commands(output, dataset))
    manifest = dict(dataset=str(dataset), dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    counts=counts, selected_lambda=0, training_seeds=[0,1,2],
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
                    commands=dict(plan))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Final training output:', output, flush=True)
    env = dict(os.environ, GEOMSTATS_BACKEND='jax', JAX_ENABLE_X64='true',
               JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false', HYDRA_FULL_ERROR='1')
    env.pop('JAX_PLATFORM_NAME', None)
    env['PYTHONPATH'] = str(ROOT/'geomstats')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    for name, command in plan:
        print('TRAIN:', name, flush=True)
        with (output/(name+'_train.log')).open('x') as log:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (output/(name+'_process.json')).write_text(json.dumps(dict(returncode=proc.returncode))+'\n')
        if proc.returncode:
            raise SystemExit(f'FAILED: {name}; inspect {output}/{name}_train.log. Existing runs are preserved.')
        print('DONE:', name, flush=True)
    print('ALL TRAINING PROCESSES DONE; verify checkpoint steps and finiteness before evaluation.', flush=True)


if __name__ == '__main__':
    main()
