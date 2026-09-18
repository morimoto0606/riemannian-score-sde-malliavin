#!/usr/bin/env python3
"""Final held-out Taxi evaluation of three methods and three training seeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from validate_spd_taxi import ROOT, digest, checkpoint_hashes, write_json, worker

METHODS = ('varadhan', 'ism', 'malliavin_lambda0')
METRICS = ('airm_squared_frechet_to_target', 'frobenius_frechet_to_target',
           'frobenius_arithmetic_to_target', 'sample_airm_to_target_mean',
           'energy_airm_diagnostic')


def summarize(output, manifest):
    import numpy as np
    from riemannian_score_sde.spd_validation_metrics import distribution_metrics
    from riemannian_score_sde.taxi_data import select_splits
    result = dict(complete=True, runs={}, across_training_seeds={},
                  note='Final test evaluation, not hyperparameter selection. Unconverged Frechet means are excluded and counted. Rejection conditions on validity. Pooled log-Euclidean MMD/NN are marginal diagnostics, not conditional accuracy.')
    with np.load(manifest['dataset'], allow_pickle=False) as z:
        splits = select_splits({s:z[s+'_indices'] for s in ('train','val','test')}, 'published_train')
        reference = z['covariances'][manifest['dataset_rows']]
        train = z['covariances'][splits['train']]
    for seed in range(3):
        for method in METHODS:
            name = f'{method}_seed{seed}'
            folder = output/f'seed{seed}'/method
            records, arrays = [], []
            for index in manifest['test_indices']:
                path = folder/f'test_{index:04d}.json'
                if not path.exists():
                    continue
                report = json.loads(path.read_text())
                if report.get('sample_sha256'):
                    if digest(path.with_suffix('.npy')) != report['sample_sha256']:
                        raise ValueError('Samples changed: '+str(path))
                records.append(report)
                if report['sampling']['complete'] and report.get('metrics') is not None:
                    arrays.append(np.load(path.with_suffix('.npy'), allow_pickle=False))
            valid = [r for r in records if r['sampling']['complete'] and r.get('metrics') is not None]
            converged = sum(r['metrics']['airm_squared_frechet_to_target'] is not None for r in valid)
            done = len(valid) == len(manifest['test_indices'])
            attempted = sum(r['sampling']['attempted'] for r in records)
            rejected = sum(r['sampling']['rejected'] for r in records)
            info = dict(conditions_recorded=len(records), conditions_evaluated=len(valid),
                        frechet_converged=converged, complete=done and converged == len(reference),
                        attempted=attempted, rejected=rejected,
                        rejection_rate=rejected/attempted if attempted else None)
            result['complete'] &= info['complete']
            for metric in METRICS:
                values = [r['metrics'][metric] for r in valid if r['metrics'][metric] is not None]
                info[metric] = dict(n=len(values), mean=float(np.mean(values)) if values else None)
            if done:
                samples = np.concatenate(arrays)
                # Bound quadratic MMD cost, using the same fixed sample positions for all runs.
                positions = np.random.RandomState(2026).permutation(len(samples))[:2000]
                pooled = distribution_metrics(samples[positions], reference, train)
                pooled['bandwidth_source'] = 'all selected test targets; common across methods and seeds'
                for old, new in [('generated_to_validation_nn','generated_to_test_nn'),
                                 ('validation_to_generated_nn','test_to_generated_nn'),
                                 ('validation_to_train_nn','test_to_train_nn')]:
                    pooled[new] = pooled.pop(old)
                info['pooled_distribution'] = pooled
            result['runs'][name] = info
    for method in METHODS:
        aggregate = {}
        for metric in METRICS:
            entries = [result['runs'][f'{method}_seed{s}'][metric] for s in range(3)]
            # Never report a three-seed comparison of partial, differing test sets.
            values = [e['mean'] for e in entries if e['n'] == len(reference)]
            aggregate[metric] = dict(seeds_complete=len(values),
                mean=float(np.mean(values)) if len(values)==3 else None,
                std=float(np.std(values, ddof=1)) if len(values)==3 else None)
        for metric in ('mmd2_biased', 'generated_to_test_nn', 'test_to_generated_nn', 'generated_to_train_nn'):
            values = []
            for seed in range(3):
                pooled = result['runs'][f'{method}_seed{seed}'].get('pooled_distribution', {})
                value = pooled.get(metric)
                if isinstance(value, dict):
                    value = value.get('mean')
                if value is not None:
                    values.append(value)
            aggregate[metric] = dict(seeds_complete=len(values),
                mean=float(np.mean(values)) if len(values)==3 else None,
                std=float(np.std(values, ddof=1)) if len(values)==3 else None)
        result['across_training_seeds'][method] = aggregate
    write_json(output/'summary.json', result)
    print('Summary:', output/'summary.json', 'complete=', result['complete'], flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path)
    p.add_argument('--dataset', type=Path, default=ROOT/'data/spd_taxi/nyc_taxi.npz')
    p.add_argument('--output', type=Path, help='Matching existing output resumes')
    p.add_argument('--timeout', type=int, default=86400, help='Seconds per method/seed')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--worker', choices=METHODS, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        manifest = json.loads((args.output/'manifest.json').read_text())
        if digest(Path(manifest['dataset'])) != manifest['dataset_sha256']:
            raise ValueError('Dataset changed')
        worker(SimpleNamespace(worker=args.worker, output=args.output), manifest)
        return
    if args.summarize_only:
        if not args.output:
            p.error('--summarize-only requires --output')
        manifest = json.loads((args.output/'manifest.json').read_text())
        if digest(Path(manifest['dataset'])) != manifest['dataset_sha256']:
            raise ValueError('Dataset changed')
        summarize(args.output, manifest)
        return
    if not args.run_root or args.timeout < 1:
        p.error('--run-root required; timeout must be positive')
    import numpy as np
    from riemannian_score_sde.taxi_data import validate, select_splits
    root, dataset = args.run_root.resolve(), args.dataset.resolve()
    with np.load(dataset, allow_pickle=False) as z:
        splits = {s:z[s+'_indices'] for s in ('train','val','test')}
        validate(z['covariances'], z['contexts'], splits)
        final = select_splits(splits, 'published_train')
        if {s:len(v) for s,v in final.items()} != dict(train=7600,val=0,test=1159):
            raise ValueError('Unexpected final Taxi partition')
        rows = final['test'].tolist()
    training = json.loads((root/'manifest.json').read_text())
    if training['dataset_sha256'] != digest(dataset):
        raise ValueError('Dataset differs from final training manifest')
    manifest = dict(version=1, run_root=str(root), dataset=str(dataset), dataset_sha256=digest(dataset),
                    test_indices=list(range(len(rows))), dataset_rows=rows,
                    evaluation_split='test', split_protocol='published_train',
                    generation_seed=123, samples=20, steps=64,
                    checkpoints={}, saved_config_sha256={})
    for seed in range(3):
        for method in METHODS:
            name = f'{method}_seed{seed}'
            manifest['checkpoints'][name] = checkpoint_hashes(root/name/'ckpt')
            manifest['saved_config_sha256'][name] = digest(root/name/'.hydra/config.yaml')
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='test_evaluation_',dir=root))
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        if json.loads((output/'manifest.json').read_text()) != manifest:
            raise ValueError('Output manifest differs; use a fresh output')
    elif any(output.iterdir()):
        raise ValueError('Output is not empty')
    else:
        write_json(output/'manifest.json', manifest)
    print('Final test output:',output,flush=True)
    env = dict(os.environ, GEOMSTATS_BACKEND='jax', JAX_ENABLE_X64='true',
               JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false', HYDRA_FULL_ERROR='1')
    env.pop('JAX_PLATFORM_NAME', None)
    env['PYTHONPATH'] = str(ROOT/'geomstats')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    failures = []
    for seed in range(3):
        dest = output/f'seed{seed}'
        dest.mkdir(exist_ok=True)
        child = dict(manifest, training_seed=seed, val_indices=manifest['test_indices'],
                     checkpoints={m:manifest['checkpoints'][f'{m}_seed{seed}'] for m in METHODS},
                     saved_config_sha256={m:manifest['saved_config_sha256'][f'{m}_seed{seed}'] for m in METHODS})
        write_json(dest/'manifest.json',child)
        for method in METHODS:
            name = f'{method}_seed{seed}'
            print('EVALUATE:',name,flush=True)
            try:
                with (output/(name+'.log')).open('a') as log:
                    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',method,
                                    '--output',str(dest)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                   check=True,timeout=args.timeout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                failures.append(name)
                print('FAILED:',name,'inspect its log',flush=True)
            if checkpoint_hashes(root/name/'ckpt') != manifest['checkpoints'][name]:
                raise RuntimeError('Checkpoint changed: '+name)
    report = summarize(output, manifest)
    if failures or not report['complete']:
        raise SystemExit('Incomplete evaluation; inspect logs and summary. Retain all artifacts.')


if __name__ == '__main__':
    main()
