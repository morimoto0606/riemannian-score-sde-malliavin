#!/usr/bin/env python3
"""Prepare or execute fresh lambda=5 baselines (Earthquake only by default)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def prepare(repo, source, output, check_only, datasets):
    from omegaconf import OmegaConf
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    records = []
    for dataset, steps in [('earthquake', 600000), ('flood', 300000), ('volcano', 600000)]:
        if dataset not in datasets:
            continue
        for method in ['varadhan', 'ism', 'spectrum']:
            for seed in range(3):
                path = source / f'{dataset}_{method}_seed{seed}' / 'input_config/config.yaml'
                cfg = OmegaConf.load(path)
                assert int(cfg.seed) == seed and int(cfg.steps) == steps, str(path)
                assert str(cfg.dataset.name) == dataset, str(path)
                assert not cfg.loss.time_weighting and float(cfg.loss.time_weight_lambda) == 0, str(path)
                if method == 'ism':
                    assert cfg.loss._target_.endswith('.get_ism_loss_fn'), str(path)
                else:
                    target = 'VaradhanTeacher' if method == 'varadhan' else 'SpectrumTeacher'
                    assert cfg.teacher._target_.endswith('.' + target), str(path)
                name = f'{dataset}_{method}_lambda5_seed{seed}'
                dest = output / name
                cfg.loss.time_weighting = True
                cfg.loss.time_weight_lambda = 5.0
                cfg.mode = 'train'
                cfg.resume = False
                cfg.ckpt_dir = str(dest / 'ckpt')
                cfg.generated_samples_path = str(dest / 'generated_samples.npy')
                cfg.logs_dir = str(dest / 'logs')
                cfg.logdir = str(dest)
                cfg.logger = {'csv': {'_target_': 'score_sde.utils.loggers_pl.CSVLogger',
                    'save_dir': str(dest/'logs'), 'name': '', 'flush_logs_every_n_steps': 1000}}
                if method == 'spectrum':
                    cfg.enable_x64 = True
                record = {'name': name, 'method': method, 'seed': seed, 'steps': steps,
                    'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'like_w': bool(cfg.loss.like_w), 'x64_required': method == 'spectrum'}
                records.append((record, cfg))
                print(len(records)-1, name, 'steps=', steps, 'lambda=5', 'like_w=', cfg.loss.like_w)
    if check_only:
        return
    output.mkdir(parents=True)
    for record, cfg in records:
        dest = output / record['name']
        (dest / 'input_config').mkdir(parents=True)
        path = dest / 'input_config/config.yaml'
        OmegaConf.save(cfg, path)
        record['config_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (dest/'source_config.yaml').write_bytes(Path(record['source']).read_bytes())
    (output/'manifest.json').write_text(json.dumps({'repo': str(repo), 'runs': [r for r,c in records],
        'note': 'Fresh training; preserve source settings except time weighting, run paths, resume/mode, and required Spectrum x64.'}, indent=2))
    print('Prepared:', output)


def run_one(output, index):
    manifest = json.loads((output/'manifest.json').read_text())
    assert len(manifest['runs']) in (9, 18, 27) and 0 <= index < len(manifest['runs'])
    record = manifest['runs'][index]
    dest = output / record['name']
    cfg = dest/'input_config/config.yaml'
    assert hashlib.sha256(cfg.read_bytes()).hexdigest() == record['config_sha256']
    if (dest/'ckpt').exists():
        raise ValueError('Checkpoint already exists; refusing to overwrite')
    with (dest/'started.json').open('x') as f:
        json.dump({'job_id': os.environ.get('PBS_JOBID'), 'index': index}, f)
    env = os.environ.copy()
    if record['x64_required']:
        env['JAX_ENABLE_X64'] = 'true'
    with (dest/'launcher.log').open('x') as log:
        result = subprocess.run([sys.executable, '-u', 'main.py', '--config-path', str(cfg.parent),
            '--config-name', 'config', 'hydra.run.dir='+str(dest)], cwd=manifest['repo'],
            env=env, stdout=log, stderr=subprocess.STDOUT)
    (dest/'process_exit.json').write_text(json.dumps({'returncode':result.returncode}))
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source', type=Path, default=Path('results/earth_long_both_v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=['earthquake', 'flood', 'volcano'],
                        default=['earthquake'])
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if args.index is not None:
        if args.check_only:
            parser.error('--check-only cannot be combined with --index')
        run_one(output, args.index)
    else:
        source = args.source if args.source.is_absolute() else repo/args.source
        prepare(repo, source.resolve(), output, args.check_only, args.datasets)


if __name__ == '__main__':
    main()
