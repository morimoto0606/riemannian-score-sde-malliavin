#!/usr/bin/env python3
"""Prepare 72 fresh 100k S2 fits; run only an explicitly selected index."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

DATASETS = ('earthquake', 'flood', 'volcano')
METHODS = ('malliavin', 'varadhan', 'ism', 'spectrum')


def configurations():
    for dataset in DATASETS:
        for method in METHODS:
            for weight in (0, 5):
                for seed in range(3):
                    yield dataset, method, weight, seed


def configure(raw, dataset, method, weight, seed, dest, smoke=False):
    cfg = copy.deepcopy(raw)
    if cfg['dataset']['name'] != dataset or int(cfg['seed']) != seed:
        raise ValueError('Source dataset or seed mismatch')
    expected = {'malliavin': 'Malliavin', 'varadhan': 'VaradhanTeacher',
                'spectrum': 'SpectrumTeacher'}
    if method == 'ism':
        if not cfg['loss']['_target_'].endswith('.get_ism_loss_fn'):
            raise ValueError('Source is not ISM')
    elif expected[method] not in cfg['teacher']['_target_']:
        raise ValueError('Source teacher mismatch')
    cfg.update(mode='train', resume=False, steps=1 if smoke else 100000,
               warmup_steps=1000, train_plot=False, test_plot=False,
               test_val=False, test_test=False)
    if smoke:
        cfg['train_val'] = False
    cfg['scheduler'] = {
        '_target_': 'optax.join_schedules',
        'schedules': [
            {'_target_': 'optax.linear_schedule', 'init_value': 0.,
             'end_value': 1., 'transition_steps': 1000},
            {'_target_': 'optax.cosine_decay_schedule', 'init_value': 1.,
             'decay_steps': 99000, 'alpha': 0.}],
        'boundaries': [1000]}
    cfg['loss'].update(time_weighting=bool(weight), time_weight_lambda=float(weight))
    if method == 'spectrum':
        cfg['enable_x64'] = True
    for key, suffix in [('ckpt_dir', 'ckpt'), ('generated_samples_path', 'generated_samples.npy'),
                        ('logs_dir', 'logs'), ('logdir', '')]:
        cfg[key] = str(dest / suffix)
    cfg['logger'] = {'csv': {'_target_': 'score_sde.utils.loggers_pl.CSVLogger',
        'save_dir': str(dest/'logs'), 'name': '', 'flush_logs_every_n_steps': 1000}}
    if 'generation' in cfg:
        cfg['generation']['enabled'] = False
    return cfg


def prepare(repo, source, output, check_only=False, smoke=False):
    from omegaconf import OmegaConf
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    records = []
    for index, (dataset, method, weight, seed) in enumerate(configurations()):
        source_method = 'malliavin_lambda0' if method == 'malliavin' else method
        path = source / f'{dataset}_{source_method}_seed{seed}' / 'input_config/config.yaml'
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        name = f'{dataset}_{method}_lambda{weight}_seed{seed}'
        cfg = configure(raw, dataset, method, weight, seed, output/name, smoke)
        record = dict(name=name, method=method, seed=seed, steps=cfg['steps'],
                      time_weight_lambda=weight, like_w=cfg['loss']['like_w'],
                      x64_required=method == 'spectrum', source=str(path),
                      source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        records.append((record, OmegaConf.create(cfg)))
        print(index, name, 'steps=', cfg['steps'], 'lambda=', weight,
              'like_w=', cfg['loss']['like_w'], 'x64=', method == 'spectrum')
    if check_only:
        return
    output.mkdir(parents=True)
    for record, cfg in records:
        dest = output/record['name']
        (dest/'input_config').mkdir(parents=True)
        path = dest/'input_config/config.yaml'
        OmegaConf.save(cfg, path)
        record['config_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (dest/'source_config.yaml').write_bytes(Path(record['source']).read_bytes())
    (output/'manifest.json').write_text(json.dumps(dict(
        repo=str(repo), smoke=smoke, runs=[r for r, _ in records],
        note='Fresh 100k schedule: 1000 warmup + 99000 cosine. Native like_w preserved; Spectrum uses x64. Smoke only truncates updates.'), indent=2))
    print('Prepared:', output)


def run_one(output, index):
    manifest = json.loads((output/'manifest.json').read_text())
    assert len(manifest['runs']) == 72 and 0 <= index < len(manifest['runs'])
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
    parser.add_argument('--smoke', action='store_true', help='Prepare separate one-update configs')
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.index is not None:
        if args.check_only or args.smoke:
            parser.error('Run mode cannot use --check-only or --smoke')
        run_one(args.output.resolve(), args.index)
    else:
        source = args.source if args.source.is_absolute() else repo/args.source
        prepare(repo, source.resolve(), args.output.resolve(), args.check_only, args.smoke)


if __name__ == '__main__':
    main()
