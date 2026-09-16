#!/usr/bin/env python3
"""Replay the nine saved S2 lambda5 configurations with lambda=0 (PBS array)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('earthquake', 'flood', 'volcano')


def prepare(index, results):
    from omegaconf import OmegaConf
    dataset, seed = DATASETS[index // 3], index % 3
    candidates = [results / f'{dataset}_malliavin_lambda5_seed{seed}']
    if dataset == 'volcano':
        # Read-only support for historical server output names.
        candidates.append(results / f'{"volcano" + "e"}_malliavin_lambda5_seed{seed}')
    sources = [p for p in candidates if (p / '.hydra/config.yaml').is_file()]
    if len(sources) != 1:
        raise ValueError(f'Expected exactly one saved config for {dataset} seed {seed}: {sources}')
    source = sources[0] / '.hydra/config.yaml'
    cfg = OmegaConf.load(source)
    if int(cfg.seed) != seed or float(cfg.loss.time_weight_lambda) != 5 or not cfg.loss.time_weighting:
        raise ValueError(f'Saved seed/lambda mismatch: {source}')
    teacher = str(OmegaConf.to_container(cfg.get('teacher'), resolve=False)).lower()
    if 'malliavin' not in teacher:
        raise ValueError(f'Expected a Malliavin teacher: {source}')
    dest = results / f'{dataset}_malliavin_lambda0_seed{seed}'
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {dest}')
    # Keep weighting enabled: exp(-0*t)=1. All other loss settings are retained.
    cfg.loss.time_weight_lambda = 0.0
    cfg.mode = 'train'
    cfg.resume = False
    cfg.ckpt_dir = str(dest / 'ckpt')
    cfg.generated_samples_path = str(dest / 'generated_samples.npy')
    cfg.logs_dir = str(dest / 'logs')
    cfg.logdir = str(dest)
    if 'csv' not in cfg.logger or len(cfg.logger) != 1:
        raise ValueError('Expected the original CSV-only logger')
    cfg.logger.csv.save_dir = str(dest / 'logs')
    if dataset == 'volcano':
        cfg.dataset.name = 'volcano'
        cfg.name = 'volcano'
        cfg.experiment = 'volcano'
    return source, dest, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=range(9))
    parser.add_argument('--check-all', action='store_true', help='Read-only preflight; no training')
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if args.check_all:
        for index in range(9):
            source, dest, cfg = prepare(index, args.results.resolve())
            print(f'{index}: {source.parent.parent.name} -> {dest.name}; steps={cfg.steps}')
        return
    if args.index is None:
        parser.error('Specify --index or --check-all')
    from omegaconf import OmegaConf
    source, dest, cfg = prepare(args.index, args.results.resolve())
    dest.mkdir()  # Atomic reservation: concurrent/repeated jobs cannot overwrite.
    config_dir = dest / 'input_config'
    config_dir.mkdir()
    OmegaConf.save(cfg, config_dir / 'config.yaml')
    (dest / 'lambda0_provenance.json').write_text(json.dumps(dict(
        source_config=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_config_text=source.read_text(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        changes='lambda 5 to 0; fresh training; isolated output paths; canonical volcano name',
        note='Saved hyperparameters retained; historical source-code identity is not guaranteed.'
    ), indent=2)+'\n')
    command = [sys.executable, '-u', 'main.py', '--config-path', str(config_dir),
               '--config-name', 'config', 'hydra.run.dir='+str(dest)]
    print('Training:', dest, flush=True)
    with (dest / 'launcher.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    (dest / 'training_process_completed.json').write_text(json.dumps({'returncode': 0})+'\n')


if __name__ == '__main__':
    main()
