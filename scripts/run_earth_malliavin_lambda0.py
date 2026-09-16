#!/usr/bin/env python3
"""Replay the nine saved S2 lambda5 configurations with lambda=0 (PBS array)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('earthquake', 'flood', 'volcano')


# Only these administrative fields may differ between resumed training records.
ADMIN_FIELDS = {'now', 'resume', 'mode', 'ckpt_dir', 'generated_samples_path',
                'logs_dir', 'logdir', 'logger', 'hydra'}


def select_training_config(run, load):
    records = [(p, load(p)) for p in sorted(run.glob('logs/**/hparams.yaml'))]
    training = [(p, c) for p, c in records if c.get('mode') == 'train']
    if not training:
        p = run / '.hydra/config.yaml'
        if p.is_file():
            c = load(p)
            if c.get('mode') == 'train':
                training = [(p, c)]
    if not training:
        raise ValueError(f'No mode=train config found in {run}; test configs are never reused')
    baseline = {k: v for k, v in training[0][1].items() if k not in ADMIN_FIELDS}
    for p, c in training[1:]:
        current = {k: v for k, v in c.items() if k not in ADMIN_FIELDS}
        if current != baseline:
            raise ValueError(f'Conflicting training configurations: {training[0][0]} and {p}')
    return training[0]


def validate_training_config(cfg, dataset, seed):
    if cfg.get('mode') != 'train' or cfg.get('seed') != seed:
        raise ValueError('Expected matching training seed and mode=train')
    loss, teacher = cfg.get('loss', {}), cfg.get('teacher', {})
    if loss.get('time_weight_lambda') != 5 or loss.get('time_weighting') is not True:
        raise ValueError('Expected enabled lambda=5 in source training config')
    if teacher.get('_target_') != 'riemannian_score_sde.teachers.MalliavinTeacher':
        raise ValueError('Expected MalliavinTeacher in training config')
    target = {'earthquake': 'Earthquake', 'flood': 'Flood', 'volcano': 'VolcanicErruption'}[dataset]
    if cfg.get('dataset', {}).get('_target_') != 'riemannian_score_sde.datasets.earth.' + target:
        raise ValueError('Source dataset target does not match requested run')
    if cfg.get('steps') != 100000:
        raise ValueError('Expected 100000-step three-seed baseline, not historical 600k experiments')


def prepare(index, results):
    from omegaconf import OmegaConf
    dataset, seed = DATASETS[index // 3], index % 3
    candidates = [results / f'{dataset}_malliavin_lambda5_seed{seed}']
    if dataset == 'volcano':
        # Read-only support for historical server output names.
        candidates.append(results / f'{"volcano" + "e"}_malliavin_lambda5_seed{seed}')
    sources = [p for p in candidates if p.is_dir()]
    if len(sources) != 1:
        raise ValueError(f'Expected exactly one source run for {dataset} seed {seed}: {sources}')
    source, raw = select_training_config(sources[0], lambda p:
        OmegaConf.to_container(OmegaConf.load(p), resolve=False))
    validate_training_config(raw, dataset, seed)
    cfg = OmegaConf.create(raw)
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
            print(f'{index}: {source} -> {dest.name}; steps={cfg.steps}; '
                  f'probes={cfg.teacher.get("hutchinson_probes")}; '
                  f'like_w={cfg.loss.get("like_w")}; splits={cfg.get("splits")}')
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
        git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
