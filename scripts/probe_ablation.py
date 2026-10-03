#!/usr/bin/env python3
"""Preserve paper runs and compare R=1,4,8 at lambda_time=0 on MIMS.

prepare reads saved training hparams, freezes source files, and writes fresh
configs. batch trains R4/R8 serially and evaluates every run on CPU. No original
checkpoint, config, sample, or metric file is modified. Python 3.9 compatible.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ADMIN = {'ckpt_dir', 'logs_dir', 'logdir', 'generated_samples_path', 'logger'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(path):
    files = sorted(p for p in Path(path).rglob('*') if p.is_file())
    if not files:
        raise ValueError('No checkpoint files: ' + str(path))
    return {str(p.relative_to(path)): digest(p) for p in files}


def write_json(path, value, exclusive=True):
    with Path(path).open('x' if exclusive else 'w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def load_config(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def config_difference(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = {}
        for key in sorted(set(a) | set(b)):
            name = prefix + '.' + key if prefix else key
            if key not in a or key not in b:
                result[name] = [a.get(key), b.get(key)]
            else:
                result.update(config_difference(a[key], b[key], name))
        return result
    return {} if a == b else {prefix: [a, b]}


def validate_source(cfg, suite, seed):
    t = cfg['teacher']
    required = (cfg['mode'] == 'train', not cfg['resume'], cfg['steps'] == 100000,
                cfg['seed'] == seed, t['_target_'].endswith('.MalliavinTeacher'),
                t['divergence_mode'] == 'hutchinson', t['hutchinson_probes'] == 1,
                t['hutchinson_noise'] == 'rademacher', not t.get('rb_enabled', False),
                t['covariance_regularization'] == 1e-6, cfg['flow']['N'] == 100,
                cfg['batch_size'] == 512, not cfg['loss']['like_w'],
                float(cfg['loss'].get('time_weight_lambda', 0)) == 0)
    if not all(required):
        raise ValueError('Source does not match the paper R1/lambda0/100k settings')
    if suite == 'so3':
        if cfg['dataset']['seed'] != 0 or cfg['manifold'].get('n') != 3:
            raise ValueError('SO3 target distribution mismatch')
    elif cfg['dataset']['name'] != 'earthquake':
        raise ValueError('Expected Earthquake source')


def redirect(cfg, dest, checkpoint):
    cfg['ckpt_dir'] = str(checkpoint)
    cfg['logs_dir'] = str(dest / 'logs')
    cfg['logdir'] = str(dest)
    cfg['generated_samples_path'] = str(dest / 'generated_samples.npy')
    for logger in cfg['logger'].values():
        if not logger['_target_'].endswith('.CSVLogger'):
            raise ValueError('Expected the saved CSV logger')
        logger['save_dir'] = str(dest / 'logs')


def training_config(raw, probes, dest):
    cfg = copy.deepcopy(raw)
    cfg['teacher']['hutchinson_probes'] = probes
    redirect(cfg, dest, dest / 'ckpt')
    difference = config_difference(raw, cfg)
    scientific = {k:v for k,v in difference.items() if k.split('.')[0] not in ADMIN}
    if scientific != {'teacher.hutchinson_probes': [1, probes]}:
        raise ValueError('Unexpected scientific configuration difference: ' + str(scientific))
    # Deliberately retain train_val/train_plot/test_plot: dataset consumption can
    # affect training even if a flag appears to be only for visualization.
    return cfg, difference


def environment(source, cpu):
    env = os.environ.copy()
    env.update(GEOMSTATS_BACKEND='jax', JAX_ENABLE_X64='false',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', MPLBACKEND='Agg',
               HYDRA_FULL_ERROR='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=str(source / 'geomstats') + os.pathsep + str(source))
    if cpu:
        env.update(JAX_PLATFORM_NAME='cpu', JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
    else:
        env.pop('JAX_PLATFORMS', None)
        env['JAX_PLATFORM_NAME'] = 'gpu'  # fail instead of silently training on CPU
    return env


def snapshot(repo, output):
    dest = output / 'source'
    dest.mkdir()
    for name in ('main.py', 'run.py', 'config', 'score_sde', 'riemannian_score_sde', 'scripts'):
        src = repo / name
        if src.is_dir():
            shutil.copytree(src, dest / name, ignore=shutil.ignore_patterns('__pycache__', '.git'))
        else:
            shutil.copy2(src, dest / name)
    geo = repo / 'geomstats/geomstats'
    if not (geo / '__init__.py').exists():
        raise FileNotFoundError('Initialize the existing Geomstats submodule first')
    shutil.copytree(geo, dest / 'geomstats/geomstats', ignore=shutil.ignore_patterns('__pycache__'))
    (dest / 'docs').mkdir()
    shutil.copy2(repo / 'docs/probe_ablation_proposals.txt', dest / 'docs/probe_ablation_proposals.txt')
    (dest / 'data').symlink_to(repo / 'data', target_is_directory=True)
    files = {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*')
             if p.is_file() and 'data' != p.relative_to(dest).parts[0]}
    write_json(output / 'source_fingerprints.json', files)
    return dest


def check_source(output):
    expected = json.loads((output / 'source_fingerprints.json').read_text())
    for name, value in expected.items():
        if digest(output / 'source' / name) != value:
            raise ValueError('Frozen source changed: ' + name)


def source_record(repo, suite, seed):
    if suite == 'so3':
        paper = repo / 'results/so3_complete_100k_v1'
        name = 'so3_malliavin_lambda0_seed' + str(seed)
        record = next(r for r in json.loads((paper / 'manifest.json').read_text())['runs']
                      if r['name'] == name)
        hparams, ckpt = Path(record['source']), Path(record['checkpoint'])
        if digest(hparams) != record['source_sha256']:
            raise ValueError('Original SO3 hparams differ from paper manifest')
        evaluation = paper / 'evaluation' / name
        reference = evaluation / 'evaluation/reference_samples.npy'
    else:
        paper = repo / 'results/earth_100k_both_v1'
        name = 'earthquake_malliavin_lambda0_seed' + str(seed)
        record = next(r for r in json.loads((paper / 'manifest.json').read_text())['runs']
                      if r['name'] == name)
        hparams = paper / name / 'logs/version_0/hparams.yaml'
        original_input = paper / name / 'input_config/config.yaml'
        if digest(original_input) != record['config_sha256']:
            raise ValueError('Original Earth input config differs from manifest')
        if load_config(hparams) != load_config(original_input):
            raise ValueError('Earth training hparams and prepared config differ')
        ckpt = paper / name / 'ckpt'
        evaluation = repo / 'results/earth_100k_evaluation_s7sCvS' / name
        reference = repo / 'data/quakes_all.csv'
    if not (evaluation / 'COMPLETE').exists():
        raise ValueError('Paper evaluation is incomplete: ' + str(evaluation))
    for path in (reference, evaluation / 'generated_samples.npy'):
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = load_config(hparams)
    validate_source(cfg, suite, seed)
    if suite == 'earthquake' and (Path(cfg['dataset']['data_dir']) / 'quakes_all.csv').resolve() != reference.resolve():
        raise ValueError('Earth training and evaluation data locations differ')
    evaluation_config = evaluation / '.hydra/config.yaml'
    saved_eval = load_config(evaluation_config)
    expected_generation_seed = 0 if suite == 'so3' else seed
    if (saved_eval['seed'] != expected_generation_seed or saved_eval['flow']['N'] != 100
            or saved_eval['mode'] != 'test' or saved_eval['batch_size'] != 512
            or saved_eval['eval_batch_size'] != 512 or not saved_eval['test_plot']
            or saved_eval['test_val'] or saved_eval['test_test']):
        raise ValueError('Unexpected saved generation configuration')
    for key in ('teacher', 'dataset', 'manifold', 'loss', 'architecture', 'flow', 'beta_schedule', 'eps', 'ema_rate'):
        if cfg[key] != saved_eval[key]:
            raise ValueError('Training/evaluation mismatch in ' + key)
    return cfg, dict(paper_name=name, original_training_config=str(hparams),
        original_training_config_sha256=digest(hparams), original_checkpoint=str(ckpt),
        original_checkpoint_sha256=fingerprint(ckpt), original_evaluation=str(evaluation),
        original_generated=str(evaluation / 'generated_samples.npy'),
        original_generated_sha256=digest(evaluation / 'generated_samples.npy'),
        original_evaluation_config=str(evaluation_config),
        original_evaluation_config_sha256=digest(evaluation_config),
        generation_seed=expected_generation_seed,
        reference=str(reference), reference_sha256=digest(reference),
        training_seed=seed, suite=suite)


def prepare(args):
    repo, output = args.repo.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh output directory (do not use mktemp -d)')
    if not (repo / 'geomstats/geomstats/__init__.py').is_file():
        raise FileNotFoundError('The existing Geomstats submodule is missing')
    records = []
    seeds = args.seeds if args.seeds is not None else ([0, 1, 2] if args.suite == 'so3' else [0])
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in (0, 1, 2) for s in seeds):
        raise ValueError('Seeds must be distinct members of 0,1,2')
    for seed in seeds:
        raw, evidence = source_record(repo, args.suite, seed)
        for probes in (1, 4, 8):
            name = '{}_R{}_lambda0_seed{}'.format(args.suite, probes, seed)
            dest = output / 'training' / name
            cfg, difference = (copy.deepcopy(raw), {}) if probes == 1 else training_config(raw, probes, dest)
            records.append(dict(evidence, name=name, probes=probes, reuse=probes == 1,
                                config=cfg, config_difference=difference,
                                checkpoint=evidence['original_checkpoint'] if probes == 1 else str(dest / 'ckpt')))
    if args.suite == 'so3' and len({r['reference_sha256'] for r in records}) != 1:
        raise ValueError('Paper reference samples differ across seeds')
    # New outputs must not be nested inside a paper run.
    for r in records:
        for protected in (Path(r['original_training_config']).parents[2], Path(r['original_evaluation'])):
            if output == protected or protected in output.parents:
                raise ValueError('Choose an output outside existing paper runs')
    for r in records:
        print(r['name'], 'REUSE' if r['reuse'] else 'NEW 100k', 'R=', r['probes'])
    if args.check_only:
        return
    output.mkdir(parents=True, exist_ok=False)
    source = snapshot(repo, output)
    for r in records:
        dest = output / 'configs' / r['name']
        dest.mkdir(parents=True)
        cfg = r.pop('config')
        write_json(dest / 'config.yaml', cfg)  # JSON is valid YAML for Hydra
        r.update(config_path=str(dest / 'config.yaml'), config_sha256=digest(dest / 'config.yaml'))
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    geomstats_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo/'geomstats', text=True).strip()
    write_json(output / 'manifest.json', dict(repo=str(repo), source=str(source), git_commit=revision,
        geomstats_commit=geomstats_revision,
        suite=args.suite, seeds=seeds, lambda_time=0, steps=100000, runs=records,
        generation_seed='0 for SO3; matched training seed for Earthquake',
        reference_seed=10000 if args.suite == 'so3' else None,
        sample_count=16384, metric_subsample=2000, metric_seed=0,
        training_order='sequential: seed0 R4,R8; seed1 R4,R8; seed2 R4,R8',
        precision='JAX float32; SO3 chordal MMD sums float64'))
    with (output / 'environment.txt').open('x') as handle:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=handle, check=True)
    summarize(output)
    write_json(output / 'PREPARED', {'runs': len(records), 'complete': True})
    print('Prepared:', output)


def protect_original(r):
    checks = [(r['original_training_config'], r['original_training_config_sha256']),
              (r['original_evaluation_config'], r['original_evaluation_config_sha256']),
              (r['original_generated'], r['original_generated_sha256']),
              (r['reference'], r['reference_sha256'])]
    if any(digest(p) != h for p, h in checks):
        raise ValueError('An original paper input changed')
    if fingerprint(Path(r['original_checkpoint'])) != r['original_checkpoint_sha256']:
        raise ValueError('Original paper checkpoint changed')


def verify_checkpoint(path):
    import jax
    import numpy as np
    from score_sde.utils import restore
    state = restore(str(path))
    if int(state.step) != 100000:
        raise ValueError('Checkpoint step {}, expected 100000'.format(int(state.step)))
    for field in ('params', 'params_ema', 'model_state', 'opt_state'):
        for a in jax.tree_util.tree_leaves(getattr(state, field)):
            a = np.asarray(a)
            if np.issubdtype(a.dtype, np.number) and not np.isfinite(a).all():
                raise ValueError('Nonfinite checkpoint ' + field)
    print(json.dumps({'step': 100000, 'finite': True}))


def check_checkpoint_subprocess(source, checkpoint):
    subprocess.run([sys.executable, str(source / 'scripts/probe_ablation.py'), 'checkpoint',
                    '--output', str(checkpoint)], cwd=source, env=environment(source, True), check=True)


def execute(cfg, dest, source, cpu):
    conf = dest / 'input_config'
    conf.mkdir()
    write_json(conf / 'config.yaml', cfg)
    cmd = [sys.executable, '-u', 'main.py', '--config-path', str(conf),
           '--config-name', 'config', 'hydra.run.dir=' + str(dest)]
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    observations, peak, gpu_errors = 0, None, []
    with (dest / 'run.log').open('x') as log:
        process = subprocess.Popen(cmd, cwd=source, env=environment(source, cpu),
                                   stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if not cpu:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                    if result.returncode:
                        raise RuntimeError(result.stderr.strip())
                    values = [float(row.split(',')[1]) for row in result.stdout.splitlines()
                              if row.split(',')[0].strip() == str(process.pid)]
                    if values:
                        observations += 1
                        peak = max(peak or 0, sum(values))
                except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError) as exc:
                    if not gpu_errors: gpu_errors.append(str(exc))
            time.sleep(2)
    timing = dict(returncode=process.returncode, process_wall_seconds=time.monotonic()-start,
                  started_at_utc=started_at, ended_at_utc=datetime.now(timezone.utc).isoformat(),
                  sampled_peak_gpu_memory_mib=peak, gpu_memory_observations=observations,
                  gpu_memory_note='2-second nvidia-smi sampling; process device allocation, not exact peak live tensors',
                  gpu_monitor_errors=gpu_errors)
    write_json(dest / 'process.json', timing)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def log_metrics(run):
    metrics = {}
    for p in sorted(Path(run).glob('logs/version_*/metrics.csv')):
        with p.open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            try: step = int(float(row.get('step', '')))
            except (ValueError, TypeError): continue
            for key in ('train/loss', 'train/total_time', 'train/wall_time',
                        'train/mean_time_per_it', 'train/time_per_it'):
                try: value = float(row.get(key, ''))
                except (ValueError, TypeError): continue
                if math.isfinite(value): metrics.setdefault(key, []).append((step, value, str(p)))
    result = {}
    for key, values in metrics.items():
        last_step = max(x[0] for x in values)
        last = [x for x in values if x[0] == last_step]
        distinct = {x[1] for x in last}
        result[key] = dict(step=last_step, value=last[0][1] if len(distinct)==1 else None,
                           ambiguous=len(distinct)!=1, source=[x[2] for x in last])
    if 'train/loss' in metrics:
        by_step = {}
        for step, value, _ in metrics['train/loss']: by_step.setdefault(step, set()).add(value)
        tail = sorted(by_step)[-10:]
        if all(len(by_step[s]) == 1 for s in tail):
            result['train/loss_last10_mean'] = statistics.mean(next(iter(by_step[s])) for s in tail)
            result['train/loss_last10_steps'] = tail
    return result


def evaluate_record(output, manifest, r):
    source = Path(manifest['source'])
    holder = output / 'evaluation' / r['name']
    if (holder / 'COMPLETE').exists(): return
    holder.mkdir(parents=True, exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix='attempt_', dir=holder))
    checkpoint = Path(r['checkpoint'])
    check_checkpoint_subprocess(source, checkpoint)
    before = fingerprint(checkpoint)
    if r['reuse']:
        generated = Path(r['original_generated'])
        training = Path(r['original_training_config']).parents[2]
    else:
        # Clone the actual paper generation configuration, including its seed.
        cfg = load_config(Path(r['original_evaluation_config']))
        cfg['teacher']['hutchinson_probes'] = r['probes']
        redirect(cfg, dest, checkpoint)
        try: execute(cfg, dest, source, True)
        finally:
            unchanged = before == fingerprint(checkpoint)
            write_json(dest / 'checkpoint_check.json', {'unchanged': unchanged})
            if not unchanged: raise RuntimeError('Checkpoint changed during generation')
        generated = dest / 'generated_samples.npy'
        training = output / 'training' / r['name']
    subprocess.run([sys.executable, str(source / 'scripts/probe_ablation_metrics.py'),
                    '--suite', manifest['suite'], '--generated', str(generated),
                    '--reference', r['reference'],
                    '--output', str(dest / 'metrics.json')], cwd=source,
                   env=environment(source, True), check=True)
    if before != fingerprint(checkpoint): raise RuntimeError('Checkpoint changed')
    protect_original(r)
    result = dict(name=r['name'], probes=r['probes'], seed=r['training_seed'], reuse=r['reuse'],
                  training_directory=str(training), evaluation_directory=str(dest),
                  generated_samples=str(generated), generation_seed=r['generation_seed'],
                  metrics=json.loads((dest / 'metrics.json').read_text()),
                  training_logs=log_metrics(training), checkpoint_unchanged=True)
    if (training / 'process.json').exists():
        result['training_process'] = json.loads((training / 'process.json').read_text())
    write_json(dest / 'result.json', result)
    write_json(holder / 'COMPLETE', {'result': str(dest / 'result.json')})


def run_one(output, index):
    manifest = json.loads((output / 'manifest.json').read_text())
    if not 0 <= index < len(manifest['runs']):
        raise ValueError('Run index out of range')
    r = manifest['runs'][index]
    check_source(output)
    protect_original(r)
    if digest(r['config_path']) != r['config_sha256']:
        raise ValueError('Prepared configuration changed')
    source = Path(manifest['source'])
    if not r['reuse']:
        dest = output / 'training' / r['name']
        if not (dest / 'COMPLETE').exists():
            dest.mkdir(parents=True, exist_ok=False)
            print('TRAIN:', r['name'], flush=True)
            execute(json.loads(Path(r['config_path']).read_text()), dest, source, False)
            check_checkpoint_subprocess(source, Path(r['checkpoint']))
            write_json(dest / 'COMPLETE', {'step': 100000, 'finite': True})
    print('EVALUATE:', r['name'], flush=True)
    evaluate_record(output, manifest, r)
    print('COMPLETE:', r['name'], flush=True)
    summarize(output)


def summarize(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    rows = []
    for r in manifest['runs']:
        path = output / 'evaluation' / r['name'] / 'COMPLETE'
        result = json.loads(Path(json.loads(path.read_text())['result']).read_text()) if path.exists() else None
        rows.append(dict(name=r['name'], probes=r['probes'], seed=r['training_seed'],
                         config_difference=r['config_difference'], checkpoint=r['checkpoint'],
                         config_path=r['config_path'],
                         status='complete' if result else 'pending', result=result))
    groups, paired = {}, {}
    for probes in (1, 4, 8):
        completed = [r['result'] for r in rows if r['probes']==probes and r['result']]
        aggregate = {}
        if completed:
            for key in completed[0]['metrics']['values']:
                values = [r['metrics']['values'][key] for r in completed]
                aggregate[key] = dict(n=len(values), mean=statistics.mean(values),
                                      std=statistics.stdev(values) if len(values)>1 else None)
        groups[str(probes)] = aggregate
    for probes in (4, 8):
        matches = []
        for seed in manifest['seeds']:
            baseline = next(r['result'] for r in rows if r['probes']==1 and r['seed']==seed)
            new = next(r['result'] for r in rows if r['probes']==probes and r['seed']==seed)
            if baseline and new:
                matches.append(dict(seed=seed, differences={k:new['metrics']['values'][k]-v
                    for k,v in baseline['metrics']['values'].items()}))
        paired[str(probes)] = matches
    complete = all(r['result'] for r in rows)
    summary = dict(complete=complete, runs=rows, groups=groups, paired_against_R1=paired,
                   decision='Review generation metrics and paired seeds; training loss alone is not a replacement criterion.')
    write_json(output / 'summary.json', summary, exclusive=False)
    per_seed = []
    for row in rows:
        if row['result']:
            per_seed.append(dict(name=row['name'], probes=row['probes'], seed=row['seed'],
                                 **row['result']['metrics']['values']))
    if per_seed:
        with (output / 'comparison_per_seed.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(per_seed[0]))
            writer.writeheader()
            writer.writerows(per_seed)
        with (output / 'comparison_summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['probes', 'metric', 'n', 'mean', 'std'])
            writer.writeheader()
            for probes, values in groups.items():
                for metric, v in values.items():
                    writer.writerow(dict(probes=probes, metric=metric, **v))
    proposal = ROOT / 'docs/probe_ablation_proposals.txt'
    text = proposal.read_text() if proposal.exists() else ''
    text += '\n\n実験結果（保存済み結果のみ）\nOutput: ' + str(output) + '\ncomplete: ' + str(complete) + '\n'
    for row in rows:
        text += '\n' + row['name'] + ': ' + row['status'] + '\n'
        text += 'Config: ' + row['config_path'] + '\nCheckpoint: ' + row['checkpoint'] + '\n'
        text += 'Differences: ' + json.dumps(row['config_difference'], ensure_ascii=False) + '\n'
        r = row['result']
        if r:
            brief = {k:v for k,v in r.items() if k != 'metrics'}
            brief['generation_metrics'] = r['metrics']['values']
            text += json.dumps(brief, ensure_ascii=False, indent=2) + '\n'
    text += '\nseed集計（nは実際に完了したseed数）\n' + json.dumps(groups, indent=2)
    text += '\n各seedのR1との差（負は当該距離・MMDの低下）\n' + json.dumps(paired, indent=2)
    text += '\n\n判断：' + ('全条件の数値を上に記録。複数seedでのMMD²と双方向NN、計算費用を合わせて差し替えを判断する。' if complete else
                            '未完了の条件があるため、R4/R8の改善・R1の効率・論文差し替えはまだ判断しない。')
    text += '\n既存論文のR=1は実runを正しく記述する。性能上の最適性はこの比較の結果と区別する。\n'
    text += '時間：process_wall_secondsは起動・コンパイル・検証・描画を含む経過時間。学習logの時間も別に記録。'
    text += '既存R1は過去の実行なのでGPU共有状況等が異なり、厳密な速度比には使わない。\n'
    text += 'Supplement表：生成指標と費用の比較が揃えば追加する価値がある。R8への差し替えは、複数seedの主要生成指標の改善を確認してから判断する。\n'
    (output / 'report.txt').write_text(text)
    print('Report:', output / 'report.txt', 'complete=', complete, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'batch', 'run', 'summarize', 'checkpoint'])
    p.add_argument('--repo', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suite', choices=['so3', 'earthquake'], default='so3')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--index', type=int)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    if args.action == 'prepare': prepare(args)
    elif args.action == 'checkpoint': verify_checkpoint(out)
    elif args.action == 'summarize': summarize(out)
    elif args.action == 'run':
        if args.index is None or args.index < 0: p.error('A nonnegative --index is required')
        run_one(out, args.index)
    else:
        # One job, one GPU training process at a time. Do not automatically launch
        # optional Earthquake work while the same GPU is occupied by SO3.
        if not (out / 'PREPARED').is_file():
            raise ValueError('Preparation did not finish')
        with (out / 'BATCH_STARTED').open('x') as f:
            f.write(os.environ.get('PBS_JOBID', 'interactive') + '\n')
        manifest = json.loads((out / 'manifest.json').read_text())
        order = sorted(range(len(manifest['runs'])), key=lambda i: (not manifest['runs'][i]['reuse'], i))
        for i in order: run_one(out, i)
        print('ALL PROBE ABLATION RUNS COMPLETE', flush=True)


if __name__ == '__main__': main()
