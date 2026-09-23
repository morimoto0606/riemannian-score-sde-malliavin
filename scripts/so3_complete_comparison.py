#!/usr/bin/env python3
"""Prepare missing SO3 fits, evaluate preserved checkpoints, summarize 18 runs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


ADMIN_FIELDS = {'now', 'resume', 'mode', 'ckpt_dir', 'generated_samples_path',
                'logs_dir', 'logdir', 'logger', 'hydra'}


def select_training_config(run, load):
    training = [(p, load(p)) for p in sorted(run.glob('logs/**/hparams.yaml'))]
    training = [(p, c) for p, c in training if c.get('mode') == 'train']
    if not training:
        raise ValueError(f'No training hparams in {run}; never use overwritten test config')
    baseline = {k:v for k,v in training[0][1].items() if k not in ADMIN_FIELDS}
    for path, cfg in training[1:]:
        if {k:v for k,v in cfg.items() if k not in ADMIN_FIELDS} != baseline:
            raise ValueError(f'Conflicting training settings: {training[0][0]} and {path}')
    return training[0]


def conditions():
    return [(m, w, s) for m in ('malliavin', 'varadhan', 'ism')
            for w in (0, 5) for s in range(3)]


def validate(cfg, method, weight, seed):
    if cfg.mode != 'train' or int(cfg.seed) != seed or int(cfg.steps) != 100000:
        raise ValueError('Expected matching mode=train, seed and 100k budget')
    if int(cfg.dataset.seed) != 0 or cfg.manifold._target_.split('.')[-1] != 'SpecialOrthogonal':
        raise ValueError('Expected target seed 0 on SO3')
    if int(cfg.manifold.n) != 3:
        raise ValueError('Expected SO3')
    actual = float(cfg.loss.get('time_weight_lambda', 0)) if cfg.loss.get('time_weighting', False) else 0.
    if actual != weight:
        raise ValueError('Source time weighting mismatch')
    if method == 'ism':
        valid = cfg.loss._target_.endswith('.get_ism_loss_fn')
    else:
        valid = cfg.teacher._target_.endswith('.' + ('MalliavinTeacher' if method == 'malliavin' else 'VaradhanTeacher'))
    if not valid:
        raise ValueError('Source objective mismatch')


def redirect(cfg, dest):
    cfg.logs_dir = str(dest/'logs')
    cfg.logdir = str(dest)
    cfg.generated_samples_path = str(dest/'generated_samples.npy')
    cfg.logger = {'csv': {'_target_': 'score_sde.utils.loggers_pl.CSVLogger',
                         'save_dir': str(dest/'logs'), 'name': '', 'flush_logs_every_n_steps': 1000}}
    cfg.train_plot = False
    cfg.test_plot = False
    cfg.test_val = False
    cfg.test_test = False
    if 'generation' in cfg:
        cfg.generation.enabled = False


def prepare(output, check_only):
    from omegaconf import OmegaConf
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    records = []
    signature = None
    for method, weight, seed in conditions():
        fresh = method != 'malliavin' and weight == 5
        old = f'so3_malliavin_lambda{weight}' if method == 'malliavin' else f'so3_{method}'
        source = ROOT/'results'/f'{old}_target0_seed{seed}'
        path, raw = select_training_config(source, lambda p: OmegaConf.to_container(OmegaConf.load(p), resolve=False))
        cfg = OmegaConf.create(raw)
        validate(cfg, method, 0 if fresh else weight, seed)
        # Resolve target construction independently of generation RNG.
        dataset = OmegaConf.to_container(cfg.dataset, resolve=True)
        dataset.pop('batch_dims', None)
        if signature is None:
            signature = dataset
        elif dataset != signature:
            raise ValueError('Target distribution differs across source configurations')
        name = f'so3_{method}_lambda{weight}_seed{seed}'
        dest = output/'training'/name
        if fresh:
            cfg.loss.time_weighting = True
            cfg.loss.time_weight_lambda = 5.
            cfg.resume = False
            cfg.ckpt_dir = str(dest/'ckpt')
            redirect(cfg, dest)
        else:
            cfg.ckpt_dir = str(source/'ckpt')
        record = dict(name=name, method=method, weight=weight, training_seed=seed,
                      new_training=fresh, checkpoint=str(cfg.ckpt_dir),
                      source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        records.append((record, cfg))
        print(len(records)-1, name, 'NEW TRAIN' if fresh else 'REUSE', record['checkpoint'])
    if check_only:
        return
    output.mkdir(parents=True)
    for record, cfg in records:
        dest=output/'configs'/record['name']
        dest.mkdir(parents=True)
        path=dest/'config.yaml'
        OmegaConf.save(cfg,path)
        record['config']=str(path)
        record['config_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps({'runs':[r for r,c in records],
        'target_signature':signature,'generation_seed':0,'reference_seed':10000},indent=2))


def records(output):
    rs=json.loads((output/'manifest.json').read_text())['runs']
    if len(rs)!=18:
        raise ValueError('Expected 18 conditions')
    return rs


def checked_config(record):
    from omegaconf import OmegaConf
    p=Path(record['config'])
    if hashlib.sha256(p.read_bytes()).hexdigest()!=record['config_sha256']:
        raise ValueError('Prepared configuration changed')
    return OmegaConf.load(p)


def verify_checkpoint(path):
    # CPU-only inspection; subprocess training/generation chooses its own backend.
    os.environ['JAX_PLATFORM_NAME']='cpu'
    os.environ['JAX_PLATFORMS']='cpu'
    import jax
    import numpy as np
    from score_sde.utils import restore
    if not path.exists():
        raise FileNotFoundError(path)
    state=restore(str(path))
    if int(state.step)!=100000:
        raise ValueError(f'{path}: step={int(state.step)}, expected 100000')
    for field in ('params','params_ema','model_state','opt_state'):
        for leaf in jax.tree_util.tree_leaves(getattr(state,field)):
            a=np.asarray(leaf)
            if np.issubdtype(a.dtype,np.number) and not np.isfinite(a).all():
                raise ValueError(f'{path}: nonfinite {field}')
    return {'step':int(state.step),'finite':True}


def fingerprint(path):
    files=sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise ValueError(f'No checkpoint files: {path}')
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def execute(cfg, dest, cpu):
    from omegaconf import OmegaConf
    (dest/'input_config').mkdir()
    OmegaConf.save(cfg,dest/'input_config/config.yaml')
    env=os.environ.copy()
    if cpu:
        env.update(JAX_PLATFORM_NAME='cpu',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
    with (dest/'run.log').open('x') as log:
        subprocess.run([sys.executable,'-u','main.py','--config-path',str(dest/'input_config'),
            '--config-name','config','hydra.run.dir='+str(dest)],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)


def train(output,index):
    rs=[r for r in records(output) if r['new_training']]
    if not 0<=index<len(rs):
        raise ValueError('Invalid training index')
    r=rs[index];cfg=checked_config(r)
    dest=output/'training'/r['name']
    if Path(r['checkpoint']).exists():
        raise FileExistsError('Refusing checkpoint overwrite')
    dest.mkdir(parents=True,exist_ok=False)
    execute(cfg,dest,False)
    report=verify_checkpoint(Path(r['checkpoint']))
    (dest/'COMPLETE').write_text(json.dumps(report))


def evaluate(output,index):
    import numpy as np
    if not 0<=index<18:
        raise ValueError('Invalid evaluation index')
    r=records(output)[index];cfg=checked_config(r)
    ckpt=Path(r['checkpoint'])
    verification=verify_checkpoint(ckpt)
    before=fingerprint(ckpt)
    dest=output/'evaluation'/r['name']
    dest.mkdir(parents=True,exist_ok=False)
    cfg.mode='test';cfg.seed=0;cfg.dataset.seed=0
    cfg.batch_size=512;cfg.eval_batch_size=512
    cfg.ckpt_dir=str(ckpt)
    redirect(cfg,dest)
    cfg.test_plot=True
    try:
        execute(cfg,dest,True)
    finally:
        unchanged=before==fingerprint(ckpt)
        (dest/'checkpoint_check.json').write_text(json.dumps(dict(verification,unchanged=unchanged)))
        if not unchanged:
            raise RuntimeError('Checkpoint changed')
    x=np.load(dest/'generated_samples.npy',allow_pickle=False)
    if x.shape not in ((16384,3,3),(16384,9)) or not np.isfinite(x).all():
        raise ValueError('Unexpected generated samples')
    with (dest/'postprocess.log').open('x') as log:
        subprocess.run([sys.executable,'scripts/evaluate_so3_generation.py','--run-dir',str(dest),
            '--reference-samples','16384','--reference-seed','10000','--metric-subsample','2000',
            '--metric-seed','0','--mmd-sigma','1.0'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    p=dest/'evaluation/metrics.json'
    report=json.loads(p.read_text())
    report['seeds']['training_seed']=r['training_seed']
    report['seeds']['generation_seed']=0
    report['checkpoint_provenance']=dict(verification,path=str(ckpt),sha256=before,
                                        training_config=r['source'])
    p.write_text(json.dumps(report,indent=2))
    # Keep the evaluator's CSV seed consistent with the corrected JSON.
    cp=dest/'evaluation/metrics.csv'
    with cp.open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        row['training_seed']=str(r['training_seed'])
    with cp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (dest/'COMPLETE').write_text('OK\n')


def summarize(output):
    rows=[]
    for r in records(output):
        dest=output/'evaluation'/r['name']
        if not (dest/'COMPLETE').exists():
            raise ValueError(f'Incomplete: {dest}')
        report=json.loads((dest/'evaluation/metrics.json').read_text())
        metrics={'rbf_mmd':report['rbf_mmd']}
        for direction,key in [('generated_to_reference','nearest_neighbor_geodesic_distance'),
                              ('reference_to_generated','reference_to_generated_nearest_neighbor')]:
            for stat in ('mean','median','max'):
                metrics[direction+'_nn_'+stat]=report[key][stat]
        for metric,value in metrics.items():
            rows.append(dict(method=r['method'],lambda_value=r['weight'],seed=r['training_seed'],metric=metric,value=value))
    groups={}
    for r in rows:
        groups.setdefault((r['method'],r['lambda_value'],r['metric']),[]).append(r['value'])
    summary=[]
    for (m,w,k),values in groups.items():
        summary.append(dict(method=m,lambda_value=w,metric=k,n=len(values),mean=statistics.mean(values),std=statistics.stdev(values)))
    for name,items in [('comparison_per_seed.csv',rows),('comparison_summary.csv',summary)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('rbf_mmd','RBF MMD estimate'),
              ('generated_to_reference_nn_mean','Generated to reference: Mean'),
              ('reference_to_generated_nn_mean','Reference to generated: Mean'),
              ('generated_to_reference_nn_max','Generated to reference: Max'),
              ('reference_to_generated_nn_max','Reference to generated: Max')]
    lookup={(r['method'],r['lambda_value'],r['metric']):r for r in summary}
    methods=('malliavin','varadhan','ism')
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,(key,title) in zip(axes.flat,panels):
        for weight,color,offset,marker in [(0,'#2463A6',-.1,'o'),(5,'#D57525',.1,'s')]:
            selected=[lookup[(m,weight,key)] for m in methods]
            ax.errorbar([i+offset for i in range(3)],[r['mean'] for r in selected],
                        yerr=[r['std'] for r in selected],fmt=marker,color=color,
                        capsize=4,label=f'lambda={weight}')
        ax.set_xticks(range(3),['Malliavin','Varadhan','ISM'])
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
        if key!='rbf_mmd':ax.set_ylabel('Matrix-log Frobenius distance')
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.suptitle('SO(3): 100k training updates, 3 seeds',fontsize=16)
    fig.text(.5,.015,'Mean across training seeds; error bars: +/-1 sample SD. Common generation seed 0. Lower is better.',ha='center')
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.savefig(output/'comparison.png',dpi=200)
    fig.savefig(output/'comparison.pdf')
    plt.close(fig)
    print('Saved:',output/'comparison_summary.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','audit','train','evaluate','summarize'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--index',type=int)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if a.action=='prepare':prepare(out,a.check_only)
    elif a.action=='audit':
        for r in records(out):
            if not r['new_training']:
                print(r['name'],verify_checkpoint(Path(r['checkpoint'])),flush=True)
    elif a.action=='summarize':summarize(out)
    else:
        if a.index is None:p.error('--index required')
        (train if a.action=='train' else evaluate)(out,a.index)

if __name__=='__main__':main()
