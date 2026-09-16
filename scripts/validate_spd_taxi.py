#!/usr/bin/env python3
"""Fixed validation contexts; one model restore/JIT per method, resumable artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('varadhan', 'ism', 'malliavin_lambda0', 'malliavin_lambda5')
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_hashes(path):
    return {n:digest(path/n) for n in ('arrays.npy', 'tree.pkl')}


def write_json(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def worker(args, manifest):
    # Import only after configuring the process environment.
    import runpy
    import numpy as np
    import jax
    import jax.numpy as jnp
    from riemannian_score_sde.spd_rejection import collect_spd_samples
    from riemannian_score_sde.spd_validation_metrics import evaluate
    import riemannian_score_sde.spd_generation as generation
    method = args.worker
    root = Path(manifest['run_root'])
    base = root/(method+'_seed0')
    ckpt = base/'ckpt'
    dest = args.output/method
    dest.mkdir(exist_ok=True)
    def generate(cfg, pushforward, model, state):
        if cfg.mode != 'test' or int(state.step) != 100000:
            raise ValueError('Requires mode=test and a 100000-update checkpoint')
        if cfg.dataset.get('split_protocol', 'development') != 'development':
            raise ValueError('Lambda selection requires development partition')
        expected_experiment = 'spd_taxi_'+('malliavin_hutchinson' if method.startswith('malliavin') else method)
        expected_lambda = 5.0 if method == 'malliavin_lambda5' else 0.0
        if str(cfg.experiment) != expected_experiment or int(cfg.seed) != 0:
            raise ValueError('Saved experiment/seed does not match run name')
        if float(cfg.loss.time_weight_lambda) != expected_lambda or bool(cfg.loss.time_weighting) != (expected_lambda != 0):
            raise ValueError('Saved lambda configuration does not match run name')
        sampler = pushforward.get_sampler((model,state.params_ema,state.model_state),
                                          train=False,N=manifest['steps'],eps=cfg.eps,predictor='GRW')
        sampler = jax.jit(sampler,static_argnums=(1,))
        with np.load(manifest['dataset'],allow_pickle=False) as z:
            train_hash = hashlib.sha256(np.ascontiguousarray(z['covariances'][z['train_indices']],dtype=np.float64).tobytes()).hexdigest()
            contexts = z['contexts'][manifest['dataset_rows']]
            targets = z['covariances'][manifest['dataset_rows']]
        actual_hash = hashlib.sha256(np.ascontiguousarray(pushforward.sde.limiting.data,dtype=np.float64).tobytes()).hexdigest()
        if train_hash != actual_hash:
            raise ValueError('Empirical terminal does not match development training data')
        for pos, (index, row) in enumerate(zip(manifest['val_indices'],manifest['dataset_rows'])):
            report_path = dest/f'val_{index:04d}.json'
            sample_path = dest/f'val_{index:04d}.npy'
            if report_path.exists():
                old = json.loads(report_path.read_text())
                if old['sampling']['complete'] and old.get('metrics') is not None:
                    if old['sample_sha256'] != digest(sample_path):
                        raise ValueError('Saved samples changed: '+str(sample_path))
                    continue
            # Same context-specific key across methods and across restarts.
            key = jax.random.fold_in(jax.random.PRNGKey(manifest['generation_seed']),index)
            def draw(size):
                nonlocal key
                key, batch_key = jax.random.split(key)
                context = jnp.tile(jnp.asarray(contexts[pos]),(size,1))
                return np.asarray(sampler(batch_key,(size,),context))
            samples, sampling = collect_spd_samples(draw,manifest['samples'],manifest['samples'],2*manifest['samples'])
            report = dict(method=method, val_index=index, dataset_row=row, context=contexts[pos].tolist(),
                          checkpoint_step=int(state.step), sampling=sampling,
                          terminal_law='context_independent_empirical_train_forward_GRW',
                          training_data_sha256=train_hash, metrics=None)
            if samples is not None:
                temp = sample_path.with_suffix('.tmp')
                with temp.open('wb') as f:
                    np.save(f,samples)
                temp.replace(sample_path)
                report['sample_sha256'] = digest(sample_path)
            write_json(report_path,report)
            if sampling['complete']:
                try:
                    report['metrics'] = evaluate(samples,targets[pos])
                except (ValueError,np.linalg.LinAlgError) as exc:
                    report['metric_error'] = str(exc)
                write_json(report_path,report)
            print(method, f'{pos+1}/{len(contexts)}', 'val=',index,
                  'accepted=',sampling['accepted'],'rejected=',sampling['rejected'],flush=True)
    generation.save_generation = generate
    config_dir = base/'.hydra'
    # Saved configs already contain the composed logger mapping and no defaults
    # group selection. logger=csv here would replace that mapping with a string.
    sys.argv = ['main.py','--config-path',str(config_dir),'--config-name','config',
                'mode=test','resume=false','generation.enabled=true',
                'train_val=false','test_val=false','test_test=false','train_plot=false','test_plot=false',
                'dataset.data_path='+manifest['dataset'], 'ckpt_dir='+str(ckpt),
                'hydra.run.dir='+str(dest/'runtime')]
    os.chdir(ROOT)
    before = checkpoint_hashes(ckpt)
    if before != manifest['checkpoints'][method]:
        raise ValueError('Checkpoint differs from manifest')
    try:
        runpy.run_path(str(ROOT/'main.py'),run_name='__main__')
    finally:
        unchanged = before == checkpoint_hashes(ckpt)
        write_json(dest/'checkpoint_check.json',dict(unchanged=unchanged))
        if not unchanged:
            raise RuntimeError('Checkpoint changed')


def summarize(output, manifest):
    import numpy as np
    result = {'complete':True,'methods':{},'selection_metric':'mean validation squared AIRM distance of converged Frechet means',
              'note':'Single training seed; rejected draws imply conditioning on validity. Energy is a diagnostic, not asserted strictly proper for AIRM.'}
    paired = {}
    for method in METHODS:
        rows = []
        for idx in manifest['val_indices']:
            p = output/method/f'val_{idx:04d}.json'
            if p.exists():
                saved=json.loads(p.read_text())
                if saved.get('sample_sha256') and digest(p.with_suffix('.npy'))!=saved['sample_sha256']:
                    raise ValueError('Saved samples changed: '+str(p))
                rows.append(saved)
        valid = [r for r in rows if r['sampling']['complete'] and r.get('metrics') is not None]
        converged = [r for r in valid if r['metrics']['airm_frechet_to_target'] is not None]
        complete = len(converged)==len(manifest['val_indices'])
        result['complete'] &= complete
        attempted=sum(r['sampling']['attempted'] for r in rows)
        rejected=sum(r['sampling']['rejected'] for r in rows)
        info=dict(conditions_recorded=len(rows), conditions_generated=len(valid),
                  frechet_converged=len(converged), complete=complete,
                  attempted=attempted,rejected=rejected,rejection_rate=rejected/attempted if attempted else None)
        for metric in ('airm_squared_frechet_to_target','frobenius_frechet_to_target','airm_frechet_to_target','frobenius_arithmetic_to_target','sample_airm_to_target_mean','sample_pairwise_airm_mean','energy_airm_diagnostic'):
            values=[r['metrics'][metric] for r in valid if r['metrics'][metric] is not None]
            info[metric]={'mean':float(np.mean(values)), 'median':float(np.median(values))} if values else None
        if len(valid)==len(manifest['val_indices']):
            from riemannian_score_sde.spd_validation_metrics import distribution_metrics
            samples=np.concatenate([np.load(output/method/f"val_{r['val_index']:04d}.npy",allow_pickle=False) for r in valid])
            with np.load(manifest['dataset'],allow_pickle=False) as z:
                reference=z['covariances'][manifest['dataset_rows']]
                train=z['covariances'][z['train_indices']]
            info['pooled_distribution']=distribution_metrics(samples,reference,train)
        result['methods'][method]=info
        paired[method]={r['val_index']:r['metrics']['airm_squared_frechet_to_target'] for r in converged}
    keys=manifest['val_indices']
    if all(len(paired[m])==len(keys) for m in ('malliavin_lambda0','malliavin_lambda5')):
        delta=np.array([paired['malliavin_lambda5'][i]-paired['malliavin_lambda0'][i] for i in keys])
        result['lambda5_minus_lambda0_airm_squared']=dict(mean=float(delta.mean()), median=float(np.median(delta)),
                                                 lambda5_better_fraction=float(np.mean(delta<0)))
    write_json(output/'summary.json',result)
    print(json.dumps(result,indent=2))


def recompute_metrics(source, manifest):
    """CPU-only revision: copy verified arrays and retain original reports untouched."""
    import shutil
    import numpy as np
    from riemannian_score_sde.spd_validation_metrics import evaluate
    if digest(Path(manifest['dataset'])) != manifest['dataset_sha256']:
        raise ValueError('Dataset differs from manifest')
    # Validate all required artifacts before creating a revision.
    records = []
    for method in METHODS:
        for index, row in zip(manifest['val_indices'], manifest['dataset_rows']):
            path = source/method/f'val_{index:04d}.json'
            report = json.loads(path.read_text())
            sample_path = path.with_suffix('.npy')
            if not report['sampling']['complete']:
                raise ValueError('Generation incomplete: '+str(path))
            if report['sample_sha256'] != digest(sample_path):
                raise ValueError('Saved samples changed: '+str(sample_path))
            if report['val_index'] != index or report['dataset_row'] != row:
                raise ValueError('Condition differs from manifest: '+str(path))
            records.append((method, row, path, report))
    output = Path(tempfile.mkdtemp(prefix='metrics_recomputed_', dir=source))
    write_json(output/'manifest.json', manifest)
    write_json(output/'metric_revision.json', dict(source=str(source.resolve()),
               frechet_tolerance=1e-6, frechet_solver_version=2, original_report_sha256={
                   str(path.relative_to(source)):digest(path) for _, _, path, _ in records}))
    print('Recomputed evaluation output:', output, flush=True)
    with np.load(manifest['dataset'], allow_pickle=False) as dataset:
        for method, row, path, report in records:
            dest = output/method/path.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(path.with_suffix('.npy'), dest.with_suffix('.npy'))
            samples = np.load(dest.with_suffix('.npy'), allow_pickle=False)
            report.pop('metric_error', None)
            report['metrics'] = None
            try:
                report['metrics'] = evaluate(samples, dataset['covariances'][row])
            except (ValueError, np.linalg.LinAlgError) as exc:
                report['metric_error'] = str(exc)
            write_json(dest, report)
    summarize(output, manifest)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/spd_taxi/nyc_taxi.npz')
    p.add_argument('--output',type=Path,help='Existing matching output resumes; otherwise creates a new directory')
    p.add_argument('--conditions',type=int,default=100)
    p.add_argument('--samples',type=int,default=20)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--timeout',type=int,default=7200,help='Seconds per method')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--recompute-metrics', action='store_true', help='CPU-only reevaluation of all saved arrays into a new revision directory')
    modes.add_argument('--summarize-only', action='store_true', help='Recompute summary from saved samples, without JAX or generation')
    p.add_argument('--worker',choices=METHODS,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.recompute_metrics:
        if args.output is None:
            p.error('--recompute-metrics requires --output')
        manifest=json.loads((args.output/'manifest.json').read_text())
        output=recompute_metrics(args.output, manifest)
        if not json.loads((output/'summary.json').read_text())['complete']:
            raise SystemExit('Incomplete metrics: inspect the new summary; do not select a winner')
        return
    if args.summarize_only:
        if args.output is None:
            p.error('--summarize-only requires --output')
        manifest=json.loads((args.output/'manifest.json').read_text())
        if digest(Path(manifest['dataset'])) != manifest['dataset_sha256']:
            p.error('Dataset differs from manifest')
        summarize(args.output,manifest)
        return
    if args.worker:
        worker(args,json.loads((args.output/'manifest.json').read_text()))
        return
    import numpy as np
    from riemannian_score_sde.taxi_data import validate
    if args.run_root is None or min(args.conditions,args.steps,args.timeout)<1 or args.samples<2:
        p.error('Specify --run-root; counts must be positive and samples >= 2')
    args.run_root=args.run_root.resolve(); args.dataset=args.dataset.resolve()
    with np.load(args.dataset,allow_pickle=False) as z:
        indices={s:z[s+'_indices'] for s in ('train','val','test')}
        validate(z['covariances'],z['contexts'],indices)
        if [len(indices[s]) for s in ('train','val','test')]!=[6460,1140,1159]:
            p.error('Expected the original development Taxi dataset')
        if args.conditions>len(indices['val']):
            p.error('Too many validation conditions')
        chosen=np.random.RandomState(2026).permutation(len(indices['val']))[:args.conditions]
        selected_rows=indices['val'][chosen].tolist()
    manifest=dict(version=1,run_root=str(args.run_root),dataset=str(args.dataset),
                  dataset_sha256=digest(args.dataset),val_indices=chosen.tolist(),dataset_rows=selected_rows,
                  selection_seed=2026,generation_seed=123,samples=args.samples,steps=args.steps,
                  split_protocol='development', checkpoints={},saved_config_sha256={})
    for method in METHODS:
        base=args.run_root/(method+'_seed0')
        manifest['checkpoints'][method]=checkpoint_hashes(base/'ckpt')
        manifest['saved_config_sha256'][method]=digest(base/'.hydra/config.yaml')
    args.output=args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='validation100_',dir=args.run_root))
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=manifest:
            p.error('Output manifest differs; use a new output directory')
    elif any(args.output.iterdir()):
        p.error('Nonempty output has no manifest')
    else:
        write_json(manifest_path,manifest)
    print('Evaluation output:',args.output,flush=True)
    env=dict(os.environ,GEOMSTATS_BACKEND='jax',JAX_ENABLE_X64='true',HYDRA_FULL_ERROR='1')
    env['PYTHONPATH']=str(ROOT/'geomstats')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    failures=[]
    for method in METHODS:
        log=args.output/(method+'.log')
        try:
            with log.open('a') as f:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',method,'--output',str(args.output)],
                               cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
            print('Finished:',method,flush=True)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
            failures.append(method)
            print('FAILED:',method,'\n'+'\n'.join(log.read_text().splitlines()[-30:]),flush=True)
        if checkpoint_hashes(args.run_root/(method+'_seed0')/'ckpt')!=manifest['checkpoints'][method]:
            raise RuntimeError('Checkpoint changed')
    summarize(args.output,manifest)
    report=json.loads((args.output/'summary.json').read_text())
    if failures or not report['complete']:
        raise SystemExit('Incomplete evaluation: inspect logs/summary; no winner should be selected')


if __name__=='__main__':
    main()
