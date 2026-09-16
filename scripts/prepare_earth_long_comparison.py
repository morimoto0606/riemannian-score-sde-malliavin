#!/usr/bin/env python3
"""Prepare 36 fresh S2 fits with paper-aligned training budgets; never trains."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from run_earth_malliavin_lambda0 import select_training_config, validate_training_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    from omegaconf import OmegaConf
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--malliavin-lambda', type=float, choices=(0., 5.), required=True)
    p.add_argument('--spectrum-nmax', type=int, default=10)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if args.spectrum_nmax < 1:
        p.error('spectrum-nmax must be positive')
    output = args.output.resolve()
    if output.exists():
        p.error('Output already exists; choose a new directory')
    records = []
    for dataset, steps in [('earthquake',600000), ('flood',300000), ('volcano',600000)]:
        for method in ('malliavin','spectrum','varadhan','ism'):
            for seed in range(3):
                names = [dataset] + (['volcano'+'e'] if dataset == 'volcano' else [])
                runs = [ROOT/'results'/f'{n}_malliavin_lambda5_seed{seed}' for n in names]
                runs = [r for r in runs if r.is_dir()]
                if len(runs) != 1:
                    raise ValueError(f'Ambiguous/missing baseline: {dataset} seed{seed}')
                source, raw = select_training_config(runs[0], lambda path: OmegaConf.to_container(OmegaConf.load(path), resolve=False))
                validate_training_config(raw, dataset, seed)
                cfg = OmegaConf.create(raw)
                name = f'{dataset}_{method}_seed{seed}'
                dest = output/name
                cfg.mode='train'; cfg.resume=False; cfg.steps=steps
                cfg.batch_size=512; cfg.eval_batch_size=512; cfg.warmup_steps=1000
                cfg.val_freq=10000; cfg.splits=[.8,.1,.1]; cfg.ema_rate=.999
                cfg.optim={'_target_':'optax.adam','learning_rate':2e-4,'b1':.9,'b2':.999,'eps':1e-8}
                cfg.scheduler={'_target_':'optax.join_schedules','schedules':[
                    {'_target_':'optax.linear_schedule','init_value':0.,'end_value':1.,'transition_steps':1000},
                    {'_target_':'optax.cosine_decay_schedule','init_value':1.,'decay_steps':steps-1000,'alpha':0.}],
                    'boundaries':[1000]}
                cfg.loss.time_weighting=method=='malliavin' and args.malliavin_lambda!=0
                cfg.loss.time_weight_lambda=args.malliavin_lambda if method=='malliavin' else 0.
                cfg.loss.like_w=False
                if method=='spectrum':
                    cfg.teacher={'_target_':'riemannian_score_sde.teachers.SpectrumTeacher','n_max':args.spectrum_nmax}
                elif method=='varadhan':
                    cfg.teacher={'_target_':'riemannian_score_sde.teachers.VaradhanTeacher'}
                elif method=='ism':
                    cfg.teacher=None
                    cfg.loss={'_target_':'riemannian_score_sde.losses.get_ism_loss_fn','like_w':True,
                              'hutchinson_type':'None','eps':cfg.eps,'time_weighting':False,'time_weight_lambda':0.}
                else:
                    cfg.teacher.hutchinson_probes=1
                    cfg.teacher.rb_enabled=False
                cfg.name=dataset; cfg.experiment=dataset; cfg.dataset.name=dataset
                cfg.ckpt_dir=str(dest/'ckpt'); cfg.generated_samples_path=str(dest/'generated_samples.npy')
                cfg.logs_dir=str(dest/'logs'); cfg.logdir=str(dest)
                cfg.logger={'csv':{'_target_':'score_sde.utils.loggers_pl.CSVLogger','save_dir':str(dest/'logs'),
                                  'name':'','flush_logs_every_n_steps':1000}}
                cfg.train_val=True; cfg.train_plot=False
                cfg.test_val=False; cfg.test_test=False; cfg.test_plot=False
                records.append((name,cfg,source))
                print(len(records)-1,name,'steps=',steps,'lambda=',cfg.loss.time_weight_lambda)
    if args.check_only:
        return
    output.mkdir(parents=True)
    manifest={'malliavin_lambda':args.malliavin_lambda,'spectrum_nmax':args.spectrum_nmax,
              'note':'Paper-aligned budget/batch/schedule, not exact reproduction; pure teachers and fixed lr/beta are our comparison choices.',
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True), 'runs':[]}
    for name,cfg,source in records:
        dest=output/name
        (dest/'input_config').mkdir(parents=True)
        OmegaConf.save(cfg,dest/'input_config/config.yaml')
        (dest/'source_training.yaml').write_bytes(source.read_bytes())
        manifest['runs'].append({'name':name,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared:', output)


if __name__=='__main__':
    main()
