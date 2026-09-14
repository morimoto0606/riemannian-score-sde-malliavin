#!/usr/bin/env python3
"""One-update Taxi check for three objectives; optional generation is separate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'data/spd_taxi/nyc_taxi.npz')
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--split-protocol', choices=('development', 'published_train'), default='development')
    p.add_argument('--generation', action='store_true', help='Also test one-update model sampling, not quality')
    args = p.parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_file():
        p.error('Run build_spd_taxi_dataset.py first')
    (ROOT/'results').mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='spd_taxi_smoke_', dir=ROOT/'results'))
    print('Output:', output, flush=True)
    env = dict(os.environ, GEOMSTATS_BACKEND='jax', JAX_ENABLE_X64='true', HYDRA_FULL_ERROR='1')
    env['PYTHONPATH'] = str(ROOT/'geomstats')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    rows = []
    def run(overrides, log):
        print('Running:', log.name, flush=True)
        try:
            with log.open('w') as f:
                subprocess.run([sys.executable, 'main.py']+overrides, cwd=ROOT, env=env,
                               stdout=f, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print('\n'.join(log.read_text().splitlines()[-50:]), flush=True)
            raise
    def hashes(ckpt):
        return {n:hashlib.sha256((ckpt/n).read_bytes()).hexdigest() for n in ('arrays.npy','tree.pkl')}
    for method in ('varadhan', 'ism', 'malliavin_hutchinson'):
        base = output/method
        ckpt = base/'ckpt'
        common = ['experiment=spd_taxi_'+method, 'logger=csv', 'seed=0', 'steps=1',
                  'batch_size=2', 'eval_batch_size=2', 'architecture.hidden_shapes=[16,16]',
                  'dataset.data_path='+str(dataset), 'dataset.split_protocol='+args.split_protocol]
        run(['--cfg','job']+common, output/(method+'_config.log'))
        run(common+['mode=train','generation.enabled=false','ckpt_dir='+str(ckpt),
                    'hydra.run.dir='+str(base)], output/(method+'_train.log'))
        # Verify checkpoint numerical contents in the same CUDA environment.
        code = ('import jax,numpy as np; from score_sde.utils import restore; '
                's=restore('+repr(str(ckpt))+'); assert int(s.step)==1; '
                'assert all(np.isfinite(np.asarray(v)).all() for n in '
                '("params","params_ema","model_state","opt_state") '
                'for v in jax.tree_util.tree_leaves(getattr(s,n)))')
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,check=True,timeout=args.timeout)
        before = hashes(ckpt)
        if args.generation:
            dest = output/(method+'_generation')
            run(common+['mode=test','generation.enabled=true','generation.count=8',
                        'generation.batch_size=8','generation.steps=64','generation.max_attempts=16',
                        'ckpt_dir='+str(ckpt),'hydra.run.dir='+str(dest),
                        'generated_samples_path='+str(dest/'generated_samples.npy')],
                output/(method+'_generation.log'))
            report = json.loads((dest/'generated_samples.metadata.json').read_text())
            assert report['sampling']['complete'] and report['spd']['count']==8
            assert len(report['context'])==13 and before==hashes(ckpt)
        rows.append(dict(method=method, checkpoint_step=1, checkpoint_finite=True,
                         generation_checked=args.generation, split_protocol=args.split_protocol))
        (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('PASS:',method,flush=True)
    print('ALL PASS:',output/'summary.json')


if __name__ == '__main__':
    main()
