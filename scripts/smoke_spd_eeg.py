#!/usr/bin/env python3
"""Server smoke: real prepared EEG, three one-update runs, both labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/spd_eeg/bnci2014_002.npz')
    p.add_argument('--timeout',type=int,default=600)
    args=p.parse_args()
    dataset=args.dataset.resolve()
    if not dataset.is_file():p.error('Prepare the EEG dataset first')
    (ROOT/'results').mkdir(exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='spd_eeg_smoke_',dir=ROOT/'results'))
    env=dict(os.environ,GEOMSTATS_BACKEND='jax',JAX_ENABLE_X64='true')
    env['PYTHONPATH']=str(ROOT/'geomstats')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    def run(overrides,log):
        with log.open('w') as f:
            subprocess.run([sys.executable,'main.py']+overrides,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
    def hashes(path):
        return {n:hashlib.sha256((path/n).read_bytes()).hexdigest() for n in ['arrays.npy','tree.pkl']}
    for method in ['varadhan','ism','malliavin_hutchinson']:
        base=output/method;ckpt=base/'ckpt'
        common=['experiment=spd_eeg_'+method,'logger=csv','seed=0','steps=1','batch_size=2','eval_batch_size=2','flow.N=1','architecture.hidden_shapes=[16,16]','dataset.data_path='+str(dataset)]
        if method=='malliavin_hutchinson':common+=['loss.time_weighting=false','loss.time_weight_lambda=0.0']
        run(['--cfg','job']+common,output/(method+'_config.log'))
        run(common+['mode=train','generation.enabled=false','hydra.run.dir='+str(base),'ckpt_dir='+str(ckpt)],output/(method+'_train.log'))
        before=hashes(ckpt)
        for label in [0,1]:
            dest=output/(method+'_class'+str(label))
            run(common+['mode=test','generation.enabled=true','generation.count=8','generation.batch_size=8','generation.steps=4','generation.max_attempts=16','generation.class_label='+str(label),'hydra.run.dir='+str(dest),'ckpt_dir='+str(ckpt),'generated_samples_path='+str(dest/'generated_samples.npy')],output/(method+'_class'+str(label)+'.log'))
            report=json.loads((dest/'generated_samples.metadata.json').read_text())
            assert report['class_label']==label and report['checkpoint_step']==1
            assert report['sampling']['complete'] and report['spd']['count']==8
            assert before==hashes(ckpt)
        print('PASS:',method,flush=True)
    print('ALL PASS:',output)

if __name__=='__main__':main()
