"""Run the predeclared six development screens sequentially and resumably."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
from src.common import ARTIFACTS_ROOT,DATA_ROOT,ROOT,read_json

TABULAR=(('v1-cheap','cheap',False),('v1-logistic','logistic',False),('v1-logistic-id-only','logistic',True))
NEURAL=(('v1-shared-lr001','shared',.001),('v1-separate-lr001','separate',.001),('v1-separate-lr0003','separate',.0003))

def run(data:Path,phase:str,device:str,threads:int,max_seconds:int):
    data_hash=read_json(data/'manifest.json')['data_hash']
    env={**os.environ,'OMP_NUM_THREADS':str(threads),'MKL_NUM_THREADS':str(threads),'OPENBLAS_NUM_THREADS':str(threads)}
    if phase in ('all','tabular'):
        for run_id,model,id_only in TABULAR:
            dest=ARTIFACTS_ROOT/'runs'/run_id
            if (dest/'manifest.json').exists():
                m=read_json(dest/'manifest.json')
                assert m['data_hash']==data_hash and m['status']=='complete'
                print(f'skip verified completed {run_id}',flush=True);continue
            cmd=[sys.executable,'-m','src.training.tabular','--data',str(data),'--run-id',run_id,'--model',model]
            if id_only:cmd.append('--id-only')
            print('start '+run_id,flush=True)
            subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    if phase in ('all','neural'):
        for run_id,mode,lr in NEURAL:
            dest=ARTIFACTS_ROOT/'runs'/run_id
            cmd=[sys.executable,'-m','src.training.neural','--data',str(data),'--run-id',run_id,'--mode',mode,
                '--lr',str(lr),'--steps','1000','--batch-size','32','--validation-every','250','--threads',str(threads),
                '--max-seconds',str(max_seconds),'--device',device,'--question','bounded joint-versus-separate baseline screen']
            if (dest/'metrics.json').exists():
                m=read_json(dest/'metrics.json');assert m['data_hash']==data_hash
                if m['status']=='complete' and m['completed_steps']==1000:
                    print(f'skip verified completed {run_id}',flush=True);continue
            if (dest/'latest.pt').exists():cmd.extend(['--resume','latest'])
            print('start '+run_id,flush=True)
            subprocess.run(cmd,cwd=ROOT,env=env,check=True)
            m=read_json(dest/'metrics.json')
            if m['status']!='complete' or m['completed_steps']!=1000:
                print(f'{run_id} stopped safely at {m["completed_steps"]} updates. Re-run this command to resume, or transfer artifacts and use the same device/runtime in cloud.',flush=True)
                return False
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=DATA_ROOT/'prepared/pilot-v1')
    p.add_argument('--phase',choices=('all','tabular','neural'),default='all')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-seconds',type=int,default=600)
    a=p.parse_args();complete=run(a.data.resolve(),a.phase,a.device,a.threads,a.max_seconds)
    print('Baseline phase complete' if complete else 'Baseline phase paused at a resumable budget boundary',flush=True)
