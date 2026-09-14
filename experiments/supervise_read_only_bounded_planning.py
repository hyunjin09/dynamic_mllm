"""Resumable direct four-GPU execution of the single authorized Phase85 plan."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
EXT=Path('/mnt/hyemin/qwen_train_eval/outputs/read_only_bounded_planning_v1')
LOG=EXT/'logs'; LOG.mkdir(exist_ok=True)
PYTHON=PROJECT/'.venv/bin/python'
RUN='experiments.run_read_only_bounded_planning'


def state(**values):
    values['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    p=EXT/'supervisor_status.json'; temp=p.with_suffix('.tmp');temp.write_text(json.dumps(values,indent=2));temp.replace(p)


def base_env():
    env=os.environ.copy()
    for folder in ['tmp','hf_home','mpl']:(EXT/folder).mkdir(exist_ok=True)
    env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(EXT/'hf_home'),TMPDIR=str(EXT/'tmp'),MPLCONFIGDIR=str(EXT/'mpl'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env


def cpu(args):
    subprocess.run([str(PYTHON),'-m',*args],cwd=PROJECT,env=base_env(),check=True)


def stage(mode):
    jobs=[]
    with (LOG/f'{mode}_gpu_occupancy.txt').open('w') as f:
        subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv'],stdout=f,stderr=subprocess.STDOUT,check=True)
    for rank in range(4):
        env=base_env();env['CUDA_VISIBLE_DEVICES']=str(rank)
        f=(LOG/f'{mode}_rank{rank}.log').open('a')
        job=subprocess.Popen([str(PYTHON),'-m',RUN,'worker','--mode',mode,'--rank',str(rank)],cwd=PROJECT,env=env,stdout=f,stderr=subprocess.STDOUT)
        jobs.append((rank,job,f))
    state(stage=mode,status='running',workers=[dict(rank=r,pid=j.pid) for r,j,_ in jobs])
    while any(j.poll() is None for _,j,_ in jobs):
        state(stage=mode,status='running',workers=[dict(rank=r,pid=j.pid,returncode=j.poll()) for r,j,_ in jobs])
        time.sleep(20)
    failures=[]
    for rank,j,f in jobs:
        f.close()
        if j.returncode:failures.append((rank,j.returncode))
    if failures:raise RuntimeError(f'{mode} workers failed: {failures}')


try:
    if not (PROJECT/'analysis/read_only_bounded_planning/controls/smoke_complete.json').exists():
        stage('smoke');cpu([RUN,'finalize-smoke'])
    stage('128')
    cpu([RUN,'extension','--budget','256']);stage('256')
    cpu([RUN,'extension','--budget','512']);stage('512')
    state(stage='aggregation',status='running')
    cpu(['experiments.finalize_read_only_bounded_planning'])
    state(stage='complete',status='complete')
except BaseException as exc:
    state(stage='stopped',status='failed',error=repr(exc))
    raise
