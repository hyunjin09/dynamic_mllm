"""Finish already-authorized W3 stages, stopping on any failed command."""
import os,subprocess,time,json,argparse
from experiments.run_write_harm_structure_learnability import ROOT,OUT,EXT,atomic_json


def main(pids):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLCONFIGDIR=str(EXT/'mpl'),HF_HOME=str(EXT/'hf_home'),TMPDIR=str(EXT/'tmp'),PYTHONUNBUFFERED='1');started=time.time()
    def status(stage,**kw):atomic_json(OUT/'work/continuation_status.json',{'stage':stage,'updated_unix':time.time(),'started_unix':started,**kw});print(stage,kw,flush=True)
    def run(module,args,name):
        status(name)
        with open(EXT/'logs'/f'{name}.log','w') as log:subprocess.run([str(ROOT/'.venv/bin/python'),'-m',module,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        status('awaiting_complete_W3_rollout',pids=pids)
        while not all((OUT/f'work/propagation_full_rank{r}_complete.json').exists() for r in range(4)):
            for r,pid in enumerate(pids):
                if (OUT/f'work/propagation_full_rank{r}_complete.json').exists():continue
                try:os.kill(pid,0)
                except ProcessLookupError:raise RuntimeError(f'rollout rank {r} exited without completion')
            time.sleep(5)
        run('experiments.run_write_harm_propagation',['finalize','--mode','full'],'propagation_full_finalize')
        run('experiments.analyze_write_harm_propagation',['prepare'],'propagation_training_prepare')
        with open(EXT/'logs/propagation_training_gpu_launch.csv','w') as log:subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv'],stdout=log,stderr=subprocess.STDOUT,check=True)
        jobs=[]
        for rank in range(4):
            child_env=env.copy();child_env['CUDA_VISIBLE_DEVICES']=str(rank);log=open(EXT/'logs'/f'propagation_train_rank{rank}.log','w');p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','experiments.analyze_write_harm_propagation','train','--rank',str(rank)],cwd=ROOT,env=child_env,stdout=log,stderr=subprocess.STDOUT);jobs.append(p)
        status('W3_fixed_training',pids=[p.pid for p in jobs])
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in [None,0] for p in jobs):raise RuntimeError('W3 training worker failed; inspect its log')
            time.sleep(5)
        assert all(p.returncode==0 for p in jobs)
        run('experiments.analyze_write_harm_propagation',['aggregate'],'propagation_aggregate')
        run('experiments.summarize_write_harm_structure_learnability',['propagation'],'propagation_summaries')
        status('analysis_ready_for_interpretation')
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pids',nargs=4,type=int,required=True);a=p.parse_args();main(a.pids)
