"""Sequential stage supervisor; terminates only its own workers on failure."""
from __future__ import annotations
import os
import subprocess
import time
from experiments.prepare_benchmark_fixed_schedule import ROOT, OUT, EXT
from experiments.run_full_benchmark_end_to_end_eval import atomic_json
from experiments.run_benchmark_fixed_schedule import check, read_json


def status(stage,**extra):
    atomic_json(EXT/'continuation_status.json',dict(stage=stage,updated_unix=time.time(),supervisor_pid=os.getpid(),**extra))


def run(stage,module,args,gpus):
    logs=EXT/'logs';logs.mkdir(parents=True,exist_ok=True)
    processes=[];handles=[]
    for rank,gpu in enumerate(gpus):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        command=[str(ROOT/'.venv/bin/python'),'-m',module,*args]
        if len(gpus)==4:command+=['--rank',str(rank)]
        handle=(logs/f'{stage}_rank{rank}.log').open('a');handles.append(handle)
        processes.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True))
    status(stage,worker_pids=[p.pid for p in processes])
    try:
        while True:
            codes=[p.poll() for p in processes]
            if any(code is not None and code!=0 for code in codes):
                raise RuntimeError(f'{stage} worker failed: {codes}')
            if all(code==0 for code in codes):break
            time.sleep(5)
    except BaseException:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill()
        raise
    finally:
        for handle in handles:handle.close()


def main():
    module='experiments.run_benchmark_fixed_schedule'
    try:
        contract=check()
        def passed(relative):
            path=OUT/relative
            return path.exists() and read_json(path).get('passed') and read_json(path).get('contract_sha256')==contract['contract_sha256']
        if not passed('parity/dense_smoke.json'):run('dense_smoke',module,['smoke','--stage','dense'],[0])
        for stage in ['dense','R','W','methods','global','random']:
            if stage=='R' and not passed('parity/interventions_smoke.json'):run('interventions_smoke',module,['smoke','--stage','interventions'],[0])
            if stage=='methods' and not passed('schedules/schedule_freeze.json'):run('schedule_selection','experiments.analyze_benchmark_fixed_schedule',['select'],[''])
            if not (OUT/('parity' if stage=='dense' else 'calibration' if stage in ['R','W'] else 'heldout')/f'{stage}_complete.json').exists():
                occupancy=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv'],text=True)
                atomic_json(EXT/f'{stage}_launch_gpu_state.json',dict(observed_unix=time.time(),gpu_state=occupancy))
                run(stage,module,['worker','--stage',stage],[0,1,2,3])
                run(stage+'_finalize',module,['finalize','--stage',stage],[''])
        run('aggregate','experiments.analyze_benchmark_fixed_schedule',['aggregate'],[''])
        status('analysis_ready_for_interpretation')
    except BaseException as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':main()
