"""Read-only compute monitoring; append evidence only while report is pending."""
import collections
import datetime
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[1]
overlay = root / 'concurrency_v1'
submission = json.loads((overlay / 'submission.json').read_text())
job = submission['new_job_id']
report = submission['report_job_id']
target = overlay / f'job_{job}'
accounting = subprocess.check_output(['sacct','-n','-X','-j',f'{job},{report}',
    '--format=JobID,State,Elapsed,Start,End','-P'],text=True).strip()
gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
result = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), accounting=accounting,
    gpu=[dict(zip(['index','utilization_percent','used_mib','total_mib'],map(int,line.split(',')))) for line in gpu.strip().splitlines()],
    process_counts=dict(collections.Counter(line.split(',')[0].strip() for line in processes.strip().splitlines())),
    gate_workers=len(list(target.glob('gate_worker*_complete.json'))), pilot_workers=len(list(target.glob('pilot_worker[0-9]*.json'))),
    failures=[str(p.relative_to(root)) for p in target.glob('*failure.json')])
for name in ['gate_decision.json','pilot_decision.json','status.json','complete.json']:
    p = target / name
    if p.exists():
        result[name] = json.loads(p.read_text())
if 'status.json' in result:
    workers = result['status.json']['workers']
    result['max_checkpoint_age_seconds'] = max(time.time()-w['updated_unix'] for w in workers if not w['complete']) if any(not w['complete'] for w in workers) else 0
if f'{report}|PENDING|' in accounting:
    (root/'monitoring/concurrency_latest.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'monitoring/concurrency_history.jsonl').open('a') as f:
        f.write(json.dumps(result)+'\n')
compact = {k:v for k,v in result.items() if k not in ['gpu','status.json','process_counts']}
compact['gpu_utilization_percent'] = [g['utilization_percent'] for g in result['gpu']]
compact['gpu_memory_used_mib'] = [g['used_mib'] for g in result['gpu']]
compact['processes_per_gpu'] = list(result['process_counts'].values())
if 'status.json' in result:
    compact['progress'] = {k:v for k,v in result['status.json'].items() if k!='workers'}
    compact['new_samples_per_worker'] = [w['new_samples'] for w in result['status.json']['workers']]
print(json.dumps(compact))
