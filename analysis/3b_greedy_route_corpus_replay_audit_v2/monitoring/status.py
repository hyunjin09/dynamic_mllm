"""Read Slurm and durable progress; no experiment submission or inference."""
import datetime,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
submission=json.loads((root/'pipeline_submission.json').read_text())
ids=','.join(str(r['job_id']) for r in submission['stages'])
accounting=subprocess.check_output(['sacct','-n','-X','-j',ids,'--format=JobID,State,ExitCode,Elapsed,Start,End','-P'],text=True).strip()
result={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accounting':accounting,'stages':{}}
for stage,folder in [('dense','dense'),('routes','replay')]:
    progress=[dict(json.loads(p.read_text()),progress_file=p.name,checkpoint_age_seconds=time.time()-p.stat().st_mtime) for p in (root/folder).glob('rank*_progress.json')]
    complete=[json.loads(p.read_text()) for p in (root/folder).glob('rank*_complete.json')]
    if progress or complete:
        result['stages'][stage]={'workers_reporting':len(progress),'workers_complete':len(complete),
            'saved_records':sum(r['success']+r.get('reused',0) for r in progress),
            'completed_samples':sum(r['samples'] for r in progress),
            'worker_details':progress}
result['alerts']=[]
for item in submission['stages']:
    if item['stage'] not in ['dense','routes']:continue
    line=next((line for line in accounting.splitlines() if line.startswith(str(item['job_id'])+'|')), '')
    if '|RUNNING|' not in line:continue
    folder='dense' if item['stage']=='dense' else 'replay'
    for record in result['stages'].get(item['stage'],{}).get('worker_details',[]):
        completed=root/folder/record['progress_file'].replace('_progress.json','_complete.json')
        if record['checkpoint_age_seconds']>600 and not completed.exists():
            result['alerts'].append('STALE_WORKER '+record['progress_file'])
    for path in (root/'replay_gate').glob('job_'+str(item['job_id'])+'_rank*_failure.json'):
        result['alerts'].append('RUNTIME_FAILURE '+path.name)
for name in ['dense/stage_complete.json','final_verification.json']:
    path=root/name
    if path.exists():result[name]=json.loads(path.read_text())
# Freeze monitoring artifacts while the reporter builds its output manifest.
report_id=next(r['job_id'] for r in submission['stages'] if r['stage']=='report')
report_line=next((line for line in accounting.splitlines() if line.startswith(str(report_id)+'|')), '')
if '|PENDING|' in report_line:
    target=root/'monitoring/latest.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(target)
    with (root/'monitoring/history.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
print(accounting)
print(json.dumps({k:{a:b for a,b in v.items() if a!='worker_details'} for k,v in result['stages'].items()}))

if result['alerts']:print('MONITOR_ALERT',json.dumps(result['alerts']))
