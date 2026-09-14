"""Read stage logs for execution progress; never inspect scientific outcomes."""
from __future__ import annotations
from pathlib import Path
import json
import time


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'analysis/benchmark_calibrated_fixed_rw_schedule'
    work=out/'work'
    state=json.loads((work/'continuation_status.json').read_text())
    support=json.loads((out/'splits/complete.json').read_text())['support']
    stage=state['stage']
    totals={'dense':sum(r['calibration_uids']+r['test_uids'] for r in support),'R':sum(r['calibration_uids'] for r in support),'W':sum(r['calibration_uids'] for r in support)}
    for name in ['methods','global','random']:totals[name]=sum(r['test_uids'] for r in support)
    report=dict(stage=stage,supervisor_pid=state.get('supervisor_pid'),checked_unix=time.time(),error=state.get('error'))
    if stage in totals:
        ranks=[]
        for rank in range(4):
            path=work/f'logs/{stage}_rank{rank}.log'
            if not path.exists():continue
            with path.open('rb') as handle:
                handle.seek(max(0,path.stat().st_size-65536));lines=handle.read().decode(errors='replace').splitlines()
            valid=[json.loads(line) for line in lines if line.startswith('{"stage"')]
            if not valid:continue
            last=valid[-1];n=(totals[stage]+3-rank)//4
            rate=last.get('new_completed',last['completed'])/last['worker_seconds'] if last['worker_seconds']>0 else 0
            ranks.append(dict(rank=rank,completed=last['completed'],total=n,uids_per_second=rate,eta_seconds=(n-last['completed'])/rate if rate else None,last_progress_age_seconds=time.time()-path.stat().st_mtime))
        report.update(total_uids=totals[stage],completed_uids=sum(r['completed'] for r in ranks),ranks=ranks,current_stage_eta_seconds=max((r['eta_seconds'] for r in ranks),default=None),eta_scope='Current stage only; estimate from completed execution timings, not scientific extrapolation')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
