"""Submit the authorized V2 stage chain only after a reviewed repair gate passes."""
import datetime
import subprocess
from common import *


def main():
    assert not (OUT/'pipeline_submission.json').exists(), 'Preserve and monitor the existing chain'
    base=read_contract();repair=read_json(OUT/'contracts/parity_repair_v1.json')
    job=read_json(OUT/'repair_submission.json')['gate_job_id']
    gate_path=OUT/'parity'/f'job_{job}'/'gate_a_result.json'
    gate=read_json(gate_path)
    assert gate['passed'] and gate['exact_anchor_matches']==gate['anchors']==32
    assert gate['sparse_repeat_passed'] and gate['contract_sha256']==repair['contract_sha256']
    review=read_json(OUT/'parity/repair_gate_review.json')
    assert review['gate_result_sha256']==file_hash(gate_path) and review['admit_dense_replay']
    active=subprocess.check_output(['squeue','-h','-u','hyunjin','-o','%i|%j'],text=True)
    assert not any('3b-v2-dense' in line or '3b-v2-routes' in line or '3b-v2-final-report' in line for line in active.splitlines())
    c={k:v for k,v in base.items() if k not in ['contract_sha256','code_hashes','schema','created_unix','stage']}
    c.update(schema='current_server_3b_dense_routes_replay_v2',stage='DENSE_THEN_ROUTES',
        diagnostic_contract_sha256=base['contract_sha256'],passed_repair_gate=str(gate_path),
        full_replay_authorized_by_gate=True,world_size=8,
        code_hashes={str(p):file_hash(p) for p in sorted((OUT/'code').glob('*')) if p.is_file()},
        prerequisite_hashes={str(p):file_hash(p) for p in [gate_path,OUT/'contracts/parity_repair_v1.json',OUT/'parity/repair_gate_review.json']},
        gate_rule='Passed repaired32-anchor Gate A; repeat32 anchors and four sparse probes across8 workers before each GPU stage',
        dense_rule='All10000 dense routes before routed inference, CPU-certified manifest; dense route reused exactly once in canonical route stream',
        checkpoint_routes=16,hidden_capture=False,
        reporting='Exact old/current transitions, current cohorts, semantic changes, action distances, storage and V3.1 audit; main-agent final interpretation')
    c['contract_sha256']=digest(c)
    atomic_json(OUT/'contracts/replay_v1.json',c,immutable=True)
    record=dict(submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),contract_sha256=c['contract_sha256'],
                stages=[],source_immutable=True,full_replay_complete=False)
    atomic_json(OUT/'pipeline_submission.json',record)
    previous=None
    for stage,filename in [('dense','dense.slurm'),('dense_report','dense_report.slurm'),('routes','routes.slurm'),('report','report.slurm')]:
        command=['sbatch','--parsable']
        if previous is not None:command.append(f'--dependency=afterok:{previous}')
        command.append(str(OUT/'code'/filename))
        result=subprocess.check_output(command,text=True).strip()
        job=int(result.split(';')[0])
        record['stages'].append(dict(stage=stage,job_id=job,afterok=previous))
        atomic_json(OUT/'pipeline_submission.json',record)
        write_text(OUT/'logs'/f'{stage}_{job}_submission.txt',subprocess.check_output(['scontrol','show','job',str(job)],text=True))
        print(stage,job,'afterok',previous,flush=True)
        previous=job


if __name__=='__main__':main()
