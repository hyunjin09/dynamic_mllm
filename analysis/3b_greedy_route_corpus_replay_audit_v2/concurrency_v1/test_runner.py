import copy
import importlib.util
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent))
import runner as r


def test_assignment_is_disjoint_and_balances_remaining_work():
    items = [dict(uid=f'u{i}', rank=i%8) for i in range(240)]
    weights = {x['uid']: float(i%31) for i, x in enumerate(items)}
    parts, loads = r.assign(items, weights)
    assert sorted(uid for part in parts for uid in part) == sorted(weights)
    assert max(loads)-min(loads) <= max(weights.values())
    assert sum(loads) == sum(weights.values())
    assert all(x['rank'] == i%8 for i, x in enumerate(items))


def test_pilot_requires_full_sequence_labels_and_action_identity():
    ref = dict(current_generated_ids=[1,2], current_answer='x', current_score=1., current_correctness=True,
               action_trace=[1,0], visual_tokens=8, text_tokens=5)
    actual = dict(generated_ids=[1,2], answer='x', score=1., correctness=True,
                  action_trace=[1,0], visual_tokens=8, text_tokens=5)
    assert r.same_result(actual, ref)
    for key, bad in [('generated_ids',[1,3]), ('answer','y'), ('score',0.), ('correctness',False),
                     ('action_trace',[0,1]), ('visual_tokens',9), ('text_tokens',6)]:
        candidate = dict(actual, **{key:bad})
        assert not r.same_result(candidate, ref)


def test_completion_requires_all_workers_and_exact_source_rank_coverage(monkeypatch):
    writes = {}
    monkeypatch.setattr(r, 'atomic_json', lambda path,value: writes.update({str(path):copy.deepcopy(value)}))
    c = dict(contract_sha256='inference')
    overlay = dict(overlay_sha256='schedule')
    progress = [dict(job_id='new', overlay_sha256='schedule', elapsed_seconds=10, complete=True,
        by_source_rank={str(k):dict(samples=int(k==worker%8),success=2*int(k==worker%8),reused=int(k==worker%8)) for k in range(8)}) for worker in range(24)]
    totals = {k:dict(samples=3,routes=9) for k in range(8)}
    combined = r.aggregate(progress,c,overlay,'new',totals,complete=True)
    assert all(v == dict(samples=3,success=6,reused=3) for v in combined.values())
    assert len([p for p in writes if p.endswith('_complete.json')]) == 8
    with pytest.raises(AssertionError):
        r.aggregate(progress[:-1],c,overlay,'new',totals,complete=True)
    corrupt = copy.deepcopy(progress)
    corrupt[0]['by_source_rank']['0']['reused'] += 1
    with pytest.raises(AssertionError):
        r.aggregate(corrupt,c,overlay,'new',totals,complete=True)


def test_overlay_record_fields_preserve_resume_integrity():
    source_record = dict(mask_key='1'*36,route_id='original-id')
    source = {'key':dict(source_record=source_record)}
    record = dict(sample_uid='u',route_key='key',contract_sha256='base',source_record_sha256=r.digest(source_record),
        replay_status='COMPLETE',mask_key='1'*36,source_route_id='original-id',
        concurrency_overlay_sha256='overlay',worker_id=23,gpu_index=7)
    record['record_sha256'] = r.digest(record)
    payload = dict(sample_uid='u',contract_sha256='base',records=[record],complete=True)
    assert r.validate_resume(payload,source,'base','u') == [record]
    corrupt = copy.deepcopy(payload)
    corrupt['records'][0]['worker_id'] = 22
    with pytest.raises(ValueError,match='hash mismatch'):
        r.validate_resume(corrupt,source,'base','u')
