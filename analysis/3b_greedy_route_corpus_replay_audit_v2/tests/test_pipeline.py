import csv
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import pytest
import common
import finish
from common import digest, stem
from replay import validate_resume, require_dense_record


def fixture_record(uid,mask,old,current,contract='fixture'):
    key=digest(dict(uid=uid,mask=mask))
    source=dict(route_id=uid+':'+mask,mask_key=mask)
    record=dict(sample_uid=uid,route_key=key,source_route_id=source['route_id'],mask_key=mask,
        source_record_sha256=digest(source),contract_sha256=contract,replay_status='COMPLETE',
        old_correctness=old,current_correctness=current,old_answer=str(old),current_answer=str(current),
        dataset='gqa',label_transition=common.transition(old,current),phase='phase1',provenance=['fixture'],
        route_length=36,interventions=mask.count('0'),visual_tokens=2)
    record['record_sha256']=digest(record)
    return dict(route_key=key,source_record=source),record


def test_dense_reuse_requires_exact_hash_identity_and_success():
    source,r=fixture_record('a','1'*36,False,True)
    item=dict(uid='a',dense_route_key=r['route_key']);c=dict(contract_sha256='fixture')
    assert require_dense_record(r,item,{source['route_key']:source},c)==r
    bad=dict(r,current_correctness=False)
    with pytest.raises(ValueError):require_dense_record(bad,item,{source['route_key']:source},c)
    wrong_source,wrong=fixture_record('a','0'+'1'*35,False,True)
    with pytest.raises(AssertionError):require_dense_record(wrong,item,{wrong_source['route_key']:wrong_source},c)


def test_resume_rejects_duplicate_and_false_completion():
    source,r=fixture_record('a','1'*36,False,True)
    second,r2=fixture_record('a','0'+'1'*35,True,True)
    sources={x['route_key']:x for x in [source,second]}
    base=dict(sample_uid='a',contract_sha256='fixture',complete=False,records=[r])
    assert len(validate_resume(base,sources,'fixture','a'))==1
    with pytest.raises(ValueError):validate_resume(dict(base,records=[r,r]),sources,'fixture','a')
    with pytest.raises(ValueError):validate_resume(dict(base,complete=True),sources,'fixture','a')


def test_pair_audit_preserves_one_sided_changes_and_efficiency_semantics():
    pair=dict(chosen_mask_key='a',rejected_mask_key='b',chosen_correct=True,rejected_correct=False,pair_type='correctness')
    def current(a,b):return {'a':dict(replay_status='COMPLETE',current_correctness=a),'b':dict(replay_status='COMPLETE',current_correctness=b)}
    assert finish.pair_status(pair,current(False,False))==('PREFERRED_CHANGED_ONLY',False)
    assert finish.pair_status(pair,current(True,True))==('REJECTED_CHANGED_ONLY',False)
    assert finish.pair_status(pair,current(False,True))==('PREFERRED_REJECTED_LABELS_REVERSED',False)
    efficiency=dict(pair,rejected_correct=True,pair_type='efficiency',chosen_budget=10,rejected_budget=20)
    assert finish.pair_status(efficiency,current(True,True))==('BOTH_LABELS_STABLE',True)
    assert finish.pair_status(efficiency,current(False,False))==('BOTH_LABELS_CHANGED',False)


def test_report_uses_current_dense_and_preserves_certified_manifest(tmp_path,monkeypatch):
    # End-to-end CPU report with one old-W/new-C and one old-C/new-W sample.
    monkeypatch.setattr(common,'OUT',tmp_path);monkeypatch.setattr(finish,'OUT',tmp_path)
    monkeypatch.setattr(finish,'WORLD_SIZE',1);monkeypatch.setattr(finish,'PAIRS',tmp_path/'pairs')
    def js(name,value):common.atomic_json(tmp_path/name,value)
    def jl(name,value):common.write_jsonl(tmp_path/name,value)
    indices=[];dense_manifest=[];pairs=[]
    for uid,olds,currents in [('a',[False,True,False],[True,True,False]),('b',[True,False,True],[False,True,False])]:
        sources=[];records=[]
        for mask,old,current in zip(['1'*36,'0'+'1'*35,'00'+'1'*34],olds,currents):
            source,record=fixture_record(uid,mask,old,current);sources.append(source);records.append(record)
        jl(f'source/{uid}.jsonl',sources)
        indices.append(dict(uid=uid,path=str(tmp_path/f'source/{uid}.jsonl'),sha256=common.file_hash(tmp_path/f'source/{uid}.jsonl'),
                            routes=3,dense_route_key=records[0]['route_key']))
        js(f'replay/by_sample/{stem(uid)}.json',dict(sample_uid=uid,contract_sha256='fixture',complete=True,records=records))
        dense=records[0]
        dense_manifest.append(dict(sample_uid=uid,dense_route_key=dense['route_key'],old_dense_answer=dense['old_answer'],
            current_dense_answer=dense['current_answer'],old_dense_correctness=olds[0],current_dense_correctness=currents[0]))
        pairs.append(dict(pair_id=uid,uid=uid,benchmark='gqa',pair_type='correctness',chosen_mask_key=records[1]['mask_key'],
            rejected_mask_key=records[2]['mask_key'],chosen_correct=olds[1],rejected_correct=olds[2],
            chosen_route_id=records[1]['source_route_id'],rejected_route_id=records[2]['source_route_id']))
    jl('source_inventory/prepared_index.jsonl',indices)
    jl('source_inventory/relocated_samples.jsonl',[dict(uid=u,benchmark='gqa') for u in ['a','b']])
    jl('dense/current_dense_manifest.jsonl',dense_manifest)
    dense_hash=common.file_hash(tmp_path/'dense/current_dense_manifest.jsonl')
    js('dense/stage_complete.json',dict(passed=True,contract_sha256='fixture',manifest_sha256=dense_hash))
    js('replay/rank0_complete.json',dict(passed=True,contract_sha256='fixture'))
    js('replay_gate/gate_result.json',dict(passed=True,contract_sha256='fixture'))
    js('census/global_census.json',dict(unique_samples=2,unique_routes=6,unique_image_groups=2,routes_per_sample={},saved_correct=3,saved_wrong=3))
    jl('pairs/train_preference_pairs.jsonl',pairs);jl('pairs/validation_preference_pairs.jsonl',[])
    contract=dict(contract_sha256='fixture',prepared_index_sha256=common.file_hash(tmp_path/'source_inventory/prepared_index.jsonl'),
                  hidden_size=2048,source_inventory=[])
    monkeypatch.setattr(finish,'read_contract',lambda:contract)
    finish.finish()
    assert common.file_hash(tmp_path/'dense/current_dense_manifest.jsonl')==dense_hash
    manifest=list(common.rows(tmp_path/'filtering/current_route_manifest.jsonl'))
    a=next(r for r in manifest if r['sample_uid']=='a' and r['mask_key']=='0'+'1'*35)
    assert a['old_dense_correctness'] is False and a['current_dense_correctness'] is True
    assert a['current_route_category']=='C_TO_C'
    assert [r['sample_uid'] for r in common.rows(tmp_path/'geometry_readiness/w_to_c_samples.jsonl')]==['b']
    summary=common.read_json(tmp_path/'summaries/complete_replay_summary.json')
    assert summary['dense_transitions']=={'W_TO_C':1,'C_TO_W':1}
    assert summary['pairs']['C_W']['total']==4
    assert summary['RS_category']=='PENDING_QUALITATIVE_REVIEW'
