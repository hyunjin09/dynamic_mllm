import copy
import hashlib
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from common import digest, eligibility, route_key, transition, validate_route, LAYERS
from replay import gate_passes, validate_resume
from finish import hamming_histograms, pair_status
import pytest


def source(mask='1'*LAYERS):
    uid='gqa:example'
    return dict(uid=uid,route_id=uid+':mask:'+hashlib.sha256(f'{uid}:{mask}'.encode()).hexdigest()[:16],
        mask_key=mask,visual_on_mask=[int(x) for x in mask],num_visual_on_layers=mask.count('1'),
        is_all_on=mask=='1'*LAYERS,is_all_off=mask=='0'*LAYERS,result_correct=True)


def test_identity_includes_uid_and_complete_mask():
    r=source();assert validate_route(r)==route_key(r['uid'],r['mask_key'])
    assert route_key('other',r['mask_key'])!=route_key(r['uid'],r['mask_key'])
    assert route_key(r['uid'],'0'+'1'*(LAYERS-1))!=route_key(r['uid'],r['mask_key'])
    with pytest.raises(ValueError):route_key(r['uid'],'1'*28)
    r['visual_on_mask'][0]=0
    with pytest.raises(ValueError):validate_route(r)


def durable_record():
    r=source();key=route_key(r['uid'],r['mask_key'])
    record=dict(sample_uid=r['uid'],route_key=key,mask_key=r['mask_key'],source_route_id=r['route_id'],
        source_record_sha256=digest(r),contract_sha256='contract',replay_status='COMPLETE',current_correctness=False)
    record['record_sha256']=digest(record)
    return {key:dict(source_record=r)},dict(sample_uid=r['uid'],contract_sha256='contract',records=[record],complete=True)


def test_resume_detects_tampering_duplicates_and_false_completion():
    source_map,p=durable_record()
    validate_resume(p,source_map,'contract','gqa:example')
    corrupt=copy.deepcopy(p);corrupt['records'][0]['current_correctness']=True
    with pytest.raises(ValueError):validate_resume(corrupt,source_map,'contract','gqa:example')
    corrupt=copy.deepcopy(p);corrupt['records']*=2
    with pytest.raises(ValueError):validate_resume(corrupt,source_map,'contract','gqa:example')
    corrupt=copy.deepcopy(p);corrupt['records']=[]
    with pytest.raises(ValueError):validate_resume(corrupt,source_map,'contract','gqa:example')


def test_gate_requires_exact_census_and_all_rank_agreement():
    anchors=[dict(uid=str(i)) for i in range(32)]
    records=[dict(uid=str(i),passed=True) for i in range(32)]
    sparse=[dict(passed=True,outputs=[1,2,3]) for _ in range(8)]
    assert gate_passes(records,sparse,anchors)
    assert not gate_passes(records[:-1],sparse,anchors)
    assert not gate_passes(records[:-1]+[records[0]],sparse,anchors)
    sparse[-1]['outputs']=[1,2,4]
    assert not gate_passes(records,sparse,anchors)


def test_hamming_counts_are_unordered_and_class_separated():
    rs=[dict(mask_key='0'*LAYERS,replay_status='COMPLETE',current_correctness=True),
        dict(mask_key='0'*(LAYERS-1)+'1',replay_status='COMPLETE',current_correctness=True),
        dict(mask_key='1'*LAYERS,replay_status='COMPLETE',current_correctness=False)]
    h=hamming_histograms(rs)
    assert h['C_C'][1]==1 and sum(h['C_C'])==1
    assert h['C_W'][35]==1 and h['C_W'][36]==1 and sum(h['C_W'])==2
    assert sum(h['W_W'])==0
    assert eligibility(2,1)['C_C_plus_C_W'] and not eligibility(2,1)['full_pairwise']


def test_pair_semantics_keep_correctness_and_efficiency_distinct():
    pair=dict(chosen_mask_key='a',rejected_mask_key='b',chosen_correct=True,rejected_correct=False,
              pair_type='correctness',chosen_budget=36,rejected_budget=1)
    current={'a':dict(replay_status='COMPLETE',current_correctness=False),
             'b':dict(replay_status='COMPLETE',current_correctness=True)}
    assert pair_status(pair,current)==('PREFERRED_REJECTED_LABELS_REVERSED',False)
    current['a']['current_correctness']=True
    assert pair_status(pair,current)==('ONE_OR_BOTH_LABELS_CHANGED',False)
    pair.update(pair_type='efficiency',rejected_correct=True,chosen_budget=1,rejected_budget=36)
    assert pair_status(pair,current)==('BOTH_LABELS_STABLE',True)
    assert pair_status(pair,{})==('ROUTE_MISSING_OR_REPLAY_ERROR',False)
    assert transition(None,False)=='UNKNOWN'
