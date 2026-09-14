"""Conditional W3 WRITE rollout; horizons count FULL layers after intervention."""
from __future__ import annotations
import argparse,json,time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from experiments.run_write_harm_structure_learnability import *
from dense_failure_stage2.write_harm_learnability import horizons_for_layer
from dense_failure_stage2.read_short_horizon import pool_horizon_state


def check():
    c=verify();contract=read_json(OUT/'propagation/frozen_contract.json')
    for path,digest in contract['files'].items():assert file_sha256(Path(path))==digest,path
    assert read_json(OUT/'learnability/local_decision_gate.json')['category']=='W-LOCAL-WEAK'
    return c


def prepare():
    c=verify();assert read_json(OUT/'learnability/local_decision_gate.json')['category']=='W-LOCAL-WEAK'
    rows=read_jsonl(OUT/'population/dense_write_state_manifest.jsonl');support=[]
    for h in HORIZONS:
        rr=[r for r in rows if h in horizons_for_layer(r['layer'])]
        support.append(dict(horizon=h,states=len(rr),uids=len({r['uid'] for r in rr}),image_groups=len({r['image_group_id'] for r in rr}),dense_w_states=sum(r['dense_wrong'] for r in rr),max_layer=27-h))
    atomic_csv(OUT/'propagation/horizon_support.csv',support);atomic_jsonl(OUT/'propagation/h8_common_support_manifest.jsonl',[r for r in rows if r['layer']<=19])
    paths=[ROOT/'experiments/run_write_harm_propagation.py',ROOT/'experiments/analyze_write_harm_propagation.py',ROOT/'dense_failure_stage2/read_short_horizon.py',OUT/'local_analysis_contract.json',OUT/'learnability/local_decision_gate.json',ROOT/'tests/test_write_harm_propagation.py',ROOT/'tests/test_write_bootstrap.py',ROOT/'dense_failure_stage2/write_bootstrap.py',OUT/'parity/propagation_protocol_review.json']
    atomic_json(OUT/'propagation/frozen_contract.json',{'parent':c['contract_sha256'],'files':{str(p):file_sha256(p) for p in paths},'horizons':list(HORIZONS),'raw_ON':'exact Phase82 FULL reached-layer tensor references; smoke replays all ON horizons','raw_OFF':'per-UID compact last-query and all visual tokens','no_new_targets':True})


@torch.inference_mode()
def rollout(wrapped,baseline,prepared,layer,action0,horizons):
    from binary_policy.executor.cache import BinaryRouteCache
    from binary_policy.executor.four_action import four_action_layer
    from binary_policy.executor.inputs import resolve_decoder
    from dense_failure_stage2.closed_loop_trajectory_set import compact_router_state
    decoder=resolve_decoder(wrapped);cache=BinaryRouteCache(len(decoder.layers));text,visual=[t.detach().clone() for t in baseline.pre_layer_states[layer]];trace=[];captures={}
    for h in range(max(horizons)+1):
        action=action0 if h==0 else 'FULL'
        text,visual,e=four_action_layer(wrapped,decoder.layers[layer+h],text,visual,prepared,action=action,layer_index=layer+h,cache=cache,use_cache=True,native_causal=baseline.native_causal)
        assert (bool(e.read_on),bool(e.write_on))==(True,action=='FULL');trace.append(action)
        if h in horizons:
            compact=compact_router_state(text,visual,prepared.text_valid_mask,prepared.visual_valid_mask)
            tx=compact['text_states'].detach().cpu().to(torch.bfloat16).contiguous();vx=compact['visual_states'].detach().cpu().to(torch.bfloat16).contiguous()
            captures[h]={'text':tx,'visual':vx,'pooled':pool_horizon_state(tx,vx).to(torch.bfloat16),'text_sha256':tensor_sha256(tx),'visual_sha256':tensor_sha256(vx),'action_trace':list(trace)}
    return captures


def worker(rank,mode):
    from dense_failure_stage1.runtime import build_dense_inputs,configure_dense_determinism
    from experiments.run_predictability_stepA_measurement import _load_model,_cached_state
    from binary_policy.executor import capture_four_action_route
    from binary_policy.executor.inputs import build_binary_inputs
    c=check();cfg=c['config'];torch.set_num_threads(4);configure_dense_determinism(cfg['seed'],cfg['backend_settings']);device=torch.device('cuda:0');torch.cuda.set_device(device)
    if mode=='full':assert read_json(OUT/'propagation/smoke_complete.json')['passed']
    schedule=read_jsonl(OUT/'work/schedule.jsonl');assigned={r['uid'] for r in schedule if r['rank']==rank and (mode=='full' or r['smoke'])}
    byuid=defaultdict(list)
    for r in read_jsonl(OUT/'population/dense_write_state_manifest.jsonl'):
        if r['uid'] in assigned:byuid[r['uid']].append(r)
    samples={r['uid']:r for r in read_jsonl(STEP/'manifests/internal_sample_manifest.jsonl') if r['uid'] in assigned};processor,base,wrapped=_load_model(cfg,device);maps=open_maps();started=time.monotonic()
    for i,u in enumerate(sorted(assigned)):
        dest=EXT/'propagation'/uid_slug(u);dest.mkdir(exist_ok=True);manifest=dest/'manifest.json'
        if manifest.exists():continue
        rows=sorted(byuid[u],key=lambda r:r['layer']);lookup={r['layer']:r for r in rows};inputs,_=build_dense_inputs(processor,samples[u]['sample'],device);prepared=build_binary_inputs(wrapped,inputs)
        baseline=capture_four_action_route(wrapped,{},['FULL']*28,prepared_inputs=prepared,use_cache=True,native_full_rows=True);records=[];raw={}
        for r in rows:
            l=r['layer'];assert _cached_state(*baseline.pre_layer_states[l],prepared.text_valid_mask,prepared.visual_valid_mask)['state_sha256']==r['pre_state_sha256']
            hs=horizons_for_layer(l);off=rollout(wrapped,baseline,prepared,l,'READ_ONLY',hs)
            if mode=='smoke':
                on=rollout(wrapped,baseline,prepared,l,'FULL',hs);repeat=rollout(wrapped,baseline,prepared,l,'READ_ONLY',hs)
            for h in hs:
                parent=lookup[l+h];tf=branch_tensor(maps,parent,0,'text');vf=branch_tensor(maps,parent,0,'visual');o=off[h]
                assert o['action_trace']==['READ_ONLY']+['FULL']*h
                if h==0:
                    assert torch.equal(tf,o['text']) and torch.equal(o['text'],branch_tensor(maps,r,2,'text')) and torch.equal(o['visual'],branch_tensor(maps,r,2,'visual'))
                if mode=='smoke':
                    assert torch.equal(on[h]['text'],tf) and torch.equal(on[h]['visual'],vf)
                    assert repeat[h]['text_sha256']==o['text_sha256'] and repeat[h]['visual_sha256']==o['visual_sha256']
                dt=tf.float()-o['text'].float();dv=vf.float()-o['visual'].float();key=f'{l}:{h}';raw[key]={'text':o['text'],'visual':o['visual']}
                records.append({'state_id':r['state_id'],'uid':u,'layer':l,'horizon':h,'reached_layer':l+h,'on_parent_state_id':parent['state_id'],'off_raw_key':key,'on_text_sha256':tensor_sha256(tf),'on_visual_sha256':tensor_sha256(vf),'off_text_sha256':o['text_sha256'],'off_visual_sha256':o['visual_sha256'],'action_trace':o['action_trace'],'text_l2':float(dt.norm()),'visual_l2':float(dv.norm()),'text_relative_l2':float(dt.norm()/tf.float().norm().clamp_min(1e-12)),'visual_relative_l2':float(dv.norm()/vf.float().norm().clamp_min(1e-12)),'on_pooled':pool_horizon_state(tf,vf).to(torch.bfloat16).float().tolist(),'off_pooled':o['pooled'].float().tolist()})
        atomic_torch(dest/'off_raw.pt',raw);atomic_json(manifest,{'uid':u,'mode':mode,'records':records,'raw_sha256':file_sha256(dest/'off_raw.pt'),'contract_sha256':c['contract_sha256']});del baseline,prepared,inputs,raw
        print(json.dumps({'rank':rank,'mode':mode,'done':i+1,'assigned':len(assigned),'elapsed':time.monotonic()-started}),flush=True)
    atomic_json(OUT/f'work/propagation_{mode}_rank{rank}_complete.json',{'passed':True,'rank':rank,'mode':mode,'uids':len(assigned),'elapsed':time.monotonic()-started})


def finalize(mode):
    c=check();schedule=read_jsonl(OUT/'work/schedule.jsonl');selected=[r for r in schedule if mode=='full' or r['smoke']];rows=read_jsonl(OUT/'population/dense_write_state_manifest.jsonl');byid={r['state_id']:r for r in rows};records=[];rollouts=[];matrices={};indices={}
    if mode=='full':
        for h in HORIZONS:
            rr=[r for r in rows if h in horizons_for_layer(r['layer'])];indices[h]={r['state_id']:i for i,r in enumerate(rr)}
            matrices[h]=np.lib.format.open_memmap(OUT/f'work/propagation_h{h}_pooled.npy',mode='w+',dtype=np.float32,shape=(len(rr),2,7168));atomic_jsonl(OUT/f'work/propagation_h{h}_rows.jsonl',rr)
    for s in selected:
        dest=EXT/'propagation'/uid_slug(s['uid']);m=read_json(dest/'manifest.json');assert m['contract_sha256']==c['contract_sha256'];assert file_sha256(dest/'off_raw.pt')==m['raw_sha256']
        for r in m['records']:
            if mode=='full':matrices[r['horizon']][indices[r['horizon']][r['state_id']]]=np.array([r['on_pooled'],r['off_pooled']],np.float32)
            meta={k:v for k,v in r.items() if k not in ['on_pooled','off_pooled']};meta.update({k:byid[r['state_id']][k] for k in ['dataset','source_regime','image_group_id','dense_wrong','h_w','cohort','write_sign']});meta['off_raw_file']=str(dest/'off_raw.pt');records.append(meta)
        rollouts.append({'uid':s['uid'],'records':len(m['records']),'raw_file':str(dest/'off_raw.pt'),'raw_sha256':m['raw_sha256'],'fresh_smoke':m['mode']=='smoke'})
    assert len({(r['state_id'],r['horizon']) for r in records})==len(records)
    expected=sum(len(horizons_for_layer(r['layer'])) for r in rows if r['uid'] in {s['uid'] for s in selected});assert len(records)==expected
    if mode=='smoke':atomic_json(OUT/'propagation/smoke_complete.json',{'passed':True,'uids':32,'records':len(records),'checks':'all prestate hashes; exact H0 pair caches/text equality; exact canonical ON all horizons; repeated OFF; action bits and trace; no future routing'})
    else:
        for m in matrices.values():m.flush()
        atomic_jsonl(OUT/'propagation/rollout_manifest.jsonl',rollouts);atomic_jsonl(OUT/'propagation/propagated_features.jsonl',records)
        atomic_csv(OUT/'propagation/action_trace_validation.csv',[{k:r[k] for k in ['state_id','horizon','reached_layer','action_trace']} for r in records])
        df=pd.DataFrame(records);summaries=[]
        for family in ['write_sign','cohort']:
            for (h,label),g in df.groupby(['horizon',family]):summaries.append(dict(horizon=h,family=family,cohort=label,states=len(g),text_l2_mean=g.text_l2.mean(),text_l2_median=g.text_l2.median(),visual_l2_mean=g.visual_l2.mean(),visual_l2_median=g.visual_l2.median(),text_relative_l2_mean=g.text_relative_l2.mean(),visual_relative_l2_mean=g.visual_relative_l2.mean(),nonzero_text_fraction=(g.text_l2>0).mean()))
        atomic_csv(OUT/'propagation/effect_growth_statistics.csv',summaries);atomic_json(OUT/'propagation/extraction_complete.json',{'passed':True,'records':len(records),'uids':len(selected)})
    print('PASS propagation',mode,len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','finalize']);p.add_argument('--rank',type=int,default=0);p.add_argument('--mode',choices=['smoke','full'],default='smoke');a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='worker':worker(a.rank,a.mode)
    else:finalize(a.mode)
