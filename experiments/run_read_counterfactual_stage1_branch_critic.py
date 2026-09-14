"""Frozen one-layer representation extraction and Stage-1 branch scoring; no suffix labels."""
from __future__ import annotations
import argparse, json, os, time
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.run_counterfactual_effect_identifiability import atomic_json, atomic_jsonl, atomic_csv, atomic_torch, file_sha256, read_json, read_jsonl, _canonical_post, tensor_sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/read_counterfactual_stage1_branch_critic'
EXT=Path('/mnt/hyemin/qwen_train_eval/outputs/read_counterfactual_stage1_branch_critic_v1')
OLD=ROOT/'analysis/dense_failure_stage2/counterfactual_effect_identifiability'
STEP=ROOT/'analysis/predictability_generalization/stepA_measurement'
GATE=ROOT/'analysis/dense_failure_stage1/all_source_threshold_calibration/frozen/robust_stage1_gate.json'
TAU=0.9061332901863008
FEATURES=('text_final','text_mean','visual_mean')


def prepare():
    OUT.mkdir(exist_ok=True);EXT.mkdir(exist_ok=True)
    for name in ['population','parity','scores','paired','policy','statistics','figures','summaries','work']:(OUT/name).mkdir(exist_ok=True)
    for name in ['features','logs','tmp','hf_home','mpl']:(EXT/name).mkdir(exist_ok=True)
    if not (OUT/'work/features').exists():(OUT/'work/features').symlink_to(EXT/'features',target_is_directory=True)
    rows=read_jsonl(OLD/'states/dense_state_manifest.jsonl'); samples={x['uid']:x for x in read_jsonl(STEP/'manifests/internal_sample_manifest.jsonl')}
    assert len(rows)==15185 and len({x['uid'] for x in rows})==1413 and len({x['image_group_id'] for x in rows})==1385
    assert all(x['text_tokens']==1 for x in rows)
    uids=sorted({x['uid'] for x in rows});selected=[]
    def take(candidates):
        available=[u for u in candidates if u not in selected]
        if available:selected.append(min(available,key=lambda u:sha256(('phase86-smoke:'+u).encode()).hexdigest()))
    for source in ['canonical','historical']:
      for dataset in ['chartqa','gqa','textvqa']:
       for wrong in [False,True]:take([u for u in uids if samples[u]['source_regime']==source and samples[u]['dataset']==dataset and samples[u]['dense_wrong']==wrong])
    for lo,hi in [(0,8),(9,18),(19,27)]:take([u for u in uids if lo<=samples[u]['p90']['first_trigger_layer']<=hi])
    while len(selected)<32:take(uids)
    smoke_set=set(selected); loads=[0]*4; schedule=[]
    byuid=defaultdict(list)
    for r in rows:byuid[r['uid']].append(r)
    for u in sorted(uids,key=lambda u:(-sum(r['visual_tokens'] for r in byuid[u]),u)):
        rank=min(range(4),key=lambda k:loads[k]);cost=sum(r['visual_tokens'] for r in byuid[u]);loads[rank]+=cost
        schedule.append(dict(uid=u,rank=rank,smoke=u in smoke_set,states=len(byuid[u]),cost=cost))
    atomic_jsonl(OUT/'work/schedule.jsonl',schedule)
    atomic_jsonl(OUT/'population/eligible_state_manifest.jsonl',({**r,'stage1_post_action_index':r['layer']} for r in rows))
    atomic_jsonl(OUT/'population/excluded_state_manifest.jsonl',[])
    atomic_jsonl(OUT/'population/first_trigger_uid_manifest.jsonl',({k:r[k] for k in ['uid','state_id','layer','trigger_layer','dataset','source_regime','dense_wrong','image_group_id']} for r in rows if r['layer']==r['trigger_layer']))
    counts=pd.DataFrame(rows).groupby(['dataset','source_regime','dense_wrong']).agg(states=('state_id','size'),uids=('uid','nunique')).reset_index();counts.to_csv(OUT/'population/cohort_counts.csv',index=False)
    audit={'dense_states':15185,'uids':1413,'image_groups':1385,'eligible':15185,'layer27_eligible':sum(r['layer']==27 for r in rows),'excluded':0,'cached_compact_branches':30370,'ON_stage1_features_available':15185,'OFF_complete_stage1_features_missing':15185,'OFF_missing_fraction':1.0,'explanation':'Phase82 stores only last text/control token plus visual states, not final user token or user-text mean. Reuse native ON Stage1 features and regenerate only one-layer OFF representations from a dense baseline; no counterfactual suffix or output generation.'}
    atomic_json(OUT/'population/cache_coverage_audit.json',audit)
    indexing='''# Stage-1 post-action indexing contract\n\nThe original DenseFeatureCollector registers forward hooks on decoder layers 0..27. Its entry l is the raw residual OUTPUT of layer l, before final decoder RMSNorm; this is also the input of decoder l+1 when l<27. SharedFailurePredictor uses the matching embedding index l. Thus action l -> post-action state -> Stage-1 index l, NOT l+1. Layer 27 has an explicitly trained output representation and embedding 27: all 1,413 layer-27 states remain eligible; no synthetic index 28.\n\nFeature order is [final user-text token; mean user-text tokens; mean visual tokens], BF16 pooling then concatenation and FP32 normalization with frozen historical mean/std. The final user token is located by the original prompt token-position helper; it is not the final assistant/control prompt token stored in the compact Stage2 cache. Raw states are before decoder final RMSNorm.\n\nFrozen model is the robust ALL-source five-checkpoint probability ensemble. Each head receives the same representation and layer index. Convert each FP32 raw logit to sigmoid in FP64 and average five probabilities, exactly as the P90 trigger pipeline. No additional calibration, retraining or normalization fit. P90 tau=0.9061332901863008 (not the stricter P98 gate default). Store five raw logits, five probabilities, their probability mean and effective ensemble logit. Primary AUROC uses mean probability.\n\nTiming limitation: the frozen P90 trigger l was itself detected from Dense post-layer-l features, while frozen StepA labels intervene at pre-layer l. Preserve the plan's exact UID/action/label identities. This is a retrospective offline rollback comparison; it does not validate a causal forward-only controller. At first-trigger l, ON reproduces the original triggering score, so ON-safe quadrants should be absent there. No label/index shift is allowed to manufacture a next-state-safe transition.\n\nHistorical population admission uses strict >tau. This new plan explicitly defines quadrants with >=tau; retain admission and implement its quadrant rule, reporting exact tau ties. All other score ties select ON.\n'''
    (OUT/'stage1_post_action_indexing_contract.md').write_text(indexing)
    protocol='''# Frozen branch-critic diagnostic protocol\n\nExecute only the user-selected plan, after safely stopping Phase85. All 15,185 exact dense states / 1,413 UIDs / 1,385 image groups are primary. No data subset, retraining, recalibration, threshold optimization, search, suffix generation or controller. Existing in-sample five-head gate reuse is an offline diagnostic, not held-out training generalization.\n\nHard gate: fresh deterministic 32-UID stratified parity, covering both sources, all datasets, Dense W/C, early/middle/late and layer27. At every included state verify exact live compact prestate, common branch prestate, action bits, ON canonical output, native Stage1 feature parity and both compact post-state hashes against Phase82. The full extraction repeats these same checks, without resampling. Missing OFF Stage1 representations (100%) require one-layer replay; native ON cached features are reused and checked exactly. No new branch suffix/answer evaluation.\n\nMetrics: state-micro branch failure AUROC and average precision (AUPRC), Brier, 10 fixed equal-width ECE bins; per-UID macro descriptive checks; probability-difference Pearson/Spearman and strict sign agreement, with exact zero targets excluded and score ties counted as no strict preference. Fixed tiny-q sensitivity |H_R|>1e-4,1e-3,1e-2 is descriptive. Behavioral flips are the primary preference check; balanced accuracy is half the two class accuracies, discordant AUROC labels harmful=1. All key intervals use 5,000 paired image-group bootstrap draws, seed2026091301, resampling full groups; missing-class replicates omitted and counted. No subgroup cherry-picking or arbitrary new decision threshold.\n\nQuadrants use the plan's >=tau rule. First-trigger policy chooses OFF iff ON>=tau and OFF<ON; exact ties choose ON. Fixed branches then FULL continuation use frozen q/correctness only. Report all 1,413 decisions, treated denominators and W→C minus C→W. Descriptive layer bins0–8/9–18/19–27; exact absolute and trigger-relative layers, Dense class and dataset/source cells.\n\nStop after full verification, BC-A/B/C/D evidence assessment and exactly one unexecuted next recommendation.\n'''
    (OUT/'protocol.md').write_text(protocol)
    sources=[ROOT/'plans/read_counterfactual_stage1_branch_critic_diagnostic_plan.md',GATE,OLD/'frozen_contract.json',OLD/'states/dense_state_manifest.jsonl',OLD/'states/dense_branch_execution_manifest.jsonl',OLD/'features/dense_cache_manifest.json',STEP/'manifests/internal_sample_manifest.jsonl',STEP/'stage2_dense/utility_labels.csv',STEP/'stage2_dense/four_branch_results.jsonl',OUT/'stage1_post_action_indexing_contract.md',OUT/'protocol.md',OUT/'population/eligible_state_manifest.jsonl',OUT/'work/schedule.jsonl',ROOT/'configs/counterfactual_effect_identifiability_v1.json']
    gate=read_json(GATE)
    for r in gate['checkpoints']+[gate['normalization']]:
        assert file_sha256(r['path'])==r['sha256'];sources.append(ROOT/r['path'])
    sources += [ROOT/p for p in ['dense_failure_stage1/runtime.py','dense_failure_stage1/shared_global_gate.py','dense_failure_stage1/layerwise_probe.py','binary_policy/executor/four_action.py','binary_policy/executor/inputs.py','binary_policy/executor/layers.py','dense_failure_stage2/closed_loop_trajectory_set.py','experiments/run_read_counterfactual_stage1_branch_critic.py']]
    contract=dict(schema='read_counterfactual_stage1_branch_critic_v1',tau=TAU,eligible_states=15185,first_trigger_uids=1413,bootstrap_draws=5000,bootstrap_seed=2026091301,sources={str(p):file_sha256(p) for p in sources},created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),raw_root=str(EXT))
    contract['contract_sha256']=sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if (OUT/'frozen_contract.json').exists():raise RuntimeError('refuse to overwrite frozen contract')
    atomic_json(OUT/'frozen_contract.json',contract);print(json.dumps(audit),flush=True)


def verify():
    c=read_json(OUT/'frozen_contract.json')
    for p,h in c['sources'].items():assert file_sha256(p)==h, p
    return c


def stage1_pool(text,visual,prepared,positions):
    lookup={int(v):i for i,v in enumerate(prepared.text_indices[0].tolist())}
    user=[lookup[int(v)] for v in positions.user_text]
    return torch.cat([text[0,lookup[positions.final_user_token]],text[0,user].mean(0),visual[0,prepared.visual_valid_mask[0]].mean(0)]).detach().cpu()


def worker(rank,mode):
    from binary_policy.executor import capture_four_action_route,one_step_four_action_from_baseline
    from binary_policy.executor.inputs import build_binary_inputs
    from dense_failure_stage1.runtime import build_dense_inputs,configure_dense_determinism,token_positions
    from dense_failure_stage2.closed_loop_trajectory_set import compact_router_state
    from experiments.run_predictability_stepA_measurement import _cached_state,_load_model
    c=verify();config=read_json(ROOT/'configs/counterfactual_effect_identifiability_v1.json')
    torch.set_num_threads(4);configure_dense_determinism(20260913,config['backend_settings']);device=torch.device('cuda:0');torch.cuda.set_device(device)
    if mode=='full':assert read_json(OUT/'parity/smoke_complete.json')['passed']
    schedule=read_jsonl(OUT/'work/schedule.jsonl');assigned={r['uid'] for r in schedule if r['rank']==rank and (mode=='full' or r['smoke'])}
    rows=read_jsonl(OUT/'population/eligible_state_manifest.jsonl');byuid=defaultdict(list)
    for r in rows:
        if r['uid'] in assigned:byuid[r['uid']].append(r)
    samples={r['uid']:r for r in read_jsonl(STEP/'manifests/internal_sample_manifest.jsonl') if r['uid'] in assigned}
    branches={(r['state_id'],r['action']):r for r in read_jsonl(OLD/'states/dense_branch_execution_manifest.jsonl') if r['uid'] in assigned and r['action'] in ['FULL','WRITE_ONLY']}
    processor,base,wrapped=_load_model(config,device);shard_path=None;shard=None;start=time.monotonic()
    for index,u in enumerate(sorted(assigned)):
        dest=EXT/'features'/f'{sha256(u.encode()).hexdigest()}.pt'
        if dest.exists():continue
        sample=samples[u];inputs,_=build_dense_inputs(processor,sample['sample'],device);prepared=build_binary_inputs(wrapped,inputs);positions=token_positions(processor,base,inputs['input_ids'])
        baseline=capture_four_action_route(wrapped,{},['FULL']*28,prepared_inputs=prepared,use_cache=True,native_full_rows=True)
        if shard_path!=sample['stage1_feature_shard']:
            shard_path=sample['stage1_feature_shard'];shard=torch.load(shard_path,map_location='cpu',weights_only=True)
        si=sample['stage1_feature_row_index'];assert str(shard['uids'][si])==u
        results=[];f_on=[];f_off=[]
        for row in sorted(byuid[u],key=lambda r:r['layer']):
            l=row['layer'];pre=baseline.pre_layer_states[l]
            assert _cached_state(*pre,prepared.text_valid_mask,prepared.visual_valid_mask)['state_sha256']==row['pre_state_sha256']
            canonical=_canonical_post(baseline,l)
            cached_on=torch.cat([shard[k][si,l] for k in FEATURES])
            on_pool=stage1_pool(*canonical,prepared,positions)
            assert torch.equal(on_pool,cached_on),(u,l,'Stage1 native ON feature mismatch')
            check={'state_id':row['state_id'],'uid':u,'layer':l,'stage1_index':l,'dataset':row['dataset'],'source_regime':row['source_regime'],'pre_exact':True,'on_stage1_native_exact':True}
            for action in ['FULL','WRITE_ONLY']:
                result=one_step_four_action_from_baseline(wrapped,baseline,l,action)
                assert result.pre_text_state.data_ptr()==pre[0].data_ptr() and result.pre_visual_state.data_ptr()==pre[1].data_ptr()
                assert (result.execution.read_on,result.execution.write_on)==((True,True) if action=='FULL' else (False,True))
                if action=='FULL':assert torch.equal(result.post_text_state,canonical[0]) and torch.equal(result.post_visual_state,canonical[1])
                compact=compact_router_state(result.post_text_state,result.post_visual_state,prepared.text_valid_mask,prepared.visual_valid_mask)
                expected=branches[(row['state_id'],action)]
                assert tensor_sha256(compact['text_states'])==expected['post_text_sha256'],(u,l,action,'text hash')
                assert tensor_sha256(compact['visual_states'])==expected['post_visual_sha256'],(u,l,action,'visual hash')
                check[action+'_compact_cache_exact']=True
                pool=stage1_pool(result.post_text_state,result.post_visual_state,prepared,positions)
                if mode=='smoke':
                    repeat=one_step_four_action_from_baseline(wrapped,baseline,l,action)
                    assert torch.equal(result.post_text_state,repeat.post_text_state) and torch.equal(result.post_visual_state,repeat.post_visual_state)
                    del repeat
                if action=='FULL':assert torch.equal(pool,cached_on);f_on.append(cached_on)
                else:f_off.append(pool)
                del result
            results.append(check)
        atomic_torch(dest,dict(uid=u,contract_sha256=c['contract_sha256'],state_ids=[r['state_id'] for r in sorted(byuid[u],key=lambda r:r['layer'])],ON=torch.stack(f_on),OFF=torch.stack(f_off),parity=results,smoke=mode=='smoke',source_stage1_shard=shard_path,source_stage1_shard_sha256=file_sha256(shard_path)))
        del baseline,prepared,inputs
        print(json.dumps({'rank':rank,'mode':mode,'uid':u,'done':index+1,'assigned':len(assigned),'elapsed':time.monotonic()-start}),flush=True)
    atomic_json(OUT/f'work/{mode}_rank{rank}_complete.json',dict(passed=True,rank=rank,mode=mode,uids=len(assigned),contract_sha256=c['contract_sha256'],elapsed=time.monotonic()-start))


def finalize_smoke():
    c=verify();schedule=read_jsonl(OUT/'work/schedule.jsonl');rows=[]
    for s in schedule:
      if s['smoke']:
        p=torch.load(EXT/'features'/f"{sha256(s['uid'].encode()).hexdigest()}.pt",map_location='cpu',weights_only=True)
        assert p['smoke'] and p['contract_sha256']==c['contract_sha256'];rows.extend(p['parity'])
    assert len({r['uid'] for r in rows})==32
    assert {r['dataset'] for r in rows}=={'gqa','chartqa','textvqa'} and {r['source_regime'] for r in rows}=={'canonical','historical'}
    assert any(r['layer']<=8 for r in rows) and any(9<=r['layer']<=18 for r in rows) and any(r['layer']==27 for r in rows)
    atomic_csv(OUT/'parity/fresh_smoke_state_parity.csv',rows)
    atomic_json(OUT/'parity/smoke_complete.json',dict(passed=True,uids=32,states=len(rows),contract_sha256=c['contract_sha256']))
    (OUT/'parity/smoke_report.md').write_text(f'# Fresh parity smoke\n\nPASS: 32 stratified UIDs / {len(rows)} states. Exact live prestate, shared branch prestate, action bits, canonical ON output, original native Stage1 ON features and Phase82 compact ON/OFF post-state hashes. Both actions repeated exactly. All datasets, both sources, early/middle/late and layer27 covered. No suffix generation or scientific metrics.\n')
    print('PASS smoke',len(rows),flush=True)


def score():
    from dense_failure_stage1.shared_global_gate import SharedFailurePredictor
    c=verify();assert read_json(OUT/'parity/smoke_complete.json')['passed'];torch.set_num_threads(4)
    gate=read_json(GATE);norm=torch.load(ROOT/gate['normalization']['path'],map_location='cpu',weights_only=True)
    heads=[]
    for ck in gate['checkpoints']:
        head=SharedFailurePredictor(variant='state_layer_random4',**{k:gate['architecture'][k] for k in ['input_size','projection_size','layer_embedding_size','hidden_size']})
        checkpoint=torch.load(ROOT/ck['path'],map_location='cpu',weights_only=True);head.load_state_dict(checkpoint['model_state_dict']);head.eval();heads.append(head)
    rows=read_jsonl(OUT/'population/eligible_state_manifest.jsonl');byid={r['state_id']:r for r in rows}
    util=pd.read_csv(STEP/'stage2_dense/utility_labels.csv').set_index('state_id');samples={r['uid']:r for r in read_jsonl(STEP/'manifests/internal_sample_manifest.jsonl')}
    output=[];parity=[];cache_records=[];max_parity=0.
    for u in sorted({r['uid'] for r in rows}):
        path=EXT/'features'/f'{sha256(u.encode()).hexdigest()}.pt';p=torch.load(path,map_location='cpu',weights_only=True);assert p['contract_sha256']==c['contract_sha256'] and p['uid']==u
        cache_records.append(dict(uid=u,path=str(path),sha256=file_sha256(path)));parity.extend(p['parity'])
        layer=torch.tensor([byid[s]['layer'] for s in p['state_ids']],dtype=torch.long)
        data={}
        for b in ['ON','OFF']:
            x=(p[b].float()-norm['mean'].float())/norm['std'].float()
            with torch.inference_mode():logits=np.stack([h(x,layer).numpy().astype(np.float64) for h in heads],axis=1)
            probs=1/(1+np.exp(-logits));data[b]=(logits,probs,probs.mean(1),torch.linalg.vector_norm(p[b].float(),dim=1).numpy(),torch.linalg.vector_norm(x,dim=1).numpy())
        for i,sid in enumerate(p['state_ids']):
            r=byid[sid];v=util.loc[sid];assert v.uid==u and v.layer==r['layer'] and bool(v.c_full)==r['dense_correct']
            rec={k:r[k] for k in ['state_id','uid','layer','trigger_layer','trigger_relative_depth','dataset','source_regime','image_group_id','dense_wrong','dense_correct']}
            rec.update(q_ON=float(v.q_full),q_OFF=float(v.q_write_only),correct_ON=bool(v.c_full),correct_OFF=bool(v.c_write_only),H_R=float(v.q_write_only-v.q_full),stage1_index=r['layer'])
            for b in ['ON','OFF']:
                logits,probs,mean,rawnorm,normnorm=data[b];rec.update({f'logits_{b}':logits[i].tolist(),f'head_probabilities_{b}':probs[i].tolist(),f'p_{b}':float(mean[i]),f'ensemble_logit_{b}':float(np.log(mean[i]/(1-mean[i]))),f'feature_norm_{b}':float(rawnorm[i]),f'normalized_feature_norm_{b}':float(normnorm[i])})
            rec['delta_p']=rec['p_ON']-rec['p_OFF'];rec['original_dense_score']=samples[u]['p90']['scores'][r['layer']];diff=abs(rec['p_ON']-rec['original_dense_score']);max_parity=max(max_parity,diff)
            assert diff<1e-6,(u,r['layer'],diff)
            output.append(rec)
    assert len(output)==15185 and len({r['state_id'] for r in output})==15185
    atomic_jsonl(OUT/'scores/branch_scores.jsonl',output);atomic_jsonl(OUT/'work/feature_cache_manifest.jsonl',cache_records)
    atomic_csv(OUT/'parity/state_alignment.csv',parity)
    for action,name in [('FULL','on_full_parity'),('WRITE_ONLY','off_wo_parity')]:atomic_csv(OUT/f'parity/{name}.csv',[{k:r[k] for k in ['state_id','uid','layer','stage1_index',action+'_compact_cache_exact']} for r in parity])
    atomic_json(OUT/'parity/full_scoring_verification.json',dict(passed=True,states=len(output),uids=1413,layer27=1413,original_stage1_max_probability_difference=max_parity,threshold_exact_ties=sum(r['p_ON']==TAU or r['p_OFF']==TAU for r in output),missing=0,contract_sha256=c['contract_sha256']))
    print('PASS scored',len(output),'max original score difference',max_parity,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','finalize-smoke','score']);p.add_argument('--rank',type=int,default=0);p.add_argument('--mode',choices=['smoke','full'],default='smoke');a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='worker':worker(a.rank,a.mode)
    elif a.command=='finalize-smoke':finalize_smoke()
    else:score()
