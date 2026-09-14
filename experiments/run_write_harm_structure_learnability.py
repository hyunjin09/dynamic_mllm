"""Plan-driven WRITE cache audit and exact local feature extraction."""
from __future__ import annotations
import argparse,json,os,time
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.run_counterfactual_effect_identifiability import atomic_json,atomic_jsonl,atomic_csv,atomic_torch,read_json,read_jsonl,file_sha256,tensor_sha256,_canonical_post
from dense_failure_stage2.write_harm_learnability import feature_schema,write_features,HORIZONS,SEEDS
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/write_harm_structure_learnability';EXT=Path('/mnt/hyemin/qwen_train_eval/outputs/write_harm_structure_learnability_v1')
P82=ROOT/'analysis/dense_failure_stage2/counterfactual_effect_identifiability';STEP=ROOT/'analysis/predictability_generalization/stepA_measurement';B=ROOT/'analysis/predictability_generalization/stepB_id_learnability'
CONFIG=ROOT/'configs/write_harm_structure_learnability_v1.json'

def uid_slug(u):return sha256(u.encode()).hexdigest()

def prepare():
    cfg=read_json(ROOT/'configs/read_harm_structure_learnability_v1.json')
    cfg.update(run_id='write_harm_structure_learnability_v1',output_root=str(OUT),external_work_root=str(EXT),seed=20260914)
    cfg['target']={'name':'H_W','formula':'q_read_only-q_full','parent_column':'u_write_r1','parent_transform':'negative'}
    cfg['features']=feature_schema();cfg['training']['seeds']=list(SEEDS)
    cfg['sources']['plan']='plans/write_harm_structure_learnability_plan.md';cfg['sources']['generic_oof']=str(B/'stage2_dense/oof_predictions_write.jsonl');cfg['sources']['one_step_oof']=str(P82/'write/delta/oof_predictions.jsonl')
    cfg['gate']={'primary':'F_ALL/mlp','minimum_spearman_gain':.1,'minimum_harmful_auc_gain':.08,'require_positive_group_bootstrap_gain_ci':True,'DenseW':'same gate versus refitted pre/delta DenseW baselines','minimum_recall_at_90_precision':.05,'minimum_top10_precision_gain_over_prevalence':.1,'primary_harmful_definition':'H_W>0 including zeros as nonharmful','score_aggregation':'pooled OOF, mean of three raw-unit predictions per state','top_fraction_ties':'state_id lexicographic order, stable descending prediction','precision_recall_threshold_ties':'include every score tie'}
    cfg['propagation']={'horizons':list(HORIZONS),'convention':'H counts common FULL layers after intervention; reached layer=l+H','conditions':['on','off','visual_delta','text_delta','delta','pair'],'models':['mlp'],'primary':'delta/mlp','common_support':'layer<=19','require_DenseW_refit':True,'baseline_transform':'all reused raw predictions and truths explicitly negated; exact UID/layer/state/target join','materiality':{'spearman_gain':.1,'harmful_auc_gain':.08,'counterfactual_over_single_spearman':.05,'counterfactual_over_single_auc':.04},'no_architecture_or_horizon_tuning':True}
    cfg['matching']['bootstrap_unit']='union_of_both_arms_image_groups_with_shared_multiplicities';cfg['matching']['matching']='exact one-to-one state matching; fixed cohort; original pair weights equal; bootstrap arm means normalized separately'
    for name in ['population','parity','structure','features','matched_mechanism','learnability','propagation','generalization','routed_secondary','statistics','figures','summaries']: (OUT/name).mkdir(parents=True,exist_ok=True)
    for name in ['work','features','logs','tmp','hf_home','mpl','coords','propagation']:(EXT/name).mkdir(parents=True,exist_ok=True)
    if not (OUT/'work').exists():(OUT/'work').symlink_to(EXT/'work',target_is_directory=True)
    atomic_json(CONFIG,cfg)
    internal={r['uid']:r for r in read_jsonl(STEP/'manifests/internal_sample_manifest.jsonl')}
    for domain in ['dense','routed']:
        states=read_jsonl(P82/f'states/{domain}_state_manifest.jsonl');utility=pd.read_csv(STEP/f'stage2_{domain}/utility_labels.csv').set_index('state_id');rows=[]
        for r in states:
            v=utility.loc[r['state_id']];assert v.uid==r['uid'] and int(v.layer)==r['layer']
            cF=bool(v.c_full);cRO=bool(v.c_read_only);h=float(v.q_read_only-v.q_full)
            assert abs(h+float(v.u_write_r1))<1e-12
            cohort='write_harmful_flip' if not cF and cRO else 'write_beneficial_flip' if cF and not cRO else 'stable_correct' if cF else 'stable_wrong'
            dense=internal[r['uid']]['dense'];text_count=int(dense['prompt_token_count'])-int(dense['visual_token_count'])
            rows.append({**r,'domain':domain,'h_w':h,'q_full':float(v.q_full),'q_read_only':float(v.q_read_only),'full_correct':cF,'read_only_correct':cRO,'cohort':cohort,'write_sign':'harmful' if h>0 else 'beneficial' if h<0 else 'zero','text_token_count':text_count,'visual_token_count':r['visual_tokens']})
        rows.sort(key=lambda r:r['state_id']);atomic_jsonl(OUT/f'population/{domain}_write_state_manifest.jsonl',rows)
        if domain=='dense':dense_rows=rows
    assert len(dense_rows)==15185 and len({r['uid'] for r in dense_rows})==1413 and len({r['image_group_id'] for r in dense_rows})==1385
    uids={r['uid'] for r in dense_rows};folds=[r for r in read_jsonl(B/'splits/group_fold_registry.jsonl') if r['uid'] in uids];atomic_jsonl(OUT/'population/fold_registry.jsonl',folds)
    pd.DataFrame(dense_rows).groupby(['dataset','source_regime','cohort','write_sign'],observed=True).agg(states=('state_id','size'),uids=('uid','nunique')).reset_index().to_csv(OUT/'population/cohort_counts.csv',index=False)
    schema=cfg['features'];atomic_json(OUT/'features/feature_group_manifest.json',schema)
    # Static sign/identity audit; baseline rates will be recomputed on the new exact denominator.
    indexed={r['state_id']:r for r in dense_rows};baseline=[]
    for name,path,model in [('generic',B/'stage2_dense/oof_predictions_write.jsonl','mlp'),('delta',P82/'write/delta/oof_predictions.jsonl','mlp')]:
        source=read_jsonl(path);models=sorted({r.get('model','') for r in source})
        if name=='generic':model='m2_text_visual' if 'm2_text_visual' in models else None
        if not model:raise RuntimeError(('baseline model selection missing',models))
        vals=[r for r in source if r['model']==model];assert len(vals)==15185,(name,model,len(vals),models)
        for v in vals:
            r=indexed[v['state_id']];assert v['uid']==r['uid'] and abs(-v['truth']-r['h_w'])<1e-9
            if 'layer' in v:assert v['layer']==r['layer']
            baseline.append({'baseline':name,'state_id':v['state_id'],'uid':r['uid'],'layer':r['layer'],'truth':r['h_w'],'prediction':-float(v['prediction']),'old_truth':float(v['truth']),'old_prediction':float(v['prediction']),'source_model':model,'transform':'negative_raw_units'})
    atomic_jsonl(OUT/'learnability/frozen_baseline_predictions.jsonl',baseline)
    atomic_json(OUT/'parity/baseline_sign_alignment.json',{'passed':True,'baseline_rows':len(baseline),'transform':'new truth=-old truth; new prediction=-old prediction; raw units','identity':'exact state_id/UID/layer and target; fold registry unchanged'})
    # Schedule based only on token-count workload; stratified smoke independent of feature values.
    byuid=defaultdict(list)
    for r in dense_rows:byuid[r['uid']].append(r)
    selected=[]
    def pick(candidates):
        candidates=[u for u in candidates if u not in selected]
        if candidates:selected.append(min(candidates,key=lambda u:uid_slug('write-smoke:'+u)))
    for source in ['historical','canonical']:
      for ds in ['chartqa','gqa','textvqa']:
       for wrong in [False,True]:pick([u for u,rs in byuid.items() if rs[0]['source_regime']==source and rs[0]['dataset']==ds and rs[0]['dense_wrong']==wrong])
    for low,high in [(0,8),(9,18),(19,27)]:pick([u for u,rs in byuid.items() if low<=rs[0]['trigger_layer']<=high])
    while len(selected)<32:pick(byuid)
    loads=[0]*4;schedule=[]
    for u in sorted(byuid,key=lambda u:(-sum(r['visual_tokens']**3 for r in byuid[u]),u)):
        rank=min(range(4),key=lambda k:loads[k]);cost=sum(r['visual_tokens']**3 for r in byuid[u]);loads[rank]+=cost;schedule.append({'uid':u,'rank':rank,'smoke':u in selected,'states':len(byuid[u]),'cost':cost})
    atomic_jsonl(OUT/'work/schedule.jsonl',schedule)
    cache=read_json(P82/'features/dense_cache_manifest.json')
    audit={'dense_states':15185,'FULL_READ_ONLY_immediate_cached_states':30370,'missing_immediate_branches':0,'missing_fraction':0,'prestate_cache':'all referenced UID payloads exist','short_horizon_READ_cache':'READ intervention WRITE_ONLY; cannot substitute for READ_ONLY WRITE propagation','routed_cache':'35565 states present; analysis gated on positive dense result','F6':'optional omitted','Stage1_control':'omitted: READ_ONLY Stage1 user-text summaries absent, so not essentially free'}
    assert all(Path(r['pre_state_file']).is_file() for r in dense_rows)
    for entry in cache['files'].values():assert Path(entry['path']).stat().st_size==entry['bytes']
    atomic_json(OUT/'parity/cache_reuse_audit.json',audit)
    (OUT/'parity/write_semantics_contract.md').write_text('# WRITE semantics\n\nFULL=(READ1,WRITE1); READ_ONLY=(READ1,WRITE0). Both READ paths use identical pre-WRITE K/V. Require exact FULL/READ_ONLY text equality at H0, and READ_ONLY visual equals pre-visual. Audit all compact cache states, plus full-text and visual equality on fresh 32-UID replay. Frozen HW=q_READ_ONLY-q_FULL; never substitute the marginal two-READ-regime utility. H0 is immediately after layer l; Hk after k common FULL layers, so H8 requires l<=19. No synthetic layer28, suffix outcome regeneration or search.\n')
    (OUT/'features/write_feature_contract.md').write_text('# Frozen WRITE features\n\nF1 magnitude and log1p magnitudes; F2 token and pooled cosine directions; F3 update concentration/entropy/Gini; F4 exact centered covariance spectrum, effective rank, spectral entropy, pair cosine, variance and centroid distances for pre/FULL/READ_ONLY and changes; F5 fixed pre-query alignment; F7 original merged image-grid spatial concentration. Use all valid visual tokens and frozen last text/control query. F6 is optional and omitted. No target, correctness, dataset/source, future outcome, answer logit or action sequence enters these feature groups. Geometry is FP32 with nonnegative covariance-eigenvalue clipping for numerical roundoff; no dimensionality reduction. Full scalar list is in feature_group_manifest.json.\n')
    (OUT/'protocol.md').write_text('''# WRITE structure and learnability protocol

Exact user-selected staged plan; all 15,185 dense states / 1,413 UIDs / 1,385 image groups. Reuse complete Phase82 FULL/READ_ONLY/pre caches and frozen StepA labels. W0 hard parity precedes full features and interpretation. READ stays ON. Old Phase85 search remains stopped.

W1: 5,000 UID-preserving sign-layer shuffles; continuous HW retained, zero distinct. Global and each dataset/source cell structural census. W2: freeze F1–5,F7 before labels/features are joined for learning. F4 exact spectrum; F6 optional omitted. Strict one-to-one exact nuisance matching (dataset/source/layer/relative-depth/token-count bins), no feature matching. Matched bootstrap resamples union of image groups for both arms with common multiplicities, fixed matched cohort and separately normalized arm means; point estimate mean pair difference, median and pooled-SD effect also reported.

Inherited StepB outer/inner image-group roles, UID-equal weighting, fold-local normalization, same Phase83 linear/128-hidden MLP optimizer/early-stop recipe and three seeds. Primary local model F_ALL/MLP, fixed before results; all family ablations, nuisance and fusion linear/MLP controls. Dense-W-only refits F_ALL plus same-capacity generic-pre and delta baselines. Strict-matched logistic/MLP uses inherited outer folds and frozen Phase83 group calibration split, never a state-random split.

Reuse full-population generic and one-step delta baseline predictions only after explicit NEGATION of raw-unit old predictions and truths, because old target is q_FULL-q_READ_ONLY. Exact state/UID/layer/target joins required. Primary harmful label HW>0 includes exact zero as nonharmful; nonzero-only legacy metrics separately qualified. Pooled OOF score is mean three raw-unit seed predictions per state. Sort state_id lexicographically then stable descending prediction for Top5/10/20%; precision-threshold recall includes entire score ties. Prevalence denominator is the full current analysis cohort (full or Dense-W).

Local positive requires F_ALL/MLP material gain versus BOTH baselines: rho gain>=.10 OR harmful AUROC gain>=.08 and paired 5,000 image-group bootstrap difference lower bound>0. The same requirement must survive Dense-W-only refits; useful high precision is recall>=5% at>=90% precision AND Top10 precision>=cohort prevalence+.10. Gate comparisons and seed metrics saved. Stop propagation if local positive; then execute only gated source/semantic/LODO checks authorized by this plan.

If local weak, W3 H0/H1/H2/H4/H8 FULL versus READ_ONLY intervention with common FULL continuation. Primary emergence is exact H8 common support (l<=19), with native support descriptive. Fixed pooled last-query/mean-visual representations, six ON/OFF/visual-delta/text-delta/combined-delta/pair conditions, same MLP and three seeds throughout; full and Dense-W-only refits. No token architecture search. Material improvement over H0 uses the same rho .10 / AUROC .08 and positive group-bootstrap rule. W-PROP-A further requires counterfactual advantage over best single branch (rho .05 or AUROC .04 with positive CI), Dense-W survival and useful precision. W-PROP-B allows a material single-branch gain with little extra paired gain and Dense-W survival. Otherwise retain weak/mixed qualifications and avoid forcing a positive category.

Generalization and routed transfer only after positive dense evidence; no external deployment automatically. Stop after WRITE characterization, READ-versus-WRITE synthesis and exactly one unexecuted recommendation. Parent READ H8 means 8 total layers, whereas WRITE H8 means 9 total layers; synthesis must retain this difference.
''')
    sources=[CONFIG,ROOT/'plans/write_harm_structure_learnability_plan.md',P82/'frozen_contract.json',P82/'features/dense_cache_manifest.json',P82/'states/dense_branch_execution_manifest.jsonl',STEP/'stage2_dense/utility_labels.csv',STEP/'manifests/internal_sample_manifest.jsonl',B/'splits/group_fold_registry.jsonl',B/'stage2_dense/oof_predictions_write.jsonl',P82/'write/delta/oof_predictions.jsonl',OUT/'population/dense_write_state_manifest.jsonl',OUT/'features/feature_group_manifest.json',OUT/'protocol.md',OUT/'parity/write_semantics_contract.md',OUT/'features/write_feature_contract.md']
    sources += [ROOT/p for p in ['dense_failure_stage2/write_harm_learnability.py','dense_failure_stage2/read_harm_learnability.py','binary_policy/executor/four_action.py','binary_policy/executor/inputs.py','binary_policy/executor/layers.py','experiments/run_write_harm_structure_learnability.py']]
    c={'schema':'write_harm_structure_learnability_v1','created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'config':cfg,'sources':{str(p):file_sha256(p) for p in sources}}
    c['contract_sha256']=sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    if (OUT/'frozen_contract.json').exists():raise RuntimeError('refuse frozen overwrite')
    atomic_json(OUT/'frozen_contract.json',c);print(json.dumps(audit),flush=True)

def verify():
    c=read_json(OUT/'frozen_contract.json')
    for p,h in c['sources'].items():assert file_sha256(p)==h,p
    return c

def open_maps():
    layout=read_json(P82/'features/dense_cache_manifest.json')
    return {k:np.memmap(v['path'],dtype=np.uint16,mode='r',shape=tuple(v['shape'])) for k,v in layout['files'].items()}

def branch_tensor(maps,row,action,stream):
    offset=row['text_offset'] if stream=='text' else row['visual_offset'];count=row['text_tokens'] if stream=='text' else row['visual_tokens']
    return torch.from_numpy(np.array(maps['post_'+stream][offset:offset+count,action],copy=True)).view(torch.bfloat16).unsqueeze(0)

def coordinates(inputs,merge):
    grid=inputs['image_grid_thw'].detach().cpu();assert len(grid)==1 and int(grid[0,0])==1
    h,w=int(grid[0,1])//merge,int(grid[0,2])//merge
    return torch.stack(torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij'),dim=-1).reshape(-1,2)

def worker(rank,mode):
    from transformers import AutoProcessor
    from dense_failure_stage1.runtime import build_dense_inputs,configure_dense_determinism
    from dense_failure_stage2.closed_loop_trajectory_set import compact_router_state
    from experiments.run_predictability_stepA_measurement import _cached_state,_load_model
    from binary_policy.executor import capture_four_action_route,one_step_four_action_from_baseline
    from binary_policy.executor.inputs import build_binary_inputs
    c=verify();cfg=c['config'];torch.set_num_threads(4);configure_dense_determinism(cfg['seed'],cfg['backend_settings']);device=torch.device('cuda:0');torch.cuda.set_device(device)
    if mode=='full':assert read_json(OUT/'parity/smoke_complete.json')['passed']
    schedule=read_jsonl(OUT/'work/schedule.jsonl');assigned={r['uid'] for r in schedule if r['rank']==rank and (mode=='full' or r['smoke'])}
    byuid=defaultdict(list)
    for r in read_jsonl(OUT/'population/dense_write_state_manifest.jsonl'):
        if r['uid'] in assigned:byuid[r['uid']].append(r)
    samples={r['uid']:r for r in read_jsonl(STEP/'manifests/internal_sample_manifest.jsonl') if r['uid'] in assigned}
    if mode=='smoke':processor,base,wrapped=_load_model(cfg,device)
    else:processor=AutoProcessor.from_pretrained(cfg['model']['snapshot_path'],local_files_only=True,use_fast=False)
    merge=int(processor.image_processor.merge_size);maps=open_maps();schema=cfg['features'];started=time.monotonic()
    expected={(r['state_id'],r['action']):r for r in read_jsonl(P82/'states/dense_branch_execution_manifest.jsonl') if r['uid'] in assigned}
    for i,u in enumerate(sorted(assigned)):
        path=EXT/'features'/f'{uid_slug(u)}.json'
        if path.exists():continue
        rows=sorted(byuid[u],key=lambda r:r['layer']);sample=samples[u]
        inputs,_=build_dense_inputs(processor,sample['sample'],device if mode=='smoke' else torch.device('cpu'));xy=coordinates(inputs,merge);assert len(xy)==rows[0]['visual_tokens']
        atomic_torch(EXT/'coords'/f'{uid_slug(u)}.pt',{'uid':u,'coords':xy})
        payload=torch.load(rows[0]['pre_state_file'],map_location='cpu',weights_only=False)
        if mode=='smoke':
            prepared=build_binary_inputs(wrapped,inputs);baseline=capture_four_action_route(wrapped,{},['FULL']*28,prepared_inputs=prepared,use_cache=True,native_full_rows=True)
            actual=prepared.visual_position_ids[:,0,prepared.visual_valid_mask[0]].T[:,1:3].cpu();actual=actual-actual.min(0).values;assert torch.equal(actual,xy)
        output=[]
        for r in rows:
            l=r['layer'];pre=payload['states'][r['state_id']];assert _cached_state(pre['text_states'],pre['visual_states'],pre['text_mask'],pre['visual_mask'])['state_sha256']==r['pre_state_sha256']
            tf=branch_tensor(maps,r,0,'text');tr=branch_tensor(maps,r,2,'text');vf=branch_tensor(maps,r,0,'visual');vr=branch_tensor(maps,r,2,'visual');vp=pre['visual_states'];query=pre['text_states'][0,-1]
            assert torch.equal(tf,tr),(r['state_id'],'WRITE changed same-layer text')
            assert torch.equal(vr,vp),(r['state_id'],'READ_ONLY changed visual state')
            for a,tx,vx in [('FULL',tf,vf),('READ_ONLY',tr,vr)]:
                e=expected[(r['state_id'],a)];assert tensor_sha256(tx)==e['post_text_sha256'] and tensor_sha256(vx)==e['post_visual_sha256']
            parity={'state_id':r['state_id'],'uid':u,'layer':l,'text_l2':float((tf.float()-tr.float()).norm()),'text_max_abs':float((tf.float()-tr.float()).abs().max()),'visual_l2':float((vf.float()-vr.float()).norm()),'off_visual_equals_pre':True,'cached_hashes_exact':True,'fresh_full_text_exact':mode=='smoke'}
            if mode=='smoke':
                live=baseline.pre_layer_states[l];assert _cached_state(*live,prepared.text_valid_mask,prepared.visual_valid_mask)['state_sha256']==r['pre_state_sha256']
                on=one_step_four_action_from_baseline(wrapped,baseline,l,'FULL');off=one_step_four_action_from_baseline(wrapped,baseline,l,'READ_ONLY')
                assert torch.equal(on.post_text_state,off.post_text_state) and torch.equal(off.post_visual_state,live[1])
                canonical=_canonical_post(baseline,l);assert torch.equal(on.post_text_state,canonical[0]) and torch.equal(on.post_visual_state,canonical[1])
                for result,tx,vx,action in [(on,tf,vf,'FULL'),(off,tr,vr,'READ_ONLY')]:
                    comp=compact_router_state(result.post_text_state,result.post_visual_state,prepared.text_valid_mask,prepared.visual_valid_mask)
                    assert torch.equal(comp['text_states'].cpu(),tx) and torch.equal(comp['visual_states'].cpu(),vx)
                    rep=one_step_four_action_from_baseline(wrapped,baseline,l,action);assert torch.equal(rep.post_text_state,result.post_text_state) and torch.equal(rep.post_visual_state,result.post_visual_state);del rep
                del on,off
            args=[t[0].to(device) for t in [vp,vf,vr]]+[query.to(device),xy.to(device)]
            features=write_features(*args);assert list(features)==schema['F_ALL']
            if mode=='smoke':
                repeat=write_features(*args);assert all(abs(features[k]-repeat[k])<=1e-6*max(1,abs(features[k])) for k in features)
            output.append({**r,'features':features,'parity':parity,'contract_sha256':c['contract_sha256']})
        atomic_json(path,{'uid':u,'rows':output,'mode':mode,'contract_sha256':c['contract_sha256']})
        if mode=='smoke':del baseline,prepared
        del payload,inputs
        print(json.dumps({'mode':mode,'rank':rank,'done':i+1,'assigned':len(assigned),'elapsed':time.monotonic()-started,'uid':u}),flush=True)
    atomic_json(OUT/f'work/{mode}_rank{rank}_complete.json',{'passed':True,'rank':rank,'mode':mode,'uids':len(assigned),'elapsed':time.monotonic()-started})

def finalize(mode):
    c=verify();schedule=read_jsonl(OUT/'work/schedule.jsonl');selected=[r for r in schedule if mode=='full' or r['smoke']];allrows=[]
    for r in selected:
        p=read_json(EXT/'features'/f"{uid_slug(r['uid'])}.json");assert p['contract_sha256']==c['contract_sha256'];allrows.extend(p['rows'])
    assert len({r['state_id'] for r in allrows})==len(allrows)
    if mode=='smoke':
        assert len(selected)==32 and any(r['layer']<=8 for r in allrows) and any(r['layer']==27 for r in allrows)
        atomic_json(OUT/'parity/smoke_complete.json',{'passed':True,'uids':32,'states':len(allrows),'contract_sha256':c['contract_sha256']})
        (OUT/'parity/smoke_report.md').write_text(f'# WRITE fresh smoke\n\nPASS 32 UIDs / {len(allrows)} states. Exact full-text equality for FULL/READ_ONLY, READ_ONLY visual carry, shared dense prestate, canonical FULL, cached compact poststate hashes, repeated branches and repeated geometry. Original MRoPE spatial positions equal the processor-derived merged image-grid coordinates. All datasets/sources/classes and early/middle/late/layer27 covered.\n')
        atomic_csv(OUT/'parity/fresh_smoke_rows.csv',[r['parity'] for r in allrows])
    else:
        assert len(allrows)==15185 and len(selected)==1413
        allrows.sort(key=lambda r:r['state_id']);atomic_jsonl(OUT/'features/write_operation_features.jsonl',allrows)
        atomic_csv(OUT/'parity/text_stream_parity.csv',[{k:r['parity'][k] for k in ['state_id','uid','layer','text_l2','text_max_abs','cached_hashes_exact']} for r in allrows])
        atomic_csv(OUT/'parity/visual_stream_difference.csv',[{k:r['parity'][k] for k in ['state_id','uid','layer','visual_l2','off_visual_equals_pre']} for r in allrows])
        atomic_json(OUT/'features/complete.json',{'passed':True,'states':15185,'uids':1413,'contract_sha256':c['contract_sha256']})
    print('PASS',mode,len(allrows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','finalize']);p.add_argument('--rank',type=int,default=0);p.add_argument('--mode',choices=['smoke','full'],default='smoke');a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='worker':worker(a.rank,a.mode)
    else:finalize(a.mode)
