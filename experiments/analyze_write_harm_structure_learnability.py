"""Frozen WRITE structure, matched geometry and local OOF learning."""
from __future__ import annotations
import argparse,json,math,time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.run_write_harm_structure_learnability import ROOT,OUT,EXT,P82,B,verify,read_json,read_jsonl,atomic_json,atomic_jsonl,atomic_csv,atomic_torch,file_sha256
from dense_failure_stage2.write_harm_learnability import harm_metrics,SEEDS
from dense_failure_stage2.read_harm_learnability import fit_predict_fold,exact_nuisance_matches,contiguous_run_lengths,adjacent_transition_counts
from experiments.analyze_read_harm_structure_learnability import _flat_sequence_metrics,_sequence_metrics,_role_indices,_matched_role_indices,_uid_weights,_bin_index,_depth_bin
from experiments.analyze_read_counterfactual_stage1_branch_critic import BinaryMetric,correlation


def require_parity():
    c=verify();assert read_json(OUT/'parity/smoke_complete.json')['passed']
    supplement=OUT/'local_analysis_contract.json'
    if supplement.exists():
        for path,digest in read_json(supplement)['files'].items():assert file_sha256(Path(path))==digest,path
    return c


def structure():
    c=require_parity();cfg=c['config'];d=pd.DataFrame(read_jsonl(OUT/'population/dense_write_state_manifest.jsonl'));byuid={u:g.sort_values('layer') for u,g in d.groupby('uid')};maps=[];burden=[];neighborhood=[];spanrows=[];trans=Counter()
    for u,g in byuid.items():
        values=g.h_w.to_numpy();assert (np.diff(g.layer)==1).all();lookup=g.set_index('layer')
        maps.append({'uid':u,'dataset':g.iloc[0].dataset,'source_regime':g.iloc[0].source_regime,'dense_wrong':bool(g.iloc[0].dense_wrong),'states':g[['state_id','layer','h_w','full_correct','read_only_correct','cohort']].to_dict('records')})
        harm=g[g.h_w>0];flip=g[g.cohort=='write_harmful_flip'];burden.append(dict(uid=u,image_group_id=g.iloc[0].image_group_id,dataset=g.iloc[0].dataset,source_regime=g.iloc[0].source_regime,dense_wrong=bool(g.iloc[0].dense_wrong),states=len(g),harmful_fraction=len(harm)/len(g),mean_positive_H_W=harm.h_w.mean() if len(harm) else 0.,max_H_W=g.h_w.max(),harmful_flips=len(flip),first_harmful_layer=int(harm.layer.min()) if len(harm) else None,first_harmful_flip_layer=int(flip.layer.min()) if len(flip) else None))
        trans.update(adjacent_transition_counts(values))
        for sign in ['harmful','beneficial']:
            for length in contiguous_run_lengths(values,sign=sign):spanrows.append(dict(uid=u,sign=sign,length=length))
        for _,row in flip.iterrows():
            for offset in range(-3,4):
                l=int(row.layer)+offset
                if l in lookup.index:neighborhood.append(dict(uid=u,center_state_id=row.state_id,offset=offset,h_w=float(lookup.loc[l].h_w)))
    atomic_jsonl(OUT/'structure/per_uid_write_harm_maps.jsonl',maps);pd.DataFrame(burden).to_csv(OUT/'structure/sample_write_burden.csv',index=False)
    pd.DataFrame([dict(from_sign=a,to_sign=b,count=trans[(a,b)]) for a in ['harmful','beneficial','zero'] for b in ['harmful','beneficial','zero']]).to_csv(OUT/'structure/sign_transition_matrix.csv',index=False)
    spans=pd.DataFrame(spanrows);spans.to_csv(OUT/'structure/span_records.csv',index=False);summary=[]
    for sign,g in spans.groupby('sign'):
        summary.append(dict(sign=sign,spans=len(g),mean_length=g.length.mean(),median_length=g.length.median(),max_length=g.length.max(),fraction_states_in_ge2=g.loc[g.length>=2,'length'].sum()/g.length.sum(),fraction_states_in_ge3=g.loc[g.length>=3,'length'].sum()/g.length.sum()))
    pd.DataFrame(summary).to_csv(OUT/'structure/harmful_span_statistics.csv',index=False)
    neighbor=pd.DataFrame(neighborhood);neighbor.to_csv(OUT/'structure/strong_flip_neighborhood_records.csv',index=False)
    neighbor.assign(harmful=neighbor.h_w>0).groupby('offset').agg(states=('h_w','size'),mean_H_W=('h_w','mean'),median_H_W=('h_w','median'),harmful_prevalence=('harmful','mean')).reset_index().to_csv(OUT/'structure/strong_flip_neighborhood.csv',index=False)
    def table(data,cols):
        return data.assign(harmful=data.h_w>0,harmful_flip=data.cohort=='write_harmful_flip').groupby(cols,observed=True).agg(states=('state_id','size'),uids=('uid','nunique'),mean_H_W=('h_w','mean'),median_H_W=('h_w','median'),harmful_prevalence=('harmful','mean'),harmful_flip_prevalence=('harmful_flip','mean')).reset_index()
    d['depth_bin']=pd.cut(d.layer,[-1,8,18,27],labels=['Early','Middle','Late']);d['relative_bin']=pd.cut(d.trigger_relative_depth,[-1,0,2,5,27],labels=['d0','d1_2','d3_5','d6_plus'])
    for cols,name in [(['layer'],'absolute_layer_summary'),(['depth_bin'],'depth_summary'),(['relative_bin'],'trigger_relative_summary'),(['dataset','source_regime'],'dataset_source_prevalence'),(['dataset','source_regime','layer'],'dataset_source_layer_structure')]:table(d,cols).to_csv(OUT/f'structure/{name}.csv',index=False)
    groups=[('ALL',d)]+[(f'{ds}|{src}',g) for (ds,src),g in d.groupby(['dataset','source_regime'])];nullrows=[]
    for label,g in groups:
        seq=[v.sort_values('layer').h_w.to_numpy() for _,v in g.groupby('uid')];flat=np.concatenate([np.sign(v).astype(np.int8) for v in seq]);starts=np.zeros(len(flat),bool);starts[np.cumsum([0]+[len(v) for v in seq[:-1]])]=True
        observed=_sequence_metrics(seq);rng=np.random.default_rng(2026091411);draws=defaultdict(list)
        for _ in range(5000):
            shuffled=np.concatenate([rng.permutation(np.sign(v).astype(np.int8)) for v in seq]);m=_flat_sequence_metrics(shuffled,starts)
            for k,v in m.items():draws[k].append(v)
        for k,v in observed.items():
            a=np.array(draws[k]);nullrows.append(dict(cell=label,metric=k,observed=v,null_mean=np.nanmean(a),null_q025=np.nanquantile(a,.025),null_q975=np.nanquantile(a,.975),enrichment_ratio=v/np.nanmean(a) if np.nanmean(a)!=0 else np.nan,empirical_p_ge=(1+(a>=v).sum())/5001))
        print('structure null',label,flush=True)
    n=pd.DataFrame(nullrows);n.to_csv(OUT/'structure/shuffled_null_statistics.csv',index=False);n[n.metric.str.contains('persistence')].to_csv(OUT/'structure/adjacent_persistence.csv',index=False);n[n.cell!='ALL'].to_csv(OUT/'structure/dataset_source_structure.csv',index=False)
    atomic_json(OUT/'structure/complete.json',{'passed':True,'states':len(d),'uids':len(byuid),'sign_counts':d.write_sign.value_counts().to_dict(),'cohort_counts':d.cohort.value_counts().to_dict(),'shuffle_draws':5000})


def matched_effects(matches,features,names,label):
    if not matches:return []
    t=[features[r['treated_state_id']] for r in matches];c=[features[r['control_state_id']] for r in matches]
    tv=np.array([[r['features'][n] for n in names] for r in t]);cv=np.array([[r['features'][n] for n in names] for r in c]);delta=tv-cv
    groups=sorted({r['image_group_id'] for r in t+c});index={g:i for i,g in enumerate(groups)};ti=np.array([index[r['image_group_id']] for r in t]);ci=np.array([index[r['image_group_id']] for r in c]);rng=np.random.default_rng(2026091422);boot=[]
    for _ in range(5000):
        w=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups));wt=w[ti];wc=w[ci]
        if wt.sum() and wc.sum():boot.append(wt@tv/wt.sum()-wc@cv/wc.sum())
    b=np.array(boot);sd=np.sqrt((tv.var(0)+cv.var(0))/2)
    return [dict(comparison=label,feature=n,pairs=len(matches),matched_mean_difference=delta[:,i].mean(),matched_median_difference=np.median(delta[:,i]),standardized_effect_size=delta[:,i].mean()/sd[i] if sd[i]>0 else 0.,bootstrap_ci_low=np.quantile(b[:,i],.025),bootstrap_ci_high=np.quantile(b[:,i],.975),valid_bootstrap_draws=len(b),bootstrap_unit='union_image_groups_both_arms') for i,n in enumerate(names)]


def prepare_training():
    c=require_parity();cfg=c['config'];assert read_json(OUT/'features/complete.json')['passed'];rows=read_jsonl(OUT/'features/write_operation_features.jsonl');rows.sort(key=lambda r:r['state_id']);schema=cfg['features'];names=schema['F_ALL'];index={r['state_id']:r for r in rows};effects=[];matched=[]
    for r in rows:
        parts=[r['dataset'],r['source_regime'],str(r['layer']),_depth_bin(r['trigger_relative_depth'],cfg['structure']['trigger_relative_bins']),_bin_index(r['visual_token_count'],cfg['matching']['visual_token_count_edges']),_bin_index(r['text_token_count'],cfg['matching']['text_token_count_edges'])];r['cell']='|'.join(parts)
    for label,cohort in [('harmful_vs_beneficial','write_beneficial_flip'),('harmful_vs_stable_wrong','stable_wrong'),('harmful_vs_stable_correct','stable_correct')]:
        matches=exact_nuisance_matches(rows,treated='write_harmful_flip',control=cohort,cell_key='cell',seed=2026091421)
        atomic_jsonl(OUT/f'matched_mechanism/{label}_matches.jsonl',matches);effects.extend(matched_effects(matches,index,names,label))
        if label=='harmful_vs_beneficial':matched=matches
    pd.DataFrame(effects).to_csv(OUT/'matched_mechanism/feature_effect_sizes.csv',index=False)
    distributions=[]
    for name in names:
        for cohort in ['ALL','write_harmful_flip','write_beneficial_flip','stable_wrong','stable_correct']:
            vals=np.array([r['features'][name] for r in rows if cohort=='ALL' or r['cohort']==cohort]);distributions.append(dict(feature=name,cohort=cohort,states=len(vals),mean=vals.mean() if len(vals) else np.nan,median=np.median(vals) if len(vals) else np.nan,std=vals.std() if len(vals) else np.nan))
    pd.DataFrame(distributions).to_csv(OUT/'features/feature_distribution_summary.csv',index=False)
    scalar=np.asarray([[r['features'][n] for n in names] for r in rows],np.float32);np.save(OUT/'work/local_scalar.npy',scalar)
    spec=read_json(P82/'features/dense_cache_manifest.json')['files']['pooled'];pooled=np.memmap(spec['path'],dtype=np.uint16,mode='r',shape=tuple(spec['shape']))
    pre=np.lib.format.open_memmap(OUT/'work/generic_pre.npy',mode='w+',dtype=np.float32,shape=(15185,7168));delta=np.lib.format.open_memmap(OUT/'work/one_step_delta.npy',mode='w+',dtype=np.float32,shape=(15185,7168))
    for start in range(0,len(rows),128):
        ii=[r['row_index'] for r in rows[start:start+128]];raw=torch.from_numpy(np.array(pooled[ii],copy=True)).view(torch.bfloat16).float().numpy();pre[start:start+len(ii)]=raw[:,0];delta[start:start+len(ii)]=raw[:,1]-raw[:,3]
    pre.flush();delta.flush();np.save(OUT/'work/fusion.npy',np.concatenate([pre,scalar],axis=1))
    nuisance=[]
    for r in rows:nuisance.append([r['layer']/27,r['trigger_layer']/27,r['trigger_relative_depth']/27,np.log1p(r['text_token_count']),np.log1p(r['visual_token_count'])]+[int(r['dataset']==ds) for ds in ['chartqa','gqa','textvqa']]+[int(r['source_regime']=='canonical')])
    np.save(OUT/'work/nuisance.npy',np.asarray(nuisance,np.float32))
    meta=[{k:v for k,v in r.items() if k not in ['features','parity']} for r in rows];atomic_jsonl(OUT/'work/training_rows.jsonl',meta)
    matched_ids={m[k] for m in matched for k in ['treated_state_id','control_state_id']};atomic_jsonl(OUT/'work/matched_state_ids.jsonl',[{'state_id':s} for s in sorted(matched_ids)])
    tasks=[]
    def add(pop,group,model):
        for fold in range(5):
            if pop=='matched' and not any(r['state_id'] in matched_ids and r['fold']==fold for r in rows):continue
            for seed in SEEDS:tasks.append(dict(task_index=len(tasks),population=pop,group=group,model=model,fold=fold,seed=seed))
    for group in list(schema['groups'])+['F_ALL','NUISANCE','FUSION']:
        for model in ['linear','mlp']:add('full',group,model)
    for group in ['F_ALL','PRE','DELTA']:add('dense_w',group,'mlp')
    add('dense_w','F_ALL','linear')
    if matched_ids:
        for model in ['linear','mlp']:add('matched','F_ALL',model)
    atomic_jsonl(OUT/'work/local_training_tasks.jsonl',tasks)
    atomic_json(OUT/'work/local_training_prepared.json',{'passed':True,'tasks':len(tasks),'states':len(rows),'matched_states':len(matched_ids),'matched_pairs':len(matched),'matrices':{n:file_sha256(OUT/f'work/{n}.npy') for n in ['local_scalar','generic_pre','one_step_delta','fusion','nuisance']}})
    print('Prepared local training',len(tasks),'tasks; matched',len(matched_ids),flush=True)


def feature_matrix(group,cfg):
    lookup={'PRE':'generic_pre','DELTA':'one_step_delta','FUSION':'fusion','NUISANCE':'nuisance'}
    if group in lookup:return torch.from_numpy(np.load(OUT/f'work/{lookup[group]}.npy'))
    full=np.load(OUT/'work/local_scalar.npy');names=cfg['features']['F_ALL'];chosen=names if group=='F_ALL' else cfg['features']['groups'][group]
    return torch.from_numpy(full[:,[names.index(k) for k in chosen]])


def train(rank):
    c=require_parity();cfg=c['config'];torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0');tasks=read_jsonl(OUT/'work/local_training_tasks.jsonl');rows=read_jsonl(OUT/'work/training_rows.jsonl');matched={r['state_id'] for r in read_jsonl(OUT/'work/matched_state_ids.jsonl')};cache={};started=time.monotonic();dest=OUT/'work/local_fits';dest.mkdir(exist_ok=True)
    for task in tasks:
        if task['task_index']%4!=rank:continue
        path=dest/f"task_{task['task_index']:04d}.json"
        if path.exists():continue
        key=task['group']
        if key not in cache:cache={key:feature_matrix(key,cfg)}
        ii=[i for i,r in enumerate(rows) if task['population']=='full' or (task['population']=='dense_w' and r['dense_wrong']) or (task['population']=='matched' and r['state_id'] in matched)]
        rr=[rows[i] for i in ii];x=cache[key].index_select(0,torch.tensor(ii));classification=task['population']=='matched'
        try:roles=_matched_role_indices(cfg,rr,task['fold']) if classification else _role_indices(cfg,rr,task['fold'])
        except RuntimeError as exc:
            if classification:atomic_json(path,{**task,'status':'unestimable_sparse_matched_roles','reason':str(exc),'test_state_ids':[r['state_id'] for r in rr if r['fold']==task['fold']]});continue
            raise
        target=np.array([int(r['cohort']=='write_harmful_flip') if classification else r['h_w'] for r in rr])
        spec=cfg['training'][task['model']]
        result=fit_predict_fold(x,target,fit_indices=roles['fit'],calibration_indices=roles['calibration'],test_indices=roles['outer_test'],fit_weights=_uid_weights(rr,roles['fit']),calibration_weights=_uid_weights(rr,roles['calibration']),kind=task['model'],spec=spec,hidden_size=128,dropout=.1,target_scale_floor=.0001,gradient_clip_norm=1.,seed=task['seed'],device=device,classification=classification)
        prediction=[{'state_id':rr[int(j)]['state_id'],'truth':float(target[int(j)]),'prediction':float(v)} for j,v in zip(result['test_indices'],result['test_prediction'])]
        atomic_torch(dest/f"task_{task['task_index']:04d}_normalization.pt",{'mean':result['standardizer_mean'],'std':result['standardizer_std']})
        atomic_json(path,{**task,'status':'complete','predictions':prediction,'best_epoch':result['best_epoch'],'best_calibration_loss':result['best_calibration_loss'],'history':result['history'],'fit_states':len(roles['fit']),'calibration_states':len(roles['calibration']),'test_states':len(prediction),'contract_sha256':c['contract_sha256']})
        print(json.dumps({'rank':rank,'task':task['task_index'],'population':task['population'],'group':key,'model':task['model'],'elapsed':time.monotonic()-started}),flush=True)
    atomic_json(OUT/f'work/local_training_rank{rank}_complete.json',{'passed':True,'rank':rank,'elapsed':time.monotonic()-started})


def bootstrap_comparison(rows,truth,left,right,seed=2026091499):
    groups=sorted({r['image_group_id'] for r in rows});index={g:i for i,g in enumerate(groups)};gi=np.array([index[r['image_group_id']] for r in rows]);rng=np.random.default_rng(seed)
    am=BinaryMetric(truth>0,left);bm=BinaryMetric(truth>0,right);data=[]
    for _ in range(5000):
        counts=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups));w=counts[gi]
        data.append([correlation(truth,left,w,True)-correlation(truth,right,w,True),am(w)['AUROC']-bm(w)['AUROC']])
    a=np.asarray(data);return {'spearman_gain':harm_metrics(truth,left)['spearman']-harm_metrics(truth,right)['spearman'],'spearman_ci_low':float(np.nanquantile(a[:,0],.025)),'spearman_ci_high':float(np.nanquantile(a[:,0],.975)),'harmful_auc_gain':am(np.ones(len(rows)))['AUROC']-bm(np.ones(len(rows)))['AUROC'],'harmful_auc_ci_low':float(np.nanquantile(a[:,1],.025)),'harmful_auc_ci_high':float(np.nanquantile(a[:,1],.975)),'draws':5000,'bootstrap_unit':'image_group_id'}


def aggregate():
    c=require_parity();cfg=c['config'];rows=read_jsonl(OUT/'work/training_rows.jsonl');byid={r['state_id']:r for r in rows};tasks=read_jsonl(OUT/'work/local_training_tasks.jsonl');collected=defaultdict(lambda:defaultdict(dict));seedpred=defaultdict(dict);incomplete=[]
    for task in tasks:
        path=OUT/f"work/local_fits/task_{task['task_index']:04d}.json";result=read_json(path)
        if result['status']!='complete':incomplete.append(result);continue
        key=(task['population'],task['group'],task['model']);sk=(*key,task['seed'])
        for r in result['predictions']:
            assert r['state_id'] not in seedpred[sk],('duplicate OOF',sk,r['state_id']);seedpred[sk][r['state_id']]=r['prediction'];collected[key][r['state_id']][task['seed']]=r['prediction']
    atomic_json(OUT/'matched_mechanism/probe_support.json',{'unestimable_tasks':incomplete,'complete_if_all_matched_states_have_all_seed_predictions':not incomplete})
    metrics=[];predictions=[];ensembles={};seedmetrics=[]
    for key,values in collected.items():
        pop,group,model=key;ids=sorted(values);rr=[byid[s] for s in ids];truth=np.array([int(r['cohort']=='write_harmful_flip') if pop=='matched' else r['h_w'] for r in rr]);prediction=np.array([np.mean(list(values[s].values())) for s in ids])
        assert all(len(values[s])==3 for s in ids)
        expected=[r for r in rows if pop=='full' or (pop=='dense_w' and r['dense_wrong'])]
        if pop!='matched':assert len(rr)==len(expected)
        ensembles[key]=(rr,truth,prediction)
        if pop=='matched':
            m=BinaryMetric(truth,prediction)(np.ones(len(rr)));order=np.argsort(-prediction,kind='stable');met={'harmful_auroc':m['AUROC'],'harmful_auprc':m['AUPRC'],'precision_at_0.1':truth[order[:max(1,math.ceil(len(rr)*.1))]].mean(),'states':len(rr)}
        else:met=harm_metrics(truth,prediction,[r['cohort']=='write_harmful_flip' for r in rr])
        metrics.append(dict(population=pop,feature_group=group,model=model,**met))
        predictions.extend({'population':pop,'feature_group':group,'model':model,'state_id':s,'truth':float(y),'prediction':float(p)} for s,y,p in zip(ids,truth,prediction))
        for seed in SEEDS:
            pp=np.array([seedpred[(*key,seed)][s] for s in ids]);sm=harm_metrics(truth,pp) if pop!='matched' else {'auroc':BinaryMetric(truth,pp)(np.ones(len(ids)))['AUROC']};seedmetrics.append(dict(population=pop,feature_group=group,model=model,seed=seed,**sm))
    baseline=read_jsonl(OUT/'learnability/frozen_baseline_predictions.jsonl')
    for name,group in [('generic','B1_GENERIC'),('delta','B2_DELTA')]:
        bi={r['state_id']:r for r in baseline if r['baseline']==name};rr=rows;y=np.array([r['h_w'] for r in rr]);p=np.array([bi[r['state_id']]['prediction'] for r in rr]);ensembles[('full',group,'mlp')]=(rr,y,p);met=harm_metrics(y,p,[r['cohort']=='write_harmful_flip' for r in rr]);metrics.append(dict(population='full',feature_group=group,model='mlp',**met))
        pd.DataFrame([met]).to_csv(OUT/f"learnability/{'generic_prestate' if name=='generic' else 'one_step_write_delta'}_baseline.csv",index=False)
        mask=y!=0;pd.DataFrame([harm_metrics(y[mask],p[mask])]).to_csv(OUT/f'learnability/{name}_legacy_nonzero_metrics.csv',index=False)
    frame=pd.DataFrame(metrics);frame.to_csv(OUT/'learnability/all_local_metrics.csv',index=False);atomic_jsonl(OUT/'learnability/oof_predictions.jsonl',predictions);pd.DataFrame(seedmetrics).to_csv(OUT/'statistics/seed_metrics.csv',index=False)
    for name,subset in [('nuisance_baseline',frame[frame.feature_group=='NUISANCE']),('write_linear_metrics',frame[(frame.population=='full')&(frame.model=='linear')]),('write_mlp_metrics',frame[(frame.population=='full')&(frame.model=='mlp')]),('feature_group_ablation',frame[frame.feature_group.str.startswith('F')]),('generic_plus_write_features',frame[frame.feature_group=='FUSION']),('dense_w_only_metrics',frame[frame.population=='dense_w'])]:subset.to_csv(OUT/f'learnability/{name}.csv',index=False)
    frame[frame.population=='matched'].to_csv(OUT/'matched_mechanism/matched_probe_metrics.csv',index=False)
    differences=[];gates=[]
    for pop,basenames in [('full',['B1_GENERIC','B2_DELTA']),('dense_w',['PRE','DELTA'])]:
        rr,y,p=ensembles[(pop,'F_ALL','mlp')];met=harm_metrics(y,p);checks=[]
        for baseline_name in basenames:
            br,by,bp=ensembles[(pop,baseline_name,'mlp')];assert [r['state_id'] for r in br]==[r['state_id'] for r in rr] and np.array_equal(y,by)
            comp=bootstrap_comparison(rr,y,p,bp);differences.append(dict(population=pop,model='F_ALL/mlp',baseline=baseline_name,**comp));passed=(comp['spearman_gain']>=.1 and comp['spearman_ci_low']>0) or (comp['harmful_auc_gain']>=.08 and comp['harmful_auc_ci_low']>0);checks.append(passed);print('local gate comparison',pop,baseline_name,comp,flush=True)
        high=met['recall_at_precision_0.9']>=.05 and met['precision_at_0.1']>=met['harmful_prevalence']+.1
        gates.append(dict(population=pop,material_over_both=all(checks),useful_high_precision=high,passes=all(checks) and high,metrics=met))
    pd.DataFrame(differences).to_csv(OUT/'statistics/pairwise_model_differences.csv',index=False);pd.DataFrame(differences).to_csv(OUT/'statistics/group_bootstrap_ci.csv',index=False)
    positive=all(g['passes'] for g in gates);decision={'category':'W-LOCAL-POSITIVE' if positive else 'W-LOCAL-WEAK','gates':gates,'propagation_authorized_by_plan':not positive,'contract_sha256':c['contract_sha256']};atomic_json(OUT/'learnability/local_decision_gate.json',decision)
    (OUT/'learnability/local_decision_gate.md').write_text('# '+decision['category']+'\n\n'+pd.DataFrame(differences).to_markdown(index=False)+'\n\n'+json.dumps(gates,indent=2)+'\n\n'+('Stop before propagation; gated transfer only.' if positive else 'Proceed only to the user-authorized W3 propagation stage. No search or controller.')+'\n')
    rr,y,p=ensembles[('full','F_ALL','mlp')];df=pd.DataFrame(rr);df['prediction']=p;df['depth_bin']=pd.cut(df.layer,[-1,8,18,27],labels=['Early','Middle','Late']);df['relative_bin']=pd.cut(df.trigger_relative_depth,[-1,0,2,5,27],labels=['d0','d1_2','d3_5','d6_plus'])
    for cols,name in [(['layer'],'absolute_layer_breakdown'),(['depth_bin'],'layer_breakdown'),(['relative_bin'],'trigger_relative_breakdown'),(['dataset','source_regime'],'dataset_source_breakdown')]:
        result=[]
        for key,g in df.groupby(cols,observed=True):
            if not isinstance(key,tuple):key=(key,)
            result.append({**dict(zip(cols,key)),**harm_metrics(g.h_w,g.prediction,g.cohort=='write_harmful_flip')})
        pd.DataFrame(result).to_csv(OUT/f'learnability/{name}.csv',index=False)
    df.groupby('cohort').agg(states=('state_id','size'),median_predicted_H_W=('prediction','median'),mean_predicted_H_W=('prediction','mean')).reset_index().to_csv(OUT/'learnability/strong_flip_ranking.csv',index=False)
    atomic_json(OUT/'learnability/complete.json',{'passed':True,'tasks':len(tasks),'unestimable_matched_tasks':len(incomplete),'decision':decision['category']});print(decision['category'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['structure','prepare-training','train','aggregate']);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    if a.command=='structure':structure()
    elif a.command=='prepare-training':prepare_training()
    elif a.command=='train':train(a.rank)
    else:aggregate()
