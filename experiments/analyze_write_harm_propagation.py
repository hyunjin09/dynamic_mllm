"""Fixed W3 OOF probes on native and exact H8 common support."""
from __future__ import annotations
import argparse,json,time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from experiments.run_write_harm_structure_learnability import *
from experiments.run_write_harm_propagation import check
from dense_failure_stage2.write_bootstrap import bootstrap_comparison
from experiments.analyze_read_harm_structure_learnability import _role_indices,_uid_weights
from dense_failure_stage2.read_harm_learnability import fit_predict_fold
from dense_failure_stage2.read_short_horizon import construct_horizon_feature
from dense_failure_stage2.write_harm_learnability import harm_metrics


def prepare():
    c=check();assert read_json(OUT/'propagation/extraction_complete.json')['passed'];tasks=[]
    for h in HORIZONS:
        for support in ['native','common']:
            if h==8 and support=='common':continue # Identical support: share exact fitted prediction.
            for population in ['full','dense_w']:
                for condition in c['config']['propagation']['conditions']:
                    for fold in range(5):
                        for seed in SEEDS:tasks.append(dict(task_index=len(tasks),horizon=h,support=support,population=population,condition=condition,fold=fold,seed=seed))
    atomic_jsonl(OUT/'work/propagation_tasks.jsonl',tasks);atomic_json(OUT/'propagation/training_protocol.json',{'tasks':len(tasks),'H8_deduplication':'common equals native exactly; reuse same fits','model':'fixed Phase83 MLP all conditions/horizons','primary_emergence':'common support delta versus its H0; Full and DenseW refits','pair_family':'secondary prespecified control, never replace primary after outcomes','matched_permutation_tests':'not required in WRITE plan','training_matrix_hashes':{h:file_sha256(OUT/f'work/propagation_h{h}_pooled.npy') for h in HORIZONS}})


def train(rank):
    c=check();cfg=c['config'];torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0');tasks=read_jsonl(OUT/'work/propagation_tasks.jsonl');dest=OUT/'work/propagation_fits';dest.mkdir(exist_ok=True);cache={};started=time.monotonic()
    for task in tasks:
        if task['task_index']%4!=rank:continue
        path=dest/f"task_{task['task_index']:04d}.json"
        if path.exists():continue
        key=(task['horizon'],task['support'],task['population'],task['condition'])
        if key not in cache:
            h,support,pop,condition=key;rows=read_jsonl(OUT/f'work/propagation_h{h}_rows.jsonl');ii=[i for i,r in enumerate(rows) if (support=='native' or r['layer']<=19) and (pop=='full' or r['dense_wrong'])];rr=[rows[i] for i in ii];pooled=torch.from_numpy(np.array(np.load(OUT/f'work/propagation_h{h}_pooled.npy',mmap_mode='r')[ii],copy=True));x=construct_horizon_feature(pooled[:,0],pooled[:,1],condition);cache={key:(rr,x)}
        rr,x=cache[key];roles=_role_indices(cfg,rr,task['fold']);target=np.array([r['h_w'] for r in rr]);result=fit_predict_fold(x,target,fit_indices=roles['fit'],calibration_indices=roles['calibration'],test_indices=roles['outer_test'],fit_weights=_uid_weights(rr,roles['fit']),calibration_weights=_uid_weights(rr,roles['calibration']),kind='mlp',spec=cfg['training']['mlp'],hidden_size=128,dropout=.1,target_scale_floor=.0001,gradient_clip_norm=1.,seed=task['seed'],device=device,classification=False)
        prediction=[dict(state_id=rr[int(i)]['state_id'],truth=float(target[int(i)]),prediction=float(v)) for i,v in zip(result['test_indices'],result['test_prediction'])]
        atomic_torch(dest/f"task_{task['task_index']:04d}_normalization.pt",{'mean':result['standardizer_mean'],'std':result['standardizer_std']});atomic_json(path,{**task,'status':'complete','predictions':prediction,'best_epoch':result['best_epoch'],'history':result['history'],'fit_states':len(roles['fit']),'calibration_states':len(roles['calibration']),'test_states':len(prediction)})
        print(json.dumps({'rank':rank,'task':task['task_index'],'horizon':task['horizon'],'support':task['support'],'population':task['population'],'condition':task['condition'],'elapsed':time.monotonic()-started}),flush=True)
    atomic_json(OUT/f'work/propagation_training_rank{rank}_complete.json',{'passed':True,'rank':rank,'elapsed':time.monotonic()-started})


def aggregate():
    c=check();tasks=read_jsonl(OUT/'work/propagation_tasks.jsonl');rows=read_jsonl(OUT/'population/dense_write_state_manifest.jsonl');byid={r['state_id']:r for r in rows};values=defaultdict(lambda:defaultdict(dict));metrics=[];predictions=[];seedmetrics=[];ensembles={}
    for t in tasks:
        result=read_json(OUT/f"work/propagation_fits/task_{t['task_index']:04d}.json");assert result['status']=='complete';key=(t['horizon'],t['support'],t['population'],t['condition'])
        for p in result['predictions']:
            assert t['seed'] not in values[key][p['state_id']];assert abs(p['truth']-byid[p['state_id']]['h_w'])<1e-12;values[key][p['state_id']][t['seed']]=p['prediction']
    for key,v in list(values.items()):
        if key[0]==8:values[(8,'common',*key[2:])]=v
    for key,v in values.items():
        h,support,pop,condition=key;ids=sorted(v);rr=[byid[s] for s in ids];expected=[r for r in rows if r['layer']+h<28 and (support=='native' or r['layer']<=19) and (pop=='full' or r['dense_wrong'])];assert len(rr)==len(expected) and all(len(v[s])==3 for s in ids)
        y=np.array([r['h_w'] for r in rr]);p=np.array([np.mean(list(v[s].values())) for s in ids]);ensembles[key]=(rr,y,p);meta=dict(horizon=h,support=support,population=pop,condition=condition);metrics.append({**meta,**harm_metrics(y,p,[r['cohort']=='write_harmful_flip' for r in rr])})
        for seed in SEEDS:seedmetrics.append({**meta,'seed':seed,**harm_metrics(y,[v[s][seed] for s in ids])})
        predictions.extend({**meta,'state_id':s,'truth':float(yy),'prediction':float(pp)} for s,yy,pp in zip(ids,y,p))
    df=pd.DataFrame(metrics);df.to_csv(OUT/'propagation/horizon_metrics.csv',index=False);df[df.population=='dense_w'].to_csv(OUT/'propagation/dense_w_horizon_metrics.csv',index=False);df[['horizon','support','population','condition','states','harmful_flip_auroc','harmful_flip_auprc']].to_csv(OUT/'propagation/strong_flip_horizon_metrics.csv',index=False);df.to_csv(OUT/'propagation/single_vs_counterfactual_controls.csv',index=False);pd.DataFrame(seedmetrics).to_csv(OUT/'statistics/propagation_seed_metrics.csv',index=False)
    # Large OOF table lives on allowed external storage.
    predpath=EXT/'propagation/oof_predictions.jsonl';atomic_jsonl(predpath,predictions);link=OUT/'propagation/oof_predictions.jsonl'
    if not link.exists():link.symlink_to(predpath)
    differences=[];gates=[]
    def comparison(pop,h,left,rh,right):
        rr,y,p=ensembles[(h,'common',pop,left)];br,by,bp=ensembles[(rh,'common',pop,right)];assert [r['state_id'] for r in rr]==[r['state_id'] for r in br] and np.array_equal(y,by);result=bootstrap_comparison(rr,y,p,bp);differences.append(dict(population=pop,horizon=h,left=left,reference_horizon=rh,right=right,**result));return result
    def material(d,rho=.1,auc=.08):return (d['spearman_gain']>=rho and d['spearman_ci_low']>0) or (d['harmful_auc_gain']>=auc and d['harmful_auc_ci_low']>0)
    def high(pop,h,condition):
        rr,y,p=ensembles[(h,'common',pop,condition)];m=harm_metrics(y,p);return m['recall_at_precision_0.9']>=.05 and m['precision_at_0.1']>=m['harmful_prevalence']+.1
    for h in [1,2,4,8]:
        perpop=[]
        for pop in ['full','dense_w']:
            emergence=comparison(pop,h,'delta',0,'delta');advantages=[comparison(pop,h,'delta',h,single) for single in ['on','off']];singlechecks=[]
            for single in ['on','off']:singlechecks.append(material(comparison(pop,h,single,0,single)) and high(pop,h,single))
            perpop.append(dict(population=pop,delta_emerges=material(emergence),delta_beats_both_singles=all(material(d,.05,.04) for d in advantages),delta_high_precision=high(pop,h,'delta'),delta_increment_bounded_small=all(d['spearman_ci_high']<.05 and d['harmful_auc_ci_high']<.04 for d in advantages),single_checks=singlechecks,single_material_and_high_precision=any(singlechecks)))
        a=all(p['delta_emerges'] and p['delta_beats_both_singles'] and p['delta_high_precision'] for p in perpop);b=any(all(p['single_checks'][j] for p in perpop) for j in range(2)) and not a and all(p['delta_increment_bounded_small'] for p in perpop);gates.append(dict(horizon=h,passes_A=a,passes_B=b,populations=perpop));print('propagation gate',h,json.dumps(gates[-1]),flush=True)
    # An emergence must be present at the same horizon in full and DenseW.
    aa=[g['horizon'] for g in gates if g['passes_A']];bb=[g['horizon'] for g in gates if g['passes_B']];secondary_flags=[]
    for pop in ['full','dense_w']:
        for condition in c['config']['propagation']['conditions']:
            base=df[(df.support=='common')&(df.population==pop)&(df.condition==condition)&(df.horizon==0)].iloc[0]
            for h in HORIZONS:
                row=df[(df.support=='common')&(df.population==pop)&(df.condition==condition)&(df.horizon==h)].iloc[0]
                if high(pop,h,condition) or row.spearman-base.spearman>=.1 or row.harmful_auroc-base.harmful_auroc>=.08:secondary_flags.append(dict(population=pop,condition=condition,horizon=h,reason='material point gain or useful precision; prevents blanket weak claim, does not authorize promotion'))
    category='W-PROP-A' if aa else 'W-PROP-B' if bb else 'W-PROP-INCONCLUSIVE' if secondary_flags else 'W-PROP-C';decision={'category':category,'smallest_successful_horizon':min(aa or bb) if aa or bb else None,'gates':gates,'mixed_evidence_flags':secondary_flags,'B_no_increment_rule':'both delta-minus-single CI upper bounds below .05 rho and .04 AUROC for both singles/full/DenseW','interpretation_scope':'fixed pooled representations and fixed predictor; secondary pair/family positives must be qualified, not promoted posthoc','generalization_gate':bool(aa or bb)}
    atomic_json(OUT/'propagation/propagation_decision.json',decision);(OUT/'propagation/propagation_decision.md').write_text('# '+category+'\n\n'+json.dumps(decision,indent=2)+'\n');pd.DataFrame(differences).to_csv(OUT/'statistics/propagation_pairwise_differences.csv',index=False);atomic_json(OUT/'propagation/complete.json',{'passed':True,'tasks':len(tasks),'decision':category});print(category,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','aggregate']);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='train':train(a.rank)
    else:aggregate()
