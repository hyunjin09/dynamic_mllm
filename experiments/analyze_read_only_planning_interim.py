"""PRELIMINARY, immutable completed-UID snapshot and descriptive interim analysis.

No GPU calls, search changes, partially evaluated UID policies, or population
extrapolation. This module is separate from the frozen execution/final analysis.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dense_failure_stage2.read_only_planning import ALGORITHMS, SEEDS, curve, enumeration, route_hash

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'analysis/read_only_bounded_planning'
EXT=Path('/mnt/hyemin/qwen_train_eval/outputs/read_only_bounded_planning_v1')
BUDGETS=(4,8,16,32,64,128)
STATUS='PRELIMINARY'
EXPECTED={(a,s) for a in ALGORITHMS for s in (SEEDS if a.startswith('random') or a=='binary_mcts' else SEEDS[:1])}


def digest(data): return sha256(data).hexdigest()
def slug(uid): return sha256(uid.encode()).hexdigest()[:24]
def jread(p): return json.loads(Path(p).read_text())
def jlines(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def jwrite(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def save(root,name,data):
    d=pd.DataFrame(data) if not isinstance(data,pd.DataFrame) else data.copy()
    d.insert(0,'status',STATUS)
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False)
    return d


def read_stable(p):
    p=Path(p);before=p.stat();data=p.read_bytes();after=p.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise RuntimeError(f'source changed during snapshot: {p}')
    return data


def validated_routes(records,uid,pop,algorithm,seed,execution_contract,parent_contract):
    if not records:raise ValueError('empty completed route history')
    trigger=pop['first_trigger_layer']
    assert len({x['suffix'] for x in records})==len(records)
    for i,r in enumerate(records,1):
        assert r['uid']==uid and r['trigger_layer']==trigger
        assert r['route_hash']==route_hash(uid,trigger,r['suffix'])
        assert r['evaluation_rank']==i and r['search_algorithm']==algorithm and r['seed']==seed
        assert isinstance(r['correct'],bool) and math.isfinite(r['q'])
        if r['cache_source']=='stepA':assert r['source_contract']==parent_contract
        else:assert r['contract_sha256']==execution_contract
    assert records[0]['suffix']=='1'*pop['suffix_length']
    assert records[0]['correct']==pop['dense_correct']


def cohort_audit(out,pop,completed,h1):
    meta=[]
    for uid,r in pop.items():
        meta.append(dict(uid=uid,dense_group='W' if r['dense_wrong'] else 'C',cohort='completed' if uid in completed else 'pending',dataset=r['dataset'],source=r['source_regime'],dataset_source=r['source_regime']+'/'+r['dataset'],trigger=r['first_trigger_layer'],T=r['suffix_length'],hamming1_rescued=bool(h1[uid]),trigger_bin='0' if r['first_trigger_layer']==0 else '1-8' if r['first_trigger_layer']<=8 else '9-18' if r['first_trigger_layer']<=18 else '19-27'))
    df=pd.DataFrame(meta);save(out,'cohort_membership.csv',df)
    numeric=[];categorical=[]
    for group in ['ALL','W','C']:
        part=df if group=='ALL' else df[df.dense_group==group]
        for feature in ['trigger','T','hamming1_rescued']:
            summary={}
            for cohort in ['completed','pending']:
                vals=part.loc[part.cohort==cohort,feature].astype(float)
                summary[cohort]=vals
                numeric.append(dict(group=group,feature=feature,cohort=cohort,n=len(vals),mean=float(vals.mean()) if len(vals) else None,median=float(vals.median()) if len(vals) else None,p25=float(vals.quantile(.25)) if len(vals) else None,p75=float(vals.quantile(.75)) if len(vals) else None,total=float(vals.sum()) if len(vals) else 0))
            a,b=summary['completed'],summary['pending']
            if len(a) and len(b):
                scale=math.sqrt((float(a.var(ddof=0))+float(b.var(ddof=0)))/2)
                numeric.append(dict(group=group,feature=feature,cohort='completed_minus_pending',n=len(part),mean=float(a.mean()-b.mean()),median=None,p25=None,p75=None,total=None,standardized_mean_difference=float((a.mean()-b.mean())/scale) if scale else None))
        for feature in ['dataset','source','dataset_source','trigger_bin','trigger','T']:
            for value in sorted(part[feature].unique()):
                c=part[part.cohort=='completed'];p=part[part.cohort=='pending'];nc=int((c[feature]==value).sum());npending=int((p[feature]==value).sum())
                categorical.append(dict(group=group,feature=feature,value=value,completed_n=nc,completed_denominator=len(c),completed_fraction=nc/len(c) if len(c) else None,pending_n=npending,pending_denominator=len(p),pending_fraction=npending/len(p) if len(p) else None,difference_pp=100*(nc/len(c)-npending/len(p)) if len(c) and len(p) else None))
    save(out,'cohort_numeric_comparison.csv',numeric);save(out,'cohort_categorical_comparison.csv',categorical)
    return df,pd.DataFrame(numeric),pd.DataFrame(categorical)


def snapshot():
    cutoff=time.time_ns();stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/'interim'/f'PRELIMINARY_{stamp}';raw=EXT/'interim'/f'PRELIMINARY_{stamp}'
    out.mkdir(parents=True,exist_ok=False);raw.mkdir(parents=True,exist_ok=False);(out/'snapshot').symlink_to(raw,target_is_directory=True)
    contract=jread(ROOT/'frozen_contract.json');parent=contract['parent_contract'];ch=contract['contract_sha256']
    pop={r['uid']:r for g in ['w','c'] for r in jlines(ROOT/f'population/triggered_{g}_manifest.jsonl')}
    h1df=pd.read_csv(ROOT/'single_off/hamming1_uid_summary.csv');h1=dict(zip(h1df.uid,h1df.any_single_correct))
    work={};evidence=[]
    # Freeze completion membership before reading any search outcomes.
    for p in sorted((EXT/'work/128').glob('*.json')):
        if p.name.startswith('rank') or p.stat().st_mtime_ns>cutoff:continue
        data=read_stable(p);r=json.loads(data)
        assert r['complete'] and r['contract_sha256']==ch and r['uid'] in pop
        assert {(x['algorithm'],x['seed']) for x in r['sessions']}==EXPECTED
        work[r['uid']]=r;(raw/'completion').mkdir(exist_ok=True);(raw/'completion'/p.name).write_bytes(data)
        evidence.append(dict(source=str(p),sha256=digest(data),snapshot='completion/'+p.name))
    df,num,cat=cohort_audit(out,pop,set(work),h1)
    common=sorted(u for u in work if pop[u]['dense_wrong'])
    if not common:raise RuntimeError('no fully completed common W cohort')
    # Freeze eligible completed Hamming2-file membership at the same timestamp.
    h2elig=set(jread(ROOT/'hamming2/eligibility.json')['uids'])
    h2files=[p for p in sorted((EXT/'search/enumeration').glob('*.jsonl')) if p.stat().st_mtime_ns<=cutoff]
    def copy_uid(uid):
        r=pop[uid];histories=[];sources=[]
        for session in sorted(work[uid]['sessions'],key=lambda x:(x['algorithm'],x['seed'])):
            a,s=session['algorithm'],session['seed'];p=EXT/f'search/{a}/{s}/{slug(uid)}.jsonl';data=read_stable(p)
            full=[json.loads(x) for x in data.splitlines() if x.strip()];count=session['routes'];assert len(full)>=count
            records=full[:count];validated_routes(records,uid,r,a,s,ch,parent)
            observed=curve(records,128)
            assert observed==session['curve'],(uid,a,s,observed,session['curve'])
            histories.append(dict(algorithm=a,seed=s,exhausted=session['exhausted'],rows=records))
            sources.append(dict(source=str(p),sha256=digest(data),base_rows=count))
        dest=raw/f'completed_w/{slug(uid)}.json';jwrite(dest,dict(status=STATUS,uid=uid,histories=histories))
        return dict(uid=uid,snapshot=str(dest),sha256=digest(dest.read_bytes()),sources=sources)
    with ThreadPoolExecutor(max_workers=8) as executor:uid_evidence=list(executor.map(copy_uid,common))
    h2evidence=[]
    for p in h2files:
        data=read_stable(p);rr=[json.loads(x) for x in data.splitlines() if x.strip()];uid=rr[0]['uid']
        if uid not in h2elig:continue
        r=pop[uid];assert r['dense_wrong'] and not h1[uid]
        validated_routes(rr,uid,r,'enumeration',SEEDS[0],ch,parent)
        assert [x['suffix'] for x in rr]==list(enumeration(r['suffix_length'],2))
        target=raw/f'hamming2/{slug(uid)}.json';jwrite(target,dict(status=STATUS,uid=uid,rows=rr))
        h2evidence.append(dict(uid=uid,source=str(p),source_sha256=digest(data),snapshot=str(target),sha256=digest(target.read_bytes()),expected_distance2=math.comb(r['suffix_length'],2),in_common_w=uid in common))
    inputs={}
    for name,path in [('execution_contract',ROOT/'frozen_contract.json'),('analysis_contract',ROOT/'analysis_contract.json'),('hamming1',ROOT/'single_off/hamming1_uid_summary.csv'),('w_manifest',ROOT/'population/triggered_w_manifest.jsonl'),('c_manifest',ROOT/'population/triggered_c_manifest.jsonl'),('hamming2_eligibility',ROOT/'hamming2/eligibility.json')]:
        data=read_stable(path);dest=raw/'inputs'/path.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data);inputs[name]=dict(source=str(path),sha256=digest(data),snapshot=str(dest))
    manifest=dict(status=STATUS,cutoff_unix_ns=cutoff,cutoff_local=datetime.fromtimestamp(cutoff/1e9).astimezone().isoformat(),execution_contract_sha256=ch,common_w_uids=common,completed_all_uids=sorted(work),population_total=len(pop),population_w=sum(x['dense_wrong'] for x in pop.values()),uid_evidence=uid_evidence,hamming2_evidence=h2evidence,completion_evidence=evidence,inputs=inputs,budgets=BUDGETS,seeds=SEEDS,cohort_rule='only atomic complete base128 UID records with all13 algorithm/seed sessions; exact intersection; no partial UID inclusion',statistics='5000 paired UID bootstrap draws, seed-averaged within UID, exploratory unadjusted intervals; no population extrapolation')
    jwrite(out/'snapshot_manifest.json',manifest)
    pre='# PRELIMINARY: completed-versus-pending cohort verification\n\nSnapshot '+manifest['cutoff_local']+f'. Common completed W={len(common)}; pending W={manifest["population_w"]-len(common)}. Completion is schedule-selected, not a random sample.\n\n'
    pre+=cat[(cat.group=='W')&(cat.feature=='dataset_source')].to_markdown(index=False)+'\n\n'+num[(num.group=='W')&(num.cohort!='completed_minus_pending')].to_markdown(index=False)+'\n\nTrigger and suffix length are deterministically related: T=28-trigger. No hypothesis test is needed to establish observed cohort composition differences.\n'
    (out/'PRELIMINARY_cohort_verification.md').write_text(pre)
    print('SNAPSHOT',str(out),flush=True);print(pre,flush=True)
    return out


def bootstrap(values):
    v=np.asarray(values,dtype=float);rng=np.random.default_rng(20260913);sample=[]
    for _ in range(50):
        ix=rng.integers(0,len(v),size=(100,len(v)));sample.extend(v[ix].mean(axis=1).tolist())
    return float(np.quantile(sample,.025)),float(np.quantile(sample,.975))


def analyze(out):
    out=Path(out);m=jread(out/'snapshot_manifest.json');uids=m['common_w_uids'];N=len(uids)
    for inp in m['inputs'].values():assert digest(Path(inp['snapshot']).read_bytes())==inp['sha256']
    h1frame=pd.read_csv(m['inputs']['hamming1']['snapshot']);h1=dict(zip(h1frame.uid,h1frame.any_single_correct))
    histories={};exhaust={};perseed=[];first=[]
    for entry in m['uid_evidence']:
        data=Path(entry['snapshot']).read_bytes();assert digest(data)==entry['sha256'];u=entry['uid']
        for h in json.loads(data)['histories']:
            a,s,rr=h['algorithm'],h['seed'],h['rows'];histories[(u,a,s)]=rr;exhaust[(u,a,s)]=h['exhausted']
            for b in BUDGETS:perseed.append(dict(uid=u,algorithm=a,seed=s,exhausted=h['exhausted'],exact_budget_supported=len(rr)>=b,**curve(rr,b)))
            f=next((r['evaluation_rank'] for r in rr if r['correct']),None)
            first.append(dict(uid=u,algorithm=a,seed=s,first_rescue_rank=f,censored=f is None,censor_at_actual_evaluations=len(rr),algorithm_exhausted=h['exhausted']))
    df=pd.DataFrame(perseed);assert len(df.uid.unique())==N and len(df)==N*13*len(BUDGETS)
    save(out,'per_uid_seed_budget_metrics.csv',df)
    uid=df.groupby(['uid','algorithm','budget'],as_index=False).agg(any_correct=('any_correct','mean'),selected_correct=('selected_correct','mean'),actual_unique_routes=('actual_unique_routes','mean'),exact_budget_support=('exact_budget_supported','mean'))
    uid['success']=uid.selected_correct;uid['ranking_failure']=uid.any_correct-uid.selected_correct;uid['generation_failure']=1-uid.any_correct
    assert np.allclose(uid[['success','ranking_failure','generation_failure']].sum(axis=1),1)
    save(out,'per_uid_budget_metrics.csv',uid)
    main=uid.groupby(['algorithm','budget'],as_index=False).agg(uids=('uid','nunique'),ANY_CORRECT=('any_correct','mean'),SELECTED_CORRECT=('selected_correct','mean'),ranking_failure=('ranking_failure','mean'),generation_failure=('generation_failure','mean'),mean_actual_evaluations=('actual_unique_routes','mean'),exact_budget_supported_fraction=('exact_budget_support','mean'))
    for i,row in main.iterrows():
        part=uid[(uid.algorithm==row.algorithm)&(uid.budget==row.budget)];lo,hi=bootstrap(part.any_correct);main.loc[i,'ANY_ci_low']=lo;main.loc[i,'ANY_ci_high']=hi
    save(out,'planning_curves.csv',main)
    save(out,'per_seed_curves.csv',df.groupby(['algorithm','seed','budget'],as_index=False).agg(uids=('uid','nunique'),ANY_CORRECT=('any_correct','mean'),SELECTED_CORRECT=('selected_correct','mean'),actual_mean=('actual_unique_routes','mean')))
    differences=[];pairrows=[]
    for a in ('beam2','beam4','beam8'):
        for other in ('random_uniform','random_sparse'):
            for b in BUDGETS:
                nominal=[];matched=[];nom_selected=[];matched_selected=[];caps=[]
                for u in uids:
                    ar=histories[(u,a,SEEDS[0])];br=[histories[(u,other,s)] for s in SEEDS]
                    cap=min([b,len(ar)]+[len(x) for x in br]);aa=curve(ar,b);bb=[curve(x,b) for x in br];ma=curve(ar,cap);mb=[curve(x,cap) for x in br]
                    nd=float(aa['any_correct'])-np.mean([x['any_correct'] for x in bb]);md=float(ma['any_correct'])-np.mean([x['any_correct'] for x in mb]);ns=float(aa['selected_correct'])-np.mean([x['selected_correct'] for x in bb]);ms=float(ma['selected_correct'])-np.mean([x['selected_correct'] for x in mb])
                    nominal.append(nd);matched.append(md);nom_selected.append(ns);matched_selected.append(ms);caps.append(cap)
                    pairrows.append(dict(uid=u,beam=a,random=other,budget=b,shared_actual_cap=cap,nominal_ANY_difference=nd,matched_ANY_difference=md,matched_SELECTED_difference=ms))
                for label,values,sel in [('nominal_at_most_B',nominal,nom_selected),('matched_actual_cap',matched,matched_selected)]:
                    lo,hi=bootstrap(values);slo,shi=bootstrap(sel)
                    differences.append(dict(beam=a,random=other,budget=b,comparison=label,uids=N,ANY_difference=float(np.mean(values)),ANY_ci_low=lo,ANY_ci_high=hi,SELECTED_difference=float(np.mean(sel)),SELECTED_ci_low=slo,SELECTED_ci_high=shi,mean_shared_actual_cap=float(np.mean(caps))))
    save(out,'beam_vs_random.csv',differences);save(out,'beam_vs_random_per_uid.csv',pairrows)
    gains=[];gainuids=[]
    for a in ALGORITHMS:
        for lo,hi in zip(BUDGETS,BUDGETS[1:]):
            low=uid[(uid.algorithm==a)&(uid.budget==lo)].set_index('uid').loc[uids];high=uid[(uid.algorithm==a)&(uid.budget==hi)].set_index('uid').loc[uids]
            delta=high.any_correct-low.any_correct;selected=high.selected_correct-low.selected_correct;added=high.actual_unique_routes-low.actual_unique_routes;eligible=added>0
            dl,dh=bootstrap(delta);sl,sh=bootstrap(selected)
            assert (delta>=-1e-12).all()
            gains.append(dict(algorithm=a,from_budget=lo,to_budget=hi,uids=N,ANY_gain=float(delta.mean()),ANY_ci_low=dl,ANY_ci_high=dh,SELECTED_gain=float(selected.mean()),SELECTED_ci_low=sl,SELECTED_ci_high=sh,mean_added_evaluations=float(added.mean()),uids_with_any_new_evaluations=int(eligible.sum()),ANY_gain_on_extension_capable=float(delta[eligible].mean()) if eligible.any() else None,relative_rescue_increase=float(delta.mean()/low.any_correct.mean()) if low.any_correct.mean()>0 else None))
            for u in uids:gainuids.append(dict(uid=u,algorithm=a,from_budget=lo,to_budget=hi,ANY_gain=delta.loc[u],SELECTED_gain=selected.loc[u],added_evaluations=added.loc[u]))
    save(out,'marginal_gains.csv',gains);save(out,'marginal_gains_per_uid.csv',gainuids)
    f=pd.DataFrame(first);save(out,'first_rescue_ranks.csv',f)
    fuid=f.groupby(['uid','algorithm'],as_index=False).agg(mean_observed_seed_rank=('first_rescue_rank','mean'),seed_runs=('seed','count'),rescued_seed_runs=('first_rescue_rank','count'),min_actual_evaluations=('censor_at_actual_evaluations','min'),max_actual_evaluations=('censor_at_actual_evaluations','max'))
    fuid['qualification']='one UID; rank averaged only over observed-rescue seeds; censored seeds counted separately'
    save(out,'first_rescue_per_uid.csv',fuid)
    first_summary=[]
    for a,part in fuid.groupby('algorithm'):
        v=part.mean_observed_seed_rank.dropna();raw=f[f.algorithm==a];rankseed=raw.groupby('seed').first_rescue_rank.median().dropna()
        first_summary.append(dict(algorithm=a,cohort_uids=N,uids_with_rescue_in_any_seed=len(v),uids_with_no_observed_rescue=N-len(v),median=float(v.median()) if len(v) else None,p75=float(v.quantile(.75)) if len(v) else None,p90=float(v.quantile(.90)) if len(v) else None,min_seed_conditional_median=float(rankseed.min()) if len(rankseed) else None,max_seed_conditional_median=float(rankseed.max()) if len(rankseed) else None))
    save(out,'first_rescue_distribution.csv',first_summary)
    save(out,'first_rescue_distribution_per_seed.csv',f.groupby(['algorithm','seed'],as_index=False).agg(uids=('uid','count'),rescued_uids=('first_rescue_rank','count'),median=('first_rescue_rank','median'),p75=('first_rescue_rank',lambda x:x.quantile(.75)),p90=('first_rescue_rank',lambda x:x.quantile(.90))))
    h2rows=[]
    for entry in m['hamming2_evidence']:
        data=Path(entry['snapshot']).read_bytes();assert digest(data)==entry['sha256'];v=json.loads(data);rr=v['rows'];h2=[x for x in rr if x['suffix'].count('0')==2]
        assert len(h2)==entry['expected_distance2'] and not h1[v['uid']]
        assert not any(x['correct'] for x in rr if x['suffix'].count('0')<=1)
        top=max(rr,key=lambda x:(x['q'],-x['evaluation_rank']))
        h2rows.append(dict(uid=v['uid'],in_common_w=v['uid'] in uids,complete_census=True,distance2_routes=len(h2),ANY_CORRECT=any(x['correct'] for x in h2),SELECTED_CORRECT=top['correct'],correct_distance2_routes=sum(x['correct'] for x in h2)))
    h2=pd.DataFrame(h2rows);save(out,'hamming2_complete_census_uids.csv',h2)
    h2summary=[]
    for label,part in [('all_completed_H2_censuses',h2),('common_completed_W_H1_unrescued',h2[h2.in_common_w])]:
        h2summary.append(dict(cohort=label,uids=len(part),actual_distance2_routes=int(part.distance2_routes.sum()),rescued_uids=int(part.ANY_CORRECT.sum()),ANY_CORRECT=float(part.ANY_CORRECT.mean()) if len(part) else None,SELECTED_CORRECT=float(part.SELECTED_CORRECT.mean()) if len(part) else None,uids_with_zero_distance2_routes=int((part.distance2_routes==0).sum())))
    save(out,'hamming2_summary.csv',h2summary)
    # A measured within-cohort union, explicitly neither high-budget ceiling nor population estimate.
    union={u:any(x['correct'] for (v,a,s),rr in histories.items() if v==u for x in rr) or bool(h1[u]) for u in uids}
    for r in h2rows:
        if r['uid'] in union:union[r['uid']]|=r['ANY_CORRECT']
    # Standalone PRELIMINARY plots, immutable common W denominator at every point.
    figures=out/'figures';figures.mkdir(exist_ok=True)
    for metric,filename,ylabel in [('ANY_CORRECT','PRELIMINARY_any_correct.png','ANY_CORRECT (%)'),('SELECTED_CORRECT','PRELIMINARY_selected_correct.png','q-SELECTED_CORRECT (%)')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for a,part in main.groupby('algorithm'):
            ax.plot(part.budget,part[metric]*100,marker='o',label=a)
        ax.set_xscale('log',base=2);ax.set_xticks(BUDGETS,labels=BUDGETS);ax.set_ylabel(ylabel);ax.set_xlabel('Maximum unique route evaluations (actual support reported separately)');ax.set_title(f'PRELIMINARY — common completed W cohort, n={N}');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(figures/filename,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for a,part in f.groupby('algorithm'):
        # This unconditional first-hit CDF equals seed-averaged ANY at each rank;
        # failures remain in the denominator rather than disappearing as censored.
        seeds=len(part.seed.unique());ranks=range(1,129);yy=[float((part.first_rescue_rank<=b).sum())/(N*seeds) for b in ranks]
        ax.plot(list(ranks),yy,label=a)
    ax.set_xscale('log',base=2);ax.set_title(f'PRELIMINARY first-rescue CDF, n={N} W UIDs');ax.set_xlabel('First correct unique-evaluation rank');ax.set_ylabel('Fraction of cohort rescued (seed mean)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'PRELIMINARY_first_rescue_cdf.png',dpi=180);plt.close(fig)
    ends=main[main.budget==128].set_index('algorithm');plot=ends[['SELECTED_CORRECT','ranking_failure','generation_failure']]*100
    ax=plot.plot.bar(stacked=True,figsize=(9,5),rot=25);ax.set_title(f'PRELIMINARY outcomes at maximum B=128, n={N}');ax.set_ylabel('Cohort percentage');ax.figure.tight_layout();ax.figure.savefig(figures/'PRELIMINARY_failure_decomposition.png',dpi=180);plt.close(ax.figure)
    for metric in ['ANY_CORRECT','SELECTED_CORRECT']:
        table=main.pivot(index='algorithm',columns='budget',values=metric).reindex(ALGORITHMS)*100
        save(out,f'{metric}_table_percent.csv',table.reset_index())
    result=dict(status=STATUS,snapshot=m['cutoff_local'],common_w=N,completed_all=len(m['completed_all_uids']),pending_w=m['population_w']-N,hamming1_rescued_common=sum(bool(h1[u]) for u in uids),measured_common_union_rescued=sum(union.values()),union_not_ceiling=True,hamming2_summary=h2summary,main_B128=ends.reset_index().to_dict('records'),last_increment=[x for x in gains if x['to_budget']==128],beam8_random_B128=[x for x in differences if x['beam']=='beam8' and x['budget']==128 and x['comparison']=='matched_actual_cap'])
    jwrite(out/'PRELIMINARY_result_summary.json',result)
    # Save a compact decision packet now; interpretation is reconciled separately.
    text='# PRELIMINARY interim analysis\n\nSnapshot '+m['cutoff_local']+f'; common completed W cohort n={N}. No partial UID or full-population extrapolation.\n\n'
    text+='## Completed versus pending\n\nSee PRELIMINARY_cohort_verification.md and cohort comparison CSVs.\n\n'
    text+='## ANY_CORRECT (%)\n\n'+(main.pivot(index='algorithm',columns='budget',values='ANY_CORRECT').reindex(ALGORITHMS)*100).round(2).to_markdown()+'\n\n'
    text+='## SELECTED_CORRECT (%)\n\n'+(main.pivot(index='algorithm',columns='budget',values='SELECTED_CORRECT').reindex(ALGORITHMS)*100).round(2).to_markdown()+'\n\n'
    text+='All denominators are the same completed W UIDs. Stochastic results average3 seeds within UID; they are not a best-seed/union rate. B means an at-most budget. Completed natural algorithm exhaustion belowB is retained and explicitly reported, never padded with fictional evaluations. Dense is candidate1.\n\n'
    text+='## PRELIMINARY marginal gains64→128\n\n'+pd.DataFrame(result['last_increment']).round(4).to_markdown(index=False)+'\n\n'
    text+='## PRELIMINARY matched-actual beam8 versus random at128\n\n'+pd.DataFrame(result['beam8_random_B128']).round(4).to_markdown(index=False)+'\n\n'
    text+='Paired5000-draw UID bootstrap intervals are exploratory and unadjusted for multiple comparisons or interim looks. Full beam2/4/8 comparisons and all budgets are in beam_vs_random.csv.\n\n'
    text+='## PRELIMINARY first-rescue distribution\n\n'+pd.DataFrame(first_summary).round(2).to_markdown(index=False)+'\n\nConditional ranks describe UIDs with at least one observed rescue. Censored seeds remain explicit in per-UID/per-seed files; the CDF figure retains every UID in the denominator.\n\n'
    text+='## PRELIMINARY ranking versus generation failures at128\n\n'+ends[['SELECTED_CORRECT','ranking_failure','generation_failure','mean_actual_evaluations','exact_budget_supported_fraction']].round(4).to_markdown()+'\n\n'
    text+='## PRELIMINARY complete Hamming2 censuses only\n\n'+pd.DataFrame(h2summary).round(4).to_markdown(index=False)+'\n\nThese are Hamming1-unrescued W UIDs whose entire prescribed distance2 census is complete. No partial census enters either numerator or denominator. No rate is applied to pending UIDs.\n\n'
    text+='## PRELIMINARY scope limits\n\nNo final P-READ or READ-ceiling category is assigned. Gold-answer q is an oracle. Pending UIDs and all not-yet-evaluated high-budget routes are unknown. Observed algorithm exhaustion and local budget flattening do not prove landscape or population saturation. Necessity assessment follows independent reconciliation in PRELIMINARY_interpretation.md.\n'
    (out/'PRELIMINARY_report.md').write_text(text)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['snapshot','analyze']);p.add_argument('--snapshot');a=p.parse_args()
    if a.command=='snapshot':snapshot()
    else:analyze(Path(a.snapshot))
