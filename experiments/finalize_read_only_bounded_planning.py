"""Aggregate Phase85 measured route histories; no model calls or new search."""
from __future__ import annotations
from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from dense_failure_stage1.historical_shortcut_audit import binary_auroc as roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dense_failure_stage2.read_only_planning import ALGORITHMS,BUDGETS,SEEDS,curve,complexity
from experiments.run_read_only_bounded_planning import ROOT,EXT,PARENT,slug,rows,verify,write_text
from experiments.run_predictability_stepA_measurement import atomic_json,atomic_jsonl,atomic_csv,file_sha256,utc_now


def save(name,data):
    p=ROOT/name; p.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(data,pd.DataFrame): data.to_csv(p,index=False)
    else: pd.DataFrame(data).to_csv(p,index=False)


def pair_boot(values,seed=20260912,draws=5000):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    draws_out=[]
    for _ in range(0,draws,100):
        idx=rng.integers(0,len(values),size=(min(100,draws-len(draws_out)),len(values)))
        draws_out.extend(values[idx].mean(1).tolist())
    return float(np.quantile(draws_out,.025)),float(np.quantile(draws_out,.975))


def matched_any(ar, br, budget):
    """Equal actual per-UID cap across both algorithms and all paired seeds."""
    cap=min([budget]+[len(x) for x in ar+br])
    if cap<1: raise ValueError('comparison requires evaluated Dense candidate')
    av=float(np.mean([curve(x,cap)['any_correct'] for x in ar]))
    bv=float(np.mean([curve(x,cap)['any_correct'] for x in br]))
    return cap,av,bv


def aggregate():
    contract=verify()
    analysis_contract=json.loads((ROOT/'analysis_contract.json').read_text())
    assert analysis_contract['execution_contract_sha256']==contract['contract_sha256']
    for path,digest in analysis_contract['files'].items():
        assert file_sha256(path)==digest, f'analysis contract drift: {path}'
    pop={r['uid']:r for g in ('w','c') for r in rows(ROOT/f'population/triggered_{g}_manifest.jsonl')}
    wids=sorted(u for u,r in pop.items() if r['dense_wrong'])
    # All base UIDs and all prospectively selected high-budget sessions must finish.
    for mode in ('128','256','512'):
        ids=list(pop) if mode=='128' else list(json.loads((ROOT/f'work/extension_{mode}.json').read_text()))
        for uid in ids:
            r=json.loads((EXT/f'work/{mode}/{slug(uid)}.json').read_text())
            assert r['complete'] and r['contract_sha256']==contract['contract_sha256']
    extensions={b:json.loads((ROOT/f'work/extension_{b}.json').read_text()) for b in (256,512)}
    measures=[]; actual_routes={}; first=[]; qquality=[]; multiplicity=[]; complexities=[]; prefixes=[]; union={}; enum_status={}
    streams={}
    for alg in (*ALGORITHMS,'enumeration'):
        name='hamming2/routes.jsonl' if alg=='enumeration' else f'search/{alg}_routes.jsonl'
        p=ROOT/name; p.parent.mkdir(parents=True,exist_ok=True)
        # Large route exports also reside externally.
        if alg=='enumeration':
            p=EXT/'search/enumeration_routes.jsonl'
        streams[alg]=p.open('w')
    for uid,r in pop.items():
        t=r['suffix_length']; initial=list(rows(EXT/f'initial_cache/{slug(uid)}.jsonl'))
        merged={x['suffix']:x for x in initial}
        extra=EXT/f'route_cache/{slug(uid)}.jsonl'
        if extra.exists(): merged.update({x['suffix']:x for x in rows(extra)})
        union[uid]=merged
        algorithms=[('enumeration',SEEDS[0])]+[(a,s) for a in ALGORITHMS for s in (SEEDS if a.startswith('random') or a=='binary_mcts' else SEEDS[:1])]
        for alg,seed in algorithms:
            p=EXT/f'search/enumeration/{slug(uid)}.jsonl' if alg=='enumeration' else EXT/f'search/{alg}/{seed}/{slug(uid)}.jsonl'
            rr=list(rows(p)); actual_routes[(uid,alg,seed)]=rr
            assert len({x['suffix'] for x in rr})==len(rr)
            assert [x['evaluation_rank'] for x in rr]==list(range(1,len(rr)+1))
            for x in rr: streams[alg].write(json.dumps(x,sort_keys=True)+'\n')
            budgets=[b for b in BUDGETS if b<=128]
            if alg=='enumeration': budgets=list(BUDGETS)
            elif r['dense_wrong']:
                for b in (256,512):
                    ex=extensions[b]
                    if any(x['algorithm']==alg and x['seed']==seed for x in ex.get(uid,[])): budgets.append(b)
            for b in budgets:
                cc=curve(rr,b)
                measures.append(dict(uid=uid,algorithm=alg,seed=seed,dataset=r['dataset'],source_regime=r['source_regime'],dense_wrong=r['dense_wrong'],trigger=r['first_trigger_layer'],T=t,cohort='adaptive_evaluated' if b>128 and alg!='enumeration' else 'enumeration_stopping_policy' if alg=='enumeration' else 'full_population_at_most_budget',**cc))
            successes=[x for x in rr if x['correct']]
            first.append(dict(uid=uid,algorithm=alg,seed=seed,dense_wrong=r['dense_wrong'],first_rescue_rank=successes[0]['evaluation_rank'] if successes else None,evaluated=len(rr),censored=not successes))
            at128=rr[:128]; good=[x for x in at128 if x['correct']]
            bits=np.array([[int(b) for b in x['suffix']] for x in good],dtype=float)
            n=len(good)
            mean_hamming=float((2*bits.sum(0)*(n-bits.sum(0))).sum()/(n*(n-1))) if n>1 else None
            multiplicity.append(dict(uid=uid,algorithm=alg,seed=seed,budget=128,actual_evaluations=len(at128),correct_routes=n,solution_fraction=n/len(at128),pairwise_hamming_mean=mean_hamming,distinct_correct_first_actions=len({x['suffix'][0] for x in good}),dense_wrong=r['dense_wrong']))
            labels=[x['correct'] for x in at128]; qs=[x['q'] for x in at128]
            ranked=sorted(at128,key=lambda x:(-x['q'],x['evaluation_rank']))
            qquality.append(dict(uid=uid,algorithm=alg,seed=seed,budget=128,dense_wrong=r['dense_wrong'],q_auc=roc_auc_score(labels,qs) if len(set(labels))==2 else None,mean_q_correct=np.mean([x['q'] for x in good]) if good else None,mean_q_wrong=np.mean([x['q'] for x in at128 if not x['correct']]) if any(not x['correct'] for x in at128) else None,best_correct_q_rank=next((i+1 for i,x in enumerate(ranked) if x['correct']),None)))
            if successes:
                shortest=min(successes,key=lambda x:(x['suffix'].count('0'),-x['q'],x['evaluation_rank']))
                complexities.append(dict(uid=uid,algorithm=alg,seed=seed,dense_wrong=r['dense_wrong'],route_hash=shortest['route_hash'],**complexity(shortest['suffix'],r['first_trigger_layer'])))
            for depth in range(1,t+1):
                if not good: break
                cc=Counter(x['suffix'][:depth] for x in good)
                entropy=-sum((v/n)*math.log2(v/n) for v in cc.values())
                shared=sum(v*(v-1) for v in cc.values())/(n*(n-1)) if n>1 else None
                prefixes.append(dict(uid=uid,algorithm=alg,seed=seed,depth=depth,correct_routes=n,prefix_entropy=entropy,pair_shared_prefix_fraction=shared,read_off_fraction=sum(x['suffix'][depth-1]=='0' for x in good)/n,dense_wrong=r['dense_wrong']))
    for s in streams.values(): s.close()
    h2link=ROOT/'hamming2/routes.jsonl'
    if not h2link.exists(): h2link.symlink_to(EXT/'search/enumeration_routes.jsonl')
    df=pd.DataFrame(measures)
    save('metrics/per_uid_budget_metrics.csv',df)
    ceiling={u for u in wids if any(x['correct'] for x in union[u].values())}; ceiling_n=len(ceiling)
    alignment=json.loads((ROOT/'population/alignment.json').read_text()); prior=set(alignment['prior_correctable_uids'])
    atomic_json(ROOT/'metrics/empirical_ceiling.json',dict(w=1307,empirical_binary_rescues=ceiling_n,rescued_uids=sorted(ceiling),coverage=ceiling_n/1307,prior_four_action=463,overlap_with_prior=len(ceiling&prior),new_beyond_prior=len(ceiling-prior),coverage_ratio_to_prior=ceiling_n/463,ceiling_status='READ-CEILING-LIMITED' if ceiling_n<.5*463 else 'READ-CEILING-SUFFICIENT',qualification='empirical adaptive union lower bound, not exhaustive true ceiling'))
    base=df[df.dense_wrong & (df.budget<=128)]
    # Average seeds within UID before treating UIDs as observations.
    u=base.groupby(['uid','algorithm','budget'],as_index=False).agg(any_correct=('any_correct','mean'),selected_correct=('selected_correct','mean'),actual_unique_routes=('actual_unique_routes','mean'))
    agg=u.groupby(['algorithm','budget'],as_index=False).agg(uids=('uid','nunique'),any_correct=('any_correct','mean'),selected_correct=('selected_correct','mean'),mean_actual_evaluations=('actual_unique_routes','mean'))
    agg['ranking_gap']=agg.any_correct-agg.selected_correct
    agg['ceiling_recovery']=agg.any_correct*1307/ceiling_n if ceiling_n else np.nan
    agg['qualified_policy']='at_most_budget; natural_exhaustion/enum_stop_retained'
    save('metrics/correction_curve_any_correct.csv',agg[['algorithm','budget','uids','any_correct','mean_actual_evaluations','qualified_policy']])
    save('metrics/correction_curve_selected_correct.csv',agg[['algorithm','budget','uids','selected_correct','mean_actual_evaluations','qualified_policy']])
    save('metrics/ceiling_recovery.csv',agg)
    decom=agg.copy(); decom['success']=decom.selected_correct; decom['ranking_failure']=decom.ranking_gap; decom['generation_failure']=1-decom.any_correct
    save('metrics/ranking_failure_decomposition.csv',decom)
    save('metrics/first_rescue_budget.csv',first)
    save('metrics/route_complexity.csv',complexities); save('metrics/solution_multiplicity.csv',multiplicity); save('metrics/prefix_entropy.csv',prefixes); save('metrics/q_ranking_quality.csv',qquality)
    save('hamming2/uid_summary.csv',[dict(uid=uid,evaluated=len(actual_routes[(uid,'enumeration',SEEDS[0])]),hamming2_evaluated=sum(x['suffix'].count('0')==2 for x in actual_routes[(uid,'enumeration',SEEDS[0])]),any_correct=any(x['correct'] for x in actual_routes[(uid,'enumeration',SEEDS[0])])) for uid in wids])
    cdf=df[~df.dense_wrong & df.budget.isin([8,32,128])].copy(); cdf['regression']=~cdf.selected_correct; cdf['dense_remains_selected']=[s=='1'*t for s,t in zip(cdf['selected_suffix'],cdf['T'])]
    save('controls/dense_c_preservation.csv',cdf)
    for grouping,name in [(['dataset','source_regime'],'dataset_source_breakdown'),(['trigger','T'],'trigger_depth_breakdown')]:
        temp=base.groupby(['uid','algorithm','budget',*grouping],as_index=False)[['any_correct','selected_correct']].mean()
        save(f'metrics/{name}.csv',temp.groupby(['algorithm','budget',*grouping],as_index=False).agg(uids=('uid','nunique'),any_correct=('any_correct','mean'),selected_correct=('selected_correct','mean')))
    seed_metrics=base.groupby(['algorithm','seed','budget'],as_index=False).agg(uids=('uid','nunique'),any_correct=('any_correct','mean'),selected_correct=('selected_correct','mean'))
    save('statistics/seed_metrics.csv',seed_metrics)
    # Pair UID effects at the same at-most-B resource cap; explicit actual count support.
    tests=[]; matched_counts=[]
    structured=('greedy','beam2','beam4','beam8','binary_mcts')
    for b in [1,4,8,16,32,64,128]:
        for alg,other in [(a,r) for a in structured for r in ('random_uniform','random_sparse')]+[(a,'greedy') for a in ('beam2','beam4','beam8')]:
            differences=[]; aa_values=[]; caps=[]
            for uid in wids:
                aseeds=SEEDS if alg.startswith('random') or alg=='binary_mcts' else SEEDS[:1]
                bseeds=SEEDS if other.startswith('random') or other=='binary_mcts' else SEEDS[:1]
                ar=[actual_routes[(uid,alg,seed)] for seed in aseeds]
                br=[actual_routes[(uid,other,seed)] for seed in bseeds]
                cap,av,bv=matched_any(ar,br,b)
                differences.append(av-bv); aa_values.append(av);caps.append(cap)
                matched_counts.append(dict(uid=uid,algorithm=alg,comparison=other,nominal_budget=b,shared_actual_cap=cap,algorithm_any=av,comparison_any=bv))
            delta=np.asarray(differences); low,high=pair_boot(delta)
            tests.append(dict(algorithm=alg,comparison=other,budget=b,uids=len(delta),mean_any_difference=float(delta.mean()),ci_low=low,ci_high=high,ceiling_recovery_difference=float(delta.mean()*1307/ceiling_n) if ceiling_n else None,matched_structured_ceiling_recovery=float(np.mean(aa_values)*1307/ceiling_n) if ceiling_n else None,mean_shared_actual_cap=float(np.mean(caps)),matched='per_UID_shared_actual_cap_across_all_paired_seeds'))
    save('controls/matched_actual_counts.csv',matched_counts)
    tests=pd.DataFrame(tests)
    save('statistics/paired_uid_bootstrap.csv',tests)
    save('controls/random_search_comparison.csv',tests[tests.comparison.str.startswith('random')])
    save('controls/greedy_vs_beam.csv',tests[tests.comparison=='greedy'])
    # High-budget changes computed on exactly the evaluated extension cohort.
    saturation=[]
    for (alg,seed),part in df[df.dense_wrong & (df.algorithm!='enumeration')].groupby(['algorithm','seed']):
        for lo,hi in zip(BUDGETS,BUDGETS[1:]):
            right=part[part.budget==hi].set_index('uid'); left=part[part.budget==lo].set_index('uid')
            common=right.index.intersection(left.index)
            if not len(common): continue
            saturation.append(dict(algorithm=alg,seed=seed,from_budget=lo,to_budget=hi,uids=len(common),cohort='adaptive_evaluated' if hi>128 else 'full_population',any_gain=float((right.loc[common,'any_correct'].astype(float)-left.loc[common,'any_correct'].astype(float)).mean()),selected_gain=float((right.loc[common,'selected_correct'].astype(float)-left.loc[common,'selected_correct'].astype(float)).mean()),mean_actual_routes=float(right.loc[common,'actual_unique_routes'].mean())))
    save('metrics/search_saturation.csv',saturation)
    save('metrics/adaptive_cohort_curves.csv',df[df.budget>128])
    # Prospective categories: apply conjunctions, preserve unclassified results.
    candidates=[]; firstdf=pd.DataFrame(first)
    for _,cell in agg[agg.algorithm.isin(['greedy','beam2','beam4','beam8','binary_mcts'])].iterrows():
        tt=tests[(tests.algorithm==cell.algorithm)&(tests.budget==cell.budget)&tests.comparison.str.startswith('random')]
        clear=len(tt)==2 and bool((tt.ci_low>0).all())
        delta10=len(tt)==2 and bool((tt.ceiling_recovery_difference>=.10).all())
        ff=firstdf[(firstdf.algorithm==cell.algorithm)&firstdf.dense_wrong]['first_rescue_rank'].dropna()
        median=float(ff.median()) if len(ff) else math.inf
        per_seed=firstdf[(firstdf.algorithm==cell.algorithm)&firstdf.dense_wrong].groupby('seed').first_rescue_rank.median()
        rank_gate=bool(len(per_seed) and per_seed.notna().all() and (per_seed<=32).all())
        recovery=float(tt.matched_structured_ceiling_recovery.min()) if len(tt)==2 else 0.
        a=cell.budget<=32 and recovery>=.70 and clear and delta10 and rank_gate
        b=cell.budget<=128 and recovery>=.70 and clear
        candidates.append(dict(algorithm=cell.algorithm,budget=int(cell.budget),A=bool(a),B=bool(b),clear_random_advantage=bool(clear),median_first_rescue=median))
    if any(c['A'] for c in candidates): category='P-READ-A'
    elif any(c['B'] for c in candidates): category='P-READ-B'
    else:
        top=agg[(agg.budget==128)&agg.algorithm.isin(['greedy','beam2','beam4','beam8','binary_mcts'])]
        high_gain=any(x['to_budget']>=256 and x['any_gain']>=.01 for x in saturation)
        low_recovery=not ceiling_n or float(top.ceiling_recovery.max())<.70
        strong=any(c['clear_random_advantage'] for c in candidates)
        dens=pd.DataFrame(multiplicity); med=dens[dens.dense_wrong & (dens.correct_routes>0)].solution_fraction.median()
        # C is conjunctive; unresolved rule gaps remain inconclusive.
        enum_easy=bool((agg[(agg.algorithm=='enumeration')&(agg.budget<=128)].ceiling_recovery>=.70).any())
        category='P-READ-C' if (low_recovery or high_gain) and (not strong or med<=.01) and not enum_easy else 'INCONCLUSIVE'
    ceiling_status='READ-CEILING-LIMITED' if ceiling_n<231.5 else 'READ-CEILING-SUFFICIENT'
    atomic_json(ROOT/'metrics/decision.json',dict(category=category,ceiling_status=ceiling_status,candidate_rules=candidates,empirical_rescues=ceiling_n,denominator=1307,prior=463))
    # Standalone figures.
    figroot=ROOT/'figures'; figroot.mkdir(exist_ok=True)
    def lines(data,y,name,ylabel):
        fig,ax=plt.subplots(figsize=(8,5))
        for alg,rr in data.groupby('algorithm'):
            rr=rr.sort_values('budget'); ax.plot(rr.budget,rr[y],marker='o',label=alg)
        ax.set_xscale('log',base=2); ax.set_xlabel('Maximum unique complete-route evaluations'); ax.set_ylabel(ylabel); ax.legend(fontsize=7,ncol=2); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figroot/name,dpi=180); plt.close(fig)
    lines(agg,'any_correct','read_search_budget_curve_any_correct.png','ANY_CORRECT (seed mean per UID)')
    lines(agg,'selected_correct','read_search_budget_curve_selected.png','q-SELECTED_CORRECT')
    lines(agg,'ceiling_recovery','read_ceiling_recovery.png','Fraction of empirical union recovered')
    lines(agg[agg.algorithm.isin(['beam8','random_uniform','random_sparse','enumeration'])],'any_correct','read_beam_vs_random.png','ANY_CORRECT')
    first_uid=firstdf.groupby(['uid','algorithm','dense_wrong'],as_index=False).agg(first_rescue_rank=('first_rescue_rank','mean'),seed_runs=('seed','count'),rescued_seed_runs=('first_rescue_rank','count'),maximum_evaluated=('evaluated','max'),minimum_evaluated=('evaluated','min'))
    first_uid['qualification']='mean_rank_over_observed_rescue_seeds; adaptive_coverage_reported; no_imputation_for_censoring'
    save('metrics/first_rescue_per_uid.csv',first_uid)
    fig,ax=plt.subplots(figsize=(8,5))
    for alg,rr in first_uid[first_uid.dense_wrong].groupby('algorithm'):
        vals=np.sort(rr.first_rescue_rank.dropna());
        if len(vals): ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=alg)
    ax.set_xscale('log',base=2); ax.set_xlabel('First correct evaluation rank'); ax.set_ylabel('CDF conditional on observed rescue'); ax.legend(fontsize=7); fig.tight_layout();fig.savefig(figroot/'read_first_rescue_cdf.png',dpi=180);plt.close(fig)
    def bars(frame,x,ys,name):
        ax=frame.set_index(x)[ys].plot.bar(figsize=(9,5),rot=35); ax.figure.tight_layout(); ax.figure.savefig(figroot/name,dpi=180);plt.close(ax.figure)
    bars(decom[decom.budget==128],'algorithm',['success','ranking_failure','generation_failure'],'read_ranking_vs_generation_failure.png')
    comp=pd.DataFrame(complexities); bars(comp[comp.dense_wrong].groupby('algorithm',as_index=False).hamming_distance.mean(),'algorithm',['hamming_distance'],'read_route_hamming_distance.png')
    dens=pd.DataFrame(multiplicity);bars(dens[dens.dense_wrong].groupby('algorithm',as_index=False).solution_fraction.mean(),'algorithm',['solution_fraction'],'read_solution_density.png')
    sat=pd.DataFrame(saturation); bars(sat.groupby('to_budget',as_index=False)[['any_gain','selected_gain']].mean(),'to_budget',['any_gain','selected_gain'],'read_search_saturation.png')
    depth=base[(base.budget==128)&(base.algorithm=='beam8')].groupby('T',as_index=False).any_correct.mean();bars(depth,'T',['any_correct'],'read_trigger_depth_difficulty.png')
    h1=pd.read_csv(ROOT/'single_off/hamming1_uid_summary.csv'); n1=int((h1.dense_wrong & h1.any_single_correct).sum())
    enum_rescues={uid for uid in wids if any(x['correct'] for x in actual_routes[(uid,'enumeration',SEEDS[0])])}; h2extra=len(enum_rescues)-n1
    first_summary=first_uid[first_uid.dense_wrong].groupby('algorithm').first_rescue_rank.agg(median='median',p75=lambda x:x.quantile(.75),p90=lambda x:x.quantile(.90),observed_rescues='count').reset_index()
    save('metrics/first_rescue_budget_summary.csv',first_summary)
    summary=f'''# READ-only bounded planning result

Frozen contract `{contract['contract_sha256']}`. **{category} / {ceiling_status}**.

1. Population: exactly1307 P90-triggered Dense-W and106 Dense-C; same UID/trigger registry as prior four-action search.
2. Cached single-OFF rescue: {n1}/1307. Exact gated Hamming2 adds {h2extra} UIDs; enumeration is an explicitly outcome-stopped policy.
3. Empirical READ binary union: {ceiling_n}/1307 ({ceiling_n/1307:.2%}), versus prior four-action463/1307. Coverage ratio {ceiling_n/463:.3f}; overlap {len(ceiling&prior)}, new outside prior {len(ceiling-prior)}. Both are bounded empirical opportunities, not exhaustive ceilings.
4. Full W ANY/SELECTED, ranking failures, ceiling recovery and actual counts: metrics/correction_curve_any_correct.csv, correction_curve_selected_correct.csv, ranking_failure_decomposition.csv, ceiling_recovery.csv. Base policies run on all W through maximumB128; natural exhaustion is retained with actual counts.
5. First-rescue ranks and censoring: metrics/first_rescue_budget.csv. Route complexity includes OFF count, first delay, last OFF, span and switches; see metrics/route_complexity.csv.
6. Solution count/fraction and mean pairwise Hamming: metrics/solution_multiplicity.csv; successful prefix entropy/sharing and ON/OFF fractions: metrics/prefix_entropy.csv. These describe discovered candidates, not the true solution-space density.
7. q separation and best-correct rank: metrics/q_ranking_quality.csv. q is annotated-answer likelihood; correctness always comes from actual LMMS-scored generated answers.
8. Dense-C q-selected regression, Dense retention and outcomes: controls/dense_c_preservation.csv; route wrong fractions follow metrics/solution_multiplicity.csv.
9. Beam versus both random baselines and greedy: controls/random_search_comparison.csv and greedy_vs_beam.csv with5000 paired UID bootstrap draws at per-UID shared actual evaluation caps (seeds averaged per UID).
10. Marginal gains and evaluated adaptive cohorts: metrics/search_saturation.csv and adaptive_cohort_curves.csv. B256/512 values are evaluated-cohort/stopping-policy evidence, never imputed fixed-budget full-population selected correctness.
11. Trigger/suffix-length and all six dataset/source cells: metrics/trigger_depth_breakdown.csv and dataset_source_breakdown.csv. Sparse cells remain descriptive.
12. Planning category and individual prospective rules: metrics/decision.json. Oracle search does not establish label-free planning, learned-policy generalization, causal mechanism, compute savings, external gain, or WRITE behavior.

## Base maximum-budget128 results

{agg[agg.budget==128].to_markdown(index=False)}

## First-rescue ranks (one mean observed-seed rank per UID; conditional on discovered rescue)

{first_summary.to_markdown(index=False)}

All required search and analysis work is stopped. No next experiment has run.
'''
    write_text(ROOT/'summaries/read_bounded_planning_summary.md',summary)
    write_text(ROOT/'summaries/read_problem_characterization.md',f'''# READ problem characterization

Existence is supported: READ suppression can improve q and correct answers. Population adjacency is mostly isolated (R-STRUCT-B); nuisance-matched operation features provide no robust local mechanism (R-MECH-B). Generic, operation-specific, one-step and H<=8 predictions remain weak (R-LEARN-C, CaseD, H-READ-D). This study adds bounded gold-q trajectory search: {category}, {ceiling_status}, empirical correction coverage{ceiling_n}/1307. See the main summary and paired comparisons for the scope of trajectory-structure claims. Oracle plan discovery and q-selection are reported separately; neither establishes a deployable method.\n''')
    if ceiling_status=='READ-CEILING-LIMITED': nextstep='A separately authorized WRITE-only characterization, maintaining the distinction between existence, learnability and planning. Do not return directly to a four-action router.'
    elif category in ('P-READ-A','P-READ-B'): nextstep='A separately authorized bounded READ planner study with a label-free proposal/value signal. A learned trajectory critic trained only on internal counterfactual labels could replace gold-answer q at inference; its held-out ranking and Dense-C preservation would remain unproven until tested.'
    else: nextstep='A separately authorized trajectory proposal/value learnability study before any deployment; preserve unresolved planning-ease evidence rather than claim a label-free planner.'
    write_text(ROOT/'summaries/next_research_direction.md','# One unexecuted next direction\n\n'+nextstep+'\n\nThis recommendation is not execution authorization.\n')
    # Hash compact reports/code and all durable worker records; external routes are hashed too.
    paths=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json']
    external=[p for p in EXT.rglob('*.jsonl') if p.is_file()]+[p for p in (EXT/'work').rglob('*.json') if p.is_file()]
    paths=sorted(set(p.resolve() for p in paths+external+[Path(__file__),ROOT/'analysis_contract.json']))
    manifest=dict(contract_sha256=contract['contract_sha256'],created_at=utc_now(),files={str(p):file_sha256(p) for p in paths})
    atomic_json(ROOT/'artifact_manifest.json',manifest)
    phase=ROOT.parents[1]/'workspace/phase_memory/phase_85_read_only_bounded_planning.md'
    with phase.open('a') as f:
        f.write(f'\n## Completed research-action result\n- Completed and stopped: {category} / {ceiling_status}; empirical READ rescue {ceiling_n}/1307.\n- Evidence: `analysis/read_only_bounded_planning/summaries/read_bounded_planning_summary.md`; all required outputs and artifact manifest saved.\n- Next implication: see `summaries/next_research_direction.md`; recommendation only, no follow-on authorized or run.\n')
    workflow=ROOT.parents[1]/'workspace/workflow_state.md'
    text=workflow.read_text(); workflow.write_text(text.replace('# Workflow State\n\n',f'# Workflow State\n\n- Phase85 completed/stopped: {category} / {ceiling_status}, empirical rescue {ceiling_n}/1307. Full result: `analysis/read_only_bounded_planning/summaries/read_bounded_planning_summary.md`. No follow-on execution.\n\n',1))
    print(json.dumps(dict(category=category,ceiling_status=ceiling_status,rescued=ceiling_n,artifact_files=len(paths))),flush=True)


if __name__=='__main__': aggregate()
