"""Render and verify completed fixed-schedule evidence; never launch experiments."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dense_failure_stage2.benchmark_fixed_schedule import FAMILIES
from experiments.run_benchmark_fixed_schedule import OUT, EXT, ROOT, check, read_json, read_jsonl, atomic_json, file_sha256


def read(path):return pd.read_csv(OUT/path)
def table(df):return df.to_markdown(index=False,floatfmt='.5f')
def layer(value):return None if pd.isna(value) else int(value)


def decision():
    assert read_json(OUT/'metrics/complete.json')['passed']
    selected=read('calibration/selected_layers.csv');halves=read('calibration/split_half_stability.csv');boot=pd.read_csv(OUT/'calibration/bootstrap_layer_selection.csv',dtype={'layer':str});nested=read('calibration/calibration_size_stability.csv')
    cis=read('metrics/paired_bootstrap_ci.csv');macro=read('metrics/macro_summary.csv')
    rand=read('metrics/benchmark_vs_random.csv')
    raw=pd.read_csv(OUT/'heldout/random_schedule_results.csv',usecols=['benchmark','benchmark_family','method','correct'])
    combined=pd.read_csv(OUT/'heldout/combined_results.csv',usecols=['benchmark','benchmark_family','correct'])
    for i,r in rand.iterrows():
        scope=raw[raw['benchmark_family' if r.scope=='family' else 'benchmark']==r.benchmark]
        key='benchmark_family' if r.scope=='family' else 'benchmark'
        calibrated=combined[combined[key]==r.benchmark]
        counts=scope.groupby('method').correct.agg(['sum','count'])
        assert (counts['count']==len(calibrated)).all()
        wins=counts['sum'].to_numpy();m3_wins=int(calibrated.correct.sum())
        values=wins/len(calibrated)
        at_least=int(np.sum(wins>=m3_wins))
        rand.loc[i,'calibrated_percentile']=100*(np.sum(wins<m3_wins)+.5*np.sum(wins==m3_wins))/len(wins)
        rand.loc[i,'random_max']=float(values.max());rand.loc[i,'random_at_least_calibrated']=at_least;rand.loc[i,'monte_carlo_p']=(1+at_least)/(1+len(values));rand.loc[i,'strictly_beats_all_random']=at_least==0
    rand.to_csv(OUT/'metrics/benchmark_vs_random.csv',index=False)
    def macro_low(name):return float(macro[macro.comparison==name].iloc[0].ci_low)
    def positive(benchmark,name):return float(cis[(cis.scope=='family')&(cis.benchmark==benchmark)&(cis.comparison==name)].iloc[0].ci_low)>0
    stability=[]
    for benchmark in FAMILIES:
        for bit in ['R','W']:
            pick=layer(selected[(selected.benchmark==benchmark)&(selected.bit==bit)].iloc[0].selected_layer)
            h=halves[(halves.benchmark==benchmark)&(halves.bit==bit)].iloc[0]
            frequency=float(boot[(boot.benchmark==benchmark)&(boot.bit==bit)&(boot.layer==('NONE' if pick is None else str(pick)))].iloc[0].frequency)
            unique=len({layer(v) for v in nested[(nested.benchmark==benchmark)&(nested.bit==bit)].selected_layer})
            finite=bool(pd.notna(h.spearman) and math.isfinite(float(h.spearman)))
            stable=bool(pick is not None and finite and h.spearman>=.3 and frequency>=.5)
            unstable=bool(pick is not None and finite and h.spearman<.3 and frequency<.5 and unique>1)
            stability.append(dict(benchmark=benchmark,bit=bit,selected_layer=pick,split_half_spearman=None if not finite else float(h.spearman),selected_bootstrap_frequency=frequency,nested_distinct_choices=unique,stable=stable,unstable=unstable))
    by={(r['benchmark'],r['bit']):r for r in stability}
    replicated=[b for b in FAMILIES if positive(b,'M3-Dense')]
    m3_replicates=len(replicated)>=2 and macro_low('M3-Dense')>0
    supported_a=[]
    for b in replicated:
        active=[by[(b,bit)] for bit in ['R','W'] if by[(b,bit)]['selected_layer'] is not None]
        random_pass=bool(rand[(rand.scope=='family')&(rand.benchmark==b)].iloc[0].strictly_beats_all_random)
        if active and all(r['stable'] for r in active) and random_pass:supported_a.append(b)
    bits={bit:[b for b in FAMILIES if by[(b,bit)]['stable'] and positive(b,method+'-Dense')] for bit,method in [('R','M1'),('W','M2')]}
    qualifying_bits=[bit for bit,method in [('R','M1'),('W','M2')] if len(bits[bit])>=2 and macro_low(method+'-Dense')>0]
    unstable_benchmarks=[b for b in FAMILIES if all(by[(b,bit)]['unstable'] for bit in ['R','W'])]
    if m3_replicates and len(supported_a)>=2 and macro_low('M3-GLOBAL')>0:category='CAL-A'
    elif qualifying_bits and not m3_replicates:category='CAL-B'
    elif len(unstable_benchmarks)>=2 and not m3_replicates:category='CAL-C'
    else:category='CAL-D'
    chosen_bit=None
    if category=='CAL-B':
        def bit_key(bit):
            m=macro[macro.comparison==('M1-Dense' if bit=='R' else 'M2-Dense')].iloc[0]
            return (-m.ci_low,-m.macro_delta,bit!='R')
        chosen_bit=min(qualifying_bits,key=bit_key)
    a_cal_possible=sum(bool(active) and all(r['stable'] for r in active) for active in [[by[(b,bit)] for bit in ['R','W'] if by[(b,bit)]['selected_layer'] is not None] for b in FAMILIES])>=2
    b_cal_possible=any(sum(by[(b,bit)]['stable'] for b in FAMILIES)>=2 for bit in ['R','W'])
    c_cal_possible=len(unstable_benchmarks)>=2
    category_cal_determined=not (a_cal_possible or b_cal_possible or c_cal_possible)
    result=dict(category=category,category_determined_by_calibration=category_cal_determined,calibration_allows_A=a_cal_possible,calibration_allows_B=b_cal_possible,calibration_allows_C=c_cal_possible,m3_replicates=m3_replicates,positive_m3_families=replicated,supported_A_families=supported_a,qualifying_B_bits=qualifying_bits,recommended_B_bit=chosen_bit,unstable_benchmarks=unstable_benchmarks,stability_checks=stability,mixed_positive_dense_evidence=bool(category=='CAL-D' and (replicated or macro_low('M3-Dense')>0)),D_scope='no demonstrated control-supported benefit under the full frozen rubric; not necessarily zero Dense improvement')
    atomic_json(OUT/'metrics/decision.json',result)
    return result


def figures():
    for benchmark in FAMILIES:
        fig,ax=plt.subplots(figsize=(8,4))
        for bit in ['read','write']:
            d=read(f'calibration/{benchmark}_{bit}_layer_effects.csv');ax.plot(d.layer,100*d.gain,marker='.',label=bit.upper()+' OFF')
        ax.axhline(0,color='gray',ls=':');ax.set(title=benchmark,xlabel='Intervention layer',ylabel='Calibration gain (percentage points)');ax.legend();fig.tight_layout();fig.savefig(OUT/f'figures/{benchmark}_read_write_layer_calibration.png',dpi=180);plt.close(fig)
    nested=read('calibration/calibration_size_stability.csv');fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,benchmark in zip(axes.flat,FAMILIES):
        for bit,g in nested[nested.benchmark==benchmark].groupby('bit'):ax.plot(g.target_n,g.selected_layer.fillna(-1),marker='o',label=bit)
        ax.set(title=benchmark,xlabel='Target CAL size (whole-group actual N may differ)',ylabel='Selected layer (-1 = NONE)',xticks=[32,64,128,256]);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'figures/calibration_size_stability.png',dpi=180);plt.close(fig)
    halves=read('calibration/split_half_stability.csv');fig,ax=plt.subplots(figsize=(8,4));labels=halves.benchmark+' '+halves.bit;ax.bar(labels,halves.spearman);ax.axhline(.3,color='gray',ls=':');ax.set(ylabel='Split-half layer Spearman');ax.tick_params(axis='x',rotation=30);fig.tight_layout();fig.savefig(OUT/'figures/split_half_layer_stability.png',dpi=180);plt.close(fig)
    m=read('metrics/benchmark_accuracy_summary.csv');ci=read('metrics/paired_bootstrap_ci.csv');m=m[m.scope=='family'];ci=ci[ci.scope=='family'];fig,ax=plt.subplots(figsize=(9,4));x=np.arange(4)
    for j,method in enumerate(['M1','M2','M3']):
        g=m[m.method==method].set_index('benchmark').loc[list(FAMILIES)];c=ci[ci.comparison==method+'-Dense'].set_index('benchmark').loc[list(FAMILIES)];point=100*g.delta_accuracy.to_numpy();lower=100*c.ci_low.to_numpy();upper=100*c.ci_high.to_numpy();ax.errorbar(x+(j-1)*.18,point,yerr=np.maximum(0,np.stack([point-lower,upper-point])),fmt='o',capsize=3,label=method)
    ax.axhline(0,color='gray',ls=':');ax.set(xticks=x,xticklabels=FAMILIES,ylabel='Held-out delta accuracy (pp), 95% group CI');ax.legend();fig.tight_layout();fig.savefig(OUT/'figures/heldout_accuracy_delta.png',dpi=180);plt.close(fig)
    g=m[m.method=='M3'].set_index('benchmark').loc[list(FAMILIES)];fig,ax=plt.subplots(figsize=(8,4));ax.bar(x-.18,g.w_to_c,.36,label='W→C');ax.bar(x+.18,g.c_to_w,.36,label='C→W');ax.set(xticks=x,xticklabels=FAMILIES,ylabel='M3 correctness transitions');ax.legend();fig.tight_layout();fig.savefig(OUT/'figures/correction_vs_regression.png',dpi=180);plt.close(fig)
    c=ci[ci.comparison=='M3-GLOBAL'].set_index('benchmark').loc[list(FAMILIES)];fig,ax=plt.subplots(figsize=(8,4));ax.bar(FAMILIES,100*c.delta);ax.axhline(0,color='gray',ls=':');ax.set(ylabel='Benchmark M3 minus global accuracy (pp)');fig.tight_layout();fig.savefig(OUT/'figures/benchmark_vs_global_schedule.png',dpi=180);plt.close(fig)
    r=read('metrics/benchmark_vs_random.csv');r=r[r.scope=='family'].set_index('benchmark').loc[list(FAMILIES)];fig,ax=plt.subplots(figsize=(8,4));ax.bar(FAMILIES,r.calibrated_percentile);ax.set(ylim=(0,105),ylabel='M3 midrank percentile among 20 random schedules',title='Ordinal diagnostic; not a significance level');fig.tight_layout();fig.savefig(OUT/'figures/calibrated_vs_random_percentile.png',dpi=180);plt.close(fig)


def dense_parity_audit():
    c=check()
    rows=[]
    for entry in read_jsonl(OUT/'parity/dense_execution_manifest.jsonl'):
        record=read_json(Path(entry['path']));dense=record['results']['dense']
        assert record['contract_sha256']==c['contract_sha256']
        assert dense['actions']==['FULL']*28
        rows.append(dict(benchmark=record['benchmark'],benchmark_family=record['benchmark_family'],split=record['split'],uids=1,native_token_exact=int(not dense['unified_token_mismatch']),scorer_equivalent_dual_output=int(dense['unified_token_mismatch']),valid_q=int(dense['q'] is not None)))
    frame=pd.DataFrame(rows)
    frame.groupby(['benchmark_family','benchmark','split'],as_index=False).sum(numeric_only=True).to_csv(OUT/'parity/dense_population_parity.csv',index=False)


def cost():
    rows=[]
    for stage in ['dense','R','W','methods','global','random']:
        workers=[read_json(EXT/f'{stage}_rank{r}_complete.json') for r in range(4)]
        seconds=sum(r['elapsed_seconds'] for r in workers);count=sum(r['uids'] for r in workers)
        prior_seconds=0.
        if stage=='dense':
            config=read_json(OUT/'frozen_contract.json')['config']
            if 'accepted_dense_parent' in config:
                prior_seconds=sum(read_json(Path(r['path']))['elapsed_seconds'] for r in read_jsonl(OUT/config['accepted_dense_parent']['manifest']))
        fresh=sum(r.get('fresh_uids',r['uids']) for r in workers)
        logical_per_uid={'dense':1,'R':28,'W':28,'methods':3,'global':1,'random':read_json(OUT/'execution_config.json')['random_schedules']}[stage]
        rows.append(dict(stage=stage,uids=count,logical_requested_routes=count*logical_per_uid,fresh_uids_this_launch=fresh,worker_gpu_wall_hours=(seconds+prior_seconds)/3600,retained_parent_uid_execution_seconds=prior_seconds,four_worker_wall_seconds=max(r['elapsed_seconds'] for r in workers),aggregate_fresh_uids_per_second=fresh/max(r['elapsed_seconds'] for r in workers),timing_scope='worker loop excluding model initialization; retained parent UID execution added separately; includes I/O, prefix reuse and all routes per UID'))
    df=pd.DataFrame(rows);df[df.stage.isin(['R','W'])].to_csv(OUT/'efficiency/calibration_cost.csv',index=False);df[~df.stage.isin(['R','W'])].to_csv(OUT/'efficiency/inference_cost.csv',index=False)
    (OUT/'efficiency/accounting_limits.md').write_text('# Cost accounting\n\nWorker-loop GPU wall-hours include I/O and CPU waiting but exclude model initialization, smoke and discarded failed attempts. Retained parent Dense UID timings are added to resumed-worker costs, and throughput uses newly executed UIDs only. These are observed execution costs, not kernel-active GPU-hours or complete project cost. Logical requested routes include duplicate and all-FULL schedules whose outputs are reused. Held-out methods share a full prefix during evaluation; the measured throughput is evaluator throughput, not the latency of one deployed fixed route. No FLOP/token reduction or end-to-end speedup is claimed. Calibration must be amortized before any efficiency claim.\n')


def summaries(result):
    scope_note='\n\nThe residual CAL-D category was already determined by the calibration stability gates before held-out efficacy was analyzed. It is not an outcome-based finding of no held-out benefit. Accuracy, interaction and global/random-control evidence below must be interpreted separately from this category.' if result['category_determined_by_calibration'] else ''
    support=read('splits/benchmark_support.csv');selected=read('calibration/selected_layers.csv');nested=read('calibration/calibration_size_stability.csv');half=read('calibration/split_half_stability.csv');boot=read('calibration/bootstrap_layer_selection.csv');dominant=boot.sort_values('frequency',ascending=False).groupby(['benchmark','bit']).head(3)
    summary=read('metrics/benchmark_accuracy_summary.csv');cis=read('metrics/paired_bootstrap_ci.csv');macro=read('metrics/macro_summary.csv');rand=read('metrics/benchmark_vs_random.csv');overfit=read('metrics/calibration_vs_test.csv');global_schedule=read_json(OUT/'schedules/global_schedule.json')
    (OUT/'summaries/calibration_summary.md').write_text('# Calibration complete\n\nNo Stage1, learning, or correctness-based population filtering. Image/content groups, native counterparts and RGB duplicates are disjoint across CAL/TEST. ChartQA/TextVQA use locally available official train frames; their existing evaluation splits are intact. MMMU/POPE exclude calibration groups from reported TEST. POPE CAL contains only 14 independent images; 252 questions are not 252 independent samples.\n\n'+table(support)+'\n\nSelected primary layers (blank = NONE; WRITE27 excluded):\n\n'+table(selected)+'\n\nNested whole-group prefixes and full-pool ranking:\n\n'+table(nested)+'\n\nSplit-half profile stability (constant correlations are unestimable):\n\n'+table(half)+'\n\nTop bootstrap selection frequencies, including NONE where present; entropy is in nats:\n\n'+table(dominant)+'\n\nGlobal schedule:\n\n```json\n'+json.dumps(global_schedule,indent=2)+'\n```\n\nFive TextVQA rows (one CAL, four TEST) have empty normalized q references. Their correctness is retained. As frozen, any incomplete q pool disables the q tie-break for that entire bit/pool; this applies to primary TextVQA and pooled global selection. Nested or half pools without that row may retain q tie-breaking.\n\nAll schedules were frozen after complete single-bit calibration and before TEST interventions; see [freeze report](../schedules/schedule_freeze_report.md). Smaller N is diagnostic only and was not selected for deployment on held-out results.\n')
    interaction=[]
    for (scope,benchmark),group in summary.groupby(['scope','benchmark']):
        g=group.set_index('method')
        d1=float(g.loc['M1','delta_accuracy']);d2=float(g.loc['M2','delta_accuracy']);d3=float(g.loc['M3','delta_accuracy'])
        interaction.append(dict(scope=scope,benchmark=benchmark,combined_minus_read=d3-d1,combined_minus_write=d3-d2,combined_beats_both=d3>d1 and d3>d2,departure_from_additive_delta=d3-d1-d2))
    interaction=pd.DataFrame(interaction)
    interaction.to_csv(OUT/'metrics/combined_interaction.csv',index=False)
    (OUT/'summaries/heldout_result_summary.md').write_text('# Held-out fixed-schedule result: '+result['category']+scope_note+'\n\nAll methods use exactly the same TEST UIDs and frozen benchmark-level schedules. Accuracy below is binary correctness under the inherited native thresholds; native_score separately preserves fractional TextVQA consensus. MMMU-Pro Standard/Vision and POPE variants appear separately. Family-pooled values enter the four-family macro; no calibration UID enters TEST.\n\n'+table(summary)+'\n\nPaired image/content-group bootstrap intervals:\n\n'+table(cis)+'\n\nBenchmark-specific versus matched random controls:\n\n'+table(rand)+'\n\nThe MonteCarlo p has minimum 1/21 for 20 draws. The midrank percentile is ordinal; a high percentile alone is not a 5% test. Repeated random routes retain their sampling weight.\n\nCalibration-to-TEST replication:\n\n'+table(overfit)+'\n\nFour-family macro (not pooled across unequal UID counts):\n\n'+table(macro)+'\n\nDirect combined-versus-bit comparisons (descriptive point differences, without a separate interaction significance test):\n\n'+table(interaction)+'\n\nM3 is directly evaluated; isolated bit gains are not added. The table compares combined accuracy against both bit ablations without retuning. Intervals condition on the frozen calibration choices; calibration uncertainty is characterized separately by grouped stability diagnostics.\n')
    plateaus=[]
    for (benchmark,bit),g in nested.groupby(['benchmark','bit']):
        g=g.sort_values('target_n');values=[layer(v) for v in g.selected_layer];primary=values[-1]
        start=next((i for i in range(len(values)-1) if all(v==primary for v in values[i:])),None)
        plateaus.append(dict(benchmark=benchmark,bit=bit,primary_layer=primary,agreement_from_target_n=None if start is None else int(g.iloc[start].target_n),agreement_from_actual_n=None if start is None else int(g.iloc[start].actual_n),tested_sizes_in_agreement=1 if start is None else len(values)-start))
    plateaus=pd.DataFrame(plateaus)
    plateaus.to_csv(OUT/'calibration/nested_selection_agreement.csv',index=False)
    stable=[f"{r['benchmark']}:{r['bit']}" for r in result['stability_checks'] if r['stable']]
    unique={bit:sorted({str(layer(v)) for v in selected[selected.bit==bit].selected_layer}) for bit in ['R','W']}
    (OUT/'summaries/benchmark_structure_characterization.md').write_text('# Benchmark-level structure characterization'+scope_note+'\n\nFrozen decision details:\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n\nStable selected READ/WRITE profiles under the prospective half-rho/frequency checks: '+(', '.join(stable) if stable else 'none')+'. This does not substitute for held-out replication. Selected layer diversity by bit is '+json.dumps(unique)+'. Distinct selected layers alone do not establish benchmark specificity: global and random controls are required.\n\nIndependently combined M3 replicates across at least two benchmarks with positive macro CI: '+str(result['m3_replicates'])+'. Consult the complete bit/combined and overfitting tables for positive and negative interactions. No fixed calibration size is claimed sufficient merely because a layer repeats; actual N, effective group count and grouped uncertainty are reported in calibration_summary.md.\n\nCoarse structure surviving weak per-instance prediction requires the full frozen A/B evidence, not a selected calibration gain. Residual D means no demonstrated control-supported benefit under this rubric. Positive Dense effects with failed control/stability criteria remain mixed evidence, not zero benefit. No result establishes instance-specific routing, unseen-benchmark generalization, causal mechanism, universal layers, or system savings.\n\n'+result['category']+'\n')
    recommendation={'CAL-A':'Formalize benchmark-calibrated sparse visual computation with exactly one separately authorized calibration-sample-efficiency test.','CAL-B':f"Retain only the stable {result['recommended_B_bit']} bit and test that minimal fixed schedule under a separately authorized plan.",'CAL-C':'Test coarser stage/region-level calibration under a separately authorized plan.','CAL-D':'Close benchmark-level exact-layer fixed calibration as insufficient under the tested rubric and return to the unresolved trajectory/nonlocal intervention question.'}[result['category']]
    (OUT/'summaries/next_research_direction.md').write_text('# Exactly one unexecuted next step\n\n'+recommendation+'\n\nBasis: '+result['category']+'. No new experiment, top-2 schedule, learned router, search, or strategic pivot is executed automatically. The current named plan ends with this evidence and recommendation. Preserve benchmark-specific positives, negative controls, low-effective-N uncertainty, and any mixed residual D qualification.\n')


def verify():
    from dense_failure_stage1.contract import model_file_hashes
    c=check();assert read_json(OUT/'metrics/complete.json')['passed']
    assert model_file_hashes(Path(c['config']['model']['snapshot_path']))==c['model_hashes'], 'model/processor content drift'
    frozen=read_json(OUT/'schedules/schedule_freeze.json')
    assert frozen['contract_sha256']==c['contract_sha256'] and frozen['no_test_intervention_execution_yet']
    for path,digest in frozen['files'].items():assert file_sha256(OUT/path)==digest
    assert read_json(EXT/'methods_launch_gpu_state.json')['observed_unix']>frozen['frozen_unix']
    for stage in ['dense','R','W','methods','global','random']:
        folder='parity' if stage=='dense' else 'calibration' if stage in ['R','W'] else 'heldout'
        assert read_json(OUT/f'{folder}/{stage}_complete.json')['passed']
        for record in read_jsonl(OUT/f'{folder}/{stage}_execution_manifest.jsonl'):
            assert file_sha256(Path(record['path']))==record['sha256'],record['uid']
    required=['calibration_summary','heldout_result_summary','benchmark_structure_characterization','next_research_direction']
    assert all((OUT/f'summaries/{name}.md').exists() for name in required)
    assert len(list((OUT/'figures').glob('*.png')))==10
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in ['artifact_manifest.json','final_verification.json']:
            files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=file_sha256(p)))
    atomic_json(OUT/'artifact_manifest.json',dict(contract_sha256=c['contract_sha256'],files=files,additional_code={name:file_sha256(ROOT/name) for name in ['experiments/continue_benchmark_fixed_schedule.py','experiments/summarize_benchmark_fixed_schedule.py','experiments/status_benchmark_fixed_schedule.py','tests/test_benchmark_fixed_reporting.py']},raw_execution='All durable UID records verified against stage execution manifests; model files bound in frozen contract'))
    atomic_json(OUT/'final_verification.json',dict(passed=True,artifact_files=len(files),figures=10,test_uids=read_json(OUT/'metrics/complete.json')['test_uids'],manifest_sha256=file_sha256(OUT/'artifact_manifest.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['report','verify']);args=parser.parse_args()
    if args.command=='report':dense_parity_audit();result=decision();figures();cost();summaries(result)
    else:verify()
