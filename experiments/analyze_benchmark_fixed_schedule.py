"""Calibration-only freeze and paired held-out analysis for fixed schedules."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dense_failure_stage2.benchmark_fixed_schedule import FAMILIES, SEED, bootstrap_deltas, layer_effects, ranked_layers, schedule, select_layer
from experiments.run_benchmark_fixed_schedule import OUT, EXT, ROOT, check, read_json, read_jsonl, atomic_json, record_path, file_sha256
from experiments.run_full_benchmark_end_to_end_eval import atomic_jsonl


def csv(path,rows):
    pd.DataFrame(rows).to_csv(OUT/path,index=False)


def calibration_records():
    assert read_json(OUT/'calibration/R_complete.json')['passed'] and read_json(OUT/'calibration/W_complete.json')['passed']
    rows=[]
    for meta in read_jsonl(OUT/'splits/split_registry.jsonl'):
        if meta['split']!='CAL':continue
        rows.append({**meta,'dense':read_json(record_path('dense',meta['uid']))['results']['dense'],'branches':{**read_json(record_path('R',meta['uid']))['results'],**read_json(record_path('W',meta['uid']))['results']}})
    return rows


def bootstrap_selection(rows,bit,draws):
    layers=28 if bit=='R' else 27
    ids=sorted({r['image_group_id'] for r in rows});idx={g:i for i,g in enumerate(ids)}
    group=np.array([idx[r['image_group_id']] for r in rows]);counts=np.bincount(group,minlength=len(ids))
    dense=np.array([r['dense']['correct'] for r in rows],bool)
    correct=np.array([[r['branches'][f'{bit}{l}']['correct'] for l in range(layers)] for r in rows],bool)
    gain=correct.astype(float)-dense[:,None]
    reg=dense[:,None]&(~correct)
    q_valid=all(r['dense'].get('q') is not None and all(r['branches'][f'{bit}{l}'].get('q') is not None for l in range(layers)) for r in rows)
    q=np.array([[r['branches'][f'{bit}{l}']['q']-r['dense']['q'] for l in range(layers)] for r in rows]) if q_valid else np.zeros((len(rows),layers))
    def grouped(x):return np.stack([np.bincount(group,weights=x[:,l],minlength=len(ids)) for l in range(layers)],axis=1)
    rng=np.random.default_rng(SEED+(1 if bit=='R' else 2));weights=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=draws)
    gains=weights@grouped(gain);regressions=weights@grouped(reg);qs=(weights@grouped(q))/(weights@counts)[:,None]
    picks=[]
    for g,r,qrow in zip(gains,regressions,qs):
        picks.append('NONE' if g.max()<=0 else str(min(range(layers),key=lambda l:(-g[l],-qrow[l],r[l],l))))
    freq=Counter(picks)
    entropy=-sum((n/draws)*math.log(n/draws) for n in freq.values())
    return [dict(layer=l,frequency=freq[l]/draws,count=freq[l],draws=draws,entropy=entropy) for l in ['NONE']+[str(i) for i in range(layers)]]


def select():
    c=check();rows=calibration_records()
    if (OUT/'schedules/schedule_freeze.json').exists():raise RuntimeError('schedules already frozen')
    selected=[];stability=[];halves=[];boot=[];schedules={}
    for benchmark in FAMILIES:
        rr=[r for r in rows if r['benchmark_family']==benchmark];layers={}
        for bit,name in [('R','read'),('W','write')]:
            effects=layer_effects(rr,bit);csv(f'calibration/{benchmark}_{name}_layer_effects.csv',effects)
            chosen=select_layer(effects);layers[bit]=chosen
            chosen_effect=next((r for r in effects if r['layer']==chosen),None)
            selected.append(dict(benchmark=benchmark,bit=bit,selected_layer=chosen,n=len(rr),**{k:chosen_effect[k] if chosen_effect else 0 for k in ['gain','net','w_to_c','c_to_w','mean_q']}))
            ranks={r['layer']:i+1 for i,r in enumerate(ranked_layers(effects))}
            for n in [32,64,128,256]:
                subset=[r for r in rr if n in r['nested_sizes']]
                e=layer_effects(subset,bit);pick=select_layer(e)
                stability.append(dict(benchmark=benchmark,bit=bit,target_n=n,actual_n=len(subset),image_groups=len({r['image_group_id'] for r in subset}),selected_layer=pick,selected_gain=next((r['gain'] for r in e if r['layer']==pick),0),rank_under_full_pool=ranks.get(pick)))
            ab=[layer_effects([r for r in rr if r['calibration_half']==half],bit) for half in ['A','B']]
            vectors=[np.array([r['gain'] for r in e if r['selectable']]) for e in ab]
            rho=float(spearmanr(*vectors).statistic) if all(np.ptp(v)>0 for v in vectors) else None
            top=[select_layer(e) for e in ab];top3=[{r['layer'] for r in ranked_layers(e)[:3]} for e in ab]
            halves.append(dict(benchmark=benchmark,bit=bit,spearman=rho,half_a_layer=top[0],half_b_layer=top[1],top1_agreement=top[0]==top[1],top3_overlap=len(top3[0]&top3[1]),sign_agreement=float(np.mean(np.sign(vectors[0])==np.sign(vectors[1])))))
            boot.extend(dict(benchmark=benchmark,bit=bit,**r) for r in bootstrap_selection(rr,bit,c['config']['selection_bootstrap_draws']))
        schedules[benchmark]=dict(read_layer=layers['R'],write_layer=layers['W'],methods={'M1':schedule(layers['R'],None),'M2':schedule(None,layers['W']),'M3':schedule(layers['R'],layers['W'])})
    global_layers={bit:select_layer(layer_effects(rows,bit)) for bit in ['R','W']}
    global_schedule=dict(read_layer=global_layers['R'],write_layer=global_layers['W'],actions=schedule(global_layers['R'],global_layers['W']),pool_uids=len(rows))
    random=[]
    for bi,benchmark in enumerate(FAMILIES):
        for i in range(c['config']['random_schedules']):
            seed=SEED+10000+bi*100+i;rng=np.random.default_rng(seed)
            r=int(rng.integers(28)) if schedules[benchmark]['read_layer'] is not None else None
            w=int(rng.integers(27)) if schedules[benchmark]['write_layer'] is not None else None
            random.append(dict(benchmark=benchmark,method=f'RANDOM_{i:03d}',seed=seed,read_layer=r,write_layer=w,actions=schedule(r,w)))
    csv('calibration/selected_layers.csv',selected);csv('calibration/calibration_size_stability.csv',stability);csv('calibration/split_half_stability.csv',halves);csv('calibration/bootstrap_layer_selection.csv',boot)
    atomic_json(OUT/'schedules/benchmark_specific_schedules.json',schedules);atomic_json(OUT/'schedules/global_schedule.json',global_schedule);atomic_jsonl(OUT/'schedules/random_schedule_registry.jsonl',random)
    files=['schedules/benchmark_specific_schedules.json','schedules/global_schedule.json','schedules/random_schedule_registry.jsonl']
    atomic_json(OUT/'schedules/schedule_freeze.json',dict(passed=True,contract_sha256=c['contract_sha256'],frozen_unix=time.time(),files={p:file_sha256(OUT/p) for p in files},calibration_execution_manifests={bit:file_sha256(OUT/f'calibration/{bit}_execution_manifest.jsonl') for bit in ['R','W']},no_test_intervention_execution_yet=not (EXT/'methods').exists()))
    assert not (EXT/'methods').exists()
    (OUT/'schedules/schedule_freeze_report.md').write_text('# Schedule freeze\n\nBenchmark, pooled global and20seeded matched-bit schedules frozen after complete READ/WRITE calibration and before any held-out intervention. No combined calibration gain was used to retune isolated choices. All NONE decisions and duplicate random draws retained.\n\n'+pd.DataFrame(selected).to_markdown(index=False)+'\n\nGlobal: '+json.dumps(global_schedule)+'\n')
    print(json.dumps(schedules,indent=2),flush=True)


def aggregate():
    c=check()
    for stage in ['methods','global','random']:assert read_json(OUT/f'heldout/{stage}_complete.json')['passed']
    manifest=[r for r in read_jsonl(OUT/'splits/split_registry.jsonl') if r['split']=='TEST']
    rows=[];flat=[]
    for meta in manifest:
        uid=meta['uid'];results={'M0':read_json(record_path('dense',uid))['results']['dense']}
        for stage in ['methods','global','random']:results.update(read_json(record_path(stage,uid))['results'])
        row={k:meta[k] for k in ['uid','benchmark','benchmark_family','image_group_id']};row['results']=results;rows.append(row)
        for method,result in results.items():flat.append({k:row[k] for k in ['uid','benchmark','benchmark_family','image_group_id']}|dict(method=method,correct=result['correct'],native_score=result['score'],generated_answer=result['generated_answer'],generated_token_ids=result['generated_token_ids'],actions=result['actions']))
    full=EXT/'heldout_per_uid_results.jsonl';atomic_jsonl(full,flat)
    link=OUT/'heldout/per_uid_results.jsonl'
    if not link.exists():link.symlink_to(full)
    frame=pd.DataFrame(flat)
    for method,name in [('M0','dense'),('M1','read_only'),('M2','write_only'),('M3','combined'),('GLOBAL','global_schedule')]:
        frame[frame.method==method].to_csv(OUT/f'heldout/{name}_results.csv',index=False)
    frame[frame.method.str.startswith('RANDOM')].to_csv(OUT/'heldout/random_schedule_results.csv',index=False)
    summary=[];cis=[];random=[];global_diffs=[];draws_by_family={};overfit=[]
    scopes=[('family',b,[r for r in rows if r['benchmark_family']==b]) for b in FAMILIES]+[('variant',b,[r for r in rows if r['benchmark']==b]) for b in sorted({r['benchmark'] for r in rows})]
    selections=pd.read_csv(OUT/'calibration/selected_layers.csv')
    for kind,benchmark,rr in scopes:
        dense=np.array([r['results']['M0']['correct'] for r in rr],bool)
        names=['M1','M2','M3','GLOBAL']
        correct={name:np.array([r['results'][name]['correct'] for r in rr],bool) for name in names}
        for name in ['M0']+names:
            values=dense if name=='M0' else correct[name];delta=values.astype(float)-dense
            summary.append(dict(scope=kind,benchmark=benchmark,method=name,n=len(rr),image_groups=len({r['image_group_id'] for r in rr}),accuracy=float(values.mean()),native_score=float(np.mean([r['results'][name]['score'] for r in rr])),dense_accuracy=float(dense.mean()),delta_accuracy=float(delta.mean()),w_to_c=int(((~dense)&values).sum()),c_to_w=int((dense&(~values)).sum()),net=int(delta.sum())))
        differences=np.stack([correct[name].astype(float)-dense for name in names]+[correct['M3'].astype(float)-correct['GLOBAL']],axis=1)
        draws=bootstrap_deltas(rr,differences,draws=c['config']['bootstrap_draws'],seed=SEED+len(cis))
        comparisons=['M1-Dense','M2-Dense','M3-Dense','GLOBAL-Dense','M3-GLOBAL']
        for j,comparison in enumerate(comparisons):
            lo,hi=np.quantile(draws[:,j],[.025,.975]);cis.append(dict(scope=kind,benchmark=benchmark,comparison=comparison,delta=float(differences[:,j].mean()),ci_low=float(lo),ci_high=float(hi)))
        random_acc=np.array([np.mean([r['results'][f'RANDOM_{i:03d}']['correct'] for r in rr]) for i in range(c['config']['random_schedules'])]);m3=float(correct['M3'].mean())
        random.append(dict(scope=kind,benchmark=benchmark,schedules=len(random_acc),random_mean=float(random_acc.mean()),random_std=float(random_acc.std(ddof=1)),calibrated_accuracy=m3,calibrated_percentile=float(100*(np.sum(random_acc<m3)+.5*np.sum(random_acc==m3))/len(random_acc)),calibrated_minus_random_mean=float(m3-random_acc.mean())))
        global_diffs.append(dict(scope=kind,benchmark=benchmark,calibrated_accuracy=m3,global_accuracy=float(correct['GLOBAL'].mean()),difference=float((correct['M3'].astype(float)-correct['GLOBAL']).mean())))
        if kind=='family':
            draws_by_family[benchmark]=draws
            for bit,method in [('R','M1'),('W','M2')]:
                selected=selections[(selections.benchmark==benchmark)&(selections.bit==bit)].iloc[0]
                held=next(x for x in summary if x['scope']==kind and x['benchmark']==benchmark and x['method']==method)
                overfit.append(dict(benchmark=benchmark,bit=bit,selected_layer=selected.selected_layer,calibration_n=int(selected.n),calibration_gain=float(selected.gain),calibration_w_to_c=float(selected.w_to_c),calibration_c_to_w=float(selected.c_to_w),heldout_delta=held['delta_accuracy'],heldout_w_to_c=held['w_to_c'],heldout_c_to_w=held['c_to_w']))
    macro_draws=np.mean(np.stack([draws_by_family[b] for b in FAMILIES]),axis=0);macro=[]
    for j,name in enumerate(comparisons):
        points=[r['delta'] for r in cis if r['scope']=='family' and r['comparison']==name];lo,hi=np.quantile(macro_draws[:,j],[.025,.975]);macro.append(dict(comparison=name,macro_delta=float(np.mean(points)),ci_low=float(lo),ci_high=float(hi),benchmarks=4))
    csv('metrics/benchmark_accuracy_summary.csv',summary);csv('metrics/correction_regression_summary.csv',summary);csv('metrics/paired_bootstrap_ci.csv',cis);csv('metrics/benchmark_vs_global.csv',global_diffs);csv('metrics/benchmark_vs_random.csv',random);csv('metrics/calibration_vs_test.csv',overfit);csv('metrics/macro_summary.csv',macro)
    atomic_json(OUT/'metrics/complete.json',dict(passed=True,test_uids=len(rows),per_uid_method_rows=len(flat),contract_sha256=c['contract_sha256']))
    print('Heldout aggregation complete',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['select','aggregate']);args=p.parse_args();select() if args.command=='select' else aggregate()
