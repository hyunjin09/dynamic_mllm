"""Prepare and execute the frozen READ-only bounded planning characterization."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
import math
import os
from pathlib import Path
import time

from dense_failure_stage2.read_only_planning import (ALGORITHMS, BUDGETS, SEEDS, SearchSession, actions, complexity, curve, enumeration, route_hash)
from experiments.run_predictability_stepA_measurement import (PROJECT_ROOT, resolve_path, file_sha256, read_json, read_jsonl, atomic_json, atomic_jsonl, atomic_csv, canonical_hash, utc_now)

ROOT=PROJECT_ROOT/'analysis/read_only_bounded_planning'
EXT=Path('/mnt/hyemin/qwen_train_eval/outputs/read_only_bounded_planning_v1')
PARENT=PROJECT_ROOT/'analysis/predictability_generalization/stepA_measurement'
CONFIG=PROJECT_ROOT/'configs/read_only_bounded_planning_v1.json'


def slug(uid): return sha256(uid.encode()).hexdigest()[:24]


def rows(path):
    with resolve_path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def write_text(path, text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)


def select_smoke(population, count):
    cells=defaultdict(list)
    for r in population:
        t=28-r['first_trigger_layer']
        cells[(r['dataset'],r['source_regime'], 'short' if t<=8 else 'middle' if t<=18 else 'long')].append(r)
    for c in cells.values(): c.sort(key=lambda r:sha256(('smoke'+r['uid']).encode()).hexdigest())
    chosen=[]
    while len(chosen)<count:
        for key in sorted(cells):
            if cells[key] and len(chosen)<count: chosen.append(cells[key].pop(0))
    return chosen


def prepare():
    if (ROOT/'frozen_contract.json').exists():
        raise RuntimeError('contract already exists; do not silently refreeze')
    ROOT.mkdir(parents=True,exist_ok=True); EXT.mkdir(parents=True,exist_ok=True)
    for name in ['work','search']:
        target=EXT/name; target.mkdir(exist_ok=True)
        link=ROOT/name
        if not link.exists(): link.symlink_to(target,target_is_directory=True)
    parent=read_json(PARENT/'frozen_contract.json')
    # Phase82 added the one-step API after StepA; its exact executor is also
    # the Phase84 parity-tested executor. Permit only that documented lineage.
    phase82_path=PROJECT_ROOT/'analysis/dense_failure_stage2/counterfactual_effect_identifiability/frozen_contract.json'
    phase82=read_json(phase82_path)
    inherited_code=dict(parent['bound_code_sha256'])
    inherited_code['binary_policy/executor/four_action.py']=phase82['bound_code_sha256']['binary_policy/executor/four_action.py']
    for p,h in inherited_code.items():
        if file_sha256(p)!=h: raise RuntimeError(f'parent code drift: {p}')
    manifest=PARENT/'manifests/p90_trigger_manifest.jsonl'
    pop=[r for r in rows(manifest) if r['triggered']]
    assert len({r['uid'] for r in pop})==1413
    for r in pop:
        assert r['first_trigger_layer']==next(i for i,s in enumerate(r['scores']) if s>0.9061332901863008)
        r['suffix_length']=28-r['first_trigger_layer']
    w=[r for r in pop if r['dense_wrong']]; c=[r for r in pop if r['dense_correct']]
    assert len(w)==1307 and len(c)==106
    by_uid={r['uid']:r for r in pop}
    for group,rr in [('w',w),('c',c)]: atomic_jsonl(ROOT/f'population/triggered_{group}_manifest.jsonl',rr)
    prior_path=PROJECT_ROOT/'analysis/dense_failure_stage2/robust_gate_corrective_search/work/manifests/triggered_wrong_pairs.jsonl'
    prior=[r for r in rows(prior_path) if r['operating_point']=='P90']
    aligned={(r['uid'],r['trigger_layer']) for r in prior}=={(r['uid'],r['first_trigger_layer']) for r in w}
    assert aligned
    route_store=PROJECT_ROOT/'analysis/dense_failure_stage2/robust_gate_corrective_search/routes/global_route_store.jsonl'
    prior_success={r['uid'] for r in rows(route_store) if r.get('valid_for_P90') and r.get('correct') and r['uid'] in {x['uid'] for x in w}}
    assert len(prior_success)==463
    atomic_json(ROOT/'population/alignment.json',dict(exact_uid_trigger_alignment=aligned,prior_correctable_uids=sorted(prior_success),w=1307,c=106))
    write_text(ROOT/'population/population_alignment_report.md',f'# Population alignment\n\nExact UID and trigger equality with robust P90 W registry: {aligned}. W=1307, C=106. Prior replay-valid four-action correction coverage is 463/1307. Dataset/source and dense correctness derive from the same hash-bound Step-A population.\n')
    dist=Counter((r['dataset'],r['source_regime'],r['dense_wrong'],r['suffix_length']) for r in pop)
    atomic_csv(ROOT/'population/suffix_length_distribution.csv',[dict(dataset=k[0],source=k[1],dense_wrong=k[2],T=k[3],uids=v) for k,v in sorted(dist.items())])
    branch_path=PARENT/'stage2_dense/four_branch_results.jsonl'
    cache=defaultdict(dict)
    for r in rows(branch_path):
        if r['action'] not in ('FULL','WRITE_ONLY'): continue
        uid=r['uid']; trigger=by_uid[uid]['first_trigger_layer']; t=28-trigger
        suffix=['1']*t
        if r['action']=='WRITE_ONLY': suffix[r['layer']-trigger]='0'
        suffix=''.join(suffix)
        item=dict(uid=uid,trigger_layer=trigger,suffix=suffix,route_hash=route_hash(uid,trigger,suffix),q=r['mean_logprob'],correct=r['correct'],generated_answer=r['generated_answer'],generated_token_ids=r['generated_token_ids'],cache_source='stepA',source_state_id=r['state_id'],source_action=r['action'],source_contract=parent['contract_sha256'],**complexity(suffix,trigger))
        old=cache[uid].get(suffix)
        if old and (old['q']!=item['q'] or old['generated_token_ids']!=item['generated_token_ids'] or old['correct']!=item['correct']):
            raise RuntimeError('inconsistent repeated FULL cache')
        cache[uid][suffix]=item
    singles=[]; summaries=[]; cost=[]
    for r in pop:
        uid=r['uid']; t=r['suffix_length']
        assert len(cache[uid])==t+1
        atomic_jsonl(EXT/f'initial_cache/{slug(uid)}.jsonl',cache[uid].values())
        h1=[cache[uid][s] for s in enumeration(t,1) if s!='1'*t]
        singles.extend(h1)
        correct=[x for x in h1 if x['correct']]
        best=max(h1,key=lambda x:x['q'])
        summaries.append(dict(uid=uid,dense_wrong=r['dense_wrong'],T=t,any_single_correct=bool(correct),best_single_suffix=best['suffix'],best_single_q=best['q'],best_single_correct=best['correct'],first_rescue_layer=min(x['first_read_off_layer'] for x in correct) if correct else None))
        if r['dense_wrong'] and not correct:
            cost.append(dict(uid=uid,T=t,distance2_routes=math.comb(t,2),expected_cache_hits=0,new_routes=math.comb(t,2)))
    atomic_jsonl(ROOT/'single_off/hamming1_routes.jsonl',singles)
    atomic_csv(ROOT/'single_off/hamming1_uid_summary.csv',summaries)
    atomic_csv(ROOT/'single_off/hamming1_rescue_layers.csv',[dict(uid=x['uid'],layer=x['first_read_off_layer'],delay=x['first_off_delay'],dense_wrong=by_uid[x['uid']]['dense_wrong']) for x in singles if x['correct']])
    atomic_csv(ROOT/'hamming2/projected_counts.csv',cost)
    total=sum(x['new_routes'] for x in cost)
    h2uids=[x['uid'] for x in cost] if total<=250000 else []
    atomic_json(ROOT/'hamming2/eligibility.json',dict(projected_new=total,cap=250000,run=total<=250000,uids=h2uids))
    write_text(ROOT/'hamming2/projected_cost.md',f'# Hamming-2 prospective cost\n\n{len(cost)} Hamming-1-unrescued W UIDs; {total:,} exact new distance-2 candidates; 0 complete q+generation cache hits beyond Step-A. Gate <=250,000: {total<=250000}. Provisional cost at 1 second/candidate: {total/3600:.2f} GPU-hours; replace this timing assumption with infrastructure-smoke timing before any Hamming-2 launch. No subsampling.\n')
    # Audit only declared candidate stores. Missing q excludes an outcome from full reuse.
    inventories=[('Step-A',str(branch_path),'eligible exact FULL/Hamming1 q+generation'),
        ('prior corrective/MCTS',str(route_store),'generation-only; no frozen gold q; not reused as complete evaluation'),
        ('historical MCTS','analysis/dense_failure_stage2/full_corrective_labels/routes/successful_mcts_routes.jsonl','no frozen q; exclude'),
        ('canonical MCTS','analysis/dense_failure_stage2/data_scale_search/routes/new_successful_mcts_routes.jsonl','no frozen q; exclude'),
        ('completeness','analysis/dense_failure_stage2/treatment_label_completeness/search/mcts_results.jsonl','state-local four-action binary-correctness score, not complete frozen-q contract; exclude'),
        ('short horizon','analysis/read_harm_short_horizon_propagation/frozen_contract.json','single-toggle intermediate states only; no new complete route outcomes')]
    provenance=[]
    for name,p,decision in inventories:
        first=next(rows(p),{}) if p.endswith('.jsonl') else read_json(p)
        provenance.append(dict(source=name,path=p,sha256=file_sha256(p),top_level_fields='|'.join(sorted(first)),decision=decision))
    atomic_csv(ROOT/'cache/cache_provenance.csv',provenance)
    atomic_jsonl(ROOT/'cache/route_cache_manifest.jsonl',[dict(uid=r['uid'],path=str(EXT/f'initial_cache/{slug(r["uid"])}.jsonl'),sha256=file_sha256(EXT/f'initial_cache/{slug(r["uid"])}.jsonl'),routes=len(cache[r['uid']])) for r in pop])
    smoke=select_smoke(w,32)+select_smoke(c,8)
    atomic_jsonl(ROOT/'population/smoke_manifest.jsonl',smoke)
    # Long expensive prompts distributed by estimated token*suffix work.
    internal={r['uid']:r for r in rows(PARENT/'manifests/internal_sample_manifest.jsonl') if r['uid'] in by_uid}
    loads=[0]*4; schedule=[]
    for r in sorted(pop,key=lambda r:(-int(internal[r['uid']]['dense']['prompt_token_count'])*r['suffix_length'],r['uid'])):
        rank=min(range(4),key=lambda k:loads[k]); loads[rank]+=int(internal[r['uid']]['dense']['prompt_token_count'])*r['suffix_length']
        schedule.append(dict(uid=r['uid'],rank=rank))
    atomic_jsonl(ROOT/'population/schedule.jsonl',schedule)
    cfg=dict(schema_version='read_only_bounded_planning_v1',seed=20260912,world_size=4,model=parent['static_config']['model'],backend_settings=parent['static_config']['backend_settings'],measurement=parent['static_config']['measurement'],algorithms=ALGORITHMS,seeds=SEEDS,budgets=BUDGETS,dense_candidate_first=True,q_tie='first_seen',hamming2_new_cap=250000,resolved_saturation_fraction=.10,extension_gain_threshold=.01,bootstrap_draws=5000,mcts_reward='raw_frozen_q',mcts_rollout='uniform_binary_among_unexhausted_prefix_children',unique_cache_hit_counts=True)
    atomic_json(CONFIG,cfg)
    write_text(ROOT/'protocol.md',PROTOCOL)
    source_paths=[str(phase82_path),str(manifest),str(branch_path),str(prior_path),str(route_store),str(PARENT/'manifests/internal_sample_manifest.jsonl'),str(PARENT/'frozen_contract.json'),str(ROOT/'population/smoke_manifest.jsonl'),str(ROOT/'population/schedule.jsonl'),str(ROOT/'protocol.md'),str(ROOT/'hamming2/eligibility.json'),str(ROOT/'cache/route_cache_manifest.jsonl')]
    code_paths=['dense_failure_stage2/read_only_planning.py','experiments/run_read_only_bounded_planning.py','tests/test_read_only_planning.py','experiments/analyze_read_only_bounded_planning.py','plans/read_only_bounded_planning_search_plan.md',str(CONFIG)]
    contract=dict(schema_version='read_only_bounded_planning_contract_v1',created_at=utc_now(),static_config=cfg,parent_contract=parent['contract_sha256'],source_sha256={p:file_sha256(p) for p in source_paths},bound_code_sha256={**inherited_code,**{p:file_sha256(p) for p in code_paths}},model_snapshot_sha256=parent['model_snapshot_sha256'],review=dict(verdict='revise',accepted='Hamming<=2 first-class budgeted stopping-policy baseline; explicit adaptive denominators',reference='/root/phase85_protocol_review'))
    contract['contract_sha256']=canonical_hash(contract)
    atomic_json(ROOT/'frozen_contract.json',contract)
    print(json.dumps(dict(contract=contract['contract_sha256'],w=1307,c=106,hamming1_rescued=sum(x['dense_wrong'] and x['any_single_correct'] for x in summaries),hamming2_new=total)),flush=True)


PROTOCOL='''# Frozen Phase-85 execution protocol

Execute only the user-authorized READ-only bounded planning/search plan. WRITE is always ON. Inherit exact Step-A model, prompt/image, robust P90, teacher-forced q aggregation, 16-token greedy generation, and LMMS correctness. No retraining or external evaluation.

Every search starts with Dense as candidate 1. Cache hits count toward each algorithm's unique complete-route budget; duplicate requests within one algorithm do not. Cross-algorithm caches supply outcomes only for routes independently requested; no shared best-route or correctness seeding. Highest q wins, ties first evaluated. Structured scores always correspond to the exact executed prefix plus all-ON completion. States are reconstructed through exact sequential routed execution, never substituted with dense states after a deviation.

Greedy and beam2/4/8 run deterministically with ON-before-OFF expansion and stable prefix tie order. They stop at natural tree exhaustion or their capped budget; no random padding. Report actual counts and stop-policy curves explicitly. Random-uniform, random-sparse and binary UCB1 MCTS each use three fixed seeds (2026091201/2/3). MCTS uses raw q terminal reward, UCB exploration sqrt(2), deterministic prefix expansion, uniform binary rollouts, and excludes fully evaluated subtrees. No correctness-dependent stopping within B<=128. Seeds are paired and averaged per UID for population/bootstraps, never pooled as independent UIDs.

All W and C run B<=128. C reports B8/32/128 with Dense always available. W extensions are per algorithm/seed: B256 for unresolved ANY@128 plus a deterministic hash-selected 10% of resolved UIDs. B512 only for still-unresolved evaluated UIDs if ANY gain from128 to256 is >=0.01 on that algorithm/seed's B128-unresolved extension cohort (resolved saturation samples reported separately). Preserve cohort IDs, stop-policy estimates, and actual counts. No extrapolated fixed-budget SELECTED correctness for unextended UIDs. Greedy/beam may have exhausted before an extension; record it.

Hamming<=2 enumeration order is Dense, ascending single-OFF position, then lexicographic OFF-position pairs. Hamming2 runs only for W unrescued by Hamming1 and only if aggregate new evaluations <=250000. It is a first-class outcome-gated enumeration baseline, not disguised full-population Hamming2. Include its measured routes in the empirical binary rescue union. Successful stopping-policy curves use actual counts; pair comparisons use explicitly matched supported budgets. An easy enumeration outcome must not be characterized as needle-like solely because another algorithm failed.

32W+8C stratified smoke validates fresh FULL/single q+tokens+correctness, image/input parity, full-route vs prefix-cloned multi-OFF execution, repeat/cache isolation, beam accounting and binary MCTS. Smoke is infrastructure-only. No main search launches before all40 pass. Timing estimates for Hamming2 use smoke measurements and are recorded before census launch.

Freeze bootstrap5000 paired UID draws; report image-group clustered sensitivity where feasible. P-READ-A uses the plan's >=70% ceiling recovery, >=0.10 advantage versus BOTH randoms at a matched B<=32 and median rescue rank<=32; require positive paired bootstrap lower bounds for structure claims. P-READ-B requires >=70% by128, not A, and positive lower bounds versus BOTH randoms. For P-READ-C define substantial high-budget gain as >=.01 cohort ANY improvement; very low solution density as median correct fraction<=.01 among rescued UIDs at the matched reported budget. Report inconclusive if categories overlap or do not apply; never force a label. Empirical union is a lower bound, includes seeds/census, and is adaptive-budget-qualified. Ceiling-limited means <50% of exactly aligned prior four-action463/1307 coverage. A gold-q oracle result does not establish label-free learnability, compute gain, or deployment utility.

Independent review: revise accepted; promote the already requested Hamming census into budgeted reports and distinguish adaptive/stopping-policy from fixed-budget estimates. User explicitly authorizes all four GPUs even occupied; no other processes are terminated. Large outputs live under /mnt/hyemin. Stop after required artifacts, characterization, and one unexecuted next recommendation.
'''


def verify():
    c=read_json(ROOT/'frozen_contract.json')
    assert canonical_hash(c)==c['contract_sha256']
    for p,h in {**c['source_sha256'],**c['bound_code_sha256']}.items():
        if file_sha256(p)!=h: raise RuntimeError(f'contract drift: {p}')
    return c


class Evaluator:
    def __init__(self, processor, wrapped, sample_row, poprow, cache_root, contract):
        import torch
        from dense_failure_stage1.runtime import build_dense_inputs
        from binary_policy.executor.inputs import build_binary_inputs
        from binary_policy.executor import capture_four_action_route
        self.processor,self.wrapped=processor,wrapped
        self.sample=sample_row['sample']; self.pop=poprow; self.uid=poprow['uid']; self.trigger=poprow['first_trigger_layer']
        self.contract=contract['contract_sha256']; self.cfg=contract['static_config']
        self.inputs,metadata=build_dense_inputs(processor,self.sample,torch.device('cuda:0'))
        assert metadata['consumed_image_sha256']==self.sample['image_content_sha256']
        self.prepared=build_binary_inputs(wrapped,self.inputs)
        self.baseline=capture_four_action_route(wrapped,{},['FULL']*28,prepared_inputs=self.prepared,use_cache=True,native_full_rows=True)
        cache_path=EXT/f'initial_cache/{slug(self.uid)}.jsonl'
        cache_manifest=next(x for x in rows(ROOT/'cache/route_cache_manifest.jsonl') if x['uid']==self.uid)
        assert file_sha256(cache_path)==cache_manifest['sha256']
        self.cache={}
        for item in rows(cache_path):
            suffix=item['suffix']
            assert item['uid']==self.uid and item['trigger_layer']==self.trigger
            assert item['source_contract']==contract['parent_contract']
            assert item['route_hash']==route_hash(self.uid,self.trigger,suffix)
            assert suffix.count('0')<=1 and suffix not in self.cache
            self.cache[suffix]=item
        assert len(self.cache)==29-self.trigger
        self.path=cache_root/f'{slug(self.uid)}.jsonl'; self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():
            for r in rows(self.path):
                assert r['contract_sha256']==self.contract
                assert r['uid']==self.uid and r['trigger_layer']==self.trigger
                assert r['route_hash']==route_hash(self.uid,self.trigger,r['suffix'])
                self.cache[r['suffix']]=r
        self.new_count=0; self.new_seconds=0.; self.hits=0

    def execute(self,suffix,full_replay=False):
        from binary_policy.executor import capture_four_action_route, capture_four_action_suffix_from_full_baseline
        from experiments.run_predictability_stepA_measurement import _measure_output
        aa=actions(self.trigger,suffix)
        if full_replay:
            out=capture_four_action_route(self.wrapped,{},aa,prepared_inputs=self.prepared,use_cache=True,native_full_rows=True)
        elif '0' not in suffix: out=self.baseline
        else:
            first=self.trigger+suffix.index('0')
            out=capture_four_action_suffix_from_full_baseline(self.wrapped,self.baseline,first,aa[first:])
        assert tuple(out.layer_actions)==aa
        assert all(s.action in ('FULL','WRITE_ONLY') for s in out.layer_stats)
        result=_measure_output(self.processor,self.wrapped,out,self.inputs['input_ids'],self.sample,max_new_tokens=16)
        return result,out

    def __call__(self,suffix):
        route_hash(self.uid,self.trigger,suffix)
        if suffix in self.cache:
            self.hits+=1
            return {**self.cache[suffix],'cache_hit':True}
        started=time.monotonic()
        measured,out=self.execute(suffix)
        del out
        elapsed=time.monotonic()-started
        result=dict(uid=self.uid,trigger_layer=self.trigger,suffix=suffix,route_hash=route_hash(self.uid,self.trigger,suffix),q=measured['mean_logprob'],correct=measured['correct'],generated_answer=measured['generated_answer'],generated_token_ids=measured['generated_token_ids'],lmms_raw_score=measured['lmms_raw_score'],cache_source='phase85_new',contract_sha256=self.contract,elapsed_seconds=elapsed,**complexity(suffix,self.trigger))
        with self.path.open('a') as f: f.write(json.dumps(result,sort_keys=True)+'\n'); f.flush()
        self.cache[suffix]=result; self.new_count+=1; self.new_seconds+=elapsed
        return {**result,'cache_hit':False}


def parity(a,b):
    assert abs(float(a.get('q',a.get('mean_logprob')))-float(b.get('q',b.get('mean_logprob'))))<=1e-6
    assert a['generated_token_ids']==b['generated_token_ids']
    assert bool(a['correct'])==bool(b['correct'])


def worker(mode,rank):
    import torch
    from dense_failure_stage1.runtime import configure_dense_determinism
    from experiments.run_stage2_v1_training_revised import _load_model
    from experiments.run_predictability_stepA_measurement import tensor_sha256
    contract=verify(); cfg=contract['static_config']
    if mode!='smoke':
        smoke=read_json(ROOT/'controls/smoke_complete.json')
        assert smoke['passed'] and smoke['uids']==40 and smoke['contract_sha256']==contract['contract_sha256']
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    configure_dense_determinism(cfg['seed'],cfg['backend_settings'])
    processor,base,wrapped=_load_model(cfg,torch.device('cuda:0'))
    pop={r['uid']:r for group in ('w','c') for r in rows(ROOT/f'population/triggered_{group}_manifest.jsonl')}
    internal={r['uid']:r for r in rows(PARENT/'manifests/internal_sample_manifest.jsonl') if r['uid'] in pop}
    if mode=='smoke': selected=list(rows(ROOT/'population/smoke_manifest.jsonl'))[rank::4]
    else:
        ids=[r['uid'] for r in rows(ROOT/'population/schedule.jsonl') if r['rank']==rank]
        selected=[pop[u] for u in ids]
    h2=read_json(ROOT/'hamming2/eligibility.json'); h2uids=set(h2['uids'])
    extensions=read_json(ROOT/f'work/extension_{mode}.json') if mode in ('256','512') else {}
    for i,r in enumerate(selected):
        uid=r['uid']; t=r['suffix_length']; outpath=EXT/f'work/{mode}/{slug(uid)}.json'
        if outpath.exists():
            old=read_json(outpath)
            assert old['contract_sha256']==contract['contract_sha256'] and old['complete']
            continue
        if mode in ('256','512') and uid not in extensions: continue
        started=time.monotonic()
        evaluator=Evaluator(processor,wrapped,internal[uid],r,EXT/('smoke_cache' if mode=='smoke' else 'route_cache'),contract)
        result=dict(uid=uid,contract_sha256=contract['contract_sha256'],mode=mode,complete=False)
        if mode=='smoke':
            validations=[]
            for suffix in ('1'*t,'0'+'1'*(t-1),'1'*(t-1)+'0'):
                actual,out=evaluator.execute(suffix)
                parity(actual,evaluator.cache[suffix]); del out
                validations.append(suffix)
            multi=''.join('0' if j in (0,t-1) else '1' for j in range(t))
            a,oa=evaluator.execute(multi); b,ob=evaluator.execute(multi,full_replay=True)
            parity(a,b)
            # Closed-loop prestates match an independently reconstructed route at every layer.
            for pa,pb in zip(oa.pre_layer_states,ob.pre_layer_states):
                assert all(torch.equal(x,y) for x,y in zip(pa,pb))
            state_hashes=[dict(layer=l,text=tensor_sha256(x),visual=tensor_sha256(y)) for l,(x,y) in enumerate(oa.pre_layer_states)]
            del oa,ob
            repeated,oo=evaluator.execute(multi); parity(a,repeated); del oo
            sessions=[]
            for alg in ALGORITHMS:
                s=SearchSession(alg,t,uid); s.advance(8,evaluator)
                assert len({x['route_hash'] for x in s.rows})==len(s.rows)
                repeat=SearchSession(alg,t,uid); repeat.advance(8,evaluator)
                assert [(x['suffix'],x['q'],x['correct']) for x in s.rows]==[(x['suffix'],x['q'],x['correct']) for x in repeat.rows]
                sessions.append(dict(algorithm=alg,routes=len(s.rows),requests=s.requests,exhausted=s.exhausted,deterministic_replay=True))
            result.update(validations=validations,multi_suffix=multi,closed_loop_state_hashes=state_hashes,sessions=sessions,passed=True)
        else:
            # Fresh Dense parity per UID on every resumed GPU worker as a runtime guard.
            actual,out=evaluator.execute('1'*t); parity(actual,evaluator.cache['1'*t]); del out
            sessions=[]
            if mode=='128':
                # Existing Hamming1 is fully cached; add exhaustive pairs only to the eligible census.
                enum=[]
                for suffix in enumeration(t,2 if uid in h2uids else 1):
                    rr=evaluator(suffix); rr.update(evaluation_rank=len(enum)+1,search_algorithm='enumeration',seed=SEEDS[0]); enum.append(rr)
                atomic_jsonl(EXT/f'search/enumeration/{slug(uid)}.jsonl',enum)
                result['enumeration']=dict(routes=len(enum),hamming2_executed=uid in h2uids)
                requested=[(a,s) for a in ALGORITHMS for s in (SEEDS if a.startswith('random') or a=='binary_mcts' else SEEDS[:1])]
            else: requested=[(x['algorithm'],x['seed']) for x in extensions[uid]]
            for alg,seed in requested:
                s=SearchSession(alg,t,uid,seed); s.advance(int(mode),evaluator)
                path=EXT/f'search/{alg}/{seed}/{slug(uid)}.jsonl'
                atomic_jsonl(path,s.rows)
                sessions.append(dict(algorithm=alg,seed=seed,routes=len(s.rows),requests=s.requests,exhausted=s.exhausted,curve=curve(s.rows,int(mode))))
            result.update(sessions=sessions)
        result.update(complete=True,new_route_evaluations=evaluator.new_count,new_route_seconds=evaluator.new_seconds,cache_hits=evaluator.hits,elapsed_seconds=time.monotonic()-started)
        atomic_json(outpath,result)
        print(json.dumps(dict(event='uid_complete',rank=rank,mode=mode,index=i,uid=uid,seconds=result['elapsed_seconds'],new=result['new_route_evaluations'])),flush=True)
        del evaluator
        torch.cuda.empty_cache()
    atomic_json(EXT/f'work/{mode}/rank{rank}_complete.json',dict(complete=True,contract_sha256=contract['contract_sha256'],rank=rank,mode=mode))


def finalize_smoke():
    c=verify()
    smoke=list(rows(ROOT/'population/smoke_manifest.jsonl'))
    results=[read_json(EXT/f'work/smoke/{slug(r["uid"])}.json') for r in smoke]
    assert all(r['passed'] and r['complete'] and r['contract_sha256']==c['contract_sha256'] for r in results)
    seconds=sum(r['new_route_seconds'] for r in results); count=sum(r['new_route_evaluations'] for r in results)
    per=seconds/max(1,count)
    h2=read_json(ROOT/'hamming2/eligibility.json')
    atomic_json(ROOT/'controls/smoke_complete.json',dict(passed=True,uids=40,contract_sha256=c['contract_sha256'],new_routes=count,new_route_seconds=seconds,seconds_per_new_route=per))
    atomic_csv(ROOT/'controls/cache_parity.csv',[dict(uid=r['uid'],passed=True,single_q_token_correctness=True,full_vs_prefix_closed_loop=True,fresh_repeat=True) for r in results])
    for name in ('cache/parity_report.md','controls/smoke_report.md'):
        write_text(ROOT/name,f'# Infrastructure smoke\n\nPassed40/40 (32W,8C): FULL/single q<=1e-6, exact tokens/LMMS, image parity, multi-OFF full versus cached-prefix state equality, repeat isolation, binary algorithm routes and unique accounting. New-route mean {per:.3f}s; {count} measured. Infrastructure only; not scientific outcomes.\n')
    write_text(ROOT/'hamming2/projected_cost.md',f'# Hamming-2 projected cost before launch\n\nExact new candidates: {h2["projected_new"]:,}; expected complete-cache hits:0. Cap250000 passes:{h2["run"]}. Infrastructure smoke measured {per:.3f}s/new route, implying {h2["projected_new"]*per/3600:.2f} GPU-hours ({h2["projected_new"]*per/14400:.2f} ideal four-GPU wall hours). Occupancy and prompt mixture can change throughput. Exact census only; no subsampling.\n')
    print(json.dumps(dict(smoke='passed',seconds_per_route=per,hamming2_gpu_hours=h2['projected_new']*per/3600)),flush=True)


def extension(budget):
    c=verify(); previous=128 if budget==256 else 256
    w=list(rows(ROOT/'population/triggered_w_manifest.jsonl'))
    selected=defaultdict(list); groups=defaultdict(list)
    for r in w:
        p=EXT/f'work/{previous}/{slug(r["uid"])}.json'
        if not p.exists():
            if previous==128: raise RuntimeError('incomplete base population')
            continue
        result=read_json(p)
        for s in result['sessions']: groups[(s['algorithm'],s['seed'])].append((r,s))
    decisions=[]
    for (alg,seed),pairs in groups.items():
        gain=None
        if budget==512:
            gains=[]
            for r,s in pairs:
                rr=list(rows(EXT/f'search/{alg}/{seed}/{slug(r["uid"])}.jsonl'))
                if not curve(rr,128)['any_correct']:
                    gains.append(int(curve(rr,256)['any_correct']))
            gain=sum(gains)/len(gains) if gains else 0.
        for r,s in pairs:
            unresolved=not s['curve']['any_correct']
            sample_resolved=int(sha256(f'saturation:{alg}:{seed}:{r["uid"]}'.encode()).hexdigest()[:8],16)/2**32<.10
            if (budget==256 and (unresolved or sample_resolved)) or (budget==512 and unresolved and gain>=.01):
                selected[r['uid']].append(dict(algorithm=alg,seed=seed,reason='unresolved' if unresolved else 'resolved_saturation_sample'))
        decisions.append(dict(algorithm=alg,seed=seed,previous_evaluated_cohort=len(pairs),gain_denominator='B128_unresolved_only',unresolved_cohort=len(gains) if budget==512 else None,any_gain=gain,extend=budget==256 or gain>=.01))
    atomic_json(ROOT/f'work/extension_{budget}.json',dict(selected))
    atomic_json(ROOT/f'work/extension_{budget}_decision.json',dict(contract_sha256=c['contract_sha256'],decisions=decisions,selected_uids=len(selected)))
    print(json.dumps(dict(budget=budget,uids=len(selected),sessions=sum(map(len,selected.values())))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','worker','finalize-smoke','extension']); p.add_argument('--mode',default='smoke',choices=['smoke','128','256','512']); p.add_argument('--rank',type=int,default=0); p.add_argument('--budget',type=int,choices=[256,512]); a=p.parse_args()
    if a.command=='prepare': prepare()
    elif a.command=='worker': worker(a.mode,a.rank)
    elif a.command=='finalize-smoke': finalize_smoke()
    else: extension(a.budget)
