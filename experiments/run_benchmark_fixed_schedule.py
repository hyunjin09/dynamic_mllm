"""Fixed-schedule execution without Stage1, a learned policy, or test tuning."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import torch

from binary_policy.executor.four_action import capture_four_action_route, capture_four_action_suffix_from_full_baseline, score_token_ids_from_cached_prompt
from binary_policy.executor.inputs import build_binary_inputs
from dense_failure_stage1.contract import model_file_hashes
from dense_failure_stage1.runtime import configure_dense_determinism
from dense_failure_stage2.benchmark_fixed_schedule import SEED, stable_key
from dense_failure_stage2.predictability_external_transfer import external_answer_specs
from dense_failure_stage2.predictability_measurement import aggregate_reference_mean_logprobs
from experiments import run_full_benchmark_end_to_end_eval as reference
from experiments.prepare_benchmark_fixed_schedule import ROOT, OUT, EXT

read_json = reference.read_json
read_jsonl = reference.read_jsonl
atomic_json = reference.atomic_json
file_sha256 = reference.file_sha256

BOUND_CODE = [
    'dense_failure_stage2/benchmark_fixed_schedule.py',
    'experiments/prepare_benchmark_fixed_schedule.py',
    'experiments/run_benchmark_fixed_schedule.py',
    'experiments/analyze_benchmark_fixed_schedule.py',
    'binary_policy/executor/four_action.py','binary_policy/executor/inputs.py',
    'binary_policy/executor/layers.py','binary_policy/executor/cache.py',
    'binary_policy/executor/generation.py','binary_policy/executor/model.py',
    'experiments/run_full_benchmark_end_to_end_eval.py',
    'experiments/run_stage2_v1_training_revised.py',
    'dense_failure_stage1/runtime.py','dense_failure_stage2/predictability_external_transfer.py',
    'dense_failure_stage2/predictability_measurement.py',
    'eval/reference/shared_prefix_eval_20260812/code/dvr_qwen/eval_metrics.py',
    'eval/reference/shared_prefix_eval_20260812/code/dvr_qwen/scripts/cache_preference_gt_router_features.py',
    'tests/test_benchmark_fixed_schedule.py',
    '.venv/lib/python3.12/site-packages/transformers/generation/utils.py',
    '.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py',
]


def freeze():
    if (OUT/'frozen_contract.json').exists():
        raise RuntimeError('contract exists')
    cfg=read_json(OUT/'execution_config.json')
    assert read_json(OUT/'splits/complete.json')['passed']
    paths=BOUND_CODE+['plans/benchmark_calibrated_fixed_read_write_schedule_plan.md']
    contract=dict(config=cfg,code_hashes={p:file_sha256(ROOT/p) for p in paths},
                  protocol_sha256=file_sha256(OUT/'protocol.md'),
                  model_hashes=model_file_hashes(Path(cfg['model']['snapshot_path'])),
                  frozen_unix=time.time())
    contract['contract_sha256']=reference.canonical_hash(contract)
    atomic_json(OUT/'frozen_contract.json',contract)
    print(contract['contract_sha256'],flush=True)


def check():
    c=read_json(OUT/'frozen_contract.json')
    assert reference.canonical_hash(c)==c['contract_sha256']
    assert read_json(OUT/'execution_config.json')==c['config']
    assert file_sha256(OUT/'splits/split_registry.jsonl')==c['config']['split_sha256']
    assert file_sha256(OUT/'protocol.md')==c['protocol_sha256']
    for path,digest in c['code_hashes'].items():
        assert file_sha256(ROOT/path)==digest,path
    if 'accepted_dense_parent' in c['config']:
        parent=c['config']['accepted_dense_parent']
        assert file_sha256(OUT/parent['manifest'])==parent['manifest_sha256']
        assert file_sha256(OUT/parent['contract_path'])==parent['contract_file_sha256']
    return c


def record_path(stage,uid):
    return EXT/stage/(sha256(uid.encode()).hexdigest()+'.json')


def align_native_decode_positions(meta):
    """HF5 generate increments the final prompt position, not its maximum.

    Multi-image prefixes can have a maximum position larger than the final text
    position. Keep prefill unchanged and encode native continuation in the
    existing cache decoder's full-length-plus-delta interface.
    """
    assert bool(meta.full_attention_mask.bool().all()), 'Phase88 requires unpadded single-UID inputs'
    last=meta.full_position_ids[:,:,-1]
    assert torch.equal(last,last[:1].expand_as(last)), 'final text rotary planes differ'
    meta.rope_deltas=last[0,:,None]+1-meta.full_prompt_len[:,None]
    return meta


class Runtime:
    def __init__(self,cfg):
        self.config=cfg
        configure_dense_determinism(SEED,cfg['backend_settings'])
        torch.set_num_threads(4)
        self.device=torch.device('cuda:0')
        self.processor,self.base,self.wrapped=reference._load_model(cfg,self.device)

    def baseline(self,row):
        reference._verify_images(row)
        inputs=reference._build_inputs(self,row)
        meta=align_native_decode_positions(build_binary_inputs(self.wrapped,inputs))
        output=capture_four_action_route(self.wrapped,{},['FULL']*28,prepared_inputs=meta,use_cache=True,native_full_rows=True)
        return inputs,output

    def measure(self,output,inputs,row,with_q):
        result=reference._generate(self,output,inputs['input_ids'],row)
        result['q']=None
        if with_q:
            try:
                specs=external_answer_specs(row)
            except ValueError as error:
                if str(error) not in ['TextVQA q references are empty','ChartQA q reference is empty']:
                    raise
                specs=[]
                result['q_invalid_reason']=str(error)
            scores=[]
            for answer in specs:
                ids=self.processor.tokenizer(str(answer['text']),add_special_tokens=False,return_tensors='pt').input_ids[0].to(self.device)
                scored=score_token_ids_from_cached_prompt(self.wrapped,output.prompt_logits,output.inputs,output.cache,ids)
                scores.append(scored.mean_logprob)
            if specs:
                result['q']=float(aggregate_reference_mean_logprobs(specs,scores))
        result['actions']=list(output.layer_actions)
        result['action_trace']=[asdict(x) for x in output.layer_stats]
        assert [x['action'] for x in result['action_trace']]==result['actions']
        result['prompt_tokens']=int(inputs['input_ids'].shape[1])
        result['visual_tokens']=int(output.inputs.visual_states.shape[1])
        return result

    def route(self,baseline,actions):
        changes=[i for i,a in enumerate(actions) if a!='FULL']
        if not changes:
            return baseline
        first=min(changes)
        return capture_four_action_suffix_from_full_baseline(self.wrapped,baseline,first,actions[first:])


def matching(left,right):
    return all(left[k]==right[k] for k in ['generated_token_ids','generated_answer','correct','score'])


def authoritative_dense(runtime,row,unified,native,source):
    """Keep native generation authoritative; q has one consistent executor path."""
    result=dict(unified)
    result['q_source']='native-causal all-FULL teacher forcing; HF5 final-prompt-position continuation'
    result['native_reference_source']=source
    if not matching(unified,native):
        replays=[]
        for _ in range(2):
            replay=reference._native_dense_generation(runtime,row,extract_features=False)
            assert matching(replay,native),f'native repeat/reference mismatch: {row["uid"]}'
            replays.append({k:replay[k] for k in ['generated_token_ids','generated_answer','score','correct']})
        assert unified['score']==native['score'] and unified['correct']==native['correct'],f'all-FULL scorer mismatch: {row["uid"]}'
        result.update(unified_generated_token_ids=unified['generated_token_ids'],unified_generated_answer=unified['generated_answer'],unified_token_mismatch=True,native_repeat_replays=replays)
        for key in ['generated_token_ids','generated_answer','score','correct']:result[key]=native[key]
    else:
        result['unified_token_mismatch']=False
    return result


def accepted_record(c,stage,path,record,parent_records):
    if record['contract_sha256']==c['contract_sha256']:return
    assert stage=='dense' and 'accepted_dense_parent' in c['config']
    assert record['contract_sha256']==c['config']['accepted_dense_parent']['contract_sha256']
    expected=parent_records[record['uid']]
    assert str(path.resolve())==expected['path'] and file_sha256(path)==expected['sha256']


def smoke(kind):
    c=check();rt=Runtime(c['config'])
    rows=read_jsonl(OUT/'splits/split_registry.jsonl')
    selected=[]
    for benchmark in sorted({r['benchmark'] for r in rows}):
        pool=[r for r in rows if r['benchmark']==benchmark and r['split']=='CAL']
        assert pool,benchmark
        selected.append(min(pool,key=lambda r:stable_key(r['uid'])))
    if kind=='dense':
        for suffix in ['Music_53','Psychology_128']:
            selected.append(next(r for r in rows if r['uid']=='mmmu_pro_standard_test:mmmu_pro_standard_test_test_'+suffix))
    output=[]
    for row in selected:
        native=reference._native_dense_generation(rt,row,extract_features=False)
        native.pop('features',None)
        inputs,base=rt.baseline(row)
        dense=rt.measure(base,inputs,row,True)
        token_parity=matching(native,dense)
        if row['split']=='TEST':assert token_parity, 'Dense continuation regression sentinel mismatch'
        dense=authoritative_dense(rt,row,dense,native,'fresh native smoke')
        again=reference._native_dense_generation(rt,row,extract_features=False)
        assert matching(native,again)
        checks=dict(uid=row['uid'],benchmark=row['benchmark'],native_full_token_parity=token_parity,native_score_correctness_parity=True,native_repeat=True,dense_only_test_sentinel=row['split']=='TEST')
        if kind=='interventions':
            assert read_json(OUT/'parity/dense_complete.json')['passed']
            for action in ['READ_ONLY','WRITE_ONLY','IGNORE']:
                actions=['FULL']*28;actions[14]=action
                first=rt.measure(rt.route(base,actions),inputs,row,True)
                second=rt.measure(rt.route(base,actions),inputs,row,True)
                direct=capture_four_action_route(rt.wrapped,{},actions,prepared_inputs=base.inputs,use_cache=True,native_full_rows=True)
                direct_result=rt.measure(direct,inputs,row,True)
                assert matching(first,second) and first['q']==second['q']
                assert matching(first,direct_result) and first['q']==direct_result['q']
            actions=['FULL']*28;actions[27]='READ_ONLY'
            terminal=rt.measure(rt.route(base,actions),inputs,row,True)
            expected=dict(dense,generated_token_ids=dense.get('unified_generated_token_ids',dense['generated_token_ids']),generated_answer=dense.get('unified_generated_answer',dense['generated_answer']))
            assert matching(terminal,expected) and terminal['q']==dense['q']
            checks.update(forced_actions_repeat=True,full_route_suffix_parity=True,terminal_write_zero=True)
        output.append(checks)
        print(json.dumps(checks),flush=True)
    atomic_json(OUT/f'parity/{kind}_smoke.json',dict(passed=True,contract_sha256=c['contract_sha256'],rows=output))


def worker(stage,rank):
    c=check();cfg=c['config']
    assert stage in ['dense','R','W','methods','global','random']
    dependency={'dense':'parity/dense_smoke.json','R':'parity/interventions_smoke.json','W':'calibration/R_complete.json','methods':'schedules/schedule_freeze.json','global':'heldout/methods_complete.json','random':'heldout/global_complete.json'}[stage]
    assert read_json(OUT/dependency)['passed']
    rows=read_jsonl(OUT/'splits/split_registry.jsonl')
    rows=[r for r in rows if stage=='dense' or r['split']==('CAL' if stage in ['R','W'] else 'TEST')]
    rows=sorted(rows,key=lambda r:stable_key(r['uid']))
    rt=Runtime(cfg)
    parent_records={r['uid']:r for r in read_jsonl(OUT/cfg['accepted_dense_parent']['manifest'])} if 'accepted_dense_parent' in cfg else {}
    prior={}
    if stage=='dense':
        for path in (ROOT/'analysis/dense_failure_stage2/full_benchmark_eval/dense').glob('*_results.jsonl'):
            prior.update({r['uid']:r for r in read_jsonl(path)})
    schedules=None
    if stage in ['methods','global','random']:
        frozen=read_json(OUT/'schedules/schedule_freeze.json')
        for path,digest in frozen['files'].items():
            assert file_sha256(OUT/path)==digest
        schedules=read_json(OUT/'schedules/benchmark_specific_schedules.json')
        global_actions=read_json(OUT/'schedules/global_schedule.json')['actions']
        random_rows=read_jsonl(OUT/'schedules/random_schedule_registry.jsonl')
    completed=0;fresh=0;reused=0;started=time.monotonic()
    for i,row in enumerate(rows):
        if i%4!=rank:
            continue
        path=record_path(stage,row['uid'])
        if path.exists():
            saved=read_json(path);accepted_record(c,stage,path,saved,parent_records);completed+=1;reused+=1;continue
        tick=time.monotonic()
        inputs,base=rt.baseline(row)
        results={}
        if stage=='dense':
            dense=rt.measure(base,inputs,row,True)
            if row['uid'] in prior:
                dense=authoritative_dense(rt,row,dense,prior[row['uid']],'Phase69 exact same-UID native; fresh unified token parity or explicit scorer-equivalent dual-output replay')
            else:
                native=reference._native_dense_generation(rt,row,extract_features=False)
                dense=authoritative_dense(rt,row,dense,native,'fresh native dense generation')
            results['dense']=dense
        elif stage in ['R','W']:
            dense=read_json(record_path('dense',row['uid']))['results']['dense']
            # One native-causal canonical prefix per UID; intervention changes only the declared bit.
            for layer in range(28):
                actions=['FULL']*28;actions[layer]='WRITE_ONLY' if stage=='R' else 'READ_ONLY'
                branch=rt.measure(rt.route(base,actions),inputs,row,True)
                if stage=='W' and layer==27:
                    expected=dict(dense,generated_token_ids=dense.get('unified_generated_token_ids',dense['generated_token_ids']),generated_answer=dense.get('unified_generated_answer',dense['generated_answer']))
                    assert matching(branch,expected) and branch['q']==dense['q'],row['uid']
                results[f'{stage}{layer}']=branch
        else:
            benchmark=row['benchmark_family']
            requested=schedules[benchmark]['methods'] if stage=='methods' else {'GLOBAL':global_actions} if stage=='global' else {r['method']:r['actions'] for r in random_rows if r['benchmark']==benchmark}
            dense=read_json(record_path('dense',row['uid']))['results']['dense']
            reuse={tuple(['FULL']*28):dense}
            for earlier in ['methods','global']:
                saved_path=record_path(earlier,row['uid'])
                if earlier!=stage and saved_path.exists():
                    for value in read_json(saved_path)['results'].values():
                        reuse[tuple(value['actions'])]=value
            for method,actions in requested.items():
                key=tuple(actions)
                if key not in reuse:
                    begin=time.monotonic()
                    reuse[key]=rt.measure(rt.route(base,actions),inputs,row,False)
                    reuse[key]['suffix_decode_seconds']=time.monotonic()-begin
                results[method]=reuse[key]
        elapsed=time.monotonic()-tick
        atomic_json(path,dict(uid=row['uid'],benchmark=row['benchmark'],benchmark_family=row['benchmark_family'],image_group_id=row['image_group_id'],split=row['split'],stage=stage,contract_sha256=c['contract_sha256'],results=results,elapsed_seconds=elapsed,rank=rank))
        completed+=1;fresh+=1
        print(json.dumps(dict(stage=stage,rank=rank,uid=row['uid'],completed=completed,new_completed=fresh,reused_uids=reused,elapsed_seconds=elapsed,worker_seconds=time.monotonic()-started)),flush=True)
        del base,inputs,results
    atomic_json(EXT/f'{stage}_rank{rank}_complete.json',dict(passed=True,rank=rank,stage=stage,uids=completed,fresh_uids=fresh,reused_uids=reused,elapsed_seconds=time.monotonic()-started,contract_sha256=c['contract_sha256']))


def finalize(stage):
    c=check();rows=read_jsonl(OUT/'splits/split_registry.jsonl')
    rows=[r for r in rows if stage=='dense' or r['split']==('CAL' if stage in ['R','W'] else 'TEST')]
    for rank in range(4):
        report=read_json(EXT/f'{stage}_rank{rank}_complete.json');assert report['passed'] and report['contract_sha256']==c['contract_sha256']
    evidence=[]
    parent_records={r['uid']:r for r in read_jsonl(OUT/c['config']['accepted_dense_parent']['manifest'])} if 'accepted_dense_parent' in c['config'] else {}
    for row in rows:
        path=record_path(stage,row['uid']);r=read_json(path)
        assert r['uid']==row['uid'];accepted_record(c,stage,path,r,parent_records)
        expected={'dense'} if stage=='dense' else {f'{stage}{i}' for i in range(28)} if stage in ['R','W'] else {'M1','M2','M3'} if stage=='methods' else {'GLOBAL'} if stage=='global' else {f'RANDOM_{i:03d}' for i in range(c['config']['random_schedules'])}
        assert set(r['results'])==expected
        evidence.append(dict(uid=row['uid'],path=str(path),sha256=file_sha256(path)))
    target=OUT/('parity' if stage=='dense' else 'calibration' if stage in ['R','W'] else 'heldout')
    reference.atomic_jsonl(target/f'{stage}_execution_manifest.jsonl',evidence)
    atomic_json(target/f'{stage}_complete.json',dict(passed=True,uids=len(rows),contract_sha256=c['contract_sha256']))
    print(stage,'complete',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','smoke','worker','finalize']);parser.add_argument('--stage',default='dense');parser.add_argument('--rank',type=int,default=0);args=parser.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='smoke':smoke(args.stage)
    elif args.command=='worker':worker(args.stage,args.rank)
    else:finalize(args.stage)
