"""Slurm-only eight-worker current-runtime gate and resumable binary route replay."""
import argparse
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from common import *


def runtime_imports():
    # Prevent any import from writing into the immutable source package.
    sys.dont_write_bytecode = True
    runtime = PACKAGE/'07_RESUME_TOOLS'
    sys.path.insert(0,str(runtime))
    import torch
    from transformers import AutoProcessor
    from dvr_qwen.modeling_dvr_qwen2_5_vl import DVRQwen2_5_VLForConditionalGeneration
    from dvr_qwen.binary_generate import binary_dvrc_greedy_generate, prepare_binary_dvrc_inputs
    from dvr_qwen.eval_metrics import score_prediction
    from dvr_qwen.scripts.cache_preference_gt_router_features import build_processor_inputs
    return torch, AutoProcessor, DVRQwen2_5_VLForConditionalGeneration, binary_dvrc_greedy_generate, prepare_binary_dvrc_inputs, score_prediction, build_processor_inputs


def validate_resume(payload, source_by_key, contract_hash, uid):
    if payload['sample_uid'] != uid or payload['contract_sha256'] != contract_hash:
        raise ValueError('Resume sample/contract mismatch')
    records = payload['records']
    seen=set()
    for r in records:
        key=r['route_key']
        if key in seen or key not in source_by_key:
            raise ValueError('Duplicate or unknown durable replay record')
        seen.add(key)
        actual=digest({k:v for k,v in r.items() if k!='record_sha256'})
        if actual!=r['record_sha256']:
            raise ValueError('Resume record hash mismatch')
        if r['source_record_sha256']!=digest(source_by_key[key]['source_record']):
            raise ValueError('Resume source record mismatch')
        if r['sample_uid']!=uid or r['contract_sha256']!=contract_hash:
            raise ValueError('Resume route binding mismatch')
        if r['replay_status'] not in ['COMPLETE','ERROR']:
            raise ValueError('Incomplete durable route record')
        expected=source_by_key[key]['source_record']
        if r['mask_key']!=expected['mask_key'] or r['source_route_id']!=expected['route_id']:
            raise ValueError('Resume route identity mismatch')
    if payload.get('complete') and seen!=set(source_by_key):
        raise ValueError('False sample completion marker')
    return records


def gate_passes(anchor_records, sparse_records, anchors, world_size=WORLD_SIZE):
    uids=[r['uid'] for r in anchor_records]
    return (len(uids)==len(anchors) and len(set(uids))==len(anchors)
            and set(uids)=={r['uid'] for r in anchors}
            and all(r.get('passed') for r in anchor_records)
            and len(sparse_records)==world_size
            and all(r.get('passed') for r in sparse_records)
            and len({digest(r['outputs']) for r in sparse_records})==1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--imports-only',action='store_true')
    args=parser.parse_args()
    if args.imports_only:
        runtime_imports()
        print('Packaged runtime/input builder/scorer import successfully; no model or GPU computation.')
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU execution requires Slurm')
    rank=int(os.environ['LOCAL_RANK'])
    world=int(os.environ['WORLD_SIZE'])
    assert world==WORLD_SIZE
    c=read_contract()
    assert file_hash(OUT/'source_inventory/prepared_index.jsonl')==c['prepared_index_sha256']
    assert file_hash(OUT/'source_inventory/relocated_samples.jsonl')==c['relocated_samples_sha256']
    assert file_hash(OUT/'replay_gate/frozen_anchor_rows.jsonl')==c['anchor_sha256']
    assert file_hash(OUT/'replay_gate/frozen_sparse_probes.jsonl')==c['sparse_probe_sha256']
    torch,AutoProcessor,Model,generate,prepare,score_prediction,build_inputs=runtime_imports()
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    device=torch.device('cuda',rank)
    assert torch.cuda.device_count()==WORLD_SIZE
    import importlib.metadata as md
    for name, version in c['installed_versions'].items():
        assert md.version(name)==version, f'Environment drift: {name}'
    job=os.environ['SLURM_JOB_ID']
    gate_dir=OUT/'replay_gate'/f'job_{job}'
    env=dict(hostname=socket.gethostname(),gpu=torch.cuda.get_device_name(rank),cuda_runtime=torch.version.cuda,
             torch=torch.__version__,python=sys.version,transformers=md.version('transformers'),rank=rank,
             slurm_job_id=job,model_snapshot=str(MODEL),model_revision=REVISION,
             processor_snapshot=str(MODEL),tokenizer_snapshot=str(MODEL),
             critical_environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONHASHSEED']},
             small_model_hashes=c['small_model_hashes'],weights='size-verified; full hashes intentionally not recomputed',
             cuda_driver_report=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())
    atomic_json(gate_dir/f'env_rank{rank}.json',env)
    if rank==0:
        atomic_json(OUT/'replay_gate/current_server_env.json',env)
    processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True,use_fast=False)
    model=Model.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,device_map={'':str(device)},attn_implementation='sdpa',local_files_only=True)
    model.eval()
    assert model.base_model.generation_config.repetition_penalty==1.05
    samples={r['uid']:r for r in rows(OUT/'source_inventory/relocated_samples.jsonl')}

    def inputs_for(sample):
        assert file_hash(sample['local_image_path'])==sample['image_content_sha256']
        # Paths are already relocated and verified. Never probe original paths.
        inputs=build_inputs(processor,sample,data_root=PACKAGE/'01_SOURCE_DATA/vqa_train_10k')
        return {k:v.to(device) if torch.is_tensor(v) else v for k,v in inputs.items()}

    def binary(sample,inputs,mask,prepared):
        with torch.inference_mode():
            output=generate(model,inputs,visual_on_mask=torch.tensor([[int(x) for x in mask]],dtype=torch.bool,device=device),
                            max_new_tokens=int(sample['max_new_tokens']),eos_token_ids=[151645],repetition_penalty=1.05,
                            prepared_binary_inputs=prepared)
        ids=output.generated_ids.detach().cpu().view(-1).tolist()
        answer=processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
        score=float(score_prediction(sample['metric_name'],answer,sample['answer'],sample.get('all_answer_norms')))
        realized=output.state.route_binary.detach().cpu().view(-1).tolist()
        assert [int(v) for v in realized]==[int(v) for v in mask]
        result=dict(generated_ids=ids,answer=answer,score=score,correctness=score>=float(sample['correctness_threshold']),
                    visual_tokens=int(prepared.visual_valid_mask.sum().item()),
                    text_tokens=int(prepared.text_valid_mask.sum().item()),action_trace=[int(v) for v in realized])
        del output
        return result

    anchors=list(rows(OUT/'replay_gate/frozen_anchor_rows.jsonl'))
    anchor_records=[]
    for index,anchor in enumerate(anchors):
        if index%world!=rank:
            continue
        sample=samples[anchor['uid']]
        result=dict(uid=sample['uid'],passed=False)
        try:
            inputs=inputs_for(sample)
            with torch.inference_mode():
                native=model.base_model.generate(**{k:v for k,v in inputs.items() if k!='instruction_token_mask'},
                    max_new_tokens=int(sample['max_new_tokens']),do_sample=False,num_beams=1,repetition_penalty=1.05,
                    eos_token_id=151645,return_dict_in_generate=True)
                ids=native.sequences[:,inputs['input_ids'].shape[1]:].detach().cpu().view(-1).tolist()
                prepared=prepare(model,inputs)
            answer=processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
            score=float(score_prediction(sample['metric_name'],answer,sample['answer'],sample.get('all_answer_norms')))
            current=binary(sample,inputs,'1'*LAYERS,prepared)
            source=anchor['current_binary_all_on']
            result.update(current_hf=dict(generated_ids=ids,answer=answer,score=score),current_binary=current,
                          exact_token_match=ids==current['generated_ids'],exact_answer_match=answer==current['answer'],
                          exact_score_match=score==current['score'],
                          correctness_match=(score>=float(sample['correctness_threshold']))==current['correctness'],
                          saved_3b_token_match=current['generated_ids']==source['generated_ids'],
                          saved_3b_answer_match=current['answer']==source['prediction'],
                          saved_3b_correctness_match=current['correctness']==source['result_correct'])
            result['passed']=all(result[k] for k in ['exact_token_match','exact_answer_match','exact_score_match','correctness_match'])
            del inputs,prepared,native
        except Exception as e:
            result['error']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
        anchor_records.append(result)
        print('anchor',rank,result['uid'],result['passed'],flush=True)
    write_jsonl(gate_dir/f'anchors_rank{rank}.jsonl',anchor_records)
    probes=list(rows(OUT/'replay_gate/frozen_sparse_probes.jsonl'))
    sparse_outputs=[]
    sparse_passed=True
    for probe in probes:
        sample=samples[probe['uid']]
        record=dict(uid=sample['uid'],route_key=probe['route_key'])
        try:
            inputs=inputs_for(sample)
            with torch.inference_mode():prepared=prepare(model,inputs)
            first=binary(sample,inputs,probe['source_record']['mask_key'],prepared)
            second=binary(sample,inputs,probe['source_record']['mask_key'],prepared)
            record.update(first=first,second=second,passed=first==second)
            del prepared,inputs
        except Exception as e:
            record.update(passed=False,error=dict(type=type(e).__name__,message=str(e)))
        sparse_passed &= record['passed']
        sparse_outputs.append(record)
    atomic_json(gate_dir/f'sparse_rank{rank}.json',dict(passed=sparse_passed,outputs=sparse_outputs))
    atomic_json(gate_dir/f'gate_rank{rank}_complete.json',dict(passed=all(r['passed'] for r in anchor_records) and sparse_passed,contract_sha256=c['contract_sha256']))
    deadline=time.monotonic()+3600
    while not all((gate_dir/f'gate_rank{k}_complete.json').exists() for k in range(world)):
        if time.monotonic()>deadline:raise TimeoutError('Waiting for peer gate results')
        time.sleep(1)
    if rank==0:
        all_anchors=[r for k in range(world) for r in rows(gate_dir/f'anchors_rank{k}.jsonl')]
        all_sparse=[read_json(gate_dir/f'sparse_rank{k}.json') for k in range(world)]
        passed=gate_passes(all_anchors,all_sparse,anchors)
        gate=dict(passed=passed,contract_sha256=c['contract_sha256'],job_id=job,anchors=len(all_anchors),
                  exact_token_matches=sum(r.get('exact_token_match',False) for r in all_anchors),
                  exact_answer_matches=sum(r.get('exact_answer_match',False) for r in all_anchors),
                  correctness_matches=sum(r.get('correctness_match',False) for r in all_anchors),
                  sparse_repeat_cross_rank_pass=all(r['passed'] for r in all_sparse) and len({digest(r['outputs']) for r in all_sparse})==1,
                  anchor_rows=all_anchors,scope='current all-on HF/binary equivalence plus sparse repeatability; not historical sparse-output parity')
        atomic_json(OUT/'replay_gate/gate_result.json',gate)
        write_text(OUT/'replay_gate/gate_report.md',f"Generation-anchor gate {'PASS' if passed else 'FAIL'} for job{job}.\n\n{gate['exact_token_matches']}/32 exact current HF/binary token matches. Sparse repeated cross-rank check: {gate['sparse_repeat_cross_rank_pass']}. Historical matches remain descriptive. Full replay {'admitted' if passed else 'STOPPED; diagnose before retry'}.\n")
        atomic_json(gate_dir/'decision.json',gate)
    while not (gate_dir/'decision.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Waiting for gate decision')
        time.sleep(1)
    if not read_json(gate_dir/'decision.json')['passed']:
        raise RuntimeError('Hard generation/sparse gate failed; full replay not started')
    selected=[r for r in rows(OUT/'source_inventory/prepared_index.jsonl') if r['rank']==rank]
    n_success=n_errors=n_reused=0
    started=time.monotonic()
    for si,item in enumerate(selected):
        assert file_hash(item['path'])==item['sha256']
        source=list(rows(item['path']))
        source_by_key={r['route_key']:r for r in source}
        assert len(source_by_key)==item['routes']
        dense_source=source_by_key[item['dense_route_key']]['source_record']
        uid=item['uid'];sample=samples[uid]
        path=OUT/'replay/by_sample'/f'{stem(uid)}.json'
        existing=read_json(path) if path.exists() else dict(sample_uid=uid,contract_sha256=c['contract_sha256'],records=[],complete=False)
        records=validate_resume(existing,source_by_key,c['contract_sha256'],uid)
        completed={r['route_key'] for r in records}
        n_reused+=len(records)
        n_success+=sum(r['replay_status']=='COMPLETE' for r in records)
        n_errors+=sum(r['replay_status']=='ERROR' for r in records)
        if existing['complete']:
            continue
        inputs=prepared=None
        preparation_error=None
        try:
            inputs=inputs_for(sample)
            with torch.inference_mode():prepared=prepare(model,inputs)
        except Exception as e:
            preparation_error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
        for source_item in source:
            key=source_item['route_key']
            if key in completed:continue
            r=source_item['source_record'];t=time.monotonic()
            record=dict(sample_uid=uid,route_key=key,source_route_id=r['route_id'],mask_key=r['mask_key'],action_sequence=r['visual_on_mask'],
                        dataset=r['benchmark'],old_correctness=old_label(r),current_correctness=None,old_answer=r['prediction'],current_answer=None,
                        old_generated_answer=r['prediction'],current_generated_answer=None,old_generated_ids=r['generated_ids'],
                        old_score=r['score'],current_score=None,source_record_sha256=digest(r),contract_sha256=c['contract_sha256'],
                        model_revision=REVISION,evaluation_contract=dict(metric=sample['metric_name'],threshold=sample['correctness_threshold']),
                        phase=r['observed_phase'],provenance=sorted({o['family'] for o in r.get('phase1_origins',[])+r.get('phase2_origins',[])}),
                        route_length=LAYERS,interventions=LAYERS-r['num_visual_on_layers'],attempt_count=1,slurm_job_id=job,rank=rank)
            try:
                if preparation_error:raise RuntimeError('Sample preparation failed: '+json.dumps(preparation_error))
                current=binary(sample,inputs,r['mask_key'],prepared)
                assert current['visual_tokens']==r.get('visual_tokens',dense_source['visual_tokens']), 'Processor visual token count drift'
                assert current['text_tokens']==r.get('text_tokens',dense_source['text_tokens']), 'Processor text token count drift'
                record.update(replay_status='COMPLETE',current_answer=current['answer'],current_generated_answer=current['answer'],
                              current_generated_ids=current['generated_ids'],current_correctness=current['correctness'],current_score=current['score'],
                              visual_tokens=current['visual_tokens'],text_tokens=current['text_tokens'],action_trace=current['action_trace'])
                n_success+=1
            except Exception as e:
                record.update(replay_status='ERROR',error_type=type(e).__name__,error_message=str(e))
                n_errors+=1
                # Stop a systematic failure after preserving its first failed route.
                # The remaining census stays explicit as pending, never mislabeled wrong.
            record['runtime_seconds']=time.monotonic()-t
            record['label_transition']=transition(record['old_correctness'],record['current_correctness'])
            record['record_sha256']=digest(record)
            records.append(record)
            completed.add(key)
            if len(records)%c['checkpoint_routes']==0 or record['replay_status']=='ERROR':
                atomic_json(path,dict(existing,records=records,complete=False))
            if record['replay_status']=='ERROR':
                atomic_json(OUT/'replay'/f'rank{rank}_failure.json',record)
                raise RuntimeError(f'Replay stopped at first runtime error; record retained: {uid}/{key}')
        atomic_json(path,dict(existing,records=records,complete=len(records)==item['routes']))
        atomic_json(OUT/'replay'/f'rank{rank}_progress.json',dict(samples=si+1,total_samples=len(selected),success=n_success,errors=n_errors,reused=n_reused,elapsed_seconds=time.monotonic()-started,job_id=job,contract_sha256=c['contract_sha256']))
        del inputs,prepared
        if (si+1)%10==0:print('replay',rank,si+1,len(selected),n_success,'routes',flush=True)
    atomic_json(OUT/'replay'/f'rank{rank}_complete.json',dict(passed=True,samples=len(selected),success=n_success,errors=n_errors,reused=n_reused,elapsed_seconds=time.monotonic()-started,contract_sha256=c['contract_sha256'],job_id=job))


if __name__=='__main__':
    main()
