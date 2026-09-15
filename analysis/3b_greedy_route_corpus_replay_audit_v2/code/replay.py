"""Eight-worker, strictly staged current Dense and routed replay."""
from common import *
import argparse
import importlib.metadata as md
import time
import traceback
from runtime import runtime_imports
from parity_repair import install
from execution import read_execution_contract

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


class Engine:
    def __init__(self,c,rank):
        torch,Processor,Model,generate,prepare,score,build_inputs=runtime_imports()
        self.torch=torch;self.generate=generate;self.prepare=prepare;self.score=score;self.build_inputs=build_inputs
        self.device=torch.device('cuda',rank)
        assert torch.cuda.device_count()==WORLD_SIZE
        torch.cuda.set_device(rank);torch.set_num_threads(4)
        for name,version in c['installed_versions'].items():assert md.version(name)==version,name
        self.processor=Processor.from_pretrained(str(MODEL),local_files_only=True,use_fast=False)
        self.model=Model.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,device_map={'':str(self.device)},
                                        attn_implementation='sdpa',local_files_only=True).eval()
        install(self.model)
        assert self.model.base_model.generation_config.repetition_penalty==1.05

    def inputs(self,sample):
        assert file_hash(sample['local_image_path'])==sample['image_content_sha256']
        inputs={k:v.to(self.device) if self.torch.is_tensor(v) else v for k,v in self.build_inputs(
            self.processor,sample,data_root=PACKAGE/'01_SOURCE_DATA/vqa_train_10k').items()}
        with self.torch.inference_mode():prepared=self.prepare(self.model,inputs)
        return inputs,prepared

    def label(self,ids,sample):
        answer=self.processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
        score=float(self.score(sample['metric_name'],answer,sample['answer'],sample.get('all_answer_norms')))
        return dict(generated_ids=ids,answer=answer,score=score,correctness=score>=float(sample['correctness_threshold']))

    def native(self,sample,inputs):
        with self.torch.inference_mode():
            output=self.model.base_model.generate(**{k:v for k,v in inputs.items() if k!='instruction_token_mask'},
                max_new_tokens=int(sample['max_new_tokens']),do_sample=False,num_beams=1,repetition_penalty=1.05,
                eos_token_id=151645,return_dict_in_generate=True)
        return self.label(output.sequences[:,inputs['input_ids'].shape[1]:].view(-1).tolist(),sample)

    def binary(self,sample,inputs,mask,prepared):
        with self.torch.inference_mode():
            output=self.generate(self.model,inputs,visual_on_mask=self.torch.tensor([[int(x) for x in mask]],
                dtype=self.torch.bool,device=self.device),max_new_tokens=int(sample['max_new_tokens']),
                eos_token_ids=[151645],repetition_penalty=1.05,prepared_binary_inputs=prepared)
        actual=[int(x) for x in output.state.route_binary.view(-1).tolist()]
        assert actual==[int(x) for x in mask]
        return dict(**self.label(output.generated_ids.view(-1).tolist(),sample),action_trace=actual,
                    visual_tokens=int(prepared.visual_valid_mask.sum().item()),text_tokens=int(prepared.text_valid_mask.sum().item()))


def preflight(engine,c,samples,rank,job,stage):
    target=OUT/'replay_gate'/f'job_{job}_{stage}'
    anchors=list(rows(OUT/'replay_gate/frozen_anchor_rows.jsonl'));records=[]
    for index,anchor in enumerate(anchors):
        if index%WORLD_SIZE!=rank:continue
        sample=samples[anchor['uid']];inputs,prepared=engine.inputs(sample)
        hf=engine.native(sample,inputs);custom=engine.binary(sample,inputs,'1'*LAYERS,prepared)
        records.append(dict(uid=sample['uid'],passed=all(hf[k]==custom[k] for k in hf),native=hf,custom=custom))
        del inputs,prepared
    write_jsonl(target/f'anchors_rank{rank}.jsonl',records)
    probes=[]
    for probe in rows(OUT/'replay_gate/frozen_sparse_probes.jsonl'):
        sample=samples[probe['uid']];inputs,prepared=engine.inputs(sample)
        first=engine.binary(sample,inputs,probe['source_record']['mask_key'],prepared)
        second=engine.binary(sample,inputs,probe['source_record']['mask_key'],prepared)
        probes.append(dict(uid=sample['uid'],route_key=probe['route_key'],first=first,second=second,passed=first==second))
        del inputs,prepared
    atomic_json(target/f'sparse_rank{rank}.json',dict(passed=all(r['passed'] for r in probes),outputs=probes))
    atomic_json(target/f'rank{rank}_complete.json',dict(contract_sha256=c['contract_sha256']))
    deadline=time.monotonic()+1800
    while not all((target/f'rank{k}_complete.json').exists() for k in range(WORLD_SIZE)):
        if time.monotonic()>deadline:raise TimeoutError('Peer preflight did not complete')
        time.sleep(1)
    if rank==0:
        all_anchors=[r for k in range(WORLD_SIZE) for r in rows(target/f'anchors_rank{k}.jsonl')]
        all_sparse=[read_json(target/f'sparse_rank{k}.json') for k in range(WORLD_SIZE)]
        gate=dict(passed=gate_passes(all_anchors,all_sparse,anchors),contract_sha256=c['contract_sha256'],
                  job_id=job,stage=stage,anchors=32,exact_matches=sum(r['passed'] for r in all_anchors),world_size=WORLD_SIZE,
                  sparse_cross_worker_pass=all(r['passed'] for r in all_sparse) and len({digest(r['outputs']) for r in all_sparse})==1)
        atomic_json(target/'decision.json',gate)
        atomic_json(OUT/'replay_gate/gate_result.json',gate)
    while not (target/'decision.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Preflight decision missing')
        time.sleep(1)
    assert read_json(target/'decision.json')['passed'], 'Eight-worker preflight failed; corpus replay blocked'


def replay_record(engine,c,sample,source_item,dense_source,inputs,prepared,rank,job,stage):
    r=source_item['source_record'];started=time.monotonic()
    record=dict(sample_uid=sample['uid'],route_key=source_item['route_key'],source_route_id=r['route_id'],
        mask_key=r['mask_key'],action_sequence=r['visual_on_mask'],dataset=r['benchmark'],
        old_correctness=old_label(r),current_correctness=None,old_answer=r['prediction'],current_answer=None,
        old_generated_answer=r['prediction'],current_generated_answer=None,old_generated_ids=r['generated_ids'],
        old_score=r['score'],current_score=None,source_record_sha256=digest(r),contract_sha256=c['contract_sha256'],
        model_revision=REVISION,evaluation_contract=dict(metric=sample['metric_name'],threshold=sample['correctness_threshold']),
        phase=r['observed_phase'],provenance=sorted({o['family'] for o in r.get('phase1_origins',[])+r.get('phase2_origins',[])}),
        route_length=LAYERS,interventions=LAYERS-r['num_visual_on_layers'],attempt_count=1,slurm_job_id=job,rank=rank,stage=stage)
    try:
        current=engine.binary(sample,inputs,r['mask_key'],prepared)
        assert current['visual_tokens']==r.get('visual_tokens',dense_source['visual_tokens'])
        assert current['text_tokens']==r.get('text_tokens',dense_source['text_tokens'])
        record.update(replay_status='COMPLETE',current_answer=current['answer'],current_generated_answer=current['answer'],
            current_generated_ids=current['generated_ids'],current_correctness=current['correctness'],current_score=current['score'],
            visual_tokens=current['visual_tokens'],text_tokens=current['text_tokens'],action_trace=current['action_trace'])
    except Exception as exc:
        record.update(replay_status='ERROR',error_type=type(exc).__name__,error_message=str(exc),traceback=traceback.format_exc())
    record['runtime_seconds']=time.monotonic()-started
    record['label_transition']=transition(record['old_correctness'],record['current_correctness'])
    record['record_sha256']=digest(record)
    return record


def require_dense_record(record,item,source,c):
    payload=dict(sample_uid=item['uid'],contract_sha256=c['contract_sha256'],records=[record],complete=False)
    validate_resume(payload,source,c['contract_sha256'],item['uid'])
    assert record['route_key']==item['dense_route_key'] and record['mask_key']=='1'*LAYERS
    assert record['replay_status']=='COMPLETE', 'Dense replay error blocks routed replay'
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['dense','routes'])
    stage=parser.parse_args().stage
    assert os.environ.get('SLURM_JOB_ID')
    rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==WORLD_SIZE
    job=os.environ['SLURM_JOB_ID'];c=read_execution_contract()
    if stage=='routes':
        done=read_json(OUT/'dense/stage_complete.json')
        assert done['passed'] and done['samples']==10000 and done['contract_sha256']==c['contract_sha256']
        assert file_hash(OUT/'dense/current_dense_manifest.jsonl')==done['manifest_sha256']
    samples={r['uid']:r for r in rows(OUT/'source_inventory/relocated_samples.jsonl')}
    engine=Engine(c,rank);preflight(engine,c,samples,rank,job,stage)
    selected=[r for r in rows(OUT/'source_inventory/prepared_index.jsonl') if r['rank']==rank]
    base=OUT/('dense' if stage=='dense' else 'replay')
    started=time.monotonic();success=reused=0
    for si,item in enumerate(selected):
        assert file_hash(item['path'])==item['sha256']
        source=list(rows(item['path']));source_by_key={r['route_key']:r for r in source}
        assert len(source_by_key)==item['routes']
        dense_item=source_by_key[item['dense_route_key']];dense_path=OUT/'dense/by_sample'/f"{stem(item['uid'])}.json"
        sample=samples[item['uid']]
        if stage=='dense' and dense_path.exists():
            require_dense_record(read_json(dense_path),item,source_by_key,c);reused+=1;continue
        if stage=='routes':
            dense=require_dense_record(read_json(dense_path),item,source_by_key,c)
            path=OUT/'replay/by_sample'/f"{stem(item['uid'])}.json"
            existing=read_json(path) if path.exists() else dict(sample_uid=item['uid'],contract_sha256=c['contract_sha256'],records=[dense],complete=False)
            records=validate_resume(existing,source_by_key,c['contract_sha256'],item['uid'])
            assert all(r['replay_status']=='COMPLETE' for r in records), 'Preserved route error requires diagnosis before retry'
            bound_dense=next(r for r in records if r['route_key']==item['dense_route_key'])
            assert bound_dense==dense, 'Dense record changed between stages'
            reused+=len(records)
            if existing['complete']:continue
            done_keys={r['route_key'] for r in records}
        inputs,prepared=engine.inputs(sample)
        if stage=='dense':
            record=replay_record(engine,c,sample,dense_item,dense_item['source_record'],inputs,prepared,rank,job,stage)
            atomic_json(dense_path,record)
            if record['replay_status']!='COMPLETE':raise RuntimeError('Dense replay failed; error preserved')
            success+=1
        else:
            for source_item in source:
                if source_item['route_key'] in done_keys:continue
                record=replay_record(engine,c,sample,source_item,dense_item['source_record'],inputs,prepared,rank,job,stage)
                records.append(record);done_keys.add(record['route_key'])
                if record['replay_status']=='COMPLETE':success+=1
                if len(records)%c['checkpoint_routes']==0 or record['replay_status']=='ERROR':
                    atomic_json(path,dict(existing,records=records,complete=False))
                    atomic_json(base/f'rank{rank}_progress.json',dict(samples=si,total_samples=len(selected),success=success,reused=reused,
                        elapsed_seconds=time.monotonic()-started,job_id=job,stage=stage,contract_sha256=c['contract_sha256']))
                if record['replay_status']=='ERROR':raise RuntimeError('Route replay failed; error preserved')
            atomic_json(path,dict(existing,records=records,complete=len(records)==item['routes']))
        atomic_json(base/f'rank{rank}_progress.json',dict(samples=si+1,total_samples=len(selected),success=success,reused=reused,
            elapsed_seconds=time.monotonic()-started,job_id=job,stage=stage,contract_sha256=c['contract_sha256']))
        del inputs,prepared
        if (si+1)%10==0:print(stage,rank,si+1,len(selected),success,'new',reused,'reused',flush=True)
    atomic_json(base/f'rank{rank}_complete.json',dict(passed=True,samples=len(selected),success=success,reused=reused,
        elapsed_seconds=time.monotonic()-started,contract_sha256=c['contract_sha256'],job_id=job))


if __name__=='__main__':
    try:main()
    except Exception:
        if os.environ.get('SLURM_JOB_ID'):
            atomic_json(OUT/'replay_gate'/f"job_{os.environ['SLURM_JOB_ID']}_rank{os.environ.get('LOCAL_RANK','unknown')}_failure.json",
                        dict(traceback=traceback.format_exc()))
        raise
