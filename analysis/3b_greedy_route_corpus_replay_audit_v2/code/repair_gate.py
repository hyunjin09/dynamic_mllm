"""Verify the supported FULL-mask repair on the complete fixed anchor set."""
import os
import importlib.metadata as md
import sys
import traceback
from common import *
from runtime import runtime_imports
from parity_repair import install
from diagnose import Trace, trace_comparison


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'GPU computation requires Slurm'
    parent=read_contract()
    contract=read_json(OUT/'contracts/parity_repair_v1.json')
    assert digest({k:v for k,v in contract.items() if k!='contract_sha256'})==contract['contract_sha256']
    assert contract['parent_contract_sha256']==parent['contract_sha256']
    for path,expected in contract['code_hashes'].items():assert file_hash(path)==expected,path
    for path,expected in contract['evidence_hashes'].items():assert file_hash(path)==expected,path
    for name in ['anchor_sha256','sparse_probe_sha256','relocated_samples_sha256']:
        assert contract[name]==parent[name]
    assert file_hash(OUT/'replay_gate/frozen_anchor_rows.jsonl')==contract['anchor_sha256']
    assert file_hash(OUT/'replay_gate/frozen_sparse_probes.jsonl')==contract['sparse_probe_sha256']
    assert file_hash(OUT/'source_inventory/relocated_samples.jsonl')==contract['relocated_samples_sha256']
    torch,Processor,Model,generate,prepare,score,build_inputs=runtime_imports()
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0);torch.set_num_threads(4)
    for name,version in parent['installed_versions'].items():assert md.version(name)==version,name
    from dvr_qwen.modeling_dvr_qwen2_5_vl import qwen_text_model
    job=os.environ['SLURM_JOB_ID'];target=OUT/'parity'/f'job_{job}'
    assert not (target/'gate_a_result.json').exists(), 'Preserve completed gate'
    processor=Processor.from_pretrained(str(MODEL),local_files_only=True,use_fast=False)
    model=Model.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},
                              attn_implementation='sdpa',local_files_only=True).eval()
    install(model)
    samples={r['uid']:r for r in rows(OUT/'source_inventory/relocated_samples.jsonl')}
    anchors=list(rows(OUT/'replay_gate/frozen_anchor_rows.jsonl'))
    assert len(anchors)==len({r['uid'] for r in anchors})==32
    def inputs_for(sample):
        assert file_hash(sample['local_image_path'])==sample['image_content_sha256']
        return {k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in build_inputs(
            processor,sample,data_root=PACKAGE/'01_SOURCE_DATA/vqa_train_10k').items()}
    def label(ids,sample):
        answer=processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
        result=float(score(sample['metric_name'],answer,sample['answer'],sample.get('all_answer_norms')))
        return dict(ids=ids,answer=answer,score=result,correctness=result>=float(sample['correctness_threshold']))
    def binary(inputs,sample,mask,prepared):
        output=generate(model,inputs,visual_on_mask=torch.tensor([[int(x) for x in mask]],dtype=torch.bool,device='cuda:0'),
            max_new_tokens=int(sample['max_new_tokens']),eos_token_ids=[151645],repetition_penalty=1.05,prepared_binary_inputs=prepared)
        actual=[int(x) for x in output.state.route_binary.view(-1).tolist()]
        assert actual==[int(x) for x in mask]
        return dict(**label(output.generated_ids.view(-1).tolist(),sample),action_trace=actual)
    records=[];repair_trace=None
    with torch.inference_mode():
        for anchor in anchors:
            sample=samples[anchor['uid']];result=dict(uid=sample['uid'],passed=False)
            try:
                inputs=inputs_for(sample)
                traced=sample['uid']==parent['diagnostic_uid']
                if traced:
                    with Trace(torch,qwen_text_model(model)) as native_trace:
                        native=model.base_model.generate(**{k:v for k,v in inputs.items() if k!='instruction_token_mask'},
                            max_new_tokens=int(sample['max_new_tokens']),do_sample=False,num_beams=1,
                            repetition_penalty=1.05,eos_token_id=151645,return_dict_in_generate=True)
                else:
                    native=model.base_model.generate(**{k:v for k,v in inputs.items() if k!='instruction_token_mask'},
                        max_new_tokens=int(sample['max_new_tokens']),do_sample=False,num_beams=1,
                        repetition_penalty=1.05,eos_token_id=151645,return_dict_in_generate=True)
                hf=label(native.sequences[:,inputs['input_ids'].shape[1]:].view(-1).tolist(),sample)
                prepared=prepare(model,inputs)
                if traced:
                    with Trace(torch,qwen_text_model(model)) as custom_trace:
                        custom=binary(inputs,sample,'1'*LAYERS,prepared)
                    comparisons,first=trace_comparison(native_trace.records,custom_trace.records)
                    write_jsonl(target/'repaired_failing_anchor_trace.jsonl',comparisons)
                    repair_trace=dict(first_observed_numerical_divergence=first,
                        compared_layers=len(comparisons),uid=sample['uid'])
                    del native_trace,custom_trace,comparisons
                else:custom=binary(inputs,sample,'1'*LAYERS,prepared)
                result.update(hf=hf,custom=custom,passed=all(hf[k]==custom[k] for k in ['ids','answer','score','correctness']))
                del inputs,prepared,native
            except Exception:
                result['error']=traceback.format_exc()
            records.append(result)
            write_jsonl(target/'anchors.jsonl',records)
            print('anchor',sample['uid'],result['passed'],flush=True)
        sparse=[]
        for probe in rows(OUT/'replay_gate/frozen_sparse_probes.jsonl'):
            sample=samples[probe['uid']];result=dict(uid=sample['uid'],route_key=probe['route_key'],passed=False)
            try:
                inputs=inputs_for(sample);prepared=prepare(model,inputs)
                first=binary(inputs,sample,probe['source_record']['mask_key'],prepared)
                second=binary(inputs,sample,probe['source_record']['mask_key'],prepared)
                result.update(first=first,second=second,passed=first==second)
                del inputs,prepared
            except Exception:result['error']=traceback.format_exc()
            sparse.append(result)
    atomic_json(target/'sparse_repeat.json',sparse)
    passed=len(records)==32 and all(r['passed'] for r in records)
    outcome=dict(job_id=job,passed=passed,exact_anchor_matches=sum(r['passed'] for r in records),anchors=32,
        contract_sha256=contract['contract_sha256'],repair='native SDPA unpadded FULL causal-mask convention',
        repaired_failing_anchor_trace=repair_trace,sparse_repeat_passed=all(r['passed'] for r in sparse),
        cross_worker_repeatability='PENDING_EIGHT_GPU_PREFLIGHT',full_replay_started=False)
    atomic_json(target/'gate_a_result.json',outcome)
    atomic_json(OUT/'parity/repaired_gate_a_result.json',outcome)
    write_csv(OUT/'parity/current_hf_vs_custom_full.csv',[
        dict(sample_uid=r['uid'],slurm_job_id=job,passed=r['passed'],hf_answer=r.get('hf',{}).get('answer'),
             custom_answer=r.get('custom',{}).get('answer'),hf_ids=json.dumps(r.get('hf',{}).get('ids')),
             custom_ids=json.dumps(r.get('custom',{}).get('ids')),error=r.get('error')) for r in records])
    write_text(OUT/'parity/gate_a_report.md',f"Repaired Gate A: {'PASS' if passed else 'FAIL'}, job{job}.\n\n"
        f"Exact complete generated sequences/answers/scores/correctness: {outcome['exact_anchor_matches']}/32.\n"
        f"Four sparse probes repeat: {outcome['sparse_repeat_passed']}.\n"
        'Eight-worker preflight and full Dense/routed replay have not run. Canonical package unchanged.\n')
    print(json.dumps(outcome,indent=2),flush=True)
    assert passed and outcome['sparse_repeat_passed'], 'Repair validation failed; do not start corpus replay'


if __name__=='__main__':
    main()
