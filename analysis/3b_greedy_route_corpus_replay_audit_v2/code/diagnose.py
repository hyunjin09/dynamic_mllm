"""One-example, unmodified-executor parity trace. Never starts corpus replay."""
import argparse
import hashlib
import importlib.metadata as md
import os
import socket
import sys
import traceback

from common import *
from runtime import runtime_imports


def snapshot(t):
    return None if t is None else t.detach().cpu().clone()


def tensor_summary(t):
    if t is None:
        return None
    import torch
    t=t.detach().cpu().contiguous()
    return dict(shape=list(t.shape),dtype=str(t.dtype),
                sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                norm=float(t.double().norm()),min=float(t.min()),max=float(t.max()))


def difference(a,b):
    import torch
    if a is None or b is None:
        return dict(equal=a is None and b is None,comparable=a is None and b is None)
    if a.shape!=b.shape:
        return dict(equal=False,comparable=False,shapes=[list(a.shape),list(b.shape)])
    d=(a.double()-b.double()).abs()
    return dict(equal=bool(torch.equal(a,b)),comparable=True,max_abs=float(d.max()),
                mean_abs=float(d.mean()),different_elements=int((a!=b).sum()))


def normalized_kv(t,heads):
    if t.shape[1]==heads:
        return t
    assert heads%t.shape[1]==0
    return t.repeat_interleave(heads//t.shape[1],dim=1)


def cache_tensors(cache,index):
    if cache is None:
        return None,None
    if hasattr(cache,'key_cache'):
        return cache.key_cache[index],cache.value_cache[index]
    layer=cache.layers[index]
    return layer.keys,layer.values


class Trace:
    """Observe the first two language-model calls without changing outputs."""
    def __init__(self,torch,text_model):
        self.torch=torch
        self.text_model=text_model
        self.handles=[]
        self.records={}
        self.calls=Counter()
        self.active=None
        self.positions=None

    def __enter__(self):
        def rotary_pre(module,args,kwargs):
            self.positions=snapshot(kwargs.get('position_ids',args[1] if len(args)>1 else None))
        self.handles.append(self.text_model.rotary_emb.register_forward_pre_hook(rotary_pre,with_kwargs=True))
        for index,layer in enumerate(self.text_model.layers):
            def before(module,args,kwargs,index=index):
                step=self.calls[index]; self.calls[index]+=1
                self.active=(step,index) if step<2 else None
                if self.active is None:return
                hidden=kwargs.get('hidden_states',args[0] if args else None)
                cos,sin=kwargs['position_embeddings']
                self.records[self.active]=dict(hidden_in=snapshot(hidden),
                    attention_mask=snapshot(kwargs.get('attention_mask')),
                    position_ids=snapshot(self.positions),
                    layer_position_ids=snapshot(kwargs.get('position_ids')),
                    cache_position=snapshot(kwargs.get('cache_position')),
                    rope_cos=snapshot(cos),rope_sin=snapshot(sin))
            def after(module,args,kwargs,output,index=index):
                if self.active is None:return
                record=self.records[self.active]
                record['hidden_out']=snapshot(output[0] if isinstance(output,tuple) else output)
                cache=kwargs.get('past_key_values',kwargs.get('past_key_value'))
                k,v=cache_tensors(cache,index)
                record.update(cache_key=snapshot(k),cache_value=snapshot(v))
                self.active=None
            self.handles.append(layer.register_forward_pre_hook(before,with_kwargs=True))
            self.handles.append(layer.register_forward_hook(after,with_kwargs=True))
        self.original_sdpa=self.torch.nn.functional.scaled_dot_product_attention
        def sdpa(query,key,value,*args,**kwargs):
            output=self.original_sdpa(query,key,value,*args,**kwargs)
            if self.active is not None:
                record=self.records[self.active]
                mask=kwargs.get('attn_mask',args[0] if args else None)
                record.update(query=snapshot(query),key=snapshot(key),value=snapshot(value),
                    sdpa_out=snapshot(output),sdpa_mask=snapshot(mask),
                    sdpa_flags=dict(is_causal=kwargs.get('is_causal',args[2] if len(args)>2 else False),
                                    enable_gqa=kwargs.get('enable_gqa',False),scale=kwargs.get('scale'),
                                    dropout_p=kwargs.get('dropout_p',args[1] if len(args)>1 else 0.0)))
            return output
        self.torch.nn.functional.scaled_dot_product_attention=sdpa
        return self

    def __exit__(self,*exc):
        self.torch.nn.functional.scaled_dot_product_attention=self.original_sdpa
        for handle in self.handles:handle.remove()


def trace_comparison(native,custom):
    result=[]
    first=None
    # Masks/cache container interfaces can differ without numerical differences.
    order=['hidden_in','rope_cos','rope_sin','query','key','value','sdpa_out','hidden_out','cache_key','cache_value']
    for step,index in sorted(set(native)&set(custom)):
        a,b=native[(step,index)],custom[(step,index)]
        diffs={}
        for name in order+['position_ids','layer_position_ids','cache_position','attention_mask','sdpa_mask']:
            x,y=a.get(name),b.get(name)
            if name in ['key','value'] and x is not None and y is not None:
                heads=a['query'].shape[1]
                x,y=normalized_kv(x,heads),normalized_kv(y,heads)
            diffs[name]=difference(x,y)
            if first is None and name in order and not diffs[name]['equal']:
                first=dict(step=step,stage='prefill' if step==0 else 'decode_after_first_token',layer=index,field=name,**diffs[name])
        result.append(dict(step=step,layer=index,differences=diffs,
            native={k:tensor_summary(v) for k,v in a.items() if k!='sdpa_flags'},
            custom={k:tensor_summary(v) for k,v in b.items() if k!='sdpa_flags'},
            native_sdpa_flags=a.get('sdpa_flags'),custom_sdpa_flags=b.get('sdpa_flags')))
    return result,first


def logits_record(torch,logits,processed):
    values,ids=processed.float().topk(10,dim=-1)
    return dict(raw=tensor_summary(logits),processed=tensor_summary(processed),
                top10_ids=ids.view(-1).tolist(),top10_scores=values.view(-1).tolist(),
                selected=int(processed.argmax(dim=-1).item()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--imports-only',action='store_true')
    args=parser.parse_args()
    if args.imports_only:
        runtime_imports();print('Imports passed; no GPU execution.');return
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required'
    c=read_contract()
    assert c['stage']=='FOCUSED_PARITY_DIAGNOSTIC_ONLY'
    for name,expected in c['inherited_artifact_hashes'].items():
        assert file_hash(OUT/name)==expected,name
    torch,Processor,Model,generate,prepare,score,build_inputs=runtime_imports()
    assert torch.cuda.device_count()==1, 'Focused diagnostic requires exactly one visible GPU'
    torch.cuda.set_device(0);torch.set_num_threads(4)
    for package,version in c['installed_versions'].items():assert md.version(package)==version,package
    from dvr_qwen.modeling_dvr_qwen2_5_vl import qwen_text_model
    from dvr_qwen.binary_generate import binary_dvrc_prefill,logits_from_binary_prefill,binary_dvrc_decode_one_step
    from dvr_qwen.generate import apply_repetition_penalty,generation_token_history,next_decode_position_ids
    job=os.environ['SLURM_JOB_ID']; target=OUT/'parity'/f'job_{job}'
    assert not (target/'diagnostic_result.json').exists(), 'Preserve completed diagnostic'
    env=dict(job_id=job,hostname=socket.gethostname(),torch=torch.__version__,cuda=torch.version.cuda,
             transformers=md.version('transformers'),python=sys.version,gpu=torch.cuda.get_device_name(0),
             contract_sha256=c['contract_sha256'],model_revision=REVISION)
    atomic_json(target/'environment.json',env)
    atomic_json(OUT/'replay_gate/current_server_env.json',env)
    sample=next(r for r in rows(OUT/'source_inventory/relocated_samples.jsonl') if r['uid']==c['diagnostic_uid'])
    assert file_hash(sample['local_image_path'])==sample['image_content_sha256']
    processor=Processor.from_pretrained(str(MODEL),local_files_only=True,use_fast=False)
    model=Model.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
    inputs={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in build_inputs(processor,sample,data_root=PACKAGE/'01_SOURCE_DATA/vqa_train_10k').items()}
    native_inputs={k:v for k,v in inputs.items() if k!='instruction_token_mask'}
    generation=dict(do_sample=False,num_beams=1,repetition_penalty=1.05,eos_token_id=151645,return_dict_in_generate=True)
    mask=torch.ones(1,LAYERS,dtype=torch.bool,device='cuda:0')
    with torch.inference_mode():
        # Uninstrumented reference ensures tracing is checked for behavioral changes.
        hf=model.base_model.generate(**native_inputs,max_new_tokens=int(sample['max_new_tokens']),**generation)
        hf_ids=hf.sequences[:,inputs['input_ids'].shape[1]:].view(-1).tolist()
        prepared=prepare(model,inputs)
        baseline=generate(model,inputs,visual_on_mask=mask,max_new_tokens=int(sample['max_new_tokens']),
                          eos_token_ids=[151645],repetition_penalty=1.05,prepared_binary_inputs=prepared)
        custom_ids=baseline.generated_ids.view(-1).tolist()
        del hf,baseline
        atomic_json(target/'baseline.json',dict(uid=sample['uid'],hf_ids=hf_ids,custom_ids=custom_ids))
        with Trace(torch,qwen_text_model(model)) as native_trace:
            hf=model.base_model.generate(**native_inputs,max_new_tokens=2,output_logits=True,output_scores=True,**generation)
        native_ids=hf.sequences[:,inputs['input_ids'].shape[1]:].view(-1).tolist()
        native_logits=[snapshot(x) for x in hf.logits];native_scores=[snapshot(x) for x in hf.scores]
        del hf
        with Trace(torch,qwen_text_model(model)) as custom_trace:
            state=binary_dvrc_prefill(model,inputs,visual_on_mask=mask,prepared_binary_inputs=prepared)
            first_logits=logits_from_binary_prefill(model,state)
            first_scores=apply_repetition_penalty(first_logits,inputs['input_ids'],1.05)
            first=first_scores.argmax(dim=-1)
            second_logits,_=binary_dvrc_decode_one_step(model,first,state,generated_step=0)
            history=generation_token_history(inputs['input_ids'],[first[:,None]],device=first.device)
            second_scores=apply_repetition_penalty(second_logits,history,1.05)
            trace_ids=[int(first.item()),int(second_scores.argmax(dim=-1).item())]
        custom_logits=[snapshot(first_logits),snapshot(second_logits)]
        custom_scores=[snapshot(first_scores),snapshot(second_scores)]
        geometry=dict(full_position_ids=tensor_summary(prepared.full_position_ids),
            visual_positions=prepared.visual_indices.detach().cpu().tolist(),
            text_positions=prepared.text_indices.detach().cpu().tolist(),
            input_attention_mask=tensor_summary(inputs['attention_mask']),
            rope_deltas=prepared.rope_deltas.detach().cpu().tolist(),
            custom_decode_position_ids=next_decode_position_ids(prepared.dvr_inputs,0).detach().cpu().tolist(),
            physical_prompt_length=int(inputs['input_ids'].shape[1]))
    assert len(native_trace.records)==len(custom_trace.records)==2*LAYERS, 'Incomplete layer trace'
    comparisons,first_divergence=trace_comparison(native_trace.records,custom_trace.records)
    write_jsonl(target/'layer_comparisons.jsonl',comparisons)
    logits=[dict(step=i,native=logits_record(torch,native_logits[i],native_scores[i]),
                 custom=logits_record(torch,custom_logits[i],custom_scores[i]),
                 raw_difference=difference(native_logits[i],custom_logits[i]),
                 processed_difference=difference(native_scores[i],custom_scores[i])) for i in range(2)]
    atomic_json(target/'logits.json',logits)
    atomic_json(target/'input_geometry.json',geometry)
    # A local counterfactual on identical Q/K/V tests SDPA calling conventions,
    # without changing the model, rerunning the corpus, or applying a repair.
    counterfactual=None
    if first_divergence and first_divergence['field']=='sdpa_out':
        key=(first_divergence['step'],first_divergence['layer'])
        a,b=native_trace.records[key],custom_trace.records[key]
        same_qkv=all(difference(a['query'] if name=='query' else normalized_kv(a[name],a['query'].shape[1]),
                                b['query'] if name=='query' else normalized_kv(b[name],b['query'].shape[1]))['equal']
                     for name in ['query','key','value'])
        if same_qkv:
            with torch.inference_mode():
                # Use native head layout/flags with the custom values after exact
                # canonical Q/K/V equality has been established above.
                q=b['query'].to('cuda:0');k=b['key'];v=b['value']
                if a['key'].shape[1]!=k.shape[1]:
                    heads=a['key'].shape[1]
                    if k.shape[1]>heads:
                        group=k.shape[1]//heads
                        assert torch.equal(k,k[:,::group].repeat_interleave(group,1))
                        assert torch.equal(v,v[:,::group].repeat_interleave(group,1))
                        k,v=k[:,::group],v[:,::group]
                    else:k,v=normalized_kv(k,heads),normalized_kv(v,heads)
                native_mask=a['sdpa_mask']
                recovered=torch.nn.functional.scaled_dot_product_attention(q,k.to('cuda:0'),v.to('cuda:0'),
                    attn_mask=None if native_mask is None else native_mask.to('cuda:0'),**a['sdpa_flags']).cpu()
            counterfactual=dict(same_qkv=True,layer=key[1],step=key[0],
                native_convention_on_custom_qkv_vs_native=difference(recovered,a['sdpa_out']),
                native_convention_on_custom_qkv_vs_custom=difference(recovered,b['sdpa_out']),
                scope='isolated SDPA call, not an executor repair or full Gate-A pass')
    reproduced=hf_ids==[17,17,151645] and custom_ids==[17,18,151645]
    trace_matches=native_ids==hf_ids[:2] and trace_ids==custom_ids[:2]
    result=dict(job_id=job,uid=sample['uid'],contract_sha256=c['contract_sha256'],
        baseline_hf_ids=hf_ids,baseline_custom_ids=custom_ids,known_failure_reproduced=reproduced,
        trace_matches_uninstrumented=trace_matches,trace_hf_ids=native_ids,trace_custom_ids=trace_ids,
        first_observed_numerical_divergence=first_divergence,sdpa_counterfactual=counterfactual,
        diagnosis='unknown_pending_review',full_replay_started=False,repair_applied=False,
        limitation='First observed divergence is among recorded layer/QKV/SDPA/hidden/cache boundaries; shared token does not prove equal prefill logits.')
    atomic_json(target/'diagnostic_result.json',result)
    report=(f"Focused parity diagnostic for {sample['uid']}, Slurm job {job}.\n\n"
            f"Known failure reproduced: {reproduced}. Instrumented prefixes match baseline: {trace_matches}.\n\n"
            f"First observed numerical divergence:\n```json\n{json.dumps(first_divergence,indent=2)}\n```\n\n"
            f"Isolated SDPA convention check:\n```json\n{json.dumps(counterfactual,indent=2)}\n```\n\n"
            'Layer checksums/norms, Q/K/V/cache comparisons, position/RoPE/mask comparisons,\n'
            'and both-step top10 logits are in the job directory. No raw hidden tensor\n'
            'corpus was saved. Cache-position arguments absent from a layer call are\n'
            'recorded as null; physical KV lengths and actual rotary positions are recorded.\n'
            'Diagnosis requires review; no executor repair or full replay was performed.\n')
    write_text(target/'diagnostic.md',report)
    write_text(OUT/'parity/current_executor_parity_diagnostic.md',report)
    print(json.dumps(result,indent=2),flush=True)
    assert trace_matches, 'Instrumentation did not reproduce baseline; diagnostic is inconclusive'


if __name__=='__main__':
    try:main()
    except Exception:
        if os.environ.get('SLURM_JOB_ID'):
            atomic_json(OUT/'parity'/f"job_{os.environ['SLURM_JOB_ID']}"/'runtime_failure.json',dict(traceback=traceback.format_exc()))
        raise
