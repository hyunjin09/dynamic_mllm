"""Bounded Dense-only validity localization; no schedules or calibration."""
import json
import torch
from experiments.run_benchmark_fixed_schedule import Runtime, check, OUT, read_jsonl, reference, atomic_json

@torch.inference_mode()
def main():
    c=check();rt=Runtime(c['config'])
    row=next(r for r in read_jsonl(OUT/'splits/split_registry.jsonl') if r['uid']=='mmmu_pro_standard_test:mmmu_pro_standard_test_test_Psychology_128')
    mode='native';saved={};diffs=[];calls={};handles=[];pre=[]
    def hook(name):
        def capture(module,args,kwargs,output):
            nonlocal mode
            count=calls.get((mode,name),0);calls[mode,name]=count+1
            value=output[0] if isinstance(output,tuple) else output
            if not torch.is_tensor(value):return
            value=value.detach().cpu()
            key=(name,count)
            if mode=='native':saved[key]=value
            elif key in saved:
                old=saved[key]
                if old.shape==value.shape:
                    d=(old.float()-value.float()).abs()
                    diffs.append(dict(name=name,call=count,shape=list(value.shape),exact=torch.equal(old,value),max_abs=float(d.max()),last_max_abs=float(d[:,-1].max()) if d.ndim>=3 else None))
                else:diffs.append(dict(name=name,call=count,native_shape=list(old.shape),unified_shape=list(value.shape)))
        return capture
    def before(module,args,kwargs):
        def info(x):
            if x is None:return None
            if torch.is_tensor(x):return dict(shape=list(x.shape),values=x.detach().cpu().flatten()[:12].tolist(),min=float(x.min()),max=float(x.max()))
            return str(type(x))
        cache=kwargs.get('past_key_values')
        pre.append(dict(mode=mode,hidden=info(kwargs.get('hidden_states',args[0] if args else None)),mask=info(kwargs.get('attention_mask')),positions=info(kwargs.get('position_ids')),cache_position=info(kwargs.get('cache_position')),rope=[info(x) for x in kwargs['position_embeddings']],cache_len=cache.get_seq_length(0) if cache is not None else None))
    def model_before(module,args,kwargs):
        p=kwargs.get('position_ids')
        if p is not None:pre.append(dict(mode=mode,language_model_position_last=p[:,:,-1].tolist(),position_shape=list(p.shape)))
    handles.append(rt.base.model.language_model.register_forward_pre_hook(model_before,with_kwargs=True))
    handles.append(rt.base.model.language_model.layers[0].register_forward_pre_hook(before,with_kwargs=True))
    for i,layer in enumerate(rt.base.model.language_model.layers):handles.append(layer.register_forward_hook(hook(f'layer{i}'),with_kwargs=True))
    handles.append(rt.base.lm_head.register_forward_hook(hook('lm_head'),with_kwargs=True))
    native=reference._native_dense_generation(rt,row,extract_features=False);native.pop('features',None)
    mode='unified'
    inputs,base=rt.baseline(row);unified=rt.measure(base,inputs,row,False)
    for h in handles:h.remove()
    last=base.inputs.full_position_ids[:,:,-1]
    assert torch.equal(last,last[:1].expand_as(last))
    old_delta=base.inputs.rope_deltas.clone()
    base.inputs.rope_deltas=last[0,:,None]+1-base.inputs.full_prompt_len[:,None]
    repaired=rt.measure(base,inputs,row,False)
    # Native cached first-step logits from actual generate forward hook.
    first=saved['lm_head',0][:,-1].float();current=base.prompt_logits.cpu().float()
    first_info=dict(max_abs=float((first-current).abs().max()),native_top=torch.topk(first[0],10).indices.tolist(),unified_top=torch.topk(current[0],10).indices.tolist(),native_values=torch.topk(first[0],10).values.tolist(),unified_values=torch.topk(current[0],10).values.tolist())
    result=dict(uid=row['uid'],contract_sha256=c['contract_sha256'],old_rope_delta=old_delta.tolist(),prefill_last_positions=last.tolist(),repaired={k:repaired[k] for k in ['generated_token_ids','generated_answer','score','correct']},native={k:native[k] for k in ['generated_token_ids','generated_answer','score','correct']},unified={k:unified[k] for k in ['generated_token_ids','generated_answer','score','correct']},prompt_logits=first_info,layer_differences=diffs,layer0_inputs=pre,meta_rope_delta=base.inputs.rope_deltas.tolist(),meta_prompt_length=base.inputs.full_prompt_len.tolist())
    atomic_json(OUT/'parity/dense_scorer_mismatch_diagnostic.json',result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
