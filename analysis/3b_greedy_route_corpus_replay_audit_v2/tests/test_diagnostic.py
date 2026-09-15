import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import pytest
import torch
from diagnose import Trace, difference, normalized_kv, tensor_summary, trace_comparison


def test_kv_comparison_handles_gqa_layout_without_hiding_value_drift():
    grouped=torch.arange(12,dtype=torch.float32).reshape(1,2,3,2)
    repeated=grouped.repeat_interleave(4,dim=1)
    assert difference(normalized_kv(grouped,8),repeated)['equal']
    repeated[0,0,1,1]+=1
    assert not difference(normalized_kv(grouped,8),repeated)['equal']


def test_first_numerical_divergence_excludes_equivalent_mask_representation():
    base={k:torch.ones(1,2,3,2) for k in ['hidden_in','rope_cos','rope_sin','query','key','value','sdpa_out','hidden_out','cache_key','cache_value']}
    a=dict(base,attention_mask=None,sdpa_flags={'is_causal':True})
    b=dict(base,attention_mask=torch.zeros(1,1,3,3),sdpa_flags={'is_causal':False})
    b['sdpa_out']=base['sdpa_out']+0.01
    rows,first=trace_comparison({(0,0):a},{(0,0):b})
    assert first['field']=='sdpa_out' and first['step']==0
    assert not rows[0]['differences']['attention_mask']['equal']


def test_trace_preserves_outputs_captures_cache_and_restores_sdpa_on_exception():
    class Rotary(torch.nn.Module):
        def forward(self,x,position_ids):return torch.ones_like(x),torch.zeros_like(x)
    class Cache:
        key_cache=[None]
        value_cache=[None]
    class Layer(torch.nn.Module):
        def forward(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None):
            q=hidden_states.unsqueeze(1)
            past_key_values.key_cache[0]=q
            past_key_values.value_cache[0]=q
            return (torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True).squeeze(1),)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.rotary_emb=Rotary();self.layers=torch.nn.ModuleList([Layer()])
        def forward(self,x):
            pos=self.rotary_emb(x,torch.arange(x.shape[1]).reshape(1,1,-1))
            return self.layers[0](hidden_states=x,position_embeddings=pos,past_key_values=Cache())[0]
    model=Model();x=torch.randn(1,3,4);expected=model(x)
    original=torch.nn.functional.scaled_dot_product_attention
    with pytest.raises(RuntimeError):
        with Trace(torch,model) as trace:
            assert torch.equal(model(x),expected)
            assert torch.equal(trace.records[(0,0)]['cache_key'],x.unsqueeze(1))
            assert trace.records[(0,0)]['position_ids'].shape==(1,1,3)
            raise RuntimeError('test cleanup')
    assert torch.nn.functional.scaled_dot_product_attention is original
    assert len(model.layers[0]._forward_hooks)==0


def test_bf16_checksum_distinguishes_values_and_scalar_supported():
    a=tensor_summary(torch.tensor([1,2],dtype=torch.bfloat16))
    b=tensor_summary(torch.tensor([1,3],dtype=torch.bfloat16))
    assert a['sha256']!=b['sha256']
    assert tensor_summary(torch.tensor(1.0))['norm']==1
