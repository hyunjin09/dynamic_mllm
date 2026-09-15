import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import torch
from parity_repair import native_full_mask


def test_unpadded_full_uses_native_causal_convention():
    def original(*args,**kwargs):raise AssertionError('Explicit mask is unnecessary')
    for dtype in [torch.long,torch.bool]:
        assert native_full_mask(original)(torch.ones(1,5,dtype=dtype),torch.bfloat16,'cpu') is None


def test_padding_and_nonbinary_masks_preserve_original_behavior():
    sentinel=object();calls=[]
    def original(mask,**kwargs):calls.append((mask,kwargs));return sentinel
    for mask in [torch.tensor([[1,1,0]]),torch.tensor([[0,1,1]]),torch.tensor([[1,2,1]])]:
        assert native_full_mask(original)(mask,torch.float32,'cpu') is sentinel
        assert calls[-1][0] is mask


def test_mask_elision_preserves_full_causal_connectivity():
    # Attention to future rows is excluded in both representations.
    q=torch.zeros(1,1,4,2);v=torch.arange(8,dtype=torch.float64).reshape(1,1,4,2)
    q=q.double();k=q.clone()
    mask=torch.full((4,4),-float('inf'),dtype=torch.float64).triu(1)
    explicit=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    native=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
    assert torch.allclose(explicit,native,atol=1e-12,rtol=0)
