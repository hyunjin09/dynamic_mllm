"""Match native SDPA dispatch for unpadded FULL prefill; preserve sparse semantics."""


def native_full_mask(original):
    def mask(attention_mask, dtype, device):
        # The packaged FULL path is causal in original sequence order. With no
        # padding, HF represents it by attn_mask=None and is_causal=True. Its
        # SDPA integration then also chooses native GQA instead of expanded KV.
        # Keep the original explicit mask for every padded/nonstandard input.
        if attention_mask.ndim == 2 and bool((attention_mask == 1).all().item()):
            return None
        return original(attention_mask, dtype=dtype, device=device)
    return mask


def install(model):
    import dvr_qwen.binary_layer as layer
    from dvr_qwen.modeling_dvr_qwen2_5_vl import qwen_text_model
    text=qwen_text_model(model)
    assert text.config._attn_implementation == 'sdpa'
    assert not model.training
    assert all(block.self_attn.is_causal for block in text.layers)
    original=layer.make_full_causal_mask
    assert not getattr(original,'_v2_native_full_mask',False), 'Repair already installed'
    replacement=native_full_mask(original)
    replacement._v2_native_full_mask=True
    layer.make_full_causal_mask=replacement
    return original
