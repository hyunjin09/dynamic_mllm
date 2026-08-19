from __future__ import annotations

import torch
from torch import nn

from binary_policy.executor.inputs import BinaryInputs, resolve_causal_lm, scatter_streams, split_streams
from binary_policy.executor.layers import visual_off_layer, visual_on_layer


class FakeRotary(nn.Module):
    def forward(self, hidden, position_ids):
        shape = (position_ids.shape[1], position_ids.shape[2], hidden.shape[-1])
        return torch.ones(shape, device=hidden.device), torch.zeros(shape, device=hidden.device)


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.last_attention_mask = "unset"
        self.last_cache_position = "unset"

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        past_key_value=None,
        use_cache=False,
        cache_position=None,
    ):
        self.last_attention_mask = attention_mask
        self.last_cache_position = cache_position
        del position_embeddings, use_cache
        if past_key_value is not None:
            batch, rows, _ = hidden_states.shape
            kv = hidden_states.new_zeros(batch, 1, rows, 1)
            past_key_value.update(kv, kv, 0)
        return (hidden_states + 1.0,)


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotary_emb = FakeRotary()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeDecoder()


def fixture_inputs():
    full = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    text_indices = torch.tensor([[0, 2]])
    visual_indices = torch.tensor([[1, 3]])
    text = full[:, [0, 2]]
    visual = full[:, [1, 3]]
    positions = torch.arange(4).view(1, 1, 4).expand(3, 1, 4)
    return BinaryInputs(
        full_inputs_embeds=full,
        text_states=text,
        visual_states=visual,
        text_indices=text_indices,
        visual_indices=visual_indices,
        text_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        visual_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        full_position_ids=positions,
        text_position_ids=positions[:, :, [0, 2]],
        visual_position_ids=positions[:, :, [1, 3]],
        full_attention_mask=torch.ones(1, 4, dtype=torch.long),
        full_prompt_len=torch.tensor([4]),
        rope_deltas=torch.tensor([[0]]),
    )


def test_split_scatter_identity():
    meta = fixture_inputs()
    reconstructed = scatter_streams(meta.text_states, meta.visual_states, meta)
    assert torch.equal(reconstructed, meta.full_inputs_embeds)
    text, visual = split_streams(reconstructed, meta)
    assert torch.equal(text, meta.text_states)
    assert torch.equal(visual, meta.visual_states)


def test_visual_off_is_compacted_text_oracle_and_visual_bypass():
    meta = fixture_inputs()
    model = FakeModel()
    layer = FakeLayer()
    text, visual, stats = visual_off_layer(
        model, layer, meta.text_states, meta.visual_states, meta, layer_index=0
    )
    assert torch.equal(text, meta.text_states + 1)
    assert torch.equal(visual, meta.visual_states)
    assert not stats.visual_on


def test_visual_on_runs_native_full_rows_then_splits():
    meta = fixture_inputs()
    model = FakeModel()
    layer = FakeLayer()
    text, visual, stats = visual_on_layer(
        model, layer, meta.text_states, meta.visual_states, meta, layer_index=0
    )
    assert torch.equal(text, meta.text_states + 1)
    assert torch.equal(visual, meta.visual_states + 1)
    assert stats.visual_on
    assert torch.is_tensor(layer.last_attention_mask)


def test_visual_on_native_all_on_uses_native_maskless_causal_dispatch():
    meta = fixture_inputs()
    model = FakeModel()
    layer = FakeLayer()
    text, visual, stats = visual_on_layer(
        model,
        layer,
        meta.text_states,
        meta.visual_states,
        meta,
        layer_index=0,
        native_causal=True,
    )
    assert torch.equal(text, meta.text_states + 1)
    assert torch.equal(visual, meta.visual_states + 1)
    assert stats.visual_on
    assert layer.last_attention_mask is None
    assert torch.equal(layer.last_cache_position, torch.arange(4))


def test_native_conditional_lm_is_not_unwrapped_via_hf_base_model_property():
    class Native:
        model = object()
        lm_head = object()
        base_model = object()

    native = Native()
    assert resolve_causal_lm(native) is native
