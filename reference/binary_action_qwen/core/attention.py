"""Static DVR attention paths for Qwen2.5-VL language layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dvr_qwen.core.cache import DVRCache

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv, rotate_half


@dataclass
class DVRAttentionStats:
    key_value_length: int
    visual_kv_projected: bool


def qwen_mrope_reorder(values: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Match Qwen2.5-VL channel reordering for multimodal RoPE sections."""
    sections = mrope_section * 2
    return torch.cat([chunk[i % 3] for i, chunk in enumerate(values.split(sections, dim=-1))], dim=-1)


def apply_mrope_one(
    states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.to(device=states.device, dtype=states.dtype)
    sin = sin.to(device=states.device, dtype=states.dtype)
    cos = qwen_mrope_reorder(cos, mrope_section).unsqueeze(unsqueeze_dim)
    sin = qwen_mrope_reorder(sin, mrope_section).unsqueeze(unsqueeze_dim)
    return (states * cos) + (rotate_half(states) * sin)


def _project_qkv(attn: nn.Module, hidden_states: torch.Tensor):
    batch_size, seq_len, _ = hidden_states.shape
    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)

    query_states = query_states.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    return query_states, key_states, value_states


def _project_kv(attn: nn.Module, hidden_states: torch.Tensor):
    batch_size, seq_len, _ = hidden_states.shape
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)
    key_states = key_states.view(batch_size, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    return key_states, value_states


def _eager_attention(
    attn: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    key_states = repeat_kv(key_states, attn.num_key_value_groups)
    value_states = repeat_kv(value_states, attn.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attn.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    batch_size, query_len = query_states.shape[0], query_states.shape[2]
    return attn.o_proj(attn_output.reshape(batch_size, query_len, -1))


def forward_text_only(
    attn: nn.Module,
    text_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings_text: tuple[torch.Tensor, torch.Tensor],
    cache: DVRCache | None = None,
    layer_idx: int | None = None,
    use_cache: bool = False,
) -> tuple[torch.Tensor, DVRAttentionStats]:
    query_states, key_states, value_states = _project_qkv(attn, text_hidden)
    cos_text, sin_text = position_embeddings_text
    mrope_section = attn.config.rope_parameters["mrope_section"]
    query_states = apply_mrope_one(query_states, cos_text, sin_text, mrope_section)
    key_states = apply_mrope_one(key_states, cos_text, sin_text, mrope_section)
    if use_cache:
        if cache is None or layer_idx is None:
            raise ValueError("cache and layer_idx are required when use_cache=True")
        key_states, value_states = cache.update(key_states, value_states, layer_idx)
    output = _eager_attention(attn, query_states, key_states, value_states, attention_mask)
    return output, DVRAttentionStats(key_value_length=key_states.shape[-2], visual_kv_projected=False)


def forward_visual_read(
    attn: nn.Module,
    text_hidden: torch.Tensor,
    visual_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings_text: tuple[torch.Tensor, torch.Tensor],
    position_embeddings_visual: tuple[torch.Tensor, torch.Tensor],
    cache: DVRCache | None = None,
    layer_idx: int | None = None,
    use_cache: bool = False,
    visual_attention_bias: float = 0.0,
    visual_value_scale: float = 1.0,
) -> tuple[torch.Tensor, DVRAttentionStats]:
    query_states, key_text, value_text = _project_qkv(attn, text_hidden)
    key_visual, value_visual = _project_kv(attn, visual_hidden)

    cos_text, sin_text = position_embeddings_text
    cos_visual, sin_visual = position_embeddings_visual
    mrope_section = attn.config.rope_parameters["mrope_section"]
    query_states = apply_mrope_one(query_states, cos_text, sin_text, mrope_section)
    key_text = apply_mrope_one(key_text, cos_text, sin_text, mrope_section)
    key_visual = apply_mrope_one(key_visual, cos_visual, sin_visual, mrope_section)
    if visual_value_scale != 1.0:
        value_visual = value_visual * visual_value_scale

    key_states = torch.cat([key_visual, key_text], dim=-2)
    value_states = torch.cat([value_visual, value_text], dim=-2)
    if visual_attention_bias != 0.0:
        attention_mask = attention_mask.clone()
        attention_mask[..., : key_visual.shape[-2]] = attention_mask[..., : key_visual.shape[-2]] + visual_attention_bias
    if use_cache:
        if cache is None or layer_idx is None:
            raise ValueError("cache and layer_idx are required when use_cache=True")
        key_states, value_states = cache.update(
            key_states,
            value_states,
            layer_idx,
            has_visual=True,
            num_visual_tokens=key_visual.shape[-2],
        )
    output = _eager_attention(attn, query_states, key_states, value_states, attention_mask)
    return output, DVRAttentionStats(key_value_length=key_states.shape[-2], visual_kv_projected=True)


def forward_decode_text(
    attn: nn.Module,
    text_hidden: torch.Tensor,
    position_embeddings_text: tuple[torch.Tensor, torch.Tensor],
    cache: DVRCache,
    layer_idx: int,
    visual_attention_bias: float = 0.0,
) -> tuple[torch.Tensor, DVRAttentionStats]:
    query_states, key_states, value_states = _project_qkv(attn, text_hidden)
    cos_text, sin_text = position_embeddings_text
    mrope_section = attn.config.rope_parameters["mrope_section"]
    query_states = apply_mrope_one(query_states, cos_text, sin_text, mrope_section)
    key_states = apply_mrope_one(key_states, cos_text, sin_text, mrope_section)
    key_states, value_states = cache.update(key_states, value_states, layer_idx)
    attention_mask = None
    if visual_attention_bias != 0.0 and cache.has_visual[layer_idx]:
        attention_mask = torch.zeros(
            query_states.shape[0],
            1,
            query_states.shape[-2],
            key_states.shape[-2],
            dtype=query_states.dtype,
            device=query_states.device,
        )
        attention_mask[..., : cache.num_visual_tokens[layer_idx]] = visual_attention_bias
    output = _eager_attention(attn, query_states, key_states, value_states, attention_mask=attention_mask)
    return output, DVRAttentionStats(key_value_length=key_states.shape[-2], visual_kv_projected=False)
