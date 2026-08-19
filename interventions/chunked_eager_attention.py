from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


def chunked_eager_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Qwen2.5-VL eager attention, evaluated in independent query chunks."""
    del position_ids, use_cache
    if position_embeddings is None:
        raise RuntimeError("Chunked eager attention requires external position embeddings")

    batch_size, query_length, _ = hidden_states.shape
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        batch_size, query_length, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
    )
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    key_length = key_states.shape[-2]
    key_positions = torch.arange(key_length, device=hidden_states.device)
    if cache_position is None:
        absolute_query_positions = torch.arange(query_length, device=hidden_states.device)
    else:
        absolute_query_positions = cache_position[-query_length:].to(hidden_states.device)

    output_chunks: list[torch.Tensor] = []
    attention_chunks: list[torch.Tensor] = []
    chunk_size = int(getattr(self, "stage_a_query_chunk_size", 32))
    for query_start in range(0, query_length, chunk_size):
        query_stop = min(query_start + chunk_size, query_length)
        weights = torch.matmul(
            query_states[:, :, query_start:query_stop, :],
            key_states.transpose(2, 3),
        ) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, query_start:query_stop, :key_length]
        else:
            causal = key_positions.unsqueeze(0) <= absolute_query_positions[
                query_start:query_stop
            ].unsqueeze(1)
            weights = weights.masked_fill(
                ~causal.unsqueeze(0).unsqueeze(0),
                torch.finfo(weights.dtype).min,
            )
        if query_states.dtype == torch.float16:
            weights = torch.where(torch.isinf(weights), torch.zeros_like(weights), weights)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        output_chunks.append(torch.matmul(weights, value_states))
        if output_attentions:
            attention_chunks.append(weights)

    attention_output = torch.cat(output_chunks, dim=2)
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(
        batch_size, query_length, -1
    )
    attention_output = self.o_proj(attention_output)
    returned_weights = torch.cat(attention_chunks, dim=2) if output_attentions else None
    return attention_output, returned_weights, past_key_value


def install_chunked_eager_attention(decoder, query_chunk_size: int = 32) -> None:
    for layer in decoder.layers:
        attention = layer.self_attn
        attention.stage_a_query_chunk_size = int(query_chunk_size)
        attention.forward = MethodType(chunked_eager_attention_forward, attention)
