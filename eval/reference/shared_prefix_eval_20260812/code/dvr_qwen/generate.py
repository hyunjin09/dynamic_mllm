"""Custom cache-aware generation for DVR-Qwen."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, Mapping

import torch

from dvr_qwen.attention import forward_decode_text, forward_text_only, forward_visual_read
from dvr_qwen.cache import DVRCache
from dvr_qwen.input_builder import DVRInputs, build_dvr_inputs
from dvr_qwen.masks import make_text_causal_mask, make_visual_read_mask
from dvr_qwen.modeling_dvr_qwen2_5_vl import (
    DVRLayerStats,
    DVRQwen2_5_VLForConditionalGeneration,
    qwen_num_hidden_layers,
    qwen_text_model,
)


@dataclass
class DVRPrefillState:
    last_hidden_state: torch.Tensor
    cache: DVRCache
    dvr_inputs: DVRInputs
    route_binary: torch.Tensor
    prefill_layer_stats: list[DVRLayerStats]


@dataclass
class DVRGenerationOutput:
    generated_ids: torch.Tensor
    state: DVRPrefillState
    decode_layer_stats: list[list[DVRLayerStats]] = field(default_factory=list)


def _normalize_static_route(
    model: DVRQwen2_5_VLForConditionalGeneration,
    visual_route_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    num_layers = qwen_num_hidden_layers(model.config)
    route = visual_route_mask.to(device=device, dtype=torch.bool)
    if route.ndim == 1:
        route = route.unsqueeze(0)
    if route.shape != (batch_size, num_layers):
        raise ValueError(f"visual_route_mask must have shape {(batch_size, num_layers)}, got {tuple(route.shape)}")
    return route


def _lm_logits(model: DVRQwen2_5_VLForConditionalGeneration, hidden_states: torch.Tensor) -> torch.Tensor:
    lm_device = next(model.lm_head.parameters()).device
    return model.lm_head(hidden_states.to(lm_device))


def _visual_memory_for_layer(
    dvr_inputs: DVRInputs,
    visual_memory_by_layer: Sequence[torch.Tensor] | torch.Tensor | None,
    layer_idx: int,
) -> torch.Tensor:
    if visual_memory_by_layer is None:
        return dvr_inputs.visual_memory
    if torch.is_tensor(visual_memory_by_layer):
        return visual_memory_by_layer[layer_idx]
    return visual_memory_by_layer[layer_idx]


@torch.inference_mode()
def dvr_prefill(
    model: DVRQwen2_5_VLForConditionalGeneration,
    inputs: Mapping[str, Any],
    visual_route_mask: torch.Tensor,
    route_mode: str = "static_route",
    visual_attention_bias: float = 0.0,
    visual_value_scale: float = 1.0,
    visual_memory_by_layer: Sequence[torch.Tensor] | torch.Tensor | None = None,
) -> DVRPrefillState:
    if route_mode != "static_route":
        raise ValueError("Phase 3 custom generation supports only route_mode='static_route'")

    dvr_inputs = build_dvr_inputs(
        model.base_model,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        mm_token_type_ids=inputs["mm_token_type_ids"],
        pixel_values=inputs.get("pixel_values"),
        pixel_values_videos=inputs.get("pixel_values_videos"),
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        second_per_grid_ts=inputs.get("second_per_grid_ts"),
    )

    batch_size = dvr_inputs.text_hidden.shape[0]
    if batch_size != 1:
        raise NotImplementedError("Phase 3 custom generation is validated for batch size 1 only")

    hidden_states = dvr_inputs.text_hidden
    route = _normalize_static_route(model, visual_route_mask, batch_size, hidden_states.device)
    text_model = qwen_text_model(model)
    cache = DVRCache(num_layers=qwen_num_hidden_layers(model.config))
    layer_stats: list[DVRLayerStats] = []

    for layer_idx, layer in enumerate(text_model.layers):
        layer_device = next(layer.parameters()).device
        hidden_states = hidden_states.to(layer_device)
        text_position_ids = dvr_inputs.text_position_ids.to(layer_device)
        visual_position_ids = dvr_inputs.visual_position_ids.to(layer_device)
        text_valid_mask = dvr_inputs.text_valid_mask.to(layer_device)
        visual_valid_mask = dvr_inputs.visual_valid_mask.to(layer_device)
        text_orig_idx = dvr_inputs.text_orig_idx.to(layer_device)
        visual_orig_idx = dvr_inputs.visual_orig_idx.to(layer_device)

        residual = hidden_states
        text_norm = layer.input_layernorm(hidden_states)
        text_position_embeddings = text_model.rotary_emb(text_norm, text_position_ids)

        gate = bool(route[0, layer_idx].item())
        if gate:
            visual_memory = _visual_memory_for_layer(dvr_inputs, visual_memory_by_layer, layer_idx).to(layer_device)
            visual_norm = layer.input_layernorm(visual_memory)
            visual_position_embeddings = text_model.rotary_emb(visual_norm, visual_position_ids)
            attn_mask = make_visual_read_mask(
                text_orig_idx,
                visual_orig_idx,
                text_valid_mask,
                visual_valid_mask,
                dtype=text_norm.dtype,
                device=layer_device,
            )
            attn_out, attn_stats = forward_visual_read(
                layer.self_attn,
                text_norm,
                visual_norm,
                attn_mask,
                text_position_embeddings,
                visual_position_embeddings,
                cache=cache,
                layer_idx=layer_idx,
                use_cache=True,
                visual_attention_bias=visual_attention_bias,
                visual_value_scale=visual_value_scale,
            )
        else:
            attn_mask = make_text_causal_mask(
                text_orig_idx,
                text_valid_mask,
                dtype=text_norm.dtype,
                device=layer_device,
            )
            attn_out, attn_stats = forward_text_only(
                layer.self_attn,
                text_norm,
                attn_mask,
                text_position_embeddings,
                cache=cache,
                layer_idx=layer_idx,
                use_cache=True,
            )

        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        layer_stats.append(
            DVRLayerStats(
                layer_idx=layer_idx,
                visual_read_on=gate,
                key_value_length=attn_stats.key_value_length,
                residual_sequence_length=hidden_states.shape[1],
                visual_kv_projected=attn_stats.visual_kv_projected,
            )
        )

    norm_device = next(text_model.norm.parameters()).device
    hidden_states = text_model.norm(hidden_states.to(norm_device))
    return DVRPrefillState(
        last_hidden_state=hidden_states,
        cache=cache,
        dvr_inputs=dvr_inputs,
        route_binary=route,
        prefill_layer_stats=layer_stats,
    )


def logits_from_prefill(
    model: DVRQwen2_5_VLForConditionalGeneration,
    state: DVRPrefillState,
) -> torch.Tensor:
    last_idx = state.dvr_inputs.text_valid_mask.long().sum(dim=1) - 1
    batch_idx = torch.arange(state.last_hidden_state.shape[0], device=state.last_hidden_state.device)
    last_hidden = state.last_hidden_state[
        batch_idx,
        last_idx.to(state.last_hidden_state.device),
    ]
    return _lm_logits(model, last_hidden[:, None, :])[:, -1, :]


def next_decode_position_ids(dvr_inputs: DVRInputs, generated_step: int) -> torch.Tensor:
    pos = dvr_inputs.full_prompt_len + generated_step
    if dvr_inputs.rope_deltas is not None:
        pos = pos + dvr_inputs.rope_deltas.squeeze(-1).to(pos.device)
    return pos.view(1, -1, 1).expand(3, -1, 1)


@torch.inference_mode()
def dvr_decode_one_step(
    model: DVRQwen2_5_VLForConditionalGeneration,
    token_id: torch.Tensor,
    state: DVRPrefillState,
    generated_step: int,
    visual_attention_bias: float = 0.0,
) -> tuple[torch.Tensor, list[DVRLayerStats]]:
    text_model = qwen_text_model(model)
    embed_device = next(model.model.get_input_embeddings().parameters()).device
    hidden_states = model.model.get_input_embeddings()(token_id.to(embed_device).view(-1, 1))
    position_ids = next_decode_position_ids(state.dvr_inputs, generated_step)
    layer_stats: list[DVRLayerStats] = []

    for layer_idx, layer in enumerate(text_model.layers):
        layer_device = next(layer.parameters()).device
        hidden_states = hidden_states.to(layer_device)
        text_position_ids = position_ids.to(layer_device)

        residual = hidden_states
        text_norm = layer.input_layernorm(hidden_states)
        text_position_embeddings = text_model.rotary_emb(text_norm, text_position_ids)
        attn_out, attn_stats = forward_decode_text(
            layer.self_attn,
            text_norm,
            text_position_embeddings,
            cache=state.cache,
            layer_idx=layer_idx,
            visual_attention_bias=visual_attention_bias,
        )

        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        layer_stats.append(
            DVRLayerStats(
                layer_idx=layer_idx,
                visual_read_on=bool(state.route_binary[0, layer_idx].item()),
                key_value_length=attn_stats.key_value_length,
                residual_sequence_length=hidden_states.shape[1],
                visual_kv_projected=attn_stats.visual_kv_projected,
            )
        )

    norm_device = next(text_model.norm.parameters()).device
    hidden_states = text_model.norm(hidden_states.to(norm_device))
    state.last_hidden_state = hidden_states
    logits = _lm_logits(model, hidden_states)[:, -1, :]
    return logits, layer_stats


def _eos_token_set(model: DVRQwen2_5_VLForConditionalGeneration, eos_token_ids: int | list[int] | None) -> set[int]:
    if eos_token_ids is None:
        eos_token_ids = getattr(model.config, "eos_token_id", None)
    if eos_token_ids is None and hasattr(model, "base_model"):
        eos_token_ids = getattr(model.base_model.generation_config, "eos_token_id", None)
    if eos_token_ids is None:
        return set()
    if isinstance(eos_token_ids, int):
        return {eos_token_ids}
    return {int(token_id) for token_id in eos_token_ids}


def generation_repetition_penalty(
    model: DVRQwen2_5_VLForConditionalGeneration,
    repetition_penalty: float | None = None,
) -> float:
    if repetition_penalty is None:
        repetition_penalty = 1.0
        if hasattr(model, "base_model"):
            generation_config = getattr(model.base_model, "generation_config", None)
            repetition_penalty = getattr(generation_config, "repetition_penalty", 1.0)
        if repetition_penalty is None:
            repetition_penalty = 1.0
    penalty = float(repetition_penalty)
    if penalty <= 0.0:
        raise ValueError("repetition_penalty must be strictly positive")
    return penalty


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty == 1.0:
        return logits

    adjusted = logits.to(copy=True, dtype=torch.float32)
    history = token_history.to(device=adjusted.device, dtype=torch.long)
    if history.ndim == 1:
        history = history.unsqueeze(0)
    for batch_idx in range(adjusted.shape[0]):
        token_ids = torch.unique(history[batch_idx])
        token_scores = adjusted[batch_idx, token_ids]
        adjusted[batch_idx, token_ids] = torch.where(
            token_scores < 0,
            token_scores * repetition_penalty,
            token_scores / repetition_penalty,
        )
    return adjusted


def generation_token_history(
    prompt_input_ids: torch.Tensor,
    generated: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    prompt_ids = prompt_input_ids.to(device=device, dtype=torch.long)
    if not generated:
        return prompt_ids
    generated_ids = torch.cat([token.to(device=device, dtype=torch.long) for token in generated], dim=1)
    return torch.cat([prompt_ids, generated_ids], dim=1)


def generation_policy_record(
    model: DVRQwen2_5_VLForConditionalGeneration,
    eos_token_ids: int | list[int] | None = None,
    repetition_penalty: float | None = None,
) -> dict[str, Any]:
    return {
        "eos_token_ids": sorted(_eos_token_set(model, eos_token_ids)),
        "repetition_penalty": generation_repetition_penalty(model, repetition_penalty),
    }


@torch.inference_mode()
def dvr_greedy_generate(
    model: DVRQwen2_5_VLForConditionalGeneration,
    inputs: Mapping[str, Any],
    visual_route_mask: torch.Tensor,
    max_new_tokens: int,
    route_mode: str = "static_route",
    eos_token_ids: int | list[int] | None = None,
    stop_on_eos: bool = True,
    visual_attention_bias: float = 0.0,
    visual_value_scale: float = 1.0,
    visual_memory_by_layer: Sequence[torch.Tensor] | torch.Tensor | None = None,
) -> DVRGenerationOutput:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    state = dvr_prefill(
        model,
        inputs,
        visual_route_mask=visual_route_mask,
        route_mode=route_mode,
        visual_attention_bias=visual_attention_bias,
        visual_value_scale=visual_value_scale,
        visual_memory_by_layer=visual_memory_by_layer,
    )
    logits = logits_from_prefill(model, state)
    eos_ids = _eos_token_set(model, eos_token_ids)
    generated: list[torch.Tensor] = []
    decode_stats: list[list[DVRLayerStats]] = []

    for step in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1)
        generated.append(next_token[:, None])
        logits, step_stats = dvr_decode_one_step(
            model,
            next_token,
            state,
            generated_step=step,
            visual_attention_bias=visual_attention_bias,
        )
        decode_stats.append(step_stats)
        if stop_on_eos and eos_ids and int(next_token[0].item()) in eos_ids:
            break

    if generated:
        generated_ids = torch.cat(generated, dim=1)
    else:
        generated_ids = torch.empty(
            state.route_binary.shape[0],
            0,
            dtype=torch.long,
            device=state.route_binary.device,
        )
    return DVRGenerationOutput(
        generated_ids=generated_ids,
        state=state,
        decode_layer_stats=decode_stats,
    )
