"""Static-route forward scoring and deterministic generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

from ..actions import normalize_visual_on_mask
from .cache import BinaryRouteCache
from .inputs import BinaryInputs, build_binary_inputs, resolve_causal_lm, resolve_decoder, scatter_streams
from .layers import LayerExecution, decode_text_layer, visual_off_layer, visual_on_layer


@dataclass
class BinaryPrefill:
    text_hidden_state: torch.Tensor
    visual_hidden_state: torch.Tensor
    route: torch.BoolTensor
    inputs: BinaryInputs
    cache: BinaryRouteCache | None
    layer_stats: list[LayerExecution]


@dataclass
class BinaryForwardOutput:
    logits: torch.Tensor
    full_hidden_state: torch.Tensor
    prefill: BinaryPrefill


@dataclass
class BinaryGenerationOutput:
    generated_ids: torch.LongTensor
    prefill_logits: torch.Tensor
    prefill: BinaryPrefill
    decode_stats: list[list[LayerExecution]]


def _num_layers(model) -> int:
    return len(resolve_decoder(model).layers)


def _normalize_final(model, text_states, visual_states, meta):
    decoder = resolve_decoder(model)
    device = next(decoder.norm.parameters()).device
    full = scatter_streams(text_states.to(device), visual_states.to(device), meta)
    return decoder.norm(full)


@torch.inference_mode()
def binary_prefill(
    model,
    inputs: Mapping[str, Any],
    visual_on_mask,
    *,
    use_cache: bool = False,
    prepared_inputs: BinaryInputs | None = None,
) -> BinaryPrefill:
    meta = prepared_inputs or build_binary_inputs(model, dict(inputs))
    if meta.text_states.shape[0] != 1:
        raise NotImplementedError("binary executor is validated only for batch size one")
    decoder = resolve_decoder(model)
    route = normalize_visual_on_mask(
        visual_on_mask,
        num_layers=len(decoder.layers),
        batch_size=1,
        device=meta.text_states.device,
    )
    cache = BinaryRouteCache(len(decoder.layers)) if use_cache else None
    native_all_on = bool(route.all().item()) and bool(meta.full_attention_mask.bool().all().item())
    text_states = meta.text_states
    visual_states = meta.visual_states
    stats = []
    for layer_index, layer in enumerate(decoder.layers):
        function = visual_on_layer if bool(route[0, layer_index]) else visual_off_layer
        text_states, visual_states, current = function(
            model,
            layer,
            text_states,
            visual_states,
            meta,
            layer_index=layer_index,
            cache=cache,
            use_cache=use_cache,
            **({"native_causal": True} if function is visual_on_layer and native_all_on else {}),
        )
        stats.append(current)
    return BinaryPrefill(text_states, visual_states, route, meta, cache, stats)


@torch.inference_mode()
def binary_route_forward(
    model,
    inputs: Mapping[str, Any],
    visual_on_mask,
    *,
    prepared_inputs: BinaryInputs | None = None,
) -> BinaryForwardOutput:
    prefill = binary_prefill(
        model,
        inputs,
        visual_on_mask,
        use_cache=False,
        prepared_inputs=prepared_inputs,
    )
    hidden = _normalize_final(model, prefill.text_hidden_state, prefill.visual_hidden_state, prefill.inputs)
    causal_lm = resolve_causal_lm(model)
    lm_device = next(causal_lm.lm_head.parameters()).device
    logits = causal_lm.lm_head(hidden.to(lm_device))
    return BinaryForwardOutput(logits=logits, full_hidden_state=hidden, prefill=prefill)


def _last_text_logits(model, prefill: BinaryPrefill) -> torch.Tensor:
    decoder = resolve_decoder(model)
    norm_device = next(decoder.norm.parameters()).device
    hidden = decoder.norm(prefill.text_hidden_state.to(norm_device))
    last = prefill.inputs.text_valid_mask.long().sum(dim=1) - 1
    selected = hidden[torch.arange(hidden.shape[0], device=hidden.device), last.to(hidden.device)]
    causal_lm = resolve_causal_lm(model)
    lm_device = next(causal_lm.lm_head.parameters()).device
    return causal_lm.lm_head(selected[:, None].to(lm_device))[:, -1]


def _decode_position_ids(meta: BinaryInputs, generated_step: int) -> torch.Tensor:
    position = meta.full_prompt_len + generated_step
    if meta.rope_deltas is not None:
        position = position + meta.rope_deltas.squeeze(-1).to(position.device)
    return position.view(1, -1, 1).expand(3, -1, 1)


def _eos_ids(model, supplied) -> set[int]:
    value = supplied
    causal_lm = resolve_causal_lm(model)
    if value is None:
        value = getattr(getattr(causal_lm, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(causal_lm.config, "eos_token_id", None)
    if value is None:
        return set()
    return {int(value)} if isinstance(value, int) else {int(item) for item in value}


def _repetition_penalty(model, supplied: float | None) -> float:
    if supplied is not None:
        value = float(supplied)
    else:
        causal_lm = resolve_causal_lm(model)
        value = float(getattr(getattr(causal_lm, "generation_config", None), "repetition_penalty", 1.0) or 1.0)
    if value <= 0:
        raise ValueError("repetition_penalty must be positive")
    return value


def _apply_repetition_penalty(logits: torch.Tensor, history: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    output = logits.float().clone()
    for batch_idx in range(output.shape[0]):
        ids = torch.unique(history[batch_idx].to(output.device))
        scores = output[batch_idx, ids]
        output[batch_idx, ids] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return output


@torch.inference_mode()
def binary_greedy_generate(
    model,
    inputs: Mapping[str, Any],
    visual_on_mask,
    *,
    max_new_tokens: int,
    eos_token_ids=None,
    repetition_penalty: float | None = None,
    prepared_inputs: BinaryInputs | None = None,
) -> BinaryGenerationOutput:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    prefill = binary_prefill(
        model,
        inputs,
        visual_on_mask,
        use_cache=True,
        prepared_inputs=prepared_inputs,
    )
    assert prefill.cache is not None
    logits = _last_text_logits(model, prefill)
    prefill_logits = logits.detach().cpu()
    decoder = resolve_decoder(model)
    causal_lm = resolve_causal_lm(model)
    embed = decoder.get_input_embeddings() if hasattr(decoder, "get_input_embeddings") else decoder.embed_tokens
    eos = _eos_ids(model, eos_token_ids)
    penalty = _repetition_penalty(model, repetition_penalty)
    generated = []
    decode_stats = []
    for step in range(max_new_tokens):
        history = inputs["input_ids"].to(logits.device)
        if generated:
            history = torch.cat([history, *[item.to(logits.device) for item in generated]], dim=1)
        token = _apply_repetition_penalty(logits, history, penalty).argmax(dim=-1)
        generated.append(token[:, None])
        hidden = embed(token.to(next(embed.parameters()).device).view(-1, 1))
        position_ids = _decode_position_ids(prefill.inputs, step)
        current_stats = []
        for layer_index, layer in enumerate(decoder.layers):
            hidden, stat = decode_text_layer(
                model,
                layer,
                hidden,
                position_ids,
                layer_index=layer_index,
                cache=prefill.cache,
            )
            stat = replace(stat, visual_on=bool(prefill.route[0, layer_index]))
            current_stats.append(stat)
        norm_device = next(decoder.norm.parameters()).device
        hidden = decoder.norm(hidden.to(norm_device))
        logits = causal_lm.lm_head(hidden.to(next(causal_lm.lm_head.parameters()).device))[:, -1]
        decode_stats.append(current_stats)
        if eos and int(token[0]) in eos:
            break
    ids = torch.cat(generated, dim=1) if generated else torch.empty((1, 0), dtype=torch.long)
    return BinaryGenerationOutput(ids, prefill_logits, prefill, decode_stats)
