"""Exact layer-local READ/WRITE interventions for the binary Qwen executor.

The implementation deliberately composes the frozen binary executor's native
full-row and compacted text/control-row calls.  It does not use the older
attention-output subtraction intervention in :mod:`interventions.read_path`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from ..actions import normalize_visual_on_mask
from .cache import BinaryRouteCache
from .generation import (
    _apply_repetition_penalty,
    _decode_position_ids,
    _eos_ids,
    _repetition_penalty,
)
from .inputs import BinaryInputs, build_binary_inputs, resolve_causal_lm, resolve_decoder, scatter_streams
from .layers import decode_text_layer, visual_off_layer, visual_on_layer


FOUR_ACTIONS = ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")


def normalize_four_action(action: str) -> str:
    normalized = str(action).strip().upper()
    if normalized not in FOUR_ACTIONS:
        raise ValueError(f"action must be one of {FOUR_ACTIONS}, got {action!r}")
    return normalized


@dataclass(frozen=True)
class FourActionLayerExecution:
    layer_index: int
    action: str
    read_on: bool
    write_on: bool
    residual_rows: int
    cache_rows: int | None
    decoder_calls: int


@dataclass
class FullBaseline:
    inputs: BinaryInputs
    pre_layer_states: list[tuple[torch.Tensor, torch.Tensor]]
    text_hidden_state: torch.Tensor
    visual_hidden_state: torch.Tensor
    full_hidden_state: torch.Tensor
    prompt_logits: torch.Tensor
    cache: BinaryRouteCache | None
    layer_stats: list[FourActionLayerExecution]
    native_causal: bool
    layer_actions: tuple[str, ...]


@dataclass
class LocalFourActionPrefill:
    text_hidden_state: torch.Tensor
    visual_hidden_state: torch.Tensor
    inputs: BinaryInputs
    cache: BinaryRouteCache | None
    layer_stats: list[FourActionLayerExecution]
    target_layer: int
    action: str
    target_pre_text_state: torch.Tensor
    target_pre_visual_state: torch.Tensor
    target_post_text_state: torch.Tensor
    target_post_visual_state: torch.Tensor
    post_layer_text_states: list[torch.Tensor]


@dataclass
class LocalFourActionForwardOutput:
    prompt_logits: torch.Tensor
    full_hidden_state: torch.Tensor
    prefill: LocalFourActionPrefill


@dataclass
class FourActionGenerationOutput:
    generated_ids: torch.LongTensor
    decode_cache: BinaryRouteCache
    decode_stats: list[list[Any]]


@dataclass(frozen=True)
class FourActionTokenScore:
    token_ids: list[int]
    token_logprobs: list[float]
    sequence_logprob: float
    mean_logprob: float


def _layer_stat(
    *,
    layer_index: int,
    action: str,
    residual_rows: int,
    cache: BinaryRouteCache | None,
    decoder_calls: int,
) -> FourActionLayerExecution:
    return FourActionLayerExecution(
        layer_index=layer_index,
        action=action,
        read_on=action in {"READ_ONLY", "FULL"},
        write_on=action in {"WRITE_ONLY", "FULL"},
        residual_rows=residual_rows,
        cache_rows=None if cache is None else cache.get_seq_length(layer_index),
        decoder_calls=decoder_calls,
    )


def _clone_cache_prefix(source: BinaryRouteCache, through_layer_exclusive: int) -> BinaryRouteCache:
    if through_layer_exclusive < 0 or through_layer_exclusive > len(source.key_cache):
        raise ValueError("invalid cache layer boundary")
    output = BinaryRouteCache(len(source.key_cache))
    for layer_index in range(through_layer_exclusive):
        key = source.key_cache[layer_index]
        value = source.value_cache[layer_index]
        if key is None or value is None:
            raise RuntimeError(f"baseline cache is empty at prefix layer {layer_index}")
        output.key_cache[layer_index] = key.detach().clone()
        output.value_cache[layer_index] = value.detach().clone()
    return output


def clone_four_action_cache(source: BinaryRouteCache) -> BinaryRouteCache:
    """Clone every populated layer of a prompt cache for independent decoding."""
    output = BinaryRouteCache(len(source.key_cache))
    for layer_index, (key, value) in enumerate(zip(source.key_cache, source.value_cache)):
        if (key is None) != (value is None):
            raise RuntimeError(f"incomplete cache entry at layer {layer_index}")
        if key is not None:
            output.key_cache[layer_index] = key.detach().clone()
            output.value_cache[layer_index] = value.detach().clone()
    return output


def four_action_layer(
    model,
    layer,
    text_states: torch.Tensor,
    visual_states: torch.Tensor,
    meta: BinaryInputs,
    *,
    action: str,
    layer_index: int,
    cache: BinaryRouteCache | None = None,
    use_cache: bool = False,
    native_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, FourActionLayerExecution]:
    """Execute one exact factorial action from a shared pre-layer state.

    ``WRITE_ONLY`` needs two native layer calls: the full-row call supplies the
    updated visual rows, while the compacted call supplies text/control rows
    and the READ-off target-layer K/V cache.  Both calls start from the same
    incoming states.
    """
    action = normalize_four_action(action)
    if action == "FULL":
        next_text, next_visual, _ = visual_on_layer(
            model,
            layer,
            text_states,
            visual_states,
            meta,
            layer_index=layer_index,
            cache=cache,
            use_cache=use_cache,
            native_causal=native_causal,
        )
        calls = 1
    elif action == "IGNORE":
        next_text, next_visual, _ = visual_off_layer(
            model,
            layer,
            text_states,
            visual_states,
            meta,
            layer_index=layer_index,
            cache=cache,
            use_cache=use_cache,
        )
        calls = 1
    elif action == "READ_ONLY":
        next_text, _, _ = visual_on_layer(
            model,
            layer,
            text_states,
            visual_states,
            meta,
            layer_index=layer_index,
            cache=cache,
            use_cache=use_cache,
            native_causal=native_causal,
        )
        next_visual = visual_states.to(next_text.device)
        calls = 1
    else:
        # The full-row call is cache-free because decoding at this target layer
        # must not retain visual K/V when READ is off.
        _, next_visual, _ = visual_on_layer(
            model,
            layer,
            text_states,
            visual_states,
            meta,
            layer_index=layer_index,
            cache=None,
            use_cache=False,
            native_causal=native_causal,
        )
        next_text, _, _ = visual_off_layer(
            model,
            layer,
            text_states,
            visual_states,
            meta,
            layer_index=layer_index,
            cache=cache,
            use_cache=use_cache,
        )
        calls = 2

    residual_rows = int(next_text.shape[1]) + (int(next_visual.shape[1]) if action in {"READ_ONLY", "WRITE_ONLY", "FULL"} else 0)
    return next_text, next_visual, _layer_stat(
        layer_index=layer_index,
        action=action,
        residual_rows=residual_rows,
        cache=cache,
        decoder_calls=calls,
    )


def unified_target_four_action_layer(
    model,
    layer,
    text_states: torch.Tensor,
    visual_states: torch.Tensor,
    meta: BinaryInputs,
    *,
    action: str,
    layer_index: int,
    prefix_cache: BinaryRouteCache | None,
    use_cache: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, BinaryRouteCache | None, FourActionLayerExecution]:
    """Run the same two materialized target calls for every factorial action.

    Every branch executes, in this fixed order, (1) a materialized full-row
    layer call and (2) the frozen compacted text/control-row layer call.  The
    action only selects which text output, visual output, and target-layer cache
    survive.  This prevents action-dependent kernel dispatch outside the
    intended READ/WRITE selection while retaining exact binary-OFF semantics.
    """
    action = normalize_four_action(action)
    full_cache = None if prefix_cache is None else clone_four_action_cache(prefix_cache)
    text_cache = None if prefix_cache is None else clone_four_action_cache(prefix_cache)
    full_text, full_visual, _ = visual_on_layer(
        model,
        layer,
        text_states,
        visual_states,
        meta,
        layer_index=layer_index,
        cache=full_cache,
        use_cache=use_cache,
        native_causal=False,
    )
    off_text, carried_visual, _ = visual_off_layer(
        model,
        layer,
        text_states,
        visual_states,
        meta,
        layer_index=layer_index,
        cache=text_cache,
        use_cache=use_cache,
    )
    read_on = action in {"READ_ONLY", "FULL"}
    write_on = action in {"WRITE_ONLY", "FULL"}
    next_text = full_text if read_on else off_text
    next_visual = full_visual if write_on else carried_visual
    selected_cache = full_cache if read_on else text_cache
    residual_rows = int(next_text.shape[1]) + (
        int(next_visual.shape[1]) if action != "IGNORE" else 0
    )
    return next_text, next_visual, selected_cache, _layer_stat(
        layer_index=layer_index,
        action=action,
        residual_rows=residual_rows,
        cache=selected_cache,
        decoder_calls=2,
    )


def _native_causal(meta: BinaryInputs) -> bool:
    return bool(meta.full_attention_mask.bool().all().item())


def _normalize_full(model, text_states: torch.Tensor, visual_states: torch.Tensor, meta: BinaryInputs) -> torch.Tensor:
    decoder = resolve_decoder(model)
    device = next(decoder.norm.parameters(), torch.empty(0, device=text_states.device)).device
    full = scatter_streams(text_states.to(device), visual_states.to(device), meta)
    return decoder.norm(full)


def _last_text_logits(model, full_hidden: torch.Tensor, meta: BinaryInputs) -> torch.Tensor:
    valid_count = meta.text_valid_mask.long().sum(dim=1)
    compact_last = valid_count - 1
    full_last = meta.text_indices[
        torch.arange(meta.text_indices.shape[0], device=meta.text_indices.device),
        compact_last.to(meta.text_indices.device),
    ]
    selected = full_hidden[
        torch.arange(full_hidden.shape[0], device=full_hidden.device),
        full_last.to(full_hidden.device),
    ]
    causal_lm = resolve_causal_lm(model)
    lm_device = next(causal_lm.lm_head.parameters()).device
    return causal_lm.lm_head(selected[:, None].to(lm_device))[:, -1]


@torch.inference_mode()
def capture_full_baseline(
    model,
    inputs: Mapping[str, Any],
    *,
    prepared_inputs: BinaryInputs | None = None,
    use_cache: bool = True,
    native_causal: bool = False,
) -> FullBaseline:
    """Run FULL once and retain every shared pre-layer state.

    The default materialized causal mask matches the current mixed binary-route
    executor, making single-layer IGNORE exactly comparable to binary OFF.
    ``native_causal=True`` is reserved for the native all-FULL parity control.
    """
    meta = prepared_inputs or build_binary_inputs(model, dict(inputs))
    if meta.text_states.shape[0] != 1:
        raise NotImplementedError("four-action executor is validated only for batch size one")
    decoder = resolve_decoder(model)
    cache = BinaryRouteCache(len(decoder.layers)) if use_cache else None
    text_states = meta.text_states
    visual_states = meta.visual_states
    pre_layer_states: list[tuple[torch.Tensor, torch.Tensor]] = []
    stats: list[FourActionLayerExecution] = []
    if native_causal and not _native_causal(meta):
        raise ValueError("native maskless causal dispatch requires an unpadded prompt")
    for layer_index, layer in enumerate(decoder.layers):
        pre_layer_states.append((text_states.detach().clone(), visual_states.detach().clone()))
        text_states, visual_states, execution = four_action_layer(
            model,
            layer,
            text_states,
            visual_states,
            meta,
            action="FULL",
            layer_index=layer_index,
            cache=cache,
            use_cache=use_cache,
            native_causal=native_causal,
        )
        stats.append(execution)
    full_hidden = _normalize_full(model, text_states, visual_states, meta)
    return FullBaseline(
        inputs=meta,
        pre_layer_states=pre_layer_states,
        text_hidden_state=text_states,
        visual_hidden_state=visual_states,
        full_hidden_state=full_hidden,
        prompt_logits=_last_text_logits(model, full_hidden, meta),
        cache=cache,
        layer_stats=stats,
        native_causal=native_causal,
        layer_actions=tuple("FULL" for _ in decoder.layers),
    )


@torch.inference_mode()
def capture_four_action_route(
    model,
    inputs: Mapping[str, Any],
    layer_actions,
    *,
    prepared_inputs: BinaryInputs | None = None,
    use_cache: bool = True,
) -> FullBaseline:
    """Run one complete four-action route through the unified executor."""
    meta = prepared_inputs or build_binary_inputs(model, dict(inputs))
    if meta.text_states.shape[0] != 1:
        raise NotImplementedError("four-action executor is validated only for batch size one")
    decoder = resolve_decoder(model)
    actions = tuple(normalize_four_action(action) for action in layer_actions)
    if len(actions) != len(decoder.layers):
        raise ValueError(
            f"layer_actions must contain exactly {len(decoder.layers)} actions, "
            f"got {len(actions)}"
        )
    cache = BinaryRouteCache(len(decoder.layers)) if use_cache else None
    text_states = meta.text_states
    visual_states = meta.visual_states
    pre_layer_states: list[tuple[torch.Tensor, torch.Tensor]] = []
    stats: list[FourActionLayerExecution] = []
    for layer_index, (layer, action) in enumerate(zip(decoder.layers, actions)):
        pre_layer_states.append((text_states.detach().clone(), visual_states.detach().clone()))
        text_states, visual_states, execution = four_action_layer(
            model,
            layer,
            text_states,
            visual_states,
            meta,
            action=action,
            layer_index=layer_index,
            cache=cache,
            use_cache=use_cache,
            native_causal=False,
        )
        stats.append(execution)
    full_hidden = _normalize_full(model, text_states, visual_states, meta)
    return FullBaseline(
        inputs=meta,
        pre_layer_states=pre_layer_states,
        text_hidden_state=text_states,
        visual_hidden_state=visual_states,
        full_hidden_state=full_hidden,
        prompt_logits=_last_text_logits(model, full_hidden, meta),
        cache=cache,
        layer_stats=stats,
        native_causal=False,
        layer_actions=actions,
    )


@torch.inference_mode()
def capture_online_four_action_route(
    model,
    inputs: Mapping[str, Any],
    action_selector: Callable[[int, torch.Tensor, torch.Tensor, BinaryInputs], str],
    *,
    prepared_inputs: BinaryInputs | None = None,
    use_cache: bool = True,
) -> FullBaseline:
    """Choose each action from the actual state produced by its routed prefix."""
    meta = prepared_inputs or build_binary_inputs(model, dict(inputs))
    if meta.text_states.shape[0] != 1:
        raise NotImplementedError("four-action executor is validated only for batch size one")
    decoder = resolve_decoder(model)
    cache = BinaryRouteCache(len(decoder.layers)) if use_cache else None
    text_states = meta.text_states
    visual_states = meta.visual_states
    pre_layer_states: list[tuple[torch.Tensor, torch.Tensor]] = []
    stats: list[FourActionLayerExecution] = []
    actions: list[str] = []
    for layer_index, layer in enumerate(decoder.layers):
        pre_layer_states.append((text_states.detach().clone(), visual_states.detach().clone()))
        action = normalize_four_action(
            action_selector(layer_index, text_states, visual_states, meta)
        )
        actions.append(action)
        text_states, visual_states, execution = four_action_layer(
            model,
            layer,
            text_states,
            visual_states,
            meta,
            action=action,
            layer_index=layer_index,
            cache=cache,
            use_cache=use_cache,
            native_causal=False,
        )
        stats.append(execution)
    full_hidden = _normalize_full(model, text_states, visual_states, meta)
    return FullBaseline(
        inputs=meta,
        pre_layer_states=pre_layer_states,
        text_hidden_state=text_states,
        visual_hidden_state=visual_states,
        full_hidden_state=full_hidden,
        prompt_logits=_last_text_logits(model, full_hidden, meta),
        cache=cache,
        layer_stats=stats,
        native_causal=False,
        layer_actions=tuple(actions),
    )


@torch.inference_mode()
def capture_route_baseline(
    model,
    inputs: Mapping[str, Any],
    visual_on_mask,
    *,
    prepared_inputs: BinaryInputs | None = None,
    use_cache: bool = True,
) -> FullBaseline:
    """Run one arbitrary binary anchor route through unified machinery."""
    meta = prepared_inputs or build_binary_inputs(model, dict(inputs))
    if meta.text_states.shape[0] != 1:
        raise NotImplementedError("four-action executor is validated only for batch size one")
    decoder = resolve_decoder(model)
    route = normalize_visual_on_mask(
        visual_on_mask,
        num_layers=len(decoder.layers),
        batch_size=1,
        device=meta.text_states.device,
    )
    actions = tuple("FULL" if bool(value) else "IGNORE" for value in route[0])
    return capture_four_action_route(
        model,
        inputs,
        actions,
        prepared_inputs=meta,
        use_cache=use_cache,
    )


def _four_action_forward_from_baseline(
    model,
    baseline: FullBaseline,
    target_layer: int,
    action: str,
) -> LocalFourActionForwardOutput:
    decoder = resolve_decoder(model)
    if target_layer < 0 or target_layer >= len(decoder.layers):
        raise ValueError(f"target_layer must be in [0, {len(decoder.layers) - 1}]")
    if baseline.cache is None:
        raise ValueError("local four-action forward requires a cached FULL baseline")
    action = normalize_four_action(action)
    meta = baseline.inputs
    if baseline.native_causal:
        raise ValueError("unified local factorial requires a materialized-mask FULL baseline")
    prefix_cache = _clone_cache_prefix(baseline.cache, target_layer)
    pre_text, pre_visual = baseline.pre_layer_states[target_layer]
    text_states, visual_states, cache, target_stat = unified_target_four_action_layer(
        model,
        decoder.layers[target_layer],
        pre_text,
        pre_visual,
        meta,
        action=action,
        layer_index=target_layer,
        prefix_cache=prefix_cache,
        use_cache=True,
    )
    assert cache is not None
    target_post_text = text_states.detach().clone()
    target_post_visual = visual_states.detach().clone()
    stats = list(baseline.layer_stats[:target_layer]) + [target_stat]
    post_layer_text_states = [
        baseline.pre_layer_states[index + 1][0]
        for index in range(target_layer)
    ] + [target_post_text]
    for layer_index in range(target_layer + 1, len(decoder.layers)):
        suffix_action = baseline.layer_actions[layer_index]
        text_states, visual_states, execution = four_action_layer(
            model,
            decoder.layers[layer_index],
            text_states,
            visual_states,
            meta,
            action=suffix_action,
            layer_index=layer_index,
            cache=cache,
            use_cache=True,
            native_causal=False,
        )
        stats.append(execution)
        post_layer_text_states.append(text_states.detach().clone())
    full_hidden = _normalize_full(model, text_states, visual_states, meta)
    prefill = LocalFourActionPrefill(
        text_hidden_state=text_states,
        visual_hidden_state=visual_states,
        inputs=meta,
        cache=cache,
        layer_stats=stats,
        target_layer=target_layer,
        action=action,
        target_pre_text_state=pre_text,
        target_pre_visual_state=pre_visual,
        target_post_text_state=target_post_text,
        target_post_visual_state=target_post_visual,
        post_layer_text_states=post_layer_text_states,
    )
    return LocalFourActionForwardOutput(
        prompt_logits=_last_text_logits(model, full_hidden, meta),
        full_hidden_state=full_hidden,
        prefill=prefill,
    )


@torch.inference_mode()
def local_four_action_forward(
    model,
    baseline: FullBaseline,
    target_layer: int,
    action: str,
) -> LocalFourActionForwardOutput:
    """Change exactly one layer in an all-FULL baseline."""
    if any(value != "FULL" for value in baseline.layer_actions):
        raise ValueError("local FULL-context forward requires an all-FULL baseline")
    return _four_action_forward_from_baseline(model, baseline, target_layer, action)


@torch.inference_mode()
def route_conditioned_four_action_forward(
    model,
    baseline: FullBaseline,
    target_layer: int,
    action: str,
) -> LocalFourActionForwardOutput:
    """Override one anchor-OFF layer and restore the anchor route elsewhere."""
    if target_layer < 0 or target_layer >= len(baseline.layer_actions):
        raise ValueError(f"target_layer must be in [0, {len(baseline.layer_actions) - 1}]")
    if baseline.layer_actions[target_layer] != "IGNORE":
        raise ValueError("route-conditioned target layer must be OFF in the anchor route")
    return _four_action_forward_from_baseline(model, baseline, target_layer, action)


def full_baseline_post_layer_text_states(baseline: FullBaseline) -> list[torch.Tensor]:
    """Return post-layer text states for layers 0..L-1."""
    return [
        *[baseline.pre_layer_states[index][0] for index in range(1, len(baseline.pre_layer_states))],
        baseline.text_hidden_state,
    ]


def _logits_from_layer_text_states(model, states: list[torch.Tensor], meta: BinaryInputs) -> torch.Tensor:
    decoder = resolve_decoder(model)
    causal_lm = resolve_causal_lm(model)
    norm_device = next(decoder.norm.parameters(), torch.empty(0, device=states[0].device)).device
    lm_device = next(causal_lm.lm_head.parameters()).device
    valid_count = meta.text_valid_mask.long().sum(dim=1) - 1
    rows = []
    for hidden in states:
        row_index = (
            torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
            if hidden.shape[1] == 1
            else valid_count.to(hidden.device)
        )
        selected = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            row_index,
        ]
        normalized = decoder.norm(selected[:, None].to(norm_device))
        rows.append(causal_lm.lm_head(normalized.to(lm_device))[:, -1])
    return torch.stack(rows, dim=0)


@torch.inference_mode()
def layerwise_token_scores_from_cached_prompt(
    model,
    post_layer_text_states: list[torch.Tensor],
    meta: BinaryInputs,
    prompt_cache: BinaryRouteCache,
    token_ids: torch.Tensor,
) -> list[float]:
    """Teacher-force one target through a final-norm/LM-head logit lens.

    Each returned value is the length-normalized target log probability read
    from the post-layer hidden state at the corresponding decoder depth.
    """
    decoder = resolve_decoder(model)
    if len(post_layer_text_states) != len(decoder.layers):
        raise ValueError("post-layer trajectory must contain exactly one state per decoder layer")
    ids = token_ids.reshape(-1)
    if ids.numel() < 1:
        raise ValueError("trajectory token_ids must be nonempty")
    cache = clone_four_action_cache(prompt_cache)
    logits_by_layer = _logits_from_layer_text_states(model, post_layer_text_states, meta)
    sums = torch.zeros(len(decoder.layers), dtype=torch.float64)
    for position, token in enumerate(ids):
        target = token.to(logits_by_layer.device)
        values = torch.log_softmax(logits_by_layer.float(), dim=-1)[:, 0, target]
        sums += values.detach().cpu().double()
        if position + 1 == ids.numel():
            continue
        embed = decoder.get_input_embeddings() if hasattr(decoder, "get_input_embeddings") else decoder.embed_tokens
        hidden = embed(token.to(next(embed.parameters()).device).view(-1, 1))
        position_ids = _decode_position_ids(meta, position)
        next_logits = []
        for layer_index, layer in enumerate(decoder.layers):
            hidden, _ = decode_text_layer(
                model,
                layer,
                hidden,
                position_ids,
                layer_index=layer_index,
                cache=cache,
            )
            next_logits.append(_logits_from_layer_text_states(model, [hidden], meta)[0])
        logits_by_layer = torch.stack(next_logits, dim=0)
    return [float(value / ids.numel()) for value in sums.tolist()]


def _decode_logits(
    model,
    token: torch.Tensor,
    *,
    meta: BinaryInputs,
    cache: BinaryRouteCache,
    generated_step: int,
) -> tuple[torch.Tensor, list[Any]]:
    decoder = resolve_decoder(model)
    embed = decoder.get_input_embeddings() if hasattr(decoder, "get_input_embeddings") else decoder.embed_tokens
    hidden = embed(token.to(next(embed.parameters()).device).view(-1, 1))
    position_ids = _decode_position_ids(meta, generated_step)
    stats = []
    for layer_index, layer in enumerate(decoder.layers):
        hidden, execution = decode_text_layer(
            model,
            layer,
            hidden,
            position_ids,
            layer_index=layer_index,
            cache=cache,
        )
        stats.append(execution)
    norm_device = next(decoder.norm.parameters(), torch.empty(0, device=hidden.device)).device
    hidden = decoder.norm(hidden.to(norm_device))
    causal_lm = resolve_causal_lm(model)
    lm_device = next(causal_lm.lm_head.parameters()).device
    return causal_lm.lm_head(hidden.to(lm_device))[:, -1], stats


@torch.inference_mode()
def greedy_generate_from_local_forward(
    model,
    output: LocalFourActionForwardOutput,
    prompt_input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids=None,
    repetition_penalty: float | None = None,
) -> FourActionGenerationOutput:
    """Greedily decode without mutating the reusable local prompt cache."""
    source_cache = output.prefill.cache
    if source_cache is None:
        raise ValueError("generation requires a cached local forward")
    return greedy_generate_from_cached_prompt(
        model,
        output.prompt_logits,
        output.prefill.inputs,
        source_cache,
        prompt_input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        repetition_penalty=repetition_penalty,
    )


@torch.inference_mode()
def greedy_generate_from_cached_prompt(
    model,
    prompt_logits: torch.Tensor,
    meta: BinaryInputs,
    prompt_cache: BinaryRouteCache,
    prompt_input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids=None,
    repetition_penalty: float | None = None,
) -> FourActionGenerationOutput:
    """Greedily decode from any four-action-compatible prompt cache clone."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    cache = clone_four_action_cache(prompt_cache)
    logits = prompt_logits
    eos = _eos_ids(model, eos_token_ids)
    penalty = _repetition_penalty(model, repetition_penalty)
    generated: list[torch.Tensor] = []
    decode_stats: list[list[Any]] = []
    for step in range(max_new_tokens):
        history = prompt_input_ids.to(logits.device)
        if generated:
            history = torch.cat([history, *[item.to(logits.device) for item in generated]], dim=1)
        token = _apply_repetition_penalty(logits, history, penalty).argmax(dim=-1)
        generated.append(token[:, None])
        if eos and int(token[0]) in eos:
            break
        logits, stats = _decode_logits(
            model,
            token,
            meta=meta,
            cache=cache,
            generated_step=step,
        )
        decode_stats.append(stats)
    ids = torch.cat(generated, dim=1) if generated else torch.empty((1, 0), dtype=torch.long)
    return FourActionGenerationOutput(ids, cache, decode_stats)


@torch.inference_mode()
def score_token_ids_from_local_forward(
    model,
    output: LocalFourActionForwardOutput,
    token_ids: torch.Tensor,
) -> FourActionTokenScore:
    """Length-normalized teacher-forced score under a local intervention."""
    source_cache = output.prefill.cache
    if source_cache is None:
        raise ValueError("teacher-forced scoring requires a cached local forward")
    return score_token_ids_from_cached_prompt(
        model,
        output.prompt_logits,
        output.prefill.inputs,
        source_cache,
        token_ids,
    )


@torch.inference_mode()
def score_token_ids_from_cached_prompt(
    model,
    prompt_logits: torch.Tensor,
    meta: BinaryInputs,
    prompt_cache: BinaryRouteCache,
    token_ids: torch.Tensor,
) -> FourActionTokenScore:
    """Teacher-force token IDs from a cloned heterogeneous per-layer cache."""
    ids = token_ids.reshape(-1).to(prompt_logits.device)
    if ids.numel() < 1:
        raise ValueError("answer token_ids must be nonempty")
    cache = clone_four_action_cache(prompt_cache)
    logits = prompt_logits
    logprobs: list[float] = []
    for position, token in enumerate(ids):
        value = F.log_softmax(logits.float(), dim=-1)[0, token]
        logprobs.append(float(value.item()))
        if position + 1 < ids.numel():
            logits, _ = _decode_logits(
                model,
                token.view(1),
                meta=meta,
                cache=cache,
                generated_step=position,
            )
    sequence = float(sum(logprobs))
    return FourActionTokenScore(
        token_ids=[int(value) for value in ids.detach().cpu().tolist()],
        token_logprobs=logprobs,
        sequence_logprob=sequence,
        mean_logprob=sequence / len(logprobs),
    )
