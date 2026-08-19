"""Layer-local VISUAL_ON and VISUAL_OFF execution."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import torch

from .cache import BinaryRouteCache
from .inputs import BinaryInputs, resolve_decoder, scatter_streams, split_streams
from .masks import additive_causal_mask, full_causal_mask


@dataclass(frozen=True)
class LayerExecution:
    layer_index: int
    visual_on: bool
    residual_rows: int
    cache_rows: int | None


def _device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _position_embeddings(decoder, hidden: torch.Tensor, position_ids: torch.Tensor):
    return decoder.rotary_emb(hidden, position_ids.to(hidden.device))


def call_decoder_layer(
    layer,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    position_embeddings,
    cache: BinaryRouteCache | None,
    use_cache: bool,
    cache_position: torch.Tensor | None = None,
):
    """Call both Transformers 4.51 (singular) and 5.x (plural) layer APIs."""
    parameters = inspect.signature(layer.forward).parameters
    kwargs = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "position_embeddings": position_embeddings,
        "use_cache": use_cache,
    }
    if cache_position is not None:
        kwargs["cache_position"] = cache_position
    if "past_key_value" in parameters:
        kwargs["past_key_value"] = cache
    elif "past_key_values" in parameters:
        kwargs["past_key_values"] = cache
    elif cache is not None:
        raise RuntimeError("decoder layer exposes no recognized K/V cache argument")
    return layer(**kwargs)


def visual_on_layer(
    model,
    layer,
    text_states: torch.Tensor,
    visual_states: torch.Tensor,
    meta: BinaryInputs,
    *,
    layer_index: int,
    cache: BinaryRouteCache | None = None,
    use_cache: bool = False,
    native_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, LayerExecution]:
    device = _device(layer)
    full = scatter_streams(text_states.to(device), visual_states.to(device), meta)
    # With one unpadded ALL-ON sequence, the native Qwen text model dispatches
    # SDPA with ``attention_mask=None`` and ``is_causal=True``. Materializing an
    # equivalent additive mask selects a different BF16 kernel path and breaks
    # the frozen native-logit parity tolerance despite identical semantics.
    mask = None if native_causal else full_causal_mask(
        meta.full_attention_mask,
        dtype=full.dtype,
        device=device,
    )
    cache_position = torch.arange(full.shape[1], device=device) if native_causal else None
    positions = _position_embeddings(resolve_decoder(model), full, meta.full_position_ids)
    output = call_decoder_layer(
        layer,
        full,
        attention_mask=mask,
        position_embeddings=positions,
        cache=cache,
        use_cache=use_cache,
        cache_position=cache_position,
    )[0]
    next_text, next_visual = split_streams(output, meta)
    return next_text, next_visual, LayerExecution(
        layer_index=layer_index,
        visual_on=True,
        residual_rows=int(output.shape[1]),
        cache_rows=None if cache is None else cache.get_seq_length(layer_index),
    )


def visual_off_layer(
    model,
    layer,
    text_states: torch.Tensor,
    visual_states: torch.Tensor,
    meta: BinaryInputs,
    *,
    layer_index: int,
    cache: BinaryRouteCache | None = None,
    use_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, LayerExecution]:
    device = _device(layer)
    text = text_states.to(device)
    carried_visual = visual_states.to(device)
    mask = additive_causal_mask(
        meta.text_valid_mask,
        meta.text_indices,
        dtype=text.dtype,
        device=device,
    )
    positions = _position_embeddings(resolve_decoder(model), text, meta.text_position_ids)
    output = call_decoder_layer(
        layer,
        text,
        attention_mask=mask,
        position_embeddings=positions,
        cache=cache,
        use_cache=use_cache,
    )[0]
    return output, carried_visual, LayerExecution(
        layer_index=layer_index,
        visual_on=False,
        residual_rows=int(output.shape[1]),
        cache_rows=None if cache is None else cache.get_seq_length(layer_index),
    )


def decode_text_layer(
    model,
    layer,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    layer_index: int,
    cache: BinaryRouteCache,
) -> tuple[torch.Tensor, LayerExecution]:
    device = _device(layer)
    hidden = hidden_states.to(device)
    positions = _position_embeddings(resolve_decoder(model), hidden, position_ids)
    output = call_decoder_layer(
        layer,
        hidden,
        attention_mask=None,
        position_embeddings=positions,
        cache=cache,
        use_cache=True,
    )[0]
    return output, LayerExecution(
        layer_index=layer_index,
        visual_on=False,
        residual_rows=1,
        cache_rows=cache.get_seq_length(layer_index),
    )
