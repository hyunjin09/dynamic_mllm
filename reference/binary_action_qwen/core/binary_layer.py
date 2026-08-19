"""Binary DVR-C layer execution for Qwen2.5-VL."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dvr_qwen.core.cache import DVRCache
from dvr_qwen.core.masks import make_text_causal_mask
from dvr_qwen.core.split_scatter import BinaryDVRCIndexCache, BinaryDVRCInputs, scatter_to_full, split_from_full


@dataclass
class BinaryDVRCLayerStats:
    layer_idx: int
    visual_on: bool
    residual_sequence_length: int
    cache_key_value_length: int | None


@dataclass
class BinaryDVRCStaticInputCache:
    index_cache: BinaryDVRCIndexCache | None = None
    full_masks: dict[tuple[str, torch.dtype], torch.Tensor] = field(default_factory=dict)
    text_masks: dict[tuple[str, torch.dtype], torch.Tensor] = field(default_factory=dict)
    full_position_embeddings: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    text_position_embeddings: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    @staticmethod
    def _key(device: torch.device, dtype: torch.dtype) -> tuple[str, torch.dtype]:
        return str(device), dtype

    def get_full_mask(self, meta: BinaryDVRCInputs, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = self._key(device, dtype)
        if key not in self.full_masks:
            self.full_masks[key] = make_full_causal_mask(meta.full_attention_mask, dtype=dtype, device=device)
        return self.full_masks[key]

    def get_text_mask(self, meta: BinaryDVRCInputs, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = self._key(device, dtype)
        if key not in self.text_masks:
            self.text_masks[key] = _text_causal_mask(meta, dtype=dtype, device=device)
        return self.text_masks[key]

    def get_full_position_embeddings(
        self,
        text_model,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._key(hidden_states.device, hidden_states.dtype)
        if key not in self.full_position_embeddings:
            self.full_position_embeddings[key] = _position_embeddings(text_model, hidden_states, position_ids)
        return self.full_position_embeddings[key]

    def get_text_position_embeddings(
        self,
        text_model,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._key(hidden_states.device, hidden_states.dtype)
        if key not in self.text_position_embeddings:
            self.text_position_embeddings[key] = _position_embeddings(text_model, hidden_states, position_ids)
        return self.text_position_embeddings[key]


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def normalize_visual_on_mask(
    visual_on_mask: torch.Tensor,
    batch_size: int,
    num_layers: int,
    device: torch.device,
) -> torch.Tensor:
    route = visual_on_mask.to(device=device, dtype=torch.bool)
    if route.ndim == 1:
        route = route.unsqueeze(0)
    if route.shape != (batch_size, num_layers):
        raise ValueError(f"visual_on_mask must have shape {(batch_size, num_layers)}, got {tuple(route.shape)}")
    return route


def make_full_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return additive `[B, 1, S, S]` full-sequence causal mask."""
    attention_mask = attention_mask.to(device=device).bool()
    seq_len = attention_mask.shape[1]
    idx = torch.arange(seq_len, device=device)
    allowed = (
        attention_mask[:, :, None]
        & attention_mask[:, None, :]
        & (idx[None, None, :] <= idx[None, :, None])
    )
    mask = torch.zeros((attention_mask.shape[0], 1, seq_len, seq_len), dtype=dtype, device=device)
    return mask.masked_fill(~allowed[:, None, :, :], torch.finfo(dtype).min)


def _text_causal_mask(meta: BinaryDVRCInputs, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return make_text_causal_mask(
        meta.text_indices,
        meta.text_valid_mask,
        dtype=dtype,
        device=device,
    )


def _position_embeddings(text_model, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    return text_model.rotary_emb(hidden_states, position_ids.to(hidden_states.device))


def _cache_length(cache: DVRCache | None, layer_idx: int) -> int | None:
    return None if cache is None else cache.get_seq_length(layer_idx)


def forward_visual_on_layer(
    text_model,
    layer: torch.nn.Module,
    text_states: torch.Tensor,
    visual_states: torch.Tensor,
    meta: BinaryDVRCInputs,
    layer_idx: int,
    cache: DVRCache | None = None,
    use_cache: bool = False,
    static_input_cache: BinaryDVRCStaticInputCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor, BinaryDVRCLayerStats]:
    """Run one original full-sequence Qwen decoder layer and split its output."""
    layer_device = _module_device(layer)
    text_states = text_states.to(layer_device)
    visual_states = visual_states.to(layer_device)
    index_cache = static_input_cache.index_cache if static_input_cache is not None else None
    full_states = scatter_to_full(text_states, visual_states, meta, index_cache=index_cache)
    if static_input_cache is None:
        attention_mask = make_full_causal_mask(
            meta.full_attention_mask,
            dtype=full_states.dtype,
            device=layer_device,
        )
        position_embeddings = _position_embeddings(text_model, full_states, meta.full_position_ids)
    else:
        attention_mask = static_input_cache.get_full_mask(meta, dtype=full_states.dtype, device=layer_device)
        position_embeddings = static_input_cache.get_full_position_embeddings(
            text_model,
            full_states,
            meta.full_position_ids,
        )
    outputs = layer(
        hidden_states=full_states,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        past_key_values=cache,
        use_cache=use_cache,
    )
    next_text, next_visual = split_from_full(outputs[0], meta, index_cache=index_cache)
    return (
        next_text,
        next_visual,
        BinaryDVRCLayerStats(
            layer_idx=layer_idx,
            visual_on=True,
            residual_sequence_length=int(outputs[0].shape[1]),
            cache_key_value_length=_cache_length(cache, layer_idx),
        ),
    )


def forward_text_only_layer(
    text_model,
    layer: torch.nn.Module,
    text_states: torch.Tensor,
    visual_states: torch.Tensor,
    meta: BinaryDVRCInputs,
    layer_idx: int,
    cache: DVRCache | None = None,
    use_cache: bool = False,
    static_input_cache: BinaryDVRCStaticInputCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor, BinaryDVRCLayerStats]:
    """Run one Qwen decoder layer on text/control rows only and carry visual states."""
    layer_device = _module_device(layer)
    text_states = text_states.to(layer_device)
    visual_states = visual_states.to(layer_device)
    if static_input_cache is None:
        attention_mask = _text_causal_mask(meta, dtype=text_states.dtype, device=layer_device)
        position_embeddings = _position_embeddings(text_model, text_states, meta.text_position_ids)
    else:
        attention_mask = static_input_cache.get_text_mask(meta, dtype=text_states.dtype, device=layer_device)
        position_embeddings = static_input_cache.get_text_position_embeddings(
            text_model,
            text_states,
            meta.text_position_ids,
        )
    outputs = layer(
        hidden_states=text_states,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        past_key_values=cache,
        use_cache=use_cache,
    )
    return (
        outputs[0],
        visual_states,
        BinaryDVRCLayerStats(
            layer_idx=layer_idx,
            visual_on=False,
            residual_sequence_length=int(outputs[0].shape[1]),
            cache_key_value_length=_cache_length(cache, layer_idx),
        ),
    )


def forward_decode_text_layer(
    text_model,
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    layer_idx: int,
    cache: DVRCache,
) -> tuple[torch.Tensor, BinaryDVRCLayerStats]:
    """Decode one generated text token through one layer using that layer's cache."""
    layer_device = _module_device(layer)
    hidden_states = hidden_states.to(layer_device)
    position_embeddings = _position_embeddings(text_model, hidden_states, position_ids)
    outputs = layer(
        hidden_states=hidden_states,
        attention_mask=None,
        position_embeddings=position_embeddings,
        past_key_values=cache,
        use_cache=True,
    )
    return (
        outputs[0],
        BinaryDVRCLayerStats(
            layer_idx=layer_idx,
            visual_on=False,
            residual_sequence_length=int(outputs[0].shape[1]),
            cache_key_value_length=_cache_length(cache, layer_idx),
        ),
    )
