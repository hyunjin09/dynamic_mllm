"""Split/scatter helpers for binary contextualized DVR-C."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from dvr_qwen.core.input_builder import (
    DVRInputs,
    build_dvr_inputs,
    reconstruct_full_inputs_embeds,
    reconstruct_full_position_ids,
)


@dataclass
class BinaryDVRCInputs:
    full_inputs_embeds: torch.Tensor
    text_states: torch.Tensor
    visual_states: torch.Tensor
    text_indices: torch.LongTensor
    visual_indices: torch.LongTensor
    text_valid_mask: torch.Tensor
    visual_valid_mask: torch.Tensor
    full_position_ids: torch.LongTensor
    text_position_ids: torch.LongTensor
    visual_position_ids: torch.LongTensor
    full_attention_mask: torch.Tensor
    text_attention_mask: torch.Tensor
    full_prompt_len: torch.Tensor
    rope_deltas: torch.Tensor | None
    dvr_inputs: DVRInputs


@dataclass
class BinaryDVRCDeviceIndices:
    text_valid: list[torch.Tensor]
    text_full_idx: list[torch.Tensor]
    visual_valid: list[torch.Tensor]
    visual_full_idx: list[torch.Tensor]


@dataclass
class BinaryDVRCIndexCache:
    per_device: dict[str, BinaryDVRCDeviceIndices] = field(default_factory=dict)

    def get(self, meta: BinaryDVRCInputs, device: torch.device) -> BinaryDVRCDeviceIndices:
        key = str(device)
        if key not in self.per_device:
            text_valid = []
            text_full_idx = []
            visual_valid = []
            visual_full_idx = []
            for batch_idx in range(meta.full_attention_mask.shape[0]):
                current_text_valid = meta.text_valid_mask[batch_idx].to(device=device)
                current_visual_valid = meta.visual_valid_mask[batch_idx].to(device=device)
                text_valid.append(current_text_valid)
                visual_valid.append(current_visual_valid)
                text_full_idx.append(meta.text_indices[batch_idx].to(device=device)[current_text_valid])
                visual_full_idx.append(meta.visual_indices[batch_idx].to(device=device)[current_visual_valid])
            self.per_device[key] = BinaryDVRCDeviceIndices(
                text_valid=text_valid,
                text_full_idx=text_full_idx,
                visual_valid=visual_valid,
                visual_full_idx=visual_full_idx,
            )
        return self.per_device[key]


def build_binary_dvrc_inputs(qwen_model, **inputs: Any) -> BinaryDVRCInputs:
    """Build Qwen multimodal inputs, then expose binary text/visual streams."""
    dvr_inputs = build_dvr_inputs(qwen_model, **inputs)
    return binary_inputs_from_dvr_inputs(dvr_inputs)


def binary_inputs_from_dvr_inputs(dvr_inputs: DVRInputs) -> BinaryDVRCInputs:
    text_attention_mask = dvr_inputs.text_valid_mask.to(dtype=dvr_inputs.full_attention_mask.dtype)
    return BinaryDVRCInputs(
        full_inputs_embeds=reconstruct_full_inputs_embeds(dvr_inputs),
        text_states=dvr_inputs.text_hidden,
        visual_states=dvr_inputs.visual_memory,
        text_indices=dvr_inputs.text_orig_idx,
        visual_indices=dvr_inputs.visual_orig_idx,
        text_valid_mask=dvr_inputs.text_valid_mask,
        visual_valid_mask=dvr_inputs.visual_valid_mask,
        full_position_ids=reconstruct_full_position_ids(dvr_inputs),
        text_position_ids=dvr_inputs.text_position_ids,
        visual_position_ids=dvr_inputs.visual_position_ids,
        full_attention_mask=dvr_inputs.full_attention_mask,
        text_attention_mask=text_attention_mask,
        full_prompt_len=dvr_inputs.full_prompt_len,
        rope_deltas=dvr_inputs.rope_deltas,
        dvr_inputs=dvr_inputs,
    )


def scatter_to_full(
    text_states: torch.Tensor,
    visual_states: torch.Tensor,
    meta: BinaryDVRCInputs,
    index_cache: BinaryDVRCIndexCache | None = None,
) -> torch.Tensor:
    """Scatter text/control and visual streams back to original sequence order."""
    batch_size, seq_len = meta.full_attention_mask.shape
    hidden_size = text_states.shape[-1]
    full = text_states.new_zeros(batch_size, seq_len, hidden_size)
    visual_states = visual_states.to(text_states.device)
    cached = index_cache.get(meta, text_states.device) if index_cache is not None else None

    for batch_idx in range(batch_size):
        if cached is None:
            text_valid = meta.text_valid_mask[batch_idx].to(text_states.device)
            text_full_idx = meta.text_indices[batch_idx].to(text_states.device)[text_valid]
            visual_valid = meta.visual_valid_mask[batch_idx].to(text_states.device)
            visual_full_idx = meta.visual_indices[batch_idx].to(text_states.device)[visual_valid]
        else:
            text_valid = cached.text_valid[batch_idx]
            text_full_idx = cached.text_full_idx[batch_idx]
            visual_valid = cached.visual_valid[batch_idx]
            visual_full_idx = cached.visual_full_idx[batch_idx]

        full[batch_idx, text_full_idx] = text_states[batch_idx, text_valid]
        full[batch_idx, visual_full_idx] = visual_states[batch_idx, visual_valid]

    return full


def split_from_full(
    hidden_states: torch.Tensor,
    meta: BinaryDVRCInputs,
    index_cache: BinaryDVRCIndexCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather full-sequence hidden states into padded text/control and visual streams."""
    batch_size = hidden_states.shape[0]
    text_states = hidden_states.new_zeros(batch_size, meta.text_indices.shape[1], hidden_states.shape[-1])
    visual_states = hidden_states.new_zeros(batch_size, meta.visual_indices.shape[1], hidden_states.shape[-1])
    cached = index_cache.get(meta, hidden_states.device) if index_cache is not None else None

    for batch_idx in range(batch_size):
        if cached is None:
            text_valid = meta.text_valid_mask[batch_idx].to(hidden_states.device)
            text_full_idx = meta.text_indices[batch_idx].to(hidden_states.device)[text_valid]
            visual_valid = meta.visual_valid_mask[batch_idx].to(hidden_states.device)
            visual_full_idx = meta.visual_indices[batch_idx].to(hidden_states.device)[visual_valid]
        else:
            text_valid = cached.text_valid[batch_idx]
            text_full_idx = cached.text_full_idx[batch_idx]
            visual_valid = cached.visual_valid[batch_idx]
            visual_full_idx = cached.visual_full_idx[batch_idx]

        text_states[batch_idx, text_valid] = hidden_states[batch_idx, text_full_idx]
        visual_states[batch_idx, visual_valid] = hidden_states[batch_idx, visual_full_idx]

    return text_states, visual_states


def split_scatter_max_diff(meta: BinaryDVRCInputs) -> float:
    """Return max reconstruction error over valid full-sequence positions."""
    reconstructed = scatter_to_full(meta.text_states, meta.visual_states, meta)
    valid = meta.full_attention_mask.to(reconstructed.device).bool()
    if not bool(valid.any().item()):
        return 0.0
    diff = (reconstructed - meta.full_inputs_embeds.to(reconstructed.device)).abs()
    return float(diff[valid].max().item())


def assert_split_scatter_identity(meta: BinaryDVRCInputs, atol: float = 0.0) -> None:
    max_diff = split_scatter_max_diff(meta)
    if max_diff > atol:
        raise AssertionError(f"split/scatter reconstruction max diff {max_diff} exceeds {atol}")
