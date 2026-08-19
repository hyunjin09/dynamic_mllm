"""Qwen2.5-VL input decomposition with 4.51/5.x API compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


TEXT = 0
IMAGE = 1
VIDEO = 2


@dataclass
class BinaryInputs:
    full_inputs_embeds: torch.Tensor
    text_states: torch.Tensor
    visual_states: torch.Tensor
    text_indices: torch.LongTensor
    visual_indices: torch.LongTensor
    text_valid_mask: torch.BoolTensor
    visual_valid_mask: torch.BoolTensor
    full_position_ids: torch.LongTensor
    text_position_ids: torch.LongTensor
    visual_position_ids: torch.LongTensor
    full_attention_mask: torch.Tensor
    full_prompt_len: torch.LongTensor
    rope_deltas: torch.Tensor | None


def resolve_causal_lm(model):
    if getattr(model, "_is_binary_qwen_wrapper", False):
        return model.base_model
    # Hugging Face PreTrainedModel exposes a ``base_model`` property as well;
    # a native conditional LM must not be unwrapped past its LM head.
    if hasattr(model, "lm_head") and hasattr(model, "model"):
        return model
    return model


def resolve_decoder(model):
    causal_lm = resolve_causal_lm(model)
    root = causal_lm.model
    return root.language_model if hasattr(root, "language_model") else root


def infer_mm_token_type_ids(causal_lm, input_ids: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(input_ids, dtype=torch.int32)
    output[input_ids == int(causal_lm.config.image_token_id)] = IMAGE
    video_id = getattr(causal_lm.config, "video_token_id", None)
    if video_id is not None:
        output[input_ids == int(video_id)] = VIDEO
    return output


def _split_padded(sequence: torch.Tensor, keep: torch.Tensor):
    counts = keep.long().sum(dim=1)
    max_count = int(counts.max().item())
    output = sequence.new_zeros(sequence.shape[0], max_count, *sequence.shape[2:])
    valid = torch.zeros(sequence.shape[0], max_count, dtype=torch.bool, device=sequence.device)
    indices = torch.full((sequence.shape[0], max_count), -1, dtype=torch.long, device=sequence.device)
    for batch_idx in range(sequence.shape[0]):
        current = torch.nonzero(keep[batch_idx], as_tuple=False).flatten()
        output[batch_idx, : current.numel()] = sequence[batch_idx, current]
        valid[batch_idx, : current.numel()] = True
        indices[batch_idx, : current.numel()] = current
    return output, valid, indices


def _split_positions(position_ids: torch.Tensor, keep: torch.Tensor):
    values, valid, indices = _split_padded(position_ids.permute(1, 2, 0), keep)
    return values.permute(2, 0, 1), valid, indices


def _full_embeddings(causal_lm, input_ids: torch.Tensor, inputs: dict[str, Any]) -> torch.Tensor:
    decoder = resolve_decoder(causal_lm)
    embed = decoder.get_input_embeddings() if hasattr(decoder, "get_input_embeddings") else decoder.embed_tokens
    device = next(embed.parameters()).device
    ids = input_ids.to(device)
    embeddings = embed(ids)
    pixels = inputs.get("pixel_values")
    if pixels is None:
        return embeddings
    root = causal_lm.model
    visual = root.visual
    visual_device = next(visual.parameters()).device
    pixels = pixels.to(device=visual_device, dtype=visual.dtype)
    grid = inputs["image_grid_thw"].to(visual_device)
    if hasattr(root, "get_image_features"):
        result = root.get_image_features(pixels, grid, return_dict=True)
        image_features = torch.cat(result.pooler_output, dim=0)
    else:
        image_features = visual(pixels, grid_thw=grid)
    image_features = image_features.to(device=embeddings.device, dtype=embeddings.dtype)
    mask = ids == int(causal_lm.config.image_token_id)
    if int(mask.sum().item()) != int(image_features.shape[0]):
        raise ValueError("image placeholder count does not match encoded visual rows")
    return embeddings.masked_scatter(mask.unsqueeze(-1).expand_as(embeddings), image_features)


def _position_ids(causal_lm, input_ids, attention_mask, full_embeddings, inputs):
    supplied = inputs.get("position_ids")
    if supplied is not None:
        return supplied.to(full_embeddings.device), inputs.get("rope_deltas")
    if hasattr(causal_lm, "get_rope_index"):
        return causal_lm.get_rope_index(
            input_ids=input_ids.to(full_embeddings.device),
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            attention_mask=attention_mask.to(full_embeddings.device),
        )
    root = causal_lm.model
    result = root.compute_3d_position_ids(
        input_ids=input_ids.to(full_embeddings.device),
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        second_per_grid_ts=inputs.get("second_per_grid_ts"),
        inputs_embeds=full_embeddings,
        attention_mask=attention_mask.to(full_embeddings.device),
        past_key_values=None,
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    if isinstance(result, tuple):
        return result
    return result, getattr(root, "rope_deltas", None)


def build_binary_inputs(model, inputs: dict[str, Any]) -> BinaryInputs:
    causal_lm = resolve_causal_lm(model)
    input_ids = inputs["input_ids"]
    attention = inputs.get("attention_mask", torch.ones_like(input_ids))
    token_types = inputs.get("mm_token_type_ids")
    if token_types is None:
        token_types = infer_mm_token_type_ids(causal_lm, input_ids)
    embeddings = _full_embeddings(causal_lm, input_ids, inputs)
    position_ids, rope_deltas = _position_ids(causal_lm, input_ids, attention, embeddings, inputs)
    attention = attention.to(embeddings.device)
    token_types = token_types.to(embeddings.device)
    valid = attention.bool()
    visual_mask = valid & ((token_types == IMAGE) | (token_types == VIDEO))
    text_mask = valid & ~visual_mask
    text_states, text_valid, text_indices = _split_padded(embeddings, text_mask)
    visual_states, visual_valid, visual_indices = _split_padded(embeddings, visual_mask)
    text_positions, _, _ = _split_positions(position_ids, text_mask)
    visual_positions, _, _ = _split_positions(position_ids, visual_mask)
    return BinaryInputs(
        full_inputs_embeds=embeddings,
        text_states=text_states,
        visual_states=visual_states,
        text_indices=text_indices,
        visual_indices=visual_indices,
        text_valid_mask=text_valid,
        visual_valid_mask=visual_valid,
        full_position_ids=position_ids,
        text_position_ids=text_positions,
        visual_position_ids=visual_positions,
        full_attention_mask=attention,
        full_prompt_len=attention.long().sum(dim=1),
        rope_deltas=rope_deltas,
    )


def scatter_streams(text_states: torch.Tensor, visual_states: torch.Tensor, meta: BinaryInputs) -> torch.Tensor:
    full = text_states.new_zeros(meta.full_attention_mask.shape[0], meta.full_attention_mask.shape[1], text_states.shape[-1])
    visual_states = visual_states.to(text_states.device)
    for batch_idx in range(full.shape[0]):
        tv = meta.text_valid_mask[batch_idx].to(text_states.device)
        vv = meta.visual_valid_mask[batch_idx].to(text_states.device)
        full[batch_idx, meta.text_indices[batch_idx].to(text_states.device)[tv]] = text_states[batch_idx, tv]
        full[batch_idx, meta.visual_indices[batch_idx].to(text_states.device)[vv]] = visual_states[batch_idx, vv]
    return full


def split_streams(full: torch.Tensor, meta: BinaryInputs) -> tuple[torch.Tensor, torch.Tensor]:
    text = full.new_zeros(full.shape[0], meta.text_indices.shape[1], full.shape[-1])
    visual = full.new_zeros(full.shape[0], meta.visual_indices.shape[1], full.shape[-1])
    for batch_idx in range(full.shape[0]):
        tv = meta.text_valid_mask[batch_idx].to(full.device)
        vv = meta.visual_valid_mask[batch_idx].to(full.device)
        text[batch_idx, tv] = full[batch_idx, meta.text_indices[batch_idx].to(full.device)[tv]]
        visual[batch_idx, vv] = full[batch_idx, meta.visual_indices[batch_idx].to(full.device)[vv]]
    return text, visual
