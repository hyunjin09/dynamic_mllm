"""Qwen2.5-VL input decomposition for DVR."""

from __future__ import annotations

from dataclasses import dataclass

import torch


TEXT_TOKEN_TYPE = 0
IMAGE_TOKEN_TYPE = 1
VIDEO_TOKEN_TYPE = 2


@dataclass
class DVRInputs:
    text_hidden: torch.Tensor
    visual_memory: torch.Tensor
    text_position_ids: torch.Tensor
    visual_position_ids: torch.Tensor
    text_orig_idx: torch.Tensor
    visual_orig_idx: torch.Tensor
    text_valid_mask: torch.Tensor
    visual_valid_mask: torch.Tensor
    full_attention_mask: torch.Tensor
    full_prompt_len: torch.Tensor
    rope_deltas: torch.Tensor | None


def _as_qwen_vl_model(qwen_model):
    return qwen_model.model if hasattr(qwen_model, "model") and hasattr(qwen_model.model, "visual") else qwen_model


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _device_of_input_embeddings(qwen_model) -> torch.device:
    return _module_device(qwen_model.get_input_embeddings())


def _device_of_visual(qwen_model) -> torch.device:
    return _module_device(qwen_model.visual)


def _to_device(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    return None if value is None else value.to(device)


def build_qwen_full_inputs_embeds(
    qwen_model,
    input_ids: torch.LongTensor,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Build full `inputs_embeds` exactly as Qwen2.5-VL does before its language model."""
    qwen_model = _as_qwen_vl_model(qwen_model)
    embed_device = _device_of_input_embeddings(qwen_model)
    visual_device = _device_of_visual(qwen_model)

    input_ids = input_ids.to(embed_device)
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_grid_thw = _to_device(image_grid_thw, visual_device)
        image_outputs = qwen_model.get_image_features(
            pixel_values.to(visual_device),
            image_grid_thw,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = qwen_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_grid_thw = _to_device(video_grid_thw, visual_device)
        video_outputs = qwen_model.get_video_features(
            pixel_values_videos.to(visual_device),
            video_grid_thw,
            return_dict=True,
        )
        video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = qwen_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    return inputs_embeds


def _split_padded(sequence: torch.Tensor, keep_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather variable-length masked rows into a padded batch-major tensor."""
    batch_size, seq_len = keep_mask.shape
    counts = keep_mask.long().sum(dim=1)
    max_count = int(counts.max().item()) if batch_size > 0 else 0
    out_shape = (batch_size, max_count, *sequence.shape[2:])
    gathered = sequence.new_zeros(out_shape)
    valid_mask = torch.zeros(batch_size, max_count, dtype=torch.bool, device=keep_mask.device)
    orig_idx = torch.full((batch_size, max_count), -1, dtype=torch.long, device=keep_mask.device)

    for batch_idx in range(batch_size):
        idx = torch.nonzero(keep_mask[batch_idx], as_tuple=False).flatten()
        count = idx.numel()
        if count == 0:
            continue
        gathered[batch_idx, :count] = sequence[batch_idx, idx]
        valid_mask[batch_idx, :count] = True
        orig_idx[batch_idx, :count] = idx

    return gathered, valid_mask, orig_idx


def _split_position_ids(
    position_ids_full: torch.Tensor,
    keep_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_first_positions = position_ids_full.permute(1, 2, 0)
    gathered, valid_mask, orig_idx = _split_padded(batch_first_positions, keep_mask)
    return gathered.permute(2, 0, 1), valid_mask, orig_idx


def build_dvr_inputs(
    qwen_model,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    mm_token_type_ids: torch.IntTensor,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    **kwargs,
) -> DVRInputs:
    qwen_model = _as_qwen_vl_model(qwen_model)
    embed_device = _device_of_input_embeddings(qwen_model)

    input_ids = input_ids.to(embed_device)
    attention_mask = attention_mask.to(embed_device)
    mm_token_type_ids = mm_token_type_ids.to(embed_device)

    full_inputs_embeds = build_qwen_full_inputs_embeds(
        qwen_model,
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )

    if position_ids is None:
        position_ids_full = qwen_model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=_to_device(image_grid_thw, embed_device),
            video_grid_thw=_to_device(video_grid_thw, embed_device),
            second_per_grid_ts=_to_device(second_per_grid_ts, embed_device),
            inputs_embeds=full_inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )
    else:
        position_ids_full = position_ids.to(embed_device)

    if position_ids_full is None:
        raise ValueError("Qwen2.5-VL did not produce multimodal position ids")

    valid_mask = attention_mask.bool()
    text_mask = valid_mask & (mm_token_type_ids == TEXT_TOKEN_TYPE)
    visual_mask = valid_mask & (
        (mm_token_type_ids == IMAGE_TOKEN_TYPE) | (mm_token_type_ids == VIDEO_TOKEN_TYPE)
    )

    text_hidden, text_valid_mask, text_orig_idx = _split_padded(full_inputs_embeds, text_mask)
    visual_memory, visual_valid_mask, visual_orig_idx = _split_padded(full_inputs_embeds, visual_mask)
    text_position_ids, _, _ = _split_position_ids(position_ids_full, text_mask)
    visual_position_ids, _, _ = _split_position_ids(position_ids_full, visual_mask)

    rope_deltas = None if qwen_model.rope_deltas is None else qwen_model.rope_deltas.to(embed_device).clone()

    return DVRInputs(
        text_hidden=text_hidden,
        visual_memory=visual_memory,
        text_position_ids=text_position_ids,
        visual_position_ids=visual_position_ids,
        text_orig_idx=text_orig_idx,
        visual_orig_idx=visual_orig_idx,
        text_valid_mask=text_valid_mask,
        visual_valid_mask=visual_valid_mask,
        full_attention_mask=attention_mask,
        full_prompt_len=attention_mask.long().sum(dim=1),
        rope_deltas=rope_deltas,
    )


def reconstruct_full_inputs_embeds(dvr_inputs: DVRInputs) -> torch.Tensor:
    batch_size, seq_len = dvr_inputs.full_attention_mask.shape
    hidden_size = dvr_inputs.text_hidden.shape[-1]
    full = dvr_inputs.text_hidden.new_zeros(batch_size, seq_len, hidden_size)

    for batch_idx in range(batch_size):
        text_valid = dvr_inputs.text_valid_mask[batch_idx]
        visual_valid = dvr_inputs.visual_valid_mask[batch_idx]
        full[batch_idx, dvr_inputs.text_orig_idx[batch_idx, text_valid]] = dvr_inputs.text_hidden[
            batch_idx, text_valid
        ]
        full[batch_idx, dvr_inputs.visual_orig_idx[batch_idx, visual_valid]] = dvr_inputs.visual_memory[
            batch_idx, visual_valid
        ]

    return full


def reconstruct_full_position_ids(dvr_inputs: DVRInputs) -> torch.Tensor:
    batch_size, seq_len = dvr_inputs.full_attention_mask.shape
    full = dvr_inputs.text_position_ids.new_zeros(3, batch_size, seq_len)

    for batch_idx in range(batch_size):
        text_valid = dvr_inputs.text_valid_mask[batch_idx]
        visual_valid = dvr_inputs.visual_valid_mask[batch_idx]
        full[:, batch_idx, dvr_inputs.text_orig_idx[batch_idx, text_valid]] = dvr_inputs.text_position_ids[
            :, batch_idx, text_valid
        ]
        full[:, batch_idx, dvr_inputs.visual_orig_idx[batch_idx, visual_valid]] = dvr_inputs.visual_position_ids[
            :, batch_idx, visual_valid
        ]

    return full
