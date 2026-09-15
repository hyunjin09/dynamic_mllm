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
    instruction_valid_mask: torch.Tensor | None = None


def _as_qwen_vl_model(qwen_model):
    if hasattr(qwen_model, "visual"):
        return qwen_model
    return qwen_model.model if hasattr(qwen_model, "model") and hasattr(qwen_model.model, "visual") else qwen_model


def _position_id_owner(qwen_model):
    candidates = [qwen_model]
    inner = getattr(qwen_model, "model", None)
    if inner is not None:
        candidates.append(inner)
    for candidate in candidates:
        if hasattr(candidate, "compute_3d_position_ids") or hasattr(candidate, "get_rope_index"):
            return candidate
    return None


def _visual_module(qwen_model) -> torch.nn.Module:
    if hasattr(qwen_model, "visual"):
        return qwen_model.visual
    if hasattr(qwen_model, "model") and hasattr(qwen_model.model, "visual"):
        return qwen_model.model.visual
    raise AttributeError(f"{type(qwen_model).__name__} does not expose a Qwen-VL visual tower")


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _device_of_input_embeddings(qwen_model) -> torch.device:
    return _module_device(qwen_model.get_input_embeddings())


def _device_of_visual(qwen_model) -> torch.device:
    return _module_device(_visual_module(qwen_model))


def _to_device(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    return None if value is None else value.to(device)


def _feature_outputs_to_embeds(outputs) -> torch.Tensor:
    if hasattr(outputs, "pooler_output"):
        value = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state"):
        value = outputs.last_hidden_state
    else:
        value = outputs
    if isinstance(value, (list, tuple)):
        return torch.cat(value, dim=0)
    return value


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
        if hasattr(qwen_model, "get_image_features"):
            try:
                image_outputs = qwen_model.get_image_features(
                    pixel_values.to(visual_device),
                    image_grid_thw,
                    return_dict=True,
                )
            except TypeError as exc:
                if "return_dict" not in str(exc):
                    raise
                image_outputs = qwen_model.get_image_features(pixel_values.to(visual_device), image_grid_thw)
            image_embeds = _feature_outputs_to_embeds(image_outputs)
            if hasattr(qwen_model, "get_placeholder_mask"):
                image_mask, _ = qwen_model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
                )
            else:
                image_mask = (input_ids == qwen_model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        else:
            visual = _visual_module(qwen_model)
            image_embeds = visual(pixel_values.to(visual_device).type(visual.dtype), grid_thw=image_grid_thw)
            image_mask = (input_ids == qwen_model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_grid_thw = _to_device(video_grid_thw, visual_device)
        if hasattr(qwen_model, "get_video_features"):
            try:
                video_outputs = qwen_model.get_video_features(
                    pixel_values_videos.to(visual_device),
                    video_grid_thw,
                    return_dict=True,
                )
            except TypeError as exc:
                if "return_dict" not in str(exc):
                    raise
                video_outputs = qwen_model.get_video_features(pixel_values_videos.to(visual_device), video_grid_thw)
            video_embeds = _feature_outputs_to_embeds(video_outputs)
            if hasattr(qwen_model, "get_placeholder_mask"):
                _, video_mask = qwen_model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    video_features=video_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
                )
            else:
                video_mask = (input_ids == qwen_model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        else:
            visual = _visual_module(qwen_model)
            video_embeds = visual(pixel_values_videos.to(visual_device).type(visual.dtype), grid_thw=video_grid_thw)
            video_mask = (input_ids == qwen_model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
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
    instruction_token_mask: torch.Tensor | None = None,
    **kwargs,
) -> DVRInputs:
    original_qwen_model = qwen_model
    qwen_model = _as_qwen_vl_model(qwen_model)
    embed_device = _device_of_input_embeddings(qwen_model)
    rope_deltas = None

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
        position_owner = _position_id_owner(original_qwen_model) or _position_id_owner(qwen_model)
        if position_owner is not None and hasattr(position_owner, "compute_3d_position_ids"):
            position_ids_full = position_owner.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=_to_device(image_grid_thw, embed_device),
                video_grid_thw=_to_device(video_grid_thw, embed_device),
                second_per_grid_ts=_to_device(second_per_grid_ts, embed_device),
                inputs_embeds=full_inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )
        elif position_owner is not None and hasattr(position_owner, "get_rope_index"):
            position_ids_full, rope_deltas = position_owner.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=_to_device(image_grid_thw, embed_device),
                video_grid_thw=_to_device(video_grid_thw, embed_device),
                second_per_grid_ts=_to_device(second_per_grid_ts, embed_device),
                attention_mask=attention_mask,
            )
            position_owner.rope_deltas = rope_deltas
        else:
            raise AttributeError(f"{type(qwen_model).__name__} cannot compute Qwen-VL position ids")
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
    instruction_valid_mask = _align_instruction_mask(
        instruction_token_mask,
        text_mask=text_mask,
        text_valid_mask=text_valid_mask,
        text_orig_idx=text_orig_idx,
        device=embed_device,
    )

    current_rope_deltas = rope_deltas
    if current_rope_deltas is None:
        for candidate in (position_owner if position_ids is None else None, original_qwen_model, qwen_model):
            if candidate is None:
                continue
            current_rope_deltas = getattr(candidate, "rope_deltas", None)
            if current_rope_deltas is not None:
                break
    rope_deltas = None if current_rope_deltas is None else current_rope_deltas.to(embed_device).clone()

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
        instruction_valid_mask=instruction_valid_mask,
    )


def _align_instruction_mask(
    instruction_token_mask: torch.Tensor | None,
    *,
    text_mask: torch.Tensor,
    text_valid_mask: torch.Tensor,
    text_orig_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    if instruction_token_mask is None:
        return None
    full_mask = instruction_token_mask.to(device=device, dtype=torch.bool)
    if tuple(full_mask.shape) != tuple(text_mask.shape):
        raise ValueError(
            "instruction_token_mask must match input_ids shape; "
            f"got {tuple(full_mask.shape)} versus {tuple(text_mask.shape)}"
        )
    if bool((full_mask & ~text_mask).any().item()):
        raise ValueError("instruction_token_mask must select valid text tokens only")
    aligned = torch.zeros_like(text_valid_mask, dtype=torch.bool)
    for batch_idx in range(text_valid_mask.shape[0]):
        valid = text_valid_mask[batch_idx]
        original_indices = text_orig_idx[batch_idx, valid]
        aligned[batch_idx, valid] = full_mask[batch_idx, original_indices]
        if not bool(aligned[batch_idx].any().item()):
            raise ValueError(f"instruction_token_mask selects no text tokens for batch item {batch_idx}")
    return aligned


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
