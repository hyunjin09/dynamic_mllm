from __future__ import annotations

from typing import Any

import torch


def _tokens(tokenizer, ids: list[int]) -> list[str]:
    return [str(token) for token in tokenizer.convert_ids_to_tokens(ids)]


def describe_token_layout(tokenizer, model_config, input_ids: torch.Tensor, position_ids: torch.Tensor) -> dict[str, Any]:
    ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    image_id = int(model_config.image_token_id)
    vision_start_id = int(model_config.vision_start_token_id)
    vision_end_id = int(model_config.vision_end_token_id)
    image_positions = [index for index, token_id in enumerate(ids) if token_id == image_id]
    if not image_positions:
        raise ValueError("No image-token rows were found")
    first_visual, last_visual = image_positions[0], image_positions[-1]
    contiguous = image_positions == list(range(first_visual, last_visual + 1))
    start_positions = [index for index, token_id in enumerate(ids) if token_id == vision_start_id]
    end_positions = [index for index, token_id in enumerate(ids) if token_id == vision_end_id]

    prefix_ids = ids[:first_visual]
    suffix_ids = ids[last_visual + 1 :]
    positions = position_ids[:, 0].detach().cpu()
    return {
        "sequence_length": len(ids),
        "visual_token_count": len(image_positions),
        "visual_rows": {
            "first": first_visual,
            "last": last_visual,
            "contiguous": contiguous,
        },
        "vision_start_positions": start_positions,
        "vision_end_positions": end_positions,
        "text_control_before_visual_count": first_visual,
        "text_control_after_visual_count": len(ids) - last_visual - 1,
        "prefix_token_ids": prefix_ids,
        "prefix_tokens": _tokens(tokenizer, prefix_ids),
        "suffix_token_ids": suffix_ids,
        "suffix_tokens": _tokens(tokenizer, suffix_ids),
        "mrope_position_min": [int(positions[axis].min().item()) for axis in range(3)],
        "mrope_position_max": [int(positions[axis].max().item()) for axis in range(3)],
        "visual_mrope_first": [int(positions[axis, first_visual].item()) for axis in range(3)],
        "visual_mrope_last": [int(positions[axis, last_visual].item()) for axis in range(3)],
        "first_postvisual_mrope": [
            int(positions[axis, min(last_visual + 1, len(ids) - 1)].item()) for axis in range(3)
        ],
    }
