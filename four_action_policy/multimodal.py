"""Visual-feature batching for Image+Question four-action training."""

from __future__ import annotations

from typing import Any, Callable

import torch

from .dataset import make_duplicated_action_collator, make_set_collator


def attach_visual_features(
    batch: dict[str, Any],
    rows: list[dict[str, Any]],
    feature_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    features = []
    loaded_by_path: dict[str, torch.Tensor] = {}
    for row in rows:
        uid = str(row["uid"])
        record = feature_index.get(uid)
        if record is None:
            raise KeyError(f"visual feature index has no entry for {uid!r}")
        tensor_path = str(record["path"])
        tensor = loaded_by_path.get(tensor_path)
        if tensor is None:
            tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
            loaded_by_path[tensor_path] = tensor
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            raise ValueError(f"visual feature for {uid!r} is not [V,D]")
        if list(tensor.shape) != list(record["shape"]):
            raise ValueError(f"visual feature shape mismatch for {uid!r}")
        if tensor.shape[0] < 1 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"visual feature for {uid!r} is empty or nonfinite")
        features.append(tensor)
    maximum = max(tensor.shape[0] for tensor in features)
    width = features[0].shape[1]
    if any(tensor.shape[1] != width for tensor in features):
        raise ValueError("visual feature widths differ within a batch")
    padded = torch.zeros(len(features), maximum, width, dtype=features[0].dtype)
    valid = torch.zeros(len(features), maximum, dtype=torch.bool)
    for index, tensor in enumerate(features):
        padded[index, : tensor.shape[0]] = tensor
        valid[index, : tensor.shape[0]] = True
    batch["image_features"] = padded
    batch["image_attention_mask"] = valid
    return batch


def _with_visual_features(
    base_collator: Callable[[list[dict[str, Any]]], dict[str, Any]],
    feature_index: dict[str, dict[str, Any]],
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return attach_visual_features(base_collator(rows), rows, feature_index)

    return collate


def make_multimodal_set_collator(
    tokenizer,
    feature_index: dict[str, dict[str, Any]],
    *,
    max_length: int = 512,
    route_weighting: str = "equal",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    return _with_visual_features(
        make_set_collator(
            tokenizer, max_length=max_length, route_weighting=route_weighting
        ),
        feature_index,
    )


def make_multimodal_duplicated_action_collator(
    tokenizer,
    feature_index: dict[str, dict[str, Any]],
    *,
    max_length: int = 512,
    route_weighting: str = "equal",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    return _with_visual_features(
        make_duplicated_action_collator(
            tokenizer, max_length=max_length, route_weighting=route_weighting
        ),
        feature_index,
    )
