"""Visual-feature batching for Image+Question four-action training."""

from __future__ import annotations

from typing import Any, Callable

import torch

from .actions import FOUR_ACTIONS
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


def make_persistent_boundary_collator(
    base_collator: Callable[[list[dict[str, Any]]], dict[str, Any]],
    boundary_by_uid: dict[str, dict[str, Any]],
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Attach the same set-valued mandatory-boundary target to either substrate."""

    action_index = {action: index for index, action in enumerate(FOUR_ACTIONS)}

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = base_collator(rows)
        layers = torch.full((len(rows),), -1, dtype=torch.long)
        valid = torch.zeros(len(rows), len(FOUR_ACTIONS), dtype=torch.bool)
        present = torch.zeros(len(rows), dtype=torch.bool)
        for index, row in enumerate(rows):
            boundary = boundary_by_uid.get(str(row["uid"]))
            if row.get("route_type") == "W2C":
                if boundary is None:
                    raise KeyError(f"W2C row lacks a mandatory boundary: {row['uid']}")
                layers[index] = int(boundary["boundary_layer"])
                actions = [str(value) for value in boundary["valid_nonfull_actions"]]
                if not actions or "FULL" in actions:
                    raise ValueError("mandatory boundary must be nonempty and exclude FULL")
                for action in actions:
                    valid[index, action_index[action]] = True
                present[index] = True
            elif boundary is not None:
                raise ValueError(f"C2C row unexpectedly has a boundary: {row['uid']}")
        batch["boundary_layers"] = layers
        batch["boundary_valid_actions"] = valid
        batch["boundary_present"] = present
        return batch

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
