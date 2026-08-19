"""P13 matched question/image input utilities."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import torch

from .dataset import make_duplicated_path_collator, make_set_collator


MODALITIES = ("question", "image", "image_question")


def resolve_modality_inputs(
    modality: str,
    token_features: torch.Tensor,
    token_attention_mask: torch.Tensor,
    image_features: torch.Tensor,
    image_attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Expose only the frozen P13 modality while retaining one architecture."""
    if modality not in MODALITIES:
        raise ValueError(f"unknown P13 modality {modality!r}")
    if modality == "question":
        return token_features, token_attention_mask, None, None
    if modality == "image":
        empty_tokens = token_features.new_zeros(token_features.shape[0], 1, token_features.shape[-1])
        empty_mask = token_attention_mask.new_zeros(token_features.shape[0], 1)
        return empty_tokens, empty_mask, image_features, image_attention_mask
    return token_features, token_attention_mask, image_features, image_attention_mask


def _with_visual_features(
    base_collator: Callable[[list[dict[str, Any]]], dict[str, Any]],
    feature_index: dict[str, dict[str, Any]],
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Attach padded native visual rows to a unique-input route batch."""

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = base_collator(rows)
        features = []
        for row in rows:
            feature_uid = row.get("feature_uid", row["uid"])
            record = feature_index.get(feature_uid)
            if record is None:
                raise KeyError(f"P13 feature index has no entry for {feature_uid!r}")
            tensor = torch.load(record["path"], map_location="cpu", weights_only=True)
            if not torch.is_tensor(tensor) or tensor.ndim != 2:
                raise ValueError(f"P13 feature for {row['uid']!r} is not [V,D]")
            if list(tensor.shape) != list(record["shape"]):
                raise ValueError(f"P13 feature shape mismatch for {row['uid']!r}")
            if tensor.shape[0] < 1 or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"P13 feature for {row['uid']!r} is empty or nonfinite")
            features.append(tensor)
        maximum = max(tensor.shape[0] for tensor in features)
        width = features[0].shape[1]
        if any(tensor.shape[1] != width for tensor in features):
            raise ValueError("P13 visual feature widths differ within a batch")
        padded = torch.zeros(len(features), maximum, width, dtype=features[0].dtype)
        valid = torch.zeros(len(features), maximum, dtype=torch.bool)
        for index, tensor in enumerate(features):
            padded[index, : tensor.shape[0]] = tensor
            valid[index, : tensor.shape[0]] = True
        batch["image_features"] = padded
        batch["image_attention_mask"] = valid
        return batch

    return collate


def make_multimodal_set_collator(
    tokenizer,
    feature_index: dict[str, dict[str, Any]],
    *,
    max_length: int = 512,
    route_weighting: str = "equal",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Pad native projected visual rows and the unchanged P11 valid sets."""
    return _with_visual_features(
        make_set_collator(
            tokenizer, max_length=max_length, route_weighting=route_weighting
        ),
        feature_index,
    )


def make_multimodal_duplicated_path_collator(
    tokenizer,
    feature_index: dict[str, dict[str, Any]],
    *,
    max_length: int = 512,
    route_weighting: str = "equal",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Attach native visual rows to POLAR-style duplicated-route batches."""
    return _with_visual_features(
        make_duplicated_path_collator(
            tokenizer, max_length=max_length, route_weighting=route_weighting
        ),
        feature_index,
    )


def deterministic_modality_permutations(
    rows: list[dict[str, Any]], *, seed: int
) -> dict[str, dict[str, str]]:
    """Freeze within-dataset derangements for P13's four-way diagnostic."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row["benchmark"])].append(str(row["uid"]))
    output: dict[str, dict[str, str]] = {}
    purposes = ("question_uid", "image_uid", "both_question_uid", "both_image_uid")
    for benchmark, uids in grouped.items():
        if len(uids) < 2:
            raise ValueError(f"dataset {benchmark!r} needs at least two rows for shuffling")
        donor_orders = {
            purpose: sorted(
                uids,
                key=lambda uid: sha256(f"{seed}:{benchmark}:{purpose}:{uid}".encode()).hexdigest(),
            )
            for purpose in purposes
        }
        for purpose, donor_order in donor_orders.items():
            donor = donor_order[1:] + donor_order[:1]
            for target, source in zip(donor_order, donor):
                output.setdefault(target, {})[purpose] = source
    return output


def deterministic_group_disjoint_modality_permutations(
    rows: list[dict[str, Any]], *, seed: int
) -> dict[str, dict[str, str]]:
    """Freeze within-dataset shuffles that also exclude the target image group."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        uid = str(row["uid"])
        benchmark = str(row["benchmark"])
        group = str(row.get("split_group") or "")
        if not group:
            raise ValueError(f"sample {uid!r} lacks split_group")
        grouped[benchmark].append({"uid": uid, "group": group})
    output: dict[str, dict[str, str]] = {}
    purposes = ("question_uid", "image_uid", "both_question_uid", "both_image_uid")
    for benchmark, records in grouped.items():
        if len({record["group"] for record in records}) < 2:
            raise ValueError(f"dataset {benchmark!r} needs at least two image groups")
        for purpose in purposes:
            ordered = sorted(
                records,
                key=lambda record: sha256(
                    f"{seed}:{benchmark}:{purpose}:{record['uid']}".encode()
                ).hexdigest(),
            )
            donors = None
            for shift in range(1, len(ordered)):
                candidate = ordered[shift:] + ordered[:shift]
                if all(
                    target["uid"] != donor["uid"] and target["group"] != donor["group"]
                    for target, donor in zip(ordered, candidate)
                ):
                    donors = candidate
                    break
            if donors is None:
                raise RuntimeError(
                    f"could not construct a group-disjoint {purpose} permutation for {benchmark}"
                )
            for target, donor in zip(ordered, donors):
                output.setdefault(target["uid"], {})[purpose] = donor["uid"]
    return output
