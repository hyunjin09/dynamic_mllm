"""PyTorch datasets and collators for compact binary-policy manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from .structured import mask_to_p12_targets


class BinaryPolicyManifestDataset(Dataset):
    """Load only samples with at least one MCTS-valid binary route."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        *,
        max_valid_routes: int | None = None,
    ) -> None:
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        seen_uids: set[str] = set()
        group_splits: dict[str, str] = {}
        for row in rows:
            uid = str(row.get("uid") or "")
            group = str(row.get("split_group") or "")
            row_split = str(row.get("split") or "")
            if not uid or not group or not row_split:
                raise ValueError("every predictor-manifest row requires uid, split_group, and split")
            if uid in seen_uids:
                raise ValueError(f"predictor manifest contains duplicate uid {uid!r}")
            seen_uids.add(uid)
            previous = group_splits.setdefault(group, row_split)
            if previous != row_split:
                raise ValueError(
                    f"split-group leakage: {group!r} occurs in both {previous!r} and {row_split!r}"
                )
        self.rows = [row for row in rows if row.get("valid_routes") and (split is None or row["split"] == split)]
        if max_valid_routes is not None:
            over_cap = [row["uid"] for row in self.rows if len(row["valid_routes"]) > max_valid_routes]
            if over_cap:
                raise ValueError(
                    f"derived manifest exceeds frozen {max_valid_routes}-route cap for {len(over_cap)} samples; "
                    "the dataset loader will not silently subsample"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _validated_routes(row: dict[str, Any]) -> list[dict[str, Any]]:
    routes = row.get("valid_routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError(f"sample {row.get('uid')!r} has no valid routes")
    keys: set[tuple[int, ...]] = set()
    width = None
    for route in routes:
        mask = route.get("mask")
        if not isinstance(mask, list) or not mask or any(value not in (0, 1) for value in mask):
            raise ValueError(f"sample {row.get('uid')!r} contains a malformed binary mask")
        current = tuple(int(value) for value in mask)
        if width is None:
            width = len(current)
        elif len(current) != width:
            raise ValueError(f"sample {row.get('uid')!r} contains masks with inconsistent lengths")
        if current in keys:
            raise ValueError(f"sample {row.get('uid')!r} contains a duplicate valid mask")
        keys.add(current)
    return routes


def route_weights(routes: list[dict[str, Any]], route_weighting: str) -> list[float]:
    if route_weighting == "equal":
        return [1.0 / len(routes)] * len(routes)
    if route_weighting == "manifest":
        raw = [float(route.get("weight", 0.0)) for route in routes]
    elif route_weighting == "polar_full_downweight_0.3":
        masks = [tuple(int(value) for value in route["mask"]) for route in routes]
        all_on = (1,) * len(masks[0])
        has_cheaper = all_on in masks and any(sum(mask) < len(mask) for mask in masks)
        raw = [0.3 if has_cheaper and mask == all_on else 1.0 for mask in masks]
    else:
        raise ValueError(
            "route_weighting must be 'equal', 'manifest', or 'polar_full_downweight_0.3'"
        )
    if any(weight <= 0 for weight in raw):
        raise ValueError("route weights must be positive")
    total = sum(raw)
    return [weight / total for weight in raw]


def make_set_collator(
    tokenizer,
    *,
    max_length: int = 512,
    route_weighting: str = "equal",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Create a collator for the set-likelihood objective."""

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["question"] for row in rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        route_rows = [_validated_routes(row) for row in rows]
        max_routes = max(len(routes) for routes in route_rows)
        num_layers = len(route_rows[0][0]["mask"])
        masks = torch.zeros(len(rows), max_routes, num_layers, dtype=torch.float32)
        valid = torch.zeros(len(rows), max_routes, dtype=torch.bool)
        weights = torch.zeros(len(rows), max_routes, dtype=torch.float32)
        for batch_idx, routes in enumerate(route_rows):
            if len(routes[0]["mask"]) != num_layers:
                raise ValueError("all batch masks must have the same layer count")
            selected_weights = route_weights(routes, route_weighting)
            for route_idx, (route, weight) in enumerate(zip(routes, selected_weights)):
                masks[batch_idx, route_idx] = torch.tensor(route["mask"], dtype=torch.float32)
                weights[batch_idx, route_idx] = weight
                valid[batch_idx, route_idx] = True
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "valid_masks": masks,
            "valid_mask": valid,
            "route_weights": weights,
            "uids": [row["uid"] for row in rows],
        }

    return collate


def make_structured_set_collator(
    tokenizer,
    *,
    max_length: int = 512,
    route_weighting: str = "equal",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Create the grouped P12 canonical structured-route collator."""

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["question"] for row in rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        route_rows = [_validated_routes(row) for row in rows]
        max_routes = max(len(routes) for routes in route_rows)
        num_layers = len(route_rows[0][0]["mask"])
        masks = torch.zeros(len(rows), max_routes, num_layers, dtype=torch.float32)
        boundaries = torch.zeros_like(masks)
        operations = torch.full(
            (len(rows), max_routes, num_layers), -100, dtype=torch.long
        )
        # Give padded slots a well-formed canonical placeholder; valid_mask
        # excludes their probability mass exactly.
        boundaries[:, :, 0] = 1
        operations[:, :, 0] = 0
        valid = torch.zeros(len(rows), max_routes, dtype=torch.bool)
        weights = torch.zeros(len(rows), max_routes, dtype=torch.float32)
        for batch_index, routes in enumerate(route_rows):
            selected_weights = route_weights(routes, route_weighting)
            for route_index, (route, weight) in enumerate(zip(routes, selected_weights)):
                mask = [int(value) for value in route["mask"]]
                if len(mask) != num_layers:
                    raise ValueError("all batch masks must have the same layer count")
                route_boundaries, route_operations = mask_to_p12_targets(mask)
                masks[batch_index, route_index] = torch.tensor(mask, dtype=torch.float32)
                boundaries[batch_index, route_index] = torch.tensor(
                    route_boundaries, dtype=torch.float32
                )
                operations[batch_index, route_index] = torch.tensor(
                    route_operations, dtype=torch.long
                )
                valid[batch_index, route_index] = True
                weights[batch_index, route_index] = weight
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "valid_masks": masks,
            "boundary_targets": boundaries,
            "operation_targets": operations,
            "valid_mask": valid,
            "route_weights": weights,
            "uids": [row["uid"] for row in rows],
        }

    return collate


def make_duplicated_path_collator(
    tokenizer,
    *,
    max_length: int = 512,
    route_weighting: str = "equal",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Create POLAR-style duplicated ``(input, valid route)`` training rows.

    The DataLoader still batches a fixed number of unique inputs. Expanding
    routes inside the collator makes every selected route a separate predictor
    row while normalized route weights keep every original input's total loss
    weight equal to one.
    """

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        targets: list[list[int]] = []
        weights: list[float] = []
        uids: list[str] = []
        route_sample_index: list[int] = []
        num_layers = None
        for sample_index, row in enumerate(rows):
            routes = _validated_routes(row)
            selected_weights = route_weights(routes, route_weighting)
            for route, weight in zip(routes, selected_weights):
                mask = [int(value) for value in route["mask"]]
                if num_layers is None:
                    num_layers = len(mask)
                elif len(mask) != num_layers:
                    raise ValueError("all batch masks must have the same layer count")
                targets.append(mask)
                weights.append(weight)
                uids.append(row["uid"])
                route_sample_index.append(sample_index)
        encoded = tokenizer(
            [row["question"] for row in rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "targets": torch.tensor(targets, dtype=torch.float32),
            "sample_weights": torch.tensor(weights, dtype=torch.float32),
            "route_sample_index": torch.tensor(route_sample_index, dtype=torch.long),
            "uids": uids,
            "unique_examples": len(rows),
        }

    return collate
