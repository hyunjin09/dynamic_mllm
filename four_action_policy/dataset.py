"""Datasets and collators for complete four-action route supervision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from .actions import encode_action_route


class FourActionManifestDataset(Dataset):
    """Load checksum-frozen inputs with one or more complete valid routes."""

    def __init__(self, manifest_path: str | Path, split: str | None = None) -> None:
        with Path(manifest_path).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        seen_uids: set[str] = set()
        group_splits: dict[str, str] = {}
        for row in rows:
            uid = str(row.get("uid") or "")
            group = str(row.get("split_group") or "")
            row_split = str(row.get("split") or "")
            if not uid or not group or row_split not in {"train", "validation"}:
                raise ValueError(
                    "every four-action manifest row requires uid, split_group, and train/validation split"
                )
            if uid in seen_uids:
                raise ValueError(f"four-action manifest contains duplicate uid {uid!r}")
            seen_uids.add(uid)
            previous = group_splits.setdefault(group, row_split)
            if previous != row_split:
                raise ValueError(
                    f"split-group leakage: {group!r} occurs in both {previous!r} and {row_split!r}"
                )
        for row in rows:
            _validated_routes(row)
        self.rows = [row for row in rows if split is None or row["split"] == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _validated_routes(row: dict[str, Any]) -> list[dict[str, Any]]:
    routes = row.get("valid_routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError(f"sample {row.get('uid')!r} has no valid four-action routes")
    width: int | None = None
    seen: set[tuple[int, ...]] = set()
    for route in routes:
        actions = route.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"sample {row.get('uid')!r} contains a malformed action route")
        encoded = tuple(encode_action_route(actions, expected_layers=None).tolist())
        if width is None:
            width = len(encoded)
        elif len(encoded) != width:
            raise ValueError(f"sample {row.get('uid')!r} contains routes with inconsistent lengths")
        if encoded in seen:
            raise ValueError(f"sample {row.get('uid')!r} contains a duplicate valid route")
        seen.add(encoded)
    return routes


def route_weights(routes: list[dict[str, Any]], weighting: str) -> list[float]:
    if not routes:
        raise ValueError("routes cannot be empty")
    if weighting == "equal":
        raw = [1.0] * len(routes)
    elif weighting == "manifest":
        raw = [float(route.get("weight", 0.0)) for route in routes]
    elif weighting == "polar_full_downweight_0.3":
        encoded = [tuple(encode_action_route(route["actions"], expected_layers=None).tolist()) for route in routes]
        all_full = (3,) * len(encoded[0])
        has_cheaper = all_full in encoded and any(route != all_full for route in encoded)
        raw = [0.3 if has_cheaper and route == all_full else 1.0 for route in encoded]
    else:
        raise ValueError(
            "route weighting must be equal, manifest, or polar_full_downweight_0.3"
        )
    if any(weight <= 0 for weight in raw):
        raise ValueError("route weights must be positive")
    total = sum(raw)
    return [weight / total for weight in raw]


def _tokenize(tokenizer, rows: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    return tokenizer(
        [row["question"] for row in rows],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def make_set_collator(
    tokenizer,
    *,
    max_length: int = 512,
    route_weighting: str = "equal",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = _tokenize(tokenizer, rows, max_length)
        route_rows = [_validated_routes(row) for row in rows]
        max_routes = max(len(routes) for routes in route_rows)
        num_layers = len(route_rows[0][0]["actions"])
        routes_tensor = torch.zeros(
            len(rows), max_routes, num_layers, dtype=torch.long
        )
        valid = torch.zeros(len(rows), max_routes, dtype=torch.bool)
        weights = torch.zeros(len(rows), max_routes, dtype=torch.float32)
        for batch_index, routes in enumerate(route_rows):
            selected_weights = route_weights(routes, route_weighting)
            for route_index, (route, weight) in enumerate(zip(routes, selected_weights)):
                encoded_route = encode_action_route(
                    route["actions"], expected_layers=num_layers
                )
                routes_tensor[batch_index, route_index] = encoded_route
                valid[batch_index, route_index] = True
                weights[batch_index, route_index] = weight
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "valid_routes": routes_tensor,
            "valid_mask": valid,
            "route_weights": weights,
            "uids": [str(row["uid"]) for row in rows],
            "benchmarks": [str(row.get("benchmark", row["dataset"])) for row in rows],
            "route_types": [str(row.get("route_type") or "") for row in rows],
            "unique_examples": len(rows),
        }

    return collate


def make_duplicated_action_collator(
    tokenizer,
    *,
    max_length: int = 512,
    route_weighting: str = "equal",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        targets = []
        weights = []
        sample_indices = []
        route_uids = []
        num_layers: int | None = None
        for sample_index, row in enumerate(rows):
            routes = _validated_routes(row)
            selected_weights = route_weights(routes, route_weighting)
            for route, weight in zip(routes, selected_weights):
                if num_layers is None:
                    num_layers = len(route["actions"])
                targets.append(
                    encode_action_route(route["actions"], expected_layers=num_layers)
                )
                weights.append(weight)
                sample_indices.append(sample_index)
                route_uids.append(str(row["uid"]))
        encoded = _tokenize(tokenizer, rows, max_length)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "target_actions": torch.stack(targets),
            "route_weights": torch.tensor(weights, dtype=torch.float32),
            "route_sample_index": torch.tensor(sample_indices, dtype=torch.long),
            "uids": route_uids,
            "unique_uids": [str(row["uid"]) for row in rows],
            "benchmarks": [str(row.get("benchmark", row["dataset"])) for row in rows],
            "route_types": [str(row.get("route_type") or "") for row in rows],
            "unique_examples": len(rows),
        }

    return collate
