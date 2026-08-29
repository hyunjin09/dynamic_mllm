"""Checksum-bound manifest and source metadata helpers."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from experiments.train_binary_polar import file_sha256
from four_action_policy.actions import FOUR_ACTIONS, encode_action_route
from .supervision import PrefixTrie


SOURCE_FIELDS = (
    "uid",
    "answer",
    "all_answer_norms",
    "metric_name",
    "correctness_threshold",
    "max_new_tokens",
    "image_content_sha256",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_verified_manifest(path: str | Path, expected_sha256: str) -> list[dict[str, Any]]:
    current = Path(path)
    if file_sha256(current) != expected_sha256:
        raise RuntimeError("online-router training manifest checksum mismatch")
    rows = load_jsonl(current)
    uids = [str(row.get("uid") or "") for row in rows]
    if any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise RuntimeError("online-router manifest contains empty or duplicate UIDs")
    return rows


def load_source_metadata(
    path: str | Path, expected_sha256: str, required_uids: set[str]
) -> dict[str, dict[str, Any]]:
    current = Path(path)
    if file_sha256(current) != expected_sha256:
        raise RuntimeError("four-action source manifest checksum mismatch")
    output = {}
    with current.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            uid = str(row.get("uid") or "")
            if uid not in required_uids:
                continue
            if any(field not in row for field in SOURCE_FIELDS):
                raise RuntimeError(f"source evaluator metadata is incomplete for {uid}")
            output[uid] = {field: row[field] for field in SOURCE_FIELDS}
    if set(output) != required_uids:
        raise RuntimeError("source evaluator metadata does not cover the training manifest")
    return output


def manifest_route_tensor(row: dict[str, Any], *, num_layers: int) -> torch.LongTensor:
    routes = row.get("valid_routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError(f"sample {row.get('uid')!r} has no valid routes")
    encoded = [
        encode_action_route(route["actions"], expected_layers=num_layers) for route in routes
    ]
    result = torch.stack(encoded)
    if torch.unique(result, dim=0).shape[0] != result.shape[0]:
        raise ValueError(f"sample {row.get('uid')!r} has duplicate valid routes")
    return result


def manifest_trie(row: dict[str, Any], *, num_layers: int) -> PrefixTrie:
    return PrefixTrie(manifest_route_tensor(row, num_layers=num_layers).tolist())


def mandatory_boundary_record(
    row: dict[str, Any], *, num_layers: int
) -> dict[str, Any]:
    """Describe the latest valid all-FULL prefix for one W2C sample."""

    if row.get("route_type") != "W2C":
        raise ValueError("mandatory boundaries are defined only for W2C samples")
    routes = row.get("valid_routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError(f"sample {row.get('uid')!r} has no valid routes")
    encoded = manifest_route_tensor(row, num_layers=num_layers)
    full_index = FOUR_ACTIONS.index("FULL")
    all_full = torch.full((num_layers,), full_index, dtype=torch.long)
    if bool((encoded == all_full).all(dim=1).any().item()):
        raise ValueError(f"W2C sample {row.get('uid')!r} contains an all-FULL route")

    leading_full = []
    for route in encoded:
        count = 0
        for action in route.tolist():
            if int(action) != full_index:
                break
            count += 1
        leading_full.append(count)
    boundary_layer = max(leading_full)
    if boundary_layer >= num_layers:
        raise ValueError("mandatory boundary must occur inside the decoder")
    route_indices = [
        index for index, prefix_length in enumerate(leading_full)
        if prefix_length == boundary_layer
    ]
    actions = sorted(
        {int(encoded[index, boundary_layer].item()) for index in route_indices}
    )
    if full_index in actions or not actions:
        raise RuntimeError("mandatory boundary must contain only non-FULL actions")

    route_keys = []
    source_ids = []
    for index in route_indices:
        route = routes[index]
        route_keys.append(str(route.get("route_key") or f"route_index_{index}"))
        source_ids.extend(str(value) for value in route.get("source_binary_route_ids", []))
    return {
        "uid": str(row["uid"]),
        "dataset": str(row["dataset"]),
        "boundary_layer": boundary_layer,
        "all_full_prefix_length": boundary_layer,
        "all_full_prefix": ["FULL"] * boundary_layer,
        "valid_nonfull_actions": [FOUR_ACTIONS[index] for index in actions],
        "boundary_route_indices": route_indices,
        "boundary_route_keys": route_keys,
        "source_binary_route_ids": list(dict.fromkeys(source_ids)),
        "singleton": len(actions) == 1,
    }


def boundary_teacher_route(
    row: dict[str, Any], boundary: dict[str, Any], *, num_layers: int
) -> torch.LongTensor:
    """Resolve and verify the frozen route that reaches a mandatory boundary."""

    if str(row.get("uid")) != str(boundary.get("uid")):
        raise ValueError("boundary record UID does not match its manifest row")
    routes = manifest_route_tensor(row, num_layers=num_layers)
    route_index = int(boundary["teacher_route_index"])
    eligible = {int(value) for value in boundary["boundary_route_indices"]}
    if route_index not in eligible or not 0 <= route_index < len(routes):
        raise ValueError("boundary teacher route is not an eligible route index")
    route = routes[route_index]
    layer = int(boundary["boundary_layer"])
    full_index = FOUR_ACTIONS.index("FULL")
    if layer < 0 or layer >= num_layers:
        raise ValueError("boundary layer lies outside the decoder")
    if layer and not bool((route[:layer] == full_index).all().item()):
        raise ValueError("boundary teacher route does not follow the exact all-FULL prefix")
    action = FOUR_ACTIONS[int(route[layer].item())]
    if action not in boundary["valid_nonfull_actions"] or action == "FULL":
        raise ValueError("boundary teacher route does not take a valid non-FULL action")
    return route


def _stable_uid_key(seed: int, purpose: str, uid: str) -> str:
    return sha256(f"{seed}:{purpose}:{uid}".encode()).hexdigest()


def _depth_bin(layer: int, num_layers: int = 28) -> int:
    return min(3, (4 * int(layer)) // int(num_layers))


def _round_robin_depth_order(
    rows: Sequence[dict[str, Any]], *, seed: int, purpose: str
) -> list[dict[str, Any]]:
    pools: dict[int, list[dict[str, Any]]] = {index: [] for index in range(4)}
    for row in rows:
        pools[_depth_bin(int(row["boundary_layer"]))].append(row)
    for depth, pool in pools.items():
        pool.sort(key=lambda row: _stable_uid_key(seed, f"{purpose}:{depth}", row["uid"]))
    ordered = []
    while any(pools.values()):
        for depth in range(4):
            if pools[depth]:
                ordered.append(pools[depth].pop(0))
    return ordered


def select_boundary_pilot(
    manifest_rows: Sequence[dict[str, Any]],
    boundary_rows: Sequence[dict[str, Any]],
    *,
    w2c_per_dataset: int,
    c2c_per_dataset: int,
    seed: int,
    num_layers: int,
) -> dict[str, Any]:
    """Select deterministic dataset/action/depth-diverse W2C and C2C cases."""

    if w2c_per_dataset < 1 or c2c_per_dataset < 1:
        raise ValueError("pilot counts per dataset must be positive")
    datasets = ("gqa", "chartqa", "textvqa")
    manifest_by_uid = {str(row["uid"]): row for row in manifest_rows}
    if len(manifest_by_uid) != len(manifest_rows):
        raise ValueError("pilot source manifest contains duplicate UIDs")
    boundary_by_uid = {str(row["uid"]): row for row in boundary_rows}
    if len(boundary_by_uid) != len(boundary_rows):
        raise ValueError("boundary manifest contains duplicate UIDs")

    selected_w2c: list[str] = []
    selected_c2c: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for dataset in datasets:
        candidates = [row for row in boundary_rows if row["dataset"] == dataset]
        categories: dict[str, list[dict[str, Any]]] = {
            "IGNORE": [], "READ_ONLY": [], "WRITE_ONLY": [], "MULTI": []
        }
        for row in candidates:
            category = (
                str(row["valid_nonfull_actions"][0]) if row["singleton"] else "MULTI"
            )
            categories[category].append(row)
        ordered_by_category = {
            category: _round_robin_depth_order(
                pool, seed=seed, purpose=f"w2c:{dataset}:{category}"
            )
            for category, pool in categories.items()
        }
        chosen: list[dict[str, Any]] = []
        category_order = ("IGNORE", "READ_ONLY", "WRITE_ONLY", "MULTI")
        while len(chosen) < w2c_per_dataset and any(ordered_by_category.values()):
            for category in category_order:
                if len(chosen) >= w2c_per_dataset:
                    break
                pool = ordered_by_category[category]
                if pool:
                    chosen.append(pool.pop(0))
        if len(chosen) != w2c_per_dataset:
            raise ValueError(f"insufficient W2C pilot candidates for {dataset}")
        selected_w2c.extend(str(row["uid"]) for row in chosen)

        c2c_candidates = []
        for row in manifest_rows:
            if row.get("dataset") != dataset or row.get("route_type") != "C2C":
                continue
            routes = manifest_route_tensor(row, num_layers=num_layers)
            full = torch.full((num_layers,), FOUR_ACTIONS.index("FULL"), dtype=torch.long)
            exact_full = (routes == full).all(dim=1)
            if bool(exact_full.any().item()) and bool((~exact_full).any().item()):
                c2c_candidates.append(row)
        c2c_candidates.sort(
            key=lambda row: _stable_uid_key(seed, f"c2c:{dataset}", str(row["uid"]))
        )
        if len(c2c_candidates) < c2c_per_dataset:
            raise ValueError(f"insufficient route-compatible C2C pilot candidates for {dataset}")
        selected_c2c.extend(str(row["uid"]) for row in c2c_candidates[:c2c_per_dataset])
        counts[dataset] = {"W2C": len(chosen), "C2C": c2c_per_dataset}

    selected = selected_w2c + selected_c2c
    if len(selected) != len(set(selected)) or any(uid not in manifest_by_uid for uid in selected):
        raise RuntimeError("pilot selection produced duplicate or unknown UIDs")
    return {
        "selection_seed": int(seed),
        "w2c_per_dataset": int(w2c_per_dataset),
        "c2c_per_dataset": int(c2c_per_dataset),
        "w2c_uids": selected_w2c,
        "c2c_uids": selected_c2c,
        "counts_by_dataset": counts,
    }


def choose_smoke_indices(
    rows: Sequence[dict[str, Any]], *, records: int, seed: int
) -> list[int]:
    if records < 6 or records > len(rows):
        raise ValueError("smoke records must cover the six dataset/type cells")

    def key(index: int, purpose: str) -> str:
        return sha256(f"{seed}:{purpose}:{rows[index]['uid']}".encode()).hexdigest()

    selected = []
    for route_type in ("W2C", "C2C"):
        for dataset in ("gqa", "chartqa", "textvqa"):
            candidates = [
                index
                for index, row in enumerate(rows)
                if row.get("route_type") == route_type and row.get("dataset") == dataset
            ]
            if not candidates:
                raise ValueError(f"smoke population lacks {route_type}/{dataset}")
            selected.append(min(candidates, key=lambda index: key(index, "cell")))
    remaining = [index for index in range(len(rows)) if index not in set(selected)]
    remaining.sort(key=lambda index: key(index, "fill"))
    selected.extend(remaining[: records - len(selected)])
    return selected
