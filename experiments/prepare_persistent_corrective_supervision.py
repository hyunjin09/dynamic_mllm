#!/usr/bin/env python3
"""Freeze matched subsets and machine-local manifests before training outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_binary_polar import file_sha256
from four_action_online_router.data import mandatory_boundary_record


DATASETS = ("gqa", "chartqa", "textvqa")
NONFULL_ACTIONS = ("IGNORE", "READ_ONLY", "WRITE_ONLY")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _stable_key(seed: int, purpose: str, uid: str) -> str:
    return sha256(f"{seed}:{purpose}:{uid}".encode()).hexdigest()


def _dataset_quotas(total: int) -> dict[str, int]:
    if total < len(DATASETS):
        raise ValueError("subset total must cover every training dataset")
    quotient, remainder = divmod(total, len(DATASETS))
    return {
        dataset: quotient + (position < remainder)
        for position, dataset in enumerate(DATASETS)
    }


def _boundary_category(row: dict[str, Any]) -> str:
    valid = tuple(str(value) for value in row["valid_nonfull_actions"])
    if not valid or "FULL" in valid or any(value not in NONFULL_ACTIONS for value in valid):
        raise ValueError(f"invalid mandatory-boundary action set for {row.get('uid')}")
    singleton = bool(row["singleton"])
    if singleton != (len(valid) == 1):
        raise ValueError(f"boundary singleton metadata mismatch for {row.get('uid')}")
    return valid[0] if singleton else "MULTI"


def _diverse_boundary_order(
    rows: Sequence[dict[str, Any]], *, seed: int, purpose: str
) -> list[dict[str, Any]]:
    pools: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        layer = int(row["boundary_layer"])
        if not 0 <= layer < 28:
            raise ValueError(f"boundary layer outside decoder for {row.get('uid')}")
        pools[(_boundary_category(row), min(3, (4 * layer) // 28))].append(row)
    for key, pool in pools.items():
        pool.sort(
            key=lambda row: _stable_key(seed, f"{purpose}:{key}", str(row["uid"]))
        )
    category_order = ("IGNORE", "READ_ONLY", "WRITE_ONLY", "MULTI")
    ordered = []
    next_depth = {category: 0 for category in category_order}
    while any(pools.values()):
        for category in category_order:
            for offset in range(4):
                depth = (next_depth[category] + offset) % 4
                pool = pools[(category, depth)]
                if pool:
                    ordered.append(pool.pop(0))
                    next_depth[category] = (depth + 1) % 4
                    break
    return ordered


def select_matched_subset(
    manifest_rows: Sequence[dict[str, Any]],
    boundary_rows: Sequence[dict[str, Any]],
    *,
    train_per_type: int,
    validation_per_type: int,
    seed: int,
) -> dict[str, Any]:
    """Select fixed dataset-balanced IDs without consulting model outcomes."""

    manifest_by_uid = {str(row.get("uid") or ""): row for row in manifest_rows}
    boundary_by_uid = {str(row.get("uid") or ""): row for row in boundary_rows}
    if "" in manifest_by_uid or len(manifest_by_uid) != len(manifest_rows):
        raise ValueError("manifest UIDs must be nonempty and unique")
    if "" in boundary_by_uid or len(boundary_by_uid) != len(boundary_rows):
        raise ValueError("boundary UIDs must be nonempty and unique")

    group_splits: dict[str, str] = {}
    for row in manifest_rows:
        split = str(row.get("split"))
        group = str(row.get("split_group") or "")
        if split not in {"train", "validation"} or not group:
            raise ValueError("manifest rows require split and split_group")
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise ValueError(f"source manifest has split-group leakage for {group}")

    selected_by_cell: dict[str, list[str]] = {}
    for split, total in (("train", train_per_type), ("validation", validation_per_type)):
        quotas = _dataset_quotas(total)
        for route_type in ("W2C", "C2C"):
            for dataset in DATASETS:
                candidates = [
                    row
                    for row in manifest_rows
                    if row["split"] == split
                    and row["route_type"] == route_type
                    and row["dataset"] == dataset
                ]
                if route_type == "W2C":
                    missing = [row["uid"] for row in candidates if row["uid"] not in boundary_by_uid]
                    if missing:
                        raise ValueError(f"W2C candidates lack boundaries: {missing[0]}")
                    ordered_boundaries = _diverse_boundary_order(
                        [boundary_by_uid[row["uid"]] for row in candidates],
                        seed=seed,
                        purpose=f"{split}:{dataset}",
                    )
                    ordered = [manifest_by_uid[row["uid"]] for row in ordered_boundaries]
                else:
                    ordered = sorted(
                        candidates,
                        key=lambda row: _stable_key(
                            seed, f"{split}:{dataset}:C2C", str(row["uid"])
                        ),
                    )
                requested = quotas[dataset]
                if len(ordered) < requested:
                    raise ValueError(
                        f"insufficient {split}/{dataset}/{route_type} candidates: "
                        f"{len(ordered)} < {requested}"
                    )
                selected_by_cell[f"{split}:{route_type}:{dataset}"] = [
                    str(row["uid"]) for row in ordered[:requested]
                ]

    selected_uids = [
        uid
        for split in ("train", "validation")
        for route_type in ("W2C", "C2C")
        for dataset in DATASETS
        for uid in selected_by_cell[f"{split}:{route_type}:{dataset}"]
    ]
    if len(selected_uids) != len(set(selected_uids)):
        raise RuntimeError("matched subset selection produced duplicate UIDs")
    return {
        "selection_seed": int(seed),
        "train_per_route_type": int(train_per_type),
        "validation_per_route_type": int(validation_per_type),
        "selected_uids": selected_uids,
        "selected_uids_by_cell": selected_by_cell,
    }


def _rebase_path(value: str, external_dataset_root: Path) -> str:
    marker = "Qwen2.5VL/"
    if marker not in value:
        raise ValueError(f"cannot rebase path outside the transferred Qwen2.5VL tree: {value}")
    return str(external_dataset_root / value.split(marker, 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frozen-train-boundaries", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--external-dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--train-per-type", type=int, default=512)
    parser.add_argument("--validation-per-type", type=int, default=128)
    args = parser.parse_args()

    source_manifest_path = Path(args.manifest)
    frozen_boundary_path = Path(args.frozen_train_boundaries)
    feature_manifest_path = Path(args.feature_manifest)
    rows = load_jsonl(source_manifest_path)
    frozen_train_boundaries = {
        row["uid"]: row for row in load_jsonl(frozen_boundary_path)
    }
    boundaries = [
        mandatory_boundary_record(row, num_layers=28)
        for row in rows
        if row["route_type"] == "W2C"
    ]
    computed_by_uid = {row["uid"]: row for row in boundaries}
    if set(frozen_train_boundaries) != {
        row["uid"] for row in rows if row["split"] == "train" and row["route_type"] == "W2C"
    }:
        raise RuntimeError("phase-38 frozen boundary population differs from train W2C")
    for uid, frozen in frozen_train_boundaries.items():
        computed = computed_by_uid[uid]
        for field in (
            "dataset",
            "boundary_layer",
            "all_full_prefix_length",
            "all_full_prefix",
            "valid_nonfull_actions",
            "boundary_route_indices",
            "boundary_route_keys",
            "source_binary_route_ids",
            "singleton",
        ):
            if computed[field] != frozen[field]:
                raise RuntimeError(f"recomputed boundary differs from phase 38 for {uid}: {field}")

    subset = select_matched_subset(
        rows,
        boundaries,
        train_per_type=args.train_per_type,
        validation_per_type=args.validation_per_type,
        seed=args.seed,
    )
    selected = set(subset["selected_uids"])
    external_root = Path(args.external_dataset_root)
    selected_rows = []
    for row in rows:
        if row["uid"] not in selected:
            continue
        updated = dict(row)
        updated["image_path"] = _rebase_path(str(row["image_path"]), external_root)
        if not Path(updated["image_path"]).is_file():
            raise FileNotFoundError(f"selected image is missing: {updated['image_path']}")
        selected_rows.append(updated)
    selected_rows.sort(key=lambda row: subset["selected_uids"].index(row["uid"]))
    selected_boundaries = [row for row in boundaries if row["uid"] in selected]
    selected_boundaries.sort(key=lambda row: subset["selected_uids"].index(row["uid"]))

    feature_by_uid = {row["uid"]: row for row in load_jsonl(feature_manifest_path)}
    selected_features = [feature_by_uid[uid] for uid in subset["selected_uids"]]
    for row in selected_features:
        tensor = Path(row["path"])
        if not tensor.is_file():
            raise FileNotFoundError(f"selected cached visual tensor is missing: {tensor}")

    output_dir = Path(args.output_dir)
    training_manifest = output_dir / "training_manifest.jsonl"
    boundary_manifest = output_dir / "boundary_manifest.jsonl"
    selected_feature_manifest = output_dir / "visual_feature_manifest.jsonl"
    write_jsonl(training_manifest, selected_rows)
    write_jsonl(boundary_manifest, selected_boundaries)
    write_jsonl(selected_feature_manifest, selected_features)

    selected_groups = [str(row["split_group"]) for row in selected_rows]
    selected_group_splits: dict[str, set[str]] = defaultdict(set)
    for row in selected_rows:
        selected_group_splits[str(row["split_group"])].add(str(row["split"]))
    if any(len(splits) != 1 for splits in selected_group_splits.values()):
        raise RuntimeError("selected subset violates image-group split integrity")
    subset.update(
        {
            "schema_version": "persistent_corrective_subset_v1",
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": file_sha256(source_manifest_path),
            "phase38_train_boundary_manifest": str(frozen_boundary_path),
            "phase38_train_boundary_manifest_sha256": file_sha256(frozen_boundary_path),
            "training_manifest": str(training_manifest),
            "training_manifest_sha256": file_sha256(training_manifest),
            "boundary_manifest": str(boundary_manifest),
            "boundary_manifest_sha256": file_sha256(boundary_manifest),
            "visual_feature_manifest": str(selected_feature_manifest),
            "visual_feature_manifest_sha256": file_sha256(selected_feature_manifest),
            "records": len(selected_rows),
            "unique_image_groups": len(set(selected_groups)),
            "valid_routes": sum(len(row["valid_routes"]) for row in selected_rows),
        }
    )
    write_json(output_dir / "subset_manifest.json", subset)

    counts = Counter(
        (row["split"], row["route_type"], row["dataset"]) for row in selected_rows
    )
    boundary_counts = Counter(
        action
        for row in selected_boundaries
        for action in row["valid_nonfull_actions"]
    )
    depth_counts = Counter(int(row["boundary_layer"]) for row in selected_boundaries)
    audit = {
        "schema_version": "persistent_corrective_subset_audit_v1",
        "passed": True,
        "records": len(selected_rows),
        "counts_by_split_route_type_dataset": {
            "|".join(key): value for key, value in sorted(counts.items())
        },
        "boundary_records": len(selected_boundaries),
        "boundary_action_membership_counts": dict(boundary_counts),
        "singleton_boundaries": sum(bool(row["singleton"]) for row in selected_boundaries),
        "multi_valid_boundaries": sum(not bool(row["singleton"]) for row in selected_boundaries),
        "boundary_layer_counts": {str(key): value for key, value in sorted(depth_counts.items())},
        "full_valid_at_boundary": sum(
            "FULL" in row["valid_nonfull_actions"] for row in selected_boundaries
        ),
        "missing_images": 0,
        "missing_visual_tensors": 0,
        "group_split_leakage": 0,
        "c2c_rows_modified": 0,
    }
    write_json(output_dir / "subset_audit.json", audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
