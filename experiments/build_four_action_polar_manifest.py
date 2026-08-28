#!/usr/bin/env python3
"""Freeze the completed VQA four-action labels into a training manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from four_action_policy.actions import FOUR_ACTIONS, encode_action_route
from tools.research_analysis.four_action.sequential_label_jobs import (
    file_sha256,
    safe_filename,
)


TRAINING_DATASETS = ("gqa", "chartqa", "textvqa")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _verify_sidecar(path: Path) -> str:
    digest = file_sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    fields = sidecar.read_text(encoding="utf-8").split() if sidecar.is_file() else []
    if not fields or fields[0] != digest:
        raise ValueError(f"record checksum is missing or invalid: {path}")
    return digest


def build_manifest_rows(
    source_rows: Iterable[dict[str, Any]],
    records_root: Path,
    *,
    layer_count: int = 28,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = sorted((dict(row) for row in source_rows), key=lambda row: str(row["uid"]))
    if not sources:
        raise ValueError("four-action POLAR source population is empty")
    uids = [str(row.get("uid") or "") for row in sources]
    if any(not uid for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError("source population contains an empty or duplicate UID")

    group_splits: dict[str, str] = {}
    for source in sources:
        group = str(source.get("image_group_id") or "")
        split = str(source.get("source_split") or "")
        if not group or split not in {"train", "validation"}:
            raise ValueError(f"source {source['uid']!r} lacks a valid group/split")
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise ValueError(
                f"split-group leakage: {group!r} occurs in both {previous!r} and {split!r}"
            )

    manifest = []
    action_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    route_type_counts: Counter[str] = Counter()
    contracts: Counter[str] = Counter()
    included_groups: set[str] = set()
    zero_valid_route_exclusions: list[dict[str, Any]] = []
    route_total = 0
    for source in sources:
        uid = str(source["uid"])
        record_path = records_root / safe_filename(uid)
        if not record_path.is_file():
            raise FileNotFoundError(f"completed four-action record is missing: {record_path}")
        record_sha = _verify_sidecar(record_path)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not record.get("passed") or str(record.get("uid")) != uid:
            raise ValueError(f"four-action record failed or has the wrong UID: {uid}")
        for field in ("dataset", "sample_id", "image_group_id", "source_split"):
            if str(record.get(field)) != str(source.get(field)):
                raise ValueError(f"source/record {field} mismatch for {uid}")
        image_path = Path(str(source.get("image_path") or ""))
        if not image_path.is_file():
            raise FileNotFoundError(f"source image is missing for {uid}: {image_path}")
        source_routes = record.get("unique_valid_four_action_routes")
        if not isinstance(source_routes, list):
            raise ValueError(f"four-action valid-route field is malformed: {uid}")
        if not source_routes:
            positive = int(record.get("source_positive_route_count", -1))
            replay_valid = int(record.get("source_route_replay_valid_count", -1))
            replay_failures = int(record.get("source_route_replay_failure_count", -1))
            if (
                str(record.get("route_type")) != "W2C"
                or positive < 1
                or replay_valid != 0
                or replay_failures != positive
            ):
                raise ValueError(
                    f"zero-valid-route record lacks complete replay-failure accounting: {uid}"
                )
            zero_valid_route_exclusions.append(
                {
                    "uid": uid,
                    "dataset": str(source["dataset"]),
                    "split": str(source["source_split"]),
                    "source_positive_route_count": positive,
                    "source_record_path": str(record_path.resolve()),
                    "source_record_sha256": record_sha,
                    "reason": "all_source_positive_routes_failed_current_unified_replay",
                }
            )
            continue

        valid_routes = []
        seen: set[tuple[int, ...]] = set()
        for route in source_routes:
            actions = [str(action).upper() for action in route.get("four_action_route", [])]
            encoded = tuple(
                encode_action_route(actions, expected_layers=layer_count).tolist()
            )
            if encoded in seen:
                raise ValueError(f"four-action record contains a duplicate valid route: {uid}")
            seen.add(encoded)
            if not bool(route.get("evaluation", {}).get("correct")):
                raise ValueError(f"route is not evaluator-correct for {uid}")
            route_key = "|".join(actions)
            if str(route.get("route_key")) != route_key:
                raise ValueError(f"route key mismatch for {uid}")
            action_counts.update(actions)
            valid_routes.append(
                {
                    "actions": actions,
                    "route_key": route_key,
                    "label_semantics": str(route.get("label_semantics") or ""),
                    "source_binary_route_ids": sorted(
                        str(value) for value in route.get("source_binary_route_ids", [])
                    ),
                }
            )
        route_total += len(valid_routes)
        dataset = str(source["dataset"])
        split = str(source["source_split"])
        route_type = str(record.get("route_type") or "")
        contract = str(
            record.get("execution_contract", {}).get("contract_sha256") or ""
        )
        if not contract:
            raise ValueError(f"four-action record lacks executor contract: {uid}")
        dataset_counts[dataset] += 1
        split_counts[split] += 1
        route_type_counts[route_type] += 1
        contracts[contract] += 1
        included_groups.add(str(source["image_group_id"]))
        manifest.append(
            {
                "schema_version": "four_action_polar_manifest_v1",
                "uid": uid,
                "dataset": dataset,
                "benchmark": str(source.get("benchmark") or dataset),
                "sample_id": str(source["sample_id"]),
                "question": str(source["question"]),
                "prompt": str(source.get("prompt") or source["question"]),
                "image_path": str(image_path.resolve()),
                "image_id": source.get("image_id"),
                "image_group_id": str(source["image_group_id"]),
                "split_group": str(source["image_group_id"]),
                "split": split,
                "route_type": route_type,
                "label_semantics": str(record.get("label_semantics") or ""),
                "valid_routes": valid_routes,
                "valid_route_count": len(valid_routes),
                "source_record_path": str(record_path.resolve()),
                "source_record_sha256": record_sha,
                "executor_contract_sha256": contract,
                "model_revision": str(
                    record.get("execution_contract", {}).get("model_revision") or ""
                ),
            }
        )

    audit = {
        "schema_version": "four_action_polar_manifest_audit_v1",
        "passed": True,
        "source_samples": len(sources),
        "samples": len(manifest),
        "routes": route_total,
        "unique_image_groups": len(included_groups),
        "group_leakage_count": 0,
        "zero_valid_route_exclusions": len(zero_valid_route_exclusions),
        "zero_valid_route_exclusion_uids": sorted(
            row["uid"] for row in zero_valid_route_exclusions
        ),
        "zero_valid_route_exclusion_records": zero_valid_route_exclusions,
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "route_type_counts": dict(sorted(route_type_counts.items())),
        "action_counts": {
            action: int(action_counts[action]) for action in FOUR_ACTIONS
        },
        "executor_contract_counts": dict(sorted(contracts.items())),
        "route_cap": None,
    }
    return manifest, audit


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        default=Path(
            "datasets/mcts_labels_4action/sequential_branching_v1/full/records"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/four_action_polar/preparation_v1"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = [
        row
        for row in read_jsonl(args.source_manifest)
        if str(row.get("dataset")) in TRAINING_DATASETS
    ]
    rows, audit = build_manifest_rows(source, args.records_root, layer_count=28)
    expected = {"gqa": 3333, "chartqa": 1756, "textvqa": 1722}
    if audit["dataset_counts"] != expected or audit["samples"] != 6811:
        raise RuntimeError(f"unexpected VQA training population: {audit['dataset_counts']}")
    if audit["split_counts"] != {"train": 5945, "validation": 866}:
        raise RuntimeError(f"unexpected predictor split: {audit['split_counts']}")
    if audit["source_samples"] != 6917 or audit["zero_valid_route_exclusions"] != 106:
        raise RuntimeError("unexpected zero-valid current-replay exclusion accounting")
    if audit["routes"] != 248804:
        raise RuntimeError(f"unexpected complete-route count: {audit['routes']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest_v1.jsonl"
    audit_path = args.output_dir / "manifest_audit_v1.json"
    if (manifest_path.exists() or audit_path.exists()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    audit.update(
        {
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": file_sha256(args.source_manifest),
            "records_root": str(args.records_root.resolve()),
        }
    )
    if args.resume and manifest_path.exists() and audit_path.exists():
        existing_rows = read_jsonl(manifest_path)
        existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing_rows != rows or existing_audit != audit:
            raise RuntimeError("existing four-action POLAR manifest differs from recomputation")
    else:
        _write_jsonl_atomic(manifest_path, rows)
        _write_json_atomic(audit_path, audit)
    for path in (manifest_path, audit_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps({"manifest": str(manifest_path), **audit}, sort_keys=True))


if __name__ == "__main__":
    main()
