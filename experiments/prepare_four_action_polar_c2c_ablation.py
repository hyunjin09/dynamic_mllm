#!/usr/bin/env python3
"""Freeze the matched POLAR C2C exact-all-FULL removal ablation."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.prepare_four_action_collapse import write_frozen
from experiments.train_binary_polar import file_sha256
from four_action_policy.feature_cache import read_feature_manifest, visual_cache_contract


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _is_all_full(route: dict[str, Any], *, num_layers: int) -> bool:
    return route.get("actions") == ["FULL"] * num_layers


def build_c2c_no_allfull_manifest(
    rows: Iterable[dict[str, Any]], *, num_layers: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove exact all-FULL only from train C2C, preserving validation labels."""

    derived = []
    excluded = []
    removed_routes = 0
    validation_changed = 0
    w2c_changed = 0
    source_rows = [copy.deepcopy(row) for row in rows]
    source_route_count = sum(len(row["valid_routes"]) for row in source_rows)
    for row in source_rows:
        original = copy.deepcopy(row["valid_routes"])
        if row.get("split") == "train" and row.get("route_type") == "C2C":
            row["valid_routes"] = [
                route for route in original if not _is_all_full(route, num_layers=num_layers)
            ]
            removed_routes += len(original) - len(row["valid_routes"])
            if not row["valid_routes"]:
                excluded.append(
                    {
                        "uid": str(row["uid"]),
                        "dataset": str(row["dataset"]),
                        "split": str(row["split"]),
                        "reason": "train_c2c_only_exact_all_full_route",
                    }
                )
                continue
            row["valid_route_count"] = len(row["valid_routes"])
        elif row.get("split") == "validation" and row["valid_routes"] != original:
            validation_changed += 1
        elif row.get("route_type") == "W2C" and row["valid_routes"] != original:
            w2c_changed += 1
        derived.append(row)

    uids = [str(row["uid"]) for row in derived]
    if len(uids) != len(set(uids)):
        raise RuntimeError("derived C2C ablation manifest has duplicate UIDs")
    split_counts = Counter(str(row["split"]) for row in derived)
    route_type_counts = Counter(str(row["route_type"]) for row in derived)
    dataset_counts = Counter(str(row["dataset"]) for row in derived)
    audit = {
        "schema_version": "four_action_polar_c2c_no_allfull_audit_v1",
        "passed": validation_changed == 0 and w2c_changed == 0,
        "source_records": len(source_rows),
        "records": len(derived),
        "source_routes": source_route_count,
        "routes": sum(len(row["valid_routes"]) for row in derived),
        "removed_train_c2c_allfull_routes": removed_routes,
        "excluded_train_c2c_records": len(excluded),
        "excluded_uids": sorted(row["uid"] for row in excluded),
        "exclusion_records": sorted(excluded, key=lambda row: row["uid"]),
        "validation_rows_changed": validation_changed,
        "w2c_rows_changed": w2c_changed,
        "split_counts": dict(sorted(split_counts.items())),
        "route_type_counts": dict(sorted(route_type_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "unique_image_groups": len({str(row["split_group"]) for row in derived}),
    }
    return derived, audit


def filter_feature_rows(
    rows: Iterable[dict[str, Any]], retained_uids: set[str]
) -> list[dict[str, Any]]:
    selected = [copy.deepcopy(row) for row in rows if str(row.get("uid")) in retained_uids]
    observed = {str(row.get("uid")) for row in selected}
    if observed != retained_uids or len(selected) != len(observed):
        raise RuntimeError("filtered visual feature UID coverage mismatch")
    return selected


def build_config(
    parent: dict[str, Any], *, parent_path: Path, plan_path: Path,
    manifest_path: Path, manifest_audit_path: Path, feature_manifest_path: Path,
    feature_audit_path: Path, audit: dict[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(parent)
    config["protocol_version"] = "four_action_polar_c2c_no_allfull_v1"
    config["source_plan"] = str(plan_path)
    config["source_plan_sha256"] = file_sha256(plan_path)
    config["matched_parent_config"] = str(parent_path)
    config["matched_parent_config_sha256"] = file_sha256(parent_path)
    config["data"].update(
        {
            "source_manifest": "datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl",
            "source_manifest_sha256": "a44ca6e8684bc1a559997ce0ea52b2796f3265d19be90e22439c653741f36ed7",
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "manifest_audit": str(manifest_audit_path),
            "manifest_audit_sha256": file_sha256(manifest_audit_path),
            "train_records": int(audit["split_counts"]["train"]),
            "validation_records": int(audit["split_counts"]["validation"]),
            "unique_image_groups": int(audit["unique_image_groups"]),
            "valid_routes": int(audit["routes"]),
            "c2c_exact_allfull_removed_train_routes": int(
                audit["removed_train_c2c_allfull_routes"]
            ),
            "c2c_exact_allfull_route_empty_exclusions": int(
                audit["excluded_train_c2c_records"]
            ),
        }
    )
    config["visual_features"]["manifest"] = str(feature_manifest_path)
    config["visual_features"]["cache_audit"] = str(feature_audit_path)
    config["reporting"] = {
        "output_dir": "outputs/four_action_collapse/polar_c2c_no_allfull_v1",
        "history": "analysis/4action_collapse/polar_c2c_no_allfull_history.jsonl",
        "report": "analysis/4action_collapse/polar_c2c_no_allfull_report.md",
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent-config", default="configs/four_action_polar_image_question_nll_v1.yaml"
    )
    parser.add_argument("--plan", default="plans/four_action_collapse.md")
    parser.add_argument("--output-dir", default="analysis/4action_collapse")
    parser.add_argument("--num-layers", type=int, default=28)
    args = parser.parse_args()

    parent_path = Path(args.parent_config)
    plan_path = Path(args.plan)
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    if parent["training"]["objective"] != "exact_set_nll":
        raise RuntimeError("POLAR C2C ablation requires the exact-set NLL parent")
    source_manifest = Path(parent["data"]["manifest"])
    if file_sha256(source_manifest) != parent["data"]["manifest_sha256"]:
        raise RuntimeError("parent POLAR manifest checksum mismatch")
    source_rows = [
        json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    derived, audit = build_c2c_no_allfull_manifest(
        source_rows, num_layers=args.num_layers
    )
    if (
        audit["source_records"] != 6811
        or audit["removed_train_c2c_allfull_routes"] != 3501
        or audit["excluded_train_c2c_records"] != 35
        or audit["split_counts"] != {"train": 5910, "validation": 866}
        or audit["validation_rows_changed"] != 0
        or audit["w2c_rows_changed"] != 0
    ):
        raise RuntimeError(f"unexpected C2C ablation audit: {audit}")

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "polar_c2c_no_allfull_manifest.jsonl"
    manifest_audit_path = output_dir / "polar_c2c_no_allfull_manifest_audit.json"
    feature_manifest_path = output_dir / "polar_c2c_no_allfull_feature_manifest.jsonl"
    feature_audit_path = output_dir / "polar_c2c_no_allfull_feature_audit.json"
    config_path = output_dir / "polar_c2c_no_allfull_config.yaml"
    audit.update(
        {
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": file_sha256(source_manifest),
            "source_plan": str(plan_path),
            "source_plan_sha256": file_sha256(plan_path),
        }
    )
    write_frozen(manifest_path, _jsonl_text(derived))
    write_frozen(manifest_audit_path, _json_text(audit))

    source_feature_manifest = Path(parent["visual_features"]["manifest"])
    source_feature_audit = Path(parent["visual_features"]["cache_audit"])
    source_feature_audit_payload = json.loads(
        source_feature_audit.read_text(encoding="utf-8")
    )
    if file_sha256(source_feature_manifest) != source_feature_audit_payload["manifest_sha256"]:
        raise RuntimeError("parent visual feature manifest checksum mismatch")
    retained_uids = {str(row["uid"]) for row in derived}
    features = filter_feature_rows(
        read_feature_manifest(source_feature_manifest), retained_uids
    )
    write_frozen(feature_manifest_path, _jsonl_text(features))

    config = build_config(
        parent,
        parent_path=parent_path,
        plan_path=plan_path,
        manifest_path=manifest_path,
        manifest_audit_path=manifest_audit_path,
        feature_manifest_path=feature_manifest_path,
        feature_audit_path=feature_audit_path,
        audit=audit,
    )
    write_frozen(config_path, yaml.safe_dump(config, sort_keys=False))
    feature_audit = {
        "schema_version": "four_action_polar_c2c_no_allfull_feature_audit_v1",
        "passed": True,
        "records": len(features),
        "manifest": str(feature_manifest_path),
        "manifest_sha256": file_sha256(feature_manifest_path),
        "cache_contract": visual_cache_contract(config),
        "source_feature_manifest": str(source_feature_manifest),
        "source_feature_manifest_sha256": file_sha256(source_feature_manifest),
        "source_feature_audit": str(source_feature_audit),
        "source_feature_audit_sha256": file_sha256(source_feature_audit),
        "tensor_files_reused_without_copy": True,
    }
    write_frozen(feature_audit_path, _json_text(feature_audit))
    print(
        json.dumps(
            {
                "event": "four_action_polar_c2c_ablation_prepared",
                "config": str(config_path),
                "config_sha256": file_sha256(config_path),
                "manifest_sha256": file_sha256(manifest_path),
                "feature_manifest_sha256": file_sha256(feature_manifest_path),
                **audit,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
