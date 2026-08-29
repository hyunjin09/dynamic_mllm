#!/usr/bin/env python3
"""Freeze balanced safe-FULL versus mandatory-deviation probe records."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.prepare_four_action_collapse import write_frozen
from experiments.train_binary_polar import file_sha256
from four_action_online_router.data import load_verified_manifest, mandatory_boundary_record


def _stable(seed: int, purpose: str, uid: str) -> str:
    return sha256(f"{seed}:{purpose}:{uid}".encode()).hexdigest()


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def build_matched_boundary_probe_records(
    boundary_rows: Iterable[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match positive/negative states exactly within split, dataset, and layer."""

    boundaries = [dict(row) for row in boundary_rows]
    keys = [(str(row["split"]), str(row["dataset"])) for row in boundaries]
    records = []
    excluded = []
    for split, dataset in sorted(set(keys)):
        pool = [
            row for row in boundaries
            if row["split"] == split and row["dataset"] == dataset
        ]
        layers = sorted({int(row["boundary_layer"]) for row in pool})
        for layer in layers:
            positives = [row for row in pool if int(row["boundary_layer"]) == layer]
            negatives = [row for row in pool if int(row["boundary_layer"]) > layer]
            positives.sort(
                key=lambda row: _stable(seed, f"positive:{split}:{dataset}:{layer}", row["uid"])
            )
            negatives.sort(
                key=lambda row: _stable(seed, f"negative:{split}:{dataset}:{layer}", row["uid"])
            )
            count = min(len(positives), len(negatives))
            for row in positives[count:]:
                excluded.append(
                    {
                        "uid": row["uid"],
                        "split": split,
                        "dataset": dataset,
                        "boundary_layer": layer,
                        "reason": "no_unique_same_split_dataset_layer_safe_full_match",
                    }
                )
            for positive, negative in zip(positives[:count], negatives[:count]):
                pair_id = sha256(
                    f"{seed}:{split}:{dataset}:{layer}:{positive['uid']}:{negative['uid']}".encode()
                ).hexdigest()[:20]
                records.extend(
                    [
                        {
                            "pair_id": pair_id,
                            "uid": positive["uid"],
                            "split": split,
                            "dataset": dataset,
                            "target_layer": layer,
                            "source_boundary_layer": layer,
                            "label": 1,
                            "class_name": "mandatory_deviation_now",
                        },
                        {
                            "pair_id": pair_id,
                            "uid": negative["uid"],
                            "split": split,
                            "dataset": dataset,
                            "target_layer": layer,
                            "source_boundary_layer": int(negative["boundary_layer"]),
                            "label": 0,
                            "class_name": "safe_to_continue_full",
                        },
                    ]
                )
    cell_counts = Counter(
        (row["split"], row["dataset"], row["target_layer"], row["label"])
        for row in records
    )
    cells = {(split, dataset, layer) for split, dataset, layer, _ in cell_counts}
    balanced = all(
        cell_counts[(split, dataset, layer, 0)]
        == cell_counts[(split, dataset, layer, 1)]
        for split, dataset, layer in cells
    )
    audit = {
        "schema_version": "four_action_boundary_probe_manifest_audit_v1",
        "passed": balanced and bool(records),
        "balanced": balanced,
        "records": len(records),
        "pairs": len(records) // 2,
        "feature_uids": len({str(row["uid"]) for row in records}),
        "split_counts": dict(sorted(Counter(row["split"] for row in records).items())),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in records).items())),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in records).items())),
        "target_layer_counts": dict(
            sorted(Counter(int(row["target_layer"]) for row in records).items())
        ),
        "min_target_layer": min(int(row["target_layer"]) for row in records),
        "max_target_layer": max(int(row["target_layer"]) for row in records),
        "excluded_positive_boundary_records": len(excluded),
        "excluded_positive_records": sorted(
            excluded, key=lambda row: (row["split"], row["dataset"], row["boundary_layer"], row["uid"])
        ),
    }
    return records, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-config", default="configs/four_action_online_router_v1.yaml"
    )
    parser.add_argument("--plan", default="plans/four_action_collapse.md")
    parser.add_argument("--output-dir", default="analysis/4action_collapse")
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    source_config_path = Path(args.source_config)
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    rows = load_verified_manifest(
        source_config["data"]["manifest"], source_config["data"]["manifest_sha256"]
    )
    boundaries = []
    for row in rows:
        if row["route_type"] != "W2C":
            continue
        boundary = mandatory_boundary_record(
            row, num_layers=int(source_config["router"]["num_layers"])
        )
        boundary["split"] = row["split"]
        boundaries.append(boundary)
    records, audit = build_matched_boundary_probe_records(boundaries, seed=args.seed)
    if not audit["passed"] or audit["max_target_layer"] >= 27:
        raise RuntimeError(f"boundary probe matching failed: {audit}")

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "upfront_vs_online_boundary_probe_manifest.jsonl"
    audit_path = output_dir / "upfront_vs_online_boundary_probe_manifest_audit.json"
    config_path = output_dir / "upfront_vs_online_boundary_probe_config.yaml"
    plan_path = Path(args.plan)
    audit.update(
        {
            "source_manifest": source_config["data"]["manifest"],
            "source_manifest_sha256": source_config["data"]["manifest_sha256"],
            "source_config": str(source_config_path),
            "source_config_sha256": file_sha256(source_config_path),
            "source_plan": str(plan_path),
            "source_plan_sha256": file_sha256(plan_path),
        }
    )
    write_frozen(manifest_path, _jsonl_text(records))
    write_frozen(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    config = {
        "protocol_version": "upfront_vs_online_boundary_probe_v1",
        "source_plan": str(plan_path),
        "source_plan_sha256": file_sha256(plan_path),
        "source_config": str(source_config_path),
        "source_config_sha256": file_sha256(source_config_path),
        "base_model": source_config["base_model"],
        "executor": source_config["executor"],
        "data": {
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "manifest_audit": str(audit_path),
            "manifest_audit_sha256": file_sha256(audit_path),
            "records": audit["records"],
            "pairs": audit["pairs"],
            "feature_uids": audit["feature_uids"],
            "split_counts": audit["split_counts"],
            "matching": "exact_split_dataset_target_layer_without_replacement",
            "excluded_positive_boundary_records": audit[
                "excluded_positive_boundary_records"
            ],
        },
        "features": {
            "upfront": "unified_full_pre_layer_0_final_text_plus_mean_visual",
            "online": "unified_full_pre_target_layer_final_text_plus_mean_visual",
            "layer_identity": True,
            "hidden_width": 3584,
            "dtype": "bfloat16",
        },
        "probe": {
            "architecture": "matched_two_branch_mlp_v1",
            "branch_width": 128,
            "layer_embedding_width": 32,
            "classifier_hidden_width": 128,
            "dropout": 0.1,
        },
        "training": {
            "seed": args.seed,
            "epochs": 30,
            "batch_size": 128,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "validation_every_epoch": True,
            "checkpoint_metric": "validation_auroc",
        },
        "analysis": {
            "metrics": ["auroc", "accuracy", "f1"],
            "bootstrap_method": "paired_uid_group_resampling",
            "bootstrap_draws": 2000,
            "confidence": 0.95,
            "online_advantage_rule": "lower_95ci_of_auroc_difference_greater_than_zero",
        },
        "reporting": {
            "feature_root": "outputs/four_action_collapse/boundary_probe_features_v1",
            "output_dir": "outputs/four_action_collapse/upfront_vs_online_boundary_probe_v1",
            "report": "analysis/4action_collapse/upfront_vs_online_boundary_probe_report.md",
            "summary": "analysis/4action_collapse/upfront_vs_online_boundary_probe_summary.json",
        },
    }
    write_frozen(config_path, yaml.safe_dump(config, sort_keys=False))
    print(
        json.dumps(
            {
                "event": "boundary_probe_prepared",
                "passed": audit["passed"],
                "records": audit["records"],
                "pairs": audit["pairs"],
                "feature_uids": audit["feature_uids"],
                "split_counts": audit["split_counts"],
                "dataset_counts": audit["dataset_counts"],
                "target_layer_range": [
                    audit["min_target_layer"],
                    audit["max_target_layer"],
                ],
                "excluded_positive_boundary_records": audit[
                    "excluded_positive_boundary_records"
                ],
                "manifest_sha256": config["data"]["manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
