#!/usr/bin/env python3
"""Static readiness gate for the matched Pareto BCE/NLL pipelines."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.train_binary_polar import file_sha256, validate_gate


PROJECT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_config(config: dict) -> dict:
    result = deepcopy(config)
    result["protocol_version"] = "MATCHED"
    result["training"]["objective"] = "MATCHED"
    return result


def main() -> None:
    bce_path = PROJECT / "configs/binary_pareto_full10_bce_v1.yaml"
    nll_path = PROJECT / "configs/binary_pareto_full10_nll_v1.yaml"
    configs = {
        "duplicated_bce": yaml.safe_load(bce_path.read_text(encoding="utf-8")),
        "exact_set_nll": yaml.safe_load(nll_path.read_text(encoding="utf-8")),
    }
    if normalized_config(configs["duplicated_bce"]) != normalized_config(configs["exact_set_nll"]):
        raise RuntimeError("Pareto configurations differ outside protocol/objective")
    for objective, config in configs.items():
        if config["training"]["objective"] != objective:
            raise RuntimeError(f"objective mismatch in {objective} config")
        if config["source_plan"]["sha256"] != file_sha256(PROJECT / config["source_plan"]["path"]):
            raise RuntimeError("Pareto source-plan checksum mismatch")
        for name, specification in config["gates"].items():
            validate_gate(name, specification)
        for path_value, expected in config["source_sha256"].items():
            if file_sha256(PROJECT / path_value) != expected:
                raise RuntimeError(f"source checksum mismatch: {path_value}")
        for specification in config["evidence"].values():
            if file_sha256(PROJECT / specification["path"]) != specification["sha256"]:
                raise RuntimeError(f"evidence checksum mismatch: {specification['path']}")

    config = configs["duplicated_bce"]
    manifest_path = PROJECT / config["data"]["manifest"]
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("Pareto manifest checksum mismatch")
    rows = read_jsonl(manifest_path)
    if len(rows) != 8000 or len({row["uid"] for row in rows}) != 8000:
        raise RuntimeError("Pareto manifest population/UID failure")
    if Counter(row["benchmark"] for row in rows) != Counter(
        {"gqa": 4000, "textvqa": 2000, "chartqa": 2000}
    ):
        raise RuntimeError("Pareto dataset population failure")
    if Counter(row["split"] for row in rows) != Counter({"train": 7000, "validation": 1000}):
        raise RuntimeError("Pareto split population failure")
    positive = [row for row in rows if row["valid_routes"]]
    if Counter(row["split"] for row in positive) != Counter({"train": 6043, "validation": 874}):
        raise RuntimeError("Pareto positive split failure")
    group_splits = {}
    for row in rows:
        prior = group_splits.setdefault(row["split_group"], row["split"])
        if prior != row["split"]:
            raise RuntimeError("Pareto image-group leakage")
        if row["pareto_efficient_route_count"] != len(row["valid_routes"]):
            raise RuntimeError("Pareto route-count mismatch")
        if len(row["valid_routes"]) > 50:
            raise RuntimeError("Pareto manifest exceeds frozen parent cap")
        retained = {route["key"] for route in row["valid_routes"]}
        if not retained.issubset(set(row["original_valid_mask_keys"])):
            raise RuntimeError("Pareto route is absent from original supervision")

    feature_path = PROJECT / config["visual_features"]["manifest"]
    if file_sha256(feature_path) != config["visual_features"]["manifest_sha256"]:
        raise RuntimeError("visual-feature manifest checksum mismatch")
    features = {row["uid"] for row in read_jsonl(feature_path)}
    if {row["uid"] for row in positive} - features:
        raise RuntimeError("visual-feature cache does not cover Pareto positives")

    payload = {
        "schema_version": "binary_pareto_training_readiness_v1",
        "passed": True,
        "outcome_blind": True,
        "configs": {
            objective: {
                "path": str(path.relative_to(PROJECT)),
                "sha256": file_sha256(path),
            }
            for objective, path in (
                ("duplicated_bce", bce_path),
                ("exact_set_nll", nll_path),
            )
        },
        "matched_except_objective": True,
        "population": {
            "records": len(rows),
            "positive_records": len(positive),
            "train_positive": 6043,
            "validation_positive": 874,
        },
        "pareto_manifest": {"path": str(manifest_path.relative_to(PROJECT)), "sha256": file_sha256(manifest_path)},
        "visual_feature_coverage": "PASS",
        "external_contract": {
            "bundle": "eval/reference/shared_prefix_eval_20260812",
            "records": 22307,
            "docvqa_excluded": True,
        },
        "gpu_jobs_submitted": 0,
    }
    output = PROJECT / "outputs/binary_pareto_v1/audits/training_readiness_v1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"passed": True, "output": str(output.relative_to(PROJECT))}))


if __name__ == "__main__":
    main()
