#!/usr/bin/env python3
"""Freeze four cap-specific training configs from the established BCE contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.train_binary_polar import file_sha256


PROJECT = Path(__file__).resolve().parents[1]
CAPS = (24, 22, 20, 18)
SOURCES = (
    "binary_policy/dataset.py",
    "binary_policy/decode.py",
    "binary_policy/evaluation.py",
    "binary_policy/losses.py",
    "binary_policy/multimodal.py",
    "binary_policy/predictor.py",
    "label_regeneration/cap_supervision.py",
    "experiments/train_binary_polar_full10.py",
    "experiments/preflight_binary_polar_full10.py",
    "experiments/evaluate_binary_polar_external.py",
)


def main() -> None:
    plan = PROJECT / "plans/cap_training.md"
    audit_path = PROJECT / "outputs/binary_cap_sweep_v1/audits/cap_supervision_audit_v1.json"
    geometry_path = PROJECT / "outputs/binary_cap_sweep_v1/audits/cap_geometry_v1.json"
    oracle_path = PROJECT / "outputs/binary_cap_sweep_v1/audits/cap_label_oracles_v1.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise RuntimeError("cap supervision audit has not passed")
    base = yaml.safe_load(
        (PROJECT / "configs/binary_pareto_full10_bce_v1.yaml").read_text(encoding="utf-8")
    )
    source_hashes = {path: file_sha256(PROJECT / path) for path in SOURCES}
    created = []
    for cap in CAPS:
        manifest = PROJECT / f"outputs/binary_cap_sweep_v1/manifests/cap{cap}_manifest_v1.jsonl"
        config = deepcopy(base)
        config["protocol_version"] = f"binary_cap{cap}_full10_bce_v1"
        config["authorization"] = "binary_cap_sweep_full10_image_question_only"
        config["source_plan"] = {"path": "plans/cap_training.md", "sha256": file_sha256(plan)}
        config["gates"] = {
            "cap_supervision_integrity": {
                "path": "outputs/binary_cap_sweep_v1/audits/cap_supervision_audit_v1.json",
                "sha256": file_sha256(audit_path),
            },
            "full_visual_cache": base["gates"]["full_visual_cache"],
        }
        config["data"].update(
            {
                "manifest": str(manifest.relative_to(PROJECT)),
                "manifest_sha256": file_sha256(manifest),
                "parent_manifest": audit["source"]["path"],
                "parent_manifest_sha256": audit["source"]["sha256"],
                "train_positive_records": int(audit["common_train_records"]),
                "validation_positive_records": int(audit["common_validation_records"]),
                "route_cap_policy": f"parent_max50_then_visual_on_le_{cap}_on_common_cap18_population",
                "visual_on_cap": cap,
                "common_eligibility_cap": 18,
            }
        )
        config["evidence"] = {
            "cap_geometry": {
                "path": str(geometry_path.relative_to(PROJECT)),
                "sha256": file_sha256(geometry_path),
            },
            "label_oracles": {
                "path": str(oracle_path.relative_to(PROJECT)),
                "sha256": file_sha256(oracle_path),
            },
        }
        config["predictor"]["modality"] = "image_question"
        config["evaluation"] = {
            "internal_primary": "cap_valid_hit_at_1",
            "checkpoint_selection": "max_cap_hit_then_min_nearest_then_min_objective_loss_then_earlier",
            "external_records": 22307,
            "external_suite": "unchanged_full10_no_docvqa",
            "scientific_baseline": "current_live_all_on",
        }
        config["source_sha256"] = source_hashes
        path = PROJECT / f"configs/binary_cap{cap}_full10_bce_v1.yaml"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen config: {path}")
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        created.append({"cap": cap, "path": str(path.relative_to(PROJECT)), "sha256": file_sha256(path)})
    print(json.dumps({"created": created}, sort_keys=True))


if __name__ == "__main__":
    main()
