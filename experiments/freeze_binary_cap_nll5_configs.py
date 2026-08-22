#!/usr/bin/env python3
"""Freeze CAP26/CAP24 exact-NLL five-epoch experiment configs."""

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
CAPS = (26, 24)
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
    "experiments/evaluate_binary_cap_validation_epochs.py",
    "experiments/evaluate_binary_polar_external.py",
    "experiments/summarize_binary_cap_external_eval.py",
    "experiments/run_binary_cap_nll5_train_eval.py",
)


def main() -> None:
    plan = PROJECT / "plans/binary_cap_nll5_executed_validation_plan.md"
    output_root = PROJECT / "outputs/binary_cap_nll5_v1"
    audit_path = output_root / "audits/supervision_audit_v1.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise RuntimeError("CAP-NLL5 supervision audit has not passed")
    base = yaml.safe_load(
        (PROJECT / "configs/binary_cap24_full10_bce_v1.yaml").read_text(encoding="utf-8")
    )
    source_hashes = {path: file_sha256(PROJECT / path) for path in SOURCES}
    created = []
    for cap in CAPS:
        manifest = output_root / "manifests" / f"cap{cap}_manifest_v1.jsonl"
        config = deepcopy(base)
        config["protocol_version"] = f"binary_cap{cap}_nll5_execval_v1"
        config["authorization"] = "binary_cap_nll5_executed_validation_image_question_only"
        config["source_plan"] = {"path": str(plan.relative_to(PROJECT)), "sha256": file_sha256(plan)}
        config["gates"] = {
            "cap_nll5_supervision_integrity": {
                "path": str(audit_path.relative_to(PROJECT)),
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
                "route_cap_policy": f"parent_max50_then_visual_on_le_{cap}_on_common_cap24_population",
                "visual_on_cap": cap,
                "common_eligibility_cap": 24,
            }
        )
        config["evidence"] = {
            "cap_geometry": audit["geometry"],
            "label_oracles": audit["label_oracles"],
        }
        config["predictor"]["modality"] = "image_question"
        config["training"].update(
            {
                "objective": "exact_set_nll",
                "epochs": 5,
                "no_early_stopping": True,
                "save_every_epoch": True,
            }
        )
        config["evaluation"] = {
            "internal_primary": "executed_validation_accuracy",
            "checkpoint_selection": (
                "max_executed_accuracy_then_min_mean_visual_on_then_"
                "min_validation_set_nll_then_earlier_epoch"
            ),
            "validation_execution_records": int(audit["common_validation_records"]),
            "validation_benchmarks": ["gqa", "textvqa", "chartqa"],
            "external_records": 22307,
            "external_suite": "unchanged_full10_no_docvqa",
            "scientific_baseline": "current_live_all_on",
        }
        config["source_sha256"] = source_hashes
        path = PROJECT / f"configs/binary_cap{cap}_nll5_execval_v1.yaml"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen config: {path}")
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
        )
        created.append({"cap": cap, "path": str(path.relative_to(PROJECT)), "sha256": file_sha256(path)})
    freeze_path = output_root / "audits/config_freeze_v1.json"
    freeze_path.write_text(
        json.dumps({"schema_version": "binary_cap_nll5_config_freeze_v1", "configs": created}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze_path.with_suffix(freeze_path.suffix + ".sha256").write_text(
        f"{file_sha256(freeze_path)}  {freeze_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({"created": created}, sort_keys=True))


if __name__ == "__main__":
    main()
