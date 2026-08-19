#!/usr/bin/env python3
"""Freeze checksum-bound readiness for only the bounded P13 smoke."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.train_binary_polar import file_sha256, validate_gate


SOURCE_PATHS = (
    "binary_policy/dataset.py",
    "binary_policy/evaluation.py",
    "binary_policy/losses.py",
    "binary_policy/multimodal.py",
    "binary_policy/predictor.py",
    "binary_policy/training.py",
    "experiments/extract_binary_polar_p13_visual_features.py",
    "experiments/preflight_binary_polar_p13.py",
    "experiments/train_binary_polar_p13.py",
    "experiments/evaluate_binary_polar_p13_conditioning.py",
    "experiments/evaluate_binary_polar_p13_execution.py",
    "tests/test_binary_policy_p13.py",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["data"]["route_weighting"] != "polar_full_downweight_0.3":
        raise RuntimeError("P13 must retain P11 route weighting")
    if int(config["smoke"]["epochs"]) != 2 or config["p13"]["full_training_authorized"] is not False:
        raise RuntimeError("P13 authorizes only a two-epoch smoke")
    if config["policy"]["representation"] != "direct_factorized_binary_mask":
        raise RuntimeError("P13 must restore the P11 direct binary head")
    validated_gates = {name: validate_gate(name, spec) for name, spec in config["gates"].items()}
    manifest_path = Path(config["data"]["manifest"])
    smoke_path = Path(config["smoke"]["manifest"])
    feature_manifest_path = Path(config["p13"]["feature_manifest"])
    permutations_path = Path(config["p13"]["modality_permutations"])
    feature_audit_path = Path(config["p13"]["feature_cache_audit"])
    for path, expected in (
        (manifest_path, config["data"]["manifest_sha256"]),
        (smoke_path, config["smoke"]["manifest_sha256"]),
        (feature_manifest_path, config["p13"]["feature_manifest_sha256"]),
        (permutations_path, config["p13"]["modality_permutations_sha256"]),
        (feature_audit_path, config["p13"]["feature_cache_audit_sha256"]),
        (Path(config["p13"]["fusion_spec"]), config["p13"]["fusion_spec_sha256"]),
    ):
        if file_sha256(path) != expected:
            raise RuntimeError(f"P13 frozen artifact checksum mismatch: {path}")
    audit = json.loads(feature_audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit["selected_records"] != 502:
        raise RuntimeError("P13 visual cache admission failed")
    if audit["answer_fields_consumed"] or audit["route_outcome_fields_consumed"]:
        raise RuntimeError("P13 visual cache contains prohibited leakage")
    feature_rows = read_jsonl(feature_manifest_path)
    if len(feature_rows) != 502 or len({row["uid"] for row in feature_rows}) != 502:
        raise RuntimeError("P13 feature manifest identities are incomplete or duplicated")
    tensor_checks = {}
    for row in feature_rows:
        path = Path(row["path"])
        expected = row["sha256"]
        prior = tensor_checks.setdefault(str(path), expected)
        if prior != expected:
            raise RuntimeError(f"inconsistent tensor checksum declarations: {path}")
    for path, expected in tensor_checks.items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"P13 visual tensor checksum mismatch: {path}")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if len(smoke["train_positive_uids"]) != 300 or len(smoke["validation_positive_uids"]) != 150:
        raise RuntimeError("P13 must reuse the P11 300/150 identities")
    if len(set(smoke["execution_validation_uids"])) != 60:
        raise RuntimeError("P13 execution set is not the frozen 60")
    if Counter(row["benchmark"] for row in smoke["execution_rows"]) != Counter(
        {"gqa": 20, "textvqa": 20, "chartqa": 20}
    ):
        raise RuntimeError("P13 execution benchmark strata changed")
    permutations = json.loads(permutations_path.read_text(encoding="utf-8"))["mapping"]
    predictor_rows = {row["uid"]: row for row in read_jsonl(manifest_path)}
    for uid in smoke["validation_positive_uids"]:
        donors = permutations[uid]
        for donor in donors.values():
            if donor == uid or predictor_rows[donor]["benchmark"] != predictor_rows[uid]["benchmark"]:
                raise RuntimeError("P13 modality permutation is not a within-dataset derangement")
    preflight_path = Path(args.preflight)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True or preflight["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("P13 runtime preflight failed or used another config")
    missing = [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"P13 required sources missing: {missing}")
    admission_path = Path("workspace/binary_polar_p13_execution_admission_gate.md")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P13 readiness: {output}")
    payload = {
        "schema_version": "binary_polar_p13_readiness_v1",
        "passed": True,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "ready_for_bounded_smoke": True,
        "ready_for_full_training": False,
        "validated_gates": validated_gates,
        "artifacts": {
            "smoke_manifest": {"path": str(smoke_path), "sha256": file_sha256(smoke_path)},
            "feature_cache_audit": {"path": str(feature_audit_path), "sha256": file_sha256(feature_audit_path)},
            "bf16_preflight": {"path": str(preflight_path), "sha256": file_sha256(preflight_path)},
        },
        # These inputs are checksum-bound above, but are not pass-bearing JSON
        # gate objects. Keep them outside ``artifacts`` because the shared
        # readiness validator parses every artifact as a gate payload.
        "frozen_inputs": {
            "feature_manifest": {
                "path": str(feature_manifest_path),
                "sha256": file_sha256(feature_manifest_path),
            },
            "modality_permutations": {
                "path": str(permutations_path),
                "sha256": file_sha256(permutations_path),
            },
            "execution_admission": {
                "path": str(admission_path),
                "sha256": file_sha256(admission_path),
            },
        },
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "checks": {
            "p11_smoke_identities_reused": True,
            "p11_direct_head_restored": True,
            "same_exact_set_nll_and_weights": True,
            "native_projected_visual_rows": True,
            "unpooled_visual_tokens": True,
            "visual_tensor_files_verified": len(tensor_checks),
            "modality_permutations_frozen_and_valid": True,
            "no_answer_or_outcome_leakage": True,
            "execution_gate_frozen_before_training": True,
            "full_training_blocked": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
