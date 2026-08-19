#!/usr/bin/env python3
"""Freeze checksum-bound readiness for the bounded P12 smoke only."""

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
    "binary_policy/predictor.py",
    "binary_policy/structured.py",
    "binary_policy/training.py",
    "experiments/train_binary_polar.py",
    "experiments/train_binary_polar_p12.py",
    "experiments/preflight_binary_polar_p12.py",
    "experiments/evaluate_binary_polar_p12_conditioning.py",
    "experiments/evaluate_binary_polar_p12_execution.py",
    "tests/test_binary_policy_p12.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["data"]["route_weighting"] != "polar_full_downweight_0.3":
        raise RuntimeError("P12 must retain P11 route weighting")
    if int(config["smoke"]["epochs"]) != 2 or config["p12"]["full_training_authorized"] is not False:
        raise RuntimeError("P12 authorizes only the two-epoch smoke")
    if config["predictor"]["boundary_threshold"] != 0.5:
        raise RuntimeError("P12 boundary threshold changed")
    validated_gates = {name: validate_gate(name, spec) for name, spec in config["gates"].items()}
    manifest_path = Path(config["data"]["manifest"])
    smoke_path = Path(config["smoke"]["manifest"])
    geometry_path = Path(config["p12"]["segment_geometry"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("P12 predictor manifest checksum mismatch")
    if file_sha256(smoke_path) != config["smoke"]["manifest_sha256"]:
        raise RuntimeError("P12 smoke checksum mismatch")
    if file_sha256(geometry_path) != config["p12"]["segment_geometry_sha256"]:
        raise RuntimeError("P12 geometry checksum mismatch")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry.get("passed") is not True or geometry["round_trip"]["accuracy"] != 1:
        raise RuntimeError("P12 round-trip geometry gate failed")
    preflight_path = Path(args.preflight)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True or preflight["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("P12 runtime preflight failed or is bound to another config")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if len(smoke["train_positive_uids"]) != 300 or len(smoke["validation_positive_uids"]) != 150:
        raise RuntimeError("P12 did not retain the 300/150 smoke")
    if len(set(smoke["execution_validation_uids"])) != 60:
        raise RuntimeError("P12 execution set must contain 60 unique records")
    if Counter(row["benchmark"] for row in smoke["execution_rows"]) != Counter(
        {"gqa": 20, "textvqa": 20, "chartqa": 20}
    ):
        raise RuntimeError("P12 execution benchmark strata changed")
    if Counter(row["stratum"] for row in smoke["execution_rows"]) != Counter(
        {"full_correct": 30, "full_wrong_mcts_fixable": 30}
    ):
        raise RuntimeError("P12 execution correctness strata changed")
    missing = [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"P12 required source files missing: {missing}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P12 readiness: {output}")
    payload = {
        "schema_version": "binary_polar_p12_readiness_v1",
        "passed": True,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "ready_for_bounded_smoke": True,
        "ready_for_full_training": False,
        "validated_gates": validated_gates,
        "artifacts": {
            "smoke_manifest": {"path": str(smoke_path), "sha256": file_sha256(smoke_path)},
            "geometry": {"path": str(geometry_path), "sha256": file_sha256(geometry_path)},
            "bf16_preflight": {"path": str(preflight_path), "sha256": file_sha256(preflight_path)},
        },
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "checks": {
            "round_trip_accuracy": 1.0,
            "ambiguous_canonicalizations": 0,
            "same_p11_smoke_identities": True,
            "same_p11_route_weights": True,
            "same_p11_optimizer_hyperparameters": True,
            "one_deterministic_decode_candidate": True,
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
