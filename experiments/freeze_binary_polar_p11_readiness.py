#!/usr/bin/env python3
"""Freeze checksum-bound readiness for the bounded P11 smoke only."""

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
    "binary_policy/predictor.py",
    "binary_policy/training.py",
    "binary_policy/p11.py",
    "experiments/train_binary_polar.py",
    "experiments/train_binary_polar_bias.py",
    "experiments/evaluate_binary_polar_p11_conditioning.py",
    "experiments/evaluate_binary_polar_p11_execution.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["data"]["route_weighting"] != "polar_full_downweight_0.3":
        raise RuntimeError("P11 requires the frozen 0.3 POLAR-compatible weighting")
    if config["smoke"]["epochs"] != 2:
        raise RuntimeError("P11 matched smoke must remain two epochs")
    if config["p11"]["full_training_authorized"] is not False:
        raise RuntimeError("P11 readiness may not authorize full training")
    validated_gates = {name: validate_gate(name, spec) for name, spec in config["gates"].items()}
    manifest_path = Path(config["data"]["manifest"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("P11 manifest checksum mismatch")
    smoke_path = Path(config["smoke"]["manifest"])
    geometry_path = Path(config["p11"]["label_geometry"])
    if file_sha256(smoke_path) != config["smoke"]["manifest_sha256"]:
        raise RuntimeError("P11 smoke checksum mismatch")
    if file_sha256(geometry_path) != config["p11"]["label_geometry_sha256"]:
        raise RuntimeError("P11 geometry checksum mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if len(smoke["train_positive_uids"]) != 300 or len(smoke["validation_positive_uids"]) != 150:
        raise RuntimeError("P11 must reuse the frozen 300/150 P10 smoke identities")
    if len(smoke["execution_validation_uids"]) != 60:
        raise RuntimeError("P11 bounded execution must contain exactly 60 records")
    if len(set(smoke["execution_validation_uids"])) != 60:
        raise RuntimeError("P11 execution identities are not unique")
    benchmark_counts = Counter(row["benchmark"] for row in smoke["execution_rows"])
    stratum_counts = Counter(row["stratum"] for row in smoke["execution_rows"])
    if benchmark_counts != Counter({"gqa": 20, "textvqa": 20, "chartqa": 20}):
        raise RuntimeError("P11 execution benchmark strata changed")
    if stratum_counts != Counter({"full_correct": 30, "full_wrong_mcts_fixable": 30}):
        raise RuntimeError("P11 execution correctness strata changed")

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P11 readiness gate: {output}")
    payload = {
        "schema_version": "binary_polar_p11_readiness_v1",
        "passed": True,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "ready_for_bounded_smoke": True,
        "ready_for_full_training": False,
        "validated_gates": validated_gates,
        "artifacts": {
            "smoke_manifest": {"path": str(smoke_path), "sha256": file_sha256(smoke_path)},
        },
        "label_geometry": {"path": str(geometry_path), "sha256": file_sha256(geometry_path)},
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "checks": {
            "p10_smoke_train_identities_reused": True,
            "p10_smoke_validation_identities_reused": True,
            "polar_all_on_relative_weight": 0.3,
            "route_weights_normalized_within_input": True,
            "same_relative_weights_both_objectives": True,
            "execution_identities_frozen_before_p11_outcomes": True,
            "execution_records": 60,
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
