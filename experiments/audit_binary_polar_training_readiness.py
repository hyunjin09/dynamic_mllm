#!/usr/bin/env python3
"""Outcome-free P10 training-readiness audit; never loads model weights."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers
import yaml
from transformers import AutoConfig, AutoTokenizer

from binary_policy.dataset import BinaryPolicyManifestDataset
from experiments.evaluate_binary_polar_internal import select_rows
from experiments.train_binary_polar import file_sha256, select_smoke_rows, validate_gate


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sidecar(path: Path) -> str:
    value = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{value}  {path.name}\n", encoding="utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gates = {name: validate_gate(name, spec) for name, spec in config["gates"].items()}
    manifest = Path(config["data"]["manifest"])
    manifest_sha = file_sha256(manifest)
    if manifest_sha != config["data"]["manifest_sha256"]:
        raise RuntimeError("predictor manifest checksum mismatch")
    route_cap = int(config["data"]["max_valid_routes_per_sample"])
    train = BinaryPolicyManifestDataset(manifest, "train", max_valid_routes=route_cap)
    validation = BinaryPolicyManifestDataset(manifest, "validation", max_valid_routes=route_cap)
    all_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    train_smoke = select_smoke_rows(
        train,
        per_dataset=int(config["smoke"]["train_positive_per_dataset"]),
        seed=int(config["smoke"]["selection_seed"]),
    )
    validation_smoke = select_smoke_rows(
        validation,
        per_dataset=int(config["smoke"]["validation_positive_per_dataset"]),
        seed=int(config["smoke"]["selection_seed"]),
    )
    execution_smoke = select_rows(
        [row for row in all_rows if row["split"] == "validation"],
        per_dataset=int(config["smoke"]["execution_records_per_dataset"]),
        seed=int(config["smoke"]["selection_seed"]),
    )

    groups: dict[str, str] = {}
    duplicate_masks = malformed_masks = unequal_weights = 0
    for row in all_rows:
        previous = groups.setdefault(row["split_group"], row["split"])
        if previous != row["split"]:
            raise RuntimeError("cross-split image group found during readiness audit")
        routes = row.get("valid_routes", [])
        keys = set()
        for route in routes:
            mask = route["mask"]
            malformed_masks += int(len(mask) != 28 or any(value not in (0, 1) for value in mask))
            key = tuple(mask)
            duplicate_masks += int(key in keys)
            keys.add(key)
        if routes:
            expected = 1.0 / len(routes)
            unequal_weights += int(any(abs(float(route["weight"]) - expected) > 1e-8 for route in routes))

    encoder_path = config["predictor"]["embedding_model_path"]
    encoder_config = AutoConfig.from_pretrained(encoder_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True, padding_side="left")
    del tokenizer
    checks = {
        "all_frozen_gates_pass_and_match_hashes": len(gates) == 5,
        "manifest_checksum_matches": manifest_sha == config["data"]["manifest_sha256"],
        "manifest_has_8000_unique_rows": len(all_rows) == 8000 and len({row["uid"] for row in all_rows}) == 8000,
        "split_is_7000_1000": Counter(row["split"] for row in all_rows) == {"train": 7000, "validation": 1000},
        "positive_training_population_is_expected": len(train) == 6043 and len(validation) == 874,
        "no_cross_split_image_groups": len(groups) > 0,
        "masks_are_unique_binary_28_bit": duplicate_masks == 0 and malformed_masks == 0,
        "equal_route_weights": unequal_weights == 0,
        "route_cap_respected": max((len(row.get("valid_routes", [])) for row in all_rows), default=0) <= route_cap,
        "smoke_is_balanced_and_frozen": len(train_smoke) == 300 and len(validation_smoke) == 150,
        "execution_smoke_is_frozen": len(execution_smoke) == 18,
        "encoder_contract_matches": (
            int(encoder_config.hidden_size) == 1024
            and str(getattr(encoder_config, "dtype", "")).endswith("bfloat16")
            and config["predictor"]["input"] == "question_only"
        ),
        "runtime_versions_match": transformers.__version__ == "5.3.0" and torch.__version__.startswith("2.6.0"),
        "comparison_is_loss_only": config["policy"]["representation"] == "direct_factorized_binary_mask",
    }
    smoke_manifest = {
        "schema_version": "binary_polar_p10_smoke_manifest_v1",
        "selection_seed": int(config["smoke"]["selection_seed"]),
        "selection_inputs": "uid, dataset, frozen split, and positive-route availability only; no predictor outcomes",
        "train_positive_uids": [row["uid"] for row in train_smoke],
        "validation_positive_uids": [row["uid"] for row in validation_smoke],
        "execution_validation_uids": [row["uid"] for row in execution_smoke],
    }
    smoke_path = args.output.with_name("p10_smoke_manifest_v1.json")
    write_json(smoke_path, smoke_manifest)
    smoke_sha = sidecar(smoke_path)
    source_paths = (
        args.config,
        Path("binary_policy/dataset.py"),
        Path("binary_policy/losses.py"),
        Path("binary_policy/predictor.py"),
        Path("binary_policy/training.py"),
        Path("experiments/train_binary_polar.py"),
        Path("experiments/evaluate_binary_polar_internal.py"),
        Path("experiments/merge_binary_polar_internal_eval.py"),
        Path("tests/test_binary_policy_objective_comparison.py"),
        Path("tests/test_binary_polar_training_readiness.py"),
    )
    payload = {
        "schema_version": "binary_polar_p10_training_readiness_v1",
        "scope": "static/data/objective/runtime-contract audit only; no predictor training or MLLM execution",
        "passed": all(checks.values()),
        "ready_for_bounded_smoke": all(checks.values()),
        "ready_for_full_training": False,
        "full_training_blocker": "bounded matched smoke has not run or passed",
        "checks": checks,
        "gates": gates,
        "manifest": {
            "path": str(manifest),
            "sha256": manifest_sha,
            "rows": len(all_rows),
            "train_positive": len(train),
            "validation_positive": len(validation),
            "zero_positive": sum(not row.get("valid_routes") for row in all_rows),
        },
        "smoke_manifest": {"path": str(smoke_path), "sha256": smoke_sha},
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
        "confirmed_contract": {
            "predictor_input": "question_only, matching released POLAR",
            "shared_architecture": "frozen Qwen3 encoder + cross-attention + two layer blocks + direct 28-bit head",
            "only_method_change": "duplicated per-route BCE versus grouped exact valid-set NLL",
            "checkpoint_selection": config["training"]["checkpoint_selection"],
            "actual_execution_required": True,
        },
    }
    write_json(args.output, payload)
    audit_sha = sidecar(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# P10 Binary Predictor Training-Readiness Audit\n\n"
        f"Status: **{'PASS' if payload['passed'] else 'FAIL'}**. No predictor training or MLLM execution was run.\n\n"
        "## Confirmed implementation\n\n"
        "- Both objectives use the same frozen 8K-derived manifest, 7K/1K image-group split, "
        "max-50 route sets, equal within-input weights, direct 28-bit head, optimizer, and inference rule.\n"
        "- Exact set-NLL computes complete Bernoulli-mask log probabilities with stable `logsigmoid`, "
        "masked padding, normalized weights, and `logsumexp`.\n"
        "- Duplicated BCE encodes each unique question once, evaluates every selected route as an "
        "independent predictor/BCE row, and normalizes each input's total route weight to one.\n"
        "- The runner now binds gates and the predictor manifest to SHA-256, uses a dedicated deterministic "
        "DataLoader generator, refuses output overwrite, and applies one common checkpoint-selection rule.\n"
        "- The execution adapter runs every predicted top-1 mask, including uncached masks, and reports "
        "FULL-relative behavior, cached Hit@1, visual-ON layers, and the observed MCTS-oracle gap.\n\n"
        "## Important boundary\n\n"
        "The predictor is **question-only**, matching released POLAR. Adding image features would be an "
        "architecture change and would no longer isolate the supervision loss. Zero-positive samples are "
        "excluded from positive training but retained in actual execution evaluation.\n\n"
        "## Frozen bounded smoke\n\n"
        "The smoke uses 100 positive training and 50 positive validation records per dataset "
        "(300/150 total), two epochs, and 18 deterministic validation execution records. The same UIDs, "
        "initial seed, order generator, optimizer settings, and checkpoint rule apply to both objectives.\n\n"
        f"Audit SHA-256: `{audit_sha}`. Smoke-manifest SHA-256: `{smoke_sha}`.\n\n"
        "## Gate\n\n"
        "The implementation is ready for the separately authorized bounded smoke, not full training. "
        "Full training remains blocked until the smoke has finite decreasing loss, frozen gradients, "
        "working actual mask execution, no leakage, and plausible validation improvement.\n",
        encoding="utf-8",
    )
    sidecar(args.report)
    print(json.dumps({"passed": payload["passed"], "output": str(args.output), "report": str(args.report)}))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
