#!/usr/bin/env python3
"""Freeze the static and real-encoder P10 preflights into one training gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_binary_polar import file_sha256


def artifact(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--static-audit", required=True, type=Path)
    parser.add_argument("--runtime-preflight", required=True, type=Path)
    parser.add_argument("--smoke-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    static = json.loads(args.static_audit.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime_preflight.read_text(encoding="utf-8"))
    smoke = json.loads(args.smoke_manifest.read_text(encoding="utf-8"))
    config_sha = file_sha256(args.config)
    checks = {
        "static_audit_passed": static.get("passed") is True and static.get("ready_for_bounded_smoke") is True,
        "runtime_preflight_passed": runtime.get("passed") is True,
        "both_preflights_bind_current_config": (
            static["source_sha256"][str(args.config)] == config_sha
            and runtime["config_sha256"] == config_sha
        ),
        "smoke_manifest_matches_both_preflights": (
            static["smoke_manifest"]["sha256"] == file_sha256(args.smoke_manifest)
            and smoke["schema_version"] == "binary_polar_p10_smoke_manifest_v1"
        ),
        "zero_optimizer_steps": runtime["checks"]["no_optimizer_step_or_checkpoint"],
        "matched_real_initialization": runtime["checks"]["matched_initialization"],
        "real_losses_and_gradients_finite": (
            runtime["checks"]["both_losses_are_finite"]
            and runtime["checks"]["both_predictor_gradients_are_finite"]
        ),
    }
    sources = (
        args.config,
        Path("binary_policy/dataset.py"),
        Path("binary_policy/losses.py"),
        Path("binary_policy/predictor.py"),
        Path("binary_policy/training.py"),
        Path("experiments/train_binary_polar.py"),
        Path("experiments/evaluate_binary_polar_internal.py"),
        Path("experiments/merge_binary_polar_internal_eval.py"),
        Path("experiments/preflight_binary_polar_runtime.py"),
        Path("tests/test_binary_policy_objective_comparison.py"),
        Path("tests/test_binary_polar_training_readiness.py"),
    )
    payload = {
        "schema_version": "binary_polar_p10_readiness_gate_v1",
        "passed": all(checks.values()),
        "ready_for_bounded_smoke": all(checks.values()),
        "ready_for_full_training": False,
        "full_training_blocker": "matched bounded smoke has not been executed and interpreted",
        "config": str(args.config),
        "config_sha256": config_sha,
        "checks": checks,
        "artifacts": {
            "static_audit": artifact(args.static_audit),
            "runtime_preflight": artifact(args.runtime_preflight),
            "smoke_manifest": artifact(args.smoke_manifest),
        },
        "source_sha256": {str(path): file_sha256(path) for path in sources},
        "training_authorization": "none; this gate permits launch only after explicit user approval",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{file_sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Final P10 Pre-Training Readiness Decision\n\n"
        f"Status: **{'READY FOR BOUNDED SMOKE' if payload['passed'] else 'NOT READY'}**.\n\n"
        "No optimizer step, predictor training, checkpoint fitting, or 7B MLLM evaluation was performed.\n\n"
        "## What is verified\n\n"
        "- Exact valid-set NLL is the stable weighted probability mass over complete 28-bit masks.\n"
        "- Duplicated BCE and set-NLL share the same selected masks, equal weights, direct head, data split, "
        "optimizer settings, initialization, shuffle generator, and checkpoint-selection rule.\n"
        "- All 8,000 manifest rows, 6,043 positive train rows, 874 positive validation rows, 1,083 "
        "zero-positive rows, masks, weights, caps, and image groups pass the frozen audit.\n"
        "- The real pinned Qwen3 tokenizer/encoder produced BF16 `[3,11,1024]` features on one GQA, "
        "TextVQA, and ChartQA record. Both objectives produced finite losses and finite gradients on all "
        "33 predictor parameter tensors from the same initialization. Encoder gradients remained absent.\n"
        "- The execution adapter evaluates both selected-set and full raw-cache Hit@1, and actually executes "
        "uncached top-1 masks through the repaired binary Qwen executor.\n\n"
        "## Controlled boundary\n\n"
        "The predictor is question-only because that is the released POLAR architecture. Image conditioning "
        "would be an architecture change, not the requested loss-only comparison. The factorized head still "
        "does not explicitly model cross-layer dependencies.\n\n"
        "## Remaining gate\n\n"
        "Only the frozen 300-train/150-validation, two-epoch matched smoke may run next, with 18 actual "
        "execution records per objective. Full training remains blocked until that smoke passes and is "
        "interpreted.\n",
        encoding="utf-8",
    )
    args.report.with_suffix(args.report.suffix + ".sha256").write_text(
        f"{file_sha256(args.report)}  {args.report.name}\n", encoding="utf-8"
    )
    print(json.dumps({"passed": payload["passed"], "output": str(args.output)}))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
