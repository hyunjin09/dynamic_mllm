#!/usr/bin/env python3
"""Run one BCE full10 modality, synchronize, then run the frozen external eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from experiments.train_binary_polar import file_sha256


POPE_OVERLAP_CLUSTER = "7b21e833d6fe982ef6c55c793c7c4fc8111b9a4876334ef7f4c470362d20ce55"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True)


def wait_for(path: Path, timeout_seconds: float, description: str) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(20)
    raise TimeoutError(f"timed out waiting for {description}: {path}")


def selected_checkpoint(train_root: Path) -> Path:
    summary_path = train_root / "training_summary.json"
    summary = read_json(summary_path)
    if summary.get("passed") is not True or summary.get("objective") != "duplicated_bce":
        raise RuntimeError(f"invalid BCE training summary: {summary_path}")
    epoch = int(summary["selections"]["best_hit_at_1"])
    checkpoint = train_root / f"epoch_{epoch:02d}" / "checkpoint.pt"
    expected = {
        int(row["epoch"]): str(row["checkpoint_sha256"])
        for row in summary["checkpoints"]
    }[epoch]
    if not checkpoint.exists() or file_sha256(checkpoint) != expected:
        raise RuntimeError(f"selected checkpoint integrity failure: {checkpoint}")
    return checkpoint


def freeze_evaluation_contract(
    *,
    output: Path,
    config: Path,
    bundle: Path,
    question_checkpoint: Path,
    image_question_checkpoint: Path,
) -> None:
    manifests = [
        bundle / "data/heldout_lmms_recommended_v1/samples.jsonl",
        bundle / "data/heldout_mmstar_mmmu_final_v2/samples.jsonl",
        bundle / "data/heldout_pope_v1/samples.jsonl",
    ]
    payload = {
        "schema_version": "binary_polar_full10_bce_external_contract_v1",
        "outcome_blind": True,
        "objective": "duplicated_bce",
        "selection_rule": "maximum validation Valid-Set Hit@1 with the frozen tie hierarchy",
        "active_records": 22307,
        "excluded_benchmarks": ["docvqa"],
        "config": {"path": str(config), "sha256": file_sha256(config)},
        "checkpoints": {
            "question": {
                "path": str(question_checkpoint),
                "sha256": file_sha256(question_checkpoint),
            },
            "image_question": {
                "path": str(image_question_checkpoint),
                "sha256": file_sha256(image_question_checkpoint),
            },
        },
        "bundle_manifests": [
            {"path": str(path), "sha256": file_sha256(path)} for path in manifests
        ],
        "generation": {
            "decoding": "deterministic_greedy",
            "mask_decoding": "logit_greater_than_or_equal_to_zero",
            "eos_token_ids": [151645],
            "repetition_penalty": 1.05,
        },
        "pope_image_disjoint_sensitivity": {
            "overlap_cluster": POPE_OVERLAP_CLUSTER,
            "official_records": 9000,
            "image_disjoint_records": 8982,
        },
    }
    write_json(output, payload)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-preflight", type=Path, required=True)
    parser.add_argument("--modality", choices=("question", "image_question"), required=True)
    parser.add_argument("--question-train-root", type=Path, required=True)
    parser.add_argument("--image-question-train-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--coordinator", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=36.0)
    args = parser.parse_args()

    train_root = (
        args.question_train_root
        if args.modality == "question"
        else args.image_question_train_root
    )
    train_summary = train_root / "training_summary.json"
    if not train_summary.exists():
        if train_root.exists():
            raise RuntimeError(
                f"partial training output requires explicit repair rather than overwrite: {train_root}"
            )
        run(
            [
                sys.executable,
                "experiments/train_binary_polar_full10.py",
                "--config",
                str(args.config),
                "--modality",
                args.modality,
                "--objective",
                "duplicated_bce",
                "--output-dir",
                str(train_root),
                "--preflight",
                str(args.training_preflight),
                "--confirm-full10",
            ]
        )

    timeout = args.timeout_hours * 3600.0
    question_summary = args.question_train_root / "training_summary.json"
    image_summary = args.image_question_train_root / "training_summary.json"
    wait_for(question_summary, timeout, "question BCE training")
    wait_for(image_summary, timeout, "image-question BCE training")
    question_checkpoint = selected_checkpoint(args.question_train_root)
    image_checkpoint = selected_checkpoint(args.image_question_train_root)

    contract = args.external_root / "evaluation_contract_v1.json"
    preflight_root = args.external_root / "preflight"
    preflight = preflight_root / "preflight_v1.json"
    if args.coordinator and not preflight.exists():
        freeze_evaluation_contract(
            output=contract,
            config=args.config,
            bundle=args.bundle,
            question_checkpoint=question_checkpoint,
            image_question_checkpoint=image_checkpoint,
        )
        run(
            [
                sys.executable,
                "experiments/evaluate_binary_polar_external.py",
                "--config",
                str(args.config),
                "--question-checkpoint",
                str(question_checkpoint),
                "--image-question-checkpoint",
                str(image_checkpoint),
                "--bundle",
                str(args.bundle),
                "--output-dir",
                str(preflight_root),
                "--mode",
                "preflight",
                "--modality",
                "both",
            ]
        )
    wait_for(preflight, timeout, "joint external evaluation preflight")
    preflight_payload = read_json(preflight)
    expected_hashes = {
        "question": file_sha256(question_checkpoint),
        "image_question": file_sha256(image_checkpoint),
    }
    if preflight_payload.get("passed") is not True or any(
        preflight_payload["checkpoints"][name]["sha256"] != digest
        for name, digest in expected_hashes.items()
    ):
        raise RuntimeError("external preflight is not bound to both selected BCE checkpoints")

    modality_root = args.external_root / args.modality
    shard = modality_root / "shard_000_of_001"
    metadata = shard / "metadata.json"
    if not metadata.exists():
        command = [
            sys.executable,
            "experiments/evaluate_binary_polar_external.py",
            "--config",
            str(args.config),
            "--question-checkpoint",
            str(question_checkpoint),
            "--image-question-checkpoint",
            str(image_checkpoint),
            "--bundle",
            str(args.bundle),
            "--output-dir",
            str(modality_root),
            "--mode",
            "full",
            "--modality",
            args.modality,
            "--preflight-path",
            str(preflight),
            "--num-shards",
            "1",
            "--shard-index",
            "0",
        ]
        if shard.exists():
            command.append("--resume")
        run(command)

    if args.coordinator:
        question_metadata = args.external_root / "question/shard_000_of_001/metadata.json"
        image_metadata = args.external_root / "image_question/shard_000_of_001/metadata.json"
        wait_for(question_metadata, timeout, "question external evaluation")
        wait_for(image_metadata, timeout, "image-question external evaluation")
        analysis_root = args.external_root / "analysis_v1"
        report = Path("reports/binary_polar_full10_bce_external_eval.md")
        if not (analysis_root / "analysis_manifest_v1.json").exists():
            run(
                [
                    sys.executable,
                    "experiments/merge_binary_polar_external_eval.py",
                    "--question-root",
                    str(args.external_root / "question"),
                    "--image-question-root",
                    str(args.external_root / "image_question"),
                    "--output-root",
                    str(analysis_root),
                    "--preflight-path",
                    str(preflight),
                    "--num-shards",
                    "1",
                    "--pope-overlap-cluster",
                    POPE_OVERLAP_CLUSTER,
                    "--bootstrap-draws",
                    "5000",
                    "--report",
                    str(report),
                    "--title",
                    "Full10 POLAR-Style Duplicated-BCE External Evaluation",
                ]
            )

    result = args.external_root / f"{args.modality}_pipeline_complete.json"
    write_json(
        result,
        {
            "schema_version": "binary_polar_full10_bce_pipeline_v1",
            "passed": True,
            "modality": args.modality,
            "training_summary": str(train_summary),
            "selected_checkpoint": str(
                question_checkpoint if args.modality == "question" else image_checkpoint
            ),
            "external_metadata": str(metadata),
            "coordinator": args.coordinator,
        },
    )
    result.with_suffix(result.suffix + ".sha256").write_text(
        f"{file_sha256(result)}  {result.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
