#!/usr/bin/env python3
"""Run one frozen Pareto Image+Question train-to-external-eval pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_binary_polar import file_sha256


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True)


def selected_checkpoint(training_root: Path, objective: str) -> tuple[Path, int]:
    summary = read_json(training_root / "training_summary.json")
    if summary.get("passed") is not True or summary.get("objective") != objective:
        raise RuntimeError("training summary does not match the frozen Pareto objective")
    epoch = int(summary["selections"]["best_hit_at_1"])
    checkpoint = training_root / f"epoch_{epoch:02d}/checkpoint.pt"
    expected = {
        int(row["epoch"]): str(row["checkpoint_sha256"])
        for row in summary["checkpoints"]
    }[epoch]
    if not checkpoint.exists() or file_sha256(checkpoint) != expected:
        raise RuntimeError("selected Pareto checkpoint failed checksum verification")
    return checkpoint, epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--objective", choices=("duplicated_bce", "exact_set_nll"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    readiness_path = Path("outputs/binary_pareto_v1/audits/training_readiness_v1.json")
    readiness = read_json(readiness_path)
    expected_config = readiness["configs"][args.objective]
    if (
        readiness.get("passed") is not True
        or Path(expected_config["path"]) != args.config
        or expected_config["sha256"] != file_sha256(args.config)
    ):
        raise RuntimeError("Pareto pipeline is not bound to a passed static readiness gate")

    run_name = "bce" if args.objective == "duplicated_bce" else "nll"
    training_root = args.output_root / "training" / run_name
    training_preflight = args.output_root / "preflight" / run_name / "training_preflight_v1.json"
    if not training_preflight.exists():
        run(
            [
                sys.executable,
                "experiments/preflight_binary_polar_full10.py",
                "--config",
                str(args.config),
                "--output",
                str(training_preflight),
            ]
        )
    preflight_payload = read_json(training_preflight)
    if (
        preflight_payload.get("passed") is not True
        or preflight_payload.get("objective") != args.objective
        or preflight_payload.get("config_sha256") != file_sha256(args.config)
    ):
        raise RuntimeError("Pareto training preflight is invalid or stale")

    summary_path = training_root / "training_summary.json"
    if not summary_path.exists():
        if training_root.exists():
            raise RuntimeError("partial Pareto training output requires explicit repair")
        run(
            [
                sys.executable,
                "experiments/train_binary_polar_full10.py",
                "--config",
                str(args.config),
                "--modality",
                "image_question",
                "--objective",
                args.objective,
                "--output-dir",
                str(training_root),
                "--preflight",
                str(training_preflight),
                "--confirm-full10",
            ]
        )
    checkpoint, epoch = selected_checkpoint(training_root, args.objective)

    external_root = args.output_root / "external_eval" / run_name
    external_preflight_root = external_root / "preflight"
    external_preflight = external_preflight_root / "preflight_v1.json"
    checkpoint_args = [
        "--question-checkpoint",
        str(checkpoint),
        "--image-question-checkpoint",
        str(checkpoint),
    ]
    if not external_preflight.exists():
        run(
            [
                sys.executable,
                "experiments/evaluate_binary_polar_external.py",
                "--config",
                str(args.config),
                *checkpoint_args,
                "--bundle",
                str(args.bundle),
                "--output-dir",
                str(external_preflight_root),
                "--mode",
                "preflight",
                "--modality",
                "image_question",
            ]
        )
    external_preflight_payload = read_json(external_preflight)
    if (
        external_preflight_payload.get("passed") is not True
        or external_preflight_payload["checkpoints"]["image_question"]["sha256"]
        != file_sha256(checkpoint)
    ):
        raise RuntimeError("external preflight is not bound to the selected checkpoint")

    shard_root = external_root / "image_question"
    metadata = shard_root / "shard_000_of_001/metadata.json"
    if not metadata.exists():
        command = [
            sys.executable,
            "experiments/evaluate_binary_polar_external.py",
            "--config",
            str(args.config),
            *checkpoint_args,
            "--bundle",
            str(args.bundle),
            "--output-dir",
            str(shard_root),
            "--mode",
            "full",
            "--modality",
            "image_question",
            "--preflight-path",
            str(external_preflight),
            "--num-shards",
            "1",
            "--shard-index",
            "0",
        ]
        if (shard_root / "shard_000_of_001").exists():
            command.append("--resume")
        run(command)

    analysis_root = external_root / "analysis_v1"
    if not (analysis_root / "analysis_manifest_v1.json").exists():
        run(
            [
                sys.executable,
                "experiments/summarize_binary_pareto_external_eval.py",
                "--input-root",
                str(shard_root),
                "--preflight-path",
                str(external_preflight),
                "--output-root",
                str(analysis_root),
                "--objective",
                args.objective,
                "--checkpoint",
                str(checkpoint),
                "--report",
                str(args.report),
            ]
        )
    result = args.output_root / f"{run_name}_pipeline_complete.json"
    write_json(
        result,
        {
            "schema_version": "binary_pareto_train_eval_pipeline_v1",
            "passed": True,
            "objective": args.objective,
            "modality": "image_question",
            "selected_epoch": epoch,
            "selected_checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
            "training_summary": {"path": str(summary_path), "sha256": file_sha256(summary_path)},
            "external_analysis": {
                "path": str(analysis_root / "external_analysis_v1.json"),
                "sha256": file_sha256(analysis_root / "external_analysis_v1.json"),
            },
        },
    )
    result.with_suffix(result.suffix + ".sha256").write_text(
        f"{file_sha256(result)}  {result.name}\n", encoding="utf-8"
    )
    print(json.dumps({"passed": True, "objective": args.objective, "result": str(result)}))


if __name__ == "__main__":
    main()
