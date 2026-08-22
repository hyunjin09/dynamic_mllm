#!/usr/bin/env python3
"""Run one CAP-NLL5 train, executed-validation selection, and external eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_binary_polar import file_sha256


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, choices=(26, 24), required=True)
    parser.add_argument(
        "--bundle", type=Path, default=Path("eval/reference/shared_prefix_eval_20260812")
    )
    args = parser.parse_args()
    cap = args.cap
    config = Path(f"configs/binary_cap{cap}_nll5_execval_v1.yaml")
    readiness_path = Path("outputs/binary_cap_nll5_v1/audits/training_readiness_v1.json")
    readiness = read_json(readiness_path)
    expected = readiness["configs"][str(cap)]
    if (
        readiness.get("passed") is not True
        or expected["path"] != str(config)
        or expected["sha256"] != file_sha256(config)
    ):
        raise RuntimeError("CAP-NLL5 pipeline is not bound to passed readiness")

    root = Path(f"outputs/binary_cap_nll5_v1/cap{cap}")
    preflight = Path(f"outputs/binary_cap_nll5_v1/preflight/cap{cap}/training_preflight_v1.json")
    if not preflight.exists():
        run([
            sys.executable, "experiments/preflight_binary_polar_full10.py",
            "--config", str(config), "--output", str(preflight),
        ])
    preflight_payload = read_json(preflight)
    if (
        preflight_payload.get("passed") is not True
        or preflight_payload.get("objective") != "exact_set_nll"
        or preflight_payload.get("config_sha256") != file_sha256(config)
    ):
        raise RuntimeError("CAP-NLL5 training preflight is invalid or stale")

    summary = root / "training_summary.json"
    if not summary.exists():
        if root.exists():
            raise RuntimeError("partial CAP-NLL5 training requires explicit repair")
        run([
            sys.executable, "experiments/train_binary_polar_full10.py",
            "--config", str(config), "--modality", "image_question",
            "--objective", "exact_set_nll", "--output-dir", str(root),
            "--preflight", str(preflight), "--confirm-full10",
        ])
    training = read_json(summary)
    if training.get("passed") is not True or int(training["epochs_completed"]) != 5:
        raise RuntimeError("CAP-NLL5 training did not complete five epochs")

    validation_output = root / "executed_validation_v1.json"
    selection_output = root / "selected_checkpoint_v1.json"
    if not validation_output.exists() or not selection_output.exists():
        command = [
            sys.executable, "experiments/evaluate_binary_cap_validation_epochs.py",
            "--config", str(config), "--training-root", str(root),
            "--output", str(validation_output),
            "--selection-output", str(selection_output),
        ]
        if (root / "validation_parts_v1").exists():
            command.append("--resume")
        run(command)
    selection = read_json(selection_output)
    checkpoint = Path(selection["checkpoint"])
    if (
        selection.get("passed") is not True
        or selection.get("objective") != "exact_set_nll"
        or file_sha256(checkpoint) != selection["checkpoint_sha256"]
    ):
        raise RuntimeError("executed-validation checkpoint selection is invalid")

    external_root = Path(f"outputs/binary_cap_nll5_v1/external_eval/cap{cap}")
    external_preflight = external_root / "preflight/preflight_v1.json"
    checkpoint_args = [
        "--question-checkpoint", str(checkpoint),
        "--image-question-checkpoint", str(checkpoint),
    ]
    if not external_preflight.exists():
        run([
            sys.executable, "experiments/evaluate_binary_polar_external.py",
            "--config", str(config), *checkpoint_args,
            "--bundle", str(args.bundle),
            "--output-dir", str(external_preflight.parent),
            "--mode", "preflight", "--modality", "image_question",
        ])
    preflight_external = read_json(external_preflight)
    if (
        preflight_external.get("passed") is not True
        or preflight_external["checkpoints"]["image_question"]["sha256"]
        != file_sha256(checkpoint)
    ):
        raise RuntimeError("external preflight is stale or failed")

    shard_root = external_root / "image_question"
    metadata = shard_root / "shard_000_of_001/metadata.json"
    if not metadata.exists():
        command = [
            sys.executable, "experiments/evaluate_binary_polar_external.py",
            "--config", str(config), *checkpoint_args,
            "--bundle", str(args.bundle), "--output-dir", str(shard_root),
            "--mode", "full", "--modality", "image_question",
            "--preflight-path", str(external_preflight),
            "--num-shards", "1", "--shard-index", "0",
        ]
        if (shard_root / "shard_000_of_001").exists():
            command.append("--resume")
        run(command)

    analysis = external_root / "analysis_v1"
    report = Path(f"reports/binary_cap{cap}_nll5_external_eval.md")
    if not (analysis / "analysis_manifest_v1.json").exists():
        run([
            sys.executable, "experiments/summarize_binary_cap_external_eval.py",
            "--input-root", str(shard_root),
            "--preflight-path", str(external_preflight),
            "--output-root", str(analysis), "--cap", str(cap),
            "--checkpoint", str(checkpoint), "--report", str(report),
        ])
    result = Path(f"outputs/binary_cap_nll5_v1/cap{cap}_pipeline_complete.json")
    result.write_text(
        json.dumps(
            {
                "schema_version": "binary_cap_nll5_train_eval_pipeline_v1",
                "passed": True,
                "cap": cap,
                "selected_epoch": int(selection["epoch"]),
                "selected_checkpoint": {
                    "path": str(checkpoint), "sha256": file_sha256(checkpoint)
                },
                "selection": {
                    "path": str(selection_output), "sha256": file_sha256(selection_output)
                },
                "external_analysis": {
                    "path": str(analysis / "external_analysis_v1.json"),
                    "sha256": file_sha256(analysis / "external_analysis_v1.json"),
                },
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    result.with_suffix(result.suffix + ".sha256").write_text(
        f"{file_sha256(result)}  {result.name}\n", encoding="utf-8"
    )
    print(json.dumps({"passed": True, "cap": cap, "selected_epoch": selection["epoch"]}))


if __name__ == "__main__":
    main()
