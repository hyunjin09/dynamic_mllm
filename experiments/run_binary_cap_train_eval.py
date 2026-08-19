#!/usr/bin/env python3
"""Run one frozen cap Image+Question training and external-evaluation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_binary_polar import file_sha256


PROJECT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True)


def selected_checkpoint(root: Path) -> tuple[Path, int]:
    summary = read_json(root / "training_summary.json")
    if summary.get("passed") is not True or summary.get("objective") != "duplicated_bce":
        raise RuntimeError("cap training summary is invalid")
    epoch = int(summary["selections"]["best_hit_at_1"])
    path = root / f"epoch_{epoch:02d}/checkpoint.pt"
    expected = {int(row["epoch"]): row["checkpoint_sha256"] for row in summary["checkpoints"]}[epoch]
    if file_sha256(path) != expected:
        raise RuntimeError("selected cap checkpoint checksum mismatch")
    return path, epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, choices=(24, 22, 20, 18), required=True)
    parser.add_argument("--bundle", type=Path, default=Path("eval/reference/shared_prefix_eval_20260812"))
    args = parser.parse_args()
    cap = args.cap
    config = Path(f"configs/binary_cap{cap}_full10_bce_v1.yaml")
    readiness_path = Path("outputs/binary_cap_sweep_v1/audits/training_readiness_v1.json")
    readiness = read_json(readiness_path)
    expected = readiness["configs"][str(cap)]
    if readiness.get("passed") is not True or Path(expected["path"]) != config or expected["sha256"] != file_sha256(config):
        raise RuntimeError("cap pipeline is not bound to the passed readiness gate")

    training_root = Path(f"outputs/binary_cap_sweep_v1/cap{cap}")
    preflight_root = Path(f"outputs/binary_cap_sweep_v1/preflight/cap{cap}")
    preflight = preflight_root / "training_preflight_v1.json"
    if not preflight.exists():
        run([sys.executable, "experiments/preflight_binary_polar_full10.py", "--config", str(config), "--output", str(preflight)])
    if read_json(preflight).get("passed") is not True:
        raise RuntimeError("cap runtime preflight failed")
    summary = training_root / "training_summary.json"
    if not summary.exists():
        if training_root.exists():
            raise RuntimeError("partial cap training output requires explicit repair")
        run([
            sys.executable, "experiments/train_binary_polar_full10.py",
            "--config", str(config), "--modality", "image_question",
            "--objective", "duplicated_bce", "--output-dir", str(training_root),
            "--preflight", str(preflight), "--confirm-full10",
        ])
    checkpoint, epoch = selected_checkpoint(training_root)

    external_root = Path(f"outputs/binary_cap_sweep_v1/external_eval/cap{cap}")
    external_preflight_root = external_root / "preflight"
    external_preflight = external_preflight_root / "preflight_v1.json"
    checkpoint_args = ["--question-checkpoint", str(checkpoint), "--image-question-checkpoint", str(checkpoint)]
    if not external_preflight.exists():
        run([
            sys.executable, "experiments/evaluate_binary_polar_external.py", "--config", str(config),
            *checkpoint_args, "--bundle", str(args.bundle), "--output-dir", str(external_preflight_root),
            "--mode", "preflight", "--modality", "image_question",
        ])
    payload = read_json(external_preflight)
    if payload.get("passed") is not True or payload["checkpoints"]["image_question"]["sha256"] != file_sha256(checkpoint):
        raise RuntimeError("cap external preflight is stale or failed")
    shard_root = external_root / "image_question"
    metadata = shard_root / "shard_000_of_001/metadata.json"
    if not metadata.exists():
        command = [
            sys.executable, "experiments/evaluate_binary_polar_external.py", "--config", str(config),
            *checkpoint_args, "--bundle", str(args.bundle), "--output-dir", str(shard_root),
            "--mode", "full", "--modality", "image_question", "--preflight-path", str(external_preflight),
            "--num-shards", "1", "--shard-index", "0",
        ]
        if (shard_root / "shard_000_of_001").exists():
            command.append("--resume")
        run(command)
    analysis_root = external_root / "analysis_v1"
    report = Path(f"reports/binary_cap{cap}_external_eval.md")
    if not (analysis_root / "analysis_manifest_v1.json").exists():
        run([
            sys.executable, "experiments/summarize_binary_cap_external_eval.py",
            "--input-root", str(shard_root), "--preflight-path", str(external_preflight),
            "--output-root", str(analysis_root), "--cap", str(cap), "--checkpoint", str(checkpoint),
            "--report", str(report),
        ])
    result = Path(f"outputs/binary_cap_sweep_v1/cap{cap}_pipeline_complete.json")
    result.write_text(json.dumps({
        "schema_version": "binary_cap_train_eval_pipeline_v1", "passed": True, "cap": cap,
        "selected_epoch": epoch, "selected_checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        "training_summary": {"path": str(summary), "sha256": file_sha256(summary)},
        "external_analysis": {"path": str(analysis_root / "external_analysis_v1.json"), "sha256": file_sha256(analysis_root / "external_analysis_v1.json")},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result.with_suffix(result.suffix + ".sha256").write_text(f"{file_sha256(result)}  {result.name}\n", encoding="utf-8")
    print(json.dumps({"passed": True, "cap": cap, "selected_epoch": epoch, "result": str(result)}))


if __name__ == "__main__":
    main()
