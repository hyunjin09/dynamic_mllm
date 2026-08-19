#!/usr/bin/env python3
"""Canonicalize interrupted ALL-ON partial rows to their live baseline."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=("question", "image_question"), required=True)
    args = parser.parse_args()

    parts = sorted(args.shard_dir.glob("part_*.jsonl"))
    if not parts or (args.shard_dir / "metadata.json").exists():
        raise RuntimeError("repair requires a nonempty interrupted, incomplete shard")
    backup = args.shard_dir / "pre_live_baseline_repair"
    if backup.exists():
        raise FileExistsError(f"repair backup already exists: {backup}")
    backup.mkdir()
    ledger = []
    repaired = 0
    for path in parts:
        original_hash = digest(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(row[args.modality]["mask_key"] != "1" * 28 for row in rows):
            raise RuntimeError(f"partial repair only supports verified ALL-ON rows: {path}")
        shutil.copy2(path, backup / path.name)
        for row in rows:
            executed = row[args.modality]
            row["reference_cache_prediction"] = row["baseline_prediction"]
            row["reference_cache_score"] = row["baseline_score"]
            row["reference_cache_correct"] = row["baseline_correct"]
            row["reference_cache_exact_match"] = bool(
                executed["prediction"] == row["baseline_prediction"]
                and executed["score"] == row["baseline_score"]
                and executed["correct"] == row["baseline_correct"]
            )
            row["baseline_prediction"] = executed["prediction"]
            row["baseline_score"] = executed["score"]
            row["baseline_correct"] = executed["correct"]
            row["baseline_generated_ids"] = executed["generated_ids"]
            row["baseline_source"] = "predicted_all_on_live_execution"
            executed["execution_source"] = "live_binary_executor"
            repaired += 1
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        temporary.replace(path)
        ledger.append(
            {
                "part": path.name,
                "records": len(rows),
                "original_sha256": original_hash,
                "backup_sha256": digest(backup / path.name),
                "repaired_sha256": digest(path),
            }
        )
    amendment = {
        "schema_version": "binary_polar_external_partial_live_baseline_repair_v1",
        "modality": args.modality,
        "repair_basis": "predicted route was exactly ALL-ON; its live output is the current live ALL-ON baseline",
        "scientific_outcomes_used_for_selection": False,
        "records": repaired,
        "parts": ledger,
    }
    amendment_path = args.shard_dir / "partial_live_baseline_repair_v1.json"
    amendment_path.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    amendment_path.with_suffix(amendment_path.suffix + ".sha256").write_text(
        f"{digest(amendment_path)}  {amendment_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({"modality": args.modality, "records": repaired, "backup": str(backup)}))


if __name__ == "__main__":
    main()
