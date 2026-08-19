from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from audit.sample_manifest import manifest_path, resolve_local_image, write_jsonl
from audit.stage_b_manifest import select_balanced_unique_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the deterministic Stage B candidate manifest.")
    parser.add_argument("--config", default="configs/stage_b.yaml")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_candidate(
    row: dict[str, Any], source_path: Path, source_index: int, pool_root: Path
) -> dict[str, Any]:
    source_record = row.get("source_manifest_record") or {}
    metadata = source_record.get("metadata") or {}
    types = metadata.get("types") or {}
    return {
        "id": row["id"],
        "sample_id": row["sample_id"],
        "benchmark": row["benchmark"],
        "inherited_bucket": row["bucket"],
        "inherited_difficulty": "easy" if row["bucket"] == "complete_correct" else "hard",
        "question": row["question"],
        "prompt": row["prompt"],
        "answer": row["answer"],
        "all_answer_norms": row.get("all_answer_norms"),
        "metric_name": row["metric_name"],
        "inherited_prediction": row.get("prediction"),
        "inherited_score": row.get("score"),
        "source_asset_id": row["source_asset_id"],
        "local_image_path": str(resolve_local_image(pool_root, row)),
        "source_manifest_path": str(source_path),
        "source_manifest_index": source_index,
        "gqa_semantic_str": metadata.get("semantic_str"),
        "gqa_question_type": types.get("detailed"),
        "recorded_visual_token_count": row.get("visual_token_count"),
    }


def execute(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pool_root = Path(config["source_pool"])
    excluded = {row["id"] for row in read_jsonl(Path(config["exclude_manifest"]))}
    rows_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for benchmark in config["benchmarks"]:
        for bucket in config["buckets"]:
            path = manifest_path(pool_root, benchmark, bucket)
            source_rows = read_jsonl(path)
            rows_by_cell[(benchmark, bucket)] = [
                compact_candidate(row, path, index, pool_root)
                for index, row in enumerate(source_rows)
            ]

    selected = select_balanced_unique_assets(
        rows_by_cell,
        quota_per_cell=int(config["quota_per_cell"]),
        seed=int(config["seed"]),
        excluded_ids=excluded,
    )
    expected_count = int(config["quota_per_cell"]) * len(rows_by_cell)
    if int(config["sample_count"]) != expected_count or len(selected) != expected_count:
        raise ValueError(
            f"Stage B sample-count contract mismatch: config={config['sample_count']}, "
            f"expected={expected_count}, selected={len(selected)}"
        )
    manifest_path_out = Path(config["candidate_manifest"])
    write_jsonl(manifest_path_out, selected)

    counts = Counter(row["selection_cell"] for row in selected)
    manifest_sha256 = hashlib.sha256(manifest_path_out.read_bytes()).hexdigest()
    plan_sha256 = hashlib.sha256(Path(config["source_plan"]).read_bytes()).hexdigest()
    audit = {
        "stage": "B",
        "status": "candidate_manifest_only",
        "source_plan_sha256": plan_sha256,
        "approved_sample_count_exception": {
            "source_plan_suggested": "100-200",
            "user_approved": 400,
            "approved_at": "2026-08-04",
        },
        "seed": int(config["seed"]),
        "candidate_count": len(selected),
        "counts_by_inherited_cell": dict(sorted(counts.items())),
        "unique_selection_assets": len({row["selection_asset_key"] for row in selected}),
        "unique_sample_ids": len({row["id"] for row in selected}),
        "excluded_stage_a_ids": len(excluded),
        "stage_a_overlap_count": len({row["id"] for row in selected} & excluded),
        "candidate_manifest_sha256": manifest_sha256,
        "pinned_full_relabeling_complete": False,
        "option_protocol_status": config["option_protocol_status"],
        "layer_grid_status": config["layer_grid_status"],
        "intervention_sweep_executed": False,
    }
    audit_path = Path(config["candidate_audit"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    execute(Path(parse_args().config))
