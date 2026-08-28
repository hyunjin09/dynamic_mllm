#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from tools.research_analysis.four_action.route_conditioned import build_anchor_candidate_rows


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_once(path: Path, payload: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = sha256_bytes(payload)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze route-conditioned anchor candidates.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_route_conditioned.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cohort_path = Path(config["source_cohort_manifest"])
    eligibility_path = Path(config["source_eligibility_manifest"])
    output_path = Path(config["candidate_manifest"])
    rows = build_anchor_candidate_rows(read_jsonl(cohort_path), read_jsonl(eligibility_path))
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    digest = write_once(output_path, payload)
    distances = [int(row["historical_minimum_off_count"]) for row in rows]
    summary = {
        "schema_version": "route_conditioned_anchor_candidate_summary_v1",
        "candidate_manifest": str(output_path),
        "candidate_manifest_sha256": digest,
        "source_cohort_manifest": str(cohort_path),
        "source_cohort_manifest_sha256": sha256_bytes(cohort_path.read_bytes()),
        "source_eligibility_manifest": str(eligibility_path),
        "source_eligibility_manifest_sha256": sha256_bytes(eligibility_path.read_bytes()),
        "sample_count": len(rows),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "historical_minimum_off_count_distribution": dict(sorted(Counter(distances).items())),
        "historical_expected_new_cells_3k": 3 * sum(distances),
        "selection_order": [
            "ascending_hamming_distance_from_full",
            "descending_cached_evaluator_score",
            "ascending_route_id",
            "ascending_mask_key",
        ],
        "anchor_frozen": False,
        "anchor_freeze_condition": "first current unified-route correct cached candidate",
    }
    summary_path = output_path.with_name("anchor_candidate_summary.json")
    write_once(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
