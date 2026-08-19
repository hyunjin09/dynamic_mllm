from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.research_analysis.v2.stage_c_real_donor_amendment import build_amended_match_rows
from experiments.stage_b_reference_likelihood import read_jsonl
from nulls.structured_read import DonorMetadata


AUDIT_PATH = Path("outputs/stage_c/preflight/stage_c_donor_coverage_audit_v1.json")
AUDIT_SHA256 = "fe8dac6d00ac42bb185e6025fc4aac8372e96fe1fcf8dfb528307546ceda5994"
DONOR_POOL_PATH = Path("outputs/stage_c/nulls/real_residual_donor_index_v1.jsonl")
DONOR_POOL_SHA256 = "88ba16adaacd747edacf235707768355853e3833e99521320b46dd6fbf047f25"
SEED_PATH = Path("outputs/stage_c/nulls/deterministic_null_seeds_v1.jsonl")
SEED_SHA256 = "6b9a5cb676aac187be06b03daa9e5c32e30c5a6adc7972a8a1d10374c76addd8"
OUTPUT_PATH = Path("outputs/stage_c/nulls/stage_c_real_residual_match_index_v2.jsonl")
SUMMARY_PATH = Path("outputs/stage_c/nulls/stage_c_real_residual_match_index_v2.summary.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    for path, expected in (
        (AUDIT_PATH, AUDIT_SHA256),
        (DONOR_POOL_PATH, DONOR_POOL_SHA256),
        (SEED_PATH, SEED_SHA256),
    ):
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Frozen input changed: {path}: {observed} != {expected}")
    if OUTPUT_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError("Refusing to overwrite the amended Stage C donor index")

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    donor_rows = read_jsonl(DONOR_POOL_PATH)
    donors = [
        DonorMetadata(
            sample_id=str(row["sample_id"]),
            image_id=str(row["image_id"]),
            residual_norm=float(row["residual_norm"]),
            postvisual_rows=int(row["postvisual_rows"]),
            visual_tokens=int(row["visual_tokens"]),
            prompt_tokens=int(row["prompt_tokens"]),
        )
        for row in donor_rows
    ]
    seed_rows = read_jsonl(SEED_PATH)
    tie_seeds = {
        str(row["id"]): int(row["real_donor_tie_break_seed"]) for row in seed_rows
    }
    rows, summary = build_amended_match_rows(
        audit, donors, tie_seeds, draws=8, expected_target_count=800
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary.update(
        {
            "index_path": str(OUTPUT_PATH),
            "index_sha256": sha256(OUTPUT_PATH),
            "audit_path": str(AUDIT_PATH),
            "audit_sha256": sha256(AUDIT_PATH),
            "donor_pool_path": str(DONOR_POOL_PATH),
            "donor_pool_sha256": sha256(DONOR_POOL_PATH),
            "seed_path": str(SEED_PATH),
            "seed_sha256": sha256(SEED_PATH),
            "partial_stage_c_results_loaded": False,
            "amendment_applied_to_scientific_results": False,
        }
    )
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
