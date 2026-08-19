#!/usr/bin/env python3
"""Freeze P11 label geometry and the outcome-blind bounded execution subset."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binary_policy.p11 import summarize_label_geometry
from experiments.train_binary_polar import file_sha256


BENCHMARKS = ("gqa", "textvqa", "chartqa")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_execution_rows(rows: list[dict], *, per_stratum: int, seed: int) -> list[dict]:
    selected: list[dict] = []
    for benchmark in BENCHMARKS:
        strata = {
            "full_correct": [
                row
                for row in rows
                if row["benchmark"] == benchmark and row["current_all_on_status"] == "correct"
            ],
            "full_wrong_mcts_fixable": [
                row
                for row in rows
                if row["benchmark"] == benchmark
                and row["current_all_on_status"] == "wrong"
                and row.get("valid_routes")
            ],
        }
        for stratum, candidates in strata.items():
            candidates.sort(
                key=lambda row: (
                    sha256(f"{seed}:p11-execution:{benchmark}:{stratum}:{row['uid']}".encode()).hexdigest(),
                    row["uid"],
                )
            )
            if len(candidates) < per_stratum:
                raise RuntimeError(f"insufficient {benchmark}/{stratum} rows")
            selected.extend(
                {"uid": row["uid"], "benchmark": benchmark, "stratum": stratum}
                for row in candidates[:per_stratum]
            )
    return sorted(selected, key=lambda row: row["uid"])


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen P11 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--p10-smoke-manifest", required=True)
    parser.add_argument("--geometry-output", required=True)
    parser.add_argument("--p11-smoke-output", required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--execution-per-stratum", type=int, default=10)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    p10_path = Path(args.p10_smoke_manifest)
    rows = read_jsonl(manifest_path)
    positive = [row for row in rows if row.get("valid_routes")]
    geometry = {
        "schema_version": "binary_polar_p11_label_geometry_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "selected_route_policy": "frozen diverse maximum 50 masks per positive input",
        "all_records": len(rows),
        "positive_records": len(positive),
        "zero_valid_records": len(rows) - len(positive),
        "record_counts": dict(sorted(Counter((row["split"], row["benchmark"]) for row in rows).items())),
        "positive_record_counts": dict(
            sorted(Counter((row["split"], row["benchmark"]) for row in positive).items())
        ),
        "splits": {},
    }
    # JSON cannot serialize tuple keys from Counter.
    geometry["record_counts"] = {
        f"{split}:{benchmark}": count
        for (split, benchmark), count in sorted(
            Counter((row["split"], row["benchmark"]) for row in rows).items()
        )
    }
    geometry["positive_record_counts"] = {
        f"{split}:{benchmark}": count
        for (split, benchmark), count in sorted(
            Counter((row["split"], row["benchmark"]) for row in positive).items()
        )
    }
    for split in ("train", "validation"):
        split_rows = [row for row in positive if row["split"] == split]
        geometry["splits"][split] = {
            "overall": summarize_label_geometry(split_rows),
            "by_benchmark": {
                benchmark: summarize_label_geometry(
                    [row for row in split_rows if row["benchmark"] == benchmark]
                )
                for benchmark in BENCHMARKS
            },
        }

    p10 = json.loads(p10_path.read_text(encoding="utf-8"))
    execution_rows = select_execution_rows(
        [row for row in rows if row["split"] == "validation"],
        per_stratum=args.execution_per_stratum,
        seed=args.seed,
    )
    p11_smoke = {
        "schema_version": "binary_polar_p11_smoke_manifest_v1",
        "selection_seed": args.seed,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "p10_smoke_manifest": str(p10_path),
        "p10_smoke_manifest_sha256": file_sha256(p10_path),
        "train_positive_uids": p10["train_positive_uids"],
        "validation_positive_uids": p10["validation_positive_uids"],
        "execution_rows": execution_rows,
        "execution_validation_uids": [row["uid"] for row in execution_rows],
        "execution_records": len(execution_rows),
        "execution_stratum_counts": dict(Counter(row["stratum"] for row in execution_rows)),
        "execution_benchmark_counts": dict(Counter(row["benchmark"] for row in execution_rows)),
        "outcome_blind_statement": (
            "Identities were frozen from dataset, split, current FULL-status metadata, MCTS-valid-set "
            "presence, and a deterministic hash before P11 predictor outcomes or executions were inspected."
        ),
    }

    write_json(Path(args.geometry_output), geometry)
    write_json(Path(args.p11_smoke_output), p11_smoke)


if __name__ == "__main__":
    main()
