#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools.research_analysis.four_action.three_action_jobs import file_sha256
from tools.research_analysis.four_action.three_action_labels import (
    select_diverse_three_action_routes,
)


def write_jsonl_once(path: Path, rows) -> tuple[int, str]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    digest = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return count, digest


def _sample_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "uid": record["uid"],
        "dataset": record["dataset"],
        "sample_id": record["sample_id"],
        "image_id": record.get("image_id"),
        "image_group_id": record.get("image_group_id"),
        "source_split": record["source_split"],
        "route_type": record["route_type"],
        "epsilon": record["epsilon"],
        "execution_contract_sha256": record["execution_contract"]["contract_sha256"],
        "epsilon_sha256": record["execution_contract"]["epsilon_sha256"],
    }


def _score(row: dict[str, Any], route_type: str) -> float:
    field = "answer_alignment_margin" if route_type == "W2C" else "S_correct"
    return float(row["evaluation"][field])


def _pareto(rows: list[dict[str, Any]], route_type: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        dominated = any(
            other["route_key"] != row["route_key"]
            and int(other["suppression_cost"]) <= int(row["suppression_cost"])
            and _score(other, route_type) >= _score(row, route_type)
            and (
                int(other["suppression_cost"]) < int(row["suppression_cost"])
                or _score(other, route_type) > _score(row, route_type)
            )
            for other in rows
        )
        if not dominated:
            output.append(row)
    return sorted(output, key=lambda row: (int(row["suppression_cost"]), -_score(row, route_type), row["route_key"]))


def build_final_views(
    records: list[dict[str, Any]],
    *,
    route_cap: int,
    diversity_seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    views: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_by_image: dict[str, set[str]] = defaultdict(set)
    for record in sorted(records, key=lambda row: str(row["uid"])):
        metadata = _sample_metadata(record)
        all_off_by_source = {
            str(conversion["source_binary_route_id"]): bool(conversion.get("all_off_seed"))
            for conversion in record["raw_conversions"]
        }
        if metadata["image_group_id"] is not None:
            split_by_image[str(metadata["image_group_id"])].add(str(metadata["source_split"]))
        for conversion in record["raw_conversions"]:
            views["source_conversion_view"].append({**metadata, **conversion})
            for partial in conversion.get("corrective_partial_candidates", []):
                views["corrective_partial_candidates"].append(
                    {**metadata, "source_binary_route_id": conversion["source_binary_route_id"], **partial}
                )
        unique = []
        for route in record["unique_valid_three_action_routes"]:
            all_off_source_ids = sorted(
                source_id
                for source_id in route["source_binary_route_ids"]
                if all_off_by_source.get(str(source_id), False)
            )
            unique.append(
                {
                    **metadata,
                    **route,
                    "all_off_seed": bool(all_off_source_ids),
                    "all_off_source_binary_route_ids": all_off_source_ids,
                }
            )
        views["unique_valid_route_view"].extend(unique)
        target_view = "w2c_corrective_training_view" if record["route_type"] == "W2C" else "c2c_alignment_training_view"
        views[target_view].extend(unique)
        canonical = record.get("canonical_three_action_route")
        if canonical is None:
            continue
        canonical_base = next(
            row for row in unique if row["route_key"] == canonical["route_key"]
        )
        canonical_row = {**canonical_base, **canonical}
        views["canonical_routes"].append(canonical_row)
        max_score = min(
            unique,
            key=lambda row: (-_score(row, record["route_type"]), int(row["suppression_cost"]), row["route_key"]),
        )
        views["max_score_routes"].append(max_score)
        pareto = _pareto(unique, record["route_type"])
        views["pareto_route_view"].extend(pareto)
        selected = select_diverse_three_action_routes(
            unique,
            limit=route_cap,
            seed=diversity_seed,
            uid=str(record["uid"]),
            canonical_route_key=str(canonical["route_key"]),
        )
        views["combined_training_manifest"].extend(selected)
    leakage = {
        image: sorted(splits) for image, splits in split_by_image.items() if len(splits) > 1
    }
    summary = {
        "schema_version": "three_action_answer_aligned_finalization_summary_v1",
        "sample_records": len(records),
        "view_counts": {key: len(value) for key, value in sorted(views.items())},
        "route_type_counts": dict(Counter(row["route_type"] for row in records)),
        "samples_without_positive_route": sum(
            not bool(row["unique_valid_three_action_routes"]) for row in records
        ),
        "image_group_split_leakage": leakage,
        "checks": {
            "no_image_group_split_leakage": not leakage,
            "all_training_rows_correct": all(
                bool(row["evaluation"].get("correct"))
                for row in views["combined_training_manifest"]
            ),
            "canonical_in_training_view": all(
                any(
                    candidate["uid"] == row["uid"] and candidate["route_key"] == row["route_key"]
                    for candidate in views["combined_training_manifest"]
                )
                for row in views["canonical_routes"]
            ),
        },
    }
    summary["passed"] = all(summary["checks"].values())
    return dict(views), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize three-action training and analysis views.")
    parser.add_argument("--config", type=Path, default=Path("configs/three_action_label_conversion.yaml"))
    parser.add_argument("--audit", type=Path, default=Path("analysis/three_action_answer_aligned_label_conversion/full_integrity_audit_v1.json"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not bool(audit.get("passed")):
        raise RuntimeError("full integrity audit must pass before finalization")
    record_paths = sorted((Path(config["output_root"]) / "full" / "records").glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in record_paths]
    views, summary = build_final_views(
        records,
        route_cap=int(config["training_view"]["route_cap"]),
        diversity_seed=int(config["training_view"]["diversity_seed"]),
    )
    if len(views.get("source_conversion_view", [])) != int(audit["source_routes"]):
        raise RuntimeError("source conversion view does not match audited source-route count")
    output_root = Path(config["output_root"]) / "final"
    files = {}
    for name in (
        "source_conversion_view",
        "unique_valid_route_view",
        "w2c_corrective_training_view",
        "c2c_alignment_training_view",
        "combined_training_manifest",
        "canonical_routes",
        "max_score_routes",
        "pareto_route_view",
        "corrective_partial_candidates",
    ):
        count, digest = write_jsonl_once(output_root / f"{name}.jsonl", views.get(name, []))
        files[name] = {"rows": count, "sha256": digest}
    summary["files"] = files
    summary_path = output_root / "finalization_summary_v1.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.with_suffix(summary_path.suffix + ".sha256").write_text(
        f"{file_sha256(summary_path)}  {summary_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
