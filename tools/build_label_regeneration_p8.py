#!/usr/bin/env python3
"""Build checksum-bound P8 supervision views from the frozen 8K route cache."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_regeneration.derived import (
    canonical_segment_targets,
    select_diverse_valid_routes,
    single_best_valid_route,
)


EXPECTED = {"gqa": 4000, "textvqa": 2000, "chartqa": 2000}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def checksum_sidecar(path: Path) -> str:
    value = digest(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{value}  {path.name}\n", encoding="utf-8")
    return value


def load_verified_record(index_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read each large raw record once, verify it, and decode it."""
    raw_path = Path(index_row["record_path"])
    payload = raw_path.read_bytes()
    if sha256(payload).hexdigest() != index_row["record_sha256"]:
        raise RuntimeError(f"raw record checksum mismatch: {raw_path}")
    record = json.loads(payload)
    # MCTS node/simulation traces dominate the 21 GB cache but are not inputs
    # to P8. Keep them in the immutable raw file and return only fields needed
    # for derived supervision, avoiding large inter-process transfers.
    return index_row, {
        "sample": record["sample"],
        "candidate_executions": record["candidate_executions"],
        "successful_route_ids": record["successful_route_ids"],
        "best_sparse_success_route_id": record.get("best_sparse_success_route_id"),
    }


def compact_route(route: dict[str, Any], minimum_mask: tuple[int, ...]) -> dict[str, Any]:
    mask = tuple(int(bit) for bit in route["visual_on_mask"])
    return {
        "route_id": route["route_id"],
        "key": route["mask_key"],
        "mask": list(mask),
        "num_visual_on_layers": int(route["num_visual_on_layers"]),
        "num_visual_off_layers": int(route["num_visual_off_layers"]),
        "num_transitions": int(route["num_transitions"]),
        "hamming_distance_to_all_on": int(route["hamming_distance_to_all_on"]),
        "hamming_distance_to_minimum_route": sum(a != b for a, b in zip(mask, minimum_mask)),
        "score": float(route["score"]),
        "reward": float(route["reward"]),
        "correctness_threshold": float(route["correctness_threshold"]),
    }


def base_row(sample: dict[str, Any], split: dict[str, Any], record_sha256: str) -> dict[str, Any]:
    return {
        "uid": sample["uid"],
        "sample_id": sample["sample_id"],
        "benchmark": sample["benchmark"],
        "split": split["split"],
        "split_group": sample["image_group_id"],
        "image_group_id": sample["image_group_id"],
        "question": sample["question"],
        "prompt": sample["prompt"],
        "answer": sample["answer"],
        "all_answer_norms": sample.get("all_answer_norms"),
        "image_path": sample["local_image_path"],
        "image_sha256": sample.get("image_content_sha256"),
        "source_asset_id": sample.get("source_asset_id"),
        "historical_all_on_status": sample["historical_all_on_status"],
        "current_all_on_status": sample["current_all_on_status"],
        "raw_record_sha256": record_sha256,
    }


def render_report(audit: dict[str, Any]) -> str:
    lines = [
        "# Label Regeneration P8 Derived Supervision",
        "",
        "Status: **PASS**",
        "",
        "P8 derives training views from the checksum-verified raw cache and frozen P7 split. "
        "No route was re-executed and the raw cache was not modified.",
        "",
        "## Counts",
        "",
        "| Dataset | Split | Samples | Positive samples | Raw valid routes | Selected valid routes | Evaluated ranking routes |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("gqa", "textvqa", "chartqa"):
        for split in ("train", "validation"):
            row = audit["counts"][dataset][split]
            lines.append(
                f"| {dataset.upper()} | {split} | {row['samples']:,} | {row['positive_samples']:,} | "
                f"{row['raw_valid_routes']:,} | {row['selected_valid_routes']:,} | "
                f"{row['evaluated_routes']:,} |"
            )
    totals = audit["totals"]
    lines += [
        "",
        f"- Samples: `{totals['samples']:,}`; positive: `{totals['positive_samples']:,}`; "
        f"zero-positive: `{totals['zero_positive_samples']:,}`.",
        f"- Raw valid routes: `{totals['raw_valid_routes']:,}`; selected: "
        f"`{totals['selected_valid_routes']:,}`; evaluated positive+negative ranking routes: "
        f"`{totals['evaluated_routes']:,}`.",
        f"- Samples capped from more than 50 routes: `{totals['capped_samples']:,}`.",
        "",
        "## Frozen selection",
        "",
        "The max-50 view first includes the minimum-ON route and valid ALL-OFF/ALL-ON anchors. "
        "It then balances exact ON-count strata and greedily maximizes minimum Hamming distance, "
        "transition-count coverage, and transition distance, with a seeded digest used only for ties.",
        "Duplicated BCE and exact set-NLL consume the same `binary_predictor_manifest_v1.jsonl` "
        "and therefore the identical selected masks and equal within-sample weights.",
        "",
        "## Integrity",
        "",
        f"- Raw records checksum verified: `{audit['integrity']['raw_records_checksum_verified']:,}`.",
        f"- P7 cross-split image groups: `{audit['integrity']['cross_split_image_groups']}`.",
        f"- Selected masks missing from raw valid sets: `{audit['integrity']['selected_not_raw_valid']}`.",
        f"- Minimum-route/anchor violations: `{audit['integrity']['anchor_violations']}`.",
        f"- Shared objective route-set digest: `{audit['integrity']['selected_route_set_sha256']}`.",
        "- P9 and predictor training were not executed.",
        "",
        "## Artifacts",
        "",
    ]
    for name, item in audit["artifacts"].items():
        lines.append(f"- `{name}`: `{item['path']}` (`{item['sha256']}`)")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p4-audit", type=Path, required=True)
    parser.add_argument("--record-index", type=Path, required=True)
    parser.add_argument("--p5-summary", type=Path, required=True)
    parser.add_argument("--p5-per-sample", type=Path, required=True)
    parser.add_argument("--p7-audit", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--route-cap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--io-workers", type=int, default=8)
    args = parser.parse_args()

    p4 = json.loads(args.p4_audit.read_text(encoding="utf-8"))
    p5 = json.loads(args.p5_summary.read_text(encoding="utf-8"))
    p7 = json.loads(args.p7_audit.read_text(encoding="utf-8"))
    if not p4.get("passed") or not p7.get("passed"):
        raise RuntimeError("P4 and P7 must pass before P8")
    if digest(args.record_index) != p4["record_index_sha256"]:
        raise RuntimeError("P4 record-index checksum mismatch")
    if digest(args.split_manifest) != p7["artifacts"]["split_manifest_sha256"]:
        raise RuntimeError("P7 split-manifest checksum mismatch")
    if digest(args.p5_per_sample) != p5["integrity"]["per_sample_summary_sha256"]:
        raise RuntimeError("P5 per-sample checksum mismatch")

    index_rows = read_jsonl(args.record_index)
    split_rows = read_jsonl(args.split_manifest)
    p5_rows = read_jsonl(args.p5_per_sample)
    split_by_uid = {row["uid"]: row for row in split_rows}
    p5_by_uid = {row["uid"]: row for row in p5_rows}
    uids = {row["uid"] for row in index_rows}
    if len(index_rows) != 8000 or uids != set(split_by_uid) or uids != set(p5_by_uid):
        raise RuntimeError("P4/P5/P7 populations do not match exactly")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "single_best": args.output_dir / "derived_single_best_manifest_v1.jsonl",
        "valid_set": args.output_dir / "derived_valid_set_manifest_v1.jsonl",
        "binary_predictor": args.output_dir / "binary_predictor_manifest_v1.jsonl",
        "route_ranking": args.output_dir / "derived_route_ranking_manifest_v1.jsonl",
        "polar_segment": args.output_dir / "derived_polar_segment_manifest_v1.jsonl",
    }
    temp_outputs = {name: path.with_suffix(path.suffix + ".tmp") for name, path in outputs.items()}
    for path in temp_outputs.values():
        if path.exists():
            raise RuntimeError(f"stale temporary output exists: {path}")

    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    totals: Counter[str] = Counter()
    selected_digest = sha256()
    selected_not_raw = 0
    anchor_violations = 0
    with ExitStack() as stack:
        handles = {name: stack.enter_context(path.open("w", encoding="utf-8")) for name, path in temp_outputs.items()}
        loader = stack.enter_context(ProcessPoolExecutor(max_workers=args.io_workers))
        for index_row, record in loader.map(load_verified_record, index_rows):
            sample = record["sample"]
            uid = sample["uid"]
            split = split_by_uid[uid]
            p5_row = p5_by_uid[uid]
            candidates = record["candidate_executions"]
            by_id = {route["route_id"]: route for route in candidates}
            valid_ids = set(record["successful_route_ids"])
            valid = [by_id[route_id] for route_id in valid_ids]
            if len(by_id) != len(candidates) or valid_ids != {
                route["route_id"] for route in candidates if route["result_correct"] is True
            }:
                raise RuntimeError(f"route linkage mismatch for {uid}")
            if len(valid) != p5_row["valid_route_count"] or len(candidates) != p5_row["evaluated_route_count"]:
                raise RuntimeError(f"P5 route-count mismatch for {uid}")
            best = single_best_valid_route(valid)
            best_id = None if best is None else best["route_id"]
            if best_id != record.get("best_sparse_success_route_id"):
                raise RuntimeError(f"minimum-route mismatch for {uid}")
            selected = select_diverse_valid_routes(valid, limit=args.route_cap, seed=args.seed, uid=uid)
            raw_keys = {route["mask_key"] for route in valid}
            selected_keys = {route["mask_key"] for route in selected}
            selected_not_raw += len(selected_keys - raw_keys)
            if best is not None and best["mask_key"] not in selected_keys:
                anchor_violations += 1
            for anchor in ("0" * 28, "1" * 28):
                if anchor in raw_keys and len(valid) > args.route_cap and anchor not in selected_keys:
                    anchor_violations += 1
            if len(valid) <= args.route_cap and selected_keys != raw_keys:
                raise RuntimeError(f"under-cap route loss for {uid}")

            minimum_mask = tuple(best["visual_on_mask"]) if best is not None else tuple()
            compact = [compact_route(route, minimum_mask) for route in selected]
            weight = 1.0 / len(compact) if compact else None
            for route in compact:
                route["weight"] = weight
            base = base_row(sample, split, index_row["record_sha256"])
            common = {
                **base,
                "raw_valid_route_count": len(valid),
                "selected_valid_route_count": len(compact),
                "route_cap": args.route_cap,
                "route_cap_applied": len(valid) > args.route_cap,
                "route_selection_seed": args.seed,
            }
            single_row = {
                "schema_version": "derived_single_best_manifest_v1",
                **common,
                "single_best_route": None if best is None else compact_route(best, minimum_mask),
            }
            valid_row = {
                "schema_version": "derived_valid_set_manifest_v1",
                **common,
                "valid_routes": compact,
            }
            predictor_row = {**valid_row, "schema_version": "binary_predictor_manifest_v1"}
            handles["single_best"].write(json.dumps(single_row, sort_keys=True) + "\n")
            handles["valid_set"].write(json.dumps(valid_row, sort_keys=True) + "\n")
            handles["binary_predictor"].write(json.dumps(predictor_row, sort_keys=True) + "\n")

            for route in candidates:
                ranking = {
                    "schema_version": "derived_route_ranking_manifest_v1",
                    "uid": uid,
                    "benchmark": sample["benchmark"],
                    "split": split["split"],
                    "split_group": sample["image_group_id"],
                    "route_id": route["route_id"],
                    "key": route["mask_key"],
                    "mask": route["visual_on_mask"],
                    "valid": route["route_id"] in valid_ids,
                    "score": float(route["score"]),
                    "reward": float(route["reward"]),
                    "correctness_threshold": float(route["correctness_threshold"]),
                    "num_visual_on_layers": int(route["num_visual_on_layers"]),
                    "num_visual_off_layers": int(route["num_visual_off_layers"]),
                    "num_transitions": int(route["num_transitions"]),
                    "raw_record_sha256": index_row["record_sha256"],
                }
                handles["route_ranking"].write(json.dumps(ranking, sort_keys=True) + "\n")
            for route, route_view in zip(selected, compact):
                polar = {
                    "schema_version": "derived_polar_segment_manifest_v1",
                    "uid": uid,
                    "benchmark": sample["benchmark"],
                    "split": split["split"],
                    "split_group": sample["image_group_id"],
                    "question": sample["question"],
                    "route_id": route["route_id"],
                    "key": route["mask_key"],
                    "mask": route["visual_on_mask"],
                    "weight": weight,
                    **canonical_segment_targets(route["visual_on_mask"]),
                }
                handles["polar_segment"].write(json.dumps(polar, sort_keys=True) + "\n")

            cell = counts[sample["benchmark"]][split["split"]]
            cell["samples"] += 1
            cell["positive_samples"] += bool(valid)
            cell["raw_valid_routes"] += len(valid)
            cell["selected_valid_routes"] += len(selected)
            cell["evaluated_routes"] += len(candidates)
            totals["samples"] += 1
            totals["positive_samples"] += bool(valid)
            totals["raw_valid_routes"] += len(valid)
            totals["selected_valid_routes"] += len(selected)
            totals["evaluated_routes"] += len(candidates)
            totals["capped_samples"] += len(valid) > args.route_cap
            selected_digest.update(f"{uid}:{','.join(route['mask_key'] for route in selected)}\n".encode())

    for name, path in outputs.items():
        os.replace(temp_outputs[name], path)
    artifact_info = {
        name: {"path": str(path), "sha256": checksum_sidecar(path), "bytes": path.stat().st_size}
        for name, path in outputs.items()
    }
    total_values = dict(totals)
    total_values["zero_positive_samples"] = totals["samples"] - totals["positive_samples"]
    audit = {
        "schema_version": "derived_supervision_audit_v1",
        "passed": (
            totals["samples"] == 8000
            and dict(Counter(row["benchmark"] for row in index_rows)) == EXPECTED
            and selected_not_raw == 0
            and anchor_violations == 0
            and p7["integrity"]["cross_split_image_groups"] == 0
        ),
        "selection": {
            "route_cap": args.route_cap,
            "seed": args.seed,
            "policy": "minimum_on_plus_valid_extreme_anchors_then_on_stratified_farthest_hamming_transition_v1",
            "single_best_tie_break": "minimum_visual_on_then_lexical_mask",
            "route_weights": "equal_within_selected_valid_set",
            "same_valid_set_for_objectives": ["duplicated_bce", "exact_set_nll"],
        },
        "inputs": {
            "p4_audit_sha256": digest(args.p4_audit),
            "record_index_sha256": digest(args.record_index),
            "p5_summary_sha256": digest(args.p5_summary),
            "p5_per_sample_sha256": digest(args.p5_per_sample),
            "p7_audit_sha256": digest(args.p7_audit),
            "split_manifest_sha256": digest(args.split_manifest),
            "source_plan_sha256": digest(args.source_plan),
        },
        "counts": {
            dataset: {split: dict(counts[dataset][split]) for split in ("train", "validation")}
            for dataset in ("gqa", "textvqa", "chartqa")
        },
        "totals": total_values,
        "integrity": {
            "raw_records_checksum_verified": totals["samples"],
            "cross_split_image_groups": p7["integrity"]["cross_split_image_groups"],
            "selected_not_raw_valid": selected_not_raw,
            "anchor_violations": anchor_violations,
            "selected_route_set_sha256": selected_digest.hexdigest(),
            "raw_cache_modified": False,
        },
        "artifacts": artifact_info,
    }
    if not audit["passed"]:
        raise RuntimeError(f"P8 integrity failure: {audit['integrity']}")
    audit_path = args.output_dir / "derived_supervision_audit_v1.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_sha = checksum_sidecar(audit_path)
    audit["artifacts"]["audit"] = {"path": str(audit_path), "sha256": audit_sha, "bytes": audit_path.stat().st_size}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(audit), encoding="utf-8")
    report_sha = checksum_sidecar(args.report)
    print(json.dumps({"passed": True, "totals": total_values, "audit_sha256": audit_sha, "report_sha256": report_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
