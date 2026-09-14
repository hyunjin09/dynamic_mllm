#!/usr/bin/env python3
"""Merge dense batches, audit regenerated labels, and freeze group splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path

import torch

from dense_failure_stage1.contract import canonical_json_sha256, file_sha256
from experiments.run_current_dense_regeneration import (
    feature_provenance,
    validate_feature_provenance,
)
from tools.research_analysis.dense_failure_stage1 import (
    build_image_group_disjoint_split,
    canonical_payload_sha256,
)


def validate_global_completion(
    expected_uids,
    rows,
    markers,
    *,
    expected_world_size: int,
    require_features: bool,
) -> None:
    """Fail unless every expected UID and every worker rank completed exactly once."""

    actual_uids = [row["uid"] for row in rows]
    if len(actual_uids) != len(set(actual_uids)):
        raise ValueError("global completion contains duplicate UIDs")
    if set(actual_uids) != set(expected_uids) or len(actual_uids) != len(expected_uids):
        missing = sorted(set(expected_uids) - set(actual_uids))
        extra = sorted(set(actual_uids) - set(expected_uids))
        raise ValueError(
            f"global UID coverage differs: missing={missing[:5]} extra={extra[:5]} "
            f"actual={len(actual_uids)} expected={len(expected_uids)}"
        )
    ranks = {int(marker.get("rank", -1)) for marker in markers}
    expected_ranks = set(range(expected_world_size))
    if ranks != expected_ranks:
        raise ValueError(
            f"global rank coverage differs: actual={sorted(ranks)} expected={sorted(expected_ranks)}"
        )
    expected_state = (
        "completed_with_features" if require_features else "completed_without_features"
    )
    for marker in markers:
        if int(marker.get("world_size", -1)) != expected_world_size:
            raise ValueError("global marker world size differs")
        if marker.get("completion_state") != expected_state:
            raise ValueError("global marker feature completion state differs")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_once(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to overwrite differing final artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def write_json(path: Path, value) -> None:
    write_once(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    write_once(path, "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows))


def csv_text(header: list[str], rows: list[list]) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def validate_and_merge_batches(
    output_root: Path,
    contract: dict,
    *,
    run_id: str,
    require_features: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    rows = []
    feature_index = []
    seen = set()
    markers = sorted((output_root / "generation" / "batches").glob("*.complete.json"))
    if not markers:
        raise ValueError("no completed dense-generation batch markers exist")
    for marker_path in markers:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("schema_version") != "current_dense_batch_marker_v2":
            raise ValueError(f"unsupported batch marker schema: {marker_path}")
        if marker["contract_id"] != contract["contract_sha256"]:
            raise ValueError(f"batch contract differs: {marker_path}")
        if marker.get("run_id") != run_id:
            raise ValueError(f"batch run ID differs: {marker_path}")
        if marker.get("world_size") != 4 or marker.get("rank") not in range(4):
            raise ValueError(f"batch execution topology differs: {marker_path}")
        expected_state = (
            "completed_with_features"
            if require_features
            else "completed_without_features"
        )
        if (
            marker.get("extract_features") is not require_features
            or marker.get("completion_state") != expected_state
        ):
            raise ValueError(f"batch feature completion mode differs: {marker_path}")
        generation_path = marker_path.parent / marker["generation_file"]
        if file_sha256(generation_path) != marker["generation_sha256"]:
            raise ValueError(f"generation batch hash differs: {generation_path}")
        batch_rows = read_jsonl(generation_path)
        if len(batch_rows) != marker["records"]:
            raise ValueError(f"generation batch count differs: {generation_path}")
        for row in batch_rows:
            if row["uid"] in seen:
                raise ValueError(f"duplicate generated UID: {row['uid']}")
            seen.add(row["uid"])
            if (
                row.get("contract_id") != contract["contract_sha256"]
                or row.get("model_revision")
                != contract["integrity"]["model"]["revision"]
                or row.get("source_manifest_sha256")
                != contract["integrity"]["candidate_manifest_sha256"]
                or row.get("consumed_image_sha256")
                != row.get("image_content_sha256")
            ):
                raise ValueError(f"generated row provenance differs: {row['uid']}")
        if require_features:
            feature_path = Path(marker["feature_path"])
            if file_sha256(feature_path) != marker["feature_sha256"]:
                raise ValueError(f"feature shard hash differs: {feature_path}")
            payload = torch.load(feature_path, map_location="cpu", weights_only=False)
            expected_provenance = feature_provenance(
                contract,
                run_id=run_id,
                generation_sha256=marker["generation_sha256"],
            )
            validate_feature_provenance(payload, expected_provenance)
            if payload["uids"] != [row["uid"] for row in batch_rows]:
                raise ValueError(f"feature UID order differs: {feature_path}")
            for key in ("text_final", "text_mean", "visual_mean"):
                if (
                    payload[key].shape[0] != len(batch_rows)
                    or payload[key].shape[1] != 28
                    or payload[key].shape[2]
                    != contract["features"]["schema"]["hidden_size"]
                ):
                    raise ValueError(f"feature shape differs for {key}: {feature_path}")
            for offset, uid in enumerate(payload["uids"]):
                feature_index.append(
                    {
                        "uid": uid,
                        "shard": str(feature_path.resolve()),
                        "offset": offset,
                        **expected_provenance,
                    }
                )
        rows.extend(batch_rows)
    return rows, feature_index, [
        json.loads(path.read_text(encoding="utf-8")) for path in markers
    ]


def population_table(rows: list[dict]) -> list[list]:
    output = []
    for dataset in ("gqa", "chartqa", "textvqa"):
        selected = [row for row in rows if row["dataset"] == dataset]
        correct = sum(row["current_dense_correct"] for row in selected)
        output.append([dataset, correct, len(selected) - correct, len(selected)])
    correct = sum(row["current_dense_correct"] for row in rows)
    output.append(["overall", correct, len(rows) - correct, len(rows)])
    return output


def historical_table(rows: list[dict]) -> list[list]:
    output = []
    for bucket in ("correct", "wrong"):
        selected = [row for row in rows if row["historical_bucket"] == bucket]
        current_correct = sum(row["current_dense_correct"] for row in selected)
        output.append([bucket, current_correct, len(selected) - current_correct, len(selected)])
    return output


def image_group_stats(rows: list[dict]) -> dict:
    groups = Counter(row["image_group_id"] for row in rows)
    distribution = Counter(groups.values())
    question_counts = Counter((row["dataset"], row.get("question")) for row in rows)
    exact_counts = Counter(
        (
            row["dataset"],
            row["image_group_id"],
            row.get("question"),
            json.dumps(row.get("gt_answer"), sort_keys=True),
        )
        for row in rows
    )
    return {
        "unique_uids": len({row["uid"] for row in rows}),
        "unique_image_groups": len(groups),
        "multi_question_image_groups": sum(value > 1 for value in groups.values()),
        "records_per_group_distribution": dict(sorted(distribution.items())),
        "max_records_per_group": max(groups.values(), default=0),
        "duplicate_question_records_beyond_first": sum(max(0, value - 1) for value in question_counts.values()),
        "exact_duplicate_records_beyond_first": sum(max(0, value - 1) for value in exact_counts.values()),
    }


def split_targets(count: int) -> dict[str, int]:
    val = round(count * 0.1)
    test = round(count * 0.1)
    return {"train": count - val - test, "val": val, "test": test}


def split_table(rows: list[dict]) -> list[list]:
    table = []
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        values = []
        for dataset in ("gqa", "chartqa", "textvqa"):
            cell = [row for row in selected if row["dataset"] == dataset]
            correct = sum(row["current_dense_correct"] for row in cell)
            values.extend([correct, len(cell) - correct])
        table.append([split, *values, len(selected)])
    return table


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--require-features", action="store_true")
    parser.add_argument("--split-seed", type=int, default=20260830)
    args = parser.parse_args()
    root = Path(args.output_root)
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    if canonical_payload_sha256(contract) != contract["contract_sha256"]:
        raise ValueError("frozen contract hash is invalid")
    candidates = read_jsonl(Path(args.candidate_manifest))
    candidate_index = {row["uid"]: row for row in candidates}
    if (
        file_sha256(Path(args.candidate_manifest))
        != contract["integrity"]["candidate_manifest_sha256"]
    ):
        raise ValueError("candidate manifest differs from frozen contract")
    generated, feature_index, batch_markers = validate_and_merge_batches(
        root,
        contract,
        run_id=args.run_id,
        require_features=args.require_features,
    )
    generated_index = {row["uid"]: row for row in generated}
    extra = sorted(set(generated_index) - set(candidate_index))
    missing = sorted(set(candidate_index) - set(generated_index))
    if extra:
        raise ValueError(f"generated UIDs outside candidate population: {extra[:5]}")
    validate_global_completion(
        list(candidate_index),
        generated,
        batch_markers,
        expected_world_size=4,
        require_features=args.require_features,
    )
    if missing:
        raise ValueError(f"candidate UIDs missing dense output: {len(missing)}")
    merged = [
        {**candidate_index[uid], **generated_index[uid]}
        for uid in sorted(generated_index)
    ]
    if not merged:
        raise ValueError("no valid dense outputs")
    feature_map = {row["uid"]: row for row in feature_index}
    if args.require_features and set(feature_map) != set(generated_index):
        raise ValueError("feature coverage differs from valid generated population")
    if not args.require_features and feature_index:
        raise ValueError("generation-only aggregation unexpectedly found feature shards")

    generation_path = root / "generation" / "dense_outputs.jsonl"
    write_jsonl(generation_path, merged)
    history_rows = []
    for path in sorted((root / "generation").glob("history_rank*.jsonl")):
        history_rows.extend(read_jsonl(path))
    write_jsonl(root / "generation" / "generation_history.jsonl", history_rows)

    counts = population_table(merged)
    historical = historical_table(merged)
    write_once(
        root / "audit" / "population_counts.csv",
        csv_text(["dataset", "current_correct", "current_wrong", "total"], counts),
    )
    write_once(
        root / "audit" / "historical_vs_current.csv",
        csv_text(["historical_bucket", "current_correct", "current_wrong", "total"], historical),
    )
    group_stats = image_group_stats(merged)
    per_dataset_flips = {}
    for dataset in ("gqa", "chartqa", "textvqa"):
        cell = [row for row in merged if row["dataset"] == dataset]
        c2w = sum(row["historical_bucket"] == "correct" and row["current_dense_wrong"] for row in cell)
        w2c = sum(row["historical_bucket"] == "wrong" and row["current_dense_correct"] for row in cell)
        per_dataset_flips[dataset] = {
            "historical_correct_to_current_wrong": c2w,
            "historical_wrong_to_current_correct": w2c,
            "flip_rate": (c2w + w2c) / len(cell) if cell else None,
        }
    c2w_total = sum(row["historical_bucket"] == "correct" and row["current_dense_wrong"] for row in merged)
    w2c_total = sum(row["historical_bucket"] == "wrong" and row["current_dense_correct"] for row in merged)

    targets = split_targets(len(merged))
    split_rows = build_image_group_disjoint_split(merged, targets=targets, seed=args.split_seed)
    split_counts = split_table(split_rows)
    split_group_sets = {
        split: {row["image_group_id"] for row in split_rows if row["split"] == split}
        for split in ("train", "val", "test")
    }
    split_uid_sets = {
        split: {row["uid"] for row in split_rows if row["split"] == split}
        for split in ("train", "val", "test")
    }
    group_overlap = sum(
        len(split_group_sets[left] & split_group_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    uid_overlap = sum(
        len(split_uid_sets[left] & split_uid_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    if group_overlap or uid_overlap:
        raise ValueError("split overlap invariant failed")
    split_manifest = []
    for row in split_rows:
        split_manifest.append(
            {
                "uid": row["uid"],
                "split": row["split"],
                "dataset": row["dataset"],
                "image_group_id": row["image_group_id"],
                "current_dense_correct": row["current_dense_correct"],
                "current_dense_wrong": row["current_dense_wrong"],
                "contract_id": row["contract_id"],
                "feature": feature_map.get(row["uid"]),
            }
        )
    write_jsonl(root / "splits" / "stage1_split_manifest.jsonl", split_manifest)
    write_jsonl(root / "features" / "feature_index.jsonl", feature_index)
    if feature_index:
        schema = contract["features"]["schema"]
        schema_path = root / "features" / "feature_schema.json"
        write_json(schema_path, schema)
        if (
            canonical_json_sha256(schema)
            != contract["integrity"]["feature_schema_sha256"]
        ):
            raise ValueError("feature schema differs from frozen contract")

    population_md = markdown_table(
        ["Dataset", "Current Correct", "Current Wrong", "Total"], counts
    )
    historical_md = markdown_table(
        ["Historical", "Current correct", "Current wrong", "Total"], historical
    )
    label_md = f"""# Current Dense Label-Regeneration Audit

{population_md}

{historical_md}

- Valid current-runtime outputs: {len(merged)} / {len(candidates)} candidates
- Missing/invalid candidates: {len(missing)}
- Historical correct -> current wrong: {c2w_total}
- Historical wrong -> current correct: {w2c_total}
- Per-dataset flips: `{json.dumps(per_dataset_flips, sort_keys=True)}`
- Current dense labels are authoritative; historical buckets were not used to modify them.
"""
    write_once(root / "audit" / "label_regeneration_audit.md", label_md)
    group_md = f"""# Image-Group Leakage Audit

- Unique UIDs: {group_stats['unique_uids']}
- Unique image-content groups: {group_stats['unique_image_groups']}
- Multi-question image groups: {group_stats['multi_question_image_groups']}
- Records/group distribution: `{group_stats['records_per_group_distribution']}`
- Maximum records/group: {group_stats['max_records_per_group']}
- Duplicate question records beyond first: {group_stats['duplicate_question_records_beyond_first']}
- Exact duplicate records beyond first: {group_stats['exact_duplicate_records_beyond_first']}
- Group definition: SHA-256 of physical image file content.
"""
    write_once(root / "audit" / "image_group_audit.md", group_md)
    split_md = f"""# Stage-1 Image-Group-Disjoint Split Audit

{markdown_table(['Split', 'GQA C', 'GQA W', 'ChartQA C', 'ChartQA W', 'TextVQA C', 'TextVQA W', 'Total'], split_counts)}

- Requested ratio targets: `{targets}`
- UID overlap = {uid_overlap}
- image-group overlap = {group_overlap}
- Dataset and current-label proportions were preserved greedily after enforcing whole-group assignment.
"""
    write_once(root / "splits" / "stage1_split_audit.md", split_md)

    summary = {
        "schema_version": "current_dense_generation_summary_v1",
        "candidate_records": len(candidates),
        "valid_generated_records": len(merged),
        "missing_records": len(missing),
        "missing_uids": missing,
        "population_counts": counts,
        "historical_vs_current": historical,
        "historical_correct_to_current_wrong": c2w_total,
        "historical_wrong_to_current_correct": w2c_total,
        "per_dataset_flips": per_dataset_flips,
        "image_groups": group_stats,
        "split_counts": split_counts,
        "uid_overlap": uid_overlap,
        "image_group_overlap": group_overlap,
        "feature_records": len(feature_index),
        "contract_id": contract["contract_sha256"],
        "dense_outputs_sha256": file_sha256(generation_path),
        "split_manifest_sha256": file_sha256(root / "splits" / "stage1_split_manifest.jsonl"),
    }
    write_json(root / "generation" / "generation_summary.json", summary)
    decision_md = f"""# Current Dense Stage-1 Regeneration Decision Summary

Q1. Were all 8,000 candidates executed under one contract? **{'Yes' if len(merged) == 8000 else 'No'}** ({len(merged)}/8000 valid), contract `{contract['contract_sha256']}`.

Q2. Current-runtime distribution:\n\n{population_md}

Q3. Historical disagreement: {c2w_total} historical-correct -> current-wrong and {w2c_total} historical-wrong -> current-correct; total {(c2w_total + w2c_total)} / {len(merged)}.

Q4. Current-runtime smoke determinism: see `smoke/smoke_report.md` (required PASS before this run).

Q5. Feature extraction parity: see `smoke/hook_parity_report.md` (required PASS when features are enabled).

Q6. 28-layer features: **{'saved for all valid samples' if len(feature_index) == len(merged) else 'not complete'}** ({len(feature_index)}/{len(merged)}).

Q7. Image-group-disjoint split: **Yes**; UID overlap = {uid_overlap}, image-group overlap = {group_overlap}.

Q8. Authoritative next-experiment artifacts:

- `candidate_8k_manifest.jsonl` for frozen identities and annotations;
- `frozen_dense_execution_contract.json` for execution provenance;
- `generation/dense_outputs.jsonl` for current labels;
- `features/feature_index.jsonl` plus its shards and schema for predictors;
- `splits/stage1_split_manifest.jsonl` for the frozen group-disjoint split.

No Stage-1 predictor training is authorized or performed in this phase.
"""
    write_once(root / "decision_summary.md", decision_md)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
