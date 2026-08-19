#!/usr/bin/env python3
"""Outcome-blind overlap and split-feasibility audit for the external router test."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata


TARGETS = {
    "gqa": {"train": 3500, "validation": 500, "validation_historical_correct": 250},
    "textvqa": {"train": 1750, "validation": 250, "validation_historical_correct": 125},
    "chartqa": {"train": 1750, "validation": 250, "validation_historical_correct": 125},
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def subset_feasible(groups: list[list[dict]], target_total: int, target_correct: int) -> bool:
    """Exact two-dimensional group subset feasibility without freezing identities."""
    reachable = {(0, 0)}
    for group in groups:
        size = len(group)
        correct = sum(row["historical_all_on_status"] == "correct" for row in group)
        additions = {
            (total + size, correct_count + correct)
            for total, correct_count in reachable
            if total + size <= target_total and correct_count + correct <= target_correct
        }
        reachable.update(additions)
        if (target_total, target_correct) in reachable:
            return True
    return (target_total, target_correct) in reachable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcts-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--eval-data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mcts = read_jsonl(args.mcts_manifest)
    external = read_jsonl(args.eval_manifest)
    if Counter(row["benchmark"] for row in mcts) != Counter(
        {"gqa": 4000, "textvqa": 2000, "chartqa": 2000}
    ):
        raise RuntimeError("unexpected regenerated-label population")
    if len(external) != 5807 or len({row["uid"] for row in external}) != 5807:
        raise RuntimeError("unexpected external evaluation population")

    mcts_paths = [Path(row["local_image_path"]) for row in mcts]
    external_refs = [
        (args.eval_data_root / relpath, declared)
        for row in external
        for relpath, declared in zip(row["image_relpaths"], row["image_content_sha256s"])
    ]
    all_paths = list(dict.fromkeys(mcts_paths + [path for path, _ in external_refs]))
    missing = [str(path) for path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} referenced images; first={missing[0]}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes = dict(zip(all_paths, pool.map(file_sha256, all_paths)))

    external_hash_mismatches = [
        str(path) for path, declared in external_refs if hashes[path] != declared
    ]
    if external_hash_mismatches:
        raise RuntimeError(
            f"external manifest image hash mismatch: {external_hash_mismatches[0]}"
        )
    mcts_hashes = [hashes[path] for path in mcts_paths]
    external_hashes = [hashes[path] for path, _ in external_refs]
    mcts_hash_set = set(mcts_hashes)
    external_hash_set = set(external_hashes)

    mcts_questions = {normalize_text(row["question"]) for row in mcts}
    mcts_prompts = {normalize_text(row["prompt"]) for row in mcts}
    external_instructions = {
        normalize_text(" ".join(row.get("instruction_text_chunks") or []))
        for row in external
    }
    external_prompts = {normalize_text(row["prompt"]) for row in external}
    mcts_image_question = {
        (image_hash, normalize_text(row["question"]))
        for row, image_hash in zip(mcts, mcts_hashes)
    }
    external_image_question = {
        (tuple(row["image_content_sha256s"]), normalize_text(" ".join(row.get("instruction_text_chunks") or [])))
        for row in external
    }

    group_summary = {}
    feasibility = {}
    for dataset, target in TARGETS.items():
        rows = [row for row in mcts if row["benchmark"] == dataset]
        by_group: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_group[str(row["image_group_id"])].append(row)
        groups = list(by_group.values())
        sizes = Counter(len(group) for group in groups)
        group_summary[dataset] = {
            "samples": len(rows),
            "image_groups": len(groups),
            "groups_with_multiple_questions": sum(size > 1 for size in sizes.elements()),
            "maximum_group_size": max(sizes),
            "group_size_counts": {str(size): count for size, count in sorted(sizes.items())},
        }
        feasibility[dataset] = {
            **target,
            "exact_grouped_validation_total_and_historical_balance_feasible": subset_feasible(
                groups,
                target_total=target["validation"],
                target_correct=target["validation_historical_correct"],
            ),
        }

    report = {
        "schema_version": "binary_router_external_eval_split_audit_v1",
        "inputs": {
            "mcts_manifest": str(args.mcts_manifest),
            "mcts_manifest_sha256": file_sha256(args.mcts_manifest),
            "external_manifest": str(args.eval_manifest),
            "external_manifest_sha256": file_sha256(args.eval_manifest),
            "external_data_root": str(args.eval_data_root),
        },
        "populations": {
            "mcts": {
                "records": len(mcts),
                "datasets": dict(sorted(Counter(row["benchmark"] for row in mcts).items())),
                "unique_uids": len({row["uid"] for row in mcts}),
                "image_references": len(mcts_hashes),
                "unique_image_hashes": len(mcts_hash_set),
            },
            "external_eval": {
                "records": len(external),
                "datasets": dict(sorted(Counter(row["benchmark"] for row in external).items())),
                "unique_uids": len({row["uid"] for row in external}),
                "image_references": len(external_hashes),
                "unique_image_hashes": len(external_hash_set),
                "manifest_image_hash_mismatches": 0,
            },
        },
        "overlap": {
            "uid": len({row["uid"] for row in mcts} & {row["uid"] for row in external}),
            "sample_id": len(
                {row["sample_id"] for row in mcts}
                & {row["sample_id"] for row in external}
            ),
            "benchmark_name": sorted(
                {row["benchmark"] for row in mcts}
                & {row["benchmark"] for row in external}
            ),
            "exact_image_sha256": len(mcts_hash_set & external_hash_set),
            "normalized_question_or_instruction": len(mcts_questions & external_instructions),
            "normalized_prompt": len(mcts_prompts & external_prompts),
            "exact_single_image_question_pair": len(
                mcts_image_question
                & {
                    (image_hashes[0], question)
                    for image_hashes, question in external_image_question
                    if len(image_hashes) == 1
                }
            ),
        },
        "mcts_image_groups": group_summary,
        "proposed_internal_split_feasibility": {
            "train_records": 7000,
            "validation_records": 1000,
            "internal_test_records": 0,
            "external_test_records": 5807,
            "dataset_targets": feasibility,
            "selection_inputs": ["dataset", "image_group_id", "historical_all_on_status"],
            "selection_excludes": [
                "current_all_on_status",
                "valid_route_count",
                "correction_found",
                "route_diversity",
                "predictor_outcomes",
                "external_test_outcomes",
            ],
        },
        "passed": all(
            value in (0, [], {})
            for value in (
                len({row["uid"] for row in mcts} & {row["uid"] for row in external}),
                len({row["sample_id"] for row in mcts} & {row["sample_id"] for row in external}),
                sorted({row["benchmark"] for row in mcts} & {row["benchmark"] for row in external}),
                len(mcts_hash_set & external_hash_set),
                len(
                    mcts_image_question
                    & {
                        (image_hashes[0], question)
                        for image_hashes, question in external_image_question
                        if len(image_hashes) == 1
                    }
                ),
            )
        )
        and all(
            row["exact_grouped_validation_total_and_historical_balance_feasible"]
            for row in feasibility.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = file_sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{checksum}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "overlap": report["overlap"], "output": str(args.output)}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
