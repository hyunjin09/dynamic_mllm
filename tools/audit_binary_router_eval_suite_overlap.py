#!/usr/bin/env python3
"""Outcome-blind overlap audit for all binary-router evaluation suites."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def row_question(row: dict) -> str:
    if row.get("question") is not None:
        return normalize_text(row["question"])
    return normalize_text(" ".join(row.get("instruction_text_chunks") or []))


def row_image_hashes(row: dict) -> tuple[str, ...]:
    hashes = row.get("image_content_sha256s")
    if hashes is None:
        hashes = [row.get("image_content_sha256")]
    return tuple(str(value) for value in hashes if value)


def summarize_overlap(eval_rows: list[dict], mcts_rows: list[dict]) -> dict:
    mcts_hashes = {str(row["computed_image_sha256"]) for row in mcts_rows}
    mcts_uids = {str(row["uid"]) for row in mcts_rows}
    mcts_sample_ids = {str(row["sample_id"]) for row in mcts_rows}
    mcts_questions = {row_question(row) for row in mcts_rows}
    mcts_prompts = {normalize_text(row.get("prompt")) for row in mcts_rows}
    mcts_pairs = {
        (str(row["computed_image_sha256"]), row_question(row)) for row in mcts_rows
    }

    by_benchmark = {}
    for benchmark in sorted({str(row["benchmark"]) for row in eval_rows}):
        rows = [row for row in eval_rows if str(row["benchmark"]) == benchmark]
        image_hashes = {value for row in rows for value in row_image_hashes(row)}
        exact_pairs = {
            (hashes[0], row_question(row))
            for row in rows
            if len(hashes := row_image_hashes(row)) == 1
        }
        by_benchmark[benchmark] = {
            "records": len(rows),
            "unique_image_hashes": len(image_hashes),
            "records_with_any_mcts_image": sum(
                any(value in mcts_hashes for value in row_image_hashes(row))
                for row in rows
            ),
            "shared_unique_image_hashes": len(image_hashes & mcts_hashes),
            "exact_image_question_pairs": len(exact_pairs & mcts_pairs),
            "uid_overlap": len({str(row["uid"]) for row in rows} & mcts_uids),
            "sample_id_overlap": len(
                {str(row["sample_id"]) for row in rows} & mcts_sample_ids
            ),
            "normalized_question_overlap": len(
                {row_question(row) for row in rows} & mcts_questions
            ),
            "normalized_prompt_overlap": len(
                {normalize_text(row.get("prompt")) for row in rows} & mcts_prompts
            ),
        }
    return by_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcts-manifest", type=Path, required=True)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--pope-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mcts_rows = read_jsonl(args.mcts_manifest)
    if len(mcts_rows) != 8000:
        raise RuntimeError(f"expected 8000 MCTS records, found {len(mcts_rows)}")
    image_paths = [Path(row["local_image_path"]) for row in mcts_rows]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing MCTS image: {missing[0]}")
    unique_paths = list(dict.fromkeys(image_paths))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes = dict(zip(unique_paths, pool.map(file_sha256, unique_paths)))
    for row, path in zip(mcts_rows, image_paths):
        row["computed_image_sha256"] = hashes[path]

    manifests = {
        "core_vqa": args.core_manifest,
        "external_multiple_choice": args.external_manifest,
        "pope": args.pope_manifest,
    }
    expected_counts = {
        "core_vqa": 12849,
        "external_multiple_choice": 5807,
        "pope": 9000,
    }
    suites = {}
    for name, path in manifests.items():
        rows = read_jsonl(path)
        if len(rows) != expected_counts[name]:
            raise RuntimeError(f"unexpected {name} count: {len(rows)}")
        if len({str(row["uid"]) for row in rows}) != len(rows):
            raise RuntimeError(f"duplicate UID in {name}")
        missing_hash = [str(row["uid"]) for row in rows if not row_image_hashes(row)]
        if missing_hash:
            raise RuntimeError(f"missing declared image hash in {name}: {missing_hash[0]}")
        suites[name] = {
            "manifest": str(path),
            "manifest_sha256": file_sha256(path),
            "records": len(rows),
            "benchmarks": dict(sorted(Counter(str(row["benchmark"]) for row in rows).items())),
            "overlap_by_benchmark": summarize_overlap(rows, mcts_rows),
        }

    report = {
        "schema_version": "binary_router_eval_suite_overlap_audit_v1",
        "selection_data_only": True,
        "scientific_outcomes_loaded": False,
        "mcts": {
            "manifest": str(args.mcts_manifest),
            "manifest_sha256": file_sha256(args.mcts_manifest),
            "records": len(mcts_rows),
            "unique_image_hashes": len(
                {str(row["computed_image_sha256"]) for row in mcts_rows}
            ),
        },
        "suites": suites,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    checksum = file_sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{checksum}  {args.output.name}\n"
    )
    print(json.dumps({"output": str(args.output), "suites": list(suites)}))


if __name__ == "__main__":
    main()
