#!/usr/bin/env python3
"""Validate a portable MCTS JSONL manifest without running model inference."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from PIL import Image


REQUIRED_FIELDS = {
    "uid",
    "sample_id",
    "benchmark",
    "question",
    "prompt",
    "answer",
    "metric_name",
    "correctness_threshold",
    "max_new_tokens",
    "image_group_id",
    "local_image_path",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        rows.append(value)
    return rows


def validate_row(row: dict, line_number: int, *, verify_image_hash: bool) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - row.keys())
    if missing:
        errors.append(f"line {line_number}: missing fields {missing}")
        return errors
    for name in ("uid", "sample_id", "benchmark", "question", "prompt", "metric_name", "image_group_id"):
        if not isinstance(row[name], str) or not row[name].strip():
            errors.append(f"line {line_number}: {name} must be a nonempty string")
    if not isinstance(row["answer"], str) or not row["answer"].strip():
        errors.append(f"line {line_number}: answer must be a nonempty string")
    if row.get("max_image_tokens") is not None:
        errors.append(f"line {line_number}: max_image_tokens must be absent or null")
    try:
        threshold = float(row["correctness_threshold"])
        if not 0.0 <= threshold <= 1.0:
            errors.append(f"line {line_number}: correctness_threshold must be in [0, 1]")
    except (TypeError, ValueError):
        errors.append(f"line {line_number}: correctness_threshold is not numeric")
    if not isinstance(row["max_new_tokens"], int) or row["max_new_tokens"] <= 0:
        errors.append(f"line {line_number}: max_new_tokens must be a positive integer")
    answers = row.get("all_answer_norms")
    if answers is not None and not (
        isinstance(answers, list) and all(isinstance(answer, str) for answer in answers)
    ):
        errors.append(f"line {line_number}: all_answer_norms must be null or a list of strings")
    for mask_index, mask in enumerate(row.get("mixed_masks", [])):
        if not isinstance(mask, list) or len(mask) != 28 or any(value not in (0, 1) for value in mask):
            errors.append(f"line {line_number}: mixed_masks[{mask_index}] is not a 28-bit mask")
    image_path = Path(str(row["local_image_path"]))
    if not image_path.is_absolute():
        errors.append(f"line {line_number}: local_image_path must be absolute")
    elif not image_path.is_file():
        errors.append(f"line {line_number}: image does not exist: {image_path}")
    else:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            errors.append(f"line {line_number}: unreadable image {image_path}: {exc}")
        expected_hash = row.get("image_content_sha256")
        if verify_image_hash and expected_hash:
            actual_hash = sha256(image_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                errors.append(f"line {line_number}: image SHA-256 mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--verify-image-hash", action="store_true")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    rows = read_jsonl(manifest)
    errors: list[str] = []
    if args.expected_count is not None and len(rows) != args.expected_count:
        errors.append(f"record count is {len(rows)}, expected {args.expected_count}")
    seen: set[str] = set()
    duplicate_uids: list[str] = []
    for line_number, row in enumerate(rows, 1):
        errors.extend(validate_row(row, line_number, verify_image_hash=args.verify_image_hash))
        uid = str(row.get("uid", ""))
        if uid in seen:
            duplicate_uids.append(uid)
        seen.add(uid)
    if duplicate_uids:
        errors.append(f"duplicate uids: {sorted(set(duplicate_uids))[:20]}")
    report = {
        "schema_version": "portable_mcts_manifest_audit_v1",
        "passed": not errors,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "record_count": len(rows),
        "benchmark_counts": {
            name: sum(str(row.get("benchmark")) == name for row in rows)
            for name in sorted({str(row.get("benchmark")) for row in rows})
        },
        "unique_uid_count": len(seen),
        "unique_image_group_count": len({str(row.get("image_group_id")) for row in rows}),
        "errors": errors,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output.read_bytes()).hexdigest()}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "records": len(rows), "errors": len(errors)}))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
