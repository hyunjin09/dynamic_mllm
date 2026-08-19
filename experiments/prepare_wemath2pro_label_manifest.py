#!/usr/bin/env python3
"""Materialize and freeze the complete We-Math2.0-Pro MCTS manifest."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Image as DatasetImage
from datasets import load_dataset
from PIL import Image

from label_regeneration.wemath import (
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_ROWS,
    build_wemath_record,
    deterministic_wemath_smoke_records,
    technical_invalid_reasons,
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    atomic_bytes(path, payload.encode("utf-8"))


def materialize_image(image_value: dict, path: Path) -> str:
    content = image_value.get("bytes")
    if content is None:
        source = Path(str(image_value["path"]))
        content = source.read_bytes()
    expected_hash = sha256(content).hexdigest()
    if path.is_file():
        if file_sha256(path) != expected_hash:
            raise ValueError(f"existing image hash mismatch: {path}")
    else:
        atomic_bytes(path, content)
    with Image.open(path) as image:
        image.verify()
    return expected_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    image_root = Path(args.image_root).resolve()
    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        cache_dir=str(Path(args.cache_dir).resolve()),
        split="pro",
    ).cast_column("image", DatasetImage(decode=False))
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} Pro rows, found {len(dataset)}")

    records = []
    inventory = []
    invalid_records = []
    seen_uids: set[str] = set()
    for source_index, row in enumerate(dataset):
        image_path = image_root / f"{int(row['idx']):06d}.png"
        image_hash = materialize_image(row["image"], image_path)
        invalid_reasons = technical_invalid_reasons(row)
        inventory_row = {
            "uid": f"wemath2pro:{row['idx']}",
            "sample_id": str(row["idx"]),
            "question_id": str(row["question_id"]),
            "difficulty": str(row["difficulty"]),
            "question": str(row.get("question") or ""),
            "answer": str(row.get("answer") or ""),
            "source_index": source_index,
            "local_image_path": str(image_path),
            "image_content_sha256": image_hash,
            "technical_valid": not invalid_reasons,
            "technical_invalid_reasons": invalid_reasons,
        }
        inventory.append(inventory_row)
        if invalid_reasons:
            invalid_records.append(inventory_row)
            continue
        record = build_wemath_record(
            row,
            source_index=source_index,
            image_path=image_path,
            image_sha256=image_hash,
        )
        if record["uid"] in seen_uids:
            raise ValueError(f"duplicate uid: {record['uid']}")
        seen_uids.add(record["uid"])
        records.append(record)

    if len(records) != 4544 or len(invalid_records) != 8:
        raise ValueError(
            f"approved validity rule expected 4544 valid/8 invalid, found "
            f"{len(records)} valid/{len(invalid_records)} invalid"
        )

    manifest = output_root / "manifest" / "wemath2pro_valid_mcts_v1.jsonl"
    smoke_manifest = output_root / "manifest" / "wemath2pro_smoke_v1.jsonl"
    inventory_manifest = output_root / "manifest" / "wemath2pro_all_inventory_v1.jsonl"
    atomic_jsonl(manifest, records)
    atomic_jsonl(smoke_manifest, deterministic_wemath_smoke_records(records))
    atomic_jsonl(inventory_manifest, inventory)
    summary = {
        "schema_version": "wemath2pro_label_manifest_v1",
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": "pro",
        "total_record_count": len(inventory),
        "valid_mcts_record_count": len(records),
        "technical_invalid_record_count": len(invalid_records),
        "technical_invalid_records": [
            {
                "uid": row["uid"],
                "source_index": row["source_index"],
                "reasons": row["technical_invalid_reasons"],
            }
            for row in invalid_records
        ],
        "unique_inventory_uid_count": len({row["uid"] for row in inventory}),
        "unique_valid_uid_count": len(seen_uids),
        "unique_image_hash_count": len({row["image_content_sha256"] for row in records}),
        "difficulty_counts": {
            difficulty: sum(row["difficulty"] == difficulty for row in records)
            for difficulty in sorted({row["difficulty"] for row in records})
        },
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "inventory_manifest": str(inventory_manifest),
        "inventory_manifest_sha256": file_sha256(inventory_manifest),
        "smoke_manifest": str(smoke_manifest),
        "smoke_manifest_sha256": file_sha256(smoke_manifest),
        "image_root": str(image_root),
        "native_image_processing": True,
        "custom_max_image_tokens": None,
    }
    atomic_json(output_root / "manifest" / "manifest_summary_v1.json", summary)
    for path in (
        manifest,
        smoke_manifest,
        inventory_manifest,
        output_root / "manifest" / "manifest_summary_v1.json",
    ):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
