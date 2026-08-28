#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.research_analysis.four_action.label_sources import (
    normalize_math_record,
    normalize_vqa_record,
)


DEFAULT_VQA_PREDICTOR = Path(
    "outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl"
)
DEFAULT_VQA_SOURCE = Path("datasets/mcts_labels/gqa_textvqa_chartqa_v1/source_manifest_v1.jsonl")
DEFAULT_VQA_IMAGES = Path(
    "datasets/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713/images"
)
DEFAULT_STANDARD_CACHE = Path("datasets/math_labels/wemath20_standard_mcts_max400_latest")
DEFAULT_STANDARD_SOURCE = Path("datasets/math_labels/wemath20_standard_source_v1")
DEFAULT_PRO_CACHE = Path("datasets/math_labels/wemath20_pro_mcts_max400_v2")
DEFAULT_PRO_IMAGES = Path("datasets/WeMath2Pro/pro_images_v1")
DEFAULT_OUTPUT = Path("datasets/mcts_labels_4action/source_inventory_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze authoritative four-action label inputs.")
    parser.add_argument("--vqa-predictor", type=Path, default=DEFAULT_VQA_PREDICTOR)
    parser.add_argument("--vqa-source", type=Path, default=DEFAULT_VQA_SOURCE)
    parser.add_argument("--vqa-images", type=Path, default=DEFAULT_VQA_IMAGES)
    parser.add_argument("--standard-cache", type=Path, default=DEFAULT_STANDARD_CACHE)
    parser.add_argument("--standard-source", type=Path, default=DEFAULT_STANDARD_SOURCE)
    parser.add_argument("--pro-cache", type=Path, default=DEFAULT_PRO_CACHE)
    parser.add_argument("--pro-images", type=Path, default=DEFAULT_PRO_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def write_atomic(path: Path, content: str) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = file_sha256(path)
    checksum = path.with_suffix(path.suffix + ".sha256")
    checksum.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _verified_image(path: Path, expected_sha256: str | None, cache: dict[Path, str]) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if expected_sha256:
        digest = cache.setdefault(resolved, file_sha256(resolved))
        if digest != expected_sha256:
            raise ValueError(f"image SHA-256 mismatch: {resolved}")
    return resolved


def _math_rows(
    *,
    cache_root: Path,
    image_for_sample,
    expected_dataset: str,
    image_hash_cache: dict[Path, str],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    paths = sorted(cache_root.glob("raw_route_cache/shard_*/samples/*.json"))
    rows = []
    terminal_uid_count = Counter()
    for path in paths:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        sample = raw["sample"]
        terminal_uid_count[str(sample["uid"])] += 1
        image_path = _verified_image(
            image_for_sample(sample),
            sample.get("image_content_sha256"),
            image_hash_cache,
        )
        normalized = normalize_math_record(
            raw,
            image_path=image_path,
            record_path=path,
            record_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )
        if normalized is not None:
            if normalized["dataset"] != expected_dataset:
                raise ValueError(f"unexpected dataset at {path}")
            rows.append(normalized)
    duplicates = sorted(uid for uid, count in terminal_uid_count.items() if count != 1)
    if duplicates:
        raise ValueError(f"duplicate math terminal UIDs: {duplicates[:3]}")
    return (
        rows,
        {
            "terminal_record_count": len(paths),
            "positive_sample_count": len(rows),
            "zero_positive_terminal_count": len(paths) - len(rows),
        },
        set(terminal_uid_count),
    )


def _summary(rows: list[dict[str, Any]], source_details: dict[str, Any]) -> dict[str, Any]:
    datasets = defaultdict(Counter)
    for row in rows:
        current = datasets[row["dataset"]]
        current["positive_samples"] += 1
        current["positive_routes"] += int(row["source_positive_route_count"])
        current["all_off_routes"] += sum(
            bool(route["source_all_off"]) for route in row["source_positive_routes"]
        )
        current["source_full_correct_samples"] += row.get("source_current_all_on_status") == "correct"
        current["source_full_wrong_samples"] += row.get("source_current_all_on_status") == "wrong"
        current["estimated_conversion_cost"] += int(row["estimated_conversion_cost"])
    return {
        "schema_version": "four_action_label_source_inventory_summary_v1",
        "definition": "all positive routes in the frozen training-authoritative source views",
        "datasets": {name: dict(counts) for name, counts in sorted(datasets.items())},
        "total_positive_samples": len(rows),
        "total_positive_routes": sum(row["source_positive_route_count"] for row in rows),
        "source_details": source_details,
    }


def main() -> int:
    args = parse_args()
    output_manifest = args.output_dir / "source_manifest_v1.jsonl"
    output_summary = args.output_dir / "source_inventory_summary_v1.json"
    if output_manifest.exists() or output_summary.exists():
        raise FileExistsError(f"source inventory already exists under {args.output_dir}")

    vqa_sources = {str(row["uid"]): row for row in iter_jsonl(args.vqa_source)}
    vqa_rows = []
    for predictor in iter_jsonl(args.vqa_predictor):
        uid = str(predictor["uid"])
        if uid not in vqa_sources:
            raise ValueError(f"VQA predictor UID absent from source manifest: {uid}")
        normalized = normalize_vqa_record(
            predictor,
            vqa_sources[uid],
            image_root=args.vqa_images,
            source_artifact=str(args.vqa_predictor.resolve()),
        )
        if normalized is not None:
            vqa_rows.append(normalized)

    image_hash_cache: dict[Path, str] = {}
    standard_image_root = args.standard_source / "images"
    standard_rows, standard_counts, standard_terminal_uids = _math_rows(
        cache_root=args.standard_cache,
        image_for_sample=lambda sample: standard_image_root
        / str(sample["image_content_sha256"])[:2]
        / Path(str(sample["local_image_path"])).name,
        expected_dataset="wemath20_standard",
        image_hash_cache=image_hash_cache,
    )
    pro_rows, pro_counts, _ = _math_rows(
        cache_root=args.pro_cache,
        image_for_sample=lambda sample: args.pro_images / Path(str(sample["local_image_path"])).name,
        expected_dataset="wemath2pro",
        image_hash_cache=image_hash_cache,
    )

    standard_manifest = args.standard_source / "manifest.jsonl"
    standard_expected_uids = {str(row["uid"]) for row in iter_jsonl(standard_manifest)}
    if not standard_terminal_uids <= standard_expected_uids:
        raise ValueError("Standard cache contains UIDs outside its frozen source manifest")
    source_details = {
        "vqa": {
            "predictor_manifest": str(args.vqa_predictor.resolve()),
            "predictor_manifest_sha256": file_sha256(args.vqa_predictor),
            "source_manifest": str(args.vqa_source.resolve()),
            "source_manifest_sha256": file_sha256(args.vqa_source),
            "positive_records": len(vqa_rows),
        },
        "wemath20_standard": {
            "cache_root": str(args.standard_cache.resolve()),
            "contract": str((args.standard_cache / "frozen_execution_contract.json").resolve()),
            "contract_sha256": file_sha256(args.standard_cache / "frozen_execution_contract.json"),
            "source_manifest": str(standard_manifest.resolve()),
            "source_manifest_sha256": file_sha256(standard_manifest),
            "expected_source_records": len(standard_expected_uids),
            "missing_terminal_records": len(standard_expected_uids - standard_terminal_uids),
            **standard_counts,
        },
        "wemath2pro": {
            "cache_root": str(args.pro_cache.resolve()),
            "contract": str((args.pro_cache / "frozen_execution_contract_cap400_v5.json").resolve()),
            "contract_sha256": file_sha256(
                args.pro_cache / "frozen_execution_contract_cap400_v5.json"
            ),
            **pro_counts,
        },
    }
    rows = sorted(
        [*vqa_rows, *standard_rows, *pro_rows],
        key=lambda row: (row["dataset"], row["uid"]),
    )
    seen_uids = Counter(row["uid"] for row in rows)
    duplicates = sorted(uid for uid, count in seen_uids.items() if count != 1)
    if duplicates:
        raise ValueError(f"duplicate positive source UIDs: {duplicates[:3]}")
    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    manifest_sha256 = write_atomic(output_manifest, manifest_text)
    summary = _summary(rows, source_details)
    summary["source_manifest_sha256"] = manifest_sha256
    write_atomic(output_summary, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
