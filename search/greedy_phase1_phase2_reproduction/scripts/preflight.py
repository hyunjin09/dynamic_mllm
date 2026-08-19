#!/usr/bin/env python3
"""Fail-fast validation for the relocated Phase-1+2 reproduction inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any, Iterable


EXPECTED_CELL_COUNTS = {
    ("chartqa", "train", "complete_correct"): 885,
    ("chartqa", "train", "complete_wrong"): 910,
    ("chartqa", "validation", "complete_correct"): 115,
    ("chartqa", "validation", "complete_wrong"): 90,
    ("docvqa", "train", "complete_correct"): 872,
    ("docvqa", "train", "complete_wrong"): 900,
    ("docvqa", "validation", "complete_correct"): 128,
    ("docvqa", "validation", "complete_wrong"): 100,
    ("gqa", "train", "complete_correct"): 1784,
    ("gqa", "train", "complete_wrong"): 1784,
    ("gqa", "validation", "complete_correct"): 216,
    ("gqa", "validation", "complete_wrong"): 216,
    ("textvqa", "train", "complete_correct"): 893,
    ("textvqa", "train", "complete_wrong"): 887,
    ("textvqa", "validation", "complete_correct"): 107,
    ("textvqa", "validation", "complete_wrong"): 113,
}

EXPECTED_RUNTIME = {
    "torch": "2.9.1+cu128",
    "transformers": "4.57.1",
    "accelerate": "1.11.0",
    "qwen-vl-utils": "0.0.14",
    "pillow": "12.0.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=10000)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-semantic-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--verify-image-hashes", action="store_true")
    parser.add_argument("--strict-runtime", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    model = args.model_source.resolve()
    manifest = args.manifest.resolve()
    config_path = args.config.resolve()
    required = [
        model / "config.json",
        manifest,
        config_path,
        project / "analysis_outputs" / "harmful_validation_common.py",
        project / "analysis_outputs" / "run_harmful_interventions.py",
        project / "dvr_qwen" / "generate.py",
        project / "dvr_qwen" / "binary_generate.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required files are missing: {missing}")
    if model.name != args.expected_revision:
        raise RuntimeError(f"wrong model snapshot: expected {args.expected_revision}, got {model.name}")

    model_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    text_config = model_config.get("text_config") or model_config
    num_layers = int(text_config.get("num_hidden_layers", -1))
    if num_layers != 28:
        raise RuntimeError(f"expected 28 language layers, got {num_layers}")

    collection = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(config_path)
    if config_sha256 != args.expected_config_sha256:
        raise RuntimeError(
            f"collection config hash mismatch: expected {args.expected_config_sha256}, got {config_sha256}"
        )
    expected_orders = [
        "early_to_late",
        "late_to_early",
        "center_out",
        "outside_in",
        "random:20260714",
        "random:20260715",
        "random:20260716",
        "random:20260717",
        "random:20260718",
        "random:20260719",
    ]
    if collection.get("search", {}).get("orders") != expected_orders:
        raise RuntimeError("collection config does not contain the canonical ten search orders")

    required_fields = {
        "uid",
        "benchmark",
        "data_split",
        "source_bucket",
        "local_image_path",
        "image_content_sha256",
        "question",
        "answer",
        "metric_name",
        "correctness_threshold",
        "source_full_score",
        "source_full_prediction",
        "max_new_tokens",
    }
    rows = 0
    uids: set[str] = set()
    cells: Counter[tuple[str, str, str]] = Counter()
    semantic = hashlib.sha256()
    missing_images: list[str] = []
    for row in iter_jsonl(manifest):
        absent = required_fields - row.keys()
        if absent:
            raise RuntimeError(f"manifest row {rows + 1} is missing fields: {sorted(absent)}")
        uid = str(row["uid"])
        if uid in uids:
            raise RuntimeError(f"duplicate UID: {uid}")
        uids.add(uid)
        cells[(str(row["benchmark"]), str(row["data_split"]), str(row["source_bucket"]))] += 1
        image_path = Path(str(row["local_image_path"]))
        if not image_path.is_file():
            missing_images.append(str(image_path))
        elif args.verify_image_hashes and sha256_file(image_path) != row["image_content_sha256"]:
            raise RuntimeError(f"image checksum mismatch: {image_path}")
        normalized = dict(row)
        normalized.pop("local_image_path", None)
        payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        semantic.update((payload + "\n").encode("utf-8"))
        rows += 1

    if rows != args.expected_samples:
        raise RuntimeError(f"manifest row count mismatch: expected {args.expected_samples}, got {rows}")
    if cells != Counter(EXPECTED_CELL_COUNTS):
        raise RuntimeError(f"canonical benchmark/split/source-bucket counts changed: {dict(cells)}")
    semantic_hash = semantic.hexdigest()
    if semantic_hash != args.expected_semantic_sha256:
        raise RuntimeError(
            f"path-invariant manifest hash mismatch: expected {args.expected_semantic_sha256}, got {semantic_hash}"
        )
    if missing_images:
        raise RuntimeError(f"{len(missing_images)} images are missing; first paths: {missing_images[:10]}")

    versions = {name: package_version(name) for name in EXPECTED_RUNTIME}
    runtime_mismatches = {
        name: {"expected": expected, "actual": versions[name]}
        for name, expected in EXPECTED_RUNTIME.items()
        if versions[name] != expected
    }
    if args.strict_runtime and runtime_mismatches:
        raise RuntimeError(f"canonical runtime mismatch: {runtime_mismatches}")

    report = {
        "decision": "pass_reproduction_preflight",
        "python": sys.version.split()[0],
        "runtime_versions": versions,
        "runtime_mismatches": runtime_mismatches,
        "strict_runtime": bool(args.strict_runtime),
        "project_root": str(project),
        "model_source": str(model),
        "model_revision": model.name,
        "num_layers": num_layers,
        "manifest": str(manifest),
        "manifest_rows": rows,
        "manifest_semantic_sha256": semantic_hash,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "search_orders": expected_orders,
        "image_hashes_verified": bool(args.verify_image_hashes),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
