#!/usr/bin/env python3
"""Rewrite only the local image-path prefix while preserving sample semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--new-prefix", required=True)
    parser.add_argument("--verify-images", action="store_true")
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


def semantic_digest(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        normalized = dict(row)
        normalized.pop("local_image_path", None)
        payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        digest.update((payload + "\n").encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    destination = args.output.resolve()
    if source == destination:
        raise ValueError("input and output must be different files")

    old_prefix = str(Path(args.old_prefix))
    new_prefix = str(Path(args.new_prefix))
    rows = list(iter_jsonl(source))
    source_semantic = semantic_digest(rows)
    missing: list[str] = []
    rewritten = 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            image_path = str(row.get("local_image_path") or "")
            if not image_path.startswith(old_prefix):
                raise RuntimeError(f"image path does not start with old prefix: {image_path}")
            relocated = new_prefix + image_path[len(old_prefix) :]
            row["local_image_path"] = relocated
            row["local_image_exists"] = Path(relocated).is_file()
            if not row["local_image_exists"]:
                missing.append(relocated)
            elif args.verify_images:
                expected = row.get("image_content_sha256")
                if expected and sha256_file(Path(relocated)) != expected:
                    raise RuntimeError(f"image checksum mismatch: {relocated}")
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            rewritten += 1

    if missing:
        temporary.unlink(missing_ok=True)
        preview = "\n".join(missing[:10])
        raise RuntimeError(f"{len(missing)} relocated images are missing; first paths:\n{preview}")
    os.replace(temporary, destination)

    output_rows = list(iter_jsonl(destination))
    output_semantic = semantic_digest(output_rows)
    if output_semantic != source_semantic:
        raise RuntimeError("semantic manifest hash changed during relocation")
    report = {
        "decision": "pass_path_only_manifest_relocation",
        "input": str(source),
        "input_sha256": sha256_file(source),
        "output": str(destination),
        "output_sha256": sha256_file(destination),
        "semantic_sha256": output_semantic,
        "rows": rewritten,
        "old_prefix": old_prefix,
        "new_prefix": new_prefix,
        "image_checksums_verified": bool(args.verify_images),
    }
    report_path = destination.with_suffix(destination.suffix + ".relocation.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
