#!/usr/bin/env python3
"""Write deterministic SHA-256 inventory for every transfer-bundle file."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


EXCLUDED = {"BUNDLE_SHA256SUMS", "bundle_manifest.json"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in EXCLUDED and "__pycache__" not in path.parts
    )
    entries = [(path.relative_to(root).as_posix(), sha256(path.read_bytes()).hexdigest()) for path in files]
    manifest = {
        "schema_version": "binary_visual_mcts_reproduction_bundle_v1",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "transformers_version": "5.3.0",
        "route_width": 28,
        "file_count_excluding_inventory": len(entries),
        "files": {name: digest for name, digest in entries},
    }
    (root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inventory_entries = entries + [
        ("bundle_manifest.json", sha256((root / "bundle_manifest.json").read_bytes()).hexdigest())
    ]
    (root / "BUNDLE_SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in inventory_entries), encoding="utf-8"
    )
    print(f"wrote checksums for {len(entries)} files")


if __name__ == "__main__":
    main()
