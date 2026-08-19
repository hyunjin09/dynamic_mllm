from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.research_analysis.v3.prepare_null_geometry_manifest import main as prepare_manifest


ROOT = Path("artifacts/v3_null_calibration/read_write_geometry_v1")
CALIBRATION_MANIFEST = Path("data_manifests/v3_null_calibration_geometry_400.jsonl")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    prepare_manifest()
    manifest_digest = sha256(CALIBRATION_MANIFEST)
    for shard_index in range(8):
        shard = ROOT / "shards" / f"shard_{shard_index:02d}"
        geometry_path = shard / "geometry.jsonl"
        rows = [json.loads(line) for line in geometry_path.read_text().splitlines()]
        for row in rows:
            row.pop("selection_cell", None)
        if any("selection_cell" in row for row in rows):
            raise RuntimeError("Outcome-derived selection metadata remains")
        write_jsonl(geometry_path, rows)
        manifest_path = shard / "shard_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["manifest_sha256"] = manifest_digest
        manifest["geometry_sha256"] = sha256(geometry_path)
        manifest["answer_or_action_outcomes_loaded_or_used"] = False
        manifest["sanitization"] = (
            "removed unused inherited correct/wrong sampling-cell metadata; "
            "sample IDs, prompts, images, residual tensors, and tensor checksums unchanged"
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "calibration_manifest": str(CALIBRATION_MANIFEST),
                "sha256": manifest_digest,
                "shards": 8,
            }
        )
    )


if __name__ == "__main__":
    main()
