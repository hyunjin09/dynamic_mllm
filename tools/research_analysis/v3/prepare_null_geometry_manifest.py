from __future__ import annotations

import hashlib
import json
from pathlib import Path


SOURCE = Path("data_manifests/stage_b_discovery_candidates_400.jsonl")
OUTPUT = Path("data_manifests/v3_null_calibration_geometry_400.jsonl")


def main() -> None:
    rows = []
    with SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            rows.append(
                {
                    "id": str(source["id"]),
                    "dataset": str(source["benchmark"]),
                    "image_id": str(source["source_asset_id"]),
                    "prompt": str(source["prompt"]),
                    "local_image_path": str(source["local_image_path"]),
                }
            )
    if len(rows) != 400 or len({row["id"] for row in rows}) != 400:
        raise RuntimeError("Expected 400 unique inspected Stage B calibration records")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(json.dumps({"rows": len(rows), "output": str(OUTPUT), "sha256": digest}))


if __name__ == "__main__":
    main()
