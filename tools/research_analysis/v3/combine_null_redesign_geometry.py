from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LAYERS = [0, 4, 8, 12, 16, 20, 24]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine checksummed initial and enlargement geometry roots.")
    parser.add_argument(
        "--initial-root", default="artifacts/v3_null_redesign/read_write_geometry_v2"
    )
    parser.add_argument(
        "--delta-root", default="artifacts/v3_null_redesign/read_write_geometry_delta_v2"
    )
    parser.add_argument(
        "--output-root", default="artifacts/v3_null_redesign/read_write_geometry_combined_v3"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(args: argparse.Namespace) -> None:
    roots = [Path(args.initial_root), Path(args.delta_root)]
    manifests = [json.loads((root / "manifest.json").read_text()) for root in roots]
    if any(item["sample_count"] != 2000 or item["sample_layer_count"] != 14000 for item in manifests):
        raise RuntimeError("Each source geometry root must contain 2,000 complete records")
    rows = []
    index = []
    for root, manifest in zip(roots, manifests):
        geometry = root / "geometry.jsonl"
        tensor_index = root / "tensor_index.json"
        if sha256(geometry) != manifest["geometry_sha256"] or sha256(tensor_index) != manifest["tensor_index_sha256"]:
            raise RuntimeError(f"Source root checksum mismatch: {root}")
        rows.extend(json.loads(line) for line in geometry.read_text().splitlines() if line)
        index.extend(json.loads(tensor_index.read_text()))
    keys = {(row["sample_id"], int(row["layer"])) for row in rows}
    paths = {row["path"] for row in index}
    if len(rows) != 28000 or len(keys) != 28000 or len(index) != 4000 or len(paths) != 4000:
        raise RuntimeError("Combined geometry has duplicate or missing records")
    output = Path(args.output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["dataset"], row["sample_id"], int(row["layer"])))
    geometry_output = output / "geometry.jsonl"
    with geometry_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    index_output = output / "tensor_index.json"
    index_output.write_text(json.dumps(sorted(index, key=lambda row: row["path"]), indent=2) + "\n")
    manifest_output = output / "manifest.json"
    manifest_output.write_text(
        json.dumps(
            {
                "schema_version": "v3_null_redesign_combined_geometry_v1",
                "outcome_blind": True,
                "answer_or_action_outcomes_loaded_or_used": False,
                "sample_count": 4000,
                "sample_layer_count": 28000,
                "dataset_counts": {"gqa": 2000, "textvqa": 2000},
                "layers": LAYERS,
                "max_reconstruction_error": max(item["max_reconstruction_error"] for item in manifests),
                "source_manifests": {
                    str(root / "manifest.json"): sha256(root / "manifest.json")
                    for root in roots
                },
                "geometry_sha256": sha256(geometry_output),
                "tensor_index_sha256": sha256(index_output),
                "tensor_integrity": "transitively verified by the two source manifests and tensor-index checksums",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path}\n"
            for path in (geometry_output, index_output, manifest_output)
        )
    )
    print(json.dumps({"manifest": str(manifest_output), "sha256": sha256(manifest_output)}, sort_keys=True))


if __name__ == "__main__":
    execute(parse_args())
