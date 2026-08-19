from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LAYERS = [0, 4, 8, 12, 16, 20, 24]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge checksummed v3 geometry shards.")
    parser.add_argument(
        "--root", default="artifacts/v3_null_calibration/read_write_geometry_v1"
    )
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--expected-samples", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    tensor_index = []
    shard_checksums = {}
    for index in range(args.num_shards):
        shard = root / "shards" / f"shard_{index:02d}"
        manifest_path = shard / "shard_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["shard_index"] != index or manifest["num_shards"] != args.num_shards:
            raise RuntimeError(f"Shard identity mismatch: {manifest_path}")
        geometry_path = shard / "geometry.jsonl"
        if sha256(geometry_path) != manifest["geometry_sha256"]:
            raise RuntimeError(f"Geometry checksum mismatch: {geometry_path}")
        rows.extend(json.loads(line) for line in geometry_path.read_text().splitlines())
        for raw_path, digest in manifest["tensor_checksums"].items():
            path = Path(raw_path)
            if sha256(path) != digest:
                raise RuntimeError(f"Tensor checksum mismatch: {path}")
            tensor_index.append({"path": str(path), "sha256": digest})
        shard_checksums[str(manifest_path)] = sha256(manifest_path)
    keys = [(row["sample_id"], int(row["layer"])) for row in rows]
    expected_rows = args.expected_samples * len(LAYERS)
    if len(rows) != expected_rows or len(set(keys)) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} unique sample-layer rows; got {len(rows)}"
        )
    if {int(row["layer"]) for row in rows} != set(LAYERS):
        raise RuntimeError("Layer grid mismatch")
    if len(tensor_index) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} tensor files; got {len(tensor_index)}"
        )
    rows.sort(key=lambda row: (row["dataset"], row["sample_id"], int(row["layer"])))
    geometry_path = root / "geometry.jsonl"
    with geometry_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    index_path = root / "tensor_index.json"
    index_path.write_text(
        json.dumps(sorted(tensor_index, key=lambda row: row["path"]), indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "v3_read_write_geometry_manifest_v1",
                "outcome_blind": True,
                "answer_or_action_outcomes_loaded_or_used": False,
                "sample_count": args.expected_samples,
                "sample_layer_count": expected_rows,
                "layers": LAYERS,
                "dataset_counts": {
                    dataset: len({row["sample_id"] for row in rows if row["dataset"] == dataset})
                    for dataset in ("gqa", "textvqa")
                },
                "max_reconstruction_error": max(
                    max(
                        row["read_hook_reconstruction_max_abs"],
                        row["read_layer_reconstruction_max_abs"],
                        row["write_hook_reconstruction_max_abs"],
                    )
                    for row in rows
                ),
                "geometry_sha256": sha256(geometry_path),
                "tensor_index_sha256": sha256(index_path),
                "shard_manifest_checksums": shard_checksums,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksum_path = root / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{sha256(path)}  {path}\n"
            for path in (geometry_path, index_path, manifest_path)
        ),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256(manifest_path)}))


if __name__ == "__main__":
    main()
