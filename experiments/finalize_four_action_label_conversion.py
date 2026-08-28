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

from tools.research_analysis.four_action.label_conversion import (
    select_diverse_four_action_routes,
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_filename(uid: str) -> str:
    readable = uid.replace(":", "__").replace("/", "_")
    return f"{readable}_{hashlib.sha256(uid.encode()).hexdigest()[:10]}.json"


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
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def jsonl(rows) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize four-action conversion label views.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl"),
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        default=Path("datasets/mcts_labels_4action/conversion_v1/full/records"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/mcts_labels_4action/conversion_v1/views"),
    )
    parser.add_argument("--route-cap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    sources = read_jsonl(args.source_manifest)
    results = []
    missing = []
    checksum_errors = []
    for source in sources:
        path = args.records_root / safe_filename(str(source["uid"]))
        if not path.is_file():
            missing.append(str(source["uid"]))
            continue
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.is_file() or sidecar.read_text().split()[0] != file_sha256(path):
            checksum_errors.append(str(path))
            continue
        result = json.loads(path.read_text())
        if result.get("uid") != source["uid"] or not result.get("passed"):
            raise ValueError(f"invalid sample result: {path}")
        results.append(result)
    if missing or checksum_errors:
        raise RuntimeError(
            f"conversion is incomplete: missing={len(missing)}, checksum_errors={len(checksum_errors)}"
        )
    contract_ids = sorted(
        {row["execution_contract"]["contract_sha256"] for row in results}
    )
    if len(contract_ids) != 1:
        raise RuntimeError("full results contain more than one execution contract")

    raw_rows = []
    unique_rows = []
    canonical_rows = []
    training_rows = []
    split_rows = []
    conversion_manifest_rows = []
    image_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source, result in zip(sources, results):
        contract = result["execution_contract"]
        common = {
            "uid": source["uid"],
            "dataset": source["dataset"],
            "sample_id": source["sample_id"],
            "image_id": source.get("image_id"),
            "image_group_id": source.get("image_group_id"),
            "source_split": source["source_split"],
            "label_semantics": result["label_semantics"],
            "current_unified_full": result["current_unified_full"],
            "current_unified_all_off": result["current_unified_all_off"],
            "execution_contract_sha256": contract["contract_sha256"],
            "model_revision": contract["model_revision"],
            "executor_sha256": contract["code_sha256"][
                "binary_policy/executor/four_action.py"
            ],
        }
        image_splits[(source["dataset"], str(source["image_group_id"]))].add(
            str(source["source_split"])
        )
        raw_rows.extend({**common, **row} for row in result["raw_conversions"])
        unique = result["unique_valid_four_action_routes"]
        unique_rows.append(
            {
                "schema_version": "four_action_unique_valid_sample_v1",
                **common,
                "source_positive_route_count": result["source_positive_route_count"],
                "source_route_replay_failure_count": result[
                    "source_route_replay_failure_count"
                ],
                "unique_valid_route_count": len(unique),
                "valid_routes": unique,
            }
        )
        canonical = result["canonical_4action_route"]
        if canonical is not None:
            canonical_rows.append(
                {"schema_version": "four_action_canonical_label_v1", **common, **canonical}
            )
            selected = select_diverse_four_action_routes(
                unique,
                limit=args.route_cap,
                seed=args.seed,
                uid=str(source["uid"]),
                canonical_route_key=str(canonical["route_key"]),
            )
            weight = 1.0 / len(selected)
            training_rows.append(
                {
                    "schema_version": "four_action_training_valid_set_v1",
                    **common,
                    "canonical_route_key": canonical["route_key"],
                    "full_unique_valid_route_count": len(unique),
                    "selected_valid_route_count": len(selected),
                    "route_cap": args.route_cap,
                    "route_cap_applied": len(unique) > args.route_cap,
                    "valid_routes": [
                        {**row, "weight": weight} for row in selected
                    ],
                }
            )
        split_rows.append(
            {
                "uid": source["uid"],
                "dataset": source["dataset"],
                "source_split": source["source_split"],
                "image_group_id": source.get("image_group_id"),
                "has_valid_current_conversion": canonical is not None,
            }
        )
        record_path = args.records_root / safe_filename(str(source["uid"]))
        conversion_manifest_rows.append(
            {
                "schema_version": "four_action_conversion_manifest_v1",
                "uid": source["uid"],
                "dataset": source["dataset"],
                "sample_id": source["sample_id"],
                "source_split": source["source_split"],
                "label_semantics": result["label_semantics"],
                "source_positive_route_count": result["source_positive_route_count"],
                "source_route_replay_valid_count": result[
                    "source_route_replay_valid_count"
                ],
                "source_route_replay_failure_count": result[
                    "source_route_replay_failure_count"
                ],
                "unique_valid_four_action_route_count": len(unique),
                "canonical_route_key": None if canonical is None else canonical["route_key"],
                "record_path": str(record_path.resolve()),
                "record_sha256": file_sha256(record_path),
                "execution_contract_sha256": contract["contract_sha256"],
            }
        )

    leakage = {
        f"{dataset}:{image}": sorted(splits)
        for (dataset, image), splits in image_splits.items()
        if len(splits) > 1
    }
    if leakage:
        raise RuntimeError(f"image groups cross source splits: {list(leakage)[:3]}")
    paths = {
        "raw_conversion": args.output_dir / "raw_conversion_v1.jsonl",
        "unique_valid": args.output_dir / "unique_valid_four_action_v1.jsonl",
        "canonical": args.output_dir / "canonical_four_action_v1.jsonl",
        "training": args.output_dir / "training_max50_four_action_v1.jsonl",
        "splits": args.output_dir / "split_manifest_v1.jsonl",
        "conversion_manifest": args.output_dir / "conversion_manifest_v1.jsonl",
    }
    hashes = {
        name: write_atomic(path, jsonl(rows))
        for name, path, rows in (
            ("raw_conversion", paths["raw_conversion"], raw_rows),
            ("unique_valid", paths["unique_valid"], unique_rows),
            ("canonical", paths["canonical"], canonical_rows),
            ("training", paths["training"], training_rows),
            ("splits", paths["splits"], split_rows),
            (
                "conversion_manifest",
                paths["conversion_manifest"],
                conversion_manifest_rows,
            ),
        )
    }
    summary = {
        "schema_version": "four_action_label_views_summary_v1",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "source_samples": len(sources),
        "source_positive_routes": sum(row["source_positive_route_count"] for row in sources),
        "raw_conversion_rows": len(raw_rows),
        "replay_valid_rows": sum(row["status"] == "converted" for row in raw_rows),
        "replay_failure_rows": sum(
            row["status"] == "source_route_replay_failure" for row in raw_rows
        ),
        "unique_valid_routes": sum(row["unique_valid_route_count"] for row in unique_rows),
        "canonical_samples": len(canonical_rows),
        "zero_current_valid_samples": len(sources) - len(canonical_rows),
        "training_samples": len(training_rows),
        "training_routes": sum(row["selected_valid_route_count"] for row in training_rows),
        "capped_training_samples": sum(row["route_cap_applied"] for row in training_rows),
        "dataset_source_samples": dict(Counter(row["dataset"] for row in sources)),
        "dataset_unique_routes": {
            dataset: sum(
                sample["unique_valid_route_count"]
                for sample in unique_rows
                if sample["dataset"] == dataset
            )
            for dataset in sorted({sample["dataset"] for sample in unique_rows})
        },
        "image_split_leakage_count": len(leakage),
        "execution_contract_sha256": contract_ids[0],
        "artifact_sha256": hashes,
    }
    write_atomic(
        args.output_dir / "label_views_summary_v1.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
