#!/usr/bin/env python3
"""Freeze the outcome-blind population/checkpoint contract for external eval."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path


ACTIVE = {
    "chartqa",
    "textvqa",
    "mmstar_val",
    "mmmu_val",
    "mmmu_pro_standard_test",
    "mmmu_pro_vision_test",
    "pope_adversarial",
    "pope_popular",
    "pope_random",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--question-checkpoint", type=Path, required=True)
    parser.add_argument("--image-question-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    bundle = args.bundle.resolve()
    mcts_manifest = project / "outputs/label_regeneration/v1/source_manifest_v1.jsonl"
    mcts_rows = read_jsonl(mcts_manifest)
    paths = list(dict.fromkeys(Path(row["local_image_path"]) for row in mcts_rows))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("an MCTS source image is missing")
    with ThreadPoolExecutor(max_workers=4) as pool:
        mcts_hashes = set(pool.map(file_sha256, paths))

    manifests = [
        bundle / "data/heldout_lmms_recommended_v1/samples.jsonl",
        bundle / "data/heldout_mmstar_mmmu_final_v2/samples.jsonl",
        bundle / "data/heldout_pope_v1/samples.jsonl",
    ]
    rows = [
        row
        for manifest in manifests
        for row in read_jsonl(manifest)
        if str(row["benchmark"]) in ACTIVE
    ]
    if len(rows) != 22307 or len({str(row["uid"]) for row in rows}) != 22307:
        raise RuntimeError("active external population is not exactly 22,307 unique UIDs")
    pope_overlap = []
    for row in rows:
        hashes = row.get("image_content_sha256s")
        if hashes is None:
            hashes = [row.get("image_content_sha256")]
        if str(row["benchmark"]).startswith("pope_") and any(value in mcts_hashes for value in hashes if value):
            pope_overlap.append(
                {
                    "uid": str(row["uid"]),
                    "benchmark": str(row["benchmark"]),
                    "image_hashes": [str(value) for value in hashes if value],
                }
            )
    clusters = sorted({value for row in pope_overlap for value in row["image_hashes"]})
    if len(pope_overlap) != 18 or len(clusters) != 1:
        raise RuntimeError(f"expected the frozen 18-record/one-image POPE overlap, got {len(pope_overlap)}/{len(clusters)}")

    contract = {
        "schema_version": "binary_polar_external_eval_contract_v1",
        "outcome_blind": True,
        "scientific_outcomes_loaded": False,
        "records": len(rows),
        "benchmark_counts": dict(sorted(Counter(str(row["benchmark"]) for row in rows).items())),
        "excluded_benchmarks": ["docvqa"],
        "checkpoints": {
            "question": {"path": str(args.question_checkpoint), "sha256": file_sha256(args.question_checkpoint)},
            "image_question": {"path": str(args.image_question_checkpoint), "sha256": file_sha256(args.image_question_checkpoint)},
        },
        "config": {"path": str(args.config), "sha256": file_sha256(args.config)},
        "bundle_manifests": [
            {"path": str(path), "sha256": file_sha256(path)} for path in manifests
        ],
        "mcts_source_manifest": {"path": str(mcts_manifest), "sha256": file_sha256(mcts_manifest)},
        "pope_image_overlap": {
            "cluster_keys": clusters,
            "excluded_sensitivity_uids": sorted(row["uid"] for row in pope_overlap),
            "records": len(pope_overlap),
            "official_population": 9000,
            "strict_image_disjoint_population": 8982,
        },
        "generation": {
            "decoding": "deterministic_greedy",
            "eos_token_ids": [151645],
            "repetition_penalty": 1.05,
            "mask_decoding": "logit_greater_than_or_equal_to_zero",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{file_sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "records": len(rows), "pope_overlap_records": len(pope_overlap)}))


if __name__ == "__main__":
    main()
