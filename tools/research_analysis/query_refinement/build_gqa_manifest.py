from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from tools.research_analysis.v4.build_discovery_manifest import (
    GQA_ARROW,
    PROMPT_SUFFIX,
    RESERVED_AUDIT,
    SCENE_GRAPHS,
    arrow_rows,
    farthest_point_preflight,
    is_different_evidence,
    pair_match_distance,
    parse_id_list,
    program_depth,
    resolve_image,
    semantic_object_ids,
)
from experiments.stage_a_validity import prepare_prompt
from scoring.benchmark_metrics import normalize_exact


MODEL_CONFIG = Path("configs/model.yaml")
STAGE_B_MANIFEST = Path("data_manifests/stage_b_discovery_candidates_400.jsonl")
STAGE_A_MANIFEST = Path("outputs/stage_a/stage_a_requested_samples.jsonl")
V4_MANIFEST = Path("outputs/v4_discovery/manifest/v4_gqa_discovery_manifest_v1.jsonl")
V3_CALIBRATION = Path("outputs/v3_null_redesign/calibration_pool_manifest_v2.json")
OUTPUT_DIR = Path("outputs/query_refinement")
SELECTION_SEED = 2026080717
IMAGE_COUNT = 100
DIFFERENT_EVIDENCE_COUNT = 50
PREFLIGHT_IMAGE_COUNT = 12
VALIDATED_PROMPT_MAX = 4861


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the query-refinement GQA manifest.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def inspected_gqa_images() -> dict[str, set[str]]:
    stage_b = {
        str(row.get("source_asset_id") or "").split(":", 1)[-1]
        for row in read_jsonl(STAGE_B_MANIFEST)
        if row.get("benchmark") == "gqa"
    }
    stage_a = {
        str(row.get("source_asset_id") or "").split(":", 1)[-1]
        for row in read_jsonl(STAGE_A_MANIFEST)
        if row.get("benchmark") == "gqa"
    }
    v4 = {str(row["image_id"]) for row in read_jsonl(V4_MANIFEST)}
    calibration_payload = json.loads(V3_CALIBRATION.read_text(encoding="utf-8"))
    calibration = {
        str(row.get("source_image_id") or "")
        for row in calibration_payload["records"]
        if row.get("dataset") == "gqa"
    }
    return {
        "v2_stage_a": stage_a,
        "v2_v3_stage_b_discovery": stage_b,
        "v3_geometry_calibration": calibration,
        "v4_query_discovery": v4,
    }


def select_groups(
    groups: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
    scenes: dict[str, Any],
    excluded_images: set[str],
) -> list[dict[str, Any]]:
    candidates = []
    for group in groups:
        image_id = str(group["image_id"])
        if image_id in excluded_images:
            continue
        source_ids = [str(value).rsplit("_", 1)[-1] for value in group["question_ids"]]
        rows = [rows_by_id[source_id] for source_id in source_ids if source_id in rows_by_id]
        scene_ids = set((scenes.get(image_id) or {}).get("objects", {}))
        pairs = []
        for first, second in itertools.combinations(rows, 2):
            ordered_ids = sorted((str(first["source_id"]), str(second["source_id"])))
            pairs.append(
                {
                    "first": first,
                    "second": second,
                    "different_evidence": is_different_evidence(first, second, scene_ids),
                    "match_distance": pair_match_distance(first, second),
                    "pair_hash": stable_hash(SELECTION_SEED, image_id, *ordered_ids),
                }
            )
        if not pairs:
            continue
        different = [pair for pair in pairs if pair["different_evidence"]]
        candidates.append(
            {
                "image_id": image_id,
                "reservation_hash": group["reservation_hash"],
                "different_pair": min(different, key=lambda row: row["pair_hash"])
                if different
                else None,
                "matched_pair": min(
                    pairs, key=lambda row: (row["match_distance"], row["pair_hash"])
                ),
            }
        )

    different_groups = sorted(
        (row for row in candidates if row["different_pair"] is not None),
        key=lambda row: (row["different_pair"]["pair_hash"], row["image_id"]),
    )
    if len(different_groups) < DIFFERENT_EVIDENCE_COUNT:
        raise RuntimeError("Insufficient new different-evidence image groups")
    selected = [
        {**row, "selected_pair": row["different_pair"], "pair_stratum": "different_evidence"}
        for row in different_groups[:DIFFERENT_EVIDENCE_COUNT]
    ]
    used = {row["image_id"] for row in selected}
    matched_groups = sorted(
        (row for row in candidates if row["image_id"] not in used),
        key=lambda row: (
            row["matched_pair"]["match_distance"],
            stable_hash(SELECTION_SEED, "matched", row["image_id"]),
        ),
    )
    needed = IMAGE_COUNT - len(selected)
    if len(matched_groups) < needed:
        raise RuntimeError("Insufficient new matched image groups")
    selected.extend(
        {**row, "selected_pair": row["matched_pair"], "pair_stratum": "matched_comparison"}
        for row in matched_groups[:needed]
    )
    return sorted(selected, key=lambda row: stable_hash(SELECTION_SEED, "selected", row["image_id"]))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    reserved = json.loads(RESERVED_AUDIT.read_text(encoding="utf-8"))
    groups = list(reserved["gqa"]["groups"])
    reserved_ids = {
        str(value).rsplit("_", 1)[-1] for group in groups for value in group["question_ids"]
    }
    columns = [
        "id",
        "imageId",
        "question",
        "answer",
        "fullAnswer",
        "groups",
        "entailed",
        "equivalent",
        "types",
        "annotations",
        "semantic",
        "semanticStr",
    ]
    rows_by_id: dict[str, dict[str, Any]] = {}
    for source in arrow_rows(GQA_ARROW, columns):
        source_id = str(source["id"])
        if source_id not in reserved_ids:
            continue
        source["source_id"] = source_id
        source["image_id"] = str(source["imageId"])
        source["answer"] = normalize_exact(str(source["answer"]))
        rows_by_id[source_id] = source
    if len(rows_by_id) != len(reserved_ids):
        raise RuntimeError(f"Resolved {len(rows_by_id)}/{len(reserved_ids)} reserved questions")

    model_config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(
        model_config["snapshot_path"], local_files_only=True, use_fast=False
    )
    hf_config = AutoConfig.from_pretrained(model_config["snapshot_path"], local_files_only=True)
    image_token_id = int(hf_config.image_token_id)
    for row in rows_by_id.values():
        token_ids = processor.tokenizer(row["answer"], add_special_tokens=False).input_ids
        if not token_ids:
            raise RuntimeError(f"Empty accepted-answer span: {row['source_id']}")
        row["answer_token_ids"] = [int(value) for value in token_ids]
        row["answer_token_length"] = len(token_ids)

    inspected = inspected_gqa_images()
    excluded = set().union(*inspected.values())
    scenes = json.loads(SCENE_GRAPHS.read_text(encoding="utf-8"))
    selected = select_groups(groups, rows_by_id, scenes, excluded)
    invalid = Counter()
    manifest_rows = []
    group_rows = []
    for image_index, group in enumerate(selected):
        image_id = group["image_id"]
        image_path = resolve_image(image_id)
        with Image.open(image_path) as image:
            image.verify()
        pair = group["selected_pair"]
        prepared = []
        for question_index, source in enumerate((pair["first"], pair["second"])):
            question = str(source["question"]).strip()
            record = {
                "schema_version": "query_refinement_gqa_manifest_v1",
                "id": f"gqa:gqa_val_{source['source_id']}",
                "sample_id": f"gqa_val_{source['source_id']}",
                "benchmark": "gqa",
                "dataset": "gqa",
                "source_split": "val_balanced_instructions",
                "source_revision": "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8",
                "source_id": source["source_id"],
                "image_id": image_id,
                "source_asset_id": f"gqa:{image_id}",
                "local_image_path": str(image_path),
                "question": question,
                "prompt": f"{question}\n{PROMPT_SUFFIX}",
                "answer": source["answer"],
                "metric_name": "exact_match_ignore_case_punctuation",
                "image_index": image_index,
                "question_index": question_index,
                "pair_stratum": group["pair_stratum"],
                "different_evidence": bool(pair["different_evidence"]),
                "pair_match_distance": float(pair["match_distance"]),
                "pair_selection_hash": pair["pair_hash"],
                "reservation_hash": group["reservation_hash"],
                "semantic_object_ids": sorted(semantic_object_ids(source)),
                "semantic_program_depth": program_depth(source),
                "semantic_program": source.get("semantic"),
                "semantic_str": source.get("semanticStr"),
                "question_types": source.get("types"),
                "question_groups": source.get("groups"),
                "equivalent_ids": sorted(parse_id_list(source.get("equivalent"))),
                "entailed_ids": sorted(parse_id_list(source.get("entailed"))),
                "question_word_length": len(question.split()),
                "answer_token_ids": source["answer_token_ids"],
                "answer_token_length": source["answer_token_length"],
            }
            prompt_text, inputs = prepare_prompt(processor, record, torch.device("cpu"))
            input_ids = inputs["input_ids"]
            visual_indices = torch.where(input_ids[0] == image_token_id)[0]
            if visual_indices.numel() == 0:
                invalid["no_visual_tokens"] += 1
                raise RuntimeError(f"No visual tokens: {record['id']}")
            prompt_length = int(input_ids.shape[1])
            if prompt_length > VALIDATED_PROMPT_MAX:
                invalid["prompt_too_long"] += 1
                raise RuntimeError(f"Prompt exceeds validated maximum: {record['id']}")
            record.update(
                {
                    "prompt_text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                    "expected_prompt_token_length": prompt_length,
                    "expected_visual_token_count": int(visual_indices.numel()),
                    "expected_visual_first": int(visual_indices[0]),
                    "expected_visual_last": int(visual_indices[-1]),
                    "expected_image_grid_thw": [
                        int(value) for value in inputs["image_grid_thw"].reshape(-1).tolist()
                    ],
                }
            )
            prepared.append(record)
            manifest_rows.append(record)
        if prepared[0]["expected_visual_token_count"] != prepared[1]["expected_visual_token_count"]:
            raise RuntimeError(f"Same-image visual token count mismatch: {image_id}")
        if (prepared[0]["expected_visual_first"], prepared[0]["expected_visual_last"]) != (
            prepared[1]["expected_visual_first"],
            prepared[1]["expected_visual_last"],
        ):
            raise RuntimeError(f"Same-image visual-token position mismatch: {image_id}")
        group_rows.append(
            {
                "image_id": image_id,
                "image_index": image_index,
                "question_ids": [row["id"] for row in prepared],
                "pair_stratum": group["pair_stratum"],
                "different_evidence": bool(pair["different_evidence"]),
                "pair_match_distance": float(pair["match_distance"]),
                "common_prompt_token_length": max(
                    row["expected_prompt_token_length"] for row in prepared
                ),
                "visual_token_count": prepared[0]["expected_visual_token_count"],
            }
        )

    preflight_images = farthest_point_preflight(group_rows, PREFLIGHT_IMAGE_COUNT)
    for row in manifest_rows:
        row["preflight_selected"] = row["image_id"] in preflight_images
    manifest_path = output_dir / "gqa_discovery_manifest_v1.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    manifest_sha = sha256_file(manifest_path)
    (output_dir / "gqa_discovery_manifest_v1.sha256").write_text(
        f"{manifest_sha}  {manifest_path.name}\n", encoding="utf-8"
    )

    selected_images = {row["image_id"] for row in manifest_rows}
    overlap = {name: len(selected_images & values) for name, values in inspected.items()}
    audit = {
        "schema_version": "query_refinement_gqa_manifest_audit_v1",
        "outcome_blind": True,
        "intervention_outcomes_loaded_or_computed": False,
        "selection_seed": SELECTION_SEED,
        "selection_rule": (
            "from the prospective v3 GQA multi-question reserve after excluding every known "
            "v2-v4 inspected/calibration image, choose 50 SHA256-ranked unambiguous "
            "different-scene-object pairs and 50 unused minimum-metadata-distance pairs; "
            "order by SHA256(seed:selected:image_id)"
        ),
        "counts": {
            "records": len(manifest_rows),
            "unique_images": len(selected_images),
            "questions_per_image": {"2": len(selected_images)},
            "different_evidence_stratum_images": sum(
                row["pair_stratum"] == "different_evidence" for row in group_rows
            ),
            "matched_comparison_stratum_images": sum(
                row["pair_stratum"] == "matched_comparison" for row in group_rows
            ),
            "actual_disjoint_evidence_pairs": sum(
                row["different_evidence"] for row in group_rows
            ),
            "preflight_images": len(preflight_images),
            "eligible_reserved_images_after_exclusions": sum(
                str(group["image_id"]) not in excluded for group in groups
            ),
        },
        "overlap": overlap,
        "technical_invalid_counts": dict(invalid),
        "preflight_image_ids": preflight_images,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "source_checksums": {
            str(GQA_ARROW): sha256_file(GQA_ARROW),
            str(SCENE_GRAPHS): sha256_file(SCENE_GRAPHS),
            str(RESERVED_AUDIT): sha256_file(RESERVED_AUDIT),
            str(MODEL_CONFIG): sha256_file(MODEL_CONFIG),
            str(STAGE_B_MANIFEST): sha256_file(STAGE_B_MANIFEST),
            str(STAGE_A_MANIFEST): sha256_file(STAGE_A_MANIFEST),
            str(V4_MANIFEST): sha256_file(V4_MANIFEST),
            str(V3_CALIBRATION): sha256_file(V3_CALIBRATION),
        },
        "groups": group_rows,
    }
    if len(manifest_rows) != 200 or len(selected_images) != 100:
        raise RuntimeError("Manifest must contain exactly 200 records over 100 images")
    if any(overlap.values()):
        raise RuntimeError(f"Inspected-image overlap gate failed: {overlap}")
    write_json(output_dir / "gqa_discovery_manifest_audit_v1.json", audit)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "sha256": manifest_sha,
                "records": len(manifest_rows),
                "images": len(selected_images),
                "overlap": overlap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
