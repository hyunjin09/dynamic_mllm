from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.ipc as ipc
import torch
import yaml
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from experiments.stage_a_validity import prepare_prompt
from scoring.benchmark_metrics import normalize_exact


GQA_ARROW = Path(
    "/data/dataset/huggingface/datasets/lmms-lab___gqa/"
    "val_balanced_instructions/0.0.0/"
    "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8/gqa-val.arrow"
)
SCENE_GRAPHS = Path("/data/dataset/GQA/sceneGraphs_v1.1/val_sceneGraphs.json")
RESERVED_AUDIT = Path("outputs/v3_preflight/stage_c2_reserved_pool_audit.json")
MODEL_CONFIG = Path("configs/model.yaml")
OUTPUT_DIR = Path("outputs/v4_discovery/manifest")
PROMPT_SUFFIX = "Answer the question using a single word or phrase."
SELECTION_SEED = 2026080711
IMAGE_COUNT = 120
DIFFERENT_EVIDENCE_COUNT = 60
PREFLIGHT_IMAGE_COUNT = 12
VALIDATED_PROMPT_MAX = 4861


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the outcome-blind v4 GQA manifest.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def arrow_rows(path: Path, columns: list[str]) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        reader = ipc.open_stream(handle)
        for batch in reader:
            yield from batch.select(columns).to_pylist()


def resolve_image(image_id: str) -> Path:
    for root in (Path("/data/dataset/VG/VG_100K"), Path("/data/dataset/VG/VG_100K_2")):
        candidate = root / f"{image_id}.jpg"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"GQA image unavailable: {image_id}")


def parse_id_list(raw: object) -> set[str]:
    try:
        value = ast.literal_eval(str(raw or "[]"))
    except (SyntaxError, ValueError):
        return set()
    return {str(item) for item in value} if isinstance(value, list) else set()


def semantic_object_ids(row: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    annotations = row.get("annotations") or {}
    for field in ("question", "answer", "fullAnswer"):
        for item in annotations.get(field) or []:
            value = str(item.get("value") or "")
            if value.isdigit():
                result.add(value)
    for match in re.findall(r"\((\d+)\)", str(row.get("semanticStr") or "")):
        result.add(match)
    return result


def program_depth(row: dict[str, Any]) -> int:
    return len(row.get("semantic") or [])


def question_words(row: dict[str, Any]) -> int:
    return len(str(row.get("question") or "").split())


def pair_match_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_types = first.get("types") or {}
    second_types = second.get("types") or {}
    structural = float(first_types.get("structural") != second_types.get("structural"))
    semantic = float(first_types.get("semantic") != second_types.get("semantic"))
    depth = abs(program_depth(first) - program_depth(second))
    question = abs(question_words(first) - question_words(second)) / 10.0
    answer = abs(int(first["answer_token_length"]) - int(second["answer_token_length"]))
    return 2.0 * structural + 2.0 * semantic + depth + question + answer


def is_official_paraphrase(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_id = str(first["source_id"])
    second_id = str(second["source_id"])
    linked = second_id in parse_id_list(first.get("equivalent")) or first_id in parse_id_list(
        second.get("equivalent")
    )
    return (
        linked
        and normalize_exact(str(first["answer"])) == normalize_exact(str(second["answer"]))
        and (first.get("types") or {}) == (second.get("types") or {})
        and semantic_object_ids(first) == semantic_object_ids(second)
    )


def is_different_evidence(
    first: dict[str, Any], second: dict[str, Any], scene_object_ids: set[str]
) -> bool:
    first_ids = semantic_object_ids(first)
    second_ids = semantic_object_ids(second)
    return bool(
        first_ids
        and second_ids
        and first_ids.isdisjoint(second_ids)
        and first_ids <= scene_object_ids
        and second_ids <= scene_object_ids
    )


def select_pairs(
    groups: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
    scenes: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for group in groups:
        image_id = str(group["image_id"])
        scene_ids = set((scenes.get(image_id) or {}).get("objects", {}))
        source_ids = [str(value).rsplit("_", 1)[-1] for value in group["question_ids"]]
        rows = [rows_by_id[source_id] for source_id in source_ids if source_id in rows_by_id]
        pair_rows = []
        for first, second in itertools.combinations(rows, 2):
            ids = sorted((str(first["source_id"]), str(second["source_id"])))
            pair_rows.append(
                {
                    "image_id": image_id,
                    "first": first,
                    "second": second,
                    "pair_hash": stable_hash(SELECTION_SEED, image_id, *ids),
                    "different_evidence": is_different_evidence(first, second, scene_ids),
                    "official_paraphrase": is_official_paraphrase(first, second),
                    "match_distance": pair_match_distance(first, second),
                }
            )
        if not pair_rows:
            continue
        different = [row for row in pair_rows if row["different_evidence"]]
        candidates.append(
            {
                "image_id": image_id,
                "reservation_hash": group["reservation_hash"],
                "different_pair": min(different, key=lambda row: row["pair_hash"])
                if different
                else None,
                "matched_pair": min(
                    pair_rows, key=lambda row: (row["match_distance"], row["pair_hash"])
                ),
                "available_question_count": len(rows),
            }
        )

    different_images = sorted(
        (row for row in candidates if row["different_pair"] is not None),
        key=lambda row: (row["different_pair"]["pair_hash"], row["image_id"]),
    )
    if len(different_images) < DIFFERENT_EVIDENCE_COUNT:
        raise RuntimeError(
            f"Only {len(different_images)} reserved images have unambiguous different-evidence pairs"
        )
    selected = [
        {**row, "selected_pair": row["different_pair"], "pair_stratum": "different_evidence"}
        for row in different_images[:DIFFERENT_EVIDENCE_COUNT]
    ]
    used = {row["image_id"] for row in selected}
    matched_images = sorted(
        (row for row in candidates if row["image_id"] not in used),
        key=lambda row: (
            row["matched_pair"]["match_distance"],
            stable_hash(SELECTION_SEED, "matched", row["image_id"]),
        ),
    )
    needed = IMAGE_COUNT - len(selected)
    if len(matched_images) < needed:
        raise RuntimeError(f"Only {len(matched_images)} matched image groups; {needed} required")
    selected.extend(
        {**row, "selected_pair": row["matched_pair"], "pair_stratum": "matched_comparison"}
        for row in matched_images[:needed]
    )
    selected.sort(key=lambda row: stable_hash(SELECTION_SEED, "selected", row["image_id"]))
    return selected


def farthest_point_preflight(groups: list[dict[str, Any]], count: int) -> list[str]:
    if len(groups) < count:
        raise ValueError("Not enough groups for preflight selection")
    lengths = [float(group["common_prompt_token_length"]) for group in groups]
    visuals = [float(group["visual_token_count"]) for group in groups]

    def scale(value: float, values: list[float]) -> float:
        low, high = min(values), max(values)
        return 0.0 if high == low else (value - low) / (high - low)

    coords = {
        group["image_id"]: (
            scale(float(group["common_prompt_token_length"]), lengths),
            scale(float(group["visual_token_count"]), visuals),
        )
        for group in groups
    }
    ranked = sorted(groups, key=lambda row: stable_hash(SELECTION_SEED, "preflight", row["image_id"]))
    chosen = [ranked[0]["image_id"]]
    while len(chosen) < count:
        remaining = [row["image_id"] for row in groups if row["image_id"] not in chosen]

        def minimum_distance(image_id: str) -> float:
            x, y = coords[image_id]
            return min(math.hypot(x - coords[other][0], y - coords[other][1]) for other in chosen)

        next_id = min(
            remaining,
            key=lambda image_id: (
                -minimum_distance(image_id),
                stable_hash(SELECTION_SEED, "preflight", image_id),
            ),
        )
        chosen.append(next_id)
    return chosen


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
    tokenizer = processor.tokenizer
    for row in rows_by_id.values():
        token_ids = tokenizer(row["answer"], add_special_tokens=False).input_ids
        if not token_ids:
            raise RuntimeError(f"Empty accepted-answer span: {row['source_id']}")
        row["answer_token_ids"] = [int(value) for value in token_ids]
        row["answer_token_length"] = len(token_ids)

    scenes = json.loads(SCENE_GRAPHS.read_text(encoding="utf-8"))
    selected = select_pairs(groups, rows_by_id, scenes)

    manifest_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    invalid = Counter()
    for image_index, group in enumerate(selected):
        image_id = group["image_id"]
        image_path = resolve_image(image_id)
        with Image.open(image_path) as image:
            image.verify()
        pair = group["selected_pair"]
        prepared = []
        for question_index, source in enumerate((pair["first"], pair["second"])):
            record = {
                "schema_version": "v4_gqa_discovery_manifest_v1",
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
                "question": str(source["question"]).strip(),
                "prompt": f"{str(source['question']).strip()}\n{PROMPT_SUFFIX}",
                "answer": source["answer"],
                "metric_name": "exact_match_ignore_case_punctuation",
                "image_index": image_index,
                "question_index": question_index,
                "pair_stratum": group["pair_stratum"],
                "different_evidence": bool(pair["different_evidence"]),
                "official_paraphrase": bool(pair["official_paraphrase"]),
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
        if (
            prepared[0]["expected_visual_first"],
            prepared[0]["expected_visual_last"],
        ) != (
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
                "official_paraphrase": bool(pair["official_paraphrase"]),
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
    manifest_path = output_dir / "v4_gqa_discovery_manifest_v1.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    manifest_sha = sha256_file(manifest_path)
    (output_dir / "v4_gqa_discovery_manifest_v1.sha256").write_text(
        f"{manifest_sha}  {manifest_path.name}\n", encoding="utf-8"
    )

    calibration = json.loads(
        Path("outputs/v3_null_redesign/calibration_pool_manifest_v2.json").read_text(
            encoding="utf-8"
        )
    )
    calibration_images = {
        str(row.get("source_image_id") or "")
        for row in calibration["records"]
        if row["dataset"] == "gqa"
    }
    selected_images = {row["image_id"] for row in manifest_rows}
    selected_ids = {row["id"] for row in manifest_rows}
    stage_b_rows = [
        json.loads(line)
        for line in Path("data_manifests/stage_b_discovery_candidates_400.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    stage_b_images = {
        str(row.get("source_asset_id") or "").split(":", 1)[-1]
        for row in stage_b_rows
        if row.get("benchmark") == "gqa"
    }
    stage_b_ids = {str(row["id"]) for row in stage_b_rows}
    audit = {
        "schema_version": "v4_gqa_discovery_manifest_audit_v1",
        "outcome_blind": True,
        "terminal_action_values_loaded_or_computed": False,
        "selection_seed": SELECTION_SEED,
        "selection_rule": (
            "within the frozen 800-image GQA Stage C2 reserve, select 60 SHA256-ranked "
            "unambiguous disjoint-scene-object pairs, then 60 unused minimum-metadata-distance "
            "pairs; final order is SHA256(seed:selected:image_id)"
        ),
        "counts": {
            "records": len(manifest_rows),
            "unique_images": len(selected_images),
            "questions_per_image": {"2": len(selected_images)},
            "different_evidence_images": sum(
                row["pair_stratum"] == "different_evidence" for row in group_rows
            ),
            "matched_comparison_images": sum(
                row["pair_stratum"] == "matched_comparison" for row in group_rows
            ),
            "official_paraphrase_images": sum(row["official_paraphrase"] for row in group_rows),
            "preflight_images": len(preflight_images),
        },
        "preflight_image_ids": preflight_images,
        "preflight_selection": (
            "deterministic farthest-point coverage in normalized common prompt length and "
            "visual-token-count geometry"
        ),
        "overlap": {
            "stage_b_record_ids": len(selected_ids & stage_b_ids),
            "stage_b_image_ids": len(selected_images & stage_b_images),
            "v3_independent_calibration_image_ids": len(selected_images & calibration_images),
            "reserved_audit_inspected_gqa_images": reserved["disjointness"]["inspected_gqa_images"],
            "reserved_vs_proposed_v3_stage_c_images": reserved["disjointness"][
                "gqa_reserved_vs_proposed_stage_c_images"
            ],
        },
        "technical_invalid_counts": dict(invalid),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "source_checksums": {
            str(GQA_ARROW): sha256_file(GQA_ARROW),
            str(SCENE_GRAPHS): sha256_file(SCENE_GRAPHS),
            str(RESERVED_AUDIT): sha256_file(RESERVED_AUDIT),
            str(MODEL_CONFIG): sha256_file(MODEL_CONFIG),
        },
        "groups": group_rows,
    }
    if len(manifest_rows) != 240 or len(selected_images) != 120:
        raise RuntimeError("Manifest does not contain exactly 240 records over 120 images")
    if any(audit["overlap"].values()):
        raise RuntimeError(f"Manifest overlap gate failed: {audit['overlap']}")
    write_json(output_dir / "v4_gqa_discovery_manifest_audit_v1.json", audit)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "sha256": manifest_sha,
                "records": len(manifest_rows),
                "images": len(selected_images),
                "different_evidence": audit["counts"]["different_evidence_images"],
                "official_paraphrases": audit["counts"]["official_paraphrase_images"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
