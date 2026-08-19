from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.ipc as ipc
from PIL import Image

from tools.research_analysis.v3.confirmation_preflight import (
    choose_one_record_per_image,
    deterministic_rank,
    image_group_summary,
    reserve_multi_question_groups,
)


GQA_ARROW = Path(
    "/data/dataset/huggingface/datasets/lmms-lab___gqa/"
    "val_balanced_instructions/0.0.0/"
    "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8/gqa-val.arrow"
)
TEXTVQA_DIR = Path(
    "/data/dataset/huggingface/datasets/lmms-lab___textvqa/default/0.0.0/"
    "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
)
VG_ROOTS = (
    Path("/data/dataset/VG/VG_100K"),
    Path("/data/dataset/VG/VG_100K_2"),
)
PROMPT_SUFFIX = "Answer the question using a single word or phrase."
CONFIRMATION_PER_DATASET = 800
STAGE_C2_GROUPS_PER_DATASET = 800
SELECTION_SEED = 2026080602
STAGE_C2_SEED = 2026080603


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build outcome-blind v3 pool audits.")
    parser.add_argument(
        "--output", default="outputs/v3_preflight/candidate_pool_audit.json"
    )
    parser.add_argument(
        "--stage-c2-output",
        default="outputs/v3_preflight/stage_c2_reserved_pool_audit.json",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_image_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{rgb.width}:{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def arrow_rows(paths: Iterable[Path], columns: list[str]) -> Iterable[dict[str, Any]]:
    for path in sorted(paths):
        with path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            for batch in reader:
                selected = batch.select(columns)
                yield from selected.to_pylist()


def resolve_gqa_image(image_id: str) -> Path | None:
    for root in VG_ROOTS:
        path = root / f"{image_id}.jpg"
        if path.is_file():
            return path
    return None


def inspected_sets() -> dict[str, set[str]]:
    stage_a = read_jsonl(Path("outputs/stage_a/stage_a_samples.jsonl"))
    stage_b = read_jsonl(Path("data_manifests/stage_b_discovery_candidates_400.jsonl"))
    stage_c = read_jsonl(Path("outputs/stage_c/manifest/stage_c_manifest_v1.jsonl"))
    result = {
        "gqa_image_ids": set(),
        "gqa_record_ids": set(),
        "textvqa_image_ids": set(),
        "textvqa_record_ids": set(),
        "textvqa_image_hashes": set(),
        "textvqa_stage_a_b_image_hashes": set(),
        "textvqa_stage_c_image_hashes": set(),
    }
    for row in stage_a + stage_b:
        dataset = row.get("benchmark") or row.get("dataset")
        asset = str(row.get("source_asset_id") or "")
        if dataset == "gqa" and asset:
            result["gqa_image_ids"].add(asset.split(":", 1)[-1])
            result["gqa_record_ids"].add(str(row["id"]))
        if dataset == "textvqa":
            result["textvqa_record_ids"].add(str(row["id"]))
            image_id = asset.split(":", 1)[-1] if asset else ""
            if image_id and "/" not in image_id and "." not in image_id:
                result["textvqa_image_ids"].add(image_id)
            local_path = row.get("local_image_path")
            if local_path and Path(local_path).is_file():
                with Image.open(local_path) as image:
                    image_hash = canonical_image_hash(image)
                    result["textvqa_image_hashes"].add(image_hash)
                    result["textvqa_stage_a_b_image_hashes"].add(image_hash)
    for row in stage_c:
        result["textvqa_record_ids"].add(str(row["id"]))
        result["textvqa_image_ids"].add(str(row["image_id"]))
        result["textvqa_image_hashes"].add(str(row["image_sha256"]))
        result["textvqa_stage_c_image_hashes"].add(str(row["image_sha256"]))
    return result


def audit_gqa(excluded: dict[str, set[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    invalid = Counter()
    overlaps = Counter()
    checked_images: dict[str, tuple[str | None, str | None]] = {}
    for source in arrow_rows([GQA_ARROW], ["id", "imageId", "question", "answer"]):
        record_id = f"gqa:gqa_val_{source['id']}"
        image_id = str(source.get("imageId") or "")
        question = str(source.get("question") or "").strip()
        answer = str(source.get("answer") or "").strip()
        if not question:
            invalid["question_missing"] += 1
            continue
        if not answer:
            invalid["accepted_answers_missing"] += 1
            continue
        if image_id not in checked_images:
            image_path = resolve_gqa_image(image_id)
            if image_path is None:
                checked_images[image_id] = (None, "image_unavailable")
            else:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    checked_images[image_id] = (str(image_path), None)
                except Exception:
                    checked_images[image_id] = (None, "image_unreadable")
        image_path, image_error = checked_images[image_id]
        if image_error:
            invalid[image_error] += 1
            continue
        if record_id in excluded["gqa_record_ids"]:
            overlaps["record_id"] += 1
            continue
        if image_id in excluded["gqa_image_ids"]:
            overlaps["image_id"] += 1
            continue
        rows.append(
            {
                "id": record_id,
                "source_id": str(source["id"]),
                "image_id": image_id,
                "question": question,
                "answer": answer,
                "local_image_path": image_path,
            }
        )
    audit = {
        "source_record_count": len(rows) + sum(invalid.values()) + sum(overlaps.values()),
        "eligible_metadata_record_count": len(rows),
        "eligible_metadata": image_group_summary(rows),
        "invalid_counts": dict(sorted(invalid.items())),
        "inspected_overlap_counts": dict(sorted(overlaps.items())),
        "image_availability_unique_count": sum(error is None for _, error in checked_images.values()),
        "image_unavailable_unique_count": sum(error is not None for _, error in checked_images.values()),
        "validity_level": "metadata, image availability/readability; frozen processor/token rules deferred to deterministic manifest construction",
    }
    return rows, audit


def audit_textvqa(excluded: dict[str, set[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen_audit = json.loads(
        Path("outputs/stage_c/manifest/stage_c_eligibility_overlap_audit_v1.json").read_text(
            encoding="utf-8"
        )
    )
    invalid_question_ids = {
        int(item["question_id"]): item["reason"] for item in frozen_audit["invalid_records"]
    }
    rows: list[dict[str, Any]] = []
    invalid = Counter()
    overlaps = Counter()
    hash_resolved_overlap_image_ids: set[str] = set()
    stage_a_b_hash_overlap_image_ids: set[str] = set()
    stage_c_hash_overlap_image_ids: set[str] = set()
    paths = TEXTVQA_DIR.glob("textvqa-validation-*.arrow")
    for source in arrow_rows(paths, ["image_id", "question_id", "question", "answers", "image"]):
        question_id = int(source["question_id"])
        image_id = str(source["image_id"])
        record_id = f"textvqa:textvqa_validation_{question_id}"
        if question_id in invalid_question_ids:
            invalid[invalid_question_ids[question_id]] += 1
            continue
        image_bytes = source["image"]["bytes"]
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_hash = canonical_image_hash(image)
        if image_hash in excluded["textvqa_image_hashes"]:
            hash_resolved_overlap_image_ids.add(image_id)
        if image_hash in excluded["textvqa_stage_a_b_image_hashes"]:
            stage_a_b_hash_overlap_image_ids.add(image_id)
        if image_hash in excluded["textvqa_stage_c_image_hashes"]:
            stage_c_hash_overlap_image_ids.add(image_id)
        if record_id in excluded["textvqa_record_ids"]:
            overlaps["record_id"] += 1
            continue
        if image_id in excluded["textvqa_image_ids"]:
            overlaps["image_id"] += 1
            continue
        if image_hash in excluded["textvqa_image_hashes"]:
            overlaps["canonical_image_hash"] += 1
            continue
        rows.append(
            {
                "id": record_id,
                "source_id": question_id,
                "image_id": image_id,
                "question": str(source["question"]).strip(),
                "answers": list(source["answers"]),
                "image_sha256": image_hash,
            }
        )
    audit = {
        "source_record_count": len(rows) + sum(invalid.values()) + sum(overlaps.values()),
        "eligible_record_count": len(rows),
        "eligible": image_group_summary(rows),
        "invalid_counts_reused_from_frozen_v2_audit": dict(sorted(invalid.items())),
        "inspected_overlap_counts": dict(sorted(overlaps.items())),
        "hash_resolved_inspected_image_ids": sorted(hash_resolved_overlap_image_ids),
        "hash_resolution_count": len(hash_resolved_overlap_image_ids),
        "stage_a_b_canonical_hash_overlap_count": len(stage_a_b_hash_overlap_image_ids),
        "stage_a_b_canonical_hash_overlap_image_ids": sorted(stage_a_b_hash_overlap_image_ids),
        "stage_c_canonical_hash_overlap_count": len(stage_c_hash_overlap_image_ids),
        "validity_level": "reuses every frozen v2 processor/token eligibility result and resolves inspected images by ID and canonical RGB hash",
    }
    return rows, audit


def compact_identity(row: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "id": row["id"],
        "image_id": row["image_id"],
        "selection_hash": deterministic_rank(str(row["id"]), seed),
    }


def execute(output: Path, stage_c2_output: Path) -> None:
    excluded = inspected_sets()
    gqa_rows, gqa_audit = audit_gqa(excluded)
    textvqa_rows, textvqa_audit = audit_textvqa(excluded)

    # Preserve every remaining two-question TextVQA image for Stage C2.
    text_counts = Counter(str(row["image_id"]) for row in textvqa_rows)
    text_singletons = [row for row in textvqa_rows if text_counts[str(row["image_id"])] == 1]
    proposed_gqa = choose_one_record_per_image(
        gqa_rows, CONFIRMATION_PER_DATASET, SELECTION_SEED
    )
    proposed_textvqa = choose_one_record_per_image(
        text_singletons, CONFIRMATION_PER_DATASET, SELECTION_SEED
    )
    confirm_images = {
        "gqa": {str(row["image_id"]) for row in proposed_gqa},
        "textvqa": {str(row["image_id"]) for row in proposed_textvqa},
    }
    gqa_reserved = reserve_multi_question_groups(
        gqa_rows, confirm_images["gqa"], STAGE_C2_GROUPS_PER_DATASET, STAGE_C2_SEED
    )
    text_reserved = reserve_multi_question_groups(
        textvqa_rows,
        confirm_images["textvqa"],
        STAGE_C2_GROUPS_PER_DATASET,
        STAGE_C2_SEED,
    )

    sources = {
        str(GQA_ARROW): file_sha256(GQA_ARROW),
        "outputs/stage_a/stage_a_samples.jsonl": file_sha256(Path("outputs/stage_a/stage_a_samples.jsonl")),
        "data_manifests/stage_b_discovery_candidates_400.jsonl": file_sha256(Path("data_manifests/stage_b_discovery_candidates_400.jsonl")),
        "outputs/stage_c/manifest/stage_c_manifest_v1.jsonl": file_sha256(Path("outputs/stage_c/manifest/stage_c_manifest_v1.jsonl")),
    }
    candidate_audit = {
        "schema_version": "v3_candidate_pool_audit_v1",
        "outcome_blind": True,
        "terminal_action_values_loaded_or_computed": False,
        "model_loaded": False,
        "sample_size_rule": {
            "joint_total": 2 * CONFIRMATION_PER_DATASET,
            "per_dataset": {"gqa": CONFIRMATION_PER_DATASET, "textvqa": CONFIRMATION_PER_DATASET},
            "one_record_per_unique_image": True,
        },
        "selection": {
            "seed": SELECTION_SEED,
            "rank": "ascending SHA256(seed:record_id), first technically valid record per unique image",
            "textvqa_pool_restriction": "singleton images only, preserving every remaining multi-question image for Stage C2",
            "manifest_frozen": False,
            "manifest_freeze_condition": "after explicit confirmation approval, apply frozen processor/token rules in rank order and freeze before any terminal intervention score",
        },
        "overlap_exclusions": {
            "v2_stage_a": True,
            "v2_stage_b_and_v3_discovery": True,
            "v2_stage_c": True,
            "null_calibration_stage_b": True,
            "record_and_image_overlap_allowed": False,
        },
        "gqa": gqa_audit,
        "textvqa": textvqa_audit,
        "proposed_manifest_identity_preview": {
            "not_a_frozen_manifest": True,
            "gqa": [compact_identity(row, SELECTION_SEED) for row in proposed_gqa],
            "textvqa": [compact_identity(row, SELECTION_SEED) for row in proposed_textvqa],
        },
        "proposed_preview_checks": {
            "gqa_unique_images": len(confirm_images["gqa"]),
            "textvqa_unique_images": len(confirm_images["textvqa"]),
            "cross_dataset_image_id_overlap": len(confirm_images["gqa"] & confirm_images["textvqa"]),
            "inspected_gqa_image_overlap": len(confirm_images["gqa"] & excluded["gqa_image_ids"]),
            "inspected_textvqa_image_overlap": len(confirm_images["textvqa"] & excluded["textvqa_image_ids"]),
        },
        "sources": sources,
    }
    write_json(output, candidate_audit)

    def reserved_payload(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return [
            {
                "image_id": image_id,
                "question_ids": [row["id"] for row in group],
                "available_question_count": len(group),
                "reservation_hash": deterministic_rank(image_id, STAGE_C2_SEED),
            }
            for image_id, group in groups.items()
        ]

    stage_c2_audit = {
        "schema_version": "v3_stage_c2_reserved_pool_audit_v1",
        "outcome_blind_metadata_only": True,
        "terminal_action_values_loaded_or_computed": False,
        "reservation_seed": STAGE_C2_SEED,
        "reservation_rule": "after inspected and proposed Stage C image exclusions, ascending SHA256(seed:image_id), first 800 images with at least two eligible questions; all questions retained in metadata and the first two sorted IDs define the future paired minimum",
        "gqa": {
            "available_after_proposed_stage_c": image_group_summary(
                row for row in gqa_rows if str(row["image_id"]) not in confirm_images["gqa"]
            ),
            "reserved_group_count": len(gqa_reserved),
            "reserved_question_count": sum(len(group) for group in gqa_reserved.values()),
            "feasible_image_grouped_split_sizes": [200, 400, 800],
            "groups": reserved_payload(gqa_reserved),
        },
        "textvqa": {
            "available_after_proposed_stage_c": image_group_summary(
                row for row in textvqa_rows if str(row["image_id"]) not in confirm_images["textvqa"]
            ),
            "reserved_group_count": len(text_reserved),
            "reserved_question_count": sum(len(group) for group in text_reserved.values()),
            "feasible_image_grouped_split_sizes": [200, 400, 800],
            "groups": reserved_payload(text_reserved),
        },
        "disjointness": {
            "gqa_reserved_vs_proposed_stage_c_images": len(set(gqa_reserved) & confirm_images["gqa"]),
            "textvqa_reserved_vs_proposed_stage_c_images": len(set(text_reserved) & confirm_images["textvqa"]),
            "inspected_gqa_images": len(set(gqa_reserved) & excluded["gqa_image_ids"]),
            "inspected_textvqa_images": len(set(text_reserved) & excluded["textvqa_image_ids"]),
        },
    }
    write_json(stage_c2_output, stage_c2_audit)
    print(json.dumps({
        "candidate_output": str(output),
        "stage_c2_output": str(stage_c2_output),
        "gqa_eligible_images": gqa_audit["eligible_metadata"]["image_count"],
        "textvqa_eligible_images": textvqa_audit["eligible"]["image_count"],
        "textvqa_hash_resolved_overlap_images": textvqa_audit["hash_resolution_count"],
    }, indent=2))


if __name__ == "__main__":
    args = parse_args()
    execute(Path(args.output), Path(args.stage_c2_output))
