from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from audit.stage_c_manifest import (
    blocking_overlap_reasons,
    normalize_path,
    normalize_question,
    overlap_reasons,
    record_checksum,
    select_unique_images,
)
from experiments.stage_b_reference_likelihood import answer_tokenization_audit
from scoring.reference_likelihood import accepted_answers


FROZEN_INVALID_RULES = (
    "image_unavailable_or_unreadable",
    "question_missing",
    "accepted_answers_missing",
    "accepted_answer_normalization_failed",
    "accepted_answer_weights_invalid",
    "prompt_construction_failed",
    "answer_token_span_empty",
    "answer_token_span_misaligned",
    "prompt_context_limit_exceeded",
    "image_token_indices_invalid",
    "pinned_processor_or_tokenizer_failed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the outcome-blind Stage C manifest.")
    parser.add_argument("--config", default="configs/stage_c_entry.yaml")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def canonical_image_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{rgb.width}:{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def stage_b_identifier_sets(rows: list[dict[str, Any]]) -> dict[str, set[Any]]:
    result: dict[str, set[Any]] = {
        "ids": set(),
        "question_ids": set(),
        "annotation_ids": set(),
        "image_ids": set(),
        "image_hashes": set(),
        "normalized_image_paths": set(),
        "normalized_questions": set(),
        "image_question_pairs": set(),
    }
    for row in rows:
        if row.get("benchmark") != "textvqa":
            continue
        sample_id = str(row["id"])
        question_id = int(str(row["sample_id"]).rsplit("_", 1)[-1])
        image_id = str(row["source_asset_id"]).split(":", 1)[-1]
        question = normalize_question(row["question"])
        image_path = Path(row["local_image_path"])
        with Image.open(image_path) as image:
            image_hash = canonical_image_hash(image)
        result["ids"].add(sample_id)
        result["question_ids"].add(question_id)
        result["annotation_ids"].add(question_id)
        result["image_ids"].add(image_id)
        result["image_hashes"].add(image_hash)
        result["normalized_image_paths"].add(normalize_path(str(image_path)))
        result["normalized_questions"].add(question)
        result["image_question_pairs"].add((image_id, question))
    return result


def prompt_for(processor, question: str, suffix: str) -> tuple[str, str]:
    prompt = f"{question}\n{suffix}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt, rendered


def validate_candidate(
    index: int,
    source: dict[str, Any],
    processor,
    model_config,
    prompt_suffix: str,
    max_prompt_tokens: int,
) -> tuple[dict[str, Any] | None, str | None]:
    question = str(source.get("question") or "").strip()
    if not question:
        return None, "question_missing"
    raw_answers = list(source.get("answers") or [])
    if not raw_answers:
        return None, "accepted_answers_missing"
    image = source.get("image")
    if image is None:
        return None, "image_unavailable_or_unreadable"
    try:
        image = image.convert("RGB")
        image.load()
        image_hash = canonical_image_hash(image)
    except Exception:
        return None, "image_unavailable_or_unreadable"

    scoring_record = {
        "benchmark": "textvqa",
        "answer": str(raw_answers[0]),
        "all_answer_norms": raw_answers,
    }
    try:
        answers = accepted_answers(scoring_record)
    except ValueError:
        return None, "accepted_answer_normalization_failed"
    weights = [float(item.weight) for item in answers]
    if any(weight <= 0.0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
        return None, "accepted_answer_weights_invalid"

    try:
        prompt, prompt_text = prompt_for(processor, question, prompt_suffix)
        batch = processor(
            text=[prompt_text], images=[image], padding=True, return_tensors="pt"
        )
    except Exception:
        return None, "prompt_construction_failed"
    input_ids = batch["input_ids"]
    prompt_tokens = int(input_ids.shape[1])
    if prompt_tokens > max_prompt_tokens:
        return None, "prompt_context_limit_exceeded"
    visual_indices = torch.where(input_ids[0] == int(model_config.image_token_id))[0]
    if visual_indices.numel() < 1:
        return None, "image_token_indices_invalid"
    first_visual = int(visual_indices[0].item())
    last_visual = int(visual_indices[-1].item())
    if visual_indices.numel() != last_visual - first_visual + 1:
        return None, "image_token_indices_invalid"

    tokenization = [
        answer_tokenization_audit(processor.tokenizer, prompt_text, answer.text)
        for answer in answers
    ]
    if not all(item["answer_token_length"] > 0 for item in tokenization):
        return None, "answer_token_span_empty"
    if not all(
        item["prompt_is_exact_combined_prefix"]
        and item["standalone_answer_matches_combined_suffix"]
        and item["prompt_positions_contributing_to_score"] == 0
        for item in tokenization
    ):
        return None, "answer_token_span_misaligned"

    image_id = str(source["image_id"])
    question_id = int(source["question_id"])
    normalized_question = normalize_question(question)
    image_grid = batch.get("image_grid_thw")
    row = {
        "schema_version": "stage_c_manifest_v1",
        "id": f"textvqa:textvqa_validation_{question_id}",
        "sample_id": f"textvqa_validation_{question_id}",
        "dataset": "textvqa",
        "source_dataset": "lmms-lab/textvqa",
        "source_split": "validation",
        "source_dataset_index": index,
        "image_id": image_id,
        "question_id": question_id,
        "annotation_id": question_id,
        "annotation_id_source": "TextVQA question_id annotation key",
        "question": question,
        "normalized_question": normalized_question,
        "image_question_key": f"{image_id}:{normalized_question}",
        "image_sha256": image_hash,
        "accepted_answers": [
            {"answer": item.text, "weight": item.weight}
            for item in answers
        ],
        "accepted_answer_tokenization": tokenization,
        "prompt": prompt,
        "prompt_template": "{question}\\nAnswer the question using a single word or phrase.",
        "prompt_text": prompt_text,
        "answer_prefix": "",
        "eos_in_primary_score": False,
        "prompt_token_length": prompt_tokens,
        "image_token_first": first_visual,
        "image_token_last": last_visual,
        "image_token_count": int(visual_indices.numel()),
        "postvisual_text_token_count": prompt_tokens - last_visual - 1,
        "nonvisual_token_count": prompt_tokens - int(visual_indices.numel()),
        "image_grid_thw": image_grid[0].tolist() if image_grid is not None else None,
        "image_width": int(image.width),
        "image_height": int(image.height),
        "local_image_path": None,
        "local_image_file_sha256": None,
        "target_answer_tokens_in_prompt_score": 0,
    }
    return row, None


def execute(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path(config["model_config"]).read_text(encoding="utf-8"))
    snapshot = model_cfg["snapshot_path"]
    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, use_fast=False
    )
    architecture = AutoConfig.from_pretrained(snapshot, local_files_only=True)

    dataset = load_dataset(
        config["dataset_id"],
        revision=config["dataset_revision"],
        split=config["dataset_split"],
        cache_dir=config["dataset_cache"],
    )
    stage_b_rows = read_jsonl(Path(config["stage_b_manifest"]))
    discovery = stage_b_identifier_sets(stage_b_rows)

    invalid_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    invalid_records: list[dict[str, Any]] = []
    overlap_records: list[dict[str, Any]] = []
    question_only_overlap_records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for index, source in enumerate(dataset):
        row, invalid_reason = validate_candidate(
            index,
            source,
            processor,
            architecture,
            str(config["prompt_suffix"]),
            int(config["validated_prompt_token_max"]),
        )
        if invalid_reason is not None:
            invalid_counts[invalid_reason] += 1
            invalid_records.append(
                {
                    "source_dataset_index": index,
                    "question_id": source.get("question_id"),
                    "reason": invalid_reason,
                }
            )
            continue
        assert row is not None
        reasons = overlap_reasons(row, discovery)
        if reasons:
            for reason in reasons:
                overlap_counts[reason] += 1
            blocking_reasons = blocking_overlap_reasons(reasons)
            if blocking_reasons:
                overlap_records.append({"id": row["id"], "reasons": blocking_reasons})
                continue
            question_only_overlap_records.append(
                {"id": row["id"], "normalized_question": row["normalized_question"]}
            )
            row["stage_b_normalized_question_only_match"] = True
        else:
            row["stage_b_normalized_question_only_match"] = False
        eligible.append(row)

    selected = select_unique_images(
        eligible, count=int(config["sample_count"]), seed=int(config["selection_seed"])
    )
    image_dir = Path(config["selected_image_dir"])
    image_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        source = dataset[int(row["source_dataset_index"])]
        image = source["image"].convert("RGB")
        if canonical_image_hash(image) != row["image_sha256"]:
            raise RuntimeError(f"Image pixels changed while freezing {row['id']}")
        image_path = image_dir / f"{row['image_id']}.png"
        if image_path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen image: {image_path}")
        image.save(image_path, format="PNG", optimize=False)
        row["local_image_path"] = str(image_path)
        row["local_image_file_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        row["record_sha256"] = record_checksum(row)

    manifest_path = Path(config["manifest_path"])
    write_jsonl(manifest_path, selected)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sha_path = Path(config["manifest_sha256_path"])
    sha_path.parent.mkdir(parents=True, exist_ok=True)
    sha_path.write_text(f"{manifest_sha}  {manifest_path}\n", encoding="utf-8")

    unique_eligible_images = len({row["image_id"] for row in eligible})
    audit = {
        "schema_version": "stage_c_eligibility_overlap_audit_v1",
        "outcome_blind": True,
        "stage_c_intervention_scores_computed": False,
        "source_dataset": config["dataset_id"],
        "source_dataset_revision": config["dataset_revision"],
        "source_split": config["dataset_split"],
        "source_fingerprint": getattr(dataset, "_fingerprint", None),
        "source_candidate_count": len(dataset),
        "stage_b_textvqa_count": len(discovery["ids"]),
        "frozen_invalid_rules": list(FROZEN_INVALID_RULES),
        "invalid_count": sum(invalid_counts.values()),
        "invalid_counts_by_reason": dict(sorted(invalid_counts.items())),
        "invalid_records": invalid_records,
        "overlap_record_count": len(overlap_records),
        "overlap_counts_by_identifier": dict(sorted(overlap_counts.items())),
        "overlap_records": overlap_records,
        "question_only_overlap_count": len(question_only_overlap_records),
        "question_only_overlap_records": question_only_overlap_records,
        "question_only_overlap_policy": "reported but non-blocking because question text alone does not identify a record or image",
        "eligible_record_count": len(eligible),
        "eligible_unique_image_count": unique_eligible_images,
        "selected_count": len(selected),
        "selected_unique_image_count": len({row["image_id"] for row in selected}),
        "selected_stage_b_id_overlap": len({row["id"] for row in selected} & discovery["ids"]),
        "selected_stage_b_image_overlap": len({row["image_id"] for row in selected} & discovery["image_ids"]),
        "selected_stage_b_image_hash_overlap": len({row["image_sha256"] for row in selected} & discovery["image_hashes"]),
        "selected_duplicate_image_count": len(selected) - len({row["image_id"] for row in selected}),
        "selection_seed": int(config["selection_seed"]),
        "selection_method": "ascending SHA256(seed:record_id), first record per unique image",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
    }
    write_json(Path(config["audit_json_path"]), audit)
    print(json.dumps({key: value for key, value in audit.items() if not key.endswith("records")}, indent=2))


if __name__ == "__main__":
    execute(Path(parse_args().config))
