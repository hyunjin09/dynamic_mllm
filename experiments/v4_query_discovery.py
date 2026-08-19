from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from tools.research_analysis.v3.confirmation_preflight import right_pad_prompt_inputs
from experiments.stage_a_validity import max_abs_difference, prepare_prompt, set_determinism
from experiments.stage_b_reference_likelihood import (
    accepted_answers,
    capture_prompt_with_cache,
    score_accepted_answer_set,
)
from interventions.four_state import FOUR_STATES
from interventions.prompt_cache import run_cached_prompt_state, truncate_dynamic_cache
from interventions.read_path import ReadInterventionCache
from scoring.reference_likelihood import factorial_effects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bounded v4 GQA discovery.")
    parser.add_argument("--config", default="configs/v4_discovery.yaml")
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_manifest_sha(checksum_path: Path) -> str:
    parts = checksum_path.read_text(encoding="utf-8").strip().split()
    if not parts:
        raise ValueError("Manifest checksum file is empty")
    return parts[0]


def load_model(model_config: dict[str, Any]):
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(
        model_config["snapshot_path"], local_files_only=True, use_fast=False
    )
    hf_config = AutoConfig.from_pretrained(model_config["snapshot_path"], local_files_only=True)
    hf_config._attn_implementation = {"vision_config": model_config["vision_attention_backend"]}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_config["snapshot_path"],
        config=hf_config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    for layer in model.model.layers:
        layer.self_attn.stage_a_query_chunk_size = int(
            model_config["decoder_attention_query_chunk_size"]
        )
    return model, processor


def runtime_metadata(model, processor, model_config: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "model_id": model_config["model_id"],
        "revision": model_config["revision"],
        "snapshot_path": model_config["snapshot_path"],
        "dtype": str(next(model.parameters()).dtype),
        "decoder_attention_backend": "stock_eager",
        "vision_attention_backend": model.config.vision_config._attn_implementation,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(processor.tokenizer).__name__,
        "chat_template": processor.chat_template,
        "layer_grid": layers,
        "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "common_right_padding": True,
        "padding_cache_handling": "crop every layer cache to original prompt boundary before answer continuation",
        "precision": "bfloat16 model; float32 log-softmax",
    }


def rms_relative(first: torch.Tensor, second: torch.Tensor) -> float:
    difference = (first.float() - second.float()).square().mean().sqrt()
    scale = torch.stack(
        [first.float().square().mean().sqrt(), second.float().square().mean().sqrt()]
    ).mean()
    return float((difference / scale.clamp_min(1e-12)).item())


def group_manifest(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_id"])].append(row)
    result = []
    for image_id, group in groups.items():
        ordered = sorted(group, key=lambda row: int(row["question_index"]))
        if len(ordered) != 2 or [int(row["question_index"]) for row in ordered] != [0, 1]:
            raise ValueError(f"Image {image_id} does not have exactly two ordered questions")
        result.append(ordered)
    return sorted(result, key=lambda group: int(group[0]["image_index"]))


def prepare_common_group(processor, model, group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    device = torch.device("cuda")
    prepared: list[dict[str, Any]] = []
    for record in group:
        prompt_text, raw = prepare_prompt(processor, record, device)
        original_length = int(raw["input_ids"].shape[1])
        if original_length != int(record["expected_prompt_token_length"]):
            raise RuntimeError(f"Prompt length drift for {record['id']}")
        prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        if prompt_sha != record["prompt_text_sha256"]:
            raise RuntimeError(f"Prompt text drift for {record['id']}")
        position_ids, rope_delta = model.get_rope_index(
            input_ids=raw["input_ids"],
            image_grid_thw=raw.get("image_grid_thw"),
            attention_mask=raw.get("attention_mask"),
        )
        prepared.append(
            {
                "record": record,
                "prompt_text": prompt_text,
                "raw": raw,
                "original_length": original_length,
                "original_position_ids": position_ids,
                "original_rope_delta": rope_delta,
            }
        )
    common_length = max(item["original_length"] for item in prepared)
    for item in prepared:
        padded = right_pad_prompt_inputs(
            item["raw"], common_length, int(processor.tokenizer.pad_token_id)
        )
        if not torch.equal(
            padded["input_ids"][:, : item["original_length"]], item["raw"]["input_ids"]
        ):
            raise RuntimeError("Right padding changed non-padding token IDs")
        if not bool((padded["attention_mask"][:, item["original_length"] :] == 0).all().item()):
            raise RuntimeError("Right padding attention mask is not zero")
        padded_position_ids, _ = model.get_rope_index(
            input_ids=padded["input_ids"],
            image_grid_thw=padded.get("image_grid_thw"),
            attention_mask=padded.get("attention_mask"),
        )
        item["padded"] = padded
        item["padded_position_ids"] = padded_position_ids
        item["common_length"] = common_length
    return prepared


def crop_for_answer_scoring(
    model,
    prompt_logits: torch.Tensor,
    prompt_cache,
    padded_attention_mask: torch.Tensor,
    original_length: int,
    original_rope_delta: torch.Tensor,
) -> tuple[torch.Tensor, Any, torch.Tensor]:
    if original_length > prompt_logits.shape[1]:
        raise ValueError("Original prompt boundary exceeds prompt logits")
    model.rope_deltas = original_rope_delta.detach().clone()
    return (
        prompt_logits[:, :original_length],
        truncate_dynamic_cache(prompt_cache, original_length),
        padded_attention_mask[:, :original_length],
    )


def score_state(
    model,
    tokenizer,
    prompt_logits,
    prompt_cache,
    padded_attention_mask,
    original_length,
    original_rope_delta,
    answers,
) -> dict[str, Any]:
    logits, cache, mask = crop_for_answer_scoring(
        model,
        prompt_logits,
        prompt_cache,
        padded_attention_mask,
        original_length,
        original_rope_delta,
    )
    return score_accepted_answer_set(model, tokenizer, logits, cache, mask, answers)


def verify_pair_layout(model, prepared: list[dict[str, Any]]) -> dict[str, Any]:
    if prepared[0]["padded"]["input_ids"].shape != prepared[1]["padded"]["input_ids"].shape:
        raise RuntimeError("Common-padded input shapes differ")
    rows = []
    reference_indices = None
    reference_positions = None
    for item in prepared:
        input_ids = item["padded"]["input_ids"]
        indices = torch.where(input_ids[0] == model.config.image_token_id)[0]
        positions = item["padded_position_ids"][:, 0, indices]
        if reference_indices is None:
            reference_indices = indices
            reference_positions = positions
        elif not torch.equal(indices, reference_indices) or not torch.equal(
            positions, reference_positions
        ):
            raise RuntimeError("Same-image visual indices or position encodings differ")
        rows.append(
            {
                "id": item["record"]["id"],
                "original_length": item["original_length"],
                "common_length": item["common_length"],
                "visual_first": int(indices[0]),
                "visual_last": int(indices[-1]),
                "visual_count": int(indices.numel()),
                "right_padding_count": item["common_length"] - item["original_length"],
                "literal_nonpadding_prompt_unchanged": True,
                "padding_masked_from_attention": True,
            }
        )
    return {
        "questions": rows,
        "identical_tensor_shapes": True,
        "identical_visual_indices": True,
        "identical_visual_position_encodings": True,
    }


def run_preflight(
    config: dict[str, Any], model, processor, groups: list[list[dict[str, Any]]]
) -> int:
    output_dir = Path(config["preflight_output_dir"])
    selected = [group for group in groups if bool(group[0]["preflight_selected"])]
    if len(selected) != int(config["preflight_image_count"]):
        raise RuntimeError("Preflight image count does not match the frozen manifest")
    layers = [int(value) for value in config["layer_grid"]]
    action_names = list(config["actions"])
    controls: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    score_differences_sequence: list[float] = []
    score_differences_mean: list[float] = []

    with torch.inference_mode():
        for group_index, group in enumerate(selected):
            prepared = prepare_common_group(processor, model, group)
            layout = verify_pair_layout(model, prepared)
            captured = []
            for item in prepared:
                padded = item["padded"]
                visual_mask = padded["input_ids"] == model.config.image_token_id
                baseline, contexts = capture_prompt_with_cache(model, padded, layers)
                indices = torch.where(visual_mask[0])[0]
                captured.append(
                    {
                        "item": item,
                        "visual_mask": visual_mask,
                        "baseline": baseline,
                        "contexts": contexts,
                        "visual_indices": indices,
                        "states": {},
                    }
                )

            layer_identity = []
            for layer in layers:
                first = captured[0]
                second = captured[1]
                indices = first["visual_indices"]
                first_pre = first["contexts"][layer].pre_layer_state[0, indices]
                second_pre = second["contexts"][layer].pre_layer_state[0, indices]
                first_full = first["contexts"][layer].full_layer_output[0, indices]
                second_full = second["contexts"][layer].full_layer_output[0, indices]
                first_write = first_full.float() - first_pre.float()
                second_write = second_full.float() - second_pre.float()
                layer_identity.append(
                    {
                        "layer": layer,
                        "pre_max_abs": max_abs_difference(first_pre, second_pre),
                        "pre_rms_relative": rms_relative(first_pre, second_pre),
                        "post_max_abs": max_abs_difference(first_full, second_full),
                        "post_rms_relative": rms_relative(first_full, second_full),
                        "write_max_abs": max_abs_difference(first_write, second_write),
                        "write_rms_relative": rms_relative(first_write, second_write),
                    }
                )

            for captured_item in captured:
                item = captured_item["item"]
                padded = item["padded"]
                answers = accepted_answers(item["record"])
                if [answer.text for answer in answers] != [item["record"]["answer"]]:
                    raise RuntimeError("Accepted-answer normalization drift")
                baseline_score = score_state(
                    model,
                    processor.tokenizer,
                    captured_item["baseline"].logits,
                    captured_item["baseline"].past_key_values,
                    padded["attention_mask"],
                    item["original_length"],
                    item["original_rope_delta"],
                    answers,
                )
                for layer in layers:
                    context = captured_item["contexts"][layer]
                    read_cache = ReadInterventionCache()
                    for action in action_names:
                        read_mode, write_mode = FOUR_STATES[action]
                        result = run_cached_prompt_state(
                            model,
                            context,
                            captured_item["baseline"].past_key_values,
                            captured_item["visual_mask"],
                            action,
                            read_mode,
                            write_mode,
                            read_cache,
                        )
                        score = score_state(
                            model,
                            processor.tokenizer,
                            result.prompt_logits,
                            result.past_key_values,
                            padded["attention_mask"],
                            item["original_length"],
                            item["original_rope_delta"],
                            answers,
                        )
                        repeat = run_cached_prompt_state(
                            model,
                            context,
                            captured_item["baseline"].past_key_values,
                            captured_item["visual_mask"],
                            action,
                            read_mode,
                            write_mode,
                            read_cache,
                        )
                        repeat_score = score_state(
                            model,
                            processor.tokenizer,
                            repeat.prompt_logits,
                            repeat.past_key_values,
                            padded["attention_mask"],
                            item["original_length"],
                            item["original_rope_delta"],
                            answers,
                        )
                        sequence_abs = abs(
                            repeat_score["sequence_logprob"] - score["sequence_logprob"]
                        )
                        mean_abs = abs(repeat_score["mean_logprob"] - score["mean_logprob"])
                        score_differences_sequence.append(sequence_abs)
                        score_differences_mean.append(mean_abs)
                        row = {
                            "sample_id": item["record"]["id"],
                            "image_id": item["record"]["image_id"],
                            "layer": layer,
                            "control": f"repeat_{action.lower()}",
                            "prompt_logit_max_abs": max_abs_difference(
                                repeat.prompt_logits, result.prompt_logits
                            ),
                            "sequence_score_abs": sequence_abs,
                            "mean_score_abs": mean_abs,
                            "score_finite": bool(
                                np.isfinite(score["sequence_logprob"])
                                and np.isfinite(score["mean_logprob"])
                            ),
                            "answer_token_ids_match_manifest": score[
                                "accepted_answer_scores"
                            ][0]["token_ids"]
                            == item["record"]["answer_token_ids"],
                            "prompt_positions_contributing_to_score": 0,
                            "padding_positions_contributing_to_score": 0,
                            "read_hook_identity_max_abs": result.read_hook_identity_max_abs,
                            "write_hook_identity_max_abs": result.write_hook_identity_max_abs,
                            "prestate_injection_max_abs": result.injected_prestate_max_abs,
                        }
                        controls.append(row)
                        if result.read_decomposition is not None:
                            row["visual_future_attention_mass_max"] = (
                                result.read_decomposition.visual_future_attention_mass_max
                            )
                        if action == "FULL":
                            full_sequence_abs = abs(
                                score["sequence_logprob"]
                                - baseline_score["sequence_logprob"]
                            )
                            full_mean_abs = abs(
                                score["mean_logprob"] - baseline_score["mean_logprob"]
                            )
                            score_differences_sequence.append(full_sequence_abs)
                            score_differences_mean.append(full_mean_abs)
                            controls.append(
                                {
                                    "sample_id": item["record"]["id"],
                                    "image_id": item["record"]["image_id"],
                                    "layer": layer,
                                    "control": "instrumented_full",
                                    "prompt_logit_max_abs": max_abs_difference(
                                        result.prompt_logits,
                                        captured_item["baseline"].logits,
                                    ),
                                    "sequence_score_abs": full_sequence_abs,
                                    "mean_score_abs": full_mean_abs,
                                }
                            )
                        del result, repeat
                    for control_name, read_mode, write_mode in (
                        ("read_reinsert_identity", "reconstruct", "full"),
                        ("write_reinsert_identity", "full", "reconstruct"),
                    ):
                        identity = run_cached_prompt_state(
                            model,
                            context,
                            captured_item["baseline"].past_key_values,
                            captured_item["visual_mask"],
                            control_name,
                            read_mode,
                            write_mode,
                            read_cache,
                        )
                        identity_score = score_state(
                            model,
                            processor.tokenizer,
                            identity.prompt_logits,
                            identity.past_key_values,
                            padded["attention_mask"],
                            item["original_length"],
                            item["original_rope_delta"],
                            answers,
                        )
                        sequence_abs = abs(
                            identity_score["sequence_logprob"]
                            - baseline_score["sequence_logprob"]
                        )
                        mean_abs = abs(
                            identity_score["mean_logprob"] - baseline_score["mean_logprob"]
                        )
                        score_differences_sequence.append(sequence_abs)
                        score_differences_mean.append(mean_abs)
                        controls.append(
                            {
                                "sample_id": item["record"]["id"],
                                "image_id": item["record"]["image_id"],
                                "layer": layer,
                                "control": control_name,
                                "prompt_logit_max_abs": max_abs_difference(
                                    identity.prompt_logits, captured_item["baseline"].logits
                                ),
                                "sequence_score_abs": sequence_abs,
                                "mean_score_abs": mean_abs,
                            }
                        )
            pair_rows.append(
                {
                    "image_id": group[0]["image_id"],
                    **layout,
                    "layers": layer_identity,
                }
            )
            del captured
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {"preflight_completed": group_index + 1, "total": len(selected), "image_id": group[0]["image_id"]}
                ),
                flush=True,
            )

    visual_tolerance = float(config["visual_identity_tolerance"])
    logit_tolerance = float(config["logit_tolerance"])
    score_tolerance = float(config["score_tolerance"])
    epsilon_sequence = max(
        float(config["noise_floor_sequence"]),
        float(np.quantile(np.asarray(score_differences_sequence), 0.99)),
    )
    epsilon_mean = max(
        float(config["noise_floor_mean"]),
        float(np.quantile(np.asarray(score_differences_mean), 0.99)),
    )
    checks = {
        "exact_visual_state_and_write_identity": all(
            row[key] <= visual_tolerance
            for pair in pair_rows
            for row in pair["layers"]
            for key in ("pre_max_abs", "post_max_abs", "write_max_abs")
        ),
        "identical_layout": all(
            pair["identical_tensor_shapes"]
            and pair["identical_visual_indices"]
            and pair["identical_visual_position_encodings"]
            for pair in pair_rows
        ),
        "zero_visual_query_future_attention": all(
            row.get("visual_future_attention_mass_max", 0.0) == 0.0
            for row in controls
            if row["control"].startswith("repeat_")
        ),
        "instrumented_full_parity": all(
            row["prompt_logit_max_abs"] <= logit_tolerance
            and row["sequence_score_abs"] <= score_tolerance
            and row["mean_score_abs"] <= score_tolerance
            for row in controls
            if row["control"] == "instrumented_full"
        ),
        "deterministic_four_actions": all(
            row["prompt_logit_max_abs"] == 0.0
            and row["sequence_score_abs"] == 0.0
            and row["mean_score_abs"] == 0.0
            and row["score_finite"]
            for row in controls
            if row["control"].startswith("repeat_")
        ),
        "reconstruction_identity": all(
            row["prompt_logit_max_abs"] <= logit_tolerance
            and row["sequence_score_abs"] <= score_tolerance
            and row["mean_score_abs"] <= score_tolerance
            for row in controls
            if row["control"].endswith("reinsert_identity")
        ),
        "answer_alignment_and_padding_exclusion": all(
            row["answer_token_ids_match_manifest"]
            and row["prompt_positions_contributing_to_score"] == 0
            and row["padding_positions_contributing_to_score"] == 0
            for row in controls
            if row["control"].startswith("repeat_")
        ),
    }
    summary = {
        "schema_version": "v4_common_padding_preflight_v1",
        "outcome_blind": True,
        "scientific_action_values_saved_or_aggregated": False,
        "image_count": len(pair_rows),
        "question_count": 2 * len(pair_rows),
        "layers": layers,
        "actions_executed": action_names,
        "checks": checks,
        "gate_pass": all(checks.values()),
        "noise": {
            "sequence_abs_p99": float(np.quantile(score_differences_sequence, 0.99)),
            "mean_abs_p99": float(np.quantile(score_differences_mean, 0.99)),
            "epsilon_sequence": epsilon_sequence,
            "epsilon_mean": epsilon_mean,
            "selection_rule": "max(frozen floor, empirical identity-control absolute-difference p99)",
        },
        "pair_identity": pair_rows,
        "control_count": len(controls),
    }
    write_json(output_dir / "v4_common_padding_preflight_controls_v1.json", controls)
    write_json(output_dir / "v4_common_padding_preflight_v1.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["gate_pass"] else 2


def run_full_shard(
    config: dict[str, Any],
    model,
    processor,
    groups: list[list[dict[str, Any]]],
    shard_index: int,
    shard_count: int,
    resume: bool,
) -> int:
    preflight = json.loads(
        (Path(config["preflight_output_dir"]) / "v4_common_padding_preflight_v1.json").read_text(
            encoding="utf-8"
        )
    )
    if not preflight["gate_pass"]:
        raise RuntimeError("v4 common-padding preflight gate did not pass")
    layers = [int(value) for value in config["layer_grid"]]
    actions = list(config["actions"])
    selected = [group for group in groups if int(group[0]["image_index"]) % shard_count == shard_index]
    shard_dir = Path(config["shard_output_dir"]) / f"shard_{shard_index:02d}"
    result_path = shard_dir / "results.jsonl"
    completed: set[str] = set()
    if resume and result_path.exists():
        completed = {row["id"] for row in read_jsonl(result_path)}
    elif result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path} without --resume")
    write_json(
        shard_dir / "runtime.json",
        {
            **runtime_metadata(model, processor, yaml.safe_load(Path("configs/model.yaml").read_text()), layers),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "image_count": len(selected),
            "question_count": 2 * len(selected),
        },
    )
    with torch.inference_mode():
        for group_number, group in enumerate(selected):
            prepared = prepare_common_group(processor, model, group)
            verify_pair_layout(model, prepared)
            for item in prepared:
                record = item["record"]
                if record["id"] in completed:
                    continue
                padded = item["padded"]
                visual_mask = padded["input_ids"] == model.config.image_token_id
                baseline, contexts = capture_prompt_with_cache(model, padded, layers)
                answers = accepted_answers(record)
                baseline_score = score_state(
                    model,
                    processor.tokenizer,
                    baseline.logits,
                    baseline.past_key_values,
                    padded["attention_mask"],
                    item["original_length"],
                    item["original_rope_delta"],
                    answers,
                )
                sample_result = {
                    "schema_version": "v4_gqa_four_action_q_v1",
                    "id": record["id"],
                    "image_id": record["image_id"],
                    "image_index": record["image_index"],
                    "question_index": record["question_index"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "pair_stratum": record["pair_stratum"],
                    "different_evidence": record["different_evidence"],
                    "official_paraphrase": record["official_paraphrase"],
                    "pair_match_distance": record["pair_match_distance"],
                    "question_types": record["question_types"],
                    "semantic_program_depth": record["semantic_program_depth"],
                    "semantic_object_ids": record["semantic_object_ids"],
                    "answer_token_length": record["answer_token_length"],
                    "original_prompt_token_length": item["original_length"],
                    "common_prompt_token_length": item["common_length"],
                    "visual_token_count": record["expected_visual_token_count"],
                    "accepted_answers": [
                        {"answer": answer.text, "weight": answer.weight} for answer in answers
                    ],
                    "baseline_full": baseline_score,
                    "layers": [],
                }
                for layer in layers:
                    context = contexts[layer]
                    read_cache = ReadInterventionCache()
                    state_records = {}
                    sequence_scores = {}
                    mean_scores = {}
                    for action in actions:
                        read_mode, write_mode = FOUR_STATES[action]
                        result = run_cached_prompt_state(
                            model,
                            context,
                            baseline.past_key_values,
                            visual_mask,
                            action,
                            read_mode,
                            write_mode,
                            read_cache,
                        )
                        score = score_state(
                            model,
                            processor.tokenizer,
                            result.prompt_logits,
                            result.past_key_values,
                            padded["attention_mask"],
                            item["original_length"],
                            item["original_rope_delta"],
                            answers,
                        )
                        sequence_scores[action] = score["sequence_logprob"]
                        mean_scores[action] = score["mean_logprob"]
                        state_records[action] = {
                            **score,
                            "read_hook_identity_max_abs": result.read_hook_identity_max_abs,
                            "write_hook_identity_max_abs": result.write_hook_identity_max_abs,
                            "prestate_injection_max_abs": result.injected_prestate_max_abs,
                        }
                        if action == "FULL":
                            state_records[action]["baseline_prompt_logit_max_abs"] = max_abs_difference(
                                result.prompt_logits, baseline.logits
                            )
                            state_records[action]["baseline_sequence_score_abs"] = abs(
                                score["sequence_logprob"] - baseline_score["sequence_logprob"]
                            )
                            state_records[action]["baseline_mean_score_abs"] = abs(
                                score["mean_logprob"] - baseline_score["mean_logprob"]
                            )
                    sample_result["layers"].append(
                        {
                            "layer": layer,
                            "states": state_records,
                            "sequence_effects": factorial_effects(sequence_scores),
                            "mean_effects": factorial_effects(mean_scores),
                        }
                    )
                append_jsonl(result_path, sample_result)
                del baseline, contexts
                torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "shard": shard_index,
                        "completed_images": group_number + 1,
                        "total_images": len(selected),
                        "image_id": group[0]["image_id"],
                    }
                ),
                flush=True,
            )
    write_json(
        shard_dir / "completion.json",
        {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "expected_images": len(selected),
            "expected_records": 2 * len(selected),
            "completed_records": len(read_jsonl(result_path)),
            "result_sha256": sha256_file(result_path),
            "complete": len(read_jsonl(result_path)) == 2 * len(selected),
        },
    )
    return 0


def execute(args: argparse.Namespace) -> int:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    actual_sha = sha256_file(manifest_path)
    frozen_sha = expected_manifest_sha(Path(config["manifest_checksum"]))
    if actual_sha != frozen_sha:
        raise RuntimeError("Frozen v4 manifest checksum mismatch")
    rows = read_jsonl(manifest_path)
    if len(rows) != int(config["sample_count"]):
        raise RuntimeError("Frozen v4 manifest sample count mismatch")
    groups = group_manifest(rows)
    if len(groups) != int(config["image_count"]):
        raise RuntimeError("Frozen v4 manifest image count mismatch")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    set_determinism(int(config["seed"]) + args.shard_index)
    model_config = yaml.safe_load(Path(config["model_config"]).read_text(encoding="utf-8"))
    model, processor = load_model(model_config)
    if args.mode == "preflight":
        output_dir = Path(config["preflight_output_dir"])
        write_json(
            output_dir / "runtime.json", runtime_metadata(model, processor, model_config, config["layer_grid"])
        )
        return run_preflight(config, model, processor, groups)
    return run_full_shard(
        config, model, processor, groups, args.shard_index, args.shard_count, args.resume
    )


def main() -> int:
    args = parse_args()
    try:
        return execute(args)
    except Exception as exc:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if args.mode == "preflight":
            output_dir = Path(config["preflight_output_dir"])
        else:
            output_dir = Path(config["shard_output_dir"]) / f"shard_{args.shard_index:02d}"
        write_json(
            output_dir / "failure.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
