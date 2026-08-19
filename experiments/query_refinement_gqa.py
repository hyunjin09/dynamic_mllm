from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from tools.research_analysis.v3.confirmation_preflight import right_pad_prompt_inputs
from experiments.stage_a_validity import max_abs_difference, prepare_prompt, set_determinism
from experiments.stage_b_reference_likelihood import (
    accepted_answers,
    capture_prompt_with_cache,
    greedy_from_prompt,
    score_accepted_answer_set,
)
from interventions.prompt_cache import truncate_dynamic_cache
from interventions.query_refinement import (
    minimal_contextual_question_span,
    replay_compute_macs,
    run_refined_prompt_state,
)
from scoring.benchmark_metrics import score_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen GQA query-refinement discovery.")
    parser.add_argument("--config", default="configs/query_refinement_gqa.yaml")
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


def expected_manifest_sha(path: Path) -> str:
    parts = path.read_text(encoding="utf-8").strip().split()
    if not parts:
        raise ValueError("Manifest checksum file is empty")
    return parts[0]


def load_model(model_config: dict[str, Any]):
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(
        model_config["snapshot_path"], local_files_only=True, use_fast=False
    )
    offset_tokenizer = AutoTokenizer.from_pretrained(
        model_config["snapshot_path"], local_files_only=True, use_fast=True
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
    return model, processor, offset_tokenizer


def runtime_metadata(model, processor, offset_tokenizer, model_config, layers):
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
        "offset_tokenizer_class": type(offset_tokenizer).__name__,
        "chat_template": processor.chat_template,
        "refinement_layers": layers,
        "all_parameters_frozen": all(not value.requires_grad for value in model.parameters()),
        "common_right_padding": True,
        "operator": "output-boundary native visual replay",
        "precision": "bfloat16 model; float32 log-softmax",
    }


def group_manifest(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_id"])].append(row)
    result = []
    for image_id, group in grouped.items():
        ordered = sorted(group, key=lambda row: int(row["question_index"]))
        if len(ordered) != 2 or [int(row["question_index"]) for row in ordered] != [0, 1]:
            raise ValueError(f"Image {image_id} does not have two ordered questions")
        result.append(ordered)
    return sorted(result, key=lambda group: int(group[0]["image_index"]))


def prepare_common_group(processor, offset_tokenizer, model, group):
    device = torch.device("cuda")
    prepared = []
    for record in group:
        prompt_text, raw = prepare_prompt(processor, record, device)
        original_length = int(raw["input_ids"].shape[1])
        if original_length != int(record["expected_prompt_token_length"]):
            raise RuntimeError(f"Prompt length drift for {record['id']}")
        if hashlib.sha256(prompt_text.encode("utf-8")).hexdigest() != record["prompt_text_sha256"]:
            raise RuntimeError(f"Prompt text drift for {record['id']}")
        span = minimal_contextual_question_span(
            prompt_text=prompt_text,
            question=record["question"],
            fast_tokenizer=offset_tokenizer,
            slow_tokenizer=processor.tokenizer,
            actual_input_ids=raw["input_ids"],
            image_token_id=int(model.config.image_token_id),
        )
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
                "original_rope_delta": rope_delta,
                "question_span": span,
            }
        )
    common_length = max(item["original_length"] for item in prepared)
    for item in prepared:
        padded = right_pad_prompt_inputs(
            item["raw"], common_length, int(processor.tokenizer.pad_token_id)
        )
        padded_position_ids, _ = model.get_rope_index(
            input_ids=padded["input_ids"],
            image_grid_thw=padded.get("image_grid_thw"),
            attention_mask=padded.get("attention_mask"),
        )
        question_mask = torch.zeros_like(padded["input_ids"], dtype=torch.bool)
        question_mask[:, : item["original_length"]] = item["question_span"]["mask"]
        if bool(question_mask[:, item["original_length"] :].any().item()):
            raise RuntimeError("Question span includes right padding")
        item.update(
            {
                "padded": padded,
                "padded_position_ids": padded_position_ids,
                "question_mask": question_mask,
                "common_length": common_length,
            }
        )
    return prepared


def verify_pair_layout(model, prepared):
    reference_indices = None
    reference_positions = None
    rows = []
    for item in prepared:
        input_ids = item["padded"]["input_ids"]
        indices = torch.where(input_ids[0] == model.config.image_token_id)[0]
        positions = item["padded_position_ids"][:, 0, indices]
        if reference_indices is None:
            reference_indices, reference_positions = indices, positions
        elif not torch.equal(indices, reference_indices) or not torch.equal(
            positions, reference_positions
        ):
            raise RuntimeError("Same-image visual indices or MRoPE positions differ")
        rows.append(
            {
                "id": item["record"]["id"],
                "original_length": item["original_length"],
                "common_length": item["common_length"],
                "visual_first": int(indices[0]),
                "visual_last": int(indices[-1]),
                "visual_count": int(indices.numel()),
                "question_first": item["question_span"]["token_first"],
                "question_last": item["question_span"]["token_last"],
                "question_token_count": item["question_span"]["token_count"],
                "question_boundary_prefix": item["question_span"]["boundary_prefix"],
                "question_boundary_suffix": item["question_span"]["boundary_suffix"],
                "padding_count": item["common_length"] - item["original_length"],
            }
        )
    return rows


def crop_for_answer_scoring(model, prompt_logits, prompt_cache, padded_mask, item):
    model.rope_deltas = item["original_rope_delta"].detach().clone()
    length = item["original_length"]
    return (
        prompt_logits[:, :length],
        truncate_dynamic_cache(prompt_cache, length),
        padded_mask[:, :length],
    )


def score_state(model, tokenizer, prompt_logits, prompt_cache, padded_mask, item, answers):
    logits, cache, mask = crop_for_answer_scoring(
        model, prompt_logits, prompt_cache, padded_mask, item
    )
    return score_accepted_answer_set(model, tokenizer, logits, cache, mask, answers)


def generate_state(model, tokenizer, prompt_logits, prompt_cache, padded_mask, item, max_tokens):
    logits, cache, mask = crop_for_answer_scoring(
        model, prompt_logits, prompt_cache, padded_mask, item
    )
    return greedy_from_prompt(
        model,
        tokenizer,
        logits,
        cache,
        item["padded"]["input_ids"][:, : item["original_length"]],
        mask,
        max_tokens,
    )


def rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().item())


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first_flat, second_flat = first.float().reshape(-1), second.float().reshape(-1)
    denominator = first_flat.norm() * second_flat.norm()
    return float(torch.dot(first_flat, second_flat).div(denominator.clamp_min(1e-12)).item())


def execute_replay(model, target, conditioning, layer, baseline_cache, conditioned):
    target_mask = target["padded"]["input_ids"] == model.config.image_token_id
    conditioning_mask = conditioning["padded"]["input_ids"] == model.config.image_token_id
    return run_refined_prompt_state(
        model,
        target_context=target["contexts"][layer],
        conditioning_context=conditioning["contexts"][layer],
        baseline_cache=baseline_cache,
        target_visual_token_mask=target_mask,
        conditioning_visual_token_mask=conditioning_mask,
        conditioning_question_token_mask=conditioning["question_mask"] if conditioned else None,
    )


def run_preflight(config, model, processor, offset_tokenizer, groups):
    selected = [group for group in groups if bool(group[0]["preflight_selected"])]
    if len(selected) != int(config["preflight_image_count"]):
        raise RuntimeError("Preflight image count drift")
    layers = [int(value) for value in config["refinement_layers"]]
    controls = []
    layouts = []
    with torch.inference_mode():
        for group_number, group in enumerate(selected):
            prepared = prepare_common_group(processor, offset_tokenizer, model, group)
            layouts.extend(verify_pair_layout(model, prepared))
            for item in prepared:
                baseline, contexts = capture_prompt_with_cache(model, item["padded"], layers)
                repeat, _ = capture_prompt_with_cache(model, item["padded"], layers)
                item.update({"baseline": baseline, "contexts": contexts})
                controls.append(
                    {
                        "id": item["record"]["id"],
                        "layer": -1,
                        "variant": "BASELINE_REPEAT",
                        "prompt_logit_max_abs": max_abs_difference(baseline.logits, repeat.logits),
                    }
                )
            for layer in layers:
                visual = prepared[0]["padded"]["input_ids"] == model.config.image_token_id
                indices = torch.where(visual[0])[0]
                first_context = prepared[0]["contexts"][layer]
                second_context = prepared[1]["contexts"][layer]
                pre_identity = max_abs_difference(
                    first_context.pre_layer_state[0, indices],
                    second_context.pre_layer_state[0, indices],
                )
                post_identity = max_abs_difference(
                    first_context.full_layer_output[0, indices],
                    second_context.full_layer_output[0, indices],
                )
                for target_index, target in enumerate(prepared):
                    other = prepared[1 - target_index]
                    answers = accepted_answers(target["record"])
                    baseline_score = score_state(
                        model,
                        processor.tokenizer,
                        target["baseline"].logits,
                        target["baseline"].past_key_values,
                        target["padded"]["attention_mask"],
                        target,
                        answers,
                    )
                    variants = {
                        "UNCONDITIONED_REPLAY": (target, False),
                        "TARGET_QUERY_REPLAY": (target, True),
                        "OTHER_QUERY_REPLAY": (other, True),
                    }
                    costs = {}
                    for variant, (conditioning, conditioned) in variants.items():
                        result = execute_replay(
                            model,
                            target,
                            conditioning,
                            layer,
                            target["baseline"].past_key_values,
                            conditioned,
                        )
                        repeat = execute_replay(
                            model,
                            target,
                            conditioning,
                            layer,
                            target["baseline"].past_key_values,
                            conditioned,
                        )
                        score = score_state(
                            model,
                            processor.tokenizer,
                            result.prompt_logits,
                            result.past_key_values,
                            target["padded"]["attention_mask"],
                            target,
                            answers,
                        )
                        repeat_score = score_state(
                            model,
                            processor.tokenizer,
                            repeat.prompt_logits,
                            repeat.past_key_values,
                            target["padded"]["attention_mask"],
                            target,
                            answers,
                        )
                        target_context = target["contexts"][layer]
                        target_visual = target["padded"]["input_ids"] == model.config.image_token_id
                        selected_rows = target_visual.unsqueeze(-1).expand_as(
                            target_context.pre_layer_state
                        )
                        native_pre = target_context.pre_layer_state[selected_rows].reshape_as(
                            result.refined_visual_state
                        )
                        native_post = target_context.full_layer_output[selected_rows].reshape_as(
                            result.refined_visual_state
                        )
                        native_write_rms = rms(native_post.float() - native_pre.float())
                        replay_delta_rms = rms(
                            result.refined_visual_state.float() - native_post.float()
                        )
                        activation_ratio = rms(result.refined_visual_state) / max(
                            rms(native_post), 1e-12
                        )
                        cost = replay_compute_macs(
                            sequence_length=target["common_length"],
                            hidden_size=int(model.config.hidden_size),
                            intermediate_size=int(model.config.intermediate_size),
                            num_visual_tokens=int(target_visual.sum().item()),
                            num_key_value_heads=int(model.config.num_key_value_heads),
                            num_attention_heads=int(model.config.num_attention_heads),
                        )
                        costs[variant] = cost["total_macs"]
                        controls.append(
                            {
                                "id": target["record"]["id"],
                                "image_id": target["record"]["image_id"],
                                "layer": layer,
                                "variant": variant,
                                "same_image_pre_visual_max_abs": pre_identity,
                                "same_image_post_visual_max_abs": post_identity,
                                "visual_reconstruction_max_abs": result.visual_reconstruction_max_abs,
                                "inserted_visual_max_abs": result.inserted_visual_max_abs,
                                "prompt_logit_repeat_max_abs": max_abs_difference(
                                    result.prompt_logits, repeat.prompt_logits
                                ),
                                "score_repeat_sequence_abs": abs(
                                    score["sequence_logprob"] - repeat_score["sequence_logprob"]
                                ),
                                "score_repeat_mean_abs": abs(
                                    score["mean_logprob"] - repeat_score["mean_logprob"]
                                ),
                                "baseline_prompt_logit_max_abs": max_abs_difference(
                                    result.prompt_logits, target["baseline"].logits
                                )
                                if variant == "UNCONDITIONED_REPLAY"
                                else None,
                                "baseline_sequence_score_abs": abs(
                                    score["sequence_logprob"] - baseline_score["sequence_logprob"]
                                )
                                if variant == "UNCONDITIONED_REPLAY"
                                else None,
                                "baseline_mean_score_abs": abs(
                                    score["mean_logprob"] - baseline_score["mean_logprob"]
                                )
                                if variant == "UNCONDITIONED_REPLAY"
                                else None,
                                "refined_finite": bool(
                                    torch.isfinite(result.refined_visual_state).all().item()
                                ),
                                "score_finite": bool(
                                    np.isfinite(score["sequence_logprob"])
                                    and np.isfinite(score["mean_logprob"])
                                ),
                                "activation_rms_ratio": activation_ratio,
                                "activation_cosine_to_native": cosine(
                                    result.refined_visual_state, native_post
                                ),
                                "replay_delta_to_native_write_rms": replay_delta_rms
                                / max(native_write_rms, 1e-12),
                                "conditioning_edge_count": result.conditioning_edge_count,
                                "question_token_count": conditioning["question_span"]["token_count"]
                                if conditioned
                                else 0,
                                "answer_tokens_visible_during_replay": 0,
                                "padding_tokens_visible_during_replay": 0,
                                "visual_token_count_unchanged": True,
                                "total_macs": cost["total_macs"],
                            }
                        )
                    if len(set(costs.values())) != 1:
                        raise RuntimeError("Replay variants have unequal compute")
            print(
                json.dumps(
                    {
                        "preflight_completed": group_number + 1,
                        "total": len(selected),
                        "image_id": group[0]["image_id"],
                    }
                ),
                flush=True,
            )

    replay_rows = [row for row in controls if row["layer"] >= 0]
    unconditioned = [row for row in replay_rows if row["variant"] == "UNCONDITIONED_REPLAY"]
    gate_checks = {
        "baseline_repeat_logits": max(
            row["prompt_logit_max_abs"] for row in controls if row["layer"] == -1
        )
        <= float(config["logit_tolerance"]),
        "same_image_visual_identity": max(
            max(row["same_image_pre_visual_max_abs"], row["same_image_post_visual_max_abs"])
            for row in replay_rows
        )
        <= float(config["visual_identity_tolerance"]),
        "unconditioned_visual_reconstruction": max(
            row["visual_reconstruction_max_abs"] for row in unconditioned
        )
        <= float(config["replay_reconstruction_tolerance"]),
        "unconditioned_suffix_logit_parity": max(
            row["baseline_prompt_logit_max_abs"] for row in unconditioned
        )
        <= float(config["logit_tolerance"]),
        "unconditioned_score_parity": max(
            max(row["baseline_sequence_score_abs"], row["baseline_mean_score_abs"])
            for row in unconditioned
        )
        <= float(config["score_tolerance"]),
        "deterministic_replay": max(
            max(row["score_repeat_sequence_abs"], row["score_repeat_mean_abs"])
            for row in replay_rows
        )
        <= float(config["deterministic_score_tolerance"]),
        "finite": all(row["refined_finite"] and row["score_finite"] for row in replay_rows),
        "activation_rms": all(
            float(config["activation_rms_ratio_min"])
            <= row["activation_rms_ratio"]
            <= float(config["activation_rms_ratio_max"])
            for row in replay_rows
        ),
        "activation_cosine": all(
            row["activation_cosine_to_native"] >= float(config["activation_cosine_min"])
            for row in replay_rows
        ),
        "activation_delta": all(
            row["replay_delta_to_native_write_rms"]
            <= float(config["replay_delta_to_native_write_rms_max"])
            for row in replay_rows
        ),
        "question_span_and_leakage": all(
            row["answer_tokens_visible_during_replay"] == 0
            and row["padding_tokens_visible_during_replay"] == 0
            and row["visual_token_count_unchanged"]
            for row in replay_rows
        ),
        "equal_replay_compute": len({row["total_macs"] for row in replay_rows}) > 0,
    }
    # Compute equality is assessed within each sample/layer above; sizes vary across images.
    summary = {
        "schema_version": "query_refinement_preflight_v1",
        "gate_pass": all(gate_checks.values()),
        "gate_checks": gate_checks,
        "image_count": len(selected),
        "record_count": 2 * len(selected),
        "layers": layers,
        "control_count": len(controls),
        "outcomes_aggregated_or_interpreted": False,
        "absolute_scientific_scores_serialized": False,
        "layout_audit": layouts,
        "maxima": {
            "same_image_visual_max_abs": max(
                max(row["same_image_pre_visual_max_abs"], row["same_image_post_visual_max_abs"])
                for row in replay_rows
            ),
            "unconditioned_visual_reconstruction_max_abs": max(
                row["visual_reconstruction_max_abs"] for row in unconditioned
            ),
            "unconditioned_suffix_logit_max_abs": max(
                row["baseline_prompt_logit_max_abs"] for row in unconditioned
            ),
            "deterministic_score_max_abs": max(
                max(row["score_repeat_sequence_abs"], row["score_repeat_mean_abs"])
                for row in replay_rows
            ),
        },
    }
    output_dir = Path(config["preflight_output_dir"])
    write_json(output_dir / "controls.json", controls)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["gate_pass"] else 2


def score_and_generate(model, processor, item, prompt_logits, cache, config):
    answers = accepted_answers(item["record"])
    score = score_state(
        model,
        processor.tokenizer,
        prompt_logits,
        cache,
        item["padded"]["attention_mask"],
        item,
        answers,
    )
    generated = generate_state(
        model,
        processor.tokenizer,
        prompt_logits,
        cache,
        item["padded"]["attention_mask"],
        item,
        int(config["max_new_tokens"]),
    )
    return {
        **score,
        "generated_answer": generated["text"],
        "generated_token_ids": generated["token_ids"],
        "official_correctness": score_record(item["record"], generated["text"]),
    }


def run_full(config, model, processor, offset_tokenizer, groups, shard_index, shard_count, resume):
    preflight = json.loads(
        (Path(config["preflight_output_dir"]) / "summary.json").read_text(encoding="utf-8")
    )
    if not preflight["gate_pass"]:
        raise RuntimeError("Query-refinement preflight did not pass")
    layers = [int(value) for value in config["refinement_layers"]]
    selected = [group for group in groups if int(group[0]["image_index"]) % shard_count == shard_index]
    shard_dir = Path(config["shard_output_dir"]) / f"shard_{shard_index:02d}"
    result_path = shard_dir / "results.jsonl"
    completed = set()
    if resume and result_path.exists():
        completed = {row["id"] for row in read_jsonl(result_path)}
    elif result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path} without --resume")
    model_config = yaml.safe_load(Path(config["model_config"]).read_text(encoding="utf-8"))
    write_json(
        shard_dir / "runtime.json",
        {
            **runtime_metadata(model, processor, offset_tokenizer, model_config, layers),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "image_count": len(selected),
        },
    )
    with torch.inference_mode():
        for group_number, group in enumerate(selected):
            prepared = prepare_common_group(processor, offset_tokenizer, model, group)
            verify_pair_layout(model, prepared)
            for item in prepared:
                baseline, contexts = capture_prompt_with_cache(model, item["padded"], layers)
                item.update({"baseline": baseline, "contexts": contexts})
            for target_index, target in enumerate(prepared):
                record = target["record"]
                if record["id"] in completed:
                    continue
                other = prepared[1 - target_index]
                baseline_result = score_and_generate(
                    model,
                    processor,
                    target,
                    target["baseline"].logits,
                    target["baseline"].past_key_values,
                    config,
                )
                result_row = {
                    "schema_version": "query_refinement_gqa_result_v1",
                    "id": record["id"],
                    "image_id": record["image_id"],
                    "image_index": record["image_index"],
                    "question_index": record["question_index"],
                    "paired_other_id": other["record"]["id"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "question_types": record["question_types"],
                    "semantic_object_ids": record["semantic_object_ids"],
                    "semantic_program_depth": record["semantic_program_depth"],
                    "pair_stratum": record["pair_stratum"],
                    "different_evidence": record["different_evidence"],
                    "pair_match_distance": record["pair_match_distance"],
                    "question_word_length": record["question_word_length"],
                    "answer_token_length": record["answer_token_length"],
                    "prompt_token_length": target["original_length"],
                    "common_prompt_token_length": target["common_length"],
                    "visual_token_count": record["expected_visual_token_count"],
                    "question_token_span": {
                        key: value
                        for key, value in target["question_span"].items()
                        if key != "mask"
                    },
                    "BASELINE": baseline_result,
                    "layers": [],
                }
                for layer in layers:
                    layer_row = {"layer": layer, "variants": {}}
                    for variant, conditioning, conditioned in (
                        ("UNCONDITIONED_REPLAY", target, False),
                        ("TARGET_QUERY_REPLAY", target, True),
                        ("OTHER_QUERY_REPLAY", other, True),
                    ):
                        replay = execute_replay(
                            model,
                            target,
                            conditioning,
                            layer,
                            target["baseline"].past_key_values,
                            conditioned,
                        )
                        scored = score_and_generate(
                            model,
                            processor,
                            target,
                            replay.prompt_logits,
                            replay.past_key_values,
                            config,
                        )
                        layer_row["variants"][variant] = {
                            **scored,
                            "visual_reconstruction_max_abs": replay.visual_reconstruction_max_abs,
                            "inserted_visual_max_abs": replay.inserted_visual_max_abs,
                            "conditioning_edge_count": replay.conditioning_edge_count,
                        }
                    result_row["layers"].append(layer_row)
                append_jsonl(result_path, result_row)
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
    rows = read_jsonl(result_path)
    write_json(
        shard_dir / "completion.json",
        {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "expected_records": 2 * len(selected),
            "completed_records": len(rows),
            "complete": len(rows) == 2 * len(selected),
            "result_sha256": sha256_file(result_path),
        },
    )
    return 0


def execute(args):
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    actual = sha256_file(manifest_path)
    expected = expected_manifest_sha(Path(config["manifest_checksum"]))
    if actual != expected:
        raise RuntimeError(f"Manifest checksum mismatch: {actual} != {expected}")
    rows = read_jsonl(manifest_path)
    if len(rows) != int(config["sample_count"]):
        raise RuntimeError("Manifest record count drift")
    groups = group_manifest(rows)
    if len(groups) != int(config["image_count"]):
        raise RuntimeError("Manifest image count drift")
    model_config = yaml.safe_load(Path(config["model_config"]).read_text(encoding="utf-8"))
    set_determinism(int(config["seed"]))
    model, processor, offset_tokenizer = load_model(model_config)
    if args.mode == "preflight":
        output_dir = Path(config["preflight_output_dir"])
        if (output_dir / "summary.json").exists():
            raise FileExistsError("Refusing to overwrite completed preflight")
        write_json(
            output_dir / "runtime.json",
            runtime_metadata(
                model,
                processor,
                offset_tokenizer,
                model_config,
                [int(value) for value in config["refinement_layers"]],
            ),
        )
        return run_preflight(config, model, processor, offset_tokenizer, groups)
    return run_full(
        config,
        model,
        processor,
        offset_tokenizer,
        groups,
        args.shard_index,
        args.shard_count,
        args.resume,
    )


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(execute(args))
    except Exception as error:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        failure_dir = (
            Path(config["preflight_output_dir"])
            if args.mode == "preflight"
            else Path(config["shard_output_dir"]) / f"shard_{args.shard_index:02d}"
        )
        write_json(
            failure_dir / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "mode": args.mode,
            },
        )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
