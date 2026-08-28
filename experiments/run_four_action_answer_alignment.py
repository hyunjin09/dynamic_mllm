#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch
import transformers
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from binary_policy.executor import (
    BinaryQwen25VL,
    binary_greedy_generate,
    capture_full_baseline,
    greedy_generate_from_cached_prompt,
    greedy_generate_from_local_forward,
    full_baseline_post_layer_text_states,
    layerwise_token_scores_from_cached_prompt,
    local_four_action_forward,
    score_token_ids_from_cached_prompt,
    score_token_ids_from_local_forward,
)
from binary_policy.executor.inputs import build_binary_inputs
from reference.dvr_qwen.eval_metrics import score_prediction
from scoring.reference_likelihood import factorial_effects
from tools.research_analysis.four_action.targets import (
    AnswerTarget,
    accepted_answer_targets,
    answer_targets_are_scorable,
    full_wrong_target,
)
from tools.research_analysis.four_action.parallelism import (
    artifact_names,
    partition_gpu_rows,
    worker_layout,
)


MODES = (
    "preflight",
    "smoke",
    "pilot",
    "primary",
    "control_no_correction",
    "control_vision_required",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact local four-action answer alignment.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_answer_alignment.yaml"))
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Independent model replicas per each of the eight allocated GPUs.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Safe suffix for a non-overwriting diagnostic repetition of a mode.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()

    diff = subprocess.run(["git", "diff", "--binary"], check=True, capture_output=True).stdout
    untracked_raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    untracked_paths = sorted(
        path.decode("utf-8") for path in untracked_raw.split(b"\0") if path
    )
    reproducibility_roots = {
        "binary_policy", "configs", "experiments", "plans", "tests", "tools", "workspace"
    }
    untracked_sha256 = {}
    for path in untracked_paths:
        candidate = Path(path)
        if (
            candidate.is_file()
            and candidate.parts
            and candidate.parts[0] in reproducibility_roots
            and candidate.suffix in {".py", ".yaml", ".yml", ".md", ".toml", ".txt"}
        ):
            untracked_sha256[path] = sha256_file(candidate)
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "status_porcelain": run("git", "status", "--short"),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_file_sha256": untracked_sha256,
    }


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rank_and_world(args: argparse.Namespace) -> tuple[int, int]:
    rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank is not None else 0))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    return rank, world


def select_rows(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    mode: str,
    rank: int,
    world: int,
    eligibility: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    layout = worker_layout(rank, world, gpu_count=8)
    if mode in {"preflight", "smoke", "pilot"}:
        if layout.replicas_per_gpu != 1:
            raise RuntimeError(f"{mode} validation requires exactly one worker per GPU")
        ids = summary["smoke_ids"] if mode in {"preflight", "smoke"} else summary["pilot_ids"]
        order = {uid: index for index, uid in enumerate(ids)}
        selected = [row for row in rows if row["uid"] in order]
        selected.sort(key=lambda row: order[row["uid"]])
        if len(selected) != len(ids):
            raise RuntimeError(f"{mode} IDs are not exactly present in the cohort manifest")
        return [row for index, row in enumerate(selected) if index % world == rank]
    cohort = {
        "primary": "primary_a_plus",
        "control_no_correction": "control_no_correction_found",
        "control_vision_required": "control_full_correct_all_off_wrong",
    }[mode]
    if eligibility is None:
        raise RuntimeError(f"{mode} requires a frozen unified-FULL eligibility manifest")
    gpu_rows = []
    for row in rows:
        if row["cohort"] != cohort or int(row["shard"]) != layout.gpu_index:
            continue
        eligibility_row = eligibility.get(row["uid"])
        if eligibility_row is None:
            raise RuntimeError(f"eligibility manifest is missing {row['uid']}")
        if eligibility_row["eligible"] and (
            mode != "control_no_correction" or answer_targets_are_scorable(row)
        ):
            gpu_rows.append({**row, "unified_full_eligibility": eligibility_row})
    return partition_gpu_rows(
        gpu_rows, layout.replica_index, layout.replicas_per_gpu
    )


def load_model(config: dict[str, Any], device: torch.device):
    model_config = config["model"]
    snapshot = str(Path(model_config["snapshot_path"]).resolve())
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        snapshot,
        revision=model_config["revision"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_config["attention_implementation"],
        low_cpu_mem_usage=True,
    ).to(device)
    base.eval()
    base.requires_grad_(False)
    return BinaryQwen25VL(base), processor


def prepare(processor, record: dict[str, Any], device: torch.device):
    image_path = Path(record["image_path"])
    image_content: dict[str, Any] = {"type": "image", "image": str(image_path)}
    processor_kwargs: dict[str, Any] = {}
    if record.get("max_image_tokens"):
        max_pixels = int(record["max_image_tokens"]) * 28 * 28
        image_content["max_pixels"] = max_pixels
        processor_kwargs["max_pixels"] = max_pixels
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    messages = [{"role": "user", "content": [image_content, {"type": "text", "text": record["prompt"]}]}]
    literal = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(
        text=[literal],
        images=[image],
        videos=None,
        padding=True,
        return_tensors="pt",
        return_mm_token_type_ids=True,
        **processor_kwargs,
    )
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in dict(batch).items()}
    return literal, inputs


def token_ids(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    if ids.numel() < 1:
        raise ValueError(f"target tokenized to empty content: {text!r}")
    return ids


def target_tokenization_audit(tokenizer, prompt_text: str, target: AnswerTarget) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    combined = tokenizer(prompt_text + target.text, add_special_tokens=False).input_ids
    standalone = tokenizer(target.text, add_special_tokens=False).input_ids
    prefix = combined[: len(prompt_ids)] == prompt_ids
    return {
        "text": target.text,
        "evaluator_score": target.evaluator_score,
        "normalized_key": target.normalized_key,
        "source_count": target.source_count,
        "token_ids": standalone,
        "token_length": len(standalone),
        "standalone_matches_prompt_suffix": prefix and combined[len(prompt_ids) :] == standalone,
    }


def score_targets(
    scorer: Callable[[torch.Tensor], Any],
    tokenizer,
    targets: list[AnswerTarget],
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for target in targets:
        score = scorer(token_ids(tokenizer, target.text, device))
        rows.append(
            {
                "text": target.text,
                "evaluator_score": target.evaluator_score,
                "token_ids": score.token_ids,
                "token_logprobs": score.token_logprobs,
                "sequence_logprob": score.sequence_logprob,
                "mean_logprob": score.mean_logprob,
            }
        )
    selected = max(rows, key=lambda row: (row["mean_logprob"], row["sequence_logprob"], row["text"]))
    return {"selected": selected, "candidates": rows}


def decode_generation(tokenizer, generated_ids: torch.Tensor) -> str:
    return tokenizer.batch_decode(
        generated_ids.detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def correctness(record: dict[str, Any], prediction: str) -> tuple[float, bool]:
    score = score_prediction(
        record["metric_name"], prediction, record["answer"], record.get("all_answer_norms")
    )
    return float(score), bool(score >= float(record["correctness_threshold"]))


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return float((left.float() - right.to(left.device).float()).abs().max().item())


def logit_difference_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    left_float = left.float()
    right_float = right.to(left.device).float()
    difference = left_float - right_float
    left_logprobs = torch.log_softmax(left_float, dim=-1)
    right_logprobs = torch.log_softmax(right_float, dim=-1)
    logprob_difference = left_logprobs - right_logprobs
    reference_rms = right_float.square().mean().sqrt()
    top_k = min(10, int(left_float.shape[-1]))
    left_top = set(torch.topk(left_float[0], top_k).indices.detach().cpu().tolist())
    right_top = set(torch.topk(right_float[0], top_k).indices.detach().cpu().tolist())
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "reference_rms": float(reference_rms.item()),
        "relative_rmse": float((difference.square().mean().sqrt() / reference_rms.clamp_min(1e-12)).item()),
        "logprob_max_abs": float(logprob_difference.abs().max().item()),
        "logprob_mean_abs": float(logprob_difference.abs().mean().item()),
        "logprob_rmse": float(logprob_difference.square().mean().sqrt().item()),
        "argmax_match": int(left_float.argmax(dim=-1).item()) == int(right_float.argmax(dim=-1).item()),
        "top10_overlap_fraction": len(left_top & right_top) / top_k,
    }


def baseline_state(model, processor, record, prompt_text, inputs, baseline, device):
    correct_targets = accepted_answer_targets(record)
    wrong_target = None if bool(record["full_correct"]) else full_wrong_target(record)
    audited_targets = [*correct_targets] + ([] if wrong_target is None else [wrong_target])
    audits = [target_tokenization_audit(processor.tokenizer, prompt_text, target) for target in audited_targets]
    if not all(row["token_length"] > 0 and row["standalone_matches_prompt_suffix"] for row in audits):
        raise RuntimeError(f"answer-tokenization contract failed for {record['uid']}")
    assert baseline.cache is not None
    correct = score_targets(
        lambda ids: score_token_ids_from_cached_prompt(
            model, baseline.prompt_logits, baseline.inputs, baseline.cache, ids
        ),
        processor.tokenizer,
        correct_targets,
        device,
    )
    wrong = None
    if wrong_target is not None:
        wrong = score_targets(
            lambda ids: score_token_ids_from_cached_prompt(
                model, baseline.prompt_logits, baseline.inputs, baseline.cache, ids
            ),
            processor.tokenizer,
            [wrong_target],
            device,
        )
    generation = greedy_generate_from_cached_prompt(
        model,
        baseline.prompt_logits,
        baseline.inputs,
        baseline.cache,
        inputs["input_ids"],
        max_new_tokens=int(record["max_new_tokens"]),
    )
    text = decode_generation(processor.tokenizer, generation.generated_ids)
    score, correct_flag = correctness(record, text)
    correct_value = correct["selected"]["mean_logprob"]
    wrong_value = None if wrong is None else wrong["selected"]["mean_logprob"]
    return {
        "S_correct": correct["selected"]["mean_logprob"],
        "S_full_wrong": wrong_value,
        "margin": correct_value if wrong_value is None else correct_value - wrong_value,
        "score_quantity": "S_correct" if wrong_value is None else "S_correct_minus_S_full_wrong",
        "correct_target_scores": correct,
        "full_wrong_target_score": None if wrong is None else wrong["selected"],
        "target_tokenization": audits,
        "generated_answer": text,
        "generated_ids": generation.generated_ids[0].tolist(),
        "correctness_score": score,
        "correct": correct_flag,
    }, correct_targets, wrong_target


def intervention_state(model, processor, record, inputs, output, correct_targets, wrong_target, device):
    correct = score_targets(
        lambda ids: score_token_ids_from_local_forward(model, output, ids),
        processor.tokenizer,
        correct_targets,
        device,
    )
    wrong = None
    if wrong_target is not None:
        wrong = score_targets(
            lambda ids: score_token_ids_from_local_forward(model, output, ids),
            processor.tokenizer,
            [wrong_target],
            device,
        )
    generation = greedy_generate_from_local_forward(
        model,
        output,
        inputs["input_ids"],
        max_new_tokens=int(record["max_new_tokens"]),
    )
    text = decode_generation(processor.tokenizer, generation.generated_ids)
    score, correct_flag = correctness(record, text)
    correct_value = correct["selected"]["mean_logprob"]
    wrong_value = None if wrong is None else wrong["selected"]["mean_logprob"]
    return {
        "S_correct": correct["selected"]["mean_logprob"],
        "S_full_wrong": wrong_value,
        "margin": correct_value if wrong_value is None else correct_value - wrong_value,
        "score_quantity": "S_correct" if wrong_value is None else "S_correct_minus_S_full_wrong",
        "correct_target_scores": correct,
        "full_wrong_target_score": None if wrong is None else wrong["selected"],
        "generated_answer": text,
        "generated_ids": generation.generated_ids[0].tolist(),
        "correctness_score": score,
        "correct": correct_flag,
        "cache_lengths": output.prefill.cache.lengths() if output.prefill.cache else None,
    }


def binary_single_off_semantic_state(
    model,
    processor,
    record,
    inputs,
    prepared,
    layer,
) -> dict[str, Any]:
    route = [1] * len(model.decoder.layers)
    route[layer] = 0
    binary = binary_greedy_generate(
        model,
        inputs,
        route,
        max_new_tokens=int(record["max_new_tokens"]),
        prepared_inputs=prepared,
    )
    text = decode_generation(processor.tokenizer, binary.generated_ids)
    score, correct = correctness(record, text)
    return {
        "generated_answer": text,
        "generated_ids": binary.generated_ids[0].tolist(),
        "correctness_score": score,
        "correct": correct,
    }


def external_semantic_comparison(
    unified_state: dict[str, Any], external_state: dict[str, Any]
) -> dict[str, Any]:
    return {
        "generated_ids_match": unified_state["generated_ids"] == external_state["generated_ids"],
        "generated_answer_match": unified_state["generated_answer"] == external_state["generated_answer"],
        "evaluator_score_match": float(unified_state["correctness_score"])
        == float(external_state["correctness_score"]),
        "correctness_match": bool(unified_state["correct"]) == bool(external_state["correct"]),
        "external": external_state,
    }


def native_unified_full_diagnostic(
    unified_state: dict[str, Any], native_state: dict[str, Any]
) -> dict[str, Any]:
    signed = {
        "S_correct": float(unified_state["S_correct"]) - float(native_state["S_correct"]),
        "margin": float(unified_state["margin"]) - float(native_state["margin"]),
    }
    if unified_state["S_full_wrong"] is not None and native_state["S_full_wrong"] is not None:
        signed["S_full_wrong"] = float(unified_state["S_full_wrong"]) - float(
            native_state["S_full_wrong"]
        )
    return {
        "definition": "unified_materialized_full_minus_native_maskless_full",
        "signed_drift": signed,
        "absolute_drift": {name: abs(value) for name, value in signed.items()},
        **external_semantic_comparison(unified_state, native_state),
    }


def answer_trajectory_from_cached_states(
    model, states, meta, cache, baseline_state_row, device
) -> dict[str, Any]:
    correct_ids = torch.tensor(
        baseline_state_row["correct_target_scores"]["selected"]["token_ids"],
        dtype=torch.long,
        device=device,
    )
    correct_scores = layerwise_token_scores_from_cached_prompt(
        model,
        states,
        meta,
        cache,
        correct_ids,
    )
    wrong_scores = None
    if baseline_state_row["full_wrong_target_score"] is not None:
        wrong_ids = torch.tensor(
            baseline_state_row["full_wrong_target_score"]["token_ids"],
            dtype=torch.long,
            device=device,
        )
        wrong_scores = layerwise_token_scores_from_cached_prompt(
            model,
            states,
            meta,
            cache,
            wrong_ids,
        )
    margins = correct_scores if wrong_scores is None else [
        correct - wrong for correct, wrong in zip(correct_scores, wrong_scores)
    ]
    peak_layer = max(range(len(margins)), key=lambda layer: (margins[layer], -layer))
    adjacent = [margins[layer] - margins[layer - 1] for layer in range(1, len(margins))]
    drop_offset = min(range(len(adjacent)), key=lambda index: (adjacent[index], index))
    final_difference = float(margins[-1]) - float(baseline_state_row["margin"])
    return {
        "schema_version": "unified_full_answer_trajectory_v1",
        "readout": "post_layer_hidden_then_final_decoder_norm_then_lm_head_teacher_forced",
        "correct_target_text": baseline_state_row["correct_target_scores"]["selected"]["text"],
        "correct_target_token_ids": baseline_state_row["correct_target_scores"]["selected"]["token_ids"],
        "wrong_target_text": None
        if baseline_state_row["full_wrong_target_score"] is None
        else baseline_state_row["full_wrong_target_score"]["text"],
        "wrong_target_token_ids": None
        if baseline_state_row["full_wrong_target_score"] is None
        else baseline_state_row["full_wrong_target_score"]["token_ids"],
        "S_correct_by_layer": correct_scores,
        "S_full_wrong_by_layer": wrong_scores,
        "margin_by_layer": margins,
        "maximum_intermediate_margin": float(margins[peak_layer]),
        "peak_layer": peak_layer,
        "final_margin": float(margins[-1]),
        "peak_to_final_erosion": float(margins[peak_layer] - margins[-1]),
        "largest_adjacent_change": float(adjacent[drop_offset]),
        "largest_drop_arrival_layer": drop_offset + 1,
        "final_margin_vs_factorial_baseline_abs_diff": abs(final_difference),
    }


def unified_full_answer_trajectory(model, baseline_output, baseline_state_row, device) -> dict[str, Any]:
    assert baseline_output.cache is not None
    return answer_trajectory_from_cached_states(
        model,
        full_baseline_post_layer_text_states(baseline_output),
        baseline_output.inputs,
        baseline_output.cache,
        baseline_state_row,
        device,
    )
def preflight_controls(
    model,
    base_model,
    processor,
    record,
    inputs,
    prepared,
    factorial_baseline,
    native_baseline,
    layer_grid,
):
    native_generation = greedy_generate_from_cached_prompt(
        model,
        native_baseline.prompt_logits,
        native_baseline.inputs,
        native_baseline.cache,
        inputs["input_ids"],
        max_new_tokens=int(record["max_new_tokens"]),
    )
    base_model.rope_deltas = None
    native_hf_ids = base_model.generate(
        **inputs, max_new_tokens=int(record["max_new_tokens"]), do_sample=False, use_cache=True
    )[0, inputs["input_ids"].shape[1] :].detach().cpu().tolist()
    controls = {
        "unified_full_vs_native_prompt_logit_stats": logit_difference_stats(
            factorial_baseline.prompt_logits, native_baseline.prompt_logits
        ),
        "native_cached_vs_hf_generated_ids": native_generation.generated_ids[0].tolist() == native_hf_ids,
        "native_generated_ids": native_generation.generated_ids[0].tolist(),
        "hf_generated_ids": native_hf_ids,
        "cached_full_ids_match": native_generation.generated_ids[0].tolist()
        == record["binary_routes"]["full_anchor"]["generated_ids"],
        "unified_generated_ids_match_native": (
            greedy_generate_from_cached_prompt(
                model,
                factorial_baseline.prompt_logits,
                factorial_baseline.inputs,
                factorial_baseline.cache,
                inputs["input_ids"],
                max_new_tokens=int(record["max_new_tokens"]),
            ).generated_ids.tolist()
            == native_generation.generated_ids.tolist()
        ),
        "all_parameters_frozen": all(not parameter.requires_grad for parameter in base_model.parameters()),
        "layers": [],
    }
    num_layers = len(model.decoder.layers)
    for layer in layer_grid:
        full = local_four_action_forward(model, factorial_baseline, layer, "FULL")
        ignore = local_four_action_forward(model, factorial_baseline, layer, "IGNORE")
        read_only = local_four_action_forward(model, factorial_baseline, layer, "READ_ONLY")
        write_only = local_four_action_forward(model, factorial_baseline, layer, "WRITE_ONLY")
        outputs = {
            "FULL": full,
            "IGNORE": ignore,
            "READ_ONLY": read_only,
            "WRITE_ONLY": write_only,
        }
        repeats = {
            action: local_four_action_forward(model, factorial_baseline, layer, action)
            for action in outputs
        }
        route = [1] * num_layers
        route[layer] = 0
        binary = binary_greedy_generate(
            model,
            inputs,
            route,
            max_new_tokens=int(record["max_new_tokens"]),
            prepared_inputs=prepared,
        )
        expected_full = int(record["full_prompt_token_count"])
        expected_text = int(record["text_token_count"])
        full_generation = greedy_generate_from_local_forward(
            model, full, inputs["input_ids"], max_new_tokens=int(record["max_new_tokens"])
        )
        ignore_generation = greedy_generate_from_local_forward(
            model, ignore, inputs["input_ids"], max_new_tokens=int(record["max_new_tokens"])
        )
        repeat_generation_match = {}
        repeat_prompt_errors = {}
        for action, output in outputs.items():
            repeat_prompt_errors[action] = max_abs(output.prompt_logits, repeats[action].prompt_logits)
            first_ids = greedy_generate_from_local_forward(
                model, output, inputs["input_ids"], max_new_tokens=int(record["max_new_tokens"])
            ).generated_ids
            repeat_ids = greedy_generate_from_local_forward(
                model, repeats[action], inputs["input_ids"], max_new_tokens=int(record["max_new_tokens"])
            ).generated_ids
            repeat_generation_match[action] = first_ids.tolist() == repeat_ids.tolist()
        visual_update = (
            write_only.prefill.target_post_visual_state.float()
            - write_only.prefill.target_pre_visual_state.float()
        )
        controls["layers"].append(
            {
                "layer": layer,
                "full_prompt_logit_max_abs": max_abs(full.prompt_logits, factorial_baseline.prompt_logits),
                "full_generated_ids_match_factorial_baseline": (
                    full_generation.generated_ids.tolist()
                    == greedy_generate_from_cached_prompt(
                        model,
                        factorial_baseline.prompt_logits,
                        factorial_baseline.inputs,
                        factorial_baseline.cache,
                        inputs["input_ids"],
                        max_new_tokens=int(record["max_new_tokens"]),
                    ).generated_ids.tolist()
                ),
                "factorial_full_generated_ids_match_native": (
                    full_generation.generated_ids.tolist() == native_generation.generated_ids.tolist()
                ),
                "ignore_binary_prompt_logit_max_abs": max_abs(
                    ignore.prompt_logits, binary.prefill_logits
                ),
                "ignore_binary_prompt_logit_stats": logit_difference_stats(
                    ignore.prompt_logits, binary.prefill_logits
                ),
                "ignore_binary_generated_ids_match": (
                    ignore_generation.generated_ids.tolist() == binary.generated_ids.tolist()
                ),
                "ignore_visual_bypass_max_abs": max_abs(
                    ignore.prefill.target_post_visual_state,
                    ignore.prefill.target_pre_visual_state,
                ),
                "read_only_visual_bypass_max_abs": max_abs(
                    read_only.prefill.target_post_visual_state,
                    read_only.prefill.target_pre_visual_state,
                ),
                "read_only_text_vs_full_max_abs": max_abs(
                    read_only.prefill.target_post_text_state,
                    full.prefill.target_post_text_state,
                ),
                "write_only_visual_vs_full_max_abs": max_abs(
                    write_only.prefill.target_post_visual_state,
                    full.prefill.target_post_visual_state,
                ),
                "write_only_text_vs_ignore_max_abs": max_abs(
                    write_only.prefill.target_post_text_state,
                    ignore.prefill.target_post_text_state,
                ),
                "write_only_visual_update_rms": float(
                    visual_update.square().mean().sqrt().item()
                ),
                "read_only_target_cache_rows": read_only.prefill.cache.get_seq_length(layer),
                "write_only_target_cache_rows": write_only.prefill.cache.get_seq_length(layer),
                "ignore_target_cache_rows": ignore.prefill.cache.get_seq_length(layer),
                "full_target_cache_rows": full.prefill.cache.get_seq_length(layer),
                "expected_full_rows": expected_full,
                "expected_text_rows": expected_text,
                "repeat_prompt_logit_max_abs": repeat_prompt_errors,
                "repeat_generated_ids_match": repeat_generation_match,
                "target_pre_text_shared_max_abs": max(
                    max_abs(output.prefill.target_pre_text_state, full.prefill.target_pre_text_state)
                    for output in outputs.values()
                ),
                "target_pre_visual_shared_max_abs": max(
                    max_abs(output.prefill.target_pre_visual_state, full.prefill.target_pre_visual_state)
                    for output in outputs.values()
                ),
                "target_decoder_calls": {
                    action: output.prefill.layer_stats[layer].decoder_calls
                    for action, output in outputs.items()
                },
                "non_target_actions": {
                    action: [
                        stat.action for index, stat in enumerate(output.prefill.layer_stats)
                        if index != layer
                    ]
                    for action, output in outputs.items()
                },
            }
        )
    return controls


def validate_preflight(controls: dict[str, Any], result_layers: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "unified_full_native_generation_parity": controls["unified_generated_ids_match_native"],
        "native_cached_hf_generation_parity": controls["native_cached_vs_hf_generated_ids"],
        "native_cached_label_generation_parity": controls["cached_full_ids_match"],
        "base_model_frozen": controls["all_parameters_frozen"],
        "local_full_identity": all(
            row["full_prompt_logit_max_abs"] <= float(config["full_local_prompt_logit_atol"])
            and row["full_generated_ids_match_factorial_baseline"]
            and row["factorial_full_generated_ids_match_native"]
            for row in controls["layers"]
        ),
        "ignore_binary_identity": all(
            row["ignore_binary_generated_ids_match"]
            for row in controls["layers"]
        ),
        "unified_branch_execution_consistency": all(
            row["target_pre_text_shared_max_abs"] == 0.0
            and row["target_pre_visual_shared_max_abs"] == 0.0
            and set(row["target_decoder_calls"].values()) == {2}
            and all(
                set(actions) == {"FULL"}
                for actions in row["non_target_actions"].values()
            )
            for row in controls["layers"]
        ),
        "read_write_target_state_semantics": all(
            row["ignore_visual_bypass_max_abs"] <= float(config["visual_bypass_atol"])
            and row["read_only_visual_bypass_max_abs"] <= float(config["visual_bypass_atol"])
            and row["read_only_text_vs_full_max_abs"] == 0.0
            and row["write_only_visual_vs_full_max_abs"] == 0.0
            and row["write_only_text_vs_ignore_max_abs"] == 0.0
            and row["write_only_visual_update_rms"] > 0.0
            for row in controls["layers"]
        ),
        "target_cache_geometry": all(
            row["read_only_target_cache_rows"] == row["expected_full_rows"]
            and row["full_target_cache_rows"] == row["expected_full_rows"]
            and row["write_only_target_cache_rows"] == row["expected_text_rows"]
            and row["ignore_target_cache_rows"] == row["expected_text_rows"]
            for row in controls["layers"]
        ),
        "deterministic_actions": all(
            all(value == 0.0 for value in row["repeat_prompt_logit_max_abs"].values())
            and all(row["repeat_generated_ids_match"].values())
            for row in controls["layers"]
        ),
        "finite_scores": all(
            torch.isfinite(torch.tensor(state["margin"]))
            for layer in result_layers
            for state in layer["states"].values()
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def validate_sample_result(
    result: dict[str, Any],
    record: dict[str, Any],
    expected_layers: int,
    require_external_semantic_parity: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "layer_count": len(result["layers"]) == expected_layers,
        "complete_factorial_states": all(
            set(layer["states"]) == {"IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL"}
            for layer in result["layers"]
        ),
        "finite_margins_and_effects": all(
            torch.isfinite(torch.tensor(value))
            for layer in result["layers"]
            for value in [
                *[state["margin"] for state in layer["states"].values()],
                *layer["effects"].values(),
            ]
        ),
        "target_cache_geometry": all(
            layer["states"]["IGNORE"]["cache_lengths"][layer["layer"]]
            == int(record["text_token_count"])
            and layer["states"]["WRITE_ONLY"]["cache_lengths"][layer["layer"]]
            == int(record["text_token_count"])
            and layer["states"]["READ_ONLY"]["cache_lengths"][layer["layer"]]
            == int(record["full_prompt_token_count"])
            for layer in result["layers"]
        ),
        "trajectory_final_margin_identity": result["unified_full_answer_trajectory"][
            "final_margin_vs_factorial_baseline_abs_diff"
        ]
        <= float(config["trajectory_final_margin_atol"]),
    }
    diagnostics = {}
    eligibility = record.get("unified_full_eligibility")
    if eligibility is None:
        checks["native_baseline_correctness_matches_cohort"] = (
            result["native_full_external"]["state"]["correct"]
            == bool(record["full_correct"])
        )
    else:
        checks["unified_full_matches_eligibility_freeze"] = (
            result["baseline_full"]["generated_ids"]
            == eligibility["unified_full_generated_ids"]
            and bool(result["baseline_full"]["correct"])
            == bool(eligibility["unified_full_correct"])
            and bool(eligibility["eligible"])
        )
        diagnostics["historical_cohort_correctness_matches_current_unified_full"] = (
            bool(record["full_correct"]) == bool(result["baseline_full"]["correct"])
        )
    if record.get("binary_routes") is not None:
        diagnostics["historical_full_anchor_generated_ids_match"] = (
            result["native_full_external"]["state"]["generated_ids"]
            == record["binary_routes"]["full_anchor"]["generated_ids"]
        )
    if require_external_semantic_parity:
        full_diagnostic = result["native_full_external"]["diagnostic"]
        checks["unified_full_native_semantic_parity"] = all(
            full_diagnostic[name]
            for name in (
                "generated_ids_match",
                "generated_answer_match",
                "evaluator_score_match",
                "correctness_match",
            )
        )
        checks["unified_ignore_binary_semantic_parity"] = all(
            all(
                layer["old_binary_ignore_external"][name]
                for name in (
                    "generated_ids_match",
                    "generated_answer_match",
                    "evaluator_score_match",
                    "correctness_match",
                )
            )
            for layer in result["layers"]
        )
    return {
        "checks": checks,
        "diagnostics": diagnostics,
        "passed": all(checks.values()),
    }


def process_record(model, processor, record, layers, actions, mode, device, config):
    started = time.monotonic()
    prompt_text, inputs = prepare(processor, record, device)
    prepared = build_binary_inputs(model, inputs)
    if int(prepared.visual_valid_mask.sum().item()) != int(record["visual_token_count"]):
        raise RuntimeError(f"visual token count drift for {record['uid']}")
    factorial_baseline = capture_full_baseline(
        model, inputs, prepared_inputs=prepared, use_cache=True, native_causal=False
    )
    baseline, correct_targets, wrong_target = baseline_state(
        model, processor, record, prompt_text, inputs, factorial_baseline, device
    )
    native_baseline = capture_full_baseline(
        model, inputs, prepared_inputs=prepared, use_cache=True, native_causal=True
    )
    native_state, _, _ = baseline_state(
        model, processor, record, prompt_text, inputs, native_baseline, device
    )
    result = {
        "schema_version": "four_action_sample_v1",
        "uid": record["uid"],
        "dataset": record["dataset"],
        "cohort": record["cohort"],
        "sample_id": record["sample_id"],
        "image_id": record["image_id"],
        "image_group_id": record["image_group_id"],
        "visual_token_count": record["visual_token_count"],
        "text_token_count": record["text_token_count"],
        "binary_routes": record["binary_routes"],
        "unified_full_eligibility": record.get("unified_full_eligibility"),
        "baseline_full": baseline,
        "factorial_executor": "unified_materialized_full_v1",
        "native_full_external": {
            "state": native_state,
            "diagnostic": native_unified_full_diagnostic(baseline, native_state),
        },
        "unified_full_answer_trajectory": unified_full_answer_trajectory(
            model, factorial_baseline, baseline, device
        ),
        "layers": [],
    }
    if mode == "preflight":
        result["preflight_controls"] = preflight_controls(
            model,
            model.base_model,
            processor,
            record,
            inputs,
            prepared,
            factorial_baseline,
            native_baseline,
            layers,
        )
    for layer in layers:
        states = {"FULL": baseline}
        for action in actions:
            if action == "FULL":
                continue
            output = local_four_action_forward(model, factorial_baseline, layer, action)
            states[action] = intervention_state(
                model, processor, record, inputs, output, correct_targets, wrong_target, device
            )
            del output
        margins = {name: float(states[name]["margin"]) for name in ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")}
        result["layers"].append(
            {"layer": layer, "states": states, "effects": factorial_effects(margins)}
        )
        if mode in {"preflight", "smoke", "pilot"}:
            binary_ignore = binary_single_off_semantic_state(
                model,
                processor,
                record,
                inputs,
                prepared,
                layer,
            )
            result["layers"][-1]["old_binary_ignore_external"] = external_semantic_comparison(
                states["IGNORE"], binary_ignore
            )
    result["sample_gate"] = validate_sample_result(
        result,
        record,
        len(layers),
        require_external_semantic_parity=mode in {"preflight", "smoke", "pilot"},
        config=config,
    )
    if mode == "preflight":
        result["preflight_gate"] = validate_preflight(
            result["preflight_controls"], result["layers"], config
        )
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def runtime_metadata(
    config, config_path, mode, output_tag, layout, device, selected_count
):
    return {
        "schema_version": "four_action_runtime_v1",
        "mode": mode,
        "output_tag": output_tag,
        "rank": layout.rank,
        "world_size": layout.world_size,
        "gpu_index": layout.gpu_index,
        "replica_index": layout.replica_index,
        "replicas_per_gpu": layout.replicas_per_gpu,
        "selected_count": selected_count,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_count_visible": torch.cuda.device_count(),
        "model": config["model"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "cohort_manifest_sha256": sha256_file(Path(config["cohort_manifest"])),
        "git": git_metadata(),
        "all_eight_gpu_workers_required": True,
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_tag and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.output_tag):
        raise ValueError("--output-tag must be a safe lowercase identifier")
    rank, world = rank_and_world(args)
    if not torch.cuda.is_available():
        raise RuntimeError("four-action model execution requires Slurm-allocated GPUs")
    if torch.cuda.device_count() != 8:
        raise RuntimeError(f"the allocation must expose all 8 GPUs, got {torch.cuda.device_count()}")
    layout = worker_layout(rank, world, gpu_count=8)
    if layout.replicas_per_gpu != args.workers_per_gpu:
        raise RuntimeError(
            f"torchrun created {layout.replicas_per_gpu} workers/GPU but "
            f"--workers-per-gpu={args.workers_per_gpu}"
        )
    device = torch.device(f"cuda:{layout.gpu_index}")
    torch.cuda.set_device(device)
    set_determinism(int(config["seed"]) + layout.gpu_index)
    rows = read_jsonl(Path(config["cohort_manifest"]))
    summary = json.loads(Path(config["cohort_summary"]).read_text(encoding="utf-8"))
    eligibility = None
    if args.mode not in {"preflight", "smoke", "pilot"}:
        eligibility_path = Path(config["eligibility_root"]) / "merged_results.jsonl"
        eligibility_rows = read_jsonl(eligibility_path)
        eligibility = {row["uid"]: row for row in eligibility_rows}
        if len(eligibility) != len(eligibility_rows):
            raise RuntimeError("eligibility manifest contains duplicate UIDs")
    selected = select_rows(rows, summary, args.mode, rank, world, eligibility)
    mode_directory = args.mode if not args.output_tag else f"{args.mode}__{args.output_tag}"
    output_dir = (
        Path(config["output_root"])
        / mode_directory
        / f"shard_{layout.gpu_index:02d}"
    )
    names = artifact_names(layout.replicas_per_gpu, layout.replica_index)
    result_path = output_dir / names["results"]
    completed: set[str] = set()
    existing_result_paths = sorted(output_dir.glob("results*.jsonl"))
    if existing_result_paths:
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {output_dir} without --resume")
        completed = {
            row["uid"]
            for path in existing_result_paths
            for row in read_jsonl(path)
        }
    runtime_path = output_dir / names["runtime"]
    if not runtime_path.exists():
        write_json_once(
            runtime_path,
            runtime_metadata(
                config, args.config, args.mode, args.output_tag, layout, device, len(selected)
            ),
        )
    model, processor = load_model(config, device)
    layers = config["preflight_layer_grid"] if args.mode == "preflight" else config["layer_grid"]
    actions = config["actions"] if args.mode == "preflight" else config["new_branches"]
    failures = 0
    for index, record in enumerate(selected):
        if record["uid"] in completed:
            continue
        try:
            result = process_record(model, processor, record, layers, actions, args.mode, device, config)
            append_jsonl(result_path, result)
            print(json.dumps({"rank": rank, "completed": index + 1, "total": len(selected), "uid": record["uid"], "elapsed": result["elapsed_seconds"]}), flush=True)
            if args.mode == "preflight" and not result["preflight_gate"]["passed"]:
                failures += 1
                failed = [
                    name
                    for name, passed in result["preflight_gate"]["checks"].items()
                    if not passed
                ]
                append_jsonl(
                    output_dir / names["failures"],
                    {"uid": record["uid"], "error": f"preflight gate failed: {failed}"},
                )
                break
            if not result["sample_gate"]["passed"]:
                failures += 1
                failed = [
                    name
                    for name, passed in result["sample_gate"]["checks"].items()
                    if not passed
                ]
                append_jsonl(
                    output_dir / names["failures"],
                    {"uid": record["uid"], "error": f"sample gate failed: {failed}"},
                )
                break
        except Exception as exc:
            failures += 1
            append_jsonl(
                output_dir / names["failures"],
                {"uid": record["uid"], "error": str(exc), "traceback": traceback.format_exc()},
            )
            print(json.dumps({"rank": rank, "uid": record["uid"], "error": str(exc)}), flush=True)
            break
        finally:
            torch.cuda.empty_cache()
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
