from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from audit.token_layout import describe_token_layout
from experiments.stage_a_validity import (
    load_yaml,
    max_abs_difference,
    prepare_prompt,
    set_determinism,
    write_csv,
    write_json,
)
from interventions.four_state import FOUR_STATES, LayerCapture
from interventions.prompt_cache import clone_dynamic_cache, run_cached_prompt_state
from interventions.read_path import ReadInterventionCache
from scoring.benchmark_metrics import score_record
from scoring.reference_likelihood import (
    AcceptedAnswer,
    accepted_answers,
    aggregate_accepted_scores,
    factorial_effects,
    score_reference_from_prompt,
)


TECHNICAL_INVALID_RULES = [
    "prompt token length exceeds validated stock-eager maximum",
    "normalized accepted-answer set is empty",
    "an accepted answer has an empty token span",
    "standalone accepted-answer tokenization differs from its prompt-concatenated suffix",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B reference-answer likelihood discovery.")
    parser.add_argument("--config", default="configs/stage_b.yaml")
    parser.add_argument("--mode", choices=("validity", "full"), required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_exclusions(path: Path, exclusions: list[dict[str, Any]]) -> None:
    write_json(
        path,
        {
            "rules_frozen_before_intervention_outcomes": TECHNICAL_INVALID_RULES,
            "exclusion_count": len(exclusions),
            "exclusions": exclusions,
        },
    )


def capture_prompt_with_cache(causal_lm, inputs: dict[str, Any], layer_indices: list[int]):
    captures = [LayerCapture(causal_lm.model.layers[index], index) for index in layer_indices]
    causal_lm.rope_deltas = None
    with ExitStack() as stack:
        for capture in captures:
            stack.enter_context(capture)
        outputs = causal_lm(**inputs, use_cache=True, return_dict=True)
    return outputs, {capture.layer_index: capture.context() for capture in captures}


def answer_tokenization_audit(tokenizer, prompt_text: str, answer: str) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    combined_ids = tokenizer(prompt_text + answer, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    prefix_match = combined_ids[: len(prompt_ids)] == prompt_ids
    suffix_ids = combined_ids[len(prompt_ids) :] if prefix_match else []
    return {
        "answer": answer,
        "answer_token_ids": answer_ids,
        "answer_token_length": len(answer_ids),
        "prompt_is_exact_combined_prefix": prefix_match,
        "standalone_answer_matches_combined_suffix": prefix_match and suffix_ids == answer_ids,
        "prompt_positions_contributing_to_score": 0,
    }


def score_accepted_answer_set(
    causal_lm,
    tokenizer,
    prompt_logits: torch.Tensor,
    prompt_cache,
    prompt_attention_mask: torch.Tensor,
    answers: list[AcceptedAnswer],
) -> dict[str, Any]:
    scores = [
        score_reference_from_prompt(
            causal_lm,
            tokenizer,
            prompt_logits,
            prompt_cache,
            prompt_attention_mask,
            answer.text,
        )
        for answer in answers
    ]
    aggregate = aggregate_accepted_scores(answers, scores)
    return {
        **aggregate,
        "accepted_answer_scores": [
            {
                "answer": answer.text,
                "weight": answer.weight,
                "token_ids": score.token_ids,
                "token_length": len(score.token_ids),
                "token_logprobs": score.token_logprobs,
                "sequence_logprob": score.sequence_logprob,
                "mean_logprob": score.mean_logprob,
            }
            for answer, score in zip(answers, scores)
        ],
    }


def apply_repetition_penalty(
    scores: torch.Tensor, input_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Match Transformers RepetitionPenaltyLogitsProcessor for cached greedy."""
    if penalty <= 0.0:
        raise ValueError("Repetition penalty must be strictly positive")
    if penalty == 1.0:
        return scores
    selected = torch.gather(scores, 1, input_ids)
    selected = torch.where(selected < 0, selected * penalty, selected / penalty)
    return scores.scatter(1, input_ids, selected)


def greedy_from_prompt(
    causal_lm,
    tokenizer,
    prompt_logits: torch.Tensor,
    prompt_cache,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    cache = clone_dynamic_cache(prompt_cache)
    generated: list[int] = []
    repetition_penalty = float(causal_lm.generation_config.repetition_penalty)
    next_scores = apply_repetition_penalty(
        prompt_logits[:, -1, :], prompt_input_ids, repetition_penalty
    )
    next_token = int(next_scores[0].argmax().item())
    eos_ids = causal_lm.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_set = {eos_ids}
    else:
        eos_set = {int(value) for value in eos_ids or []}
    prompt_length = int(prompt_attention_mask.shape[1])

    for _ in range(max_new_tokens):
        generated.append(next_token)
        if next_token in eos_set:
            break
        input_ids = torch.tensor([[next_token]], dtype=torch.long, device=prompt_logits.device)
        attention_mask = torch.cat(
            [
                prompt_attention_mask,
                torch.ones((1, len(generated)), dtype=prompt_attention_mask.dtype, device=prompt_logits.device),
            ],
            dim=1,
        )
        cache_position = torch.tensor(
            [prompt_length + len(generated) - 1], dtype=torch.long, device=prompt_logits.device
        )
        output = causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        history = torch.cat(
            [
                prompt_input_ids,
                torch.tensor([generated], dtype=torch.long, device=prompt_input_ids.device),
            ],
            dim=1,
        )
        next_scores = apply_repetition_penalty(
            output.logits[:, -1, :], history, repetition_penalty
        )
        next_token = int(next_scores[0].argmax().item())
    generated_tensor = torch.tensor([generated], dtype=torch.long)
    text = tokenizer.batch_decode(
        generated_tensor,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return {"token_ids": generated, "text": text}


def runtime_metadata(
    causal_lm,
    processor,
    model_config: dict[str, Any],
    layer_grid: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    first_attention = causal_lm.model.layers[0].self_attn
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "model_id": model_config["model_id"],
        "revision": model_config["revision"],
        "snapshot_path": model_config["snapshot_path"],
        "dtype": str(next(causal_lm.parameters()).dtype),
        "decoder_attention_execution": "transformers_stock_eager",
        "vision_attention_backend": causal_lm.config.vision_config._attn_implementation,
        "model_class": type(causal_lm).__name__,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(processor.tokenizer).__name__,
        "chat_template": processor.chat_template,
        "num_hidden_layers": causal_lm.config.num_hidden_layers,
        "num_attention_heads": causal_lm.config.num_attention_heads,
        "num_key_value_heads": causal_lm.config.num_key_value_heads,
        "num_key_value_groups": first_attention.num_key_value_groups,
        "layer_grid": layer_grid,
        "precision": "bfloat16 model; float32 log-softmax",
        "generation_config_from_snapshot": causal_lm.generation_config.to_dict(),
        "generation_overrides": {
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": max_new_tokens,
        },
        "all_parameters_frozen": all(not parameter.requires_grad for parameter in causal_lm.parameters()),
    }


def execute(args: argparse.Namespace) -> int:
    config = load_yaml(Path(args.config))
    model_config = load_yaml(Path(config["model_config"]))
    layer_grid = [int(value) for value in config["layer_grid"]]
    output_dir = Path(
        args.output_dir
        or (config["validity_output_dir"] if args.mode == "validity" else config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(int(config["seed"]))

    all_samples = read_jsonl(Path(config["candidate_manifest"]))
    if args.mode == "validity":
        wanted = set(config["validity_sample_ids"])
        samples = [row for row in all_samples if row["id"] in wanted]
        if {row["id"] for row in samples} != wanted:
            raise ValueError("Validity samples are not exactly present in the candidate manifest")
    else:
        validity = json.loads(
            (Path(config["validity_output_dir"]) / "stage_b_validity_summary.json").read_text(
                encoding="utf-8"
            )
        )
        if not validity["gate_pass"]:
            raise RuntimeError("Stage B validity gate did not pass")
        samples = all_samples

    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(
        model_config["snapshot_path"], local_files_only=True, use_fast=False
    )
    hf_config = AutoConfig.from_pretrained(model_config["snapshot_path"], local_files_only=True)
    hf_config._attn_implementation = {"vision_config": model_config["vision_attention_backend"]}
    causal_lm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_config["snapshot_path"],
        config=hf_config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    causal_lm.eval()
    causal_lm.requires_grad_(False)
    for layer in causal_lm.model.layers:
        layer.self_attn.stage_a_query_chunk_size = int(
            model_config["decoder_attention_query_chunk_size"]
        )
    write_json(
        output_dir / "runtime.json",
        runtime_metadata(
            causal_lm,
            processor,
            model_config,
            layer_grid,
            int(config["max_new_tokens"]),
        ),
    )

    result_path = output_dir / (
        "validity_results.jsonl" if args.mode == "validity" else "stage_b_results_v1.jsonl"
    )
    completed_ids: set[str] = set()
    if args.resume and result_path.exists():
        completed_ids = {row["id"] for row in read_jsonl(result_path)}
    elif result_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results without --resume: {result_path}"
        )

    no_op_rows: list[dict[str, Any]] = []
    layout_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    exclusions_path = output_dir / "technical_exclusions.json"
    write_exclusions(exclusions_path, exclusions)
    max_prompt_tokens = int(config["validated_prompt_token_max"])

    with torch.inference_mode():
        for sample_index, record in enumerate(samples):
            if record["id"] in completed_ids:
                continue
            prompt_text, prompt_inputs = prepare_prompt(processor, record, device)
            input_ids = prompt_inputs["input_ids"]
            prompt_length = int(input_ids.shape[1])
            if prompt_length > max_prompt_tokens:
                reason = (
                    f"prompt token length {prompt_length} exceeds validated stock-eager "
                    f"maximum {max_prompt_tokens}"
                )
                if args.mode == "validity":
                    raise RuntimeError(f"Technical invalidity for {record['id']}: {reason}")
                exclusions.append(
                    {"id": record["id"], "dataset": record["benchmark"], "reason": reason}
                )
                write_exclusions(exclusions_path, exclusions)
                continue
            visual_mask = input_ids == causal_lm.config.image_token_id
            position_ids, _ = causal_lm.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=prompt_inputs.get("image_grid_thw"),
                attention_mask=prompt_inputs.get("attention_mask"),
            )
            layout = describe_token_layout(
                processor.tokenizer, causal_lm.config, input_ids, position_ids
            )
            layout.update({"sample_id": record["id"], "dataset": record["benchmark"]})
            layout_rows.append(layout)

            try:
                answers = accepted_answers(record)
            except ValueError as exc:
                if args.mode == "validity":
                    raise
                exclusions.append(
                    {"id": record["id"], "dataset": record["benchmark"], "reason": str(exc)}
                )
                write_exclusions(exclusions_path, exclusions)
                continue
            tokenization = [
                answer_tokenization_audit(processor.tokenizer, prompt_text, answer.text)
                for answer in answers
            ]
            if not all(row["answer_token_length"] > 0 for row in tokenization):
                reason = "an accepted answer has an empty token span"
                if args.mode == "validity":
                    raise RuntimeError(f"Technical invalidity for {record['id']}: {reason}")
                exclusions.append(
                    {"id": record["id"], "dataset": record["benchmark"], "reason": reason}
                )
                write_exclusions(exclusions_path, exclusions)
                continue
            if not all(row["standalone_answer_matches_combined_suffix"] for row in tokenization):
                reason = (
                    "standalone accepted-answer tokenization differs from its "
                    "prompt-concatenated suffix"
                )
                if args.mode == "validity":
                    raise RuntimeError(f"Technical invalidity for {record['id']}: {reason}")
                exclusions.append(
                    {"id": record["id"], "dataset": record["benchmark"], "reason": reason}
                )
                write_exclusions(exclusions_path, exclusions)
                continue

            baseline, contexts = capture_prompt_with_cache(causal_lm, prompt_inputs, layer_grid)
            baseline_score = score_accepted_answer_set(
                causal_lm,
                processor.tokenizer,
                baseline.logits,
                baseline.past_key_values,
                prompt_inputs["attention_mask"],
                answers,
            )
            baseline_generation = greedy_from_prompt(
                causal_lm,
                processor.tokenizer,
                baseline.logits,
                baseline.past_key_values,
                prompt_inputs["input_ids"],
                prompt_inputs["attention_mask"],
                int(config["max_new_tokens"]),
            )
            standard_generation_match = None
            if args.mode == "validity":
                causal_lm.rope_deltas = None
                standard_generated = causal_lm.generate(
                    **prompt_inputs,
                    max_new_tokens=int(config["max_new_tokens"]),
                    do_sample=False,
                    use_cache=True,
                )
                standard_token_ids = [
                    int(value)
                    for value in standard_generated[0, prompt_length:].detach().cpu().tolist()
                ]
                standard_generation_match = (
                    baseline_generation["token_ids"] == standard_token_ids
                )
                no_op_rows.append(
                    {
                        "sample_id": record["id"],
                        "dataset": record["benchmark"],
                        "layer": -1,
                        "control": "cached_greedy_vs_standard_generate",
                        "prompt_logit_max_abs": 0.0,
                        "sequence_score_abs": 0.0,
                        "mean_score_abs": 0.0,
                        "generation_match": standard_generation_match,
                    }
                )
            baseline_correctness = score_record(record, baseline_generation["text"])

            sample_result: dict[str, Any] = {
                "schema_version": "stage_b_reference_likelihood_v1",
                "id": record["id"],
                "dataset": record["benchmark"],
                "question": record["question"],
                "inherited_bucket": record["inherited_bucket"],
                "prompt_text": prompt_text,
                "prompt_token_length": prompt_length,
                "visual_token_range": layout["visual_rows"],
                "accepted_answers": [
                    {"answer": answer.text, "weight": answer.weight}
                    for answer in answers
                ],
                "answer_tokenization": tokenization,
                "baseline_full": {
                    "sequence_logprob": baseline_score["sequence_logprob"],
                    "mean_logprob": baseline_score["mean_logprob"],
                    "accepted_answer_scores": baseline_score["accepted_answer_scores"],
                    "generated_answer": baseline_generation["text"],
                    "generated_token_ids": baseline_generation["token_ids"],
                    "official_correctness": baseline_correctness,
                    "cached_greedy_matches_standard_generate": standard_generation_match,
                },
                "layers": [],
            }

            for layer_index in layer_grid:
                context = contexts[layer_index]
                read_cache = ReadInterventionCache()
                state_records: dict[str, dict[str, Any]] = {}
                state_scores_sequence: dict[str, float] = {}
                state_scores_mean: dict[str, float] = {}

                for state_name in ("FULL", "IGNORE", "READ_ONLY", "WRITE_ONLY"):
                    read_mode, write_mode = FOUR_STATES[state_name]
                    result = run_cached_prompt_state(
                        causal_lm,
                        context,
                        baseline.past_key_values,
                        visual_mask,
                        state_name,
                        read_mode,
                        write_mode,
                        read_cache,
                    )
                    state_score = score_accepted_answer_set(
                        causal_lm,
                        processor.tokenizer,
                        result.prompt_logits,
                        result.past_key_values,
                        prompt_inputs["attention_mask"],
                        answers,
                    )
                    generation = greedy_from_prompt(
                        causal_lm,
                        processor.tokenizer,
                        result.prompt_logits,
                        result.past_key_values,
                        prompt_inputs["input_ids"],
                        prompt_inputs["attention_mask"],
                        int(config["max_new_tokens"]),
                    )
                    correctness = score_record(record, generation["text"])
                    state_scores_sequence[state_name] = state_score["sequence_logprob"]
                    state_scores_mean[state_name] = state_score["mean_logprob"]
                    state_records[state_name] = {
                        "sequence_logprob": state_score["sequence_logprob"],
                        "mean_logprob": state_score["mean_logprob"],
                        "accepted_answer_scores": state_score["accepted_answer_scores"],
                        "generated_answer": generation["text"],
                        "generated_token_ids": generation["token_ids"],
                        "official_correctness": correctness,
                        "read_hook_identity_max_abs": result.read_hook_identity_max_abs,
                        "write_hook_identity_max_abs": result.write_hook_identity_max_abs,
                        "prestate_injection_max_abs": result.injected_prestate_max_abs,
                    }

                    if args.mode == "validity":
                        repeat = run_cached_prompt_state(
                            causal_lm,
                            context,
                            baseline.past_key_values,
                            visual_mask,
                            state_name,
                            read_mode,
                            write_mode,
                            read_cache,
                        )
                        repeat_score = score_accepted_answer_set(
                            causal_lm,
                            processor.tokenizer,
                            repeat.prompt_logits,
                            repeat.past_key_values,
                            prompt_inputs["attention_mask"],
                            answers,
                        )
                        no_op_rows.append(
                            {
                                "sample_id": record["id"],
                                "dataset": record["benchmark"],
                                "layer": layer_index,
                                "control": f"repeat_{state_name.lower()}",
                                "prompt_logit_max_abs": max_abs_difference(
                                    repeat.prompt_logits, result.prompt_logits
                                ),
                                "sequence_score_abs": abs(
                                    repeat_score["sequence_logprob"]
                                    - state_score["sequence_logprob"]
                                ),
                                "mean_score_abs": abs(
                                    repeat_score["mean_logprob"] - state_score["mean_logprob"]
                                ),
                            }
                        )
                    if state_name == "FULL":
                        no_op_rows.append(
                            {
                                "sample_id": record["id"],
                                "dataset": record["benchmark"],
                                "layer": layer_index,
                                "control": "instrumented_full",
                                "prompt_logit_max_abs": max_abs_difference(
                                    result.prompt_logits, baseline.logits
                                ),
                                "sequence_score_abs": abs(
                                    state_score["sequence_logprob"]
                                    - baseline_score["sequence_logprob"]
                                ),
                                "mean_score_abs": abs(
                                    state_score["mean_logprob"] - baseline_score["mean_logprob"]
                                ),
                                "generation_match": generation["token_ids"]
                                == baseline_generation["token_ids"],
                            }
                        )
                    del result

                if args.mode == "validity":
                    for control_name, read_mode, write_mode in (
                        ("read_reinsert_identity", "reconstruct", "full"),
                        ("write_reinsert_identity", "full", "reconstruct"),
                    ):
                        identity = run_cached_prompt_state(
                            causal_lm,
                            context,
                            baseline.past_key_values,
                            visual_mask,
                            control_name,
                            read_mode,
                            write_mode,
                            read_cache,
                        )
                        identity_score = score_accepted_answer_set(
                            causal_lm,
                            processor.tokenizer,
                            identity.prompt_logits,
                            identity.past_key_values,
                            prompt_inputs["attention_mask"],
                            answers,
                        )
                        no_op_rows.append(
                            {
                                "sample_id": record["id"],
                                "dataset": record["benchmark"],
                                "layer": layer_index,
                                "control": control_name,
                                "prompt_logit_max_abs": max_abs_difference(
                                    identity.prompt_logits, baseline.logits
                                ),
                                "sequence_score_abs": abs(
                                    identity_score["sequence_logprob"]
                                    - baseline_score["sequence_logprob"]
                                ),
                                "mean_score_abs": abs(
                                    identity_score["mean_logprob"]
                                    - baseline_score["mean_logprob"]
                                ),
                            }
                        )
                        del identity

                layer_result = {
                    "layer": layer_index,
                    "states": state_records,
                    "sequence_effects": factorial_effects(state_scores_sequence),
                    "mean_effects": factorial_effects(state_scores_mean),
                }
                sample_result["layers"].append(layer_result)

            append_jsonl(result_path, sample_result)
            del baseline, contexts, prompt_inputs
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {"completed": sample_index + 1, "total": len(samples), "sample_id": record["id"]}
                ),
                flush=True,
            )

    write_json(output_dir / "token_layout.json", {"samples": layout_rows})
    if args.mode == "validity":
        write_csv(output_dir / "no_op_noise_controls.csv", no_op_rows)
        sequence_noise = np.array([row["sequence_score_abs"] for row in no_op_rows])
        mean_noise = np.array([row["mean_score_abs"] for row in no_op_rows])
        epsilon_sequence = max(
            float(config["noise_floor_sequence"]), float(np.quantile(sequence_noise, 0.99))
        )
        epsilon_mean = max(
            float(config["noise_floor_mean"]), float(np.quantile(mean_noise, 0.99))
        )
        summary = {
            "stage": "B",
            "mode": "validity",
            "sample_count": len(samples),
            "datasets": sorted({row["benchmark"] for row in samples}),
            "layer_grid": layer_grid,
            "checks": {
                "both_datasets_present": {row["benchmark"] for row in samples}
                == {"gqa", "textvqa"},
                "target_alignment": all(
                    token["standalone_answer_matches_combined_suffix"]
                    and token["prompt_positions_contributing_to_score"] == 0
                    for result in read_jsonl(result_path)
                    for token in result["answer_tokenization"]
                ),
                "instrumented_full_parity": all(
                    row["prompt_logit_max_abs"] <= float(config["logit_tolerance"])
                    and row["sequence_score_abs"] <= float(config["score_tolerance"])
                    and row["mean_score_abs"] <= float(config["score_tolerance"])
                    and row.get("generation_match", True)
                    for row in no_op_rows
                    if row["control"] == "instrumented_full"
                ),
                "deterministic_scores": all(
                    row["prompt_logit_max_abs"] == 0.0
                    and row["sequence_score_abs"] == 0.0
                    and row["mean_score_abs"] == 0.0
                    for row in no_op_rows
                    if row["control"].startswith("repeat_")
                ),
                "reinsert_identity": all(
                    row["prompt_logit_max_abs"] <= float(config["logit_tolerance"])
                    and row["sequence_score_abs"] <= float(config["score_tolerance"])
                    and row["mean_score_abs"] <= float(config["score_tolerance"])
                    for row in no_op_rows
                    if row["control"].endswith("reinsert_identity")
                ),
                "token_layout": all(
                    layout["visual_rows"]["contiguous"] for layout in layout_rows
                ),
                "cached_greedy_matches_standard_generate": all(
                    row.get("generation_match", False)
                    for row in no_op_rows
                    if row["control"] == "cached_greedy_vs_standard_generate"
                ),
            },
            "noise": {
                "control_count": len(no_op_rows),
                "sequence_abs_p99": float(np.quantile(sequence_noise, 0.99)),
                "mean_abs_p99": float(np.quantile(mean_noise, 0.99)),
                "epsilon_sequence": epsilon_sequence,
                "epsilon_mean": epsilon_mean,
                "selection_rule": "max(predeclared_floor, empirical_no_op_absolute_difference_p99)",
            },
            "full_sweep_executed": False,
        }
        summary["gate_pass"] = all(summary["checks"].values())
        write_json(output_dir / "stage_b_validity_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if summary["gate_pass"] else 2
    return 0


def main() -> int:
    args = parse_args()
    try:
        return execute(args)
    except Exception as exc:
        config = load_yaml(Path(args.config))
        output_dir = Path(
            args.output_dir
            or (
                config["validity_output_dir"]
                if args.mode == "validity"
                else config["output_dir"]
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "stage_b_failure.json",
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
