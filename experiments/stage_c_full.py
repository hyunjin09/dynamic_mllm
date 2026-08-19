from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor

from tools.research_analysis.v2.stage_c_analysis import uniform_accepted_aggregate
from tools.research_analysis.v2.stage_c_real_donor_amendment import validate_frozen_match_row
from experiments.stage_a_validity import max_abs_difference, prepare_prompt, set_determinism, to_device
from experiments.stage_b_reference_likelihood import (
    capture_prompt_with_cache,
    greedy_from_prompt,
    read_jsonl,
    runtime_metadata,
    score_accepted_answer_set,
)
from experiments.stage_c_entry_gate import derived_seed, load_model
from interventions.prompt_cache import run_cached_prompt_state
from interventions.read_path import ReadInterventionCache
from nulls.structured_read import (
    DonorMetadata,
    FixedGridCovariance,
    donor_matching_ratio,
    generate_covariance_null,
    generate_real_residual_null,
    real_donor_candidates,
    select_real_donors,
)
from scoring.benchmark_metrics import normalize_textvqa, textvqa_consensus
from scoring.contextual_reference_likelihood import (
    ContextualContinuation,
    contextual_continuation,
    score_reference_token_ids_from_prompt,
)
from scoring.reference_likelihood import (
    AcceptedAnswer,
    accepted_answers,
    aggregate_accepted_scores,
    score_reference_from_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the frozen Stage C sweep.")
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "prefix-score-preflight",
            "merge-prefix-preflight",
            "freeze-amended-execution",
            "run",
            "merge",
        ),
    )
    parser.add_argument("--config", default="configs/stage_c.yaml")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def ratio(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        return math.inf
    return max(left / right, right / left)


def accepted_from_manifest(record: dict[str, Any]) -> list[AcceptedAnswer]:
    answers = [
        AcceptedAnswer(str(item["answer"]), float(item["weight"]))
        for item in record["accepted_answers"]
    ]
    if not answers or abs(sum(item.weight for item in answers) - 1.0) > 1e-9:
        raise ValueError(f"Invalid frozen accepted-answer weights for {record['id']}")
    return answers


def frozen_file_checks(config: dict[str, Any]) -> dict[str, str]:
    pairs = {
        "source_plan": (config["source_plan"], config["source_plan_sha256"]),
        "frozen_amendment": (config["frozen_amendment"], config["frozen_amendment_sha256"]),
        "entry_gate_report": (config["entry_gate_report"], config["entry_gate_report_sha256"]),
        "scoring_spec": (config["scoring_spec"], config["scoring_spec_sha256"]),
        "structured_null_spec": (
            config["structured_null_spec"],
            config["structured_null_spec_sha256"],
        ),
        "prefix_tokenization_amendment": (
            config["prefix_tokenization_amendment"],
            config["prefix_tokenization_amendment_sha256"],
        ),
        "real_residual_caliper_amendment": (
            config["real_residual_caliper_amendment"],
            config["real_residual_caliper_amendment_sha256"],
        ),
        "donor_coverage_audit": (
            config["donor_coverage_audit"],
            config["donor_coverage_audit_sha256"],
        ),
        "model_config": (config["model_config"], config["model_config_sha256"]),
        "scoring_implementation": (
            config["scoring_implementation"],
            config["scoring_implementation_sha256"],
        ),
        "normalization_implementation": (
            config["normalization_implementation"],
            config["normalization_implementation_sha256"],
        ),
        "contextual_scoring_implementation": (
            config["contextual_scoring_implementation"],
            config["contextual_scoring_implementation_sha256"],
        ),
        "manifest": (config["manifest_path"], config["manifest_sha256"]),
        "covariance_parameters": (
            config["nulls"]["covariance_parameters"],
            config["nulls"]["covariance_parameters_sha256"],
        ),
        "calibration_residuals": (
            config["nulls"]["calibration_residuals"],
            config["nulls"]["calibration_residuals_sha256"],
        ),
        "donor_index": (
            config["nulls"]["donor_index"],
            config["nulls"]["donor_index_sha256"],
        ),
        "real_donor_match_index": (
            config["nulls"]["real_donor_match_index"],
            config["nulls"]["real_donor_match_index_sha256"],
        ),
        "real_donor_match_index_summary": (
            config["nulls"]["real_donor_match_index_summary"],
            config["nulls"]["real_donor_match_index_summary_sha256"],
        ),
        "deterministic_seeds": (
            config["nulls"]["deterministic_seeds"],
            config["nulls"]["deterministic_seeds_sha256"],
        ),
    }
    observed: dict[str, str] = {}
    for name, (raw_path, expected) in pairs.items():
        digest = file_sha256(Path(raw_path))
        if digest != expected:
            raise RuntimeError(f"Frozen {name} SHA-256 mismatch: {digest} != {expected}")
        observed[name] = digest
    return observed


def load_frozen_nulls(config: dict[str, Any]):
    null_cfg = config["nulls"]
    raw_fit = torch.load(null_cfg["covariance_parameters"], map_location="cpu", weights_only=False)
    fit = FixedGridCovariance(
        mean=raw_fit["mean"],
        basis=raw_fit["basis"],
        eigenvalues=raw_fit["eigenvalues"],
        rank=int(raw_fit["rank"]),
        explained_variance=float(raw_fit["explained_variance"]),
        grid_rows=int(raw_fit["grid_rows"]),
        hidden_size=int(raw_fit["hidden_size"]),
        calibration_samples=int(raw_fit["calibration_samples"]),
        variance_target=float(raw_fit["variance_target"]),
        eigen_shrinkage=float(raw_fit["eigen_shrinkage"]),
    )
    if (
        fit.rank != int(null_cfg["rank"])
        or fit.grid_rows != int(null_cfg["grid_rows"])
        or fit.variance_target != float(null_cfg["variance_target"])
        or fit.eigen_shrinkage != float(null_cfg["eigen_shrinkage"])
    ):
        raise RuntimeError("Frozen covariance parameter fields do not match Stage C config")

    donor_rows = read_jsonl(Path(null_cfg["donor_index"]))
    metadata = [
        DonorMetadata(
            sample_id=str(row["sample_id"]),
            image_id=str(row["image_id"]),
            residual_norm=float(row["residual_norm"]),
            postvisual_rows=int(row["postvisual_rows"]),
            visual_tokens=int(row["visual_tokens"]),
            prompt_tokens=int(row["prompt_tokens"]),
        )
        for row in donor_rows
    ]
    raw_residuals = torch.load(
        null_cfg["calibration_residuals"], map_location="cpu", weights_only=False
    )
    residuals = raw_residuals["residuals"]
    if len(metadata) != 200 or len(residuals) != 200:
        raise RuntimeError("Frozen real-residual pool must contain exactly 200 donors")
    if len({item.sample_id for item in metadata}) != 200 or len({item.image_id for item in metadata}) != 200:
        raise RuntimeError("Frozen donor sample/image IDs must both be unique")
    if raw_residuals["sample_ids"] != [item.sample_id for item in metadata]:
        raise RuntimeError("Frozen donor metadata and residual tensor indices are misaligned")
    return fit, metadata, residuals


def contextual_answer_specs(
    tokenizer,
    record: dict[str, Any],
    config: dict[str, Any],
) -> list[tuple[AcceptedAnswer, ContextualContinuation]]:
    prompt_suffix = str(config["answer_prefix_prompt_suffix"])
    target_prefix = str(config["answer_prefix_target_prefix"])
    literal_prefix = str(config["answer_prefix_robustness_literal"])
    if prompt_suffix + target_prefix != literal_prefix:
        raise RuntimeError("Contextual prefix boundary does not reconstruct the frozen literal")
    prompt_text = str(record["prompt_text"]) + prompt_suffix
    return [
        (
            answer,
            contextual_continuation(
                tokenizer,
                prompt_text=prompt_text,
                continuation_text=target_prefix + answer.text,
                expected_literal_text=(
                    str(record["prompt_text"]) + literal_prefix + answer.text
                ),
            ),
        )
        for answer in accepted_from_manifest(record)
    ]


def score_contextual_accepted_answer_set(
    causal_lm,
    prompt_logits: torch.Tensor,
    prompt_cache,
    prompt_attention_mask: torch.Tensor,
    specs: list[tuple[AcceptedAnswer, ContextualContinuation]],
) -> dict[str, Any]:
    answers = [answer for answer, _ in specs]
    scores = [
        score_reference_token_ids_from_prompt(
            causal_lm,
            prompt_logits,
            prompt_cache,
            prompt_attention_mask,
            continuation.target_token_ids,
        )
        for _, continuation in specs
    ]
    aggregate = aggregate_accepted_scores(answers, scores)
    return {
        **aggregate,
        "accepted_answer_scores": [
            {
                "answer": answer.text,
                "weight": answer.weight,
                "contextual_target_text": continuation.target_text,
                "token_ids": score.token_ids,
                "token_length": len(score.token_ids),
                "token_logprobs": score.token_logprobs,
                "sequence_logprob": score.sequence_logprob,
                "mean_logprob": score.mean_logprob,
            }
            for (answer, continuation), score in zip(specs, scores)
        ],
    }


def preflight(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    observed = frozen_file_checks(config)
    manifest = read_jsonl(Path(config["manifest_path"]))
    if len(manifest) != int(config["sample_count"]):
        raise RuntimeError("Frozen manifest count mismatch")
    if len({row["id"] for row in manifest}) != len(manifest):
        raise RuntimeError("Frozen manifest contains duplicate sample IDs")
    if len({row["image_id"] for row in manifest}) != len(manifest):
        raise RuntimeError("Frozen manifest contains duplicate image IDs")
    fit, metadata, _ = load_frozen_nulls(config)
    seed_rows = read_jsonl(Path(config["nulls"]["deterministic_seeds"]))
    seed_by_id = {row["id"]: row for row in seed_rows}
    if set(seed_by_id) != {row["id"] for row in manifest}:
        raise RuntimeError("Frozen seed rows do not align with the manifest")
    draws = int(config["nulls"]["draws_per_family"])
    for record in manifest:
        expected = [derived_seed(2026080511, record["id"], draw) for draw in range(draws)]
        seed_row = seed_by_id[record["id"]]
        if seed_row["covariance_draw_seeds"] != expected:
            raise RuntimeError(f"Frozen covariance seeds changed for {record['id']}")
        if seed_row["real_donor_tie_break_seed"] != derived_seed(2026080521, record["id"], 0):
            raise RuntimeError(f"Frozen real-donor seed changed for {record['id']}")

    model_config = yaml.safe_load(Path(config["model_config"]).read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(
        model_config["snapshot_path"], local_files_only=True, use_fast=False
    )
    template_sha = hashlib.sha256(processor.chat_template.encode("utf-8")).hexdigest()
    if template_sha != config["chat_template_sha256"]:
        raise RuntimeError("Frozen chat template SHA-256 mismatch")
    prefix_failures: list[dict[str, str]] = []
    component_count = 0
    for record in manifest:
        try:
            specs = contextual_answer_specs(processor.tokenizer, record, config)
            component_count += len(specs)
            for answer, continuation in specs:
                if not (
                    continuation.target_token_count > 0
                    and continuation.prompt_is_combined_prefix
                    and continuation.decoded_text_exact
                    and continuation.prompt_positions_contributing_to_score == 0
                    and continuation.prompt_text
                    == str(record["prompt_text"]) + config["answer_prefix_prompt_suffix"]
                    and continuation.target_text
                    == config["answer_prefix_target_prefix"] + answer.text
                ):
                    prefix_failures.append({"id": record["id"], "answer": answer.text})
        except ValueError as exc:
            prefix_failures.append({"id": record["id"], "error": str(exc)})
    if prefix_failures:
        raise RuntimeError(
            f"Amended contextual-prefix span check failed for {len(prefix_failures)} records/components"
        )

    static_preflight = {
        "schema_version": "stage_c_prefix_static_preflight_v1",
        "outcome_blind": True,
        "stage_c_primary_endpoint_computed": False,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "frozen_input_sha256": observed,
        "manifest_count": len(manifest),
        "unique_image_count": len({row["image_id"] for row in manifest}),
        "accepted_answer_component_count": component_count,
        "covariance_rank": fit.rank,
        "covariance_grid_rows": fit.grid_rows,
        "real_donor_count": len(metadata),
        "real_donor_matching_ratio_cap": config["nulls"]["real_donor_matching_ratio_cap"],
        "draws_per_family": draws,
        "prefix_span_failure_count": 0,
        "literal_prefix": config["answer_prefix_robustness_literal"],
        "prompt_suffix": config["answer_prefix_prompt_suffix"],
        "target_prefix": config["answer_prefix_target_prefix"],
        "deterministic_score_reproduction": "pending_gpu_preflight",
        "all_choices_frozen_before_stage_c_outcomes": True,
    }
    path = Path(config["output"]["prefix_static_preflight_path"])
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != static_preflight:
        raise RuntimeError("Existing contextual-prefix static preflight differs from current inputs")
    write_json(path, static_preflight)
    print(json.dumps(static_preflight, indent=2), flush=True)
    return 0


def prepare_prefixed_prompt(
    processor, record: dict[str, Any], device: torch.device, prompt_suffix: str
) -> tuple[str, dict[str, Any]]:
    image = Image.open(record["local_image_path"]).convert("RGB")
    prompt_text = str(record["prompt_text"]) + prompt_suffix
    batch = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    return prompt_text, to_device(dict(batch), device)


def prefix_score_preflight_shard(
    config_path: Path,
    shard_index: int,
    num_shards: int,
    resume: bool,
) -> int:
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("Shard index must lie in [0, num_shards)")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen_file_checks(config)
    static_path = Path(config["output"]["prefix_static_preflight_path"])
    static = json.loads(static_path.read_text(encoding="utf-8"))
    if static["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("Stage C config differs from the contextual-prefix static preflight")
    set_determinism(int(config["bootstrap"]["primary_seed"]))
    manifest = read_jsonl(Path(config["manifest_path"]))
    samples = [row for index, row in enumerate(manifest) if index % num_shards == shard_index]
    model, processor, device, model_config = load_model(config)
    shard_dir = (
        Path(config["output"]["prefix_preflight_shard_dir"])
        / f"shard_{shard_index:02d}"
    )
    result_path = shard_dir / "results.jsonl"
    runtime_path = shard_dir / "runtime.json"
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed = set()
    if result_path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {result_path} without --resume")
        completed = {row["id"] for row in read_jsonl(result_path)}
    write_json(
        runtime_path,
        {
            **runtime_metadata(model, processor, model_config, [0], int(config["max_new_tokens"])),
            "schema_version": "stage_c_prefix_score_preflight_runtime_v1",
            "config_sha256": file_sha256(config_path),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "primary_endpoint_computed": False,
        },
    )
    tolerance = float(config["full_parity_score_tolerance"])
    prompt_suffix = str(config["answer_prefix_prompt_suffix"])
    with torch.inference_mode():
        for local_index, record in enumerate(samples):
            if record["id"] in completed:
                continue
            prompt_text, inputs = prepare_prefixed_prompt(
                processor, record, device, prompt_suffix
            )
            if prompt_text != str(record["prompt_text"]) + prompt_suffix:
                raise RuntimeError(f"Contextual prefix prompt changed for {record['id']}")
            specs = contextual_answer_specs(processor.tokenizer, record, config)
            prompt_state, _ = capture_prompt_with_cache(model, inputs, [])
            first = score_contextual_accepted_answer_set(
                model,
                prompt_state.logits,
                prompt_state.past_key_values,
                inputs["attention_mask"],
                specs,
            )
            second = score_contextual_accepted_answer_set(
                model,
                prompt_state.logits,
                prompt_state.past_key_values,
                inputs["attention_mask"],
                specs,
            )
            component_differences = []
            for first_component, second_component in zip(
                first["accepted_answer_scores"], second["accepted_answer_scores"]
            ):
                if first_component["token_ids"] != second_component["token_ids"]:
                    raise RuntimeError(
                        f"Contextual target IDs changed across repeats for {record['id']}"
                    )
                component_differences.append(
                    max(
                        abs(
                            float(left) - float(right)
                        )
                        for left, right in zip(
                            first_component["token_logprobs"],
                            second_component["token_logprobs"],
                        )
                    )
                )
            max_component_difference = max(component_differences)
            sequence_difference = abs(
                float(first["sequence_logprob"]) - float(second["sequence_logprob"])
            )
            mean_difference = abs(
                float(first["mean_logprob"]) - float(second["mean_logprob"])
            )
            passed = (
                max_component_difference <= tolerance
                and sequence_difference <= tolerance
                and mean_difference <= tolerance
            )
            if not passed:
                raise RuntimeError(
                    f"Contextual deterministic score reproduction failed for {record['id']}"
                )
            append_jsonl(
                result_path,
                {
                    "schema_version": "stage_c_prefix_score_preflight_record_v1",
                    "id": record["id"],
                    "image_id": record["image_id"],
                    "accepted_answer_component_count": len(specs),
                    "target_spans_nonempty": all(
                        continuation.target_token_count > 0 for _, continuation in specs
                    ),
                    "decoded_literal_text_exact": all(
                        continuation.decoded_text_exact for _, continuation in specs
                    ),
                    "prompt_positions_contributing_to_score": 0,
                    "max_repeated_token_logprob_abs": max_component_difference,
                    "repeated_sequence_score_abs": sequence_difference,
                    "repeated_mean_score_abs": mean_difference,
                    "pass": True,
                    "primary_endpoint_computed": False,
                },
            )
            del prompt_state, inputs
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "completed": local_index + 1,
                        "shard_total": len(samples),
                        "sample_id": record["id"],
                    }
                ),
                flush=True,
            )
    return 0


def merge_prefix_score_preflight(config_path: Path, num_shards: int) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    observed = frozen_file_checks(config)
    manifest = read_jsonl(Path(config["manifest_path"]))
    static_path = Path(config["output"]["prefix_static_preflight_path"])
    static = json.loads(static_path.read_text(encoding="utf-8"))
    rows_by_id: dict[str, dict[str, Any]] = {}
    shard_dir = Path(config["output"]["prefix_preflight_shard_dir"])
    runtime_rows = []
    for shard_index in range(num_shards):
        results_path = shard_dir / f"shard_{shard_index:02d}" / "results.jsonl"
        runtime_path = shard_dir / f"shard_{shard_index:02d}" / "runtime.json"
        rows = read_jsonl(results_path)
        expected_ids = {
            row["id"]
            for index, row in enumerate(manifest)
            if index % num_shards == shard_index
        }
        if {row["id"] for row in rows} != expected_ids or len(rows) != len(expected_ids):
            raise RuntimeError(f"Contextual-prefix preflight shard {shard_index} is incomplete")
        if any(not row["pass"] or row["primary_endpoint_computed"] for row in rows):
            raise RuntimeError(f"Contextual-prefix preflight shard {shard_index} failed")
        for row in rows:
            if row["id"] in rows_by_id:
                raise RuntimeError(f"Duplicate contextual-prefix preflight ID: {row['id']}")
            rows_by_id[row["id"]] = row
        runtime_rows.append(
            {
                "path": str(runtime_path),
                "sha256": file_sha256(runtime_path),
            }
        )
    ordered = [rows_by_id[row["id"]] for row in manifest]
    component_count = sum(row["accepted_answer_component_count"] for row in ordered)
    if len(ordered) != int(config["sample_count"]) or component_count != int(
        static["accepted_answer_component_count"]
    ):
        raise RuntimeError("Contextual-prefix preflight coverage does not match static freeze")
    summary = {
        "schema_version": "stage_c_prefix_score_preflight_summary_v1",
        "outcome_blind": True,
        "primary_endpoint_computed": False,
        "config_sha256": file_sha256(config_path),
        "static_preflight_sha256": file_sha256(static_path),
        "manifest_count": len(ordered),
        "accepted_answer_component_count": component_count,
        "all_target_spans_nonempty": all(row["target_spans_nonempty"] for row in ordered),
        "all_decoded_literal_text_exact": all(
            row["decoded_literal_text_exact"] for row in ordered
        ),
        "prompt_positions_contributing_to_score": 0,
        "max_repeated_token_logprob_abs": max(
            row["max_repeated_token_logprob_abs"] for row in ordered
        ),
        "max_repeated_sequence_score_abs": max(
            row["repeated_sequence_score_abs"] for row in ordered
        ),
        "max_repeated_mean_score_abs": max(
            row["repeated_mean_score_abs"] for row in ordered
        ),
        "all_records_pass": True,
        "shard_runtimes": runtime_rows,
    }
    summary_path = Path(config["output"]["prefix_preflight_summary_path"])
    write_json(summary_path, summary)
    fit, metadata, _ = load_frozen_nulls(config)
    execution_freeze = {
        "schema_version": "stage_c_execution_freeze_v1",
        "outcome_blind": True,
        "stage_c_primary_endpoint_computed": False,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "frozen_input_sha256": observed,
        "manifest_count": len(manifest),
        "unique_image_count": len({row["image_id"] for row in manifest}),
        "covariance_rank": fit.rank,
        "covariance_grid_rows": fit.grid_rows,
        "real_donor_count": len(metadata),
        "real_donor_matching_ratio_cap": config["nulls"]["real_donor_matching_ratio_cap"],
        "draws_per_family": int(config["nulls"]["draws_per_family"]),
        "prefix_span_failure_count": 0,
        "prefix_deterministic_score_failure_count": 0,
        "prefix_preflight_summary_sha256": file_sha256(summary_path),
        "all_choices_frozen_before_outcomes": True,
    }
    freeze_path = Path(config["output"]["execution_freeze_path"])
    if freeze_path.exists() and json.loads(freeze_path.read_text(encoding="utf-8")) != execution_freeze:
        raise RuntimeError("Existing Stage C execution freeze differs from amended preflight")
    write_json(freeze_path, execution_freeze)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def freeze_amended_execution(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    observed = frozen_file_checks(config)
    manifest = read_jsonl(Path(config["manifest_path"]))
    prefix_path = Path(config["output"]["prefix_preflight_summary_path"])
    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    match_rows = read_jsonl(Path(config["nulls"]["real_donor_match_index"]))
    match_summary = json.loads(
        Path(config["nulls"]["real_donor_match_index_summary"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        len(manifest) != 800
        or len({row["id"] for row in manifest}) != 800
        or len({row["image_id"] for row in manifest}) != 800
        or not prefix["all_records_pass"]
        or prefix["primary_endpoint_computed"]
        or len(match_rows) != 800
        or [row["id"] for row in match_rows] != [row["id"] for row in manifest]
        or match_summary["original_supported_target_count"] != 798
        or match_summary["amended_supported_target_count"] != 800
        or match_summary["selection_changed_for_original_supported_count"] != 0
        or set(match_summary["wider_caliper_target_ids"])
        != {
            "textvqa:textvqa_validation_39543",
            "textvqa:textvqa_validation_36174",
        }
        or match_summary["likelihood_or_intervention_outcome_used"]
        or match_summary["partial_stage_c_results_loaded"]
    ):
        raise RuntimeError("Approved amended execution inputs failed their freeze gate")
    freeze = {
        "schema_version": "stage_c_execution_freeze_v2",
        "outcome_blind_at_amendment": True,
        "stage_c_primary_endpoint_computed_or_inspected_before_amendment": False,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "frozen_input_sha256": observed,
        "manifest_count": 800,
        "unique_image_count": 800,
        "original_caliper": float(config["nulls"]["real_donor_original_caliper"]),
        "amended_caliper": float(config["nulls"]["real_donor_matching_ratio_cap"]),
        "donors_per_target": int(config["nulls"]["draws_per_family"]),
        "original_supported_target_count": 798,
        "amended_supported_target_count": 800,
        "wider_caliper_target_ids": match_summary["wider_caliper_target_ids"],
        "original_supported_selection_changed_count": 0,
        "real_donor_match_index_sha256": file_sha256(
            Path(config["nulls"]["real_donor_match_index"])
        ),
        "prefix_preflight_reused_unaffected": True,
        "prefix_preflight_summary_sha256": file_sha256(prefix_path),
        "prior_partial_stage_c_records_loaded": False,
        "prior_partial_stage_c_records_reused": False,
        "fresh_shard_directory": config["output"]["shard_dir"],
        "all_800_records_require_fresh_execution": True,
        "secondary_798_real_residual_sensitivity_frozen": True,
        "stage_d_authorized": False,
    }
    if freeze["amended_caliper"] != 1.5833333333333333:
        raise RuntimeError("Approved amended caliper is not exact 19/12")
    freeze_path = Path(config["output"]["execution_freeze_path"])
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to overwrite amended execution freeze: {freeze_path}")
    write_json(freeze_path, freeze)
    print(json.dumps(freeze, indent=2), flush=True)
    return 0


def state_payload(score: dict[str, Any], generation: dict[str, Any], raw_answers: list[str]) -> dict[str, Any]:
    correctness = textvqa_consensus(generation["text"], raw_answers)
    return {
        "sequence_logprob": score["sequence_logprob"],
        "mean_logprob": score["mean_logprob"],
        "accepted_answer_scores": score["accepted_answer_scores"],
        "generated_answer": generation["text"],
        "normalized_generated_answer": normalize_textvqa(generation["text"]),
        "generated_token_ids": generation["token_ids"],
        "official_correctness": correctness,
        "strictly_correct": correctness >= 1.0,
    }


def score_wrong_answer(model, tokenizer, state, attention_mask, answer: str) -> dict[str, Any]:
    score = score_reference_from_prompt(
        model,
        tokenizer,
        state.prompt_logits,
        state.past_key_values,
        attention_mask,
        answer,
    )
    return {
        "answer": answer,
        "token_ids": score.token_ids,
        "token_logprobs": score.token_logprobs,
        "sequence_logprob": score.sequence_logprob,
        "mean_logprob": score.mean_logprob,
    }


def validate_official_record(record: dict[str, Any], source: dict[str, Any]) -> list[str]:
    if int(source["question_id"]) != int(record["question_id"]):
        raise RuntimeError(f"Official question ID mismatch for {record['id']}")
    if str(source["image_id"]) != str(record["image_id"]):
        raise RuntimeError(f"Official image ID mismatch for {record['id']}")
    if str(source["question"]).strip() != str(record["question"]).strip():
        raise RuntimeError(f"Official question text mismatch for {record['id']}")
    raw_answers = [str(value) for value in source["answers"]]
    computed = accepted_answers(
        {"benchmark": "textvqa", "answer": raw_answers[0], "all_answer_norms": raw_answers}
    )
    frozen = accepted_from_manifest(record)
    if len(computed) != len(frozen) or any(
        left.text != right.text or abs(left.weight - right.weight) > 1e-9
        for left, right in zip(computed, frozen)
    ):
        raise RuntimeError(f"Official accepted answers changed for {record['id']}")
    return raw_answers


def run_shard(config_path: Path, shard_index: int, num_shards: int, resume: bool) -> int:
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("Shard index must lie in [0, num_shards)")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen_file_checks(config)
    freeze = json.loads(Path(config["output"]["execution_freeze_path"]).read_text(encoding="utf-8"))
    if freeze["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("Stage C config differs from the outcome-blind execution freeze")
    if freeze.get(
        "stage_c_primary_endpoint_computed",
        freeze.get("stage_c_primary_endpoint_computed_or_inspected_before_amendment", True),
    ):
        raise RuntimeError("Stage C endpoint was opened before the amended execution freeze")
    prefix_preflight_path = Path(config["output"]["prefix_preflight_summary_path"])
    prefix_preflight = json.loads(prefix_preflight_path.read_text(encoding="utf-8"))
    if (
        not prefix_preflight["all_records_pass"]
        or prefix_preflight["primary_endpoint_computed"]
        or freeze["prefix_preflight_summary_sha256"] != file_sha256(prefix_preflight_path)
    ):
        raise RuntimeError("Amended contextual-prefix preflight is absent or invalid")
    set_determinism(int(config["bootstrap"]["primary_seed"]))
    manifest = read_jsonl(Path(config["manifest_path"]))
    samples = [row for index, row in enumerate(manifest) if index % num_shards == shard_index]
    seed_by_id = {
        row["id"]: row for row in read_jsonl(Path(config["nulls"]["deterministic_seeds"]))
    }
    match_by_id = {
        row["id"]: row
        for row in read_jsonl(Path(config["nulls"]["real_donor_match_index"]))
    }
    if set(match_by_id) != {row["id"] for row in manifest} or len(match_by_id) != 800:
        raise RuntimeError("Frozen amended donor index does not align with the manifest")
    fit, donor_metadata, donor_residuals = load_frozen_nulls(config)
    donor_position = {item.sample_id: index for index, item in enumerate(donor_metadata)}

    source_dataset = load_dataset(
        config["dataset_id"],
        revision=config["dataset_revision"],
        split=config["dataset_split"],
        cache_dir=config["dataset_cache"],
    ).remove_columns("image")
    model, processor, device, model_config = load_model(config)
    shard_dir = Path(config["output"]["shard_dir"]) / f"shard_{shard_index:02d}"
    result_path = shard_dir / "results.jsonl"
    runtime_path = shard_dir / "runtime.json"
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed = set()
    if result_path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {result_path} without --resume")
        completed = {row["id"] for row in read_jsonl(result_path)}
    write_json(
        runtime_path,
        {
            **runtime_metadata(model, processor, model_config, [0], int(config["max_new_tokens"])),
            "schema_version": "stage_c_runtime_shard_v1",
            "config_sha256": file_sha256(config_path),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "dataset_id": config["dataset_id"],
            "dataset_revision": config["dataset_revision"],
        },
    )

    layer_index = int(config["primary_layer"])
    draws = int(config["nulls"]["draws_per_family"])
    cap = float(config["nulls"]["real_donor_matching_ratio_cap"])
    max_new_tokens = int(config["max_new_tokens"])
    prefix_prompt_suffix = str(config["answer_prefix_prompt_suffix"])
    with torch.inference_mode():
        for local_index, record in enumerate(samples):
            if record["id"] in completed:
                continue
            raw_answers = validate_official_record(
                record, source_dataset[int(record["source_dataset_index"])]
            )
            answers = accepted_from_manifest(record)
            prompt_text, inputs = prepare_prompt(processor, record, device)
            if prompt_text != record["prompt_text"]:
                raise RuntimeError(f"Frozen primary prompt changed for {record['id']}")
            input_ids = inputs["input_ids"]
            visual_mask = input_ids == model.config.image_token_id
            if int(input_ids.shape[1]) != int(record["prompt_token_length"]):
                raise RuntimeError(f"Prompt length changed for {record['id']}")
            if int(visual_mask.sum().item()) != int(record["image_token_count"]):
                raise RuntimeError(f"Image-token count changed for {record['id']}")

            baseline, contexts = capture_prompt_with_cache(model, inputs, [layer_index])
            baseline_score = score_accepted_answer_set(
                model, processor.tokenizer, baseline.logits, baseline.past_key_values,
                inputs["attention_mask"], answers,
            )
            baseline_generation = greedy_from_prompt(
                model, processor.tokenizer, baseline.logits, baseline.past_key_values,
                input_ids, inputs["attention_mask"], max_new_tokens,
            )
            context = contexts[layer_index]
            full_state = run_cached_prompt_state(
                model, context, baseline.past_key_values, visual_mask,
                "FULL", "full", "full",
            )
            full_score = score_accepted_answer_set(
                model, processor.tokenizer, full_state.prompt_logits,
                full_state.past_key_values, inputs["attention_mask"], answers,
            )
            full_generation = greedy_from_prompt(
                model, processor.tokenizer, full_state.prompt_logits,
                full_state.past_key_values, input_ids, inputs["attention_mask"], max_new_tokens,
            )
            read_cache = ReadInterventionCache()
            write_only_state = run_cached_prompt_state(
                model, context, baseline.past_key_values, visual_mask,
                "WRITE_ONLY", "off", "full", read_cache,
            )
            write_only_score = score_accepted_answer_set(
                model, processor.tokenizer, write_only_state.prompt_logits,
                write_only_state.past_key_values, inputs["attention_mask"], answers,
            )
            write_only_generation = greedy_from_prompt(
                model, processor.tokenizer, write_only_state.prompt_logits,
                write_only_state.past_key_values, input_ids, inputs["attention_mask"], max_new_tokens,
            )

            parity = {
                "prompt_logit_max_abs": max_abs_difference(full_state.prompt_logits, baseline.logits),
                "sequence_score_abs": abs(full_score["sequence_logprob"] - baseline_score["sequence_logprob"]),
                "mean_score_abs": abs(full_score["mean_logprob"] - baseline_score["mean_logprob"]),
                "generation_token_ids_match": full_generation["token_ids"] == baseline_generation["token_ids"],
            }
            if not (
                parity["prompt_logit_max_abs"] <= float(config["full_parity_logit_tolerance"])
                and parity["sequence_score_abs"] <= float(config["full_parity_score_tolerance"])
                and parity["mean_score_abs"] <= float(config["full_parity_score_tolerance"])
                and parity["generation_token_ids_match"]
            ):
                raise RuntimeError(f"Frozen FULL parity failed for {record['id']}: {parity}")

            if read_cache.actual_output is None or read_cache.off_output is None:
                raise RuntimeError(f"READ cache incomplete for {record['id']}")
            actual_delta = read_cache.actual_output.float() - read_cache.off_output.float()
            visual_indices = torch.where(visual_mask[0])[0]
            post_start = int(visual_indices[-1].item()) + 1
            target_residual = actual_delta[0, post_start:].detach().cpu()
            target_norm = float(target_residual.norm().item())
            target_meta = DonorMetadata(
                sample_id=str(record["id"]), image_id=str(record["image_id"]),
                residual_norm=target_norm, postvisual_rows=int(target_residual.shape[0]),
                visual_tokens=int(visual_mask.sum().item()), prompt_tokens=int(input_ids.shape[1]),
            )
            seed_row = seed_by_id[record["id"]]
            covariance_draws: list[dict[str, Any]] = []
            for draw_index, seed in enumerate(seed_row["covariance_draw_seeds"]):
                rows = generate_covariance_null(fit, target_residual.shape[0], target_norm, int(seed))
                replacement = torch.zeros_like(actual_delta)
                replacement[0, post_start:] = rows.to(device)
                null_state = run_cached_prompt_state(
                    model, context, baseline.past_key_values, visual_mask,
                    f"COVARIANCE_NULL_{draw_index}", "replace", "full", read_cache,
                    read_replacement_delta=replacement,
                )
                null_score = score_accepted_answer_set(
                    model, processor.tokenizer, null_state.prompt_logits,
                    null_state.past_key_values, inputs["attention_mask"], answers,
                )
                covariance_draws.append({
                    "draw": draw_index,
                    "seed": int(seed),
                    "sequence_logprob": null_score["sequence_logprob"],
                    "mean_logprob": null_score["mean_logprob"],
                    "sequence_effect": null_score["sequence_logprob"] - write_only_score["sequence_logprob"],
                    "mean_effect": null_score["mean_logprob"] - write_only_score["mean_logprob"],
                    "norm_relative_error": abs(float(rows.norm().item()) - target_norm) / max(target_norm, 1e-12),
                })
                del null_state, replacement

            match_row = match_by_id[record["id"]]
            if (
                match_row["manifest_record_sha256"] != record["record_sha256"]
                or int(match_row["layer"]) != layer_index
                or match_row["hook"]
                != "decoder.layer.0.self_attn.output.postvisual_nonvisual_rows"
            ):
                raise RuntimeError(f"Frozen amended donor row changed for {record['id']}")
            eligible_donors = real_donor_candidates(
                target_meta, donor_metadata,
                seed=int(seed_row["real_donor_tie_break_seed"]), matching_ratio_cap=cap,
            )
            chosen = validate_frozen_match_row(
                target_meta,
                donor_metadata,
                match_row,
                draws=draws,
                seed=int(seed_row["real_donor_tie_break_seed"]),
                amended_caliper=cap,
            )
            real_draws: list[dict[str, Any]] = []
            for draw_index, donor in enumerate(chosen):
                rows = generate_real_residual_null(
                    donor_residuals[donor_position[donor.sample_id]],
                    target_residual.shape[0], target_norm,
                )
                replacement = torch.zeros_like(actual_delta)
                replacement[0, post_start:] = rows.to(device)
                null_state = run_cached_prompt_state(
                    model, context, baseline.past_key_values, visual_mask,
                    f"REAL_RESIDUAL_NULL_{draw_index}", "replace", "full", read_cache,
                    read_replacement_delta=replacement,
                )
                null_score = score_accepted_answer_set(
                    model, processor.tokenizer, null_state.prompt_logits,
                    null_state.past_key_values, inputs["attention_mask"], answers,
                )
                real_draws.append({
                    "draw": draw_index,
                    "donor_id": donor.sample_id,
                    "donor_image_id": donor.image_id,
                    "matching_ratio": donor_matching_ratio(target_meta, donor),
                    "norm_ratio": ratio(target_meta.residual_norm, donor.residual_norm),
                    "postvisual_row_ratio": ratio(target_meta.postvisual_rows, donor.postvisual_rows),
                    "visual_token_ratio": ratio(target_meta.visual_tokens, donor.visual_tokens),
                    "sequence_logprob": null_score["sequence_logprob"],
                    "mean_logprob": null_score["mean_logprob"],
                    "sequence_effect": null_score["sequence_logprob"] - write_only_score["sequence_logprob"],
                    "mean_effect": null_score["mean_logprob"] - write_only_score["mean_logprob"],
                    "norm_relative_error": abs(float(rows.norm().item()) - target_norm) / max(target_norm, 1e-12),
                })
                del null_state, replacement

            prefixed_text, prefixed_inputs = prepare_prefixed_prompt(
                processor, record, device, prefix_prompt_suffix
            )
            contextual_specs = contextual_answer_specs(
                processor.tokenizer, record, config
            )
            prefixed_baseline, prefixed_contexts = capture_prompt_with_cache(
                model, prefixed_inputs, [layer_index]
            )
            prefixed_full = run_cached_prompt_state(
                model, prefixed_contexts[layer_index], prefixed_baseline.past_key_values,
                prefixed_inputs["input_ids"] == model.config.image_token_id,
                "FULL_PREFIX", "full", "full",
            )
            prefixed_read_cache = ReadInterventionCache()
            prefixed_write_only = run_cached_prompt_state(
                model, prefixed_contexts[layer_index], prefixed_baseline.past_key_values,
                prefixed_inputs["input_ids"] == model.config.image_token_id,
                "WRITE_ONLY_PREFIX", "off", "full", prefixed_read_cache,
            )
            prefixed_full_score = score_contextual_accepted_answer_set(
                model, prefixed_full.prompt_logits,
                prefixed_full.past_key_values, prefixed_inputs["attention_mask"],
                contextual_specs,
            )
            prefixed_write_only_score = score_contextual_accepted_answer_set(
                model, prefixed_write_only.prompt_logits,
                prefixed_write_only.past_key_values, prefixed_inputs["attention_mask"],
                contextual_specs,
            )

            full_payload = state_payload(full_score, full_generation, raw_answers)
            write_only_payload = state_payload(write_only_score, write_only_generation, raw_answers)
            full_wrong = not full_payload["strictly_correct"]
            wrong_string = full_payload["normalized_generated_answer"]
            wrong_contrast: dict[str, Any] = {
                "eligible": bool(full_wrong and wrong_string),
                "ineligible_reason": None,
            }
            if not full_wrong:
                wrong_contrast["ineligible_reason"] = "full_strictly_correct"
            elif not wrong_string:
                wrong_contrast["ineligible_reason"] = "normalized_full_greedy_answer_empty"
            else:
                wrong_full = score_wrong_answer(
                    model, processor.tokenizer, full_state, inputs["attention_mask"], wrong_string
                )
                wrong_write_only = score_wrong_answer(
                    model, processor.tokenizer, write_only_state,
                    inputs["attention_mask"], wrong_string,
                )
                c_full_mean = full_score["mean_logprob"] - wrong_full["mean_logprob"]
                c_wo_mean = write_only_score["mean_logprob"] - wrong_write_only["mean_logprob"]
                c_full_sequence = full_score["sequence_logprob"] - wrong_full["sequence_logprob"]
                c_wo_sequence = write_only_score["sequence_logprob"] - wrong_write_only["sequence_logprob"]
                wrong_contrast.update({
                    "frozen_full_greedy_wrong_answer": wrong_string,
                    "full_wrong_answer_score": wrong_full,
                    "write_only_wrong_answer_score": wrong_write_only,
                    "reference_minus_wrong_full_mean": c_full_mean,
                    "reference_minus_wrong_write_only_mean": c_wo_mean,
                    "delta_c_mean": c_wo_mean - c_full_mean,
                    "reference_minus_wrong_full_sequence": c_full_sequence,
                    "reference_minus_wrong_write_only_sequence": c_wo_sequence,
                    "delta_c_sequence": c_wo_sequence - c_full_sequence,
                })

            uniform_full = uniform_accepted_aggregate(full_score["accepted_answer_scores"])
            uniform_write_only = uniform_accepted_aggregate(
                write_only_score["accepted_answer_scores"]
            )
            result = {
                "schema_version": "stage_c_reference_likelihood_v1",
                "id": record["id"],
                "dataset": "textvqa",
                "image_id": record["image_id"],
                "question_id": record["question_id"],
                "question": record["question"],
                "manifest_record_sha256": record["record_sha256"],
                "prompt_text": prompt_text,
                "prompt_token_length": int(input_ids.shape[1]),
                "image_token_count": int(visual_mask.sum().item()),
                "answer_length_weighted_tokens": sum(
                    float(answer["weight"]) * int(tokenization["answer_token_length"])
                    for answer, tokenization in zip(
                        record["accepted_answers"], record["accepted_answer_tokenization"]
                    )
                ),
                "accepted_answers": record["accepted_answers"],
                "answer_tokenization": record["accepted_answer_tokenization"],
                "integrity": parity,
                "states": {"FULL": full_payload, "WRITE_ONLY": write_only_payload},
                "effects": {
                    "mean": full_score["mean_logprob"] - write_only_score["mean_logprob"],
                    "sequence": full_score["sequence_logprob"] - write_only_score["sequence_logprob"],
                },
                "uniform_aggregation_robustness": {
                    "full": uniform_full,
                    "write_only": uniform_write_only,
                    "mean_effect": uniform_full["mean_logprob"] - uniform_write_only["mean_logprob"],
                    "sequence_effect": uniform_full["sequence_logprob"] - uniform_write_only["sequence_logprob"],
                },
                "answer_prefix_robustness": {
                    "literal_prefix": config["answer_prefix_robustness_literal"],
                    "prompt_suffix": prefix_prompt_suffix,
                    "target_prefix": config["answer_prefix_target_prefix"],
                    "subset": config["answer_prefix_subset"],
                    "prompt_text": prefixed_text,
                    "contextual_target_ids_identical_across_states": True,
                    "cross_prefix_raw_per_token_levels_comparable": False,
                    "full": prefixed_full_score,
                    "write_only": prefixed_write_only_score,
                    "mean_effect": prefixed_full_score["mean_logprob"] - prefixed_write_only_score["mean_logprob"],
                    "sequence_effect": prefixed_full_score["sequence_logprob"] - prefixed_write_only_score["sequence_logprob"],
                    "full_parity_logit_max_abs": max_abs_difference(
                        prefixed_full.prompt_logits, prefixed_baseline.logits
                    ),
                },
                "read_residual": {
                    "shape": list(target_residual.shape),
                    "frobenius_norm": target_norm,
                    "eligible_real_donor_count": len(eligible_donors),
                    "original_caliper": float(config["nulls"]["real_donor_original_caliper"]),
                    "amended_caliper": cap,
                    "original_eligible_real_donor_count": int(
                        match_row["original_eligible_donor_count"]
                    ),
                    "original_caliper_supplies_eight": bool(
                        match_row["original_caliper_supplies_eight"]
                    ),
                    "amended_match_index_record_schema": match_row["schema_version"],
                },
                "structured_nulls": {
                    "covariance": covariance_draws,
                    "real_residual": real_draws,
                },
                "wrong_answer_contrast": wrong_contrast,
            }
            append_jsonl(result_path, result)
            del (
                baseline, contexts, full_state, write_only_state, prefixed_baseline,
                prefixed_contexts, prefixed_full, prefixed_write_only, inputs, prefixed_inputs,
            )
            torch.cuda.empty_cache()
            print(json.dumps({
                "completed": local_index + 1,
                "shard_total": len(samples),
                "sample_id": record["id"],
            }), flush=True)
    return 0


def merge_shards(config_path: Path, num_shards: int) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen_file_checks(config)
    manifest = read_jsonl(Path(config["manifest_path"]))
    match_by_id = {
        row["id"]: row
        for row in read_jsonl(Path(config["nulls"]["real_donor_match_index"]))
    }
    result_by_id: dict[str, dict[str, Any]] = {}
    runtime_rows = []
    shard_dir = Path(config["output"]["shard_dir"])
    for shard_index in range(num_shards):
        path = shard_dir / f"shard_{shard_index:02d}" / "results.jsonl"
        runtime_path = shard_dir / f"shard_{shard_index:02d}" / "runtime.json"
        rows = read_jsonl(path)
        expected_ids = {
            row["id"] for index, row in enumerate(manifest) if index % num_shards == shard_index
        }
        if {row["id"] for row in rows} != expected_ids or len(rows) != len(expected_ids):
            raise RuntimeError(f"Shard {shard_index} is incomplete or contains duplicate IDs")
        for row in rows:
            if row["id"] in result_by_id:
                raise RuntimeError(f"Duplicate Stage C result ID: {row['id']}")
            result_by_id[row["id"]] = row
        runtime_rows.append({
            "path": str(runtime_path),
            "sha256": file_sha256(runtime_path),
            "runtime": json.loads(runtime_path.read_text(encoding="utf-8")),
        })
    if len(result_by_id) != int(config["sample_count"]):
        raise RuntimeError("Merged Stage C results do not contain exactly 800 records")
    ordered = [result_by_id[row["id"]] for row in manifest]
    if any(
        row["manifest_record_sha256"] != manifest[index]["record_sha256"]
        or row["image_id"] != manifest[index]["image_id"]
        or not row["integrity"]["generation_token_ids_match"]
        or [
            draw["donor_id"] for draw in row["structured_nulls"]["real_residual"]
        ]
        != [
            donor["sample_id"] for donor in match_by_id[row["id"]]["selected_donors"]
        ]
        for index, row in enumerate(ordered)
    ):
        raise RuntimeError("Merged Stage C results failed manifest or FULL-parity integrity")
    output_path = Path(config["output"]["results_path"])
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite merged results: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(
        Path(config["output"]["runtime_path"]),
        {
            "schema_version": "stage_c_runtime_v1",
            "config_sha256": file_sha256(config_path),
            "num_shards": num_shards,
            "result_count": len(ordered),
            "shard_runtimes": runtime_rows,
        },
    )
    print(json.dumps({
        "result_count": len(ordered),
        "results_path": str(output_path),
        "results_sha256": file_sha256(output_path),
        "scientific_aggregation_performed": False,
    }, indent=2), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    try:
        if args.command == "preflight":
            return preflight(config_path)
        if args.command == "prefix-score-preflight":
            if args.shard_index is None:
                raise ValueError("prefix-score-preflight requires --shard-index")
            return prefix_score_preflight_shard(
                config_path,
                args.shard_index,
                args.num_shards,
                args.resume,
            )
        if args.command == "merge-prefix-preflight":
            return merge_prefix_score_preflight(config_path, args.num_shards)
        if args.command == "freeze-amended-execution":
            return freeze_amended_execution(config_path)
        if args.command == "merge":
            return merge_shards(config_path, args.num_shards)
        if args.shard_index is None:
            raise ValueError("run requires --shard-index")
        return run_shard(config_path, args.shard_index, args.num_shards, args.resume)
    except Exception as exc:
        failure_dir = Path("outputs/stage_c/failures")
        failure_dir.mkdir(parents=True, exist_ok=True)
        suffix = args.command if args.shard_index is None else f"shard_{args.shard_index:02d}"
        write_json(failure_dir / f"stage_c_{suffix}_failure.json", {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
