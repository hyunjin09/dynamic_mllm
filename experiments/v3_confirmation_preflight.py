from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.ipc as ipc
import torch
import yaml
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from experiments.stage_a_validity import max_abs_difference, prepare_prompt, set_determinism
from experiments.stage_b_reference_likelihood import (
    accepted_answers,
    capture_prompt_with_cache,
    read_jsonl,
    score_accepted_answer_set,
)
from interventions.four_state import FOUR_STATES
from interventions.prompt_cache import run_cached_prompt_state
from interventions.read_path import ReadInterventionCache
from nulls.four_action_structured import generate_isotropic_null
from nulls.structured_read import (
    fit_fixed_grid_covariance,
    generate_covariance_null,
    generate_real_residual_null,
)


LAYER_SET = [0, 4, 8, 12, 16, 20, 24]
SMOKE_LAYERS = [0, 12, 24]
NONFULL_ACTIONS = ["IGNORE", "READ_ONLY", "WRITE_ONLY"]
BASE_SEED = 2026080604
GRID_ROWS = 32
VARIANCE_TARGET = 0.90
EIGEN_SHRINKAGE = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Outcome-blind v3 technical preflight.")
    parser.add_argument(
        "--output", default="outputs/v3_preflight/null_preflight_manifest.json"
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derived_seed(family: str, dataset: str, sample_id: str, layer: int, path: str) -> int:
    material = f"{BASE_SEED}:{family}:{dataset}:{sample_id}:{layer}:{path}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def load_model():
    model_config = yaml.safe_load(Path("configs/model.yaml").read_text(encoding="utf-8"))
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
    ).to(torch.device("cuda"))
    model.eval()
    model.requires_grad_(False)
    for layer in model.model.layers:
        layer.self_attn.stage_a_query_chunk_size = int(
            model_config["decoder_attention_query_chunk_size"]
        )
    return model, processor, model_config


def residual_parts(
    context,
    visual_mask: torch.Tensor,
    read_cache: ReadInterventionCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    if read_cache.actual_output is None or read_cache.off_output is None:
        raise RuntimeError("READ cache is incomplete")
    visual_indices = torch.where(visual_mask[0])[0]
    post_start = int(visual_indices[-1].item()) + 1
    read = (
        read_cache.actual_output.float() - read_cache.off_output.float()
    )[0, post_start:].detach().cpu()
    write = (
        context.full_layer_output.float() - context.pre_layer_state.float()
    )[0, visual_indices].detach().cpu()
    if read.numel() == 0 or write.numel() == 0:
        raise RuntimeError("READ or WRITE residual is empty")
    return read, write


def embed_delta(
    rows: torch.Tensor,
    reference: torch.Tensor,
    visual_mask: torch.Tensor,
    path: str,
) -> torch.Tensor:
    result = torch.zeros_like(reference, dtype=torch.float32)
    visual_indices = torch.where(visual_mask[0])[0]
    if path == "read":
        start = int(visual_indices[-1].item()) + 1
        if rows.shape != result[0, start:].shape:
            raise ValueError("READ null rows do not match target postvisual rows")
        result[0, start:] = rows.to(result.device)
    elif path == "write":
        if rows.shape != result[0, visual_indices].shape:
            raise ValueError("WRITE null rows do not match target visual rows")
        result[0, visual_indices] = rows.to(result.device)
    else:
        raise ValueError(path)
    return result


def collect_calibration(
    model,
    processor,
    device: torch.device,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    collected: list[dict[int, dict[str, Any]]] = []
    audits: list[dict[str, Any]] = []
    for record_index, record in enumerate(rows):
        _, inputs = prepare_prompt(processor, record, device)
        visual_mask = inputs["input_ids"] == model.config.image_token_id
        baseline, contexts = capture_prompt_with_cache(model, inputs, SMOKE_LAYERS)
        sample: dict[int, dict[str, Any]] = {}
        for layer in SMOKE_LAYERS:
            cache = ReadInterventionCache()
            write_only = run_cached_prompt_state(
                model,
                contexts[layer],
                baseline.past_key_values,
                visual_mask,
                "WRITE_ONLY",
                "off",
                "full",
                cache,
            )
            read, write = residual_parts(contexts[layer], visual_mask, cache)
            del write_only
            sample[layer] = {
                "record": record,
                "read": read,
                "write": write,
            }
            if record_index == 0:
                sample[layer].update(
                    {
                        "inputs": inputs,
                        "visual_mask": visual_mask,
                        "baseline": baseline,
                        "context": contexts[layer],
                        "read_cache": cache,
                    }
                )
            audits.append(
                {
                    "dataset": record["benchmark"],
                    "id": record["id"],
                    "layer": layer,
                    "read_shape": list(read.shape),
                    "write_shape": list(write.shape),
                    "read_norm": float(read.norm().item()),
                    "write_norm": float(write.norm().item()),
                }
            )
        collected.append(sample)
        if record_index != 0:
            del inputs, baseline, contexts
            torch.cuda.empty_cache()
    return collected, audits


def run_dataset_smoke(model, processor, dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    device = torch.device("cuda")
    calibration, residual_audits = collect_calibration(model, processor, device, rows[:3])
    target_by_layer = calibration[0]
    layer_rows: list[dict[str, Any]] = []
    for layer in SMOKE_LAYERS:
        target = target_by_layer[layer]
        inputs = target["inputs"]
        visual_mask = target["visual_mask"]
        baseline = target["baseline"]
        context = target["context"]
        read_cache = target["read_cache"]
        answers = accepted_answers(target["record"])

        real_states = {}
        for name in ("FULL", "IGNORE", "READ_ONLY", "WRITE_ONLY"):
            read_mode, write_mode = FOUR_STATES[name]
            real_states[name] = run_cached_prompt_state(
                model,
                context,
                baseline.past_key_values,
                visual_mask,
                name,
                read_mode,
                write_mode,
                read_cache,
            )
        full_score = score_accepted_answer_set(
            model,
            processor.tokenizer,
            real_states["FULL"].prompt_logits,
            real_states["FULL"].past_key_values,
            inputs["attention_mask"],
            answers,
        )

        read_samples = [item[layer]["read"] for item in calibration]
        write_samples = [item[layer]["write"] for item in calibration]
        read_fit = fit_fixed_grid_covariance(
            read_samples, GRID_ROWS, VARIANCE_TARGET, EIGEN_SHRINKAGE
        )
        write_fit = fit_fixed_grid_covariance(
            write_samples, GRID_ROWS, VARIANCE_TARGET, EIGEN_SHRINKAGE
        )
        target_read = target["read"]
        target_write = target["write"]
        family_pairs = {
            "isotropic": (
                generate_isotropic_null(
                    target_read.shape,
                    float(target_read.norm().item()),
                    derived_seed("isotropic", dataset, target["record"]["id"], layer, "read"),
                ),
                generate_isotropic_null(
                    target_write.shape,
                    float(target_write.norm().item()),
                    derived_seed("isotropic", dataset, target["record"]["id"], layer, "write"),
                ),
            ),
            "covariance": (
                generate_covariance_null(
                    read_fit,
                    target_read.shape[0],
                    float(target_read.norm().item()),
                    derived_seed("covariance", dataset, target["record"]["id"], layer, "read"),
                ),
                generate_covariance_null(
                    write_fit,
                    target_write.shape[0],
                    float(target_write.norm().item()),
                    derived_seed("covariance", dataset, target["record"]["id"], layer, "write"),
                ),
            ),
            "real_residual": (
                generate_real_residual_null(
                    calibration[1][layer]["read"],
                    target_read.shape[0],
                    float(target_read.norm().item()),
                ),
                generate_real_residual_null(
                    calibration[1][layer]["write"],
                    target_write.shape[0],
                    float(target_write.norm().item()),
                ),
            ),
        }
        family_rows = []
        for family, (read_rows, write_rows) in family_pairs.items():
            read_delta = embed_delta(
                read_rows, read_cache.actual_output, visual_mask, "read"
            ).to(device)
            write_delta = embed_delta(
                write_rows, context.full_layer_output, visual_mask, "write"
            ).to(device)
            action_finite = {}
            for action in NONFULL_ACTIONS:
                replace_read = action in {"IGNORE", "WRITE_ONLY"}
                replace_write = action in {"IGNORE", "READ_ONLY"}
                result = run_cached_prompt_state(
                    model,
                    context,
                    baseline.past_key_values,
                    visual_mask,
                    f"{family}:{action}",
                    "replace" if replace_read else "full",
                    "replace" if replace_write else "full",
                    read_cache,
                    read_replacement_delta=read_delta if replace_read else None,
                    write_replacement_delta=write_delta if replace_write else None,
                )
                action_finite[action] = bool(torch.isfinite(result.prompt_logits).all().item())
            family_rows.append(
                {
                    "family": family,
                    "searched_actions": list(NONFULL_ACTIONS),
                    "action_suffix_finite": action_finite,
                    "read_shape": list(read_rows.shape),
                    "write_shape": list(write_rows.shape),
                    "read_norm_relative_error": abs(
                        float(read_rows.norm().item()) - float(target_read.norm().item())
                    ) / max(float(target_read.norm().item()), 1e-12),
                    "write_norm_relative_error": abs(
                        float(write_rows.norm().item()) - float(target_write.norm().item())
                    ) / max(float(target_write.norm().item()), 1e-12),
                }
            )
        layer_rows.append(
            {
                "layer": layer,
                "full_parity_logit_max_abs": max_abs_difference(
                    real_states["FULL"].prompt_logits, baseline.logits
                ),
                "four_real_actions_finite": all(
                    torch.isfinite(state.prompt_logits).all().item()
                    for state in real_states.values()
                ),
                "accepted_answer_component_count": len(answers),
                "accepted_answer_score_finite": all(
                    torch.isfinite(torch.tensor(value)).item()
                    for value in (full_score["mean_logprob"], full_score["sequence_logprob"])
                ),
                "null_families": family_rows,
                "smoke_covariance_rank": {
                    "read": read_fit.rank,
                    "write": write_fit.rank,
                    "note": "technical three-record fit only; confirmatory rank is set by the frozen 90% calibration rule",
                },
            }
        )
    return {"dataset": dataset, "residual_extraction": residual_audits, "layers": layer_rows}


def gqa_query_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    reserve = json.loads(
        Path("outputs/v3_preflight/stage_c2_reserved_pool_audit.json").read_text(encoding="utf-8")
    )
    group = reserve["gqa"]["groups"][0]
    wanted = {item.split("_")[-1] for item in group["question_ids"][:2]}
    arrow = Path(
        "/data/dataset/huggingface/datasets/lmms-lab___gqa/val_balanced_instructions/"
        "0.0.0/a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8/gqa-val.arrow"
    )
    found = []
    with arrow.open("rb") as handle:
        for batch in ipc.open_stream(handle):
            for row in batch.select(["id", "imageId", "question", "answer"]).to_pylist():
                if str(row["id"]) in wanted:
                    image_id = str(row["imageId"])
                    roots = [Path("/data/dataset/VG/VG_100K"), Path("/data/dataset/VG/VG_100K_2")]
                    image_path = next(root / f"{image_id}.jpg" for root in roots if (root / f"{image_id}.jpg").is_file())
                    found.append(
                        {
                            "id": f"gqa:gqa_val_{row['id']}",
                            "benchmark": "gqa",
                            "question": row["question"],
                            "answer": row["answer"],
                            "prompt": f"{row['question']}\nAnswer the question using a single word or phrase.",
                            "local_image_path": str(image_path),
                        }
                    )
    if len(found) != 2:
        raise RuntimeError("Could not resolve the reserved GQA query-invariance pair")
    return found[0], found[1]


def query_invariance_smoke(model, processor) -> dict[str, Any]:
    first, second = gqa_query_pair()
    device = torch.device("cuda")
    records = []
    for record in (first, second):
        _, inputs = prepare_prompt(processor, record, device)
        visual_mask = inputs["input_ids"] == model.config.image_token_id
        outputs, contexts = capture_prompt_with_cache(model, inputs, LAYER_SET)
        indices = torch.where(visual_mask[0])[0]
        records.append(
            {
                "record": record,
                "count": int(indices.numel()),
                "layers": {
                    layer: {
                        "pre": contexts[layer].pre_layer_state[0, indices].detach().cpu(),
                        "full": contexts[layer].full_layer_output[0, indices].detach().cpu(),
                    }
                    for layer in LAYER_SET
                },
            }
        )
        del outputs, contexts, inputs, visual_mask
        torch.cuda.empty_cache()
    if records[0]["count"] != records[1]["count"]:
        raise RuntimeError("Same image produced different visual-token counts")
    rows = []
    for layer in LAYER_SET:
        first_pre = records[0]["layers"][layer]["pre"]
        second_pre = records[1]["layers"][layer]["pre"]
        first_full = records[0]["layers"][layer]["full"]
        second_full = records[1]["layers"][layer]["full"]
        rows.append(
            {
                "layer": layer,
                "pre_layer_visual_max_abs": max_abs_difference(first_pre, second_pre),
                "post_layer_visual_max_abs": max_abs_difference(first_full, second_full),
                "write_residual_max_abs": max_abs_difference(
                    first_full.float() - first_pre.float(),
                    second_full.float() - second_pre.float(),
                ),
            }
        )
    return {
        "dataset": "gqa",
        "same_image": True,
        "sample_ids": [first["id"], second["id"]],
        "questions_differ": first["question"] != second["question"],
        "visual_token_count": records[0]["count"],
        "layers": rows,
        "terminal_action_values_computed_or_inspected": False,
    }


def execute(output: Path) -> None:
    set_determinism(BASE_SEED)
    model, processor, model_config = load_model()
    manifest = read_jsonl(Path("data_manifests/stage_b_discovery_candidates_400.jsonl"))
    by_dataset = defaultdict(list)
    for row in manifest:
        by_dataset[row["benchmark"]].append(row)
    with torch.inference_mode():
        dataset_smokes = [
            run_dataset_smoke(model, processor, dataset, by_dataset[dataset][:3])
            for dataset in ("gqa", "textvqa")
        ]
        invariance = query_invariance_smoke(model, processor)
    parity_tolerance = 1e-4
    norm_tolerance = 1e-5
    technical_pass = all(
        layer["full_parity_logit_max_abs"] <= parity_tolerance
        and layer["four_real_actions_finite"]
        and layer["accepted_answer_score_finite"]
        and all(
            all(family["action_suffix_finite"].values())
            and family["read_norm_relative_error"] <= norm_tolerance
            and family["write_norm_relative_error"] <= norm_tolerance
            for family in layer["null_families"]
        )
        for dataset in dataset_smokes
        for layer in dataset["layers"]
    )
    invariance_pass = all(
        row["pre_layer_visual_max_abs"] == 0.0
        and row["post_layer_visual_max_abs"] == 0.0
        and row["write_residual_max_abs"] == 0.0
        for row in invariance["layers"]
    )
    payload = {
        "schema_version": "v3_null_preflight_manifest_v1",
        "outcome_blind": True,
        "heldout_terminal_action_values_loaded_computed_or_inspected": False,
        "calibration_outcomes": "six already-inspected Stage B records; scores used only for finite scoring-path smoke",
        "model": model_config,
        "layer_action_search": {
            "confirmatory_layers": LAYER_SET,
            "nonfull_actions": NONFULL_ACTIONS,
            "cells_per_null_replicate": len(LAYER_SET) * len(NONFULL_ACTIONS),
            "same_maximum_rule_as_real": True,
            "terminal_layer_27_excluded_as_structurally_WRITE_silent": True,
        },
        "smoke_scope": {
            "datasets": ["gqa", "textvqa"],
            "layers": SMOKE_LAYERS,
            "all_four_real_actions": True,
            "all_three_null_action_analogues_per_family": True,
            "families": ["isotropic", "covariance", "real_residual"],
        },
        "tolerances": {"full_parity_max_abs": parity_tolerance, "norm_relative_error": norm_tolerance},
        "dataset_smokes": dataset_smokes,
        "query_invariance": invariance,
        "query_invariance_pass": invariance_pass,
        "technical_smoke_pass": technical_pass and invariance_pass,
        "serialization_pass": True,
    }
    write_json(output, payload)
    print(json.dumps({
        "output": str(output),
        "technical_smoke_pass": payload["technical_smoke_pass"],
        "query_invariance_pass": invariance_pass,
    }, indent=2))
    if not payload["technical_smoke_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    execute(Path(parse_args().output))
