#!/usr/bin/env python3
"""Correct answer-position dense logit-emergence extraction and analysis."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from dense_failure_stage1.answer_logit_emergence import (
    DATASETS,
    answer_prediction_position,
    build_divergence_spec,
    cached_replay_matches,
    classify_wrong_trajectory,
    read_layer_logits_after_generated_prefix,
    read_layer_logits_at_position,
    select_position_sanity,
    stable_reference_order,
    summarize_layerwise_trajectories,
    token_metrics,
)
from dense_failure_stage1.logit_emergence import first_persistent_layer
from dense_failure_stage1.runtime import (
    build_dense_inputs,
    configure_dense_determinism,
    load_dense_runtime,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYERS = 28


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    write_text_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def write_csv_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    payload = dict(value)
    payload.pop("contract_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _terminated(ids: list[int], im_end_token_id: int) -> list[int]:
    values = [int(value) for value in ids]
    try:
        end = values.index(int(im_end_token_id))
    except ValueError:
        values.append(int(im_end_token_id))
    else:
        values = values[: end + 1]
    return values


def validate_static_files(config: dict, *, verify_model: bool) -> None:
    for key in ("candidate_manifest", "dense_outputs", "lmms_eval_contract"):
        path = resolve_path(config["inputs"][key])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        expected = config["inputs"][f"{key}_sha256"]
        if actual != expected:
            raise ValueError(f"input hash differs: {path}: {actual} != {expected}")
    if verify_model:
        model_root = Path(config["model"]["path"])
        for name, expected in config["model"]["artifact_sha256"].items():
            path = model_root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(f"model artifact hash differs: {path}")


def _population_rows(config: dict, tokenizer) -> list[dict]:
    candidates = {
        str(row["uid"]): row
        for row in read_jsonl(resolve_path(config["inputs"]["candidate_manifest"]))
    }
    outputs = read_jsonl(resolve_path(config["inputs"]["dense_outputs"]))
    im_end = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    rows = []
    for output in outputs:
        uid = str(output["uid"])
        candidate = candidates.get(uid)
        if candidate is None:
            raise ValueError(f"dense output has no candidate row: {uid}")
        for left, right, label in (
            (candidate["dataset"], output["dataset"], "dataset"),
            (candidate["answer"], output["gt_answer"], "GT answer"),
            (candidate["image_group_id"], output["image_group_id"], "image group"),
            (
                candidate["image_content_sha256"],
                output["image_content_sha256"],
                "image SHA-256",
            ),
        ):
            if left != right:
                raise ValueError(f"candidate/output {label} differs for {uid}")
        gt_ids = tokenizer.encode(str(output["gt_answer"]), add_special_tokens=False)
        if not gt_ids:
            raise ValueError(f"canonical GT tokenizes empty: {uid}")
        generated_ids = [int(value) for value in output["generated_token_ids"]]
        if not generated_ids:
            raise ValueError(f"generated answer has no token: {uid}")
        references = stable_reference_order(output.get("gt_answers") or [])
        fallback_sequences = []
        if str(output["dataset"]) == "textvqa":
            for reference_index, reference in enumerate(references):
                reference_ids = tokenizer.encode(reference, add_special_tokens=False)
                if reference_ids:
                    fallback_sequences.append(
                        (f"textvqa_reference_{reference_index}", reference_ids)
                    )
        divergence = build_divergence_spec(
            canonical_gt_ids=gt_ids,
            generated_ids=generated_ids,
            im_end_token_id=im_end,
            fallback_gt_sequences=fallback_sequences,
        )
        row = {
            "schema_version": "dense_answer_logit_population_v2",
            "uid": uid,
            "sample_id": candidate["sample_id"],
            "dataset": candidate["dataset"],
            "prompt": candidate["prompt"],
            "answer": candidate["answer"],
            "all_answer_norms": output.get("gt_answers"),
            "local_image_path": candidate["local_image_path"],
            "image_content_sha256": candidate["image_content_sha256"],
            "image_group_id": candidate["image_group_id"],
            "current_dense_correct": bool(output["current_dense_correct"]),
            "current_dense_wrong": bool(output["current_dense_wrong"]),
            "lmms_eval_per_sample_score": output["lmms_eval_per_sample_score"],
            "lmms_eval_metric": output["lmms_eval_metric"],
            "generated_answer": output["generated_answer"],
            "generated_token_ids": generated_ids,
            "generated_answer_token_ids": _terminated(generated_ids, im_end),
            "canonical_gt_token_ids": [int(value) for value in gt_ids],
            "canonical_gt_first_token_id": int(gt_ids[0]),
            "stored_first_generated_token_id": int(generated_ids[0]),
            "literal_prompt_sha256": output["literal_prompt_sha256"],
            "comparison_usable": bool(divergence.usable),
            "comparison_unusable_reason": divergence.reason,
            "comparison_gt_source": divergence.gt_source,
            "comparison_gt_sequence_ids": list(divergence.gt_sequence_ids),
            "comparison_generated_sequence_ids": list(
                divergence.generated_sequence_ids
            ),
            "comparison_common_prefix_ids": list(divergence.common_prefix_ids),
            "comparison_divergence_index": divergence.divergence_index,
            "comparison_gt_token_id": divergence.gt_token_id,
            "comparison_generated_token_id": divergence.generated_token_id,
        }
        rows.append(row)
    rows.sort(key=lambda row: row["uid"])
    expected = int(config["inputs"]["expected_records"])
    if len(rows) != expected or len({row["uid"] for row in rows}) != expected:
        raise ValueError(f"population UID coverage differs: {len(rows)} != {expected}")
    if sum(row["current_dense_correct"] for row in rows) != int(
        config["inputs"]["expected_correct"]
    ):
        raise ValueError("correct population count differs")
    if sum(row["current_dense_wrong"] for row in rows) != int(
        config["inputs"]["expected_wrong"]
    ):
        raise ValueError("wrong population count differs")
    counts = Counter(row["dataset"] for row in rows)
    if counts != Counter(
        {key: int(value) for key, value in config["inputs"]["datasets"].items()}
    ):
        raise ValueError(f"dataset population count differs: {counts}")
    return rows


def freeze(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    validate_static_files(config, verify_model=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["path"],
        revision=config["model"]["revision"],
        local_files_only=True,
        use_fast=False,
    )
    rows = _population_rows(config, tokenizer)
    output_root = Path(args.output_root)
    population_path = output_root / "analysis_population.jsonl"
    write_jsonl_atomic(population_path, rows)
    sanity_rows = select_position_sanity(
        rows,
        seed=int(config["sanity"]["seed"]),
        per_cell=int(config["sanity"]["records_per_dataset_outcome"]),
        teacher_forced_per_wrong_dataset=int(
            config["sanity"]["teacher_forced_shared_prefix_per_wrong_dataset"]
        ),
    )
    if len(sanity_rows) != int(config["sanity"]["expected_records"]):
        raise ValueError("sanity population count differs")
    sanity_path = output_root / "position_sanity_manifest.jsonl"
    write_jsonl_atomic(sanity_path, sanity_rows)
    code_paths = [
        Path("configs/dense_answer_logit_emergence_v2.json"),
        Path("dense_failure_stage1/answer_logit_emergence.py"),
        Path("experiments/analyze_dense_answer_logit_emergence_v2.py"),
        Path("tests/test_dense_answer_logit_emergence_v2.py"),
    ]
    contract = {
        "schema_version": "dense_answer_logit_emergence_v2_contract_v1",
        "frozen_at": utc_now(),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "worktree_status_porcelain_v1": _git(
                "status", "--porcelain=v1"
            ).splitlines(),
        },
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model": config["model"],
        "inputs": config["inputs"],
        "position": config["position"],
        "targets": config["targets"],
        "emergence": config["emergence"],
        "sanity": config["sanity"],
        "population_reporting": config["population_reporting"],
        "execution": config["execution"],
        "backend_settings": config["backend_settings"],
        "population": {
            "path": str(population_path),
            "sha256": file_sha256(population_path),
            "records": len(rows),
            "correct": sum(row["current_dense_correct"] for row in rows),
            "wrong": sum(row["current_dense_wrong"] for row in rows),
            "wrong_shared_prefix": sum(
                row["current_dense_wrong"]
                and row["comparison_usable"]
                and len(row["comparison_common_prefix_ids"]) > 0
                for row in rows
            ),
            "wrong_unusable_before_execution": sum(
                row["current_dense_wrong"] and not row["comparison_usable"]
                for row in rows
            ),
            "correct_canonical_gt_first_token_differs_from_generated": sum(
                row["current_dense_correct"]
                and row["canonical_gt_first_token_id"]
                != row["stored_first_generated_token_id"]
                for row in rows
            ),
            "correct_canonical_gt_first_token_mismatch_by_dataset": {
                dataset: sum(
                    row["dataset"] == dataset
                    and row["current_dense_correct"]
                    and row["canonical_gt_first_token_id"]
                    != row["stored_first_generated_token_id"]
                    for row in rows
                )
                for dataset in DATASETS
            },
            "wrong_shared_prefix_by_dataset": {
                dataset: sum(
                    row["dataset"] == dataset
                    and row["current_dense_wrong"]
                    and row["comparison_usable"]
                    and len(row["comparison_common_prefix_ids"]) > 0
                    for row in rows
                )
                for dataset in DATASETS
            },
        },
        "sanity_manifest": {
            "path": str(sanity_path),
            "sha256": file_sha256(sanity_path),
            "records": len(sanity_rows),
        },
        "code_sha256": {
            str(path): file_sha256(REPO_ROOT / path) for path in code_paths
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_models": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    write_json_atomic(output_root / "analysis_contract.json", contract)
    print(
        json.dumps(
            {
                "contract_sha256": contract["contract_sha256"],
                **contract["population"],
                "sanity_records": len(sanity_rows),
            },
            sort_keys=True,
        )
    )


def validate_contract(args: argparse.Namespace, *, verify_model: bool) -> tuple[dict, dict]:
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_root = Path(args.output_root)
    contract = read_json(output_root / "analysis_contract.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise ValueError("analysis contract self-hash differs")
    if file_sha256(config_path) != contract["config_sha256"]:
        raise ValueError("config differs from frozen contract")
    validate_static_files(config, verify_model=verify_model)
    for key in ("population", "sanity_manifest"):
        spec = contract[key]
        if file_sha256(Path(spec["path"])) != spec["sha256"]:
            raise ValueError(f"frozen {key} differs")
    for path_value, expected in contract["code_sha256"].items():
        if file_sha256(REPO_ROOT / path_value) != expected:
            raise ValueError(f"frozen code differs: {path_value}")
    if _git("rev-parse", "HEAD") != contract["git"]["commit"]:
        raise ValueError("git commit differs from frozen contract")
    if _git("branch", "--show-current") != contract["git"]["branch"]:
        raise ValueError("git branch differs from frozen contract")
    if _git("status", "--porcelain=v1").splitlines() != contract["git"][
        "worktree_status_porcelain_v1"
    ]:
        raise ValueError("git worktree status differs from frozen contract")
    actual_environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_models": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    if actual_environment != contract["environment"]:
        raise ValueError("runtime environment differs from frozen contract")
    return config, contract


def _boundary_tokens(tokenizer, input_ids: torch.Tensor, position: int) -> list[dict]:
    ids = input_ids[0].detach().cpu().tolist()
    start = max(0, position - 15)
    return [
        {
            "index": index,
            "token_id": int(ids[index]),
            "token": tokenizer.convert_ids_to_tokens(int(ids[index])),
            "decoded": tokenizer.decode(
                [int(ids[index])], clean_up_tokenization_spaces=False
            ),
            "is_answer_prediction_position": index == position,
        }
        for index in range(start, position + 1)
    ]


def _metric_lists(metrics: dict[str, torch.Tensor]) -> dict[str, list]:
    return {
        "token_logits": metrics["token_logits"].tolist(),
        "token_ranks": [int(value) for value in metrics["token_ranks"].tolist()],
        "top1_token_ids": [
            int(value) for value in metrics["top1_token_ids"].tolist()
        ],
        "top1_logits": metrics["top1_logits"].tolist(),
        "strongest_non_target_logits": metrics[
            "strongest_non_target_logits"
        ].tolist(),
    }


def _process_sample(processor, model, device, sample: dict, *, mode: str) -> dict:
    inputs, metadata = build_dense_inputs(processor, sample, device)
    if metadata["literal_prompt_sha256"] != sample["literal_prompt_sha256"]:
        raise ValueError(f"literal prompt SHA-256 differs for {sample['uid']}")
    expected_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    if not metadata["literal_prompt"].endswith(expected_suffix):
        raise ValueError(f"assistant-generation suffix differs for {sample['uid']}")
    position = answer_prediction_position(inputs["attention_mask"])
    if position != int(inputs["input_ids"].shape[1]) - 1:
        raise ValueError("answer prediction position is not the final input token")
    base = read_layer_logits_at_position(model, inputs, position=position)
    if base.layer_logits.shape != (LAYERS, int(model.config.text_config.vocab_size)):
        raise ValueError(f"unexpected layer-logit shape: {base.layer_logits.shape}")
    if not base.final_top1_matches_model:
        raise ValueError("hooked layer-27 top-1 differs from model final logits")
    top1 = int(base.layer_logits[-1].argmax().item())
    stored_first = int(sample["stored_first_generated_token_id"])
    row = {
        "schema_version": "dense_answer_logit_position_sanity_v2"
        if mode == "sanity"
        else "dense_answer_logit_trajectory_v2",
        "uid": sample["uid"],
        "dataset": sample["dataset"],
        "image_group_id": sample["image_group_id"],
        "current_dense_correct": sample["current_dense_correct"],
        "current_dense_wrong": sample["current_dense_wrong"],
        "answer_prediction_position": position,
        "prompt_token_count": int(inputs["attention_mask"].sum().item()),
        "literal_prompt_sha256": metadata["literal_prompt_sha256"],
        "consumed_image_sha256": metadata["consumed_image_sha256"],
        "stored_first_generated_token_id": stored_first,
        "layer27_top1_token_id": top1,
        "layer27_top1_matches_stored_generation": top1 == stored_first,
        "layer27_top1_matches_model_forward": base.final_top1_matches_model,
        "layer27_reconstructed_top1_matches_model_forward": (
            base.reconstructed_final_top1_matches_model
        ),
        "layer27_max_abs_logit_difference_vs_model": (
            base.final_max_abs_logit_difference
        ),
    }
    if mode == "sanity":
        row["boundary_tokens"] = _boundary_tokens(
            processor.tokenizer, inputs["input_ids"], position
        )
        row["stored_first_generated_token_text"] = processor.tokenizer.decode(
            [stored_first], clean_up_tokenization_spaces=False
        )
        row["layer27_top1_token_text"] = processor.tokenizer.decode(
            [top1], clean_up_tokenization_spaces=False
        )
        row["layers_read"] = int(base.layer_logits.shape[0])
        prefix = [int(value) for value in sample["comparison_common_prefix_ids"]]
        row["teacher_forced_divergence_checked"] = False
        row["teacher_forced_prefix_ids"] = []
        if sample["current_dense_wrong"] and prefix:
            if not sample["comparison_usable"]:
                raise ValueError("selected shared-prefix sanity comparison is unusable")
            comparison = read_layer_logits_after_generated_prefix(
                model, inputs, prefix_ids=prefix
            )
            comparison_top1 = int(comparison.layer_logits[-1].argmax().item())
            expected_generated = int(sample["comparison_generated_token_id"])
            replay_matches = cached_replay_matches(
                prefix, expected_generated, comparison.generated_ids
            )
            row.update(
                {
                    "teacher_forced_divergence_checked": True,
                    "teacher_forced_prefix_ids": prefix,
                    "teacher_forced_divergence_index": sample[
                        "comparison_divergence_index"
                    ],
                    "teacher_forced_gt_token_id": int(
                        sample["comparison_gt_token_id"]
                    ),
                    "teacher_forced_expected_generated_token_id": expected_generated,
                    "teacher_forced_layer27_top1_token_id": comparison_top1,
                    "teacher_forced_layer27_top1_matches_generated_divergence": (
                        comparison_top1 == expected_generated
                    ),
                    "teacher_forced_layer27_top1_matches_model_forward": (
                        comparison.final_top1_matches_raw_generation_logits
                    ),
                    "teacher_forced_reconstructed_layer27_top1_matches_raw_model_logits": (
                        comparison.reconstructed_final_top1_matches_raw_generation_logits
                    ),
                    "teacher_forced_layer27_max_abs_logit_difference_vs_model": (
                        comparison.final_max_abs_logit_difference
                    ),
                    "teacher_forced_answer_prediction_position": (
                        position + len(prefix)
                    ),
                    "teacher_forced_prompt_token_count": int(
                        inputs["attention_mask"].sum().item() + len(prefix)
                    ),
                    "teacher_forced_cached_replay_ids": list(
                        comparison.generated_ids
                    ),
                    "teacher_forced_cached_replay_matches_stored_sequence": (
                        replay_matches
                    ),
                }
            )
        return row
    if sample["current_dense_correct"]:
        target_id = int(sample["canonical_gt_first_token_id"])
        metrics = token_metrics(base.layer_logits, token_id=target_id)
        generated = token_metrics(base.layer_logits, token_id=stored_first)
        values = _metric_lists(metrics)
        row.update(
            {
                "comparison_usable": True,
                "comparison_position_kind": "assistant_answer_start",
                "comparison_gt_source": "canonical",
                "comparison_divergence_index": 0,
                "teacher_forced_prefix_ids": [],
                "gt_token_id": target_id,
                "generated_token_id": stored_first,
                "gt_token_logits": values["token_logits"],
                "gt_token_ranks": values["token_ranks"],
                "top1_token_ids": values["top1_token_ids"],
                "top1_logits": values["top1_logits"],
                "strongest_non_gt_logits": values[
                    "strongest_non_target_logits"
                ],
                "correct_margins": (
                    metrics["token_logits"]
                    - metrics["strongest_non_target_logits"]
                ).tolist(),
                "generated_token_logits": generated["token_logits"].tolist(),
                "generated_token_ranks": [
                    int(value) for value in generated["token_ranks"].tolist()
                ],
                "gt_generated_first_token_match": target_id == stored_first,
            }
        )
        return row
    if not sample["comparison_usable"]:
        row.update(
            {
                "comparison_usable": False,
                "comparison_unusable_reason": sample[
                    "comparison_unusable_reason"
                ],
                "comparison_gt_source": sample["comparison_gt_source"],
            }
        )
        return row
    prefix = [int(value) for value in sample["comparison_common_prefix_ids"]]
    if prefix:
        comparison = read_layer_logits_after_generated_prefix(
            model, inputs, prefix_ids=prefix
        )
        comparison_position = position + len(prefix)
    else:
        comparison_position = position
        comparison = base
    gt_token_id = int(sample["comparison_gt_token_id"])
    generated_token_id = int(sample["comparison_generated_token_id"])
    comparison_top1 = int(comparison.layer_logits[-1].argmax().item())
    comparison_internal_match = (
        comparison.final_top1_matches_raw_generation_logits
        if prefix
        else comparison.final_top1_matches_model
    )
    if not comparison_internal_match:
        raise ValueError("comparison layer-27 top-1 differs from model logits")
    if prefix and not cached_replay_matches(
        prefix, generated_token_id, comparison.generated_ids
    ):
        row.update(
            {
                "comparison_usable": False,
                "comparison_unusable_reason": (
                    "cached_greedy_replay_differs_from_stored_prefix_or_next_token"
                ),
                "comparison_gt_source": sample["comparison_gt_source"],
                "comparison_divergence_index": sample[
                    "comparison_divergence_index"
                ],
                "teacher_forced_prefix_ids": prefix,
                "cached_replay_ids": list(comparison.generated_ids),
            }
        )
        return row
    gt_metrics = token_metrics(comparison.layer_logits, token_id=gt_token_id)
    generated_metrics = token_metrics(
        comparison.layer_logits, token_id=generated_token_id
    )
    gt_values = _metric_lists(gt_metrics)
    generated_values = _metric_lists(generated_metrics)
    row.update(
        {
            "comparison_usable": True,
            "comparison_position_kind": "first_answer_divergence",
            "comparison_gt_source": sample["comparison_gt_source"],
            "comparison_divergence_index": sample["comparison_divergence_index"],
            "teacher_forced_prefix_ids": prefix,
            "comparison_answer_prediction_position": comparison_position,
            "comparison_prompt_token_count": int(
                inputs["attention_mask"].sum().item() + len(prefix)
            ),
            "gt_token_id": gt_token_id,
            "generated_token_id": generated_token_id,
            "gt_token_logits": gt_values["token_logits"],
            "gt_token_ranks": gt_values["token_ranks"],
            "generated_token_logits": generated_values["token_logits"],
            "generated_token_ranks": generated_values["token_ranks"],
            "top1_token_ids": gt_values["top1_token_ids"],
            "top1_logits": gt_values["top1_logits"],
            "wrong_margins": (
                gt_metrics["token_logits"] - generated_metrics["token_logits"]
            ).tolist(),
            "teacher_forced_layer27_top1_token_id": comparison_top1,
            "teacher_forced_layer27_top1_matches_generated_divergence": (
                comparison_top1 == generated_token_id
            ),
            "teacher_forced_cached_replay_matches_stored_sequence": (
                cached_replay_matches(
                    prefix, generated_token_id, comparison.generated_ids
                )
                if prefix
                else top1 == generated_token_id
            ),
            "teacher_forced_cached_replay_ids": (
                list(comparison.generated_ids) if prefix else []
            ),
            "teacher_forced_layer27_top1_matches_model_forward": (
                comparison_internal_match
            ),
            "teacher_forced_layer27_max_abs_logit_difference_vs_model": (
                comparison.final_max_abs_logit_difference
            ),
        }
    )
    return row


def _worker_rows(contract: dict, mode: str) -> list[dict]:
    if mode == "sanity":
        return read_jsonl(Path(contract["sanity_manifest"]["path"]))
    return read_jsonl(Path(contract["population"]["path"]))


def worker(args: argparse.Namespace) -> None:
    config, contract = validate_contract(args, verify_model=True)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["execution"]["world_size"]):
        raise ValueError(f"four workers are required: world_size={world_size}")
    if rank not in range(world_size):
        raise ValueError(f"invalid worker rank: {rank}")
    if args.mode == "full":
        gate = read_json(Path(args.output_root) / "position_sanity_gate.json")
        if (
            not gate.get("passed")
            or gate.get("contract_sha256") != contract["contract_sha256"]
        ):
            raise ValueError("full execution requires the matching passed sanity gate")
    configure_dense_determinism(
        int(config["execution"]["seed"]), config["backend_settings"]
    )
    rows = _worker_rows(contract, args.mode)
    assigned = [row for index, row in enumerate(rows) if index % world_size == rank]
    processor, model, device = load_dense_runtime(
        config["model"]["path"], config["model"]["revision"], rank
    )
    worker_root = (
        Path(args.output_root) / "sanity" / "workers"
        if args.mode == "sanity"
        else Path(args.output_root) / "workers"
    )
    batch_size = int(config["execution"]["worker_batch_records"])
    output_paths = []
    failure_paths = []
    for batch_index, start in enumerate(range(0, len(assigned), batch_size)):
        batch = assigned[start : start + batch_size]
        stem = f"rank{rank:03d}_batch{batch_index:05d}"
        marker_path = worker_root / f"{stem}.complete.json"
        expected_uids = [row["uid"] for row in batch]
        if marker_path.exists():
            marker = read_json(marker_path)
            if (
                marker.get("contract_sha256") == contract["contract_sha256"]
                and marker.get("mode") == args.mode
                and marker.get("uids") == expected_uids
            ):
                output_paths.append(marker["output_path"])
                failure_paths.append(marker["failure_path"])
                print(json.dumps({"rank": rank, "resume": stem}), flush=True)
                continue
            raise ValueError(f"incompatible existing marker: {marker_path}")
        output_rows = []
        failures = []
        for sample in batch:
            started = time.monotonic()
            try:
                result = _process_sample(
                    processor, model, device, sample, mode=args.mode
                )
                result["worker_rank"] = rank
                result["elapsed_seconds"] = time.monotonic() - started
                result["contract_sha256"] = contract["contract_sha256"]
                output_rows.append(result)
            except Exception as exc:
                failures.append(
                    {
                        "uid": sample["uid"],
                        "dataset": sample["dataset"],
                        "exception_type": type(exc).__name__,
                        "reason": str(exc),
                        "traceback": traceback.format_exc(limit=10),
                        "worker_rank": rank,
                        "elapsed_seconds": time.monotonic() - started,
                        "contract_sha256": contract["contract_sha256"],
                    }
                )
        output_path = worker_root / f"{stem}.outputs.jsonl"
        failure_path = worker_root / f"{stem}.failures.jsonl"
        write_jsonl_atomic(output_path, output_rows)
        write_jsonl_atomic(failure_path, failures)
        write_json_atomic(
            marker_path,
            {
                "schema_version": "dense_answer_logit_worker_batch_v2",
                "contract_sha256": contract["contract_sha256"],
                "mode": args.mode,
                "rank": rank,
                "world_size": world_size,
                "uids": expected_uids,
                "attempted": len(batch),
                "completed": len(output_rows),
                "failed": len(failures),
                "output_path": str(output_path),
                "failure_path": str(failure_path),
                "completed_at": utc_now(),
            },
        )
        output_paths.append(str(output_path))
        failure_paths.append(str(failure_path))
        print(
            json.dumps(
                {
                    "rank": rank,
                    "batch": batch_index + 1,
                    "batches": (len(assigned) + batch_size - 1) // batch_size,
                    "completed": len(output_rows),
                    "failed": len(failures),
                }
            ),
            flush=True,
        )
    write_json_atomic(
        worker_root / f"rank{rank:03d}.json",
        {
            "schema_version": "dense_answer_logit_worker_rank_v2",
            "contract_sha256": contract["contract_sha256"],
            "mode": args.mode,
            "rank": rank,
            "world_size": world_size,
            "records": len(assigned),
            "output_paths": output_paths,
            "failure_paths": failure_paths,
            "completed_at": utc_now(),
        },
    )
    print(json.dumps({"rank": rank, "status": "complete", "records": len(assigned)}), flush=True)


def _load_worker_rows(
    output_root: Path, contract: dict, config: dict, *, mode: str
) -> tuple[list[dict], list[dict]]:
    worker_root = (
        output_root / "sanity" / "workers" if mode == "sanity" else output_root / "workers"
    )
    outputs = []
    failures = []
    seen_paths = set()
    for rank in range(int(config["execution"]["world_size"])):
        marker = read_json(worker_root / f"rank{rank:03d}.json")
        if (
            marker["contract_sha256"] != contract["contract_sha256"]
            or marker["mode"] != mode
            or int(marker["rank"]) != rank
        ):
            raise ValueError(f"rank marker differs for rank {rank}")
        for path_value in marker["output_paths"]:
            if path_value in seen_paths:
                raise ValueError(f"duplicate worker output path: {path_value}")
            seen_paths.add(path_value)
            outputs.extend(read_jsonl(Path(path_value)))
        for path_value in marker["failure_paths"]:
            failures.extend(read_jsonl(Path(path_value)))
    expected = _worker_rows(contract, mode)
    expected_uids = {row["uid"] for row in expected}
    observed_uids = [row["uid"] for row in outputs] + [row["uid"] for row in failures]
    if len(observed_uids) != len(set(observed_uids)):
        raise ValueError("duplicate UID across worker outputs/failures")
    if set(observed_uids) != expected_uids or len(observed_uids) != len(expected):
        raise ValueError("worker UID coverage differs")
    return outputs, failures


def aggregate_sanity(args: argparse.Namespace) -> None:
    config, contract = validate_contract(args, verify_model=False)
    output_root = Path(args.output_root)
    rows, failures = _load_worker_rows(
        output_root, contract, config, mode="sanity"
    )
    rows.sort(key=lambda row: row["uid"])
    write_jsonl_atomic(output_root / "position_sanity_rows.jsonl", rows)
    matches = sum(row["layer27_top1_matches_stored_generation"] for row in rows)
    required = int(config["sanity"]["expected_records"])
    teacher_required_per_dataset = int(
        config["sanity"]["teacher_forced_shared_prefix_per_wrong_dataset"]
    )
    teacher_checked = [
        row for row in rows if row["teacher_forced_divergence_checked"]
    ]
    teacher_raw_top1_matches = sum(
        row["teacher_forced_layer27_top1_matches_generated_divergence"]
        for row in teacher_checked
    )
    teacher_replay_matches = sum(
        row["teacher_forced_cached_replay_matches_stored_sequence"]
        for row in teacher_checked
    )
    teacher_internal_matches = sum(
        row["teacher_forced_layer27_top1_matches_model_forward"]
        for row in teacher_checked
    )
    teacher_by_dataset = {
        dataset: {
            "records": len(
                [row for row in teacher_checked if row["dataset"] == dataset]
            ),
            "cached_replay_matches": sum(
                row["teacher_forced_cached_replay_matches_stored_sequence"]
                for row in teacher_checked
                if row["dataset"] == dataset
            ),
            "raw_readout_matches_raw_model_logits": sum(
                row["teacher_forced_layer27_top1_matches_model_forward"]
                for row in teacher_checked
                if row["dataset"] == dataset
            ),
            "raw_top1_matches_processed_generated_token": sum(
                row["teacher_forced_layer27_top1_matches_generated_divergence"]
                for row in teacher_checked
                if row["dataset"] == dataset
            ),
        }
        for dataset in DATASETS
    }
    teacher_gate_passed = all(
        values["records"] >= teacher_required_per_dataset
        and values["cached_replay_matches"] >= teacher_required_per_dataset
        and values["raw_readout_matches_raw_model_logits"] == values["records"]
        for values in teacher_by_dataset.values()
    )
    passed = (
        not failures
        and len(rows) == required
        and matches == required
        and all(row["layers_read"] == LAYERS for row in rows)
        and all(row["layer27_top1_matches_model_forward"] for row in rows)
        and teacher_gate_passed
    )
    by_cell = {}
    for dataset in DATASETS:
        for correct in (False, True):
            cell = [
                row
                for row in rows
                if row["dataset"] == dataset
                and bool(row["current_dense_correct"]) is correct
            ]
            by_cell[f"{dataset}/{'correct' if correct else 'wrong'}"] = {
                "records": len(cell),
                "matches": sum(
                    row["layer27_top1_matches_stored_generation"] for row in cell
                ),
            }
    gate = {
        "schema_version": "dense_answer_position_sanity_gate_v2",
        "contract_sha256": contract["contract_sha256"],
        "passed": passed,
        "records": len(rows),
        "failures": len(failures),
        "layer27_top1_matches": matches,
        "exact_match_rate": matches / len(rows) if rows else 0.0,
        "by_dataset_outcome": by_cell,
        "teacher_forced_divergence_records": len(teacher_checked),
        "teacher_forced_cached_replay_matches": teacher_replay_matches,
        "teacher_forced_cached_replay_unusable": (
            len(teacher_checked) - teacher_replay_matches
        ),
        "teacher_forced_cached_replay_exact_match_rate": (
            teacher_replay_matches / len(teacher_checked)
            if teacher_checked
            else 0.0
        ),
        "teacher_forced_raw_readout_matches_raw_model_logits": (
            teacher_internal_matches
        ),
        "teacher_forced_raw_top1_matches_processed_generated_token": (
            teacher_raw_top1_matches
        ),
        "teacher_forced_raw_top1_processed_token_match_rate": (
            teacher_raw_top1_matches / len(teacher_checked)
            if teacher_checked
            else 0.0
        ),
        "teacher_forced_by_dataset": teacher_by_dataset,
        "teacher_forced_gate_passed": teacher_gate_passed,
        "maximum_internal_layer27_logit_difference": max(
            (
                float(row["layer27_max_abs_logit_difference_vs_model"])
                for row in rows
            ),
            default=None,
        ),
    }
    write_json_atomic(output_root / "position_sanity_gate.json", gate)
    boundary = rows[0]["boundary_tokens"] if rows else []
    boundary_lines = [
        f"- `{item['index']}`: ID `{item['token_id']}`, token "
        f"`{item['token']}`, decoded `{item['decoded'].encode('unicode_escape').decode()}`"
        + (" **← position used**" if item["is_answer_prediction_position"] else "")
        for item in boundary
    ]
    write_text_atomic(
        output_root / "position_sanity_report.md",
        "\n".join(
            [
                "# Corrected assistant answer-position sanity",
                "",
                f"- Frozen contract: `{contract['contract_sha256']}`",
                f"- Result: **{'PASS' if passed else 'FAIL'}**",
                f"- Exact layer-27 top-1 / stored first-token agreement: {matches}/{len(rows)} ({gate['exact_match_rate']:.2%})",
                f"- Cached shared-prefix plus next-token replay agreement: {teacher_replay_matches}/{len(teacher_checked)} ({gate['teacher_forced_cached_replay_exact_match_rate']:.2%})",
                f"- Raw layer-27 readout / raw model-logit top-1 agreement: {teacher_internal_matches}/{len(teacher_checked)}",
                f"- Diagnostic only—raw top-1 / repetition-penalized generated-token agreement: {teacher_raw_top1_matches}/{len(teacher_checked)} ({gate['teacher_forced_raw_top1_processed_token_match_rate']:.2%})",
                f"- Execution failures: {len(failures)}",
                "- Position: final non-padding token of the complete prompt after `add_generation_prompt=true`.",
                "- Literal suffix: `<|im_end|>\\n<|im_start|>assistant\\n`; the final newline is the state used to predict answer token one.",
                "",
                "## Token boundary from the first sanity record",
                "",
                *boundary_lines,
                "",
                "## Dataset/outcome cells",
                "",
                *[
                    f"- {cell}: {values['matches']}/{values['records']} exact"
                    for cell, values in sorted(by_cell.items())
                ],
                "",
                "## Teacher-forced shared-prefix divergence checks",
                "",
                *[
                    f"- {dataset}: cached replay {values['cached_replay_matches']}/{values['records']}; raw readout parity {values['raw_readout_matches_raw_model_logits']}/{values['records']}; raw-top1/processed-token diagnostic {values['raw_top1_matches_processed_generated_token']}/{values['records']}"
                    for dataset, values in sorted(teacher_by_dataset.items())
                ],
                "",
                f"The full analysis is authorized only when every base answer-start record matches, at least {teacher_required_per_dataset} cached prefix-plus-next-token replays match per dataset, and every checked layer-27 readout matches the corresponding raw model logits. Failed continuation replays are logged and excluded from wrong-token comparison, as required by the collision skip rule. Raw top-1 need not equal a token selected after the frozen repetition penalty.",
            ]
        )
        + "\n",
    )
    if failures:
        write_jsonl_atomic(output_root / "position_sanity_failures.jsonl", failures)
    print(json.dumps(gate, sort_keys=True))
    if not passed:
        raise SystemExit("position sanity gate failed; full execution is forbidden")


def _matrix(rows: list[dict], key: str, *, dtype=np.float64) -> np.ndarray:
    values = np.asarray([row[key] for row in rows], dtype=dtype)
    if values.shape != (len(rows), LAYERS):
        raise ValueError(f"invalid {key} matrix shape: {values.shape}")
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"non-finite values in {key}")
    return values


def _layerwise_rows(rows: list[dict], *, outcome: str) -> list[dict]:
    if not rows:
        raise ValueError(f"no {outcome} rows for layer-wise summary")
    gt = _matrix(rows, "gt_token_logits")
    gt_ranks = _matrix(rows, "gt_token_ranks", dtype=np.int64)
    top1_ids = _matrix(rows, "top1_token_ids", dtype=np.int64)
    top1_logits = _matrix(rows, "top1_logits")
    gt_ids = np.asarray([int(row["gt_token_id"]) for row in rows])[:, None]
    gt_is_top1 = top1_ids == gt_ids
    if outcome == "correct":
        other = _matrix(rows, "strongest_non_gt_logits")
        margins = _matrix(rows, "correct_margins")
        other_ranks = np.where(gt_is_top1, 2, 1)
        other_is_top1 = ~gt_is_top1
        other_role = "strongest_non_gt"
    elif outcome == "wrong":
        other = _matrix(rows, "generated_token_logits")
        margins = _matrix(rows, "wrong_margins")
        other_ranks = _matrix(rows, "generated_token_ranks", dtype=np.int64)
        generated_ids = np.asarray(
            [int(row["generated_token_id"]) for row in rows]
        )[:, None]
        other_is_top1 = top1_ids == generated_ids
        other_role = "generated_wrong"
    else:
        raise ValueError(f"unsupported outcome: {outcome}")
    summaries = summarize_layerwise_trajectories(
        gt_logits=gt,
        other_logits=other,
        gt_ranks=gt_ranks,
        other_ranks=other_ranks,
        top1_logits=top1_logits,
        margins=margins,
        gt_is_top1=gt_is_top1,
        other_is_top1=other_is_top1,
    )
    for row in summaries:
        row["outcome"] = outcome
        row["other_role"] = other_role
        if outcome == "correct":
            for key in list(row):
                if "other_rank" in key:
                    row.pop(key)
    return summaries


def _event_stats(values: list[int | None]) -> dict[str, float | int | None]:
    defined = np.asarray([value for value in values if value is not None], dtype=float)
    return {
        "records": len(values),
        "defined": len(defined),
        "defined_fraction": len(defined) / len(values) if values else 0.0,
        "q25": float(np.quantile(defined, 0.25)) if len(defined) else None,
        "median": float(np.median(defined)) if len(defined) else None,
        "q75": float(np.quantile(defined, 0.75)) if len(defined) else None,
    }


def _coverage_layer(values: list[int | None], fraction: float) -> int | None:
    if not values:
        return None
    for layer in range(LAYERS):
        reached = sum(value is not None and value <= layer for value in values)
        if reached / len(values) >= fraction:
            return layer
    return None


def _format_event(value: float | int | None) -> str:
    return "not reached" if value is None else f"{value:g}"


def _plot_v2(
    output_root: Path,
    correct_rows: list[dict],
    wrong_rows: list[dict],
    correct_events: list[int | None],
    wrong_events: list[int | None],
) -> None:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    layers = np.arange(LAYERS)

    def series(rows: list[dict], key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows])

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figure_root / name, dpi=180)
        plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(layers, series(correct_rows, "mean_margin"), label="Mean")
    plt.plot(layers, series(correct_rows, "median_margin"), label="Median")
    plt.fill_between(
        layers,
        series(correct_rows, "q25_margin"),
        series(correct_rows, "q75_margin"),
        alpha=0.22,
        label="IQR",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Decoder layer")
    plt.ylabel("GT − strongest non-GT raw logit")
    plt.title("Dense-correct answer-start margin")
    plt.legend(fontsize=8)
    save("correct_margin_by_layer.png")

    plt.figure(figsize=(8, 4.8))
    plt.plot(layers, series(wrong_rows, "mean_gt_logit"), label="Mean GT")
    plt.plot(
        layers,
        series(wrong_rows, "mean_other_logit"),
        label="Mean generated-wrong",
    )
    plt.plot(
        layers,
        series(wrong_rows, "median_gt_logit"),
        linestyle="--",
        label="Median GT",
    )
    plt.plot(
        layers,
        series(wrong_rows, "median_other_logit"),
        linestyle="--",
        label="Median generated-wrong",
    )
    plt.xlabel("Decoder layer")
    plt.ylabel("Raw logit")
    plt.title("Dense-wrong GT and generated-answer logits")
    plt.legend(fontsize=8)
    save("wrong_gt_vs_generated_logits.png")

    plt.figure(figsize=(8, 4.8))
    plt.plot(layers, series(wrong_rows, "mean_margin"), label="Mean")
    plt.plot(layers, series(wrong_rows, "median_margin"), label="Median")
    plt.fill_between(
        layers,
        series(wrong_rows, "q25_margin"),
        series(wrong_rows, "q75_margin"),
        alpha=0.22,
        label="IQR",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Decoder layer")
    plt.ylabel("GT − generated-wrong raw logit")
    plt.title("Dense-wrong answer divergence margin")
    plt.legend(fontsize=8)
    save("wrong_margin_by_layer.png")

    plt.figure(figsize=(8, 4.8))
    bins = np.arange(-0.5, LAYERS + 0.5, 1)
    correct_defined = [value for value in correct_events if value is not None]
    wrong_defined = [value for value in wrong_events if value is not None]
    plt.hist(correct_defined, bins=bins, alpha=0.6, label="Correct emergence")
    plt.hist(wrong_defined, bins=bins, alpha=0.6, label="Wrong crossover")
    plt.xlabel("First persistent crossing layer")
    plt.ylabel("Samples")
    plt.title("Correct emergence and wrong crossover")
    plt.legend(fontsize=8)
    save("emergence_crossover_histogram.png")


def aggregate(args: argparse.Namespace) -> None:
    config, contract = validate_contract(args, verify_model=False)
    output_root = Path(args.output_root)
    gate = read_json(output_root / "position_sanity_gate.json")
    if not gate.get("passed") or gate.get("contract_sha256") != contract["contract_sha256"]:
        raise ValueError("matching passed position sanity gate is required")
    rows, failures = _load_worker_rows(output_root, contract, config, mode="full")
    rows.sort(key=lambda row: row["uid"])
    if failures:
        write_jsonl_atomic(output_root / "full_failures.jsonl", failures)
        raise ValueError(f"full analysis has {len(failures)} execution failures")
    expected = int(config["inputs"]["expected_records"])
    if len(rows) != expected or len({row["uid"] for row in rows}) != expected:
        raise ValueError(f"full successful UID coverage differs: {len(rows)} != {expected}")
    if any(row["contract_sha256"] != contract["contract_sha256"] for row in rows):
        raise ValueError("full output contract hash differs")
    write_jsonl_atomic(output_root / "sample_answer_trajectories.jsonl", rows)

    correct = [row for row in rows if row["current_dense_correct"]]
    wrong = [row for row in rows if row["current_dense_wrong"]]
    wrong_usable = [row for row in wrong if row.get("comparison_usable")]
    wrong_unusable = [row for row in wrong if not row.get("comparison_usable")]
    if len(correct) != int(config["inputs"]["expected_correct"]):
        raise ValueError("full correct count differs")
    if len(wrong) != int(config["inputs"]["expected_wrong"]):
        raise ValueError("full wrong count differs")

    correct_layer_rows = _layerwise_rows(correct, outcome="correct")
    wrong_layer_rows = _layerwise_rows(wrong_usable, outcome="wrong")
    write_csv_atomic(
        output_root / "layerwise_correct_summary.csv",
        correct_layer_rows,
        list(correct_layer_rows[0]),
    )
    write_csv_atomic(
        output_root / "layerwise_wrong_summary.csv",
        wrong_layer_rows,
        list(wrong_layer_rows[0]),
    )

    threshold = float(config["emergence"]["threshold_raw_logit"])
    persistence = int(config["emergence"]["persistent_layers"])
    early_cutoff = int(config["emergence"]["early_cutoff_layer"])
    sample_event_rows = []
    taxonomy_rows = []
    correct_events: list[int | None] = []
    wrong_events: list[int | None] = []
    for row in rows:
        base = {
            "uid": row["uid"],
            "dataset": row["dataset"],
            "current_dense_correct": row["current_dense_correct"],
            "comparison_usable": row.get("comparison_usable", True),
            "comparison_position_kind": row.get("comparison_position_kind"),
            "comparison_divergence_index": row.get("comparison_divergence_index"),
            "comparison_gt_source": row.get("comparison_gt_source"),
            "teacher_forced_prefix_length": len(row.get("teacher_forced_prefix_ids", [])),
        }
        if row["current_dense_correct"]:
            margins = np.asarray(row["correct_margins"], dtype=float)
            event = first_persistent_layer(
                margins,
                threshold=threshold,
                direction="greater",
                consecutive=persistence,
            )
            correct_events.append(event)
            base.update(
                {
                    "correct_emergence_layer": event,
                    "wrong_crossover_layer": None,
                    "prior_gt_preference_layer": event,
                    "wrong_trajectory_taxonomy": "not_applicable",
                    "layer0_margin": float(margins[0]),
                    "layer27_margin": float(margins[-1]),
                    "minimum_margin": float(margins.min()),
                    "maximum_margin": float(margins.max()),
                }
            )
        elif row.get("comparison_usable"):
            margins = np.asarray(row["wrong_margins"], dtype=float)
            event = first_persistent_layer(
                margins,
                threshold=-threshold,
                direction="less",
                consecutive=persistence,
            )
            prior_gt = first_persistent_layer(
                margins,
                threshold=threshold,
                direction="greater",
                consecutive=persistence,
            )
            taxonomy = classify_wrong_trajectory(
                margins,
                threshold=threshold,
                consecutive=persistence,
                early_cutoff=early_cutoff,
            )
            wrong_events.append(event)
            base.update(
                {
                    "correct_emergence_layer": None,
                    "wrong_crossover_layer": event,
                    "prior_gt_preference_layer": prior_gt,
                    "wrong_trajectory_taxonomy": taxonomy,
                    "layer0_margin": float(margins[0]),
                    "layer27_margin": float(margins[-1]),
                    "minimum_margin": float(margins.min()),
                    "maximum_margin": float(margins.max()),
                }
            )
            taxonomy_rows.append(
                {
                    **base,
                    "gt_token_id": row["gt_token_id"],
                    "generated_wrong_token_id": row["generated_token_id"],
                    "comparison_unusable_reason": None,
                }
            )
        else:
            base.update(
                {
                    "correct_emergence_layer": None,
                    "wrong_crossover_layer": None,
                    "prior_gt_preference_layer": None,
                    "wrong_trajectory_taxonomy": "unusable_comparison",
                    "layer0_margin": None,
                    "layer27_margin": None,
                    "minimum_margin": None,
                    "maximum_margin": None,
                }
            )
            taxonomy_rows.append(
                {
                    **base,
                    "gt_token_id": row.get("gt_token_id"),
                    "generated_wrong_token_id": row.get("generated_token_id"),
                    "comparison_unusable_reason": row.get(
                        "comparison_unusable_reason"
                    ),
                }
            )
        sample_event_rows.append(base)
    write_csv_atomic(
        output_root / "sample_emergence_layers.csv",
        sample_event_rows,
        list(sample_event_rows[0]),
    )
    taxonomy_fields = sorted({key for row in taxonomy_rows for key in row})
    write_csv_atomic(
        output_root / "wrong_trajectory_taxonomy.csv",
        taxonomy_rows,
        taxonomy_fields,
    )

    events_by_uid = {row["uid"]: row for row in sample_event_rows}
    dataset_rows = []
    for dataset in DATASETS:
        for outcome in ("correct", "wrong"):
            subset = [
                row
                for row in (correct if outcome == "correct" else wrong_usable)
                if row["dataset"] == dataset
            ]
            summaries = _layerwise_rows(subset, outcome=outcome)
            event_values = [
                events_by_uid[row["uid"]][
                    "correct_emergence_layer"
                    if outcome == "correct"
                    else "wrong_crossover_layer"
                ]
                for row in subset
            ]
            stats = _event_stats(event_values)
            taxonomy_counts = Counter(
                events_by_uid[row["uid"]]["wrong_trajectory_taxonomy"]
                for row in subset
            )
            for summary_row in summaries:
                dataset_rows.append(
                    {
                        "dataset": dataset,
                        **summary_row,
                        "persistent_event_defined_fraction": stats[
                            "defined_fraction"
                        ],
                        "persistent_event_q25": stats["q25"],
                        "persistent_event_median": stats["median"],
                        "persistent_event_q75": stats["q75"],
                        "early_wrong_fraction": (
                            taxonomy_counts["early_wrong"] / len(subset)
                            if outcome == "wrong"
                            else None
                        ),
                        "progressive_wrong_fraction": (
                            taxonomy_counts["progressive_wrong"] / len(subset)
                            if outcome == "wrong"
                            else None
                        ),
                        "answer_erosion_fraction": (
                            taxonomy_counts["answer_erosion"] / len(subset)
                            if outcome == "wrong"
                            else None
                        ),
                        "ambiguous_fraction": (
                            taxonomy_counts["ambiguous"] / len(subset)
                            if outcome == "wrong"
                            else None
                        ),
                    }
                )
    dataset_fields = sorted({key for row in dataset_rows for key in row})
    write_csv_atomic(
        output_root / "dataset_breakdown.csv", dataset_rows, dataset_fields
    )

    taxonomy_counts = Counter(
        row["wrong_trajectory_taxonomy"]
        for row in sample_event_rows
        if not row["current_dense_correct"]
        and row["comparison_usable"]
    )
    taxonomy_by_dataset = {
        dataset: dict(
            Counter(
                row["wrong_trajectory_taxonomy"]
                for row in sample_event_rows
                if row["dataset"] == dataset
                and not row["current_dense_correct"]
                and row["comparison_usable"]
            )
        )
        for dataset in DATASETS
    }
    correct_stats = _event_stats(correct_events)
    wrong_stats = _event_stats(wrong_events)
    coverage_levels = [
        float(value) for value in config["population_reporting"]["coverage_levels"]
    ]
    coverage = {
        str(level): {
            "correct_emergence": _coverage_layer(correct_events, level),
            "wrong_crossover": _coverage_layer(wrong_events, level),
        }
        for level in coverage_levels
    }
    summary = {
        "schema_version": "dense_answer_logit_emergence_v2_summary_v1",
        "contract_sha256": contract["contract_sha256"],
        "position_sanity": gate,
        "records": len(rows),
        "correct": len(correct),
        "wrong": len(wrong),
        "full_answer_start_layer27_top1_matches": sum(
            row["layer27_top1_matches_stored_generation"] for row in rows
        ),
        "full_answer_start_layer27_top1_match_rate": sum(
            row["layer27_top1_matches_stored_generation"] for row in rows
        )
        / len(rows),
        "wrong_comparisons_usable": len(wrong_usable),
        "wrong_comparisons_unusable": len(wrong_unusable),
        "teacher_forced_wrong_comparisons": sum(
            bool(row.get("teacher_forced_prefix_ids")) for row in wrong_usable
        ),
        "correct_canonical_gt_first_token_mismatches": sum(
            row["gt_token_id"] != row["stored_first_generated_token_id"]
            for row in correct
        ),
        "correct_emergence": correct_stats,
        "wrong_crossover": wrong_stats,
        "coverage_layers": coverage,
        "wrong_taxonomy": dict(sorted(taxonomy_counts.items())),
        "wrong_taxonomy_by_dataset": taxonomy_by_dataset,
        "wrong_unusable_reasons": dict(
            Counter(row.get("comparison_unusable_reason") for row in wrong_unusable)
        ),
        "selected_layer_curves": {
            "correct": {
                str(layer): correct_layer_rows[layer]
                for layer in config["population_reporting"]["selected_curve_layers"]
            },
            "wrong": {
                str(layer): wrong_layer_rows[layer]
                for layer in config["population_reporting"]["selected_curve_layers"]
            },
        },
    }
    write_json_atomic(output_root / "analysis_results.json", summary)
    _plot_v2(
        output_root,
        correct_layer_rows,
        wrong_layer_rows,
        correct_events,
        wrong_events,
    )

    def pct(count: int, total: int) -> str:
        return f"{count / total:.1%}" if total else "n/a"

    selected_layers = [
        int(value)
        for value in config["population_reporting"]["selected_curve_layers"]
    ]
    curve_lines = [
        f"- Layer {layer}: correct GT-minus-competitor mean/median "
        f"{correct_layer_rows[layer]['mean_margin']:.3f}/{correct_layer_rows[layer]['median_margin']:.3f}; "
        f"wrong GT-minus-generated mean/median "
        f"{wrong_layer_rows[layer]['mean_margin']:.3f}/{wrong_layer_rows[layer]['median_margin']:.3f}."
        for layer in selected_layers
    ]
    taxonomy_lines = [
        f"- `{name}`: {count:,}/{len(wrong_usable):,} ({pct(count, len(wrong_usable))})"
        for name, count in sorted(taxonomy_counts.items())
    ]
    q2 = (
        f"median layer {_format_event(correct_stats['median'])} "
        f"(IQR {_format_event(correct_stats['q25'])}–{_format_event(correct_stats['q75'])}); "
        f"no persistent emergence in {len(correct) - int(correct_stats['defined']):,}/{len(correct):,}."
    )
    q3 = (
        f"median layer {_format_event(wrong_stats['median'])} "
        f"(IQR {_format_event(wrong_stats['q25'])}–{_format_event(wrong_stats['q75'])}); "
        f"no persistent crossover in {len(wrong_usable) - int(wrong_stats['defined']):,}/{len(wrong_usable):,} usable comparisons."
    )
    report = "\n".join(
        [
            "# Corrected dense answer-logit emergence analysis",
            "",
            f"- Frozen contract: `{contract['contract_sha256']}`",
            f"- Population: {len(rows):,} current-runtime dense records ({len(correct):,} correct, {len(wrong):,} wrong).",
            f"- Wrong comparisons: {len(wrong_usable):,} usable, {len(wrong_unusable):,} excluded with logged reasons; {summary['teacher_forced_wrong_comparisons']:,} use a teacher-forced shared prefix.",
            "- This report supersedes no conclusions from the invalid `text_final` readout; those results are not reused here.",
            "",
            "## Position validation",
            "",
            f"The final prompt token after `<|im_end|>\\n<|im_start|>assistant\\n` reproduced the stored generated first token for {gate['layer27_top1_matches']}/{gate['records']} sanity samples ({gate['exact_match_rate']:.2%}) and {summary['full_answer_start_layer27_top1_matches']}/{len(rows)} full-population samples ({summary['full_answer_start_layer27_top1_match_rate']:.2%}). Cached shared-prefix plus next-token replay matched {gate['teacher_forced_cached_replay_matches']}/{gate['teacher_forced_divergence_records']} sanity cases; raw readouts matched raw generation logits in {gate['teacher_forced_raw_readout_matches_raw_model_logits']}/{gate['teacher_forced_divergence_records']}.",
            "",
            "## Main curves",
            "",
            *curve_lines,
            "",
            "## Persistent emergence and crossover",
            "",
            f"- Correct emergence: {q2}",
            f"- Wrong crossover: {q3}",
            f"- Population 50% coverage: correct layer {_format_event(coverage['0.5']['correct_emergence'])}, wrong layer {_format_event(coverage['0.5']['wrong_crossover'])}.",
            f"- Population 75% coverage: correct layer {_format_event(coverage['0.75']['correct_emergence'])}, wrong layer {_format_event(coverage['0.75']['wrong_crossover'])}.",
            "",
            "## Wrong trajectory taxonomy",
            "",
            *taxonomy_lines,
            "",
            "## Final questions",
            "",
            f"1. Layer 27 at the true answer-start position reproduced {gate['layer27_top1_matches']}/{gate['records']} stored first generated tokens in the stratified gate and {summary['full_answer_start_layer27_top1_matches']}/{len(rows)} over the full population ({summary['full_answer_start_layer27_top1_match_rate']:.2%}).",
            f"2. {q2}",
            f"3. {q3}",
            f"4. The fixed taxonomy proportions are listed above over the {len(wrong_usable):,} usable wrong comparisons; {len(wrong_unusable):,} unusable comparisons are excluded rather than called ambiguous.",
            f"5. At layer 0, only {correct_layer_rows[0]['positive_margin_fraction']:.1%} of correct examples prefer the canonical GT token, while {wrong_layer_rows[0]['negative_margin_fraction']:.1%} of usable wrong examples prefer the eventual generated token. This fixed-head lens is evidence about answer-token separability, not proof that the full early hidden state cannot support failure prediction.",
            f"6. The 50% and 75% persistent-coverage depths above provide descriptive candidate bounds. This analysis does not choose a supervision start layer; that strategy decision remains separately authorized.",
            "",
            "## Reference and interpretation limits",
            "",
            f"Canonical GT first-token and generated first-token differ in {summary['correct_canonical_gt_first_token_mismatches']:,}/{len(correct):,} evaluator-correct records. Those records remain in the requested canonical-GT analysis and are explicitly auditable.",
            "Raw logits from the frozen final norm and LM head are not calibrated probabilities. Multi-reference and evaluator-equivalent answers can differ tokenically from the canonical GT string.",
            "",
            "## Stop status",
            "",
            "The corrected answer-position analysis is complete. No Stage-1 predictor training, supervision-range choice, W→C work, four-action routing, MCTS, or external evaluation was performed.",
        ]
    ) + "\n"
    write_text_atomic(output_root / "analysis_summary.md", report)
    print(
        json.dumps(
            {
                "records": len(rows),
                "wrong_usable": len(wrong_usable),
                "correct_emergence": correct_stats,
                "wrong_crossover": wrong_stats,
                "wrong_taxonomy": dict(taxonomy_counts),
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("freeze", "sanity", "aggregate-sanity", "full", "aggregate"),
        required=True,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "freeze":
        freeze(args)
    elif args.mode in {"sanity", "full"}:
        worker(args)
    elif args.mode == "aggregate-sanity":
        aggregate_sanity(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
