#!/usr/bin/env python3
"""Evaluate one validation-selected four-action Image+Question predictor."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import yaml

from binary_policy.executor import (
    capture_four_action_route,
    greedy_generate_from_cached_prompt,
)
from binary_policy.executor.inputs import build_binary_inputs
from binary_policy.predictor import FrozenHFTokenEncoder
from experiments.train_binary_polar import file_sha256
from four_action_policy.actions import decode_action_indices
from four_action_policy.external import (
    ACTIVE_BENCHMARKS,
    TOTAL_RECORDS,
    action_statistics,
    load_active_rows,
    select_shard,
)
from four_action_policy.predictor import FourActionPolarBackbone
from label_regeneration.runtime import configure_determinism, load_frozen_model


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def native_generation_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
        if key != "instruction_token_mask"
    }


def decode_generated(processor, token_ids: torch.Tensor) -> str:
    return processor.batch_decode(
        token_ids.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def choose_preflight_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected = []
    for benchmark in ACTIVE_BENCHMARKS:
        candidates = [row for row in rows if row["benchmark"] == benchmark]
        candidates.sort(
            key=lambda row: sha256(
                f"{seed}:four-action-external-preflight:{row['uid']}".encode()
            ).hexdigest()
        )
        selected.append(candidates[0])
    return selected


def score_prediction(row: dict[str, Any], prediction: str) -> tuple[float, bool]:
    from dvr_qwen.eval_metrics import score_prediction as reference_score

    score = float(
        reference_score(
            str(row["metric_name"]),
            prediction,
            row.get("answer"),
            row.get("all_answer_norms"),
        )
    )
    return score, bool(score >= float(row["correctness_threshold"]))


def build_eval_inputs(processor, row: dict[str, Any]) -> dict[str, Any]:
    from dvr_qwen.scripts.cache_preference_gt_router_features import build_processor_inputs

    return build_processor_inputs(processor, row, data_root=Path(row["data_root"]))


def load_predictor(
    config: dict[str, Any],
    selection_path: Path,
    device: torch.device,
    *,
    expected_config_sha256: str,
):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selected_before_external_evaluation") is not True:
        raise RuntimeError("checkpoint selection was not frozen before external evaluation")
    if selection.get("config_sha256") != expected_config_sha256:
        raise RuntimeError("checkpoint selection belongs to another training config")
    checkpoint_path = Path(selection["best_checkpoint"])
    if file_sha256(checkpoint_path) != selection["best_checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint checksum mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config_sha256") != expected_config_sha256:
        raise RuntimeError("selected checkpoint belongs to another training config")
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_path, padding_side="left", local_files_only=True
    )
    encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device).eval()
    predictor = FourActionPolarBackbone(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=encoder.output_dim,
        image_dim=int(config["visual_features"]["feature_width"]),
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    ).to(device).eval()
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    return tokenizer, encoder, predictor, checkpoint_path, selection


@torch.inference_mode()
def predict_actions(
    *,
    row: dict[str, Any],
    visual_rows: torch.Tensor,
    tokenizer,
    encoder,
    predictor,
    max_question_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    encoded = tokenizer(
        [row["predictor_text"]],
        padding=True,
        truncation=True,
        max_length=max_question_tokens,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention = encoded["attention_mask"].to(device)
    question = encoder(input_ids, attention)
    image = visual_rows.to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    image_attention = torch.ones(image.shape[:2], dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = predictor(question, attention, image, image_attention)
    indices = logits[0].argmax(dim=-1)
    actions = decode_action_indices(indices)
    return {
        "actions": actions,
        "logits": logits[0].float().cpu().tolist(),
        **action_statistics(actions),
    }


@torch.inference_mode()
def execute_actions(
    *,
    wrapped_model,
    processor,
    inputs: dict[str, Any],
    prepared,
    actions,
    row: dict[str, Any],
    eos_token_ids: list[int],
    repetition_penalty: float,
) -> dict[str, Any]:
    captured = capture_four_action_route(
        wrapped_model,
        inputs,
        actions,
        prepared_inputs=prepared,
        use_cache=True,
    )
    if captured.cache is None:
        raise RuntimeError("four-action prompt capture did not produce a KV cache")
    generation = greedy_generate_from_cached_prompt(
        wrapped_model,
        captured.prompt_logits,
        captured.inputs,
        captured.cache,
        inputs["input_ids"],
        max_new_tokens=int(row["max_new_tokens"]),
        eos_token_ids=eos_token_ids,
        repetition_penalty=repetition_penalty,
    )
    prediction = decode_generated(processor, generation.generated_ids)
    score, correct = score_prediction(row, prediction)
    return {
        "generated_ids": generation.generated_ids.detach().cpu().view(-1).tolist(),
        "prediction": prediction,
        "score": score,
        "correct": correct,
        "execution_source": "live_unified_four_action_executor",
    }


def process_row(
    *,
    row: dict[str, Any],
    wrapped_model,
    processor,
    tokenizer,
    encoder,
    predictor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    inputs = build_eval_inputs(processor, row)
    prepared = build_binary_inputs(wrapped_model, inputs)
    visual = prepared.visual_states[0, prepared.visual_valid_mask[0]]
    prediction = predict_actions(
        row=row,
        visual_rows=visual,
        tokenizer=tokenizer,
        encoder=encoder,
        predictor=predictor,
        max_question_tokens=int(config["data"]["max_question_tokens"]),
        device=device,
    )
    actions = tuple(prediction.pop("actions"))
    generation_config = config["external_evaluation"]
    execution = execute_actions(
        wrapped_model=wrapped_model,
        processor=processor,
        inputs=inputs,
        prepared=prepared,
        actions=actions,
        row=row,
        eos_token_ids=list(generation_config["eos_token_ids"]),
        repetition_penalty=float(generation_config["repetition_penalty"]),
    )
    full_actions = ("FULL",) * int(config["policy"]["num_layers"])
    baseline = (
        execution
        if actions == full_actions
        else execute_actions(
            wrapped_model=wrapped_model,
            processor=processor,
            inputs=inputs,
            prepared=prepared,
            actions=full_actions,
            row=row,
            eos_token_ids=list(generation_config["eos_token_ids"]),
            repetition_penalty=float(generation_config["repetition_penalty"]),
        )
    )
    benchmark = str(row["benchmark"])
    suite = (
        "chartqa"
        if benchmark == "chartqa"
        else "mmmu_pro"
        if benchmark.startswith("mmmu_pro_")
        else "pope"
    )
    return {
        "uid": row["uid"],
        "sample_id": row["sample_id"],
        "benchmark": benchmark,
        "suite": suite,
        "cluster_key": row["cluster_key"],
        "metric_name": row["metric_name"],
        "correctness_threshold": float(row["correctness_threshold"]),
        "baseline_prediction": baseline["prediction"],
        "baseline_score": baseline["score"],
        "baseline_correct": baseline["correct"],
        "baseline_generated_ids": baseline["generated_ids"],
        "baseline_source": "current_live_unified_full",
        "predictor_text_sha256": sha256(row["predictor_text"].encode()).hexdigest(),
        "visual_tokens": int(visual.shape[0]),
        "predicted": {
            "actions": list(actions),
            **prediction,
            **execution,
        },
    }


@torch.inference_mode()
def native_unified_full_parity(
    *, row, base_model, wrapped_model, processor, config
) -> dict[str, Any]:
    inputs = build_eval_inputs(processor, row)
    prepared = build_binary_inputs(wrapped_model, inputs)
    generation_config = config["external_evaluation"]
    unified = execute_actions(
        wrapped_model=wrapped_model,
        processor=processor,
        inputs=inputs,
        prepared=prepared,
        actions=("FULL",) * int(config["policy"]["num_layers"]),
        row=row,
        eos_token_ids=list(generation_config["eos_token_ids"]),
        repetition_penalty=float(generation_config["repetition_penalty"]),
    )
    model_device = next(base_model.parameters()).device
    base_model.rope_deltas = None
    native = base_model.generate(
        **native_generation_inputs(inputs, model_device),
        max_new_tokens=int(row["max_new_tokens"]),
        do_sample=False,
        use_cache=True,
        eos_token_id=list(generation_config["eos_token_ids"]),
        repetition_penalty=float(generation_config["repetition_penalty"]),
    )
    native_ids = native[:, inputs["input_ids"].shape[1] :]
    native_prediction = decode_generated(processor, native_ids)
    native_score, native_correct = score_prediction(row, native_prediction)
    return {
        "uid": row["uid"],
        "benchmark": row["benchmark"],
        "token_sequence_equal": unified["generated_ids"]
        == native_ids.detach().cpu().view(-1).tolist(),
        "prediction_equal": unified["prediction"] == native_prediction,
        "evaluator_correctness_equal": unified["correct"] == native_correct,
        "unified_prediction": unified["prediction"],
        "native_prediction": native_prediction,
        "unified_score": unified["score"],
        "native_score": native_score,
        "score_drift": unified["score"] - native_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--preflight-path")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = Path(config["external_evaluation"]["protocol"])
    sys.path.insert(0, str(protocol / "code"))
    rows = load_active_rows(config["external_evaluation"]["data_root"])
    configure_determinism(args.seed)
    processor, base_model, wrapped_model, device = load_frozen_model(
        config["base_model"]["path"],
        config["base_model"]["revision"],
        args.device_index,
    )
    selection_path = Path(args.selection)
    tokenizer, encoder, predictor, checkpoint_path, selection = load_predictor(
        config,
        selection_path,
        device,
        expected_config_sha256=file_sha256(config_path),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "preflight":
        selected = choose_preflight_rows(rows, args.seed)
        parity = []
        determinism = []
        for row in tqdm(selected, desc="four-action external preflight", unit="sample"):
            parity.append(
                native_unified_full_parity(
                    row=row,
                    base_model=base_model,
                    wrapped_model=wrapped_model,
                    processor=processor,
                    config=config,
                )
            )
            first = process_row(
                row=row,
                wrapped_model=wrapped_model,
                processor=processor,
                tokenizer=tokenizer,
                encoder=encoder,
                predictor=predictor,
                config=config,
                device=device,
            )
            second = process_row(
                row=row,
                wrapped_model=wrapped_model,
                processor=processor,
                tokenizer=tokenizer,
                encoder=encoder,
                predictor=predictor,
                config=config,
                device=device,
            )
            determinism.append(
                {
                    "uid": row["uid"],
                    "predicted_execution_exact": first["predicted"] == second["predicted"],
                    "baseline_exact": all(
                        first[key] == second[key]
                        for key in (
                            "baseline_prediction",
                            "baseline_score",
                            "baseline_correct",
                            "baseline_generated_ids",
                        )
                    ),
                }
            )
        passed = all(
            item["token_sequence_equal"]
            and item["prediction_equal"]
            and item["evaluator_correctness_equal"]
            for item in parity
        ) and all(
            item["predicted_execution_exact"] and item["baseline_exact"]
            for item in determinism
        )
        payload = {
            "schema_version": "four_action_polar_external_preflight_v1",
            "passed": passed,
            "active_records": TOTAL_RECORDS,
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "selection": str(selection_path),
            "selection_sha256": file_sha256(selection_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "best_epoch": int(selection["best_epoch"]),
            "native_vs_unified_full": parity,
            "determinism": determinism,
        }
        path = output_dir / "preflight_v1.json"
        write_json(path, payload)
        path.with_suffix(".json.sha256").write_text(
            f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
        )
        if not passed:
            raise RuntimeError("four-action external evaluation preflight failed")
        return

    preflight_path = Path(args.preflight_path or output_dir / "preflight_v1.json")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("passed") is not True
        or preflight.get("config_sha256") != file_sha256(config_path)
        or preflight.get("selection_sha256") != file_sha256(selection_path)
        or preflight.get("checkpoint_sha256") != file_sha256(checkpoint_path)
    ):
        raise RuntimeError("full external evaluation requires its exact passed preflight")
    rows = select_shard(
        rows, num_shards=args.num_shards, shard_index=args.shard_index
    )
    shard_dir = output_dir / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    if shard_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite external shard: {shard_dir}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = shard_dir / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"external shard is already complete: {shard_dir}")
    existing_parts = sorted(shard_dir.glob("part_*.jsonl"))
    existing_rows = [row for path in existing_parts for row in load_jsonl(path)]
    completed_uids = {str(row["uid"]) for row in existing_rows}
    if len(completed_uids) != len(existing_rows):
        raise RuntimeError("resumable shard has duplicate UIDs")
    expected_uids = {str(row["uid"]) for row in rows}
    if not completed_uids <= expected_uids:
        raise RuntimeError("resumable shard contains unexpected UIDs")
    remaining = [row for row in rows if str(row["uid"]) not in completed_uids]
    buffer = []
    part = len(existing_parts)
    started = time.time()
    for row in tqdm(remaining, desc=f"four-action eval shard {args.shard_index}", unit="sample"):
        buffer.append(
            process_row(
                row=row,
                wrapped_model=wrapped_model,
                processor=processor,
                tokenizer=tokenizer,
                encoder=encoder,
                predictor=predictor,
                config=config,
                device=device,
            )
        )
        if len(buffer) >= args.chunk_size:
            write_jsonl(shard_dir / f"part_{part:05d}.jsonl", buffer)
            buffer.clear()
            part += 1
    if buffer:
        write_jsonl(shard_dir / f"part_{part:05d}.jsonl", buffer)
    metadata = {
        "schema_version": "four_action_polar_external_shard_v1",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "records": len(rows),
        "resumed_records": len(completed_uids),
        "elapsed_seconds": time.time() - started,
        "config_sha256": file_sha256(config_path),
        "selection_sha256": file_sha256(selection_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "preflight_sha256": file_sha256(preflight_path),
        "source_sha256": file_sha256(Path(__file__)),
    }
    write_json(metadata_path, metadata)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
