#!/usr/bin/env python3
"""Evaluate the validation-selected online router on the frozen three suites."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm.auto import tqdm
import yaml

from binary_policy.executor import capture_four_action_route, greedy_generate_from_cached_prompt
from binary_policy.executor.inputs import build_binary_inputs
from experiments.train_binary_polar import file_sha256
from four_action_online_router.model import OnlineFourActionRouter
from four_action_online_router.runtime import capture_online_router_route
from four_action_policy.external import (
    ACTIVE_BENCHMARKS,
    TOTAL_RECORDS,
    action_statistics,
    load_active_rows,
    select_shard,
)
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def decode_generated(processor, token_ids: torch.Tensor) -> str:
    return processor.batch_decode(
        token_ids.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def score_prediction(row: dict[str, Any], prediction: str) -> tuple[float, bool]:
    from dvr_qwen.eval_metrics import score_prediction as reference_score

    score = float(
        reference_score(
            str(row["metric_name"]), prediction, row.get("answer"),
            row.get("all_answer_norms")
        )
    )
    return score, bool(score >= float(row["correctness_threshold"]))


def build_eval_inputs(processor, row: dict[str, Any]) -> dict[str, Any]:
    from dvr_qwen.scripts.cache_preference_gt_router_features import build_processor_inputs

    return build_processor_inputs(processor, row, data_root=Path(row["data_root"]))


def make_router(config: dict[str, Any], device: torch.device) -> OnlineFourActionRouter:
    values = config["router"]
    return OnlineFourActionRouter(
        hidden_size=int(values["hidden_size"]), num_layers=int(values["num_layers"]),
        d_router=int(values["d_router"]), num_heads=int(values["num_heads"]),
        mlp_hidden_size=int(values["mlp_hidden_size"]), dropout=float(values["dropout"]),
        interaction_scale=float(values["interaction_scale"]),
    ).to(device).eval()


def load_router(config, selection_path: Path, device, *, config_sha256: str):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selected_before_external_evaluation") is not True:
        raise RuntimeError("router checkpoint was not selected before external evaluation")
    if selection.get("config_sha256") != config_sha256:
        raise RuntimeError("router selection belongs to another config")
    checkpoint_path = Path(selection["best_checkpoint"])
    if file_sha256(checkpoint_path) != selection["best_checkpoint_sha256"]:
        raise RuntimeError("selected online-router checkpoint checksum mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != config_sha256:
        raise RuntimeError("online-router checkpoint belongs to another config")
    router = make_router(config, device)
    router.load_state_dict(payload["router"], strict=True)
    return router, selection, checkpoint_path


@torch.inference_mode()
def execute_capture(
    *, wrapped_model, processor, inputs, prepared, captured, row, generation_config
) -> dict[str, Any]:
    if captured.cache is None:
        raise RuntimeError("four-action capture did not produce a prompt cache")
    generation = greedy_generate_from_cached_prompt(
        wrapped_model, captured.prompt_logits, captured.inputs, captured.cache,
        inputs["input_ids"], max_new_tokens=int(row["max_new_tokens"]),
        eos_token_ids=list(generation_config["eos_token_ids"]),
        repetition_penalty=float(generation_config["repetition_penalty"]),
    )
    prediction = decode_generated(processor, generation.generated_ids)
    score, correct = score_prediction(row, prediction)
    return {
        "prediction": prediction, "score": score, "correct": correct,
        "generated_ids": generation.generated_ids.detach().cpu().view(-1).tolist(),
    }


@torch.inference_mode()
def process_row(*, row, wrapped_model, processor, router, config) -> dict[str, Any]:
    inputs = build_eval_inputs(processor, row)
    prepared = build_binary_inputs(wrapped_model, inputs)
    routed_capture = capture_online_router_route(
        wrapped_model, inputs, router, prepared_inputs=prepared,
        amp_dtype=torch.bfloat16, use_cache=True
    )
    actions = tuple(routed_capture.layer_actions or ())
    generation_config = config["external_evaluation"]
    predicted = execute_capture(
        wrapped_model=wrapped_model, processor=processor, inputs=inputs,
        prepared=prepared, captured=routed_capture, row=row,
        generation_config=generation_config
    )
    full_actions = ("FULL",) * int(config["router"]["num_layers"])
    if actions == full_actions:
        baseline = predicted
    else:
        full_capture = capture_four_action_route(
            wrapped_model, inputs, full_actions, prepared_inputs=prepared, use_cache=True
        )
        baseline = execute_capture(
            wrapped_model=wrapped_model, processor=processor, inputs=inputs,
            prepared=prepared, captured=full_capture, row=row,
            generation_config=generation_config
        )
    benchmark = str(row["benchmark"])
    suite = "chartqa" if benchmark == "chartqa" else "mmmu_pro" if benchmark.startswith("mmmu_pro_") else "pope"
    return {
        "uid": row["uid"], "sample_id": row["sample_id"],
        "benchmark": benchmark, "suite": suite, "cluster_key": row["cluster_key"],
        "metric_name": row["metric_name"],
        "correctness_threshold": float(row["correctness_threshold"]),
        "baseline_prediction": baseline["prediction"], "baseline_score": baseline["score"],
        "baseline_correct": baseline["correct"], "baseline_generated_ids": baseline["generated_ids"],
        "baseline_source": "current_live_unified_full",
        "predictor_text_sha256": sha256(row["predictor_text"].encode()).hexdigest(),
        "visual_tokens": int(prepared.visual_valid_mask.sum().item()),
        "predicted": {"actions": list(actions), **action_statistics(actions), **predicted},
    }


def choose_preflight_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected = []
    for benchmark in ACTIVE_BENCHMARKS:
        candidates = [row for row in rows if row["benchmark"] == benchmark]
        candidates.sort(
            key=lambda row: sha256(f"{seed}:online-router-preflight:{row['uid']}".encode()).hexdigest()
        )
        selected.append(candidates[0])
    return selected


def native_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items() if key != "instruction_token_mask"
    }


@torch.inference_mode()
def native_full_parity(*, row, base_model, wrapped_model, processor, config):
    inputs = build_eval_inputs(processor, row)
    prepared = build_binary_inputs(wrapped_model, inputs)
    full = capture_four_action_route(
        wrapped_model, inputs, ("FULL",) * int(config["router"]["num_layers"]),
        prepared_inputs=prepared, use_cache=True
    )
    unified = execute_capture(
        wrapped_model=wrapped_model, processor=processor, inputs=inputs,
        prepared=prepared, captured=full, row=row,
        generation_config=config["external_evaluation"]
    )
    device = next(base_model.parameters()).device
    base_model.rope_deltas = None
    generated = base_model.generate(
        **native_inputs(inputs, device), max_new_tokens=int(row["max_new_tokens"]),
        do_sample=False, use_cache=True,
        eos_token_id=list(config["external_evaluation"]["eos_token_ids"]),
        repetition_penalty=float(config["external_evaluation"]["repetition_penalty"]),
    )
    native_ids = generated[:, inputs["input_ids"].shape[1] :]
    native_prediction = decode_generated(processor, native_ids)
    native_score, native_correct = score_prediction(row, native_prediction)
    return {
        "uid": row["uid"], "benchmark": row["benchmark"],
        "token_sequence_equal": unified["generated_ids"] == native_ids.detach().cpu().view(-1).tolist(),
        "prediction_equal": unified["prediction"] == native_prediction,
        "evaluator_correctness_equal": unified["correct"] == native_correct,
        "unified_prediction": unified["prediction"], "native_prediction": native_prediction,
        "unified_score": unified["score"], "native_score": native_score,
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
    config_sha = file_sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(config["external_evaluation"]["protocol"]) / "code"))
    rows = load_active_rows(config["external_evaluation"]["data_root"])
    if len(rows) != int(config["external_evaluation"]["expected_records"]):
        raise RuntimeError("external evaluation population differs from config")
    configure_determinism(args.seed)
    processor, base_model, wrapped_model, device = load_frozen_model(
        config["base_model"]["path"], config["base_model"]["revision"], args.device_index
    )
    router, selection, checkpoint_path = load_router(
        config, Path(args.selection), device, config_sha256=config_sha
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "preflight":
        parity, determinism = [], []
        for row in tqdm(choose_preflight_rows(rows, args.seed), desc="online router preflight", unit="sample"):
            parity.append(native_full_parity(
                row=row, base_model=base_model, wrapped_model=wrapped_model,
                processor=processor, config=config
            ))
            first = process_row(row=row, wrapped_model=wrapped_model, processor=processor, router=router, config=config)
            second = process_row(row=row, wrapped_model=wrapped_model, processor=processor, router=router, config=config)
            determinism.append({
                "uid": row["uid"], "predicted_execution_exact": first["predicted"] == second["predicted"],
                "baseline_exact": all(first[key] == second[key] for key in (
                    "baseline_prediction", "baseline_score", "baseline_correct", "baseline_generated_ids"
                )),
            })
        passed = all(
            item["token_sequence_equal"] and item["prediction_equal"] and item["evaluator_correctness_equal"]
            for item in parity
        ) and all(item["predicted_execution_exact"] and item["baseline_exact"] for item in determinism)
        payload = {
            "schema_version": "four_action_online_router_external_preflight_v1",
            "passed": passed, "active_records": TOTAL_RECORDS,
            "config": str(config_path), "config_sha256": config_sha,
            "selection": str(args.selection), "selection_sha256": file_sha256(Path(args.selection)),
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
            "best_epoch": int(selection["best_epoch"]),
            "native_vs_unified_full": parity, "determinism": determinism,
        }
        path = output_dir / "preflight_v1.json"
        write_json(path, payload)
        if not passed:
            raise RuntimeError("online-router external evaluation preflight failed")
        return

    preflight_path = Path(args.preflight_path or output_dir / "preflight_v1.json")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("passed") is not True or preflight.get("config_sha256") != config_sha
        or preflight.get("selection_sha256") != file_sha256(Path(args.selection))
        or preflight.get("checkpoint_sha256") != file_sha256(checkpoint_path)
    ):
        raise RuntimeError("full evaluation requires the exact passed preflight")
    rows = select_shard(rows, num_shards=args.num_shards, shard_index=args.shard_index)
    shard_dir = output_dir / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    if shard_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite evaluation shard: {shard_dir}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = shard_dir / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"evaluation shard is already complete: {shard_dir}")
    parts = sorted(shard_dir.glob("part_*.jsonl"))
    existing = [row for path in parts for row in load_jsonl(path)]
    complete = {str(row["uid"]) for row in existing}
    expected = {str(row["uid"]) for row in rows}
    if len(complete) != len(existing) or not complete <= expected:
        raise RuntimeError("resumable evaluation shard has duplicate or unexpected UIDs")
    remaining = [row for row in rows if str(row["uid"]) not in complete]
    buffer: list[dict[str, Any]] = []
    part = len(parts)
    started = time.time()
    for row in tqdm(remaining, desc=f"online router shard {args.shard_index}", unit="sample"):
        buffer.append(process_row(
            row=row, wrapped_model=wrapped_model, processor=processor, router=router, config=config
        ))
        if len(buffer) >= args.chunk_size:
            write_jsonl(shard_dir / f"part_{part:05d}.jsonl", buffer)
            buffer.clear()
            part += 1
    if buffer:
        write_jsonl(shard_dir / f"part_{part:05d}.jsonl", buffer)
    write_json(metadata_path, {
        "schema_version": "four_action_online_router_external_shard_v1",
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "records": len(rows), "resumed_records": len(complete),
        "elapsed_seconds": time.time() - started,
        "config_sha256": config_sha,
        "selection_sha256": file_sha256(Path(args.selection)),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "preflight_sha256": file_sha256(preflight_path),
        "source_sha256": file_sha256(Path(__file__)),
    })


if __name__ == "__main__":
    main()
