#!/usr/bin/env python3
"""Collect all-on/off anchors and full greedy permutation traces per sample."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(os.environ.get("VISUAL_INJECTION_ROOT", Path(__file__).resolve().parents[2])).resolve()
ANALYSIS_DIR = ROOT / "analysis_outputs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ANALYSIS_DIR))

os.environ.setdefault("HF_HOME", "/home/hyemin/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/home/hyemin/.cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/hyemin/.cache/huggingface/hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TMPDIR", str(ROOT / "state" / "tmp"))

from harmful_validation_common import HF_HUB_CACHE, MODEL_SOURCE, is_correct, mask_one_based  # noqa: E402
from run_harmful_interventions import (  # noqa: E402
    build_processor_inputs,
    evaluate_route as evaluate_binary_route,
    load_model,
    prepare_binary_dvrc_inputs,
)


DEFAULT_MANIFEST = ROOT / "10k_dataset_mask" / "manifests" / "all_samples.jsonl"
DEFAULT_CONFIG = ROOT / "10k_dataset_mask" / "config" / "collection_config.json"
DEFAULT_OUTPUT = ROOT / "10k_dataset_mask" / "raw" / "search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["gate", "search"], default="search")
    parser.add_argument("--gate-summary", type=Path, default=ROOT / "10k_dataset_mask" / "gate" / "summary.json")
    parser.add_argument("--gate-per-cell", type=int, default=2)
    parser.add_argument(
        "--require-source-anchor-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the current model score to match the manifest's source-model score.",
    )
    parser.add_argument(
        "--require-saved-generated-id-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require any saved source-model generated IDs to match the current model.",
    )
    parser.add_argument("--data-splits", default="train,validation")
    parser.add_argument("--benchmarks", default="chartqa,docvqa,gqa,textvqa")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--model-source", type=Path, default=MODEL_SOURCE)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--processor-use-fast",
        choices=["auto", "true", "false"],
        default="false",
        help="Use the source-compatible slow processor by default; auto preserves AutoProcessor defaults.",
    )
    parser.add_argument("--first-gpu-max-memory-gb", type=int, default=30)
    parser.add_argument("--other-gpu-max-memory-gb", type=int, default=46)
    parser.add_argument("--cpu-max-memory-gb", type=int, default=64)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def route_key(route: list[int]) -> str:
    return "".join("1" if int(value) else "0" for value in route)


def route_id(uid: str, route: list[int]) -> str:
    digest = hashlib.sha256(f"{uid}:{route_key(route)}".encode("utf-8")).hexdigest()[:16]
    return f"{uid}:mask:{digest}"


def safe_filename(uid: str) -> str:
    return uid.replace(":", "__").replace("/", "_") + ".json"


def layer_order(name: str, num_layers: int, uid: str) -> list[int]:
    if name == "early_to_late":
        return list(range(num_layers))
    if name == "late_to_early":
        return list(range(num_layers - 1, -1, -1))
    if name == "center_out":
        left = (num_layers - 1) // 2
        right = num_layers // 2
        order: list[int] = []
        while left >= 0 or right < num_layers:
            if left >= 0:
                order.append(left)
            if right < num_layers and right != left:
                order.append(right)
            left -= 1
            right += 1
        return order
    if name == "outside_in":
        order = []
        left, right = 0, num_layers - 1
        while left <= right:
            order.append(left)
            if right != left:
                order.append(right)
            left += 1
            right -= 1
        return order
    if name.startswith("random:"):
        seed = int(name.split(":", 1)[1])
        order = list(range(num_layers))
        random.Random(f"{seed}:{uid}").shuffle(order)
        return order
    raise ValueError(f"unsupported order: {name}")


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    splits = {value.strip() for value in args.data_splits.split(",") if value.strip()}
    benchmarks = {value.strip().lower() for value in args.benchmarks.split(",") if value.strip()}
    selected = [row for row in rows if row["data_split"] in splits and row["benchmark"] in benchmarks]
    selected.sort(key=lambda row: row["uid"])
    if args.mode == "gate":
        counts: Counter[tuple[str, str, str]] = Counter()
        gate_rows = []
        for row in selected:
            key = (row["data_split"], row["benchmark"], row["source_bucket"])
            if counts[key] >= args.gate_per_cell:
                continue
            counts[key] += 1
            gate_rows.append(row)
        selected = gate_rows
    else:
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("invalid shard settings")
        selected = [row for index, row in enumerate(selected) if index % args.num_shards == args.shard_index]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]
    return selected


def run_hf_generate(torch: Any, model: Any, processor: Any, processor_inputs: dict[str, Any], sample: dict[str, Any], device: Any) -> dict[str, Any]:
    device_inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in processor_inputs.items()}
    # Current Transformers Qwen2.5-VL needs mm_token_type_ids to construct
    # multimodal 3D RoPE during prefill. Dropping it silently selects the
    # rope-delta fallback and makes HF full differ from binary all-on.
    generate_inputs = device_inputs
    output = model.base_model.generate(
        **generate_inputs,
        max_new_tokens=int(sample["max_new_tokens"]),
        do_sample=False,
        return_dict_in_generate=True,
    )
    generated = output.sequences[:, generate_inputs["input_ids"].shape[1] :]
    prediction = processor.batch_decode(
        generated.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    from dvr_qwen.eval_metrics import score_prediction

    score = score_prediction(sample["metric_name"], prediction, sample["answer"], sample.get("all_answer_norms"))
    return {
        "generated_ids": generated.detach().cpu().view(-1).tolist(),
        "prediction": prediction,
        "score": float(score),
    }


def evaluate_with_cache(
    *,
    model: Any,
    processor: Any,
    processor_inputs: dict[str, Any],
    prepared: Any,
    sample: dict[str, Any],
    device: Any,
    route: list[int],
    origin: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    key = route_key(route)
    reused = key in cache
    if not reused:
        result = evaluate_binary_route(
            model=model,
            processor=processor,
            processor_inputs=processor_inputs,
            prepared_binary_inputs=prepared,
            sample=sample,
            route=route,
            embed_device=device,
        )
        cache[key] = {
            "route_id": route_id(sample["uid"], route),
            "visual_on_mask": [int(value) for value in route],
            "mask_one_based": mask_one_based(route),
            "num_visual_on_layers": int(sum(route)),
            "prediction": result["prediction"],
            "generated_ids": result["generated_ids"],
            "score": float(result["score"]),
            "result_correct": is_correct(float(result["score"]), float(sample["correctness_threshold"])),
            "cache_lengths_unique": result["cache_lengths_unique"],
            "text_tokens": result["text_tokens"],
            "visual_tokens": result["visual_tokens"],
            "full_prompt_tokens": result["full_prompt_tokens"],
            "origins": [],
        }
    if origin not in cache[key]["origins"]:
        cache[key]["origins"].append(origin)
    return cache[key], reused


def run_gate(args: argparse.Namespace, rows: list[dict[str, Any]], model: Any, processor: Any, device: Any, torch: Any) -> None:
    from dvr_qwen.binary_generate import prepare_binary_dvrc_inputs as prepare

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_rows = []
    for sample in rows:
        try:
            processor_inputs = build_processor_inputs(processor, sample)
            prepared = prepare(model, processor_inputs)
            route_cache: dict[str, dict[str, Any]] = {}
            binary, _ = evaluate_with_cache(
                model=model,
                processor=processor,
                processor_inputs=processor_inputs,
                prepared=prepared,
                sample=sample,
                device=device,
                route=[1] * int(model.config.text_config.num_hidden_layers),
                origin={"family": "anchor", "name": "all_on"},
                cache=route_cache,
            )
            del prepared, route_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            hf = run_hf_generate(torch, model, processor, processor_inputs, sample, device)
            saved_ids = sample.get("saved_generated_ids")
            row = {
                "uid": sample["uid"],
                "data_split": sample["data_split"],
                "benchmark": sample["benchmark"],
                "source_bucket": sample["source_bucket"],
                "current_hf": hf,
                "current_binary_all_on": binary,
                "current_hf_binary_ids_match": hf["generated_ids"] == binary["generated_ids"],
                "current_hf_binary_prediction_match": hf["prediction"] == binary["prediction"],
                "current_hf_binary_score_match": hf["score"] == binary["score"],
                "source_prediction_match": hf["prediction"] == sample["source_full_prediction"],
                "source_score_match": hf["score"] == sample["source_full_score"],
                "saved_generated_ids_available": saved_ids is not None,
                "saved_generated_ids_match": hf["generated_ids"] == saved_ids if saved_ids is not None else None,
            }
        except Exception as exc:
            row = {
                "uid": sample["uid"],
                "data_split": sample["data_split"],
                "benchmark": sample["benchmark"],
                "source_bucket": sample["source_bucket"],
                "error": repr(exc),
            }
        gate_rows.append(row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows_path = output_dir / "generation_anchor_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in gate_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    completed_rows = [row for row in gate_rows if "error" not in row]
    error_rows = [row for row in gate_rows if "error" in row]
    summary = {
        "model_source": str(args.model_source.resolve()),
        "num_layers": int(model.config.text_config.num_hidden_layers),
        "processor_use_fast": args.processor_use_fast,
        "require_source_anchor_match": bool(args.require_source_anchor_match),
        "require_saved_generated_id_match": bool(args.require_saved_generated_id_match),
        "samples_requested": len(gate_rows),
        "samples_completed": len(completed_rows),
        "errors": len(error_rows),
        "current_hf_binary_ids_matches": sum(row["current_hf_binary_ids_match"] for row in completed_rows),
        "current_hf_binary_prediction_matches": sum(row["current_hf_binary_prediction_match"] for row in completed_rows),
        "current_hf_binary_score_matches": sum(row["current_hf_binary_score_match"] for row in completed_rows),
        "source_prediction_matches": sum(row["source_prediction_match"] for row in completed_rows),
        "source_score_matches": sum(row["source_score_match"] for row in completed_rows),
        "saved_generated_ids_available": sum(row["saved_generated_ids_available"] for row in completed_rows),
        "saved_generated_ids_matches": sum(row["saved_generated_ids_match"] is True for row in completed_rows),
        "pass_current_hf_binary_ids": bool(completed_rows) and not error_rows and all(
            row["current_hf_binary_ids_match"] for row in completed_rows
        ),
        "pass_source_prediction_exact": bool(completed_rows) and not error_rows and all(
            row["source_prediction_match"] for row in completed_rows
        ),
        "pass_source_score": bool(completed_rows) and not error_rows and all(
            row["source_score_match"] for row in completed_rows
        ),
        "pass_available_saved_ids": bool(completed_rows) and not error_rows and all(
            not row["saved_generated_ids_available"] or row["saved_generated_ids_match"]
            for row in completed_rows
        ),
        "strict_saved_id_gate_possible": bool(completed_rows) and all(
            row["saved_generated_ids_available"] for row in completed_rows
        ),
        "decision": "pending",
    }
    required_checks_pass = (
        summary["pass_current_hf_binary_ids"]
        and summary["current_hf_binary_prediction_matches"] == summary["samples_completed"]
        and summary["current_hf_binary_score_matches"] == summary["samples_completed"]
        and (not args.require_source_anchor_match or summary["pass_source_score"])
        and (not args.require_saved_generated_id_match or summary["pass_available_saved_ids"])
    )
    if required_checks_pass:
        summary["decision"] = "canonical_current_model_anchor_gate_pass"
    else:
        summary["decision"] = "fail_stop_before_search"
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_search(args: argparse.Namespace, rows: list[dict[str, Any]], model: Any, processor: Any, device: Any, torch: Any, config: dict[str, Any]) -> None:
    from dvr_qwen.generate import generation_policy_record

    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    gate_model_source = gate.get("model_source")
    if gate_model_source and Path(str(gate_model_source)).resolve() != args.model_source.resolve():
        raise RuntimeError(
            f"gate/search model mismatch: gate={gate_model_source}, search={args.model_source.resolve()}"
        )
    gate_num_layers = gate.get("num_layers")
    current_num_layers = int(model.config.text_config.num_hidden_layers)
    if gate_num_layers is not None and int(gate_num_layers) != current_num_layers:
        raise RuntimeError(
            f"gate/search layer-count mismatch: gate={gate_num_layers}, search={current_num_layers}"
        )
    require_source_anchor_match = bool(gate.get("require_source_anchor_match", True))
    require_saved_generated_id_match = bool(gate.get("require_saved_generated_id_match", True))
    if (
        not gate.get("pass_current_hf_binary_ids")
        or gate.get("current_hf_binary_prediction_matches") != gate.get("samples_completed")
        or gate.get("current_hf_binary_score_matches") != gate.get("samples_completed")
        or (require_source_anchor_match and not gate.get("pass_source_score"))
        or (require_saved_generated_id_match and not gate.get("pass_available_saved_ids"))
    ):
        raise RuntimeError(f"generation anchor gate did not pass: {gate}")

    shard_dir = args.output_dir / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    samples_dir = shard_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    errors_path = shard_dir / "errors.jsonl"
    orders = [str(value) for value in config["search"]["orders"]]
    tolerance = float(config["search"]["score_tolerance"])
    num_layers = current_num_layers
    completed = 0
    skipped = 0
    errors = 0
    started = time.time()

    with errors_path.open("a", encoding="utf-8") as error_handle:
        for sample in rows:
            output_path = samples_dir / safe_filename(sample["uid"])
            if output_path.exists():
                skipped += 1
                continue
            try:
                processor_inputs = build_processor_inputs(processor, sample)
                prepared = prepare_binary_dvrc_inputs(model, processor_inputs)
                cache: dict[str, dict[str, Any]] = {}
                all_on, _ = evaluate_with_cache(
                    model=model,
                    processor=processor,
                    processor_inputs=processor_inputs,
                    prepared=prepared,
                    sample=sample,
                    device=device,
                    route=[1] * num_layers,
                    origin={"family": "anchor", "name": "all_on"},
                    cache=cache,
                )
                all_off, _ = evaluate_with_cache(
                    model=model,
                    processor=processor,
                    processor_inputs=processor_inputs,
                    prepared=prepared,
                    sample=sample,
                    device=device,
                    route=[0] * num_layers,
                    origin={"family": "anchor", "name": "all_off", "null_visual_route": True},
                    cache=cache,
                )
                anchor_target_score = float(all_on["score"])
                traces: list[dict[str, Any]] = []
                finals: list[dict[str, Any]] = []
                for order_name in orders:
                    current_route = [1] * num_layers
                    current = all_on
                    accepted_layers: list[int] = []
                    for step, layer_idx in enumerate(layer_order(order_name, num_layers, sample["uid"]), start=1):
                        parent_route = list(current_route)
                        candidate_route = list(current_route)
                        candidate_route[layer_idx] = 0
                        candidate, reused = evaluate_with_cache(
                            model=model,
                            processor=processor,
                            processor_inputs=processor_inputs,
                            prepared=prepared,
                            sample=sample,
                            device=device,
                            route=candidate_route,
                            origin={"family": "permutation_trace", "order": order_name, "step": step},
                            cache=cache,
                        )
                        acceptance_score = max(anchor_target_score, float(current["score"]))
                        accepted = float(candidate["score"]) + tolerance >= acceptance_score
                        traces.append(
                            {
                                "uid": sample["uid"],
                                "order": order_name,
                                "step": step,
                                "tested_layer_zero_based": layer_idx,
                                "tested_layer_one_based": layer_idx + 1,
                                "parent_route_id": route_id(sample["uid"], parent_route),
                                "candidate_route_id": candidate["route_id"],
                                "candidate_execution_reused": reused,
                                "score_before": float(current["score"]),
                                "score_after": float(candidate["score"]),
                                "acceptance_score": acceptance_score,
                                "accepted": accepted,
                            }
                        )
                        if accepted:
                            current_route = candidate_route
                            current = candidate
                            accepted_layers.append(layer_idx + 1)
                    current["origins"].append({"family": "permutation_final", "order": order_name})
                    finals.append(
                        {
                            "order": order_name,
                            "order_layers_one_based": [idx + 1 for idx in layer_order(order_name, num_layers, sample["uid"])],
                            "final_route_id": current["route_id"],
                            "final_mask_one_based": current["mask_one_based"],
                            "final_num_visual_on_layers": current["num_visual_on_layers"],
                            "final_score": current["score"],
                            "final_correct": current["result_correct"],
                            "accepted_removed_layers_one_based": accepted_layers,
                        }
                    )

                payload = {
                    "dataset_version": config["dataset_version"],
                    "phase": "anchors_and_permutation_traces",
                    "sample": sample,
                    "source_anchor_reproduced_by_binary": {
                        "prediction_match": all_on["prediction"] == sample["source_full_prediction"],
                        "score_match": float(all_on["score"]) == float(sample["source_full_score"]),
                    },
                    "target_policy": "current_binary_all_on_score",
                    "target_score": anchor_target_score,
                    "anchors": {"all_on_route_id": all_on["route_id"], "all_off_route_id": all_off["route_id"]},
                    "candidate_executions": sorted(cache.values(), key=lambda row: row["route_id"]),
                    "search_trace": traces,
                    "permutation_finals": finals,
                    "runtime": {
                        "generation_policy": generation_policy_record(model),
                        "attn_implementation": args.attn_implementation,
                        "model_source": str(args.model_source),
                        "num_layers": num_layers,
                        "processor_use_fast": args.processor_use_fast,
                        "torch_version": torch.__version__,
                    },
                }
                atomic_write_json(output_path, payload)
                completed += 1
                print(
                    json.dumps(
                        {
                            "uid": sample["uid"],
                            "candidate_executions": len(cache),
                            "successful_orders": sum(row["final_correct"] for row in finals),
                            "elapsed_seconds": round(time.time() - started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del processor_inputs, prepared, cache
            except Exception as exc:
                errors += 1
                error_handle.write(
                    json.dumps(
                        {"uid": sample["uid"], "error": repr(exc), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )
                error_handle.flush()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    atomic_write_json(
        shard_dir / "summary.json",
        {
            "selected_samples": len(rows),
            "completed_this_run": completed,
            "skipped_existing": skipped,
            "errors_this_run": errors,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "elapsed_seconds": time.time() - started,
        },
    )


def run_self_test() -> None:
    assert layer_order("early_to_late", 4, "x") == [0, 1, 2, 3]
    assert layer_order("late_to_early", 4, "x") == [3, 2, 1, 0]
    assert layer_order("center_out", 4, "x") == [1, 2, 0, 3]
    assert layer_order("outside_in", 4, "x") == [0, 3, 1, 2]
    assert sorted(layer_order("random:1", 28, "x")) == list(range(28))
    assert route_key([1, 0, 1]) == "101"
    print("phase1 candidate collector self-test ok")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = select_rows(list(iter_jsonl(args.manifest)), args)
    if not rows:
        raise RuntimeError("no samples selected")

    import torch

    model, processor, device = load_model(args)
    if args.processor_use_fast != "auto":
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(args.model_source),
            cache_dir=str(HF_HUB_CACHE),
            local_files_only=True,
            use_fast=args.processor_use_fast == "true",
        )
    if args.mode == "gate":
        run_gate(args, rows, model, processor, device, torch)
    else:
        run_search(args, rows, model, processor, device, torch, config)


if __name__ == "__main__":
    main()
