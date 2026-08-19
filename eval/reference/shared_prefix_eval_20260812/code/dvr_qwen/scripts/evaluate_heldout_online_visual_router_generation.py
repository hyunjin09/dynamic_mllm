#!/usr/bin/env python3
"""Evaluate full dense and learned online router generation on heldout manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HELDOUT_DIR = Path("/mnt/hyemin/10k_dataset_mask/heldout_lmms_recommended_v1")
DEFAULT_MODEL_SOURCE = Path(
    "/mnt/hyemin/models/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/"
    "snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
DEFAULT_HF_HUB_CACHE = Path("/mnt/hyemin/models/hub")
DEFAULT_OUT_ROOT = Path("/mnt/hyemin/10k_dataset_mask/heldout_router_generation_eval")

os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HUB_CACHE.parent))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HUB_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_HUB_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TMPDIR", str(ROOT / "state" / "tmp"))
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input-fallback-gate-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--baseline-rows-jsonl",
        type=Path,
        default=None,
        help="Reuse all-on generations from a matching heldout evaluation.",
    )
    parser.add_argument("--heldout-dir", type=Path, default=DEFAULT_HELDOUT_DIR)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--hf-hub-cache", type=Path, default=DEFAULT_HF_HUB_CACHE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="heldout_router_generation_eval")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--router-threshold", type=float, default=None)
    parser.add_argument("--fallback-threshold", type=float, default=None)
    parser.add_argument("--process-name", default="")
    parser.add_argument("--allow-initial", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--processor-use-fast", choices=["true", "false"], default="false")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", choices=["auto", "cpu", "none"], default="auto")
    parser.add_argument("--first-gpu-max-memory-gb", type=int, default=46)
    parser.add_argument("--other-gpu-max-memory-gb", type=int, default=46)
    parser.add_argument("--cpu-max-memory-gb", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def index_baseline_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = str(row["uid"])
        if uid in indexed:
            raise ValueError(f"duplicate baseline UID: {uid}")
        indexed[uid] = row
    return indexed


def baseline_record_for_row(
    row: dict[str, Any],
    baseline_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    uid = str(row["uid"])
    if uid not in baseline_rows:
        raise ValueError(f"baseline cache is missing UID: {uid}")
    cached = baseline_rows[uid]
    for field in ("benchmark", "metric_name"):
        if str(cached.get(field)) != str(row.get(field)):
            raise ValueError(
                f"baseline cache mismatch for {uid}: "
                f"{field}={cached.get(field)!r} != {row.get(field)!r}"
            )
    required = ("baseline_prediction", "baseline_score", "baseline_correct")
    missing = [field for field in required if field not in cached]
    if missing:
        raise ValueError(f"baseline cache row {uid} is missing fields: {missing}")
    return {
        "baseline_prediction": str(cached["baseline_prediction"]),
        "baseline_score": float(cached["baseline_score"]),
        "baseline_correct": bool(cached["baseline_correct"]),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_filter(value: str) -> set[str] | None:
    if value.strip().lower() in {"all", "*", ""}:
        return None
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def select_records(
    rows: list[dict[str, Any]],
    *,
    benchmarks: set[str] | None,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    filtered = [
        row
        for row in rows
        if bool(row.get("has_answer", True))
        and (benchmarks is None or str(row["benchmark"]).lower() in benchmarks)
    ]
    return [row for index, row in enumerate(filtered) if index % num_shards == shard_index]


def mask_statistics(mask: list[int] | list[bool]) -> dict[str, Any]:
    values = [int(value) for value in mask]
    if not values or any(value not in (0, 1) for value in values):
        raise ValueError("mask must be a non-empty binary sequence")
    return {
        "mask_key": "".join(str(value) for value in values),
        "num_visual_on_layers": sum(values),
        "transition_count": sum(int(left != right) for left, right in zip(values, values[1:])),
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_mean_ci(values: list[float], *, repetitions: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(int(seed))
    numeric = [float(value) for value in values]
    bootstrapped = []
    for _ in range(int(repetitions)):
        bootstrapped.append(sum(numeric[rng.randrange(len(numeric))] for _ in numeric) / len(numeric))
    bootstrapped.sort()
    return {
        "n": len(numeric),
        "mean": sum(numeric) / len(numeric),
        "ci_low": _percentile(bootstrapped, 0.025),
        "ci_high": _percentile(bootstrapped, 0.975),
    }


def summarize_rows(rows: list[dict[str, Any]], *, bootstrap_repetitions: int, bootstrap_seed: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    for row in rows:
        groups.setdefault(str(row["benchmark"]), []).append(row)

    metric_fields = {
        "baseline_score": lambda row: float(row["baseline_score"]),
        "baseline_correct_rate": lambda row: float(bool(row["baseline_correct"])),
        "router_score": lambda row: float(row["router_score"]),
        "router_correct_rate": lambda row: float(bool(row["router_correct"])),
        "paired_correct_delta": lambda row: float(bool(row["router_correct"])) - float(bool(row["baseline_correct"])),
        "avg_selected_layers": lambda row: float(row["selected_num_visual_on_layers"]),
        "avg_selected_transitions": lambda row: float(row["selected_transition_count"]),
    }
    if any(row.get("fallback_used_sparse_router") is not None for row in rows):
        metric_fields["sparse_admission_rate"] = lambda row: float(bool(row["fallback_used_sparse_router"]))
    summary: dict[str, Any] = {}
    for group_index, (name, group) in enumerate(sorted(groups.items())):
        group_summary: dict[str, Any] = {"samples": len(group)}
        for metric_index, (metric_name, getter) in enumerate(metric_fields.items()):
            group_summary[metric_name] = bootstrap_mean_ci(
                [getter(row) for row in group],
                repetitions=bootstrap_repetitions,
                seed=int(bootstrap_seed) + group_index * 100 + metric_index,
            )
        summary[name] = group_summary
    return summary


def decode_generated(processor: Any, token_ids: Any) -> str:
    return processor.batch_decode(
        token_ids.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def load_router_checkpoint(
    checkpoint_path: Path,
    *,
    allow_initial: bool,
    threshold_override: float | None,
    device: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("method", "")).startswith("drllm_visual"):
        from related_work_baselines.src.drllm_visual_router import DRLLMVisualRouter

        router = DRLLMVisualRouter(**checkpoint["router_config"])
        router.load_state_dict(checkpoint["router_state_dict"])
        runtime = {
            "router_threshold": float(0.0 if threshold_override is None else threshold_override),
            "threshold_source": "drllm_binary_logit_zero" if threshold_override is None else "command_line_override",
            "checkpoint_role": "drllm_visual_paper_baseline",
            "visual_summary_mode": "none",
            "text_summary_mode": "all_text",
        }
        router.to(device)
        router.eval()
        return router, runtime, checkpoint

    from dvr_qwen.routing import BinaryVisualOnRouter, load_binary_router_state_dict
    from dvr_qwen.scripts.evaluate_online_visual_router_generation import checkpoint_runtime

    runtime = checkpoint_runtime(checkpoint, allow_initial=allow_initial)
    if threshold_override is not None:
        runtime["router_threshold"] = float(threshold_override)
        runtime["threshold_source"] = "command_line_override"
    else:
        runtime["threshold_source"] = "checkpoint_recommendation"
    router = BinaryVisualOnRouter(**checkpoint["router_config"])
    load_binary_router_state_dict(router, checkpoint["router_state_dict"])
    router.to(device)
    router.eval()
    return router, runtime, checkpoint


def validate_checkpoint_runtime(checkpoint: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if str(checkpoint.get("method", "")).startswith("drllm_visual"):
        config = checkpoint.get("training_config", {})
        return {
            "checkpoint_family": "drllm_visual_paper_baseline",
            "model_source": str(config.get("model_source", args.model_source)),
            "generation_policy": {
                "eos_token_ids": [151643, 151645],
                "repetition_penalty": 1.05,
            },
        }
    from dvr_qwen.scripts.evaluate_online_visual_router_generation import validate_checkpoint_runtime as validate

    return validate(checkpoint, args)


def run_one_row(
    *,
    model: Any,
    processor: Any,
    router: Any,
    row: dict[str, Any],
    heldout_dir: Path,
    router_threshold: float,
    visual_summary_mode: str,
    text_summary_mode: str,
    eos_token_ids: list[int],
    repetition_penalty: float,
    cached_baseline: dict[str, Any] | None = None,
    input_fallback_gate: Any | None = None,
    fallback_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    from dvr_qwen.binary_generate import (
        binary_dvrc_greedy_generate,
        binary_dvrc_input_fallback_router_greedy_generate,
        binary_dvrc_router_greedy_generate,
    )
    from dvr_qwen.eval_metrics import score_prediction
    from dvr_qwen.modeling_dvr_qwen2_5_vl import qwen_num_hidden_layers
    from dvr_qwen.scripts.cache_preference_gt_router_features import build_processor_inputs

    started = time.time()
    processor_inputs = build_processor_inputs(processor, row, data_root=heldout_dir)
    num_layers = qwen_num_hidden_layers(model.config)
    route_device = next(model.parameters()).device
    all_on = torch.ones(1, num_layers, dtype=torch.bool, device=route_device)

    if cached_baseline is None:
        with torch.inference_mode():
            baseline_output = binary_dvrc_greedy_generate(
                model,
                processor_inputs,
                visual_on_mask=all_on,
                max_new_tokens=int(row.get("max_new_tokens") or 16),
                eos_token_ids=eos_token_ids,
                stop_on_eos=True,
                repetition_penalty=repetition_penalty,
            )
        baseline_prediction = decode_generated(processor, baseline_output.generated_ids)
        baseline_score = float(
            score_prediction(
                str(row["metric_name"]),
                baseline_prediction,
                row.get("answer"),
                row.get("all_answer_norms"),
            )
        )
        del baseline_output
    else:
        baseline_prediction = str(cached_baseline["baseline_prediction"])
        baseline_score = float(cached_baseline["baseline_score"])

    generation_args = {
        "max_new_tokens": int(row.get("max_new_tokens") or 16),
        "eos_token_ids": eos_token_ids,
        "stop_on_eos": True,
        "repetition_penalty": repetition_penalty,
        "visual_summary_mode": visual_summary_mode,
        "text_summary_mode": text_summary_mode,
        "router_threshold": float(router_threshold),
        "return_route_logits": True,
    }
    with torch.inference_mode():
        if input_fallback_gate is None:
            router_output = binary_dvrc_router_greedy_generate(
                model,
                processor_inputs,
                visual_on_router=router,
                **generation_args,
            )
        else:
            if fallback_runtime is None:
                raise ValueError("fallback_runtime is required with input_fallback_gate")
            router_output = binary_dvrc_input_fallback_router_greedy_generate(
                model,
                processor_inputs,
                visual_on_router=router,
                input_fallback_gate=input_fallback_gate,
                fallback_threshold=float(fallback_runtime["fallback_threshold"]),
                gate_text_summary_mode=str(fallback_runtime["text_summary_mode"]),
                gate_visual_summary_count=int(fallback_runtime["visual_summary_count"]),
                **generation_args,
            )
    router_prediction = decode_generated(processor, router_output.generated_ids)
    router_score = float(
        score_prediction(
            str(row["metric_name"]),
            router_prediction,
            row.get("answer"),
            row.get("all_answer_norms"),
        )
    )
    correctness_threshold = float(row["correctness_threshold"])
    mask = router_output.state.route_binary.detach().cpu().view(-1).to(dtype=torch.int64).tolist()
    stats = mask_statistics(mask)
    route_logits = (
        router_output.state.route_logits.detach().float().cpu().view(-1).tolist()
        if router_output.state.route_logits is not None
        else None
    )
    result = {
        "uid": row["uid"],
        "sample_id": row["sample_id"],
        "benchmark": row["benchmark"],
        "source_dataset": row.get("source_dataset"),
        "source_split": row.get("source_split"),
        "correctness_threshold": correctness_threshold,
        "metric_name": row["metric_name"],
        "answer": row.get("answer"),
        "all_answer_norms": row.get("all_answer_norms"),
        "baseline_prediction": baseline_prediction,
        "baseline_score": baseline_score,
        "baseline_correct": bool(baseline_score >= correctness_threshold),
        "router_prediction": router_prediction,
        "router_score": router_score,
        "router_correct": bool(router_score >= correctness_threshold),
        "selected_mask_key": stats["mask_key"],
        "selected_visual_on_mask": mask,
        "selected_num_visual_on_layers": stats["num_visual_on_layers"],
        "selected_transition_count": stats["transition_count"],
        "route_logits": route_logits,
        "router_generated_ids": router_output.generated_ids.detach().cpu().view(-1).tolist(),
        "fallback_gate_logit": (
            float(router_output.state.fallback_gate_logit.view(-1)[0].item())
            if router_output.state.fallback_gate_logit is not None
            else None
        ),
        "fallback_used_sparse_router": router_output.state.fallback_used_sparse_router,
        "elapsed_seconds": time.time() - started,
    }
    del router_output
    return result


def run_self_test() -> None:
    assert select_records(
        [{"uid": "a", "benchmark": "x", "has_answer": True}, {"uid": "b", "benchmark": "y", "has_answer": False}],
        benchmarks=None,
        num_shards=1,
        shard_index=0,
    )[0]["uid"] == "a"
    assert mask_statistics([1, 0, 1])["transition_count"] == 2
    print("self-test passed")


def main() -> int:
    args = parse_args()
    from dvr_qwen.process_name import set_process_name

    effective_process_name = set_process_name(args.process_name)
    if args.self_test:
        run_self_test()
        return 0
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --self-test is used")
    out_dir = args.out_root / args.run_id / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from dvr_qwen.scripts.cache_preference_gt_router_features import (
        ensure_min_free_cuda_memory,
        load_model_and_processor,
    )

    ensure_min_free_cuda_memory(float(args.min_free_gb), device_map=str(args.device_map))
    router_device = torch.device("cuda:0" if torch.cuda.is_available() and args.device_map != "cpu" else "cpu")
    router, runtime, checkpoint = load_router_checkpoint(
        args.checkpoint,
        allow_initial=bool(args.allow_initial),
        threshold_override=args.router_threshold,
        device=router_device,
    )
    checkpoint_provenance = validate_checkpoint_runtime(checkpoint, args)
    fallback_gate = None
    fallback_runtime = None
    if args.input_fallback_gate_checkpoint is not None:
        from dvr_qwen.scripts.evaluate_online_visual_router_generation import load_input_fallback_gate_checkpoint

        fallback_gate, fallback_runtime, _ = load_input_fallback_gate_checkpoint(
            args.input_fallback_gate_checkpoint,
            threshold_override=args.fallback_threshold,
            device=router_device,
        )
    generation_policy = checkpoint_provenance.get("generation_policy") or {}
    eos_token_ids = [int(value) for value in generation_policy.get("eos_token_ids") or [151645]]
    repetition_penalty = float(generation_policy.get("repetition_penalty") or 1.05)

    samples_path = args.heldout_dir / "samples.jsonl"
    rows = select_records(
        read_jsonl(samples_path),
        benchmarks=parse_filter(str(args.benchmarks)),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError("no heldout records selected")

    baseline_rows: dict[str, dict[str, Any]] | None = None
    if args.baseline_rows_jsonl is not None:
        baseline_rows = index_baseline_rows(read_jsonl(args.baseline_rows_jsonl))
        for row in rows:
            baseline_record_for_row(row, baseline_rows)

    model, processor = load_model_and_processor(args)
    model.eval()

    output_rows: list[dict[str, Any]] = []
    rows_path = out_dir / "heldout_generation_rows.jsonl"
    started = time.time()
    print(
        f"[setup] rows={len(rows)} checkpoint={args.checkpoint} "
        f"threshold={runtime['router_threshold']} role={runtime['checkpoint_role']}",
        flush=True,
    )
    for index, row in enumerate(rows, start=1):
        result = run_one_row(
            model=model,
            processor=processor,
            router=router,
            row=row,
            heldout_dir=args.heldout_dir,
            router_threshold=float(runtime["router_threshold"]),
            visual_summary_mode=str(runtime["visual_summary_mode"]),
            text_summary_mode=str(runtime["text_summary_mode"]),
            eos_token_ids=eos_token_ids,
            repetition_penalty=repetition_penalty,
            cached_baseline=(
                baseline_record_for_row(row, baseline_rows)
                if baseline_rows is not None
                else None
            ),
            input_fallback_gate=fallback_gate,
            fallback_runtime=fallback_runtime,
        )
        output_rows.append(result)
        if torch.cuda.is_available() and index % 8 == 0:
            torch.cuda.empty_cache()
        if index % 10 == 0 or index == len(rows):
            baseline_correct = sum(float(item["baseline_correct"]) for item in output_rows) / len(output_rows)
            router_correct = sum(float(item["router_correct"]) for item in output_rows) / len(output_rows)
            mean_layers = sum(float(item["selected_num_visual_on_layers"]) for item in output_rows) / len(output_rows)
            print(
                f"[eval] {index}/{len(rows)} baseline={baseline_correct:.4f} "
                f"router={router_correct:.4f} avg_layers={mean_layers:.2f}",
                flush=True,
            )
            write_jsonl(rows_path, output_rows)

    summary = {
        "evaluation_version": "heldout_online_visual_router_generation_eval_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_runtime": runtime,
        "checkpoint_provenance": checkpoint_provenance,
        "input_fallback_gate_checkpoint": (
            str(args.input_fallback_gate_checkpoint) if args.input_fallback_gate_checkpoint else None
        ),
        "input_fallback_gate_checkpoint_sha256": (
            sha256_file(args.input_fallback_gate_checkpoint) if args.input_fallback_gate_checkpoint else None
        ),
        "input_fallback_gate_runtime": fallback_runtime,
        "process_name": effective_process_name,
        "model_source": str(args.model_source),
        "heldout_dir": str(args.heldout_dir),
        "baseline_rows_jsonl": (
            str(args.baseline_rows_jsonl) if args.baseline_rows_jsonl is not None else None
        ),
        "baseline_rows_sha256": (
            sha256_file(args.baseline_rows_jsonl) if args.baseline_rows_jsonl is not None else None
        ),
        "samples_jsonl": str(samples_path),
        "benchmarks": str(args.benchmarks),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "generation_policy": {
            "eos_token_ids": eos_token_ids,
            "repetition_penalty": repetition_penalty,
        },
        "summary": summarize_rows(
            output_rows,
            bootstrap_repetitions=int(args.bootstrap_repetitions),
            bootstrap_seed=int(args.bootstrap_seed),
        ),
        "elapsed_seconds": time.time() - started,
        "outputs": {"rows_jsonl": str(rows_path), "summary_json": str(out_dir / "summary.json")},
    }
    write_jsonl(rows_path, output_rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
