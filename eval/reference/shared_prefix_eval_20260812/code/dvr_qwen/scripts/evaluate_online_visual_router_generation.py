#!/usr/bin/env python3
"""Evaluate a learned router with live per-layer VISUAL_ON decisions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = Path("/mnt/hyemin/10k_dataset_mask/preference_gt_correctness_first_v1")
DEFAULT_ALL_SAMPLES = Path("/mnt/hyemin/10k_dataset_mask/manifests/all_samples.jsonl")
DEFAULT_SAMPLE_INDEX = Path("/mnt/hyemin/10k_dataset_mask/final_phase1_phase2/sample_index.jsonl")
DEFAULT_DATA_ROOT = Path("/mnt/hyemin/dvr_qwen/rl/complete_correct_wrong_pools_20260713")
DEFAULT_MODEL_SOURCE = Path(
    "/mnt/hyemin/models/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/"
    "snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
DEFAULT_HF_HUB_CACHE = Path("/mnt/hyemin/models/hub")
DEFAULT_OUT_ROOT = Path("/mnt/hyemin/10k_dataset_mask/online_visual_router_generation_eval")

os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HUB_CACHE.parent))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HUB_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_HUB_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--input-fallback-gate-checkpoint",
        type=Path,
        default=None,
        help="Optional input-only gate. A non-admitted sample follows exact all-on instead of the sparse router.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--all-samples-jsonl", type=Path, default=DEFAULT_ALL_SAMPLES)
    parser.add_argument("--sample-index-jsonl", type=Path, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--hf-hub-cache", type=Path, default=DEFAULT_HF_HUB_CACHE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="online_router_generation_eval")
    parser.add_argument("--process-name", default="")
    parser.add_argument("--split", choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--eligible-only", action="store_true")
    parser.add_argument(
        "--uid-manifest",
        type=Path,
        default=None,
        help="Optional JSONL UID allowlist applied before split and limit selection.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-selection", choices=["ordered", "stratified"], default="ordered")
    parser.add_argument("--selection-seed", type=int, default=20260723)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--router-threshold", type=float, default=None)
    parser.add_argument("--fallback-threshold", type=float, default=None)
    parser.add_argument("--allow-initial", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
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


def rows_by_uid(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    indexed = {str(row["uid"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate UID in {path}")
    return indexed


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


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


def load_router_checkpoint(
    checkpoint_path: Path,
    *,
    allow_initial: bool,
    threshold_override: float | None,
    device: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch

    from dvr_qwen.routing import BinaryVisualOnRouter, load_binary_router_state_dict

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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


def load_input_fallback_gate_checkpoint(
    checkpoint_path: Path,
    *,
    threshold_override: float | None,
    device: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch

    from dvr_qwen.routing import InputFallbackGate

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("input_fallback_gate_config")
    state_dict = checkpoint.get("input_fallback_gate_state_dict")
    training_config = checkpoint.get("training_config") or {}
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError(f"not an input fallback gate checkpoint: {checkpoint_path}")
    threshold = training_config.get("fallback_threshold")
    if threshold_override is not None:
        threshold = float(threshold_override)
        source = "command_line_override"
    else:
        source = "checkpoint_recommendation"
    if threshold is None:
        raise ValueError("input fallback gate checkpoint has no recommended fallback threshold")
    gate = InputFallbackGate(**config)
    gate.load_state_dict(state_dict, strict=True)
    gate.to(device)
    gate.eval()
    return gate, {
        "fallback_threshold": float(threshold),
        "threshold_source": source,
        "text_summary_mode": str(training_config.get("text_summary_mode") or "instruction_only"),
        "visual_summary_count": int(training_config.get("visual_summary_count") or 2),
        "label_policy": training_config.get("label_policy"),
    }, checkpoint


def load_evaluation_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    from dvr_qwen.preference_gt import PreferenceGTDatasetPaths, iter_jsonl, validate_sample_target

    paths = PreferenceGTDatasetPaths(args.dataset_dir)
    paths.require_files()
    targets: list[dict[str, Any]] = []
    for row in iter_jsonl(paths.sample_targets):
        validate_sample_target(row)
        targets.append(row)
    if args.uid_manifest is not None:
        allowed_uids = {str(row["uid"]) for row in iter_jsonl(args.uid_manifest)}
        targets = [row for row in targets if str(row["uid"]) in allowed_uids]
        if not targets:
            raise ValueError("UID manifest selected no evaluation targets")
    joined = join_evaluation_metadata(
        targets,
        rows_by_uid(args.all_samples_jsonl),
        rows_by_uid(args.sample_index_jsonl),
    )
    selected = select_records(
        joined,
        split=str(args.split),
        benchmarks=parse_filter(str(args.benchmarks)),
        eligible_only=bool(args.eligible_only),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    if int(args.limit) > 0:
        if str(args.limit_selection) == "stratified":
            selected = stratified_limit_records(selected, limit=int(args.limit), seed=int(args.selection_seed))
        else:
            selected = selected[: int(args.limit)]
    if not selected:
        raise ValueError("no evaluation records selected")
    return selected


def stratified_limit_records(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    """Take a deterministic proportional sample over benchmark/source/eligibility strata."""
    if int(limit) <= 0 or len(rows) <= int(limit):
        return list(rows)
    grouped: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["benchmark"]).lower(),
            str(row["source_bucket"]),
            bool(row.get("training_eligible", False)),
        )
        grouped[key].append(row)
    rng = random.Random(int(seed))
    for group in grouped.values():
        rng.shuffle(group)
    total = float(len(rows))
    allocations: dict[tuple[str, str, bool], int] = {}
    fractions: list[tuple[float, tuple[str, str, bool]]] = []
    assigned = 0
    for key, group in grouped.items():
        exact = int(limit) * len(group) / total
        count = min(int(exact), len(group))
        allocations[key] = count
        assigned += count
        fractions.append((exact - count, key))
    for _, key in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if assigned >= int(limit):
            break
        if allocations[key] < len(grouped[key]):
            allocations[key] += 1
            assigned += 1
    selected = [row for key in sorted(grouped) for row in grouped[key][: allocations[key]]]
    rng.shuffle(selected)
    return selected


def validate_checkpoint_runtime(checkpoint: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    summary = checkpoint.get("summary") or {}
    metadata = summary.get("dataset_metadata") or {}
    model_runtime = metadata.get("model_runtime") or {}
    revision = str(model_runtime.get("hf_revision") or "")
    if revision and args.model_source.name != revision:
        raise ValueError(
            f"checkpoint HF revision {revision} does not match model source {args.model_source.name}"
        )
    return {
        "hf_revision": revision or None,
        "model_runtime_id": model_runtime.get("model_runtime_id"),
        "attn_implementation": model_runtime.get("attn_implementation"),
        "generation_policy": model_runtime.get("generation_policy"),
        "scorer_sha256": model_runtime.get("scorer_sha256"),
    }


def select_records(
    rows: list[dict[str, Any]],
    *,
    split: str,
    benchmarks: set[str] | None,
    eligible_only: bool,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    filtered = [
        row
        for row in rows
        if (split == "all" or str(row["split"]) == split)
        and (benchmarks is None or str(row["benchmark"]).lower() in benchmarks)
        and (not eligible_only or bool(row["training_eligible"]))
    ]
    return [row for index, row in enumerate(filtered) if index % num_shards == shard_index]


def checkpoint_runtime(checkpoint: dict[str, Any], *, allow_initial: bool) -> dict[str, Any]:
    role = str(checkpoint.get("checkpoint_role") or "legacy_unknown")
    if role == "initial_untrained" and not allow_initial:
        raise ValueError("refusing to evaluate an untrained initial checkpoint")
    training_config = checkpoint.get("training_config") or {}
    threshold = training_config.get("recommended_router_threshold")
    if threshold is None:
        raise ValueError("checkpoint has no independent recommended router threshold")
    return {
        "checkpoint_role": role,
        "selected_epoch": int(checkpoint.get("selected_epoch") or 0),
        "router_threshold": float(threshold),
        "visual_summary_mode": str(training_config.get("visual_summary_mode") or "none"),
        "text_summary_mode": str(training_config.get("text_summary_mode") or "all_text"),
    }


def join_evaluation_metadata(
    targets: list[dict[str, Any]],
    all_samples_by_uid: dict[str, dict[str, Any]],
    sample_index_by_uid: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for target in targets:
        uid = str(target["uid"])
        if uid not in all_samples_by_uid:
            raise KeyError(f"missing all-samples metadata for {uid}")
        if uid not in sample_index_by_uid:
            raise KeyError(f"missing sample-index metadata for {uid}")
        row = dict(all_samples_by_uid[uid])
        row.update(target)
        sample_index = sample_index_by_uid[uid]
        if sample_index.get("all_on_score") is not None:
            row["source_full_score"] = float(sample_index["all_on_score"])
            row["source_full_prediction"] = sample_index.get("all_on_prediction")
            row["source_full_provenance"] = "current_model_phase1_all_on_anchor"
        else:
            row["source_full_score"] = float(all_samples_by_uid[uid]["source_full_score"])
            row["source_full_prediction"] = all_samples_by_uid[uid].get("source_full_prediction")
            row["source_full_provenance"] = "historical_source_manifest"
        row["all_answer_norms"] = all_samples_by_uid[uid].get("all_answer_norms")
        row["all_off_score"] = float(sample_index["all_off_score"])
        joined.append(row)
    return joined


def mask_statistics(mask: list[int] | list[bool]) -> dict[str, Any]:
    values = [int(value) for value in mask]
    if not values or any(value not in (0, 1) for value in values):
        raise ValueError("mask must be a non-empty binary sequence")
    return {
        "mask_key": "".join(str(value) for value in values),
        "num_visual_on_layers": sum(values),
        "transition_count": sum(int(left != right) for left, right in zip(values, values[1:])),
    }


def mask_diversity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [str(row["selected_mask_key"]) for row in rows if row.get("selected_mask_key") is not None]
    if not keys:
        return {
            "unique_masks": 0,
            "modal_mask_key": None,
            "modal_mask_fraction": None,
            "mean_hamming_to_modal": None,
            "variable_layers_05_95": 0,
            "mean_layer_binary_entropy": None,
        }
    widths = {len(key) for key in keys}
    if len(widths) != 1 or any(set(key) - {"0", "1"} for key in keys):
        raise ValueError("selected_mask_key values must be fixed-width binary strings")
    counts = Counter(keys)
    modal_key, modal_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    layer_frequencies = [sum(key[layer] == "1" for key in keys) / float(len(keys)) for layer in range(len(modal_key))]
    entropies = []
    for probability in layer_frequencies:
        if probability in {0.0, 1.0}:
            entropies.append(0.0)
        else:
            entropies.append(-probability * math.log(probability) - (1.0 - probability) * math.log(1.0 - probability))
    return {
        "unique_masks": len(counts),
        "modal_mask_key": modal_key,
        "modal_mask_fraction": modal_count / float(len(keys)),
        "mean_hamming_to_modal": sum(
            sum(left != right for left, right in zip(key, modal_key)) for key in keys
        )
        / float(len(keys)),
        "variable_layers_05_95": sum(0.05 < probability < 0.95 for probability in layer_frequencies),
        "mean_layer_binary_entropy": sum(entropies) / float(len(entropies)),
        "layer_on_frequencies": layer_frequencies,
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_mean_ci(values: list[float], *, repetitions: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    numeric = [float(value) for value in values]
    rng = random.Random(int(seed))
    bootstrapped = []
    for _ in range(int(repetitions)):
        draw = [numeric[rng.randrange(len(numeric))] for _ in numeric]
        bootstrapped.append(sum(draw) / float(len(draw)))
    bootstrapped.sort()
    return {
        "n": len(numeric),
        "mean": sum(numeric) / float(len(numeric)),
        "ci_low": _percentile(bootstrapped, 0.025),
        "ci_high": _percentile(bootstrapped, 0.975),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_repetitions: int = 1000,
    bootstrap_seed: int = 20260720,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    for row in rows:
        groups.setdefault(str(row["benchmark"]), []).append(row)
        groups.setdefault(str(row["source_bucket"]), []).append(row)
        groups.setdefault("eligible" if row["training_eligible"] else "ineligible", []).append(row)

    metric_fields = {
        "online_score": lambda row: float(row["online_score"]),
        "online_correct_rate": lambda row: float(bool(row["online_correct"])),
        "source_full_score": lambda row: float(row["source_full_score"]),
        "source_full_correct_rate": lambda row: float(bool(row["source_full_correct"])),
        "all_off_score": lambda row: float(row["all_off_score"]),
        "avg_selected_layers": lambda row: float(row["selected_num_visual_on_layers"]),
        "avg_selected_transitions": lambda row: float(row["selected_transition_count"]),
        "training_eligible_rate": lambda row: float(bool(row["training_eligible"])),
    }
    summary: dict[str, Any] = {}
    for group_index, (name, group) in enumerate(sorted(groups.items())):
        group_summary: dict[str, Any] = {"samples": len(group)}
        for metric_index, (metric_name, getter) in enumerate(metric_fields.items()):
            group_summary[metric_name] = bootstrap_mean_ci(
                [getter(row) for row in group],
                repetitions=bootstrap_repetitions,
                seed=int(bootstrap_seed) + group_index * 100 + metric_index,
            )
        group_summary["mask_diversity"] = mask_diversity_summary(group)
        summary[name] = group_summary
    return summary


def decode_generated(processor: Any, token_ids: Any) -> str:
    return processor.batch_decode(
        token_ids.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def run_online_route(
    *,
    model: Any,
    processor: Any,
    router: Any,
    row: dict[str, Any],
    data_root: Path,
    router_threshold: float,
    visual_summary_mode: str,
    text_summary_mode: str,
    eos_token_ids: list[int],
    repetition_penalty: float,
    input_fallback_gate: Any | None = None,
    fallback_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    from dvr_qwen.binary_generate import (
        binary_dvrc_input_fallback_router_greedy_generate,
        binary_dvrc_router_greedy_generate,
    )
    from dvr_qwen.eval_metrics import score_prediction
    from dvr_qwen.scripts.cache_preference_gt_router_features import build_processor_inputs

    started = time.time()
    processor_inputs = build_processor_inputs(processor, row, data_root=data_root)
    with torch.inference_mode():
        common = {
            "max_new_tokens": int(row.get("max_new_tokens") or 16),
            "eos_token_ids": eos_token_ids,
            "stop_on_eos": True,
            "repetition_penalty": repetition_penalty,
            "visual_summary_mode": visual_summary_mode,
            "text_summary_mode": text_summary_mode,
            "router_threshold": float(router_threshold),
            "return_route_logits": True,
        }
        if input_fallback_gate is None:
            output = binary_dvrc_router_greedy_generate(
                model,
                processor_inputs,
                visual_on_router=router,
                **common,
            )
        else:
            if fallback_runtime is None:
                raise ValueError("fallback_runtime is required with input_fallback_gate")
            output = binary_dvrc_input_fallback_router_greedy_generate(
                model,
                processor_inputs,
                visual_on_router=router,
                input_fallback_gate=input_fallback_gate,
                fallback_threshold=float(fallback_runtime["fallback_threshold"]),
                gate_text_summary_mode=str(fallback_runtime["text_summary_mode"]),
                gate_visual_summary_count=int(fallback_runtime["visual_summary_count"]),
                **common,
            )
    prediction = decode_generated(processor, output.generated_ids)
    score = float(
        score_prediction(
            str(row["metric_name"]),
            prediction,
            row.get("answer"),
            row.get("all_answer_norms"),
        )
    )
    correctness_threshold = float(row["correctness_threshold"])
    mask = output.state.route_binary.detach().cpu().view(-1).to(dtype=torch.int64).tolist()
    stats = mask_statistics(mask)
    route_logits = (
        output.state.route_logits.detach().float().cpu().view(-1).tolist()
        if output.state.route_logits is not None
        else None
    )
    generated_ids = output.generated_ids.detach().cpu().view(-1).tolist()
    result = {
        "uid": row["uid"],
        "sample_id": row["sample_id"],
        "benchmark": row["benchmark"],
        "source_bucket": row["source_bucket"],
        "training_eligible": bool(row["training_eligible"]),
        "correctness_threshold": correctness_threshold,
        "metric_name": row["metric_name"],
        "answer": row.get("answer"),
        "online_prediction": prediction,
        "online_score": score,
        "online_correct": bool(score >= correctness_threshold),
        "source_full_prediction": row.get("source_full_prediction"),
        "source_full_score": float(row["source_full_score"]),
        "source_full_correct": bool(float(row["source_full_score"]) >= correctness_threshold),
        "source_full_provenance": row.get("source_full_provenance"),
        "all_off_score": float(row["all_off_score"]),
        "all_off_correct": bool(float(row["all_off_score"]) >= correctness_threshold),
        "selected_mask_key": stats["mask_key"],
        "selected_visual_on_mask": mask,
        "selected_num_visual_on_layers": stats["num_visual_on_layers"],
        "selected_transition_count": stats["transition_count"],
        "route_logits": route_logits,
        "fallback_gate_logit": (
            float(output.state.fallback_gate_logit.view(-1)[0].item())
            if output.state.fallback_gate_logit is not None
            else None
        ),
        "fallback_used_sparse_router": output.state.fallback_used_sparse_router,
        "generated_ids": generated_ids,
        "elapsed_seconds": time.time() - started,
    }
    del output
    return result


def run_self_test() -> None:
    assert mask_statistics([0, 1, 1, 0])["transition_count"] == 2
    assert checkpoint_runtime(
        {
            "checkpoint_role": "best_learned_lexicographic",
            "selected_epoch": 1,
            "training_config": {"recommended_router_threshold": 0.01},
        },
        allow_initial=False,
    )["router_threshold"] == 0.01
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if str(args.process_name):
        try:
            import setproctitle

            setproctitle.setproctitle(str(args.process_name))
        except ImportError:
            pass
    if args.self_test:
        run_self_test()
        return 0
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --self-test is used")
    out_dir = args.out_root / args.run_id
    if out_dir.exists() and not bool(args.overwrite):
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
    fallback_checkpoint = None
    if args.input_fallback_gate_checkpoint is not None:
        fallback_gate, fallback_runtime, fallback_checkpoint = load_input_fallback_gate_checkpoint(
            args.input_fallback_gate_checkpoint,
            threshold_override=args.fallback_threshold,
            device=router_device,
        )
    rows = load_evaluation_rows(args)
    generation_policy = checkpoint_provenance.get("generation_policy") or {}
    eos_token_ids = [int(value) for value in generation_policy.get("eos_token_ids") or [151645]]
    repetition_penalty = float(generation_policy.get("repetition_penalty") or 1.05)
    model, processor = load_model_and_processor(args)
    model.eval()

    output_rows: list[dict[str, Any]] = []
    rows_path = out_dir / "online_generation_rows.jsonl"
    started = time.time()
    print(
        f"[setup] rows={len(rows)} checkpoint={args.checkpoint} "
        f"threshold={runtime['router_threshold']} role={runtime['checkpoint_role']}",
        flush=True,
    )
    for index, row in enumerate(rows, start=1):
        result = run_online_route(
            model=model,
            processor=processor,
            router=router,
            row=row,
            data_root=args.data_root,
            router_threshold=float(runtime["router_threshold"]),
            visual_summary_mode=str(runtime["visual_summary_mode"]),
            text_summary_mode=str(runtime["text_summary_mode"]),
            eos_token_ids=eos_token_ids,
            repetition_penalty=repetition_penalty,
            input_fallback_gate=fallback_gate,
            fallback_runtime=fallback_runtime,
        )
        output_rows.append(result)
        if torch.cuda.is_available() and index % 8 == 0:
            torch.cuda.empty_cache()
        if index % 10 == 0 or index == len(rows):
            mean_correct = sum(float(item["online_correct"]) for item in output_rows) / len(output_rows)
            mean_layers = sum(float(item["selected_num_visual_on_layers"]) for item in output_rows) / len(output_rows)
            print(
                f"[eval] {index}/{len(rows)} correct={mean_correct:.4f} "
                f"avg_layers={mean_layers:.2f}",
                flush=True,
            )
            write_jsonl(rows_path, output_rows)

    summary = {
        "evaluation_version": "online_visual_router_generation_eval_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_runtime": runtime,
        "checkpoint_provenance": checkpoint_provenance,
        "input_fallback_gate_checkpoint": str(args.input_fallback_gate_checkpoint) if args.input_fallback_gate_checkpoint else None,
        "input_fallback_gate_runtime": fallback_runtime,
        "input_fallback_gate_checkpoint_sha256": (
            sha256_file(args.input_fallback_gate_checkpoint) if args.input_fallback_gate_checkpoint else None
        ),
        "model_source": str(args.model_source),
        "dataset_dir": str(args.dataset_dir),
        "split": str(args.split),
        "eligible_only": bool(args.eligible_only),
        "limit": int(args.limit),
        "limit_selection": str(args.limit_selection),
        "selection_seed": int(args.selection_seed),
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
