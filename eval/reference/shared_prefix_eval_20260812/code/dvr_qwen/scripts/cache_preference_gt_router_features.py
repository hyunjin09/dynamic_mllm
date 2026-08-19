#!/usr/bin/env python3
"""Cache anchor-context router features for the 10k preference GT dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = Path("/mnt/hyemin/10k_dataset_mask/preference_gt_correctness_first_v1")
DEFAULT_DATA_ROOT = Path("/mnt/hyemin/dvr_qwen/rl/complete_correct_wrong_pools_20260713")
DEFAULT_MODEL_SOURCE = Path(
    "/mnt/hyemin/models/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/"
    "snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
DEFAULT_HF_HUB_CACHE = Path("/mnt/hyemin/models/hub")
DEFAULT_OUT_ROOT = Path("/mnt/hyemin/10k_dataset_mask/preference_router_features")
DEFAULT_GATE_SUMMARY = ROOT / "MCTS" / "state" / "gate" / "summary.json"
ORIGINAL_DATA_ROOT_MARKER = "complete_correct_wrong_pools_20260713/"

os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HUB_CACHE.parent))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HUB_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_HUB_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TMPDIR", str(ROOT / "state" / "tmp"))

sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dvr_qwen.preference_gt import (  # noqa: E402
    NUM_LAYERS,
    PreferenceGTDatasetPaths,
    cooptimal_soft_label,
    mask_key_to_list,
    mask_to_key,
    minimum_budget_representative_mask_key,
    validate_sample_target,
)
from dvr_qwen.router_data import (  # noqa: E402
    collate_cached_feature_samples,
    load_cached_feature_sample,
    previous_gate_tensor,
    validate_cached_feature_sample,
)
from dvr_qwen.router_features import collect_teacher_forced_router_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--hf-hub-cache", type=Path, default=DEFAULT_HF_HUB_CACHE)
    parser.add_argument("--gate-summary", type=Path, default=DEFAULT_GATE_SUMMARY)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="preference_anchor_all_on_features_v1")
    parser.add_argument("--split", choices=["train", "validation", "all"], default="train")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--source-buckets", default="all")
    parser.add_argument("--eligible-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--route-context",
        choices=["all_on", "all_off", "cooptimal_representative"],
        default="all_on",
        help="Route executed only to define the cached hidden-state trajectory.",
    )
    parser.add_argument("--num-windows", type=int, default=8)
    parser.add_argument("--visual-summary-mode", choices=["none", "mean_abs"], default="mean_abs")
    parser.add_argument("--processor-use-fast", choices=["true", "false"], default="false")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", choices=["auto", "cpu", "none"], default="auto")
    parser.add_argument("--first-gpu-max-memory-gb", type=int, default=30)
    parser.add_argument("--other-gpu-max-memory-gb", type=int, default=46)
    parser.add_argument("--cpu-max-memory-gb", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-ungated", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def feature_path_for_row(feature_dir: Path, record_index: int, row: dict[str, Any]) -> Path:
    return feature_dir / f"{int(record_index):06d}_{safe_stem(str(row['uid']))}.pt"


def parse_filter(value: str) -> set[str] | None:
    if str(value).strip().lower() in {"", "all", "*"}:
        return None
    return {item.strip().lower() for item in str(value).split(",") if item.strip()}


def load_target_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = PreferenceGTDatasetPaths(args.dataset_dir)
    paths.require_files()
    benchmarks = parse_filter(args.benchmarks)
    source_buckets = parse_filter(args.source_buckets)
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(paths.sample_targets):
        validate_sample_target(row)
        if args.split != "all" and row["split"] != args.split:
            continue
        if benchmarks is not None and str(row["benchmark"]).lower() not in benchmarks:
            continue
        if source_buckets is not None and str(row["source_bucket"]).lower() not in source_buckets:
            continue
        if args.eligible_only and not bool(row["training_eligible"]):
            continue
        rows.append(row)
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    return rows[args.offset : args.offset + args.limit]


def anchor_route_mask(row: dict[str, Any], route_context: str) -> list[int]:
    if route_context == "all_on":
        return [1] * NUM_LAYERS
    if route_context == "all_off":
        return [0] * NUM_LAYERS
    if route_context == "cooptimal_representative":
        key = minimum_budget_representative_mask_key(row)
        if key is None:
            return [0] * NUM_LAYERS
        return mask_key_to_list(key)
    raise ValueError(f"unknown route_context {route_context!r}")


def validate_gate_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"missing replay gate summary: {path}. Run the saved-oracle replay gate before caching features."
        )
    gate = json.loads(path.read_text(encoding="utf-8"))
    required_true = ["pass_current_hf_binary_ids", "pass_source_score", "pass_available_saved_ids"]
    if "pass_reference_generated_ids" in gate:
        required_true.append("pass_reference_generated_ids")
    failed = [key for key in required_true if not bool(gate.get(key))]
    if int(gate.get("errors", 0)) != 0:
        failed.append("errors==0")
    if failed:
        raise RuntimeError(f"replay gate did not pass required fields {failed}: {path}")
    return gate


def cuda_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _ = torch.cuda.mem_get_info()
    return float(free_bytes) / (1024.0**3)


def ensure_min_free_cuda_memory(min_free_gb: float, *, device_map: str) -> None:
    if min_free_gb <= 0 or device_map == "cpu":
        return
    free_gb = cuda_free_gb()
    if free_gb < float(min_free_gb):
        raise RuntimeError(
            f"CUDA free memory {free_gb:.2f} GiB is below required {float(min_free_gb):.2f} GiB"
        )


def resolve_image_path(row: dict[str, Any], data_root: Path, raw_value: str | None = None) -> Path:
    raw = Path(str(raw_value or row["image_path"]))
    if raw.exists():
        return raw
    if ORIGINAL_DATA_ROOT_MARKER in str(raw):
        rel = str(raw).split(ORIGINAL_DATA_ROOT_MARKER, 1)[1]
        mapped = data_root / rel
        if mapped.exists():
            return mapped
    parts = raw.parts
    if "images" in parts:
        mapped = data_root.joinpath(*parts[parts.index("images") + 1 :])
        if mapped.exists():
            return mapped
    fallback = data_root / "images" / str(row["benchmark"]) / raw.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"could not resolve image for {row['uid']}: {raw}")


def resolve_image_paths(row: dict[str, Any], data_root: Path) -> list[Path]:
    raw_paths = row.get("image_paths") or [row.get("image_path")]
    return [resolve_image_path(row, data_root, str(raw)) for raw in raw_paths if raw]


def instruction_token_mask_from_ids(
    input_ids: torch.Tensor,
    instruction_token_ids: list[int] | torch.Tensor,
) -> torch.Tensor:
    """Mark the final exact instruction-token span in a rendered chat prompt."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must have shape [1, S], got {tuple(input_ids.shape)}")
    instruction = torch.as_tensor(instruction_token_ids, dtype=input_ids.dtype, device=input_ids.device).flatten()
    if instruction.numel() == 0:
        raise ValueError("instruction token sequence must not be empty")
    if instruction.numel() > input_ids.shape[1]:
        raise ValueError("instruction token sequence is longer than the rendered prompt")
    windows = input_ids[0].unfold(0, int(instruction.numel()), 1)
    matches = torch.nonzero((windows == instruction).all(dim=1), as_tuple=False).flatten()
    if matches.numel() == 0:
        raise ValueError("instruction token sequence was not found exactly in the rendered prompt")
    start = int(matches[-1].item())
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[:, start : start + instruction.numel()] = True
    return mask


def build_processor_inputs(processor: Any, row: dict[str, Any], *, data_root: Path) -> dict[str, Any]:
    image_paths = resolve_image_paths(row, data_root)
    image_contents: list[dict[str, Any]] = []
    max_pixels = row.get("max_pixels")
    for image_path in image_paths:
        image_content: dict[str, Any] = {"type": "image", "image": str(image_path)}
        if max_pixels is not None:
            image_content["max_pixels"] = int(max_pixels)
        elif int(row.get("max_image_tokens") or 0) > 0:
            image_content["max_pixels"] = int(row["max_image_tokens"]) * 28 * 28
        image_contents.append(image_content)
    prompt = str(row["prompt"])
    placeholders = re.findall(r"<image(?: \d+)?>", prompt)
    content: list[dict[str, Any]] = []
    if placeholders:
        parts = re.split(r"<image(?: \d+)?>", prompt)
        for index, part in enumerate(parts):
            if part.strip():
                content.append({"type": "text", "text": part})
            if index < len(image_contents):
                content.append(image_contents[index])
    else:
        content.extend(image_contents)
        content.append({"type": "text", "text": prompt})
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        return_mm_token_type_ids=True,
    )
    if "mm_token_type_ids" not in inputs:
        raise RuntimeError("processor did not return mm_token_type_ids")
    instruction_mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
    text_chunks = row.get("instruction_text_chunks") or [str(row["prompt"])]
    for chunk in text_chunks:
        token_ids = processor.tokenizer(str(chunk), add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        if len(token_ids) > inputs["input_ids"].shape[1]:
            continue
        windows = inputs["input_ids"][0].unfold(0, len(token_ids), 1)
        matches = torch.nonzero((windows == torch.as_tensor(token_ids, dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)).all(dim=1), as_tuple=False).flatten()
        for start in matches.tolist():
            instruction_mask[:, start : start + len(token_ids)] = True
    if not bool(instruction_mask.any().item()):
        instruction_mask = inputs["mm_token_type_ids"] == 0
    if bool((instruction_mask & (inputs["mm_token_type_ids"] != 0)).any().item()):
        raise RuntimeError("instruction token span overlaps non-text multimodal tokens")
    inputs["instruction_token_mask"] = instruction_mask
    return dict(inputs)


def model_max_memory(args: argparse.Namespace) -> dict[Any, str] | None:
    if args.device_map != "auto" or args.first_gpu_max_memory_gb <= 0:
        return None
    max_memory: dict[Any, str] = {
        device_index: f"{int(args.other_gpu_max_memory_gb)}GiB"
        for device_index in range(torch.cuda.device_count())
    }
    max_memory[0] = f"{int(args.first_gpu_max_memory_gb)}GiB"
    if int(args.cpu_max_memory_gb) > 0:
        max_memory["cpu"] = f"{int(args.cpu_max_memory_gb)}GiB"
    return max_memory


def load_model_and_processor(args: argparse.Namespace):
    if not args.model_source.exists():
        raise FileNotFoundError(f"model snapshot does not exist: {args.model_source}")
    if not args.hf_hub_cache.exists():
        raise FileNotFoundError(f"HF hub cache does not exist: {args.hf_hub_cache}")
    from transformers import AutoProcessor

    from dvr_qwen.modeling_dvr_qwen2_5_vl import DVRQwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        str(args.model_source),
        cache_dir=str(args.hf_hub_cache),
        local_files_only=True,
        use_fast=args.processor_use_fast == "true",
    )
    if args.device_map == "cpu":
        device_map: str | dict[str, str] | None = {"": "cpu"}
    elif args.device_map == "none":
        device_map = None
    else:
        device_map = "auto"
    model = DVRQwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_source),
        cache_dir=str(args.hf_hub_cache),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=model_max_memory(args),
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, processor


def feature_sample_from_row(model: Any, processor: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    route = anchor_route_mask(row, args.route_context)
    processor_inputs = build_processor_inputs(processor, row, data_root=args.data_root)
    collected = collect_teacher_forced_router_features(
        model,
        processor_inputs,
        torch.tensor([route], dtype=torch.bool),
        num_windows=args.num_windows,
        visual_summary_mode=args.visual_summary_mode,
    )
    route_key = mask_to_key(route)
    sample = {
        "id": row["uid"],
        "sample_id": row["sample_id"],
        "uid": row["uid"],
        "benchmark": row["benchmark"],
        "split": row["split"],
        "source_bucket": row["source_bucket"],
        "route_context": args.route_context,
        "route_context_mask_key": route_key,
        "route_context_visual_on_layers": int(sum(route)),
        "global_mean": collected["global_mean"],
        "window_mean": collected["window_mean"],
        "last_token": collected["last_token"],
        "labels": torch.tensor(route, dtype=torch.float32),
        "soft_labels": cooptimal_soft_label(row),
        "cooptimal_soft_labels": cooptimal_soft_label(row),
        "minimum_budget_representative_mask_key": minimum_budget_representative_mask_key(row),
        "minimum_correct_budget": row.get("minimum_correct_budget"),
        "cooptimal_route_ids": list(row.get("cooptimal_route_ids") or []),
        "prev_gates": previous_gate_tensor(torch.tensor(route, dtype=torch.bool)),
        "layer_idx": collected["layer_idx"],
        "num_visual_tokens": int(collected["num_visual_tokens"][0].item()),
        "num_text_tokens": int(collected["num_text_tokens"][0].item()),
        "full_prompt_tokens": int(collected["full_prompt_tokens"][0].item()),
        "num_visual_on_layers": int(sum(route)),
        "num_windows": int(args.num_windows),
        "feature_schema": (
            f"preference_gt_anchor_{args.route_context}_text_summary_v1"
            if args.visual_summary_mode == "none"
            else f"preference_gt_anchor_{args.route_context}_text_visual_mean_abs_summary_v1"
        ),
        "model_runtime_id": row["model_runtime_id"],
        "image_path": row["image_path"],
        "max_image_tokens": row.get("max_image_tokens"),
        "max_pixels": row.get("max_pixels"),
    }
    if "visual_summaries" in collected:
        sample["visual_summaries"] = collected["visual_summaries"]
    validate_cached_feature_sample(sample)
    return sample


def run_self_test() -> None:
    row = {
        "dataset_version": "correctness_first_preference_gt_v1",
        "model_runtime_id": "qwen25vl7b_cc594898_sdpa_greedy_v1",
        "uid": "gqa:gqa_1",
        "sample_id": "gqa_1",
        "benchmark": "gqa",
        "split": "train",
        "source_bucket": "complete_wrong",
        "correctness_threshold": 1.0,
        "training_eligible": True,
        "correct_route_count": 1,
        "incorrect_route_count": 1,
        "candidate_route_count": 2,
        "minimum_correct_budget": 1,
        "cooptimal_route_count": 1,
        "cooptimal_route_ids": ["r1"],
        "cooptimal_mask_keys": ["1000000000000000000000000000"],
        "null_visual_optimal": False,
        "preference_pair_count": 1,
        "correctness_pair_count": 1,
        "efficiency_pair_count": 0,
        "image_path": "/data/dataset/dynamic_mllm/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713/images/gqa/a.jpg",
        "prompt": "question",
        "max_image_tokens": 0,
        "max_pixels": None,
    }
    assert sum(anchor_route_mask(row, "all_on")) == NUM_LAYERS
    assert sum(anchor_route_mask(row, "all_off")) == 0
    assert anchor_route_mask(row, "cooptimal_representative")[0] == 1
    assert safe_stem("gqa:sample/id") == "gqa_sample_id"
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if not args.allow_ungated:
        gate = validate_gate_summary(args.gate_summary)
    else:
        gate = {"decision": "allow_ungated_debug_only"}
    rows = load_target_rows(args)
    if not rows:
        raise ValueError("no rows selected for feature caching")

    out_dir = args.out_root / args.run_id
    feature_dir = out_dir / "features"
    if out_dir.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"{out_dir} already exists; pass --resume or --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(exist_ok=True)

    pending: list[tuple[int, dict[str, Any], Path]] = []
    feature_paths: list[Path] = []
    for local_idx, row in enumerate(rows):
        record_index = args.offset + local_idx
        path = feature_path_for_row(feature_dir, record_index, row)
        feature_paths.append(path)
        if args.resume and path.exists():
            sample = load_cached_feature_sample(path)
            if sample["id"] != row["uid"]:
                raise ValueError(f"resume feature id mismatch at {path}")
        else:
            pending.append((record_index, row, path))

    model = None
    processor = None
    if pending:
        ensure_min_free_cuda_memory(args.min_free_gb, device_map=args.device_map)
        model, processor = load_model_and_processor(args)

    started = time.time()
    for _, row, path in pending:
        if model is None or processor is None:
            raise AssertionError("model/processor not initialized")
        sample = feature_sample_from_row(model, processor, row, args)
        torch.save(sample, path)

    index_rows: list[dict[str, Any]] = []
    for local_idx, (row, path) in enumerate(zip(rows, feature_paths)):
        sample = load_cached_feature_sample(path)
        index_rows.append(
            {
                "uid": row["uid"],
                "id": row["uid"],
                "sample_id": row["sample_id"],
                "benchmark": row["benchmark"],
                "split": row["split"],
                "source_bucket": row["source_bucket"],
                "record_index": args.offset + local_idx,
                "feature_path": str(path),
                "route_context": args.route_context,
                "route_context_mask_key": sample["route_context_mask_key"],
                "route_context_visual_on_layers": int(sample["route_context_visual_on_layers"]),
                "minimum_correct_budget": sample.get("minimum_correct_budget"),
                "num_visual_tokens": int(sample["num_visual_tokens"]),
                "num_text_tokens": int(sample["num_text_tokens"]),
                "full_prompt_tokens": int(sample["full_prompt_tokens"]),
                "feature_schema": sample["feature_schema"],
            }
        )
    write_jsonl(out_dir / "feature_index.jsonl", index_rows)

    loaded_samples = [load_cached_feature_sample(path) for path in feature_paths]
    batch = collate_cached_feature_samples(loaded_samples)
    summary = {
        "phase": "preference_gt_anchor_feature_cache",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(out_dir),
        "feature_index": str(out_dir / "feature_index.jsonl"),
        "gate_summary": str(args.gate_summary),
        "gate_decision": gate.get("decision"),
        "allow_ungated": bool(args.allow_ungated),
        "split": args.split,
        "benchmarks": args.benchmarks,
        "source_buckets": args.source_buckets,
        "eligible_only": bool(args.eligible_only),
        "offset": int(args.offset),
        "limit": int(args.limit),
        "num_samples": len(index_rows),
        "num_pending_samples_extracted": len(pending),
        "num_resumed_samples": len(index_rows) - len(pending),
        "route_context": args.route_context,
        "visual_summary_mode": args.visual_summary_mode,
        "num_windows": int(args.num_windows),
        "model_source": str(args.model_source),
        "hf_hub_cache": str(args.hf_hub_cache),
        "data_root": str(args.data_root),
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
        "cached_feature_batch_shape": {
            "global_mean": list(batch["global_mean"].shape),
            "window_mean": list(batch["window_mean"].shape),
            "last_token": list(batch["last_token"].shape),
            "labels": list(batch["labels"].shape),
            "soft_labels": list(batch["soft_labels"].shape),
            "prev_gates": list(batch["prev_gates"].shape),
            "visual_summaries": list(batch["visual_summaries"].shape) if "visual_summaries" in batch else None,
        },
        "sample_summaries": index_rows,
        "elapsed_seconds": time.time() - started,
        "verification_status": "passed",
    }
    write_json(out_dir / "feature_cache_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
