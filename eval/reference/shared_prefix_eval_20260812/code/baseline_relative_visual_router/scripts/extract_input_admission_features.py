#!/usr/bin/env python3
"""Cache deployable pre-language-layer features aligned to one frozen policy."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE / "src"))
os.environ.setdefault("HF_HOME", "/mnt/hyemin/models")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/hyemin/models/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/hyemin/models/hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch

from baseline_relative_visual_router.input_features import align_manifest_policy_rows
from dvr_qwen.binary_generate import prepare_binary_dvrc_inputs
from dvr_qwen.fallback_gate import initial_input_gate_features
from dvr_qwen.scripts.cache_preference_gt_router_features import (
    build_processor_inputs,
    ensure_min_free_cuda_memory,
    load_model_and_processor,
)


DEFAULT_MODEL = Path(
    "/mnt/hyemin/models/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/"
    "snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--policy-rows-jsonl", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hf-hub-cache", type=Path, default=Path("/mnt/hyemin/models/hub"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--process-name", default="brvr-input-features")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--processor-use-fast", choices=["true", "false"], default="false")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", choices=["auto", "cpu", "none"], default="auto")
    parser.add_argument("--first-gpu-max-memory-gb", type=int, default=20)
    parser.add_argument("--other-gpu-max-memory-gb", type=int, default=20)
    parser.add_argument("--cpu-max-memory-gb", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def outcome(policy_row: dict[str, Any]) -> str:
    baseline = bool(policy_row["baseline_correct"])
    routed = bool(policy_row["router_correct"])
    if baseline and routed:
        return "preserve"
    if baseline:
        return "harm"
    if routed:
        return "rescue"
    return "unsolved"


def existing_uids(output_dir: Path) -> tuple[set[str], int]:
    seen: set[str] = set()
    next_part = 0
    for path in sorted(output_dir.glob("input_features_shard_*_part_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        uids = [str(row["uid"]) for row in payload["metadata"]]
        overlap = seen.intersection(uids)
        if overlap:
            raise RuntimeError(f"duplicate cached UIDs: {sorted(overlap)[:3]}")
        seen.update(uids)
        if f"shard_{payload['shard_index']:02d}_" in path.name:
            next_part = max(next_part, int(path.stem.rsplit("_", 1)[-1]) + 1)
    return seen, next_part


def save_chunk(
    args: argparse.Namespace,
    part: int,
    means: list[torch.Tensor],
    lasts: list[torch.Tensor],
    visuals: list[torch.Tensor],
    metadata: list[dict[str, Any]],
) -> Path:
    path = args.output_dir / (
        f"input_features_shard_{args.shard_index:02d}_of_{args.num_shards:02d}_"
        f"part_{part:05d}.pt"
    )
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": "input_admission_features_v1",
            "feature_stage": "pre_language_layer_0",
            "feature_fields": [
                "instruction_mean",
                "instruction_last",
                "visual_mean",
                "visual_mean_abs",
            ],
            "instruction_mean": torch.stack(means),
            "instruction_last": torch.stack(lasts),
            "visual_summaries": torch.stack(visuals),
            "metadata": metadata,
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
        },
        temporary,
    )
    temporary.replace(path)
    return path


@torch.inference_mode()
def extract_one(
    model: Any,
    processor: Any,
    manifest_row: dict[str, Any],
    policy_row: dict[str, Any],
    data_root: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    inputs = build_processor_inputs(processor, manifest_row, data_root=data_root)
    binary_inputs = prepare_binary_dvrc_inputs(model, inputs)
    features = initial_input_gate_features(
        binary_inputs,
        text_summary_mode="instruction_only",
        visual_summary_count=2,
    )
    metadata = {
        "uid": str(manifest_row["uid"]),
        "sample_id": str(manifest_row.get("sample_id", "")),
        "benchmark": str(manifest_row["benchmark"]).lower(),
        "source_dataset": str(manifest_row.get("source_dataset", "")),
        "source_split": str(manifest_row.get("source_split", "")),
        "outcome": outcome(policy_row),
        "baseline_correct": bool(policy_row["baseline_correct"]),
        "router_correct": bool(policy_row["router_correct"]),
        "selected_num_visual_on_layers": int(
            policy_row["selected_num_visual_on_layers"]
        ),
        "instruction_tokens": int(binary_inputs.instruction_valid_mask.sum().item()),
        "visual_tokens": int(binary_inputs.visual_valid_mask.sum().item()),
    }
    return (
        features["instruction_mean"][0].to(dtype=torch.float16).cpu(),
        features["instruction_last"][0].to(dtype=torch.float16).cpu(),
        features["visual_summaries"][0].to(dtype=torch.float16).cpu(),
        metadata,
    )


def run_self_test() -> None:
    assert outcome({"baseline_correct": True, "router_correct": False}) == "harm"
    assert outcome({"baseline_correct": False, "router_correct": True}) == "rescue"
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned = align_manifest_policy_rows(
        read_jsonl(args.manifest_jsonl), read_jsonl(args.policy_rows_jsonl)
    )
    selected = [row for index, row in enumerate(aligned) if index % args.num_shards == args.shard_index]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]
    seen, part = existing_uids(args.output_dir)
    if seen and not args.resume:
        raise FileExistsError(f"existing input feature cache under {args.output_dir}")
    pending = [row for row in selected if str(row[0]["uid"]) not in seen]
    ensure_min_free_cuda_memory(args.min_free_gb, device_map=args.device_map)
    model, processor = load_model_and_processor(args)
    means: list[torch.Tensor] = []
    lasts: list[torch.Tensor] = []
    visuals: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    written = []
    started = time.time()
    for index, (manifest_row, policy_row) in enumerate(pending, start=1):
        mean, last, visual, meta = extract_one(
            model, processor, manifest_row, policy_row, args.data_root
        )
        means.append(mean)
        lasts.append(last)
        visuals.append(visual)
        metadata.append(meta)
        if len(metadata) >= args.chunk_size or index == len(pending):
            written.append(
                str(save_chunk(args, part, means, lasts, visuals, metadata))
            )
            part += 1
            means, lasts, visuals, metadata = [], [], [], []
        # The extractor has stable inference-only allocations.  A frequent
        # allocator flush substantially slows co-located jobs without lowering
        # the steady-state memory footprint.
        if torch.cuda.is_available() and index % 256 == 0:
            torch.cuda.empty_cache()
        if index % 50 == 0 or index == len(pending):
            elapsed = time.time() - started
            print(
                f"[input-features] shard={args.shard_index} {index}/{len(pending)} "
                f"rate={index / max(elapsed, 1e-6):.2f}/s",
                flush=True,
            )
    summary = {
        "schema_version": "input_admission_feature_shard_summary_v1",
        "manifest_jsonl": str(args.manifest_jsonl),
        "policy_rows_jsonl": str(args.policy_rows_jsonl),
        "data_root": str(args.data_root),
        "model_source": str(args.model_source),
        "feature_stage": "pre_language_layer_0",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_rows": len(selected),
        "already_cached": len(selected) - len(pending),
        "new_rows": len(pending),
        "outcomes": dict(Counter(outcome(policy) for _, policy in selected)),
        "written_parts": written,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / f"summary_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
