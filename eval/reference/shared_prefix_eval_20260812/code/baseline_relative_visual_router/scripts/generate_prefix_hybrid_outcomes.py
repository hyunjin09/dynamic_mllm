#!/usr/bin/env python3
"""Generate actual SW31 outcomes after a shared dense visual-on prefix."""

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
from dvr_qwen.binary_generate import (
    binary_dvrc_router_greedy_generate,
    prepare_binary_dvrc_inputs,
)
from dvr_qwen.eval_metrics import score_prediction
from dvr_qwen.scripts.cache_preference_gt_router_features import (
    build_processor_inputs,
    ensure_min_free_cuda_memory,
    load_model_and_processor,
)
from dvr_qwen.scripts.evaluate_heldout_online_visual_router_generation import (
    decode_generated,
    load_router_checkpoint,
    mask_statistics,
)


DEFAULT_MODEL = Path(
    "/mnt/hyemin/models/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/"
    "snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
DEFAULT_CHECKPOINT = Path(
    "/mnt/hyemin/10k_dataset_mask/online_visual_router_preference_runs/"
    "sw31_bt_leg_s41/router_epoch_001.pt"
)


def parse_csv_ints(value: str) -> list[int]:
    parsed: list[int] = []
    for item in value.split(","):
        number = int(item.strip())
        if number not in parsed:
            parsed.append(number)
    if not parsed:
        raise ValueError("at least one prefix layer count is required")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-rows-jsonl", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hf-hub-cache", type=Path, default=Path("/mnt/hyemin/models/hub"))
    parser.add_argument("--prefix-layers", default="2,4,8")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--router-threshold", type=float, default=None)
    parser.add_argument("--processor-use-fast", choices=["true", "false"], default="false")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", choices=["auto", "cpu", "none"], default="auto")
    parser.add_argument("--first-gpu-max-memory-gb", type=int, default=20)
    parser.add_argument("--other-gpu-max-memory-gb", type=int, default=20)
    parser.add_argument("--cpu-max-memory-gb", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--process-name", default="brvr-prefix-outcomes")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def outcome(baseline_correct: bool, router_correct: bool) -> str:
    if baseline_correct and router_correct:
        return "preserve"
    if baseline_correct:
        return "harm"
    if router_correct:
        return "rescue"
    return "unsolved"


def cached_uids(prefix_dir: Path) -> set[str]:
    seen: set[str] = set()
    for path in sorted(prefix_dir.glob("prefix_*_shard_*_part_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        current = {str(row["uid"]) for row in payload["rows"]}
        overlap = seen & current
        if overlap:
            raise RuntimeError(f"duplicate cached prefix UIDs: {sorted(overlap)[:3]}")
        seen.update(current)
    return seen


def next_part(prefix_dir: Path, prefix_layers: int, shard_index: int) -> int:
    paths = list(
        prefix_dir.glob(
            f"prefix_{prefix_layers:02d}_shard_{shard_index:02d}_of_*_part_*.pt"
        )
    )
    return 0 if not paths else max(int(path.stem.rsplit("_", 1)[-1]) for path in paths) + 1


def save_chunk(
    output_dir: Path,
    *,
    prefix_layers: int,
    shard_index: int,
    num_shards: int,
    part: int,
    feature_lists: dict[str, list[torch.Tensor]],
    rows: list[dict[str, Any]],
) -> Path:
    path = output_dir / (
        f"prefix_{prefix_layers:02d}_shard_{shard_index:02d}_of_{num_shards:02d}_part_{part:05d}.pt"
    )
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": "shared_dense_prefix_actual_policy_v1",
            "prefix_layers": prefix_layers,
            "shard_index": shard_index,
            "num_shards": num_shards,
            **{name: torch.stack(values) for name, values in feature_lists.items()},
            "rows": rows,
        },
        temporary,
    )
    temporary.replace(path)
    return path


@torch.inference_mode()
def run_one(
    *,
    model: Any,
    processor: Any,
    router: Any,
    inputs: dict[str, Any],
    prepared_binary_inputs: Any,
    manifest_row: dict[str, Any],
    baseline_row: dict[str, Any],
    prefix_layers: int,
    runtime: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    output = binary_dvrc_router_greedy_generate(
        model,
        inputs,
        visual_on_router=router,
        max_new_tokens=int(manifest_row.get("max_new_tokens") or 16),
        eos_token_ids=[151643, 151645],
        stop_on_eos=True,
        repetition_penalty=1.05,
        visual_summary_mode=str(runtime["visual_summary_mode"]),
        text_summary_mode=str(runtime["text_summary_mode"]),
        router_threshold=float(runtime["router_threshold"]),
        return_route_logits=True,
        forced_visual_on_prefix_layers=prefix_layers,
        capture_prefix_gate_features=True,
        prepared_binary_inputs=prepared_binary_inputs,
    )
    if output.state.prefix_gate_features is None:
        raise RuntimeError("prefix generation did not return gate features")
    prediction = decode_generated(processor, output.generated_ids)
    score = float(
        score_prediction(
            str(manifest_row["metric_name"]),
            prediction,
            manifest_row.get("answer"),
            manifest_row.get("all_answer_norms"),
        )
    )
    threshold = float(manifest_row["correctness_threshold"])
    baseline_correct = bool(baseline_row["baseline_correct"])
    router_correct = bool(score >= threshold)
    mask = output.state.route_binary.detach().cpu().view(-1).to(torch.int64).tolist()
    stats = mask_statistics(mask)
    if any(not value for value in mask[:prefix_layers]):
        raise AssertionError("shared dense-prefix invariant was violated")
    features = {
        name: value[0].to(dtype=torch.float16).cpu()
        for name, value in output.state.prefix_gate_features.items()
    }
    row = {
        "uid": str(manifest_row["uid"]),
        "sample_id": str(manifest_row.get("sample_id", "")),
        "benchmark": str(manifest_row["benchmark"]).lower(),
        "metric_name": str(manifest_row["metric_name"]),
        "correctness_threshold": threshold,
        "prefix_layers": prefix_layers,
        "baseline_prediction": str(baseline_row["baseline_prediction"]),
        "baseline_score": float(baseline_row["baseline_score"]),
        "baseline_correct": baseline_correct,
        "router_prediction": prediction,
        "router_score": score,
        "router_correct": router_correct,
        "outcome": outcome(baseline_correct, router_correct),
        "selected_visual_on_mask": mask,
        "selected_mask_key": stats["mask_key"],
        "selected_num_visual_on_layers": stats["num_visual_on_layers"],
        "selected_transition_count": stats["transition_count"],
    }
    return features, row


def run_self_test() -> None:
    assert parse_csv_ints("2,4,2,8") == [2, 4, 8]
    assert outcome(True, False) == "harm"
    assert outcome(False, True) == "rescue"
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    prefixes = parse_csv_ints(args.prefix_layers)
    manifests = read_jsonl(args.manifest_jsonl)
    baselines = read_jsonl(args.baseline_rows_jsonl)
    aligned = align_manifest_policy_rows(manifests, baselines)
    selected = [row for index, row in enumerate(aligned) if index % args.num_shards == args.shard_index]
    if args.limit:
        selected = selected[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen = {prefix: cached_uids(args.output_dir / f"prefix_{prefix:02d}") for prefix in prefixes}
    parts = {
        prefix: next_part(args.output_dir / f"prefix_{prefix:02d}", prefix, args.shard_index)
        for prefix in prefixes
    }
    ensure_min_free_cuda_memory(args.min_free_gb, device_map=args.device_map)
    model, processor = load_model_and_processor(args)
    router_device = next(model.parameters()).device
    router, runtime, _ = load_router_checkpoint(
        args.checkpoint,
        allow_initial=False,
        threshold_override=args.router_threshold,
        device=router_device,
    )
    buffers: dict[int, dict[str, Any]] = {
        prefix: {
            "features": {
                "instruction_mean": [],
                "instruction_window_mean": [],
                "instruction_last": [],
                "visual_summaries": [],
            },
            "rows": [],
        }
        for prefix in prefixes
    }
    started = time.time()
    evaluations = 0
    for sample_index, (manifest_row, baseline_row) in enumerate(selected, start=1):
        uid = str(manifest_row["uid"])
        missing_prefixes = [prefix for prefix in prefixes if uid not in seen[prefix]]
        inputs = (
            build_processor_inputs(processor, manifest_row, data_root=args.data_root)
            if missing_prefixes
            else None
        )
        prepared_binary_inputs = (
            prepare_binary_dvrc_inputs(model, inputs)
            if inputs is not None
            else None
        )
        for prefix in missing_prefixes:
            assert inputs is not None
            assert prepared_binary_inputs is not None
            features, result = run_one(
                model=model,
                processor=processor,
                router=router,
                inputs=inputs,
                prepared_binary_inputs=prepared_binary_inputs,
                manifest_row=manifest_row,
                baseline_row=baseline_row,
                prefix_layers=prefix,
                runtime=runtime,
            )
            for name, value in features.items():
                buffers[prefix]["features"][name].append(value)
            buffers[prefix]["rows"].append(result)
            evaluations += 1
            if len(buffers[prefix]["rows"]) >= args.chunk_size:
                prefix_dir = args.output_dir / f"prefix_{prefix:02d}"
                prefix_dir.mkdir(parents=True, exist_ok=True)
                save_chunk(
                    prefix_dir,
                    prefix_layers=prefix,
                    shard_index=args.shard_index,
                    num_shards=args.num_shards,
                    part=parts[prefix],
                    feature_lists=buffers[prefix]["features"],
                    rows=buffers[prefix]["rows"],
                )
                parts[prefix] += 1
                buffers[prefix] = {
                    "features": {name: [] for name in buffers[prefix]["features"]},
                    "rows": [],
                }
        if sample_index % 20 == 0 or sample_index == len(selected):
            elapsed = time.time() - started
            print(
                f"[prefix-outcomes] shard={args.shard_index} samples={sample_index}/{len(selected)} "
                f"evaluations={evaluations} rate={evaluations / max(elapsed, 1e-6):.2f}/s",
                flush=True,
            )
    for prefix in prefixes:
        if buffers[prefix]["rows"]:
            prefix_dir = args.output_dir / f"prefix_{prefix:02d}"
            prefix_dir.mkdir(parents=True, exist_ok=True)
            save_chunk(
                prefix_dir,
                prefix_layers=prefix,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                part=parts[prefix],
                feature_lists=buffers[prefix]["features"],
                rows=buffers[prefix]["rows"],
            )
    summary = {
        "schema_version": "shared_dense_prefix_shard_summary_v1",
        "prefix_layers": prefixes,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_samples": len(selected),
        "new_evaluations": evaluations,
        "already_cached": {str(prefix): len(seen[prefix]) for prefix in prefixes},
        "elapsed_seconds": time.time() - started,
        "runtime": runtime,
        "checkpoint": str(args.checkpoint),
    }
    (args.output_dir / f"summary_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
