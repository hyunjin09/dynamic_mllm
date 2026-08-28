#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

from binary_policy.executor import capture_full_baseline, greedy_generate_from_cached_prompt
from binary_policy.executor.inputs import build_binary_inputs
from experiments.run_four_action_answer_alignment import (
    append_jsonl,
    correctness,
    decode_generation,
    git_metadata,
    load_model,
    prepare,
    read_jsonl,
    set_determinism,
    sha256_file,
    write_json_once,
)
from tools.research_analysis.four_action.eligibility import expected_unified_full_correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze current unified-FULL cohort eligibility.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_answer_alignment.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser.parse_args()


def eligibility_row(model, processor, record, device) -> dict[str, Any]:
    started = time.monotonic()
    _, inputs = prepare(processor, record, device)
    prepared = build_binary_inputs(model, inputs)
    baseline = capture_full_baseline(
        model, inputs, prepared_inputs=prepared, use_cache=True, native_causal=False
    )
    assert baseline.cache is not None
    generation = greedy_generate_from_cached_prompt(
        model,
        baseline.prompt_logits,
        baseline.inputs,
        baseline.cache,
        inputs["input_ids"],
        max_new_tokens=int(record["max_new_tokens"]),
    )
    answer = decode_generation(processor.tokenizer, generation.generated_ids)
    score, correct = correctness(record, answer)
    expected_correct = expected_unified_full_correct(record["cohort"])
    return {
        "schema_version": "four_action_unified_full_eligibility_v1",
        "uid": record["uid"],
        "dataset": record["dataset"],
        "cohort": record["cohort"],
        "image_group_id": record["image_group_id"],
        "historical_full_correct": bool(record["full_correct"]),
        "historical_full_answer": record["full_prediction"],
        "unified_full_generated_ids": generation.generated_ids[0].tolist(),
        "unified_full_generated_answer": answer,
        "unified_full_correctness_score": score,
        "unified_full_correct": correct,
        "expected_unified_full_correct": expected_correct,
        "eligible": correct == expected_correct,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank is not None else 0))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    if world != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError("eligibility freeze requires exactly 8 visible GPUs and 8 workers")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_determinism(int(config["seed"]) + rank)
    candidates = [
        row for row in read_jsonl(Path(config["cohort_manifest"]))
        if int(row["shard"]) == rank
    ]
    root = Path(config["eligibility_root"]) / f"shard_{rank:02d}"
    result_path = root / "results.jsonl"
    completed: set[str] = set()
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {result_path} without --resume")
        completed = {row["uid"] for row in read_jsonl(result_path)}
    runtime_path = root / "runtime.json"
    if not runtime_path.exists():
        write_json_once(
            runtime_path,
            {
                "schema_version": "four_action_eligibility_runtime_v1",
                "rank": rank,
                "world_size": world,
                "candidate_count": len(candidates),
                "gpu": torch.cuda.get_device_name(device),
                "config_sha256": sha256_file(args.config),
                "cohort_manifest_sha256": sha256_file(Path(config["cohort_manifest"])),
                "git": git_metadata(),
                "all_eight_gpu_workers_required": True,
            },
        )
    model, processor = load_model(config, device)
    failures = 0
    for index, record in enumerate(candidates):
        if record["uid"] in completed:
            continue
        try:
            row = eligibility_row(model, processor, record, device)
            append_jsonl(result_path, row)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "completed": index + 1,
                        "total": len(candidates),
                        "uid": record["uid"],
                        "eligible": row["eligible"],
                        "elapsed": row["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            failures += 1
            append_jsonl(
                root / "failures.jsonl",
                {"uid": record["uid"], "error": str(exc), "traceback": traceback.format_exc()},
            )
            break
        finally:
            torch.cuda.empty_cache()
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
