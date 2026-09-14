#!/usr/bin/env python3
"""Prepare, execute, and aggregate the current dense Stage-1 population."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import UnidentifiedImageError
import torch

from dense_failure_stage1.current_dense_8k import (
    STAGE1_DATASETS,
    build_current_candidate_rows,
    summarize_population,
    validate_attempt_coverage,
)
from dense_failure_stage1.lmms_scoring import lmms_eval_source_metadata
from dense_failure_stage1.runtime import (
    configure_dense_determinism,
    generate_dense,
    load_dense_runtime,
)
from tools.research_analysis.dense_failure_stage1 import select_dense_smoke


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
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


def write_json_atomic(path: Path, value) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    write_text_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def prepare(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    portable = read_jsonl(Path(args.portable_manifest))
    candidates = build_current_candidate_rows(
        portable, image_root=Path(args.image_root)
    )
    config = read_json(Path(args.config))
    counts = Counter(row["dataset"] for row in candidates)
    expected = Counter({key: int(value) for key, value in config["datasets"].items()})
    if counts != expected:
        raise ValueError(f"candidate counts differ: actual={counts} expected={expected}")
    candidate_path = output_root / "candidate_manifest.jsonl"
    write_jsonl_atomic(candidate_path, candidates)
    smoke = select_dense_smoke(
        candidates,
        seed=int(config["smoke_seed"]),
        per_cell=int(config["smoke_records_per_dataset_bucket"]),
    )
    write_json_atomic(
        output_root / "smoke" / "smoke_manifest.json",
        {
            "schema_version": "current_dense_8k_smoke_v1",
            "records": smoke,
        },
    )
    lmms = lmms_eval_source_metadata()
    if lmms["version"] != config["lmms_eval"]["version"]:
        raise ValueError("installed LMMS-Eval version differs from config")
    write_json_atomic(
        output_root / "features" / "feature_schema.json",
        {
            "schema_version": "current_dense_8k_features_v1",
            "layers": list(range(28)),
            "hidden_size": int(config["feature_hidden_size"]),
            "storage_dtype": "torch.bfloat16",
            "text_final": "post-layer hidden state at final literal user-prompt token",
            "text_mean": "mean post-layer hidden state over literal user-prompt tokens",
            "visual_mean": "mean post-layer hidden state over expanded image-pad tokens",
            "confidence_features": None,
            "confidence_omission_reason": (
                "an intermediate-layer calibrated LM-head readout is not already "
                "defined by the dense model contract"
            ),
        },
    )
    contract_lines = [
        "# LMMS-Eval scoring contract",
        "",
        f"- Package: `lmms-eval=={lmms['version']}`",
        f"- Upstream tag: `{config['lmms_eval']['git_tag']}` "
        f"(`{config['lmms_eval']['git_tag_commit']}`)",
        "- GQA: official `exact_match` with `ignore_case=true` and "
        "`ignore_punctuation=true`; binary iff score is 1.0.",
        "- ChartQA: official `chartqa_process_results` / `relaxed_overall`; "
        "binary iff score is 1.0.",
        "- TextVQA: official `textvqa_process_results` EvalAI leave-one-out "
        "consensus; raw fractional score is preserved; the repository's existing "
        "8K convention defines correct as score >= 0.5.",
        "- No historical correctness field participates in scoring.",
        "",
        "## Installed source hashes",
        "",
    ]
    for name, source in sorted(lmms["sources"].items()):
        contract_lines.append(
            f"- `{name}`: `{source['sha256']}` (`{source['path']}`)"
        )
    write_text_atomic(
        output_root / "lmms_eval_contract.md", "\n".join(contract_lines) + "\n"
    )
    write_json_atomic(
        output_root / "preparation_summary.json",
        {
            "candidate_samples": len(candidates),
            "dataset_counts": dict(counts),
            "images_present_at_preparation": sum(
                row["image_present_at_preparation"] for row in candidates
            ),
            "images_missing_at_preparation": sum(
                not row["image_present_at_preparation"] for row in candidates
            ),
            "smoke_samples": len(smoke),
            "lmms_eval": lmms,
        },
    )
    print(json.dumps({"prepared": len(candidates), "smoke": len(smoke)}))


def _classify_failure(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "missing_image"
    if isinstance(exc, UnidentifiedImageError):
        return "corrupt_image"
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "cuda_out_of_memory"
    if isinstance(exc, KeyError):
        return "missing_annotation_or_input"
    if isinstance(exc, ValueError):
        text = str(exc).lower()
        if "image" in text or "sha-256" in text:
            return "image_integrity_or_decode_error"
        return "invalid_input"
    return "isolated_runtime_failure"


def _output_row(sample: dict, result, *, rank: int, run_id: str, elapsed: float) -> dict:
    return {
        "schema_version": "current_dense_8k_output_v1",
        "uid": sample["uid"],
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "image_identifier": sample["image_identifier"],
        "image_group_id": sample["image_group_id"],
        "image_content_sha256": sample["image_content_sha256"],
        "historical_bucket": sample["historical_bucket"],
        "generated_answer": result.generated_text,
        "generated_token_ids": result.generated_ids,
        "normalized_prediction": result.normalized_prediction,
        "gt_answer": sample["answer"],
        "gt_answers": sample.get("all_answer_norms"),
        "lmms_eval_task": {
            "gqa": "gqa",
            "chartqa": "chartqa",
            "textvqa": "textvqa_val",
        }[sample["dataset"]],
        "lmms_eval_metric": result.lmms_metric_name,
        "lmms_eval_per_sample_score": result.score,
        "binary_correctness_threshold": result.correctness_threshold,
        "current_dense_correct": result.correct,
        "current_dense_wrong": not result.correct,
        "generation_length": len(result.generated_ids),
        "prompt_token_count": result.prompt_token_count,
        "visual_token_count": result.visual_token_count,
        "user_text_token_count": result.user_text_token_count,
        "literal_prompt_sha256": result.input_metadata["literal_prompt_sha256"],
        "execution": "native_qwen2_5_vl_dense_all_on",
        "decoder_layers": 28,
        "worker_rank": rank,
        "run_id": run_id,
        "elapsed_seconds": elapsed,
    }


def _save_feature_shard(path: Path, rows: list[dict], features: list[dict]) -> None:
    if len(rows) != len(features):
        raise ValueError("output/feature count differs")
    count = len(rows)
    empty_shape = (0, 28, 3584)
    payload = {
        "schema_version": "current_dense_8k_features_v1",
        "uids": [row["uid"] for row in rows],
        "layer_ids": list(range(28)),
        "text_final": (
            torch.stack([item["text_final"] for item in features]).to(torch.bfloat16)
            if features
            else torch.empty(empty_shape, dtype=torch.bfloat16)
        ),
        "text_mean": (
            torch.stack([item["text_mean"] for item in features]).to(torch.bfloat16)
            if features
            else torch.empty(empty_shape, dtype=torch.bfloat16)
        ),
        "visual_mean": (
            torch.stack([item["visual_mean"] for item in features]).to(torch.bfloat16)
            if features
            else torch.empty(empty_shape, dtype=torch.bfloat16)
        ),
        "records": count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_execution_rows(path: Path, mode: str) -> list[dict]:
    if mode == "smoke":
        return read_json(path)["records"]
    return read_jsonl(path)


def worker(args: argparse.Namespace) -> None:
    config = read_json(Path(args.config))
    configure_dense_determinism(
        int(config["execution_seed"]), dict(config["backend_settings"])
    )
    rows = _load_execution_rows(Path(args.manifest), args.mode)
    rank = int(os.environ.get("LOCAL_RANK", args.rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    if world_size != 4 or rank not in range(4):
        raise ValueError(f"four GPU workers are required: rank={rank} world={world_size}")
    assigned = [row for index, row in enumerate(rows) if index % world_size == rank]
    device_index = int(os.environ.get("LOCAL_RANK", rank))
    processor, model, device = load_dense_runtime(
        args.model_path, config["model_revision"], device_index
    )
    worker_root = Path(args.output_root) / ("smoke/workers" if args.mode == "smoke" else "workers")
    feature_root = Path(args.output_root) / ("smoke/features" if args.mode == "smoke" else "features/shards")
    batch_size = int(args.batch_size)
    for batch_index, start in enumerate(range(0, len(assigned), batch_size)):
        batch = assigned[start : start + batch_size]
        stem = f"rank{rank:03d}_batch{batch_index:05d}"
        marker_path = worker_root / f"{stem}.complete.json"
        if marker_path.exists():
            marker = read_json(marker_path)
            if marker.get("uids") == [row["uid"] for row in batch] and marker.get("run_id") == args.run_id:
                print(json.dumps({"resume": stem, "records": len(batch)}), flush=True)
                continue
            raise ValueError(f"incompatible existing batch marker: {marker_path}")
        outputs: list[dict] = []
        skips: list[dict] = []
        features: list[dict] = []
        for sample in batch:
            started = time.monotonic()
            try:
                result = generate_dense(
                    processor, model, device, sample, extract_features=True
                )
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - started
                if result.features is None:
                    raise RuntimeError("successful generation has no layer features")
                outputs.append(
                    _output_row(sample, result, rank=rank, run_id=args.run_id, elapsed=elapsed)
                )
                features.append(result.features)
                print(
                    json.dumps(
                        {
                            "uid": sample["uid"],
                            "status": "completed",
                            "correct": result.correct,
                            "score": result.score,
                            "seconds": round(elapsed, 3),
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                elapsed = time.monotonic() - started
                skips.append(
                    {
                        "uid": sample["uid"],
                        "sample_id": sample["sample_id"],
                        "dataset": sample["dataset"],
                        "image_group_id": sample["image_group_id"],
                        "historical_bucket": sample["historical_bucket"],
                        "reason_code": _classify_failure(exc),
                        "exception_type": type(exc).__name__,
                        "reason": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                        "worker_rank": rank,
                        "run_id": args.run_id,
                        "elapsed_seconds": elapsed,
                    }
                )
                print(
                    json.dumps(
                        {
                            "uid": sample["uid"],
                            "status": "skipped",
                            "reason": _classify_failure(exc),
                        }
                    ),
                    flush=True,
                )
        output_path = worker_root / f"{stem}.outputs.jsonl"
        skip_path = worker_root / f"{stem}.skips.jsonl"
        feature_path = feature_root / f"{stem}.pt"
        write_jsonl_atomic(output_path, outputs)
        write_jsonl_atomic(skip_path, skips)
        _save_feature_shard(feature_path, outputs, features)
        write_json_atomic(
            marker_path,
            {
                "schema_version": "current_dense_8k_batch_v1",
                "run_id": args.run_id,
                "rank": rank,
                "world_size": world_size,
                "uids": [row["uid"] for row in batch],
                "attempted": len(batch),
                "completed": len(outputs),
                "skipped": len(skips),
                "output_path": str(output_path),
                "skip_path": str(skip_path),
                "feature_path": str(feature_path),
                "completed_at": utc_now(),
            },
        )
    print(json.dumps({"rank": rank, "assigned": len(assigned), "status": "done"}), flush=True)


def _load_worker_artifacts(root: Path, expected_ranks: set[int]):
    markers = [read_json(path) for path in sorted(root.glob("*.complete.json"))]
    if {int(marker["rank"]) for marker in markers} != expected_ranks:
        raise ValueError("worker rank coverage differs")
    outputs: list[dict] = []
    skips: list[dict] = []
    feature_index: list[dict] = []
    feature_uids: list[str] = []
    nonfinite = {key: 0 for key in ("text_final", "text_mean", "visual_mean")}
    for marker in markers:
        output_rows = read_jsonl(Path(marker["output_path"]))
        skip_rows = read_jsonl(Path(marker["skip_path"]))
        feature_path = Path(marker["feature_path"])
        payload = torch.load(feature_path, map_location="cpu", weights_only=True)
        uids = list(payload["uids"])
        if uids != [row["uid"] for row in output_rows]:
            raise ValueError(f"feature UID order differs: {feature_path}")
        for key in ("text_final", "text_mean", "visual_mean"):
            tensor = payload[key]
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape[1:]) != (28, 3584):
                raise ValueError(f"invalid feature tensor {key}: {feature_path}: {tensor.shape}/{tensor.dtype}")
            nonfinite[key] += int((~torch.isfinite(tensor)).sum().item())
        for row_index, uid in enumerate(uids):
            feature_index.append(
                {
                    "uid": uid,
                    "shard": str(feature_path),
                    "row_index": row_index,
                    "layers": 28,
                }
            )
        feature_uids.extend(uids)
        outputs.extend(output_rows)
        skips.extend(skip_rows)
    if len(feature_uids) != len(set(feature_uids)) or set(feature_uids) != {row["uid"] for row in outputs}:
        raise ValueError("feature/output UID coverage differs")
    return outputs, skips, feature_index, {
        "shards": len(markers),
        "records": len(feature_uids),
        "dtype": "torch.bfloat16",
        "per_record_shape": [28, 3584],
        "nonfinite_values": nonfinite,
        "passed": not any(nonfinite.values()),
    }


def _dataset_table(summary: dict) -> str:
    lines = [
        "| Dataset | Correct | Wrong | Total |",
        "|---|---:|---:|---:|",
    ]
    for dataset in STAGE1_DATASETS:
        row = summary["by_dataset"][dataset]
        lines.append(
            f"| {dataset.upper()} | {row['correct']:,} | {row['wrong']:,} | {row['total']:,} |"
        )
    row = summary["overall"]
    lines.append(
        f"| **Overall** | **{row['correct']:,}** | **{row['wrong']:,}** | **{row['total']:,}** |"
    )
    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    is_smoke = args.mode == "aggregate-smoke"
    manifest_path = Path(args.manifest)
    candidates = _load_execution_rows(manifest_path, "smoke" if is_smoke else "full")
    worker_root = output_root / ("smoke/workers" if is_smoke else "workers")
    outputs, skips, feature_index, feature_integrity = _load_worker_artifacts(
        worker_root, {0, 1, 2, 3}
    )
    validate_attempt_coverage(candidates, outputs, skips)
    if is_smoke:
        if skips or len(outputs) != len(candidates):
            raise ValueError(f"smoke did not complete cleanly: completed={len(outputs)} skipped={len(skips)}")
        write_json_atomic(
            output_root / "smoke" / "smoke_summary.json",
            {
                "passed": True,
                "samples": len(candidates),
                "datasets": dict(Counter(row["dataset"] for row in outputs)),
                "lmms_scored": len(outputs),
                "complete_28_layer_features": len(feature_index),
                "feature_integrity": feature_integrity,
            },
        )
        print(json.dumps({"smoke_passed": True, "samples": len(outputs)}))
        return

    outputs.sort(key=lambda row: row["uid"])
    skips.sort(key=lambda row: row["uid"])
    feature_index.sort(key=lambda row: row["uid"])
    summary = summarize_population(candidates, outputs, skips)
    summary["feature_integrity"] = feature_integrity
    summary["lmms_eval"] = read_json(Path(args.config))["lmms_eval"]
    write_jsonl_atomic(output_root / "dense_outputs.jsonl", outputs)
    write_jsonl_atomic(output_root / "skipped_samples.jsonl", skips)
    write_jsonl_atomic(output_root / "features" / "feature_index.jsonl", feature_index)
    write_json_atomic(output_root / "generation_summary.json", summary)
    write_json_atomic(
        output_root / "features" / "feature_integrity_audit.json",
        feature_integrity,
    )
    table = _dataset_table(summary)
    write_text_atomic(
        output_root / "dataset_summary.md",
        "# Current dense Stage-1 dataset summary\n\n"
        + table
        + "\n\n"
        + f"- Candidate image groups: {summary['candidate_image_groups']:,}\n"
        + f"- Completed image groups: {summary['completed_image_groups']:,}\n",
    )
    overall = summary["overall"]
    ratio = (
        float(overall["wrong"]) / float(overall["total"])
        if overall["total"]
        else 0.0
    )
    skip_lines = [
        f"- `{reason}`: {count:,}" for reason, count in summary["skip_reasons"].items()
    ] or ["- None"]
    write_text_atomic(
        output_root / "run_summary.md",
        "\n".join(
            [
                "# Current dense 8K run summary",
                "",
                f"- Candidate samples: {summary['candidate_samples']:,}",
                f"- Attempted: {summary['attempted']:,}",
                f"- Successfully completed: {summary['successfully_completed']:,}",
                f"- Skipped/failed: {summary['skipped_or_failed']:,}",
                f"- Complete 28-layer feature records: {summary['complete_28_layer_features']:,}",
                f"- Candidate/completed image groups: {summary['candidate_image_groups']:,} / {summary['completed_image_groups']:,}",
                f"- Current-runtime wrong ratio: {ratio:.6f}",
                "",
                "## Labels",
                "",
                table,
                "",
                "## LMMS-Eval metrics",
                "",
                "- GQA: `gqa` / `exact_match`, case and punctuation ignored; correct at 1.0.",
                "- ChartQA: `chartqa` / `relaxed_overall`; correct at 1.0.",
                "- TextVQA: `textvqa_val` / EvalAI leave-one-out `exact_match`; raw score preserved, correct at >= 0.5.",
                "",
                "## Historical bucket comparison (analysis only)",
                "",
                *[
                    f"- `{name}`: {count:,}"
                    for name, count in summary["historical_vs_current"].items()
                ],
                "",
                "## Skip reasons",
                "",
                *skip_lines,
                "",
                "The current LMMS-Eval dense outcome is authoritative. Historical buckets are metadata only.",
            ]
        )
        + "\n",
    )
    print(json.dumps(summary, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prepare", "smoke", "aggregate-smoke", "full", "aggregate-full"),
        required=True,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--portable-manifest")
    parser.add_argument("--image-root")
    parser.add_argument("--manifest")
    parser.add_argument("--model-path")
    parser.add_argument("--run-id", default="current_dense_8k_v1")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.mode == "prepare" and (not args.portable_manifest or not args.image_root):
        parser.error("prepare requires --portable-manifest and --image-root")
    if args.mode in {"smoke", "full"} and (not args.manifest or not args.model_path):
        parser.error(f"{args.mode} requires --manifest and --model-path")
    if args.mode.startswith("aggregate") and not args.manifest:
        parser.error(f"{args.mode} requires --manifest")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode in {"smoke", "full"}:
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
