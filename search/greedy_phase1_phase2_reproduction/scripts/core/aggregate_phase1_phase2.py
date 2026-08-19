#!/usr/bin/env python3
"""Validate and merge the completed 10k Phase-1 and Phase-2 mask datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "10k_dataset_mask" / "manifests" / "all_samples.jsonl"
DEFAULT_PHASE1 = ROOT / "10k_dataset_mask" / "raw" / "phase1"
DEFAULT_PHASE2 = ROOT / "10k_dataset_mask" / "raw" / "phase2_node06"
DEFAULT_CONFIG = ROOT / "10k_dataset_mask" / "config" / "collection_config.json"
DEFAULT_GATE = ROOT / "10k_dataset_mask" / "gate_v6" / "summary.json"
DEFAULT_OUTPUT = ROOT / "10k_dataset_mask" / "final_phase1_phase2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gate-summary", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-samples", type=int, default=10000)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def recorded_path(value: str) -> Path:
    """Resolve either the legacy ROOT-relative or relocated absolute path form."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def mask_key(mask: list[int]) -> str:
    return "".join(str(int(value)) for value in mask)


def one_based(mask: list[int]) -> list[int]:
    return [index + 1 for index, value in enumerate(mask) if value]


def validate_execution(row: dict[str, Any], *, uid: str, context: str) -> None:
    route_id = str(row.get("route_id") or "")
    if not route_id.startswith(f"{uid}:mask:"):
        raise RuntimeError(f"{context}: invalid route_id for {uid}: {route_id}")
    mask = row.get("visual_on_mask")
    if not isinstance(mask, list) or len(mask) != 28 or any(value not in (0, 1, False, True) for value in mask):
        raise RuntimeError(f"{context}: invalid 28-layer mask for {route_id}")
    normalized = [int(value) for value in mask]
    if int(row.get("num_visual_on_layers", -1)) != sum(normalized):
        raise RuntimeError(f"{context}: budget mismatch for {route_id}")
    if list(row.get("mask_one_based") or []) != one_based(normalized):
        raise RuntimeError(f"{context}: one-based mask mismatch for {route_id}")
    if not isinstance(row.get("result_correct"), bool):
        raise RuntimeError(f"{context}: result_correct is not bool for {route_id}")
    score = float(row.get("score"))
    if not math.isfinite(score):
        raise RuntimeError(f"{context}: non-finite score for {route_id}")
    generated_ids = row.get("generated_ids")
    if not isinstance(generated_ids, list) or any(not isinstance(value, int) for value in generated_ids):
        raise RuntimeError(f"{context}: invalid generated_ids for {route_id}")


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
    }


def write_line(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def open_temp(output_dir: Path, name: str) -> tuple[Path, Path, TextIO]:
    final_path = output_dir / name
    temp_path = output_dir / f"{name}.tmp"
    return temp_path, final_path, temp_path.open("w", encoding="utf-8")


def group_factory() -> dict[str, Any]:
    group: dict[str, Any] = defaultdict(int)
    group["source_buckets"] = Counter()
    for key in (
        "phase1_min_correct_budget_values",
        "phase2_new_min_correct_budget_values",
        "combined_min_correct_budget_values",
    ):
        group[key] = []
    return group


def audit_raw_shards(input_dir: Path, expected_samples: int) -> dict[str, Any]:
    shard_dirs = sorted(path for path in input_dir.glob("shard_*_of_*") if path.is_dir())
    if not shard_dirs:
        raise RuntimeError(f"no shard directories under {input_dir}")
    selected_total = sample_file_total = error_row_total = 0
    shard_rows = []
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "summary.json"
        errors_path = shard_dir / "errors.jsonl"
        if not summary_path.is_file() or not errors_path.is_file():
            raise RuntimeError(f"missing summary/errors file in {shard_dir}")
        summary = load_json(summary_path)
        selected = int(summary.get("selected_samples", summary.get("selected", -1)))
        summary_errors = int(summary.get("errors_this_run", summary.get("errors", -1)))
        sample_files = len(list((shard_dir / "samples").glob("*.json")))
        with errors_path.open("r", encoding="utf-8") as handle:
            error_rows = sum(1 for line in handle if line.strip())
        if selected != sample_files or summary_errors != 0 or error_rows != 0:
            raise RuntimeError(
                f"invalid shard completion {shard_dir}: selected={selected} "
                f"files={sample_files} summary_errors={summary_errors} error_rows={error_rows}"
            )
        selected_total += selected
        sample_file_total += sample_files
        error_row_total += error_rows
        shard_rows.append(
            {
                "shard": shard_dir.name,
                "selected": selected,
                "sample_files": sample_files,
                "error_rows": error_rows,
                "summary": relative(summary_path),
            }
        )
    if selected_total != expected_samples or sample_file_total != expected_samples:
        raise RuntimeError(
            f"raw shard total mismatch under {input_dir}: selected={selected_total} files={sample_file_total}"
        )
    return {
        "shards": len(shard_dirs),
        "selected_samples": selected_total,
        "sample_files": sample_file_total,
        "error_rows": error_row_total,
        "shard_rows": shard_rows,
    }


def add_sample_to_group(group: dict[str, Any], row: dict[str, Any]) -> None:
    group["samples"] += 1
    group["source_buckets"][row["source_bucket"]] += 1
    for key in (
        "phase1_candidates",
        "phase1_correct_candidates",
        "permutation_finals",
        "correct_permutation_finals",
        "phase2_requests",
        "phase2_reused_phase1_requests",
        "phase2_new_candidates",
        "phase2_new_correct_candidates",
        "combined_candidates",
        "combined_correct_candidates",
    ):
        group[key] += int(row[key])
    for key in (
        "all_on_correct",
        "all_off_correct",
        "phase1_has_correct",
        "permutation_has_correct",
        "phase2_new_has_correct",
        "combined_has_correct",
        "phase2_rescued",
        "phase2_reduced_min_budget",
    ):
        group[key] += int(bool(row[key]))
    for key in ("phase1_min_correct_budget", "phase2_new_min_correct_budget", "combined_min_correct_budget"):
        value = row[key]
        if value is not None:
            group[f"{key}_values"].append(float(value))


def finalize_group(benchmark: str, data_split: str, group: dict[str, Any]) -> dict[str, Any]:
    samples = int(group["samples"])
    phase1_candidates = int(group["phase1_candidates"])
    phase2_new = int(group["phase2_new_candidates"])
    combined = int(group["combined_candidates"])
    return {
        "benchmark": benchmark,
        "data_split": data_split,
        "samples": samples,
        "source_bucket_counts": dict(sorted(group["source_buckets"].items())),
        "all_on_correct_samples": int(group["all_on_correct"]),
        "all_on_accuracy": group["all_on_correct"] / samples,
        "all_off_correct_samples": int(group["all_off_correct"]),
        "all_off_accuracy": group["all_off_correct"] / samples,
        "phase1_candidates": phase1_candidates,
        "phase1_candidates_per_sample": phase1_candidates / samples,
        "phase1_correct_candidates": int(group["phase1_correct_candidates"]),
        "phase1_candidate_correct_rate": group["phase1_correct_candidates"] / phase1_candidates,
        "phase1_samples_with_correct_candidate": int(group["phase1_has_correct"]),
        "phase1_correct_coverage": group["phase1_has_correct"] / samples,
        "phase1_min_correct_budget": describe(group["phase1_min_correct_budget_values"]),
        "permutation_finals": int(group["permutation_finals"]),
        "correct_permutation_finals": int(group["correct_permutation_finals"]),
        "permutation_final_correct_rate": group["correct_permutation_finals"] / group["permutation_finals"],
        "samples_with_correct_permutation_final": int(group["permutation_has_correct"]),
        "phase2_requests": int(group["phase2_requests"]),
        "phase2_reused_phase1_requests": int(group["phase2_reused_phase1_requests"]),
        "phase2_new_candidates": phase2_new,
        "phase2_new_candidates_per_sample": phase2_new / samples,
        "phase2_new_correct_candidates": int(group["phase2_new_correct_candidates"]),
        "phase2_new_candidate_correct_rate": group["phase2_new_correct_candidates"] / phase2_new,
        "phase2_samples_with_new_correct_candidate": int(group["phase2_new_has_correct"]),
        "phase2_new_correct_coverage": group["phase2_new_has_correct"] / samples,
        "phase2_new_min_correct_budget": describe(group["phase2_new_min_correct_budget_values"]),
        "combined_candidates": combined,
        "combined_candidates_per_sample": combined / samples,
        "combined_correct_candidates": int(group["combined_correct_candidates"]),
        "combined_candidate_correct_rate": group["combined_correct_candidates"] / combined,
        "combined_samples_with_correct_candidate": int(group["combined_has_correct"]),
        "combined_correct_coverage": group["combined_has_correct"] / samples,
        "combined_min_correct_budget": describe(group["combined_min_correct_budget_values"]),
        "phase2_rescued_samples": int(group["phase2_rescued"]),
        "phase2_reduced_min_budget_samples": int(group["phase2_reduced_min_budget"]),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.phase1_dir = args.phase1_dir.resolve()
    args.phase2_dir = args.phase2_dir.resolve()
    args.config = args.config.resolve()
    args.gate_summary = args.gate_summary.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.config)
    gate = load_json(args.gate_summary)
    expected_orders = list(config["search"]["orders"])
    if not (
        gate.get("pass_current_hf_binary_ids")
        and gate.get("pass_available_saved_ids")
        and gate.get("pass_source_score")
        and int(gate.get("errors", -1)) == 0
    ):
        raise RuntimeError(f"gate did not pass: {gate}")

    manifest: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(args.manifest):
        uid = str(row["uid"])
        if uid in manifest:
            raise RuntimeError(f"duplicate manifest UID: {uid}")
        manifest[uid] = row
    if len(manifest) != args.expected_samples:
        raise RuntimeError(f"manifest count {len(manifest)} != {args.expected_samples}")

    phase1_paths = sorted(args.phase1_dir.glob("shard_*_of_*/samples/*.json"), key=lambda path: path.name)
    phase2_paths = sorted(args.phase2_dir.glob("shard_*_of_*/samples/*.json"), key=lambda path: path.name)
    if len(phase1_paths) != args.expected_samples or len(phase2_paths) != args.expected_samples:
        raise RuntimeError(f"raw sample counts phase1={len(phase1_paths)} phase2={len(phase2_paths)}")
    phase2_by_name: dict[str, Path] = {}
    for path in phase2_paths:
        if path.name in phase2_by_name:
            raise RuntimeError(f"duplicate Phase-2 filename: {path.name}")
        phase2_by_name[path.name] = path

    output_specs = [
        open_temp(args.output_dir, "evaluated_mask_candidates.jsonl"),
        open_temp(args.output_dir, "phase1_permutation_final_masks.jsonl"),
        open_temp(args.output_dir, "phase2_route_requests.jsonl"),
        open_temp(args.output_dir, "sample_index.jsonl"),
    ]
    candidate_handle = output_specs[0][2]
    final_handle = output_specs[1][2]
    request_handle = output_specs[2][2]
    sample_handle = output_specs[3][2]

    seen_phase1_uids: set[str] = set()
    seen_phase2_uids: set[str] = set()
    totals = Counter()
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(group_factory)
    family_stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    order_stats: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    phase1_runtime: dict[str, Any] | None = None
    phase2_runtime: dict[str, Any] | None = None

    try:
        for sample_index, phase1_path in enumerate(phase1_paths, 1):
            phase2_path = phase2_by_name.get(phase1_path.name)
            if phase2_path is None:
                raise RuntimeError(f"missing Phase-2 counterpart for {phase1_path.name}")
            phase1 = load_json(phase1_path)
            phase2 = load_json(phase2_path)
            if phase1_runtime is None:
                phase1_runtime = phase1["runtime"]
                phase2_runtime = phase2["runtime"]
            elif phase1["runtime"] != phase1_runtime or phase2["runtime"] != phase2_runtime:
                raise RuntimeError(f"runtime fingerprint mismatch for {phase1_path.name}")
            sample = phase1["sample"]
            uid = str(sample["uid"])
            benchmark = str(sample["benchmark"])
            data_split = str(sample["data_split"])
            source_bucket = str(sample["source_bucket"])
            if uid in seen_phase1_uids:
                raise RuntimeError(f"duplicate Phase-1 UID: {uid}")
            seen_phase1_uids.add(uid)
            phase2_uid = str(phase2["sample"]["uid"])
            if phase2_uid in seen_phase2_uids:
                raise RuntimeError(f"duplicate Phase-2 UID: {phase2_uid}")
            seen_phase2_uids.add(phase2_uid)
            if phase2_uid != uid:
                raise RuntimeError(f"Phase UID mismatch: {uid} != {phase2_uid}")
            manifest_row = manifest.get(uid)
            if manifest_row is None:
                raise RuntimeError(f"raw UID not in manifest: {uid}")
            for key in ("benchmark", "data_split", "sample_id", "source_bucket"):
                if str(sample[key]) != str(manifest_row[key]) or str(phase2["sample"][key]) != str(sample[key]):
                    raise RuntimeError(f"sample metadata mismatch for {uid}: {key}")
            if phase1_path.resolve() != recorded_path(str(phase2["phase1_sample_file"])):
                raise RuntimeError(f"Phase-2 Phase-1 path mismatch for {uid}")

            phase1_by_id: dict[str, dict[str, Any]] = {}
            phase1_by_mask: dict[str, str] = {}
            for execution in phase1["candidate_executions"]:
                validate_execution(execution, uid=uid, context="phase1")
                route_id = str(execution["route_id"])
                key = mask_key(execution["visual_on_mask"])
                if route_id in phase1_by_id or key in phase1_by_mask:
                    raise RuntimeError(f"duplicate Phase-1 route/mask for {uid}: {route_id}")
                phase1_by_id[route_id] = execution
                phase1_by_mask[key] = route_id

            all_on_id = str(phase1["anchors"]["all_on_route_id"])
            all_off_id = str(phase1["anchors"]["all_off_route_id"])
            if all_on_id not in phase1_by_id or all_off_id not in phase1_by_id:
                raise RuntimeError(f"missing anchor execution for {uid}")
            all_on = phase1_by_id[all_on_id]
            all_off = phase1_by_id[all_off_id]
            if sum(all_on["visual_on_mask"]) != 28 or sum(all_off["visual_on_mask"]) != 0:
                raise RuntimeError(f"anchor mask mismatch for {uid}")
            if not math.isclose(float(phase1["target_score"]), float(all_on["score"]), abs_tol=1e-9):
                raise RuntimeError(f"target/all-on score mismatch for {uid}")

            finals = list(phase1["permutation_finals"])
            if len(finals) != len(expected_orders) or {row["order"] for row in finals} != set(expected_orders):
                raise RuntimeError(f"permutation order mismatch for {uid}")
            final_orders_by_route: dict[str, list[str]] = defaultdict(list)
            final_rows_for_sample = []
            for final in finals:
                route_id = str(final["final_route_id"])
                execution = phase1_by_id.get(route_id)
                if execution is None:
                    raise RuntimeError(f"missing final route for {uid}: {route_id}")
                if list(final["final_mask_one_based"]) != list(execution["mask_one_based"]):
                    raise RuntimeError(f"final mask mismatch for {uid}: {final['order']}")
                if int(final["final_num_visual_on_layers"]) != int(execution["num_visual_on_layers"]):
                    raise RuntimeError(f"final budget mismatch for {uid}: {final['order']}")
                if bool(final["final_correct"]) != bool(execution["result_correct"]):
                    raise RuntimeError(f"final correctness mismatch for {uid}: {final['order']}")
                if not math.isclose(float(final["final_score"]), float(execution["score"]), abs_tol=1e-9):
                    raise RuntimeError(f"final score mismatch for {uid}: {final['order']}")
                final_orders_by_route[route_id].append(str(final["order"]))
                expanded_final = {
                    "uid": uid,
                    "sample_id": sample["sample_id"],
                    "benchmark": benchmark,
                    "data_split": data_split,
                    "source_bucket": source_bucket,
                    "order": final["order"],
                    "order_layers_one_based": final["order_layers_one_based"],
                    "accepted_removed_layers_one_based": final["accepted_removed_layers_one_based"],
                    "route_id": route_id,
                    "visual_on_mask": execution["visual_on_mask"],
                    "mask_one_based": execution["mask_one_based"],
                    "num_visual_on_layers": execution["num_visual_on_layers"],
                    "score": execution["score"],
                    "result_correct": execution["result_correct"],
                }
                write_line(final_handle, expanded_final)
                final_rows_for_sample.append(expanded_final)
                for group_key in ((benchmark, data_split), (benchmark, "all"), ("all", data_split), ("all", "all")):
                    order_group = order_stats[(group_key[0], group_key[1], str(final["order"]))]
                    order_group["samples"] += 1
                    order_group["correct"] += int(bool(execution["result_correct"]))
                    order_group["budget_sum"] += int(execution["num_visual_on_layers"])
                    if execution["result_correct"]:
                        order_group["correct_budget_sum"] += int(execution["num_visual_on_layers"])

            if len(phase1["search_trace"]) != 28 * len(expected_orders):
                raise RuntimeError(f"search trace count mismatch for {uid}")
            for trace in phase1["search_trace"]:
                if trace["candidate_route_id"] not in phase1_by_id or trace["parent_route_id"] not in phase1_by_id:
                    raise RuntimeError(f"search trace route reference mismatch for {uid}")

            request_by_id: dict[str, dict[str, Any]] = {}
            for request in phase2["route_requests"]:
                route_id = str(request["route_id"])
                if route_id in request_by_id:
                    raise RuntimeError(f"duplicate Phase-2 request for {uid}: {route_id}")
                request_by_id[route_id] = request
            phase2_new_by_id: dict[str, dict[str, Any]] = {}
            phase2_new_by_mask: dict[str, str] = {}
            for execution in phase2["new_candidate_executions"]:
                validate_execution(execution, uid=uid, context="phase2")
                route_id = str(execution["route_id"])
                key = mask_key(execution["visual_on_mask"])
                if route_id in phase2_new_by_id or key in phase2_new_by_mask:
                    raise RuntimeError(f"duplicate Phase-2 route/mask for {uid}: {route_id}")
                if route_id in phase1_by_id or key in phase1_by_mask:
                    raise RuntimeError(f"Phase-2 new route already exists in Phase 1 for {uid}: {route_id}")
                phase2_new_by_id[route_id] = execution
                phase2_new_by_mask[key] = route_id
            expected_request_ids = set(phase1_by_id).intersection(request_by_id).union(phase2_new_by_id)
            if set(request_by_id) != expected_request_ids:
                raise RuntimeError(f"Phase-2 request execution coverage mismatch for {uid}")
            for route_id, request in request_by_id.items():
                in_phase1 = route_id in phase1_by_id
                if bool(request["already_in_phase1"]) != in_phase1:
                    raise RuntimeError(f"Phase-2 reuse flag mismatch for {uid}: {route_id}")
                execution = phase1_by_id.get(route_id) or phase2_new_by_id.get(route_id)
                if execution is None:
                    raise RuntimeError(f"missing requested execution for {uid}: {route_id}")
                if not in_phase1 and list(execution["origins"]) != list(request["origins"]):
                    raise RuntimeError(f"Phase-2 origin mismatch for {uid}: {route_id}")
                write_line(
                    request_handle,
                    {
                        "uid": uid,
                        "sample_id": sample["sample_id"],
                        "benchmark": benchmark,
                        "data_split": data_split,
                        "source_bucket": source_bucket,
                        "route_id": route_id,
                        "visual_on_mask": execution["visual_on_mask"],
                        "mask_one_based": execution["mask_one_based"],
                        "num_visual_on_layers": execution["num_visual_on_layers"],
                        "already_in_phase1": in_phase1,
                        "origins": request["origins"],
                    },
                )
                request_families = {str(origin["family"]) for origin in request["origins"]}
                for family in request_families:
                    stats = family_stats[("phase2", family)]
                    stats["unique_routes"] += 1
                    stats["already_in_phase1_routes"] += int(in_phase1)
                    stats["new_routes"] += int(not in_phase1)
                    stats["new_correct_routes"] += int(not in_phase1 and execution["result_correct"])
                    stats["budget_sum"] += int(execution["num_visual_on_layers"])
                for origin in request["origins"]:
                    family_stats[("phase2", str(origin["family"]))]["origin_records"] += 1

            for route_id, execution in phase1_by_id.items():
                request = request_by_id.get(route_id)
                phase2_origins = list(request["origins"]) if request else []
                phase1_origins = list(execution["origins"])
                candidate_row = {
                    "uid": uid,
                    "sample_id": sample["sample_id"],
                    "benchmark": benchmark,
                    "data_split": data_split,
                    "source_bucket": source_bucket,
                    "route_id": route_id,
                    "mask_key": mask_key(execution["visual_on_mask"]),
                    "visual_on_mask": execution["visual_on_mask"],
                    "mask_one_based": execution["mask_one_based"],
                    "num_visual_on_layers": execution["num_visual_on_layers"],
                    "prediction": execution["prediction"],
                    "generated_ids": execution["generated_ids"],
                    "score": execution["score"],
                    "result_correct": execution["result_correct"],
                    "observed_phase": "phase1",
                    "phase1_origins": phase1_origins,
                    "phase2_origins": phase2_origins,
                    "phase2_requested": request is not None,
                    "is_all_on": route_id == all_on_id,
                    "is_all_off": route_id == all_off_id,
                    "null_visual_route": route_id == all_off_id,
                    "is_permutation_final": route_id in final_orders_by_route,
                    "permutation_final_orders": sorted(final_orders_by_route.get(route_id, [])),
                }
                for optional_key in ("cache_lengths_unique", "full_prompt_tokens", "text_tokens", "visual_tokens"):
                    if optional_key in execution:
                        candidate_row[optional_key] = execution[optional_key]
                write_line(candidate_handle, candidate_row)
                phase1_families = {str(origin["family"]) for origin in phase1_origins}
                for family in phase1_families:
                    stats = family_stats[("phase1", family)]
                    stats["unique_routes"] += 1
                    stats["correct_routes"] += int(bool(execution["result_correct"]))
                    stats["budget_sum"] += int(execution["num_visual_on_layers"])
                for origin in phase1_origins:
                    family_stats[("phase1", str(origin["family"]))]["origin_records"] += 1

            for route_id, execution in phase2_new_by_id.items():
                request = request_by_id[route_id]
                write_line(
                    candidate_handle,
                    {
                        "uid": uid,
                        "sample_id": sample["sample_id"],
                        "benchmark": benchmark,
                        "data_split": data_split,
                        "source_bucket": source_bucket,
                        "route_id": route_id,
                        "mask_key": mask_key(execution["visual_on_mask"]),
                        "visual_on_mask": execution["visual_on_mask"],
                        "mask_one_based": execution["mask_one_based"],
                        "num_visual_on_layers": execution["num_visual_on_layers"],
                        "prediction": execution["prediction"],
                        "generated_ids": execution["generated_ids"],
                        "score": execution["score"],
                        "result_correct": execution["result_correct"],
                        "observed_phase": "phase2",
                        "phase1_origins": [],
                        "phase2_origins": request["origins"],
                        "phase2_requested": True,
                        "is_all_on": False,
                        "is_all_off": False,
                        "null_visual_route": False,
                        "is_permutation_final": False,
                        "permutation_final_orders": [],
                    },
                )

            phase1_correct = [row for row in phase1_by_id.values() if row["result_correct"]]
            phase2_new_correct = [row for row in phase2_new_by_id.values() if row["result_correct"]]
            combined_correct = phase1_correct + phase2_new_correct
            phase1_min = min((int(row["num_visual_on_layers"]) for row in phase1_correct), default=None)
            phase2_min = min((int(row["num_visual_on_layers"]) for row in phase2_new_correct), default=None)
            combined_min = min((int(row["num_visual_on_layers"]) for row in combined_correct), default=None)
            sample_row = {
                "uid": uid,
                "sample_id": sample["sample_id"],
                "benchmark": benchmark,
                "data_split": data_split,
                "source_bucket": source_bucket,
                "correctness_threshold": sample["correctness_threshold"],
                "target_score": phase1["target_score"],
                "all_on_route_id": all_on_id,
                "all_on_score": all_on["score"],
                "all_on_correct": all_on["result_correct"],
                "all_off_route_id": all_off_id,
                "all_off_score": all_off["score"],
                "all_off_correct": all_off["result_correct"],
                "phase1_candidates": len(phase1_by_id),
                "phase1_correct_candidates": len(phase1_correct),
                "phase1_has_correct": bool(phase1_correct),
                "phase1_min_correct_budget": phase1_min,
                "permutation_finals": len(finals),
                "correct_permutation_finals": sum(bool(row["result_correct"]) for row in final_rows_for_sample),
                "permutation_has_correct": any(row["result_correct"] for row in final_rows_for_sample),
                "phase2_requests": len(request_by_id),
                "phase2_reused_phase1_requests": sum(bool(row["already_in_phase1"]) for row in request_by_id.values()),
                "phase2_new_candidates": len(phase2_new_by_id),
                "phase2_new_correct_candidates": len(phase2_new_correct),
                "phase2_new_has_correct": bool(phase2_new_correct),
                "phase2_new_min_correct_budget": phase2_min,
                "combined_candidates": len(phase1_by_id) + len(phase2_new_by_id),
                "combined_correct_candidates": len(combined_correct),
                "combined_has_correct": bool(combined_correct),
                "combined_min_correct_budget": combined_min,
                "phase2_rescued": not phase1_correct and bool(phase2_new_correct),
                "phase2_reduced_min_budget": phase1_min is not None and combined_min is not None and combined_min < phase1_min,
                "phase1_sample_file": relative(phase1_path),
                "phase2_sample_file": relative(phase2_path),
            }
            write_line(sample_handle, sample_row)
            for group_key in ((benchmark, data_split), (benchmark, "all"), ("all", data_split), ("all", "all")):
                add_sample_to_group(groups[group_key], sample_row)

            totals["samples"] += 1
            totals["phase1_candidates"] += len(phase1_by_id)
            totals["phase1_search_trace_rows"] += len(phase1["search_trace"])
            totals["phase1_permutation_finals"] += len(finals)
            totals["phase2_requests"] += len(request_by_id)
            totals["phase2_reused_phase1_requests"] += sum(bool(row["already_in_phase1"]) for row in request_by_id.values())
            totals["phase2_new_candidates"] += len(phase2_new_by_id)
            totals["combined_candidates"] += len(phase1_by_id) + len(phase2_new_by_id)
            if sample_index % 100 == 0:
                print(json.dumps({"processed_samples": sample_index, "combined_candidates": totals["combined_candidates"]}), flush=True)
    finally:
        for _, _, handle in output_specs:
            handle.close()

    expected_uids = set(manifest)
    if seen_phase1_uids != expected_uids or seen_phase2_uids != expected_uids:
        raise RuntimeError(
            "UID set mismatch: "
            f"phase1_missing={len(expected_uids - seen_phase1_uids)} "
            f"phase1_extra={len(seen_phase1_uids - expected_uids)} "
            f"phase2_missing={len(expected_uids - seen_phase2_uids)} "
            f"phase2_extra={len(seen_phase2_uids - expected_uids)}"
        )

    phase1_raw_audit = audit_raw_shards(args.phase1_dir, args.expected_samples)
    phase2_raw_audit = audit_raw_shards(args.phase2_dir, args.expected_samples)

    for temp_path, final_path, _ in output_specs:
        os.replace(temp_path, final_path)

    group_rows = [finalize_group(benchmark, split, group) for (benchmark, split), group in sorted(groups.items())]
    group_jsonl = args.output_dir / "benchmark_split_summary.jsonl"
    with group_jsonl.open("w", encoding="utf-8") as handle:
        for row in group_rows:
            write_line(handle, row)
    group_json = args.output_dir / "benchmark_split_summary.json"
    group_json.write_text(
        json.dumps({f"{row['data_split']}/{row['benchmark']}": row for row in group_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    family_rows = []
    for (phase, family), stats in sorted(family_stats.items()):
        unique_routes = int(stats["unique_routes"])
        row = {
            "phase": phase,
            "family": family,
            "origin_records": int(stats["origin_records"]),
            "unique_routes": unique_routes,
            "mean_visual_on_layers": stats["budget_sum"] / unique_routes,
        }
        if phase == "phase1":
            row.update(
                {
                    "correct_routes": int(stats["correct_routes"]),
                    "correct_rate": stats["correct_routes"] / unique_routes,
                }
            )
        else:
            new_routes = int(stats["new_routes"])
            row.update(
                {
                    "already_in_phase1_routes": int(stats["already_in_phase1_routes"]),
                    "new_routes": new_routes,
                    "new_correct_routes": int(stats["new_correct_routes"]),
                    "new_correct_rate": stats["new_correct_routes"] / new_routes if new_routes else None,
                }
            )
        family_rows.append(row)
    family_path = args.output_dir / "origin_family_summary.jsonl"
    with family_path.open("w", encoding="utf-8") as handle:
        for row in family_rows:
            write_line(handle, row)

    order_rows = []
    for (benchmark, split, order), stats in sorted(order_stats.items()):
        order_rows.append(
            {
                "benchmark": benchmark,
                "data_split": split,
                "order": order,
                "samples": int(stats["samples"]),
                "correct_finals": int(stats["correct"]),
                "correct_rate": stats["correct"] / stats["samples"],
                "mean_visual_on_layers": stats["budget_sum"] / stats["samples"],
                "mean_visual_on_layers_correct_only": (
                    stats["correct_budget_sum"] / stats["correct"] if stats["correct"] else None
                ),
            }
        )
    order_path = args.output_dir / "phase1_permutation_order_summary.jsonl"
    with order_path.open("w", encoding="utf-8") as handle:
        for row in order_rows:
            write_line(handle, row)

    summary = {
        "dataset_version": "10k_mask_candidates_phase1_phase2_final_v1",
        "decision": "pass_complete_no_ranking_constructed",
        "manifest": relative(args.manifest),
        "phase1_raw": relative(args.phase1_dir),
        "phase2_raw": relative(args.phase2_dir),
        "gate_summary": relative(args.gate_summary),
        "gate_decision": gate.get("decision"),
        "phase1_runtime": phase1_runtime,
        "phase2_runtime": phase2_runtime,
        "phase1_raw_audit": phase1_raw_audit,
        "phase2_raw_audit": phase2_raw_audit,
        "samples": int(totals["samples"]),
        "phase1_candidates": int(totals["phase1_candidates"]),
        "phase1_search_trace_rows": int(totals["phase1_search_trace_rows"]),
        "phase1_permutation_finals": int(totals["phase1_permutation_finals"]),
        "phase2_requests": int(totals["phase2_requests"]),
        "phase2_reused_phase1_requests": int(totals["phase2_reused_phase1_requests"]),
        "phase2_new_candidates": int(totals["phase2_new_candidates"]),
        "combined_unique_evaluated_candidates": int(totals["combined_candidates"]),
        "ranking_constructed": False,
        "preference_pairs_constructed": False,
        "golden_mask_constructed": False,
        "validation": {
            "manifest_phase1_uid_exact_match": True,
            "manifest_phase2_uid_exact_match": True,
            "phase1_phase2_uid_exact_match": True,
            "anchor_masks_and_target_scores_valid": True,
            "permutation_final_references_valid": True,
            "search_trace_references_valid": True,
            "phase2_request_references_and_reuse_flags_valid": True,
            "combined_routes_unique_within_sample": True,
            "raw_error_rows": 0,
        },
        "outputs": {
            "evaluated_mask_candidates": "evaluated_mask_candidates.jsonl",
            "phase1_permutation_final_masks": "phase1_permutation_final_masks.jsonl",
            "phase2_route_requests": "phase2_route_requests.jsonl",
            "sample_index": "sample_index.jsonl",
            "benchmark_split_summary": "benchmark_split_summary.jsonl",
            "origin_family_summary": "origin_family_summary.jsonl",
            "phase1_permutation_order_summary": "phase1_permutation_order_summary.jsonl",
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_paths = [
        path
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "checksums.sha256" and not path.name.endswith(".tmp")
    ]
    checksum_file = args.output_dir / "checksums.sha256"
    with checksum_file.open("w", encoding="utf-8") as handle:
        for path in checksum_paths:
            handle.write(f"{file_sha256(path)}  {path.name}\n")

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
