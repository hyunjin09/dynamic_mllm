#!/usr/bin/env python3
"""Run the current-runtime WeMath greedy Phase-1/Phase-2 recovery search."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from label_regeneration.greedy_recovery import (
    PHASE1_ORDERS,
    SCORE_TOLERANCE,
    acceptance_decision,
    candidate_plan,
    layer_order,
    route_id,
    route_key,
    select_diverse_valid_routes,
)
from label_regeneration.runtime import RouteEvaluator, configure_determinism, load_frozen_model


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "outputs/label_regeneration/wemath2pro_greedy_recovery_v1"
NUM_LAYERS = 28


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def safe_filename(uid: str) -> str:
    digest = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:10]
    return uid.replace(":", "__").replace("/", "_") + f"_{digest}.json"


def load_contract(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    path = Path(args.contract)
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract_hash = sha256_file(path)
    if contract["population"]["manifest_sha256"] != sha256_file(Path(args.manifest)):
        raise RuntimeError("recovery manifest checksum does not match frozen contract")
    return contract, contract_hash


def load_mcts_record(sample: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT / str(sample["mcts_record_path"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sample["mcts_record_sha256"]:
        raise RuntimeError(f"MCTS cache checksum mismatch: {sample['uid']}")
    record = json.loads(payload)
    if record["sample"]["uid"] != sample["uid"]:
        raise RuntimeError(f"MCTS cache UID mismatch: {sample['uid']}")
    return record


def normalize_cached_route(uid: str, row: dict[str, Any]) -> dict[str, Any]:
    mask = [int(value) for value in row["visual_on_mask"]]
    output = dict(row)
    output["source_route_id"] = row.get("route_id")
    output["route_id"] = route_id(uid, mask)
    output["visual_on_mask"] = mask
    output["mask_key"] = route_key(mask)
    output["execution_source"] = "mcts_cache_reuse"
    output["origins"] = [{"family": "mcts_cache", "source_route_id": row.get("route_id")}]
    return output


class RouteCache:
    def __init__(self, evaluator: RouteEvaluator, mcts_record: dict[str, Any]):
        self.evaluator = evaluator
        self.uid = str(mcts_record["sample"]["uid"])
        self.rows = {
            route_key(row["visual_on_mask"]): normalize_cached_route(self.uid, row)
            for row in mcts_record["candidate_executions"]
        }

    def add_existing(self, rows: list[dict[str, Any]]) -> None:
        for source in rows:
            key = route_key(source["visual_on_mask"])
            row = dict(source)
            row["visual_on_mask"] = [int(value) for value in source["visual_on_mask"]]
            row["mask_key"] = key
            self.rows[key] = row

    def evaluate(self, mask: list[int], origin: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        key = route_key(mask)
        reused = key in self.rows
        if not reused:
            row = self.evaluator.evaluate(tuple(mask), origin["family"])
            row["route_id"] = route_id(self.uid, mask)
            row["execution_source"] = origin["family"]
            row["origins"] = []
            self.rows[key] = row
        origins = self.rows[key].setdefault("origins", [])
        if origin not in origins:
            origins.append(origin)
        return self.rows[key], reused


def runtime_record(args: argparse.Namespace, contract_hash: str) -> dict[str, Any]:
    return {
        "contract_file_sha256": contract_hash,
        "manifest_sha256": sha256_file(Path(args.manifest)),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "num_layers": NUM_LAYERS,
        "native_image_processing": True,
        "custom_max_image_tokens": None,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "generation": {"do_sample": False, "max_new_tokens": 96},
        "scoring_timeout_seconds": float(args.scoring_timeout_seconds),
    }


def output_complete(path: Path, uid: str, phase: str, contract_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        row.get("phase") == phase
        and row.get("sample", {}).get("uid") == uid
        and row.get("runtime", {}).get("contract_file_sha256") == contract_hash
        and row.get("status") == "complete"
    )


def select_shard(rows: list[dict[str, Any]], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard settings")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def make_evaluator(args: argparse.Namespace, sample: dict[str, Any], loaded):
    processor, base, wrapped, device = loaded
    return RouteEvaluator(
        processor=processor,
        base_model=base,
        wrapped_model=wrapped,
        sample=sample,
        device=device,
        scoring_timeout_seconds=args.scoring_timeout_seconds,
    )


def run_preflight(args: argparse.Namespace, rows: list[dict[str, Any]], loaded, contract_hash: str) -> None:
    results = []
    for sample in rows[:5]:
        mcts = load_mcts_record(sample)
        evaluator = make_evaluator(args, sample, loaded)
        native = evaluator.native_all_on()
        binary = evaluator.evaluate((1,) * NUM_LAYERS, "preflight_binary_all_on")
        cached = {
            route_key(row["visual_on_mask"]): row for row in mcts["candidate_executions"]
        }
        cached_all_on = cached["1" * NUM_LAYERS]
        mixed_key = next(
            key for key in sorted(cached) if key not in {"0" * NUM_LAYERS, "1" * NUM_LAYERS}
        )
        cached_mixed = cached[mixed_key]
        live_mixed = evaluator.evaluate(tuple(int(bit) for bit in mixed_key), "preflight_cached_mixed")
        candidate = None
        for counter in range(1024):
            bits = hashlib.sha256(f"{sample['uid']}:preflight:{counter}".encode()).digest()
            proposed = [int((bits[index // 8] >> (index % 8)) & 1) for index in range(NUM_LAYERS)]
            if route_key(proposed) not in cached:
                candidate = proposed
                break
        if candidate is None:
            raise RuntimeError(f"could not construct an uncached preflight mask for {sample['uid']}")
        first = evaluator.evaluate(tuple(candidate), "preflight_new_mixed_first")
        second = evaluator.evaluate(tuple(candidate), "preflight_new_mixed_repeat")
        checks = {
            "native_binary_all_on_token_parity": native.generated_ids == binary["generated_ids"],
            "binary_cached_all_on_token_parity": binary["generated_ids"] == cached_all_on["generated_ids"],
            "binary_cached_all_on_score_parity": binary["score"] == cached_all_on["score"],
            "cached_mixed_token_parity": live_mixed["generated_ids"] == cached_mixed["generated_ids"],
            "cached_mixed_score_parity": live_mixed["score"] == cached_mixed["score"],
            "new_mixed_repeat_token_parity": first["generated_ids"] == second["generated_ids"],
            "new_mixed_repeat_score_parity": first["score"] == second["score"],
            "native_processing": evaluator.input_metadata["processor_uses_native_defaults"],
            "no_custom_image_cap": evaluator.input_metadata["custom_max_image_tokens"] is None,
        }
        results.append(
            {
                "uid": sample["uid"],
                "checks": checks,
                "passed": all(checks.values()),
                "geometry": evaluator.geometry,
                "cached_mixed_mask": mixed_key,
                "new_mixed_mask": route_key(candidate),
            }
        )
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "schema_version": "wemath2pro_greedy_recovery_preflight_v1",
        "passed": len(results) == 5 and all(row["passed"] for row in results),
        "records": results,
        "runtime": runtime_record(args, contract_hash),
        "scientific_search_started": False,
    }
    output = Path(args.output_root) / "preflight/preflight_report_v1.json"
    atomic_json(output, report)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise RuntimeError("greedy recovery preflight failed")
    print(json.dumps({"preflight": "PASS", "records": len(results)}), flush=True)


def run_phase1(args: argparse.Namespace, rows: list[dict[str, Any]], loaded, contract_hash: str) -> None:
    selected = select_shard(rows, args.shard_index, args.num_shards)
    root = Path(args.output_root) / "phase1" / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    completed = skipped = errors = 0
    started = time.time()
    for sample in selected:
        output = root / "samples" / safe_filename(sample["uid"])
        if output_complete(output, sample["uid"], "greedy_phase1", contract_hash):
            skipped += 1
            continue
        try:
            mcts = load_mcts_record(sample)
            evaluator = make_evaluator(args, sample, loaded)
            cache = RouteCache(evaluator, mcts)
            requested_keys: set[str] = set()

            def request(mask: list[int], origin: dict[str, Any]):
                requested_keys.add(route_key(mask))
                return cache.evaluate(mask, origin)

            all_on, _ = request([1] * NUM_LAYERS, {"family": "anchor", "name": "all_on"})
            all_off, _ = request([0] * NUM_LAYERS, {"family": "anchor", "name": "all_off"})
            anchor_score = float(all_on["score"])
            traces = []
            finals = []
            for order_name in PHASE1_ORDERS:
                current_mask = [1] * NUM_LAYERS
                current = all_on
                removed = []
                order = layer_order(order_name, NUM_LAYERS, sample["uid"])
                for step, layer_index in enumerate(order, start=1):
                    parent_mask = list(current_mask)
                    candidate_mask = list(current_mask)
                    candidate_mask[layer_index] = 0
                    candidate, reused = request(
                        candidate_mask,
                        {"family": "greedy_phase1", "order": order_name, "step": step},
                    )
                    accepted = acceptance_decision(
                        candidate["score"],
                        all_on_score=anchor_score,
                        current_score=current["score"],
                    )
                    traces.append(
                        {
                            "order": order_name,
                            "step": step,
                            "tested_layer_zero_based": layer_index,
                            "tested_layer_one_based": layer_index + 1,
                            "parent_mask_key": route_key(parent_mask),
                            "candidate_mask_key": route_key(candidate_mask),
                            "candidate_execution_reused": reused,
                            "score_before": float(current["score"]),
                            "score_after": float(candidate["score"]),
                            "acceptance_score": max(anchor_score, float(current["score"])),
                            "accepted": accepted,
                        }
                    )
                    if accepted:
                        current_mask = candidate_mask
                        current = candidate
                        removed.append(layer_index + 1)
                finals.append(
                    {
                        "order": order_name,
                        "order_layers_one_based": [index + 1 for index in order],
                        "final_route_id": current["route_id"],
                        "final_mask_key": route_key(current_mask),
                        "final_mask_one_based": [index + 1 for index, bit in enumerate(current_mask) if bit],
                        "final_num_visual_on_layers": sum(current_mask),
                        "final_score": float(current["score"]),
                        "final_correct": bool(current["result_correct"]),
                        "accepted_removed_layers_one_based": removed,
                    }
                )
            payload = {
                "schema_version": "wemath2pro_greedy_phase1_v1",
                "phase": "greedy_phase1",
                "status": "complete",
                "sample": sample,
                "anchors": {
                    "all_on_route_id": all_on["route_id"],
                    "all_off_route_id": all_off["route_id"],
                },
                "target_policy": "current_binary_all_on_score",
                "target_score": anchor_score,
                "candidate_executions": [cache.rows[key] for key in sorted(requested_keys)],
                "search_trace": traces,
                "permutation_finals": finals,
                "runtime": runtime_record(args, contract_hash),
            }
            atomic_json(output, payload)
            completed += 1
            del evaluator, cache
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            errors += 1
            atomic_json(
                root / "errors" / safe_filename(sample["uid"]),
                {
                    "uid": sample["uid"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        summary = {
            "status": "running",
            "selected": len(selected),
            "completed_this_run": completed,
            "skipped_existing": skipped,
            "errors_this_run": errors,
            "last_uid": sample["uid"],
            "elapsed_seconds": time.time() - started,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "contract_file_sha256": contract_hash,
        }
        atomic_json(root / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        gc.collect()
    summary["status"] = "complete" if errors == 0 else "complete_with_errors"
    atomic_json(root / "summary.json", summary)


def aggregate_phase1(args: argparse.Namespace, rows: list[dict[str, Any]], contract_hash: str) -> None:
    root = Path(args.output_root)
    paths = sorted((root / "phase1").glob("shard_*_of_*/samples/*.json"))
    by_uid = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        uid = payload.get("sample", {}).get("uid")
        if not output_complete(path, uid, "greedy_phase1", contract_hash):
            raise RuntimeError(f"invalid Phase-1 record: {path}")
        if uid in by_uid:
            raise RuntimeError(f"duplicate Phase-1 record: {uid}")
        by_uid[uid] = (path, payload)
    expected = {row["uid"] for row in rows}
    if set(by_uid) != expected:
        raise RuntimeError(f"Phase-1 incomplete: found {len(by_uid)} expected {len(expected)}")
    budgets = []
    request_rows = []
    for sample in rows:
        path, payload = by_uid[sample["uid"]]
        unique_success = {
            final["final_mask_key"]: int(final["final_num_visual_on_layers"])
            for final in payload["permutation_finals"]
            if final["final_correct"]
        }
        if unique_success:
            budgets.append(statistics.fmean(unique_success.values()))
    if not budgets:
        raise RuntimeError("Phase-1 found no successful final mask; Phase-2 budget center undefined")
    budget_center = int(round(statistics.fmean(budgets)))
    plan_args = SimpleNamespace(seed=20260720, random_per_budget=2, local_per_operation=4)
    for sample in rows:
        path, payload = by_uid[sample["uid"]]
        plan = candidate_plan(payload, budget_center, plan_args)
        request_rows.append(
            {
                "uid": sample["uid"],
                "phase1_path": str(path.relative_to(PROJECT)),
                "budget_center": budget_center,
                "requests": [
                    {"mask_key": key, "visual_on_mask": item["route"], "origins": item["origins"]}
                    for key, item in sorted(plan.items())
                ],
            }
        )
    aggregate = {
        "schema_version": "wemath2pro_greedy_phase1_aggregate_v1",
        "status": "PASS",
        "records": len(by_uid),
        "permutation_finals": sum(len(payload["permutation_finals"]) for _, payload in by_uid.values()),
        "samples_with_successful_final": len(budgets),
        "mean_sample_success_budget": statistics.fmean(budgets),
        "median_sample_success_budget": statistics.median(budgets),
        "rounded_budget_center": budget_center,
        "phase2_requests": sum(len(row["requests"]) for row in request_rows),
        "contract_file_sha256": contract_hash,
    }
    aggregate_path = root / "phase1_aggregate/summary_v1.json"
    atomic_json(aggregate_path, aggregate)
    requests_path = root / "phase2/phase2_request_manifest_v1.jsonl"
    atomic_jsonl(requests_path, request_rows)
    for path in (aggregate_path, requests_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps(aggregate, sort_keys=True), flush=True)


def run_phase2(args: argparse.Namespace, rows: list[dict[str, Any]], loaded, contract_hash: str) -> None:
    request_path = Path(args.output_root) / "phase2/phase2_request_manifest_v1.jsonl"
    request_by_uid = {row["uid"]: row for row in read_jsonl(request_path)}
    selected = select_shard(rows, args.shard_index, args.num_shards)
    root = Path(args.output_root) / "phase2" / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    completed = skipped = errors = 0
    started = time.time()
    for sample in selected:
        output = root / "samples" / safe_filename(sample["uid"])
        if output_complete(output, sample["uid"], "greedy_phase2", contract_hash):
            skipped += 1
            continue
        try:
            request_row = request_by_uid[sample["uid"]]
            phase1 = json.loads((PROJECT / request_row["phase1_path"]).read_text(encoding="utf-8"))
            mcts = load_mcts_record(sample)
            evaluator = make_evaluator(args, sample, loaded)
            cache = RouteCache(evaluator, mcts)
            cache.add_existing(phase1["candidate_executions"])
            requests = []
            new_keys = set()
            for item in request_row["requests"]:
                existed_before = item["mask_key"] in cache.rows
                row = None
                for origin in item["origins"]:
                    row, _ = cache.evaluate(item["visual_on_mask"], origin)
                if not existed_before:
                    new_keys.add(item["mask_key"])
                requests.append(
                    {
                        "route_id": row["route_id"],
                        "mask_key": item["mask_key"],
                        "already_available": existed_before,
                        "origins": item["origins"],
                    }
                )
            payload = {
                "schema_version": "wemath2pro_greedy_phase2_v1",
                "phase": "greedy_phase2",
                "status": "complete",
                "sample": sample,
                "budget_center": int(request_row["budget_center"]),
                "route_requests": requests,
                "new_candidate_executions": [cache.rows[key] for key in sorted(new_keys)],
                "runtime": runtime_record(args, contract_hash),
            }
            atomic_json(output, payload)
            completed += 1
            del evaluator, cache
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            errors += 1
            atomic_json(
                root / "errors" / safe_filename(sample["uid"]),
                {
                    "uid": sample["uid"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        summary = {
            "status": "running",
            "selected": len(selected),
            "completed_this_run": completed,
            "skipped_existing": skipped,
            "errors_this_run": errors,
            "last_uid": sample["uid"],
            "elapsed_seconds": time.time() - started,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "contract_file_sha256": contract_hash,
        }
        atomic_json(root / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        gc.collect()
    summary["status"] = "complete" if errors == 0 else "complete_with_errors"
    atomic_json(root / "summary.json", summary)


def finalize(args: argparse.Namespace, rows: list[dict[str, Any]], contract_hash: str) -> None:
    root = Path(args.output_root)
    phase1_paths = {
        json.loads(path.read_text(encoding="utf-8"))["sample"]["uid"]: path
        for path in sorted((root / "phase1").glob("shard_*_of_*/samples/*.json"))
    }
    phase2_paths = {
        json.loads(path.read_text(encoding="utf-8"))["sample"]["uid"]: path
        for path in sorted((root / "phase2").glob("shard_*_of_*/samples/*.json"))
    }
    expected = {row["uid"] for row in rows}
    if set(phase1_paths) != expected or set(phase2_paths) != expected:
        raise RuntimeError("cannot finalize incomplete Phase-1/Phase-2 outputs")
    summaries = []
    selected_rows = []
    recovered_phase1 = recovered_phase2 = 0
    for sample in rows:
        uid = sample["uid"]
        mcts = load_mcts_record(sample)
        phase1 = json.loads(phase1_paths[uid].read_text(encoding="utf-8"))
        phase2 = json.loads(phase2_paths[uid].read_text(encoding="utf-8"))
        combined = {}
        for source_name, candidates in (
            ("mcts", mcts["candidate_executions"]),
            ("greedy_phase1", phase1["candidate_executions"]),
            ("greedy_phase2", phase2["new_candidate_executions"]),
        ):
            for candidate in candidates:
                key = route_key(candidate["visual_on_mask"])
                if key not in combined:
                    combined[key] = dict(candidate)
                    combined[key]["mask_key"] = key
                    combined[key]["visual_on_mask"] = [int(bit) for bit in candidate["visual_on_mask"]]
                    combined[key]["search_sources"] = []
                if source_name not in combined[key]["search_sources"]:
                    combined[key]["search_sources"].append(source_name)
        p1_success = any(row["result_correct"] for row in phase1["candidate_executions"])
        p2_success = any(row["result_correct"] for row in phase2["new_candidate_executions"])
        recovered_phase1 += p1_success
        recovered_phase2 += bool(not p1_success and p2_success)
        valid = [row for row in combined.values() if row["result_correct"]]
        selected = select_diverse_valid_routes(valid, max_routes=50)
        summaries.append(
            {
                "uid": uid,
                "image_group_id": sample["image_group_id"],
                "difficulty": sample["difficulty"],
                "evaluated_unique_masks_combined": len(combined),
                "valid_routes_combined": len(valid),
                "new_valid_routes": sum("mcts" not in row["search_sources"] for row in valid),
                "recovered_in_phase1": bool(p1_success),
                "incrementally_recovered_in_phase2": bool(not p1_success and p2_success),
                "derived_valid_routes": len(selected),
                "derived_route_cap": 50,
            }
        )
        selected_rows.append(
            {
                "uid": uid,
                "valid_masks": [row["visual_on_mask"] for row in selected],
                "valid_mask_keys": [row["mask_key"] for row in selected],
                "raw_valid_route_count": len(valid),
                "route_cap": 50,
            }
        )
    summary_path = root / "final/per_sample_recovery_summary_v1.jsonl"
    selected_path = root / "final/derived_max50_valid_routes_v1.jsonl"
    atomic_jsonl(summary_path, summaries)
    atomic_jsonl(selected_path, selected_rows)
    report = {
        "schema_version": "wemath2pro_greedy_recovery_final_v1",
        "status": "PASS",
        "records": len(rows),
        "phase1_recovered_records": recovered_phase1,
        "incremental_phase2_recovered_records": recovered_phase2,
        "total_recovered_records": recovered_phase1 + recovered_phase2,
        "max_derived_valid_routes_per_sample": 50,
        "raw_valid_routes_truncated": False,
        "contract_file_sha256": contract_hash,
    }
    report_path = root / "final/final_audit_v1.json"
    atomic_json(report_path, report)
    for path in (summary_path, selected_path, report_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps(report, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("preflight", "phase1", "aggregate_phase1", "phase2", "finalize"), required=True
    )
    parser.add_argument("--manifest", default=str(DEFAULT_ROOT / "manifest/recovery_manifest_v1.jsonl"))
    parser.add_argument("--contract", default=str(DEFAULT_ROOT / "frozen_execution_contract_v1.json"))
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--scoring-timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()
    contract, contract_hash = load_contract(args)
    rows = read_jsonl(Path(args.manifest))
    if len(rows) != 2278:
        raise RuntimeError(f"expected 2,278 recovery rows, found {len(rows)}")
    configure_determinism(args.seed + args.shard_index)
    if args.mode in {"aggregate_phase1", "finalize"}:
        if args.mode == "aggregate_phase1":
            aggregate_phase1(args, rows, contract_hash)
        else:
            finalize(args, rows, contract_hash)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution must run inside Slurm")
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) // 2))
    loaded = load_frozen_model(contract["model"]["path"], contract["model"]["revision"], 0)
    if args.mode == "preflight":
        run_preflight(args, rows, loaded, contract_hash)
    elif args.mode == "phase1":
        preflight = Path(args.output_root) / "preflight/preflight_report_v1.json"
        if not preflight.is_file() or not json.loads(preflight.read_text())["passed"]:
            raise RuntimeError("Phase-1 blocked: preflight has not passed")
        run_phase1(args, rows, loaded, contract_hash)
    else:
        run_phase2(args, rows, loaded, contract_hash)


if __name__ == "__main__":
    main()
