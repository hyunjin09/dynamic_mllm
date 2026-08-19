"""Strict P4 completeness and executor-contract audit for regenerated labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any


NUM_LAYERS = 28
PHASE = "binary_visual_mask_graph_mcts_regenerated_v1"
DATASET_VERSION = "8k_native_qwen_unrestricted_mask_regeneration_v1"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
EXPANSION_POLICY = "choose_layer_and_visual_on_off_from_all_undecided_layers"
SOURCE_BINDING_FIELDS = (
    "uid",
    "benchmark",
    "sample_id",
    "extraction_index",
    "source_row_sha256",
    "prompt",
    "question",
    "answer",
    "metric_name",
    "correctness_threshold",
    "max_new_tokens",
    "max_image_tokens",
    "image_group_id",
    "local_image_path",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list) or len(value) != NUM_LAYERS or any(bit not in (0, 1) for bit in value):
        return None
    return tuple(int(bit) for bit in value)


def _candidate_failures(candidate: Any, index: int, threshold: float) -> list[str]:
    prefix = f"candidate_{index}"
    if not isinstance(candidate, dict):
        return [f"{prefix}_object"]
    failures: list[str] = []
    mask = _mask(candidate.get("visual_on_mask"))
    if mask is None:
        return [f"{prefix}_mask"]
    key = "".join(str(bit) for bit in mask)
    on_count = sum(mask)
    transitions = sum(mask[layer] != mask[layer - 1] for layer in range(1, NUM_LAYERS))
    expected_one_based = [layer + 1 for layer, bit in enumerate(mask) if bit]
    checks = {
        "mask_key": candidate.get("mask_key") == key,
        "mask_one_based": candidate.get("mask_one_based") == expected_one_based,
        "on_count": candidate.get("num_visual_on_layers") == on_count,
        "off_count": candidate.get("num_visual_off_layers") == NUM_LAYERS - on_count,
        "transition_count": candidate.get("num_transitions") == transitions,
        "hamming_to_all_on": candidate.get("hamming_distance_to_all_on") == NUM_LAYERS - on_count,
        "generated_ids": isinstance(candidate.get("generated_ids"), list)
        and all(isinstance(token, int) for token in candidate["generated_ids"]),
        "route_id": isinstance(candidate.get("route_id"), str) and bool(candidate["route_id"]),
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(f"{prefix}_{name}")
    score = candidate.get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        failures.append(f"{prefix}_score")
    else:
        expected_correct = float(score) >= threshold
        if candidate.get("result_correct") is not expected_correct:
            failures.append(f"{prefix}_correctness")
        if float(candidate.get("reward", math.nan)) != float(expected_correct):
            failures.append(f"{prefix}_reward")
    if float(candidate.get("correctness_threshold", math.nan)) != threshold:
        failures.append(f"{prefix}_threshold")
    text_tokens = candidate.get("text_tokens")
    visual_tokens = candidate.get("visual_tokens")
    full_tokens = candidate.get("full_prompt_tokens")
    if not all(isinstance(value, int) and value > 0 for value in (text_tokens, visual_tokens, full_tokens)):
        failures.append(f"{prefix}_token_counts")
    elif text_tokens + visual_tokens != full_tokens:
        failures.append(f"{prefix}_token_count_sum")
    cache_lengths = candidate.get("cache_lengths_unique")
    if not isinstance(cache_lengths, list) or not cache_lengths or any(
        not isinstance(value, int) or value <= 0 for value in cache_lengths
    ):
        failures.append(f"{prefix}_cache_lengths")
    return failures


def validate_record(record: Any, source: dict[str, Any], contract_sha256: str) -> list[str]:
    """Return named contract failures for one terminal record."""
    if not isinstance(record, dict):
        return ["record_object"]
    failures: list[str] = []
    if record.get("phase") != PHASE:
        failures.append("phase")
    if record.get("dataset_version") != DATASET_VERSION:
        failures.append("dataset_version")
    if record.get("root_policy") != "all_visual_on_recomputed_current_executor":
        failures.append("root_policy")

    sample = record.get("sample")
    if not isinstance(sample, dict):
        return failures + ["sample_object"]
    for field in SOURCE_BINDING_FIELDS:
        if sample.get(field) != source.get(field):
            failures.append(f"source_binding_{field}")
    if sample.get("benchmark") not in {"gqa", "textvqa", "chartqa"}:
        failures.append("sample_dataset_scope")
    if sample.get("max_image_tokens") is not None:
        failures.append("sample_visual_token_cap")

    runtime = record.get("runtime")
    if not isinstance(runtime, dict):
        failures.append("runtime_object")
    else:
        runtime_checks = {
            "contract": runtime.get("contract_sha256") == contract_sha256,
            "model_revision": runtime.get("model_revision") == MODEL_REVISION,
            "attention": runtime.get("attn_implementation") == "sdpa",
            "dtype": runtime.get("dtype") == "bfloat16",
            "visual_token_cap": runtime.get("custom_max_image_tokens") is None,
            "native_processing": runtime.get("native_image_processing") is True,
            "processor_fast": runtime.get("processor_use_fast") is False,
            "greedy": runtime.get("generation_policy", {}).get("do_sample") is False,
            "max_new_tokens": runtime.get("generation_policy", {}).get("max_new_tokens")
            == source.get("max_new_tokens"),
        }
        failures.extend(f"runtime_{name}" for name, passed in runtime_checks.items() if not passed)

    input_metadata = sample.get("input_metadata")
    if not isinstance(input_metadata, dict):
        failures.append("input_metadata")
    else:
        if input_metadata.get("custom_max_image_tokens") is not None:
            failures.append("input_metadata_visual_token_cap")
        if input_metadata.get("processor_uses_native_defaults") is not True:
            failures.append("input_metadata_native_processing")

    threshold = float(source["correctness_threshold"])
    candidates = record.get("candidate_executions")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return failures + ["candidate_executions"]
    for index, candidate in enumerate(candidates):
        failures.extend(_candidate_failures(candidate, index, threshold))
    if any(
        not isinstance(candidate, dict) or _mask(candidate.get("visual_on_mask")) is None
        for candidate in candidates
    ):
        return failures

    candidate_by_id = {candidate.get("route_id"): candidate for candidate in candidates if isinstance(candidate, dict)}
    if len(candidate_by_id) != len(candidates):
        failures.append("duplicate_route_ids")
    candidate_masks = [_mask(candidate["visual_on_mask"]) for candidate in candidates]
    if len(set(candidate_masks)) != len(candidate_masks):
        failures.append("duplicate_candidate_masks")

    root = candidate_by_id.get(record.get("root_route_id"))
    all_off = candidate_by_id.get(record.get("all_off_route_id"))
    if root is None or _mask(root.get("visual_on_mask")) != (1,) * NUM_LAYERS:
        failures.append("all_on_root")
    if all_off is None or _mask(all_off.get("visual_on_mask")) != (0,) * NUM_LAYERS:
        failures.append("all_off_anchor")
    if root is not None:
        expected_status = "correct" if root.get("result_correct") else "wrong"
        if sample.get("current_all_on_status") != expected_status:
            failures.append("current_all_on_status")
        if sample.get("current_all_on_prediction") != root.get("prediction"):
            failures.append("current_all_on_prediction")
        if float(sample.get("current_all_on_score", math.nan)) != float(root.get("score", math.nan)):
            failures.append("current_all_on_score")

    correct_ids = {candidate["route_id"] for candidate in candidates if candidate.get("result_correct") is True}
    successful_ids = record.get("successful_route_ids")
    if not isinstance(successful_ids, list) or len(successful_ids) != len(set(successful_ids)):
        failures.append("successful_route_ids")
    elif set(successful_ids) != correct_ids:
        failures.append("successful_route_linkage")
    best_id = record.get("best_sparse_success_route_id")
    if correct_ids:
        expected_best = min(
            (candidate_by_id[route_id] for route_id in correct_ids),
            key=lambda candidate: (sum(candidate["visual_on_mask"]), tuple(candidate["visual_on_mask"])),
        )["route_id"]
        if best_id != expected_best:
            failures.append("best_sparse_success_route")
    elif best_id is not None:
        failures.append("best_sparse_success_without_success")

    mcts = record.get("mcts")
    config = record.get("mcts_config")
    if not isinstance(mcts, dict) or not isinstance(config, dict):
        return failures + ["mcts_objects"]
    completed = mcts.get("completed_simulations")
    requested = mcts.get("requested_simulations")
    simulations = mcts.get("simulations")
    if completed != requested or requested not in {200, 400, 600}:
        failures.append("mcts_completion_budget")
    if not isinstance(simulations, list) or len(simulations) != completed:
        failures.append("mcts_simulation_trace_length")
    if config.get("num_simulations") != requested:
        failures.append("mcts_config_budget")
    if config.get("fixed_layer_permutation") is not False:
        failures.append("mcts_fixed_layer_order")
    if config.get("stop_on_first_success") is not False:
        failures.append("mcts_early_stop")
    if config.get("transposition_table") is not True:
        failures.append("mcts_transposition_table")
    if config.get("expansion_policy") != EXPANSION_POLICY:
        failures.append("mcts_expansion_policy")
    root_correct = bool(root and root.get("result_correct"))
    if root_correct:
        if requested != 200 or config.get("base_num_simulations") != 200 or mcts.get("extension_reason") is not None:
            failures.append("mcts_correct_root_budget")
    else:
        if requested not in {400, 600} or config.get("base_num_simulations") != 400:
            failures.append("mcts_wrong_root_budget")
        if requested == 600 and mcts.get("extension_reason") != "no_correcting_route_after_400":
            failures.append("mcts_extension_reason")
        if requested == 400 and mcts.get("extension_reason") is not None:
            failures.append("mcts_unexpected_extension_reason")

    evaluated = mcts.get("evaluated_masks")
    if not isinstance(evaluated, list):
        failures.append("mcts_evaluated_masks")
    else:
        evaluated_ids = {row.get("route_id") for row in evaluated if isinstance(row, dict)}
        evaluated_masks = {_mask(row.get("visual_on_mask")) for row in evaluated if isinstance(row, dict)}
        if evaluated_ids != set(candidate_by_id) or evaluated_masks != set(candidate_masks):
            failures.append("mcts_candidate_linkage")
    successful_masks = mcts.get("successful_masks")
    expected_successful_masks = {_mask(candidate_by_id[route_id]["visual_on_mask"]) for route_id in correct_ids}
    if not isinstance(successful_masks, list) or {_mask(mask) for mask in successful_masks} != expected_successful_masks:
        failures.append("mcts_successful_mask_linkage")
    expected_best_mask = None if best_id is None else _mask(candidate_by_id[best_id]["visual_on_mask"])
    actual_best_mask = _mask(mcts.get("best_mask")) if mcts.get("best_mask") is not None else None
    if actual_best_mask != expected_best_mask:
        failures.append("mcts_best_mask_linkage")
    if root is not None and float(mcts.get("root_reward", math.nan)) != float(root.get("reward", math.nan)):
        failures.append("mcts_root_reward")
    if all_off is not None and float(mcts.get("all_off_reward", math.nan)) != float(all_off.get("reward", math.nan)):
        failures.append("mcts_all_off_reward")

    geometry = (
        sample.get("actual_text_tokens"),
        sample.get("actual_visual_tokens"),
        sample.get("actual_full_prompt_tokens"),
    )
    if not all(isinstance(value, int) and value > 0 for value in geometry) or geometry[0] + geometry[1] != geometry[2]:
        failures.append("sample_token_geometry")
    elif any(
        (candidate.get("text_tokens"), candidate.get("visual_tokens"), candidate.get("full_prompt_tokens")) != geometry
        for candidate in candidates
    ):
        failures.append("candidate_token_geometry_drift")
    return failures


def audit_cache(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    contract_sha256: str,
    expected_dataset_counts: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit one raw cache without loading or searching any other dataset cache."""
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_uids = [str(row.get("uid") or "") for row in manifest_rows]
    source_by_uid = {str(row["uid"]): row for row in manifest_rows if row.get("uid")}
    manifest_dataset_counts = Counter(str(row.get("benchmark")) for row in manifest_rows)
    extraction_indices = [row.get("extraction_index") for row in manifest_rows]
    extraction_indices_ok = all(isinstance(index, int) for index in extraction_indices) and sorted(
        extraction_indices
    ) == list(range(len(manifest_rows)))
    manifest_population_ok = (
        dict(manifest_dataset_counts) == expected_dataset_counts
        and set(manifest_dataset_counts) == set(expected_dataset_counts)
        and len(manifest_rows) == sum(expected_dataset_counts.values())
        and len(source_by_uid) == len(manifest_rows)
        and extraction_indices_ok
    )

    raw_root = output_root / "raw_route_cache"
    record_paths = sorted(raw_root.glob("shard_*_of_*/samples/*.json"))
    error_paths = sorted(raw_root.glob("shard_*_of_*/errors/*"))
    temporary_paths = sorted(raw_root.rglob("*.tmp.*"))
    zero_byte_paths = [path for path in record_paths if path.stat().st_size == 0]
    invalid_records: list[dict[str, Any]] = []
    records_by_uid: dict[str, list[str]] = defaultdict(list)
    seen_uids: set[str] = set()
    index_rows: list[dict[str, Any]] = []
    cache_dataset_counts: Counter[str] = Counter()
    search_budget_counts: Counter[str] = Counter()
    runtime_job_counts: Counter[str] = Counter()
    for path in record_paths:
        digest = file_sha256(path)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            sample = record.get("sample")
            uid = str(sample.get("uid") or "") if isinstance(sample, dict) else ""
            seen_uids.add(uid)
            source = source_by_uid.get(uid)
            if source is None:
                invalid_records.append(
                    {"path": str(path), "uid": uid, "sha256": digest, "failed_checks": ["unexpected_uid"]}
                )
                continue
            failures = validate_record(record, source, contract_sha256)
            if failures:
                invalid_records.append(
                    {"path": str(path), "uid": uid, "sha256": digest, "failed_checks": sorted(set(failures))}
                )
                continue
            records_by_uid[uid].append(str(path))
            cache_dataset_counts[str(source["benchmark"])] += 1
            requested = int(record["mcts"]["requested_simulations"])
            search_budget_counts[str(requested)] += 1
            runtime_job_counts[str(record["runtime"].get("slurm_job_id"))] += 1
            index_rows.append(
                {
                    "uid": uid,
                    "benchmark": source["benchmark"],
                    "extraction_index": source["extraction_index"],
                    "record_path": str(path),
                    "record_sha256": digest,
                    "candidate_count": len(record["candidate_executions"]),
                    "requested_simulations": requested,
                    "contract_sha256": contract_sha256,
                }
            )
        except Exception as exc:
            invalid_records.append(
                {"path": str(path), "sha256": digest, "error": f"{type(exc).__name__}: {exc}"}
            )

    duplicates = {uid: paths for uid, paths in records_by_uid.items() if len(paths) > 1}
    completed = set(records_by_uid)
    expected = set(source_by_uid)
    checks = {
        "manifest_dataset_population": manifest_population_ok,
        "record_file_count": len(record_paths) == sum(expected_dataset_counts.values()),
        "no_zero_byte_records": not zero_byte_paths,
        "no_temporary_records": not temporary_paths,
        "no_error_records": not error_paths,
        "all_records_contract_valid": not invalid_records,
        "no_duplicate_terminal_records": not duplicates,
        "exact_uid_coverage": completed == expected,
        "cache_dataset_population": dict(cache_dataset_counts) == expected_dataset_counts,
    }
    report = {
        "schema_version": "label_regeneration_p4_cache_audit_v1",
        "scope": {
            "included_datasets": sorted(expected_dataset_counts),
            "excluded_datasets": ["wemath2pro"],
            "output_root": str(output_root),
        },
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "contract_sha256": contract_sha256,
        "expected_dataset_counts": expected_dataset_counts,
        "manifest_dataset_counts": dict(sorted(manifest_dataset_counts.items())),
        "cache_dataset_counts": dict(sorted(cache_dataset_counts.items())),
        "expected_records": len(expected),
        "record_files": len(record_paths),
        "valid_terminal_records": len(completed),
        "missing_uids": sorted(expected - completed),
        "unexpected_uids": sorted(seen_uids - expected),
        "duplicate_records": duplicates,
        "invalid_records": invalid_records,
        "zero_byte_record_paths": [str(path) for path in zero_byte_paths],
        "temporary_record_paths": [str(path) for path in temporary_paths],
        "error_record_paths": [str(path) for path in error_paths],
        "search_budget_counts": dict(sorted(search_budget_counts.items())),
        "runtime_job_counts": dict(sorted(runtime_job_counts.items())),
        "record_index_rows": len(index_rows),
    }
    index_rows.sort(key=lambda row: (row["extraction_index"], row["uid"]))
    return report, index_rows
