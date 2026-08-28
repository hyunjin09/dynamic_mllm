from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


ROUTE_CONDITIONED_TAXONOMY = (
    "redundant",
    "read_mediated",
    "write_mediated",
    "either_removal_sufficient",
    "both_required",
    "inconsistent_anchor",
)

ROUTE_ACTION_NAMES = {
    "IGNORE": ("BOTH_OFF", "M00"),
    "READ_ONLY": ("WRITE_OFF", "M10"),
    "WRITE_ONLY": ("READ_OFF", "M01"),
    "FULL": ("FULL_RESTORE", "M11"),
}


def build_anchor_candidate_rows(
    cohort_rows: Iterable[Mapping[str, Any]],
    eligibility_rows: Iterable[Mapping[str, Any]],
    *,
    layer_count: int = 28,
) -> list[dict[str, Any]]:
    """Join the authoritative cohort to its current unified-FULL freeze."""
    eligibility = {}
    for row in eligibility_rows:
        uid = str(row["uid"])
        if uid in eligibility:
            raise ValueError(f"duplicate eligibility UID: {uid}")
        eligibility[uid] = bool(row["eligible"])
    output = []
    for source in cohort_rows:
        if source.get("cohort") != "primary_a_plus":
            continue
        uid = str(source["uid"])
        if uid not in eligibility:
            raise ValueError(f"missing eligibility row: {uid}")
        if not eligibility[uid]:
            continue
        candidates = ordered_anchor_candidates(
            source["binary_routes"]["correcting_routes"],
            layer_count=layer_count,
        )
        row = dict(source)
        row["schema_version"] = "route_conditioned_anchor_candidates_v1"
        row["anchor_candidates"] = candidates
        row["historical_minimum_off_count"] = candidates[0]["hamming_distance_to_full"]
        row["historical_minimum_distance_tie_count"] = sum(
            candidate["minimum_distance_tie"] for candidate in candidates
        )
        output.append(row)
    output.sort(key=lambda row: str(row["uid"]))
    return output


def finalize_anchor_rows(
    candidate_rows: Iterable[Mapping[str, Any]],
    validation_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join complete current-runtime validation into final anchor rows."""
    candidates = {}
    for row in candidate_rows:
        uid = str(row["uid"])
        if uid in candidates:
            raise ValueError(f"duplicate candidate UID: {uid}")
        candidates[uid] = row
    validations = {}
    for row in validation_rows:
        uid = str(row["uid"])
        if uid in validations:
            raise ValueError(f"duplicate validation UID: {uid}")
        validations[uid] = row
    missing = sorted(set(candidates) - set(validations))
    extra = sorted(set(validations) - set(candidates))
    if missing:
        raise ValueError(f"missing validation rows: {missing[:3]}")
    if extra:
        raise ValueError(f"validation rows not present in candidates: {extra[:3]}")
    anchors = []
    exclusions = []
    for uid in sorted(candidates):
        source = candidates[uid]
        validation = validations[uid]
        if not bool(validation.get("passed")):
            raise ValueError(f"validation gate failed for {uid}")
        if not bool(validation["analyzable"]):
            exclusions.append(
                {
                    "schema_version": "route_conditioned_anchor_exclusion_v1",
                    "uid": uid,
                    "dataset": source["dataset"],
                    "image_group_id": source.get("image_group_id"),
                    "exclusion_reason": validation["exclusion_reason"],
                    "candidate_evaluations": validation["candidate_evaluations"],
                    "cached_candidate_count": len(source["anchor_candidates"]),
                }
            )
            continue
        selected = validation["anchor"]
        selected_ids = {str(row["route_id"]) for row in source["anchor_candidates"]}
        if str(selected["route_id"]) not in selected_ids:
            raise ValueError(f"selected anchor is not a cached candidate for {uid}")
        mask = [int(value) for value in selected["mask"]]
        off_layers = [index for index, value in enumerate(mask) if value == 0]
        if off_layers != [int(value) for value in validation["anchor_off_layers"]]:
            raise ValueError(f"anchor OFF-layer mismatch for {uid}")
        row = dict(source)
        row.update(
            {
                "schema_version": "route_conditioned_anchor_manifest_v1",
                "anchor_route_id": str(selected["route_id"]),
                "anchor_route_mask": mask,
                "anchor_cached_score": float(selected["score"]),
                "anchor_candidate_rank": int(selected["candidate_rank"]),
                "anchor_fallback_count": int(selected["fallback_count"]),
                "anchor_hamming_distance_from_full": len(off_layers),
                "anchor_off_count": len(off_layers),
                "anchor_off_layers": off_layers,
                "anchor_current_state": dict(selected["current_state"]),
                "minimum_distance_tied_candidates": [
                    dict(candidate)
                    for candidate in source["anchor_candidates"]
                    if candidate["minimum_distance_tie"]
                ],
                "anchor_candidate_evaluations": validation["candidate_evaluations"],
            }
        )
        anchors.append(row)
    return anchors, exclusions


def ordered_anchor_candidates(
    routes: Iterable[Mapping[str, Any]],
    *,
    layer_count: int = 28,
) -> list[dict[str, Any]]:
    """Return cached correcting routes in the frozen anchor-selection order."""
    normalized = []
    for source in routes:
        mask = [int(value) for value in source["mask"]]
        if len(mask) != layer_count or any(value not in {0, 1} for value in mask):
            raise ValueError("anchor candidate must have one binary action per layer")
        distance = layer_count - sum(mask)
        if distance < 1 or distance >= layer_count:
            raise ValueError("anchor candidate must be neither FULL nor ALL-OFF")
        if int(source["hamming_distance_to_full"]) != distance:
            raise ValueError("anchor candidate Hamming distance does not match its mask")
        row = dict(source)
        row["mask"] = mask
        row["hamming_distance_to_full"] = distance
        row["score"] = float(source["score"])
        normalized.append(row)
    if not normalized:
        raise ValueError("at least one correcting anchor candidate is required")
    normalized.sort(
        key=lambda row: (
            row["hamming_distance_to_full"],
            -row["score"],
            str(row["route_id"]),
            "".join(map(str, row["mask"])),
        )
    )
    minimum = normalized[0]["hamming_distance_to_full"]
    for rank, row in enumerate(normalized):
        row["candidate_rank"] = rank
        row["minimum_distance_tie"] = row["hamming_distance_to_full"] == minimum
    return normalized


def select_current_anchor(
    candidates: Sequence[Mapping[str, Any]],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Select the first current-correct candidate, or return no anchor."""
    for candidate in candidates:
        route_id = str(candidate["route_id"])
        if route_id not in evaluations:
            raise ValueError(f"missing current-runtime evaluation for {route_id}")
        state = dict(evaluations[route_id])
        if bool(state.get("correct")):
            output = dict(candidate)
            output["fallback_count"] = int(candidate["candidate_rank"])
            output["current_state"] = state
            return output
    return None


def evaluate_until_current_correct(
    candidates: Sequence[Mapping[str, Any]],
    evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate ordered candidates lazily until one is current-correct."""
    evaluations = []
    selected = None
    for candidate in candidates:
        state = dict(evaluate(candidate))
        evaluations.append(
            {
                "route_id": str(candidate["route_id"]),
                "candidate_rank": int(candidate["candidate_rank"]),
                **state,
            }
        )
        if bool(state.get("correct")):
            selected = dict(candidate)
            selected["fallback_count"] = int(candidate["candidate_rank"])
            selected["current_state"] = state
            break
    return {"selected": selected, "evaluations": evaluations}


def _evenly_spaced(rows: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    if count < 0 or count > len(rows):
        raise ValueError("stratum is too small for the requested pilot count")
    if count == 0:
        return []
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("pilot selection produced duplicate positions")
    return [rows[index] for index in indices]


def select_stratified_pilot(
    rows: Iterable[Mapping[str, Any]],
    *,
    total: int = 56,
    datasets: Sequence[str] = ("gqa", "textvqa"),
) -> list[dict[str, Any]]:
    """Select a deterministic dataset × anchor-size stratified pilot."""
    if total < len(datasets) * 3 or total % len(datasets):
        raise ValueError("pilot total must split evenly across datasets and cover three strata")
    population = list(rows)
    per_dataset = total // len(datasets)
    allocation = [per_dataset // 3 + (index < per_dataset % 3) for index in range(3)]
    output = []
    for dataset in datasets:
        current = sorted(
            (row for row in population if row["dataset"] == dataset),
            key=lambda row: (int(row["anchor_off_count"]), str(row["uid"])),
        )
        if len(current) < per_dataset:
            raise ValueError(f"not enough {dataset} rows for the pilot")
        boundaries = [0, len(current) // 3, 2 * len(current) // 3, len(current)]
        for index, name in enumerate(("small", "medium", "large")):
            selected = _evenly_spaced(current[boundaries[index] : boundaries[index + 1]], allocation[index])
            for source in selected:
                row = dict(source)
                row["pilot_off_count_stratum"] = name
                output.append(row)
    return output


def balance_work_units(
    rows: Iterable[Mapping[str, Any]],
    *,
    work_unit_count: int,
) -> list[dict[str, Any]]:
    """Greedily balance deterministic work units by expected 3K cell cost."""
    if work_unit_count < 1:
        raise ValueError("work_unit_count must be positive")
    units = [
        {"work_unit_id": f"work_unit_{index:03d}", "expected_new_cells": 0, "samples": []}
        for index in range(work_unit_count)
    ]
    ordered = sorted(
        rows,
        key=lambda row: (-int(row["anchor_off_count"]), str(row["uid"])),
    )
    seen = set()
    for source in ordered:
        uid = str(source["uid"])
        if uid in seen:
            raise ValueError(f"duplicate sample in work-unit input: {uid}")
        seen.add(uid)
        target = min(units, key=lambda unit: (unit["expected_new_cells"], unit["work_unit_id"]))
        cost = 3 * int(source["anchor_off_count"])
        row = dict(source)
        row["work_unit_id"] = target["work_unit_id"]
        row["expected_new_cells"] = cost
        target["samples"].append(row)
        target["expected_new_cells"] += cost
    return units


def select_execution_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    gpu_index: int,
    replica_index: int,
    replicas_per_gpu: int,
    gpu_count: int = 8,
) -> list[dict[str, Any]]:
    """Map pilot workers or full work units onto one GPU-local replica."""
    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be pilot or full")
    if not 0 <= gpu_index < gpu_count:
        raise ValueError("gpu_index is out of range")
    if replicas_per_gpu < 1 or not 0 <= replica_index < replicas_per_gpu:
        raise ValueError("invalid replica layout")
    selected = []
    for source in rows:
        if mode == "pilot":
            assigned_gpu = int(source["pilot_worker_index"])
        else:
            unit_id = str(source["work_unit_id"])
            try:
                assigned_gpu = int(unit_id.rsplit("_", 1)[1]) % gpu_count
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid work_unit_id: {unit_id}") from exc
        if assigned_gpu == gpu_index:
            selected.append(dict(source))
    return selected[replica_index::replicas_per_gpu]


def classify_route_conditioned_cell(correctness: Mapping[str, bool]) -> str:
    """Classify one anchor-OFF position under the plan's discrete taxonomy."""
    if not bool(correctness["IGNORE"]):
        return "inconsistent_anchor"
    if bool(correctness["FULL"]):
        return "redundant"
    read_off_correct = bool(correctness["WRITE_ONLY"])
    write_off_correct = bool(correctness["READ_ONLY"])
    if read_off_correct and write_off_correct:
        return "either_removal_sufficient"
    if read_off_correct:
        return "read_mediated"
    if write_off_correct:
        return "write_mediated"
    return "both_required"


def flatten_route_conditioned_samples(
    samples: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten nested sample cells into one row per layer and action."""
    output = []
    for sample in samples:
        for cell in sample["cells"]:
            for action, (route_name, factorial_state) in ROUTE_ACTION_NAMES.items():
                state = cell["states"][action]
                output.append(
                    {
                        "schema_version": "route_conditioned_action_cell_v1",
                        "uid": sample["uid"],
                        "dataset": sample["dataset"],
                        "image_id": sample["image_id"],
                        "image_group_id": sample["image_group_id"],
                        "work_unit_id": sample["work_unit_id"],
                        "anchor_route_id": sample["anchor_route_id"],
                        "anchor_route_mask": sample["anchor_route_mask"],
                        "anchor_off_count": sample["anchor_off_count"],
                        "anchor_hamming_distance_from_full": sample[
                            "anchor_hamming_distance_from_full"
                        ],
                        "target_layer": cell["target_layer"],
                        "action": action,
                        "route_action_name": route_name,
                        "factorial_state": factorial_state,
                        "read_on": action in {"READ_ONLY", "FULL"},
                        "write_on": action in {"WRITE_ONLY", "FULL"},
                        "new_evaluation": action != "IGNORE",
                        "fixed_correct_target_text": sample["fixed_correct_target_text"],
                        "fixed_wrong_target_text": sample["fixed_wrong_target_text"],
                        "generated_answer": state["generated_answer"],
                        "generated_ids": state["generated_ids"],
                        "correctness_score": state["correctness_score"],
                        "correct": state["correct"],
                        "S_correct": state["S_correct"],
                        "S_original_full_wrong": state["S_full_wrong"],
                        "margin": state["margin"],
                        "taxonomy": cell["taxonomy"],
                        **cell["effects"],
                        "cell_elapsed_seconds": cell["elapsed_seconds"],
                        "worker_rank": sample["worker"]["rank"],
                        "gpu_index": sample["worker"]["gpu_index"],
                        "replica_index": sample["worker"]["replica_index"],
                    }
                )
    return output


def choose_pilot_configuration(
    configurations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose the fastest gate-passing pilot by useful intervention throughput."""
    valid = [
        dict(row)
        for row in configurations
        if bool(row.get("passed"))
        and int(row.get("disqualifying_failure_count", 0)) == 0
        and row.get("useful_new_cells_per_second") is not None
    ]
    if not valid:
        raise ValueError("no passing pilot configuration has measured throughput")
    valid.sort(
        key=lambda row: (
            -float(row["useful_new_cells_per_second"]),
            int(row["replicas_per_gpu"]),
            str(row.get("name", "")),
        )
    )
    return valid[0]


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_gpu_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize bounded nvidia-smi samples without treating utilization as the objective."""
    parsed = []
    for row in rows:
        parsed.append(
            (
                str(row["gpu_index"]),
                float(row["memory_used_mib"]),
                float(row["utilization_gpu_percent"]),
            )
        )
    if not parsed:
        raise ValueError("GPU metric log is empty")
    utilization = [row[2] for row in parsed]
    per_gpu = {}
    for gpu, memory, _ in parsed:
        per_gpu[gpu] = max(memory, per_gpu.get(gpu, 0.0))
    return {
        "sample_count": len(parsed),
        "peak_memory_used_mib": max(row[1] for row in parsed),
        "per_gpu_peak_memory_used_mib": dict(sorted(per_gpu.items())),
        "mean_gpu_utilization_percent": sum(utilization) / len(utilization),
        "median_gpu_utilization_percent": _quantile(utilization, 0.5),
        "p90_gpu_utilization_percent": _quantile(utilization, 0.9),
    }
