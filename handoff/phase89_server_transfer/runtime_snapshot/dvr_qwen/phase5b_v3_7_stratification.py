"""Train-internal split stratification helpers for Phase 5B v3.7."""

from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Any


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)
    if not grouped:
        raise ValueError("rows must not be empty")
    return dict(grouped)


def _group_info(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    for group_id, group in _group_rows(rows).items():
        benchmarks = {str(row.get("benchmark", "unknown")) for row in group}
        if len(benchmarks) != 1:
            raise ValueError(f"group {group_id} has inconsistent benchmark values: {sorted(benchmarks)}")
        safe_rows = sum(1 for row in group if bool(row.get("safe_switch", False)))
        info[group_id] = {
            "id": group_id,
            "benchmark": next(iter(benchmarks)),
            "num_rows": len(group),
            "safe_rows": safe_rows,
            "has_safe": safe_rows > 0,
        }
    return info


def _bounded_calibration_count(total: int, fraction: float) -> int:
    if total <= 1:
        return 0
    count = int(round(float(total) * float(fraction)))
    return max(1, min(total - 1, count))


def stratified_safe_switch_split(
    rows: list[dict[str, Any]],
    *,
    calibration_fraction: float = 0.25,
    seed: int = 0,
    min_safe_calibration_groups: int = 1,
) -> dict[str, Any]:
    """Split groups while reserving safe-switch calibration groups per benchmark."""

    fraction = float(calibration_fraction)
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    min_safe = int(min_safe_calibration_groups)
    if min_safe < 0:
        raise ValueError("min_safe_calibration_groups must be non-negative")

    info = _group_info(rows)
    ids_by_benchmark: dict[str, list[str]] = defaultdict(list)
    for group_id, item in info.items():
        ids_by_benchmark[str(item["benchmark"])].append(group_id)

    rng = random.Random(int(seed))
    calibration_ids: set[str] = set()
    by_benchmark: dict[str, dict[str, int | bool]] = {}
    for benchmark in sorted(ids_by_benchmark):
        group_ids = sorted(ids_by_benchmark[benchmark])
        safe_ids = [group_id for group_id in group_ids if bool(info[group_id]["has_safe"])]
        non_safe_ids = [group_id for group_id in group_ids if not bool(info[group_id]["has_safe"])]
        rng.shuffle(safe_ids)
        rng.shuffle(non_safe_ids)
        target_count = _bounded_calibration_count(len(group_ids), fraction)
        safe_target = 0
        if safe_ids and target_count > 0:
            if len(safe_ids) == 1:
                safe_target = 1
            else:
                safe_target = max(min_safe, int(round(len(safe_ids) * fraction)))
                safe_target = max(1, safe_target)
                safe_target = min(safe_target, len(safe_ids) - 1, target_count)

        benchmark_calibration = list(safe_ids[:safe_target])
        remaining_slots = max(0, target_count - len(benchmark_calibration))
        fill_pool = [*non_safe_ids, *safe_ids[safe_target:]]
        benchmark_calibration.extend(fill_pool[:remaining_slots])
        calibration_ids.update(benchmark_calibration)

        benchmark_calibration_set = set(benchmark_calibration)
        fit_ids = [group_id for group_id in group_ids if group_id not in benchmark_calibration_set]
        calibration_safe_groups = sum(1 for group_id in benchmark_calibration if bool(info[group_id]["has_safe"]))
        fit_safe_groups = sum(1 for group_id in fit_ids if bool(info[group_id]["has_safe"]))
        by_benchmark[benchmark] = {
            "fit": len(fit_ids),
            "calibration": len(benchmark_calibration),
            "total": len(group_ids),
            "total_safe_groups": len(safe_ids),
            "fit_safe_groups": fit_safe_groups,
            "calibration_safe_groups": calibration_safe_groups,
            "fit_and_calibration_have_safe_groups": bool(fit_safe_groups > 0 and calibration_safe_groups > 0),
        }

    all_ids = set(info)
    fit_ids = all_ids - calibration_ids
    if not fit_ids or not calibration_ids:
        raise ValueError(
            f"split produced fit={len(fit_ids)} calibration={len(calibration_ids)} groups; "
            "adjust calibration_fraction or provide more groups"
        )
    return {
        "strategy": "benchmark_safe_switch_stratified",
        "seed": int(seed),
        "calibration_fraction": fraction,
        "min_safe_calibration_groups": min_safe,
        "fit_group_ids": sorted(fit_ids),
        "calibration_group_ids": sorted(calibration_ids),
        "fit_num_groups": len(fit_ids),
        "calibration_num_groups": len(calibration_ids),
        "total_num_groups": len(all_ids),
        "by_benchmark": by_benchmark,
    }


def safe_switch_coverage_by_benchmark(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
) -> dict[str, dict[str, int | bool]]:
    info = _group_info(rows)
    fit_ids = {str(group_id) for group_id in split.get("fit_group_ids", [])}
    calibration_ids = {str(group_id) for group_id in split.get("calibration_group_ids", [])}
    if fit_ids & calibration_ids:
        raise ValueError("fit and calibration group ids overlap")
    missing = set(info) - fit_ids - calibration_ids
    if missing:
        raise ValueError(f"split is missing {len(missing)} groups; first missing id: {sorted(missing)[0]}")

    out: dict[str, dict[str, int | bool]] = {}
    for benchmark in sorted({str(item["benchmark"]) for item in info.values()}):
        benchmark_ids = [group_id for group_id, item in info.items() if str(item["benchmark"]) == benchmark]
        fit_benchmark_ids = [group_id for group_id in benchmark_ids if group_id in fit_ids]
        calibration_benchmark_ids = [group_id for group_id in benchmark_ids if group_id in calibration_ids]
        fit_safe_groups = sum(1 for group_id in fit_benchmark_ids if bool(info[group_id]["has_safe"]))
        calibration_safe_groups = sum(1 for group_id in calibration_benchmark_ids if bool(info[group_id]["has_safe"]))
        out[benchmark] = {
            "total_groups": len(benchmark_ids),
            "total_safe_groups": sum(1 for group_id in benchmark_ids if bool(info[group_id]["has_safe"])),
            "total_safe_rows": sum(int(info[group_id]["safe_rows"]) for group_id in benchmark_ids),
            "fit_groups": len(fit_benchmark_ids),
            "fit_safe_groups": fit_safe_groups,
            "fit_safe_rows": sum(int(info[group_id]["safe_rows"]) for group_id in fit_benchmark_ids),
            "calibration_groups": len(calibration_benchmark_ids),
            "calibration_safe_groups": calibration_safe_groups,
            "calibration_safe_rows": sum(int(info[group_id]["safe_rows"]) for group_id in calibration_benchmark_ids),
            "fit_and_calibration_have_safe_groups": bool(fit_safe_groups > 0 and calibration_safe_groups > 0),
        }
    return out


def compact_coverage_table(coverage: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"benchmark": benchmark, **dict(values)}
        for benchmark, values in sorted(coverage.items())
    ]


def coverage_failure_reasons(coverage: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for benchmark, values in sorted(coverage.items()):
        if int(values.get("total_safe_groups", 0)) <= 0:
            continue
        if int(values.get("calibration_safe_groups", 0)) <= 0:
            reasons.append(f"{benchmark}: calibration has no safe-switch groups")
        if int(values.get("fit_safe_groups", 0)) <= 0:
            reasons.append(f"{benchmark}: fit has no safe-switch groups")
    return reasons
