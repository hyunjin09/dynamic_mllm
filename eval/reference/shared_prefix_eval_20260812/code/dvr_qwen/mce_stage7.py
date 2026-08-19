"""MCE-7 random best-of-N and search-bias control helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import random
from typing import Any, Iterable

from dvr_qwen.mce_stage0 import source_row_for_manifest
from dvr_qwen.mce_stage3 import mean, route_key


DEFAULT_BUDGETS: tuple[int | str, ...] = (1, 5, 10, 25, "matched")
CONTROL_METHODS = (
    "uniform_random",
    "oracle_count_random",
    "single_suppression_random",
    "contiguous_text_only",
    "fixed_global",
    "fixed_dataset",
)
SCORE_ATOL = 1e-9


def _binary(route: Iterable[int | bool]) -> list[int]:
    return [1 if value else 0 for value in route]


def _route_key(route: Iterable[int | bool]) -> str:
    return route_key([bool(value) for value in route])


def _num_layers(source_row: dict[str, Any]) -> int:
    for record in (
        source_row.get("binary_all_visual_on"),
        (source_row.get("binary_oracle") or {}).get("best"),
    ):
        if isinstance(record, dict) and isinstance(record.get("visual_on_mask"), list):
            return len(record["visual_on_mask"])
    return 28


def _stable_order_key(random_seed: int, dataset: str, sample_id: str) -> str:
    return hashlib.sha256(f"{random_seed}:{dataset}:{sample_id}".encode("utf-8")).hexdigest()


def build_calibration_plan(
    train_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    *,
    per_dataset: int = 25,
    num_layers: int = 28,
    on_count_step: int = 4,
    random_seed: int = 20260619,
) -> dict[str, Any]:
    """Build a train-only fixed-route calibration split and candidate bank."""

    if on_count_step <= 0:
        raise ValueError("on_count_step must be positive")

    diagnostic_ids = {str(row["sample_id"]) for row in diagnostic_rows}
    diagnostic_images = {
        str(row.get("image_path"))
        for row in diagnostic_rows
        if row.get("image_path") is not None
    }
    eligible = [
        row for row in train_rows
        if str(row["sample_id"]) not in diagnostic_ids
        and str(row.get("image")) not in diagnostic_images
    ]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_dataset[str(row["benchmark"])].append(row)

    calibration_source_rows: list[dict[str, Any]] = []
    for dataset in sorted(by_dataset):
        ordered = sorted(
            by_dataset[dataset],
            key=lambda row: _stable_order_key(random_seed, dataset, str(row["sample_id"])),
        )
        chosen: list[dict[str, Any]] = []
        used_images: set[str] = set()
        for row in ordered:
            image = str(row.get("image"))
            if image in used_images:
                continue
            chosen.append(row)
            used_images.add(image)
            if len(chosen) == per_dataset:
                break
        if len(chosen) != per_dataset:
            raise ValueError(f"calibration shortfall for {dataset}: {len(chosen)} < {per_dataset}")
        calibration_source_rows.extend(chosen)

    calibration_ids = {str(row["sample_id"]) for row in calibration_source_rows}
    builder_rows = [row for row in eligible if str(row["sample_id"]) not in calibration_ids]
    calibration_manifest = [
        {
            "sample_id": row["sample_id"],
            "dataset": row["benchmark"],
            "cohort": "calibration",
            "source_pool": "primary_5k",
            "source_asset_id": str(row.get("image")),
            "image_path": row.get("image"),
            "oracle_gain_used_for_sampling": False,
            "sampling_seed": random_seed,
        }
        for row in calibration_source_rows
    ]

    scopes: list[tuple[str, list[dict[str, Any]]]] = [("global", builder_rows)]
    scopes.extend(
        (dataset, [row for row in builder_rows if str(row["benchmark"]) == dataset])
        for dataset in sorted(by_dataset)
    )
    candidates_by_key: dict[str, dict[str, Any]] = {}
    for scope, rows in scopes:
        if not rows:
            continue
        frequencies = [0] * num_layers
        for row in rows:
            route = _binary(row["visual_on_mask"])
            if len(route) != num_layers:
                raise ValueError(f"route length {len(route)} != num_layers {num_layers}")
            for layer_idx, value in enumerate(route):
                frequencies[layer_idx] += value
        layer_order = sorted(range(num_layers), key=lambda idx: (-frequencies[idx], idx))
        on_counts = list(range(0, num_layers + 1, on_count_step))
        if on_counts[-1] != num_layers:
            on_counts.append(num_layers)
        for on_count in on_counts:
            selected = set(layer_order[:on_count])
            route = [1 if idx in selected else 0 for idx in range(num_layers)]
            key = _route_key(route)
            origin = {
                "scope": scope,
                "on_count": on_count,
                "builder_row_count": len(rows),
                "layer_order_one_based": [idx + 1 for idx in layer_order],
            }
            if key not in candidates_by_key:
                candidates_by_key[key] = {
                    "route_id": f"fixed_{key}",
                    "route": route,
                    "route_key": key,
                    "num_visual_on_layers": on_count,
                    "origins": [origin],
                }
            else:
                candidates_by_key[key]["origins"].append(origin)

    return {
        "calibration_manifest": calibration_manifest,
        "route_builder_sample_ids": sorted(str(row["sample_id"]) for row in builder_rows),
        "candidate_routes": list(candidates_by_key.values()),
        "diagnostic_sample_ids": sorted(diagnostic_ids),
        "random_seed": random_seed,
        "per_dataset": per_dataset,
        "num_layers": num_layers,
        "on_count_step": on_count_step,
    }


def build_calibration_specs(
    calibration_manifest: list[dict[str, Any]],
    candidate_routes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    specs = []
    for sample_index, row in enumerate(calibration_manifest):
        for candidate in candidate_routes:
            origin_scopes = {str(origin["scope"]) for origin in candidate.get("origins") or []}
            eligible_global = not origin_scopes or "global" in origin_scopes
            eligible_dataset = not origin_scopes or eligible_global or str(row["dataset"]) in origin_scopes
            if not eligible_global and not eligible_dataset:
                continue
            specs.append(
                {
                    "sample_index": sample_index,
                    "sample_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "cohort": "calibration",
                    "source_pool": "primary_5k",
                    "source_asset_id": row.get("source_asset_id"),
                    "route_id": candidate["route_id"],
                    "route": _binary(candidate["route"]),
                    "route_key": candidate["route_key"],
                    "eligible_global_selection": eligible_global,
                    "eligible_dataset_selection": eligible_dataset,
                    "intervention_id": f"{row['sample_id']}:calibration:{candidate['route_id']}",
                }
            )
    return specs


def _route_from_seed(source_row: dict[str, Any], seed: str, num_layers: int) -> list[int]:
    if seed == "all_visual_on":
        return [1] * num_layers
    if seed == "all_text_only":
        return [0] * num_layers
    if seed == "old_gt":
        selected = {int(layer) for layer in source_row.get("old_gt_mask_one_based") or []}
        return [1 if idx + 1 in selected else 0 for idx in range(num_layers)]
    raise ValueError(f"unknown oracle-search seed: {seed}")


def _seed_result(source_row: dict[str, Any], seed: str) -> dict[str, Any] | None:
    field = {
        "all_visual_on": "binary_all_visual_on",
        "all_text_only": "binary_all_text_only",
        "old_gt": "binary_old_gt",
    }[seed]
    value = source_row.get(field)
    return value if isinstance(value, dict) else None


def reconstruct_structured_candidates(source_row: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct the unique route sequence actually explored by greedy search.

    All-ON is the common first route. Later duplicate routes from independent
    search orders are not counted again because best-of-N measures unique output
    opportunities, not repeated model calls on the same mask.
    """

    num_layers = _num_layers(source_row)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append(route: list[int], result: dict[str, Any] | None, source: str) -> None:
        key = _route_key(route)
        if key in seen or result is None:
            return
        seen.add(key)
        candidates.append(
            {
                "rank": len(candidates) + 1,
                "route": list(route),
                "route_key": key,
                "score": float(result.get("score", 0.0)),
                "prediction": str(result.get("prediction", "")),
                "num_visual_on_layers": int(sum(route)),
                "source": source,
            }
        )

    all_on = [1] * num_layers
    append(all_on, source_row.get("binary_all_visual_on"), "common_all_visual_on")

    for search_index, search in enumerate(source_row.get("all_searches") or []):
        seed = str(search.get("seed", "all_visual_on"))
        current = _route_from_seed(source_row, seed, num_layers)
        append(current, _seed_result(source_row, seed), f"search_{search_index}_seed")
        for trial_index, trial in enumerate(search.get("trials") or []):
            if "layer_zero_based" in trial:
                layer_idx = int(trial["layer_zero_based"])
            else:
                layer_idx = int(trial["layer_one_based"]) - 1
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(f"trial layer outside 0..{num_layers - 1}: {layer_idx}")
            trial_route = list(current)
            trial_route[layer_idx] = 0
            append(trial_route, trial, f"search_{search_index}_trial_{trial_index}")
            if bool(trial.get("accepted", False)):
                current = trial_route

    best = (source_row.get("binary_oracle") or {}).get("best")
    if isinstance(best, dict) and isinstance(best.get("visual_on_mask"), list):
        append(_binary(best["visual_on_mask"]), best, "binary_oracle_best")
    return candidates


def _rng(sample_id: str, method: str, random_seed: int) -> random.Random:
    payload = f"{random_seed}:{sample_id}:{method}".encode("utf-8")
    return random.Random(int(hashlib.sha256(payload).hexdigest()[:16], 16))


def _random_unique_routes(
    sample_id: str,
    method: str,
    *,
    num_layers: int,
    oracle_on_count: int,
    count: int,
    random_seed: int,
) -> list[list[int]]:
    rng = _rng(sample_id, method, random_seed)
    all_on_key = "1" * num_layers

    if method == "single_suppression_random":
        routes = [[0 if idx == layer else 1 for idx in range(num_layers)] for layer in range(num_layers)]
        rng.shuffle(routes)
        return routes[:count]

    if method == "contiguous_text_only":
        routes = []
        for start in range(num_layers):
            for end in range(start, num_layers):
                routes.append([0 if start <= idx <= end else 1 for idx in range(num_layers)])
        rng.shuffle(routes)
        return routes[:count]

    if method == "oracle_count_random" and oracle_on_count in {0, num_layers}:
        route = [1] * num_layers if oracle_on_count == num_layers else [0] * num_layers
        return [] if _route_key(route) == all_on_key else [route]

    routes: list[list[int]] = []
    seen = {all_on_key}
    max_unique = (1 << num_layers) - 1
    if method == "oracle_count_random":
        # The loop is bounded naturally for Qwen's 28 layers; low-cardinality
        # edge cases are handled above or exhausted by repeated draws.
        max_attempts = max(100, count * 100)
    elif method == "uniform_random":
        max_attempts = max(100, count * 20)
    else:
        raise ValueError(f"unknown random control method: {method}")

    attempts = 0
    while len(routes) < count and len(seen) < max_unique and attempts < max_attempts:
        attempts += 1
        if method == "uniform_random":
            route = [rng.randrange(2) for _ in range(num_layers)]
        else:
            selected = set(rng.sample(range(num_layers), oracle_on_count))
            route = [1 if idx in selected else 0 for idx in range(num_layers)]
        key = _route_key(route)
        if key in seen:
            continue
        seen.add(key)
        routes.append(route)
    return routes


def build_control_specs(
    manifest_rows: list[dict[str, Any]],
    source_indexes: dict[str, dict[str, dict[str, Any]]],
    fixed_routes: dict[str, Any],
    *,
    random_seed: int = 20260619,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for sample_index, manifest_row in enumerate(manifest_rows):
        source_row = source_row_for_manifest(manifest_row, source_indexes)
        structured = reconstruct_structured_candidates(source_row)
        matched_budget = len(structured)
        num_layers = _num_layers(source_row)
        oracle_mask = _binary(source_row["binary_oracle"]["best"]["visual_on_mask"])
        random_count = max(0, matched_budget - 1)
        base = {
            "sample_index": sample_index,
            "sample_id": manifest_row["sample_id"],
            "dataset": manifest_row["dataset"],
            "cohort": manifest_row["cohort"],
            "source_pool": manifest_row["source_pool"],
            "source_asset_id": manifest_row.get("source_asset_id"),
            "matched_structured_budget": matched_budget,
        }
        for method in CONTROL_METHODS[:4]:
            routes = _random_unique_routes(
                str(manifest_row["sample_id"]),
                method,
                num_layers=num_layers,
                oracle_on_count=sum(oracle_mask),
                count=random_count,
                random_seed=random_seed,
            )
            for offset, route in enumerate(routes, start=2):
                specs.append(
                    {
                        **base,
                        "intervention_id": f"{manifest_row['sample_id']}:{method}:R{offset}",
                        "method": method,
                        "rank": offset,
                        "route": route,
                        "route_key": _route_key(route),
                        "requires_evaluation": method != "single_suppression_random",
                    }
                )

        dataset = str(manifest_row["dataset"])
        fixed = (
            ("fixed_global", fixed_routes["global"]),
            ("fixed_dataset", fixed_routes["by_dataset"][dataset]),
        )
        for method, route_value in fixed:
            route = _binary(route_value.get("route", route_value) if isinstance(route_value, dict) else route_value)
            if _route_key(route) == "1" * num_layers:
                continue
            specs.append(
                {
                    **base,
                    "intervention_id": f"{manifest_row['sample_id']}:{method}:R2",
                    "method": method,
                    "rank": 2,
                    "route": route,
                    "route_key": _route_key(route),
                    "requires_evaluation": True,
                }
            )
    return specs


def select_best_fixed_routes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("fixed-route calibration rows are empty")
    if any(
        "eligible_global_selection" not in row or "eligible_dataset_selection" not in row
        for row in rows
    ):
        raise ValueError("fixed-route calibration rows are missing eligibility metadata")

    def select(current_rows: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in current_rows:
            grouped[str(row["route_id"])].append(row)
        candidates = []
        for route_id, route_rows in grouped.items():
            route = _binary(route_rows[0]["route"])
            if any(_binary(row["route"]) != route for row in route_rows):
                raise ValueError(f"route_id maps to multiple masks: {route_id}")
            candidates.append(
                {
                    "route_id": route_id,
                    "route": route,
                    "mean_score": mean([float(row["score"]) for row in route_rows]),
                    "num_visual_on_layers": sum(route),
                    "sample_count": len(route_rows),
                }
            )
        return sorted(
            candidates,
            key=lambda row: (-row["mean_score"], row["num_visual_on_layers"], row["route_id"]),
        )[0]

    datasets = sorted({str(row["dataset"]) for row in rows})
    global_rows = [row for row in rows if bool(row["eligible_global_selection"])]
    return {
        "global": select(global_rows),
        "by_dataset": {
            dataset: select(
                [
                    row for row in rows
                    if str(row["dataset"]) == dataset
                    and bool(row["eligible_dataset_selection"])
                ]
            )
            for dataset in datasets
        },
    }


def _best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["score"]),
            int(row["num_visual_on_layers"]),
            int(row["rank"]),
            str(row.get("route_key") or _route_key(row["route"])),
        ),
    )[0]


def select_curve_winners(
    manifest_rows: list[dict[str, Any]],
    source_indexes: dict[str, dict[str, dict[str, Any]]],
    control_rows: list[dict[str, Any]],
    *,
    budgets: tuple[int | str, ...] = DEFAULT_BUDGETS,
) -> list[dict[str, Any]]:
    controls: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in control_rows:
        controls[(str(row["sample_id"]), str(row["method"]))].append(row)

    winners: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        sample_id = str(manifest_row["sample_id"])
        source_row = source_row_for_manifest(manifest_row, source_indexes)
        structured = reconstruct_structured_candidates(source_row)
        baseline = structured[0]
        for method in ("structured_oracle_search",) + CONTROL_METHODS:
            method_rows = structured if method == "structured_oracle_search" else [
                baseline,
                *sorted(controls.get((sample_id, method), []), key=lambda row: int(row["rank"])),
            ]
            for budget in budgets:
                limit = len(structured) if budget == "matched" else int(budget)
                available = [row for row in method_rows if int(row["rank"]) <= limit]
                winner = _best_candidate(available)
                winners.append(
                    {
                        "sample_id": sample_id,
                        "dataset": manifest_row["dataset"],
                        "cohort": manifest_row["cohort"],
                        "source_pool": manifest_row["source_pool"],
                        "source_asset_id": manifest_row.get("source_asset_id"),
                        "method": method,
                        "budget": budget,
                        "budget_key": str(budget),
                        "requested_budget": limit,
                        "effective_budget": len(available),
                        "matched_structured_budget": len(structured),
                        "baseline_score": float(baseline["score"]),
                        "winner_rank": int(winner["rank"]),
                        "winner_route": _binary(winner["route"]),
                        "winner_route_key": str(winner.get("route_key") or _route_key(winner["route"])),
                        "winner_score": float(winner["score"]),
                        "winner_prediction": str(winner["prediction"]),
                        "winner_num_visual_on_layers": int(winner["num_visual_on_layers"]),
                    }
                )
    return winners


def summarize_best_of_n(
    manifest_rows: list[dict[str, Any]],
    source_indexes: dict[str, dict[str, dict[str, Any]]],
    control_rows: list[dict[str, Any]],
    *,
    nll_by_key: dict[tuple[str, str], dict[str, Any]] | None = None,
    budgets: tuple[int | str, ...] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    winners = select_curve_winners(
        manifest_rows,
        source_indexes,
        control_rows,
        budgets=budgets,
    )
    nll_by_key = nll_by_key or {}
    curves: dict[str, dict[str, Any]] = {}
    for method in ("structured_oracle_search",) + CONTROL_METHODS:
        curves[method] = {}
        for budget in budgets:
            budget_key = str(budget)
            rows = [
                row for row in winners
                if row["method"] == method and row["budget_key"] == budget_key
            ]
            wrong = [row for row in rows if row["cohort"] == "wrong"]
            correct = [row for row in rows if row["cohort"] == "correct"]
            logprobs = []
            for row in rows:
                nll = nll_by_key.get((str(row["sample_id"]), str(row["winner_route_key"])))
                if nll is not None and nll.get("mean_logprob") is not None:
                    logprobs.append(float(nll["mean_logprob"]))
            curves[method][budget_key] = {
                "samples": len(rows),
                "mean_best_score": mean([float(row["winner_score"]) for row in rows]),
                "wrong_samples": len(wrong),
                "wrong_full_fixes": sum(float(row["winner_score"]) >= 1.0 - SCORE_ATOL for row in wrong),
                "wrong_full_fix_rate": (
                    sum(float(row["winner_score"]) >= 1.0 - SCORE_ATOL for row in wrong) / len(wrong)
                    if wrong else 0.0
                ),
                "wrong_improvements": sum(
                    float(row["winner_score"]) > float(row["baseline_score"]) + SCORE_ATOL for row in wrong
                ),
                "correct_samples": len(correct),
                "correct_regressions": sum(
                    float(row["winner_score"]) < float(row["baseline_score"]) - SCORE_ATOL for row in correct
                ),
                "mean_effective_budget": mean([float(row["effective_budget"]) for row in rows]),
                "mean_visual_on_layers": mean([float(row["winner_num_visual_on_layers"]) for row in rows]),
                "mean_correct_answer_logprob": mean(logprobs) if logprobs else None,
                "answer_logprob_count": len(logprobs),
                "winner_route_key_counts": dict(Counter(str(row["winner_route_key"]) for row in rows)),
            }
    return {"curves": curves, "winners": winners}
