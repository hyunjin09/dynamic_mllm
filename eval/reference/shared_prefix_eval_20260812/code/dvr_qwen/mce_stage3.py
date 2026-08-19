"""MCE-3 smoke intervention planning and summary helpers."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any

from dvr_qwen.mce_inventory import BENCHMARKS
from dvr_qwen.mce_stage0 import source_row_for_manifest


DEFAULT_SMOKE_LAYERS = (0, 13, 27)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def route_from_binary(mask: list[int | bool]) -> list[bool]:
    return [bool(value) for value in mask]


def route_binary(route: list[bool]) -> list[int]:
    return [1 if value else 0 for value in route]


def mask_one_based(route: list[bool]) -> list[int]:
    return [idx + 1 for idx, value in enumerate(route) if value]


def route_with_layer(route: list[bool], layer_idx: int, visual_on: bool) -> list[bool]:
    out = list(route)
    out[layer_idx] = visual_on
    return out


def route_key(route: list[bool]) -> str:
    return "".join("1" if value else "0" for value in route)


def delta_sign(delta: float, *, atol: float = 1e-9) -> str:
    if delta > atol:
        return "improved"
    if delta < -atol:
        return "regressed"
    return "neutral"


def select_smoke_manifest_rows(
    manifest_rows: list[dict[str, Any]],
    *,
    per_dataset_per_cohort: int = 1,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[(str(row["dataset"]), str(row["cohort"]))].append(row)

    selected: list[dict[str, Any]] = []
    for dataset in BENCHMARKS:
        for cohort in ("wrong", "correct"):
            rows = sorted(grouped[(dataset, cohort)], key=lambda row: str(row["sample_id"]))
            selected.extend(rows[:per_dataset_per_cohort])
    return selected


def _random_matched_route(num_layers: int, num_on: int, seed: int) -> list[bool]:
    if num_on <= 0:
        return [False] * num_layers
    if num_on >= num_layers:
        return [True] * num_layers
    rng = random.Random(seed)
    selected = set(rng.sample(range(num_layers), num_on))
    return [idx in selected for idx in range(num_layers)]


def stable_sample_seed(sample_id: str, random_seed: int) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()
    return random_seed + int(digest[:8], 16)


def build_intervention_specs(
    manifest_rows: list[dict[str, Any]],
    source_indexes: dict[str, dict[str, dict[str, Any]]],
    *,
    layer_indices: tuple[int, ...] = DEFAULT_SMOKE_LAYERS,
    random_seed: int = 20260619,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        source_row = source_row_for_manifest(manifest_row, source_indexes)
        oracle_route = route_from_binary(source_row["binary_oracle"]["best"]["visual_on_mask"])
        num_layers = len(oracle_route)
        all_on_route = [True] * num_layers
        base = {
            "sample_id": manifest_row["sample_id"],
            "dataset": manifest_row["dataset"],
            "cohort": manifest_row["cohort"],
            "source_pool": manifest_row["source_pool"],
        }

        specs.append(
            {
                **base,
                "intervention_id": f"{manifest_row['sample_id']}:baseline_all_visual_on",
                "mode": "baseline_all_visual_on",
                "parent_mode": None,
                "layer_zero_based": None,
                "layer_one_based": None,
                "eligible": True,
                "route": route_binary(all_on_route),
            }
        )
        specs.append(
            {
                **base,
                "intervention_id": f"{manifest_row['sample_id']}:baseline_oracle",
                "mode": "baseline_oracle",
                "parent_mode": None,
                "layer_zero_based": None,
                "layer_one_based": None,
                "eligible": True,
                "route": route_binary(oracle_route),
            }
        )

        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(f"layer index {layer_idx} outside 0..{num_layers - 1}")
            specs.append(
                {
                    **base,
                    "intervention_id": f"{manifest_row['sample_id']}:all_on_suppress:L{layer_idx + 1}",
                    "mode": "all_on_suppress",
                    "parent_mode": "baseline_all_visual_on",
                    "layer_zero_based": layer_idx,
                    "layer_one_based": layer_idx + 1,
                    "eligible": True,
                    "route": route_binary(route_with_layer(all_on_route, layer_idx, False)),
                }
            )

            addback_eligible = not oracle_route[layer_idx]
            specs.append(
                {
                    **base,
                    "intervention_id": f"{manifest_row['sample_id']}:oracle_add_back:L{layer_idx + 1}",
                    "mode": "oracle_add_back",
                    "parent_mode": "baseline_oracle",
                    "layer_zero_based": layer_idx,
                    "layer_one_based": layer_idx + 1,
                    "eligible": addback_eligible,
                    "route": route_binary(route_with_layer(oracle_route, layer_idx, True)),
                }
            )

            dropout_eligible = oracle_route[layer_idx]
            specs.append(
                {
                    **base,
                    "intervention_id": f"{manifest_row['sample_id']}:oracle_dropout:L{layer_idx + 1}",
                    "mode": "oracle_dropout",
                    "parent_mode": "baseline_oracle",
                    "layer_zero_based": layer_idx,
                    "layer_one_based": layer_idx + 1,
                    "eligible": dropout_eligible,
                    "route": route_binary(route_with_layer(oracle_route, layer_idx, False)),
                }
            )

        random_route = _random_matched_route(
            num_layers,
            int(sum(oracle_route)),
            stable_sample_seed(str(manifest_row["sample_id"]), random_seed),
        )
        specs.append(
            {
                **base,
                "intervention_id": f"{manifest_row['sample_id']}:random_matched_budget",
                "mode": "random_matched_budget",
                "parent_mode": "baseline_oracle",
                "layer_zero_based": None,
                "layer_one_based": None,
                "eligible": True,
                "route": route_binary(random_route),
            }
        )
    return specs


def summarize_intervention_specs(specs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "planned_count": len(specs),
        "eligible_count": sum(bool(spec["eligible"]) for spec in specs),
        "ineligible_count": sum(not bool(spec["eligible"]) for spec in specs),
        "planned_by_mode": dict(Counter(str(spec["mode"]) for spec in specs)),
        "eligible_by_mode": dict(Counter(str(spec["mode"]) for spec in specs if spec["eligible"])),
    }


def summarize_intervention_rows(
    rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    *,
    reused_existing_count: int = 0,
) -> dict[str, Any]:
    spec_summary = summarize_intervention_specs(specs)
    evaluated_ids = {row["intervention_id"] for row in rows}
    eligible_ids = {spec["intervention_id"] for spec in specs if spec["eligible"]}
    missing_ids = sorted(eligible_ids - evaluated_ids)
    by_mode = Counter(str(row["mode"]) for row in rows)
    by_dataset = Counter(str(row["dataset"]) for row in rows)
    score_deltas = [
        float(row["score_delta_vs_parent"])
        for row in rows
        if row.get("score_delta_vs_parent") is not None
    ]
    nll_values = [
        float(row["answer_nll"]["nll"])
        for row in rows
        if isinstance(row.get("answer_nll"), dict) and row["answer_nll"].get("nll") is not None
    ]
    return {
        **spec_summary,
        "evaluated_count": len(rows),
        "reused_existing_count": reused_existing_count,
        "missing_eligible_count": len(missing_ids),
        "missing_eligible_ids": missing_ids,
        "evaluated_by_mode": dict(by_mode),
        "evaluated_by_dataset": dict(by_dataset),
        "route_application_mismatches": sum(
            row.get("stored_generated_ids_match") is False for row in rows
        ),
        "delta_sign_missing_count": sum(
            row.get("parent_mode") is not None
            and row.get("delta_sign") not in {"improved", "regressed", "neutral"}
            for row in rows
        ),
        "score_delta_min": min(score_deltas) if score_deltas else None,
        "score_delta_max": max(score_deltas) if score_deltas else None,
        "score_delta_mean": mean(score_deltas),
        "answer_nll_missing_count": len(rows) - len(nll_values),
        "answer_nll_mean": mean(nll_values),
    }
