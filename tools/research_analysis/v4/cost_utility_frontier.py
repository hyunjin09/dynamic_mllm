from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow
import yaml


ACTIONS = ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the deterministic v4 image-only/query-conditioned cost-utility frontier."
    )
    parser.add_argument("--config", default="configs/v4_cost_utility_frontier.yaml")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def trimmed_mean(values: Iterable[float], fraction: float) -> float:
    ordered = np.sort(np.asarray(list(values), dtype=np.float64))
    trim = int(math.floor(ordered.size * fraction))
    if trim == 0:
        return float(ordered.mean())
    return float(ordered[trim:-trim].mean())


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return (base + int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")) % (2**32)


def bootstrap_mean_ci(
    values: np.ndarray, image_ids: np.ndarray, replicates: int, seed: int, parts: tuple[Any, ...]
) -> tuple[float, float]:
    unique = np.unique(image_ids)
    grouped = {image: values[image_ids == image] for image in unique}
    generator = np.random.default_rng(stable_seed(seed, *parts))
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.choice(unique, size=unique.size, replace=True)
        draws[index] = np.concatenate([grouped[image] for image in sampled]).mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def choose_with_tolerance(
    objectives: np.ndarray, costs: np.ndarray, epsilon: float, tie_indices: np.ndarray | None = None
) -> int:
    maximum = float(np.max(objectives))
    candidates = np.where(objectives >= maximum - epsilon)[0]
    if tie_indices is None:
        tie_indices = np.arange(objectives.size)
    return int(min(candidates, key=lambda index: (float(costs[index]), int(tie_indices[index]))))


def action_cost_geometry(
    record_group: list[dict[str, Any]], hidden_size: int, intermediate_size: int, kv_dim: int
) -> dict[str, int]:
    if len(record_group) != 2:
        raise ValueError("Every v4 image must have exactly two questions")
    first = record_group[0]
    visual_rows = int(first["expected_visual_token_count"])
    visual_first = int(first["expected_visual_first"])
    visual_last = int(first["expected_visual_last"])
    if any(
        int(row[field]) != int(first[field])
        for row in record_group
        for field in (
            "expected_visual_token_count",
            "expected_visual_first",
            "expected_visual_last",
        )
    ):
        raise ValueError("Same-image visual layout differs across questions")
    common_rows = max(int(row["expected_prompt_token_length"]) for row in record_group)
    postvisual_rows = common_rows - visual_last - 1
    if postvisual_rows < 1:
        raise ValueError("No post-visual query row")
    nonvisual_rows = common_rows - visual_rows
    visual_causal_edges = (
        visual_rows * visual_first + visual_rows * (visual_rows + 1) // 2
    )

    read_value_flops = 2 * postvisual_rows * visual_rows * hidden_size
    write_visual_q_flops = 2 * visual_rows * hidden_size * hidden_size
    write_visual_attention_flops = 4 * hidden_size * visual_causal_edges
    write_visual_o_flops = 2 * visual_rows * hidden_size * hidden_size
    write_visual_ffn_flops = 6 * visual_rows * hidden_size * intermediate_size
    write_flops = (
        write_visual_q_flops
        + write_visual_attention_flops
        + write_visual_o_flops
        + write_visual_ffn_flops
    )

    invariant_visual_kv_flops = 4 * visual_rows * hidden_size * kv_dim
    invariant_nonvisual_linear_flops = (
        4 * nonvisual_rows * hidden_size * hidden_size
        + 4 * nonvisual_rows * hidden_size * kv_dim
        + 6 * nonvisual_rows * hidden_size * intermediate_size
    )
    invariant_nonvisual_attention_flops = (
        4 * hidden_size * nonvisual_rows * (nonvisual_rows + 1) // 2
    )
    invariant_text_visual_qk_flops = 2 * postvisual_rows * visual_rows * hidden_size
    invariant_total = (
        invariant_visual_kv_flops
        + invariant_nonvisual_linear_flops
        + invariant_nonvisual_attention_flops
        + invariant_text_visual_qk_flops
    )
    return {
        "common_prompt_rows": common_rows,
        "visual_rows": visual_rows,
        "prefix_rows": visual_first,
        "postvisual_query_rows": postvisual_rows,
        "nonvisual_rows": nonvisual_rows,
        "visual_causal_edges": visual_causal_edges,
        "read_value_flops": read_value_flops,
        "write_visual_q_flops": write_visual_q_flops,
        "write_visual_attention_flops": write_visual_attention_flops,
        "write_visual_o_flops": write_visual_o_flops,
        "write_visual_ffn_flops": write_visual_ffn_flops,
        "write_flops": write_flops,
        "full_visual_flops": read_value_flops + write_flops,
        "invariant_visual_kv_flops": invariant_visual_kv_flops,
        "invariant_nonvisual_linear_flops": invariant_nonvisual_linear_flops,
        "invariant_nonvisual_attention_flops": invariant_nonvisual_attention_flops,
        "invariant_text_visual_qk_flops": invariant_text_visual_qk_flops,
        "invariant_total_flops": invariant_total,
    }


def action_costs(geometry: dict[str, int]) -> np.ndarray:
    read = float(geometry["read_value_flops"])
    write = float(geometry["write_flops"])
    return np.asarray([0.0, read, write, read + write], dtype=np.float64)


def build_pairs(
    q_frame: pd.DataFrame,
    manifest: list[dict[str, Any]],
    image_layer: pd.DataFrame,
    dimensions: dict[str, int],
) -> list[dict[str, Any]]:
    manifest_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        manifest_groups[str(row["image_id"])].append(row)
    robust = {
        (str(row.image_id), int(row.layer)): bool(row.robust_best_action_disagreement_mean)
        for row in image_layer.itertuples(index=False)
    }
    pairs: list[dict[str, Any]] = []
    for (image_id, layer), part in q_frame.groupby(["image_id", "layer"], sort=True):
        part = part.sort_values("question_index")
        if len(part) != 2:
            raise ValueError(f"{image_id} layer {layer} does not have two questions")
        geometry = action_cost_geometry(
            manifest_groups[str(image_id)],
            dimensions["hidden_size"],
            dimensions["intermediate_size"],
            dimensions["kv_dim"],
        )
        costs = action_costs(geometry)
        full = costs[ACTION_INDEX["FULL"]]
        q = np.asarray(
            [
                [float(row[f"q_mean_{action.lower()}"]) for action in ACTIONS]
                for _, row in part.iterrows()
            ],
            dtype=np.float64,
        )
        full_baseline = float(q[:, ACTION_INDEX["FULL"]].mean())
        questions = [str(value) for value in part["question"]]
        pairs.append(
            {
                "image_id": str(image_id),
                "layer": int(layer),
                "q": q,
                "costs": costs,
                "normalized_costs": costs / full,
                "geometry": geometry,
                "robust_non_tie": robust[(str(image_id), int(layer))],
                "mean_question_words": float(np.mean([len(value.split()) for value in questions])),
                "mean_answer_tokens": float(part["answer_token_length"].mean()),
                "any_multi_token_answer": bool((part["answer_token_length"] > 1).any()),
                "full_baseline": full_baseline,
                "full_difficulty": -full_baseline,
            }
        )
    if len(pairs) != 840:
        raise ValueError(f"Expected 840 image-layer pairs, found {len(pairs)}")
    return pairs


def select_penalized(pair: dict[str, Any], oracle: str, penalty: float, epsilon: float):
    q = pair["q"]
    costs = pair["costs"]
    normalized = pair["normalized_costs"]
    if oracle == "image_only":
        objective = q.mean(axis=0) - penalty * normalized
        action = choose_with_tolerance(objective, costs, epsilon)
        actions = (action, action)
    else:
        selected = []
        for question in range(2):
            objective = q[question] - penalty * normalized
            selected.append(choose_with_tolerance(objective, costs, epsilon))
        actions = tuple(selected)
    utility = float(np.mean([q[index, action] for index, action in enumerate(actions)]))
    cost = float(np.mean([costs[action] for action in actions]))
    return actions, utility, cost


def select_budget(pair: dict[str, Any], oracle: str, budget: float, epsilon: float):
    q = pair["q"]
    costs = pair["costs"]
    normalized = pair["normalized_costs"]
    if oracle == "image_only":
        candidates = [index for index in range(4) if normalized[index] <= budget + 1e-12]
        objectives = np.asarray([q[:, index].mean() for index in candidates])
        candidate_costs = np.asarray([costs[index] for index in candidates])
        local = choose_with_tolerance(
            objectives, candidate_costs, epsilon, np.asarray(candidates)
        )
        action = candidates[local]
        actions = (action, action)
    else:
        combinations = [
            combo
            for combo in itertools.product(range(4), repeat=2)
            if np.mean([normalized[index] for index in combo]) <= budget + 1e-12
        ]
        objectives = np.asarray(
            [np.mean([q[index, action] for index, action in enumerate(combo)]) for combo in combinations]
        )
        candidate_costs = np.asarray(
            [np.mean([costs[action] for action in combo]) for combo in combinations]
        )
        tie_order = np.asarray([combo[0] * 4 + combo[1] for combo in combinations])
        local = choose_with_tolerance(objectives, candidate_costs, epsilon, tie_order)
        actions = combinations[local]
    utility = float(np.mean([q[index, action] for index, action in enumerate(actions)]))
    cost = float(np.mean([costs[action] for action in actions]))
    return actions, utility, cost


def pair_frontier_flags(pair: dict[str, Any], epsilon: float) -> dict[str, bool]:
    q = pair["q"]
    costs = pair["costs"]
    normalized = pair["normalized_costs"]
    shared = [(float(normalized[a]), float(q[:, a].mean()), (a, a)) for a in range(4)]
    query = [
        (
            float(np.mean([normalized[a0], normalized[a1]])),
            float(np.mean([q[0, a0], q[1, a1]])),
            (a0, a1),
        )
        for a0, a1 in itertools.product(range(4), repeat=2)
    ]
    expands = any(
        utility
        > max(shared_utility for shared_cost, shared_utility, _ in shared if shared_cost <= cost + 1e-12)
        + epsilon
        for cost, utility, actions in query
        if actions[0] != actions[1]
    )
    image_actions, image_utility, image_cost = select_penalized(pair, "image_only", 0.0, epsilon)
    dominates_unconstrained = any(
        (
            cost <= image_cost + 1e-3
            and utility > image_utility + epsilon
        )
        or (
            cost < image_cost - 1e-3
            and utility >= image_utility - epsilon
        )
        for cost, utility, actions in [
            (
                float(np.mean([costs[a0], costs[a1]])),
                float(np.mean([q[0, a0], q[1, a1]])),
                (a0, a1),
            )
            for a0, a1 in itertools.product(range(4), repeat=2)
            if a0 != a1
        ]
    )
    return {
        "query_frontier_strictly_expands_shared": expands,
        "query_strictly_dominates_unconstrained_shared": dominates_unconstrained,
        "unconstrained_image_shared_action": image_actions[0],
    }


def aggregate_selection(
    pairs: list[dict[str, Any]], selections: list[tuple[tuple[int, int], float, float]], trim: float
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    utilities = np.asarray([row[1] for row in selections], dtype=np.float64)
    costs = np.asarray([row[2] for row in selections], dtype=np.float64)
    full = np.asarray([pair["full_baseline"] for pair in pairs], dtype=np.float64)
    full_costs = np.asarray([pair["costs"][ACTION_INDEX["FULL"]] for pair in pairs])
    relative = utilities - full
    actions = [action for row in selections for action in row[0]]
    counts = Counter(actions)
    metrics = {
        "pair_count": len(pairs),
        "image_count": len({pair["image_id"] for pair in pairs}),
        "mean_answer_utility": float(utilities.mean()),
        "median_answer_utility": float(np.median(utilities)),
        "trimmed_mean_20_answer_utility": trimmed_mean(utilities, trim),
        "mean_utility_relative_FULL": float(relative.mean()),
        "median_utility_relative_FULL": float(np.median(relative)),
        "trimmed_mean_20_utility_relative_FULL": trimmed_mean(relative, trim),
        "mean_local_visual_compute_gflops": float(costs.mean() / 1e9),
        "median_local_visual_compute_gflops": float(np.median(costs) / 1e9),
        "mean_normalized_compute": float(np.mean(costs / full_costs)),
    }
    for action, index in ACTION_INDEX.items():
        metrics[f"{action.lower()}_selection_rate"] = counts[index] / (2 * len(pairs))
    return metrics, {
        "utilities": utilities,
        "relative": relative,
        "costs": costs,
        "normalized_costs": costs / full_costs,
        "image_ids": np.asarray([pair["image_id"] for pair in pairs]),
    }


def scope_definitions(pairs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, list[int]]:
    scopes: dict[str, list[int]] = {
        "all": list(range(len(pairs))),
        "robust_non_tie": [index for index, pair in enumerate(pairs) if pair["robust_non_tie"]],
    }
    for layer in config["layers"]:
        scopes[f"layer_{layer}"] = [index for index, pair in enumerate(pairs) if pair["layer"] == int(layer)]
    bins = config["robustness"]["question_length_word_bins"]
    labels = config["robustness"]["question_length_labels"]
    for lower, upper, label in zip(bins[:-1], bins[1:], labels):
        scopes[f"question_length_{label}"] = [
            index
            for index, pair in enumerate(pairs)
            if float(lower) < pair["mean_question_words"] <= float(upper)
            or (float(lower) == 0 and pair["mean_question_words"] == 0)
        ]
    scopes["answer_length_all_one_token"] = [
        index for index, pair in enumerate(pairs) if not pair["any_multi_token_answer"]
    ]
    scopes["answer_length_any_multi_token"] = [
        index for index, pair in enumerate(pairs) if pair["any_multi_token_answer"]
    ]
    image_difficulty = defaultdict(list)
    for pair in pairs:
        image_difficulty[pair["image_id"]].append(pair["full_difficulty"])
    image_scores = {image: float(np.mean(values)) for image, values in image_difficulty.items()}
    ordered = sorted(image_scores, key=lambda image: (image_scores[image], image))
    thirds = np.array_split(np.asarray(ordered, dtype=object), 3)
    for label, images in zip(("easy", "medium", "hard"), thirds):
        selected = set(str(image) for image in images)
        scopes[f"full_difficulty_{label}_tertile"] = [
            index for index, pair in enumerate(pairs) if pair["image_id"] in selected
        ]
    if any(not indices for indices in scopes.values()):
        empty = [scope for scope, indices in scopes.items() if not indices]
        raise ValueError(f"Predefined scope is empty: {empty}")
    return scopes


def raw_frontiers(
    pairs: list[dict[str, Any]], scopes: dict[str, list[int]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, float], dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    arrays: dict[tuple[str, str, str, float], dict[str, np.ndarray]] = {}
    epsilon = float(config["epsilon_utility"])
    trim = float(config["trim_fraction"])
    for scope, indices in scopes.items():
        selected_pairs = [pairs[index] for index in indices]
        for mode, values in (
            ("lambda", [float(value) for value in config["lambda_grid"]]),
            ("budget", [float(value) for value in config["relative_budget_grid"]]),
        ):
            for value in values:
                oracle_results = {}
                for oracle in ("image_only", "image_query"):
                    selections = [
                        select_penalized(pair, oracle, value, epsilon)
                        if mode == "lambda"
                        else select_budget(pair, oracle, value, epsilon)
                        for pair in selected_pairs
                    ]
                    metrics, selected_arrays = aggregate_selection(selected_pairs, selections, trim)
                    key = (scope, mode, oracle, value)
                    arrays[key] = selected_arrays
                    oracle_results[oracle] = (metrics, selected_arrays)
                image_metrics, image_arrays = oracle_results["image_only"]
                query_metrics, query_arrays = oracle_results["image_query"]
                utility_difference = query_arrays["relative"] - image_arrays["relative"]
                cost_difference_gflops = (query_arrays["costs"] - image_arrays["costs"]) / 1e9
                strict = (
                    ((query_arrays["costs"] <= image_arrays["costs"] + 1e-3) & (utility_difference > epsilon))
                    | ((query_arrays["costs"] < image_arrays["costs"] - 1e-3) & (utility_difference >= -epsilon))
                )
                shared = {
                    "query_minus_image_mean_utility": float(utility_difference.mean()),
                    "query_minus_image_median_utility": float(np.median(utility_difference)),
                    "query_minus_image_trimmed_mean_20_utility": trimmed_mean(utility_difference, trim),
                    "query_minus_image_mean_compute_gflops": float(cost_difference_gflops.mean()),
                    "strict_selected_pareto_dominance_fraction": float(strict.mean()),
                }
                for oracle, (metrics, _) in oracle_results.items():
                    rows.append(
                        {
                            "record_type": "raw_grid",
                            "scope": scope,
                            "selection_mode": mode,
                            "grid_value": value,
                            "oracle_type": oracle,
                            **metrics,
                            **shared,
                        }
                    )
    return rows, arrays


def nondominated_curve(
    rows: pd.DataFrame, utility_column: str = "mean_utility_relative_FULL"
) -> tuple[np.ndarray, np.ndarray]:
    grouped = (
        rows.groupby("mean_normalized_compute", as_index=False)[utility_column]
        .max()
        .sort_values("mean_normalized_compute")
    )
    costs = []
    utilities = []
    maximum = -math.inf
    for row in grouped.itertuples(index=False):
        utility = float(getattr(row, utility_column))
        if utility > maximum + 1e-12:
            costs.append(float(row.mean_normalized_compute))
            utilities.append(utility)
            maximum = utility
    return np.asarray(costs), np.asarray(utilities)


def maximum_contiguous_width(mask: np.ndarray, spacing: float) -> float:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return max(0.0, (longest - 1) * spacing)


def interpolated_frontiers(
    raw_rows: list[dict[str, Any]],
    config: dict[str, Any],
    scope: str,
    utility_column: str = "mean_utility_relative_FULL",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = pd.DataFrame(raw_rows)
    frame = frame[(frame.scope == scope) & (frame.record_type == "raw_grid")]
    count = int(config["frontier_interpolation"]["normalized_compute_grid_points"])
    x_grid = np.linspace(0.0, 1.0, count)
    curves = {}
    source_points = {}
    for oracle in ("image_only", "image_query"):
        costs, utilities = nondominated_curve(
            frame[frame.oracle_type == oracle], utility_column
        )
        source_points[oracle] = {"cost": costs.tolist(), "utility": utilities.tolist()}
        curves[oracle] = np.interp(x_grid, costs, utilities, left=utilities[0], right=utilities[-1])
    gap = curves["image_query"] - curves["image_only"]
    threshold = float(config["interpretation_reference"]["practical_utility_nats_per_token"])
    spacing = 1.0 / (count - 1)
    rows = []
    for index, cost in enumerate(x_grid):
        for oracle in ("image_only", "image_query"):
            rows.append(
                {
                    "record_type": "interpolated_frontier",
                    "scope": scope,
                    "selection_mode": "linear_pareto_interpolation",
                    "utility_aggregation": utility_column,
                    "grid_value": float(cost),
                    "oracle_type": oracle,
                    "mean_normalized_compute": float(cost),
                    "mean_utility_relative_FULL": float(curves[oracle][index]),
                    "query_minus_image_mean_utility": float(gap[index]),
                }
            )
    utility_targets = np.linspace(
        float(curves["image_only"][0]), float(curves["image_only"].max()), count
    )
    inverse = {}
    for oracle in ("image_only", "image_query"):
        utility = curves[oracle]
        unique_utility, first_indices = np.unique(utility, return_index=True)
        inverse[oracle] = np.interp(
            utility_targets,
            unique_utility,
            x_grid[first_indices],
            left=0.0,
            right=float(x_grid[first_indices][-1]),
        )
    savings = inverse["image_only"] - inverse["image_query"]
    summary = {
        "scope": scope,
        "utility_aggregation": utility_column,
        "source_pareto_points": source_points,
        "integrated_utility_gap_normalized_compute": float(np.trapezoid(gap, x_grid)),
        "mean_matched_compute_utility_gain": float(gap.mean()),
        "maximum_matched_compute_utility_gain": float(gap.max()),
        "normalized_compute_fraction_with_gain_at_least_0_05": float(np.mean(gap >= threshold)),
        "maximum_contiguous_compute_width_with_gain_at_least_0_05": maximum_contiguous_width(
            gap >= threshold, spacing
        ),
        "mean_matched_utility_compute_saving_fraction_FULL": float(savings.mean()),
        "maximum_matched_utility_compute_saving_fraction_FULL": float(savings.max()),
        "utility_target_fraction_with_compute_saving_at_least_0_10_FULL": float(
            np.mean(
                savings
                >= float(config["interpretation_reference"]["material_compute_fraction_of_mean_FULL"])
            )
        ),
    }
    return rows, summary


def unconstrained_summary(
    pairs: list[dict[str, Any]], config: dict[str, Any], frontier_flags: list[dict[str, bool]]
) -> dict[str, Any]:
    epsilon = float(config["epsilon_utility"])
    outputs = {}
    selections_by_oracle = {}
    for oracle in ("image_only", "image_query"):
        selections = [select_penalized(pair, oracle, 0.0, epsilon) for pair in pairs]
        metrics, arrays = aggregate_selection(pairs, selections, float(config["trim_fraction"]))
        outputs[oracle] = metrics
        selections_by_oracle[oracle] = (selections, arrays)
    image_selections, image_arrays = selections_by_oracle["image_only"]
    query_selections, query_arrays = selections_by_oracle["image_query"]
    utility_gap = query_arrays["relative"] - image_arrays["relative"]
    compute_saving = image_arrays["costs"] - query_arrays["costs"]
    full_costs = np.asarray([pair["costs"][ACTION_INDEX["FULL"]] for pair in pairs])
    robust_mask = np.asarray([pair["robust_non_tie"] for pair in pairs], dtype=bool)
    config_threshold = float(config["interpretation_reference"]["material_compute_fraction_of_mean_FULL"])
    practical = float(config["interpretation_reference"]["practical_utility_nats_per_token"])
    summary = {
        "image_only": outputs["image_only"],
        "image_query": outputs["image_query"],
        "query_minus_image_utility": {
            "mean": float(utility_gap.mean()),
            "median": float(np.median(utility_gap)),
            "trimmed_mean_20": trimmed_mean(utility_gap, float(config["trim_fraction"])),
        },
        "image_minus_query_compute": {
            "mean_gflops": float(compute_saving.mean() / 1e9),
            "mean_fraction_FULL": float(np.mean(compute_saving / full_costs)),
            "median_fraction_FULL": float(np.median(compute_saving / full_costs)),
            "robust_disagreement_mean_fraction_FULL": float(
                np.mean((compute_saving / full_costs)[robust_mask])
            ),
        },
        "pairwise_frontier": {
            "strict_expansion_fraction": float(
                np.mean([row["query_frontier_strictly_expands_shared"] for row in frontier_flags])
            ),
            "unconstrained_shared_dominance_fraction": float(
                np.mean(
                    [row["query_strictly_dominates_unconstrained_shared"] for row in frontier_flags]
                )
            ),
        },
    }
    summary["conservative_FULL_overcompute_hypothesis_supported"] = bool(
        summary["image_minus_query_compute"]["mean_fraction_FULL"] >= config_threshold
        and summary["query_minus_image_utility"]["mean"] <= practical
        and summary["query_minus_image_utility"]["mean"] >= -epsilon
    )
    return summary


def action_cost_rows(
    pairs: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    epsilon = float(config["epsilon_utility"])
    rows = []
    for pair in pairs:
        image_actions, _, _ = select_penalized(pair, "image_only", 0.0, epsilon)
        query_actions, _, _ = select_penalized(pair, "image_query", 0.0, epsilon)
        geometry = pair["geometry"]
        full = pair["costs"][ACTION_INDEX["FULL"]]
        for action, action_index in ACTION_INDEX.items():
            cost = pair["costs"][action_index]
            rows.append(
                {
                    "image_id": pair["image_id"],
                    "layer": pair["layer"],
                    "action": action,
                    "read_enabled": int(action in {"READ_ONLY", "FULL"}),
                    "write_enabled": int(action in {"WRITE_ONLY", "FULL"}),
                    **geometry,
                    "action_local_visual_flops": int(cost),
                    "action_local_visual_gflops": cost / 1e9,
                    "incremental_flops_vs_IGNORE": int(cost),
                    "incremental_flops_vs_FULL": int(cost - full),
                    "normalized_cost_relative_FULL": cost / full,
                    "unconstrained_image_action": ACTIONS[image_actions[0]],
                    "unconstrained_query0_action": ACTIONS[query_actions[0]],
                    "unconstrained_query1_action": ACTIONS[query_actions[1]],
                    "robust_non_tie": pair["robust_non_tie"],
                    "current_counterfactual_runner_executes_dense_FULL_for_every_action": True,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    q_path = Path(config["source_q"])
    source_analysis = json.loads(Path(config["source_analysis_manifest"]).read_text())
    if source_analysis["artifacts"][str(q_path)] != sha256_file(q_path):
        raise RuntimeError("Frozen Q matrix checksum mismatch")
    q_frame = pd.read_parquet(q_path)
    if len(q_frame) != 1680 or sorted(q_frame.layer.unique()) != [int(x) for x in config["layers"]]:
        raise RuntimeError("Frozen Q matrix shape/layer grid mismatch")
    manifest_path = Path(config["source_manifest"])
    if sha256_file(manifest_path) != source_analysis["manifest_sha256"]:
        raise RuntimeError("Frozen discovery manifest checksum mismatch")
    manifest = read_jsonl(manifest_path)
    image_layer = pd.read_csv(
        "outputs/v4_discovery/analysis_v1/image_layer_query_dependence_v1.csv",
        dtype={"image_id": str},
    )
    model_config = yaml.safe_load(Path(config["model_config"]).read_text())
    hf_config_path = Path(model_config["snapshot_path"]) / "config.json"
    hf_config = json.loads(hf_config_path.read_text())
    dimensions = {
        "hidden_size": int(hf_config["hidden_size"]),
        "intermediate_size": int(hf_config["intermediate_size"]),
        "kv_dim": int(hf_config["hidden_size"])
        * int(hf_config["num_key_value_heads"])
        // int(hf_config["num_attention_heads"]),
    }
    pairs = build_pairs(q_frame, manifest, image_layer, dimensions)
    scopes = scope_definitions(pairs, config)
    frontier_flags = [pair_frontier_flags(pair, float(config["epsilon_utility"])) for pair in pairs]
    raw_rows, _ = raw_frontiers(pairs, scopes, config)
    interpolated_rows = []
    interpolation_summary = {}
    aggregation_robustness = {}
    for scope in ("all", "robust_non_tie"):
        rows, summary = interpolated_frontiers(
            raw_rows, config, scope, "mean_utility_relative_FULL"
        )
        interpolated_rows.extend(rows)
        interpolation_summary[scope] = summary
    for scope in scopes:
        aggregation_robustness[scope] = {}
        for utility_column in (
            "mean_utility_relative_FULL",
            "median_utility_relative_FULL",
            "trimmed_mean_20_utility_relative_FULL",
        ):
            _, summary = interpolated_frontiers(
                raw_rows, config, scope, utility_column
            )
            aggregation_robustness[scope][utility_column] = summary
    frontier_rows = raw_rows + interpolated_rows
    frontier_path = Path(config["output_frontier"])
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frontier_rows).to_csv(frontier_path, index=False)

    action_rows = action_cost_rows(pairs, config)
    action_path = Path(config["output_action_costs"])
    pd.DataFrame(action_rows).to_csv(action_path, index=False)

    full_costs = np.asarray([pair["costs"][ACTION_INDEX["FULL"]] for pair in pairs])
    read_costs = np.asarray([pair["costs"][ACTION_INDEX["READ_ONLY"]] for pair in pairs])
    write_costs = np.asarray([pair["costs"][ACTION_INDEX["WRITE_ONLY"]] for pair in pairs])
    unconstrained = unconstrained_summary(pairs, config, frontier_flags)
    primary_gain = interpolation_summary["all"]
    material_compute = float(config["interpretation_reference"]["material_compute_fraction_of_mean_FULL"])
    practical_utility = float(config["interpretation_reference"]["practical_utility_nats_per_token"])
    minimum_width = float(
        config["interpretation_reference"]["minimum_contiguous_normalized_compute_width"]
    )
    any_point_material_threshold_crossed = bool(
        primary_gain["maximum_matched_utility_compute_saving_fraction_FULL"]
        >= material_compute
        or primary_gain["maximum_matched_compute_utility_gain"] >= practical_utility
    )
    sustained_material_frontier_advantage = bool(
        (
            primary_gain["maximum_contiguous_compute_width_with_gain_at_least_0_05"]
            >= minimum_width
            and primary_gain["maximum_matched_compute_utility_gain"] >= practical_utility
        )
        or (
            primary_gain["utility_target_fraction_with_compute_saving_at_least_0_10_FULL"]
            >= minimum_width
            and primary_gain["maximum_matched_utility_compute_saving_fraction_FULL"]
            >= material_compute
        )
    )
    summary = {
        "schema_version": "v4_cost_utility_frontier_summary_v1",
        "deterministic_reanalysis_only": True,
        "new_model_inference": False,
        "source": {
            "q_sha256": sha256_file(q_path),
            "manifest_sha256": sha256_file(manifest_path),
            "source_analysis_manifest_sha256": sha256_file(Path(config["source_analysis_manifest"])),
            "config_sha256": sha256_file(config_path),
            "model_config_sha256": sha256_file(Path(config["model_config"])),
            "hf_config_sha256": sha256_file(hf_config_path),
        },
        "dimensions": dimensions,
        "counts": {
            "question_layer_q_matrices": len(q_frame),
            "image_layer_pairs": len(pairs),
            "unique_images": len({pair["image_id"] for pair in pairs}),
            "frontier_rows": len(frontier_rows),
            "action_cost_rows": len(action_rows),
        },
        "cost_distribution_gflops": {
            "READ": {
                "mean": float(read_costs.mean() / 1e9),
                "median": float(np.median(read_costs) / 1e9),
                "minimum": float(read_costs.min() / 1e9),
                "maximum": float(read_costs.max() / 1e9),
            },
            "WRITE": {
                "mean": float(write_costs.mean() / 1e9),
                "median": float(np.median(write_costs) / 1e9),
                "minimum": float(write_costs.min() / 1e9),
                "maximum": float(write_costs.max() / 1e9),
            },
            "FULL": {
                "mean": float(full_costs.mean() / 1e9),
                "median": float(np.median(full_costs) / 1e9),
                "minimum": float(full_costs.min() / 1e9),
                "maximum": float(full_costs.max() / 1e9),
            },
            "mean_READ_fraction_FULL": float(np.mean(read_costs / full_costs)),
        },
        "unconstrained": unconstrained,
        "frontier_comparison": interpolation_summary,
        "frontier_aggregation_robustness": aggregation_robustness,
        "any_point_material_threshold_crossed": any_point_material_threshold_crossed,
        "sustained_material_frontier_advantage_under_frozen_reference": sustained_material_frontier_advantage,
        "scope_pair_counts": {scope: len(indices) for scope, indices in scopes.items()},
        "artifacts": {
            str(frontier_path): sha256_file(frontier_path),
            str(action_path): sha256_file(action_path),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "claim_boundary": config["interpretation_reference"]["claim_boundary"],
    }
    summary_path = Path(config["output_summary"])
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "complete": True,
                "frontier_rows": len(frontier_rows),
                "action_cost_rows": len(action_rows),
                "any_point_material_threshold_crossed": any_point_material_threshold_crossed,
                "sustained_material_frontier_advantage": sustained_material_frontier_advantage,
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
