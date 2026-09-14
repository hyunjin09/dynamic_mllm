#!/usr/bin/env python3
"""Analyze structure, mechanism, learnability, and transfer of READ harm."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dense_failure_stage2.predictability_learnability import (  # noqa: E402
    binary_classification_metrics,
    regression_metrics,
)
from dense_failure_stage2.read_harm_learnability import (  # noqa: E402
    FEATURE_GROUP_ORDER,
    adjacent_transition_counts,
    contiguous_run_lengths,
    exact_nuisance_matches,
    fit_predict_fold,
    select_dense_winner,
    validate_feature_census,
)
from experiments.run_read_harm_structure_learnability import (  # noqa: E402
    DEFAULT_CONFIG,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    canonical_hash,
    file_sha256,
    read_csv,
    read_json,
    read_jsonl,
    resolve_path,
    verify_contract,
)


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"support": 0, "mean": float("nan"), "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "support": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)), "q75": float(np.quantile(array, 0.75)),
    }


def _depth_bin(value: int, bins: Mapping[str, Sequence[int]]) -> str:
    for name, bounds in bins.items():
        if int(bounds[0]) <= int(value) <= int(bounds[1]):
            return str(name)
    raise ValueError(f"value outside frozen bins: {value}")


def _bin_index(value: int, edges: Sequence[int]) -> str:
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if int(left) <= int(value) < int(right):
            return f"b{index}"
    raise ValueError(f"value outside frozen edges: {value}")


def _uid_weights(rows: Sequence[Mapping[str, Any]], indices: np.ndarray) -> np.ndarray:
    counts = Counter(str(rows[int(index)]["uid"]) for index in indices)
    weights = np.asarray([1.0 / counts[str(rows[int(index)]["uid"])] for index in indices], dtype=np.float64)
    return weights / weights.mean()


def _sequence_metrics(sequences: Sequence[np.ndarray]) -> dict[str, float]:
    transitions: Counter[tuple[str, str]] = Counter()
    harmful_runs: list[int] = []
    beneficial_runs: list[int] = []
    harmful_layers = harmful_in_2 = harmful_in_3 = 0
    beneficial_layers = beneficial_in_2 = beneficial_in_3 = 0
    for values in sequences:
        transitions.update(adjacent_transition_counts(values))
        hr = contiguous_run_lengths(values, sign="harmful")
        br = contiguous_run_lengths(values, sign="beneficial")
        harmful_runs.extend(hr)
        beneficial_runs.extend(br)
        harmful_layers += sum(hr)
        harmful_in_2 += sum(length for length in hr if length >= 2)
        harmful_in_3 += sum(length for length in hr if length >= 3)
        beneficial_layers += sum(br)
        beneficial_in_2 += sum(length for length in br if length >= 2)
        beneficial_in_3 += sum(length for length in br if length >= 3)
    hh = transitions[("harmful", "harmful")]
    hb = transitions[("harmful", "beneficial")] + transitions[("harmful", "zero")]
    bb = transitions[("beneficial", "beneficial")]
    bh = transitions[("beneficial", "harmful")] + transitions[("beneficial", "zero")]
    return {
        "harmful_persistence": hh / (hh + hb) if hh + hb else float("nan"),
        "beneficial_persistence": bb / (bb + bh) if bb + bh else float("nan"),
        "harmful_span_mean": float(np.mean(harmful_runs)) if harmful_runs else 0.0,
        "beneficial_span_mean": float(np.mean(beneficial_runs)) if beneficial_runs else 0.0,
        "harmful_fraction_in_span_ge2": harmful_in_2 / harmful_layers if harmful_layers else 0.0,
        "harmful_fraction_in_span_ge3": harmful_in_3 / harmful_layers if harmful_layers else 0.0,
        "beneficial_fraction_in_span_ge2": beneficial_in_2 / beneficial_layers if beneficial_layers else 0.0,
        "beneficial_fraction_in_span_ge3": beneficial_in_3 / beneficial_layers if beneficial_layers else 0.0,
    }


def _flat_sequence_metrics(signs: np.ndarray, starts: np.ndarray) -> dict[str, float]:
    """Vectorized equivalent of `_sequence_metrics` for many short paths."""

    if signs.ndim != 1 or starts.shape != signs.shape or not bool(starts[0]):
        raise ValueError("flat sequence representation differs")
    valid_transition = ~starts[1:]
    left, right = signs[:-1][valid_transition], signs[1:][valid_transition]

    def runs(target: int) -> tuple[np.ndarray, int, int, int]:
        selected = signs == target
        previous = np.empty_like(selected)
        previous[0] = False
        previous[1:] = selected[:-1] & ~starts[1:]
        following = np.empty_like(selected)
        following[-1] = False
        following[:-1] = selected[1:] & ~starts[1:]
        run_starts = np.flatnonzero(selected & ~previous)
        run_ends = np.flatnonzero(selected & ~following)
        lengths = run_ends - run_starts + 1
        layers = int(lengths.sum())
        in_two = int(lengths[lengths >= 2].sum())
        in_three = int(lengths[lengths >= 3].sum())
        return lengths, layers, in_two, in_three

    harmful_runs, harmful_layers, harmful_in_2, harmful_in_3 = runs(1)
    beneficial_runs, beneficial_layers, beneficial_in_2, beneficial_in_3 = runs(-1)
    hh = int(np.sum((left == 1) & (right == 1)))
    hb = int(np.sum((left == 1) & (right != 1)))
    bb = int(np.sum((left == -1) & (right == -1)))
    bh = int(np.sum((left == -1) & (right != -1)))
    return {
        "harmful_persistence": hh / (hh + hb) if hh + hb else float("nan"),
        "beneficial_persistence": bb / (bb + bh) if bb + bh else float("nan"),
        "harmful_span_mean": float(harmful_runs.mean()) if len(harmful_runs) else 0.0,
        "beneficial_span_mean": float(beneficial_runs.mean()) if len(beneficial_runs) else 0.0,
        "harmful_fraction_in_span_ge2": harmful_in_2 / harmful_layers if harmful_layers else 0.0,
        "harmful_fraction_in_span_ge3": harmful_in_3 / harmful_layers if harmful_layers else 0.0,
        "beneficial_fraction_in_span_ge2": beneficial_in_2 / beneficial_layers if beneficial_layers else 0.0,
        "beneficial_fraction_in_span_ge3": beneficial_in_3 / beneficial_layers if beneficial_layers else 0.0,
    }


def _structure_for_rows(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], *, domain: str
) -> dict[str, Any]:
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[str(row["uid"])].append(dict(row))
    values_by_group: list[np.ndarray] = []
    sequence_indices: list[tuple[int, np.ndarray]] = []
    maps = []
    for group_index, (uid, values) in enumerate(sorted(by_uid.items())):
        values.sort(key=lambda row: int(row["layer"]))
        array = np.asarray([float(row["h_r"]) for row in values], dtype=np.float64)
        values_by_group.append(array)
        if domain == "dense":
            layers = [int(row["layer"]) for row in values]
            if any(right != left + 1 for left, right in zip(layers[:-1], layers[1:])):
                raise RuntimeError(f"noncontiguous post-trigger layer sequence: {uid}")
            sequence_indices.append((group_index, np.arange(len(values), dtype=np.int64)))
        else:
            prefix_to_index: dict[tuple[str, ...], int] = {}
            for index, row in enumerate(values):
                prefix = tuple(str(action) for action in row["prefix_actions"])
                if int(row["layer"]) != int(row["trigger_layer"]) + len(prefix):
                    raise RuntimeError(f"routed prefix depth differs: {row['state_id']}")
                if prefix in prefix_to_index:
                    raise RuntimeError(f"duplicate routed prefix node: {uid}/{prefix}")
                prefix_to_index[prefix] = index
            parents = {prefix[:-1] for prefix in prefix_to_index if prefix}
            leaves = sorted((prefix for prefix in prefix_to_index if prefix not in parents), key=lambda prefix: (len(prefix), prefix))
            if not leaves:
                raise RuntimeError(f"routed prefix DAG has no leaves: {uid}")
            for leaf in leaves:
                path = [leaf[:depth] for depth in range(len(leaf) + 1)]
                if not all(prefix in prefix_to_index for prefix in path):
                    raise RuntimeError(f"routed prefix DAG is disconnected: {uid}/{leaf}")
                sequence_indices.append((group_index, np.asarray([prefix_to_index[prefix] for prefix in path], dtype=np.int64)))
        maps.append({
            "uid": uid, "dataset": values[0]["dataset"], "source_regime": values[0]["source_regime"],
            "image_group_id": values[0]["image_group_id"], "dense_wrong": values[0]["dense_wrong"],
            "trigger_layer": values[0]["trigger_layer"],
            "states": [{
                "state_id": row["state_id"], "layer": row["layer"], "trigger_relative_depth": row["trigger_relative_depth"],
                "h_r": row["h_r"], "full_correct": row["full_correct"], "write_only_correct": row["write_only_correct"],
                "cohort": row["cohort"], "read_sign": row["read_sign"],
            } for row in values],
        })
    sequences = [values_by_group[group][indices] for group, indices in sequence_indices]
    observed = _sequence_metrics(sequences)
    offsets = np.cumsum([0] + [len(values) for values in values_by_group[:-1]])
    flat_indices = np.concatenate([offsets[group] + indices for group, indices in sequence_indices])
    starts = np.zeros(len(flat_indices), dtype=bool)
    cursor = 0
    for _, indices in sequence_indices:
        starts[cursor] = True
        cursor += len(indices)
    signs_by_group = [np.sign(values).astype(np.int8) for values in values_by_group]
    rng = np.random.default_rng(int(config["structure"]["shuffle_seed"]))
    null: dict[str, list[float]] = defaultdict(list)
    for _ in range(int(config["structure"]["shuffle_draws"])):
        shuffled_signs = np.concatenate([rng.permutation(values) for values in signs_by_group])
        metrics = _flat_sequence_metrics(shuffled_signs[flat_indices], starts)
        for key, value in metrics.items():
            null[key].append(float(value))
    null_rows = []
    for key, observed_value in observed.items():
        values = np.asarray(null[key], dtype=np.float64)
        null_rows.append({
            "domain": domain, "metric": key, "observed": observed_value,
            "null_mean": float(values.mean()), "null_q025": float(np.quantile(values, 0.025)),
            "null_q975": float(np.quantile(values, 0.975)),
            "one_sided_p_ge_observed": float((1 + np.sum(values >= observed_value)) / (len(values) + 1)),
        })
    return {"maps": maps, "observed": observed, "null": null_rows, "sequences": sequences, "by_uid": by_uid,
            "sequence_unit": "uid_chain" if domain == "dense" else "exact_prefix_DAG_root_to_leaf_path"}


def structure(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    dense = read_jsonl(output_root / "population/dense_read_state_manifest.jsonl")
    routed = read_jsonl(output_root / "population/routed_read_state_manifest.jsonl")
    primary = _structure_for_rows(dense, config, domain="dense")
    secondary = _structure_for_rows(routed, config, domain="routed")
    atomic_jsonl(output_root / "structure/per_uid_read_harm_maps.jsonl", primary["maps"])

    transitions: Counter[tuple[str, str]] = Counter()
    for sequence in primary["sequences"]:
        transitions.update(adjacent_transition_counts(sequence))
    atomic_csv(output_root / "structure/sign_transition_matrix.csv", [
        {"from_sign": key[0], "to_sign": key[1], "transitions": value}
        for key, value in sorted(transitions.items())
    ])
    atomic_csv(output_root / "structure/adjacent_persistence.csv", [
        {"domain": domain, **result["observed"]} for domain, result in (("dense", primary), ("routed", secondary))
    ])
    atomic_csv(output_root / "structure/shuffled_null_statistics.csv", primary["null"] + secondary["null"])

    span_rows = []
    for sign in ("harmful", "beneficial"):
        lengths = [length for sequence in primary["sequences"] for length in contiguous_run_lengths(sequence, sign=sign)]
        counts = Counter(lengths)
        for length, count in sorted(counts.items()):
            span_rows.append({"sign": sign, "span_length": length, "spans": count, "fraction_of_spans": count / len(lengths)})
    atomic_csv(output_root / "structure/harmful_span_statistics.csv", span_rows)

    def aggregate(group_key: str, rows_: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows_:
            grouped[row[group_key]].append(row)
        output = []
        for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
            h = [float(row["h_r"]) for row in values]
            output.append({
                group_key: key, **_summary(h),
                "harmful": sum(value > 0 for value in h), "beneficial": sum(value < 0 for value in h),
                "zero": sum(value == 0 for value in h),
                "harmful_prevalence": sum(value > 0 for value in h) / len(h),
                "harmful_flip_prevalence": sum(row["cohort"] == "read_harmful_flip" for row in values) / len(values),
            })
        return output

    atomic_csv(output_root / "structure/layer_summary.csv", aggregate("layer", dense))
    depth_rows = []
    for row in dense:
        copy = dict(row)
        copy["trigger_relative_bin"] = _depth_bin(int(row["trigger_relative_depth"]), config["structure"]["trigger_relative_bins"])
        depth_rows.append(copy)
    atomic_csv(output_root / "structure/trigger_relative_summary.csv", aggregate("trigger_relative_bin", depth_rows))

    radius = int(config["structure"]["strong_flip_neighborhood_radius"])
    neighborhoods: dict[int, list[float]] = defaultdict(list)
    for values in primary["by_uid"].values():
        by_layer = {int(row["layer"]): row for row in values}
        for center in values:
            if center["cohort"] != "read_harmful_flip":
                continue
            for offset in range(-radius, radius + 1):
                neighbor = by_layer.get(int(center["layer"]) + offset)
                if neighbor is not None:
                    neighborhoods[offset].append(float(neighbor["h_r"]))
    atomic_csv(output_root / "structure/strong_flip_neighborhood.csv", [
        {"offset": offset, **_summary(values), "harmful_prevalence": sum(value > 0 for value in values) / len(values)}
        for offset, values in sorted(neighborhoods.items())
    ])

    burden = []
    for uid, values in sorted(primary["by_uid"].items()):
        values.sort(key=lambda row: int(row["layer"]))
        harmful = [row for row in values if float(row["h_r"]) > 0]
        burden.append({
            "uid": uid, "dataset": values[0]["dataset"], "source_regime": values[0]["source_regime"],
            "image_group_id": values[0]["image_group_id"], "dense_outcome": "W" if values[0]["dense_wrong"] else "C",
            "post_trigger_layers": len(values), "harmful_layers": len(harmful),
            "harmful_fraction": len(harmful) / len(values),
            "mean_positive_h_r": float(np.mean([float(row["h_r"]) for row in harmful])) if harmful else 0.0,
            "max_h_r": max(float(row["h_r"]) for row in values),
            "first_harmful_layer": min((int(row["layer"]) for row in harmful), default=None),
            "harmful_flips": sum(row["cohort"] == "read_harmful_flip" for row in values),
        })
    atomic_csv(output_root / "structure/sample_read_burden.csv", burden)

    source_rows = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in dense:
        grouped[(str(row["dataset"]), str(row["source_regime"]))].append(row)
    for (dataset, source), values in sorted(grouped.items()):
        h = [float(row["h_r"]) for row in values]
        source_rows.append({
            "dataset": dataset, "source_regime": source, **_summary(h),
            "uids": len({row["uid"] for row in values}), "harmful_prevalence": sum(value > 0 for value in h) / len(h),
            "harmful_flips": sum(row["cohort"] == "read_harmful_flip" for row in values),
            "beneficial_flips": sum(row["cohort"] == "read_beneficial_flip" for row in values),
        })
    atomic_csv(output_root / "structure/dataset_source_structure.csv", source_rows)
    atomic_csv(output_root / "routed_secondary/structure_summary.csv", [
        {"sequence_unit": secondary["sequence_unit"], "metric": key, "value": value}
        for key, value in secondary["observed"].items()
    ])
    print(json.dumps({"structure_complete": True, "dense_states": len(dense), "routed_states": len(routed)}, sort_keys=True))


def _bin_index(value: int, edges: Sequence[int]) -> str:
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if int(left) <= int(value) < int(right):
            return f"b{index}:{left}-{right}"
    raise ValueError(f"value outside matching bins: {value}")


def _uid_weights(rows: Sequence[Mapping[str, Any]], indices: np.ndarray) -> np.ndarray:
    counts = Counter(str(rows[int(index)]["uid"]) for index in indices)
    weights = np.asarray([1.0 / counts[str(rows[int(index)]["uid"])] for index in indices], dtype=np.float64)
    return weights / weights.mean()


def _metric_bundle(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    regression = regression_metrics(truth=truth, prediction=prediction)
    mask = truth != 0.0
    classification = binary_classification_metrics(truth=(truth[mask] > 0.0).astype(int), prediction=prediction[mask])
    output = {**regression, "harmful_auroc": classification["auroc"], "harmful_auprc": classification["auprc"], "harmful_prevalence": classification["prevalence"]}
    order = np.argsort(-prediction[mask], kind="mergesort")
    labels = (truth[mask] > 0.0).astype(int)[order]
    for coverage in (0.05, 0.1, 0.2):
        count = max(1, int(math.ceil(coverage * len(labels))))
        output[f"precision_at_{coverage:g}"] = float(labels[:count].mean())
    precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    recall = np.cumsum(labels) / max(1, int(labels.sum()))
    for target in (0.9, 0.95):
        valid = precision >= target
        output[f"recall_at_precision_{target:g}"] = float(recall[valid].max()) if valid.any() else 0.0
    return output


def _matched_effects(
    matches: Sequence[Mapping[str, Any]],
    feature_index: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    *,
    comparison: str,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not matches:
        return []
    rows = []
    rng = np.random.default_rng(int(seed))
    treated_groups = np.asarray([
        str(feature_index[str(match["treated_state_id"])]["image_group_id"]) for match in matches
    ])
    unique_groups = np.unique(treated_groups)
    for feature in feature_names:
        differences = np.asarray([
            float(feature_index[str(match["treated_state_id"])]["features"][feature])
            - float(feature_index[str(match["control_state_id"])]["features"][feature])
            for match in matches
        ], dtype=np.float64)
        treated_values = np.asarray([
            float(feature_index[str(match["treated_state_id"])]["features"][feature]) for match in matches
        ])
        control_values = np.asarray([
            float(feature_index[str(match["control_state_id"])]["features"][feature]) for match in matches
        ])
        pooled = math.sqrt((float(treated_values.var()) + float(control_values.var())) / 2.0)
        boot = np.empty(int(draws), dtype=np.float64)
        for draw in range(int(draws)):
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            selected = np.concatenate([np.flatnonzero(treated_groups == group) for group in sampled])
            boot[draw] = float(differences[selected].mean())
        rows.append({
            "comparison": comparison, "feature": feature, "pairs": len(matches),
            "matched_mean_difference": float(differences.mean()),
            "matched_median_difference": float(np.median(differences)),
            "standardized_effect_size": float(differences.mean() / pooled) if pooled > 0 else 0.0,
            "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
            "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
            "bootstrap_unit": "treated_image_group_id",
        })
    return rows


def prepare_training(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    completion = read_json(output_root / "features/dense_feature_completion.json")
    if completion.get("contract_sha256") != contract["contract_sha256"] or not completion.get("complete"):
        raise RuntimeError("dense READ features are incomplete")
    states = sorted(read_jsonl(output_root / "population/dense_read_state_manifest.jsonl"), key=lambda row: str(row["state_id"]))
    features = read_jsonl(output_root / "features/read_operation_features.jsonl")
    validate_feature_census([str(row["state_id"]) for row in states], features)
    feature_index = {str(row["state_id"]): row for row in features}
    schema = read_json(output_root / "features/feature_group_manifest.json")
    all_names = list(schema["F_ALL"])

    enriched = []
    for state in states:
        feature = feature_index[str(state["state_id"])]
        trigger_bin = _depth_bin(int(state["trigger_relative_depth"]), config["structure"]["trigger_relative_bins"])
        visual_bin = _bin_index(int(feature["visual_token_count"]), config["matching"]["visual_token_count_edges"])
        text_bin = _bin_index(int(feature["text_token_count"]), config["matching"]["text_token_count_edges"])
        cell = "|".join((str(state["dataset"]), str(state["source_regime"]), str(state["layer"]), trigger_bin, visual_bin, text_bin))
        enriched.append({**state, "cell": cell, "trigger_relative_bin": trigger_bin, "visual_token_count_bin": visual_bin, "text_token_count_bin": text_bin})

    comparisons = (
        ("harmful_vs_beneficial", "read_beneficial_flip"),
        ("harmful_vs_stable_wrong", "stable_wrong"),
        ("harmful_vs_stable_correct", "stable_correct"),
    )
    all_effects = []
    primary_matches = []
    for name, control in comparisons:
        matches = exact_nuisance_matches(
            enriched, treated="read_harmful_flip", control=control,
            cell_key="cell", seed=int(config["matching"]["seed"]),
        )
        augmented = []
        for index, match in enumerate(matches):
            treated = feature_index[str(match["treated_state_id"])]
            control_row = feature_index[str(match["control_state_id"])]
            augmented.append({
                "pair_id": f"{name}:{index:06d}", **match,
                "treated_uid": treated["uid"], "control_uid": control_row["uid"],
                "treated_image_group_id": treated["image_group_id"],
                "control_image_group_id": control_row["image_group_id"],
            })
        atomic_jsonl(output_root / f"matched_mechanism/{name}_matches.jsonl", augmented)
        if name == "harmful_vs_beneficial":
            primary_matches = augmented
        all_effects.extend(_matched_effects(
            augmented, feature_index, all_names, comparison=name,
            draws=int(config["matching"]["bootstrap_draws"]),
            seed=int(config["matching"]["bootstrap_seed"]),
        ))
        if name == "harmful_vs_beneficial":
            strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for match in augmented:
                treated = feature_index[str(match["treated_state_id"])]
                strata[f"{treated['dataset']}|{treated['source_regime']}"].append(match)
            for stratum, stratum_matches in sorted(strata.items()):
                if len(stratum_matches) < int(config["decision_rules"]["mechanism"]["minimum_stratum_pairs"]):
                    continue
                all_effects.extend(_matched_effects(
                    stratum_matches, feature_index, all_names,
                    comparison=f"{name}|{stratum}",
                    draws=int(config["matching"]["bootstrap_draws"]),
                    seed=int(config["matching"]["bootstrap_seed"]) + len(all_effects),
                ))
    if not primary_matches:
        raise RuntimeError("primary harmful-vs-beneficial exact matching has no support")
    atomic_csv(output_root / "matched_mechanism/feature_effect_sizes.csv", all_effects)

    distribution = []
    for feature in all_names:
        for cohort in ("all", "read_harmful_flip", "read_beneficial_flip", "stable_wrong", "stable_correct"):
            selected = features if cohort == "all" else [row for row in features if row["cohort"] == cohort]
            values = [float(row["features"][feature]) for row in selected]
            distribution.append({"feature": feature, "cohort": cohort, **_summary(values)})
    atomic_csv(output_root / "features/feature_distribution_summary.csv", distribution)

    order_path = output_root / "work/dense_state_order.jsonl"
    atomic_jsonl(order_path, [{
        "row_index": index, "state_id": row["state_id"], "uid": row["uid"],
        "image_group_id": row["image_group_id"], "fold": row["fold"],
    } for index, row in enumerate(states)])
    scalar_path = output_root / "work/dense_scalar_features.npy"
    scalar = np.asarray([[float(feature_index[str(row["state_id"])]["features"][name]) for name in all_names] for row in states], dtype=np.float32)
    np.save(scalar_path, scalar)

    phase82_layout = read_json(config["sources"]["phase82_dense_text_cache"])
    pooled_spec = phase82_layout["files"]["pooled"]
    pooled = np.memmap(resolve_path(pooled_spec["path"]), dtype=np.uint16, mode="r", shape=tuple(pooled_spec["shape"]))
    fusion_path = output_root / "work/dense_generic_plus_read.npy"
    fusion = np.lib.format.open_memmap(fusion_path, mode="w+", dtype=np.float32, shape=(len(states), int(pooled_spec["shape"][-1]) + len(all_names)))
    for start in range(0, len(states), 128):
        batch = states[start : start + 128]
        parent_indices = [int(row["row_index"]) for row in batch]
        raw = np.array(pooled[parent_indices, 0, :], copy=True)
        generic = torch.from_numpy(raw).view(torch.bfloat16).float().numpy()
        fusion[start : start + len(batch), : generic.shape[1]] = generic
        fusion[start : start + len(batch), generic.shape[1] :] = scalar[start : start + len(batch)]
    fusion.flush()
    del fusion, pooled
    atomic_json(output_root / "work/training_matrix_manifest.json", {
        "contract_sha256": contract["contract_sha256"], "rows": len(states),
        "feature_names": all_names, "feature_groups": schema["groups"],
        "scalar_path": str(scalar_path), "scalar_sha256": file_sha256(scalar_path),
        "fusion_path": str(fusion_path), "fusion_sha256": file_sha256(fusion_path),
        "state_order_sha256": file_sha256(order_path),
    })

    inherited = [
        row for row in read_jsonl(config["split"]["registry"])
        if str(row["uid"]) in {str(state["uid"]) for state in states}
    ]
    atomic_jsonl(output_root / "learnability/inherited_stepB_fold_registry.jsonl", inherited)
    tasks = []
    task_index = 0
    for feature_group in (*schema["groups"].keys(), "F_ALL"):
        for model in ("linear", "mlp"):
            for fold in range(int(config["split"]["folds"])):
                for seed in config["training"]["seeds"]:
                    tasks.append({"task_index": task_index, "family": "dense", "feature_group": feature_group, "model": model, "fold": fold, "seed": seed})
                    task_index += 1
    for model in ("linear", "mlp"):
        for fold in range(int(config["split"]["folds"])):
            for seed in config["training"]["seeds"]:
                tasks.append({"task_index": task_index, "family": "fusion", "feature_group": "FUSION", "model": model, "fold": fold, "seed": seed})
                task_index += 1
    matched_ids = sorted({str(match[key]) for match in primary_matches for key in ("treated_state_id", "control_state_id")})
    atomic_jsonl(output_root / "work/matched_state_order.jsonl", [
        {"state_id": state_id, "label": int(feature_index[state_id]["cohort"] == "read_harmful_flip")}
        for state_id in matched_ids
    ])
    for model in ("linear", "mlp"):
        for fold in range(int(config["split"]["folds"])):
            for seed in config["training"]["seeds"]:
                tasks.append({"task_index": task_index, "family": "matched", "feature_group": "F_ALL", "model": model, "fold": fold, "seed": seed})
                task_index += 1
    atomic_jsonl(output_root / "work/training_tasks.jsonl", tasks)
    print(json.dumps({"training_prepared": True, "dense_rows": len(states), "primary_matched_states": len(matched_ids), "tasks": len(tasks)}, sort_keys=True))


def _role_indices(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], fold: int) -> dict[str, np.ndarray]:
    role_by_uid = {
        str(row["uid"]): str(row["role"])
        for row in read_jsonl(str(config["split"]["inner_roles_pattern"]).format(fold=int(fold)))
    }
    output = {}
    for role in ("fit", "calibration", "outer_test"):
        output[role] = np.asarray([index for index, row in enumerate(rows) if role_by_uid[str(row["uid"])] == role], dtype=np.int64)
        if len(output[role]) == 0:
            raise RuntimeError(f"fold {fold} lacks {role} rows")
    return output


def _matched_role_indices(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], fold: int
) -> dict[str, np.ndarray]:
    """Keep inherited outer folds, but calibrate the sparse probe by group."""

    role_by_uid = {
        str(row["uid"]): str(row["role"])
        for row in read_jsonl(str(config["split"]["inner_roles_pattern"]).format(fold=int(fold)))
    }
    outer_groups = {
        str(row["image_group_id"]) for row in rows
        if role_by_uid[str(row["uid"])] == "outer_test"
    }
    training_groups = sorted({str(row["image_group_id"]) for row in rows} - outer_groups)
    if not outer_groups or len(training_groups) < 3:
        raise RuntimeError(f"matched fold {fold} lacks group support")
    ordered = sorted(
        training_groups,
        key=lambda group: sha256(
            f"{config['matching']['probe_calibration_seed']}|{fold}|{group}".encode()
        ).hexdigest(),
    )
    count = max(2, int(math.ceil(float(config["matching"]["probe_calibration_fraction"]) * len(ordered))))
    count = min(count, len(ordered) - 1)
    calibration_groups = set(ordered[:count])
    fit_groups = set(ordered[count:])
    output = {
        "fit": np.asarray([index for index, row in enumerate(rows) if str(row["image_group_id"]) in fit_groups], dtype=np.int64),
        "calibration": np.asarray([index for index, row in enumerate(rows) if str(row["image_group_id"]) in calibration_groups], dtype=np.int64),
        "outer_test": np.asarray([index for index, row in enumerate(rows) if str(row["image_group_id"]) in outer_groups], dtype=np.int64),
    }
    if min(map(len, output.values())) < 1:
        raise RuntimeError(f"matched fold {fold} has an empty derived role")
    role_groups = {
        role: {str(rows[int(index)]["image_group_id"]) for index in indices}
        for role, indices in output.items()
    }
    if any(role_groups[left] & role_groups[right] for left, right in (("fit", "calibration"), ("fit", "outer_test"), ("calibration", "outer_test"))):
        raise RuntimeError(f"matched fold {fold} has image-group leakage")
    return output


def _training_data(output_root: Path, family: str, group: str) -> tuple[torch.Tensor, list[dict[str, Any]], np.ndarray, bool]:
    states = sorted(read_jsonl(output_root / "population/dense_read_state_manifest.jsonl"), key=lambda row: str(row["state_id"]))
    manifest = read_json(output_root / "work/training_matrix_manifest.json")
    all_names = list(manifest["feature_names"])
    if family == "fusion":
        matrix = np.load(manifest["fusion_path"], mmap_mode="r")
        return torch.from_numpy(matrix), states, np.asarray([float(row["h_r"]) for row in states]), False
    scalar = np.load(manifest["scalar_path"], mmap_mode="r")
    groups = manifest["feature_groups"]
    names = all_names if group == "F_ALL" else list(groups[group])
    indices = [all_names.index(name) for name in names]
    if family == "dense":
        return torch.from_numpy(scalar[:, indices]), states, np.asarray([float(row["h_r"]) for row in states]), False
    matched = read_jsonl(output_root / "work/matched_state_order.jsonl")
    state_index = {str(row["state_id"]): index for index, row in enumerate(states)}
    selected = [state_index[str(row["state_id"])] for row in matched]
    selected_rows = [states[index] for index in selected]
    return torch.from_numpy(scalar[np.ix_(selected, indices)]), selected_rows, np.asarray([int(row["label"]) for row in matched], dtype=np.float64), True


def train_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    tasks = [row for row in read_jsonl(output_root / "work/training_tasks.jsonl") if int(row["task_index"]) % int(config["world_size"]) == int(rank)]
    torch.cuda.set_device(int(rank))
    device = torch.device(f"cuda:{int(rank)}")
    rank_root = output_root / f"work/training/rank{int(rank):02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    started = time.monotonic()
    cache: dict[tuple[str, str], tuple[torch.Tensor, list[dict[str, Any]], np.ndarray, bool]] = {}
    for task in tasks:
        name = f"{task['family']}__{task['feature_group']}__{task['model']}__f{task['fold']}__s{task['seed']}.json"
        path = rank_root / name
        if resume and path.is_file():
            old = read_json(path)
            if old.get("contract_sha256") == contract["contract_sha256"] and old.get("task") == task:
                completed += 1
                continue
        key = str(task["family"]), str(task["feature_group"])
        if key not in cache:
            cache[key] = _training_data(output_root, *key)
        features, rows, target, classification = cache[key]
        roles = (
            _matched_role_indices(config, rows, int(task["fold"]))
            if task["family"] == "matched"
            else _role_indices(config, rows, int(task["fold"]))
        )
        spec = config["training"][str(task["model"])]
        result = fit_predict_fold(
            features, target,
            fit_indices=roles["fit"], calibration_indices=roles["calibration"], test_indices=roles["outer_test"],
            fit_weights=_uid_weights(rows, roles["fit"]), calibration_weights=_uid_weights(rows, roles["calibration"]),
            kind=str(task["model"]), spec=spec, hidden_size=int(config["training"]["mlp"]["hidden_size"]),
            dropout=float(config["training"]["mlp"]["dropout"]), target_scale_floor=float(config["training"]["target_scale_floor"]),
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]), seed=int(task["seed"]), device=device,
            classification=classification,
        )
        test_indices = roles["outer_test"]
        atomic_json(path, {
            "schema_version": "read_harm_training_result_v1", "contract_sha256": contract["contract_sha256"],
            "task": task, "best_epoch": result["best_epoch"], "best_calibration_loss": result["best_calibration_loss"],
            "predictions": [
                {"state_id": rows[int(index)]["state_id"], "uid": rows[int(index)]["uid"], "truth": float(target[int(index)]), "prediction": float(prediction)}
                for index, prediction in zip(test_indices, result["test_prediction"])
            ],
        })
        completed += 1
        if completed % 10 == 0 or completed == len(tasks):
            print(json.dumps({"rank": rank, "completed": completed, "assigned": len(tasks), "elapsed_seconds": time.monotonic() - started}), flush=True)
    atomic_json(rank_root / "complete.json", {"contract_sha256": contract["contract_sha256"], "rank": rank, "expected": len(tasks), "completed": completed})


def _validated_training_results(
    contract: Mapping[str, Any], output_root: Path, task_file: str = "work/training_tasks.jsonl", root_name: str = "work/training"
) -> list[dict[str, Any]]:
    config = contract["static_config"]
    tasks = read_jsonl(output_root / task_file)
    results = []
    for rank in range(int(config["world_size"])):
        root = output_root / f"{root_name}/rank{rank:02d}"
        complete = read_json(root / "complete.json")
        expected = sum(int(task["task_index"]) % int(config["world_size"]) == rank for task in tasks)
        if complete.get("contract_sha256") != contract["contract_sha256"] or int(complete.get("completed", -1)) != expected:
            raise RuntimeError(f"partial training rank: {rank}")
        results.extend(read_json(path) for path in sorted(root.glob("*.json")) if path.name != "complete.json")
    observed = [int(result["task"]["task_index"]) for result in results]
    expected_indices = [int(task["task_index"]) for task in tasks]
    if len(observed) != len(set(observed)) or sorted(observed) != sorted(expected_indices):
        raise RuntimeError("training result census differs")
    return results


def _ensemble(
    results: Sequence[Mapping[str, Any]], *, family: str, feature_group: str, model: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    selected = [
        result for result in results
        if result["task"]["family"] == family
        and result["task"]["feature_group"] == feature_group
        and result["task"]["model"] == model
    ]
    for result in selected:
        for row in result["predictions"]:
            grouped[str(row["state_id"])].append(float(row["prediction"]))
    return {state_id: float(np.mean(values)) for state_id, values in grouped.items()}


def _baseline_predictions(
    config: Mapping[str, Any], states: Sequence[Mapping[str, Any]], model: str, source: str
) -> dict[str, float]:
    expected = {str(row["state_id"]): float(row["h_r"]) for row in states}
    output = {}
    for row in read_jsonl(config["sources"][source]):
        if str(row.get("model")) != model:
            continue
        state_id = str(row["state_id"])
        if state_id not in expected:
            continue
        if not math.isclose(-float(row["truth"]), expected[state_id], abs_tol=1e-9):
            raise RuntimeError(f"baseline target differs: {state_id}")
        output[state_id] = -float(row["prediction"])
    if set(output) != set(expected):
        raise RuntimeError(f"baseline {model} census differs")
    return output


def _metrics_for_prediction(
    states: Sequence[Mapping[str, Any]], prediction: Mapping[str, float]
) -> dict[str, Any]:
    expected = {str(row["state_id"]) for row in states}
    if not expected.issubset(prediction):
        raise RuntimeError("prediction census differs")
    truth = np.asarray([float(row["h_r"]) for row in states])
    score = np.asarray([float(prediction[str(row["state_id"])]) for row in states])
    return _metric_bundle(truth, score)


def _breakdown(
    states: Sequence[Mapping[str, Any]], prediction: Mapping[str, float], *, key: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in states:
        grouped[str(row[key])].append(row)
    output = []
    for value, rows in sorted(grouped.items()):
        truth = np.asarray([float(row["h_r"]) for row in rows])
        score = np.asarray([float(prediction[str(row["state_id"])]) for row in rows])
        output.append({key: value, **_metric_bundle(truth, score)})
    return output


def aggregate_dense(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    results = _validated_training_results(contract, output_root)
    states = sorted(read_jsonl(output_root / "population/dense_read_state_manifest.jsonl"), key=lambda row: str(row["state_id"]))
    schema = read_json(output_root / "features/feature_group_manifest.json")
    metric_rows = []
    predictions: dict[tuple[str, str], dict[str, float]] = {}
    for group in (*schema["groups"].keys(), "F_ALL"):
        for model in ("linear", "mlp"):
            current = _ensemble(results, family="dense", feature_group=group, model=model)
            if len(current) != len(states):
                raise RuntimeError(f"OOF prediction census differs: {group}/{model}")
            predictions[(group, model)] = current
            metric_rows.append({"feature_group": group, "model": model, **_metrics_for_prediction(states, current)})
    linear = [row for row in metric_rows if row["model"] == "linear"]
    mlp = [row for row in metric_rows if row["model"] == "mlp"]
    atomic_csv(output_root / "learnability/read_linear_metrics.csv", linear)
    atomic_csv(output_root / "learnability/read_mlp_metrics.csv", mlp)
    atomic_csv(output_root / "learnability/feature_group_ablation.csv", metric_rows)

    fusion_rows = []
    for model in ("linear", "mlp"):
        current = _ensemble(results, family="fusion", feature_group="FUSION", model=model)
        if len(current) != len(states):
            raise RuntimeError(f"fusion OOF census differs: {model}")
        fusion_rows.append({"feature_group": "generic_prestate_plus_F_ALL", "model": model, **_metrics_for_prediction(states, current)})
    atomic_csv(output_root / "learnability/generic_plus_read_features.csv", fusion_rows)

    baseline_specs = (
        ("B0_nuisance", "m0_nuisance", "generic_oof", "nuisance_baseline.csv"),
        ("B1_stepB_generic", "m3_z_RW", "generic_oof", "generic_prestate_baseline.csv"),
        ("B2_one_step_delta", "mlp", "one_step_oof", "one_step_delta_baseline.csv"),
    )
    baselines = {}
    for name, model, source, filename in baseline_specs:
        current = _baseline_predictions(config, states, model, source)
        baselines[name] = current
        atomic_csv(output_root / f"learnability/{filename}", [{"baseline": name, **_metrics_for_prediction(states, current)}])

    candidates = [
        {"feature_group": row["feature_group"], "model": row["model"], "spearman": row["spearman"], "harmful_auroc": row["harmful_auroc"]}
        for row in metric_rows
    ]
    winner = select_dense_winner(candidates)
    winner_prediction = predictions[(str(winner["feature_group"]), str(winner["model"]))]
    winner.update({
        "contract_sha256": contract["contract_sha256"],
        "selection_rule": config["winner_selection"],
        "metrics": _metrics_for_prediction(states, winner_prediction),
        "generic_baseline": _metrics_for_prediction(states, baselines["B1_stepB_generic"]),
        "one_step_baseline": _metrics_for_prediction(states, baselines["B2_one_step_delta"]),
    })
    atomic_json(output_root / "learnability/dense_winner.json", winner)
    atomic_jsonl(output_root / "learnability/winner_oof_predictions.jsonl", [
        {**row, "prediction": winner_prediction[str(row["state_id"])]} for row in states
    ])

    dense_w = [row for row in states if bool(row["dense_wrong"])]
    atomic_csv(output_root / "learnability/dense_w_only_metrics.csv", [{
        "feature_group": winner["feature_group"], "model": winner["model"], **_metrics_for_prediction(dense_w, winner_prediction)
    }])
    labels = np.asarray([int(row["cohort"] == "read_harmful_flip") for row in states])
    scores = np.asarray([winner_prediction[str(row["state_id"])] for row in states])
    flip_metrics = binary_classification_metrics(truth=labels, prediction=scores)
    category_rows = []
    for cohort in ("read_harmful_flip", "read_beneficial_flip", "stable_wrong", "stable_correct"):
        values = [winner_prediction[str(row["state_id"])] for row in states if row["cohort"] == cohort]
        category_rows.append({"cohort": cohort, **_summary(values)})
    atomic_csv(output_root / "learnability/strong_flip_ranking.csv", [{"scope": "harmful_flip_vs_all", **flip_metrics}] + category_rows)

    layer_rows = _breakdown(states, winner_prediction, key="layer")
    atomic_csv(output_root / "learnability/layer_breakdown.csv", layer_rows)
    relative_states = []
    for row in states:
        copy = dict(row)
        copy["trigger_relative_bin"] = _depth_bin(int(row["trigger_relative_depth"]), config["structure"]["trigger_relative_bins"])
        relative_states.append(copy)
    atomic_csv(output_root / "learnability/trigger_relative_breakdown.csv", _breakdown(relative_states, winner_prediction, key="trigger_relative_bin"))
    source_states = []
    for row in states:
        copy = dict(row)
        copy["dataset_source"] = f"{row['dataset']}|{row['source_regime']}"
        source_states.append(copy)
    atomic_csv(output_root / "learnability/dataset_source_breakdown.csv", _breakdown(source_states, winner_prediction, key="dataset_source"))

    matched_rows = []
    matched_order = read_jsonl(output_root / "work/matched_state_order.jsonl")
    matched_truth = {str(row["state_id"]): int(row["label"]) for row in matched_order}
    for model in ("linear", "mlp"):
        current = _ensemble(results, family="matched", feature_group="F_ALL", model=model)
        if set(current) != set(matched_truth):
            raise RuntimeError(f"matched OOF census differs: {model}")
        truth = [matched_truth[state_id] for state_id in current]
        score = [current[state_id] for state_id in current]
        metrics = binary_classification_metrics(truth=truth, prediction=score)
        order = np.argsort(-np.asarray(score), kind="mergesort")
        count = max(1, math.ceil(0.1 * len(order)))
        metrics["precision_at_top10pct"] = float(np.asarray(truth)[order[:count]].mean())
        matched_rows.append({"model": model, **metrics})
    atomic_csv(output_root / "matched_mechanism/matched_probe_metrics.csv", matched_rows)

    similarity = {str(row["uid"]): row for row in read_jsonl(config["generalization"]["similarity"])}
    similarity_rows = []
    for bin_name in ("Q1", "Q5"):
        subset = [row for row in states if similarity[str(row["uid"])]["similarity_bin"] == bin_name]
        similarity_rows.append({"similarity_bin": bin_name, **_metrics_for_prediction(subset, winner_prediction)})
    atomic_csv(output_root / "generalization/semantic_similarity_metrics.csv", similarity_rows)
    print(json.dumps({"dense_aggregation_complete": True, "winner": winner["feature_group"], "model": winner["model"], "spearman": winner["metrics"]["spearman"]}, sort_keys=True))


def prepare_generalization(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    winner = read_json(output_root / "learnability/dense_winner.json")
    tasks = []
    index = 0
    for fold in range(5):
        for seed in config["training"]["seeds"]:
            tasks.append({"task_index": index, "family": "cluster_ood", "name": f"cluster_fold_{fold}", "fold": fold, "seed": seed})
            index += 1
    for train, test in (("historical", "canonical"), ("canonical", "historical")):
        for seed in config["training"]["seeds"]:
            tasks.append({"task_index": index, "family": "source", "name": f"{train}_to_{test}", "train_value": train, "test_value": test, "seed": seed})
            index += 1
    for held_out in ("gqa", "chartqa", "textvqa"):
        for seed in config["training"]["seeds"]:
            tasks.append({"task_index": index, "family": "lodo", "name": f"held_out_{held_out}", "test_value": held_out, "seed": seed})
            index += 1
    atomic_jsonl(output_root / "work/generalization_tasks.jsonl", tasks)
    atomic_json(output_root / "work/generalization_contract.json", {
        "contract_sha256": contract["contract_sha256"], "winner": winner,
        "tasks": len(tasks), "task_sha256": file_sha256(output_root / "work/generalization_tasks.jsonl"),
    })
    print(json.dumps({"generalization_prepared": True, "tasks": len(tasks), "winner": winner["feature_group"], "model": winner["model"]}, sort_keys=True))


def _holdout_roles(
    task: Mapping[str, Any], states: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    family = str(task["family"])
    if family == "cluster_ood":
        cluster = {str(row["uid"]): int(row["cluster_fold"]) for row in read_jsonl(config["generalization"]["cluster_registry"])}
        test_candidate = np.asarray([cluster[str(row["uid"])] == int(task["fold"]) for row in states])
    elif family == "source":
        test_candidate = np.asarray([str(row["source_regime"]) == str(task["test_value"]) for row in states])
    elif family == "lodo":
        test_candidate = np.asarray([str(row["dataset"]) == str(task["test_value"]) for row in states])
    else:
        raise ValueError(f"unknown transfer family: {family}")
    if family == "source":
        train_candidate = np.asarray([str(row["source_regime"]) == str(task["train_value"]) for row in states])
    else:
        train_candidate = ~test_candidate
    test_groups = {str(row["image_group_id"]) for row, selected in zip(states, test_candidate) if selected}
    test = np.asarray([
        index for index, row in enumerate(states)
        if bool(test_candidate[index]) and str(row["image_group_id"]) in test_groups
    ], dtype=np.int64)
    training_groups = sorted({
        str(row["image_group_id"]) for index, row in enumerate(states)
        if bool(train_candidate[index]) and str(row["image_group_id"]) not in test_groups
    })
    if len(test) < int(config["generalization"]["minimum_test_states"]) or not training_groups:
        raise RuntimeError(f"unsupported transfer split: {task['name']}")
    calibration_count = max(1, int(math.ceil(0.1 * len(training_groups))))
    ordered = sorted(training_groups, key=lambda group: sha256(f"{config['seed']}|{task['name']}|{group}".encode()).hexdigest())
    calibration_groups = set(ordered[:calibration_count])
    fit = np.asarray([index for index, row in enumerate(states) if str(row["image_group_id"]) in set(training_groups) - calibration_groups], dtype=np.int64)
    calibration = np.asarray([index for index, row in enumerate(states) if str(row["image_group_id"]) in calibration_groups], dtype=np.int64)
    if min(len(fit), len(calibration), len(test)) < 1:
        raise RuntimeError(f"empty transfer role: {task['name']}")
    return {"fit": fit, "calibration": calibration, "outer_test": test}


def generalization_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    frozen = read_json(output_root / "work/generalization_contract.json")
    if frozen.get("contract_sha256") != contract["contract_sha256"] or frozen.get("task_sha256") != file_sha256(output_root / "work/generalization_tasks.jsonl"):
        raise RuntimeError("generalization task contract differs")
    winner = frozen["winner"]
    tasks = [task for task in read_jsonl(output_root / "work/generalization_tasks.jsonl") if int(task["task_index"]) % int(config["world_size"]) == int(rank)]
    features, states, target, _ = _training_data(output_root, "dense", str(winner["feature_group"]))
    torch.cuda.set_device(int(rank))
    device = torch.device(f"cuda:{int(rank)}")
    rank_root = output_root / f"work/generalization/rank{int(rank):02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for task in tasks:
        path = rank_root / f"{task['name']}__s{task['seed']}.json"
        if resume and path.is_file():
            old = read_json(path)
            if old.get("contract_sha256") == contract["contract_sha256"] and old.get("task") == task:
                completed += 1
                continue
        roles = _holdout_roles(task, states, config)
        model = str(winner["model"])
        spec = config["training"][model]
        result = fit_predict_fold(
            features, target, fit_indices=roles["fit"], calibration_indices=roles["calibration"], test_indices=roles["outer_test"],
            fit_weights=_uid_weights(states, roles["fit"]), calibration_weights=_uid_weights(states, roles["calibration"]),
            kind=model, spec=spec, hidden_size=int(config["training"]["mlp"]["hidden_size"]), dropout=float(config["training"]["mlp"]["dropout"]),
            target_scale_floor=float(config["training"]["target_scale_floor"]), gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            seed=int(task["seed"]), device=device, classification=False,
        )
        atomic_json(path, {
            "schema_version": "read_harm_generalization_result_v1", "contract_sha256": contract["contract_sha256"], "task": task,
            "best_epoch": result["best_epoch"], "best_calibration_loss": result["best_calibration_loss"],
            "predictions": [{"state_id": states[int(index)]["state_id"], "uid": states[int(index)]["uid"], "truth": float(target[int(index)]), "prediction": float(prediction)} for index, prediction in zip(roles["outer_test"], result["test_prediction"])],
        })
        completed += 1
    atomic_json(rank_root / "complete.json", {"contract_sha256": contract["contract_sha256"], "rank": rank, "expected": len(tasks), "completed": completed})


def _bootstrap_difference(
    states: Sequence[Mapping[str, Any]], truth: np.ndarray, left: np.ndarray, right: np.ndarray,
    *, draws: int, seed: int, metric: str
) -> dict[str, float]:
    groups = np.asarray([str(row["image_group_id"]) for row in states])
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        selected = np.concatenate([indices[group] for group in rng.choice(unique, size=len(unique), replace=True)])
        if metric == "spearman":
            left_value = float(regression_metrics(truth=truth[selected], prediction=left[selected])["spearman"])
            right_value = float(regression_metrics(truth=truth[selected], prediction=right[selected])["spearman"])
        else:
            mask = truth[selected] != 0.0
            labels = (truth[selected][mask] > 0).astype(int)
            left_value = float(binary_classification_metrics(truth=labels, prediction=left[selected][mask])["auroc"])
            right_value = float(binary_classification_metrics(truth=labels, prediction=right[selected][mask])["auroc"])
        values[draw] = left_value - right_value
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975))}


def aggregate_generalization(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    tasks = read_jsonl(output_root / "work/generalization_tasks.jsonl")
    results = []
    for rank in range(int(config["world_size"])):
        root = output_root / f"work/generalization/rank{rank:02d}"
        complete = read_json(root / "complete.json")
        expected = sum(int(task["task_index"]) % int(config["world_size"]) == rank for task in tasks)
        if complete.get("contract_sha256") != contract["contract_sha256"] or int(complete.get("completed", -1)) != expected:
            raise RuntimeError(f"partial generalization rank: {rank}")
        results.extend(read_json(path) for path in sorted(root.glob("*.json")) if path.name != "complete.json")
    if len(results) != len(tasks):
        raise RuntimeError("generalization result census differs")
    by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        by_name[str(result["task"]["name"])].append(result)
    metric_rows = []
    for name, seed_results in sorted(by_name.items()):
        predictions: dict[str, list[float]] = defaultdict(list)
        truth = {}
        family = str(seed_results[0]["task"]["family"])
        for result in seed_results:
            for row in result["predictions"]:
                predictions[str(row["state_id"])].append(float(row["prediction"]))
                truth[str(row["state_id"])] = float(row["truth"])
        ids = sorted(predictions)
        metric_rows.append({"family": family, "split": name, **_metric_bundle(np.asarray([truth[state_id] for state_id in ids]), np.asarray([np.mean(predictions[state_id]) for state_id in ids]))})
    atomic_csv(output_root / "generalization/question_cluster_ood.csv", [row for row in metric_rows if row["family"] == "cluster_ood"])
    atomic_csv(output_root / "generalization/historical_to_canonical.csv", [row for row in metric_rows if row["split"] == "historical_to_canonical"])
    atomic_csv(output_root / "generalization/canonical_to_historical.csv", [row for row in metric_rows if row["split"] == "canonical_to_historical"])
    atomic_csv(output_root / "generalization/dataset_lodo.csv", [row for row in metric_rows if row["family"] == "lodo"])

    states = sorted(read_jsonl(output_root / "population/dense_read_state_manifest.jsonl"), key=lambda row: str(row["state_id"]))
    winner_rows = read_jsonl(output_root / "learnability/winner_oof_predictions.jsonl")
    winner = np.asarray([float(row["prediction"]) for row in winner_rows])
    truth = np.asarray([float(row["h_r"]) for row in states])
    baseline_map = _baseline_predictions(config, states, "m3_z_RW", "generic_oof")
    baseline = np.asarray([baseline_map[str(row["state_id"])] for row in states])
    draws = int(config["evaluation"]["bootstrap_draws"])
    spearman = _bootstrap_difference(states, truth, winner, baseline, draws=draws, seed=int(config["evaluation"]["bootstrap_seed"]), metric="spearman")
    auroc = _bootstrap_difference(states, truth, winner, baseline, draws=draws, seed=int(config["evaluation"]["bootstrap_seed"]) + 1, metric="auroc")
    difference_rows = [
        {"comparison": "winner_minus_stepB_generic", "metric": "spearman", **spearman},
        {"comparison": "winner_minus_stepB_generic", "metric": "harmful_auroc", **auroc},
    ]
    atomic_csv(output_root / "statistics/group_bootstrap_ci.csv", difference_rows)
    atomic_csv(output_root / "statistics/pairwise_model_differences.csv", difference_rows)
    winner_metrics = read_json(output_root / "learnability/dense_winner.json")["metrics"]
    baseline_metrics = _metrics_for_prediction(states, baseline_map)
    delta_s = float(winner_metrics["spearman"]) - float(baseline_metrics["spearman"])
    delta_a = float(winner_metrics["harmful_auroc"]) - float(baseline_metrics["harmful_auroc"])
    gate = ((delta_s >= float(config["external_transfer_gate"]["spearman_improvement"]) and spearman["ci_low"] > 0.0) or (delta_a >= float(config["external_transfer_gate"]["harmful_auroc_improvement"]) and auroc["ci_low"] > 0.0))
    text = f"""# External-transfer gate\n\n- Winner-minus-generic Spearman: `{delta_s:.6f}`; bootstrap 95% CI `[{spearman['ci_low']:.6f}, {spearman['ci_high']:.6f}]`.\n- Winner-minus-generic harmful AUROC: `{delta_a:.6f}`; bootstrap 95% CI `[{auroc['ci_low']:.6f}, {auroc['ci_high']:.6f}]`.\n- Frozen gate passed: **{str(gate).upper()}**.\n- No external counterfactual measurement was launched.\n"""
    (output_root / "generalization/external_transfer_gate.md").write_text(text)
    atomic_json(output_root / "generalization/external_transfer_gate.json", {"contract_sha256": contract["contract_sha256"], "passed": gate, "spearman_delta": delta_s, "harmful_auroc_delta": delta_a, "spearman_bootstrap": spearman, "harmful_auroc_bootstrap": auroc})
    print(json.dumps({"generalization_complete": True, "external_transfer_gate_passed": gate}, sort_keys=True))


def prepare_routed_training(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    completion = read_json(output_root / "routed_secondary/feature_completion.json")
    if completion.get("contract_sha256") != contract["contract_sha256"] or not completion.get("complete"):
        raise RuntimeError("routed READ features are incomplete")
    winner = read_json(output_root / "learnability/dense_winner.json")
    group = str(winner["feature_group"])
    schema = read_json(output_root / "features/feature_group_manifest.json")
    all_names = list(schema["F_ALL"])
    names = all_names if group == "F_ALL" else list(schema["groups"][group])
    matrices: dict[str, Any] = {}
    for domain, feature_file in (
        ("dense", output_root / "features/read_operation_features.jsonl"),
        ("routed", output_root / "routed_secondary/read_operation_features.jsonl"),
    ):
        states = sorted(read_jsonl(output_root / f"population/{domain}_read_state_manifest.jsonl"), key=lambda row: str(row["state_id"]))
        features = read_jsonl(feature_file)
        validate_feature_census([str(row["state_id"]) for row in states], features)
        by_id = {str(row["state_id"]): row for row in features}
        matrix = np.asarray([
            [float(by_id[str(row["state_id"])]["features"][name]) for name in names]
            for row in states
        ], dtype=np.float32)
        path = output_root / f"work/{domain}_routed_selected_features.npy"
        np.save(path, matrix)
        order = output_root / f"work/{domain}_routed_state_order.jsonl"
        atomic_jsonl(order, [
            {"row_index": index, "state_id": row["state_id"], "uid": row["uid"],
             "image_group_id": row["image_group_id"], "fold": row["fold"]}
            for index, row in enumerate(states)
        ])
        matrices[domain] = {
            "rows": len(states), "path": str(path), "sha256": file_sha256(path),
            "state_order": str(order), "state_order_sha256": file_sha256(order),
        }
    tasks = []
    index = 0
    for family in ("routed_oof", "dense_to_routed"):
        for fold in range(int(config["split"]["folds"])):
            for seed in config["training"]["seeds"]:
                tasks.append({
                    "task_index": index, "family": family, "feature_group": group,
                    "model": winner["model"], "fold": fold, "seed": seed,
                })
                index += 1
    task_path = output_root / "work/routed_training_tasks.jsonl"
    atomic_jsonl(task_path, tasks)
    atomic_json(output_root / "work/routed_training_manifest.json", {
        "contract_sha256": contract["contract_sha256"], "winner": winner,
        "feature_names": names, "matrices": matrices, "tasks": len(tasks),
        "task_sha256": file_sha256(task_path),
    })
    print(json.dumps({"routed_training_prepared": True, "feature_group": group, "model": winner["model"], "tasks": len(tasks)}, sort_keys=True))


def _routed_training_data(output_root: Path) -> tuple[torch.Tensor, list[dict[str, Any]], np.ndarray, torch.Tensor, list[dict[str, Any]], np.ndarray]:
    manifest = read_json(output_root / "work/routed_training_manifest.json")
    values = []
    for domain in ("dense", "routed"):
        spec = manifest["matrices"][domain]
        if file_sha256(spec["path"]) != spec["sha256"] or file_sha256(spec["state_order"]) != spec["state_order_sha256"]:
            raise RuntimeError(f"routed training matrix provenance differs: {domain}")
        matrix = torch.from_numpy(np.load(spec["path"], mmap_mode="r"))
        states = sorted(read_jsonl(output_root / f"population/{domain}_read_state_manifest.jsonl"), key=lambda row: str(row["state_id"]))
        if matrix.shape[0] != len(states):
            raise RuntimeError(f"routed training row census differs: {domain}")
        target = np.asarray([float(row["h_r"]) for row in states], dtype=np.float64)
        values.extend((matrix, states, target))
    return tuple(values)  # type: ignore[return-value]


def routed_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    frozen = read_json(output_root / "work/routed_training_manifest.json")
    task_path = output_root / "work/routed_training_tasks.jsonl"
    if frozen.get("contract_sha256") != contract["contract_sha256"] or frozen.get("task_sha256") != file_sha256(task_path):
        raise RuntimeError("routed training contract differs")
    tasks = [task for task in read_jsonl(task_path) if int(task["task_index"]) % int(config["world_size"]) == int(rank)]
    dense_x, dense_rows, dense_y, routed_x, routed_rows, routed_y = _routed_training_data(output_root)
    combined_x = torch.cat((dense_x, routed_x), dim=0)
    combined_y = np.concatenate((dense_y, routed_y))
    combined_rows = dense_rows + routed_rows
    torch.cuda.set_device(int(rank))
    device = torch.device(f"cuda:{int(rank)}")
    rank_root = output_root / f"work/routed_training/rank{int(rank):02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for task in tasks:
        path = rank_root / f"{task['family']}__f{task['fold']}__s{task['seed']}.json"
        if resume and path.is_file():
            old = read_json(path)
            if old.get("contract_sha256") == contract["contract_sha256"] and old.get("task") == task:
                completed += 1
                continue
        fold = int(task["fold"])
        if task["family"] == "routed_oof":
            roles = _role_indices(config, routed_rows, fold)
            features, target, rows = routed_x, routed_y, routed_rows
            fit_indices, calibration_indices, test_indices = roles["fit"], roles["calibration"], roles["outer_test"]
        else:
            dense_roles = _role_indices(config, dense_rows, fold)
            routed_roles = _role_indices(config, routed_rows, fold)
            features, target, rows = combined_x, combined_y, combined_rows
            fit_indices = dense_roles["fit"]
            calibration_indices = dense_roles["calibration"]
            test_indices = routed_roles["outer_test"] + len(dense_rows)
        spec = config["training"][str(task["model"])]
        result = fit_predict_fold(
            features, target, fit_indices=fit_indices, calibration_indices=calibration_indices,
            test_indices=test_indices, fit_weights=_uid_weights(rows, fit_indices),
            calibration_weights=_uid_weights(rows, calibration_indices), kind=str(task["model"]),
            spec=spec, hidden_size=int(config["training"]["mlp"]["hidden_size"]),
            dropout=float(config["training"]["mlp"]["dropout"]),
            target_scale_floor=float(config["training"]["target_scale_floor"]),
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            seed=int(task["seed"]), device=device, classification=False,
        )
        atomic_json(path, {
            "schema_version": "read_harm_routed_training_result_v1",
            "contract_sha256": contract["contract_sha256"], "task": task,
            "best_epoch": result["best_epoch"], "best_calibration_loss": result["best_calibration_loss"],
            "predictions": [
                {"state_id": rows[int(index)]["state_id"], "uid": rows[int(index)]["uid"],
                 "truth": float(target[int(index)]), "prediction": float(prediction)}
                for index, prediction in zip(test_indices, result["test_prediction"])
            ],
        })
        completed += 1
    atomic_json(rank_root / "complete.json", {
        "contract_sha256": contract["contract_sha256"], "rank": rank,
        "expected": len(tasks), "completed": completed,
    })


def aggregate_routed(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    results = _validated_training_results(
        contract, output_root, task_file="work/routed_training_tasks.jsonl",
        root_name="work/routed_training",
    )
    routed = sorted(read_jsonl(output_root / "population/routed_read_state_manifest.jsonl"), key=lambda row: str(row["state_id"]))
    winner = read_json(output_root / "learnability/dense_winner.json")
    rows = []
    for family, destination in (
        ("routed_oof", "learnability_metrics.csv"),
        ("dense_to_routed", "dense_to_routed_transfer.csv"),
    ):
        prediction = _ensemble(
            results, family=family, feature_group=str(winner["feature_group"]),
            model=str(winner["model"]),
        )
        if set(prediction) != {str(row["state_id"]) for row in routed}:
            raise RuntimeError(f"routed prediction census differs: {family}")
        result = {
            "family": family, "selection_qualified": True,
            "feature_group": winner["feature_group"], "model": winner["model"],
            **_metrics_for_prediction(routed, prediction),
        }
        rows.append(result)
        atomic_csv(output_root / f"routed_secondary/{destination}", [result])
    print(json.dumps({"routed_aggregation_complete": True, "routed_oof_spearman": rows[0]["spearman"], "dense_to_routed_spearman": rows[1]["spearman"]}, sort_keys=True))


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def _make_figures(output_root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    maps = read_jsonl(output_root / "structure/per_uid_read_harm_maps.jsonl")
    examples = sorted(maps, key=lambda row: max(abs(float(state["h_r"])) for state in row["states"]), reverse=True)[:6]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in examples:
        ax.plot([state["layer"] for state in row["states"]], [state["h_r"] for state in row["states"]], marker="o", label=str(row["uid"])[:12])
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Layer", ylabel="H_R"); ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(figure_root / "read_harm_map_examples.png", dpi=180); plt.close(fig)

    def line_plot(rows: Sequence[Mapping[str, Any]], xkey: str, filename: str, xlabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(rows)) if xkey != "layer" else [int(row[xkey]) for row in rows]
        labels = [str(row[xkey]) for row in rows]
        ax.plot(x, [_float(row, "mean") for row in rows], marker="o")
        ax.fill_between(x, [_float(row, "q25") for row in rows], [_float(row, "q75") for row in rows], alpha=.2)
        if xkey != "layer": ax.set_xticks(x, labels, rotation=20)
        ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel=xlabel, ylabel="H_R mean / IQR"); fig.tight_layout()
        fig.savefig(figure_root / filename, dpi=180); plt.close(fig)
    line_plot(read_csv(output_root / "structure/layer_summary.csv"), "layer", "read_harm_by_layer.png", "Layer")
    line_plot(read_csv(output_root / "structure/trigger_relative_summary.csv"), "trigger_relative_bin", "read_harm_by_trigger_depth.png", "Trigger-relative depth")

    spans = read_csv(output_root / "structure/harmful_span_statistics.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for sign in ("harmful", "beneficial"):
        selected = [row for row in spans if row["sign"] == sign]
        ax.plot([int(row["span_length"]) for row in selected], [float(row["fraction_of_spans"]) for row in selected], marker="o", label=sign)
    ax.set(xlabel="Contiguous span length", ylabel="Fraction of spans"); ax.legend(); fig.tight_layout(); fig.savefig(figure_root / "harmful_span_length.png", dpi=180); plt.close(fig)
    neighborhood = read_csv(output_root / "structure/strong_flip_neighborhood.csv")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot([int(row["offset"]) for row in neighborhood], [float(row["harmful_prevalence"]) for row in neighborhood], marker="o")
    ax.set(xlabel="Layer offset from harmful flip", ylabel="Harmful prevalence"); fig.tight_layout(); fig.savefig(figure_root / "strong_flip_neighborhood.png", dpi=180); plt.close(fig)

    features = read_jsonl(output_root / "features/read_operation_features.jsonl")
    def cohort_box(feature: str, filename: str, ylabel: str) -> None:
        values = [[float(row["features"][feature]) for row in features if row["cohort"] == cohort] for cohort in ("read_harmful_flip", "read_beneficial_flip")]
        fig, ax = plt.subplots(figsize=(5, 4)); ax.boxplot(values, tick_labels=["harmful", "beneficial"], showfliers=False); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(figure_root / filename, dpi=180); plt.close(fig)
    cohort_box("f4_visual_attention_entropy", "read_attention_entropy_harmful_vs_beneficial.png", "Visual attention entropy")
    cohort_box("f1_read_update_norm", "read_update_norm_harmful_vs_beneficial.png", "READ update norm")
    effects = [row for row in read_csv(output_root / "matched_mechanism/feature_effect_sizes.csv") if row["comparison"] == "harmful_vs_beneficial"]
    effects = sorted(effects, key=lambda row: abs(float(row["standardized_effect_size"])), reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.barh([row["feature"] for row in reversed(effects)], [float(row["standardized_effect_size"]) for row in reversed(effects)]); ax.axvline(0, color="black", linewidth=.8); ax.set_xlabel("Matched standardized effect"); fig.tight_layout(); fig.savefig(figure_root / "read_feature_effect_sizes.png", dpi=180); plt.close(fig)

    metrics = read_csv(output_root / "learnability/feature_group_ablation.csv")
    baselines = [read_csv(output_root / f"learnability/{name}.csv")[0] for name in ("nuisance_baseline", "generic_prestate_baseline", "one_step_delta_baseline")]
    names = [f"{row['feature_group']}-{row['model']}" for row in metrics] + [row["baseline"] for row in baselines]
    values = [float(row["spearman"]) for row in metrics] + [float(row["spearman"]) for row in baselines]
    order = np.argsort(values)[-12:]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh([names[index] for index in order], [values[index] for index in order]); ax.set_xlabel("OOF Spearman"); fig.tight_layout(); fig.savefig(figure_root / "read_learnability_comparison.png", dpi=180); plt.close(fig)

    winner_rows = read_jsonl(output_root / "learnability/winner_oof_predictions.jsonl")
    ordered = sorted(winner_rows, key=lambda row: float(row["prediction"]), reverse=True)
    labels = np.asarray([float(row["h_r"]) > 0 for row in ordered], dtype=float)
    coverage = np.arange(1, len(labels) + 1) / len(labels)
    precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(coverage, precision); ax.axhline(labels.mean(), color="black", linestyle="--", label="prevalence"); ax.set(xlabel="Selected coverage", ylabel="Harmful precision", xlim=(0, .5)); ax.legend(); fig.tight_layout(); fig.savefig(figure_root / "read_high_precision_harmful_subset.png", dpi=180); plt.close(fig)

    ladder = []
    winner = read_json(output_root / "learnability/dense_winner.json")
    ladder.append(("ID", float(winner["metrics"]["spearman"])))
    for filename, label in (("question_cluster_ood.csv", "Cluster OOD"), ("historical_to_canonical.csv", "H→C"), ("canonical_to_historical.csv", "C→H")):
        rows = read_csv(output_root / f"generalization/{filename}")
        ladder.append((label, float(np.mean([float(row["spearman"]) for row in rows]))))
    rows = read_csv(output_root / "generalization/dataset_lodo.csv")
    ladder.append(("LODO", float(np.mean([float(row["spearman"]) for row in rows]))))
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar([name for name, _ in ladder], [value for _, value in ladder]); ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("Spearman"); fig.tight_layout(); fig.savefig(figure_root / "read_generalization_ladder.png", dpi=180); plt.close(fig)


def finalize(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    dense = read_jsonl(output_root / "population/dense_read_state_manifest.jsonl")
    counts = Counter(str(row["read_sign"]) for row in dense)
    cohorts = Counter(str(row["cohort"]) for row in dense)
    persistence_rows = {row["domain"]: row for row in read_csv(output_root / "structure/adjacent_persistence.csv")}
    null_rows = {
        (row["domain"], row["metric"]): row
        for row in read_csv(output_root / "structure/shuffled_null_statistics.csv")
    }
    dense_persistence = persistence_rows["dense"]
    structure_rule = config["decision_rules"]["structure"]
    persistence_pass = float(dense_persistence["harmful_persistence"]) > float(null_rows[("dense", "harmful_persistence")]["null_q975"])
    span_pass = float(dense_persistence["harmful_span_mean"]) > float(null_rows[("dense", "harmful_span_mean")]["null_q975"])
    neighborhoods = read_csv(output_root / "structure/strong_flip_neighborhood.csv")
    population_harmful = counts["harmful"] / len(dense)
    neighbor_values = [float(row["harmful_prevalence"]) for row in neighborhoods if int(row["offset"]) in (-1, 1)]
    neighborhood_pass = bool(neighbor_values) and float(np.mean(neighbor_values)) > population_harmful
    structure_case = "R-STRUCT-A" if persistence_pass and span_pass and neighborhood_pass else "R-STRUCT-B"

    effect_rows = read_csv(output_root / "matched_mechanism/feature_effect_sizes.csv")
    pooled = [row for row in effect_rows if row["comparison"] == "harmful_vs_beneficial"]
    significant = [row for row in pooled if float(row["bootstrap_ci_low"]) > 0 or float(row["bootstrap_ci_high"]) < 0]
    largest = max(pooled, key=lambda row: abs(float(row["standardized_effect_size"])))
    strata = [row for row in effect_rows if row["comparison"].startswith("harmful_vs_beneficial|") and row["feature"] == largest["feature"]]
    direction = np.sign(float(largest["matched_mean_difference"]))
    consistent = sum(np.sign(float(row["matched_mean_difference"])) == direction for row in strata) / len(strata) if strata else 0.0
    probe_rows = read_csv(output_root / "matched_mechanism/matched_probe_metrics.csv")
    probe_best = max(float(row["auroc"]) for row in probe_rows)
    mech_rules = config["decision_rules"]["mechanism"]
    mechanism_case = "R-MECH-A" if significant and probe_best >= float(mech_rules["minimum_matched_probe_auroc"]) and consistent >= float(mech_rules["minimum_sign_consistent_dataset_source_fraction"]) else "R-MECH-B"

    winner = read_json(output_root / "learnability/dense_winner.json")
    dense_w = read_csv(output_root / "learnability/dense_w_only_metrics.csv")[0]
    gate = read_json(output_root / "generalization/external_transfer_gate.json")
    learn_rules = config["decision_rules"]["learnability"]
    metrics = winner["metrics"]
    high_precision = float(metrics["precision_at_0.1"]) - float(metrics["harmful_prevalence"]) >= float(learn_rules["minimum_top10_precision_gain_over_prevalence"])
    learn_a = bool(gate["passed"]) and high_precision and float(dense_w["spearman"]) >= float(learn_rules["minimum_dense_w_spearman"])
    restricted = float(metrics["spearman"]) >= float(learn_rules["restricted_minimum_spearman"]) or float(metrics["harmful_auroc"]) >= float(learn_rules["restricted_minimum_harmful_auroc"])
    learn_case = "R-LEARN-A" if learn_a else ("R-LEARN-B" if restricted else "R-LEARN-C")

    burden = read_csv(output_root / "structure/sample_read_burden.csv")
    layer = read_csv(output_root / "structure/layer_summary.csv")
    trigger = read_csv(output_root / "structure/trigger_relative_summary.csv")
    source = read_csv(output_root / "structure/dataset_source_structure.csv")
    spans = read_csv(output_root / "structure/harmful_span_statistics.csv")
    harmful_span_mean = float(dense_persistence["harmful_span_mean"])
    structure_md = f"""# READ-harm structure summary

- Primary census: **{len({row['uid'] for row in dense}):,} UIDs / {len(dense):,} states**.
- READ signs: **{counts['harmful']:,} harmful / {counts['beneficial']:,} beneficial / {counts['zero']:,} zero**.
- Strong flips: **{cohorts['read_harmful_flip']:,} harmful / {cohorts['read_beneficial_flip']:,} beneficial**.
- Harmful adjacent persistence is `{float(dense_persistence['harmful_persistence']):.4f}` versus shuffled q97.5 `{float(null_rows[('dense', 'harmful_persistence')]['null_q975']):.4f}`; gate **{persistence_pass}**.
- Mean harmful span is `{harmful_span_mean:.3f}` versus shuffled q97.5 `{float(null_rows[('dense', 'harmful_span_mean')]['null_q975']):.3f}`; the complete span distribution is in `harmful_span_statistics.csv` ({len(spans)} bins).
- Immediate-neighbor harmful prevalence averages `{float(np.mean(neighbor_values)):.4f}` versus population `{population_harmful:.4f}`; gate **{neighborhood_pass}**.
- Trigger-relative evidence is frozen in `trigger_relative_summary.csv`; harm by exact layer is in `layer_summary.csv`.
- UID fragility is explicit in `sample_read_burden.csv`; maximum harmful-layer fraction is `{max(float(row['harmful_fraction']) for row in burden):.4f}`.
- Dataset/source variation is reported over {len(source)} fixed cells in `dataset_source_structure.csv`.

Decision: **{structure_case}**. The fixed rule requires persistence, span enrichment, and harmful-neighborhood enrichment; component gates are `{persistence_pass}/{span_pass}/{neighborhood_pass}`.
"""
    (output_root / "summaries/read_harm_structure_summary.md").write_text(structure_md)

    mechanism_md = f"""# READ-harm mechanism summary

- The largest pooled nuisance-matched harmful-minus-beneficial feature is `{largest['feature']}` with standardized effect `{float(largest['standardized_effect_size']):.4f}` and group-bootstrap CI on the raw difference `[{float(largest['bootstrap_ci_low']):.6f}, {float(largest['bootstrap_ci_high']):.6f}]`.
- **{len(significant)} / {len(pooled)}** preregistered features have pooled matched CIs excluding zero.
- Dataset/source sign consistency for the largest feature is `{consistent:.3f}` over {len(strata)} supported strata.
- Best image-group-disjoint harmful-vs-beneficial matched-probe AUROC is `{probe_best:.4f}`.
- F1/F2/F3 quantify text/control update magnitude, alignment, and token concentration; F4 attention mass/entropy, F5 q-k compatibility, F6 projected READ output/value magnitude, and F7 visual spatial concentration. Their complete effects are in `feature_effect_sizes.csv`.

Decision: **{mechanism_case}**. This supports only associations in measured READ-operation statistics. It does not establish attention to a semantically wrong object, causality, benchmark gain, or a deployable READ policy.
"""
    (output_root / "summaries/read_harm_mechanism_summary.md").write_text(mechanism_md)

    ablations = read_csv(output_root / "learnability/feature_group_ablation.csv")
    generic = read_csv(output_root / "learnability/generic_prestate_baseline.csv")[0]
    one_step = read_csv(output_root / "learnability/one_step_delta_baseline.csv")[0]
    nuisance = read_csv(output_root / "learnability/nuisance_baseline.csv")[0]
    fusion = max(read_csv(output_root / "learnability/generic_plus_read_features.csv"), key=lambda row: float(row["spearman"]))
    best_layer = max(read_csv(output_root / "learnability/layer_breakdown.csv"), key=lambda row: float(row["spearman"]))
    best_trigger = max(read_csv(output_root / "learnability/trigger_relative_breakdown.csv"), key=lambda row: float(row["spearman"]))
    best_source = max(read_csv(output_root / "learnability/dataset_source_breakdown.csv"), key=lambda row: float(row["spearman"]))
    routed_oof = read_csv(output_root / "routed_secondary/learnability_metrics.csv")[0]
    routed_transfer = read_csv(output_root / "routed_secondary/dense_to_routed_transfer.csv")[0]
    learnability_md = f"""# READ-harm learnability summary

- Baselines (Spearman / harmful AUROC): nuisance `{float(nuisance['spearman']):.4f}/{float(nuisance['harmful_auroc']):.4f}`, generic pre-state `{float(generic['spearman']):.4f}/{float(generic['harmful_auroc']):.4f}`, one-step delta `{float(one_step['spearman']):.4f}/{float(one_step['harmful_auroc']):.4f}`.
- Dense-only selected winner: **{winner['feature_group']} / {winner['model']}**, OOF Spearman `{float(metrics['spearman']):.4f}`, harmful AUROC `{float(metrics['harmful_auroc']):.4f}`, precision@top10% `{float(metrics['precision_at_0.1']):.4f}` versus prevalence `{float(metrics['harmful_prevalence']):.4f}`.
- Dense-W Spearman / harmful AUROC: `{float(dense_w['spearman']):.4f}/{float(dense_w['harmful_auroc']):.4f}`.
- Best generic+F_ALL fusion Spearman is `{float(fusion['spearman']):.4f}`. All {len(ablations)} fixed feature/model ablations are in `feature_group_ablation.csv`.
- Best exact layer is `{best_layer['layer']}` (Spearman `{float(best_layer['spearman']):.4f}`); best trigger-relative bin is `{best_trigger['trigger_relative_bin']}` (`{float(best_trigger['spearman']):.4f}`); strongest dataset/source cell is `{best_source['dataset_source']}` (`{float(best_source['spearman']):.4f}`).
- Semantic Q1/Q5, cluster OOD, historical↔canonical, and LODO results are under `generalization/`. The material external-transfer gate is **{str(gate['passed']).upper()}**; no external branch measurement ran.
- Routed evidence is secondary and selection-qualified: routed OOF Spearman `{float(routed_oof['spearman']):.4f}`, Dense-to-routed `{float(routed_transfer['spearman']):.4f}`.

Decision: **{learn_case}**. This phase does not establish final benchmark improvement, causal optimality, compute savings, or WRITE behavior.
"""
    (output_root / "summaries/read_harm_learnability_summary.md").write_text(learnability_md)
    recommendations = {
        "R-LEARN-A": "train one minimal READ-specific critic/router using the identified mechanism features, with WRITE always ON",
        "R-LEARN-B": "restrict READ control to the predictable regime and test one selective intervention",
        "R-LEARN-C": "run one short-horizon READ effect-propagation/planning audit instead of another local classifier",
    }
    recommendation = recommendations[learn_case]
    (output_root / "summaries/next_read_method_recommendation.md").write_text(
        f"# Next READ-method recommendation\n\nRecommend exactly one next step: **{recommendation}**.\n\nThis is a recommendation, not authorization to execute it. Do not return directly to a four-action router.\n"
    )
    atomic_json(output_root / "summaries/decision_categories.json", {
        "contract_sha256": contract["contract_sha256"], "structure": structure_case,
        "mechanism": mechanism_case, "learnability": learn_case,
        "component_gates": {"persistence": persistence_pass, "span": span_pass, "neighborhood": neighborhood_pass,
                            "matched_significant_features": len(significant), "matched_probe_auroc": probe_best,
                            "stratum_sign_consistency": consistent, "external_transfer": bool(gate["passed"]),
                            "high_precision": high_precision, "dense_w": float(dense_w["spearman"])},
    })
    _make_figures(output_root)

    required = [
        "protocol.md", "frozen_contract.json", "population/dense_read_state_manifest.jsonl",
        "population/routed_read_state_manifest.jsonl", "population/cohort_counts.csv", "population/group_registry.jsonl",
        "structure/per_uid_read_harm_maps.jsonl", "structure/layer_summary.csv", "structure/trigger_relative_summary.csv",
        "structure/sign_transition_matrix.csv", "structure/adjacent_persistence.csv", "structure/harmful_span_statistics.csv",
        "structure/shuffled_null_statistics.csv", "structure/strong_flip_neighborhood.csv", "structure/sample_read_burden.csv",
        "structure/dataset_source_structure.csv", "features/read_feature_contract.md", "features/read_operation_features.jsonl",
        "features/feature_group_manifest.json", "features/feature_distribution_summary.csv",
        "matched_mechanism/harmful_vs_beneficial_matches.jsonl", "matched_mechanism/harmful_vs_stable_wrong_matches.jsonl",
        "matched_mechanism/harmful_vs_stable_correct_matches.jsonl", "matched_mechanism/feature_effect_sizes.csv",
        "matched_mechanism/matched_probe_metrics.csv", "learnability/inherited_stepB_fold_registry.jsonl",
        "learnability/nuisance_baseline.csv", "learnability/generic_prestate_baseline.csv", "learnability/one_step_delta_baseline.csv",
        "learnability/read_linear_metrics.csv", "learnability/read_mlp_metrics.csv", "learnability/feature_group_ablation.csv",
        "learnability/generic_plus_read_features.csv", "learnability/dense_w_only_metrics.csv", "learnability/strong_flip_ranking.csv",
        "learnability/layer_breakdown.csv", "learnability/trigger_relative_breakdown.csv", "learnability/dataset_source_breakdown.csv",
        "generalization/semantic_similarity_metrics.csv", "generalization/question_cluster_ood.csv",
        "generalization/historical_to_canonical.csv", "generalization/canonical_to_historical.csv", "generalization/dataset_lodo.csv",
        "generalization/external_transfer_gate.md", "routed_secondary/structure_summary.csv",
        "routed_secondary/learnability_metrics.csv", "routed_secondary/dense_to_routed_transfer.csv",
        "statistics/group_bootstrap_ci.csv", "statistics/pairwise_model_differences.csv",
        *[f"figures/{name}" for name in (
            "read_harm_map_examples.png", "read_harm_by_layer.png", "read_harm_by_trigger_depth.png",
            "harmful_span_length.png", "strong_flip_neighborhood.png", "read_attention_entropy_harmful_vs_beneficial.png",
            "read_update_norm_harmful_vs_beneficial.png", "read_feature_effect_sizes.png", "read_learnability_comparison.png",
            "read_high_precision_harmful_subset.png", "read_generalization_ladder.png")],
        "summaries/read_harm_structure_summary.md", "summaries/read_harm_mechanism_summary.md",
        "summaries/read_harm_learnability_summary.md", "summaries/next_read_method_recommendation.md",
    ]
    missing = [name for name in required if not (output_root / name).is_file()]
    if missing:
        raise RuntimeError(f"required artifact set incomplete: {missing}")
    files = {
        str(path.relative_to(output_root)): file_sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and "work" not in path.relative_to(output_root).parts and path.name != "artifact_manifest.json"
    }
    artifact = {
        "schema_version": "read_harm_structure_learnability_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"], "required_artifacts": required,
        "decision_categories": {"structure": structure_case, "mechanism": mechanism_case, "learnability": learn_case},
        "files": files,
    }
    artifact["artifact_manifest_sha256"] = canonical_hash(artifact)
    atomic_json(output_root / "artifact_manifest.json", artifact)
    print(json.dumps({"complete": True, "contract_sha256": contract["contract_sha256"], "artifact_manifest_sha256": artifact["artifact_manifest_sha256"], "structure": structure_case, "mechanism": mechanism_case, "learnability": learn_case}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "structure", "prepare-training", "train", "aggregate-dense",
        "prepare-generalization", "generalization", "aggregate-generalization",
        "prepare-routed", "routed-train", "aggregate-routed", "finalize",
    ))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    actions = {
        "structure": lambda: structure(args.config),
        "prepare-training": lambda: prepare_training(args.config),
        "train": lambda: train_worker(args.config, args.rank, resume=args.resume),
        "aggregate-dense": lambda: aggregate_dense(args.config),
        "prepare-generalization": lambda: prepare_generalization(args.config),
        "generalization": lambda: generalization_worker(args.config, args.rank, resume=args.resume),
        "aggregate-generalization": lambda: aggregate_generalization(args.config),
        "prepare-routed": lambda: prepare_routed_training(args.config),
        "routed-train": lambda: routed_worker(args.config, args.rank, resume=args.resume),
        "aggregate-routed": lambda: aggregate_routed(args.config),
        "finalize": lambda: finalize(args.config),
    }
    actions[args.command]()


if __name__ == "__main__":
    main()
