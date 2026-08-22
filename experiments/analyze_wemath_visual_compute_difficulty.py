#!/usr/bin/env python3
"""Analyze WeMath2.0-Pro difficulty against raw minimum valid visual depth.

This script is deliberately label-only.  It consumes the checksum-bound
per-sample index produced by the full raw-cache audit; it never loads a model,
executes a route, or reads the max-50 predictor supervision view.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "outputs/wemath2pro_mcts_label_analysis_v1"
MANIFEST = (
    PROJECT
    / "outputs/label_regeneration/wemath2pro_cap400_v2/manifest/wemath2pro_valid_mcts_v1.jsonl"
)
OUTPUT = PROJECT / "outputs/wemath2pro_visual_compute_difficulty_v1"
DIFFICULTIES = ("base", "x", "y", "z", "xy", "xz", "yz", "xyz")
DEGREE = {label: 0 if label == "base" else len(label) for label in DIFFICULTIES}
TRANSITIONS = (
    ("base", "x"), ("base", "y"), ("base", "z"),
    ("x", "xy"), ("x", "xz"),
    ("y", "xy"), ("y", "yz"),
    ("z", "xz"), ("z", "yz"),
    ("xy", "xyz"), ("xz", "xyz"), ("yz", "xyz"),
)
BUDGETS = tuple(range(0, 29, 2))
REPORT_BUDGETS = {8, 12, 16, 18, 20, 22, 24, 28}
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 20260822


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {
            "n": 0, "mean": None, "median": None, "std": None,
            "p10": None, "q25": None, "q75": None, "p90": None,
        }
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _cluster_parts(values: Sequence[float], clusters: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    if len(values) != len(clusters) or not values:
        raise ValueError("cluster bootstrap requires equally sized nonempty values/clusters")
    sums: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for value, cluster in zip(values, clusters):
        sums[str(cluster)] += float(value)
        counts[str(cluster)] += 1
    names = sorted(counts)
    return (
        np.asarray([sums[name] for name in names], dtype=float),
        np.asarray([counts[name] for name in names], dtype=float),
    )


def cluster_bootstrap_mean_ci(
    values: Sequence[float], clusters: Sequence[str], *, draws: int, seed: int
) -> tuple[float, float]:
    sums, counts = _cluster_parts(values, clusters)
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    chunk = 500
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        indices = rng.integers(0, len(sums), size=(size, len(sums)))
        estimates.append(sums[indices].sum(axis=1) / counts[indices].sum(axis=1))
    combined = np.concatenate(estimates)
    return float(np.quantile(combined, 0.025)), float(np.quantile(combined, 0.975))


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return float("nan")
    x = average_ranks(values_x)
    y = average_ranks(values_y)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cluster_bootstrap_spearman_ci(
    values_x: Sequence[float], values_y: Sequence[float], clusters: Sequence[str], *, draws: int, seed: int
) -> tuple[float, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[str(cluster)].append(index)
    names = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = []
    x = np.asarray(values_x, dtype=float)
    y = np.asarray(values_y, dtype=float)
    for _ in range(draws):
        chosen = rng.integers(0, len(names), size=len(names))
        indices = [index for cluster_index in chosen for index in grouped[names[cluster_index]]]
        estimate = spearman(x[indices], y[indices])
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan")
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def fit_ols(y: Sequence[float], columns: Sequence[Sequence[float]]) -> tuple[np.ndarray, float]:
    response = np.asarray(y, dtype=float)
    matrix = np.column_stack([np.ones(len(response)), *[np.asarray(col, dtype=float) for col in columns]])
    beta, *_ = np.linalg.lstsq(matrix, response, rcond=None)
    fitted = matrix @ beta
    denominator = float(((response - response.mean()) ** 2).sum())
    r2 = 1.0 - float(((response - fitted) ** 2).sum()) / denominator if denominator else 0.0
    return beta, r2


def cluster_bootstrap_ols_ci(
    y: Sequence[float], columns: Sequence[Sequence[float]], clusters: Sequence[str], *, draws: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[str(cluster)].append(index)
    names = sorted(grouped)
    rng = np.random.default_rng(seed)
    response = np.asarray(y, dtype=float)
    arrays = [np.asarray(column, dtype=float) for column in columns]
    estimates = []
    for _ in range(draws):
        chosen = rng.integers(0, len(names), size=len(names))
        indices = [index for cluster_index in chosen for index in grouped[names[cluster_index]]]
        beta, _ = fit_ols(response[indices], [column[indices] for column in arrays])
        estimates.append(beta)
    matrix = np.vstack(estimates)
    return np.quantile(matrix, 0.025, axis=0), np.quantile(matrix, 0.975, axis=0)


def grouped_rows(
    rows: Sequence[dict[str, Any]], *, include_degree: bool = True
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    output = [
        ("difficulty", label, [row for row in rows if row["difficulty"] == label])
        for label in DIFFICULTIES
    ]
    if include_degree:
        output.extend(
            ("degree", str(degree), [row for row in rows if row["difficulty_degree"] == degree])
            for degree in range(4)
        )
    return output


def summarize_min_on(
    members: Sequence[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    values = [float(row["raw_min_on"]) for row in members]
    result = distribution_summary(values)
    if members:
        low, high = cluster_bootstrap_mean_ci(
            values, [row["question_id"] for row in members], draws=BOOTSTRAP_DRAWS, seed=seed
        )
        cheaper = [float(value < 28) for value in values]
        cheaper_low, cheaper_high = cluster_bootstrap_mean_ci(
            cheaper, [row["question_id"] for row in members], draws=BOOTSTRAP_DRAWS, seed=seed + 1
        )
        removable = [28.0 - value for value in values]
        result.update({
            "mean_ci_low": low,
            "mean_ci_high": high,
            "mean_removable_layers": float(np.mean(removable)),
            "median_removable_layers": float(np.median(removable)),
            "cheaper_route_n": int(sum(cheaper)),
            "cheaper_route_fraction": float(np.mean(cheaper)),
            "cheaper_route_ci_low": cheaper_low,
            "cheaper_route_ci_high": cheaper_high,
            "full_only_n": int(len(values) - sum(cheaper)),
            "full_only_fraction": float(1.0 - np.mean(cheaper)),
        })
    return result


def feasibility_rows(
    rows: Sequence[dict[str, Any]], *, cohort: str, seed_offset: int
) -> list[dict[str, Any]]:
    output = []
    for group_index, (group_type, group, members) in enumerate(grouped_rows(rows)):
        values = [int(row["raw_min_on"]) for row in members]
        clusters = [row["question_id"] for row in members]
        for budget in BUDGETS:
            indicators = [float(value <= budget) for value in values]
            if indicators:
                low, high = cluster_bootstrap_mean_ci(
                    indicators,
                    clusters,
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED + seed_offset + group_index * 100 + budget,
                )
                fraction = float(np.mean(indicators))
            else:
                low = high = fraction = None
            output.append({
                "cohort": cohort,
                "group_type": group_type,
                "group": group,
                "budget_visual_on": budget,
                "n": len(values),
                "feasible_n": int(sum(indicators)),
                "feasible_fraction": fraction,
                "ci_low": low,
                "ci_high": high,
                "in_compact_report_table": budget in REPORT_BUDGETS,
            })
    return output


def load_and_validate() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_manifest = json.loads((SOURCE / "analysis_manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((SOURCE / "completion_audit_v1.json").read_text(encoding="utf-8"))
    sample_path = SOURCE / "per_sample_training_suitability_v1.jsonl"
    expected_outputs = source_manifest["output_hashes_before_manifest"]
    if sha256_file(sample_path) != expected_outputs[str(sample_path.relative_to(PROJECT))]:
        raise RuntimeError("raw-derived per-sample index checksum mismatch")
    if sha256_file(MANIFEST) != source_manifest["source_hashes"]["manifest"]:
        raise RuntimeError("frozen WeMath manifest checksum mismatch")
    if completion.get("status") != "PASS" or completion.get("sample_records") != 4544:
        raise RuntimeError("source completion audit is not PASS for 4,544 records")
    samples = read_jsonl(sample_path)
    manifest = read_jsonl(MANIFEST)
    if len(samples) != 4544 or len(manifest) != 4544:
        raise RuntimeError("expected exactly 4,544 source and derived records")
    sample_by_uid = {row["uid"]: row for row in samples}
    manifest_by_uid = {row["uid"]: row for row in manifest}
    if len(sample_by_uid) != 4544 or set(sample_by_uid) != set(manifest_by_uid):
        raise RuntimeError("source UID reconciliation failed")
    rows = []
    for uid in sorted(sample_by_uid):
        row = dict(sample_by_uid[uid])
        metadata = manifest_by_uid[uid]
        if row["difficulty"] != metadata["difficulty"]:
            raise RuntimeError(f"difficulty binding mismatch: {uid}")
        row["question_id"] = str(metadata["question_id"])
        row["knowledge_points"] = metadata.get("knowledge_points", [])
        row["difficulty_degree"] = DEGREE.get(row["difficulty"])
        if row["difficulty_degree"] is None:
            raise RuntimeError(f"unexpected difficulty label: {row['difficulty']}")
        rows.append(row)
    if set(row["difficulty"] for row in rows) != set(DIFFICULTIES):
        raise RuntimeError("observed difficulty strata do not match the frozen eight-label design")
    expected_groups = {"A": 1425, "B": 784, "C": 57, "D": 2278}
    groups = Counter(row["group"] for row in rows)
    if groups != Counter(expected_groups):
        raise RuntimeError(f"A/B/C/D population mismatch: {dict(groups)}")
    if sum(row["raw_valid_routes"] for row in rows) != 107_671:
        raise RuntimeError("raw valid-route total does not match authoritative audit")
    if sum(row["evaluated_routes"] for row in rows) != 1_658_485:
        raise RuntimeError("evaluated-route total does not match authoritative audit")
    if any(row["scoring_timeout_route_occurrences"] for row in rows):
        raise RuntimeError("source cache contains an unexpected scoring timeout")
    full_correct = [row for row in rows if row["current_all_on_status"] == "correct"]
    if len(full_correct) != 841 or any(row["raw_min_on"] is None for row in full_correct):
        raise RuntimeError("FULL-correct primary cohort is incomplete")
    return rows, {
        "source_analysis_manifest_sha256": sha256_file(SOURCE / "analysis_manifest.json"),
        "source_completion_audit_sha256": sha256_file(SOURCE / "completion_audit_v1.json"),
        "source_per_sample_sha256": sha256_file(sample_path),
        "source_manifest_sha256": sha256_file(MANIFEST),
        "source_integrity": "PASS",
    }


def population_analysis(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for label in DIFFICULTIES:
        members = [row for row in rows if row["difficulty"] == label]
        full_correct = [row for row in members if row["current_all_on_status"] == "correct"]
        positive = [row for row in members if row["raw_valid_routes"] > 0]
        output.append({
            "difficulty": label,
            "degree": DEGREE[label],
            "total_eligible": len(members),
            "full_correct": len(full_correct),
            "full_correct_fraction": len(full_correct) / len(members),
            "full_wrong": len(members) - len(full_correct),
            "positive_route_samples": len(positive),
            "positive_route_fraction": len(positive) / len(members),
            "zero_positive_samples": len(members) - len(positive),
            "zero_positive_fraction": 1.0 - len(positive) / len(members),
        })
    return output


def full_correct_analysis(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary = [row for row in rows if row["current_all_on_status"] == "correct"]
    summaries = []
    for index, (group_type, group, members) in enumerate(grouped_rows(primary)):
        summary = summarize_min_on(members, seed=BOOTSTRAP_SEED + index * 10)
        summaries.append({"group_type": group_type, "group": group, **summary})
    return summaries, feasibility_rows(primary, cohort="full_correct", seed_offset=1_000)


def full_wrong_analysis(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    full_wrong = [row for row in rows if row["current_all_on_status"] == "wrong"]
    correction_rows = []
    for index, (group_type, group, members) in enumerate(grouped_rows(full_wrong)):
        corrected = [float(row["raw_valid_routes"] > 0) for row in members]
        clusters = [row["question_id"] for row in members]
        low, high = cluster_bootstrap_mean_ci(
            corrected, clusters, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED + 2_000 + index
        )
        correction_rows.append({
            "group_type": group_type,
            "group": group,
            "full_wrong_n": len(members),
            "correction_found_n": int(sum(corrected)),
            "correction_found_fraction": float(np.mean(corrected)),
            "correction_found_ci_low": low,
            "correction_found_ci_high": high,
            "no_correction_n": int(len(members) - sum(corrected)),
            "no_correction_fraction": float(1.0 - np.mean(corrected)),
        })
    zero = [row for row in full_wrong if row["raw_valid_routes"] == 0]
    zero_rows = []
    for label in DIFFICULTIES:
        stratum_wrong = [row for row in full_wrong if row["difficulty"] == label]
        stratum_zero = [row for row in zero if row["difficulty"] == label]
        zero_rows.append({
            "difficulty": label,
            "degree": DEGREE[label],
            "zero_positive_n": len(stratum_zero),
            "fraction_of_all_zero_positive": len(stratum_zero) / len(zero),
            "full_wrong_n": len(stratum_wrong),
            "zero_positive_within_full_wrong_fraction": len(stratum_zero) / len(stratum_wrong),
        })
    group_a = [row for row in full_wrong if row["raw_valid_routes"] > 0]
    group_a_rows = []
    for index, (group_type, group, members) in enumerate(grouped_rows(group_a)):
        group_a_rows.append({
            "row_type": "distribution",
            "group_type": group_type,
            "group": group,
            **summarize_min_on(members, seed=BOOTSTRAP_SEED + 3_000 + index * 10),
        })
    group_a_rows.extend(
        {"row_type": "budget_feasibility", **row}
        for row in feasibility_rows(group_a, cohort="group_a_correcting", seed_offset=4_000)
    )
    return correction_rows, zero_rows, group_a_rows


def paired_family_analysis(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary = [row for row in rows if row["current_all_on_status"] == "correct"]
    by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in primary:
        by_family[row["question_id"]][row["difficulty"]].append(row)
    output = []
    all_pairs: list[dict[str, Any]] = []
    for transition_index, (source, target) in enumerate(TRANSITIONS):
        pairs = []
        for family, labels in by_family.items():
            if len(labels.get(source, [])) != 1 or len(labels.get(target, [])) != 1:
                continue
            left, right = labels[source][0], labels[target][0]
            pairs.append({
                "family": family,
                "delta": float(right["raw_min_on"] - left["raw_min_on"]),
                "same_image": left["image_group_id"] == right["image_group_id"],
            })
        all_pairs.extend({"transition": f"{source}->{target}", **pair} for pair in pairs)
        for scope, selected in (
            ("same_family", pairs),
            ("same_family_same_image", [pair for pair in pairs if pair["same_image"]]),
        ):
            deltas = [pair["delta"] for pair in selected]
            if deltas:
                low, high = cluster_bootstrap_mean_ci(
                    deltas,
                    [pair["family"] for pair in selected],
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED + 5_000 + transition_index * 10 + (scope != "same_family"),
                )
            else:
                low = high = None
            output.append({
                "scope": scope,
                "transition": f"{source}->{target}",
                "source_degree": DEGREE[source],
                "target_degree": DEGREE[target],
                "usable_families": len(selected),
                "paired_transitions": len(selected),
                "increase_n": sum(delta > 0 for delta in deltas),
                "equal_n": sum(delta == 0 for delta in deltas),
                "decrease_n": sum(delta < 0 for delta in deltas),
                "increase_fraction": sum(delta > 0 for delta in deltas) / len(deltas) if deltas else None,
                "equal_fraction": sum(delta == 0 for delta in deltas) / len(deltas) if deltas else None,
                "decrease_fraction": sum(delta < 0 for delta in deltas) / len(deltas) if deltas else None,
                "mean_paired_delta": float(np.mean(deltas)) if deltas else None,
                "median_paired_delta": float(np.median(deltas)) if deltas else None,
                "mean_delta_ci_low": low,
                "mean_delta_ci_high": high,
            })
    unique_families = sorted({pair["family"] for pair in all_pairs})
    deltas = [pair["delta"] for pair in all_pairs]
    low, high = cluster_bootstrap_mean_ci(
        deltas,
        [pair["family"] for pair in all_pairs],
        draws=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED + 5_999,
    )
    aggregate = {
        "usable_families": len(unique_families),
        "paired_transition_occurrences": len(all_pairs),
        "increase_fraction": float(np.mean([delta > 0 for delta in deltas])),
        "equal_fraction": float(np.mean([delta == 0 for delta in deltas])),
        "decrease_fraction": float(np.mean([delta < 0 for delta in deltas])),
        "mean_paired_delta": float(np.mean(deltas)),
        "median_paired_delta": float(np.median(deltas)),
        "mean_delta_ci_low": low,
        "mean_delta_ci_high": high,
        "note": "Both endpoints must be FULL-correct; duplicate family/difficulty cells are excluded per transition.",
    }
    output.append({"scope": "all_supported_transitions", "transition": "aggregate", **aggregate})
    return output, aggregate


def visual_token_analysis(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    for cohort, cohort_rows in (
        ("all_eligible", list(rows)),
        ("full_correct", [row for row in rows if row["current_all_on_status"] == "correct"]),
    ):
        for group_type, group, members in grouped_rows(cohort_rows):
            values = [float(row["actual_visual_tokens"]) for row in members]
            output.append({
                "row_type": "token_distribution",
                "cohort": cohort,
                "group_type": group_type,
                "group": group,
                "n": len(values),
                "mean_visual_tokens": float(np.mean(values)) if values else None,
                "median_visual_tokens": float(np.median(values)) if values else None,
                "p90_visual_tokens": percentile(values, 0.90),
            })
    primary = [row for row in rows if row["current_all_on_status"] == "correct"]
    token_values = np.asarray([row["actual_visual_tokens"] for row in primary], dtype=float)
    boundaries = np.quantile(token_values, [0.25, 0.50, 0.75])
    for degree in range(4):
        for bin_index in range(4):
            lower = -math.inf if bin_index == 0 else boundaries[bin_index - 1]
            upper = math.inf if bin_index == 3 else boundaries[bin_index]
            members = [
                row for row in primary
                if row["difficulty_degree"] == degree
                and row["actual_visual_tokens"] > lower
                and row["actual_visual_tokens"] <= upper
            ]
            values = [float(row["raw_min_on"]) for row in members]
            output.append({
                "row_type": "full_correct_token_bin",
                "cohort": "full_correct",
                "group_type": "degree_by_token_quartile",
                "group": str(degree),
                "token_bin": bin_index + 1,
                "token_lower_exclusive": None if not math.isfinite(lower) else float(lower),
                "token_upper_inclusive": None if not math.isfinite(upper) else float(upper),
                "n": len(values),
                "mean_min_valid_on": float(np.mean(values)) if values else None,
                "median_min_valid_on": float(np.median(values)) if values else None,
            })
    y = [float(row["raw_min_on"]) for row in primary]
    degree = [float(row["difficulty_degree"]) for row in primary]
    log_tokens = [math.log1p(float(row["actual_visual_tokens"])) for row in primary]
    clusters = [row["question_id"] for row in primary]
    beta, r2 = fit_ols(y, [degree, log_tokens])
    low, high = cluster_bootstrap_ols_ci(
        y, [degree, log_tokens], clusters, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED + 6_000
    )
    names = ("intercept", "difficulty_degree", "log1p_visual_tokens")
    for index, name in enumerate(names):
        output.append({
            "row_type": "ols_coefficient",
            "cohort": "full_correct",
            "group_type": "min_on_degree_plus_log_tokens",
            "group": name,
            "n": len(primary),
            "coefficient": float(beta[index]),
            "ci_low": float(low[index]),
            "ci_high": float(high[index]),
            "model_r_squared": r2,
        })
    return output, {
        "n": len(primary),
        "difficulty_degree_coefficient": float(beta[1]),
        "difficulty_degree_ci": [float(low[1]), float(high[1])],
        "log_visual_tokens_coefficient": float(beta[2]),
        "log_visual_tokens_ci": [float(low[2]), float(high[2])],
        "r_squared": r2,
        "token_quartile_boundaries": [float(value) for value in boundaries],
    }


def route_geometry_analysis(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group_type, group, eligible in grouped_rows(rows):
        positive = [row for row in eligible if row["raw_valid_routes"] > 0]
        pairwise = [
            row["raw_mean_pairwise_hamming"]
            for row in positive
            if row["raw_mean_pairwise_hamming"] is not None
        ]
        output.append({
            "group_type": group_type,
            "group": group,
            "eligible_n": len(eligible),
            "positive_n": len(positive),
            "mean_raw_valid_routes_all_eligible": float(np.mean([row["raw_valid_routes"] for row in eligible])) if eligible else None,
            "median_raw_valid_routes_all_eligible": float(np.median([row["raw_valid_routes"] for row in eligible])) if eligible else None,
            "mean_raw_valid_routes_positive": float(np.mean([row["raw_valid_routes"] for row in positive])) if positive else None,
            "median_raw_valid_routes_positive": float(np.median([row["raw_valid_routes"] for row in positive])) if positive else None,
            "mean_route_on_positive": float(np.mean([row["raw_mean_on"] for row in positive])) if positive else None,
            "median_of_sample_median_route_on_positive": float(np.median([row["raw_median_on"] for row in positive])) if positive else None,
            "mean_pairwise_hamming_positive": float(np.mean(pairwise)) if pairwise else None,
            "mean_bit_entropy_positive": float(np.mean([row["raw_mean_bit_entropy"] for row in positive])) if positive else None,
        })
    return output


def trend_analysis(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    primary = [row for row in rows if row["current_all_on_status"] == "correct"]
    degree = [float(row["difficulty_degree"]) for row in primary]
    min_on = [float(row["raw_min_on"]) for row in primary]
    removable = [28.0 - value for value in min_on]
    clusters = [row["question_id"] for row in primary]
    output = []
    for index, (name, values) in enumerate((("min_valid_on", min_on), ("removable_layers", removable))):
        rho = spearman(degree, values)
        low, high = cluster_bootstrap_spearman_ci(
            degree,
            values,
            clusters,
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED + 7_000 + index,
        )
        output.append({
            "cohort": "full_correct",
            "difficulty_variable": "degree_0_to_3",
            "outcome": name,
            "n": len(primary),
            "spearman_rho": rho,
            "cluster_bootstrap_ci_low": low,
            "cluster_bootstrap_ci_high": high,
            "cluster": "question_id_seed_family",
        })
    return output


def _svg_frame(title: str, body: str, *, width: int = 880, height: int = 520) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>'
        f'{body}</svg>\n'
    )


def svg_boxplot(path: Path, grouped_values: dict[str, Sequence[float]], title: str, y_label: str) -> None:
    width, height = 880, 520
    left, right, top, bottom = 80, 30, 55, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.0, 28.0
    body = []
    for tick in range(0, 29, 4):
        y = top + plot_h * (1 - (tick - y_min) / (y_max - y_min))
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick}</text>')
    labels = list(grouped_values)
    for index, label in enumerate(labels):
        values = list(grouped_values[label])
        if not values:
            continue
        x = left + plot_w * (index + 0.5) / len(labels)
        p10, q25, median, q75, p90 = [float(np.quantile(values, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
        scale = lambda value: top + plot_h * (1 - value / 28.0)
        body.append(f'<line x1="{x:.1f}" y1="{scale(p90):.1f}" x2="{x:.1f}" y2="{scale(p10):.1f}" stroke="#333"/>')
        body.append(f'<rect x="{x-35:.1f}" y="{scale(q75):.1f}" width="70" height="{scale(q25)-scale(q75):.1f}" fill="#9ecae1" stroke="#2171b5"/>')
        body.append(f'<line x1="{x-35:.1f}" y1="{scale(median):.1f}" x2="{x+35:.1f}" y2="{scale(median):.1f}" stroke="#cb181d" stroke-width="2"/>')
        body.append(f'<text x="{x:.1f}" y="{height-42}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(label)}</text>')
        body.append(f'<text x="{x:.1f}" y="{height-25}" text-anchor="middle" font-family="sans-serif" font-size="11">N={len(values)}</text>')
    body.append(f'<text transform="translate(20,{top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>')
    path.write_text(_svg_frame(title, "".join(body), width=width, height=height), encoding="utf-8")


def svg_feasibility(path: Path, rows: Sequence[dict[str, Any]], *, group_type: str, title: str) -> None:
    width, height = 880, 520
    left, right, top, bottom = 80, 30, 55, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#1f78b4")
    body = []
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h * (1 - tick)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.1f}</text>')
    groups = sorted({row["group"] for row in rows if row["group_type"] == group_type}, key=lambda value: (len(value), value))
    for group_index, group in enumerate(groups):
        members = sorted(
            [row for row in rows if row["group_type"] == group_type and row["group"] == group],
            key=lambda row: row["budget_visual_on"],
        )
        points = []
        for row in members:
            x = left + plot_w * row["budget_visual_on"] / 28
            y = top + plot_h * (1 - row["feasible_fraction"])
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[group_index % len(colors)]
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = top + 18 * group_index
        body.append(f'<line x1="{width-150}" y1="{legend_y}" x2="{width-125}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="{width-120}" y="{legend_y+4}" font-family="sans-serif" font-size="12">{html.escape(group)}</text>')
    for budget in range(0, 29, 4):
        x = left + plot_w * budget / 28
        body.append(f'<text x="{x:.1f}" y="{height-38}" text-anchor="middle" font-family="sans-serif" font-size="12">{budget}</text>')
    body.append(f'<text x="{left+plot_w/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="13">VISUAL_ON budget C</text>')
    body.append(f'<text transform="translate(20,{top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">P(min ON ≤ C)</text>')
    path.write_text(_svg_frame(title, "".join(body), width=width, height=height), encoding="utf-8")


def create_figures(
    rows: Sequence[dict[str, Any]], feasibility: Sequence[dict[str, Any]], group_a: Sequence[dict[str, Any]]
) -> list[Path]:
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = [row for row in rows if row["current_all_on_status"] == "correct"]
    degree_values = {
        f"degree {degree}": [row["raw_min_on"] for row in primary if row["difficulty_degree"] == degree]
        for degree in range(4)
    }
    paths = [
        figure_dir / "01_full_correct_min_on_by_degree.svg",
        figure_dir / "02_full_correct_removable_by_degree.svg",
        figure_dir / "03_full_correct_budget_feasibility_degree.svg",
        figure_dir / "04_full_correct_budget_feasibility_stratum.svg",
        figure_dir / "05_group_a_budget_feasibility_degree.svg",
    ]
    svg_boxplot(paths[0], degree_values, "FULL-correct minimum discovered visual depth", "minimum valid VISUAL_ON layers")
    svg_boxplot(
        paths[1],
        {key: [28 - value for value in values] for key, values in degree_values.items()},
        "FULL-correct removable visual layers",
        "28 - minimum valid VISUAL_ON layers",
    )
    svg_feasibility(paths[2], feasibility, group_type="degree", title="FULL-correct visual-budget feasibility by degree")
    svg_feasibility(paths[3], feasibility, group_type="difficulty", title="FULL-correct visual-budget feasibility by difficulty stratum")
    group_a_feasibility = [row for row in group_a if row["row_type"] == "budget_feasibility"]
    svg_feasibility(paths[4], group_a_feasibility, group_type="degree", title="FULL-wrong corrected-route budget feasibility")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", choices=("PENDING", "A", "B", "C", "D", "E"), default="PENDING")
    parser.add_argument("--outcome-rationale", default="Interpretation pending aggregate review.")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, integrity = load_and_validate()
    population = population_analysis(rows)
    full_correct, feasibility = full_correct_analysis(rows)
    full_wrong, zero_positive, group_a = full_wrong_analysis(rows)
    paired, paired_aggregate = paired_family_analysis(rows)
    token_control, regression = visual_token_analysis(rows)
    geometry = route_geometry_analysis(rows)
    trends = trend_analysis(rows)

    paths = {
        "difficulty_population": OUTPUT / "difficulty_population.csv",
        "full_correct": OUTPUT / "full_correct_min_on_by_difficulty.csv",
        "feasibility": OUTPUT / "full_correct_budget_feasibility.csv",
        "full_wrong": OUTPUT / "full_wrong_correction_by_difficulty.csv",
        "zero_positive": OUTPUT / "zero_positive_by_difficulty.csv",
        "group_a": OUTPUT / "group_a_min_correcting_on.csv",
        "paired": OUTPUT / "paired_family_analysis.csv",
        "token_control": OUTPUT / "visual_token_control.csv",
        "geometry": OUTPUT / "route_geometry_by_difficulty.csv",
        "trends": OUTPUT / "trend_tests.csv",
    }
    for path, payload in (
        (paths["difficulty_population"], population),
        (paths["full_correct"], full_correct),
        (paths["feasibility"], feasibility),
        (paths["full_wrong"], full_wrong),
        (paths["zero_positive"], zero_positive),
        (paths["group_a"], group_a),
        (paths["paired"], paired),
        (paths["token_control"], token_control),
        (paths["geometry"], geometry),
        (paths["trends"], trends),
    ):
        write_csv(path, payload)
    figures = create_figures(rows, feasibility, group_a)

    primary_by_degree = {
        row["group"]: row for row in full_correct if row["group_type"] == "degree"
    }
    correction_by_degree = {
        row["group"]: row for row in full_wrong if row["group_type"] == "degree"
    }
    family_cells: Counter[tuple[str, str]] = Counter(
        (row["question_id"], row["difficulty"]) for row in rows
    )
    family_sizes = Counter(row["question_id"] for row in rows)
    complete_unique_families = sum(
        family_sizes[family] == len(DIFFICULTIES)
        and all(family_cells[(family, label)] == 1 for label in DIFFICULTIES)
        for family in family_sizes
    )
    summary = {
        "schema_version": "wemath2pro_visual_compute_difficulty_v1",
        "integrity": integrity,
        "population": {
            "eligible": len(rows),
            "full_correct": sum(row["current_all_on_status"] == "correct" for row in rows),
            "full_wrong": sum(row["current_all_on_status"] == "wrong" for row in rows),
            "positive_route_samples": sum(row["raw_valid_routes"] > 0 for row in rows),
            "zero_positive_samples": sum(row["raw_valid_routes"] == 0 for row in rows),
            "group_counts": dict(sorted(Counter(row["group"] for row in rows).items())),
            "difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
            "family_count": len({row["question_id"] for row in rows}),
            "image_group_count": len({row["image_group_id"] for row in rows}),
        },
        "family_metadata_audit": {
            "grouping_key": "official source question_id",
            "families": len(family_sizes),
            "complete_unique_eight_stratum_families": complete_unique_families,
            "families_with_fewer_than_eight_eligible_records": sum(size < 8 for size in family_sizes.values()),
            "duplicate_family_difficulty_cells": sum(count > 1 for count in family_cells.values()),
            "paired_rule": "require exactly one FULL-correct record at both transition endpoints",
        },
        "primary_full_correct_by_degree": primary_by_degree,
        "full_wrong_correction_by_degree": correction_by_degree,
        "trend_tests": trends,
        "paired_family_aggregate": paired_aggregate,
        "visual_token_adjusted_model": regression,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "cluster": "question_id seed family",
            "interval": "percentile",
        },
        "claim_boundaries": [
            "Minimum discovered ON is search-dependent, not an identified physical requirement.",
            "Zero-positive means no correction in the frozen finite search/action space.",
            "The analysis cannot assess repeat, recurrence, or more than 28 visual executions.",
        ],
        "outcome": args.outcome,
        "outcome_rationale": args.outcome_rationale,
    }
    summary_path = OUTPUT / "analysis_summary.json"
    write_json(summary_path, summary)
    output_paths = [*paths.values(), *figures, summary_path]
    manifest = {
        "schema_version": "wemath2pro_visual_compute_difficulty_analysis_manifest_v1",
        "status": "PASS",
        "source_integrity": integrity,
        "source_policy": "checksum-bound raw-route-derived fields only; no max-50 supervision",
        "difficulty_mapping": DEGREE,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "output_hashes": {
            str(path.relative_to(PROJECT)): sha256_file(path) for path in output_paths
        },
    }
    manifest_path = OUTPUT / "analysis_manifest.json"
    write_json(manifest_path, manifest)
    for path in (summary_path, manifest_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps({
        "status": "PASS",
        "records": len(rows),
        "full_correct": len([row for row in rows if row["current_all_on_status"] == "correct"]),
        "output": str(OUTPUT.relative_to(PROJECT)),
        "outcome": args.outcome,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
