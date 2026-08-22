#!/usr/bin/env python3
"""Analyze visual-access placement in the frozen WeMath2.0-Pro route cache.

This is a route-cache-only analysis.  It validates every authoritative raw
record and uses all discovered valid masks; it never loads Qwen, executes a
route, or reads the max-50 predictor-supervision view.
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
from typing import Any, Iterable, Sequence

import numpy as np

from experiments import analyze_wemath_visual_compute_difficulty as prior
from experiments import analyze_wemath_visual_dependence as dependence
from experiments.analyze_wemath2pro_mcts_labels import validate_record


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "outputs/wemath2pro_mcts_label_analysis_v1"
PRIOR_DEPENDENCE = PROJECT / "outputs/wemath2pro_visual_dependence_reanalysis_v1"
OUTPUT = PROJECT / "outputs/wemath2pro_visual_access_placement_v1"
REPORT = PROJECT / "reports/wemath2pro_visual_access_placement_v1.md"
MANIFEST = (
    PROJECT
    / "outputs/label_regeneration/wemath2pro_cap400_v2/manifest/wemath2pro_valid_mcts_v1.jsonl"
)
DIFFICULTIES = prior.DIFFICULTIES
DEGREE = prior.DEGREE
TRANSITIONS = prior.TRANSITIONS
AXIS_NAMES = {
    "x": "contextual complexity",
    "y": "visual complexity",
    "z": "step complexity",
}
DELTAS = (0, 2, 4)
LAYERS = tuple(range(28))
METRICS = (
    "first_on", "last_on", "centroid", "normalized_centroid", "span",
    "early_on", "middle_on", "late_on",
    "early_fraction", "middle_fraction", "late_fraction",
    "late_access", "very_late_access", "on_segments", "reentries",
    "max_internal_off_gap", "has_reentry", "has_late_reentry",
)
PRIMARY_METRICS = (
    "first_on", "last_on", "centroid", "normalized_centroid", "span",
    "early_fraction", "middle_fraction", "late_fraction", "late_access",
    "very_late_access", "on_segments", "reentries", "max_internal_off_gap",
    "has_reentry", "has_late_reentry",
)
PAIRED_METRICS = (
    "normalized_centroid", "first_on", "last_on", "late_fraction",
    "on_segments", "has_late_reentry",
)
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 20260824
ALL_OFF = (0,) * 28
ALL_ON = (1,) * 28
BUDGET_BINS = (
    (1, 8, "1-8"), (9, 12, "9-12"), (13, 16, "13-16"),
    (17, 20, "17-20"), (21, 27, "21-27"), (28, 28, "28"),
)


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


def route_metrics(mask: Sequence[int]) -> dict[str, float]:
    """Return placement/re-entry metrics for a nonempty 28-layer mask."""
    bits = tuple(int(value) for value in mask)
    if len(bits) != 28 or any(value not in (0, 1) for value in bits):
        raise ValueError("route mask must contain exactly 28 binary values")
    on = [index for index, value in enumerate(bits) if value]
    if not on:
        raise ValueError("placement metrics are undefined for ALL-OFF")
    segment_starts = [
        index for index in on if index == 0 or bits[index - 1] == 0
    ]
    gaps: list[int] = []
    previous_end: int | None = None
    for start in segment_starts:
        if previous_end is not None:
            gaps.append(start - previous_end - 1)
        end = start
        while end + 1 < len(bits) and bits[end + 1]:
            end += 1
        previous_end = end
    n_on = len(on)
    early = sum(index <= 8 for index in on)
    middle = sum(9 <= index <= 18 for index in on)
    late = sum(index >= 19 for index in on)
    centroid = float(np.mean(on))
    segments = len(segment_starts)
    return {
        "first_on": float(on[0]),
        "last_on": float(on[-1]),
        "centroid": centroid,
        "normalized_centroid": centroid / 27.0,
        "span": float(on[-1] - on[0]),
        "early_on": float(early),
        "middle_on": float(middle),
        "late_on": float(late),
        "early_fraction": early / n_on,
        "middle_fraction": middle / n_on,
        "late_fraction": late / n_on,
        "late_access": float(late > 0),
        "very_late_access": float(any(index >= 24 for index in on)),
        "on_segments": float(segments),
        "reentries": float(segments - 1),
        "max_internal_off_gap": float(max(gaps, default=0)),
        "has_reentry": float(segments > 1),
        "has_late_reentry": float(any(start >= 19 for start in segment_starts[1:])),
    }


def select_route_set(
    masks: Sequence[Sequence[int]], *, minimum_on: int, delta: int
) -> list[tuple[int, ...]]:
    limit = min(28, int(minimum_on) + int(delta))
    selected = [tuple(int(value) for value in mask) for mask in masks if sum(mask) <= limit]
    if not selected or min(sum(mask) for mask in selected) != minimum_on:
        raise ValueError("minimum/near-minimum route set is incomplete")
    return selected


def sample_route_summary(
    *, row: dict[str, Any], masks: Sequence[Sequence[int]], cohort: str, delta: int
) -> dict[str, Any]:
    minimum_on = min(sum(mask) for mask in masks)
    if minimum_on <= 0:
        raise ValueError(f"{cohort} placement received nonpositive route: {row['uid']}")
    selected = select_route_set(masks, minimum_on=minimum_on, delta=delta)
    route_values = [route_metrics(mask) for mask in selected]
    profile = np.asarray(selected, dtype=float).mean(axis=0)
    result: dict[str, Any] = {
        "cohort": cohort,
        "delta": delta,
        "uid": row["uid"],
        "difficulty": row["difficulty"],
        "difficulty_degree": row["difficulty_degree"],
        "question_id": row["question_id"],
        "image_group_id": row["image_group_id"],
        "min_positive_on": minimum_on,
        "route_on_limit": min(28, minimum_on + delta),
        "selected_route_count": len(selected),
        "all_valid_route_count": len(masks),
        "knowledge_point_count": len(row.get("knowledge_points") or []),
        "axis_x_present": int("x" in row["difficulty"]),
        "axis_y_present": int("y" in row["difficulty"]),
        "axis_z_present": int("z" in row["difficulty"]),
    }
    for metric in METRICS:
        result[metric] = float(np.mean([values[metric] for values in route_values]))
    for layer, probability in enumerate(profile):
        result[f"layer_{layer:02d}"] = float(probability)
    return result


def cluster_bootstrap_vector_ci(
    matrix: np.ndarray,
    clusters: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if len(matrix) != len(clusters) or not len(matrix):
        raise ValueError("cluster bootstrap requires aligned nonempty data")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[str(cluster)].append(index)
    names = sorted(grouped)
    sums = np.vstack([matrix[grouped[name]].sum(axis=0) for name in names])
    counts = np.asarray([len(grouped[name]) for name in names], dtype=float)
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for start in range(0, draws, 250):
        size = min(250, draws - start)
        indices = rng.integers(0, len(names), size=(size, len(names)))
        numerator = sums[indices].sum(axis=1)
        denominator = counts[indices].sum(axis=1)[:, None]
        estimates.append(numerator / denominator)
    combined = np.vstack(estimates)
    return np.quantile(combined, 0.025, axis=0), np.quantile(combined, 0.975, axis=0)


def summarize_sample_metrics(
    members: Sequence[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(members)}
    if not members:
        for metric in PRIMARY_METRICS:
            for suffix in ("mean", "median", "q25", "q75", "ci_low", "ci_high"):
                result[f"{metric}_{suffix}"] = None
        return result
    matrix = np.asarray([[row[metric] for metric in PRIMARY_METRICS] for row in members])
    low, high = cluster_bootstrap_vector_ci(
        matrix, [row["question_id"] for row in members], draws=BOOTSTRAP_DRAWS, seed=seed
    )
    for index, metric in enumerate(PRIMARY_METRICS):
        values = matrix[:, index]
        result.update({
            f"{metric}_mean": float(np.mean(values)),
            f"{metric}_median": float(np.median(values)),
            f"{metric}_q25": float(np.quantile(values, 0.25)),
            f"{metric}_q75": float(np.quantile(values, 0.75)),
            f"{metric}_ci_low": float(low[index]),
            f"{metric}_ci_high": float(high[index]),
        })
    return result


def layer_profile_rows(
    rows: Sequence[dict[str, Any]], *, delta: int, include_all_valid: bool = False
) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("difficulty", label, [row for row in rows if row["difficulty"] == label])
        for label in DIFFICULTIES
    ] + [
        ("degree", str(degree), [row for row in rows if row["difficulty_degree"] == degree])
        for degree in range(4)
    ]
    if include_all_valid:
        groups.append(("overall", "all", list(rows)))
    output: list[dict[str, Any]] = []
    for group_index, (group_type, group, members) in enumerate(groups):
        if not members:
            continue
        matrix = np.asarray([[row[f"layer_{layer:02d}"] for layer in LAYERS] for row in members])
        low, high = cluster_bootstrap_vector_ci(
            matrix,
            [row["question_id"] for row in members],
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED + delta * 1000 + group_index,
        )
        means = matrix.mean(axis=0)
        for layer in LAYERS:
            output.append({
                "delta": delta,
                "group_type": group_type,
                "group": group,
                "layer": layer,
                "n_samples": len(members),
                "sample_balanced_access_probability": float(means[layer]),
                "ci_low": float(low[layer]),
                "ci_high": float(high[layer]),
            })
    return output


def load_valid_masks() -> tuple[list[dict[str, Any]], dict[str, list[tuple[int, ...]]], dict[str, Any]]:
    rows, prior_integrity = prior.load_and_validate()
    dependence_manifest = json.loads(
        (PRIOR_DEPENDENCE / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    for relative, expected in dependence_manifest["output_hashes"].items():
        path = PROJECT / relative
        if sha256_file(path) != expected:
            raise RuntimeError(f"phase-28 output checksum mismatch: {relative}")
    dependence_summary = json.loads(
        (PRIOR_DEPENDENCE / "analysis_summary.json").read_text(encoding="utf-8")
    )
    if dependence_summary["status"] != "PASS":
        raise RuntimeError("phase-28 source is not PASS")

    source_manifest = json.loads((SOURCE / "analysis_manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((SOURCE / "completion_audit_v1.json").read_text(encoding="utf-8"))
    index_path = SOURCE / "cache_record_index_v1.jsonl"
    expected_index_hash = source_manifest["output_hashes_before_manifest"][
        str(index_path.relative_to(PROJECT))
    ]
    if sha256_file(index_path) != expected_index_hash:
        raise RuntimeError("raw-record index checksum mismatch")
    index = read_jsonl(index_path)
    manifest_rows = read_jsonl(MANIFEST)
    by_manifest = {row["uid"]: row for row in manifest_rows}
    by_row = {row["uid"]: row for row in rows}
    if {item["uid"] for item in index} != set(by_row) or set(by_manifest) != set(by_row):
        raise RuntimeError("raw index / manifest / derived UID mismatch")
    accepted_contracts = set(completion["accepted_contracts"])
    valid_masks: dict[str, list[tuple[int, ...]]] = {}
    hash_rollup = hashlib.sha256()
    candidate_total = 0
    valid_total = 0
    full_correct = 0
    all_off_correct = 0
    for position, item in enumerate(index, start=1):
        path = PROJECT / item["record_path"]
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != item["record_sha256"]:
            raise RuntimeError(f"raw-record checksum mismatch: {item['uid']}")
        hash_rollup.update(f"{item['uid']}:{digest}\n".encode())
        record = json.loads(payload)
        candidates, valid = validate_record(
            record, by_manifest[item["uid"]], accepted_contracts=accepted_contracts
        )
        masks = [tuple(int(value) for value in route["visual_on_mask"]) for route in valid]
        valid_masks[item["uid"]] = masks
        candidate_total += len(candidates)
        valid_total += len(valid)
        anchors = {tuple(route["visual_on_mask"]): route for route in candidates}
        row = by_row[item["uid"]]
        if bool(anchors[ALL_ON]["result_correct"]) != (row["current_all_on_status"] == "correct"):
            raise RuntimeError(f"FULL anchor mismatch with derived row: {item['uid']}")
        if bool(anchors[ALL_OFF]["result_correct"]) != bool(row["all_off_correct"]):
            raise RuntimeError(f"ALL-OFF anchor mismatch with derived row: {item['uid']}")
        full_correct += bool(anchors[ALL_ON]["result_correct"])
        all_off_correct += bool(anchors[ALL_OFF]["result_correct"])
        if position % 250 == 0:
            print(f"validated raw records {position}/4544", flush=True)
    if candidate_total != 1_658_485 or valid_total != 107_671:
        raise RuntimeError("raw route totals do not reproduce the frozen audit")

    regime_counts = dependence.classify_visual_regimes(rows)
    expected = dependence_summary["population"]["regime_counts"]
    if regime_counts != expected:
        raise RuntimeError(f"phase-28 visual-regime mismatch: {regime_counts} != {expected}")
    vplus = [row for row in rows if row.get("visual_regime") == "V+"]
    aplus = [row for row in rows if row.get("correction_regime") == "A+"]
    if len(vplus) != 428 or len(aplus) != 1263:
        raise RuntimeError("V+/A+ population mismatch")
    for row in [*vplus, *aplus]:
        masks = valid_masks[row["uid"]]
        if not masks or min(map(sum, masks)) != int(row["raw_min_on"]):
            raise RuntimeError(f"raw minimum-ON mismatch: {row['uid']}")
        if min(map(sum, masks)) <= 0:
            raise RuntimeError(f"positive cohort contains ALL-OFF valid route: {row['uid']}")
    integrity = {
        **prior_integrity,
        "status": "PASS",
        "raw_records_verified": len(index),
        "candidate_routes_verified": candidate_total,
        "valid_routes_verified": valid_total,
        "exact_full_anchors": len(index),
        "exact_all_off_anchors": len(index),
        "full_correct": full_correct,
        "all_off_correct": all_off_correct,
        "vplus": len(vplus),
        "aplus": len(aplus),
        "raw_record_index_sha256": expected_index_hash,
        "raw_record_hash_rollup_sha256": hash_rollup.hexdigest(),
        "phase28_analysis_manifest_sha256": sha256_file(PRIOR_DEPENDENCE / "analysis_manifest.json"),
        "phase28_analysis_summary_sha256": sha256_file(PRIOR_DEPENDENCE / "analysis_summary.json"),
    }
    return rows, valid_masks, integrity


def grouped_metric_rows(rows: Sequence[dict[str, Any]], *, group_type: str) -> list[dict[str, Any]]:
    if group_type == "difficulty":
        groups: Iterable[tuple[str, list[dict[str, Any]]]] = (
            (label, [row for row in rows if row["difficulty"] == label]) for label in DIFFICULTIES
        )
    elif group_type == "degree":
        groups = (
            (str(degree), [row for row in rows if row["difficulty_degree"] == degree])
            for degree in range(4)
        )
    else:
        raise ValueError(group_type)
    output = []
    for index, (group, members) in enumerate(groups):
        output.append({
            "group_type": group_type,
            "group": group,
            "delta": members[0]["delta"] if members else None,
            "mean_min_positive_on": float(np.mean([row["min_positive_on"] for row in members])) if members else None,
            "median_min_positive_on": float(np.median([row["min_positive_on"] for row in members])) if members else None,
            **summarize_sample_metrics(members, seed=BOOTSTRAP_SEED + 20_000 + index),
        })
    return output


def fit_adjusted_models(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    target_metrics = ("normalized_centroid", "late_fraction", "has_late_reentry")
    for delta in DELTAS:
        members = [row for row in rows if row["delta"] == delta]
        clusters = [row["question_id"] for row in members]
        min_on = [float(row["min_positive_on"]) for row in members]
        model_specs: list[tuple[str, list[str], list[list[float]]]] = [
            (
                "difficulty_categorical_plus_min_on",
                [*DIFFICULTIES[1:], "min_positive_on"],
                [[float(row["difficulty"] == label) for row in members] for label in DIFFICULTIES[1:]]
                + [min_on],
            ),
            (
                "degree_plus_min_on",
                ["difficulty_degree", "min_positive_on"],
                [[float(row["difficulty_degree"]) for row in members], min_on],
            ),
        ]
        for axis in "xyz":
            model_specs.append((
                f"axis_{axis}_plus_min_on",
                [f"axis_{axis}_present", "min_positive_on"],
                [[float(row[f"axis_{axis}_present"]) for row in members], min_on],
            ))
        for metric_index, metric in enumerate(target_metrics):
            y = [float(row[metric]) for row in members]
            for model_index, (model, names, columns) in enumerate(model_specs):
                beta, r2 = prior.fit_ols(y, columns)
                low, high = prior.cluster_bootstrap_ols_ci(
                    y,
                    columns,
                    clusters,
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED + 30_000 + delta * 100 + metric_index * 10 + model_index,
                )
                for coefficient_index, name in enumerate(("intercept", *names)):
                    output.append({
                        "delta": delta,
                        "outcome_metric": metric,
                        "model": model,
                        "coefficient_name": name,
                        "n": len(members),
                        "coefficient": float(beta[coefficient_index]),
                        "ci_low": float(low[coefficient_index]),
                        "ci_high": float(high[coefficient_index]),
                        "r_squared": float(r2),
                        "interpretation": "descriptive family-clustered adjustment; not causal",
                    })
    return output


def budget_bin_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for delta in DELTAS:
        delta_rows = [row for row in rows if row["delta"] == delta]
        group_specs = [("overall", "all", delta_rows)]
        group_specs += [
            ("degree", str(degree), [row for row in delta_rows if row["difficulty_degree"] == degree])
            for degree in range(4)
        ]
        group_specs += [
            ("difficulty", label, [row for row in delta_rows if row["difficulty"] == label])
            for label in DIFFICULTIES
        ]
        for group_type, group, group_members in group_specs:
            for lower, upper, label in BUDGET_BINS:
                members = [row for row in group_members if lower <= row["min_positive_on"] <= upper]
                output.append({
                    "delta": delta,
                    "group_type": group_type,
                    "group": group,
                    "min_on_bin": label,
                    "n": len(members),
                    "sparse_cell": len(members) < 10,
                    "normalized_centroid_mean": float(np.mean([row["normalized_centroid"] for row in members])) if members else None,
                    "normalized_centroid_median": float(np.median([row["normalized_centroid"] for row in members])) if members else None,
                    "last_on_mean": float(np.mean([row["last_on"] for row in members])) if members else None,
                    "late_fraction_mean": float(np.mean([row["late_fraction"] for row in members])) if members else None,
                    "on_segments_mean": float(np.mean([row["on_segments"] for row in members])) if members else None,
                    "late_reentry_rate": float(np.mean([row["has_late_reentry"] for row in members])) if members else None,
                })
    return output


def _paired_samples(
    sample_rows: Sequence[dict[str, Any]], transition: tuple[str, str]
) -> list[dict[str, Any]]:
    source, target = transition
    by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in sample_rows:
        by_family[row["question_id"]][row["difficulty"]].append(row)
    pairs: list[dict[str, Any]] = []
    for family, labels in by_family.items():
        if len(labels.get(source, [])) != 1 or len(labels.get(target, [])) != 1:
            continue
        left, right = labels[source][0], labels[target][0]
        pair: dict[str, Any] = {
            "family": family,
            "source_uid": left["uid"],
            "target_uid": right["uid"],
            "source_difficulty": source,
            "target_difficulty": target,
            "same_image": left["image_group_id"] == right["image_group_id"],
            "source_min_on": left["min_positive_on"],
            "target_min_on": right["min_positive_on"],
        }
        for metric in PAIRED_METRICS:
            pair[f"delta_{metric}"] = float(right[metric] - left[metric])
        left_profile = np.asarray([left[f"layer_{layer:02d}"] for layer in LAYERS], dtype=float)
        right_profile = np.asarray([right[f"layer_{layer:02d}"] for layer in LAYERS], dtype=float)
        difference = right_profile - left_profile
        denominator = float(np.linalg.norm(left_profile) * np.linalg.norm(right_profile))
        pair["profile_l1"] = float(np.abs(difference).sum())
        pair["profile_l2"] = float(np.linalg.norm(difference))
        pair["profile_cosine_similarity"] = (
            float(left_profile @ right_profile / denominator) if denominator else None
        )
        for layer in LAYERS:
            pair[f"delta_layer_{layer:02d}"] = float(difference[layer])
        pairs.append(pair)
    return pairs


def paired_analyses(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    transition_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    same_image_rows: list[dict[str, Any]] = []
    pairs_by_delta: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for delta in DELTAS:
        delta_rows = [row for row in rows if row["delta"] == delta]
        for transition_index, transition in enumerate(TRANSITIONS):
            transition_name = f"{transition[0]}->{transition[1]}"
            pairs = _paired_samples(delta_rows, transition)
            pairs_by_delta[delta].extend({"transition": transition_name, **pair} for pair in pairs)
            for scope, selected in (
                ("same_family", pairs),
                ("same_family_same_image", [pair for pair in pairs if pair["same_image"]]),
            ):
                if not selected:
                    for metric in PAIRED_METRICS:
                        target = same_image_rows if scope.endswith("same_image") else transition_rows
                        target.append({
                            "delta": delta, "scope": scope, "transition": transition_name,
                            "metric": metric, "n": 0,
                        })
                    continue
                matrix = np.asarray([
                    [pair[f"delta_{metric}"] for metric in PAIRED_METRICS] for pair in selected
                ])
                low, high = cluster_bootstrap_vector_ci(
                    matrix,
                    [pair["family"] for pair in selected],
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED + 40_000 + delta * 1000 + transition_index * 10 + int(scope.endswith("same_image")),
                )
                for metric_index, metric in enumerate(PAIRED_METRICS):
                    values = matrix[:, metric_index]
                    row = {
                        "delta": delta,
                        "scope": scope,
                        "transition": transition_name,
                        "source_degree": DEGREE[transition[0]],
                        "target_degree": DEGREE[transition[1]],
                        "metric": metric,
                        "n": len(values),
                        "mean_delta": float(np.mean(values)),
                        "median_delta": float(np.median(values)),
                        "increase_n": int(sum(values > 1e-12)),
                        "equal_n": int(sum(np.abs(values) <= 1e-12)),
                        "decrease_n": int(sum(values < -1e-12)),
                        "increase_fraction": float(np.mean(values > 1e-12)),
                        "equal_fraction": float(np.mean(np.abs(values) <= 1e-12)),
                        "decrease_fraction": float(np.mean(values < -1e-12)),
                        "ci_low": float(low[metric_index]),
                        "ci_high": float(high[metric_index]),
                    }
                    (same_image_rows if scope.endswith("same_image") else transition_rows).append(row)
            if pairs:
                profile_matrix = np.asarray([
                    [pair[f"delta_layer_{layer:02d}"] for layer in LAYERS] for pair in pairs
                ])
                low, high = cluster_bootstrap_vector_ci(
                    profile_matrix,
                    [pair["family"] for pair in pairs],
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED + 50_000 + delta * 1000 + transition_index,
                )
                means = profile_matrix.mean(axis=0)
                for layer in LAYERS:
                    layer_rows.append({
                        "delta": delta,
                        "transition": transition_name,
                        "layer": layer,
                        "n": len(pairs),
                        "mean_harder_minus_easier_access_probability": float(means[layer]),
                        "ci_low": float(low[layer]),
                        "ci_high": float(high[layer]),
                    })
        all_pairs = pairs_by_delta[delta]
        if all_pairs:
            # Families can contribute to several supported transitions. Cluster by
            # family so the aggregate does not pretend they are independent.
            matrix = np.asarray([
                [pair[f"delta_{metric}"] for metric in PAIRED_METRICS] for pair in all_pairs
            ])
            low, high = cluster_bootstrap_vector_ci(
                matrix,
                [pair["family"] for pair in all_pairs],
                draws=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED + 60_000 + delta,
            )
            for metric_index, metric in enumerate(PAIRED_METRICS):
                values = matrix[:, metric_index]
                transition_rows.append({
                    "delta": delta, "scope": "all_supported_transitions",
                    "transition": "aggregate", "metric": metric,
                    "n": len(values), "unique_families": len({pair["family"] for pair in all_pairs}),
                    "mean_delta": float(np.mean(values)), "median_delta": float(np.median(values)),
                    "increase_n": int(sum(values > 1e-12)),
                    "equal_n": int(sum(np.abs(values) <= 1e-12)),
                    "decrease_n": int(sum(values < -1e-12)),
                    "increase_fraction": float(np.mean(values > 1e-12)),
                    "equal_fraction": float(np.mean(np.abs(values) <= 1e-12)),
                    "decrease_fraction": float(np.mean(values < -1e-12)),
                    "ci_low": float(low[metric_index]), "ci_high": float(high[metric_index]),
                })
            for shape_metric in ("profile_l1", "profile_l2", "profile_cosine_similarity"):
                values = np.asarray([pair[shape_metric] for pair in all_pairs if pair[shape_metric] is not None])
                low_shape, high_shape = prior.cluster_bootstrap_mean_ci(
                    values.tolist(),
                    [pair["family"] for pair in all_pairs if pair[shape_metric] is not None],
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED + 61_000 + delta + len(shape_metric),
                )
                transition_rows.append({
                    "delta": delta, "scope": "all_supported_transitions",
                    "transition": "aggregate", "metric": shape_metric,
                    "n": len(values), "mean_delta": float(np.mean(values)),
                    "median_delta": float(np.median(values)),
                    "ci_low": low_shape, "ci_high": high_shape,
                })
            image_pairs = [pair for pair in all_pairs if pair["same_image"]]
            if image_pairs:
                image_matrix = np.asarray([
                    [pair[f"delta_{metric}"] for metric in PAIRED_METRICS] for pair in image_pairs
                ])
                image_low, image_high = cluster_bootstrap_vector_ci(
                    image_matrix,
                    [pair["family"] for pair in image_pairs],
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED + 62_000 + delta,
                )
                for metric_index, metric in enumerate(PAIRED_METRICS):
                    values = image_matrix[:, metric_index]
                    same_image_rows.append({
                        "delta": delta,
                        "scope": "all_supported_same_image_transitions",
                        "transition": "aggregate",
                        "metric": metric,
                        "n": len(values),
                        "unique_families": len({pair["family"] for pair in image_pairs}),
                        "mean_delta": float(np.mean(values)),
                        "median_delta": float(np.median(values)),
                        "increase_n": int(sum(values > 1e-12)),
                        "equal_n": int(sum(np.abs(values) <= 1e-12)),
                        "decrease_n": int(sum(values < -1e-12)),
                        "increase_fraction": float(np.mean(values > 1e-12)),
                        "equal_fraction": float(np.mean(np.abs(values) <= 1e-12)),
                        "decrease_fraction": float(np.mean(values < -1e-12)),
                        "ci_low": float(image_low[metric_index]),
                        "ci_high": float(image_high[metric_index]),
                    })
    return transition_rows, layer_rows, same_image_rows, pairs_by_delta


def axis_added(source: str, target: str) -> str:
    added = set(target) - set() if source == "base" else set(target) - set(source)
    if source == "base":
        added = set(target)
    if len(added) != 1:
        raise ValueError(f"transition does not add exactly one axis: {source}->{target}")
    return next(iter(added))


def axis_summary_rows(
    sample_rows: Sequence[dict[str, Any]], pairs_by_delta: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metric_names = ("normalized_centroid", "last_on", "late_fraction", "has_late_reentry", "on_segments")
    transition_axis = {f"{source}->{target}": axis_added(source, target) for source, target in TRANSITIONS}
    for delta in DELTAS:
        rows = [row for row in sample_rows if row["delta"] == delta]
        for axis_index, axis in enumerate("xyz"):
            absent = [row for row in rows if not row[f"axis_{axis}_present"]]
            present = [row for row in rows if row[f"axis_{axis}_present"]]
            for metric_index, metric in enumerate(metric_names):
                # Unpaired amount-adjusted coefficient is copied from a direct
                # two-predictor model to keep this artifact self-contained.
                y = [float(row[metric]) for row in rows]
                columns = [
                    [float(row[f"axis_{axis}_present"]) for row in rows],
                    [float(row["min_positive_on"]) for row in rows],
                ]
                beta, _ = prior.fit_ols(y, columns)
                low, high = prior.cluster_bootstrap_ols_ci(
                    y, columns, [row["question_id"] for row in rows],
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED + 70_000 + delta * 100 + axis_index * 10 + metric_index,
                )
                selected_pairs = [
                    pair for pair in pairs_by_delta[delta]
                    if transition_axis[pair["transition"]] == axis
                ]
                values = [pair[f"delta_{metric}"] for pair in selected_pairs]
                if values:
                    paired_low, paired_high = prior.cluster_bootstrap_mean_ci(
                        values,
                        [pair["family"] for pair in selected_pairs],
                        draws=BOOTSTRAP_DRAWS,
                        seed=BOOTSTRAP_SEED + 71_000 + delta * 100 + axis_index * 10 + metric_index,
                    )
                else:
                    paired_low = paired_high = None
                by_transition = {}
                for transition in [name for name, value in transition_axis.items() if value == axis]:
                    transition_values = [
                        pair[f"delta_{metric}"] for pair in selected_pairs if pair["transition"] == transition
                    ]
                    by_transition[transition] = {
                        "n": len(transition_values),
                        "mean": float(np.mean(transition_values)) if transition_values else None,
                        "direction": (
                            "increase" if transition_values and np.mean(transition_values) > 1e-12
                            else "decrease" if transition_values and np.mean(transition_values) < -1e-12
                            else "equal_or_unavailable"
                        ),
                    }
                populated_directions = [
                    item["direction"] for item in by_transition.values()
                    if item["n"] >= 5 and item["direction"] != "equal_or_unavailable"
                ]
                output.append({
                    "delta": delta,
                    "axis": axis,
                    "axis_semantics": AXIS_NAMES[axis],
                    "metric": metric,
                    "axis_absent_n": len(absent),
                    "axis_present_n": len(present),
                    "axis_absent_mean": float(np.mean([row[metric] for row in absent])),
                    "axis_present_mean": float(np.mean([row[metric] for row in present])),
                    "adjusted_axis_coefficient": float(beta[1]),
                    "adjusted_axis_ci_low": float(low[1]),
                    "adjusted_axis_ci_high": float(high[1]),
                    "paired_transition_occurrences": len(values),
                    "paired_unique_families": len({pair["family"] for pair in selected_pairs}),
                    "paired_mean_delta": float(np.mean(values)) if values else None,
                    "paired_median_delta": float(np.median(values)) if values else None,
                    "paired_ci_low": paired_low,
                    "paired_ci_high": paired_high,
                    "non_tiny_transition_directions": json.dumps(populated_directions),
                    "multiple_transition_direction_agreement": (
                        len(populated_directions) >= 2 and len(set(populated_directions)) == 1
                    ),
                    "transition_details": json.dumps(by_transition, sort_keys=True),
                })
    return output


def all_valid_profile_rows(
    cohort_rows: Sequence[dict[str, Any]], valid_masks: dict[str, list[tuple[int, ...]]]
) -> list[dict[str, Any]]:
    converted = []
    for row in cohort_rows:
        masks = valid_masks[row["uid"]]
        profile = np.asarray(masks, dtype=float).mean(axis=0)
        converted.append({
            **row,
            **{f"layer_{layer:02d}": float(profile[layer]) for layer in LAYERS},
        })
    return layer_profile_rows(converted, delta=-1)


def metadata_audit(rows: Sequence[dict[str, Any]], sample_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    coverage = sum(bool(row.get("knowledge_points")) for row in rows)
    counts = Counter(len(row.get("knowledge_points") or []) for row in rows)
    unique = sorted({point for row in rows for point in (row.get("knowledge_points") or [])})
    vplus0 = [row for row in sample_rows if row["delta"] == 0]
    model_rows = []
    for metric_index, metric in enumerate(("normalized_centroid", "late_fraction", "has_late_reentry", "last_on")):
        y = [float(row[metric]) for row in vplus0]
        columns = [
            [float(row["knowledge_point_count"]) for row in vplus0],
            [float(row["min_positive_on"]) for row in vplus0],
        ]
        beta, r2 = prior.fit_ols(y, columns)
        low, high = prior.cluster_bootstrap_ols_ci(
            y, columns, [row["question_id"] for row in vplus0],
            draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED + 80_000 + metric_index,
        )
        model_rows.append({
            "metric": metric,
            "knowledge_point_count_coefficient": float(beta[1]),
            "ci_low": float(low[1]), "ci_high": float(high[1]), "r_squared": float(r2),
        })
    return {
        "axis_mapping": AXIS_NAMES,
        "axis_mapping_provenance": (
            "We-Math/We-Math2.0 released file dynamic_scheduling/verl/utils/dataset.py, "
            "DynamicScheduler docstring lines 303-311 at repository main"
        ),
        "official_dataset_card": (
            "We-Math/We-Math2.0-Pro revision c1d9f3ccea7361069f0442362e781d1ae7a28e94"
        ),
        "available_fields": [
            "question_id", "question", "image", "difficulty", "knowledge points", "idx", "answer"
        ],
        "knowledge_point_coverage": coverage,
        "eligible_records": len(rows),
        "knowledge_point_count_distribution": dict(sorted(counts.items())),
        "unique_knowledge_points": len(unique),
        "reasoning_type_analysis_status": "UNAVAILABLE",
        "reasoning_type_reason": (
            "The authoritative source exposes difficulty axes and knowledge-point lists but no "
            "independent categorical problem/reasoning/geometry subtype or reasoning-step field. "
            "Knowledge-point count is retained only as a descriptive covariate."
        ),
        "knowledge_point_count_adjusted_models_delta0": model_rows,
    }


def _svg(path: Path, body: str, title: str, *, width: int = 1000, height: int = 560) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>'
        f'{body}</svg>\n', encoding="utf-8"
    )


def svg_lines(path: Path, series: Sequence[tuple[str, Sequence[float], str]], title: str, *, y_label: str = "value") -> None:
    width, height = 1000, 560
    left, right, top, bottom = 72, 30, 50, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [float(value) for _, line, _ in series for value in line]
    ymin, ymax = min(values + [0.0]), max(values + [0.0])
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0
    def x(index: int, n: int) -> float:
        return left + index * plot_w / max(1, n - 1)
    def y(value: float) -> float:
        return top + (ymax - value) * plot_h / (ymax - ymin)
    parts = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>',
        f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(y_label)}</text>',
    ]
    for tick in range(6):
        value = ymin + (ymax - ymin) * tick / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy}" x2="{left+plot_w}" y2="{yy}" stroke="#eee"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.2f}</text>')
    for series_index, (name, line, color) in enumerate(series):
        points = " ".join(f"{x(i,len(line)):.1f},{y(float(value)):.1f}" for i, value in enumerate(line))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        lx, ly = left + 10 + (series_index % 4) * 215, height - 25 - (series_index // 4) * 20
        parts.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+22}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+28}" y="{ly}" font-family="sans-serif" font-size="11">{html.escape(name)}</text>')
    n = max(len(line) for _, line, _ in series)
    for index in range(0, n, max(1, n // 7)):
        xx = x(index, n)
        parts.append(f'<text x="{xx}" y="{top+plot_h+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{index}</text>')
    _svg(path, "".join(parts), title, width=width, height=height)


def svg_bars(path: Path, labels: Sequence[str], series: Sequence[tuple[str, Sequence[float], str]], title: str, *, y_label: str = "value") -> None:
    width, height = 1000, 560
    left, right, top, bottom = 75, 30, 50, 100
    plot_w, plot_h = width-left-right, height-top-bottom
    values = [float(value) for _, vals, _ in series for value in vals]
    ymin, ymax = min(values + [0.0]), max(values + [0.0])
    if math.isclose(ymin, ymax): ymax = ymin + 1.0
    def y(value: float) -> float: return top + (ymax-value)*plot_h/(ymax-ymin)
    base_y = y(0)
    group_w = plot_w / max(1, len(labels)); bar_w = group_w * 0.75 / max(1, len(series))
    parts = [f'<line x1="{left}" y1="{base_y}" x2="{left+plot_w}" y2="{base_y}" stroke="#222"/>']
    for group_index, label in enumerate(labels):
        center = left + (group_index + .5) * group_w
        parts.append(f'<text x="{center}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{html.escape(label)}</text>')
        for series_index, (_, vals, color) in enumerate(series):
            value = float(vals[group_index]); xx = center-group_w*.375+series_index*bar_w
            yy = y(value); top_y=min(yy,base_y); h=abs(base_y-yy)
            parts.append(f'<rect x="{xx}" y="{top_y}" width="{bar_w-2}" height="{max(1,h)}" fill="{color}"/>')
    for series_index, (name, _, color) in enumerate(series):
        xx=left+series_index*220; yy=height-25
        parts.append(f'<rect x="{xx}" y="{yy-11}" width="14" height="10" fill="{color}"/><text x="{xx+20}" y="{yy-2}" font-family="sans-serif" font-size="11">{html.escape(name)}</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(y_label)}</text>')
    _svg(path, "".join(parts), title, width=width, height=height)


def create_figures(
    profiles: dict[int, list[dict[str, Any]]],
    by_stratum: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
    layerwise: Sequence[dict[str, Any]],
    aplus_profiles: Sequence[dict[str, Any]],
) -> list[Path]:
    figure_dir = OUTPUT / "figures"
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")
    def profile_series(delta: int, group_type: str, labels: Sequence[str]) -> list[tuple[str, list[float], str]]:
        rows = profiles[delta]
        return [
            (label, [
                next(row["sample_balanced_access_probability"] for row in rows if row["group_type"] == group_type and row["group"] == label and row["layer"] == layer)
                for layer in LAYERS
            ], colors[index % len(colors)])
            for index, label in enumerate(labels)
        ]
    paths = [figure_dir / name for name in (
        "01_vplus_layer_access_profile_by_degree.svg",
        "02_vplus_layer_access_profile_by_stratum.svg",
        "03_vplus_centroid_by_stratum.svg",
        "04_vplus_latest_on_by_stratum.svg",
        "05_vplus_early_mid_late_fraction.svg",
        "06_vplus_on_segment_count.svg",
        "07_vplus_late_reentry_rate.svg",
        "08_vplus_paired_centroid_delta.svg",
        "09_vplus_paired_late_fraction_delta.svg",
        "10_vplus_layerwise_paired_delta_profiles.svg",
        "11_delta0_delta2_delta4_sensitivity.svg",
        "12_aplus_layer_access_profile_by_stratum.svg",
    )]
    svg_lines(paths[0], profile_series(0, "degree", ["0", "1", "2", "3"]), "V+ exact-min access profile by degree", y_label="access probability")
    svg_lines(paths[1], profile_series(0, "difficulty", DIFFICULTIES), "V+ exact-min access profile by stratum", y_label="access probability")
    exact = {row["group"]: row for row in by_stratum if row["delta"] == 0}
    svg_bars(paths[2], DIFFICULTIES, [("centroid", [exact[label]["centroid_mean"] for label in DIFFICULTIES], "#3182bd")], "V+ exact-min centroid by stratum", y_label="layer")
    svg_bars(paths[3], DIFFICULTIES, [("latest ON", [exact[label]["last_on_mean"] for label in DIFFICULTIES], "#756bb1")], "V+ exact-min latest ON by stratum", y_label="layer")
    svg_bars(paths[4], DIFFICULTIES, [
        ("early 0-8", [exact[label]["early_fraction_mean"] for label in DIFFICULTIES], "#9ecae1"),
        ("middle 9-18", [exact[label]["middle_fraction_mean"] for label in DIFFICULTIES], "#74c476"),
        ("late 19-27", [exact[label]["late_fraction_mean"] for label in DIFFICULTIES], "#fb6a4a"),
    ], "V+ exact-min access fractions by stratum", y_label="fraction")
    svg_bars(paths[5], DIFFICULTIES, [("ON segments", [exact[label]["on_segments_mean"] for label in DIFFICULTIES], "#31a354")], "V+ exact-min ON segments", y_label="segments")
    svg_bars(paths[6], DIFFICULTIES, [("late reentry", [exact[label]["has_late_reentry_mean"] for label in DIFFICULTIES], "#de2d26")], "V+ exact-min late reentry rate", y_label="fraction")

    transitions = [f"{source}->{target}" for source, target in TRANSITIONS]
    def paired_values(metric: str, delta: int = 0) -> list[float]:
        return [
            next((row["mean_delta"] for row in paired if row["delta"] == delta and row["scope"] == "same_family" and row["transition"] == transition and row["metric"] == metric and row.get("mean_delta") is not None), 0.0)
            for transition in transitions
        ]
    svg_bars(paths[7], transitions, [("harder-easier", paired_values("normalized_centroid"), "#3182bd")], "Paired normalized-centroid delta", y_label="delta")
    svg_bars(paths[8], transitions, [("harder-easier", paired_values("late_fraction"), "#de2d26")], "Paired late-fraction delta", y_label="delta")
    selected_transitions = ["base->x", "base->y", "base->z", "xy->xyz", "xz->xyz", "yz->xyz"]
    series = []
    for index, transition in enumerate(selected_transitions):
        values = [
            next((row["mean_harder_minus_easier_access_probability"] for row in layerwise if row["delta"] == 0 and row["transition"] == transition and row["layer"] == layer), 0.0)
            for layer in LAYERS
        ]
        series.append((transition, values, colors[index]))
    svg_lines(paths[9], series, "V+ paired layerwise harder-easier profiles", y_label="access-probability delta")
    sensitivity_series = []
    for index, delta in enumerate(DELTAS):
        rows = profiles[delta]
        line = [
            float(np.mean([row["sample_balanced_access_probability"] for row in rows if row["layer"] == layer and row["group_type"] == "degree"]))
            for layer in LAYERS
        ]
        sensitivity_series.append((f"min+{delta}", line, colors[index]))
    svg_lines(paths[10], sensitivity_series, "V+ route-set sensitivity", y_label="mean degree-profile access")
    aplus_series = []
    for index, label in enumerate(DIFFICULTIES):
        line = [
            next(row["sample_balanced_access_probability"] for row in aplus_profiles if row["group_type"] == "difficulty" and row["group"] == label and row["layer"] == layer)
            for layer in LAYERS
        ]
        aplus_series.append((label, line, colors[index]))
    svg_lines(paths[11], aplus_series, "A+ minimum correcting-route access profiles", y_label="access probability")
    return paths


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(summary: dict[str, Any], *, outcome: str, rationale: str) -> None:
    stratum = summary["vplus_exact_min_by_stratum"]
    degree = summary["vplus_exact_min_by_degree"]
    paired = summary["paired_aggregate"]
    axes = summary["axis_key_results"]
    sensitivity = summary["sensitivity_key_results"]
    outcome_names = {
        "PENDING": "Interpretation pending",
        "A": "later/deeper visual access with difficulty",
        "B": "earlier/front-loaded visual access with difficulty",
        "C": "difficulty-axis-specific placement",
        "D": "route placement varies, but not with difficulty",
        "E": "no stable placement structure",
    }
    report = f"""# WeMath2.0-Pro Visual-Access Placement Analysis

## Executive result

This read-only analysis passed every frozen-cache integrity gate and used all
raw valid routes rather than the capped predictor view. The primary cohort is
exactly **{summary['population']['vplus']} V+ samples** (FULL correct and exact
ALL-OFF wrong). Each sample contributes equal weight across all discovered
minimum-budget routes; `min+2` and `min+4` sets are frozen sensitivities.

Current classification: **Outcome {outcome} — {outcome_names[outcome]}**.
{rationale}

## Why this analysis was run

Phase 28 showed that difficulty does not robustly predict the *amount* of
positive direct visual access after separating ALL-OFF-solvable records. This
analysis asks the orthogonal question: with comparable ON budgets, do harder or
specific difficulty variants move direct visual access earlier, later, or into
separated re-access episodes?

## Sources, integrity, and semantics

- 4,544/4,544 authoritative raw records, 1,658,485 evaluated routes, and
  107,671 valid routes were checksum-verified and contract-validated.
- Exact FULL and ALL-OFF anchors reproduce phases 27/28; V+=428 and A+=1,263.
- `x = contextual complexity`, `y = visual complexity`, and `z = step
  complexity`, from the authors' released `DynamicScheduler` documentation.
- The source provides complete knowledge-point lists, but no independent
  categorical reasoning-type field. Knowledge-point count is descriptive only;
  categorical reasoning-type analysis is unavailable.

## Exact-minimum V+ placement by stratum

{_table(['stratum','n','min ON','centroid','last ON','late frac','segments','late reentry'], [[label, int(stratum[label]['n']), _fmt(stratum[label]['mean_min_positive_on'],2), _fmt(stratum[label]['centroid_mean'],2), _fmt(stratum[label]['last_on_mean'],2), _fmt(stratum[label]['late_fraction_mean']), _fmt(stratum[label]['on_segments_mean'],2), _fmt(stratum[label]['has_late_reentry_mean'])] for label in DIFFICULTIES])}

The profiles and summaries are sample-balanced: a sample with many discovered
minimum routes does not outweigh a sample with one route. `late` means layers
19--27; a late re-entry is a new ON segment beginning at layer 19 or later.

## Coarse degree summary and amount control

{_table(['degree','n','centroid','centroid CI','last ON','late frac','late reentry'], [[degree_label, int(degree[degree_label]['n']), _fmt(degree[degree_label]['normalized_centroid_mean']), f"[{_fmt(degree[degree_label]['normalized_centroid_ci_low'])}, {_fmt(degree[degree_label]['normalized_centroid_ci_high'])}]", _fmt(degree[degree_label]['last_on_mean'],2), _fmt(degree[degree_label]['late_fraction_mean']), _fmt(degree[degree_label]['has_late_reentry_mean'])] for degree_label in ('0','1','2','3')])}

The amount-adjusted tables fit normalized centroid, late fraction, and late
re-entry on categorical difficulty (and separately degree or each axis) plus
minimum positive ON. These are descriptive family-clustered models, not causal
effects. Predeclared ON-budget-bin tables retain counts and mark cells with
fewer than ten observations as sparse.

## Minimum versus near-minimum sensitivity

{_table(['route set','centroid','last ON','late frac','segments','late reentry'], [[f"min+{delta}", _fmt(sensitivity[str(delta)]['normalized_centroid_mean']), _fmt(sensitivity[str(delta)]['last_on_mean'],2), _fmt(sensitivity[str(delta)]['late_fraction_mean']), _fmt(sensitivity[str(delta)]['on_segments_mean'],2), _fmt(sensitivity[str(delta)]['has_late_reentry_mean'])] for delta in DELTAS])}

Patterns used for the scientific classification must survive all three route-set
definitions. Exact-min-only spikes are treated as fragile finite-search
structure.

## Same-family and same-image evidence

Across all supported V+ family transitions at exact minimum, the paired mean
normalized-centroid delta is **{_fmt(paired['normalized_centroid']['mean_delta'])}**
(95% CI **[{_fmt(paired['normalized_centroid']['ci_low'])},
{_fmt(paired['normalized_centroid']['ci_high'])}]**), the paired late-fraction
delta is **{_fmt(paired['late_fraction']['mean_delta'])}** (CI
**[{_fmt(paired['late_fraction']['ci_low'])}, {_fmt(paired['late_fraction']['ci_high'])}]**),
and the late-reentry delta is **{_fmt(paired['has_late_reentry']['mean_delta'])}**
(CI **[{_fmt(paired['has_late_reentry']['ci_low'])},
{_fmt(paired['has_late_reentry']['ci_high'])}]**). Transition-specific and
same-image results are preserved in the CSVs rather than pooled as independent
sample-layer observations.

## Axis-specific results

{_table(['axis','meaning','metric','paired N','paired delta','paired CI','multi-transition agreement'], [[axis, AXIS_NAMES[axis], metric, int(axes[axis][metric]['paired_transition_occurrences']), _fmt(axes[axis][metric]['paired_mean_delta']), f"[{_fmt(axes[axis][metric]['paired_ci_low'])}, {_fmt(axes[axis][metric]['paired_ci_high'])}]", axes[axis][metric]['multiple_transition_direction_agreement']] for axis in 'xyz' for metric in ('normalized_centroid','late_fraction','has_late_reentry')])}

An axis is called stable only if multiple non-tiny transitions agree, the
family-clustered paired aggregate supports the same direction, and the pattern
persists under min/min+2/min+4.

## Secondary A+ corrections

The A+ cohort contains {summary['population']['aplus']} FULL-wrong,
ALL-OFF-wrong samples with at least one positive correcting route. Its
minimum-budget correcting placement and per-layer profiles remain separate
because its search budget and estimand differ from V+ correctness preservation.

## Direct answers required by the plan

1. **Does difficulty change placement?** {summary['direct_answers']['1']}
2. **Later/re-access heavy or earlier/front-loaded?** {summary['direct_answers']['2']}
3. **After controlling minimum ON?** {summary['direct_answers']['3']}
4. **Across min/min+2/min+4?** {summary['direct_answers']['4']}
5. **Same-family paired?** {summary['direct_answers']['5']}
6. **Same-image control?** {summary['direct_answers']['6']}
7. **Reproducible axes?** {summary['direct_answers']['7']}
8. **Reasoning-type metadata?** {summary['direct_answers']['8']}
9. **A+ placement?** {summary['direct_answers']['9']}
10. **Schedule-router motivation?** {summary['direct_answers']['10']}

## Claim boundary

These are discovered finite-MCTS route schedules. They do not prove that any
specific layer is causally necessary. Late ON is direct visual access late in
the decoder; it does not by itself establish semantic re-grounding,
verification, or backtracking.

Outcome {outcome} — {outcome_names[outcome]}
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def direct_answers_for(outcome: str, summary: dict[str, Any]) -> dict[str, str]:
    paired = summary["paired_aggregate"]
    same = summary["same_image_aggregate"]
    if outcome == "PENDING":
        return {str(index): "Pending aggregate interpretation." for index in range(1, 11)}
    if outcome == "A":
        global_answer = "Yes: the robust summaries support a later/deeper schedule as difficulty increases."
        direction = "Higher difficulty shifts access later and/or increases late re-entry; the opposite front-loaded account is unsupported."
    elif outcome == "B":
        global_answer = "Yes: the robust summaries support an earlier/front-loaded schedule as difficulty increases."
        direction = "Higher difficulty shifts access earlier and reduces late access/re-entry; the later-vision account is unsupported."
    elif outcome == "C":
        global_answer = "No single aggregate trend is stable, but at least one verified difficulty axis has a reproducible placement shift."
        direction = "Direction depends on the authoritative difficulty axis rather than coarse degree."
    elif outcome == "D":
        global_answer = "No stable global or axis-specific difficulty relationship survives the primary controls."
        direction = "Schedules are heterogeneous across samples, but neither consistently later nor earlier with difficulty."
    else:
        global_answer = "No stable placement relationship is supported by the finite cache."
        direction = "Apparent directions are unstable across exact-min and near-min route sets."
    centroid = paired.get("normalized_centroid", {})
    same_centroid = same.get("normalized_centroid", {})
    aplus = summary["aplus_degree_comparison"]
    robust = summary["route_set_stability"]
    return {
        "1": global_answer,
        "2": direction,
        "3": (
            "See the categorical/degree/axis models adjusted for minimum ON and the predeclared ON-budget bins; "
            + ("the reported relationship remains." if outcome in ("A", "B", "C") else "they do not rescue a stable relationship.")
        ),
        "4": (
            f"The median sample profile L1 change from exact-min to min+4 is {robust['delta0_to_delta4_profile_l1_median']:.3f}. "
            + ("The classification is stable across all three sets." if outcome in ("A", "B", "C", "D") else "The classification is instability/no-structure rather than a directional effect.")
        ),
        "5": (
            f"The aggregate family-paired normalized-centroid delta is {centroid.get('mean_delta', float('nan')):.3f} "
            f"with CI [{centroid.get('ci_low', float('nan')):.3f}, {centroid.get('ci_high', float('nan')):.3f}]."
        ),
        "6": (
            f"The same-image aggregate uses {same_centroid.get('n', 0)} transitions; normalized-centroid delta is "
            f"{same_centroid.get('mean_delta', float('nan')):.3f} with CI "
            f"[{same_centroid.get('ci_low', float('nan')):.3f}, {same_centroid.get('ci_high', float('nan')):.3f}]."
            if same_centroid else "No same-image V+ transition aggregate was available."
        ),
        "7": (
            "At least one axis satisfies the predeclared multi-transition, paired, amount-controlled, and sensitivity requirements."
            if outcome == "C" else
            "No axis satisfies all predeclared multi-transition, paired, amount-controlled, and sensitivity requirements."
        ),
        "8": (
            "The official axes are contextual/visual/step complexity and knowledge-point lists are complete, but no independent "
            "categorical reasoning-type field exists; reasoning-type analysis is therefore unavailable."
        ),
        "9": (
            f"A+ remains secondary: degree-3 minus degree-0 exact-min normalized centroid is "
            f"{aplus['degree3_minus_degree0_normalized_centroid']:.3f}, and late-fraction difference is "
            f"{aplus['degree3_minus_degree0_late_fraction']:.3f}; see the stratum/profile tables for nonmonotonicity."
        ),
        "10": (
            "No schedule router is motivated by difficulty alone; any future predictor would need other input properties and independent execution validation."
            if outcome in ("D", "E") else
            "The result can motivate a bounded schedule-prediction hypothesis, but it is not evidence that such a router is learnable or beneficial."
        ),
    }


def finalize_existing(*, outcome: str, rationale: str) -> None:
    manifest_path = OUTPUT / "analysis_manifest.json"
    summary_path = OUTPUT / "analysis_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["output_hashes"].items():
        if relative in (str(summary_path.relative_to(PROJECT)), str(REPORT.relative_to(PROJECT))):
            continue
        if sha256_file(PROJECT / relative) != expected:
            raise RuntimeError(f"cannot finalize after output drift: {relative}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["outcome"] = outcome
    summary["outcome_rationale"] = rationale
    summary["direct_answers"] = direct_answers_for(outcome, summary)
    write_json(summary_path, summary)
    write_report(summary, outcome=outcome, rationale=rationale)
    manifest["outcome"] = outcome
    manifest["outcome_rationale"] = rationale
    manifest["output_hashes"] = {
        relative: sha256_file(PROJECT / relative)
        for relative in manifest["output_hashes"]
    }
    write_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", choices=("PENDING", "A", "B", "C", "D", "E"), default="PENDING")
    parser.add_argument("--outcome-rationale", default="Interpretation pending aggregate review.")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        if args.outcome == "PENDING":
            raise ValueError("--finalize-only requires a non-PENDING outcome")
        finalize_existing(outcome=args.outcome, rationale=args.outcome_rationale)
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, valid_masks, integrity = load_valid_masks()
    vplus = [row for row in rows if row.get("visual_regime") == "V+"]
    aplus = [row for row in rows if row.get("correction_regime") == "A+"]

    vplus_samples = [
        sample_route_summary(row=row, masks=valid_masks[row["uid"]], cohort="V+", delta=delta)
        for delta in DELTAS for row in vplus
    ]
    aplus_samples = [
        sample_route_summary(row=row, masks=valid_masks[row["uid"]], cohort="A+", delta=0)
        for row in aplus
    ]
    profiles = {
        delta: layer_profile_rows([row for row in vplus_samples if row["delta"] == delta], delta=delta)
        for delta in DELTAS
    }
    by_stratum = [
        item
        for delta in DELTAS
        for item in grouped_metric_rows(
            [row for row in vplus_samples if row["delta"] == delta], group_type="difficulty"
        )
    ]
    by_degree = [
        item
        for delta in DELTAS
        for item in grouped_metric_rows(
            [row for row in vplus_samples if row["delta"] == delta], group_type="degree"
        )
    ]
    adjusted = fit_adjusted_models(vplus_samples)
    bins = budget_bin_rows(vplus_samples)
    paired, layerwise, same_image, pairs_by_delta = paired_analyses(vplus_samples)
    axes = axis_summary_rows(vplus_samples, pairs_by_delta)
    aplus_by_stratum = grouped_metric_rows(aplus_samples, group_type="difficulty")
    aplus_by_degree = grouped_metric_rows(aplus_samples, group_type="degree")
    aplus_profiles = layer_profile_rows(aplus_samples, delta=0)
    all_valid = all_valid_profile_rows(vplus, valid_masks)
    metadata = metadata_audit(rows, vplus_samples)

    exact_by_uid = {row["uid"]: row for row in vplus_samples if row["delta"] == 0}
    plus2_by_uid = {row["uid"]: row for row in vplus_samples if row["delta"] == 2}
    plus4_by_uid = {row["uid"]: row for row in vplus_samples if row["delta"] == 4}
    def profile_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        return float(sum(abs(left[f"layer_{layer:02d}"] - right[f"layer_{layer:02d}"]) for layer in LAYERS))
    d02 = [profile_distance(exact_by_uid[uid], plus2_by_uid[uid]) for uid in exact_by_uid]
    d04 = [profile_distance(exact_by_uid[uid], plus4_by_uid[uid]) for uid in exact_by_uid]
    route_set_stability = {
        "delta0_to_delta2_profile_l1_mean": float(np.mean(d02)),
        "delta0_to_delta2_profile_l1_median": float(np.median(d02)),
        "delta0_to_delta4_profile_l1_mean": float(np.mean(d04)),
        "delta0_to_delta4_profile_l1_median": float(np.median(d04)),
        "fraction_exact_profile_unchanged_at_delta2": float(np.mean(np.asarray(d02) <= 1e-12)),
        "fraction_exact_profile_unchanged_at_delta4": float(np.mean(np.asarray(d04) <= 1e-12)),
    }

    exact_stratum = {row["group"]: row for row in by_stratum if row["delta"] == 0}
    exact_degree = {row["group"]: row for row in by_degree if row["delta"] == 0}
    paired_aggregate = {
        metric: next(
            row for row in paired
            if row["delta"] == 0 and row["scope"] == "all_supported_transitions"
            and row["transition"] == "aggregate" and row["metric"] == metric
        ) for metric in PAIRED_METRICS
    }
    same_image_aggregate = {
        metric: next(
            (row for row in same_image
             if row["delta"] == 0 and row["scope"] == "all_supported_same_image_transitions"
             and row["transition"] == "aggregate" and row["metric"] == metric),
            {},
        ) for metric in PAIRED_METRICS
    }
    axis_key = {
        axis: {
            metric: next(row for row in axes if row["delta"] == 0 and row["axis"] == axis and row["metric"] == metric)
            for metric in ("normalized_centroid", "late_fraction", "has_late_reentry")
        } for axis in "xyz"
    }
    sensitivity = {}
    for delta in DELTAS:
        members = [row for row in vplus_samples if row["delta"] == delta]
        sensitivity[str(delta)] = {
            metric + "_mean": float(np.mean([row[metric] for row in members]))
            for metric in ("normalized_centroid", "last_on", "late_fraction", "on_segments", "has_late_reentry")
        }
    aplus_degree = {row["group"]: row for row in aplus_by_degree}
    aplus_comparison = {
        "degree3_minus_degree0_normalized_centroid": (
            aplus_degree["3"]["normalized_centroid_mean"] - aplus_degree["0"]["normalized_centroid_mean"]
        ),
        "degree3_minus_degree0_late_fraction": (
            aplus_degree["3"]["late_fraction_mean"] - aplus_degree["0"]["late_fraction_mean"]
        ),
    }

    paths = {
        "profile0": OUTPUT / "vplus_minroute_layer_profiles.csv",
        "profile2": OUTPUT / "vplus_nearmin_layer_profiles_delta2.csv",
        "profile4": OUTPUT / "vplus_nearmin_layer_profiles_delta4.csv",
        "sample": OUTPUT / "vplus_placement_metrics_by_sample.csv",
        "stratum": OUTPUT / "vplus_placement_by_stratum.csv",
        "degree": OUTPUT / "vplus_placement_by_degree.csv",
        "adjusted": OUTPUT / "vplus_amount_adjusted_models.csv",
        "bins": OUTPUT / "vplus_budget_bin_analysis.csv",
        "paired": OUTPUT / "vplus_family_paired_transitions.csv",
        "layerwise": OUTPUT / "vplus_family_layerwise_deltas.csv",
        "same_image": OUTPUT / "vplus_same_image_analysis.csv",
        "axes": OUTPUT / "axis_placement_summary.csv",
        "aplus_stratum": OUTPUT / "aplus_placement_by_stratum.csv",
        "aplus_profiles": OUTPUT / "aplus_layer_profiles.csv",
        "all_valid": OUTPUT / "vplus_all_valid_layer_profiles_secondary.csv",
        "metadata": OUTPUT / "authoritative_metadata_audit.json",
    }
    payloads: dict[str, Sequence[dict[str, Any]]] = {
        "profile0": profiles[0], "profile2": profiles[2], "profile4": profiles[4],
        "sample": vplus_samples, "stratum": by_stratum, "degree": by_degree,
        "adjusted": adjusted, "bins": bins, "paired": paired, "layerwise": layerwise,
        "same_image": same_image, "axes": axes, "aplus_stratum": aplus_by_stratum,
        "aplus_profiles": aplus_profiles, "all_valid": all_valid,
    }
    for key, payload in payloads.items():
        write_csv(paths[key], payload)
    write_json(paths["metadata"], metadata)
    figures = create_figures(profiles, by_stratum, paired, layerwise, aplus_profiles)

    summary = {
        "schema_version": "wemath2pro_visual_access_placement_v1",
        "status": "PASS",
        "outcome": args.outcome,
        "outcome_rationale": args.outcome_rationale,
        "integrity": integrity,
        "population": {"eligible": len(rows), "vplus": len(vplus), "aplus": len(aplus)},
        "axis_semantics": AXIS_NAMES,
        "metadata_audit": metadata,
        "route_set_definitions": {"deltas": list(DELTAS), "sample_balanced": True, "all_raw_valid_routes": True},
        "vplus_exact_min_by_stratum": exact_stratum,
        "vplus_exact_min_by_degree": exact_degree,
        "paired_aggregate": paired_aggregate,
        "same_image_aggregate": same_image_aggregate,
        "axis_key_results": axis_key,
        "sensitivity_key_results": sensitivity,
        "route_set_stability": route_set_stability,
        "aplus_degree_comparison": aplus_comparison,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "confidence": 0.95, "cluster": "question_id seed family"},
        "claim_boundaries": [
            "Discovered minimum-budget routes are finite-MCTS observations, not exhaustive causal necessities.",
            "Late ON denotes direct visual access late in depth, not semantic re-grounding.",
            "V+ and A+ have different estimands and search-budget regimes and are not pooled.",
        ],
    }
    summary["direct_answers"] = direct_answers_for(args.outcome, summary)
    summary_path = OUTPUT / "analysis_summary.json"
    write_json(summary_path, summary)
    write_report(summary, outcome=args.outcome, rationale=args.outcome_rationale)

    output_paths = [*paths.values(), summary_path, REPORT, *figures]
    analysis_manifest = {
        "schema_version": "wemath2pro_visual_access_placement_analysis_manifest_v1",
        "status": "PASS",
        "outcome": args.outcome,
        "outcome_rationale": args.outcome_rationale,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "plan_sha256": sha256_file(PROJECT / "plans/motivation_check3.md"),
        "source_integrity": integrity,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "axis_mapping": AXIS_NAMES,
        "axis_mapping_provenance_url": "https://github.com/We-Math/We-Math2.0/blob/main/dynamic_scheduling/verl/utils/dataset.py#L303-L311",
        "local_dataset_card_sha256": sha256_file(
            Path("/data/dataset/huggingface/hub/datasets--We-Math--We-Math2.0-Pro/snapshots/c1d9f3ccea7361069f0442362e781d1ae7a28e94/README.md")
        ),
        "output_hashes": {
            str(path.relative_to(PROJECT)): sha256_file(path) for path in output_paths
        },
    }
    write_json(OUTPUT / "analysis_manifest.json", analysis_manifest)


if __name__ == "__main__":
    main()
