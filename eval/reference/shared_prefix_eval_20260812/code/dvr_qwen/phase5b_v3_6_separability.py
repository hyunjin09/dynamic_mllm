"""Cache-only separability diagnostics for Phase 5B v3.6."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

import torch

from dvr_qwen.route_selector import (
    NUM_LAYERS,
    candidate_on,
    candidate_rank,
    early_mid_late_counts,
    group_rows_by_id,
    route_mask_from_layers,
    transition_count,
)


REFERENCE_BUCKETS = ("safe_switch", "regression", "cost_only_preserve")
HARM_BUCKETS = ("regression", "cost_only_preserve")


def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _as_bool(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key, False))


def candidate_bucket(row: dict[str, Any]) -> str:
    """Return the v3.6 separability class for one route candidate."""

    if _as_bool(row, "selected_full_qwen_fallback"):
        return "fallback"
    if _as_bool(row, "safe_switch"):
        return "safe_switch"
    if _as_bool(row, "regression") or _as_bool(row, "is_regression") or _as_bool(row, "full_correct_regression"):
        return "regression"
    if _as_bool(row, "cost_only_preserve"):
        return "cost_only_preserve"
    if _as_bool(row, "preserve") or _as_bool(row, "full_correct_preserved"):
        return "preserve_other"
    return "other"


def binary_safe_vs_harm_label(row: dict[str, Any]) -> int | None:
    bucket = candidate_bucket(row)
    if bucket == "safe_switch":
        return 1
    if bucket in HARM_BUCKETS:
        return 0
    return None


def quantiles(values: list[float], points: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)) -> list[float | None]:
    if not values:
        return [None for _ in points]
    tensor = torch.tensor(values, dtype=torch.float32)
    return [float(item) for item in torch.quantile(tensor, torch.tensor(points, dtype=torch.float32)).tolist()]


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": None if not values else mean(float(value) for value in values),
        "quantiles_0_25_50_75_90_100": quantiles(values),
    }


def route_metrics(row: dict[str, Any], *, num_layers: int = NUM_LAYERS) -> dict[str, Any]:
    layers = [int(layer) for layer in row.get("layers_one_based", [])]
    mask = route_mask_from_layers(layers, num_layers=num_layers)
    early, middle, late = early_mid_late_counts(mask)
    return {
        "on_count": candidate_on(row),
        "transition_count": transition_count(mask),
        "candidate_rank": candidate_rank(row),
        "candidate_index": int(row.get("candidate_index", -1)),
        "early_on_count": early,
        "middle_on_count": middle,
        "late_on_count": late,
        "first_on": int(layers[0]) if layers else 0,
        "last_on": int(layers[-1]) if layers else 0,
    }


def annotate_rows_for_audit(
    rows: list[dict[str, Any]],
    scores: torch.Tensor,
    *,
    split_name: str,
    num_layers: int = NUM_LAYERS,
) -> list[dict[str, Any]]:
    """Attach bucket, selector score, fallback gap, and route metrics."""

    if int(scores.numel()) != len(rows):
        raise ValueError(f"scores length {scores.numel()} != rows length {len(rows)}")
    fallback_score_by_id: dict[str, float] = {}
    for score, row in zip(scores.float().tolist(), rows, strict=True):
        if _as_bool(row, "selected_full_qwen_fallback"):
            fallback_score_by_id[str(row["id"])] = float(score)
    out: list[dict[str, Any]] = []
    for score, row in zip(scores.float().tolist(), rows, strict=True):
        sample_id = str(row["id"])
        if sample_id not in fallback_score_by_id:
            raise ValueError(f"group {sample_id} is missing fallback score")
        selector_score = float(score)
        out.append(
            {
                **row,
                "split_name": str(split_name),
                "v3_6_bucket": candidate_bucket(row),
                "v3_6_selector_score": selector_score,
                "v3_6_score_gap_to_fallback": selector_score - fallback_score_by_id[sample_id],
                **{f"route_{key}": value for key, value in route_metrics(row, num_layers=num_layers).items()},
            }
        )
    return out


def summarize_bucket_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "v3_6_selector_score",
        "v3_6_score_gap_to_fallback",
        "route_on_count",
        "route_transition_count",
        "route_candidate_rank",
        "route_candidate_index",
        "route_early_on_count",
        "route_middle_on_count",
        "route_late_on_count",
    ]
    by_bucket: dict[str, dict[str, Any]] = {}
    for bucket in sorted({candidate_bucket(row) for row in rows}):
        bucket_rows = [row for row in rows if candidate_bucket(row) == bucket]
        by_bucket[bucket] = {
            "num_rows": len(bucket_rows),
            "num_groups": len({str(row["id"]) for row in bucket_rows}),
            "by_benchmark": dict(sorted(Counter(str(row.get("benchmark", "unknown")) for row in bucket_rows).items())),
            "numeric": {
                key: _numeric_summary([_as_float(row, key) for row in bucket_rows if row.get(key) is not None])
                for key in numeric_keys
            },
        }
    return {
        "num_rows": len(rows),
        "num_groups": len({str(row["id"]) for row in rows}),
        "bucket_counts": dict(sorted(Counter(candidate_bucket(row) for row in rows).items())),
        "by_bucket": by_bucket,
    }


def layer_frequency_by_bucket(rows: list[dict[str, Any]], *, num_layers: int = NUM_LAYERS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bucket in sorted({candidate_bucket(row) for row in rows}):
        bucket_rows = [row for row in rows if candidate_bucket(row) == bucket]
        if not bucket_rows:
            continue
        masks = torch.stack(
            [
                route_mask_from_layers([int(layer) for layer in row.get("layers_one_based", [])], num_layers=num_layers)
                for row in bucket_rows
            ]
        )
        freq = masks.float().mean(dim=0)
        top_layers = sorted(
            [{"layer": idx + 1, "frequency": float(value)} for idx, value in enumerate(freq.tolist())],
            key=lambda item: (-float(item["frequency"]), int(item["layer"])),
        )[:8]
        out[bucket] = {
            "num_rows": len(bucket_rows),
            "mean_on_count": float(masks.sum(dim=1).float().mean().item()),
            "top_layers": top_layers,
            "layer_frequencies": [float(value) for value in freq.tolist()],
        }
    return out


def _features_for_bucket(features: torch.Tensor, rows: list[dict[str, Any]], bucket: str) -> torch.Tensor:
    indices = [idx for idx, row in enumerate(rows) if candidate_bucket(row) == bucket]
    if not indices:
        return torch.empty((0, int(features.shape[1])), dtype=torch.float32)
    return features[torch.tensor(indices, dtype=torch.long)].float()


def _prototype_map(reference_features: torch.Tensor, reference_rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    prototypes: dict[str, torch.Tensor] = {}
    for bucket in REFERENCE_BUCKETS:
        bucket_features = _features_for_bucket(reference_features, reference_rows, bucket)
        if int(bucket_features.shape[0]):
            prototypes[bucket] = bucket_features.mean(dim=0)
    return prototypes


def prototype_distance_audit(
    *,
    reference_features: torch.Tensor,
    reference_rows: list[dict[str, Any]],
    eval_features: torch.Tensor,
    eval_rows: list[dict[str, Any]],
    reference_name: str,
    eval_name: str,
) -> dict[str, Any]:
    """Compare eval candidates to fit prototypes in standardized feature space."""

    if int(reference_features.shape[0]) != len(reference_rows):
        raise ValueError("reference feature rows do not match reference rows")
    if int(eval_features.shape[0]) != len(eval_rows):
        raise ValueError("eval feature rows do not match eval rows")
    prototypes = _prototype_map(reference_features.float(), reference_rows)
    if "safe_switch" not in prototypes:
        raise ValueError("reference rows must include at least one safe_switch")

    nearest_counts: dict[str, Counter[str]] = defaultdict(Counter)
    distance_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    safe_margin_values: list[float] = []
    for idx, row in enumerate(eval_rows):
        bucket = candidate_bucket(row)
        if bucket not in REFERENCE_BUCKETS:
            continue
        feature = eval_features[idx].float()
        distances = {
            proto_bucket: float(torch.linalg.vector_norm(feature - prototype).item())
            for proto_bucket, prototype in prototypes.items()
        }
        nearest = min(distances, key=lambda key: (distances[key], key))
        nearest_counts[bucket][nearest] += 1
        for proto_bucket, distance in distances.items():
            distance_values[bucket][f"distance_to_{proto_bucket}"].append(distance)
        harm_distances = [distances[name] for name in HARM_BUCKETS if name in distances]
        if bucket == "safe_switch" and harm_distances:
            safe_margin_values.append(distances["safe_switch"] - min(harm_distances))

    return {
        "reference_name": str(reference_name),
        "eval_name": str(eval_name),
        "prototype_buckets": sorted(prototypes),
        "bucket_counts": dict(sorted(Counter(candidate_bucket(row) for row in eval_rows).items())),
        "nearest_prototype_counts_by_bucket": {
            bucket: dict(sorted(counts.items())) for bucket, counts in sorted(nearest_counts.items())
        },
        "distance_summary_by_bucket": {
            bucket: {key: _numeric_summary(values) for key, values in sorted(metrics.items())}
            for bucket, metrics in sorted(distance_values.items())
        },
        "safe_distance_margin_to_nearest_harm": _numeric_summary(safe_margin_values),
        "safe_rows_closer_to_safe_than_harm": sum(1 for value in safe_margin_values if value < 0.0),
        "safe_rows_with_distance_margin": len(safe_margin_values),
    }


def nearest_neighbor_audit(
    *,
    reference_features: torch.Tensor,
    reference_rows: list[dict[str, Any]],
    eval_features: torch.Tensor,
    eval_rows: list[dict[str, Any]],
    reference_name: str,
    eval_name: str,
) -> dict[str, Any]:
    """Compare each eval row to nearest fit examples by candidate bucket."""

    reference_by_bucket = {
        bucket: _features_for_bucket(reference_features.float(), reference_rows, bucket) for bucket in REFERENCE_BUCKETS
    }
    nearest_counts: dict[str, Counter[str]] = defaultdict(Counter)
    safe_margin_values: list[float] = []
    for idx, row in enumerate(eval_rows):
        bucket = candidate_bucket(row)
        if bucket not in REFERENCE_BUCKETS:
            continue
        feature = eval_features[idx].float().view(1, -1)
        distances: dict[str, float] = {}
        for ref_bucket, ref_features in reference_by_bucket.items():
            if int(ref_features.shape[0]) == 0:
                continue
            values = torch.cdist(feature, ref_features).view(-1)
            distances[ref_bucket] = float(values.min().item())
        if not distances:
            continue
        nearest = min(distances, key=lambda key: (distances[key], key))
        nearest_counts[bucket][nearest] += 1
        harm_distances = [distances[name] for name in HARM_BUCKETS if name in distances]
        if bucket == "safe_switch" and "safe_switch" in distances and harm_distances:
            safe_margin_values.append(distances["safe_switch"] - min(harm_distances))
    return {
        "reference_name": str(reference_name),
        "eval_name": str(eval_name),
        "nearest_neighbor_counts_by_bucket": {
            bucket: dict(sorted(counts.items())) for bucket, counts in sorted(nearest_counts.items())
        },
        "safe_nearest_margin_to_nearest_harm": _numeric_summary(safe_margin_values),
        "safe_rows_closer_to_fit_safe_than_fit_harm": sum(1 for value in safe_margin_values if value < 0.0),
        "safe_rows_with_nearest_margin": len(safe_margin_values),
    }


def _binary_metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    if len(predictions) != len(labels):
        raise ValueError("prediction and label counts differ")
    tp = sum(1 for pred, label in zip(predictions, labels, strict=True) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(predictions, labels, strict=True) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(predictions, labels, strict=True) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels, strict=True) if pred == 0 and label == 1)
    pos = tp + fn
    neg = tn + fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / pos if pos else 0.0
    specificity = tn / neg if neg else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {
        "num_examples": len(labels),
        "positive": pos,
        "negative": neg,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 * (recall + specificity) if labels else 0.0,
    }


def _candidate_thresholds(values: torch.Tensor, *, max_thresholds: int = 64) -> list[float]:
    unique = torch.unique(values.float()).sort().values
    if int(unique.numel()) <= 1:
        return [float(unique[0].item())] if int(unique.numel()) else [0.0]
    if int(unique.numel()) <= int(max_thresholds):
        mids = (unique[:-1] + unique[1:]) * 0.5
        return [float(item) for item in mids.tolist()]
    points = torch.linspace(0.02, 0.98, int(max_thresholds))
    return [float(item) for item in torch.quantile(values.float(), points).unique().tolist()]


def fit_single_feature_threshold_rules(
    features: torch.Tensor,
    rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    max_rules: int = 20,
    max_thresholds: int = 64,
) -> list[dict[str, Any]]:
    """Fit simple one-feature safe-vs-harm threshold rules on the provided rows."""

    if int(features.shape[0]) != len(rows):
        raise ValueError("feature rows do not match metadata rows")
    if int(features.shape[1]) != len(feature_names):
        raise ValueError("feature columns do not match feature_names")
    selected_indices: list[int] = []
    labels: list[int] = []
    for idx, row in enumerate(rows):
        label = binary_safe_vs_harm_label(row)
        if label is not None:
            selected_indices.append(idx)
            labels.append(label)
    if not selected_indices:
        return []
    selected_features = features[torch.tensor(selected_indices, dtype=torch.long)].float()
    label_tensor = torch.tensor(labels, dtype=torch.long)
    rules: list[dict[str, Any]] = []
    for feature_idx, feature_name in enumerate(feature_names):
        values = selected_features[:, feature_idx]
        if not bool(torch.isfinite(values).all().item()):
            continue
        for threshold in _candidate_thresholds(values, max_thresholds=max_thresholds):
            for direction in (">=", "<="):
                if direction == ">=":
                    preds = (values >= float(threshold)).long()
                else:
                    preds = (values <= float(threshold)).long()
                metrics = _binary_metrics(preds.tolist(), label_tensor.tolist())
                rules.append(
                    {
                        "feature_index": int(feature_idx),
                        "feature_name": str(feature_name),
                        "threshold": float(threshold),
                        "direction": direction,
                        **{f"fit_{key}": value for key, value in metrics.items()},
                    }
                )
    rules.sort(
        key=lambda rule: (
            float(rule["fit_balanced_accuracy"]),
            float(rule["fit_recall"]),
            float(rule["fit_specificity"]),
            float(rule["fit_precision"]),
            -int(rule["feature_index"]),
        ),
        reverse=True,
    )
    return rules[: int(max_rules)]


def evaluate_single_feature_rule(
    features: torch.Tensor,
    rows: list[dict[str, Any]],
    rule: dict[str, Any],
) -> dict[str, Any]:
    selected_indices: list[int] = []
    labels: list[int] = []
    for idx, row in enumerate(rows):
        label = binary_safe_vs_harm_label(row)
        if label is not None:
            selected_indices.append(idx)
            labels.append(label)
    if not selected_indices:
        return _binary_metrics([], [])
    values = features[torch.tensor(selected_indices, dtype=torch.long), int(rule["feature_index"])].float()
    if str(rule["direction"]) == ">=":
        preds = (values >= float(rule["threshold"])).long()
    else:
        preds = (values <= float(rule["threshold"])).long()
    return _binary_metrics(preds.tolist(), labels)


def group_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contrasts: list[dict[str, Any]] = []
    for sample_id, group in group_rows_by_id(rows).items():
        fallback = [row for row in group if candidate_bucket(row) == "fallback"]
        safe_rows = [row for row in group if candidate_bucket(row) == "safe_switch"]
        if len(fallback) != 1 or not safe_rows:
            continue
        fallback_row = fallback[0]
        regression_rows = [row for row in group if candidate_bucket(row) == "regression"]
        cost_rows = [row for row in group if candidate_bucket(row) == "cost_only_preserve"]
        non_safe_rows = [row for row in group if candidate_bucket(row) not in {"fallback", "safe_switch"}]

        def best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
            if not candidates:
                return None
            return max(
                candidates,
                key=lambda row: (
                    _as_float(row, "v3_6_selector_score"),
                    _as_float(row, "delta_q", _as_float(row, "delta_score")),
                    -candidate_on(row),
                    -candidate_rank(row),
                ),
            )

        best_safe = best(safe_rows)
        best_regression = best(regression_rows)
        best_cost = best(cost_rows)
        best_non_safe = best(non_safe_rows)
        all_nonfallback = [row for row in group if candidate_bucket(row) != "fallback"]
        top_nonfallback = best(all_nonfallback)
        fallback_score = _as_float(fallback_row, "v3_6_selector_score")
        best_safe_score = _as_float(best_safe or {}, "v3_6_selector_score")
        best_non_safe_score = None if best_non_safe is None else _as_float(best_non_safe, "v3_6_selector_score")
        contrasts.append(
            {
                "id": sample_id,
                "benchmark": group[0].get("benchmark"),
                "safe_switch_count": len(safe_rows),
                "regression_count": len(regression_rows),
                "cost_only_preserve_count": len(cost_rows),
                "best_safe_score": best_safe_score,
                "fallback_score": fallback_score,
                "best_safe_minus_fallback": best_safe_score - fallback_score,
                "best_safe_minus_best_regression": None
                if best_regression is None
                else best_safe_score - _as_float(best_regression, "v3_6_selector_score"),
                "best_safe_minus_best_cost_only": None
                if best_cost is None
                else best_safe_score - _as_float(best_cost, "v3_6_selector_score"),
                "best_safe_minus_best_non_safe": None
                if best_non_safe_score is None
                else best_safe_score - best_non_safe_score,
                "top_nonfallback_bucket": None if top_nonfallback is None else candidate_bucket(top_nonfallback),
                "best_safe_above_fallback": best_safe_score > fallback_score,
                "top_nonfallback_is_safe": top_nonfallback is best_safe,
                "best_safe_on": candidate_on(best_safe or {}),
                "best_safe_transition_count": int((best_safe or {}).get("transition_count", 0)),
                "best_safe_layers_one_based": list((best_safe or {}).get("layers_one_based", [])),
            }
        )
    return contrasts


def summarize_group_contrasts(contrasts: list[dict[str, Any]]) -> dict[str, Any]:
    by_benchmark: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in contrasts:
        benchmark = str(row.get("benchmark", "unknown"))
        by_benchmark[benchmark]["safe_switch_groups"] += 1
        by_benchmark[benchmark]["best_safe_above_fallback"] += int(bool(row.get("best_safe_above_fallback", False)))
        by_benchmark[benchmark]["top_nonfallback_is_safe"] += int(bool(row.get("top_nonfallback_is_safe", False)))
    gap_keys = [
        "best_safe_minus_fallback",
        "best_safe_minus_best_regression",
        "best_safe_minus_best_cost_only",
        "best_safe_minus_best_non_safe",
    ]
    return {
        "safe_switch_groups": len(contrasts),
        "best_safe_above_fallback_groups": sum(int(bool(row.get("best_safe_above_fallback", False))) for row in contrasts),
        "top_nonfallback_is_safe_groups": sum(int(bool(row.get("top_nonfallback_is_safe", False))) for row in contrasts),
        "top_nonfallback_bucket_counts": dict(sorted(Counter(str(row.get("top_nonfallback_bucket")) for row in contrasts).items())),
        "by_benchmark": {key: dict(value) for key, value in sorted(by_benchmark.items())},
        "gap_summaries": {
            key: _numeric_summary([float(row[key]) for row in contrasts if row.get(key) is not None])
            for key in gap_keys
        },
    }
