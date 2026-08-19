"""MCE-8 aggregate decision helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from typing import Any, Callable


SCORE_ATOL = 1e-9


def _rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else 0.0


def natural_accuracy_delta(*, full_accuracy: float, fix_rate: float, regression_rate: float) -> float:
    return (1.0 - full_accuracy) * fix_rate - full_accuracy * regression_rate


def sample_any_rate(
    rows: list[dict[str, Any]],
    *,
    cohort: str,
    event_key: str,
) -> tuple[int, int, float]:
    by_sample: dict[str, bool] = {}
    for row in rows:
        if str(row.get("cohort")) != cohort:
            continue
        sample_id = str(row["sample_id"])
        by_sample[sample_id] = by_sample.get(sample_id, False) or bool(row.get(event_key))
    numer = sum(by_sample.values())
    denom = len(by_sample)
    return numer, denom, _rate(numer, denom)


def sample_event_records(
    rows: list[dict[str, Any]],
    *,
    cohort: str | None,
    event_key: str,
) -> list[dict[str, Any]]:
    by_sample: dict[str, dict[str, Any]] = {}
    for row in rows:
        if cohort is not None and str(row.get("cohort")) != cohort:
            continue
        sample_id = str(row["sample_id"])
        record = by_sample.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "source_asset_id": row.get("source_asset_id") or sample_id,
                "dataset": row.get("dataset"),
                "cohort": row.get("cohort"),
                "event": False,
            },
        )
        record["event"] = bool(record["event"]) or bool(row.get(event_key))
    return list(by_sample.values())


def boolean_event_rate(rows: list[dict[str, Any]], event_key: str) -> tuple[int, int, float]:
    numer = sum(bool(row.get(event_key)) for row in rows)
    denom = len(rows)
    return numer, denom, _rate(numer, denom)


def label_rate(rows: list[dict[str, Any]], label: str) -> tuple[int, int, float]:
    numer = sum(str(row.get("counterfactual_label")) == label for row in rows)
    denom = len(rows)
    return numer, denom, _rate(numer, denom)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    frac = position - lower
    return float(sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac)


def cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    metric_fn: Callable[[list[dict[str, Any]]], float],
    *,
    cluster_key: str = "source_asset_id",
    iterations: int = 2000,
    seed: int = 20260619,
    ci: float = 0.95,
) -> dict[str, Any]:
    if not rows:
        return {
            "observed": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "iterations": iterations,
            "cluster_count": 0,
        }
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row.get(cluster_key) or row.get("sample_id"))].append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    observed = float(metric_fn(rows))
    estimates: list[float] = []
    for _ in range(iterations):
        sampled_rows: list[dict[str, Any]] = []
        for _ in keys:
            sampled_rows.extend(clusters[rng.choice(keys)])
        estimates.append(float(metric_fn(sampled_rows)))
    estimates.sort()
    alpha = (1.0 - ci) / 2.0
    ci_low = _percentile(estimates, alpha)
    ci_high = _percentile(estimates, 1.0 - alpha)
    return {
        "observed": observed,
        "ci_low": min(ci_low, observed),
        "ci_high": max(ci_high, observed),
        "iterations": iterations,
        "cluster_count": len(keys),
    }


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_enrichment_pvalue(
    *,
    event_in_group: int,
    total_in_group: int,
    event_total: int,
    total: int,
) -> float:
    if total <= 0 or total_in_group <= 0 or event_total <= 0:
        return 1.0
    min_x = max(0, total_in_group - (total - event_total))
    max_x = min(total_in_group, event_total)
    if event_in_group <= min_x:
        return 1.0
    log_den = _log_comb(total, total_in_group)
    terms = [
        _log_comb(event_total, x) + _log_comb(total - event_total, total_in_group - x) - log_den
        for x in range(event_in_group, max_x + 1)
    ]
    max_log = max(terms)
    return min(1.0, sum(math.exp(term - max_log) for term in terms) * math.exp(max_log))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted_sorted: list[tuple[int, float]] = []
    running_min = 1.0
    for rank_from_end, (index, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        adjusted = min(running_min, float(p_value) * m / rank)
        running_min = adjusted
        adjusted_sorted.append((index, min(1.0, adjusted)))
    result = [0.0] * m
    for index, adjusted in adjusted_sorted:
        result[index] = adjusted
    return result


def layerwise_enrichment(
    rows: list[dict[str, Any]],
    *,
    family: str,
    event_label: str,
    num_layers: int = 28,
) -> list[dict[str, Any]]:
    total = len(rows)
    event_total = sum(str(row.get("counterfactual_label")) == event_label for row in rows)
    records: list[dict[str, Any]] = []
    p_values: list[float] = []
    for layer in range(1, num_layers + 1):
        layer_rows = [row for row in rows if int(row.get("layer_one_based", -1)) == layer]
        event_count = sum(str(row.get("counterfactual_label")) == event_label for row in layer_rows)
        p_value = fisher_enrichment_pvalue(
            event_in_group=event_count,
            total_in_group=len(layer_rows),
            event_total=event_total,
            total=total,
        )
        p_values.append(p_value)
        records.append(
            {
                "family": family,
                "layer_one_based": layer,
                "event_label": event_label,
                "event_count": event_count,
                "total_count": len(layer_rows),
                "event_rate": _rate(event_count, len(layer_rows)),
                "p_value": p_value,
            }
        )
    q_values = benjamini_hochberg(p_values)
    for record, q_value in zip(records, q_values):
        record["q_value"] = q_value
        record["significant_fdr_0_05"] = q_value <= 0.05
    return records


def choose_decision_outcome(
    *,
    search_bias_gate_result: str,
    mce4_redundant_rate: float,
    mce6_critical_rate: float,
) -> dict[str, str]:
    if search_bias_gate_result == "FAIL":
        return {
            "outcome": "C",
            "title": "random best-of-N/search-bias explains the diagnostic gains",
            "rationale": (
                "MCE-7 failed because the strongest matched random/fixed control beat "
                "structured search on both task score and wrong-case correction rate, "
                "so random best-of-N/search-bias remains a sufficient explanation. "
                "Local harmful and critical counterfactuals exist, but the current "
                "diagnostic does not support claiming that structured harmful-update "
                "suppression is the specific source of the oracle accuracy gain."
            ),
            "phase5b_action": (
                "Do not resume harm-aware accuracy-improving router training as-is. "
                "Reframe Phase 5B toward conservative full-Qwen fallback and safe sparse "
                "contextualization, with random-search-aware controls as a required gate."
            ),
        }
    if mce6_critical_rate > 0.5:
        return {
            "outcome": "D",
            "title": "harm exists but critical regressions dominate",
            "rationale": "The search-bias gate passed, but oracle dropout shows many retained updates are critical.",
            "phase5b_action": "Use conservative fallback and regression guards before sparse routing.",
        }
    if mce4_redundant_rate > 0.8:
        return {
            "outcome": "B",
            "title": "mostly redundant visual contextualization",
            "rationale": "Most all-ON single-layer suppressions are neutral after the search-bias gate passes.",
            "phase5b_action": "Reframe toward safe sparse contextualization rather than accuracy gains.",
        }
    return {
        "outcome": "A",
        "title": "strong harmfulness evidence",
        "rationale": "Structured search beats controls and counterfactual harmfulness is not dominated by redundancy.",
        "phase5b_action": "Resume harm-aware Phase 5B with MCE-derived regression gates.",
    }


def winner_sample_records(
    rows: list[dict[str, Any]],
    *,
    method: str,
    budget_key: str = "matched",
    full_fix_score: float = 1.0,
    atol: float = SCORE_ATOL,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("method")) != method or str(row.get("budget_key")) != budget_key:
            continue
        cohort = str(row["cohort"])
        baseline_score = float(row["baseline_score"])
        winner_score = float(row["winner_score"])
        records.append(
            {
                "sample_id": row["sample_id"],
                "source_asset_id": row.get("source_asset_id") or row["sample_id"],
                "dataset": row["dataset"],
                "cohort": cohort,
                "fix_event": (
                    cohort == "wrong"
                    and baseline_score < full_fix_score - atol
                    and winner_score >= full_fix_score - atol
                ),
                "regression_event": cohort == "correct" and winner_score < baseline_score - atol,
                "winner_score": winner_score,
                "baseline_score": baseline_score,
            }
        )
    return records


def summarize_winner_records(
    records: list[dict[str, Any]],
    *,
    full_accuracy: float,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    wrong = [row for row in records if str(row["cohort"]) == "wrong"]
    correct = [row for row in records if str(row["cohort"]) == "correct"]
    fix_n = sum(bool(row["fix_event"]) for row in wrong)
    reg_n = sum(bool(row["regression_event"]) for row in correct)
    fix_rate = _rate(fix_n, len(wrong))
    regression_rate = _rate(reg_n, len(correct))
    return {
        "wrong_fixes": fix_n,
        "wrong_samples": len(wrong),
        "wrong_fix_rate": fix_rate,
        "correct_regressions": reg_n,
        "correct_samples": len(correct),
        "correct_regression_rate": regression_rate,
        "natural_accuracy_delta": natural_accuracy_delta(
            full_accuracy=full_accuracy,
            fix_rate=fix_rate,
            regression_rate=regression_rate,
        ),
        "wrong_fix_rate_ci": cluster_bootstrap_ci(
            wrong,
            lambda sample: _rate(sum(bool(row["fix_event"]) for row in sample), len(sample)),
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "correct_regression_rate_ci": cluster_bootstrap_ci(
            correct,
            lambda sample: _rate(sum(bool(row["regression_event"]) for row in sample), len(sample)),
            iterations=bootstrap_iterations,
            seed=seed + 1,
        ),
    }
