"""Pure contracts for Stage-2 non-FULL abstention-margin calibration."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence


ACTION_NAMES = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")


def artifact_hash(value: Mapping[str, Any], *, hash_field: str) -> str:
    """Hash an artifact while excluding exactly its declared self-hash field."""
    import json

    payload = {key: item for key, item in value.items() if key != str(hash_field)}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def choose_action_from_logits(logits: Sequence[float], *, delta: float) -> dict[str, Any]:
    """Apply the frozen strict best-non-FULL versus FULL margin rule."""
    if len(logits) != len(ACTION_NAMES):
        raise ValueError("router logits must follow the frozen four-action order")
    values = [float(value) for value in logits]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("router logits must be finite")
    threshold = float(delta)
    if math.isnan(threshold) or threshold < 0:
        raise ValueError("abstention margin must be nonnegative")
    best_nonfull = max(range(1, len(values)), key=lambda index: values[index])
    raw_margin = values[best_nonfull] - values[0]
    selected = best_nonfull if raw_margin > threshold else 0
    return {
        "action_index": selected,
        "action": ACTION_NAMES[selected],
        "best_nonfull_index": best_nonfull,
        "best_nonfull_action": ACTION_NAMES[best_nonfull],
        "raw_margin": raw_margin,
        "full_logit": values[0],
        "best_nonfull_logit": values[best_nonfull],
    }


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantiles require at least one value")
    p = float(probability)
    if p < 0 or p > 1:
        raise ValueError("quantile probabilities must lie in [0, 1]")
    position = (len(sorted_values) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low]) * (1.0 - weight) + float(sorted_values[high]) * weight


def build_margin_grid(
    margins: Sequence[float],
    *,
    quantiles: Sequence[float] = (0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
) -> list[dict[str, Any]]:
    """Freeze 0, positive training-margin quantiles, and the dense +infinity control."""
    positive = sorted(float(value) for value in margins if math.isfinite(float(value)) and float(value) > 0)
    if not positive:
        raise ValueError("no positive training-side margins are available")
    candidates: list[tuple[str, float, float | None]] = [("zero", 0.0, None)]
    for probability in quantiles:
        value = _linear_quantile(positive, float(probability))
        candidates.append((f"q{round(float(probability) * 100):02d}", value, float(probability)))
    deduplicated: list[dict[str, Any]] = []
    seen: set[float] = set()
    for label, value, quantile in sorted(candidates, key=lambda item: item[1]):
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(
            {
                "margin_id": f"margin_{len(deduplicated):02d}",
                "label": label,
                "delta": value,
                "source_quantile": quantile,
            }
        )
    deduplicated.append(
        {
            "margin_id": f"margin_{len(deduplicated):02d}",
            "label": "infinity",
            "delta": math.inf,
            "source_quantile": None,
        }
    )
    return deduplicated


def summarize_margin_rollout(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rollout summary requires at least one sample")
    transitions = Counter(
        ("C" if bool(row["dense_correct"]) else "W")
        + "→"
        + ("C" if bool(row["routed_correct"]) else "W")
        for row in rows
    )
    dense_correct = transitions["C→C"] + transitions["C→W"]
    dense_wrong = transitions["W→C"] + transitions["W→W"]
    routed_correct = transitions["C→C"] + transitions["W→C"]
    interventions = sum(bool(row.get("any_non_full", False)) for row in rows)
    triggered = [row for row in rows if bool(row.get("triggered", False))]
    post_counts = Counter()
    for row in triggered:
        post_counts.update(row.get("post_trigger_action_counts", {}))
    total_post = sum(post_counts.values())
    total_nonfull = sum(post_counts[action] for action in ACTION_NAMES[1:])
    return {
        "samples": len(rows),
        "dense_correct": dense_correct,
        "dense_wrong": dense_wrong,
        "routed_correct": routed_correct,
        "dense_accuracy": dense_correct / len(rows),
        "routed_accuracy": routed_correct / len(rows),
        "delta_accuracy": (routed_correct - dense_correct) / len(rows),
        "w_to_c": transitions["W→C"],
        "w_to_w": transitions["W→W"],
        "c_to_c": transitions["C→C"],
        "c_to_w": transitions["C→W"],
        "net_corrections": transitions["W→C"] - transitions["C→W"],
        "w_to_c_rate": transitions["W→C"] / dense_wrong if dense_wrong else None,
        "c_to_c_preservation_rate": transitions["C→C"] / dense_correct if dense_correct else None,
        "rescue_to_regression_ratio": transitions["W→C"] / max(transitions["C→W"], 1),
        "stage1_triggered_count": len(triggered),
        "intervened_samples": interventions,
        "intervention_rate": interventions / len(rows),
        "total_non_full_count": total_nonfull,
        "post_trigger_full_fraction": post_counts["FULL"] / total_post if total_post else None,
        "mean_non_full_actions": total_nonfull / len(triggered) if triggered else None,
    }


def select_margin(
    summaries: Sequence[Mapping[str, Any]],
    *,
    min_preservation: float = 0.995,
) -> dict[str, Any]:
    """Select conservatively; a positive finite margin must improve net over delta=0."""
    baseline_rows = [row for row in summaries if float(row["delta"]) == 0.0]
    if len(baseline_rows) != 1:
        raise ValueError("selection requires exactly one delta=0 baseline")
    baseline = dict(baseline_rows[0])
    eligible = [
        dict(row)
        for row in summaries
        if float(row["c_to_c_preservation_rate"]) >= float(min_preservation)
        and math.isfinite(float(row["delta"]))
        and float(row["delta"]) > 0.0
        and int(row["net_corrections"]) > int(baseline["net_corrections"])
    ]
    if not eligible:
        return baseline
    return max(
        eligible,
        key=lambda row: (
            int(row["net_corrections"]),
            -int(row["c_to_w"]),
            int(row["w_to_c"]),
            -int(row["intervened_samples"]),
            float(row["delta"]),
        ),
    )


def assign_group_folds(
    rows: Sequence[Mapping[str, Any]], *, folds: int, seed: int
) -> dict[str, int]:
    """Deterministically balance indivisible image groups across folds."""
    if int(folds) < 2:
        raise ValueError("at least two folds are required")
    grouped: dict[str, list[str]] = defaultdict(list)
    seen_uids: set[str] = set()
    for row in rows:
        uid = str(row["uid"])
        group = str(row["image_group_id"])
        if uid in seen_uids:
            raise ValueError(f"duplicate UID in fold population: {uid}")
        seen_uids.add(uid)
        grouped[group].append(uid)
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            sha256(f"{int(seed)}:{item[0]}".encode()).hexdigest(),
            item[0],
        ),
    )
    loads = [0] * int(folds)
    assignment: dict[str, int] = {}
    for _group, uids in ranked:
        fold = min(range(int(folds)), key=lambda index: (loads[index], index))
        for uid in uids:
            assignment[uid] = fold
        loads[fold] += len(uids)
    return assignment


def validate_result_matrix(
    expected_uids: Sequence[str],
    margin_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    contract_sha256: str,
) -> None:
    if any(str(row.get("contract_sha256")) != str(contract_sha256) for row in rows):
        raise ValueError("result contract differs from the frozen contract")
    expected = Counter(
        (str(uid), str(margin_id)) for uid in expected_uids for margin_id in margin_ids
    )
    observed = Counter((str(row["uid"]), str(row["margin_id"])) for row in rows)
    if observed != expected:
        missing = list((expected - observed).elements())[:5]
        duplicate = list((observed - expected).elements())[:5]
        raise ValueError(f"result matrix is incomplete or duplicated: missing={missing}, extra={duplicate}")
