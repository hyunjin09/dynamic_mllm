"""Pure contracts and summaries for the frozen full-benchmark paired evaluation."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

import numpy as np


TASK_COUNTS = {
    "chartqa": 2500,
    "textvqa": 5000,
    "mmmu_pro_standard_test": 1730,
    "mmmu_pro_vision_test": 1730,
    "pope_adversarial": 3000,
    "pope_popular": 3000,
    "pope_random": 3000,
}
FAMILY_ORDER = ("chartqa", "textvqa", "mmmu_pro", "pope")
ACTION_NAMES = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")


def feature_positions_from_masks(
    instruction_mask: Sequence[bool],
    multimodal_token_types: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve native full-sequence instruction and visual feature positions."""
    if len(instruction_mask) != len(multimodal_token_types):
        raise ValueError("instruction mask and multimodal token types differ in length")
    instruction = tuple(index for index, selected in enumerate(instruction_mask) if bool(selected))
    visual = tuple(index for index, value in enumerate(multimodal_token_types) if int(value) != 0)
    if not instruction or not visual:
        raise ValueError("native feature positions require nonempty instruction and visual spans")
    if any(int(multimodal_token_types[index]) != 0 for index in instruction):
        raise ValueError("reference instruction mask includes a visual position")
    return instruction, visual


def benchmark_family(benchmark: str) -> str:
    name = str(benchmark).lower()
    if name in {"chartqa", "textvqa"}:
        return name
    if name in {"mmmu_pro_standard_test", "mmmu_pro_vision_test"}:
        return "mmmu_pro"
    if name in {"pope_adversarial", "pope_popular", "pope_random"}:
        return "pope"
    raise ValueError(f"benchmark is outside the frozen evaluation scope: {benchmark}")


def validate_population(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["benchmark"]).lower() for row in rows)
    if counts != Counter(TASK_COUNTS):
        raise ValueError(f"full benchmark population differs: {dict(counts)}")
    uids = [str(row["uid"]) for row in rows]
    if len(uids) != len(set(uids)):
        raise ValueError("full benchmark population contains duplicate UIDs")
    return dict(counts)


def transition(dense_correct: bool, routed_correct: bool) -> str:
    return ("C" if dense_correct else "W") + "→" + ("C" if routed_correct else "W")


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def summarize_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("paired summary requires at least one row")
    counts = Counter(str(row["transition"]) for row in rows)
    allowed = {"C→C", "C→W", "W→C", "W→W"}
    if set(counts) - allowed:
        raise ValueError(f"unsupported transitions: {sorted(set(counts) - allowed)}")
    n = len(rows)
    dense_correct = counts["C→C"] + counts["C→W"]
    routed_correct = counts["C→C"] + counts["W→C"]
    dense_wrong = n - dense_correct
    net = counts["W→C"] - counts["C→W"]
    return {
        "n": n,
        "dense_correct": dense_correct,
        "routed_correct": routed_correct,
        "dense_accuracy": dense_correct / n,
        "routed_accuracy": routed_correct / n,
        "delta_accuracy": net / n,
        "w_to_c": counts["W→C"],
        "c_to_w": counts["C→W"],
        "c_to_c": counts["C→C"],
        "w_to_w": counts["W→W"],
        "net_correction": net,
        "w_to_c_rate_among_dense_wrong": safe_rate(counts["W→C"], dense_wrong),
        "c_to_c_preservation": safe_rate(counts["C→C"], dense_correct),
        "rescue_precision": safe_rate(counts["W→C"], counts["W→C"] + counts["C→W"]),
    }


def percentile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def paired_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> list[dict[str, Any]]:
    if not rows or draws < 1:
        raise ValueError("paired bootstrap requires nonempty rows and positive draws")
    dense = np.asarray([bool(row["dense_correct"]) for row in rows], dtype=np.float64)
    routed = np.asarray([bool(row["routed_correct"]) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    values = {name: [] for name in ("delta_accuracy", "net_correction_rate", "w_to_c_rate", "c_to_c_preservation")}
    n = len(rows)
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        indices = rng.integers(0, n, size=(count, n))
        d = dense[indices]
        r = routed[indices]
        delta = (r - d).mean(axis=1)
        values["delta_accuracy"].extend(delta.tolist())
        values["net_correction_rate"].extend(delta.tolist())
        wrong = (d == 0)
        correct = ~wrong
        w_den = wrong.sum(axis=1)
        c_den = correct.sum(axis=1)
        w_num = (wrong & (r == 1)).sum(axis=1)
        c_num = (correct & (r == 1)).sum(axis=1)
        values["w_to_c_rate"].extend(
            np.divide(w_num, w_den, out=np.full(count, np.nan), where=w_den > 0).tolist()
        )
        values["c_to_c_preservation"].extend(
            np.divide(c_num, c_den, out=np.full(count, np.nan), where=c_den > 0).tolist()
        )
    summary = summarize_pairs(rows)
    estimates = {
        "delta_accuracy": summary["delta_accuracy"],
        "net_correction_rate": summary["delta_accuracy"],
        "w_to_c_rate": summary["w_to_c_rate_among_dense_wrong"],
        "c_to_c_preservation": summary["c_to_c_preservation"],
    }
    output = []
    for metric, samples in values.items():
        array = np.asarray(samples, dtype=np.float64)
        array = array[np.isfinite(array)]
        if not len(array) or estimates[metric] is None:
            low = high = None
        else:
            low, high = percentile(array, 0.025), percentile(array, 0.975)
        output.append(
            {
                "metric": metric,
                "estimate": estimates[metric],
                "ci_low": low,
                "ci_high": high,
                "bootstrap_draws": int(draws),
                "seed": int(seed),
            }
        )
    return output


def stage1_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Stage-1 summary requires rows")
    triggered = [row for row in rows if bool(row["triggered"])]
    dense_c = [row for row in rows if bool(row["dense_correct"])]
    dense_w = [row for row in rows if not bool(row["dense_correct"])]
    layers = sorted(int(row["trigger_layer"]) for row in triggered)
    thirds = Counter(
        "early" if layer <= 8 else "middle" if layer <= 18 else "late"
        for layer in layers
    )
    return {
        "n": len(rows),
        "triggered": len(triggered),
        "trigger_rate": len(triggered) / len(rows),
        "p_trigger_given_dense_c": safe_rate(sum(bool(row["triggered"]) for row in dense_c), len(dense_c)),
        "p_trigger_given_dense_w": safe_rate(sum(bool(row["triggered"]) for row in dense_w), len(dense_w)),
        "trigger_precision": safe_rate(sum(not bool(row["dense_correct"]) for row in triggered), len(triggered)),
        "median_trigger_layer": float(np.median(layers)) if layers else None,
        "early_triggers": thirds["early"],
        "middle_triggers": thirds["middle"],
        "late_triggers": thirds["late"],
    }


def stage2_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    triggered = [row for row in rows if bool(row["triggered"])]
    actions = Counter()
    for row in triggered:
        actions.update(row["post_trigger_action_counts"])
    post_total = sum(actions.values())
    nonfull_counts = [int(row["non_full_count"]) for row in triggered]
    first = [int(row["first_non_full_layer"]) for row in triggered if row["first_non_full_layer"] is not None]
    delays = [int(row["trigger_to_first_non_full_delay"]) for row in triggered if row["trigger_to_first_non_full_delay"] is not None]
    triggered_w = [row for row in triggered if not bool(row["dense_correct"])]
    triggered_c = [row for row in triggered if bool(row["dense_correct"])]
    return {
        "triggered": len(triggered),
        "triggered_any_non_full": sum(bool(row["any_non_full"]) for row in triggered),
        "fraction_any_non_full": safe_rate(sum(bool(row["any_non_full"]) for row in triggered), len(triggered)),
        "post_trigger_full_fraction": safe_rate(actions["FULL"], post_total),
        "mean_non_full_actions": float(np.mean(nonfull_counts)) if nonfull_counts else None,
        "mean_first_non_full_layer": float(np.mean(first)) if first else None,
        "mean_trigger_to_first_non_full_delay": float(np.mean(delays)) if delays else None,
        **{f"{name.lower()}_count": actions[name] for name in ACTION_NAMES},
        "triggered_w": len(triggered_w),
        "triggered_w_any_non_full": sum(bool(row["any_non_full"]) for row in triggered_w),
        "w_to_c": sum(str(row["transition"]) == "W→C" for row in rows),
        "triggered_c": len(triggered_c),
        "triggered_c_any_non_full": sum(bool(row["any_non_full"]) for row in triggered_c),
        "c_to_w": sum(str(row["transition"]) == "C→W" for row in rows),
    }


def validate_complete_rows(
    expected_uids: Sequence[str], rows: Sequence[Mapping[str, Any]], *, contract_sha256: str
) -> None:
    expected = Counter(map(str, expected_uids))
    observed = Counter(str(row["uid"]) for row in rows)
    if expected != observed:
        missing = list((expected - observed).elements())[:5]
        duplicate = [uid for uid, count in observed.items() if count != 1][:5]
        unexpected = list((observed - expected).elements())[:5]
        raise ValueError(
            f"paired coverage differs: missing={missing} duplicate={duplicate} unexpected={unexpected}"
        )
    if any(str(row.get("contract_sha256")) != str(contract_sha256) for row in rows):
        raise ValueError("one or more rows belong to another frozen contract")


def method_decision(family_summaries: Mapping[str, Mapping[str, Any]]) -> str:
    overall = int(family_summaries["overall"]["net_correction"])
    positive = sum(int(family_summaries[name]["net_correction"]) > 0 for name in FAMILY_ORDER)
    if overall > 0 and positive >= 2:
        return "A"
    if overall > 0:
        return "B"
    if overall < 0:
        return "D"
    return "C"


def recommendation_direction(
    overall_pairs: Mapping[str, Any], overall_stage2: Mapping[str, Any]
) -> str:
    if int(overall_pairs["net_correction"]) < 0:
        return "conservative action-selection calibration focused on preservation"
    if int(overall_pairs["w_to_c"]) <= 4 and int(overall_pairs["net_correction"]) > 0:
        return "coverage expansion under a frozen preservation constraint"
    fraction = overall_stage2.get("fraction_any_non_full")
    if fraction is not None and float(fraction) < 0.1:
        return "Stage-2 abstention/generalization diagnosis"
    if int(overall_pairs["w_to_c"]) <= int(overall_pairs["c_to_w"]):
        return "corrective action-and-timing quality diagnosis"
    return "coverage expansion under a frozen preservation constraint"
