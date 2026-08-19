from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def _correct(row: Mapping[str, Any], field: str) -> bool:
    if field in row:
        return bool(row[field])
    aliases = {"router_correct": "online_correct"}
    alias = aliases.get(field)
    if alias and alias in row:
        return bool(row[alias])
    raise KeyError(f"missing correctness field {field!r}")


def oracle_select(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Choose correctness first, then the smallest visual-on budget."""
    if not candidates:
        raise ValueError("oracle selection requires at least one candidate")
    return min(
        candidates,
        key=lambda row: (
            not bool(row["correct"]),
            int(row["budget"]),
            str(row["policy"]),
        ),
    )


def align_policy_rows(
    policy_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Align policy outputs and enforce a shared deterministic baseline contract."""
    if not policy_rows:
        raise ValueError("at least one policy is required")
    by_policy: dict[str, dict[str, Mapping[str, Any]]] = {}
    for policy, rows in policy_rows.items():
        indexed = {str(row["uid"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"duplicate UID in policy {policy!r}")
        by_policy[policy] = indexed

    reference_policy = next(iter(by_policy))
    reference_uids = set(by_policy[reference_policy])
    for policy, indexed in by_policy.items():
        if set(indexed) != reference_uids:
            missing = len(reference_uids - set(indexed))
            extra = len(set(indexed) - reference_uids)
            raise ValueError(
                f"UID set mismatch for {policy!r}: missing={missing}, extra={extra}"
            )

    aligned: list[dict[str, Any]] = []
    for uid in sorted(reference_uids):
        reference = by_policy[reference_policy][uid]
        baseline_correct = _correct(reference, "baseline_correct")
        benchmark = str(reference["benchmark"])
        routes: dict[str, dict[str, Any]] = {}
        for policy, indexed in by_policy.items():
            row = indexed[uid]
            if _correct(row, "baseline_correct") != baseline_correct:
                raise ValueError(f"baseline correctness mismatch for UID {uid!r}")
            if str(row["benchmark"]) != benchmark:
                raise ValueError(f"benchmark mismatch for UID {uid!r}")
            budget = int(row["selected_num_visual_on_layers"])
            if not 0 <= budget <= 28:
                raise ValueError(f"invalid route budget {budget} for UID {uid!r}")
            routes[policy] = {
                "correct": _correct(row, "router_correct"),
                "budget": budget,
                "mask": row.get("selected_mask_key"),
            }
        aligned.append(
            {
                "uid": uid,
                "benchmark": benchmark,
                "baseline_correct": baseline_correct,
                "routes": routes,
            }
        )
    return aligned


def _point_arrays(
    aligned: Sequence[Mapping[str, Any]], policies: Sequence[str]
) -> dict[str, np.ndarray]:
    baseline = np.asarray([row["baseline_correct"] for row in aligned], dtype=np.int8)
    selected_correct = np.zeros(len(aligned), dtype=np.int8)
    selected_budget = np.zeros(len(aligned), dtype=np.float64)
    selected_policy: list[str] = []
    for index, row in enumerate(aligned):
        candidates: list[dict[str, Any]] = [
            {"policy": "all_on", "correct": bool(row["baseline_correct"]), "budget": 28}
        ]
        candidates.extend(
            {
                "policy": policy,
                "correct": bool(row["routes"][policy]["correct"]),
                "budget": int(row["routes"][policy]["budget"]),
            }
            for policy in policies
        )
        chosen = oracle_select(candidates)
        selected_correct[index] = int(chosen["correct"])
        selected_budget[index] = int(chosen["budget"])
        selected_policy.append(str(chosen["policy"]))
    return {
        "baseline_correct": baseline,
        "selected_correct": selected_correct,
        "selected_budget": selected_budget,
        "selected_policy": np.asarray(selected_policy, dtype=object),
    }


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
    }


def evaluate_oracle(
    aligned: Sequence[Mapping[str, Any]],
    policies: Sequence[str],
    *,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 20260812,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not policies:
        raise ValueError("oracle evaluation requires at least one sparse policy")
    arrays = _point_arrays(aligned, policies)
    baseline = arrays["baseline_correct"]
    selected = arrays["selected_correct"]
    budget = arrays["selected_budget"]
    chosen_policy = arrays["selected_policy"]
    n = len(aligned)
    if n == 0:
        raise ValueError("cannot evaluate an empty population")

    rng = np.random.default_rng(bootstrap_seed)
    delta_samples = np.empty(bootstrap_repetitions, dtype=np.float64)
    budget_samples = np.empty(bootstrap_repetitions, dtype=np.float64)
    for repetition in range(bootstrap_repetitions):
        sample = rng.integers(0, n, size=n)
        delta_samples[repetition] = float((selected[sample] - baseline[sample]).mean())
        budget_samples[repetition] = float(budget[sample].mean())

    rows = []
    for index, source in enumerate(aligned):
        rows.append(
            {
                "uid": source["uid"],
                "benchmark": source["benchmark"],
                "baseline_correct": bool(baseline[index]),
                "oracle_correct": bool(selected[index]),
                "oracle_budget": int(budget[index]),
                "oracle_policy": str(chosen_policy[index]),
            }
        )

    rescue = int(((baseline == 0) & (selected == 1)).sum())
    harm = int(((baseline == 1) & (selected == 0)).sum())
    policy_counts = {
        str(policy): int((chosen_policy == policy).sum())
        for policy in sorted(set(chosen_policy.tolist()))
    }
    summary = {
        "n": n,
        "policies": list(policies),
        "baseline_accuracy": float(baseline.mean()),
        "oracle_accuracy": float(selected.mean()),
        "accuracy_delta": float((selected - baseline).mean()),
        "accuracy_delta_bootstrap_95_ci": _percentile_interval(delta_samples),
        "rescue_count": rescue,
        "harm_count": harm,
        "mean_visual_on_layers": float(budget.mean()),
        "mean_visual_on_layers_bootstrap_95_ci": _percentile_interval(budget_samples),
        "route_sensitive_layer_saving_fraction": float((28.0 - budget.mean()) / 28.0),
        "sparse_route_fraction": float((chosen_policy != "all_on").mean()),
        "selected_policy_counts": policy_counts,
    }
    return summary, rows


def benchmark_summaries(
    aligned: Sequence[Mapping[str, Any]], policies: Sequence[str]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for benchmark in sorted({str(row["benchmark"]) for row in aligned}):
        subset = [row for row in aligned if row["benchmark"] == benchmark]
        summary, _ = evaluate_oracle(
            subset,
            policies,
            bootstrap_repetitions=500,
            bootstrap_seed=20260812,
        )
        output[benchmark] = summary
    return output
