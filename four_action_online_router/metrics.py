"""Node- and execution-level metrics for online four-action routing."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import torch

from four_action_policy.actions import FOUR_ACTIONS
from .supervision import set_valued_action_loss


READ_BIT = torch.tensor([0, 1, 0, 1], dtype=torch.bool)
WRITE_BIT = torch.tensor([0, 0, 1, 1], dtype=torch.bool)


def summarize_node_predictions(
    logits: torch.Tensor, valid_actions: torch.BoolTensor
) -> dict[str, Any]:
    if logits.ndim != 2 or logits.shape[-1] != 4 or logits.shape != valid_actions.shape:
        raise ValueError("node logits and valid masks must have shape [states, 4]")
    predictions = logits.argmax(dim=-1)
    selected_valid = valid_actions.to(logits.device).gather(1, predictions[:, None]).squeeze(1)
    counts = Counter(int(value) for value in predictions.detach().cpu().tolist())
    per_action = {}
    for index, action in enumerate(FOUR_ACTIONS):
        predicted = predictions == index
        valid = valid_actions[:, index].to(predictions.device)
        true_positive = int((predicted & valid).sum().item())
        precision = true_positive / int(predicted.sum().item()) if bool(predicted.any()) else 0.0
        recall = true_positive / int(valid.sum().item()) if bool(valid.any()) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_action[action] = {"precision": precision, "recall": recall, "f1": f1}

    def bit_accuracy(bits: torch.BoolTensor) -> float:
        bits = bits.to(predictions.device)
        predicted_bits = bits[predictions]
        valid_bits = valid_actions.to(predictions.device) & bits[None, :]
        invalid_bits = valid_actions.to(predictions.device) & ~bits[None, :]
        accepts_true = valid_bits.any(dim=1)
        accepts_false = invalid_bits.any(dim=1)
        correct = torch.where(predicted_bits, accepts_true, accepts_false)
        return float(correct.float().mean().item())

    return {
        "states": int(logits.shape[0]),
        "negative_log_valid_action_probability": float(
            set_valued_action_loss(logits, valid_actions).detach().item()
        ),
        "valid_action_at_1": float(selected_valid.float().mean().item()),
        "predicted_action_counts": {
            action: counts[index] for index, action in enumerate(FOUR_ACTIONS)
        },
        "per_action": per_action,
        "read_bit_accuracy": bit_accuracy(READ_BIT),
        "write_bit_accuracy": bit_accuracy(WRITE_BIT),
    }


def _summarize_execution_population(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(
        str(action) for row in rows for action in row.get("actions", [])
    )
    records = len(rows)
    return {
        "records": records,
        "routed_accuracy": sum(bool(row["correct"]) for row in rows) / records if records else None,
        "action_counts": {action: action_counts[action] for action in FOUR_ACTIONS},
        "mean_action_layers": {
            action: action_counts[action] / records if records else None for action in FOUR_ACTIONS
        },
    }


def summarize_execution_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("execution validation rows cannot be empty")
    uids = [str(row.get("uid") or "") for row in rows]
    if any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError("execution validation UIDs are empty or duplicated")
    w2c = [row for row in rows if row.get("route_type") == "W2C"]
    c2c = [row for row in rows if row.get("route_type") == "C2C"]
    if not w2c or not c2c or len(w2c) + len(c2c) != len(rows):
        raise ValueError("execution validation requires only nonempty W2C and C2C pools")
    w2c_rescues = sum(bool(row["correct"]) for row in w2c)
    c2c_preserved = sum(bool(row["correct"]) for row in c2c)
    overall_correct = w2c_rescues + c2c_preserved
    overall_actions = _summarize_execution_population(rows)
    return {
        "records": len(rows),
        "w2c_records": len(w2c),
        "c2c_records": len(c2c),
        "w2c_rescues": w2c_rescues,
        "c2c_preserved": c2c_preserved,
        "c2c_regressions": len(c2c) - c2c_preserved,
        "w2c_rescue_rate": w2c_rescues / len(w2c),
        "c2c_preservation_rate": c2c_preserved / len(c2c),
        "balanced_execution_score": 0.5 * (w2c_rescues / len(w2c))
        + 0.5 * (c2c_preserved / len(c2c)),
        "overall_routed_accuracy": overall_correct / len(rows),
        "action_counts": overall_actions["action_counts"],
        "mean_action_layers": overall_actions["mean_action_layers"],
        "by_dataset": {
            dataset: _summarize_execution_population(
                [row for row in rows if row.get("dataset") == dataset]
            )
            for dataset in ("gqa", "chartqa", "textvqa")
        },
    }


def execution_checkpoint_key(epoch_row: dict[str, Any]) -> tuple[float, float, int, float, int]:
    metrics = epoch_row["execution"]
    return (
        float(metrics["balanced_execution_score"]),
        float(metrics["overall_routed_accuracy"]),
        -int(metrics["c2c_regressions"]),
        -float(metrics["mean_action_layers"]["FULL"]),
        -int(epoch_row["epoch"]),
    )


def _boundary_population(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records = len(rows)
    valid = sum(
        str(row["predicted_boundary_action"]) in row["valid_nonfull_actions"]
        for row in rows
    )
    nonfull = sum(str(row["predicted_boundary_action"]) != "FULL" for row in rows)
    return {
        "records": records,
        "valid_action_at_1": valid / records if records else None,
        "nonfull_recall": nonfull / records if records else None,
    }


def mandatory_boundary_metrics(
    rows: Sequence[dict[str, Any]], *, num_layers: int
) -> dict[str, Any]:
    """Summarize exact-boundary predictions and free-rollout deviation timing."""

    if not rows:
        raise ValueError("mandatory-boundary metrics require records")
    if len({str(row["uid"]) for row in rows}) != len(rows):
        raise ValueError("mandatory-boundary metric UIDs must be unique")
    for row in rows:
        if "FULL" in row["valid_nonfull_actions"] or not row["valid_nonfull_actions"]:
            raise ValueError("boundary valid actions must be nonempty and exclude FULL")
        if not 0 <= int(row["boundary_layer"]) < num_layers:
            raise ValueError("boundary layer lies outside the decoder")

    singleton = [row for row in rows if bool(row["singleton"])]
    multi = [row for row in rows if not bool(row["singleton"])]
    action_records = {}
    action_recall = {}
    for action in ("IGNORE", "READ_ONLY", "WRITE_ONLY"):
        selected = [
            row for row in singleton if row["valid_nonfull_actions"] == [action]
        ]
        action_records[action] = len(selected)
        action_recall[action] = (
            sum(row["predicted_boundary_action"] == action for row in selected)
            / len(selected)
            if selected else None
        )

    deltas: list[int] = []
    left_full = 0
    exact = 0
    within_one = 0
    within_two = 0
    early = 0
    late_or_none = 0
    for row in rows:
        actions = list(row["actions"])
        if len(actions) != num_layers:
            raise ValueError("free-rollout action route has the wrong width")
        predicted_layer = next(
            (index for index, action in enumerate(actions) if action != "FULL"), None
        )
        if predicted_layer is None:
            late_or_none += 1
            continue
        left_full += 1
        delta = predicted_layer - int(row["boundary_layer"])
        deltas.append(delta)
        exact += delta == 0
        within_one += abs(delta) <= 1
        within_two += abs(delta) <= 2
        early += delta < 0
        late_or_none += delta > 0

    overall = _boundary_population(rows)
    return {
        **overall,
        "by_dataset": {
            dataset: _boundary_population(
                [row for row in rows if row["dataset"] == dataset]
            )
            for dataset in ("gqa", "chartqa", "textvqa")
        },
        "by_valid_action": {
            action: _boundary_population(
                [row for row in rows if action in row["valid_nonfull_actions"]]
            )
            for action in ("IGNORE", "READ_ONLY", "WRITE_ONLY")
        },
        "singleton": _boundary_population(singleton),
        "multi_valid": _boundary_population(multi),
        "singleton_action_records": action_records,
        "singleton_action_recall": action_recall,
        "free_rollout": {
            "left_all_full_fraction": left_full / len(rows),
            "exact_boundary_fraction": exact / len(rows),
            "within_1_fraction": within_one / len(rows),
            "within_2_fraction": within_two / len(rows),
            "early_fraction": early / len(rows),
            "late_or_no_deviation_fraction": late_or_none / len(rows),
            "observed_delta_count": len(deltas),
            "mean_delta_when_observed": sum(deltas) / len(deltas) if deltas else None,
        },
    }


def mandatory_boundary_pilot_gate(
    metrics: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    """Apply the frozen A1 behavioral gate without inspecting later outcomes."""

    boundary = metrics["boundary"]
    execution = metrics["execution"]
    checks = {
        "boundary_valid_action_at_1": boundary["valid_action_at_1"]
        >= float(gates["boundary_valid_action_at_1"]),
        "boundary_nonfull_recall": boundary["nonfull_recall"]
        >= float(gates["boundary_nonfull_recall"]),
        "free_rollout_leave_full": boundary["free_rollout"]["left_all_full_fraction"]
        >= float(gates["free_rollout_leave_full"]),
        "w2c_rescue_rate": execution["w2c_rescue_rate"]
        >= float(gates["w2c_rescue_rate"]),
        "c2c_preservation_rate": execution["c2c_preservation_rate"]
        >= float(gates["c2c_preservation_rate"]),
    }
    minimum = int(gates["singleton_min_records"])
    for action in ("IGNORE", "READ_ONLY", "WRITE_ONLY"):
        records = int(boundary["singleton_action_records"][action])
        recall = boundary["singleton_action_recall"][action]
        if records >= minimum:
            checks[f"singleton_{action}_recall"] = recall >= float(
                gates["singleton_action_recall"]
            )
    return {"passed": all(checks.values()), "checks": checks, "thresholds": gates}
