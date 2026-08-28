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
