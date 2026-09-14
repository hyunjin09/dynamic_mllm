"""Prospectively frozen rules for dense first-token logit emergence."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def first_persistent_layer(
    values: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    direction: str,
    consecutive: int = 3,
) -> int | None:
    """Return the first zero-based layer starting a strict persistent crossing."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("emergence values must be one-dimensional")
    if consecutive < 1:
        raise ValueError("consecutive must be positive")
    if direction == "greater":
        passes = array > float(threshold)
    elif direction == "less":
        passes = array < float(threshold)
    else:
        raise ValueError(f"unsupported direction: {direction}")
    for layer in range(max(0, len(array) - consecutive + 1)):
        if bool(passes[layer : layer + consecutive].all()):
            return layer
    return None


def strongest_non_target_from_top2(
    top_values: np.ndarray,
    top_ids: np.ndarray,
    target_ids: np.ndarray,
) -> np.ndarray:
    """Select the strongest non-target value from sorted top-2 logits."""

    values = np.asarray(top_values)
    ids = np.asarray(top_ids)
    targets = np.asarray(target_ids)
    if values.shape != ids.shape or values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("top_values/top_ids must have matching [N,2] shapes")
    if targets.shape != (values.shape[0],):
        raise ValueError("target_ids must have shape [N]")
    return np.where(ids[:, 0] == targets, values[:, 1], values[:, 0])


def classify_wrong_trajectory(
    margins: Sequence[float] | np.ndarray,
    *,
    gt_token_id: int,
    pred_token_id: int,
    delta: float = 1.0,
    consecutive: int = 3,
    early_cutoff: int = 4,
) -> str:
    """Assign the frozen wrong-sample trajectory taxonomy."""

    if int(gt_token_id) == int(pred_token_id):
        return "ambiguous"
    wrong_layer = first_persistent_layer(
        margins,
        threshold=-float(delta),
        direction="less",
        consecutive=consecutive,
    )
    gt_layer = first_persistent_layer(
        margins,
        threshold=float(delta),
        direction="greater",
        consecutive=consecutive,
    )
    if (
        wrong_layer is not None
        and gt_layer is not None
        and gt_layer < wrong_layer
    ):
        return "answer_erosion"
    if wrong_layer is None:
        return "ambiguous"
    if wrong_layer <= int(early_cutoff):
        return "early_wrong"
    return "progressive_wrong"
