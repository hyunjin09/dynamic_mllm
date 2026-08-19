from __future__ import annotations

import numpy as np


def conservative_utility_score(
    harm_members: np.ndarray,
    rescue_members: np.ndarray,
    *,
    uncertainty_beta: float,
    rescue_weight: float,
) -> np.ndarray:
    """Return lower-is-better treatment risk from ensemble probabilities."""

    if harm_members.shape != rescue_members.shape or harm_members.ndim != 2:
        raise ValueError("harm/rescue members must have the same [members, samples] shape")
    beta = float(uncertainty_beta)
    harm_ucb = harm_members.mean(axis=0) + beta * harm_members.std(axis=0)
    rescue_lcb = np.maximum(
        0.0, rescue_members.mean(axis=0) - beta * rescue_members.std(axis=0)
    )
    return harm_ucb - float(rescue_weight) * rescue_lcb
