"""Shared Stage-1 predictor and trajectory-level risk calibration helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from dense_failure_stage1.layerwise_probe import binary_metrics
from dense_failure_stage1.sequential_gate import gate_metrics


VARIANTS = (
    "state_only",
    "layer_only",
    "state_layer_all28",
    "state_layer_random4",
)
MAIN_VARIANTS = ("state_layer_all28", "state_layer_random4")
CHECKPOINT_SCHEMA = "shared_stage1_global_risk_gate_checkpoint_v1"


class SharedFailurePredictor(nn.Module):
    """One small predictor shared across all decoder layers."""

    def __init__(
        self,
        *,
        variant: str,
        input_size: int,
        projection_size: int,
        layer_embedding_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unsupported shared-predictor variant: {variant}")
        if min(input_size, projection_size, layer_embedding_size, hidden_size) < 1:
            raise ValueError("shared-predictor dimensions must be positive")
        self.variant = variant
        self.uses_state = variant != "layer_only"
        self.uses_layer = variant != "state_only"
        self.projection = (
            nn.Linear(input_size, projection_size) if self.uses_state else None
        )
        self.layer_embedding = (
            nn.Embedding(28, layer_embedding_size) if self.uses_layer else None
        )
        head_input = (
            (projection_size if self.uses_state else 0)
            + (layer_embedding_size if self.uses_layer else 0)
        )
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, states: torch.Tensor | None, layers: torch.Tensor
    ) -> torch.Tensor:
        if layers.ndim != 1 or layers.dtype != torch.long:
            raise ValueError("layer indices must be a rank-1 long tensor")
        if len(layers) == 0 or int(layers.min()) < 0 or int(layers.max()) >= 28:
            raise ValueError("layer indices must lie in 0..27")
        pieces = []
        if self.uses_state:
            if states is None or states.ndim != 2 or len(states) != len(layers):
                raise ValueError("state-using variants require aligned rank-2 states")
            pieces.append(self.projection(states))
        elif states is not None:
            raise ValueError("layer-only predictor must not receive sample state")
        if self.uses_layer:
            pieces.append(self.layer_embedding(layers))
        return self.head(torch.cat(pieces, dim=-1)).squeeze(-1)


def deterministic_random_layers(
    records: int, *, epoch: int, seed: int, count: int = 4
) -> torch.Tensor:
    """Sample balanced-in-expectation layers without replacement per record."""

    if records < 1 or epoch < 0 or not 1 <= count <= 28:
        raise ValueError("invalid deterministic random-layer request")
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + 1_000_003 * int(epoch)
    )
    priorities = torch.rand((records, 28), generator=generator)
    return priorities.argsort(dim=1, stable=True)[:, :count]


def validate_score_matrix(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 28:
        raise ValueError("shared gate requires an N x 28 score matrix")
    if y.ndim != 1 or len(y) != len(values) or len(y) == 0:
        raise ValueError("scores and labels must be nonempty and aligned")
    if set(np.unique(y)) != {0, 1} or not np.isfinite(values).all():
        raise ValueError("shared gate requires finite scores and both classes")
    return values, y


def first_trigger_layers(
    scores: Sequence[Sequence[float]] | np.ndarray,
    *,
    threshold: float,
    window: Sequence[int],
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    layers = np.asarray(tuple(int(layer) for layer in window), dtype=np.int64)
    tau = float(threshold)
    if values.ndim != 2 or values.shape[1] != 28 or not np.isfinite(values).all():
        raise ValueError("trigger scores must be finite N x 28")
    if layers.ndim != 1 or len(layers) == 0 or not np.array_equal(
        layers, np.unique(layers)
    ) or not np.isin(layers, np.arange(28)).all():
        raise ValueError("gate window must be sorted unique layers in 0..27")
    if not math.isfinite(tau):
        raise ValueError("global threshold must be finite")
    crossings = values[:, layers] > tau
    triggered = crossings.any(axis=1)
    first_offsets = crossings.argmax(axis=1)
    first = layers[first_offsets]
    first[~triggered] = -1
    return first.astype(np.int64)


def trajectory_operating_point(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    target_preservation: float,
    window: Sequence[int],
) -> dict[str, Any]:
    """Choose the most permissive tie-safe global threshold on trajectories."""

    values, y = validate_score_matrix(scores, labels)
    target = float(target_preservation)
    if not 0.0 < target <= 1.0:
        raise ValueError("target preservation must lie in (0, 1]")
    layers = np.asarray(tuple(int(layer) for layer in window), dtype=np.int64)
    if len(layers) == 0 or not np.array_equal(layers, np.unique(layers)):
        raise ValueError("trajectory window is invalid")
    maxima = values[:, layers].max(axis=1)
    correct = np.sort(maxima[y == 0])[::-1]
    allowed = int(math.floor((1.0 - target) * len(correct) + 1e-12))
    if allowed >= len(correct):
        threshold = float(np.nextafter(correct[-1], -np.inf))
    else:
        threshold = float(correct[allowed])
    first = first_trigger_layers(values, threshold=threshold, window=layers)
    metrics = gate_metrics(y, first)
    if float(metrics["correct_preservation"]) + 1e-12 < target:
        raise RuntimeError("global trajectory threshold violates preservation target")
    return {
        "target_preservation": target,
        "allowed_correct_false_triggers": allowed,
        "threshold": threshold,
        "window": layers.tolist(),
        **metrics,
    }


def threshold_sweep(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    window: Sequence[int],
) -> list[dict[str, Any]]:
    """Evaluate every attainable strict-crossing trajectory threshold."""

    values, y = validate_score_matrix(scores, labels)
    layers = np.asarray(tuple(int(layer) for layer in window), dtype=np.int64)
    maxima = values[:, layers].max(axis=1)
    thresholds = sorted(set(maxima.tolist()), reverse=True)
    output = []
    for threshold in thresholds:
        first = first_trigger_layers(values, threshold=threshold, window=layers)
        output.append({"threshold": float(threshold), **gate_metrics(y, first)})
    return output


def layerwise_ranking_metrics(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> list[dict[str, Any]]:
    values, y = validate_score_matrix(scores, labels)
    return [
        {"layer": layer, **binary_metrics(y, values[:, layer])}
        for layer in range(28)
    ]


def score_space_rows(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> list[dict[str, Any]]:
    values, y = validate_score_matrix(scores, labels)
    output = []
    for layer in range(28):
        output.append(
            {
                "row_type": "distribution",
                "layer": layer,
                "next_layer": "",
                "class": "correct",
                "mean": float(values[y == 0, layer].mean()),
                "std": float(values[y == 0, layer].std()),
                "correlation": "",
            }
        )
        output.append(
            {
                "row_type": "distribution",
                "layer": layer,
                "next_layer": "",
                "class": "wrong",
                "mean": float(values[y == 1, layer].mean()),
                "std": float(values[y == 1, layer].std()),
                "correlation": "",
            }
        )
    for layer in range(27):
        for name, mask in (("overall", np.ones(len(y), dtype=bool)), ("correct", y == 0), ("wrong", y == 1)):
            left = values[mask, layer]
            right = values[mask, layer + 1]
            correlation = (
                float(np.corrcoef(left, right)[0, 1])
                if float(left.std()) > 0.0 and float(right.std()) > 0.0
                else None
            )
            output.append(
                {
                    "row_type": "neighbor_correlation",
                    "layer": layer,
                    "next_layer": layer + 1,
                    "class": name,
                    "mean": "",
                    "std": "",
                    "correlation": correlation,
                }
            )
    return output


def select_candidate(
    candidates: Sequence[Mapping[str, Any]], *, preferred: str
) -> dict[str, Any]:
    """Freeze a validation candidate by the prospective 99%-risk rule."""

    if not candidates:
        raise ValueError("cannot select from no validation candidates")

    def key(row: Mapping[str, Any]) -> tuple[float, float, float, int]:
        return (
            float(row["wrong_detection_recall"]),
            float(row["correct_preservation"]),
            float(row["failure_precision"]),
            int(str(row["name"]) == preferred),
        )

    return dict(max(candidates, key=key))
