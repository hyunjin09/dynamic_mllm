"""Pure helpers for the layer-wise current-dense failure probe diagnostic."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = ("text_final", "text_mean", "visual_mean")
CHECKPOINT_SCHEMA = "layerwise_dense_failure_probe_checkpoint_v1"


def _as_binary_vectors(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.shape != y.shape or y.size == 0:
        raise ValueError("binary metrics require equal nonempty vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(values).all():
        raise ValueError("binary metrics require binary labels and finite scores")
    if len(np.unique(y)) != 2:
        raise ValueError("binary ranking metrics require both classes")
    return y, values


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    ranks = _average_ranks(scores)
    statistic = ranks[positives].sum() - positive_count * (positive_count + 1) / 2
    return float(statistic / (positive_count * negative_count))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    previous_recall = 0.0
    result = 0.0
    for threshold in sorted(set(scores.tolist()), reverse=True):
        predicted = scores >= threshold
        true_positive = int(np.logical_and(predicted, labels == 1).sum())
        false_positive = int(np.logical_and(predicted, labels == 0).sum())
        recall = true_positive / positives
        precision = true_positive / max(true_positive + false_positive, 1)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return float(result)


def binary_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Return rank and fixed-threshold metrics with wrong as the positive class."""

    y, values = _as_binary_vectors(labels, scores)
    if not math.isfinite(float(threshold)):
        raise ValueError("binary decision threshold must be finite")
    predicted = values >= float(threshold)
    truth = y == 1
    true_positive = int(np.logical_and(predicted, truth).sum())
    true_negative = int(np.logical_and(~predicted, ~truth).sum())
    false_positive = int(np.logical_and(predicted, ~truth).sum())
    false_negative = int(np.logical_and(~predicted, truth).sum())
    wrong_recall = true_positive / max(true_positive + false_negative, 1)
    correct_preservation = true_negative / max(true_negative + false_positive, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    return {
        "records": int(len(y)),
        "correct": int((y == 0).sum()),
        "wrong": int((y == 1).sum()),
        "auroc": _auroc(y, values),
        "auprc": _average_precision(y, values),
        "threshold": float(threshold),
        "balanced_accuracy": float((wrong_recall + correct_preservation) / 2),
        "wrong_recall": float(wrong_recall),
        "failure_precision": float(precision),
        "correct_preservation": float(correct_preservation),
        "false_deviation_rate": float(1.0 - correct_preservation),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def preservation_operating_point(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    target_preservation: float,
) -> dict[str, float | int]:
    """Choose the most permissive tie-safe threshold meeting preservation."""

    y, values = _as_binary_vectors(labels, scores)
    target = float(target_preservation)
    if not 0.0 < target <= 1.0:
        raise ValueError("target preservation must lie in (0, 1]")
    correct_scores = np.sort(values[y == 0])[::-1]
    allowed = int(math.floor((1.0 - target) * len(correct_scores) + 1e-12))
    if allowed >= len(correct_scores):
        threshold = float(np.nextafter(correct_scores[-1], -np.inf))
    else:
        boundary = float(correct_scores[allowed])
        threshold = float(np.nextafter(boundary, np.inf))
    metrics = binary_metrics(y, values, threshold=threshold)
    if float(metrics["correct_preservation"]) + 1e-12 < target:
        raise RuntimeError("selected threshold violates requested preservation")
    return {
        **metrics,
        "target_preservation": target,
        "allowed_false_deviations": allowed,
    }


def compose_layer_features(
    shard: Mapping[str, Any],
    *,
    row_indices: Sequence[int] | torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Concatenate the three frozen summaries for selected rows at one layer."""

    layer_index = int(layer)
    tensors = []
    expected_prefix: tuple[int, int] | None = None
    indices = torch.as_tensor(row_indices, dtype=torch.long)
    for name in FEATURE_NAMES:
        value = shard.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(f"feature shard has no rank-3 tensor for {name}")
        if not 0 <= layer_index < value.shape[1]:
            raise ValueError(f"layer {layer_index} lies outside {name}")
        prefix = (int(value.shape[0]), int(value.shape[1]))
        if expected_prefix is None:
            expected_prefix = prefix
        elif prefix != expected_prefix:
            raise ValueError("feature tensors have incompatible record/layer shapes")
        tensors.append(value.index_select(0, indices)[:, layer_index, :])
    output = torch.cat(tensors, dim=-1)
    if not torch.isfinite(output.float()).all():
        raise ValueError("composed layer features contain non-finite values")
    return output


def _validate_probe_data(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError(f"{name} probe features/labels have incompatible shapes")
    if len(features) == 0 or not torch.isfinite(features.float()).all():
        raise ValueError(f"{name} probe features must be finite and nonempty")
    labels = labels.to(dtype=torch.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"{name} probe labels must contain both binary classes")
    return features, labels


def fit_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Fit one deterministic linear logit with validation-loss checkpointing."""

    train_features, train_labels = _validate_probe_data(
        train_features, train_labels, name="train"
    )
    validation_features, validation_labels = _validate_probe_data(
        validation_features, validation_labels, name="validation"
    )
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("train and validation probe dimensions differ")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    train_x = train_features.to(device=device, dtype=torch.float32)
    validation_x = validation_features.to(device=device, dtype=torch.float32)
    train_y = train_labels.to(device=device, dtype=torch.float32)
    validation_y = validation_labels.to(device=device, dtype=torch.float32)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    floor = float(config.get("normalization_std_floor", 1e-6))
    std = torch.where(std < floor, torch.ones_like(std), std)
    train_x = (train_x - mean) / std
    validation_x = (validation_x - mean) / std

    model = nn.Linear(train_x.shape[1], 1, bias=True, device=device, dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    maximum_epochs = int(config["maximum_epochs"])
    minimum_epochs = int(config["minimum_epochs"])
    patience = int(config["early_stopping_patience"])
    batch_size = int(config["batch_size"])
    if not 1 <= minimum_epochs <= maximum_epochs or patience < 1 or batch_size < 1:
        raise ValueError("invalid linear-probe optimization schedule")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    best_loss = float("inf")
    best_epoch = -1
    best_weight: torch.Tensor | None = None
    best_bias: torch.Tensor | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(maximum_epochs):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        accumulated_loss = 0.0
        for start in range(0, len(permutation), batch_size):
            batch_cpu = permutation[start : start + batch_size]
            batch = batch_cpu.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x.index_select(0, batch)).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, train_y.index_select(0, batch)
            )
            loss.backward()
            optimizer.step()
            accumulated_loss += float(loss.detach().cpu()) * len(batch_cpu)

        model.eval()
        with torch.inference_mode():
            validation_logits = model(validation_x).squeeze(-1)
            validation_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    validation_logits, validation_y
                ).cpu()
            )
        train_loss = accumulated_loss / len(train_x)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_weight = model.weight.detach().cpu().clone()
            best_bias = model.bias.detach().cpu().clone()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch + 1 >= minimum_epochs and stale_epochs >= patience:
            break

    if best_weight is None or best_bias is None or best_epoch < 0:
        raise RuntimeError("linear probe did not produce a validation checkpoint")
    result = {
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
        "weight": best_weight,
        "bias": best_bias,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "history": history,
    }
    train_scores = score_linear_probe(train_features, result, device=device)
    validation_scores = score_linear_probe(validation_features, result, device=device)
    train_logits = torch.logit(train_scores.clamp(1e-7, 1 - 1e-7))
    validation_logits = torch.logit(validation_scores.clamp(1e-7, 1 - 1e-7))
    result["train_loss"] = float(
        nn.functional.binary_cross_entropy_with_logits(
            train_logits, train_labels.float().cpu()
        )
    )
    result["validation_loss"] = float(
        nn.functional.binary_cross_entropy_with_logits(
            validation_logits, validation_labels.float().cpu()
        )
    )
    return result


def score_linear_probe(
    features: torch.Tensor,
    state: Mapping[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Apply a frozen train-normalized linear probe and return CPU probabilities."""

    required = ("mean", "std", "weight", "bias")
    if any(not isinstance(state.get(name), torch.Tensor) for name in required):
        raise ValueError("linear probe state is incomplete")
    values = features.to(device=device, dtype=torch.float32)
    mean = state["mean"].to(device=device, dtype=torch.float32)
    std = state["std"].to(device=device, dtype=torch.float32)
    weight = state["weight"].to(device=device, dtype=torch.float32)
    bias = state["bias"].to(device=device, dtype=torch.float32)
    if values.ndim != 2 or values.shape[1] != mean.numel():
        raise ValueError("features do not match frozen probe normalization")
    with torch.inference_mode():
        logits = nn.functional.linear((values - mean) / std, weight, bias).squeeze(-1)
        probabilities = torch.sigmoid(logits)
    return probabilities.detach().cpu()


def validate_complete_layer_records(
    records: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int] = tuple(range(28)),
) -> list[dict[str, Any]]:
    expected = [int(layer) for layer in layers]
    observed = Counter(int(row["layer"]) for row in records)
    if set(observed) != set(expected) or any(observed[layer] != 1 for layer in expected):
        raise ValueError("expected exactly one result for every declared layer")
    return sorted((dict(row) for row in records), key=lambda row: int(row["layer"]))


def validate_checkpoint_provenance(
    payload: Mapping[str, Any],
    *,
    layer: int,
    expected: Mapping[str, str],
) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported probe checkpoint schema")
    if int(payload.get("layer", -1)) != int(layer):
        raise ValueError("probe checkpoint layer mismatch")
    if tuple(payload.get("input_features", ())) != FEATURE_NAMES:
        raise ValueError("probe checkpoint input feature contract mismatch")
    mismatches = [
        key
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(f"probe checkpoint provenance mismatch: {mismatches}")
