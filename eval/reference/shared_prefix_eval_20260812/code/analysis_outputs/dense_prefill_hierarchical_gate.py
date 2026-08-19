"""Shared feature and model utilities for the dense-prefill hierarchical gate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


HIDDEN_SIZE = 3584
CONFIDENCE_SIZE = 3
INPUT_SIZE = 2 * HIDDEN_SIZE + CONFIDENCE_SIZE


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_feature_parts(feature_dir: Path) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    paths = sorted(feature_dir.glob("features_shard_*_part_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no feature parts under {feature_dir}")
    means: list[torch.Tensor] = []
    lasts: list[torch.Tensor] = []
    confidence: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(payload["confidence_fields"]) != (
            "top1_logprob",
            "top1_top2_margin",
            "entropy",
        ):
            raise RuntimeError(f"feature schema mismatch: {path}")
        means.append(payload["instruction_mean"])
        lasts.append(payload["instruction_last"])
        confidence.append(payload["confidence"])
        metadata.extend(payload["metadata"])
    tensors = {
        "instruction_mean": torch.cat(means).float(),
        "instruction_last": torch.cat(lasts).float(),
        "confidence": torch.cat(confidence).float(),
    }
    n = len(metadata)
    if any(len(value) != n for value in tensors.values()):
        raise RuntimeError("feature and metadata row counts differ")
    uids = [str(row["uid"]) for row in metadata]
    if len(uids) != len(set(uids)):
        raise RuntimeError("duplicate feature UIDs")
    order = np.argsort(np.asarray(uids, dtype=object))
    tensors = {key: value[torch.as_tensor(order)] for key, value in tensors.items()}
    metadata = [metadata[int(index)] for index in order]
    return tensors, metadata


@dataclass(frozen=True)
class FeatureScaler:
    confidence_mean: torch.Tensor
    confidence_std: torch.Tensor

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "confidence_mean": self.confidence_mean.cpu(),
            "confidence_std": self.confidence_std.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "FeatureScaler":
        return cls(
            confidence_mean=torch.as_tensor(state["confidence_mean"]).float(),
            confidence_std=torch.as_tensor(state["confidence_std"]).float(),
        )


def fit_scaler(tensors: dict[str, torch.Tensor], indices: np.ndarray) -> FeatureScaler:
    confidence = tensors["confidence"][torch.as_tensor(indices)].float()
    std = confidence.std(dim=0, unbiased=False).clamp_min(1e-6)
    return FeatureScaler(confidence.mean(dim=0), std)


def transform_features(
    tensors: dict[str, torch.Tensor], scaler: FeatureScaler, indices: np.ndarray | None = None
) -> torch.Tensor:
    if indices is None:
        mean = tensors["instruction_mean"].float()
        last = tensors["instruction_last"].float()
        confidence = tensors["confidence"].float()
    else:
        index = torch.as_tensor(indices, dtype=torch.long)
        mean = tensors["instruction_mean"][index].float()
        last = tensors["instruction_last"][index].float()
        confidence = tensors["confidence"][index].float()
    mean = torch.nn.functional.normalize(mean, p=2, dim=-1)
    last = torch.nn.functional.normalize(last, p=2, dim=-1)
    confidence = (confidence - scaler.confidence_mean) / scaler.confidence_std
    result = torch.cat((mean, last, confidence), dim=-1)
    if result.shape[1] != INPUT_SIZE or not torch.isfinite(result).all():
        raise RuntimeError(f"invalid transformed feature tensor {tuple(result.shape)}")
    return result


class GateHead(nn.Module):
    def __init__(self, architecture: str, input_size: int = INPUT_SIZE, hidden_size: int = 256):
        super().__init__()
        self.architecture = architecture
        if architecture == "linear":
            self.network = nn.Linear(input_size, 1)
        elif architecture == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1),
            )
        else:
            raise ValueError(f"unknown architecture {architecture!r}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@torch.inference_mode()
def predict_ensemble(
    features: torch.Tensor,
    state_dicts: list[dict[str, torch.Tensor]],
    *,
    architecture: str,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    if not state_dicts:
        raise ValueError("ensemble must contain at least one member")
    probabilities = torch.zeros(len(features), dtype=torch.float32)
    for state_dict in state_dicts:
        model = GateHead(architecture).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        member = []
        for start in range(0, len(features), batch_size):
            logits = model(features[start : start + batch_size].to(device))
            member.append(torch.sigmoid(logits).cpu())
        probabilities += torch.cat(member) / len(state_dicts)
    return probabilities.numpy()


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    prediction = scores >= threshold
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives and negatives:
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
        start = 0
        while start < len(scores):
            end = start + 1
            while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[start:end] = ranks[start:end].mean()
            start = end
        rank_by_original = np.empty_like(ranks)
        rank_by_original[order] = ranks
        auroc = (
            rank_by_original[labels == 1].sum() - positives * (positives + 1) / 2
        ) / (positives * negatives)
    else:
        auroc = float("nan")
    if positives:
        descending = np.argsort(-scores, kind="stable")
        sorted_labels = labels[descending]
        precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
        auprc = float((precision * sorted_labels).sum() / positives)
    else:
        auprc = float("nan")
    return {
        "n": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "auroc": float(auroc),
        "auprc": auprc,
        "accuracy_at_0_5": float((prediction == labels).mean()),
    }
