from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def fixed_uid_train_calibration_split(
    metadata: Sequence[dict[str, Any]], *, train_fraction: float, seed: int
) -> dict[str, np.ndarray]:
    """Split consistently across treatments whose outcomes may differ.

    K-specific hybrid policies can change preserve/harm/rescue labels.  The
    comparison split must therefore depend only on immutable UID, benchmark,
    and all-on correctness, rather than on a treatment outcome.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, bool], list[int]] = {}
    for index, row in enumerate(metadata):
        key = (str(row["benchmark"]), bool(row["baseline_correct"]))
        groups.setdefault(key, []).append(index)
    train: list[int] = []
    calibration: list[int] = []
    for key in sorted(groups):
        values = np.asarray(
            sorted(groups[key], key=lambda index: str(metadata[index]["uid"])),
            dtype=np.int64,
        )
        if len(values) < 2:
            raise ValueError(f"stratum {key!r} needs at least two samples")
        rng.shuffle(values)
        cut = min(len(values) - 1, max(1, int(np.floor(train_fraction * len(values)))))
        train.extend(values[:cut].tolist())
        calibration.extend(values[cut:].tolist())
    return {
        "train": np.asarray(sorted(train), dtype=np.int64),
        "calibration": np.asarray(sorted(calibration), dtype=np.int64),
    }


def stratified_train_calibration_split(
    metadata: Sequence[dict[str, Any]], *, train_fraction: float, seed: int
) -> dict[str, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(metadata):
        key = (str(row["benchmark"]), str(row["outcome"]))
        groups.setdefault(key, []).append(index)
    train: list[int] = []
    calibration: list[int] = []
    for key in sorted(groups):
        values = np.asarray(groups[key], dtype=np.int64)
        if len(values) < 2:
            raise ValueError(f"stratum {key!r} needs at least two samples")
        rng.shuffle(values)
        cut = min(len(values) - 1, max(1, int(np.floor(train_fraction * len(values)))))
        train.extend(values[:cut].tolist())
        calibration.extend(values[cut:].tolist())
    return {
        "train": np.asarray(sorted(train), dtype=np.int64),
        "calibration": np.asarray(sorted(calibration), dtype=np.int64),
    }


def override_scores_with_safe_admission(
    utility_scores: np.ndarray, safe_admission: np.ndarray
) -> np.ndarray:
    score = np.asarray(utility_scores, dtype=np.float64).copy()
    safe = np.asarray(safe_admission, dtype=bool)
    if score.shape != safe.shape or score.ndim != 1:
        raise ValueError("utility scores and safe admission must be aligned vectors")
    if not np.isfinite(score).all():
        raise ValueError("utility scores must be finite")
    score[safe] = float(score.min() - max(1.0, np.ptp(score) + 1.0))
    return score


def compose_admission_score(
    mode: str,
    harm_members: np.ndarray,
    rescue_members: np.ndarray,
    *,
    harm_beta: float,
    harm_threshold: float,
    utility_beta: float,
    rescue_weight: float,
) -> np.ndarray:
    """Build a lower-is-better admission score without train/eval drift."""

    from baseline_relative_visual_router.utility import conservative_utility_score

    harm = np.asarray(harm_members, dtype=np.float64)
    rescue = np.asarray(rescue_members, dtype=np.float64)
    if harm.shape != rescue.shape or harm.ndim != 2:
        raise ValueError("harm/rescue members must have the same [members, samples] shape")
    harm_score = harm.mean(0) + float(harm_beta) * harm.std(0)
    if mode == "harm_only":
        return harm_score
    utility = conservative_utility_score(
        harm,
        rescue,
        uncertainty_beta=float(utility_beta),
        rescue_weight=float(rescue_weight),
    )
    if mode == "utility_only":
        return utility
    if mode == "hierarchical":
        return override_scores_with_safe_admission(
            utility, harm_score <= float(harm_threshold)
        )
    raise ValueError(f"unknown admission score mode: {mode}")


def load_input_feature_cache(
    feature_dir: Path,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    paths = sorted(feature_dir.glob("input_features_shard_*_part_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no input feature parts under {feature_dir}")
    fields = {"instruction_mean": [], "instruction_last": [], "visual_summaries": []}
    metadata: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != "input_admission_features_v1":
            raise RuntimeError(f"feature schema mismatch: {path}")
        for field in fields:
            fields[field].append(payload[field])
        metadata.extend(payload["metadata"])
    tensors = {field: torch.cat(parts).float() for field, parts in fields.items()}
    uids = [str(row["uid"]) for row in metadata]
    if len(uids) != len(set(uids)):
        raise RuntimeError("duplicate input feature UIDs")
    if any(len(tensor) != len(metadata) for tensor in tensors.values()):
        raise RuntimeError("input feature and metadata counts differ")
    order = np.argsort(np.asarray(uids, dtype=object))
    tensors = {field: tensor[torch.as_tensor(order)] for field, tensor in tensors.items()}
    metadata = [metadata[int(index)] for index in order]
    return tensors, metadata


def input_feature_matrix(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    fields = [tensors["instruction_mean"], tensors["instruction_last"]]
    visual = tensors["visual_summaries"]
    if visual.ndim != 3 or visual.shape[1] != 2:
        raise ValueError("visual_summaries must have shape [N, 2, D]")
    fields.extend([visual[:, 0], visual[:, 1]])
    normalized = [torch.nn.functional.normalize(field.float(), p=2, dim=-1) for field in fields]
    result = torch.cat(normalized, dim=-1)
    if not torch.isfinite(result).all():
        raise RuntimeError("input feature matrix contains non-finite values")
    return result


def load_prefix_feature_cache(
    prefix_dir: Path,
    *,
    expected_prefix_layers: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    paths = sorted(prefix_dir.glob("prefix_*_shard_*_part_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no shared-prefix feature parts under {prefix_dir}")
    fields = {
        "instruction_mean": [],
        "instruction_window_mean": [],
        "instruction_last": [],
        "visual_summaries": [],
    }
    metadata: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != "shared_dense_prefix_actual_policy_v1":
            raise RuntimeError(f"prefix feature schema mismatch: {path}")
        if int(payload.get("prefix_layers", -1)) != int(expected_prefix_layers):
            raise RuntimeError(f"prefix layer mismatch: {path}")
        rows = payload["rows"]
        for row in rows:
            if int(row["prefix_layers"]) != int(expected_prefix_layers):
                raise RuntimeError(f"row prefix layer mismatch: {row.get('uid')}")
            mask = [int(value) for value in row["selected_visual_on_mask"]]
            if any(value != 1 for value in mask[:expected_prefix_layers]):
                raise RuntimeError(f"forced-prefix route invariant failed: {row.get('uid')}")
        for field in fields:
            fields[field].append(payload[field])
        metadata.extend(rows)
    tensors = {field: torch.cat(parts).float() for field, parts in fields.items()}
    uids = [str(row["uid"]) for row in metadata]
    if len(uids) != len(set(uids)):
        raise RuntimeError("duplicate shared-prefix feature UIDs")
    if any(len(tensor) != len(metadata) for tensor in tensors.values()):
        raise RuntimeError("shared-prefix feature and metadata counts differ")
    order = np.argsort(np.asarray(uids, dtype=object))
    tensors = {field: tensor[torch.as_tensor(order)] for field, tensor in tensors.items()}
    metadata = [metadata[int(index)] for index in order]
    return tensors, metadata


def prefix_feature_matrix(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    visual = tensors["visual_summaries"]
    if visual.ndim != 3 or visual.shape[1] != 2:
        raise ValueError("visual_summaries must have shape [N, 2, D]")
    fields = [
        tensors["instruction_mean"],
        tensors["instruction_window_mean"],
        tensors["instruction_last"],
        visual[:, 0],
        visual[:, 1],
    ]
    normalized = [torch.nn.functional.normalize(field.float(), p=2, dim=-1) for field in fields]
    result = torch.cat(normalized, dim=-1)
    if not torch.isfinite(result).all():
        raise RuntimeError("shared-prefix feature matrix contains non-finite values")
    return result
