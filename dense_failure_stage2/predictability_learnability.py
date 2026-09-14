"""Pure contracts for the frozen predictability Step-B learnability study."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


def _stable_key(seed: int, *parts: object) -> str:
    payload = "|".join((str(int(seed)), *(str(part) for part in parts)))
    return sha256(payload.encode()).hexdigest()


def _group_key(row: Mapping[str, Any]) -> str:
    return str(row.get("image_group_id") or row["uid"])


def _stratum(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row["dataset"]),
            str(row["source_regime"]),
            str(int(bool(row["dense_wrong"]))),
            str(int(bool(row["p90_triggered"]))),
        )
    )


def assign_shared_group_folds(
    rows: Sequence[Mapping[str, Any]], *, folds: int, seed: int
) -> list[dict[str, Any]]:
    """Assign base UIDs to deterministic approximately stratified group folds."""

    if folds < 2:
        raise ValueError("at least two folds are required")
    by_uid: dict[str, Mapping[str, Any]] = {}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        uid = str(row["uid"])
        if uid in by_uid:
            raise ValueError(f"duplicate UID: {uid}")
        by_uid[uid] = row
        groups[_group_key(row)].append(row)
    if len(groups) < folds:
        raise ValueError("fewer groups than folds")

    totals = Counter(_stratum(row) for row in rows)
    target_cells = {key: value / folds for key, value in totals.items()}
    target_uids = len(rows) / folds
    target_groups = len(groups) / folds
    fold_cells = [Counter() for _ in range(folds)]
    fold_uids = [0] * folds
    fold_groups = [0] * folds

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            min(totals[_stratum(row)] for row in item[1]),
            _stable_key(seed, item[0]),
        ),
    )
    assignment: dict[str, int] = {}
    for group_index, (group, members) in enumerate(ordered_groups):
        counts = Counter(_stratum(row) for row in members)
        if group_index < folds:
            candidates = [group_index]
        else:
            candidates = list(range(folds))

        def cost(fold: int) -> tuple[float, int, str]:
            cell_cost = sum(
                (
                    (
                        fold_cells[candidate][key]
                        + (counts.get(key, 0) if candidate == fold else 0)
                        - target_cells[key]
                    )
                    / max(target_cells[key], 1.0)
                )
                ** 2
                for candidate in range(folds)
                for key in target_cells
            )
            uid_cost = sum(
                (
                    (
                        fold_uids[candidate]
                        + (len(members) if candidate == fold else 0)
                        - target_uids
                    )
                    / max(target_uids, 1.0)
                )
                ** 2
                for candidate in range(folds)
            )
            group_cost = sum(
                (
                    (
                        fold_groups[candidate]
                        + (1 if candidate == fold else 0)
                        - target_groups
                    )
                    / max(target_groups, 1.0)
                )
                ** 2
                for candidate in range(folds)
            )
            return (cell_cost + 0.25 * uid_cost + 0.05 * group_cost, fold_uids[fold], _stable_key(seed, group, fold))

        selected = min(candidates, key=cost)
        assignment[group] = selected
        fold_cells[selected].update(counts)
        fold_uids[selected] += len(members)
        fold_groups[selected] += 1

    output = []
    for uid, row in sorted(by_uid.items()):
        output.append(
            {
                "uid": uid,
                "image_group_id": _group_key(row),
                "dataset": str(row["dataset"]),
                "source_regime": str(row["source_regime"]),
                "dense_wrong": bool(row["dense_wrong"]),
                "p90_triggered": bool(row["p90_triggered"]),
                "fold": int(assignment[_group_key(row)]),
            }
        )
    return output


def assign_inner_group_roles(
    registry: Sequence[Mapping[str, Any]],
    *,
    outer_fold: int,
    calibration_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Freeze fit/calibration roles without ever admitting the outer-test fold."""

    reciprocal = round(1.0 / float(calibration_fraction))
    if not 0.0 < calibration_fraction < 0.5 or not math.isclose(
        reciprocal * float(calibration_fraction), 1.0, abs_tol=1e-9
    ):
        raise ValueError("calibration fraction must be the reciprocal of an integer >=3")
    outer = int(outer_fold)
    train_rows = [dict(row) for row in registry if int(row["fold"]) != outer]
    if not train_rows:
        raise ValueError("outer fold leaves no training rows")
    local = assign_shared_group_folds(
        train_rows, folds=int(reciprocal), seed=int(seed) + 1009 * outer
    )
    local_fold = {str(row["uid"]): int(row["fold"]) for row in local}
    output = []
    for row in sorted(registry, key=lambda value: str(value["uid"])):
        if int(row["fold"]) == outer:
            role = "outer_test"
        else:
            role = "calibration" if local_fold[str(row["uid"])] == 0 else "fit"
        output.append({**dict(row), "outer_fold": outer, "role": role})
    return output


def uid_equal_state_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return per-state weights whose total is exactly one for every UID."""

    if not rows:
        raise ValueError("cannot weight an empty state population")
    counts = Counter(str(row["uid"]) for row in rows)
    if any(value < 1 for value in counts.values()):
        raise ValueError("invalid UID state count")
    return np.asarray([1.0 / counts[str(row["uid"])] for row in rows], dtype=np.float64)


@dataclass(frozen=True)
class RobustTargetScale:
    center: float
    scale: float

    def transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.center) / self.scale

    def inverse(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.center


def robust_target_scale(
    train_values: Sequence[float] | np.ndarray, *, floor: float
) -> RobustTargetScale:
    values = np.asarray(train_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("training targets must be a finite nonempty vector")
    if floor <= 0.0:
        raise ValueError("target scale floor must be positive")
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return RobustTargetScale(center=center, scale=max(1.4826 * mad, float(floor)))


@dataclass(frozen=True)
class FoldStandardizer:
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values.to(dtype=torch.float32) - self.mean.to(values.device)) / self.std.to(
            values.device
        )


def fit_standardizer(
    features: torch.Tensor, row_indices: Sequence[int] | np.ndarray, *, floor: float
) -> FoldStandardizer:
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must be a nonempty matrix")
    indices = torch.as_tensor(row_indices, dtype=torch.long)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("fit indices must be a nonempty vector")
    selected = features.index_select(0, indices).float()
    if not torch.isfinite(selected).all():
        raise ValueError("fit features contain non-finite values")
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, unbiased=False)
    std = torch.where(std < float(floor), torch.ones_like(std), std)
    return FoldStandardizer(mean=mean.cpu(), std=std.cpu())


def primary_utility_targets(row: Mapping[str, Any]) -> dict[str, float]:
    """Return the two prospectively fixed one-bit deviations from dense FULL."""

    read = float(row["u_read_w1"])
    write = float(row["u_write_r1"])
    if not math.isclose(read, float(row["q_full"]) - float(row["q_write_only"]), abs_tol=1e-9):
        raise ValueError("READ primary utility differs from q_F-q_WO")
    if not math.isclose(write, float(row["q_full"]) - float(row["q_read_only"]), abs_tol=1e-9):
        raise ValueError("WRITE primary utility differs from q_F-q_RO")
    return {"read": read, "write": write}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("binary labels and scores must be aligned vectors")
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return float("nan"), float("nan")
    ranks = _average_ranks(scores)
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    auroc = (float(ranks[positives].sum()) - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(ordered) + 1)
    auprc = float(precision[ordered == 1].mean())
    return float(auroc), auprc


def binary_classification_metrics(
    *, truth: Sequence[int] | np.ndarray, prediction: Sequence[float] | np.ndarray
) -> dict[str, float | int]:
    labels = np.asarray(truth, dtype=np.int64)
    scores = np.asarray(prediction, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("classification truth and prediction must be aligned nonempty vectors")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("classification inputs must be binary and finite")
    auroc, auprc = _binary_metrics(labels, scores)
    positives = int(labels.sum())
    return {
        "support": int(len(labels)),
        "positives": positives,
        "negatives": int(len(labels) - positives),
        "prevalence": float(positives / len(labels)),
        "auroc": auroc,
        "auprc": auprc,
    }


def harmful_ranking_metrics(
    *, truth: Sequence[float] | np.ndarray, prediction: Sequence[float] | np.ndarray
) -> dict[str, float | int]:
    truth_values = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(prediction, dtype=np.float64)
    if truth_values.ndim != 1 or predictions.ndim != 1 or len(truth_values) != len(predictions):
        raise ValueError("truth and prediction must be aligned vectors")
    if not np.isfinite(truth_values).all() or not np.isfinite(predictions).all():
        raise ValueError("harmful-ranking inputs must be finite")
    mask = truth_values != 0.0
    labels = (truth_values[mask] < 0.0).astype(np.int64)
    auroc, auprc = _binary_metrics(labels, -predictions[mask])
    return {
        "support": int(mask.sum()),
        "harmful": int(labels.sum()),
        "beneficial": int((labels == 0).sum()),
        "neutral_excluded": int((~mask).sum()),
        "auroc": auroc,
        "auprc": auprc,
    }


def regression_metrics(
    *, truth: Sequence[float] | np.ndarray, prediction: Sequence[float] | np.ndarray
) -> dict[str, float | int]:
    truth_values = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(prediction, dtype=np.float64)
    if truth_values.ndim != 1 or predictions.ndim != 1 or len(truth_values) != len(predictions):
        raise ValueError("regression truth and prediction must be aligned vectors")
    if len(truth_values) == 0 or not np.isfinite(truth_values).all() or not np.isfinite(predictions).all():
        raise ValueError("regression inputs must be finite and nonempty")
    centered_truth = truth_values - truth_values.mean()
    centered_prediction = predictions - predictions.mean()
    denominator = float(
        np.sqrt(np.square(centered_truth).sum() * np.square(centered_prediction).sum())
    )
    pearson = (
        float(np.dot(centered_truth, centered_prediction) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    truth_ranks = _average_ranks(truth_values)
    prediction_ranks = _average_ranks(predictions)
    rank_truth = truth_ranks - truth_ranks.mean()
    rank_prediction = prediction_ranks - prediction_ranks.mean()
    rank_denominator = float(
        np.sqrt(np.square(rank_truth).sum() * np.square(rank_prediction).sum())
    )
    spearman = (
        float(np.dot(rank_truth, rank_prediction) / rank_denominator)
        if rank_denominator > 0.0
        else float("nan")
    )
    error = predictions - truth_values
    return {
        "support": int(len(truth_values)),
        "spearman": spearman,
        "pearson": pearson,
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
    }


def uid_macro_regression_metrics(
    *,
    truth: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    uids: Sequence[str] | np.ndarray,
    minimum_nonconstant_states: int,
) -> dict[str, float | int]:
    truth_values = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(prediction, dtype=np.float64)
    uid_values = np.asarray(uids, dtype=str)
    if (
        truth_values.ndim != 1
        or predictions.ndim != 1
        or uid_values.ndim != 1
        or len(truth_values) != len(predictions)
        or len(truth_values) != len(uid_values)
    ):
        raise ValueError("UID-macro inputs must be aligned vectors")
    if minimum_nonconstant_states < 2:
        raise ValueError("minimum nonconstant states must be at least two")
    per_uid = []
    all_uids = sorted(set(uid_values.tolist()))
    for uid in all_uids:
        mask = uid_values == uid
        if int(mask.sum()) < int(minimum_nonconstant_states):
            continue
        if len(np.unique(truth_values[mask])) < 2 or len(np.unique(predictions[mask])) < 2:
            continue
        per_uid.append(regression_metrics(truth=truth_values[mask], prediction=predictions[mask]))
    return {
        "uids_total": len(all_uids),
        "uids_included": len(per_uid),
        "spearman": (
            float(np.nanmean([float(row["spearman"]) for row in per_uid]))
            if per_uid else float("nan")
        ),
        "pearson": (
            float(np.nanmean([float(row["pearson"]) for row in per_uid]))
            if per_uid else float("nan")
        ),
        "mae": (
            float(np.mean([float(row["mae"]) for row in per_uid]))
            if per_uid else float("nan")
        ),
        "rmse": (
            float(np.mean([float(row["rmse"]) for row in per_uid]))
            if per_uid else float("nan")
        ),
    }


def high_precision_harmful_metrics(
    *,
    truth: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    coverages: Sequence[float],
    precision_targets: Sequence[float],
) -> dict[str, float | int]:
    truth_values = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(prediction, dtype=np.float64)
    if truth_values.ndim != 1 or predictions.ndim != 1 or len(truth_values) != len(predictions):
        raise ValueError("harmful subset inputs must be aligned vectors")
    mask = truth_values != 0.0
    labels = (truth_values[mask] < 0.0).astype(np.int64)
    scores = -predictions[mask]
    if len(scores) == 0:
        raise ValueError("harmful subset has no non-neutral states")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    precision_curve = cumulative / np.arange(1, len(ordered) + 1)
    positives = int(labels.sum())
    recall_curve = cumulative / positives if positives else np.zeros_like(cumulative, dtype=float)
    output: dict[str, float | int] = {
        "support": int(len(labels)), "harmful": positives, "neutral_excluded": int((~mask).sum())
    }
    for coverage in coverages:
        if not 0.0 < float(coverage) <= 1.0:
            raise ValueError("coverage must lie in (0,1]")
        count = max(1, int(math.ceil(float(coverage) * len(labels))))
        output[f"precision_at_{float(coverage):g}"] = float(ordered[:count].mean())
    for target in precision_targets:
        valid = precision_curve >= float(target)
        output[f"recall_at_precision_{float(target):g}"] = (
            float(recall_curve[valid].max()) if valid.any() else 0.0
        )
    return output


def select_preservation_threshold(
    scores: Sequence[float] | np.ndarray,
    dense_wrong: Sequence[int] | np.ndarray,
    *,
    target_correct_preservation: float,
) -> float:
    """Select a strict-greater trigger threshold from calibration correct rows."""

    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(dense_wrong, dtype=np.int64)
    if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("preservation scores and labels must be aligned vectors")
    if not 0.0 < float(target_correct_preservation) <= 1.0:
        raise ValueError("preservation target must lie in (0,1]")
    correct = np.sort(values[labels == 0])
    if len(correct) == 0:
        raise ValueError("calibration population contains no correct samples")
    preserve = max(1, int(math.ceil(float(target_correct_preservation) * len(correct))))
    return float(correct[preserve - 1])


class SummaryScalarPredictor(nn.Module):
    """Prospectively fixed linear or two-layer summary predictor."""

    def __init__(
        self, *, kind: str, input_size: int, hidden_size: int, dropout: float
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("summary predictor dimensions must be positive")
        if kind == "linear":
            self.network = nn.Linear(input_size, 1)
        elif kind == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
        else:
            raise ValueError(f"unsupported summary predictor kind: {kind}")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError("summary predictor requires a feature matrix")
        return self.network(values).squeeze(-1)


class RouterStyleUtilityRegressor(nn.Module):
    """Current READ/WRITE cross-modal topology with one frozen-form scalar head."""

    REPRESENTATIONS = ("z_R", "z_W", "z_RW")

    def __init__(
        self,
        *,
        hidden_size: int,
        router_size: int,
        num_heads: int,
        dropout: float,
        readout_hidden_size: int,
        representation: str,
    ) -> None:
        super().__init__()
        if representation not in self.REPRESENTATIONS:
            raise ValueError(f"unsupported router representation: {representation}")
        if min(hidden_size, router_size, num_heads, readout_hidden_size) < 1:
            raise ValueError("router dimensions must be positive")
        if router_size % num_heads:
            raise ValueError("router size must be divisible by attention heads")
        self.hidden_size = int(hidden_size)
        self.router_size = int(router_size)
        self.representation = str(representation)
        self.text_query_projection = nn.Linear(hidden_size, router_size)
        self.visual_projection = nn.Linear(hidden_size, router_size)
        self.read_attention = nn.MultiheadAttention(
            router_size, num_heads, dropout=dropout, batch_first=True
        )
        self.write_attention = nn.MultiheadAttention(
            router_size, num_heads, dropout=dropout, batch_first=True
        )
        self.write_query = nn.Parameter(torch.empty(1, 1, router_size))
        nn.init.normal_(self.write_query, std=router_size**-0.5)
        input_size = 2 * router_size if representation == "z_RW" else router_size
        self.readout = nn.Sequential(
            nn.Linear(input_size, readout_hidden_size),
            nn.GELU(),
            nn.LayerNorm(readout_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden_size, 1),
        )

    def forward(
        self,
        text_states: torch.Tensor,
        visual_states: torch.Tensor,
        *,
        text_mask: torch.Tensor,
        visual_mask: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if text_states.ndim != 3 or visual_states.ndim != 3:
            raise ValueError("text and visual states must be [batch,tokens,hidden]")
        if text_states.shape[0] != visual_states.shape[0]:
            raise ValueError("text and visual batch sizes differ")
        if text_states.shape[-1] != self.hidden_size or visual_states.shape[-1] != self.hidden_size:
            raise ValueError("state width differs from router contract")
        if text_mask.shape != text_states.shape[:2] or visual_mask.shape != visual_states.shape[:2]:
            raise ValueError("state masks are not aligned")
        text_mask = text_mask.bool()
        visual_mask = visual_mask.bool()
        if not bool(text_mask.any(dim=1).all()) or not bool(visual_mask.any(dim=1).all()):
            raise ValueError("every row needs valid text and visual tokens")
        dtype = self.text_query_projection.weight.dtype
        text_states = text_states.to(dtype=dtype)
        visual_states = visual_states.to(dtype=dtype)
        batch = len(text_states)
        last = text_mask.long().sum(dim=1) - 1
        query = text_states[torch.arange(batch, device=text_states.device), last]
        query = self.text_query_projection(query).unsqueeze(1)
        visual = self.visual_projection(visual_states)
        padding = ~visual_mask
        read, _ = self.read_attention(
            query, visual, visual, key_padding_mask=padding, need_weights=False
        )
        write_query = self.write_query.expand(batch, -1, -1)
        write, _ = self.write_attention(
            write_query, visual, visual, key_padding_mask=padding, need_weights=False
        )
        z_read = read[:, 0]
        z_write = write[:, 0]
        selected = {
            "z_R": z_read,
            "z_W": z_write,
            "z_RW": torch.cat((z_read, z_write), dim=-1),
        }[self.representation]
        prediction = self.readout(selected).squeeze(-1)
        if return_details:
            return prediction, {"z_R": z_read, "z_W": z_write}
        return prediction


def validate_oof_completeness(
    expected: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> None:
    seen: set[str] = set()
    for row in rows:
        state_id = str(row["state_id"])
        if state_id in seen:
            raise ValueError(f"duplicate OOF state: {state_id}")
        seen.add(state_id)
        if state_id not in expected:
            raise ValueError(f"unexpected OOF state: {state_id}")
        if int(row["fold"]) != int(expected[state_id]["fold"]):
            raise ValueError(f"OOF fold mismatch: {state_id}")
    missing = sorted(set(expected).difference(seen))
    if missing:
        raise ValueError(f"missing OOF states: {missing[:3]}")


def validate_prediction_roundtrip(
    expected: Sequence[float] | np.ndarray,
    observed: Sequence[float] | np.ndarray,
    *,
    float32_ulp_multiplier: float,
) -> dict[str, float]:
    left = np.asarray(expected, dtype=np.float64)
    right = np.asarray(observed, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("roundtrip predictions must be aligned finite vectors")
    if float32_ulp_multiplier <= 0.0:
        raise ValueError("roundtrip ULP multiplier must be positive")
    difference = np.abs(left - right)
    scale = np.maximum(1.0, np.abs(left))
    scaled_ulps = difference / (np.finfo(np.float32).eps * scale)
    maximum = float(np.max(difference, initial=0.0))
    maximum_ulps = float(np.max(scaled_ulps, initial=0.0))
    if np.any(scaled_ulps > float(float32_ulp_multiplier)):
        raise ValueError(
            "checkpoint prediction roundtrip exceeds float32 ULP bound: "
            f"{maximum_ulps} > {float32_ulp_multiplier}"
        )
    return {"maximum_absolute_difference": maximum, "maximum_scaled_float32_ulps": maximum_ulps}
