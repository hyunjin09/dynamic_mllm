"""Cache-only content-feature probes for Phase 5B v3.7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from dvr_qwen.phase5b_v3_6_separability import binary_safe_vs_harm_label
from dvr_qwen.route_selector import NUM_LAYERS, route_mask_from_layers


EPS = 1e-8


@dataclass
class LayerContentProjection:
    mean: torch.Tensor
    scale: torch.Tensor
    components: torch.Tensor
    output_dim: int
    actual_rank: int
    input_feature_names: list[str]
    explained_variance: float
    seed: int


@dataclass
class LinearSafeHarmProbe:
    weights: torch.Tensor
    bias: torch.Tensor
    fit_summary: dict[str, Any]


def _mean_per_layer(value: torch.Tensor) -> torch.Tensor:
    return value.float().mean(dim=-1)


def _rms_per_layer(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(value.float().pow(2).mean(dim=-1).clamp_min(0.0))


def _abs_mean_per_layer(value: torch.Tensor) -> torch.Tensor:
    return value.float().abs().mean(dim=-1)


def _cosine_per_layer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=-1, eps=EPS)


def _diff_rms_per_layer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _rms_per_layer(left.float() - right.float())


def layer_content_scalar_matrix(sample: dict[str, Any]) -> tuple[torch.Tensor, list[str]]:
    """Build cheap per-layer text/visual content scalars from one cached sample."""

    global_mean = torch.as_tensor(sample["global_mean"]).float()
    window_mean = torch.as_tensor(sample["window_mean"]).float()
    last_token = torch.as_tensor(sample["last_token"]).float()
    if global_mean.ndim != 2:
        raise ValueError(f"global_mean must have shape (L, D), got {tuple(global_mean.shape)}")
    if tuple(window_mean.shape) != tuple(global_mean.shape) or tuple(last_token.shape) != tuple(global_mean.shape):
        raise ValueError("text summary tensors must share shape")
    num_layers = int(global_mean.shape[0])
    visual_summaries = torch.as_tensor(sample.get("visual_summaries", torch.zeros(num_layers, 2, global_mean.shape[1]))).float()
    if visual_summaries.ndim == 2:
        visual_mean = visual_summaries
        visual_abs = torch.zeros_like(visual_mean)
    elif visual_summaries.ndim == 3:
        if int(visual_summaries.shape[0]) != num_layers or int(visual_summaries.shape[-1]) != int(global_mean.shape[-1]):
            raise ValueError("visual_summaries shape is incompatible with text summaries")
        visual_mean = visual_summaries[:, 0, :]
        visual_abs = visual_summaries[:, min(1, int(visual_summaries.shape[1]) - 1), :]
    else:
        raise ValueError(f"visual_summaries must have rank 2 or 3, got {visual_summaries.ndim}")

    vectors = {
        "global": global_mean,
        "window": window_mean,
        "last": last_token,
        "visual_mean": visual_mean,
        "visual_abs": visual_abs,
    }
    columns: list[torch.Tensor] = []
    names: list[str] = []
    for name, value in vectors.items():
        columns.extend([_mean_per_layer(value), _rms_per_layer(value), _abs_mean_per_layer(value)])
        names.extend([f"content_mean_{name}", f"content_rms_{name}", f"content_abs_mean_{name}"])

    pairs = [
        ("global", "visual_mean"),
        ("global", "visual_abs"),
        ("window", "visual_mean"),
        ("window", "visual_abs"),
        ("last", "visual_mean"),
        ("last", "visual_abs"),
        ("global", "window"),
        ("global", "last"),
        ("window", "last"),
    ]
    for left_name, right_name in pairs:
        columns.append(_cosine_per_layer(vectors[left_name], vectors[right_name]))
        names.append(f"content_cos_{left_name}_{right_name}")
    for left_name, right_name in [("global", "visual_mean"), ("window", "visual_mean"), ("last", "visual_mean")]:
        columns.append(_diff_rms_per_layer(vectors[left_name], vectors[right_name]))
        names.append(f"content_diff_rms_{left_name}_{right_name}")

    layer_position = torch.linspace(0.0, 1.0, steps=num_layers, dtype=torch.float32)
    columns.append(layer_position)
    names.append("content_layer_position_norm")
    matrix = torch.stack(columns, dim=1).float()
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("content scalar matrix contains non-finite values")
    return matrix, names


def fit_layer_content_projection(
    fit_stats_by_id: dict[str, torch.Tensor],
    *,
    feature_names: list[str],
    output_dim: int,
    seed: int,
) -> LayerContentProjection:
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if not fit_stats_by_id:
        raise ValueError("cannot fit content projection from zero samples")
    matrices = []
    for sample_id, matrix in sorted(fit_stats_by_id.items()):
        current = torch.as_tensor(matrix).float()
        if current.ndim != 2:
            raise ValueError(f"content stats for {sample_id} must have shape (L, F)")
        if int(current.shape[1]) != len(feature_names):
            raise ValueError(f"content stats for {sample_id} do not match feature_names")
        matrices.append(current)
    train_matrix = torch.cat(matrices, dim=0)
    mean = train_matrix.mean(dim=0)
    scale = train_matrix.std(dim=0).clamp_min(1e-6)
    standardized = (train_matrix - mean) / scale
    rank = min(int(output_dim), int(standardized.shape[0]), int(standardized.shape[1]))
    if rank <= 0:
        raise ValueError("content projection rank must be positive")
    torch.manual_seed(int(seed))
    _, singular_values, components = torch.pca_lowrank(standardized, q=rank, center=False, niter=2)
    total_variance = float(standardized.pow(2).sum().item())
    explained = float(singular_values[:rank].pow(2).sum().item())
    return LayerContentProjection(
        mean=mean.float(),
        scale=scale.float(),
        components=components[:, :rank].float().contiguous(),
        output_dim=int(output_dim),
        actual_rank=int(rank),
        input_feature_names=list(feature_names),
        explained_variance=explained / total_variance if total_variance > 0.0 else 0.0,
        seed=int(seed),
    )


def project_layer_content(stats: torch.Tensor, projection: LayerContentProjection) -> torch.Tensor:
    stats = torch.as_tensor(stats).float()
    if stats.ndim != 2:
        raise ValueError("stats must have shape (L, F)")
    if int(stats.shape[1]) != int(projection.mean.numel()):
        raise ValueError("stats feature dimension does not match projection")
    projected = ((stats - projection.mean) / projection.scale.clamp_min(1e-6)) @ projection.components
    if int(projected.shape[1]) < int(projection.output_dim):
        projected = F.pad(projected, (0, int(projection.output_dim) - int(projected.shape[1])))
    return projected[:, : int(projection.output_dim)].float().contiguous()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().view(-1, 1)
    denom = weights.sum().clamp_min(1.0)
    return (values.float() * weights).sum(dim=0) / denom


def _region_mean(values: torch.Tensor, route: torch.Tensor, start: int, end: int) -> torch.Tensor:
    region_mask = torch.zeros_like(route.float())
    region_mask[start:end] = route.float()[start:end]
    return _masked_mean(values, region_mask)


def route_conditioned_content_features(
    row: dict[str, Any],
    projected_layers: torch.Tensor,
    *,
    num_layers: int = NUM_LAYERS,
) -> tuple[torch.Tensor, list[str]]:
    projected = torch.as_tensor(projected_layers).float()
    if projected.ndim != 2 or int(projected.shape[0]) != int(num_layers):
        raise ValueError(f"projected_layers must have shape ({num_layers}, D), got {tuple(projected.shape)}")
    dim = int(projected.shape[1])
    route = route_mask_from_layers([int(layer) for layer in row.get("layers_one_based", [])], num_layers=num_layers).float()
    off_route = 1.0 - route
    first_cut = int(num_layers) // 3
    second_cut = (2 * int(num_layers)) // 3 + (1 if int(num_layers) % 3 else 0)
    selected_indices = torch.nonzero(route > 0.5, as_tuple=False).view(-1)

    all_mean = projected.mean(dim=0)
    selected_mean = _masked_mean(projected, route)
    unselected_mean = _masked_mean(projected, off_route)
    selected_sum_norm = (projected * route.view(-1, 1)).sum(dim=0) / float(num_layers)
    early_selected_mean = _region_mean(projected, route, 0, first_cut)
    middle_selected_mean = _region_mean(projected, route, first_cut, second_cut)
    late_selected_mean = _region_mean(projected, route, second_cut, int(num_layers))
    if int(selected_indices.numel()):
        first_selected = projected[int(selected_indices[0].item())]
        last_selected = projected[int(selected_indices[-1].item())]
    else:
        first_selected = torch.zeros(dim, dtype=torch.float32)
        last_selected = torch.zeros(dim, dtype=torch.float32)

    blocks = {
        "all_mean": all_mean,
        "selected_mean": selected_mean,
        "unselected_mean": unselected_mean,
        "selected_minus_unselected": selected_mean - unselected_mean,
        "selected_sum_norm": selected_sum_norm,
        "early_selected_mean": early_selected_mean,
        "middle_selected_mean": middle_selected_mean,
        "late_selected_mean": late_selected_mean,
        "first_selected": first_selected,
        "last_selected": last_selected,
    }
    names = [f"content_pca_{block}_{idx}" for block in blocks for idx in range(dim)]
    return torch.cat([value.float().view(-1) for value in blocks.values()], dim=0), names


def build_route_conditioned_content_feature_matrix(
    rows: list[dict[str, Any]],
    projected_by_id: dict[str, torch.Tensor],
    *,
    num_layers: int = NUM_LAYERS,
) -> tuple[torch.Tensor, list[str]]:
    if not rows:
        raise ValueError("cannot build content features for zero rows")
    features: list[torch.Tensor] = []
    feature_names: list[str] | None = None
    for row in rows:
        sample_id = str(row["id"])
        if sample_id not in projected_by_id:
            raise ValueError(f"missing projected content features for {sample_id}")
        feature, names = route_conditioned_content_features(row, projected_by_id[sample_id], num_layers=num_layers)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("inconsistent content feature schema")
        features.append(feature)
    if feature_names is None:
        raise ValueError("no content features were built")
    return torch.stack(features, dim=0).float(), feature_names


def safe_harm_indices_and_labels(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    indices: list[int] = []
    labels: list[int] = []
    for idx, row in enumerate(rows):
        label = binary_safe_vs_harm_label(row)
        if label is None:
            continue
        indices.append(idx)
        labels.append(int(label))
    return torch.tensor(indices, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)


def _binary_metrics_from_predictions(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    preds = preds.long().view(-1)
    labels = labels.long().view(-1)
    if int(preds.numel()) != int(labels.numel()):
        raise ValueError("prediction and label counts differ")
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    pos = tp + fn
    neg = tn + fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / pos if pos else 0.0
    specificity = tn / neg if neg else 0.0
    return {
        "num_examples": int(labels.numel()),
        "positive": pos,
        "negative": neg,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": (tp + tn) / int(labels.numel()) if int(labels.numel()) else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity) if int(labels.numel()) else 0.0,
        "f1": 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall),
    }


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    pos = scores[labels > 0.5].float()
    neg = scores[labels <= 0.5].float()
    if int(pos.numel()) == 0 or int(neg.numel()) == 0:
        return None
    comparisons = (pos.view(-1, 1) > neg.view(1, -1)).float()
    ties = (pos.view(-1, 1) == neg.view(1, -1)).float() * 0.5
    return float((comparisons + ties).mean().item())


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    labels = labels.float().view(-1)
    if int((labels > 0.5).sum().item()) == 0:
        return None
    order = torch.argsort(scores.float().view(-1), descending=True)
    sorted_labels = labels[order]
    cumulative_tp = torch.cumsum((sorted_labels > 0.5).float(), dim=0)
    positions = torch.arange(1, int(labels.numel()) + 1, dtype=torch.float32)
    precision_at_k = cumulative_tp / positions
    return float(precision_at_k[sorted_labels > 0.5].mean().item())


def evaluate_probe_scores(scores: torch.Tensor, rows: list[dict[str, Any]], *, threshold: float = 0.0) -> dict[str, Any]:
    indices, labels = safe_harm_indices_and_labels(rows)
    if int(indices.numel()) == 0:
        return _binary_metrics_from_predictions(torch.empty(0), torch.empty(0))
    selected_scores = scores.float().view(-1)[indices]
    preds = (selected_scores >= float(threshold)).long()
    metrics = _binary_metrics_from_predictions(preds, labels)
    metrics["roc_auc"] = _roc_auc(selected_scores, labels)
    metrics["average_precision"] = _average_precision(selected_scores, labels)
    pos_count = int((labels > 0.5).sum().item())
    top_k = min(pos_count, int(labels.numel()))
    if top_k > 0:
        order = torch.argsort(selected_scores, descending=True)[:top_k]
        top_preds = torch.zeros_like(labels, dtype=torch.long)
        top_preds[order] = 1
        metrics["top_k_at_positive_count"] = _binary_metrics_from_predictions(top_preds, labels)
    else:
        metrics["top_k_at_positive_count"] = _binary_metrics_from_predictions(torch.zeros_like(labels), labels)
    return metrics


def train_linear_safe_harm_probe(
    features: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    steps: int = 200,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> LinearSafeHarmProbe:
    indices, labels = safe_harm_indices_and_labels(rows)
    if int(indices.numel()) == 0:
        raise ValueError("cannot train probe without safe/harm examples")
    positives = int((labels > 0.5).sum().item())
    negatives = int((labels <= 0.5).sum().item())
    if positives == 0 or negatives == 0:
        raise ValueError("safe/harm probe requires both positive and negative examples")
    x = features.float()[indices]
    y = labels.float()
    torch.manual_seed(int(seed))
    linear = torch.nn.Linear(int(x.shape[1]), 1)
    pos_weight = torch.tensor([negatives / max(float(positives), 1.0)], dtype=torch.float32)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    initial_loss = None
    for step_idx in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = linear(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        if initial_loss is None:
            initial_loss = float(loss.detach().item())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_logits = linear(x).view(-1)
        final_loss = F.binary_cross_entropy_with_logits(final_logits, y, pos_weight=pos_weight)
    return LinearSafeHarmProbe(
        weights=linear.weight.detach().view(-1).float().contiguous(),
        bias=linear.bias.detach().view(()).float().contiguous(),
        fit_summary={
            "objective": "diagnostic_safe_vs_harm_linear_probe",
            "steps": int(steps),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
            "positive": positives,
            "negative": negatives,
            "initial_loss": initial_loss,
            "final_loss": float(final_loss.item()),
        },
    )
