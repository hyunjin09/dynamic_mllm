"""Cached-feature router training helpers for Phase 5B."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from dvr_qwen.losses import compute_route_loss
from dvr_qwen.router_data import NUM_LAYERS
from dvr_qwen.routing import BinaryVisualOnRouter


def freeze_qwen_for_router_training(
    qwen_model: torch.nn.Module,
    router: torch.nn.Module,
) -> dict[str, Any]:
    """Freeze Qwen/base-model parameters while keeping the router trainable."""
    for param in qwen_model.parameters():
        param.requires_grad_(False)
    for param in router.parameters():
        param.requires_grad_(True)
    return assert_only_router_trainable(qwen_model, router)


def assert_only_router_trainable(
    qwen_model: torch.nn.Module,
    router: torch.nn.Module,
) -> dict[str, Any]:
    qwen_trainable = int(sum(param.numel() for param in qwen_model.parameters() if param.requires_grad))
    router_trainable = int(sum(param.numel() for param in router.parameters() if param.requires_grad))
    if qwen_trainable != 0:
        raise AssertionError(f"expected 0 trainable Qwen parameters, found {qwen_trainable}")
    if router_trainable <= 0:
        raise AssertionError("router has no trainable parameters")
    return {
        "qwen_trainable_parameters": qwen_trainable,
        "router_trainable_parameters": router_trainable,
    }


def apply_prev_gate_mode_to_batch(batch: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return a shallow batch copy with the requested previous-gate input mode."""
    if "prev_gates" not in batch:
        raise ValueError("batch is missing prev_gates")
    if mode == "teacher":
        return dict(batch)
    if mode in {"zero", "drop"}:
        updated = dict(batch)
        updated["prev_gates"] = torch.zeros_like(batch["prev_gates"])
        return updated
    raise ValueError(f"unknown prev_gate_mode {mode!r}; expected 'teacher', 'zero', or 'drop'")


def attach_normalized_visual_count_features(
    batch: dict[str, Any],
    *,
    max_visual_tokens: float | None = None,
) -> tuple[dict[str, Any], float]:
    """Attach log-normalized visual-token count as a one-dimensional scalar feature."""
    if "num_visual_tokens" not in batch:
        raise ValueError("batch is missing num_visual_tokens")
    visual_tokens = torch.as_tensor(batch["num_visual_tokens"], dtype=torch.float32)
    if visual_tokens.ndim != 1:
        raise ValueError(f"num_visual_tokens must have shape [B], got {tuple(visual_tokens.shape)}")
    observed_max = float(visual_tokens.max().item())
    if max_visual_tokens is None:
        max_visual_tokens = observed_max
    max_visual_tokens = float(max_visual_tokens)
    if max_visual_tokens <= 0.0:
        raise ValueError("max_visual_tokens must be positive")
    scalar = torch.log1p(visual_tokens) / math.log1p(max_visual_tokens)
    updated = dict(batch)
    updated["scalar_features"] = scalar.unsqueeze(-1)
    return updated, max_visual_tokens


def count_label_mix(labels: torch.Tensor) -> dict[str, Any]:
    if labels.ndim != 2 or labels.shape[1] != NUM_LAYERS:
        raise ValueError(f"labels must have shape [B, {NUM_LAYERS}], got {tuple(labels.shape)}")
    labels = labels.float()
    per_sample = labels.sum(dim=1)
    return {
        "num_samples": int(labels.shape[0]),
        "num_zero_label_samples": int((per_sample == 0).sum().item()),
        "num_positive_label_samples": int((per_sample > 0).sum().item()),
        "min_visual_on_layers": float(per_sample.min().item()),
        "max_visual_on_layers": float(per_sample.max().item()),
        "avg_visual_on_layers": float(per_sample.mean().item()),
    }


def _router_logits(router: BinaryVisualOnRouter, batch: dict[str, Any]) -> torch.Tensor:
    return router(
        batch["global_mean"].float(),
        batch["window_mean"].float(),
        batch["last_token"].float(),
        batch["layer_idx"],
        batch["prev_gates"],
        scalar_features=batch.get("scalar_features"),
        visual_summaries=batch.get("visual_summaries"),
    )


def _slice_batch_to_device(
    batch: dict[str, Any],
    start: int,
    end: int,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            current = value[start:end] if value.ndim > 0 and int(value.shape[0]) == batch_size else value
            sliced[key] = current.to(device)
        else:
            sliced[key] = value
    return sliced


def _router_logits_in_chunks(
    router: BinaryVisualOnRouter,
    batch: dict[str, Any],
    *,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    num_samples = int(batch["global_mean"].shape[0])
    outputs = []
    was_training = router.training
    router.eval()
    with torch.no_grad():
        for start in range(0, num_samples, chunk_size):
            chunk = _slice_batch_to_device(
                batch,
                start,
                min(start + chunk_size, num_samples),
                batch_size=num_samples,
                device=device,
            )
            outputs.append(_router_logits(router, chunk).detach().cpu())
    router.train(was_training)
    return torch.cat(outputs, dim=0)


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in module.named_parameters()}


def _parameter_l2_delta(module: torch.nn.Module, before: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, param in module.named_parameters():
        total += float((param.detach() - before[name]).float().pow(2).sum().item())
    return total**0.5


def _topk_probability_budget_predictions(probs: torch.Tensor) -> torch.Tensor:
    if probs.ndim != 2:
        raise ValueError(f"probs must have shape [B, L], got {tuple(probs.shape)}")
    batch_size, num_layers = probs.shape
    counts = torch.round(probs.sum(dim=1)).to(dtype=torch.long).clamp(min=0, max=num_layers)
    pred = torch.zeros_like(probs, dtype=torch.bool)
    for row_idx in range(batch_size):
        count = int(counts[row_idx].item())
        if count <= 0:
            continue
        top_indices = torch.topk(probs[row_idx], k=count, largest=True).indices
        pred[row_idx, top_indices] = True
    return pred


def _normalize_budget_counts(
    budget_counts: torch.Tensor,
    *,
    batch_size: int,
    num_layers: int,
    device: torch.device,
) -> torch.Tensor:
    counts = torch.as_tensor(budget_counts, device=device)
    if counts.ndim != 1 or int(counts.shape[0]) != batch_size:
        raise ValueError(f"budget_counts must have shape [{batch_size}], got {tuple(counts.shape)}")
    if torch.is_floating_point(counts):
        counts = torch.round(counts)
    return counts.to(dtype=torch.long).clamp(min=0, max=num_layers)


def _topk_rank_positions(logits: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(logits, dim=1, descending=True)
    rank_positions = torch.empty_like(order)
    positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0).expand_as(order)
    rank_positions.scatter_(1, order, positions)
    return rank_positions


def _budget_topk_predictions_from_ranks(rank_positions: torch.Tensor, budget_counts: torch.Tensor) -> torch.Tensor:
    if rank_positions.ndim != 2:
        raise ValueError(f"rank_positions must have shape [B, L], got {tuple(rank_positions.shape)}")
    counts = _normalize_budget_counts(
        budget_counts,
        batch_size=int(rank_positions.shape[0]),
        num_layers=int(rank_positions.shape[1]),
        device=rank_positions.device,
    )
    return rank_positions < counts.unsqueeze(1)


def budget_topk_predictions_from_logits(logits: torch.Tensor, budget_counts: torch.Tensor) -> torch.Tensor:
    """Select the top logit layers per sample using externally predicted budgets."""
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, L], got {tuple(logits.shape)}")
    return _budget_topk_predictions_from_ranks(_topk_rank_positions(logits), budget_counts)


def _repair_pairwise_diffs(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, L], got {tuple(logits.shape)}")
    positive = torch.as_tensor(positive_mask, device=logits.device, dtype=torch.bool)
    negative = torch.as_tensor(negative_mask, device=logits.device, dtype=torch.bool)
    if positive.shape != logits.shape or negative.shape != logits.shape:
        raise ValueError(
            "repair positive/negative masks must match logits shape; "
            f"got logits={tuple(logits.shape)}, positive={tuple(positive.shape)}, negative={tuple(negative.shape)}"
        )
    valid = positive.unsqueeze(2) & negative.unsqueeze(1)
    diffs = logits.float().unsqueeze(2) - logits.float().unsqueeze(1)
    return diffs[valid]


def repair_pairwise_margin_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Rank row-specific repair-positive layers above repair-negative layers."""
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    diffs = _repair_pairwise_diffs(logits, positive_mask, negative_mask)
    if diffs.numel() == 0:
        return logits.float().sum() * 0.0
    return torch.relu(float(margin) - diffs).mean()


def repair_pairwise_margin_metrics(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    margin: float = 1.0,
) -> dict[str, Any]:
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    diffs = _repair_pairwise_diffs(logits, positive_mask, negative_mask)
    if diffs.numel() == 0:
        return {
            "pair_count": 0,
            "strictly_ordered_pairs": 0,
            "margin_satisfied_pairs": 0,
            "strict_order_rate": 0.0,
            "margin_satisfied_rate": 0.0,
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "repair_margin_loss": 0.0,
        }
    pair_count = int(diffs.numel())
    strictly_ordered = int((diffs > 0).sum().item())
    margin_satisfied = int((diffs >= float(margin)).sum().item())
    return {
        "pair_count": pair_count,
        "strictly_ordered_pairs": strictly_ordered,
        "margin_satisfied_pairs": margin_satisfied,
        "strict_order_rate": strictly_ordered / pair_count,
        "margin_satisfied_rate": margin_satisfied / pair_count,
        "mean_margin": float(diffs.mean().item()),
        "min_margin": float(diffs.min().item()),
        "repair_margin_loss": float(torch.relu(float(margin) - diffs).mean().item()),
    }


def _normalize_route_repair_inputs(
    logits: torch.Tensor,
    positive_routes: torch.Tensor,
    negative_routes: torch.Tensor,
    positive_route_mask: torch.Tensor,
    negative_route_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, L], got {tuple(logits.shape)}")
    positive = torch.as_tensor(positive_routes, device=logits.device, dtype=torch.bool)
    negative = torch.as_tensor(negative_routes, device=logits.device, dtype=torch.bool)
    positive_mask = torch.as_tensor(positive_route_mask, device=logits.device, dtype=torch.bool)
    negative_mask = torch.as_tensor(negative_route_mask, device=logits.device, dtype=torch.bool)
    batch_size, num_layers = int(logits.shape[0]), int(logits.shape[1])
    if positive.ndim != 3 or int(positive.shape[0]) != batch_size or int(positive.shape[2]) != num_layers:
        raise ValueError(
            "positive_routes must have shape [B, P, L] matching logits; "
            f"got logits={tuple(logits.shape)}, positive_routes={tuple(positive.shape)}"
        )
    if negative.ndim != 3 or int(negative.shape[0]) != batch_size or int(negative.shape[2]) != num_layers:
        raise ValueError(
            "negative_routes must have shape [B, N, L] matching logits; "
            f"got logits={tuple(logits.shape)}, negative_routes={tuple(negative.shape)}"
        )
    if positive_mask.shape != positive.shape[:2]:
        raise ValueError(
            f"positive_route_mask must have shape {tuple(positive.shape[:2])}, got {tuple(positive_mask.shape)}"
        )
    if negative_mask.shape != negative.shape[:2]:
        raise ValueError(
            f"negative_route_mask must have shape {tuple(negative.shape[:2])}, got {tuple(negative_mask.shape)}"
        )
    return positive, negative, positive_mask, negative_mask


def _route_log_probability_scores(logits: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
    log_on = F.logsigmoid(logits.float()).unsqueeze(1)
    log_off = F.logsigmoid(-logits.float()).unsqueeze(1)
    return torch.where(routes.bool(), log_on, log_off).sum(dim=-1)


def _route_repair_contrastive_diffs(
    logits: torch.Tensor,
    positive_routes: torch.Tensor,
    negative_routes: torch.Tensor,
    positive_route_mask: torch.Tensor,
    negative_route_mask: torch.Tensor,
) -> torch.Tensor:
    positive, negative, positive_mask, negative_mask = _normalize_route_repair_inputs(
        logits,
        positive_routes,
        negative_routes,
        positive_route_mask,
        negative_route_mask,
    )
    valid = positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)
    if valid.numel() == 0:
        return logits.float().new_zeros(0)
    positive_scores = _route_log_probability_scores(logits, positive)
    negative_scores = _route_log_probability_scores(logits, negative)
    diffs = positive_scores.unsqueeze(2) - negative_scores.unsqueeze(1)
    return diffs[valid]


def route_repair_contrastive_loss(
    logits: torch.Tensor,
    positive_routes: torch.Tensor,
    negative_routes: torch.Tensor,
    positive_route_mask: torch.Tensor,
    negative_route_mask: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Rank complete fixing routes above failed/non-fixing routes for the same row."""
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    diffs = _route_repair_contrastive_diffs(
        logits,
        positive_routes,
        negative_routes,
        positive_route_mask,
        negative_route_mask,
    )
    if diffs.numel() == 0:
        return logits.float().sum() * 0.0
    return torch.relu(float(margin) - diffs).mean()


def route_repair_contrastive_metrics(
    logits: torch.Tensor,
    positive_routes: torch.Tensor,
    negative_routes: torch.Tensor,
    positive_route_mask: torch.Tensor,
    negative_route_mask: torch.Tensor,
    *,
    margin: float = 1.0,
) -> dict[str, Any]:
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    positive, negative, positive_mask, negative_mask = _normalize_route_repair_inputs(
        logits,
        positive_routes,
        negative_routes,
        positive_route_mask,
        negative_route_mask,
    )
    diffs = _route_repair_contrastive_diffs(
        logits,
        positive,
        negative,
        positive_mask,
        negative_mask,
    )
    if diffs.numel() == 0:
        return {
            "pair_count": 0,
            "strictly_ordered_pairs": 0,
            "margin_satisfied_pairs": 0,
            "strict_order_rate": 0.0,
            "margin_satisfied_rate": 0.0,
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "route_repair_contrastive_loss": 0.0,
        }
    pair_count = int(diffs.numel())
    strictly_ordered = int((diffs > 0).sum().item())
    margin_satisfied = int((diffs >= float(margin)).sum().item())
    return {
        "pair_count": pair_count,
        "strictly_ordered_pairs": strictly_ordered,
        "margin_satisfied_pairs": margin_satisfied,
        "strict_order_rate": strictly_ordered / pair_count,
        "margin_satisfied_rate": margin_satisfied / pair_count,
        "mean_margin": float(diffs.mean().item()),
        "min_margin": float(diffs.min().item()),
        "route_repair_contrastive_loss": float(torch.relu(float(margin) - diffs).mean().item()),
    }


def _repair_mask_summary(
    positive_mask: torch.Tensor | None,
    negative_mask: torch.Tensor | None,
) -> dict[str, Any]:
    if positive_mask is None or negative_mask is None:
        return {
            "num_rows_with_constraints": 0,
            "num_positive_layers": 0,
            "num_negative_layers": 0,
            "pair_count": 0,
        }
    positive = positive_mask.bool()
    negative = negative_mask.bool()
    if positive.shape != negative.shape:
        raise ValueError(
            f"repair positive/negative masks must have the same shape, got {positive.shape} and {negative.shape}"
        )
    per_row_pairs = positive.long().sum(dim=1) * negative.long().sum(dim=1)
    return {
        "num_rows_with_constraints": int(((positive | negative).any(dim=1)).sum().item()),
        "num_rows_with_pairs": int((per_row_pairs > 0).sum().item()),
        "num_positive_layers": int(positive.sum().item()),
        "num_negative_layers": int(negative.sum().item()),
        "pair_count": int(per_row_pairs.sum().item()),
    }


def _route_repair_summary(
    positive_routes: torch.Tensor | None,
    negative_routes: torch.Tensor | None,
    positive_route_mask: torch.Tensor | None,
    negative_route_mask: torch.Tensor | None,
) -> dict[str, Any]:
    if (
        positive_routes is None
        or negative_routes is None
        or positive_route_mask is None
        or negative_route_mask is None
    ):
        return {
            "num_rows_with_constraints": 0,
            "num_rows_with_pairs": 0,
            "num_positive_routes": 0,
            "num_negative_routes": 0,
            "pair_count": 0,
        }
    positive = torch.as_tensor(positive_routes, dtype=torch.bool)
    negative = torch.as_tensor(negative_routes, dtype=torch.bool)
    positive_mask = torch.as_tensor(positive_route_mask, dtype=torch.bool)
    negative_mask = torch.as_tensor(negative_route_mask, dtype=torch.bool)
    if positive.ndim != 3 or negative.ndim != 3:
        raise ValueError("route repair positive/negative routes must have shape [B, R, L]")
    if positive.shape[0] != negative.shape[0] or positive.shape[2] != negative.shape[2]:
        raise ValueError(
            f"route repair positive/negative routes are incompatible: {positive.shape} vs {negative.shape}"
        )
    if positive_mask.shape != positive.shape[:2]:
        raise ValueError(
            f"route repair positive mask must have shape {tuple(positive.shape[:2])}, "
            f"got {tuple(positive_mask.shape)}"
        )
    if negative_mask.shape != negative.shape[:2]:
        raise ValueError(
            f"route repair negative mask must have shape {tuple(negative.shape[:2])}, "
            f"got {tuple(negative_mask.shape)}"
        )
    per_row_pairs = positive_mask.long().sum(dim=1) * negative_mask.long().sum(dim=1)
    return {
        "num_rows_with_constraints": int((positive_mask.any(dim=1) | negative_mask.any(dim=1)).sum().item()),
        "num_rows_with_pairs": int((per_row_pairs > 0).sum().item()),
        "num_positive_routes": int(positive_mask.sum().item()),
        "num_negative_routes": int(negative_mask.sum().item()),
        "pair_count": int(per_row_pairs.sum().item()),
    }


def normalize_forced_visual_on_layers(forced_layers: list[int] | tuple[int, ...], *, num_layers: int) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    normalized = sorted({int(layer_idx) for layer_idx in forced_layers})
    for layer_idx in normalized:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"forced layer {layer_idx} is out of range for {num_layers} layers")
    return normalized


def force_visual_on_layers(predictions: torch.Tensor, forced_layers: list[int] | tuple[int, ...]) -> torch.Tensor:
    """Return a copy of binary route predictions with selected layers forced VISUAL_ON."""
    if predictions.ndim != 2:
        raise ValueError(f"predictions must have shape [B, L], got {tuple(predictions.shape)}")
    forced = normalize_forced_visual_on_layers(forced_layers, num_layers=int(predictions.shape[1]))
    output = predictions.to(dtype=torch.bool).clone()
    if forced:
        output[:, forced] = True
    return output


def anchored_budget_topk_predictions_from_logits(
    logits: torch.Tensor,
    budget_counts: torch.Tensor,
    forced_layers: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Select budgeted top-K layers while always including requested anchor layers."""
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, L], got {tuple(logits.shape)}")
    batch_size, num_layers = logits.shape
    forced = normalize_forced_visual_on_layers(forced_layers, num_layers=int(num_layers))
    counts = _normalize_budget_counts(
        budget_counts,
        batch_size=int(batch_size),
        num_layers=int(num_layers),
        device=logits.device,
    )
    pred = torch.zeros_like(logits, dtype=torch.bool)
    if forced:
        pred[:, forced] = True
    masked_logits = logits.float().clone()
    if forced:
        masked_logits[:, forced] = float("-inf")
    available = int(num_layers) - len(forced)
    for row_idx in range(int(batch_size)):
        remaining = int(counts[row_idx].item()) - len(forced)
        if remaining <= 0 or available <= 0:
            continue
        top_indices = torch.topk(masked_logits[row_idx], k=min(remaining, available), largest=True).indices
        pred[row_idx, top_indices] = True
    return pred


def _route_metrics_from_predictions(
    pred_bool: torch.Tensor,
    targets_bool: torch.Tensor,
    probs: torch.Tensor,
    *,
    route_policy: str,
    threshold: float | None,
    budget_counts: torch.Tensor | None = None,
) -> dict[str, Any]:
    if pred_bool.shape != targets_bool.shape:
        raise ValueError(
            f"predictions and targets must have the same shape, got {pred_bool.shape} and {targets_bool.shape}"
        )
    tp = int((pred_bool & targets_bool).sum().item())
    fp = int((pred_bool & ~targets_bool).sum().item())
    fn = int((~pred_bool & targets_bool).sum().item())
    tn = int((~pred_bool & ~targets_bool).sum().item())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    target_count = targets_bool.float().sum(dim=1)
    pred_count = pred_bool.float().sum(dim=1)
    metrics = {
        "route_policy": route_policy,
        "threshold": None if route_policy != "threshold" or threshold is None else float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "route_precision": precision,
        "route_recall": recall,
        "route_f1": f1,
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "route_layer_accuracy": float((pred_bool == targets_bool).float().mean().item()),
        "route_exact_match_accuracy": float((pred_bool == targets_bool).all(dim=1).float().mean().item()),
        "budget_mae": float((pred_count - target_count).abs().mean().item()),
        "target_avg_visual_on_layers": float(target_count.mean().item()),
        "predicted_avg_visual_on_layers": float(pred_count.mean().item()),
        "prob_avg_visual_on_layers": float(probs.sum(dim=1).mean().item()),
        "per_layer_visual_on_rate": [float(value) for value in pred_bool.float().mean(dim=0).cpu().tolist()],
        "per_layer_target_visual_on_rate": [float(value) for value in targets_bool.float().mean(dim=0).cpu().tolist()],
    }
    if budget_counts is not None:
        counts = _normalize_budget_counts(
            budget_counts,
            batch_size=int(pred_bool.shape[0]),
            num_layers=int(pred_bool.shape[1]),
            device=pred_bool.device,
        )
        metrics["budget_counts_mean"] = float(counts.float().mean().item())
        metrics["budget_counts_min"] = int(counts.min().item())
        metrics["budget_counts_max"] = int(counts.max().item())
    return metrics


def route_predictions_from_logits(
    logits: torch.Tensor,
    *,
    threshold: float | None = 0.5,
    route_policy: str = "threshold",
    budget_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, L], got {tuple(logits.shape)}")
    probs = torch.sigmoid(logits)
    if route_policy == "threshold":
        if threshold is None:
            raise ValueError("threshold route_policy requires a threshold")
        return probs >= float(threshold)
    if route_policy == "topk_prob_budget":
        return _topk_probability_budget_predictions(probs)
    if route_policy == "budget_topk":
        if budget_counts is None:
            raise ValueError("budget_topk route_policy requires budget_counts")
        return budget_topk_predictions_from_logits(logits, budget_counts)
    raise ValueError(f"unknown route_policy {route_policy!r}")


def compute_route_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float | None = 0.5,
    route_policy: str = "threshold",
    budget_counts: torch.Tensor | None = None,
) -> dict[str, Any]:
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have the same shape, got {logits.shape} and {targets.shape}")
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, L], got {tuple(logits.shape)}")
    targets_bool = targets.to(device=logits.device).float() >= 0.5
    probs = torch.sigmoid(logits)
    pred_bool = route_predictions_from_logits(
        logits,
        threshold=threshold,
        route_policy=route_policy,
        budget_counts=budget_counts,
    )
    return _route_metrics_from_predictions(
        pred_bool,
        targets_bool,
        probs,
        route_policy=route_policy,
        threshold=threshold,
        budget_counts=budget_counts,
    )


def build_budget_policy_features(
    logits: torch.Tensor,
    scalar_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build cheap per-sample features for predicting the number of VISUAL_ON layers."""
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, L], got {tuple(logits.shape)}")
    logits = logits.float()
    probs = torch.sigmoid(logits)
    parts = [
        probs.sum(dim=1, keepdim=True),
        probs.mean(dim=1, keepdim=True),
        probs.max(dim=1, keepdim=True).values,
        probs.std(dim=1, unbiased=False, keepdim=True),
        logits.mean(dim=1, keepdim=True),
        logits.max(dim=1, keepdim=True).values,
        logits.std(dim=1, unbiased=False, keepdim=True),
    ]
    if scalar_features is not None:
        scalar = torch.as_tensor(scalar_features, dtype=torch.float32, device=logits.device)
        if scalar.ndim == 1:
            scalar = scalar.unsqueeze(-1)
        if scalar.ndim != 2 or int(scalar.shape[0]) != int(logits.shape[0]):
            raise ValueError(
                f"scalar_features must have shape [{logits.shape[0]}, S], got {tuple(scalar.shape)}"
            )
        parts.append(scalar)
    return torch.cat(parts, dim=1)


def fit_budget_ridge_predictor(
    features: torch.Tensor,
    target_counts: torch.Tensor,
    *,
    l2: float = 1e-3,
) -> dict[str, Any]:
    """Fit a tiny linear budget head with standardized features and ridge penalty."""
    if features.ndim != 2:
        raise ValueError(f"features must have shape [B, F], got {tuple(features.shape)}")
    counts = torch.as_tensor(target_counts, dtype=torch.float32, device=features.device)
    if counts.ndim != 1 or int(counts.shape[0]) != int(features.shape[0]):
        raise ValueError(f"target_counts must have shape [{features.shape[0]}], got {tuple(counts.shape)}")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    features = features.float()
    feature_mean = features.mean(dim=0)
    feature_scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardized = (features - feature_mean) / feature_scale
    design = torch.cat([torch.ones(features.shape[0], 1, device=features.device), standardized], dim=1)
    regularizer = torch.eye(design.shape[1], dtype=design.dtype, device=design.device) * float(l2)
    regularizer[0, 0] = 0.0
    xtx = design.T @ design + regularizer
    xty = design.T @ counts
    try:
        weights = torch.linalg.solve(xtx, xty)
    except RuntimeError:
        weights = torch.linalg.pinv(xtx) @ xty
    return {
        "weights": weights.detach(),
        "feature_mean": feature_mean.detach(),
        "feature_scale": feature_scale.detach(),
        "l2": float(l2),
    }


def predict_budget_counts(
    features: torch.Tensor,
    predictor: dict[str, Any],
    *,
    num_layers: int,
    budget_bias: float = 0.0,
    min_count: int = 0,
    max_count: int | None = None,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"features must have shape [B, F], got {tuple(features.shape)}")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    upper = num_layers if max_count is None else min(int(max_count), num_layers)
    lower = max(0, int(min_count))
    if lower > upper:
        raise ValueError(f"min_count {lower} cannot exceed max_count {upper}")
    weights = predictor["weights"].to(device=features.device, dtype=torch.float32)
    feature_mean = predictor["feature_mean"].to(device=features.device, dtype=torch.float32)
    feature_scale = predictor["feature_scale"].to(device=features.device, dtype=torch.float32)
    if int(weights.shape[0]) != int(features.shape[1]) + 1:
        raise ValueError(
            f"predictor has {weights.shape[0]} weights for {features.shape[1]} features"
        )
    standardized = (features.float() - feature_mean) / feature_scale
    design = torch.cat([torch.ones(features.shape[0], 1, device=features.device), standardized], dim=1)
    raw = design @ weights + float(budget_bias)
    return torch.round(raw).to(dtype=torch.long).clamp(min=lower, max=upper)


def _default_budget_biases() -> list[float]:
    return [idx / 4.0 for idx in range(-16, 17)]


def _default_min_counts() -> list[int]:
    return [0, 1, 2, 3, 4]


def _default_max_counts(num_layers: int) -> list[int]:
    candidates = {num_layers}
    for fraction in [0.35, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85]:
        candidates.add(max(1, min(num_layers, int(round(num_layers * fraction)))))
    return sorted(candidates)


def fit_budget_topk_policy(
    train_logits: torch.Tensor,
    train_targets: torch.Tensor,
    val_logits: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    train_scalar_features: torch.Tensor | None = None,
    val_scalar_features: torch.Tensor | None = None,
    ridge_lambdas: list[float] | None = None,
    budget_biases: list[float] | None = None,
    min_counts: list[int] | None = None,
    max_counts: list[int] | None = None,
    max_extra_avg_visual_on_layers: float = 1.0,
    top_k: int = 10,
) -> dict[str, Any]:
    if train_logits.shape != train_targets.shape:
        raise ValueError(
            f"train logits and targets must have the same shape, got {train_logits.shape} and {train_targets.shape}"
        )
    if val_logits.shape != val_targets.shape:
        raise ValueError(
            f"validation logits and targets must have the same shape, got {val_logits.shape} and {val_targets.shape}"
        )
    if train_logits.ndim != 2 or val_logits.ndim != 2:
        raise ValueError("train and validation logits must have shape [B, L]")
    if int(train_logits.shape[1]) != int(val_logits.shape[1]):
        raise ValueError("train and validation logits must have the same number of layers")
    if max_extra_avg_visual_on_layers < 0.0:
        raise ValueError("max_extra_avg_visual_on_layers must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if (train_scalar_features is None) != (val_scalar_features is None):
        raise ValueError("train and validation scalar features must both be provided or both be omitted")

    train_logits = train_logits.float()
    val_logits = val_logits.float()
    train_targets = train_targets.float()
    val_targets = val_targets.float()
    num_layers = int(train_logits.shape[1])
    if ridge_lambdas is None:
        ridge_lambdas = [1e-6, 1e-4, 1e-2, 1.0]
    if budget_biases is None:
        budget_biases = _default_budget_biases()
    if min_counts is None:
        min_counts = _default_min_counts()
    if max_counts is None:
        max_counts = _default_max_counts(num_layers)

    train_features = build_budget_policy_features(train_logits, train_scalar_features)
    val_features = build_budget_policy_features(val_logits, val_scalar_features)
    train_target_counts = train_targets.sum(dim=1)
    train_targets_bool = train_targets.to(device=train_logits.device) >= 0.5
    val_targets_bool = val_targets.to(device=val_logits.device) >= 0.5
    train_probs = torch.sigmoid(train_logits)
    val_probs = torch.sigmoid(val_logits)
    train_rank_positions = _topk_rank_positions(train_logits)
    val_rank_positions = _topk_rank_positions(val_logits)
    train_default = compute_route_metrics(train_logits, train_targets, threshold=0.5)
    train_prob_topk = compute_route_metrics(train_logits, train_targets, threshold=None, route_policy="topk_prob_budget")
    val_default = compute_route_metrics(val_logits, val_targets, threshold=0.5)
    val_prob_topk = compute_route_metrics(val_logits, val_targets, threshold=None, route_policy="topk_prob_budget")
    target_avg = train_default["target_avg_visual_on_layers"]
    max_predicted_avg = target_avg + float(max_extra_avg_visual_on_layers)

    candidates = []
    predictors: dict[float, dict[str, Any]] = {}
    for ridge_lambda in ridge_lambdas:
        predictor = fit_budget_ridge_predictor(train_features, train_target_counts, l2=float(ridge_lambda))
        predictors[float(ridge_lambda)] = predictor
        for budget_bias in budget_biases:
            for min_count in min_counts:
                for max_count in max_counts:
                    if int(min_count) > int(max_count):
                        continue
                    counts = predict_budget_counts(
                        train_features,
                        predictor,
                        num_layers=num_layers,
                        budget_bias=float(budget_bias),
                        min_count=int(min_count),
                        max_count=int(max_count),
                    )
                    pred_bool = _budget_topk_predictions_from_ranks(train_rank_positions, counts)
                    metrics = _route_metrics_from_predictions(
                        pred_bool,
                        train_targets_bool,
                        train_probs,
                        route_policy="budget_topk",
                        threshold=None,
                        budget_counts=counts,
                    )
                    metrics.update(
                        {
                            "policy_name": (
                                f"budget_topk_l2{float(ridge_lambda):g}_b{float(budget_bias):g}_"
                                f"min{int(min_count)}_max{int(max_count)}"
                            ),
                            "ridge_lambda": float(ridge_lambda),
                            "budget_bias": float(budget_bias),
                            "min_count": int(min_count),
                            "max_count": int(max_count),
                            "within_avg_visual_on_budget": (
                                metrics["predicted_avg_visual_on_layers"] <= max_predicted_avg
                            ),
                            "improves_false_negative_rate_vs_threshold": (
                                metrics["false_negative_rate"] < train_default["false_negative_rate"]
                            ),
                            "improves_false_negative_rate_vs_prob_topk": (
                                metrics["false_negative_rate"] < train_prob_topk["false_negative_rate"]
                            ),
                        }
                    )
                    candidates.append(metrics)

    viable = [
        item
        for item in candidates
        if item["within_avg_visual_on_budget"] and item["improves_false_negative_rate_vs_threshold"]
    ]
    within_budget = [item for item in candidates if item["within_avg_visual_on_budget"]]
    ranked = viable or within_budget or candidates
    selected_candidate = min(
        ranked,
        key=lambda item: (
            item["false_negative_rate"],
            item["budget_mae"],
            -item["route_f1"],
            abs(item["predicted_avg_visual_on_layers"] - target_avg),
            item["ridge_lambda"],
            abs(item["budget_bias"]),
            item["min_count"],
            item["max_count"],
        ),
    )
    selected_predictor = predictors[selected_candidate["ridge_lambda"]]
    selected_train_counts = predict_budget_counts(
        train_features,
        selected_predictor,
        num_layers=num_layers,
        budget_bias=selected_candidate["budget_bias"],
        min_count=selected_candidate["min_count"],
                        max_count=selected_candidate["max_count"],
    )
    selected_val_counts = predict_budget_counts(
        val_features,
        selected_predictor,
        num_layers=num_layers,
        budget_bias=selected_candidate["budget_bias"],
        min_count=selected_candidate["min_count"],
        max_count=selected_candidate["max_count"],
    )
    selected_train_pred = _budget_topk_predictions_from_ranks(train_rank_positions, selected_train_counts)
    selected_val_pred = _budget_topk_predictions_from_ranks(val_rank_positions, selected_val_counts)
    selected_train = {
        **_route_metrics_from_predictions(
            selected_train_pred,
            train_targets_bool,
            train_probs,
            route_policy="budget_topk",
            threshold=None,
            budget_counts=selected_train_counts,
        ),
        "policy_name": selected_candidate["policy_name"],
        "ridge_lambda": selected_candidate["ridge_lambda"],
        "budget_bias": selected_candidate["budget_bias"],
        "min_count": selected_candidate["min_count"],
        "max_count": selected_candidate["max_count"],
    }
    selected_val = {
        **_route_metrics_from_predictions(
            selected_val_pred,
            val_targets_bool,
            val_probs,
            route_policy="budget_topk",
            threshold=None,
            budget_counts=selected_val_counts,
        ),
        "policy_name": selected_candidate["policy_name"],
        "ridge_lambda": selected_candidate["ridge_lambda"],
        "budget_bias": selected_candidate["budget_bias"],
        "min_count": selected_candidate["min_count"],
        "max_count": selected_candidate["max_count"],
    }
    top_candidates = sorted(
        candidates,
        key=lambda item: (
            not item["within_avg_visual_on_budget"],
            item["false_negative_rate"],
            item["budget_mae"],
            -item["route_f1"],
            abs(item["predicted_avg_visual_on_layers"] - target_avg),
            item["ridge_lambda"],
            abs(item["budget_bias"]),
            item["min_count"],
            item["max_count"],
        ),
    )[:top_k]
    return {
        "verification_status": "passed",
        "route_policy": "budget_topk",
        "max_extra_avg_visual_on_layers": float(max_extra_avg_visual_on_layers),
        "max_predicted_avg_visual_on_layers": max_predicted_avg,
        "num_features": int(train_features.shape[1]),
        "num_ridge_lambdas_swept": len(ridge_lambdas),
        "num_budget_biases_swept": len(budget_biases),
        "num_min_counts_swept": len(min_counts),
        "num_max_counts_swept": len(max_counts),
        "num_candidates": len(candidates),
        "num_viable_candidates": len(viable),
        "train_default_metrics": train_default,
        "train_prob_topk_metrics": train_prob_topk,
        "val_default_metrics": val_default,
        "val_prob_topk_metrics": val_prob_topk,
        "selected_train": selected_train,
        "selected_val_metrics": selected_val,
        "top_train_candidates": top_candidates,
        "selected_budget_predictor": {
            "weights": selected_predictor["weights"].detach().cpu(),
            "feature_mean": selected_predictor["feature_mean"].detach().cpu(),
            "feature_scale": selected_predictor["feature_scale"].detach().cpu(),
            "l2": float(selected_predictor["l2"]),
        },
        "selected_train_predicted_counts": selected_train_counts.detach().cpu(),
        "selected_val_predicted_counts": selected_val_counts.detach().cpu(),
        "selection_policy": (
            "Fit a standardized ridge count predictor on train cached logits plus optional scalar "
            "features, sweep global count bias and count clamps on train labels, and route each "
            "sample by top-K layer logits using the predicted budget. Prefer train candidates within "
            "target_avg + max_extra_avg_visual_on_layers that reduce threshold false-negative rate; "
            "rank by false-negative rate, budget MAE, route F1, budget closeness, ridge lambda, "
            "budget-bias magnitude, and count clamps. Validation is evaluated without refitting."
        ),
    }


def compute_layerwise_base_rate_bias(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float = 1e-6,
) -> torch.Tensor:
    """Fit per-layer logit biases that match train-set target counts at threshold 0.5."""
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have the same shape, got {logits.shape} and {targets.shape}")
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, L], got {tuple(logits.shape)}")
    if margin <= 0.0:
        raise ValueError("margin must be positive")

    logits = logits.float()
    targets_bool = targets.to(device=logits.device).float() >= 0.5
    batch_size, num_layers = logits.shape
    layer_bias = []
    for layer_idx in range(num_layers):
        scores = logits[:, layer_idx]
        target_count = int(targets_bool[:, layer_idx].sum().item())
        if target_count <= 0:
            bias = -float(scores.max().item()) - float(margin)
        elif target_count >= batch_size:
            bias = -float(scores.min().item()) + float(margin)
        else:
            sorted_scores = torch.sort(scores, descending=True).values
            threshold = 0.5 * (
                float(sorted_scores[target_count - 1].item()) + float(sorted_scores[target_count].item())
            )
            bias = -threshold
        layer_bias.append(bias)
    return torch.tensor(layer_bias, dtype=logits.dtype, device=logits.device)


def apply_layerwise_logit_calibration(
    logits: torch.Tensor,
    layer_bias: torch.Tensor,
    *,
    layer_bias_scale: float = 1.0,
    global_bias: float = 0.0,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, L], got {tuple(logits.shape)}")
    layer_bias = torch.as_tensor(layer_bias, dtype=logits.dtype, device=logits.device)
    if layer_bias.ndim != 1 or int(layer_bias.shape[0]) != int(logits.shape[1]):
        raise ValueError(f"layer_bias must have shape [{logits.shape[1]}], got {tuple(layer_bias.shape)}")
    return logits + float(layer_bias_scale) * layer_bias.unsqueeze(0) + float(global_bias)


def _half_threshold_selection_metrics(logits: torch.Tensor, targets_bool: torch.Tensor) -> dict[str, Any]:
    pred_bool = logits >= 0.0
    tp = int((pred_bool & targets_bool).sum().item())
    fp = int((pred_bool & ~targets_bool).sum().item())
    fn = int((~pred_bool & targets_bool).sum().item())
    tn = int((~pred_bool & ~targets_bool).sum().item())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    target_count = targets_bool.float().sum(dim=1)
    pred_count = pred_bool.float().sum(dim=1)
    return {
        "route_policy": "threshold",
        "threshold": 0.5,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "route_precision": precision,
        "route_recall": recall,
        "route_f1": f1,
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "route_layer_accuracy": float((pred_bool == targets_bool).float().mean().item()),
        "route_exact_match_accuracy": float((pred_bool == targets_bool).all(dim=1).float().mean().item()),
        "budget_mae": float((pred_count - target_count).abs().mean().item()),
        "target_avg_visual_on_layers": float(target_count.mean().item()),
        "predicted_avg_visual_on_layers": float(pred_count.mean().item()),
    }


def _default_layer_bias_scales() -> list[float]:
    return [idx / 20.0 for idx in range(0, 41)]


def _default_global_biases() -> list[float]:
    return [idx / 20.0 for idx in range(-80, 81)]


def calibrate_layerwise_bias_policy(
    train_logits: torch.Tensor,
    train_targets: torch.Tensor,
    val_logits: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    layer_bias_scales: list[float] | None = None,
    global_biases: list[float] | None = None,
    max_extra_avg_visual_on_layers: float = 1.0,
    top_k: int = 10,
) -> dict[str, Any]:
    if train_logits.shape != train_targets.shape:
        raise ValueError(
            f"train logits and targets must have the same shape, got {train_logits.shape} and {train_targets.shape}"
        )
    if val_logits.shape != val_targets.shape:
        raise ValueError(
            f"validation logits and targets must have the same shape, got {val_logits.shape} and {val_targets.shape}"
        )
    if train_logits.ndim != 2 or val_logits.ndim != 2:
        raise ValueError("train and validation logits must have shape [B, L]")
    if int(train_logits.shape[1]) != int(val_logits.shape[1]):
        raise ValueError("train and validation logits must have the same number of layers")
    if max_extra_avg_visual_on_layers < 0.0:
        raise ValueError("max_extra_avg_visual_on_layers must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    train_logits = train_logits.float()
    val_logits = val_logits.float()
    train_targets = train_targets.float()
    val_targets = val_targets.float()
    train_targets_bool = train_targets.to(device=train_logits.device) >= 0.5
    if layer_bias_scales is None:
        layer_bias_scales = _default_layer_bias_scales()
    if global_biases is None:
        global_biases = _default_global_biases()

    train_default = compute_route_metrics(train_logits, train_targets, threshold=0.5)
    val_default = compute_route_metrics(val_logits, val_targets, threshold=0.5)
    layer_bias = compute_layerwise_base_rate_bias(train_logits, train_targets)
    target_avg = train_default["target_avg_visual_on_layers"]
    max_predicted_avg = target_avg + float(max_extra_avg_visual_on_layers)

    candidates = []
    for layer_bias_scale in layer_bias_scales:
        scaled = train_logits + float(layer_bias_scale) * layer_bias.unsqueeze(0)
        for global_bias in global_biases:
            metrics = _half_threshold_selection_metrics(scaled + float(global_bias), train_targets_bool)
            metrics.update(
                {
                    "policy_name": f"layer_bias_s{float(layer_bias_scale):g}_gb{float(global_bias):g}",
                    "layer_bias_scale": float(layer_bias_scale),
                    "global_bias": float(global_bias),
                    "within_avg_visual_on_budget": metrics["predicted_avg_visual_on_layers"] <= max_predicted_avg,
                    "improves_false_negative_rate": (
                        metrics["false_negative_rate"] < train_default["false_negative_rate"]
                    ),
                }
            )
            candidates.append(metrics)

    viable = [
        item
        for item in candidates
        if item["within_avg_visual_on_budget"] and item["improves_false_negative_rate"]
    ]
    within_budget = [item for item in candidates if item["within_avg_visual_on_budget"]]
    ranked = viable or within_budget or candidates
    selected_candidate = min(
        ranked,
        key=lambda item: (
            item["false_negative_rate"],
            item["budget_mae"],
            -item["route_f1"],
            abs(item["predicted_avg_visual_on_layers"] - target_avg),
            item["layer_bias_scale"],
            item["global_bias"],
        ),
    )

    selected_train_logits = apply_layerwise_logit_calibration(
        train_logits,
        layer_bias,
        layer_bias_scale=selected_candidate["layer_bias_scale"],
        global_bias=selected_candidate["global_bias"],
    )
    selected_val_logits = apply_layerwise_logit_calibration(
        val_logits,
        layer_bias.to(device=val_logits.device),
        layer_bias_scale=selected_candidate["layer_bias_scale"],
        global_bias=selected_candidate["global_bias"],
    )
    selected_train = {
        **compute_route_metrics(selected_train_logits, train_targets, threshold=0.5),
        "policy_name": selected_candidate["policy_name"],
        "layer_bias_scale": selected_candidate["layer_bias_scale"],
        "global_bias": selected_candidate["global_bias"],
    }
    selected_val = {
        **compute_route_metrics(selected_val_logits, val_targets, threshold=0.5),
        "policy_name": selected_candidate["policy_name"],
        "layer_bias_scale": selected_candidate["layer_bias_scale"],
        "global_bias": selected_candidate["global_bias"],
    }
    top_candidates = sorted(
        candidates,
        key=lambda item: (
            not item["within_avg_visual_on_budget"],
            item["false_negative_rate"],
            item["budget_mae"],
            -item["route_f1"],
            abs(item["predicted_avg_visual_on_layers"] - target_avg),
            item["layer_bias_scale"],
            item["global_bias"],
        ),
    )[:top_k]
    return {
        "verification_status": "passed",
        "route_policy": "layerwise_bias_threshold",
        "max_extra_avg_visual_on_layers": float(max_extra_avg_visual_on_layers),
        "max_predicted_avg_visual_on_layers": max_predicted_avg,
        "num_layer_bias_scales_swept": len(layer_bias_scales),
        "num_global_biases_swept": len(global_biases),
        "num_candidates": len(candidates),
        "num_viable_candidates": len(viable),
        "base_layer_bias": [float(value) for value in layer_bias.detach().cpu().tolist()],
        "train_default_metrics": train_default,
        "val_default_metrics": val_default,
        "selected_train": selected_train,
        "selected_val_metrics": selected_val,
        "top_train_candidates": top_candidates,
        "selected_val_logits": selected_val_logits.detach().cpu(),
        "selection_policy": (
            "Fit per-layer base-rate biases on train logits, sweep layer-bias scale and global "
            "bias on train logits, prefer candidates within target_avg + max_extra_avg_visual_on_layers "
            "that reduce train false-negative rate, then rank by train false-negative rate, budget MAE, "
            "route F1, budget closeness, layer-bias scale, and global bias. Evaluate the selected "
            "calibration on validation logits without refitting."
        ),
    }


def threshold_to_global_bias(threshold: float) -> float:
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    return -math.log(threshold / (1.0 - threshold))


def _default_calibration_thresholds() -> list[float]:
    return [idx / 1000.0 for idx in range(1, 1000)]


def calibrate_route_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    default_threshold: float = 0.5,
    thresholds: list[float] | None = None,
    max_extra_avg_visual_on_layers: float = 1.0,
    top_k: int = 10,
) -> dict[str, Any]:
    default_metrics = compute_route_metrics(logits, targets, threshold=default_threshold)
    if thresholds is None:
        thresholds = _default_calibration_thresholds()
    if max_extra_avg_visual_on_layers < 0.0:
        raise ValueError("max_extra_avg_visual_on_layers must be non-negative")

    target_avg = default_metrics["target_avg_visual_on_layers"]
    max_predicted_avg = target_avg + float(max_extra_avg_visual_on_layers)
    candidates = []
    for threshold in thresholds:
        metrics = compute_route_metrics(logits, targets, threshold=threshold)
        metrics["global_bias_for_threshold_0_5"] = threshold_to_global_bias(threshold)
        metrics["within_avg_visual_on_budget"] = metrics["predicted_avg_visual_on_layers"] <= max_predicted_avg
        metrics["improves_false_negative_rate"] = (
            metrics["false_negative_rate"] < default_metrics["false_negative_rate"]
        )
        candidates.append(metrics)

    viable = [
        item
        for item in candidates
        if item["within_avg_visual_on_budget"] and item["improves_false_negative_rate"]
    ]
    selected = None
    if viable:
        selected = min(
            viable,
            key=lambda item: (
                item["false_negative_rate"],
                item["budget_mae"],
                -item["route_f1"],
                abs(item["predicted_avg_visual_on_layers"] - target_avg),
                -item["threshold"],
            ),
        )

    best_f1 = max(candidates, key=lambda item: (item["route_f1"], -item["budget_mae"], -item["threshold"]))
    top_candidates = sorted(
        candidates,
        key=lambda item: (
            not item["within_avg_visual_on_budget"],
            item["false_negative_rate"],
            item["budget_mae"],
            -item["route_f1"],
            -item["threshold"],
        ),
    )[:top_k]
    return {
        "default_threshold": float(default_threshold),
        "default_metrics": default_metrics,
        "max_extra_avg_visual_on_layers": float(max_extra_avg_visual_on_layers),
        "max_predicted_avg_visual_on_layers": max_predicted_avg,
        "num_thresholds_swept": len(thresholds),
        "num_viable_thresholds": len(viable),
        "selected": selected,
        "best_f1": best_f1,
        "top_candidates": top_candidates,
        "selection_policy": (
            "Among thresholds that reduce false-negative rate and keep predicted average "
            "VISUAL_ON layers within target_avg + max_extra_avg_visual_on_layers, choose "
            "lowest false-negative rate, then lowest budget MAE, then highest F1, then "
            "closest average budget, then highest threshold."
        ),
    }


def compare_route_policies(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    default_threshold: float = 0.5,
    max_extra_avg_visual_on_layers: float = 1.0,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    default_metrics = compute_route_metrics(logits, targets, threshold=default_threshold, route_policy="threshold")
    calibration = calibrate_route_threshold(
        logits,
        targets,
        default_threshold=default_threshold,
        thresholds=thresholds,
        max_extra_avg_visual_on_layers=max_extra_avg_visual_on_layers,
    )
    target_avg = default_metrics["target_avg_visual_on_layers"]
    max_predicted_avg = target_avg + float(max_extra_avg_visual_on_layers)

    policies: dict[str, dict[str, Any]] = {
        f"threshold_{default_threshold:g}": {
            **default_metrics,
            "policy_name": f"threshold_{default_threshold:g}",
        },
        "topk_prob_budget": {
            **compute_route_metrics(logits, targets, threshold=None, route_policy="topk_prob_budget"),
            "policy_name": "topk_prob_budget",
        },
    }
    if calibration["selected"] is not None:
        selected_threshold = calibration["selected"]["threshold"]
        policies[f"guarded_threshold_{selected_threshold:g}"] = {
            **calibration["selected"],
            "policy_name": f"guarded_threshold_{selected_threshold:g}",
        }

    for item in policies.values():
        item["within_avg_visual_on_budget"] = item["predicted_avg_visual_on_layers"] <= max_predicted_avg
        item["improves_false_negative_rate_vs_threshold"] = (
            item["false_negative_rate"] < default_metrics["false_negative_rate"]
        )

    candidates = list(policies.values())
    within_budget = [item for item in candidates if item["within_avg_visual_on_budget"]]
    ranked = within_budget or candidates
    selected = min(
        ranked,
        key=lambda item: (
            item["false_negative_rate"],
            item["budget_mae"],
            -item["route_f1"],
            abs(item["predicted_avg_visual_on_layers"] - target_avg),
            item["policy_name"],
        ),
    )
    return {
        "default_threshold": float(default_threshold),
        "max_extra_avg_visual_on_layers": float(max_extra_avg_visual_on_layers),
        "max_predicted_avg_visual_on_layers": max_predicted_avg,
        "policies": policies,
        "threshold_calibration": calibration,
        "selected": selected,
        "selection_policy": (
            "Compare default threshold, guarded threshold if available, and top-K by rounded "
            "probability mass. Prefer policies within target_avg + max_extra_avg_visual_on_layers; "
            "choose lowest false-negative rate, then lowest budget MAE, then highest F1."
        ),
    }


def evaluate_cached_router_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda_budget: float,
    lambda_fn: float,
    lambda_repair_margin: float = 0.0,
    repair_margin: float = 1.0,
    lambda_route_repair_contrastive: float = 0.0,
    route_repair_margin: float = 1.0,
    alpha_pos: float,
    alpha_neg: float,
    gamma: float,
    threshold: float,
    positive_layer_weights: torch.Tensor | None = None,
    repair_positive_mask: torch.Tensor | None = None,
    repair_negative_mask: torch.Tensor | None = None,
    route_repair_positive_routes: torch.Tensor | None = None,
    route_repair_negative_routes: torch.Tensor | None = None,
    route_repair_positive_mask: torch.Tensor | None = None,
    route_repair_negative_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    parts = compute_route_loss(
        logits,
        labels,
        lambda_budget=lambda_budget,
        lambda_fn=lambda_fn,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        gamma=gamma,
        positive_layer_weights=positive_layer_weights,
    )
    if (repair_positive_mask is None) != (repair_negative_mask is None):
        raise ValueError("repair_positive_mask and repair_negative_mask must be provided together")
    if lambda_repair_margin < 0.0:
        raise ValueError("lambda_repair_margin must be non-negative")
    if repair_margin < 0.0:
        raise ValueError("repair_margin must be non-negative")
    route_repair_items = [
        route_repair_positive_routes,
        route_repair_negative_routes,
        route_repair_positive_mask,
        route_repair_negative_mask,
    ]
    if any(item is None for item in route_repair_items) and not all(item is None for item in route_repair_items):
        raise ValueError("route repair routes and masks must be provided together")
    if lambda_route_repair_contrastive < 0.0:
        raise ValueError("lambda_route_repair_contrastive must be non-negative")
    if route_repair_margin < 0.0:
        raise ValueError("route_repair_margin must be non-negative")
    if repair_positive_mask is None:
        repair_loss = logits.float().sum() * 0.0
        repair_metrics = repair_pairwise_margin_metrics(
            logits,
            torch.zeros_like(logits, dtype=torch.bool),
            torch.zeros_like(logits, dtype=torch.bool),
            margin=repair_margin,
        )
    else:
        repair_loss = repair_pairwise_margin_loss(
            logits,
            repair_positive_mask,
            repair_negative_mask,
            margin=repair_margin,
        )
        repair_metrics = repair_pairwise_margin_metrics(
            logits,
            repair_positive_mask,
            repair_negative_mask,
            margin=repair_margin,
        )
    if route_repair_positive_routes is None:
        route_repair_loss = logits.float().sum() * 0.0
        empty_routes = torch.zeros(logits.shape[0], 0, logits.shape[1], dtype=torch.bool, device=logits.device)
        empty_mask = torch.zeros(logits.shape[0], 0, dtype=torch.bool, device=logits.device)
        route_repair_metrics = route_repair_contrastive_metrics(
            logits,
            empty_routes,
            empty_routes,
            empty_mask,
            empty_mask,
            margin=route_repair_margin,
        )
    else:
        route_repair_loss = route_repair_contrastive_loss(
            logits,
            route_repair_positive_routes,
            route_repair_negative_routes,
            route_repair_positive_mask,
            route_repair_negative_mask,
            margin=route_repair_margin,
        )
        route_repair_metrics = route_repair_contrastive_metrics(
            logits,
            route_repair_positive_routes,
            route_repair_negative_routes,
            route_repair_positive_mask,
            route_repair_negative_mask,
            margin=route_repair_margin,
        )
    total_loss = (
        parts["loss"]
        + float(lambda_repair_margin) * repair_loss
        + float(lambda_route_repair_contrastive) * route_repair_loss
    )
    metrics = compute_route_metrics(logits, labels, threshold=threshold)
    return {
        "loss": float(total_loss.item()),
        "route_loss": float(parts["loss"].item()),
        "focal_bce": float(parts["focal_bce"].item()),
        "budget": float(parts["budget"].item()),
        "false_negative_loss": float(parts["false_negative"].item()),
        "repair_margin_loss": float(repair_loss.item()),
        "weighted_repair_margin_loss": float((float(lambda_repair_margin) * repair_loss).item()),
        "repair_margin": repair_metrics,
        "route_repair_contrastive_loss": float(route_repair_loss.item()),
        "weighted_route_repair_contrastive_loss": float(
            (float(lambda_route_repair_contrastive) * route_repair_loss).item()
        ),
        "route_repair_contrastive": route_repair_metrics,
        **metrics,
    }


def evaluate_cached_router_batch(
    router: BinaryVisualOnRouter,
    batch: dict[str, Any],
    *,
    lambda_budget: float,
    lambda_fn: float,
    lambda_repair_margin: float = 0.0,
    repair_margin: float = 1.0,
    lambda_route_repair_contrastive: float = 0.0,
    route_repair_margin: float = 1.0,
    alpha_pos: float,
    alpha_neg: float,
    gamma: float,
    threshold: float,
    positive_layer_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    with torch.no_grad():
        logits = _router_logits(router, batch)
        labels = batch["soft_labels"].float()
        return evaluate_cached_router_logits(
            logits,
            labels,
            lambda_budget=lambda_budget,
            lambda_fn=lambda_fn,
            lambda_repair_margin=lambda_repair_margin,
            repair_margin=repair_margin,
            lambda_route_repair_contrastive=lambda_route_repair_contrastive,
            route_repair_margin=route_repair_margin,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma,
            threshold=threshold,
            positive_layer_weights=positive_layer_weights,
            repair_positive_mask=batch.get("repair_positive_mask"),
            repair_negative_mask=batch.get("repair_negative_mask"),
            route_repair_positive_routes=batch.get("route_repair_positive_routes"),
            route_repair_negative_routes=batch.get("route_repair_negative_routes"),
            route_repair_positive_mask=batch.get("route_repair_positive_mask"),
            route_repair_negative_mask=batch.get("route_repair_negative_mask"),
        )


def run_cached_router_train_val_pilot(
    train_batch: dict[str, Any],
    val_batch: dict[str, Any],
    *,
    hidden_dim: int = 256,
    steps: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    lambda_budget: float = 0.05,
    lambda_fn: float = 0.0,
    lambda_repair_margin: float = 0.0,
    repair_margin: float = 1.0,
    lambda_route_repair_contrastive: float = 0.0,
    route_repair_margin: float = 1.0,
    alpha_pos: float = 3.0,
    alpha_neg: float = 1.0,
    gamma: float = 2.0,
    max_grad_norm: float = 1.0,
    threshold: float = 0.5,
    calibration_thresholds: list[float] | None = None,
    calibration_max_extra_avg_visual_on_layers: float = 1.0,
    seed: int = 0,
    return_artifacts: bool = False,
    use_prev_gate: bool = True,
    scalar_features_dim: int | None = None,
    visual_summary_count: int | None = None,
    device: torch.device | str | None = None,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    positive_layer_weights: torch.Tensor | None = None,
    train_repair_positive_mask: torch.Tensor | None = None,
    train_repair_negative_mask: torch.Tensor | None = None,
    train_route_repair_positive_routes: torch.Tensor | None = None,
    train_route_repair_negative_routes: torch.Tensor | None = None,
    train_route_repair_positive_mask: torch.Tensor | None = None,
    train_route_repair_negative_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if train_batch_size is not None and train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive when provided")
    if eval_batch_size is not None and eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive when provided")
    if lambda_repair_margin < 0.0:
        raise ValueError("lambda_repair_margin must be non-negative")
    if repair_margin < 0.0:
        raise ValueError("repair_margin must be non-negative")
    if (train_repair_positive_mask is None) != (train_repair_negative_mask is None):
        raise ValueError("train repair positive and negative masks must be provided together")
    route_repair_items = [
        train_route_repair_positive_routes,
        train_route_repair_negative_routes,
        train_route_repair_positive_mask,
        train_route_repair_negative_mask,
    ]
    if any(item is None for item in route_repair_items) and not all(item is None for item in route_repair_items):
        raise ValueError("train route repair routes and masks must be provided together")
    if lambda_route_repair_contrastive < 0.0:
        raise ValueError("lambda_route_repair_contrastive must be non-negative")
    if route_repair_margin < 0.0:
        raise ValueError("route_repair_margin must be non-negative")
    torch.manual_seed(seed)
    d_model = int(train_batch["global_mean"].shape[-1])
    if int(val_batch["global_mean"].shape[-1]) != d_model:
        raise ValueError("train and validation feature dimensions differ")
    inferred_scalar_dim = 0
    if "scalar_features" in train_batch or "scalar_features" in val_batch:
        if "scalar_features" not in train_batch or "scalar_features" not in val_batch:
            raise ValueError("train and validation batches must both include scalar_features or neither")
        train_scalar = train_batch["scalar_features"]
        val_scalar = val_batch["scalar_features"]
        if train_scalar.ndim != 2 or val_scalar.ndim != 2:
            raise ValueError("scalar_features must have shape [B, S] for cached training")
        inferred_scalar_dim = int(train_scalar.shape[-1])
        if int(val_scalar.shape[-1]) != inferred_scalar_dim:
            raise ValueError("train and validation scalar feature dimensions differ")
    if scalar_features_dim is None:
        scalar_features_dim = inferred_scalar_dim
    if int(scalar_features_dim) != inferred_scalar_dim:
        raise ValueError(
            f"scalar_features_dim={scalar_features_dim} does not match batch scalar dimension {inferred_scalar_dim}"
        )
    inferred_visual_summary_count = 0
    if "visual_summaries" in train_batch or "visual_summaries" in val_batch:
        if "visual_summaries" not in train_batch or "visual_summaries" not in val_batch:
            raise ValueError("train and validation batches must both include visual_summaries or neither")
        train_visual = train_batch["visual_summaries"]
        val_visual = val_batch["visual_summaries"]
        if train_visual.ndim != 4 or val_visual.ndim != 4:
            raise ValueError("visual_summaries must have shape [B, L, N, D] for cached training")
        if int(train_visual.shape[1]) != NUM_LAYERS or int(val_visual.shape[1]) != NUM_LAYERS:
            raise ValueError("visual_summaries must have NUM_LAYERS layer entries")
        inferred_visual_summary_count = int(train_visual.shape[2])
        if inferred_visual_summary_count <= 0:
            raise ValueError("visual_summaries must contain at least one summary")
        if int(val_visual.shape[2]) != inferred_visual_summary_count:
            raise ValueError("train and validation visual_summary_count differ")
        if int(train_visual.shape[-1]) != d_model or int(val_visual.shape[-1]) != d_model:
            raise ValueError("visual_summaries feature dimension must match d_model")
    if visual_summary_count is None:
        visual_summary_count = inferred_visual_summary_count
    if int(visual_summary_count) != inferred_visual_summary_count:
        raise ValueError(
            f"visual_summary_count={visual_summary_count} does not match batch visual summary count "
            f"{inferred_visual_summary_count}"
        )

    train_labels = train_batch["soft_labels"].float()
    val_labels = val_batch["soft_labels"].float()
    positive_layer_weights_tensor = (
        None if positive_layer_weights is None else torch.as_tensor(positive_layer_weights, dtype=torch.float32)
    )
    repair_positive_tensor = None
    repair_negative_tensor = None
    if train_repair_positive_mask is not None:
        repair_positive_tensor = torch.as_tensor(train_repair_positive_mask, dtype=torch.bool)
        repair_negative_tensor = torch.as_tensor(train_repair_negative_mask, dtype=torch.bool)
        if tuple(repair_positive_tensor.shape) != tuple(train_labels.shape):
            raise ValueError(
                f"train_repair_positive_mask must have shape {tuple(train_labels.shape)}, "
                f"got {tuple(repair_positive_tensor.shape)}"
            )
        if tuple(repair_negative_tensor.shape) != tuple(train_labels.shape):
            raise ValueError(
                f"train_repair_negative_mask must have shape {tuple(train_labels.shape)}, "
                f"got {tuple(repair_negative_tensor.shape)}"
            )
        train_batch = dict(train_batch)
        train_batch["repair_positive_mask"] = repair_positive_tensor
        train_batch["repair_negative_mask"] = repair_negative_tensor
    route_repair_positive_routes_tensor = None
    route_repair_negative_routes_tensor = None
    route_repair_positive_mask_tensor = None
    route_repair_negative_mask_tensor = None
    if train_route_repair_positive_routes is not None:
        route_repair_positive_routes_tensor = torch.as_tensor(train_route_repair_positive_routes, dtype=torch.bool)
        route_repair_negative_routes_tensor = torch.as_tensor(train_route_repair_negative_routes, dtype=torch.bool)
        route_repair_positive_mask_tensor = torch.as_tensor(train_route_repair_positive_mask, dtype=torch.bool)
        route_repair_negative_mask_tensor = torch.as_tensor(train_route_repair_negative_mask, dtype=torch.bool)
        _route_repair_summary(
            route_repair_positive_routes_tensor,
            route_repair_negative_routes_tensor,
            route_repair_positive_mask_tensor,
            route_repair_negative_mask_tensor,
        )
        if int(route_repair_positive_routes_tensor.shape[0]) != int(train_labels.shape[0]):
            raise ValueError(
                "train_route_repair_positive_routes batch dimension must match train labels; "
                f"got {tuple(route_repair_positive_routes_tensor.shape)} and {tuple(train_labels.shape)}"
            )
        if int(route_repair_negative_routes_tensor.shape[0]) != int(train_labels.shape[0]):
            raise ValueError(
                "train_route_repair_negative_routes batch dimension must match train labels; "
                f"got {tuple(route_repair_negative_routes_tensor.shape)} and {tuple(train_labels.shape)}"
            )
        if int(route_repair_positive_routes_tensor.shape[2]) != int(train_labels.shape[1]):
            raise ValueError(
                "train_route_repair_positive_routes layer dimension must match train labels; "
                f"got {tuple(route_repair_positive_routes_tensor.shape)} and {tuple(train_labels.shape)}"
            )
        if int(route_repair_negative_routes_tensor.shape[2]) != int(train_labels.shape[1]):
            raise ValueError(
                "train_route_repair_negative_routes layer dimension must match train labels; "
                f"got {tuple(route_repair_negative_routes_tensor.shape)} and {tuple(train_labels.shape)}"
            )
        train_batch = dict(train_batch)
        train_batch["route_repair_positive_routes"] = route_repair_positive_routes_tensor
        train_batch["route_repair_negative_routes"] = route_repair_negative_routes_tensor
        train_batch["route_repair_positive_mask"] = route_repair_positive_mask_tensor
        train_batch["route_repair_negative_mask"] = route_repair_negative_mask_tensor
    train_repair_summary = _repair_mask_summary(repair_positive_tensor, repair_negative_tensor)
    train_route_repair_summary = _route_repair_summary(
        route_repair_positive_routes_tensor,
        route_repair_negative_routes_tensor,
        route_repair_positive_mask_tensor,
        route_repair_negative_mask_tensor,
    )
    train_label_mix = count_label_mix(train_labels)
    val_label_mix = count_label_mix(val_labels)
    for split_name, label_mix in [("train", train_label_mix), ("val", val_label_mix)]:
        if label_mix["num_zero_label_samples"] == 0 or label_mix["num_positive_label_samples"] == 0:
            raise ValueError(f"{split_name} split requires both zero-label and positive-label samples")

    router_device = torch.device(device) if device is not None else train_batch["global_mean"].device
    router = BinaryVisualOnRouter(
        d_model=d_model,
        num_layers=NUM_LAYERS,
        hidden_dim=hidden_dim,
        use_prev_gate=use_prev_gate,
        scalar_features_dim=int(scalar_features_dim),
        visual_summary_count=int(visual_summary_count),
    )
    router.to(device=router_device)
    router.train()
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
    before = _parameter_snapshot(router)
    effective_eval_batch_size = eval_batch_size or train_batch_size

    if effective_eval_batch_size is None:
        initial_train = evaluate_cached_router_batch(
            router,
            train_batch,
            lambda_budget=lambda_budget,
            lambda_fn=lambda_fn,
            lambda_repair_margin=lambda_repair_margin,
            repair_margin=repair_margin,
            lambda_route_repair_contrastive=lambda_route_repair_contrastive,
            route_repair_margin=route_repair_margin,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma,
            threshold=threshold,
            positive_layer_weights=positive_layer_weights_tensor,
        )
        initial_val = evaluate_cached_router_batch(
            router,
            val_batch,
            lambda_budget=lambda_budget,
            lambda_fn=lambda_fn,
            lambda_repair_margin=0.0,
            repair_margin=repair_margin,
            lambda_route_repair_contrastive=0.0,
            route_repair_margin=route_repair_margin,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma,
            threshold=threshold,
            positive_layer_weights=positive_layer_weights_tensor,
        )
    else:
        initial_train_logits = _router_logits_in_chunks(
            router,
            train_batch,
            chunk_size=effective_eval_batch_size,
            device=router_device,
        )
        initial_val_logits = _router_logits_in_chunks(
            router,
            val_batch,
            chunk_size=effective_eval_batch_size,
            device=router_device,
        )
        initial_train = evaluate_cached_router_logits(
            initial_train_logits,
            train_labels.cpu(),
            lambda_budget=lambda_budget,
            lambda_fn=lambda_fn,
            lambda_repair_margin=lambda_repair_margin,
            repair_margin=repair_margin,
            lambda_route_repair_contrastive=lambda_route_repair_contrastive,
            route_repair_margin=route_repair_margin,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma,
            threshold=threshold,
            positive_layer_weights=positive_layer_weights_tensor,
            repair_positive_mask=repair_positive_tensor,
            repair_negative_mask=repair_negative_tensor,
            route_repair_positive_routes=route_repair_positive_routes_tensor,
            route_repair_negative_routes=route_repair_negative_routes_tensor,
            route_repair_positive_mask=route_repair_positive_mask_tensor,
            route_repair_negative_mask=route_repair_negative_mask_tensor,
        )
        initial_val = evaluate_cached_router_logits(
            initial_val_logits,
            val_labels.cpu(),
            lambda_budget=lambda_budget,
            lambda_fn=lambda_fn,
            lambda_repair_margin=0.0,
            repair_margin=repair_margin,
            lambda_route_repair_contrastive=0.0,
            route_repair_margin=route_repair_margin,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma,
            threshold=threshold,
            positive_layer_weights=positive_layer_weights_tensor,
        )

    loss_history = []
    last_grad_norm = 0.0
    num_train_samples = int(train_batch["global_mean"].shape[0])
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if train_batch_size is None:
            logits = _router_logits(router, train_batch)
            parts = compute_route_loss(
                logits,
                train_labels,
                lambda_budget=lambda_budget,
                lambda_fn=lambda_fn,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                gamma=gamma,
                positive_layer_weights=positive_layer_weights_tensor,
            )
            repair_loss = (
                logits.float().sum() * 0.0
                if repair_positive_tensor is None
                else repair_pairwise_margin_loss(
                    logits,
                    repair_positive_tensor.to(device=logits.device),
                    repair_negative_tensor.to(device=logits.device),
                    margin=repair_margin,
                )
            )
            route_repair_loss = (
                logits.float().sum() * 0.0
                if route_repair_positive_routes_tensor is None
                else route_repair_contrastive_loss(
                    logits,
                    route_repair_positive_routes_tensor,
                    route_repair_negative_routes_tensor,
                    route_repair_positive_mask_tensor,
                    route_repair_negative_mask_tensor,
                    margin=route_repair_margin,
                )
            )
            total_loss = (
                parts["loss"]
                + float(lambda_repair_margin) * repair_loss
                + float(lambda_route_repair_contrastive) * route_repair_loss
            )
            total_loss.backward()
            step_loss = float(total_loss.detach().item())
        else:
            step_loss = 0.0
            for start in range(0, num_train_samples, train_batch_size):
                end = min(start + train_batch_size, num_train_samples)
                chunk = _slice_batch_to_device(
                    train_batch,
                    start,
                    end,
                    batch_size=num_train_samples,
                    device=router_device,
                )
                logits = _router_logits(router, chunk)
                parts = compute_route_loss(
                    logits,
                    chunk["soft_labels"].float(),
                    lambda_budget=lambda_budget,
                    lambda_fn=lambda_fn,
                    alpha_pos=alpha_pos,
                    alpha_neg=alpha_neg,
                    gamma=gamma,
                    positive_layer_weights=positive_layer_weights_tensor,
                )
                repair_loss = (
                    logits.float().sum() * 0.0
                    if "repair_positive_mask" not in chunk
                    else repair_pairwise_margin_loss(
                        logits,
                        chunk["repair_positive_mask"],
                        chunk["repair_negative_mask"],
                        margin=repair_margin,
                    )
                )
                route_repair_loss = (
                    logits.float().sum() * 0.0
                    if "route_repair_positive_routes" not in chunk
                    else route_repair_contrastive_loss(
                        logits,
                        chunk["route_repair_positive_routes"],
                        chunk["route_repair_negative_routes"],
                        chunk["route_repair_positive_mask"],
                        chunk["route_repair_negative_mask"],
                        margin=route_repair_margin,
                    )
                )
                weight = float(end - start) / float(num_train_samples)
                weighted_loss = (
                    parts["loss"]
                    + float(lambda_repair_margin) * repair_loss
                    + float(lambda_route_repair_contrastive) * route_repair_loss
                ) * weight
                weighted_loss.backward()
                step_loss += float(weighted_loss.detach().item())
        last_grad_norm = float(clip_grad_norm_(router.parameters(), max_grad_norm).item())
        optimizer.step()
        loss_history.append(step_loss)

    with torch.no_grad():
        if effective_eval_batch_size is None:
            final_train_logits = _router_logits(router, train_batch)
            final_val_logits = _router_logits(router, val_batch)
            metric_train_labels = train_labels
            metric_val_labels = val_labels
        else:
            final_train_logits = _router_logits_in_chunks(
                router,
                train_batch,
                chunk_size=effective_eval_batch_size,
                device=router_device,
            )
            final_val_logits = _router_logits_in_chunks(
                router,
                val_batch,
                chunk_size=effective_eval_batch_size,
                device=router_device,
            )
            metric_train_labels = train_labels.cpu()
            metric_val_labels = val_labels.cpu()
        final_train = evaluate_cached_router_logits(
            final_train_logits,
            metric_train_labels,
            lambda_budget=lambda_budget,
            lambda_fn=lambda_fn,
            lambda_repair_margin=lambda_repair_margin,
            repair_margin=repair_margin,
            lambda_route_repair_contrastive=lambda_route_repair_contrastive,
            route_repair_margin=route_repair_margin,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma,
            threshold=threshold,
            positive_layer_weights=positive_layer_weights_tensor,
            repair_positive_mask=repair_positive_tensor,
            repair_negative_mask=repair_negative_tensor,
            route_repair_positive_routes=route_repair_positive_routes_tensor,
            route_repair_negative_routes=route_repair_negative_routes_tensor,
            route_repair_positive_mask=route_repair_positive_mask_tensor,
            route_repair_negative_mask=route_repair_negative_mask_tensor,
        )
        final_val = evaluate_cached_router_logits(
            final_val_logits,
            metric_val_labels,
            lambda_budget=lambda_budget,
            lambda_fn=lambda_fn,
            lambda_repair_margin=0.0,
            repair_margin=repair_margin,
            lambda_route_repair_contrastive=0.0,
            route_repair_margin=route_repair_margin,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma,
            threshold=threshold,
            positive_layer_weights=positive_layer_weights_tensor,
        )
        train_calibration = calibrate_route_threshold(
            final_train_logits,
            metric_train_labels,
            default_threshold=threshold,
            thresholds=calibration_thresholds,
            max_extra_avg_visual_on_layers=calibration_max_extra_avg_visual_on_layers,
        )
        train_policy_comparison = compare_route_policies(
            final_train_logits,
            metric_train_labels,
            default_threshold=threshold,
            thresholds=calibration_thresholds,
            max_extra_avg_visual_on_layers=calibration_max_extra_avg_visual_on_layers,
        )
        val_policy_comparison = compare_route_policies(
            final_val_logits,
            metric_val_labels,
            default_threshold=threshold,
            thresholds=calibration_thresholds,
            max_extra_avg_visual_on_layers=calibration_max_extra_avg_visual_on_layers,
        )
        val_calibration = calibrate_route_threshold(
            final_val_logits,
            metric_val_labels,
            default_threshold=threshold,
            thresholds=calibration_thresholds,
            max_extra_avg_visual_on_layers=calibration_max_extra_avg_visual_on_layers,
        )

    parameter_delta = _parameter_l2_delta(router, before)
    if parameter_delta <= 0.0:
        raise AssertionError("router parameters did not update")
    if not torch.isfinite(torch.tensor(final_train["loss"])).item():
        raise AssertionError("final train loss is not finite")
    if not torch.isfinite(torch.tensor(final_val["loss"])).item():
        raise AssertionError("final validation loss is not finite")

    trainable_parameters = int(sum(param.numel() for param in router.parameters() if param.requires_grad))
    optimizer_parameters = int(sum(param.numel() for group in optimizer.param_groups for param in group["params"]))
    train_loss_decreased = final_train["loss"] < initial_train["loss"]
    if not train_loss_decreased:
        raise AssertionError(
            f"train loss did not decrease: initial={initial_train['loss']}, final={final_train['loss']}"
        )

    summary = {
        "verification_status": "passed",
        "num_layers": NUM_LAYERS,
        "d_model": d_model,
        "hidden_dim": int(hidden_dim),
        "use_prev_gate": bool(use_prev_gate),
        "scalar_features_dim": int(scalar_features_dim),
        "visual_summary_count": int(visual_summary_count),
        "device": str(router_device),
        "train_batch_size": None if train_batch_size is None else int(train_batch_size),
        "eval_batch_size": None if effective_eval_batch_size is None else int(effective_eval_batch_size),
        "steps": int(steps),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "lambda_budget": float(lambda_budget),
        "lambda_fn": float(lambda_fn),
        "lambda_repair_margin": float(lambda_repair_margin),
        "repair_margin": float(repair_margin),
        "lambda_route_repair_contrastive": float(lambda_route_repair_contrastive),
        "route_repair_margin": float(route_repair_margin),
        "alpha_pos": float(alpha_pos),
        "alpha_neg": float(alpha_neg),
        "gamma": float(gamma),
        "positive_layer_weights": (
            None
            if positive_layer_weights_tensor is None
            else [float(value) for value in positive_layer_weights_tensor.detach().cpu().view(-1).tolist()]
        ),
        "max_grad_norm": float(max_grad_norm),
        "threshold": float(threshold),
        "calibration_max_extra_avg_visual_on_layers": float(calibration_max_extra_avg_visual_on_layers),
        "initial_train_loss": initial_train["loss"],
        "final_train_loss": final_train["loss"],
        "initial_val_loss": initial_val["loss"],
        "final_val_loss": final_val["loss"],
        "train_loss_decreased": train_loss_decreased,
        "val_loss_decreased": final_val["loss"] < initial_val["loss"],
        "loss_history_first": loss_history[:5],
        "loss_history_last": loss_history[-5:],
        "initial_train_metrics": initial_train,
        "final_train_metrics": final_train,
        "initial_val_metrics": initial_val,
        "final_val_metrics": final_val,
        "train_repair_margin": train_repair_summary,
        "initial_train_repair_margin": initial_train["repair_margin"],
        "final_train_repair_margin": final_train["repair_margin"],
        "train_route_repair_contrastive": train_route_repair_summary,
        "initial_train_route_repair_contrastive": initial_train["route_repair_contrastive"],
        "final_train_route_repair_contrastive": final_train["route_repair_contrastive"],
        "final_train_calibration": train_calibration,
        "final_val_calibration": val_calibration,
        "final_train_policy_comparison": train_policy_comparison,
        "final_val_policy_comparison": val_policy_comparison,
        "router_parameter_l2_delta": parameter_delta,
        "last_grad_norm_before_clip": last_grad_norm,
        "router_trainable_parameters": trainable_parameters,
        "optimizer_parameter_count": optimizer_parameters,
        "optimizer_updates_only_router_parameters": optimizer_parameters == trainable_parameters,
        "optimizer_parameter_scope": "router_only",
        "qwen_parameters_loaded": 0,
        "uses_cached_features_only": True,
        "train_label_mix": train_label_mix,
        "val_label_mix": val_label_mix,
    }
    if return_artifacts:
        summary["_artifacts"] = {
            "router_state_dict": {
                name: param.detach().cpu().clone() for name, param in router.state_dict().items()
            },
            "final_train_logits": final_train_logits.detach().cpu().clone(),
            "final_val_logits": final_val_logits.detach().cpu().clone(),
            "train_labels": train_labels.detach().cpu().clone(),
            "val_labels": val_labels.detach().cpu().clone(),
        }
    return summary


def run_cached_router_optimizer_pilot(
    batch: dict[str, Any],
    *,
    hidden_dim: int = 256,
    steps: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    lambda_budget: float = 0.05,
    lambda_fn: float = 0.0,
    alpha_pos: float = 3.0,
    alpha_neg: float = 1.0,
    gamma: float = 2.0,
    max_grad_norm: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    d_model = int(batch["global_mean"].shape[-1])
    router = BinaryVisualOnRouter(d_model=d_model, num_layers=NUM_LAYERS, hidden_dim=hidden_dim)
    router.to(device=batch["global_mean"].device)
    router.train()
    labels = batch["soft_labels"].float()
    label_mix = count_label_mix(labels)
    if label_mix["num_zero_label_samples"] == 0 or label_mix["num_positive_label_samples"] == 0:
        raise ValueError("optimizer pilot requires both zero-label and positive-label samples")

    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
    before = _parameter_snapshot(router)

    with torch.no_grad():
        initial_logits = _router_logits(router, batch)
        initial_parts = compute_route_loss(
            initial_logits,
            labels,
            lambda_budget=lambda_budget,
            lambda_fn=lambda_fn,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma,
        )
        initial_loss = float(initial_parts["loss"].item())

    history = []
    last_grad_norm = 0.0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = _router_logits(router, batch)
        parts = compute_route_loss(
            logits,
            labels,
            lambda_budget=lambda_budget,
            lambda_fn=lambda_fn,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma,
        )
        parts["loss"].backward()
        last_grad_norm = float(clip_grad_norm_(router.parameters(), max_grad_norm).item())
        optimizer.step()
        history.append(float(parts["loss"].detach().item()))

    with torch.no_grad():
        final_logits = _router_logits(router, batch)
        final_parts = compute_route_loss(
            final_logits,
            labels,
            lambda_budget=lambda_budget,
            lambda_fn=lambda_fn,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma,
        )
        final_loss = float(final_parts["loss"].item())
        final_probs = torch.sigmoid(final_logits)

    parameter_delta = _parameter_l2_delta(router, before)
    loss_decreased = final_loss < initial_loss
    if not torch.isfinite(torch.tensor(final_loss)).item():
        raise AssertionError("final loss is not finite")
    if parameter_delta <= 0.0:
        raise AssertionError("router parameters did not update")
    if not loss_decreased:
        raise AssertionError(f"loss did not decrease: initial={initial_loss}, final={final_loss}")

    trainable_parameters = int(sum(param.numel() for param in router.parameters() if param.requires_grad))
    optimizer_parameters = int(sum(param.numel() for group in optimizer.param_groups for param in group["params"]))
    return {
        "verification_status": "passed",
        "num_layers": NUM_LAYERS,
        "d_model": d_model,
        "hidden_dim": int(hidden_dim),
        "steps": int(steps),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "lambda_budget": float(lambda_budget),
        "lambda_fn": float(lambda_fn),
        "alpha_pos": float(alpha_pos),
        "alpha_neg": float(alpha_neg),
        "gamma": float(gamma),
        "max_grad_norm": float(max_grad_norm),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": loss_decreased,
        "loss_history_first": history[:5],
        "loss_history_last": history[-5:],
        "initial_budget": float(initial_parts["budget"].item()),
        "final_budget": float(final_parts["budget"].item()),
        "initial_focal_bce": float(initial_parts["focal_bce"].item()),
        "final_focal_bce": float(final_parts["focal_bce"].item()),
        "target_avg_visual_on_layers": label_mix["avg_visual_on_layers"],
        "predicted_avg_visual_on_layers": float(final_probs.sum(dim=1).mean().item()),
        "router_parameter_l2_delta": parameter_delta,
        "last_grad_norm_before_clip": last_grad_norm,
        "router_trainable_parameters": trainable_parameters,
        "optimizer_parameter_count": optimizer_parameters,
        "optimizer_updates_only_router_parameters": optimizer_parameters == trainable_parameters,
        "optimizer_parameter_scope": "router_only",
        "qwen_parameters_loaded": 0,
        "uses_cached_features_only": True,
        **label_mix,
    }
