"""Bounded trainer for a frozen-encoder binary POLAR predictor."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

from .evaluation import batch_offline_metrics, mask_diversity_metrics
from .losses import multi_valid_set_nll, polar_path_bce_per_path
from .multimodal import MODALITIES, resolve_modality_inputs
from .structured import structured_batch_metrics, structured_valid_set_nll


def predictor_state_sha256(predictor) -> str:
    """Hash predictor tensor contents for matched-initialization verification."""
    digest = sha256()
    for name, tensor in sorted(predictor.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _amp_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is not None and device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, dtype=amp_dtype)
    return nullcontext()


def train_epoch(
    predictor,
    frozen_encoder,
    loader,
    optimizer,
    *,
    device: torch.device,
    objective: str = "exact_set_nll",
    amp_dtype: torch.dtype | None = None,
    gradient_clip_norm: float = 1.0,
    duplicated_route_microbatch_size: int | None = None,
) -> dict[str, Any]:
    if objective not in {"duplicated_bce", "exact_set_nll"}:
        raise ValueError("objective must be 'duplicated_bce' or 'exact_set_nll'")
    if duplicated_route_microbatch_size is not None and duplicated_route_microbatch_size < 1:
        raise ValueError("duplicated_route_microbatch_size must be positive or None")
    predictor.train()
    frozen_encoder.eval()
    total_loss = 0.0
    examples = 0
    for batch in loader:
        batch = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            token_features = frozen_encoder(batch["input_ids"], batch["attention_mask"])
        if objective == "exact_set_nll":
            with _amp_context(device, amp_dtype):
                logits = predictor(token_features, batch["attention_mask"])
                loss = multi_valid_set_nll(
                    logits,
                    batch["valid_masks"],
                    valid_mask=batch["valid_mask"],
                    route_weights=batch["route_weights"],
                )
            loss.backward()
            batch_size = int(batch["input_ids"].shape[0])
            loss_value = float(loss.detach())
        else:
            batch_size = int(batch["unique_examples"])
            route_count = int(batch["targets"].shape[0])
            chunk_size = duplicated_route_microbatch_size or route_count
            loss_value = 0.0
            for start in range(0, route_count, chunk_size):
                end = min(start + chunk_size, route_count)
                sample_indices = batch["route_sample_index"][start:end]
                with _amp_context(device, amp_dtype):
                    logits = predictor(
                        token_features.index_select(0, sample_indices),
                        batch["attention_mask"].index_select(0, sample_indices),
                    )
                    per_path = polar_path_bce_per_path(logits, batch["targets"][start:end])
                    chunk_loss = (
                        per_path * batch["sample_weights"][start:end].to(per_path.dtype)
                    ).sum() / batch_size
                chunk_loss.backward()
                loss_value += float(chunk_loss.detach())
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), gradient_clip_norm)
        optimizer.step()
        total_loss += loss_value * batch_size
        examples += batch_size
    mean_loss = total_loss / max(examples, 1)
    return {
        "loss": mean_loss,
        "objective": objective,
        "examples": examples,
        ("set_nll" if objective == "exact_set_nll" else "duplicated_bce"): mean_loss,
    }


@torch.no_grad()
def evaluate_epoch(
    predictor,
    frozen_encoder,
    loader,
    *,
    device: torch.device,
    top_k: int = 5,
    amp_dtype: torch.dtype | None = None,
):
    predictor.eval()
    frozen_encoder.eval()
    loss_sum = 0.0
    metrics_sum: dict[str, float] = {}
    top1_mask_counts: dict[str, int] = {}
    examples = 0
    for batch in loader:
        batch = _move(batch, device)
        token_features = frozen_encoder(batch["input_ids"], batch["attention_mask"])
        with _amp_context(device, amp_dtype):
            logits = predictor(token_features, batch["attention_mask"])
            loss = multi_valid_set_nll(
                logits,
                batch["valid_masks"],
                valid_mask=batch["valid_mask"],
                route_weights=batch["route_weights"],
            )
            current = batch_offline_metrics(
                logits, batch["valid_masks"], batch["valid_mask"], top_k=top_k
            )
        batch_size = int(logits.shape[0])
        loss_sum += float(loss) * batch_size
        examples += batch_size
        for key, value in current.items():
            if key == "top1_mask_counts":
                for mask, count in value.items():
                    top1_mask_counts[mask] = top1_mask_counts.get(mask, 0) + int(count)
                continue
            if key in {
                "unique_top1_masks",
                "fraction_top1_all_on",
                "fraction_top1_all_off",
                "average_predicted_visual_on",
                "top1_mask_entropy_nats",
                "top5_masks",
            }:
                continue
            metrics_sum[key] = metrics_sum.get(key, 0.0) + value * batch_size
    diversity = mask_diversity_metrics(
        Counter({tuple(int(bit) for bit in mask): count for mask, count in top1_mask_counts.items()})
    )
    return {
        "set_nll": loss_sum / max(examples, 1),
        "examples": examples,
        **{key: value / max(examples, 1) for key, value in metrics_sum.items()},
        **diversity,
    }


def train_structured_epoch(
    predictor,
    frozen_encoder,
    loader,
    optimizer,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    gradient_clip_norm: float = 1.0,
) -> dict[str, Any]:
    """Train one epoch with the P12 canonical exact valid-set likelihood."""
    predictor.train()
    frozen_encoder.eval()
    total_loss = 0.0
    examples = 0
    for batch in loader:
        batch = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            token_features = frozen_encoder(batch["input_ids"], batch["attention_mask"])
        with _amp_context(device, amp_dtype):
            boundary_logits, operation_logits = predictor(
                token_features, batch["attention_mask"]
            )
            loss = structured_valid_set_nll(
                boundary_logits,
                operation_logits,
                batch["boundary_targets"],
                batch["operation_targets"],
                valid_mask=batch["valid_mask"],
                route_weights=batch["route_weights"],
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_size = int(batch["input_ids"].shape[0])
        total_loss += float(loss.detach()) * batch_size
        examples += batch_size
    mean_loss = total_loss / max(examples, 1)
    return {
        "loss": mean_loss,
        "set_nll": mean_loss,
        "objective": "structured_exact_set_nll",
        "examples": examples,
    }


@torch.no_grad()
def evaluate_structured_epoch(
    predictor,
    frozen_encoder,
    loader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Evaluate P12 exact likelihood and its frozen deterministic top-1 decode."""
    predictor.eval()
    frozen_encoder.eval()
    loss_sum = 0.0
    examples = 0
    sample_sums: dict[str, float] = {}
    raw_sums: dict[str, float] = {}
    top1_mask_counts: Counter[tuple[int, ...]] = Counter()
    sample_metric_keys = {
        "top1_valid_route_coverage",
        "topk_valid_route_coverage",
        "nearest_valid_hamming",
        "average_predicted_segments",
    }
    raw_metric_keys = {
        "_boundary_correct",
        "_boundary_total",
        "_boundary_tp",
        "_boundary_fp",
        "_boundary_fn",
        "_operation_correct",
        "_operation_total",
    }
    for batch in loader:
        batch = _move(batch, device)
        token_features = frozen_encoder(batch["input_ids"], batch["attention_mask"])
        with _amp_context(device, amp_dtype):
            boundary_logits, operation_logits = predictor(
                token_features, batch["attention_mask"]
            )
            loss = structured_valid_set_nll(
                boundary_logits,
                operation_logits,
                batch["boundary_targets"],
                batch["operation_targets"],
                valid_mask=batch["valid_mask"],
                route_weights=batch["route_weights"],
            )
        current = structured_batch_metrics(
            boundary_logits,
            operation_logits,
            batch["valid_masks"],
            batch["boundary_targets"],
            batch["operation_targets"],
            batch["valid_mask"],
            batch["route_weights"],
        )
        batch_size = int(boundary_logits.shape[0])
        loss_sum += float(loss) * batch_size
        examples += batch_size
        for key in sample_metric_keys:
            sample_sums[key] = sample_sums.get(key, 0.0) + float(current[key]) * batch_size
        for key in raw_metric_keys:
            raw_sums[key] = raw_sums.get(key, 0.0) + float(current[key])
        for serialized, count in current["_top1_mask_counts"].items():
            top1_mask_counts[tuple(int(bit) for bit in serialized)] += int(count)
    diversity = mask_diversity_metrics(top1_mask_counts)
    boundary_total = raw_sums["_boundary_total"]
    boundary_tp = raw_sums["_boundary_tp"]
    boundary_fp = raw_sums["_boundary_fp"]
    boundary_fn = raw_sums["_boundary_fn"]
    operation_total = raw_sums["_operation_total"]
    return {
        "set_nll": loss_sum / max(examples, 1),
        "examples": examples,
        **{key: value / max(examples, 1) for key, value in sample_sums.items()},
        "topk_candidate_count": 1,
        "top5_available": False,
        "boundary_accuracy": raw_sums["_boundary_correct"] / max(boundary_total, 1e-12),
        "boundary_precision": boundary_tp / max(boundary_tp + boundary_fp, 1e-12),
        "boundary_recall": boundary_tp / max(boundary_tp + boundary_fn, 1e-12),
        "segment_operation_accuracy_at_gt_boundaries": raw_sums["_operation_correct"]
        / max(operation_total, 1e-12),
        **diversity,
    }


def _multimodal_inputs(batch, frozen_encoder, modality: str):
    if modality not in MODALITIES:
        raise ValueError(f"unknown P13 modality {modality!r}")
    if modality == "image":
        token_features = batch["image_features"].new_zeros(
            batch["input_ids"].shape[0], 1, int(frozen_encoder.output_dim)
        )
        token_mask = batch["attention_mask"].new_zeros(batch["input_ids"].shape[0], 1)
    else:
        with torch.no_grad():
            token_features = frozen_encoder(batch["input_ids"], batch["attention_mask"])
        token_mask = batch["attention_mask"]
    return resolve_modality_inputs(
        modality,
        token_features,
        token_mask,
        batch["image_features"],
        batch["image_attention_mask"],
    )


def train_multimodal_epoch(
    predictor,
    frozen_encoder,
    loader,
    optimizer,
    *,
    modality: str,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    gradient_clip_norm: float = 1.0,
) -> dict[str, Any]:
    """Train one P13 modality with the unchanged direct exact-set objective."""
    predictor.train()
    frozen_encoder.eval()
    total_loss = 0.0
    examples = 0
    for batch in loader:
        batch = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        inputs = _multimodal_inputs(batch, frozen_encoder, modality)
        with _amp_context(device, amp_dtype):
            logits = predictor(*inputs)
            loss = multi_valid_set_nll(
                logits,
                batch["valid_masks"],
                valid_mask=batch["valid_mask"],
                route_weights=batch["route_weights"],
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_size = int(logits.shape[0])
        total_loss += float(loss.detach()) * batch_size
        examples += batch_size
    mean_loss = total_loss / max(examples, 1)
    return {
        "loss": mean_loss,
        "set_nll": mean_loss,
        "objective": "exact_set_nll",
        "modality": modality,
        "examples": examples,
    }


@torch.no_grad()
def evaluate_multimodal_epoch(
    predictor,
    frozen_encoder,
    loader,
    *,
    modality: str,
    device: torch.device,
    top_k: int = 5,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Evaluate one P13 modality with common complete-mask metrics."""
    predictor.eval()
    frozen_encoder.eval()
    loss_sum = 0.0
    metrics_sum: dict[str, float] = {}
    top1_mask_counts: dict[str, int] = {}
    examples = 0
    for batch in loader:
        batch = _move(batch, device)
        inputs = _multimodal_inputs(batch, frozen_encoder, modality)
        with _amp_context(device, amp_dtype):
            logits = predictor(*inputs)
            loss = multi_valid_set_nll(
                logits,
                batch["valid_masks"],
                valid_mask=batch["valid_mask"],
                route_weights=batch["route_weights"],
            )
            current = batch_offline_metrics(
                logits, batch["valid_masks"], batch["valid_mask"], top_k=top_k
            )
        batch_size = int(logits.shape[0])
        loss_sum += float(loss) * batch_size
        examples += batch_size
        for key, value in current.items():
            if key == "top1_mask_counts":
                for mask, count in value.items():
                    top1_mask_counts[mask] = top1_mask_counts.get(mask, 0) + int(count)
                continue
            if key in {
                "unique_top1_masks",
                "fraction_top1_all_on",
                "fraction_top1_all_off",
                "average_predicted_visual_on",
                "top1_mask_entropy_nats",
                "top5_masks",
            }:
                continue
            metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value) * batch_size
    diversity = mask_diversity_metrics(
        Counter({tuple(int(bit) for bit in mask): count for mask, count in top1_mask_counts.items()})
    )
    return {
        "set_nll": loss_sum / max(examples, 1),
        "examples": examples,
        "modality": modality,
        **{key: value / max(examples, 1) for key, value in metrics_sum.items()},
        **diversity,
    }


def save_checkpoint(path: str | Path, predictor, optimizer, *, epoch: int, config: dict, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps({"epoch": epoch, "config": config, "metrics": metrics}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
