"""Matched Image+Question training runtime for categorical four-action POLAR."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch

from experiments.train_binary_polar import file_sha256
from .evaluation import FourActionMetricAccumulator
from .losses import exact_valid_set_nll, polar_action_bce_per_route


OBJECTIVES = ("duplicated_action_bce", "exact_set_nll")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is not None and device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def forward_from_features(
    predictor,
    question: torch.Tensor,
    batch: dict[str, Any],
    *,
    sample_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    attention = batch["attention_mask"]
    image = batch["image_features"]
    image_attention = batch["image_attention_mask"]
    if sample_indices is not None:
        question = question.index_select(0, sample_indices)
        attention = attention.index_select(0, sample_indices)
        image = image.index_select(0, sample_indices)
        image_attention = image_attention.index_select(0, sample_indices)
    return predictor(question, attention, image, image_attention)


def _scale_gradients(module, denominator: int) -> None:
    if denominator < 1:
        raise ValueError("gradient denominator must be positive")
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


def train_epoch(
    predictor,
    encoder,
    loader,
    optimizer,
    scheduler,
    *,
    device: torch.device,
    objective: str,
    accumulation_steps: int,
    gradient_clip_norm: float,
    duplicated_route_microbatch_size: int,
    amp_dtype: torch.dtype | None,
    epoch: int,
    global_step: int,
    progress_first_batches: int = 0,
    progress_every_batches: int = 0,
) -> tuple[dict[str, Any], int]:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if accumulation_steps < 1 or duplicated_route_microbatch_size < 1:
        raise ValueError("accumulation and route microbatch sizes must be positive")
    if progress_first_batches < 0 or progress_every_batches < 0:
        raise ValueError("progress batch intervals must be nonnegative")
    predictor.train()
    encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    accumulated_examples = 0
    total_examples = 0
    total_loss = 0.0
    loader_length = len(loader)
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        with torch.no_grad():
            question = encoder(batch["input_ids"], batch["attention_mask"])
        batch_size = int(batch["unique_examples"])
        batch_loss_sum_value = 0.0
        if objective == "exact_set_nll":
            with _amp_context(device, amp_dtype):
                logits = forward_from_features(predictor, question, batch)
                mean_loss = exact_valid_set_nll(
                    logits,
                    batch["valid_routes"],
                    valid_mask=batch["valid_mask"],
                    route_weights=batch["route_weights"],
                )
                batch_loss_sum = mean_loss * batch_size
            if not bool(torch.isfinite(batch_loss_sum).item()):
                raise FloatingPointError(
                    f"nonfinite exact-set NLL at epoch {epoch} batch {batch_index}"
                )
            batch_loss_sum.backward()
            batch_loss_sum_value = float(batch_loss_sum.detach())
        else:
            route_count = int(batch["target_actions"].shape[0])
            for start in range(0, route_count, duplicated_route_microbatch_size):
                stop = min(start + duplicated_route_microbatch_size, route_count)
                indices = batch["route_sample_index"][start:stop]
                with _amp_context(device, amp_dtype):
                    logits = forward_from_features(
                        predictor, question, batch, sample_indices=indices
                    )
                    per_route = polar_action_bce_per_route(
                        logits, batch["target_actions"][start:stop]
                    )
                    chunk_loss_sum = (
                        per_route
                        * batch["route_weights"][start:stop].to(per_route.dtype)
                    ).sum()
                if not bool(torch.isfinite(chunk_loss_sum).item()):
                    raise FloatingPointError(
                        f"nonfinite duplicated-action BCE at epoch {epoch} batch {batch_index}"
                    )
                chunk_loss_sum.backward()
                batch_loss_sum_value += float(chunk_loss_sum.detach())
        accumulated_examples += batch_size
        total_examples += batch_size
        total_loss += batch_loss_sum_value
        should_step = (
            batch_index % accumulation_steps == 0 or batch_index == loader_length
        )
        if should_step:
            _scale_gradients(predictor, accumulated_examples)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_examples = 0
            global_step += 1
        if batch_index <= progress_first_batches or (
            progress_every_batches > 0 and batch_index % progress_every_batches == 0
        ):
            print(
                json.dumps(
                    {
                        "event": "four_action_polar_train_batch",
                        "epoch": epoch,
                        "objective": objective,
                        "batch": batch_index,
                        "batches": loader_length,
                        "examples_seen": total_examples,
                        "optimizer_step": should_step,
                        "global_step": global_step,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "mean_loss_so_far": total_loss / total_examples,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("frozen question encoder received gradients")
    mean = total_loss / total_examples
    return {
        "loss": mean,
        "objective": objective,
        "examples": total_examples,
        ("set_nll" if objective == "exact_set_nll" else "duplicated_action_bce"): mean,
        "optimizer_steps": global_step,
    }, global_step


@torch.inference_mode()
def validate_epoch(
    predictor,
    encoder,
    loader,
    *,
    device: torch.device,
    objective: str,
    top_k: int,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    predictor.eval()
    encoder.eval()
    accumulators = {"overall": FourActionMetricAccumulator(top_k=top_k)}
    route_type_accumulators: dict[str, FourActionMetricAccumulator] = {}
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        question = encoder(batch["input_ids"], batch["attention_mask"])
        with _amp_context(device, amp_dtype):
            logits = forward_from_features(predictor, question, batch).float()
        accumulators["overall"].update(
            logits,
            batch["valid_routes"],
            batch["valid_mask"],
            batch["route_weights"],
        )
        for benchmark in sorted(set(batch["benchmarks"])):
            indices = torch.tensor(
                [
                    index
                    for index, value in enumerate(batch["benchmarks"])
                    if value == benchmark
                ],
                dtype=torch.long,
                device=device,
            )
            accumulator = accumulators.setdefault(
                benchmark, FourActionMetricAccumulator(top_k=top_k)
            )
            accumulator.update(
                logits.index_select(0, indices),
                batch["valid_routes"].index_select(0, indices),
                batch["valid_mask"].index_select(0, indices),
                batch["route_weights"].index_select(0, indices),
            )
        for route_type in sorted(set(batch.get("route_types", [])) - {""}):
            indices = torch.tensor(
                [
                    index
                    for index, value in enumerate(batch["route_types"])
                    if value == route_type
                ],
                dtype=torch.long,
                device=device,
            )
            accumulator = route_type_accumulators.setdefault(
                route_type, FourActionMetricAccumulator(top_k=top_k)
            )
            accumulator.update(
                logits.index_select(0, indices),
                batch["valid_routes"].index_select(0, indices),
                batch["valid_mask"].index_select(0, indices),
                batch["route_weights"].index_select(0, indices),
            )
    return {
        "overall": accumulators["overall"].finalize(objective=objective),
        "by_benchmark": {
            benchmark: accumulator.finalize(objective=objective)
            for benchmark, accumulator in accumulators.items()
            if benchmark != "overall"
        },
        "by_route_type": {
            route_type: accumulator.finalize(objective=objective)
            for route_type, accumulator in route_type_accumulators.items()
        },
    }


def save_epoch_checkpoint(
    output_dir: Path,
    predictor,
    optimizer,
    scheduler,
    *,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    config_sha256: str,
    resolved_assets: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    epoch_dir = output_dir / f"epoch_{epoch:02d}"
    epoch_dir.mkdir(parents=False, exist_ok=False)
    checkpoint = epoch_dir / "checkpoint.pt"
    torch.save(
        {
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "config_sha256": config_sha256,
            "resolved_assets": resolved_assets,
            "metrics": metrics,
        },
        checkpoint,
    )
    digest = file_sha256(checkpoint)
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "metrics": metrics,
    }
    metadata_path = epoch_dir / "metadata.json"
    import json

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (epoch_dir / "checkpoint.sha256").write_text(
        f"{digest}  checkpoint.pt\n", encoding="utf-8"
    )
    return metadata
