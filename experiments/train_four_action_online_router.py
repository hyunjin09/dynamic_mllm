#!/usr/bin/env python3
"""Smoke-test and train the state-conditioned four-action router with DDP."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import get_cosine_schedule_with_warmup
import yaml

from binary_policy.executor import (
    capture_four_action_route,
    greedy_generate_from_cached_prompt,
)
from binary_policy.executor.inputs import build_binary_inputs
from experiments.train_binary_polar import file_sha256
from four_action_online_router.data import (
    choose_smoke_indices,
    load_source_metadata,
    load_verified_manifest,
    manifest_route_tensor,
    manifest_trie,
)
from four_action_online_router.metrics import (
    execution_checkpoint_key,
    summarize_execution_rows,
    summarize_node_predictions,
)
from four_action_online_router.model import OnlineFourActionRouter
from four_action_online_router.runtime import (
    capture_online_router_route,
    replay_teacher_forced_states,
    router_logits_for_trajectory,
    select_last_text_state,
)
from four_action_online_router.supervision import (
    balanced_epoch_indices,
    deterministic_route_index,
    set_valued_action_loss,
)
from four_action_policy.actions import decode_action_indices
from four_action_policy.external import action_statistics
from label_regeneration.runtime import (
    build_native_processor_inputs,
    configure_determinism,
    load_frozen_model,
    score_prediction_with_timeout,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distributed_context() -> tuple[int, int, int, torch.device]:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if any(name not in os.environ for name in required):
        raise RuntimeError("launch online-router training with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available() or world_size != 8:
        raise RuntimeError("the frozen online-router contract requires exactly eight GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def make_router(config: dict[str, Any], device: torch.device) -> OnlineFourActionRouter:
    values = config["router"]
    return OnlineFourActionRouter(
        hidden_size=int(values["hidden_size"]),
        num_layers=int(values["num_layers"]),
        d_router=int(values["d_router"]),
        num_heads=int(values["num_heads"]),
        mlp_hidden_size=int(values["mlp_hidden_size"]),
        dropout=float(values["dropout"]),
        interaction_scale=float(values["interaction_scale"]),
    ).to(device)


def sample_with_evaluator(
    row: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    return {
        **row,
        **source,
        "local_image_path": row["image_path"],
    }


def prepare_sample(processor, wrapped_model, row, source, device):
    sample = sample_with_evaluator(row, source)
    inputs, input_metadata = build_native_processor_inputs(processor, sample, device)
    return sample, inputs, input_metadata, build_binary_inputs(wrapped_model, inputs)


def decode_generated(processor, token_ids: torch.Tensor) -> str:
    return processor.batch_decode(
        token_ids.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


@torch.inference_mode()
def execute_router(
    *, processor, wrapped_model, router, inputs, prepared, sample, config
) -> dict[str, Any]:
    captured = capture_online_router_route(
        wrapped_model,
        inputs,
        router,
        prepared_inputs=prepared,
        amp_dtype=torch.bfloat16,
        use_cache=True,
    )
    if captured.cache is None:
        raise RuntimeError("online-router prompt capture did not create a cache")
    generation = greedy_generate_from_cached_prompt(
        wrapped_model,
        captured.prompt_logits,
        captured.inputs,
        captured.cache,
        inputs["input_ids"],
        max_new_tokens=int(sample["max_new_tokens"]),
        eos_token_ids=list(config["external_evaluation"]["eos_token_ids"]),
        repetition_penalty=float(config["external_evaluation"]["repetition_penalty"]),
    )
    prediction = decode_generated(processor, generation.generated_ids)
    score, timed_out = score_prediction_with_timeout(
        sample["metric_name"],
        prediction,
        sample["answer"],
        sample.get("all_answer_norms"),
        timeout_seconds=5.0,
    )
    actions = tuple(captured.layer_actions or ())
    if len(actions) != int(config["router"]["num_layers"]):
        raise RuntimeError("online router did not select exactly one action per layer")
    for action, stats in zip(actions, captured.layer_stats):
        if stats.read_on != (action in {"READ_ONLY", "FULL"}) or stats.write_on != (
            action in {"WRITE_ONLY", "FULL"}
        ):
            raise RuntimeError("executor READ/WRITE semantics disagree with selected action")
    return {
        "actions": list(actions),
        **action_statistics(actions),
        "prediction": prediction,
        "generated_ids": generation.generated_ids.detach().cpu().view(-1).tolist(),
        "score": float(score),
        "correct": bool(score >= float(sample["correctness_threshold"])),
        "scoring_timed_out": timed_out,
    }


def gather_flat(local_rows: list[dict[str, Any]], world_size: int) -> list[dict[str, Any]]:
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local_rows)
    return [row for part in gathered for row in part]


def validate(
    *,
    epoch: int,
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    processor,
    wrapped_model,
    router,
    config: dict[str, Any],
    rank: int,
    world_size: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    router.eval()
    local_rows = []
    teacher_seed = int(config["validation"]["node_metrics_teacher_route_seed"])
    for position in range(rank, len(rows), world_size):
        row = rows[position]
        sample, inputs, _input_metadata, prepared = prepare_sample(
            processor, wrapped_model, row, sources[row["uid"]], next(router.parameters()).device
        )
        routes = manifest_route_tensor(row, num_layers=int(config["router"]["num_layers"]))
        route_index = deterministic_route_index(
            uid=row["uid"], route_count=len(routes), seed=teacher_seed, epoch=epoch
        )
        trajectory = replay_teacher_forced_states(
            wrapped_model, prepared, routes[route_index], manifest_trie(
                row, num_layers=int(config["router"]["num_layers"])
            )
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = router_logits_for_trajectory(router, trajectory)
        execution = execute_router(
            processor=processor,
            wrapped_model=wrapped_model,
            router=router,
            inputs=inputs,
            prepared=prepared,
            sample=sample,
            config=config,
        )
        local_rows.append(
            {
                "uid": row["uid"],
                "dataset": row["dataset"],
                "route_type": row["route_type"],
                "teacher_route_index": route_index,
                "node_logits": logits.float().cpu().tolist(),
                "valid_action_mask": trajectory.valid_action_mask.cpu().tolist(),
                **execution,
            }
        )
        if len(local_rows) <= 3:
            print(
                json.dumps(
                    {
                        "event": "online_router_validation_sample",
                        "rank": rank,
                        "epoch": epoch,
                        "uid": row["uid"],
                        "correct": execution["correct"],
                        "route_key": execution["route_key"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    combined = gather_flat(local_rows, world_size)
    if rank != 0:
        return None, None
    if len(combined) != int(config["validation"]["expected_records"]):
        raise RuntimeError("distributed validation did not cover the frozen population")
    combined.sort(key=lambda row: row["uid"])
    logits = torch.tensor(
        [node for row in combined for node in row.pop("node_logits")], dtype=torch.float32
    )
    masks = torch.tensor(
        [node for row in combined for node in row.pop("valid_action_mask")], dtype=torch.bool
    )
    metrics = {
        "node": summarize_node_predictions(logits, masks),
        "execution": summarize_execution_rows(combined),
    }
    return metrics, combined


def checkpoint_payload(router, optimizer, scheduler, **metadata) -> dict[str, Any]:
    core = router.module if isinstance(router, DistributedDataParallel) else router
    return {
        "router": {key: value.detach().cpu() for key, value in core.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        **metadata,
    }


def save_epoch_artifacts(
    output_dir: Path,
    epoch: int,
    payload: dict[str, Any],
    validation_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    epoch_dir = output_dir / f"epoch_{epoch:02d}"
    if epoch_dir.exists():
        raise FileExistsError(f"completed epoch directory already exists: {epoch_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".epoch_{epoch:02d}_", dir=output_dir))
    temporary_checkpoint = temporary / "router_checkpoint.pt"
    torch.save(payload, temporary_checkpoint)
    write_jsonl(temporary / "validation_outputs.jsonl", validation_outputs)
    final_checkpoint = epoch_dir / "router_checkpoint.pt"
    metadata = {
        "epoch": epoch,
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": file_sha256(temporary_checkpoint),
    }
    write_json(temporary / "metadata.json", metadata)
    temporary.rename(epoch_dir)
    return metadata


def prepare_smoke_output_dir(output_dir: Path, *, rank: int) -> None:
    """Create a fresh shared smoke directory without a cross-rank check race."""
    if rank == 0:
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite smoke output: {output_dir}")
        output_dir.mkdir(parents=True)
    dist.barrier()


def run_smoke(
    *, config, config_path, output_dir, rows, sources, processor, wrapped_model,
    router, rank, world_size, device
) -> None:
    started = time.time()
    prepare_smoke_output_dir(output_dir, rank=rank)
    selected = choose_smoke_indices(
        rows, records=int(config["smoke"]["records"]), seed=int(config["training"]["seed"])
    )
    row = rows[selected[rank]]
    sample, inputs, input_metadata, prepared = prepare_sample(
        processor, wrapped_model, row, sources[row["uid"]], device
    )
    routes = manifest_route_tensor(row, num_layers=int(config["router"]["num_layers"]))
    chosen = deterministic_route_index(
        uid=row["uid"], route_count=len(routes), seed=int(config["training"]["seed"]), epoch=0
    )
    trie = manifest_trie(row, num_layers=int(config["router"]["num_layers"]))
    trajectory = replay_teacher_forced_states(wrapped_model, prepared, routes[chosen], trie)
    multi_valid = bool((trajectory.valid_action_mask.sum(dim=-1) > 1).any().item())

    full = capture_four_action_route(
        wrapped_model,
        inputs,
        ("FULL",) * int(config["router"]["num_layers"]),
        prepared_inputs=prepared,
        use_cache=False,
    )
    routed_differs = False
    for layer, (text, visual) in enumerate(full.pre_layer_states):
        full_query = select_last_text_state(text, prepared.text_valid_mask.to(text.device))
        if not torch.equal(full_query, trajectory.text_queries[layer : layer + 1]) or not torch.equal(
            visual, trajectory.visual_states[layer : layer + 1]
        ):
            routed_differs = True
            break

    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    router.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        initial_loss = set_valued_action_loss(
            router_logits_for_trajectory(router, trajectory), trajectory.valid_action_mask
        )
    initial_mean = initial_loss.detach().clone()
    dist.all_reduce(initial_mean)
    initial_mean /= world_size
    gradient_checks = []
    router.train()
    for _ in range(int(config["smoke"]["optimization_repeats"])):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = set_valued_action_loss(
                router_logits_for_trajectory(router, trajectory), trajectory.valid_action_mask
            )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite smoke loss")
        loss.backward()
        core = router.module
        gradient_checks.append(
            (
                float(core.read_layer_queries.weight.grad.abs().sum().item()),
                float(core.write_layer_queries.weight.grad.abs().sum().item()),
            )
        )
        torch.nn.utils.clip_grad_norm_(router.parameters(), float(config["training"]["gradient_clip_norm"]))
        optimizer.step()
    router.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        final_loss = set_valued_action_loss(
            router_logits_for_trajectory(router, trajectory), trajectory.valid_action_mask
        )
    final_mean = final_loss.detach().clone()
    dist.all_reduce(final_mean)
    final_mean /= world_size

    execution = execute_router(
        processor=processor, wrapped_model=wrapped_model, router=router,
        inputs=inputs, prepared=prepared, sample=sample, config=config
    )
    text_mask = prepared.text_valid_mask[0]
    compact_index = int(text_mask.long().sum().item()) - 1
    full_index = int(prepared.text_indices[0, compact_index].item())
    token_id = int(inputs["input_ids"][0, full_index].item())
    local = {
        "rank": rank,
        "uid": row["uid"],
        "dataset": row["dataset"],
        "route_type": row["route_type"],
        "teacher_route_index": chosen,
        "teacher_route_sampling_repeat": deterministic_route_index(
            uid=row["uid"], route_count=len(routes), seed=int(config["training"]["seed"]), epoch=0
        ),
        "multi_valid_state": multi_valid,
        "routed_prefix_differs_from_all_full": routed_differs,
        "initial_loss": float(initial_loss.item()),
        "final_loss": float(final_loss.item()),
        "read_query_gradient_sums": [item[0] for item in gradient_checks],
        "write_query_gradient_sums": [item[1] for item in gradient_checks],
        "backbone_trainable_parameters": sum(p.requires_grad for p in wrapped_model.parameters()),
        "backbone_parameters_with_grad": sum(p.grad is not None for p in wrapped_model.parameters()),
        "query_compact_text_index": compact_index,
        "query_full_token_index": full_index,
        "query_token_id": token_id,
        "query_token_text": processor.decode([token_id], clean_up_tokenization_spaces=False),
        "input_metadata": input_metadata,
        "execution": execution,
    }
    records = gather_flat([local], world_size)
    flags = torch.tensor([multi_valid, routed_differs], dtype=torch.int32, device=device)
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    if rank == 0:
        core = router.module
        smoke_checkpoint = output_dir / "router_smoke_checkpoint.pt"
        torch.save({"router": {k: v.detach().cpu() for k, v in core.state_dict().items()}}, smoke_checkpoint)
        roundtrip = make_router(config, torch.device("cpu"))
        roundtrip.load_state_dict(torch.load(smoke_checkpoint, map_location="cpu", weights_only=False)["router"], strict=True)
        roundtrip_equal = all(
            torch.equal(left.cpu(), right)
            for left, right in zip(core.state_dict().values(), roundtrip.state_dict().values())
        )
        passed = (
            bool(flags[0].item())
            and bool(flags[1].item())
            and float(final_mean.item()) < float(initial_mean.item())
            and all(item["teacher_route_index"] == item["teacher_route_sampling_repeat"] for item in records)
            and all(max(item["read_query_gradient_sums"]) > 0 for item in records)
            and all(max(item["write_query_gradient_sums"]) > 0 for item in records)
            and all(item["backbone_trainable_parameters"] == 0 for item in records)
            and all(item["backbone_parameters_with_grad"] == 0 for item in records)
            and roundtrip_equal
        )
        payload = {
            "schema_version": "four_action_online_router_smoke_v1",
            "passed": passed,
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "records": records,
            "global_initial_mean_loss": float(initial_mean.item()),
            "global_final_mean_loss": float(final_mean.item()),
            "multi_valid_supervision_exercised": bool(flags[0].item()),
            "routed_state_conditioning_exercised": bool(flags[1].item()),
            "checkpoint_roundtrip_exact": roundtrip_equal,
            "checkpoint_sha256": file_sha256(smoke_checkpoint),
            "measured_smoke_body_seconds": time.time() - started,
            "qwen_route_equivalents_per_rank": 3,
        }
        write_json(output_dir / "smoke_report.json", payload)
        if not passed:
            raise RuntimeError("online-router GPU smoke failed")
    dist.barrier()


def run_training(
    *, config, config_path, output_dir, train_rows, validation_rows, sources,
    processor, wrapped_model, router, rank, world_size, device, resume
) -> None:
    training = config["training"]
    restart_before_first_epoch = output_dir.exists() and not resume
    if restart_before_first_epoch:
        entries = list(output_dir.iterdir())
        unexpected = [
            path
            for path in entries
            if path.name != "initialization.json" and not path.name.startswith(".epoch_")
        ]
        if unexpected:
            raise FileExistsError(
                f"refusing to overwrite nonempty training output: {unexpected[0]}"
            )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"])
    )
    total_steps = int(training["total_optimizer_steps"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(training["warmup_steps"]),
        num_training_steps=total_steps
    )
    history_path = output_dir / "history.json"
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    start_epoch = 1
    global_step = 0
    if resume:
        if (output_dir / "training_summary.json").is_file():
            raise FileExistsError("training is already complete")
        metadata_paths = sorted(output_dir.glob("epoch_[0-9][0-9]/metadata.json"))
        completed_epochs = [int(path.parent.name.removeprefix("epoch_")) for path in metadata_paths]
        if completed_epochs != list(range(1, len(completed_epochs) + 1)):
            raise RuntimeError("completed resume checkpoints are non-contiguous")
        payload = None
        for epoch, metadata_path in zip(completed_epochs, metadata_paths):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            checkpoint_path = Path(metadata["checkpoint"])
            if file_sha256(checkpoint_path) != metadata["checkpoint_sha256"]:
                raise RuntimeError(f"resume checkpoint checksum mismatch at epoch {epoch}")
            current = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if current["config_sha256"] != file_sha256(config_path):
                raise RuntimeError("resume config checksum mismatch")
            history.append(current["metrics"])
            checkpoints.append(metadata)
            payload = current
        if payload is None:
            raise RuntimeError("resume requires at least one atomically completed epoch")
        router.module.load_state_dict(payload["router"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        global_step = int(payload["global_step"])
        start_epoch = len(history) + 1
        if rank == 0:
            write_json(history_path, history)
    elif rank == 0:
        initialization = {
                "schema_version": "four_action_online_router_initialization_v1",
                "config": str(config_path),
                "config_sha256": file_sha256(config_path),
                "manifest_sha256": config["data"]["manifest_sha256"],
                "world_size": world_size,
                "train_records": len(train_rows),
                "validation_records": len(validation_rows),
                "balanced_samples_per_epoch": int(training["balanced_samples_per_epoch"]),
                "optimizer_steps_per_epoch": int(training["optimizer_steps_per_epoch"]),
                "total_optimizer_steps": total_steps,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }
        initialization_path = output_dir / "initialization.json"
        if initialization_path.is_file():
            previous = json.loads(initialization_path.read_text(encoding="utf-8"))
            if previous != initialization:
                raise RuntimeError("pre-epoch restart initialization contract mismatch")
        else:
            write_json(initialization_path, initialization)
    dist.barrier()

    accumulation = int(training["gradient_accumulation_steps"])
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        indices = balanced_epoch_indices(
            train_rows,
            samples_per_epoch=int(training["balanced_samples_per_epoch"]),
            seed=int(training["seed"]), epoch=epoch, world_size=world_size,
        )[rank::world_size]
        if len(indices) % accumulation:
            raise RuntimeError("per-rank epoch samples must divide gradient accumulation")
        router.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        started = time.time()
        for microstep, row_index in enumerate(indices, start=1):
            row = train_rows[row_index]
            _sample, _inputs, _metadata, prepared = prepare_sample(
                processor, wrapped_model, row, sources[row["uid"]], device
            )
            routes = manifest_route_tensor(row, num_layers=int(config["router"]["num_layers"]))
            route_index = deterministic_route_index(
                uid=row["uid"], route_count=len(routes), seed=int(training["seed"]), epoch=epoch
            )
            trajectory = replay_teacher_forced_states(
                wrapped_model, prepared, routes[route_index],
                manifest_trie(row, num_layers=int(config["router"]["num_layers"]))
            )
            synchronize = microstep % accumulation == 0
            sync_context = nullcontext() if synchronize else router.no_sync()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = set_valued_action_loss(
                        router_logits_for_trajectory(router, trajectory),
                        trajectory.valid_action_mask,
                    )
                (loss / accumulation).backward()
            loss_sum += float(loss.detach().item())
            if synchronize:
                torch.nn.utils.clip_grad_norm_(router.parameters(), float(training["gradient_clip_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if microstep <= 3 or microstep % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "online_router_train_sample", "rank": rank,
                            "epoch": epoch, "microstep": microstep, "global_step": global_step,
                            "uid": row["uid"], "route_index": route_index,
                            "loss": float(loss.detach().item()),
                        }, sort_keys=True
                    ), flush=True
                )
        totals = torch.tensor([loss_sum, len(indices)], dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        metrics, validation_outputs = validate(
            epoch=epoch, rows=validation_rows, sources=sources, processor=processor,
            wrapped_model=wrapped_model, router=router, config=config,
            rank=rank, world_size=world_size
        )
        if rank == 0:
            epoch_row = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
                "train": {"samples": int(totals[1].item()), "mean_loss": float(totals[0].item() / totals[1].item())},
                **metrics,
            }
            history.append(epoch_row)
            checkpoint = save_epoch_artifacts(
                output_dir, epoch,
                checkpoint_payload(
                    router, optimizer, scheduler, epoch=epoch, global_step=global_step,
                    config_sha256=file_sha256(config_path), manifest_sha256=config["data"]["manifest_sha256"],
                    metrics=epoch_row,
                ),
                validation_outputs,
            )
            checkpoints.append(checkpoint)
            write_json(history_path, history)
            print(json.dumps({"event": "online_router_epoch_complete", **epoch_row}, sort_keys=True), flush=True)
        dist.barrier()

    if global_step != total_steps:
        raise RuntimeError(f"optimizer-step mismatch: {global_step} != {total_steps}")
    if rank == 0:
        best = max(history, key=execution_checkpoint_key)
        checkpoint = checkpoints[int(best["epoch"]) - 1]
        selection = {
            "schema_version": "four_action_online_router_checkpoint_selection_v1",
            "selected_before_external_evaluation": True,
            "config": str(config_path), "config_sha256": file_sha256(config_path),
            "best_epoch": int(best["epoch"]),
            "best_checkpoint": checkpoint["checkpoint"],
            "best_checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "checkpoint_order": config["validation"]["tie_break"],
            "execution": best["execution"], "node": best["node"],
        }
        write_json(output_dir / "best_checkpoint.json", selection)
        write_json(
            output_dir / "training_summary.json",
            {
                "schema_version": "four_action_online_router_training_v1",
                "passed": True, "epochs_completed": len(history),
                "global_steps": global_step, "best_epoch": int(best["epoch"]),
                "best_checkpoint": checkpoint["checkpoint"],
                "external_evaluation_started": False,
            },
        )
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--smoke-report")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args()
    if args.mode == "train" and not args.confirm_training:
        raise RuntimeError("main training requires --confirm-training")
    if args.mode == "train" and not args.smoke_report:
        raise RuntimeError("main training requires its passed GPU smoke report")

    rank, world_size, local_rank, device = distributed_context()
    try:
        config_path = Path(args.config)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if args.mode == "train":
            smoke = json.loads(Path(args.smoke_report).read_text(encoding="utf-8"))
            if (
                smoke.get("passed") is not True
                or smoke.get("config_sha256") != file_sha256(config_path)
            ):
                raise RuntimeError("training smoke report is failed or belongs to another config")
        if int(config["training"]["world_size"]) != world_size:
            raise RuntimeError("runtime world size differs from frozen config")
        configure_determinism(int(config["training"]["seed"]))
        rows = load_verified_manifest(config["data"]["manifest"], config["data"]["manifest_sha256"])
        if len(rows) != int(config["data"]["records"]):
            raise RuntimeError("manifest record count differs from frozen config")
        sources = load_source_metadata(
            config["data"]["source_manifest"], config["data"]["source_manifest_sha256"],
            {str(row["uid"]) for row in rows}
        )
        train_rows = [row for row in rows if row["split"] == "train"]
        validation_rows = [row for row in rows if row["split"] == "validation"]
        processor, base_model, wrapped_model, _ = load_frozen_model(
            config["base_model"]["path"], config["base_model"]["revision"], local_rank
        )
        base_model.requires_grad_(False).eval()
        seed_everything(int(config["training"]["seed"]))
        core = make_router(config, device)
        router = DistributedDataParallel(
            core, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, gradient_as_bucket_view=True
        )
        if args.mode == "smoke":
            run_smoke(
                config=config, config_path=config_path, output_dir=Path(args.output_dir),
                rows=train_rows, sources=sources, processor=processor,
                wrapped_model=wrapped_model, router=router, rank=rank,
                world_size=world_size, device=device
            )
        else:
            run_training(
                config=config, config_path=config_path, output_dir=Path(args.output_dir),
                train_rows=train_rows, validation_rows=validation_rows, sources=sources,
                processor=processor, wrapped_model=wrapped_model, router=router,
                rank=rank, world_size=world_size, device=device, resume=args.resume
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
