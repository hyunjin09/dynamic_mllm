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
    boundary_teacher_route,
    choose_smoke_indices,
    load_jsonl,
    load_source_metadata,
    load_verified_manifest,
    mandatory_boundary_record,
    manifest_route_tensor,
    manifest_trie,
)
from four_action_online_router.metrics import (
    execution_checkpoint_key,
    mandatory_boundary_metrics,
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
    guaranteed_boundary_epoch_schedule,
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


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device=device)


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
        num_layers = int(config["router"]["num_layers"])
        routes = manifest_route_tensor(row, num_layers=num_layers)
        boundary = None
        if (
            config["data"].get("route_sampling")
            == "guaranteed_mandatory_boundary_once_plus_deterministic_valid_route"
            and row["route_type"] == "W2C"
        ):
            boundary = mandatory_boundary_record(row, num_layers=num_layers)
            boundary["teacher_route_index"] = min(boundary["boundary_route_indices"])
            route_index = int(boundary["teacher_route_index"])
            teacher_route = boundary_teacher_route(
                row, boundary, num_layers=num_layers
            )
        else:
            route_index = deterministic_route_index(
                uid=row["uid"], route_count=len(routes), seed=teacher_seed, epoch=epoch
            )
            teacher_route = routes[route_index]
        trajectory = replay_teacher_forced_states(
            wrapped_model, prepared, teacher_route, manifest_trie(
                row, num_layers=num_layers
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
        output_row = {
                "uid": row["uid"],
                "dataset": row["dataset"],
                "route_type": row["route_type"],
                "teacher_route_index": route_index,
                "node_logits": logits.float().cpu().tolist(),
                "valid_action_mask": trajectory.valid_action_mask.cpu().tolist(),
                **execution,
        }
        if boundary is not None:
            layer = int(boundary["boundary_layer"])
            output_row.update(
                {
                    "boundary_layer": layer,
                    "valid_nonfull_actions": boundary["valid_nonfull_actions"],
                    "singleton": boundary["singleton"],
                    "predicted_boundary_action": decode_action_indices(
                        [int(logits[layer].argmax().item())]
                    )[0],
                }
            )
        local_rows.append(output_row)
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
    boundary_rows = [row for row in combined if "boundary_layer" in row]
    if boundary_rows:
        metrics["boundary"] = mandatory_boundary_metrics(
            boundary_rows, num_layers=int(config["router"]["num_layers"])
        )
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


def build_training_optimizer_and_scheduler(parameters, training):
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training["warmup_steps"]),
        num_training_steps=int(training["total_optimizer_steps"]),
    )
    return optimizer, scheduler


def render_boundary_coverage_report(
    *, config_path: Path, output_dir: Path, history: list[dict[str, Any]]
) -> str:
    best = max(history, key=execution_checkpoint_key)
    execution = best["execution"]
    boundary = best.get("boundary") or {}
    return "\n".join(
        [
            "# Online Four-Action Guaranteed-Boundary Training",
            "",
            f"- Config: `{config_path}`",
            f"- Config SHA-256: `{file_sha256(config_path)}`",
            f"- Output: `{output_dir}`",
            f"- Epochs completed: {len(history)}",
            f"- Selected epoch: {best['epoch']}",
            "",
            "## Primary routed behavior",
            "",
            f"- W2C free-rollout rescue: {execution['w2c_rescue_rate']:.6f}",
            f"- C2C preservation: {execution['c2c_preservation_rate']:.6f}",
            f"- Overall routed accuracy: {execution['overall_routed_accuracy']:.6f}",
            f"- Mean FULL layers: {execution['mean_action_layers']['FULL']:.6f}",
            "",
            "## Mandatory-boundary behavior",
            "",
            f"- Validation boundary Valid-Action@1: {boundary.get('valid_action_at_1')}",
            f"- Validation boundary non-FULL recall: {boundary.get('nonfull_recall')}",
            f"- Validation free rollout left all-FULL: {(boundary.get('free_rollout') or {}).get('left_all_full_fraction')}",
            "",
            "Every training W2C sample received exactly one scheduled visit to its",
            "latest all-FULL-prefix mandatory-deviation boundary. All remaining",
            "visits retained the original deterministic valid-route sampler.",
            "",
        ]
    )


def verify_boundary_coverage_contract(
    config: dict[str, Any], config_path: Path
) -> None:
    """Fail closed unless A2 differs from its parent only in route exposure."""

    if config.get("protocol_version") != "four_action_online_boundary_coverage_v2":
        return
    plan = Path(config["source_plan"])
    parent_path = Path(config["matched_parent_config"])
    if file_sha256(plan) != config["source_plan_sha256"]:
        raise RuntimeError("collapse plan checksum mismatch")
    if file_sha256(parent_path) != config["matched_parent_config_sha256"]:
        raise RuntimeError("matched parent config checksum mismatch")
    summary_path = Path(config["a1_gate"]["summary"])
    if file_sha256(summary_path) != config["a1_gate"]["summary_sha256"]:
        raise RuntimeError("A1 gate summary checksum mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("passed") is not True or summary.get("full_online_a2_authorized") is not True:
        raise RuntimeError("A1 did not authorize full online A2")
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    for section in ("base_model", "executor", "router", "training"):
        if config[section] != parent[section]:
            raise RuntimeError(f"A2 matched parent section differs: {section}")
    for key in (
        "manifest", "manifest_sha256", "manifest_audit", "manifest_audit_sha256",
        "source_manifest", "source_manifest_sha256", "records", "train_records",
        "validation_records", "valid_routes", "unique_image_groups",
        "zero_valid_route_exclusions", "supervision", "sample_balance",
    ):
        if config["data"][key] != parent["data"][key]:
            raise RuntimeError(f"A2 matched parent data field differs: {key}")
    configured_output = Path(config["reporting"]["output_dir"])
    if not configured_output.is_absolute():
        configured_output = Path.cwd() / configured_output
    if not config_path.is_file() or not configured_output.parent.exists():
        raise RuntimeError("A2 config/output contract is incomplete")


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

    optimizer, scheduler = build_training_optimizer_and_scheduler(
        router.parameters(), config["training"]
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
        scheduler.step()
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
    coverage_enabled = (
        config["data"].get("route_sampling")
        == "guaranteed_mandatory_boundary_once_plus_deterministic_valid_route"
    )
    boundary_by_uid: dict[str, dict[str, Any]] = {}
    epoch_schedule: list[list[dict[str, Any]]] | None = None
    if coverage_enabled:
        boundary_path = Path(config["data"]["boundary_manifest"])
        if file_sha256(boundary_path) != config["data"]["boundary_manifest_sha256"]:
            raise RuntimeError("mandatory-boundary manifest checksum mismatch")
        boundary_rows = load_jsonl(boundary_path)
        boundary_by_uid = {str(row["uid"]): row for row in boundary_rows}
        expected_w2c = {
            str(row["uid"]) for row in train_rows if row["route_type"] == "W2C"
        }
        if set(boundary_by_uid) != expected_w2c:
            raise RuntimeError("mandatory-boundary manifest does not match train W2C UIDs")
        epoch_schedule = guaranteed_boundary_epoch_schedule(
            train_rows,
            samples_per_epoch=int(training["balanced_samples_per_epoch"]),
            seed=int(training["seed"]),
            epochs=int(training["epochs"]),
            world_size=world_size,
        )
        exposure_count = sum(
            bool(visit["mandatory_boundary"])
            for visits in epoch_schedule for visit in visits
        )
        if exposure_count != len(expected_w2c):
            raise RuntimeError("mandatory-boundary schedule exposure count mismatch")
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
    optimizer, scheduler = build_training_optimizer_and_scheduler(
        router.parameters(), training
    )
    total_steps = int(training["total_optimizer_steps"])
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
        states = payload.get("rng_states")
        if isinstance(states, list) and len(states) == world_size:
            restore_rng_state(states[rank], device)
        elif coverage_enabled:
            raise RuntimeError("resume checkpoint lacks complete per-rank RNG states")
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
                "route_sampling": config["data"]["route_sampling"],
                "mandatory_boundary_manifest_sha256": config["data"].get(
                    "boundary_manifest_sha256"
                ),
                "mandatory_boundary_exposures": (
                    len(boundary_by_uid) if coverage_enabled else 0
                ),
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
        if epoch_schedule is None:
            visits = [
                {"row_index": index, "mandatory_boundary": False}
                for index in balanced_epoch_indices(
                    train_rows,
                    samples_per_epoch=int(training["balanced_samples_per_epoch"]),
                    seed=int(training["seed"]), epoch=epoch, world_size=world_size,
                )
            ][rank::world_size]
        else:
            visits = epoch_schedule[epoch - 1][rank::world_size]
        if len(visits) % accumulation:
            raise RuntimeError("per-rank epoch samples must divide gradient accumulation")
        router.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        started = time.time()
        local_boundary_exposures = 0
        local_ordinary_w2c_visits = 0
        for microstep, visit in enumerate(visits, start=1):
            row_index = int(visit["row_index"])
            row = train_rows[row_index]
            _sample, _inputs, _metadata, prepared = prepare_sample(
                processor, wrapped_model, row, sources[row["uid"]], device
            )
            routes = manifest_route_tensor(row, num_layers=int(config["router"]["num_layers"]))
            if bool(visit["mandatory_boundary"]):
                boundary = dict(boundary_by_uid[row["uid"]])
                boundary["teacher_route_index"] = min(
                    int(value) for value in boundary["boundary_route_indices"]
                )
                route_index = int(boundary["teacher_route_index"])
                teacher_route = boundary_teacher_route(
                    row, boundary, num_layers=int(config["router"]["num_layers"])
                )
                route_source = "mandatory_boundary"
                local_boundary_exposures += 1
            else:
                route_index = deterministic_route_index(
                    uid=row["uid"], route_count=len(routes),
                    seed=int(training["seed"]), epoch=epoch
                )
                teacher_route = routes[route_index]
                route_source = "ordinary_valid_route"
                local_ordinary_w2c_visits += row["route_type"] == "W2C"
            trajectory = replay_teacher_forced_states(
                wrapped_model, prepared, teacher_route,
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
                            "route_source": route_source,
                            "loss": float(loss.detach().item()),
                        }, sort_keys=True
                    ), flush=True
                )
        totals = torch.tensor(
            [loss_sum, len(visits), local_boundary_exposures, local_ordinary_w2c_visits],
            dtype=torch.float64, device=device
        )
        dist.all_reduce(totals)
        metrics, validation_outputs = validate(
            epoch=epoch, rows=validation_rows, sources=sources, processor=processor,
            wrapped_model=wrapped_model, router=router, config=config,
            rank=rank, world_size=world_size
        )
        rng_states: list[Any] = [None] * world_size
        dist.all_gather_object(rng_states, capture_rng_state(device))
        if rank == 0:
            epoch_row = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
                "train": {
                    "samples": int(totals[1].item()),
                    "mean_loss": float(totals[0].item() / totals[1].item()),
                    "mandatory_boundary_exposures": int(totals[2].item()),
                    "ordinary_w2c_visits": int(totals[3].item()),
                },
                **metrics,
            }
            history.append(epoch_row)
            checkpoint = save_epoch_artifacts(
                output_dir, epoch,
                checkpoint_payload(
                    router, optimizer, scheduler, epoch=epoch, global_step=global_step,
                    config_sha256=file_sha256(config_path), manifest_sha256=config["data"]["manifest_sha256"],
                    metrics=epoch_row, rng_states=rng_states,
                ),
                validation_outputs,
            )
            checkpoints.append(checkpoint)
            write_json(history_path, history)
            if config.get("reporting", {}).get("history"):
                write_jsonl(Path(config["reporting"]["history"]), history)
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
        if config.get("reporting", {}).get("report"):
            report_path = Path(config["reporting"]["report"])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                render_boundary_coverage_report(
                    config_path=config_path, output_dir=output_dir, history=history
                ),
                encoding="utf-8",
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
        verify_boundary_coverage_contract(config, config_path)
        if (
            args.mode == "train"
            and config.get("protocol_version") == "four_action_online_boundary_coverage_v2"
        ):
            if Path(args.output_dir).resolve() != Path(
                config["reporting"]["output_dir"]
            ).resolve():
                raise RuntimeError("A2 must use its canonical output directory")
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
