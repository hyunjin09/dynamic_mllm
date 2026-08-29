#!/usr/bin/env python3
"""Overfit the unchanged online router on exact mandatory-boundary states."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml

from experiments.train_binary_polar import file_sha256
from experiments.train_four_action_online_router import (
    build_training_optimizer_and_scheduler,
    checkpoint_payload,
    execute_router,
    gather_flat,
    make_router,
    prepare_sample,
    save_epoch_artifacts,
    seed_everything,
    write_json,
    write_jsonl,
)
from four_action_online_router.data import (
    boundary_teacher_route,
    load_jsonl,
    load_source_metadata,
    load_verified_manifest,
    manifest_route_tensor,
    manifest_trie,
)
from four_action_online_router.metrics import (
    mandatory_boundary_metrics,
    mandatory_boundary_pilot_gate,
    summarize_execution_rows,
)
from four_action_online_router.runtime import (
    replay_teacher_forced_states,
    router_logits_for_trajectory,
)
from four_action_online_router.supervision import (
    deterministic_route_index,
    set_valued_action_loss,
)
from four_action_policy.actions import FOUR_ACTIONS
from label_regeneration.runtime import configure_determinism, load_frozen_model


MATCHED_PARENT_FIELDS = (
    ("base_model",),
    ("executor",),
    ("data", "manifest"),
    ("data", "manifest_sha256"),
    ("data", "source_manifest"),
    ("data", "source_manifest_sha256"),
    ("data", "supervision"),
    ("router",),
    ("training", "optimizer"),
    ("training", "learning_rate"),
    ("training", "weight_decay"),
    ("training", "scheduler"),
    ("training", "warmup_steps"),
    ("training", "gradient_clip_norm"),
    ("training", "precision"),
    ("training", "deterministic_algorithms"),
)


def _nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        current = current[key]
    return current


def verify_parent_contract(config: dict[str, Any], parent: dict[str, Any]) -> None:
    for path in MATCHED_PARENT_FIELDS:
        if _nested(config, path) != _nested(parent, path):
            raise RuntimeError(f"matched parent field differs: {'.'.join(path)}")
    if config["data"]["c2c_route_sampling"] != parent["data"]["route_sampling"]:
        raise RuntimeError("matched parent field differs: data.c2c_route_sampling")


def require_slurm_environment(environment: Mapping[str, str]) -> None:
    if not str(environment.get("SLURM_JOB_ID") or ""):
        raise RuntimeError("GPU training requires a live Slurm allocation")


def verify_output_contract(requested: Path, configured: Path) -> None:
    if requested.resolve() != configured.resolve():
        raise RuntimeError("boundary pilot must use its canonical output directory")


def acquire_run_lock(output_dir: Path, *, resume: bool):
    if resume:
        if not output_dir.is_dir():
            raise RuntimeError("pilot resume requires its existing output directory")
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_dir.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite pilot output: {output_dir}") from exc
    handle = (output_dir / ".run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError("another boundary-pilot run is already active") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"slurm_job_id={os.environ.get('SLURM_JOB_ID', 'test')}\n")
    handle.flush()
    return handle


def release_run_lock(handle) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def latest_resume_metadata(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("epoch_[0-9][0-9]/metadata.json"))
    if not checkpoints:
        return None
    epochs = [int(path.parent.name.removeprefix("epoch_")) for path in checkpoints]
    interval = epochs[0]
    if interval < 1 or epochs != list(range(interval, epochs[-1] + 1, interval)):
        raise RuntimeError("pilot resume checkpoints are non-contiguous")
    return checkpoints[-1]


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


def load_resume_payload(
    metadata_path: Path, *, config_sha256: str, world_size: int
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = Path(metadata["checkpoint"])
    if file_sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise RuntimeError("pilot resume checkpoint checksum mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != config_sha256:
        raise RuntimeError("pilot resume config checksum mismatch")
    states = payload.get("rng_states")
    if not isinstance(states, list) or len(states) != world_size:
        raise RuntimeError("pilot resume RNG-state population mismatch")
    if int(payload.get("epoch", -1)) != int(metadata_path.parent.name.removeprefix("epoch_")):
        raise RuntimeError("pilot resume epoch metadata mismatch")
    return payload


def distributed_context(expected_world_size: int) -> tuple[int, int, int, torch.device]:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if any(name not in os.environ for name in required):
        raise RuntimeError("launch the boundary pilot with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available() or world_size != expected_world_size:
        raise RuntimeError(
            f"boundary pilot requires exactly {expected_world_size} visible processes"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return rank, world_size, local_rank, device


def load_pilot_contract(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = config["data"]
    pilot_path = Path(values["pilot_subset"])
    if file_sha256(pilot_path) != values["pilot_subset_sha256"]:
        raise RuntimeError("pilot subset checksum mismatch")
    boundary_path = Path(values["boundary_manifest"])
    if file_sha256(boundary_path) != values["boundary_manifest_sha256"]:
        raise RuntimeError("boundary manifest checksum mismatch")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    boundaries = load_jsonl(boundary_path)
    boundary_by_uid = {row["uid"]: row for row in boundaries}
    for frozen in pilot["w2c_records"]:
        source = boundary_by_uid.get(frozen["uid"])
        if source is None or any(
            source[key] != frozen[key]
            for key in (
                "dataset", "boundary_layer", "all_full_prefix_length",
                "valid_nonfull_actions", "boundary_route_indices", "singleton",
            )
        ):
            raise RuntimeError(f"pilot boundary metadata mismatch for {frozen['uid']}")
    return pilot, boundaries


def build_pilot_rows(
    manifest_rows: list[dict[str, Any]], pilot: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_uid = {row["uid"]: row for row in manifest_rows}
    w2c_records = list(pilot["w2c_records"])
    c2c_records = list(pilot["c2c_records"])
    ordered_uids = [row["uid"] for row in w2c_records + c2c_records]
    if len(ordered_uids) != len(set(ordered_uids)):
        raise RuntimeError("pilot contract contains duplicate UIDs")
    try:
        rows = [by_uid[uid] for uid in ordered_uids]
    except KeyError as exc:
        raise RuntimeError(f"pilot UID is absent from the training manifest: {exc}") from exc
    boundary = {row["uid"]: row for row in w2c_records}
    if any(row["split"] != "train" for row in rows):
        raise RuntimeError("pilot must contain only training-split samples")
    if sum(row["route_type"] == "W2C" for row in rows) != len(w2c_records):
        raise RuntimeError("pilot W2C population mismatch")
    if sum(row["route_type"] == "C2C" for row in rows) != len(c2c_records):
        raise RuntimeError("pilot C2C population mismatch")
    return rows, boundary


def render_report(
    *, config_path: Path, config_sha256: str, history: list[dict[str, Any]],
    completed_epoch: int, passed: bool, output_dir: Path,
) -> str:
    validations = [row for row in history if row.get("validation") is not None]
    if not validations:
        raise RuntimeError("pilot report requires at least one validation")
    best = max(
        validations,
        key=lambda row: (
            bool(row["validation"]["gate"]["passed"]),
            float(row["validation"]["boundary"]["valid_action_at_1"]),
            float(row["validation"]["execution"]["w2c_rescue_rate"]),
            -int(row["epoch"]),
        ),
    )
    boundary = best["validation"]["boundary"]
    execution = best["validation"]["execution"]
    answer = "YES" if passed else "NO"
    return "\n".join(
        [
            "# Mandatory-Boundary Overfit Pilot",
            "",
            f"- Config: `{config_path}`",
            f"- Config SHA-256: `{config_sha256}`",
            f"- Output: `{output_dir}`",
            f"- Completed epoch: {completed_epoch}",
            f"- Best evaluated epoch: {best['epoch']}",
            f"- Prospective gate passed: **{passed}**",
            "",
            "## Question 1",
            "",
            "Can the unchanged online router recognize a mandatory deviation state when explicitly trained on it?",
            "",
            f"Answer: **{answer}**",
            "",
            "## Best-checkpoint evidence",
            "",
            f"- Boundary Valid-Action@1: {boundary['valid_action_at_1']:.6f}",
            f"- Boundary non-FULL recall: {boundary['nonfull_recall']:.6f}",
            f"- Singleton action recall: `{json.dumps(boundary['singleton_action_recall'], sort_keys=True)}`",
            f"- Free rollout left all-FULL: {boundary['free_rollout']['left_all_full_fraction']:.6f}",
            f"- W2C rescue rate: {execution['w2c_rescue_rate']:.6f}",
            f"- C2C preservation rate: {execution['c2c_preservation_rate']:.6f}",
            "",
            "The full online A2 retrain is authorized only when every frozen gate passes.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args()
    if not args.confirm_training:
        raise RuntimeError("boundary pilot requires --confirm-training")

    config_path = Path(args.config)
    config_sha = file_sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if file_sha256(Path(config["source_plan"])) != config["source_plan_sha256"]:
        raise RuntimeError("collapse plan checksum mismatch")
    parent_path = Path(config["matched_parent_config"])
    if file_sha256(parent_path) != config["matched_parent_config_sha256"]:
        raise RuntimeError("matched parent config checksum mismatch")
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    verify_parent_contract(config, parent)
    require_slurm_environment(os.environ)
    verify_output_contract(
        Path(args.output_dir), Path(config["reporting"]["output_dir"])
    )
    rank, world_size, local_rank, device = distributed_context(
        int(config["training"]["world_size"])
    )
    run_lock = None
    try:
        output_dir = Path(args.output_dir)
        if rank == 0:
            run_lock = acquire_run_lock(output_dir, resume=args.resume)
        dist.barrier()
        configure_determinism(int(config["training"]["seed"]))
        manifest_rows = load_verified_manifest(
            config["data"]["manifest"], config["data"]["manifest_sha256"]
        )
        pilot, _boundaries = load_pilot_contract(config)
        rows, boundary_by_uid = build_pilot_rows(manifest_rows, pilot)
        if len(rows) != int(config["data"]["records"]):
            raise RuntimeError("pilot record count differs from config")
        sources = load_source_metadata(
            config["data"]["source_manifest"],
            config["data"]["source_manifest_sha256"],
            {row["uid"] for row in rows},
        )
        processor, base_model, wrapped_model, _ = load_frozen_model(
            config["base_model"]["path"], config["base_model"]["revision"], local_rank
        )
        base_model.requires_grad_(False).eval()
        seed_everything(int(config["training"]["seed"]))
        core = make_router(config, device)
        router = DistributedDataParallel(
            core, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, gradient_as_bucket_view=True,
        )

        local_rows = rows[rank::world_size]
        expected_local = int(config["training"]["per_gpu_samples"])
        if len(local_rows) != expected_local:
            raise RuntimeError("pilot rows do not shard evenly across ranks")
        cached = []
        for position, row in enumerate(local_rows, start=1):
            sample, inputs, _metadata, prepared = prepare_sample(
                processor, wrapped_model, row, sources[row["uid"]], device
            )
            trie = manifest_trie(row, num_layers=int(config["router"]["num_layers"]))
            boundary = boundary_by_uid.get(row["uid"])
            trajectory = None
            if boundary is not None:
                route = boundary_teacher_route(
                    row, boundary, num_layers=int(config["router"]["num_layers"])
                )
                trajectory = replay_teacher_forced_states(
                    wrapped_model, prepared, route, trie
                )
            cached.append(
                {
                    "row": row, "sample": sample, "inputs": inputs,
                    "prepared": prepared, "trie": trie,
                    "boundary": boundary, "w2c_trajectory": trajectory,
                }
            )
            if position <= 3 or position % 10 == 0:
                print(
                    json.dumps(
                        {"event": "boundary_pilot_cache", "rank": rank,
                         "position": position, "records": len(local_rows), "uid": row["uid"]},
                        sort_keys=True,
                    ), flush=True,
                )

        training = config["training"]
        optimizer, scheduler = build_training_optimizer_and_scheduler(
            router.parameters(), training
        )
        history: list[dict[str, Any]] = []
        start_epoch = 1
        global_step = 0
        restored_passed = False
        if rank == 0:
            initialization = {
                "schema_version": "four_action_boundary_pilot_initialization_v1",
                "config": str(config_path), "config_sha256": config_sha,
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "world_size": world_size, "records": len(rows),
                "pilot_subset_sha256": config["data"]["pilot_subset_sha256"],
                "boundary_manifest_sha256": config["data"]["boundary_manifest_sha256"],
            }
            initialization_path = output_dir / "initialization.json"
            if initialization_path.is_file():
                if json.loads(initialization_path.read_text(encoding="utf-8")) != initialization:
                    raise RuntimeError("pilot initialization contract mismatch")
            else:
                write_json(initialization_path, initialization)
        dist.barrier()
        if args.resume:
            metadata_path = latest_resume_metadata(output_dir)
            if metadata_path is not None:
                payload = load_resume_payload(
                    metadata_path, config_sha256=config_sha, world_size=world_size
                )
                router.module.load_state_dict(payload["router"], strict=True)
                optimizer.load_state_dict(payload["optimizer"])
                scheduler.load_state_dict(payload["scheduler"])
                global_step = int(payload["global_step"])
                start_epoch = int(payload["epoch"]) + 1
                restored_passed = bool(
                    payload["metrics"]["validation"]["gate"]["passed"]
                )
                restore_rng_state(payload["rng_states"][rank], device)
            if rank == 0 and metadata_path is not None:
                history_path = Path(config["reporting"]["history"])
                history = [
                    json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and int(json.loads(line)["epoch"]) < start_epoch
                ]
                write_jsonl(history_path, history)
            elif rank == 0:
                write_jsonl(Path(config["reporting"]["history"]), [])
        dist.barrier()

        accumulation = int(training["gradient_accumulation_steps"])
        passed = restored_passed
        completed_epoch = start_epoch - 1
        epoch_range = (
            range(start_epoch, int(training["epochs"]) + 1) if not passed else ()
        )
        for epoch in epoch_range:
            router.train()
            optimizer.zero_grad(set_to_none=True)
            order = list(range(len(cached)))
            random.Random(int(training["seed"]) + 10_000 * rank + epoch).shuffle(order)
            if len(order) % accumulation:
                raise RuntimeError("per-rank pilot samples must divide accumulation")
            loss_sum = 0.0
            started = time.time()
            for microstep, cache_index in enumerate(order, start=1):
                item = cached[cache_index]
                if item["w2c_trajectory"] is not None:
                    trajectory = item["w2c_trajectory"]
                    route_index = int(item["boundary"]["teacher_route_index"])
                else:
                    routes = manifest_route_tensor(
                        item["row"], num_layers=int(config["router"]["num_layers"])
                    )
                    route_index = deterministic_route_index(
                        uid=item["row"]["uid"], route_count=len(routes),
                        seed=int(training["seed"]), epoch=epoch,
                    )
                    trajectory = replay_teacher_forced_states(
                        wrapped_model, item["prepared"], routes[route_index], item["trie"]
                    )
                synchronize = microstep % accumulation == 0
                context = nullcontext() if synchronize else router.no_sync()
                with context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = set_valued_action_loss(
                            router_logits_for_trajectory(router, trajectory),
                            trajectory.valid_action_mask,
                        )
                    (loss / accumulation).backward()
                loss_sum += float(loss.detach().item())
                if synchronize:
                    torch.nn.utils.clip_grad_norm_(
                        router.parameters(), float(training["gradient_clip_norm"])
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                if microstep <= 3 or microstep % 10 == 0:
                    print(
                        json.dumps(
                            {"event": "boundary_pilot_train", "rank": rank,
                             "epoch": epoch, "microstep": microstep,
                             "global_step": global_step, "uid": item["row"]["uid"],
                             "route_index": route_index, "loss": float(loss.detach().item())},
                            sort_keys=True,
                        ), flush=True,
                    )
            totals = torch.tensor([loss_sum, len(order)], dtype=torch.float64, device=device)
            dist.all_reduce(totals)
            validation = None
            validation_outputs = None
            if epoch % int(training["validation_every_epochs"]) == 0:
                router.eval()
                local_outputs = []
                for item in cached:
                    boundary = item["boundary"]
                    row = item["row"]
                    boundary_fields: dict[str, Any] = {}
                    if boundary is not None:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = router_logits_for_trajectory(
                                router, item["w2c_trajectory"]
                            )
                        layer = int(boundary["boundary_layer"])
                        predicted = FOUR_ACTIONS[int(logits[layer].argmax().item())]
                        mask_actions = [
                            FOUR_ACTIONS[index] for index, value in enumerate(
                                item["w2c_trajectory"].valid_action_mask[layer].tolist()
                            ) if value
                        ]
                        if mask_actions != boundary["valid_nonfull_actions"]:
                            raise RuntimeError("runtime boundary mask differs from frozen metadata")
                        boundary_fields = {
                            "boundary_layer": layer,
                            "valid_nonfull_actions": boundary["valid_nonfull_actions"],
                            "singleton": bool(boundary["singleton"]),
                            "predicted_boundary_action": predicted,
                            "boundary_logits": logits[layer].float().cpu().tolist(),
                        }
                    execution = execute_router(
                        processor=processor, wrapped_model=wrapped_model, router=router,
                        inputs=item["inputs"], prepared=item["prepared"],
                        sample=item["sample"], config=config,
                    )
                    local_outputs.append(
                        {"uid": row["uid"], "dataset": row["dataset"],
                         "route_type": row["route_type"], **boundary_fields, **execution}
                    )
                    if len(local_outputs) <= 3:
                        print(
                            json.dumps(
                                {"event": "boundary_pilot_validation", "rank": rank,
                                 "epoch": epoch, "uid": row["uid"],
                                 "correct": execution["correct"],
                                 "route_key": execution["route_key"]},
                                sort_keys=True,
                            ), flush=True,
                        )
                combined = gather_flat(local_outputs, world_size)
                local_rng_state = capture_rng_state(device)
                rng_states: list[Any] = [None] * world_size
                dist.all_gather_object(rng_states, local_rng_state)
                if rank == 0:
                    combined.sort(key=lambda row: row["uid"])
                    if len(combined) != len(rows):
                        raise RuntimeError("pilot validation coverage mismatch")
                    boundary_summary = mandatory_boundary_metrics(
                        [row for row in combined if row["route_type"] == "W2C"],
                        num_layers=int(config["router"]["num_layers"]),
                    )
                    execution_summary = summarize_execution_rows(combined)
                    validation = {
                        "boundary": boundary_summary,
                        "execution": execution_summary,
                    }
                    validation["gate"] = mandatory_boundary_pilot_gate(
                        validation, config["gates"]
                    )
                    validation_outputs = combined
            epoch_row = None
            if rank == 0:
                epoch_row = {
                    "epoch": epoch, "global_step": global_step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.time() - started,
                    "train": {"samples": int(totals[1].item()),
                              "mean_loss": float(totals[0].item() / totals[1].item())},
                    "validation": validation,
                }
                history.append(epoch_row)
                write_jsonl(Path(config["reporting"]["history"]), history)
                if validation is not None:
                    save_epoch_artifacts(
                        output_dir, epoch,
                        checkpoint_payload(
                            router, optimizer, scheduler, epoch=epoch,
                            global_step=global_step, config_sha256=config_sha,
                            metrics=epoch_row, rng_states=rng_states,
                        ),
                        validation_outputs,
                    )
                    passed = bool(validation["gate"]["passed"])
                print(
                    json.dumps(
                        {"event": "boundary_pilot_epoch_complete", **epoch_row,
                         "prospective_gate_passed": passed}, sort_keys=True
                    ), flush=True,
                )
            flag = torch.tensor(int(passed), dtype=torch.int32, device=device)
            dist.broadcast(flag, src=0)
            passed = bool(flag.item())
            completed_epoch = epoch
            dist.barrier()
            if passed:
                break

        if rank == 0:
            write_json(
                output_dir / "training_summary.json",
                {"schema_version": "four_action_boundary_pilot_training_v1",
                 "passed": passed, "completed_epoch": completed_epoch,
                 "global_step": global_step, "max_epochs": int(training["epochs"]),
                 "full_online_a2_authorized": passed},
            )
            report = render_report(
                config_path=config_path, config_sha256=config_sha, history=history,
                completed_epoch=completed_epoch, passed=passed, output_dir=output_dir,
            )
            Path(config["reporting"]["report"]).write_text(report, encoding="utf-8")
        dist.barrier()
    finally:
        if run_lock is not None:
            release_run_lock(run_lock)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
