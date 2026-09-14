#!/usr/bin/env python3
"""Fit and aggregate the frozen counterfactual-effect identifiability ladder."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dense_failure_stage2.counterfactual_identifiability import (
    CONDITIONS,
    PairedTokenComparator,
    classify_case,
    construct_summary_feature,
    feature_width,
    matched_random_pair_assignment,
    validate_prediction_census,
)
from dense_failure_stage2.predictability_learnability import (
    SummaryScalarPredictor,
    binary_classification_metrics,
    harmful_ranking_metrics,
    high_precision_harmful_metrics,
    regression_metrics,
    robust_target_scale,
    uid_macro_regression_metrics,
)
from experiments.run_counterfactual_effect_identifiability import (
    DEFAULT_CONFIG,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    atomic_torch,
    canonical_hash,
    file_sha256,
    read_csv,
    read_json,
    read_jsonl,
    resolve_path,
    verify_contract,
)


def configure_determinism(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def configure_worker_runtime(config: Mapping[str, Any]) -> None:
    runtime = config["execution_runtime"]
    torch.set_num_threads(int(runtime["torch_intraop_threads_per_worker"]))
    torch.set_num_interop_threads(int(runtime["torch_interop_threads_per_worker"]))


def _u16_to_bf16(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(value, copy=True)).view(torch.bfloat16)


class EffectStore:
    def __init__(self, manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
        self.manifest = dict(manifest)
        self.rows = [dict(row) for row in rows]
        self.hidden = int(manifest["hidden_size"])
        files = manifest["files"]
        self.pooled = np.memmap(
            resolve_path(files["pooled"]["path"]), dtype=np.uint16, mode="r",
            shape=tuple(files["pooled"]["shape"]),
        )
        self.text = np.memmap(
            resolve_path(files["post_text"]["path"]), dtype=np.uint16, mode="r",
            shape=tuple(files["post_text"]["shape"]),
        )
        self.visual = np.memmap(
            resolve_path(files["post_visual"]["path"]), dtype=np.uint16, mode="r",
            shape=tuple(files["post_visual"]["shape"]),
        )

    def summaries(self, indices: Sequence[int] | np.ndarray) -> torch.Tensor:
        return _u16_to_bf16(self.pooled[np.asarray(indices, dtype=np.int64)])

    def summary_feature(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        target: str,
        condition: str,
        random_pairs: np.ndarray | None = None,
        swapped: bool = False,
    ) -> torch.Tensor:
        indices = np.asarray(indices, dtype=np.int64)
        summaries = self.summaries(indices)
        paired = None
        if random_pairs is not None:
            paired_indices = random_pairs[indices]
            paired_summaries = self.summaries(paired_indices)
            off_index = 2 if target == "read" else 3
            paired = paired_summaries[:, off_index].float()
        return construct_summary_feature(
            summaries, target=target, condition=condition,
            paired_off_summaries=paired, swapped=swapped,
        )

    def collate_pair(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        target: str,
        pin_memory: bool,
    ) -> tuple[torch.Tensor, ...]:
        selected = [self.rows[int(index)] for index in indices]
        max_text = max(int(row["text_tokens"]) for row in selected)
        max_visual = max(int(row["visual_tokens"]) for row in selected)
        batch = len(selected)
        full_text = torch.zeros((batch, max_text, self.hidden), dtype=torch.bfloat16, pin_memory=pin_memory)
        off_text = torch.zeros_like(full_text)
        full_visual = torch.zeros((batch, max_visual, self.hidden), dtype=torch.bfloat16, pin_memory=pin_memory)
        off_visual = torch.zeros_like(full_visual)
        text_mask = torch.zeros((batch, max_text), dtype=torch.bool, pin_memory=pin_memory)
        visual_mask = torch.zeros((batch, max_visual), dtype=torch.bool, pin_memory=pin_memory)
        off_action = 1 if target == "read" else 2
        for output_index, row in enumerate(selected):
            nt, nv = int(row["text_tokens"]), int(row["visual_tokens"])
            to, vo = int(row["text_offset"]), int(row["visual_offset"])
            full_text[output_index, :nt] = _u16_to_bf16(self.text[to : to + nt, 0])
            off_text[output_index, :nt] = _u16_to_bf16(self.text[to : to + nt, off_action])
            full_visual[output_index, :nv] = _u16_to_bf16(self.visual[vo : vo + nv, 0])
            off_visual[output_index, :nv] = _u16_to_bf16(self.visual[vo : vo + nv, off_action])
            text_mask[output_index, :nt] = True
            visual_mask[output_index, :nv] = True
        return full_text, full_visual, off_text, off_visual, text_mask, visual_mask


def _verify_extraction(contract: Mapping[str, Any], output_root: Path, domain: str) -> dict[str, Any]:
    manifest = read_json(output_root / f"features/{domain}_cache_manifest.json")
    if manifest.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError(f"{domain} cache contract differs")
    for spec in manifest["files"].values():
        path = resolve_path(spec["path"])
        if not path.is_file() or path.stat().st_size != int(spec["bytes"]) or file_sha256(path) != spec["sha256"]:
            raise RuntimeError(f"{domain} cache file differs: {path}")
    rows = read_jsonl(output_root / f"states/{domain}_state_manifest.jsonl")
    branches = read_jsonl(output_root / f"states/{domain}_branch_execution_manifest.jsonl")
    if len(branches) != 3 * len(rows):
        raise RuntimeError(f"{domain} branch census differs")
    return manifest


def _task_id(task: Mapping[str, Any]) -> str:
    return "__".join(
        str(task[key]) for key in ("phase", "target", "family", "condition", "model", "fold", "seed")
    )


def _assign_tasks(tasks: list[dict[str, Any]], world_size: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[(task["phase"], task["target"], task["family"], task["condition"], task["fold"])].append(task)
    loads = [0] * int(world_size)
    output = []
    for key, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), str(item[0]))):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        loads[rank] += len(members)
        output.extend({**task, "worker_rank": rank} for task in sorted(members, key=_task_id))
    return sorted(output, key=lambda task: (int(task["worker_rank"]), _task_id(task)))


def prepare_training(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    config = contract["static_config"]
    manifests = {domain: _verify_extraction(contract, output_root, domain) for domain in ("dense", "routed")}
    dense = read_jsonl(output_root / "states/dense_state_manifest.jsonl")
    pairs, tiers = matched_random_pair_assignment(dense, seed=int(config["controls"]["random_pair_seed"]))
    pair_rows = [
        {
            "state_id": row["state_id"], "paired_state_id": dense[int(pairs[index])]["state_id"],
            "uid": row["uid"], "paired_uid": dense[int(pairs[index])]["uid"],
            "match_tier": tiers[index],
        }
        for index, row in enumerate(dense)
    ]
    atomic_jsonl(output_root / "controls/random_pair_registry.jsonl", pair_rows)
    tasks = []
    seeds = [int(value) for value in config["training"]["seeds"]]
    folds = range(int(config["split"]["folds"]))
    for target in ("read", "write"):
        for condition in CONDITIONS:
            for model in ("linear", "mlp"):
                for fold in folds:
                    for seed in seeds:
                        task = {"phase": "primary", "domain": "dense", "target": target, "family": "primary", "condition": condition, "model": model, "fold": fold, "seed": seed}
                        task["task_id"] = _task_id(task)
                        tasks.append(task)
        for fold in folds:
            for seed in seeds:
                task = {"phase": "primary", "domain": "dense", "target": target, "family": "token", "condition": "token_comparator", "model": "token_comparator", "fold": fold, "seed": seed}
                task["task_id"] = _task_id(task)
                tasks.append(task)
        for condition in config["features"]["delta_ablation_conditions"]:
            for fold in folds:
                for seed in seeds:
                    task = {"phase": "primary", "domain": "dense", "target": target, "family": "ablation", "condition": condition, "model": "mlp", "fold": fold, "seed": seed}
                    task["task_id"] = _task_id(task)
                    tasks.append(task)
        for condition in config["controls"]["random_pair_conditions"]:
            for fold in folds:
                for seed in seeds:
                    task = {"phase": "primary", "domain": "dense", "target": target, "family": "random_pair", "condition": condition, "model": "mlp", "fold": fold, "seed": seed}
                    task["task_id"] = _task_id(task)
                    tasks.append(task)
        for fold in folds:
            for seed in seeds:
                task = {"phase": "primary", "domain": "dense", "target": target, "family": "swapped_order", "condition": str(config["controls"]["branch_order_condition"]), "model": "mlp", "fold": fold, "seed": seed}
                task["task_id"] = _task_id(task)
                tasks.append(task)
    tasks = _assign_tasks(tasks, int(config["world_size"]))
    atomic_jsonl(output_root / "work/primary_training_tasks.jsonl", tasks)
    ready = {
        "schema_version": "counterfactual_effect_training_ready_v1",
        "contract_sha256": contract["contract_sha256"],
        "dense_cache_manifest_sha256": file_sha256(output_root / "features/dense_cache_manifest.json"),
        "routed_cache_manifest_sha256": file_sha256(output_root / "features/routed_cache_manifest.json"),
        "cache_mtime_ns": {
            domain: {name: resolve_path(spec["path"]).stat().st_mtime_ns for name, spec in manifest["files"].items()}
            for domain, manifest in manifests.items()
        },
        "primary_tasks": len(tasks),
        "random_pair_match_tiers": dict(Counter(tiers)),
    }
    atomic_json(output_root / "work/training_ready.json", ready)
    print(json.dumps(ready, sort_keys=True))


def _require_training_ready(contract: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    ready = read_json(output_root / "work/training_ready.json")
    if ready.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("training-ready contract differs")
    for domain in ("dense", "routed"):
        manifest_path = output_root / f"features/{domain}_cache_manifest.json"
        if file_sha256(manifest_path) != ready[f"{domain}_cache_manifest_sha256"]:
            raise RuntimeError(f"{domain} cache manifest changed after validation")
        manifest = read_json(manifest_path)
        for name, spec in manifest["files"].items():
            path = resolve_path(spec["path"])
            if path.stat().st_size != int(spec["bytes"]) or path.stat().st_mtime_ns != int(ready["cache_mtime_ns"][domain][name]):
                raise RuntimeError(f"{domain}/{name} cache changed after validation")
    return ready


def _role_indices(output_root: Path, rows: Sequence[Mapping[str, Any]], fold: int) -> dict[str, np.ndarray]:
    roles = {str(row["uid"]): str(row["role"]) for row in read_jsonl(output_root / f"splits/inner_roles_fold{fold}.jsonl")}
    result = {}
    for role in ("fit", "calibration", "outer_test"):
        result[role] = np.asarray([index for index, row in enumerate(rows) if roles[str(row["uid"])] == role], dtype=np.int64)
        if len(result[role]) == 0:
            raise RuntimeError(f"fold {fold} has no {role} states")
    return result


def _uid_weights(rows: Sequence[Mapping[str, Any]], indices: np.ndarray) -> np.ndarray:
    counts = Counter(str(rows[int(index)]["uid"]) for index in indices)
    weights = np.asarray([1.0 / counts[str(rows[int(index)]["uid"])] for index in indices])
    return weights / weights.mean()


def _summary_standardizer(
    store: EffectStore,
    indices: np.ndarray,
    task: Mapping[str, Any],
    random_pairs: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = feature_width(str(task["condition"]), store.hidden)
    total = torch.zeros(width, dtype=torch.float64)
    square = torch.zeros(width, dtype=torch.float64)
    count = 0
    for start in range(0, len(indices), 512):
        batch = indices[start : start + 512]
        features = store.summary_feature(
            batch, target=str(task["target"]), condition=str(task["condition"]),
            random_pairs=random_pairs if task["family"] == "random_pair" else None,
            swapped=task["family"] == "swapped_order",
        ).to(device=device, dtype=torch.float32)
        total += features.sum(dim=0).double().cpu()
        square += features.square().sum(dim=0).double().cpu()
        count += len(batch)
    mean = total / count
    variance = torch.clamp(square / count - mean.square(), min=0)
    std = torch.where(variance.sqrt() < 1e-6, torch.ones_like(mean), variance.sqrt())
    return mean.float(), std.float()


def _model(task: Mapping[str, Any], config: Mapping[str, Any], hidden: int) -> torch.nn.Module:
    if task["model"] in {"linear", "mlp"}:
        spec = config["training"][str(task["model"])]
        return SummaryScalarPredictor(
            kind=str(task["model"]), input_size=feature_width(str(task["condition"]), hidden),
            hidden_size=int(spec.get("hidden_size", 1)), dropout=float(spec.get("dropout", 0)),
        )
    spec = config["features"]["token_comparator"]
    return PairedTokenComparator(
        hidden_size=hidden, projection_size=int(spec["projection_size"]),
        attention_heads=int(spec["attention_heads"]), readout_hidden_size=int(spec["readout_hidden_size"]),
        dropout=float(spec["dropout"]),
    )


def _spec(task: Mapping[str, Any], config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["training"][str(task["model"])]


def _token_forward(model, packed: tuple[torch.Tensor, ...], device: torch.device) -> torch.Tensor:
    full_text, full_visual, off_text, off_visual, text_mask, visual_mask = packed
    non_blocking = bool(full_text.is_pinned())
    return model(
        full_text.to(device, non_blocking=non_blocking),
        full_visual.to(device, non_blocking=non_blocking),
        off_text.to(device, non_blocking=non_blocking),
        off_visual.to(device, non_blocking=non_blocking),
        text_mask=text_mask.to(device, non_blocking=non_blocking),
        visual_mask=visual_mask.to(device, non_blocking=non_blocking),
    )


def _iter_token_batches(
    store: EffectStore, batches: Iterable[np.ndarray], *, target: str
) -> Iterable[tuple[np.ndarray, tuple[torch.Tensor, ...]]]:
    iterator = iter(batches)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cf-token") as executor:
        try:
            current = np.asarray(next(iterator), dtype=np.int64)
        except StopIteration:
            return
        future: Future[tuple[torch.Tensor, ...]] = executor.submit(store.collate_pair, current, target=target, pin_memory=True)
        while True:
            packed = future.result()
            try:
                following = np.asarray(next(iterator), dtype=np.int64)
            except StopIteration:
                yield current, packed
                break
            future = executor.submit(store.collate_pair, following, target=target, pin_memory=True)
            yield current, packed
            current = following


def _predict(
    model: torch.nn.Module,
    task: Mapping[str, Any],
    store: EffectStore,
    indices: np.ndarray,
    *,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    random_pairs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    batches = [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]
    outputs = []
    with torch.inference_mode():
        if task["model"] == "token_comparator":
            iterator = _iter_token_batches(store, batches, target=str(task["target"]))
            for _batch, packed in iterator:
                outputs.append(_token_forward(model, packed, device).float().cpu())
        else:
            if mean is None or std is None:
                raise RuntimeError("summary prediction lacks normalization")
            for batch in batches:
                features = store.summary_feature(
                    batch, target=str(task["target"]), condition=str(task["condition"]),
                    random_pairs=random_pairs if task["family"] == "random_pair" else None,
                    swapped=task["family"] == "swapped_order",
                ).to(device=device, dtype=torch.float32)
                outputs.append(model((features - mean.to(device)) / std.to(device)).float().cpu())
    return torch.cat(outputs).numpy().astype(np.float64)


def train_task(
    contract: Mapping[str, Any], output_root: Path, external_root: Path,
    task: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], store: EffectStore,
    random_pairs: np.ndarray, device: torch.device,
    stats_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    config = contract["static_config"]
    configure_determinism(int(task["seed"]))
    roles = _role_indices(output_root, rows, int(task["fold"]))
    target = np.asarray([float(row[f"u_{task['target']}"]) for row in rows], dtype=np.float64)
    scale = robust_target_scale(target[roles["fit"]], floor=float(config["targets"]["target_scale_floor"]))
    scaled = scale.transform(target)
    model = _model(task, config, store.hidden).to(device=device, dtype=torch.float32)
    spec = _spec(task, config)
    batch_size = int(spec.get("batch_size", spec.get("state_microbatch")))
    if task["model"] == "token_comparator":
        mean = std = None
    else:
        key = (task["domain"], task["fold"], task["target"], task["family"], task["condition"])
        if key not in stats_cache:
            stats_cache[key] = _summary_standardizer(store, roles["fit"], task, random_pairs, device)
        mean, std = stats_cache[key]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    fit_weights = _uid_weights(rows, roles["fit"])
    calibration_weights = _uid_weights(rows, roles["calibration"])
    generator = torch.Generator(device="cpu").manual_seed(int(task["seed"]))
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(int(spec["maximum_epochs"])):
        model.train()
        local_order = torch.randperm(len(roles["fit"]), generator=generator).numpy()
        batches = [roles["fit"][local_order[start : start + batch_size]] for start in range(0, len(local_order), batch_size)]
        batch_weights = [fit_weights[local_order[start : start + batch_size]] for start in range(0, len(local_order), batch_size)]
        if task["model"] == "token_comparator":
            iterator = _iter_token_batches(store, batches, target=str(task["target"]))
        else:
            iterator = ((batch, None) for batch in batches)
        epoch_sum = epoch_weight = 0.0
        for batch_number, (indices, packed) in enumerate(iterator):
            weights = torch.tensor(batch_weights[batch_number], dtype=torch.float32, device=device)
            truth = torch.tensor(scaled[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            if task["model"] == "token_comparator":
                prediction = _token_forward(model, packed, device)
            else:
                features = store.summary_feature(
                    indices, target=str(task["target"]), condition=str(task["condition"]),
                    random_pairs=random_pairs if task["family"] == "random_pair" else None,
                    swapped=task["family"] == "swapped_order",
                ).to(device=device, dtype=torch.float32)
                prediction = model((features - mean.to(device)) / std.to(device))
            losses = torch.nn.functional.huber_loss(prediction, truth, delta=1.0, reduction="none")
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            epoch_sum += float((losses.detach() * weights).sum().cpu())
            epoch_weight += float(weights.sum().cpu())
        calibration_raw = _predict(
            model, task, store, roles["calibration"], mean=mean, std=std,
            random_pairs=random_pairs, device=device, batch_size=batch_size,
        )
        cal_loss = torch.nn.functional.huber_loss(
            torch.tensor(calibration_raw), torch.tensor(scaled[roles["calibration"]]),
            delta=1.0, reduction="none",
        ).numpy()
        calibration_loss = float(np.dot(cal_loss, calibration_weights) / calibration_weights.sum())
        history.append({"epoch": epoch + 1, "fit_loss": epoch_sum / epoch_weight, "calibration_loss": calibration_loss})
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= int(spec["minimum_epochs"]) and stale >= int(spec["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"training produced no checkpoint: {task['task_id']}")
    model.load_state_dict(best_state)
    test_raw = _predict(
        model, task, store, roles["outer_test"], mean=mean, std=std,
        random_pairs=random_pairs, device=device, batch_size=batch_size,
    )
    test_prediction = scale.inverse(test_raw)
    payload = {
        "schema_version": "counterfactual_effect_checkpoint_v1", "contract_sha256": contract["contract_sha256"],
        "task": dict(task), "task_sha256": canonical_hash(task), "model_state": best_state,
        "normalization_mean": mean, "normalization_std": std,
        "target_center": scale.center, "target_scale": scale.scale,
        "best_epoch": best_epoch, "best_calibration_loss": best_loss, "history": history,
        "test_indices": roles["outer_test"], "test_prediction": test_prediction,
    }
    checkpoint = external_root / "models" / str(task["phase"]) / f"{task['task_id']}.pt"
    atomic_torch(checkpoint, payload)
    return {
        "contract_sha256": contract["contract_sha256"], "task_id": task["task_id"],
        "task_sha256": canonical_hash(task), "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint), "best_epoch": best_epoch,
        "best_calibration_loss": best_loss, "test_states": len(roles["outer_test"]),
        "elapsed_seconds": time.monotonic() - started,
    }


def train_worker(config_path: Path, phase: str, rank: int, *, resume: bool) -> None:
    contract, output_root, external_root = verify_contract(config_path)
    _require_training_ready(contract, output_root)
    config = contract["static_config"]
    configure_worker_runtime(config)
    rank = int(rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tasks = [
        row for row in read_jsonl(output_root / f"work/{phase}_training_tasks.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    rows_by_domain = {}
    stores = {}
    for domain in {str(task["domain"]) for task in tasks}:
        rows_by_domain[domain] = read_jsonl(output_root / f"states/{domain}_state_manifest.jsonl")
        stores[domain] = EffectStore(read_json(output_root / f"features/{domain}_cache_manifest.json"), rows_by_domain[domain])
    dense_rows = rows_by_domain.get("dense") or read_jsonl(output_root / "states/dense_state_manifest.jsonl")
    random_pairs = matched_random_pair_assignment(dense_rows, seed=int(config["controls"]["random_pair_seed"]))[0]
    rank_root = output_root / f"work/training/{phase}/rank{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    stats_cache = {}
    completed = 0
    started = time.monotonic()
    for task in tasks:
        result_path = rank_root / f"{task['task_id']}.json"
        if resume and result_path.is_file():
            old = read_json(result_path)
            checkpoint = resolve_path(old.get("checkpoint", ""))
            if old.get("contract_sha256") == contract["contract_sha256"] and old.get("task_sha256") == canonical_hash(task) and checkpoint.is_file() and file_sha256(checkpoint) == old.get("checkpoint_sha256"):
                completed += 1
                continue
        result = train_task(
            contract, output_root, external_root, task,
            rows_by_domain[str(task["domain"])], stores[str(task["domain"])], random_pairs,
            device, stats_cache,
        )
        atomic_json(result_path, result)
        completed += 1
        torch.cuda.empty_cache()
        print(json.dumps({"phase": phase, "rank": rank, "completed": completed, "assigned": len(tasks), "task_id": task["task_id"], "elapsed_seconds": time.monotonic() - started}), flush=True)
    atomic_json(rank_root / "complete.json", {"contract_sha256": contract["contract_sha256"], "phase": phase, "rank": rank, "expected_tasks": len(tasks), "completed_tasks": completed})


def _collect_task_results(contract: Mapping[str, Any], output_root: Path, phase: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    tasks = read_jsonl(output_root / f"work/{phase}_training_tasks.jsonl")
    by_id = {str(task["task_id"]): task for task in tasks}
    results = []
    for rank in range(int(contract["static_config"]["world_size"])):
        rank_root = output_root / f"work/training/{phase}/rank{rank:02d}"
        complete = read_json(rank_root / "complete.json")
        if complete.get("contract_sha256") != contract["contract_sha256"] or complete.get("expected_tasks") != complete.get("completed_tasks"):
            raise RuntimeError(f"partial training rank: {phase}/{rank}")
        for path in sorted(rank_root.glob("*.json")):
            if path.name == "complete.json":
                continue
            result = read_json(path)
            task_id = str(result["task_id"])
            if task_id not in by_id or result.get("task_sha256") != canonical_hash(by_id[task_id]):
                raise RuntimeError(f"task result differs: {task_id}")
            checkpoint = resolve_path(result["checkpoint"])
            if file_sha256(checkpoint) != result["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint differs: {task_id}")
            results.append((by_id[task_id], result))
    if len(results) != len(tasks) or len({task[0]["task_id"] for task in results}) != len(tasks):
        raise RuntimeError(f"global task completion differs: {phase}")
    return results


def _oof_predictions(
    rows: Sequence[Mapping[str, Any]], entries: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    seeds = sorted({int(task["seed"]) for task, _ in entries})
    per_seed = {}
    seed_rows = []
    truth = np.asarray([float(row[f"u_{entries[0][0]['target']}"]) for row in rows])
    for seed in seeds:
        prediction = np.full(len(rows), np.nan)
        for task, result in entries:
            if int(task["seed"]) != seed:
                continue
            checkpoint = torch.load(resolve_path(result["checkpoint"]), map_location="cpu", weights_only=False)
            indices = np.asarray(checkpoint["test_indices"], dtype=np.int64)
            values = np.asarray(checkpoint["test_prediction"], dtype=np.float64)
            if np.isfinite(prediction[indices]).any():
                raise RuntimeError("OOF predictions overlap")
            prediction[indices] = values
        if not np.isfinite(prediction).all():
            raise RuntimeError("OOF prediction census is incomplete")
        per_seed[seed] = prediction
        seed_rows.append({"seed": seed, **regression_metrics(truth=truth, prediction=prediction)})
    return np.mean(np.stack(list(per_seed.values())), axis=0), seed_rows


def aggregate_primary(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    config = contract["static_config"]
    results = _collect_task_results(contract, output_root, "primary")
    rows = read_jsonl(output_root / "states/dense_state_manifest.jsonl")
    grouped: dict[tuple[str, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for task, result in results:
        grouped[(str(task["target"]), str(task["family"]), str(task["condition"]), str(task["model"]))].append((task, result))
    metric_rows = []
    seed_metric_rows = []
    predictions = {}
    for key, entries in sorted(grouped.items()):
        target, family, condition, model = key
        prediction, seed_rows = _oof_predictions(rows, entries)
        truth = np.asarray([float(row[f"u_{target}"]) for row in rows])
        metrics = regression_metrics(truth=truth, prediction=prediction)
        harmful = harmful_ranking_metrics(truth=truth, prediction=prediction)
        precision = high_precision_harmful_metrics(
            truth=truth, prediction=prediction,
            coverages=config["evaluation"]["harmful_coverages"],
            precision_targets=config["evaluation"]["harmful_precision_targets"],
        )
        macro = uid_macro_regression_metrics(
            truth=truth, prediction=prediction, uids=[str(row["uid"]) for row in rows],
            minimum_nonconstant_states=int(config["evaluation"]["uid_macro_minimum_nonconstant_states"]),
        )
        metric_rows.append(
            {"target": target, "family": family, "condition": condition, "model": model,
             **metrics, "harmful_auroc": harmful["auroc"], "harmful_auprc": harmful["auprc"],
             **{key_: value for key_, value in precision.items() if key_.startswith("precision_") or key_.startswith("recall_")},
             **{f"uid_macro_{key_}": value for key_, value in macro.items()},
            }
        )
        seed_metric_rows.extend({"target": target, "family": family, "condition": condition, "model": model, **row} for row in seed_rows)
        predictions[key] = prediction
    for target in ("read", "write"):
        target_rows = [row for row in metric_rows if row["target"] == target and row["family"] in {"primary", "token"}]
        atomic_csv(output_root / f"{target}/metrics.csv", target_rows)
        atomic_csv(output_root / f"{target}/text_visual_delta_ablation.csv", [row for row in metric_rows if row["target"] == target and row["family"] == "ablation"])
        canonical_directories = {
            "pre": "pre_state",
            "off_post": "write_only_post" if target == "read" else "read_only_post",
        }
        for condition in (*CONDITIONS, "token_comparator"):
            directory = output_root / target / condition
            directory.mkdir(parents=True, exist_ok=True)
            subset = [row for row in target_rows if row["condition"] == condition]
            if subset:
                atomic_csv(directory / "metrics.csv", subset)
            prediction_rows = []
            for key, values in predictions.items():
                if key[0] == target and key[2] == condition and key[1] in {"primary", "token"}:
                    for index, row in enumerate(rows):
                        prediction_rows.append({"state_id": row["state_id"], "uid": row["uid"], "target": target, "family": key[1], "condition": condition, "model": key[3], "truth": row[f"u_{target}"], "prediction": float(values[index]), "fold": row["fold"]})
            if prediction_rows:
                atomic_jsonl(directory / "oof_predictions.jsonl", prediction_rows)
            alias_name = canonical_directories.get(condition)
            if alias_name:
                alias = output_root / target / alias_name
                alias.mkdir(parents=True, exist_ok=True)
                if subset:
                    atomic_csv(alias / "metrics.csv", subset)
                if prediction_rows:
                    atomic_jsonl(alias / "oof_predictions.jsonl", prediction_rows)
    atomic_csv(output_root / "statistics/seed_metrics.csv", seed_metric_rows)
    atomic_csv(output_root / "work/all_primary_metrics.csv", metric_rows)
    selection = {}
    tie = {name: index for index, name in enumerate(config["controls"]["secondary_condition_tiebreak"])}
    for target in ("read", "write"):
        candidates = [row for row in metric_rows if row["target"] == target and row["family"] == "primary" and row["model"] == "mlp"]
        winner = max(candidates, key=lambda row: (float(row["spearman"]), -tie[str(row["condition"])]))
        selection[target] = {"condition": winner["condition"], "dense_oof_spearman": winner["spearman"]}
    atomic_json(output_root / "work/secondary_selection.json", {"contract_sha256": contract["contract_sha256"], "selection": selection})
    tasks = []
    for target in ("read", "write"):
        for model, condition, family in (
            ("mlp", selection[target]["condition"], "selected_mlp"),
            ("token_comparator", "token_comparator", "token"),
        ):
            for fold in range(int(config["split"]["folds"])):
                for seed in config["training"]["seeds"]:
                    task = {"phase": "secondary", "domain": "routed", "target": target, "family": family, "condition": condition, "model": model, "fold": fold, "seed": int(seed)}
                    task["task_id"] = _task_id(task)
                    tasks.append(task)
    atomic_jsonl(output_root / "work/secondary_training_tasks.jsonl", _assign_tasks(tasks, int(config["world_size"])))
    print(json.dumps({"primary_tasks": len(results), "metric_rows": len(metric_rows), "secondary_selection": selection, "secondary_tasks": len(tasks)}, sort_keys=True))


def _load_primary_model_for_transfer(
    output_root: Path, target: str, model: str, condition: str, fold: int, seed: int,
    config: Mapping[str, Any], hidden: int, device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    task = {"phase": "primary", "domain": "dense", "target": target, "family": "token" if model == "token_comparator" else "primary", "condition": condition, "model": model, "fold": fold, "seed": seed}
    task["task_id"] = _task_id(task)
    for rank in range(int(config["world_size"])):
        result_path = output_root / f"work/training/primary/rank{rank:02d}/{task['task_id']}.json"
        if result_path.is_file():
            result = read_json(result_path)
            payload = torch.load(resolve_path(result["checkpoint"]), map_location="cpu", weights_only=False)
            instance = _model(task, config, hidden).to(device=device, dtype=torch.float32)
            instance.load_state_dict(payload["model_state"])
            return instance, payload
    raise RuntimeError(f"primary transfer checkpoint is missing: {task['task_id']}")


def dense_to_routed_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root, _ = verify_contract(config_path)
    _require_training_ready(contract, output_root)
    config = contract["static_config"]
    configure_worker_runtime(config)
    rank = int(rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tasks = [row for row in read_jsonl(output_root / "work/secondary_training_tasks.jsonl") if int(row["worker_rank"]) == rank]
    rows = read_jsonl(output_root / "states/routed_state_manifest.jsonl")
    store = EffectStore(read_json(output_root / "features/routed_cache_manifest.json"), rows)
    random_pairs = np.arange(len(rows), dtype=np.int64)
    rank_root = output_root / f"work/transfer/rank{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for task in tasks:
        result_path = rank_root / f"{task['task_id']}.json"
        if resume and result_path.is_file() and read_json(result_path).get("contract_sha256") == contract["contract_sha256"]:
            completed += 1
            continue
        model, payload = _load_primary_model_for_transfer(
            output_root, str(task["target"]), str(task["model"]), str(task["condition"]),
            int(task["fold"]), int(task["seed"]), config, store.hidden, device,
        )
        roles = _role_indices(output_root, rows, int(task["fold"]))
        spec = _spec(task, config)
        prediction = _predict(
            model, task, store, roles["outer_test"], mean=payload["normalization_mean"],
            std=payload["normalization_std"], random_pairs=random_pairs, device=device,
            batch_size=int(spec.get("batch_size", spec.get("state_microbatch"))),
        )
        prediction = prediction * float(payload["target_scale"]) + float(payload["target_center"])
        atomic_json(result_path, {"contract_sha256": contract["contract_sha256"], "task_id": task["task_id"], "task_sha256": canonical_hash(task), "test_indices": roles["outer_test"].tolist(), "test_prediction": prediction.tolist()})
        completed += 1
        del model, payload
        torch.cuda.empty_cache()
    atomic_json(rank_root / "complete.json", {"contract_sha256": contract["contract_sha256"], "rank": rank, "expected_tasks": len(tasks), "completed_tasks": completed})


def _collect_transfer(contract: Mapping[str, Any], output_root: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    tasks = read_jsonl(output_root / "work/secondary_training_tasks.jsonl")
    by_id = {str(task["task_id"]): task for task in tasks}
    output = []
    for rank in range(int(contract["static_config"]["world_size"])):
        rank_root = output_root / f"work/transfer/rank{rank:02d}"
        complete = read_json(rank_root / "complete.json")
        if complete.get("contract_sha256") != contract["contract_sha256"] or complete.get("expected_tasks") != complete.get("completed_tasks"):
            raise RuntimeError(f"partial transfer rank: {rank}")
        for path in sorted(rank_root.glob("*.json")):
            if path.name == "complete.json":
                continue
            result = read_json(path)
            task = by_id.get(str(result["task_id"]))
            if task is None or result.get("task_sha256") != canonical_hash(task):
                raise RuntimeError(f"transfer result differs: {result.get('task_id')}")
            output.append((task, result))
    if len(output) != len(tasks) or len({task[0]["task_id"] for task in output}) != len(tasks):
        raise RuntimeError("global transfer completion differs")
    return output


def _transfer_oof(
    rows: Sequence[Mapping[str, Any]], entries: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    seeds = sorted({int(task["seed"]) for task, _ in entries})
    truth = np.asarray([float(row[f"u_{entries[0][0]['target']}"]) for row in rows])
    seed_predictions = []
    seed_metrics = []
    for seed in seeds:
        prediction = np.full(len(rows), np.nan)
        for task, result in entries:
            if int(task["seed"]) != seed:
                continue
            indices = np.asarray(result["test_indices"], dtype=np.int64)
            values = np.asarray(result["test_prediction"], dtype=np.float64)
            if np.isfinite(prediction[indices]).any():
                raise RuntimeError("transfer predictions overlap")
            prediction[indices] = values
        if not np.isfinite(prediction).all():
            raise RuntimeError("transfer prediction census is incomplete")
        seed_predictions.append(prediction)
        seed_metrics.append({"seed": seed, **regression_metrics(truth=truth, prediction=prediction)})
    return np.mean(np.stack(seed_predictions), axis=0), seed_metrics


def _prediction_rows(path: Path, *, model: str) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if str(row["model"]) == model]
    validate_prediction_census(
        [str(row["state_id"]) for row in read_jsonl(path.parent.parent.parent / "states/dense_state_manifest.jsonl")],
        rows,
    )
    return rows


def _weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    mx, my = float(np.dot(weights, x) / total), float(np.dot(weights, y) / total)
    xc, yc = x - mx, y - my
    denominator = math.sqrt(float(np.dot(weights, xc * xc) * np.dot(weights, yc * yc)))
    return float(np.dot(weights, xc * yc) / denominator) if denominator else float("nan")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _bootstrap_rows(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    truth = np.asarray([float(row["truth"]) for row in rows])
    prediction = np.asarray([float(row["prediction"]) for row in rows])
    groups = sorted({str(row["image_group_id"]) for row in rows})
    lookup = {group: index for index, group in enumerate(groups)}
    codes = np.asarray([lookup[str(row["image_group_id"])] for row in rows])
    tr, pr = _average_ranks(truth), _average_ranks(prediction)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for draw in range(draws):
        counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        values[draw] = _weighted_correlation(tr, pr, counts[codes].astype(np.float64))
    finite = values[np.isfinite(values)]
    return {
        "draws_requested": draws, "draws_valid": len(finite),
        "ci_low": float(np.quantile(finite, 0.025)),
        "median": float(np.quantile(finite, 0.5)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "bootstrap_unit": "image_group_id",
    }


def _pairwise_bootstrap(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    left = sorted(left, key=lambda row: str(row["state_id"]))
    right = sorted(right, key=lambda row: str(row["state_id"]))
    if [row["state_id"] for row in left] != [row["state_id"] for row in right]:
        raise RuntimeError("pairwise bootstrap rows do not align")
    truth = np.asarray([float(row["truth"]) for row in left])
    lp = np.asarray([float(row["prediction"]) for row in left])
    rp = np.asarray([float(row["prediction"]) for row in right])
    groups = sorted({str(row["image_group_id"]) for row in left})
    lookup = {group: index for index, group in enumerate(groups)}
    codes = np.asarray([lookup[str(row["image_group_id"])] for row in left])
    tr, lr, rr = _average_ranks(truth), _average_ranks(lp), _average_ranks(rp)
    rng = np.random.default_rng(seed)
    differences = np.empty(draws)
    for draw in range(draws):
        counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        weights = counts[codes].astype(np.float64)
        differences[draw] = _weighted_correlation(tr, lr, weights) - _weighted_correlation(tr, rr, weights)
    finite = differences[np.isfinite(differences)]
    return {"draws_requested": draws, "draws_valid": len(finite), "ci_low": float(np.quantile(finite, .025)), "median": float(np.quantile(finite, .5)), "ci_high": float(np.quantile(finite, .975)), "bootstrap_unit": "image_group_id"}


def _metric_subset(rows: Sequence[Mapping[str, Any]], mask: np.ndarray) -> dict[str, Any]:
    truth = np.asarray([float(row["truth"]) for row in rows])[mask]
    prediction = np.asarray([float(row["prediction"]) for row in rows])[mask]
    if len(truth) < 2 or len(np.unique(truth)) < 2 or len(np.unique(prediction)) < 2:
        return {"support": len(truth), "spearman": float("nan"), "pearson": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    return regression_metrics(truth=truth, prediction=prediction)


def _plot_primary(output_root: Path, primary_metrics: Sequence[Mapping[str, Any]]) -> None:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = ["PRE", "FULL", "OFF", "PAIR", "DELTA", "PAIR+DELTA", "TOKEN"]
    condition_names = [*CONDITIONS, "token_comparator"]
    for target in ("read", "write"):
        values = []
        for condition in condition_names:
            model = "token_comparator" if condition == "token_comparator" else "mlp"
            match = [row for row in primary_metrics if row["target"] == target and row["condition"] == condition and row["model"] == model and row["family"] in {"primary", "token"}]
            values.append(float(match[0]["spearman"]))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_ylabel("OOF Spearman")
        ax.set_title(f"{target.upper()} one-step identifiability")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(figures / f"{target}_identifiability_comparison.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for target, marker in (("read", "o"), ("write", "s")):
        values = []
        for condition in condition_names:
            model = "token_comparator" if condition == "token_comparator" else "mlp"
            row = next(row for row in primary_metrics if row["target"] == target and row["condition"] == condition and row["model"] == model and row["family"] in {"primary", "token"})
            values.append(float(row["spearman"]))
        ax.plot(labels, values, marker=marker, label=target.upper())
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_ylabel("OOF Spearman")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures / "pre_vs_post_vs_counterfactual.png", dpi=180)
    plt.close(fig)


def finalize(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    config = contract["static_config"]
    _require_training_ready(contract, output_root)
    primary_results = _collect_task_results(contract, output_root, "primary")
    secondary_results = _collect_task_results(contract, output_root, "secondary")
    transfer_results = _collect_transfer(contract, output_root)
    dense = read_jsonl(output_root / "states/dense_state_manifest.jsonl")
    routed = read_jsonl(output_root / "states/routed_state_manifest.jsonl")
    primary_metrics = read_csv(output_root / "work/all_primary_metrics.csv")
    selection = read_json(output_root / "work/secondary_selection.json")["selection"]
    prediction_sets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for target in ("read", "write"):
        for condition in (*CONDITIONS, "token_comparator"):
            model = "token_comparator" if condition == "token_comparator" else "mlp"
            rows = [row for row in read_jsonl(output_root / f"{target}/{condition}/oof_predictions.jsonl") if row["model"] == model]
            by_state = {str(row["state_id"]): row for row in dense}
            for row in rows:
                source = by_state[str(row["state_id"])]
                row.update({"image_group_id": source["image_group_id"], "dataset": source["dataset"], "source_regime": source["source_regime"], "dense_wrong": source["dense_wrong"], "layer": source["layer"], "trigger_layer": source["trigger_layer"], "trigger_relative_depth": source["trigger_relative_depth"]})
            validate_prediction_census([str(row["state_id"]) for row in dense], rows)
            prediction_sets[(target, condition)] = rows
    secondary_grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for task, result in secondary_results:
        secondary_grouped[(str(task["target"]), str(task["family"]))].append((task, result))
    routed_metrics = []
    routed_predictions = {}
    for key, entries in sorted(secondary_grouped.items()):
        prediction, seeds = _oof_predictions(routed, entries)
        target, family = key
        truth = np.asarray([float(row[f"u_{target}"]) for row in routed])
        routed_metrics.append({"target": target, "family": family, "condition": entries[0][0]["condition"], **regression_metrics(truth=truth, prediction=prediction), **{f"harmful_{key_}": value for key_, value in harmful_ranking_metrics(truth=truth, prediction=prediction).items() if key_ in {"auroc", "auprc"}}})
        routed_predictions[key] = prediction
    transfer_grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for task, result in transfer_results:
        transfer_grouped[(str(task["target"]), str(task["family"]))].append((task, result))
    transfer_metrics = []
    for key, entries in sorted(transfer_grouped.items()):
        prediction, _ = _transfer_oof(routed, entries)
        target, family = key
        truth = np.asarray([float(row[f"u_{target}"]) for row in routed])
        transfer_metrics.append({"target": target, "family": family, "condition": entries[0][0]["condition"], **regression_metrics(truth=truth, prediction=prediction)})
    atomic_csv(output_root / "routed_secondary/metrics_read.csv", [row for row in routed_metrics if row["target"] == "read"])
    atomic_csv(output_root / "routed_secondary/metrics_write.csv", [row for row in routed_metrics if row["target"] == "write"])
    atomic_csv(output_root / "routed_secondary/dense_to_routed_transfer.csv", transfer_metrics)

    conditional = []
    layer_rows = []
    trigger_rows = []
    for (target, condition), rows in prediction_sets.items():
        for outcome in (False, True):
            mask = np.asarray([bool(row["dense_wrong"]) == outcome for row in rows])
            conditional.append({"target": target, "condition": condition, "dense_outcome": "W" if outcome else "C", **_metric_subset(rows, mask)})
        for layer in range(28):
            mask = np.asarray([int(row["layer"]) == layer for row in rows])
            if int(mask.sum()) >= int(config["evaluation"]["minimum_breakdown_rows"]):
                layer_rows.append({"target": target, "condition": condition, "breakdown": "exact_layer", "value": layer, **_metric_subset(rows, mask)})
        for name, bounds in config["evaluation"]["depth_bins"].items():
            mask = np.asarray([int(bounds[0]) <= int(row["layer"]) <= int(bounds[1]) for row in rows])
            layer_rows.append({"target": target, "condition": condition, "breakdown": "depth_bin", "value": name, **_metric_subset(rows, mask)})
        for name, bounds in config["evaluation"]["trigger_relative_bins"].items():
            mask = np.asarray([int(bounds[0]) <= int(row["trigger_relative_depth"]) <= int(bounds[1]) for row in rows])
            trigger_rows.append({"target": target, "condition": condition, "value": name, **_metric_subset(rows, mask)})
    atomic_csv(output_root / "controls/dense_cw_conditional.csv", conditional)
    atomic_csv(output_root / "controls/layer_breakdown.csv", layer_rows)
    atomic_csv(output_root / "controls/trigger_relative_breakdown.csv", trigger_rows)

    random_rows = [row for row in primary_metrics if row["family"] == "random_pair"]
    swapped_rows = [row for row in primary_metrics if row["family"] == "swapped_order"]
    atomic_csv(output_root / "controls/random_pair_control.csv", random_rows)
    atomic_csv(output_root / "controls/branch_order_control.csv", swapped_rows)
    flips = {target: {str(row["state_id"]): row for row in read_csv(config["sources"]["dense_flips"])} for target in ("read", "write")}
    flip_metrics = []
    for target in ("read", "write"):
        condition = str(selection[target]["condition"])
        rows = prediction_sets[(target, condition)]
        if target == "read":
            harmful_col, beneficial_col = "read_harmful_flip_w1", "read_beneficial_flip_w1"
        else:
            harmful_col, beneficial_col = "write_harmful_flip_r1", "write_beneficial_flip_r1"
        prediction = np.asarray([float(row["prediction"]) for row in rows])
        for label_name, column, score in (("harmful_flip", harmful_col, -prediction), ("beneficial_flip", beneficial_col, prediction)):
            truth = np.asarray([str(flips[target][str(row["state_id"])][column]).lower() == "true" for row in rows], dtype=np.int64)
            flip_metrics.append({"target": target, "condition": condition, "label": label_name, **binary_classification_metrics(truth=truth, prediction=score)})
    atomic_csv(output_root / "controls/strong_correctness_flip_analysis.csv", flip_metrics)

    draws = int(config["evaluation"]["bootstrap_draws"])
    bootstrap = []
    pairwise = []
    seed_base = int(config["evaluation"]["bootstrap_seed"])
    counter = 0
    for target in ("read", "write"):
        for condition in (*CONDITIONS, "token_comparator"):
            rows = prediction_sets[(target, condition)]
            bootstrap.append({"target": target, "condition": condition, "model": "token_comparator" if condition == "token_comparator" else "mlp", "metric": "spearman", **_bootstrap_rows(rows, draws=draws, seed=seed_base + counter)})
            counter += 1
        best_single = max(("full_post", "off_post"), key=lambda condition: float(next(row["spearman"] for row in primary_metrics if row["target"] == target and row["family"] == "primary" and row["model"] == "mlp" and row["condition"] == condition)))
        for left, right in (("delta", "pre"), ("delta", best_single), ("token_comparator", str(selection[target]["condition"]))):
            pairwise.append({"target": target, "left": left, "right": right, "metric": "spearman_difference", **_pairwise_bootstrap(prediction_sets[(target, left)], prediction_sets[(target, right)], draws=draws, seed=seed_base + counter)})
            counter += 1
    atomic_csv(output_root / "statistics/uid_bootstrap_ci.csv", bootstrap)
    atomic_csv(output_root / "statistics/pairwise_model_differences.csv", pairwise)

    case_results = {}
    for target in ("read", "write"):
        def rho(condition: str) -> float:
            model = "token_comparator" if condition == "token_comparator" else "mlp"
            family = "token" if condition == "token_comparator" else "primary"
            return float(next(row["spearman"] for row in primary_metrics if row["target"] == target and row["family"] == family and row["condition"] == condition and row["model"] == model))
        random_best = max(float(row["spearman"]) for row in random_rows if row["target"] == target)
        case, reason = classify_case(
            pre=rho("pre"), full_post=rho("full_post"), off_post=rho("off_post"),
            pair=rho("pair"), delta=rho("delta"), pair_plus_delta=rho("pair_plus_delta"),
            token=rho("token_comparator"), random_pair_best=random_best, thresholds=config["decision"],
        )
        case_results[target] = {"case": case, "reason": reason}
    joint_case = case_results["read"]["case"] if case_results["read"]["case"] == case_results["write"]["case"] else "D"
    joint_reason = "both targets agree" if case_results["read"]["case"] == case_results["write"]["case"] else "READ and WRITE do not support one common positive one-step pattern"
    atomic_json(output_root / "summaries/decision_category.json", {"contract_sha256": contract["contract_sha256"], "read": case_results["read"], "write": case_results["write"], "joint_case": joint_case, "joint_reason": joint_reason})
    _plot_primary(output_root, primary_metrics)

    # Remaining required figures use already frozen aggregate rows only.
    for target in ("read", "write"):
        fig, ax = plt.subplots(figsize=(6, 4))
        conditions = [*CONDITIONS, "token_comparator"]
        precisions = []
        for condition in conditions:
            model = "token_comparator" if condition == "token_comparator" else "mlp"
            family = "token" if condition == "token_comparator" else "primary"
            row = next(row for row in primary_metrics if row["target"] == target and row["condition"] == condition and row["model"] == model and row["family"] == family)
            precisions.append(float(row["precision_at_0.1"]))
        ax.bar(conditions, precisions)
        ax.set_ylabel("Harmful precision @ 10%")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output_root / f"figures/harmful_precision_coverage_{target}.png", dpi=180)
        plt.close(fig)
    ablation = [row for row in primary_metrics if row["family"] == "ablation"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    names = list(config["features"]["delta_ablation_conditions"])
    for offset, target in ((-.18, "read"), (.18, "write")):
        values = [float(next(row["spearman"] for row in ablation if row["target"] == target and row["condition"] == name)) for name in names]
        ax.bar(x + offset, values, width=.36, label=target.upper())
    ax.set_xticks(x, names)
    ax.set_ylabel("OOF Spearman")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "figures/text_vs_visual_delta.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for target in ("read", "write"):
        condition = str(selection[target]["condition"])
        subset = [row for row in layer_rows if row["target"] == target and row["condition"] == condition and row["breakdown"] == "exact_layer"]
        ax.plot([int(row["value"]) for row in subset], [float(row["spearman"]) for row in subset], marker="o", label=f"{target.upper()} {condition}")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "figures/layerwise_counterfactual_identifiability.png", dpi=180)
    plt.close(fig)

    def metric(target: str, condition: str, model: str = "mlp", family: str = "primary") -> Mapping[str, Any]:
        return next(row for row in primary_metrics if row["target"] == target and row["condition"] == condition and row["model"] == model and row["family"] == family)
    read_rows = {condition: metric("read", condition) for condition in CONDITIONS}
    read_rows["token_comparator"] = metric("read", "token_comparator", "token_comparator", "token")
    write_rows = {condition: metric("write", condition) for condition in CONDITIONS}
    write_rows["token_comparator"] = metric("write", "token_comparator", "token_comparator", "token")
    summary = f"""# Counterfactual effect identifiability summary

## Validity and population

1. Primary population: **{len(dense):,} states / {len(set(row['uid'] for row in dense)):,} UIDs / {len(set(row['image_group_id'] for row in dense)):,} image groups**. Secondary routed population: **{len(routed):,} states / {len(set(row['uid'] for row in routed)):,} UIDs**.
2. Exact same-prestate/action/repeat parity passed for the frozen smoke and complete extraction.
3. FULL post-state exactly reproduced the canonical Dense next-layer state for all {len(dense):,} primary states, including layer 27's actual final-layer output; no synthetic layer 28 was created.

## READ primary OOF Spearman

| PRE | FULL post | WO post | Pair | Delta | Pair+Delta | Token comparator |
|---:|---:|---:|---:|---:|---:|---:|
| {float(read_rows['pre']['spearman']):.4f} | {float(read_rows['full_post']['spearman']):.4f} | {float(read_rows['off_post']['spearman']):.4f} | {float(read_rows['pair']['spearman']):.4f} | {float(read_rows['delta']['spearman']):.4f} | {float(read_rows['pair_plus_delta']['spearman']):.4f} | {float(read_rows['token_comparator']['spearman']):.4f} |

## WRITE primary OOF Spearman

| PRE | FULL post | RO post | Pair | Delta | Pair+Delta | Token comparator |
|---:|---:|---:|---:|---:|---:|---:|
| {float(write_rows['pre']['spearman']):.4f} | {float(write_rows['full_post']['spearman']):.4f} | {float(write_rows['off_post']['spearman']):.4f} | {float(write_rows['pair']['spearman']):.4f} | {float(write_rows['delta']['spearman']):.4f} | {float(write_rows['pair_plus_delta']['spearman']):.4f} | {float(write_rows['token_comparator']['spearman']):.4f} |

## Answers to the remaining protocol questions

11. The single-post versus explicit-counterfactual conclusion follows the frozen cases below; no architecture is credited without its same-capacity controls.
12. READ is **Case {case_results['read']['case']}** ({case_results['read']['reason']}); WRITE is **Case {case_results['write']['case']}** ({case_results['write']['reason']}).
13. Linear and MLP results are reported side-by-side in `read/metrics.csv` and `write/metrics.csv`.
14. The token-aware comparison is reported above and uses a fixed shared projection/attention pair comparator.
15-16. Text-only, visual-only, and combined delta ablations are in each target's `text_visual_delta_ablation.csv`; they are treated as hypotheses, not assumed stream specialization.
17. Dense-C and Dense-W results are in `controls/dense_cw_conditional.csv`.
18-19. Exact-layer, depth-bin, and trigger-relative results are in `controls/layer_breakdown.csv` and `controls/trigger_relative_breakdown.csv`.
20. The matched random-pair control is in `controls/random_pair_control.csv`; its registry records all match relaxations prospectively.
21. The swapped-order diagnostic is in `controls/branch_order_control.csv`; primary signs remain FULL-minus-OFF and are never averaged across orders.
22. Routed-state OOF results are in `routed_secondary/metrics_read.csv` and `metrics_write.csv`.
23. Dense-trained routed-state transfer is in `routed_secondary/dense_to_routed_transfer.csv` with the inherited group-disjoint folds.
24. The joint result is **Case {joint_case}**: {joint_reason}. Per-target categories are retained rather than hidden by pooling.
25. This experiment does not establish benchmark gain, compute savings, external transfer, causal correctness, or global optimality of one-step probing.
"""
    (output_root / "summaries").mkdir(parents=True, exist_ok=True)
    (output_root / "summaries/counterfactual_effect_identifiability_summary.md").write_text(summary)
    recommendations = {
        "A": "one-layer counterfactual probe-and-route Stage-2",
        "B": "one-layer post-action verification / delayed-routing controller",
        "C": "token-aware one-layer counterfactual comparator",
        "D": "2-layer / short-horizon counterfactual identifiability audit",
    }
    recommendation = recommendations[joint_case]
    recommendation_md = f"""# Next-method recommendation

Recommend exactly one next experiment: **{recommendation}**.

This is the smallest follow-up implied by joint Case {joint_case}: {joint_reason}. Positive evidence would show that the selected additional horizon/representation exposes stable utility information beyond the frozen one-step controls. Negative evidence would rule out that bounded extension and argue against investing in a deployment controller of this family. Benchmark gain, external transfer, compute savings, and causal optimality remain unproven.

This recommendation is not authorization to run it.
"""
    (output_root / "summaries/next_method_recommendation.md").write_text(recommendation_md)
    manifest_files = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or "work" in path.relative_to(output_root).parts or path.name == "artifact_manifest.json":
            continue
        manifest_files[str(path.relative_to(output_root))] = file_sha256(path)
    external_files = {}
    for domain in ("dense", "routed"):
        cache = read_json(output_root / f"features/{domain}_cache_manifest.json")
        external_files.update({str(spec["path"]): str(spec["sha256"]) for spec in cache["files"].values()})
    artifact = {"schema_version": "counterfactual_effect_identifiability_artifact_manifest_v1", "contract_sha256": contract["contract_sha256"], "joint_case": joint_case, "read_case": case_results["read"]["case"], "write_case": case_results["write"]["case"], "files": manifest_files, "external_files_sha256": external_files}
    artifact["artifact_manifest_sha256"] = canonical_hash(artifact)
    atomic_json(output_root / "artifact_manifest.json", artifact)
    print(json.dumps({"complete": True, "contract_sha256": contract["contract_sha256"], "artifact_manifest_sha256": artifact["artifact_manifest_sha256"], "joint_case": joint_case, "read_case": case_results["read"]["case"], "write_case": case_results["write"]["case"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare-training", "train", "aggregate-primary", "transfer", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("primary", "secondary"), default="primary")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-training":
        prepare_training(args.config)
    elif args.command == "train":
        train_worker(args.config, args.phase, args.rank, resume=args.resume)
    elif args.command == "aggregate-primary":
        aggregate_primary(args.config)
    elif args.command == "transfer":
        dense_to_routed_worker(args.config, args.rank, resume=args.resume)
    else:
        finalize(args.config)


if __name__ == "__main__":
    main()
