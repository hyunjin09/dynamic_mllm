#!/usr/bin/env python3
"""Train and aggregate the frozen READ short-horizon propagation study."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from hashlib import sha256
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

from dense_failure_stage2.counterfactual_identifiability import PairedTokenComparator
from dense_failure_stage2.predictability_learnability import (
    SummaryScalarPredictor,
    binary_classification_metrics,
    regression_metrics,
    robust_target_scale,
)
from dense_failure_stage2.read_short_horizon import (
    HORIZONS,
    classify_h_read,
    construct_horizon_feature,
    matched_random_pair_indices,
    monotonic_nondecreasing,
)
from experiments.run_counterfactual_effect_identifiability import (
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
)
from experiments.run_read_short_horizon_propagation import DEFAULT_CONFIG, verify_contract


def _canonical_json_hash(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


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


def _task_id(task: Mapping[str, Any]) -> str:
    fields = ("support", "horizon", "family", "condition", "model", "fold", "seed")
    return "__".join(str(task[field]) for field in fields)


def build_task_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the fully frozen fold/seed grid used by tests and execution."""

    conditions = [
        *config["features"]["pooled_conditions"],
        *config["features"]["delta_conditions"],
    ]
    folds = range(int(config["split"]["folds"]))
    seeds = [int(value) for value in config["training"]["seeds"]]
    cells: list[dict[str, Any]] = []
    for support in ("native", "common_h8"):
        for horizon in HORIZONS:
            for condition in conditions:
                for model in ("linear", "mlp"):
                    cells.append({
                        "support": support, "horizon": horizon, "family": "standard",
                        "condition": condition, "model": model,
                    })
            cells.append({
                "support": support, "horizon": horizon, "family": "standard",
                "condition": "token_comparator", "model": "token_comparator",
            })
    for horizon in config["controls"]["random_pair_horizons"]:
        cells.append({
            "support": "common_h8", "horizon": int(horizon), "family": "random_pair",
            "condition": str(config["controls"]["random_pair_condition"]), "model": "mlp",
        })
    for horizon in config["controls"]["permuted_target_horizons"]:
        cells.append({
            "support": "common_h8", "horizon": int(horizon), "family": "permuted_target",
            "condition": str(config["controls"]["permuted_target_condition"]), "model": "mlp",
        })
    output = []
    task_index = 0
    for cell in cells:
        for fold in folds:
            for seed in seeds:
                task = {"task_index": task_index, **cell, "fold": fold, "seed": seed}
                task["task_id"] = _task_id(task)
                output.append(task)
                task_index += 1
    return output


def _assign_tasks(tasks: Sequence[Mapping[str, Any]], world_size: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        groups[(task["support"], task["horizon"], task["family"], task["condition"], task["model"])].append(task)
    loads = [0] * int(world_size)
    output = []
    # Keep the 15 related fits together so feature/cache reads are reused.
    for key, members in sorted(groups.items(), key=lambda item: (-len(item[1]), str(item[0]))):
        rank = min(range(int(world_size)), key=lambda value: (loads[value], value))
        loads[rank] += len(members)
        output.extend({**dict(member), "worker_rank": rank} for member in members)
    return sorted(output, key=lambda row: (int(row["worker_rank"]), int(row["task_index"])))


class HorizonStore:
    """Read one native/common-support horizon view from the finalized raw caches."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        all_feature_rows: Sequence[Mapping[str, Any]],
        *,
        horizon: int,
        support: str,
    ) -> None:
        self.horizon = int(horizon)
        self.support = str(support)
        self.hidden = int(manifest["hidden_size"])
        selected = [
            dict(row) for row in all_feature_rows
            if int(row["horizon"]) == self.horizon
            and (self.support == "native" or bool(row["common_h8"]))
        ]
        if self.support not in {"native", "common_h8"} or not selected:
            raise RuntimeError(f"unsupported or empty horizon view: {support}/H={horizon}")
        self.rows = sorted(selected, key=lambda row: str(row["state_id"]))
        self.cache_indices = np.asarray([int(row["horizon_row_index"]) for row in self.rows], dtype=np.int64)
        specs = manifest["horizons"][str(self.horizon)]["files"]
        self.pooled = np.memmap(
            resolve_path(specs["pooled"]["path"]), dtype=np.uint16, mode="r",
            shape=tuple(specs["pooled"]["shape"]),
        )
        self.text = np.memmap(
            resolve_path(specs["text"]["path"]), dtype=np.uint16, mode="r",
            shape=tuple(specs["text"]["shape"]),
        )
        self.visual = np.memmap(
            resolve_path(specs["visual"]["path"]), dtype=np.uint16, mode="r",
            shape=tuple(specs["visual"]["shape"]),
        )

    def summaries(self, local_indices: Sequence[int] | np.ndarray) -> torch.Tensor:
        local = np.asarray(local_indices, dtype=np.int64)
        return _u16_to_bf16(self.pooled[self.cache_indices[local]])

    def summary_feature(
        self,
        local_indices: Sequence[int] | np.ndarray,
        *,
        condition: str,
        random_pairs: np.ndarray | None = None,
    ) -> torch.Tensor:
        local = np.asarray(local_indices, dtype=np.int64)
        values = self.summaries(local).float()
        on = values[:, 0]
        if random_pairs is None:
            off = values[:, 1]
        else:
            donor = np.asarray(random_pairs, dtype=np.int64)[local]
            off = self.summaries(donor)[:, 1].float()
        return construct_horizon_feature(on, off, str(condition))

    def collate_pair(
        self,
        local_indices: Sequence[int] | np.ndarray,
        *,
        pin_memory: bool,
    ) -> tuple[torch.Tensor, ...]:
        local = np.asarray(local_indices, dtype=np.int64)
        selected = [self.rows[int(index)] for index in local]
        max_visual = max(int(row["visual_tokens"]) for row in selected)
        batch = len(selected)
        on_text = torch.empty((batch, 1, self.hidden), dtype=torch.bfloat16, pin_memory=pin_memory)
        off_text = torch.empty_like(on_text)
        on_visual = torch.zeros((batch, max_visual, self.hidden), dtype=torch.bfloat16, pin_memory=pin_memory)
        off_visual = torch.zeros_like(on_visual)
        text_mask = torch.ones((batch, 1), dtype=torch.bool, pin_memory=pin_memory)
        visual_mask = torch.zeros((batch, max_visual), dtype=torch.bool, pin_memory=pin_memory)
        for output_index, (index, row) in enumerate(zip(local, selected)):
            cache_index = int(self.cache_indices[int(index)])
            offset = int(row["visual_offset"])
            count = int(row["visual_tokens"])
            on_text[output_index, 0] = _u16_to_bf16(self.text[cache_index, 0])
            off_text[output_index, 0] = _u16_to_bf16(self.text[cache_index, 1])
            on_visual[output_index, :count] = _u16_to_bf16(self.visual[offset : offset + count, 0])
            off_visual[output_index, :count] = _u16_to_bf16(self.visual[offset : offset + count, 1])
            visual_mask[output_index, :count] = True
        return on_text, on_visual, off_text, off_visual, text_mask, visual_mask


def _verify_extraction(contract: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    complete = read_json(output_root / "work/extraction_complete.json")
    manifest_path = output_root / "features/cache_manifest.json"
    manifest = read_json(manifest_path)
    if (
        not complete.get("complete")
        or complete.get("contract_sha256") != contract["contract_sha256"]
        or complete.get("cache_manifest_sha256") != file_sha256(manifest_path)
        or manifest.get("contract_sha256") != contract["contract_sha256"]
    ):
        raise RuntimeError("short-horizon extraction is not complete under this contract")
    for horizon in HORIZONS:
        for spec in manifest["horizons"][str(horizon)]["files"].values():
            path = resolve_path(spec["path"])
            if not path.is_file() or path.stat().st_size != int(spec["bytes"]):
                raise RuntimeError(f"short-horizon cache differs: {path}")
    return manifest


def prepare_training(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    config = contract["static_config"]
    manifest = _verify_extraction(contract, output_root)
    feature_rows = read_jsonl(output_root / "features/pooled_horizon_features.jsonl")
    expected = sum(int(value) for value in contract["population"]["native_support"].values())
    if len(feature_rows) != expected:
        raise RuntimeError("feature-row census differs")
    random_registry = []
    common = [dict(row) for row in feature_rows if int(row["horizon"]) == 1 and bool(row["common_h8"])]
    common.sort(key=lambda row: str(row["state_id"]))
    pairs = matched_random_pair_indices(common, seed=int(config["controls"]["random_pair_seed"]))
    for index, donor in enumerate(pairs):
        random_registry.append({
            "row_index": index, "state_id": common[index]["state_id"], "uid": common[index]["uid"],
            "paired_row_index": int(donor), "paired_state_id": common[int(donor)]["state_id"],
            "paired_uid": common[int(donor)]["uid"],
        })
    if any(row["uid"] == row["paired_uid"] for row in random_registry):
        raise RuntimeError("random-pair registry contains a same-UID pair")
    atomic_jsonl(output_root / "controls/random_pair_registry.jsonl", random_registry)
    tasks = _assign_tasks(build_task_grid(config), int(config["world_size"]))
    atomic_jsonl(output_root / "work/training_tasks.jsonl", tasks)
    ready = {
        "schema_version": "read_short_horizon_training_ready_v1",
        "contract_sha256": contract["contract_sha256"],
        "cache_manifest_sha256": file_sha256(output_root / "features/cache_manifest.json"),
        "feature_manifest_sha256": file_sha256(output_root / "features/pooled_horizon_features.jsonl"),
        "cache_mtime_ns": {
            f"h{horizon}/{name}": resolve_path(spec["path"]).stat().st_mtime_ns
            for horizon in HORIZONS
            for name, spec in manifest["horizons"][str(horizon)]["files"].items()
        },
        "tasks": len(tasks),
        "tasks_by_rank": dict(Counter(str(task["worker_rank"]) for task in tasks)),
    }
    atomic_json(output_root / "work/training_ready.json", ready)
    print(json.dumps(ready, sort_keys=True))


def _require_training_ready(contract: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    ready = read_json(output_root / "work/training_ready.json")
    manifest_path = output_root / "features/cache_manifest.json"
    feature_path = output_root / "features/pooled_horizon_features.jsonl"
    if (
        ready.get("contract_sha256") != contract["contract_sha256"]
        or ready.get("cache_manifest_sha256") != file_sha256(manifest_path)
        or ready.get("feature_manifest_sha256") != file_sha256(feature_path)
    ):
        raise RuntimeError("training-ready provenance differs")
    manifest = read_json(manifest_path)
    for horizon in HORIZONS:
        for name, spec in manifest["horizons"][str(horizon)]["files"].items():
            path = resolve_path(spec["path"])
            if (
                path.stat().st_size != int(spec["bytes"])
                or path.stat().st_mtime_ns != int(ready["cache_mtime_ns"][f"h{horizon}/{name}"])
            ):
                raise RuntimeError(f"cache changed after training preparation: H={horizon}/{name}")
    return ready


def _role_indices(output_root: Path, rows: Sequence[Mapping[str, Any]], fold: int) -> dict[str, np.ndarray]:
    roles = {
        str(row["uid"]): str(row["role"])
        for row in read_jsonl(output_root / f"splits/inner_roles_fold{int(fold)}.jsonl")
    }
    output = {}
    for role in ("fit", "calibration", "outer_test"):
        output[role] = np.asarray(
            [index for index, row in enumerate(rows) if roles[str(row["uid"])] == role],
            dtype=np.int64,
        )
        if len(output[role]) == 0:
            raise RuntimeError(f"fold {fold} has no {role} rows")
    return output


def _uid_weights(rows: Sequence[Mapping[str, Any]], indices: np.ndarray) -> np.ndarray:
    counts = Counter(str(rows[int(index)]["uid"]) for index in indices)
    weights = np.asarray([1.0 / counts[str(rows[int(index)]["uid"])] for index in indices], dtype=np.float64)
    return weights / weights.mean()


def _permuted_targets_for_role(
    target: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    subset = [rows[int(index)] for index in indices]
    donors = matched_random_pair_indices(subset, seed=int(seed), uid_permutation=True)
    return target[indices[np.asarray(donors, dtype=np.int64)]]


def _feature_width(condition: str, hidden: int) -> int:
    if condition in {"on", "off", "delta", "text_visual_delta"}:
        return 2 * hidden
    if condition == "pair":
        return 4 * hidden
    if condition == "pair_plus_delta":
        return 6 * hidden
    if condition in {"text_delta", "visual_delta"}:
        return hidden
    raise ValueError(f"unknown pooled condition: {condition}")


def _model(task: Mapping[str, Any], config: Mapping[str, Any], hidden: int) -> torch.nn.Module:
    if task["model"] in {"linear", "mlp"}:
        spec = config["training"][str(task["model"])]
        return SummaryScalarPredictor(
            kind=str(task["model"]), input_size=_feature_width(str(task["condition"]), hidden),
            hidden_size=int(spec.get("hidden_size", 1)), dropout=float(spec.get("dropout", 0.0)),
        )
    spec = config["features"]["token_comparator"]
    return PairedTokenComparator(
        hidden_size=hidden, projection_size=int(spec["projection_size"]),
        attention_heads=int(spec["attention_heads"]),
        readout_hidden_size=int(spec["readout_hidden_size"]), dropout=float(spec["dropout"]),
    )


def _spec(task: Mapping[str, Any], config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["training"][str(task["model"])]


def _summary_standardizer(
    store: HorizonStore,
    indices: np.ndarray,
    task: Mapping[str, Any],
    random_pairs: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = _feature_width(str(task["condition"]), store.hidden)
    total = torch.zeros(width, dtype=torch.float64)
    square = torch.zeros(width, dtype=torch.float64)
    count = 0
    for start in range(0, len(indices), 512):
        batch = indices[start : start + 512]
        features = store.summary_feature(
            batch, condition=str(task["condition"]),
            random_pairs=random_pairs if task["family"] == "random_pair" else None,
        ).to(device=device, dtype=torch.float32)
        total += features.sum(dim=0).double().cpu()
        square += features.square().sum(dim=0).double().cpu()
        count += len(batch)
    mean = total / count
    variance = torch.clamp(square / count - mean.square(), min=0.0)
    std = torch.where(variance.sqrt() < 1e-6, torch.ones_like(mean), variance.sqrt())
    return mean.float(), std.float()


def _token_forward(model: torch.nn.Module, packed: tuple[torch.Tensor, ...], device: torch.device) -> torch.Tensor:
    on_text, on_visual, off_text, off_visual, text_mask, visual_mask = packed
    non_blocking = bool(on_text.is_pinned())
    return model(
        on_text.to(device, non_blocking=non_blocking),
        on_visual.to(device, non_blocking=non_blocking),
        off_text.to(device, non_blocking=non_blocking),
        off_visual.to(device, non_blocking=non_blocking),
        text_mask=text_mask.to(device, non_blocking=non_blocking),
        visual_mask=visual_mask.to(device, non_blocking=non_blocking),
    )


def _iter_token_batches(
    store: HorizonStore, batches: Iterable[np.ndarray]
) -> Iterable[tuple[np.ndarray, tuple[torch.Tensor, ...]]]:
    iterator = iter(batches)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="horizon-token") as executor:
        try:
            current = np.asarray(next(iterator), dtype=np.int64)
        except StopIteration:
            return
        future: Future[tuple[torch.Tensor, ...]] = executor.submit(
            store.collate_pair, current, pin_memory=True
        )
        while True:
            packed = future.result()
            try:
                following = np.asarray(next(iterator), dtype=np.int64)
            except StopIteration:
                yield current, packed
                break
            future = executor.submit(store.collate_pair, following, pin_memory=True)
            yield current, packed
            current = following


def _predict_raw(
    model: torch.nn.Module,
    task: Mapping[str, Any],
    store: HorizonStore,
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
            for _, packed in _iter_token_batches(store, batches):
                outputs.append(_token_forward(model, packed, device).float().cpu())
        else:
            if mean is None or std is None:
                raise RuntimeError("pooled prediction lacks a fold-local standardizer")
            mean_device, std_device = mean.to(device), std.to(device)
            for batch in batches:
                values = store.summary_feature(
                    batch, condition=str(task["condition"]),
                    random_pairs=random_pairs if task["family"] == "random_pair" else None,
                ).to(device=device, dtype=torch.float32)
                outputs.append(model((values - mean_device) / std_device).float().cpu())
    return torch.cat(outputs).numpy().astype(np.float64)


def _train_task(
    contract: Mapping[str, Any],
    output_root: Path,
    external_root: Path,
    task: Mapping[str, Any],
    store: HorizonStore,
    random_pairs: np.ndarray,
    device: torch.device,
    stats_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    config = contract["static_config"]
    configure_determinism(int(task["seed"]))
    rows = store.rows
    roles = _role_indices(output_root, rows, int(task["fold"]))
    h_r = np.asarray([float(row["h_r"]) for row in rows], dtype=np.float64)
    # Exact Phase-82 target parameterization: fit u_read=-H_R, invert at readout.
    internal_target = -h_r
    fit_target = internal_target[roles["fit"]]
    calibration_target = internal_target[roles["calibration"]]
    if task["family"] == "permuted_target":
        permutation_seed = int(config["controls"]["permuted_target_seed"]) + int(task["fold"])
        fit_target = _permuted_targets_for_role(
            internal_target, rows, roles["fit"], seed=permutation_seed
        )
        calibration_target = _permuted_targets_for_role(
            internal_target, rows, roles["calibration"], seed=permutation_seed + 1000
        )
    scale = robust_target_scale(
        fit_target, floor=float(config["training"]["target_scale_floor"])
    )
    scaled_fit = scale.transform(fit_target)
    scaled_calibration = scale.transform(calibration_target)
    model = _model(task, config, store.hidden).to(device=device, dtype=torch.float32)
    spec = _spec(task, config)
    batch_size = int(spec.get("batch_size", spec.get("state_microbatch")))
    if task["model"] == "token_comparator":
        mean = std = None
    else:
        stats_key = (
            task["support"], task["horizon"], task["family"], task["condition"], task["fold"]
        )
        if stats_key not in stats_cache:
            stats_cache[stats_key] = _summary_standardizer(
                store, roles["fit"], task, random_pairs, device
            )
        mean, std = stats_cache[stats_key]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
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
        order = torch.randperm(len(roles["fit"]), generator=generator).numpy()
        batches = [
            roles["fit"][order[start : start + batch_size]]
            for start in range(0, len(order), batch_size)
        ]
        batch_weights = [
            fit_weights[order[start : start + batch_size]]
            for start in range(0, len(order), batch_size)
        ]
        if task["model"] == "token_comparator":
            iterator = _iter_token_batches(store, batches)
        else:
            iterator = ((batch, None) for batch in batches)
        epoch_sum = 0.0
        epoch_weight = 0.0
        fit_position = {int(index): position for position, index in enumerate(roles["fit"])}
        for batch_number, (indices, packed) in enumerate(iterator):
            local_positions = np.asarray([fit_position[int(index)] for index in indices], dtype=np.int64)
            truth = torch.tensor(scaled_fit[local_positions], dtype=torch.float32, device=device)
            weights = torch.tensor(batch_weights[batch_number], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            if task["model"] == "token_comparator":
                prediction = _token_forward(model, packed, device)
            else:
                values = store.summary_feature(
                    indices, condition=str(task["condition"]),
                    random_pairs=random_pairs if task["family"] == "random_pair" else None,
                ).to(device=device, dtype=torch.float32)
                prediction = model((values - mean.to(device)) / std.to(device))
            losses = torch.nn.functional.huber_loss(
                prediction, truth, delta=1.0, reduction="none"
            )
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            epoch_sum += float((losses.detach() * weights).sum().cpu())
            epoch_weight += float(weights.sum().cpu())
        calibration_raw = _predict_raw(
            model, task, store, roles["calibration"], mean=mean, std=std,
            random_pairs=random_pairs, device=device, batch_size=batch_size,
        )
        calibration_losses = torch.nn.functional.huber_loss(
            torch.tensor(calibration_raw), torch.tensor(scaled_calibration),
            delta=1.0, reduction="none",
        ).numpy()
        calibration_loss = float(
            np.dot(calibration_losses, calibration_weights) / calibration_weights.sum()
        )
        history.append({
            "epoch": epoch + 1,
            "fit_loss": epoch_sum / epoch_weight,
            "calibration_loss": calibration_loss,
        })
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= int(spec["minimum_epochs"]) and stale >= int(spec["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"training produced no checkpoint: {task['task_id']}")
    model.load_state_dict(best_state)
    test_raw = _predict_raw(
        model, task, store, roles["outer_test"], mean=mean, std=std,
        random_pairs=random_pairs, device=device, batch_size=batch_size,
    )
    test_prediction = -scale.inverse(test_raw)
    checkpoint = external_root / "models" / f"{task['task_id']}.pt"
    payload = {
        "schema_version": "read_short_horizon_checkpoint_v1",
        "contract_sha256": contract["contract_sha256"],
        "task": dict(task), "task_sha256": canonical_hash(task),
        "model_state": best_state, "normalization_mean": mean, "normalization_std": std,
        "internal_target_center": scale.center, "internal_target_scale": scale.scale,
        "best_epoch": best_epoch, "best_calibration_loss": best_loss, "history": history,
        "test_indices": roles["outer_test"], "test_prediction_h_r": test_prediction,
    }
    atomic_torch(checkpoint, payload)
    return {
        "contract_sha256": contract["contract_sha256"],
        "task_id": task["task_id"], "task_sha256": canonical_hash(task),
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "best_epoch": best_epoch, "best_calibration_loss": best_loss,
        "test_states": len(roles["outer_test"]), "elapsed_seconds": time.monotonic() - started,
    }


def train_worker(config_path: Path, rank: int, device_index: int, *, resume: bool) -> None:
    contract, output_root, external_root = verify_contract(config_path)
    _require_training_ready(contract, output_root)
    config = contract["static_config"]
    configure_worker_runtime(config)
    rank = int(rank)
    torch.cuda.set_device(int(device_index))
    device = torch.device(f"cuda:{int(device_index)}")
    tasks = [
        row for row in read_jsonl(output_root / "work/training_tasks.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    manifest = read_json(output_root / "features/cache_manifest.json")
    feature_rows = read_jsonl(output_root / "features/pooled_horizon_features.jsonl")
    stores: dict[tuple[str, int], HorizonStore] = {}
    pairs: dict[tuple[str, int], np.ndarray] = {}
    rank_root = output_root / f"work/training/rank{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    stats_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    completed = 0
    started = time.monotonic()
    for task in tasks:
        result_path = rank_root / f"{task['task_id']}.json"
        if resume and result_path.is_file():
            old = read_json(result_path)
            checkpoint = resolve_path(old.get("checkpoint", ""))
            if (
                old.get("contract_sha256") == contract["contract_sha256"]
                and old.get("task_sha256") == canonical_hash(task)
                and checkpoint.is_file()
                and file_sha256(checkpoint) == old.get("checkpoint_sha256")
            ):
                completed += 1
                continue
        key = str(task["support"]), int(task["horizon"])
        if key not in stores:
            stores[key] = HorizonStore(
                manifest, feature_rows, horizon=key[1], support=key[0]
            )
            pairs[key] = matched_random_pair_indices(
                stores[key].rows, seed=int(config["controls"]["random_pair_seed"])
            )
        result = _train_task(
            contract, output_root, external_root, task, stores[key], pairs[key], device,
            stats_cache,
        )
        atomic_json(result_path, result)
        completed += 1
        torch.cuda.empty_cache()
        print(json.dumps({
            "rank": rank, "device_index": int(device_index), "completed": completed,
            "assigned": len(tasks), "task_id": task["task_id"],
            "elapsed_seconds": time.monotonic() - started,
        }), flush=True)
    atomic_json(rank_root / "complete.json", {
        "contract_sha256": contract["contract_sha256"], "rank": rank,
        "expected_tasks": len(tasks), "completed_tasks": completed,
    })


def _collect_results(
    contract: Mapping[str, Any], output_root: Path
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    tasks = read_jsonl(output_root / "work/training_tasks.jsonl")
    by_id = {str(task["task_id"]): task for task in tasks}
    output = []
    for rank in range(int(contract["static_config"]["world_size"])):
        root = output_root / f"work/training/rank{rank:02d}"
        complete = read_json(root / "complete.json")
        if (
            complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("expected_tasks") != complete.get("completed_tasks")
        ):
            raise RuntimeError(f"partial training rank: {rank}")
        for path in sorted(root.glob("*.json")):
            if path.name == "complete.json":
                continue
            result = read_json(path)
            task = by_id.get(str(result.get("task_id")))
            if task is None or result.get("task_sha256") != canonical_hash(task):
                raise RuntimeError(f"training result task differs: {path}")
            checkpoint = resolve_path(result["checkpoint"])
            if file_sha256(checkpoint) != result["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint differs: {checkpoint}")
            output.append((task, result))
    if len(output) != len(tasks) or len({task[0]["task_id"] for task in output}) != len(tasks):
        raise RuntimeError("global training task census differs")
    return output


def _metric_bundle(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if len(truth) < 2 or len(np.unique(truth)) < 2 or len(np.unique(prediction)) < 2:
        return {
            "support": int(len(truth)), "spearman": float("nan"), "pearson": float("nan"),
            "mae": float("nan"), "rmse": float("nan"), "harmful_auroc": float("nan"),
            "harmful_auprc": float("nan"), "harmful_prevalence": float(np.mean(truth > 0)) if len(truth) else float("nan"),
            "precision_at_0.05": float("nan"), "precision_at_0.1": float("nan"),
            "precision_at_0.2": float("nan"), "recall_at_precision_0.9": 0.0,
            "recall_at_precision_0.95": 0.0,
        }
    result = dict(regression_metrics(truth=truth, prediction=prediction))
    mask = truth != 0.0
    labels = (truth[mask] > 0.0).astype(np.int64)
    scores = prediction[mask]
    classification = binary_classification_metrics(truth=labels, prediction=scores)
    result.update({
        "harmful_auroc": classification["auroc"],
        "harmful_auprc": classification["auprc"],
        "harmful_prevalence": classification["prevalence"],
    })
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    for coverage in (0.05, 0.1, 0.2):
        count = max(1, int(math.ceil(coverage * len(ordered))))
        result[f"precision_at_{coverage:g}"] = float(ordered[:count].mean())
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(ordered) + 1)
    recall = cumulative / max(1, int(labels.sum()))
    for target in (0.9, 0.95):
        valid = precision >= target
        result[f"recall_at_precision_{target:g}"] = float(recall[valid].max()) if valid.any() else 0.0
    return result


def _oof_predictions(
    rows: Sequence[Mapping[str, Any]],
    entries: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    truth = np.asarray([float(row["h_r"]) for row in rows], dtype=np.float64)
    seeds = sorted({int(task["seed"]) for task, _ in entries})
    seed_predictions = []
    seed_metrics = []
    for seed in seeds:
        prediction = np.full(len(rows), np.nan, dtype=np.float64)
        for task, result in entries:
            if int(task["seed"]) != seed:
                continue
            payload = torch.load(resolve_path(result["checkpoint"]), map_location="cpu", weights_only=False)
            if (
                payload.get("contract_sha256") != result["contract_sha256"]
                or payload.get("task_sha256") != result["task_sha256"]
            ):
                raise RuntimeError(f"checkpoint provenance differs: {task['task_id']}")
            indices = np.asarray(payload["test_indices"], dtype=np.int64)
            values = np.asarray(payload["test_prediction_h_r"], dtype=np.float64)
            if np.isfinite(prediction[indices]).any():
                raise RuntimeError(f"OOF fold overlap: {task['task_id']}")
            prediction[indices] = values
        if not np.isfinite(prediction).all():
            raise RuntimeError("OOF prediction census is incomplete")
        seed_predictions.append(prediction)
        seed_metrics.append({"seed": seed, **_metric_bundle(truth, prediction)})
    return np.mean(np.stack(seed_predictions), axis=0), seed_metrics


def _aggregate_effect_growth(
    output_root: Path,
    manifest: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    stores = {
        horizon: HorizonStore(manifest, feature_rows, horizon=horizon, support="native")
        for horizon in HORIZONS
    }
    h1_index = {str(row["state_id"]): index for index, row in enumerate(stores[1].rows)}
    output = []
    for horizon, store in stores.items():
        local = np.arange(len(store.rows), dtype=np.int64)
        current = store.summaries(local).float().numpy()
        baseline = stores[1].summaries(
            [h1_index[str(row["state_id"])] for row in store.rows]
        ).float().numpy()
        current_delta = current[:, 0] - current[:, 1]
        baseline_delta = baseline[:, 0] - baseline[:, 1]
        hidden = store.hidden
        streams = {
            "text": (current_delta[:, :hidden], baseline_delta[:, :hidden]),
            "visual": (current_delta[:, hidden:], baseline_delta[:, hidden:]),
            "pooled": (current_delta, baseline_delta),
        }
        cohorts = {
            "all": np.ones(len(store.rows), dtype=bool),
            "harmful": np.asarray([float(row["h_r"]) > 0 for row in store.rows]),
            "beneficial": np.asarray([float(row["h_r"]) < 0 for row in store.rows]),
            "read_harmful_flip": np.asarray([row["cohort"] == "read_harmful_flip" for row in store.rows]),
            "read_beneficial_flip": np.asarray([row["cohort"] == "read_beneficial_flip" for row in store.rows]),
        }
        for stream, (delta, h1_delta) in streams.items():
            norm = np.linalg.norm(delta, axis=1)
            h1_norm = np.linalg.norm(h1_delta, axis=1)
            ratio = np.divide(norm, h1_norm, out=np.full_like(norm, np.nan), where=h1_norm > 0)
            denominator = norm * h1_norm
            cosine = np.divide(
                np.einsum("ij,ij->i", delta, h1_delta), denominator,
                out=np.full_like(norm, np.nan), where=denominator > 0,
            )
            for cohort, mask in cohorts.items():
                values = {"norm": norm[mask], "ratio_to_h1": ratio[mask], "cosine_to_h1": cosine[mask]}
                for measure, array in values.items():
                    finite = array[np.isfinite(array)]
                    output.append({
                        "horizon": horizon, "stream": stream, "cohort": cohort, "measure": measure,
                        "support": int(mask.sum()), "finite": len(finite),
                        "mean": float(finite.mean()) if len(finite) else float("nan"),
                        "median": float(np.median(finite)) if len(finite) else float("nan"),
                        "q25": float(np.quantile(finite, 0.25)) if len(finite) else float("nan"),
                        "q75": float(np.quantile(finite, 0.75)) if len(finite) else float("nan"),
                    })
    atomic_csv(output_root / "features/effect_growth_statistics.csv", output)
    return output


def _bootstrap_pairwise(
    rows: Sequence[Mapping[str, Any]],
    truth: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["image_group_id"])].append(index)
    keys = sorted(groups)
    arrays = [np.asarray(groups[key], dtype=np.int64) for key in keys]
    rng = np.random.default_rng(int(seed))
    spearman = np.empty(int(draws), dtype=np.float64)
    auroc = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        sampled = rng.integers(0, len(arrays), size=len(arrays))
        indices = np.concatenate([arrays[int(index)] for index in sampled])
        left_metrics = _metric_bundle(truth[indices], left[indices])
        right_metrics = _metric_bundle(truth[indices], right[indices])
        spearman[draw] = float(left_metrics["spearman"]) - float(right_metrics["spearman"])
        auroc[draw] = float(left_metrics["harmful_auroc"]) - float(right_metrics["harmful_auroc"])
    output = {}
    for name, values in (("spearman", spearman), ("harmful_auroc", auroc)):
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise RuntimeError(f"bootstrap has no finite {name} draws")
        output[name] = {
            "draws_requested": int(draws), "draws_valid": len(finite),
            "ci_low": float(np.quantile(finite, 0.025)),
            "median": float(np.quantile(finite, 0.5)),
            "ci_high": float(np.quantile(finite, 0.975)),
        }
    return output


def _subset_metrics(
    rows: Sequence[Mapping[str, Any]], prediction: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    truth = np.asarray([float(row["h_r"]) for row in rows], dtype=np.float64)
    return _metric_bundle(truth[mask], prediction[mask])


def _phase82_pre_common(
    common_rows: Sequence[Mapping[str, Any]], model: str
) -> dict[str, Any]:
    path = Path("analysis/dense_failure_stage2/counterfactual_effect_identifiability/read/pre/oof_predictions.jsonl")
    state_ids = {str(row["state_id"]) for row in common_rows}
    selected = [row for row in read_jsonl(path) if row["model"] == model and str(row["state_id"]) in state_ids]
    index = {str(row["state_id"]): row for row in selected}
    if set(index) != state_ids:
        raise RuntimeError("Phase2 PRE common-support prediction census differs")
    truth = np.asarray([float(row["h_r"]) for row in common_rows])
    prediction = np.asarray([-float(index[str(row["state_id"])]["prediction"]) for row in common_rows])
    return _metric_bundle(truth, prediction)


def _write_figures(
    output_root: Path,
    metrics: Sequence[Mapping[str, Any]],
    dense_w: Sequence[Mapping[str, Any]],
    trigger: Sequence[Mapping[str, Any]],
    growth: Sequence[Mapping[str, Any]],
) -> None:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def row(condition: str, horizon: int, model: str = "mlp") -> Mapping[str, Any]:
        return next(
            value for value in metrics
            if value["support_name"] == "common_h8" and value["family"] == "standard"
            and value["condition"] == condition and value["model"] == model
            and int(value["horizon"]) == int(horizon)
        )

    conditions = ("on", "off", "pair", "delta", "token_comparator")
    for metric, filename, ylabel in (
        ("spearman", "read_horizon_spearman.png", "OOF Spearman"),
        ("harmful_auroc", "read_horizon_harmful_auroc.png", "Harmful READ AUROC"),
    ):
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        for condition in conditions:
            model = "token_comparator" if condition == "token_comparator" else "mlp"
            ax.plot(HORIZONS, [float(row(condition, h, model)[metric]) for h in HORIZONS], marker="o", label=condition)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Propagation horizon")
        ax.set_ylabel(ylabel)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for condition in ("on", "off", "pair", "delta"):
        ax.plot(HORIZONS, [float(row(condition, h)["spearman"]) for h in HORIZONS], marker="o", label=condition)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Propagation horizon")
    ax.set_ylabel("OOF Spearman")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "read_single_vs_counterfactual_by_horizon.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    chosen = [value for value in dense_w if value["condition"] == "delta" and value["model"] == "mlp"]
    chosen.sort(key=lambda value: int(value["horizon"]))
    ax.plot([int(value["horizon"]) for value in chosen], [float(value["spearman"]) for value in chosen], marker="o", label="Spearman")
    ax.plot([int(value["horizon"]) for value in chosen], [float(value["harmful_auroc"]) for value in chosen], marker="s", label="Harmful AUROC")
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Propagation horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "read_dense_w_horizon_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for condition in ("delta", "token_comparator"):
        model = "token_comparator" if condition == "token_comparator" else "mlp"
        ax.plot(HORIZONS, [float(row(condition, h, model)["precision_at_0.1"]) for h in HORIZONS], marker="o", label=condition)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Propagation horizon")
    ax.set_ylabel("Harmful precision @ top 10%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "read_harmful_precision_by_horizon.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for stream in ("text", "visual", "pooled"):
        selected = [
            value for value in growth
            if value["stream"] == stream and value["cohort"] == "all" and value["measure"] == "norm"
        ]
        selected.sort(key=lambda value: int(value["horizon"]))
        ax.plot([int(value["horizon"]) for value in selected], [float(value["median"]) for value in selected], marker="o", label=stream)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Propagation horizon")
    ax.set_ylabel("Median ON-minus-OFF norm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "read_delta_norm_growth.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for value in sorted({str(row_["value"]) for row_ in trigger}):
        selected = [row_ for row_ in trigger if str(row_["value"]) == value]
        selected.sort(key=lambda row_: int(row_["horizon"]))
        ax.plot([int(row_["horizon"]) for row_ in selected], [float(row_["spearman"]) for row_ in selected], marker="o", label=value)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Propagation horizon")
    ax.set_ylabel("Delta-MLP Spearman")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "read_horizon_by_trigger_depth.png", dpi=180)
    plt.close(fig)


def aggregate(config_path: Path) -> None:
    contract, output_root, external_root = verify_contract(config_path)
    _require_training_ready(contract, output_root)
    config = contract["static_config"]
    results = _collect_results(contract, output_root)
    manifest = read_json(output_root / "features/cache_manifest.json")
    feature_rows = read_jsonl(output_root / "features/pooled_horizon_features.jsonl")
    grouped: dict[tuple[str, int, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for task, result in results:
        grouped[(str(task["support"]), int(task["horizon"]), str(task["family"]), str(task["condition"]), str(task["model"]))].append((task, result))
    stores: dict[tuple[str, int], HorizonStore] = {}
    predictions: dict[tuple[str, int, str, str, str], np.ndarray] = {}
    metric_rows = []
    seed_rows = []
    prediction_root = external_root / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction_manifest = []
    for key, entries in sorted(grouped.items()):
        support, horizon, family, condition, model = key
        store_key = support, horizon
        if store_key not in stores:
            stores[store_key] = HorizonStore(manifest, feature_rows, horizon=horizon, support=support)
        store = stores[store_key]
        prediction, seeds = _oof_predictions(store.rows, entries)
        truth = np.asarray([float(row["h_r"]) for row in store.rows], dtype=np.float64)
        metrics = _metric_bundle(truth, prediction)
        metric_rows.append({
            "support_name": support, "horizon": horizon, "family": family,
            "condition": condition, "model": model,
            "uids": len({str(row["uid"]) for row in store.rows}),
            "image_groups": len({str(row["image_group_id"]) for row in store.rows}),
            **metrics,
        })
        seed_rows.extend({
            "support_name": support, "horizon": horizon, "family": family,
            "condition": condition, "model": model, **seed,
        } for seed in seeds)
        predictions[key] = prediction
        prediction_path = prediction_root / ("__".join(map(str, key)) + ".npz")
        np.savez_compressed(prediction_path, prediction=prediction.astype(np.float32))
        prediction_manifest.append({
            "support_name": support, "horizon": horizon, "family": family,
            "condition": condition, "model": model, "path": str(prediction_path),
            "sha256": file_sha256(prediction_path), "rows": len(prediction),
            "state_order_sha256": _canonical_json_hash(
                [str(row["state_id"]) for row in store.rows]
            ),
        })
    atomic_jsonl(output_root / "work/prediction_manifest.jsonl", prediction_manifest)
    standard = [row for row in metric_rows if row["family"] == "standard"]
    atomic_csv(output_root / "metrics/native_support_metrics.csv", [row for row in standard if row["support_name"] == "native"])
    atomic_csv(output_root / "metrics/common_support_metrics.csv", [row for row in standard if row["support_name"] == "common_h8"])
    atomic_csv(output_root / "metrics/harmful_ranking_metrics.csv", [{
        key: row[key] for key in (
            "support_name", "horizon", "family", "condition", "model", "support",
            "harmful_prevalence", "harmful_auroc", "harmful_auprc", "precision_at_0.05",
            "precision_at_0.1", "precision_at_0.2", "recall_at_precision_0.9",
            "recall_at_precision_0.95",
        )
    } for row in standard])
    atomic_csv(output_root / "statistics/seed_metrics.csv", seed_rows)
    atomic_csv(output_root / "work/all_metrics.csv", metric_rows)

    common_metrics = [row for row in standard if row["support_name"] == "common_h8"]
    single_control = [
        row for row in common_metrics
        if row["model"] == "mlp" and row["condition"] in {"on", "off", "pair", "delta"}
    ]
    atomic_csv(output_root / "controls/single_branch_depth_control.csv", single_control)
    atomic_csv(output_root / "controls/random_pair_control.csv", [row for row in metric_rows if row["family"] == "random_pair"])
    atomic_csv(output_root / "controls/permuted_target_control.csv", [row for row in metric_rows if row["family"] == "permuted_target"])
    atomic_csv(output_root / "controls/text_visual_delta_ablation.csv", [
        row for row in common_metrics
        if row["model"] == "mlp" and row["condition"] in {"text_delta", "visual_delta", "text_visual_delta"}
    ])

    def prediction_for(horizon: int, condition: str = "delta", model: str = "mlp") -> tuple[HorizonStore, np.ndarray]:
        key = ("common_h8", int(horizon), "standard", condition, model)
        return stores[("common_h8", int(horizon))], predictions[key]

    dense_w_rows = []
    strong_flip_rows = []
    layer_rows = []
    trigger_rows = []
    dataset_rows = []
    depth_bins = config["evaluation"]["depth_bins"]
    trigger_bins = config["evaluation"]["trigger_relative_bins"]
    minimum = int(config["evaluation"]["minimum_breakdown_rows"])
    for horizon in HORIZONS:
        store, primary_prediction = prediction_for(horizon)
        for condition in ("on", "off", "pair", "delta", "token_comparator"):
            model = "token_comparator" if condition == "token_comparator" else "mlp"
            _, prediction = prediction_for(horizon, condition, model)
            dense_w_mask = np.asarray([bool(row["dense_wrong"]) for row in store.rows])
            dense_w_rows.append({
                "horizon": horizon, "condition": condition, "model": model,
                **_subset_metrics(store.rows, prediction, dense_w_mask),
            })
            labels = np.asarray([row["cohort"] == "read_harmful_flip" for row in store.rows], dtype=np.int64)
            flip_metric = binary_classification_metrics(truth=labels, prediction=prediction)
            cohorts = defaultdict(list)
            for row, value in zip(store.rows, prediction):
                cohorts[str(row["cohort"])].append(float(value))
            strong_flip_rows.append({
                "horizon": horizon, "condition": condition, "model": model,
                "label": "read_harmful_flip_vs_all", **flip_metric,
                "cohort_prediction_medians_json": json.dumps({
                    name: float(np.median(values)) for name, values in sorted(cohorts.items())
                }, sort_keys=True),
            })
        for name, bounds in depth_bins.items():
            mask = np.asarray([int(bounds[0]) <= int(row["layer"]) <= int(bounds[1]) for row in store.rows])
            if int(mask.sum()) >= minimum:
                layer_rows.append({"horizon": horizon, "breakdown": "depth_bin", "value": name, **_subset_metrics(store.rows, primary_prediction, mask)})
        for layer in range(28):
            mask = np.asarray([int(row["layer"]) == layer for row in store.rows])
            if int(mask.sum()) >= minimum:
                layer_rows.append({"horizon": horizon, "breakdown": "exact_layer", "value": layer, **_subset_metrics(store.rows, primary_prediction, mask)})
        for name, bounds in trigger_bins.items():
            mask = np.asarray([int(bounds[0]) <= int(row["trigger_relative_depth"]) <= int(bounds[1]) for row in store.rows])
            if int(mask.sum()) >= minimum:
                trigger_rows.append({"horizon": horizon, "value": name, **_subset_metrics(store.rows, primary_prediction, mask)})
        cells: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(store.rows):
            cells[(str(row["dataset"]), str(row["source_regime"]))].append(index)
        for (dataset, source), indices in sorted(cells.items()):
            mask = np.zeros(len(store.rows), dtype=bool)
            mask[indices] = True
            if len(indices) >= minimum:
                dataset_rows.append({
                    "horizon": horizon, "dataset": dataset, "source_regime": source,
                    **_subset_metrics(store.rows, primary_prediction, mask),
                })
    atomic_csv(output_root / "metrics/dense_w_only_metrics.csv", dense_w_rows)
    atomic_csv(output_root / "metrics/strong_flip_metrics.csv", strong_flip_rows)
    atomic_csv(output_root / "metrics/layer_breakdown.csv", layer_rows)
    atomic_csv(output_root / "metrics/trigger_relative_breakdown.csv", trigger_rows)
    atomic_csv(output_root / "metrics/dataset_source_breakdown.csv", dataset_rows)

    emergence = []
    monotonicity = []
    for condition in ("on", "off", "pair", "delta", "pair_plus_delta", "text_delta", "visual_delta", "text_visual_delta", "token_comparator"):
        model = "token_comparator" if condition == "token_comparator" else "mlp"
        selected = [row for row in common_metrics if row["condition"] == condition and row["model"] == model]
        selected.sort(key=lambda row: int(row["horizon"]))
        baseline = next(row for row in selected if int(row["horizon"]) == 1)
        for row in selected:
            emergence.append({
                "condition": condition, "model": model, "horizon": row["horizon"],
                "spearman": row["spearman"], "harmful_auroc": row["harmful_auroc"],
                "delta_spearman_vs_h1": float(row["spearman"]) - float(baseline["spearman"]),
                "delta_harmful_auroc_vs_h1": float(row["harmful_auroc"]) - float(baseline["harmful_auroc"]),
            })
        monotonicity.append({
            "condition": condition, "model": model,
            "spearman_nondecreasing": monotonic_nondecreasing({int(row["horizon"]): float(row["spearman"]) for row in selected}),
            "harmful_auroc_nondecreasing": monotonic_nondecreasing({int(row["horizon"]): float(row["harmful_auroc"]) for row in selected}),
        })
    atomic_csv(output_root / "metrics/horizon_emergence.csv", emergence)
    atomic_csv(output_root / "metrics/monotonicity.csv", monotonicity)

    bootstrap_rows = []
    pairwise_rows = []
    bootstrap_by_condition: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    counter = 0
    for condition in ("delta", "token_comparator"):
        model = "token_comparator" if condition == "token_comparator" else "mlp"
        base_store, base_prediction = prediction_for(1, condition, model)
        truth = np.asarray([float(row["h_r"]) for row in base_store.rows])
        for horizon in (2, 4, 8):
            store, prediction = prediction_for(horizon, condition, model)
            if [row["state_id"] for row in store.rows] != [row["state_id"] for row in base_store.rows]:
                raise RuntimeError("common-support horizon order differs")
            intervals = _bootstrap_pairwise(
                store.rows, truth, prediction, base_prediction,
                draws=int(config["evaluation"]["bootstrap_draws"]),
                seed=int(config["evaluation"]["bootstrap_seed"]) + counter,
            )
            counter += 1
            bootstrap_by_condition[condition][horizon] = intervals
            current_metric = next(row for row in common_metrics if row["horizon"] == horizon and row["condition"] == condition and row["model"] == model)
            base_metric = next(row for row in common_metrics if int(row["horizon"]) == 1 and row["condition"] == condition and row["model"] == model)
            for metric in ("spearman", "harmful_auroc"):
                record = {
                    "condition": condition, "model": model, "horizon": horizon, "metric": metric,
                    "observed_difference": float(current_metric[metric]) - float(base_metric[metric]),
                    "bootstrap_unit": "image_group_id", **intervals[metric],
                }
                bootstrap_rows.append(record)
                pairwise_rows.append(record)
    atomic_csv(output_root / "statistics/group_bootstrap_ci.csv", bootstrap_rows)
    atomic_csv(output_root / "statistics/horizon_pairwise_differences.csv", pairwise_rows)

    growth = _aggregate_effect_growth(output_root, manifest, feature_rows)

    def common_metric(horizon: int, condition: str, model: str = "mlp") -> Mapping[str, Any]:
        return next(row for row in common_metrics if int(row["horizon"]) == horizon and row["condition"] == condition and row["model"] == model)

    primary = {h: common_metric(h, "delta") for h in HORIZONS}
    token = {h: common_metric(h, "token_comparator", "token_comparator") for h in HORIZONS}
    best_single = {h: max(float(common_metric(h, "on")["spearman"]), float(common_metric(h, "off")["spearman"])) for h in HORIZONS}
    random = {
        int(row["horizon"]): float(row["spearman"])
        for row in metric_rows if row["family"] == "random_pair"
    }
    thresholds = {
        "spearman_gain": float(config["decision"]["spearman_gain"]),
        "auroc_gain": float(config["decision"]["harmful_auroc_gain"]),
        "pair_over_single": float(config["decision"]["pair_over_single_spearman"]),
        "random_pair_gap": float(config["decision"]["random_pair_gap_spearman"]),
        "useful_precision_gain": float(config["decision"]["useful_top10_precision_gain"]),
    }
    decision = classify_h_read(
        h1_spearman=float(primary[1]["spearman"]), h1_auroc=float(primary[1]["harmful_auroc"]),
        horizon_spearman={h: float(primary[h]["spearman"]) for h in HORIZONS},
        horizon_auroc={h: float(primary[h]["harmful_auroc"]) for h in HORIZONS},
        best_single_spearman=best_single, random_pair_spearman=random,
        token_spearman={h: float(token[h]["spearman"]) for h in HORIZONS},
        ci_lower_spearman={h: float(bootstrap_by_condition["delta"][h]["spearman"]["ci_low"]) for h in (2, 4, 8)},
        ci_lower_auroc={h: float(bootstrap_by_condition["delta"][h]["harmful_auroc"]["ci_low"]) for h in (2, 4, 8)},
        ci_lower_token_spearman={h: float(bootstrap_by_condition["token_comparator"][h]["spearman"]["ci_low"]) for h in (2, 4, 8)},
        precision_top10={h: float(primary[h]["precision_at_0.1"]) for h in HORIZONS},
        prevalence=float(primary[1]["harmful_prevalence"]), thresholds=thresholds,
    )
    decision.update({"contract_sha256": contract["contract_sha256"], "thresholds": thresholds})
    atomic_json(output_root / "summaries/decision_category.json", decision)
    _write_figures(output_root, common_metrics, dense_w_rows, trigger_rows, growth)

    support = read_csv(output_root / "population/horizon_support.csv")
    native_support = {int(row["horizon"]): row for row in support if row["support"] == "native"}
    previous_h1 = next(
        row for row in read_csv("analysis/dense_failure_stage2/counterfactual_effect_identifiability/read/metrics.csv")
        if row["family"] == "primary" and row["condition"] == "delta" and row["model"] == "mlp"
    )
    pre = _phase82_pre_common(stores[("common_h8", 1)].rows, "mlp")
    action_validation = read_csv(output_root / "branches/action_trace_validation.csv")
    action_exact = all(str(row["trace_exact"]).lower() == "true" for row in action_validation)
    parity = read_csv(output_root / "branches/state_hash_parity.csv")
    h1_exact = all(str(row["phase82_h1_exact"]).lower() in {"true", "none", ""} for row in parity if int(row["horizon"]) == 1)
    delta_dense_w = {int(row["horizon"]): row for row in dense_w_rows if row["condition"] == "delta" and row["model"] == "mlp"}
    random_rows = [row for row in metric_rows if row["family"] == "random_pair"]
    permuted_rows = [row for row in metric_rows if row["family"] == "permuted_target"]
    strongest_horizon = max(HORIZONS, key=lambda h: float(primary[h]["spearman"]))
    harmful_growth = next(row for row in growth if int(row["horizon"]) == 8 and row["stream"] == "pooled" and row["cohort"] == "harmful" and row["measure"] == "ratio_to_h1")
    beneficial_growth = next(row for row in growth if int(row["horizon"]) == 8 and row["stream"] == "pooled" and row["cohort"] == "beneficial" and row["measure"] == "ratio_to_h1")

    def curve(condition: str, metric: str = "spearman", model: str = "mlp") -> str:
        return ", ".join(f"H{h}={float(common_metric(h, condition, model)[metric]):.4f}" for h in HORIZONS)

    def table(metric: str) -> str:
        labels = (("ON-only", "on", "mlp"), ("OFF-only", "off", "mlp"), ("Pair", "pair", "mlp"), ("Delta", "delta", "mlp"), ("Token comparator", "token_comparator", "token_comparator"))
        lines = ["| Input | PRE | H=1 | H=2 | H=4 | H=8 |", "|---|---:|---:|---:|---:|---:|"]
        for label, condition, model in labels:
            pre_value = float(pre[metric]) if label in {"ON-only", "OFF-only"} else float("nan")
            formatted_pre = f"{pre_value:.4f}" if math.isfinite(pre_value) else "—"
            lines.append(f"| {label} | {formatted_pre} | " + " | ".join(f"{float(common_metric(h, condition, model)[metric]):.4f}" for h in HORIZONS) + " |")
        return "\n".join(lines)

    summary = f"""# READ short-horizon counterfactual propagation summary

## Outcome

The frozen decision is **{decision['category']}**. {decision['reason']} The strongest preregistered pooled-delta point estimate was H={strongest_horizon} (Spearman {float(primary[strongest_horizon]['spearman']):.4f}); material horizons were {decision['material_horizons']}.

## Common-support main tables

Spearman:

{table('spearman')}

Harmful READ AUROC:

{table('harmful_auroc')}

Precision at top 10%:

{table('precision_at_0.1')}

## Required questions

1. Native eligibility is H1 **{int(native_support[1]['states']):,} states / {int(native_support[1]['uids']):,} UIDs**, H2 **{int(native_support[2]['states']):,} / {int(native_support[2]['uids']):,}**, H4 **{int(native_support[4]['states']):,} / {int(native_support[4]['uids']):,}**, and H8 **{int(native_support[8]['states']):,} / {int(native_support[8]['uids']):,}**.
2. **Yes.** Stored H=1 ON/OFF hashes exactly reproduced Phase-82 (`{h1_exact}`); the native delta-MLP Spearman is {float(next(row['spearman'] for row in standard if row['support_name']=='native' and int(row['horizon'])==1 and row['condition']=='delta' and row['model']=='mlp')):.4f} versus {float(previous_h1['spearman']):.4f} previously.
3. **Yes.** Every recorded trace passed the fixed intervention-then-FULL contract (`{action_exact}`).
4. The median pooled ON/OFF norm ratio at H8 is {float(next(row['median'] for row in growth if int(row['horizon'])==8 and row['stream']=='pooled' and row['cohort']=='all' and row['measure']=='ratio_to_h1')):.3f}× H1; stream-wise curves are in `features/effect_growth_statistics.csv`.
5. Median H8/H1 pooled growth is {float(harmful_growth['median']):.3f}× for harmful and {float(beneficial_growth['median']):.3f}× for beneficial states; this is descriptive, not a predictability claim.
6. ON-only: {curve('on')}.
7. OFF-only: {curve('off')}.
8. PAIR: {curve('pair')}.
9. DELTA: {curve('delta')}.
10. Token comparator: {curve('token_comparator', model='token_comparator')}.
11. Pooled-delta Spearman monotonicity is **{monotonic_nondecreasing({h: float(primary[h]['spearman']) for h in HORIZONS})}**; AUROC monotonicity is **{monotonic_nondecreasing({h: float(primary[h]['harmful_auroc']) for h in HORIZONS})}**.
12. Materiality gate passed at: **{decision['material_horizons']}**. A gate requires the fixed absolute gain and a strictly positive image-group bootstrap lower bound.
13. At H={strongest_horizon}, delta exceeds the best single branch by {float(primary[strongest_horizon]['spearman']) - best_single[strongest_horizon]:+.4f} Spearman.
14. Random-pair Spearman values are {', '.join(f"H{int(row['horizon'])}={float(row['spearman']):.4f}" for row in sorted(random_rows, key=lambda value: int(value['horizon'])))}.
15. UID-permuted-target Spearman values are {', '.join(f"H{int(row['horizon'])}={float(row['spearman']):.4f}" for row in sorted(permuted_rows, key=lambda value: int(value['horizon'])))}.
16. Dense-W delta results are {', '.join(f"H{h}: rho={float(delta_dense_w[h]['spearman']):.4f}, AUROC={float(delta_dense_w[h]['harmful_auroc']):.4f}" for h in HORIZONS)}.
17. Strong harmful-flip AUROC/AUPRC and cohort score medians at every horizon are in `metrics/strong_flip_metrics.csv`.
18. Exact-layer, depth-bin, and trigger-relative results are in `metrics/layer_breakdown.csv` and `metrics/trigger_relative_breakdown.csv`; they are descriptive subgroup checks.
19. Historical/canonical GQA, ChartQA, and TextVQA cells are all reported without best-cell selection in `metrics/dataset_source_breakdown.csv`.
20. The supported category is **{decision['category']}**: {decision['reason']}.
21. This phase does not establish benchmark gain, deployment utility, compute savings, external transfer, causal optimality, or WRITE identifiability. Routed-state evidence was not promoted into the primary dense decision.

## Validity

The 16-state fresh-cache smoke, swapped-order H8 check, exact H1 parent parity, exact ON canonical parity, global branch census, and cache readback hashes all passed before training. Primary comparisons use the same H8-common population, fixed model capacity, inherited image-group folds, UID-balanced Huber training, and three fixed seeds.
"""
    (output_root / "summaries").mkdir(parents=True, exist_ok=True)
    (output_root / "summaries/read_short_horizon_propagation_summary.md").write_text(summary)
    recommendations = {
        "H-READ-A": "minimal short-lookahead counterfactual READ critic",
        "H-READ-B": "delayed verification / rollback READ controller",
        "H-READ-C": "token-aware short-lookahead READ critic",
        "H-READ-D": "bounded longer-horizon READ planning/search study",
    }
    recommendation = recommendations[str(decision["category"])]
    recommendation_md = f"""# Next READ-method recommendation

Recommend exactly one next research action: **{recommendation}**.

This follows frozen category **{decision['category']}**: {decision['reason']} A positive result in that separately authorized study would show that the bounded method exposes stable READ-harm information beyond this horizon/representation family. A negative result would rule out that bounded extension and weigh against investment in a READ deployment controller of the same family. Benchmark gain, external transfer, compute savings, and causal/global optimality remain unproven.

This recommendation is not authorization to run it.
"""
    (output_root / "summaries/next_read_method_recommendation.md").write_text(recommendation_md)

    files = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or "work" in path.relative_to(output_root).parts or path.name == "artifact_manifest.json":
            continue
        files[str(path.relative_to(output_root))] = file_sha256(path)
    external_files = {
        str(spec["path"]): str(spec["sha256"])
        for horizon in HORIZONS
        for spec in manifest["horizons"][str(horizon)]["files"].values()
    }
    artifact = {
        "schema_version": "read_short_horizon_propagation_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"], "decision": decision,
        "files": files, "external_raw_cache_sha256": external_files,
        "training_checkpoints": len(results),
    }
    artifact["artifact_manifest_sha256"] = canonical_hash(artifact)
    atomic_json(output_root / "artifact_manifest.json", artifact)
    print(json.dumps({
        "complete": True, "contract_sha256": contract["contract_sha256"],
        "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
        "decision": decision,
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare-training", "train", "aggregate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-training":
        prepare_training(args.config)
    elif args.command == "train":
        train_worker(args.config, args.rank, args.device_index, resume=args.resume)
    else:
        aggregate(args.config)


if __name__ == "__main__":
    main()



def _oof_predictions(
    rows: Sequence[Mapping[str, Any]],
    entries: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    truth = np.asarray([float(row["h_r"]) for row in rows], dtype=np.float64)
    per_seed = []
    seed_metrics = []
    for seed in sorted({int(task["seed"]) for task, _ in entries}):
        prediction = np.full(len(rows), np.nan, dtype=np.float64)
        for task, result in entries:
            if int(task["seed"]) != seed:
                continue
            payload = torch.load(
                resolve_path(result["checkpoint"]), map_location="cpu", weights_only=False
            )
            indices = np.asarray(payload["test_indices"], dtype=np.int64)
            values = np.asarray(payload["test_prediction_h_r"], dtype=np.float64)
            if np.isfinite(prediction[indices]).any():
                raise RuntimeError("OOF prediction overlap")
            prediction[indices] = values
        if not np.isfinite(prediction).all():
            raise RuntimeError("OOF prediction census is incomplete")
        per_seed.append(prediction)
        seed_metrics.append({"seed": seed, **_metric_bundle(truth, prediction)})
    return np.mean(np.stack(per_seed), axis=0), seed_metrics
