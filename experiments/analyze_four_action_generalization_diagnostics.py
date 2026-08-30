#!/usr/bin/env python3
"""Run Priority-1/2 diagnostics and freeze the bounded label-audit subset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import yaml

from experiments.prepare_four_action_collapse import write_frozen
from experiments.train_binary_polar import file_sha256
from four_action_online_router.data import load_jsonl, load_verified_manifest
from four_action_policy.actions import FOUR_ACTIONS
from four_action_policy.generalization_diagnostics import (
    binary_metrics,
    build_label_incompleteness_subset,
    candidate_suffix_routes,
    compact_knn_label_consistency,
    first_deviation_bucket,
    knn_label_consistency,
    layer_only_binary_scores,
    multiclass_metrics,
)


ARCHITECTURES = ("polar", "online")
REPRESENTATIONS = ("upfront", "online", "z_R", "z_W")
MECHANISM_CLASSES = ("IGNORE", "READ_ONLY", "WRITE_ONLY")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty diagnostic table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise ValueError(f"diagnostic CSV rows have inconsistent columns: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_joined(config: dict[str, Any], config_sha: str) -> list[dict[str, Any]]:
    states = load_jsonl(config["data"]["state_manifest"])
    payload = torch.load(
        config["reporting"]["state_outputs"], map_location="cpu", weights_only=False
    )
    if (
        payload.get("schema_version") != "four_action_generalization_state_outputs_v1"
        or payload.get("config_sha256") != config_sha
        or payload.get("state_manifest_sha256") != config["data"]["state_manifest_sha256"]
    ):
        raise RuntimeError("diagnostic state outputs are not config-bound")
    by_id = {row["state_id"]: row for row in payload["records"]}
    if set(by_id) != {row["state_id"] for row in states}:
        raise RuntimeError("diagnostic state outputs do not cover the state manifest")
    joined = []
    for state in states:
        output = by_id[state["state_id"]]
        joined.append(
            {
                **state,
                **{
                    key: value.detach().cpu().float().numpy()
                    if torch.is_tensor(value)
                    else value
                    for key, value in output.items()
                    if key not in {"state_id", "uid"}
                },
            }
        )
    return joined


def probabilities(row: dict[str, Any], architecture: str, *, shuffled: bool = False) -> np.ndarray:
    key = "online_shuffled_probabilities" if shuffled else f"{architecture}_probabilities"
    values = np.asarray(row[key], dtype=np.float64)
    if values.shape != (4,) or not np.isclose(values.sum(), 1.0, atol=1e-5):
        raise RuntimeError("diagnostic action probabilities are malformed")
    return values


def architecture_values(
    row: dict[str, Any], architecture: str, *, shuffled: bool = False
) -> dict[str, Any]:
    values = probabilities(row, architecture, shuffled=shuffled)
    prediction = int(np.argmax(values))
    return {
        "predicted_action": FOUR_ACTIONS[prediction],
        "when": float(1.0 - values[FOUR_ACTIONS.index("FULL")]),
        "read_off": float(values[FOUR_ACTIONS.index("WRITE_ONLY")] + values[FOUR_ACTIONS.index("IGNORE")]),
        "write_off": float(values[FOUR_ACTIONS.index("READ_ONLY")] + values[FOUR_ACTIONS.index("IGNORE")]),
        "valid": bool(row["valid_action_mask"][prediction]),
    }


def scoped(rows: Sequence[dict[str, Any]], *, include_layer: bool = True):
    yield "overall", "all", list(rows)
    for dataset in ("gqa", "chartqa", "textvqa"):
        selected = [row for row in rows if row["dataset"] == dataset]
        if selected:
            yield "dataset", dataset, selected
    for depth, value in (("early", 0), ("middle", 1), ("late", 2)):
        selected = [row for row in rows if int(row["depth_bin"]) == value]
        if selected:
            yield "depth", depth, selected
    if include_layer:
        for layer in range(28):
            selected = [row for row in rows if int(row["target_layer"]) == layer]
            if selected:
                yield "layer", str(layer), selected


def when_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = []
    primary: dict[str, dict[str, Any]] = defaultdict(dict)
    for architecture in ARCHITECTURES:
        for split in ("train", "validation"):
            split_rows = [row for row in rows if row["split"] == split]
            for scope_type, scope_value, selected in scoped(split_rows):
                metrics = binary_metrics(
                    [int(row["when_label"]) for row in selected],
                    [architecture_values(row, architecture)["when"] for row in selected],
                )
                record = {
                    "architecture": architecture,
                    "split": split,
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    **metrics,
                }
                table.append(record)
                if scope_type == "overall":
                    primary[architecture][split] = metrics
    return table, primary


def what_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = []
    primary: dict[str, dict[str, Any]] = defaultdict(dict)
    for architecture in ARCHITECTURES:
        for split in ("train", "validation"):
            positives = [
                row
                for row in rows
                if row["split"] == split and row["state_kind"] == "mandatory_deviation"
            ]
            for scope_type, scope_value, selected in scoped(positives, include_layer=False):
                values = [architecture_values(row, architecture) for row in selected]
                deviated = [value for value in values if value["predicted_action"] != "FULL"]
                action_counts = Counter(value["predicted_action"] for value in values)
                record = {
                    "architecture": architecture,
                    "split": split,
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "records": len(selected),
                    "deviated": len(deviated),
                    "deviate_recall": len(deviated) / len(selected),
                    "unconditional_valid_action_at_1": sum(value["valid"] for value in values) / len(values),
                    "conditional_valid_action_at_1": (
                        sum(value["valid"] for value in deviated) / len(deviated)
                        if deviated
                        else float("nan")
                    ),
                    **{f"predicted_{action}": action_counts[action] for action in FOUR_ACTIONS},
                }
                table.append(record)
                if scope_type == "overall":
                    primary[architecture][split] = record
    return table, primary


def singleton_confusion(rows: list[dict[str, Any]], architecture: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    classes = list(FOUR_ACTIONS)
    table = []
    summaries = {}
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split and row["singleton"]]
        truth = [row["mechanism_class"] for row in selected]
        predicted = [architecture_values(row, architecture)["predicted_action"] for row in selected]
        metrics = multiclass_metrics(truth, predicted, classes=classes)
        summaries[split] = metrics
        for target in classes:
            values = metrics["by_class"][target]
            table.append(
                {
                    "architecture": architecture,
                    "split": split,
                    "target_action": target,
                    **{f"predicted_{action}": metrics["confusion"][target][action] for action in classes},
                    "support": values["support"],
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1"],
                    "predicted_full_fraction": (
                        metrics["confusion"][target]["FULL"] / values["support"]
                        if values["support"]
                        else float("nan")
                    ),
                }
            )
    return table, summaries


def bit_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = []
    primary: dict[str, dict[str, Any]] = defaultdict(dict)
    for architecture in ARCHITECTURES:
        for split in ("train", "validation"):
            split_rows = [row for row in rows if row["split"] == split]
            for bit in ("read_off", "write_off"):
                eligible = [row for row in split_rows if row[f"{bit}_label"] is not None]
                ambiguous = len(split_rows) - len(eligible)
                for scope_type, scope_value, selected in scoped(eligible):
                    metrics = binary_metrics(
                        [int(row[f"{bit}_label"]) for row in selected],
                        [architecture_values(row, architecture)[bit] for row in selected],
                    )
                    record = {
                        "architecture": architecture,
                        "split": split,
                        "bit": bit,
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "ambiguous_excluded_from_split": ambiguous,
                        **metrics,
                    }
                    table.append(record)
                    if scope_type == "overall":
                        primary[architecture][f"{split}:{bit}"] = metrics
    return table, primary


def both_off_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = []
    primary: dict[str, dict[str, Any]] = defaultdict(dict)
    for architecture in ARCHITECTURES:
        for split in ("train", "validation"):
            selected_all = [
                row
                for row in rows
                if row["split"] == split
                and row["state_kind"] == "mandatory_deviation"
                and row["mechanism_class"] == "IGNORE"
            ]
            for scope_type, scope_value, selected in scoped(selected_all, include_layer=False):
                counts = Counter(
                    architecture_values(row, architecture)["predicted_action"]
                    for row in selected
                )
                record = {
                    "architecture": architecture,
                    "split": split,
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "records": len(selected),
                    "ignore_only_recall": counts["IGNORE"] / len(selected),
                    "partial_suppression_rate": (counts["READ_ONLY"] + counts["WRITE_ONLY"]) / len(selected),
                    "full_error_rate": counts["FULL"] / len(selected),
                    **{f"predicted_{action}": counts[action] for action in FOUR_ACTIONS},
                }
                table.append(record)
                if scope_type == "overall":
                    primary[architecture][split] = record
    return table, primary


def train_validation_gap(
    when: dict[str, Any],
    what: dict[str, Any],
    bits: dict[str, Any],
    confusion: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for architecture in ARCHITECTURES:
        metrics = {
            "KEEP_vs_DEVIATE_AUROC": (
                when[architecture]["train"]["auroc"],
                when[architecture]["validation"]["auroc"],
            ),
            "READ_OFF_AUROC": (
                bits[architecture]["train:read_off"]["auroc"],
                bits[architecture]["validation:read_off"]["auroc"],
            ),
            "WRITE_OFF_AUROC": (
                bits[architecture]["train:write_off"]["auroc"],
                bits[architecture]["validation:write_off"]["auroc"],
            ),
            "READ_ONLY_only_recall": (
                confusion[architecture]["train"]["by_class"]["READ_ONLY"]["recall"],
                confusion[architecture]["validation"]["by_class"]["READ_ONLY"]["recall"],
            ),
            "WRITE_ONLY_only_recall": (
                confusion[architecture]["train"]["by_class"]["WRITE_ONLY"]["recall"],
                confusion[architecture]["validation"]["by_class"]["WRITE_ONLY"]["recall"],
            ),
            "IGNORE_only_recall": (
                confusion[architecture]["train"]["by_class"]["IGNORE"]["recall"],
                confusion[architecture]["validation"]["by_class"]["IGNORE"]["recall"],
            ),
            "conditional_valid_given_deviation": (
                what[architecture]["train"]["conditional_valid_action_at_1"],
                what[architecture]["validation"]["conditional_valid_action_at_1"],
            ),
        }
        for metric, (train, validation) in metrics.items():
            rows.append(
                {
                    "architecture": architecture,
                    "metric": metric,
                    "train": train,
                    "validation": validation,
                    "gap_train_minus_validation": train - validation,
                }
            )
    return rows


def first_deviation_table(
    config: dict[str, Any], states: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    boundary_by_uid = {
        row["uid"]: row
        for row in states
        if row["split"] == "validation" and row["state_kind"] == "mandatory_deviation"
    }
    parent = {
        name: yaml.safe_load(Path(config["parent_configs"][name]["path"]).read_text(encoding="utf-8"))
        for name in ARCHITECTURES
    }
    execution_paths = {
        "polar": Path(parent["polar"]["reporting"]["execution"]),
        "online": Path(parent["online"]["reporting"]["execution"]),
    }
    selected_epochs = {name: int(config["checkpoints"][name]["epoch"]) for name in ARCHITECTURES}
    table = []
    summaries = {}
    for architecture in ARCHITECTURES:
        outputs = [
            row
            for row in load_jsonl(execution_paths[architecture])
            if int(row["epoch"]) == selected_epochs[architecture] and row["route_type"] == "W2C"
        ]
        if len(outputs) != len(boundary_by_uid):
            raise RuntimeError("timing analysis does not cover validation W2C exactly")
        for output in outputs:
            boundary = boundary_by_uid[output["uid"]]
            actions = [str(value) for value in output["actions"]]
            predicted_layer = next(
                (index for index, action in enumerate(actions) if action != "FULL"), None
            )
            target_layer = int(boundary["target_layer"])
            delta = None if predicted_layer is None else predicted_layer - target_layer
            bucket = first_deviation_bucket(predicted_layer, target_layer)
            table.append(
                {
                    "architecture": architecture,
                    "uid": output["uid"],
                    "dataset": output["dataset"],
                    "boundary_layer": target_layer,
                    "predicted_first_deviation_layer": predicted_layer,
                    "error_layers": delta,
                    "bucket": bucket,
                    "within_1": bool(delta is not None and abs(delta) <= 1),
                    "within_2": bool(delta is not None and abs(delta) <= 2),
                    "predicted_first_action": (
                        "FULL" if predicted_layer is None else actions[predicted_layer]
                    ),
                    "rescued": bool(output["correct"]),
                }
            )
        current = [row for row in table if row["architecture"] == architecture]
        counts = Counter(row["bucket"] for row in current)
        near = [row for row in current if row["within_2"]]
        summaries[architecture] = {
            "records": len(current),
            "exact": counts["exact"] / len(current),
            "within_1": sum(row["within_1"] for row in current) / len(current),
            "within_2": sum(row["within_2"] for row in current) / len(current),
            "too_early": counts["too_early"] / len(current),
            "too_late": counts["too_late"] / len(current),
            "never": counts["never"] / len(current),
            "rescue_given_within_2": (
                sum(row["rescued"] for row in near) / len(near) if near else float("nan")
            ),
            "bucket_counts": dict(sorted(counts.items())),
        }
    return table, summaries


def layer_baseline(
    rows: list[dict[str, Any]], *, alpha: float
) -> dict[str, Any]:
    output = {}
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    for task, label_key in (
        ("when", "when_label"),
        ("read_off", "read_off_label"),
        ("write_off", "write_off_label"),
    ):
        task_train = [row for row in train if row[label_key] is not None]
        task_validation = [row for row in validation if row[label_key] is not None]
        scores = layer_only_binary_scores(
            task_train,
            task_validation,
            label_key=label_key,
            alpha=alpha,
            num_layers=28,
        )
        output[task] = {
            "train_records": len(task_train),
            "validation_records": len(task_validation),
            "validation": binary_metrics(
                [int(row[label_key]) for row in task_validation], scores
            ),
            "validation_scores": scores,
        }
    class_order = list(FOUR_ACTIONS)
    train_singleton = [row for row in train if row["singleton"]]
    val_singleton = [row for row in validation if row["singleton"]]
    counts = {layer: Counter() for layer in range(28)}
    for row in train_singleton:
        counts[int(row["target_layer"])][row["mechanism_class"]] += 1
    predictions = []
    for row in val_singleton:
        layer_counts = counts[int(row["target_layer"])]
        probabilities_by_class = {
            value: (layer_counts[value] + alpha)
            / (sum(layer_counts.values()) + alpha * len(class_order))
            for value in class_order
        }
        predictions.append(max(class_order, key=lambda value: (probabilities_by_class[value], -class_order.index(value))))
    output["singleton_four_action"] = {
        "train_records": len(train_singleton),
        "validation_records": len(val_singleton),
        "validation": multiclass_metrics(
            [row["mechanism_class"] for row in val_singleton],
            predictions,
            classes=class_order,
        ),
    }
    return output


def state_shuffle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {"shuffle": "joint_text_and_visual_within_split_dataset_layer", "rows": []}
    for split in ("train", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        for scope_type, scope_value, selected in scoped(split_rows, include_layer=False):
            entry: dict[str, Any] = {
                "split": split,
                "scope_type": scope_type,
                "scope_value": scope_value,
                "records": len(selected),
            }
            for task, label_key in (
                ("when", "when_label"),
                ("read_off", "read_off_label"),
                ("write_off", "write_off_label"),
            ):
                eligible = [row for row in selected if row[label_key] is not None]
                labels = [int(row[label_key]) for row in eligible]
                original = [architecture_values(row, "online")[task] for row in eligible]
                shuffled = [architecture_values(row, "online", shuffled=True)[task] for row in eligible]
                original_metrics = binary_metrics(labels, original)
                shuffled_metrics = binary_metrics(labels, shuffled)
                entry[task] = {
                    "original": original_metrics,
                    "shuffled": shuffled_metrics,
                    "auroc_drop": original_metrics["auroc"] - shuffled_metrics["auroc"],
                }
            original_actions = [architecture_values(row, "online")["predicted_action"] for row in selected]
            shuffled_actions = [architecture_values(row, "online", shuffled=True)["predicted_action"] for row in selected]
            singleton = [row for row in selected if row["singleton"]]
            entry.update(
                {
                    "prediction_unchanged_fraction": sum(a == b for a, b in zip(original_actions, shuffled_actions)) / len(selected),
                    "mean_kl_original_to_shuffled": float(np.mean([row["online_shuffle_kl"] for row in selected])),
                    "singleton_valid_action_at_1_original": sum(architecture_values(row, "online")["valid"] for row in singleton) / len(singleton),
                    "singleton_valid_action_at_1_shuffled": sum(architecture_values(row, "online", shuffled=True)["valid"] for row in singleton) / len(singleton),
                }
            )
            output["rows"].append(entry)
    return output


def representation(row: dict[str, Any], name: str) -> np.ndarray:
    if name == "upfront":
        return np.asarray(row["polar_feature"], dtype=np.float32)
    if name == "online":
        return np.concatenate(
            (
                np.asarray(row["online_z_read"], dtype=np.float32),
                np.asarray(row["online_z_write"], dtype=np.float32),
            )
        )
    if name == "z_R":
        return np.asarray(row["online_z_read"], dtype=np.float32)
    if name == "z_W":
        return np.asarray(row["online_z_write"], dtype=np.float32)
    raise ValueError(f"unknown representation: {name}")


def task_rows(rows: list[dict[str, Any]], task: str, split: str) -> tuple[list[dict[str, Any]], list[Any]]:
    selected = [row for row in rows if row["split"] == split]
    if task == "when":
        return selected, [int(row["when_label"]) for row in selected]
    if task in {"read_off", "write_off"}:
        selected = [row for row in selected if row[f"{task}_label"] is not None]
        return selected, [int(row[f"{task}_label"]) for row in selected]
    if task == "mechanism":
        selected = [
            row
            for row in selected
            if row["state_kind"] == "mandatory_deviation" and row["singleton"]
        ]
        return selected, [MECHANISM_CLASSES.index(row["mechanism_class"]) for row in selected]
    raise ValueError(f"unknown probe task: {task}")


class DiagnosticProbe(nn.Module):
    def __init__(self, input_size: int, output_size: int, *, model_type: str, hidden_size: int, dropout: float) -> None:
        super().__init__()
        if model_type == "linear":
            self.model = nn.Linear(input_size, output_size)
        elif model_type == "mlp":
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            )
        else:
            raise ValueError("diagnostic probe type must be linear or mlp")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.model(values)


def _probe_seed(base_seed: int, representation_name: str, task: str, model_type: str) -> int:
    return int.from_bytes(
        sha256(f"{base_seed}:{representation_name}:{task}:{model_type}".encode()).digest()[:4]
    )


def train_probe(
    train_features: np.ndarray,
    train_labels: Sequence[int],
    validation_features: np.ndarray,
    validation_labels: Sequence[int],
    *,
    task: str,
    model_type: str,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-6] = 1.0
    train_x = torch.from_numpy(((train_features - mean) / std).astype(np.float32))
    validation_x = torch.from_numpy(((validation_features - mean) / std).astype(np.float32))
    train_y = torch.tensor(train_labels, dtype=torch.long)
    validation_y = torch.tensor(validation_labels, dtype=torch.long)
    output_size = 1 if task != "mechanism" else len(MECHANISM_CLASSES)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = DiagnosticProbe(
        train_x.shape[1],
        output_size,
        model_type=model_type,
        hidden_size=int(config["mlp_hidden_size"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = int(config["batch_size"])
    for _epoch in range(int(config["epochs"])):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x[indices].to(device))
            if task == "mechanism":
                loss = nn.functional.cross_entropy(logits, train_y[indices].to(device))
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), train_y[indices].float().to(device)
                )
            loss.backward()
            optimizer.step()

    def evaluate(values: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        model.eval()
        with torch.inference_mode():
            logits = model(values.to(device)).cpu()
        if task == "mechanism":
            predicted = logits.argmax(dim=-1).tolist()
            return multiclass_metrics(
                [MECHANISM_CLASSES[int(value)] for value in labels.tolist()],
                [MECHANISM_CLASSES[int(value)] for value in predicted],
                classes=MECHANISM_CLASSES,
            )
        scores = torch.sigmoid(logits.squeeze(-1)).tolist()
        return binary_metrics(labels.tolist(), scores)

    return {
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train": evaluate(train_x, train_y),
        "validation": evaluate(validation_x, validation_y),
    }


def probe_results(rows: list[dict[str, Any]], config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    results = {
        "contract": config,
        "representations": {},
    }
    for representation_name in REPRESENTATIONS:
        results["representations"][representation_name] = {}
        for task in ("when", "read_off", "write_off", "mechanism"):
            train_rows, train_labels = task_rows(rows, task, "train")
            val_rows, val_labels = task_rows(rows, task, "validation")
            train_x = np.stack([representation(row, representation_name) for row in train_rows])
            val_x = np.stack([representation(row, representation_name) for row in val_rows])
            results["representations"][representation_name][task] = {}
            for model_type in config["models"]:
                seed = _probe_seed(int(config["seed"]), representation_name, task, model_type)
                results["representations"][representation_name][task][model_type] = train_probe(
                    train_x,
                    train_labels,
                    val_x,
                    val_labels,
                    task=task,
                    model_type=model_type,
                    config=config,
                    seed=seed,
                    device=device,
                )
    return results


def _conditional_entropy(rows: list[dict[str, Any]], label: Callable[[dict[str, Any]], Any], keys: Sequence[str]) -> float:
    groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for row in rows:
        value = label(row)
        if value is None:
            continue
        groups[tuple(row[key] for key in keys)].append(value)
    total = sum(len(values) for values in groups.values())
    entropy = 0.0
    for values in groups.values():
        counts = Counter(values)
        cell_entropy = -sum(
            (count / len(values)) * math.log2(count / len(values))
            for count in counts.values()
        )
        entropy += len(values) / total * cell_entropy
    return entropy


def knn_results(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    output = {"contract": config, "representations": {}, "entropy": {}}
    for representation_name in REPRESENTATIONS:
        output["representations"][representation_name] = {}
        for task in ("when", "read_off", "write_off", "mechanism"):
            train_rows, train_labels = task_rows(rows, task, "train")
            val_rows, val_labels = task_rows(rows, task, "validation")
            train_x = np.stack([representation(row, representation_name) for row in train_rows]).astype(np.float64)
            val_x = np.stack([representation(row, representation_name) for row in val_rows]).astype(np.float64)
            mean = train_x.mean(axis=0)
            std = train_x.std(axis=0)
            std[std < 1e-6] = 1.0
            train_x = (train_x - mean) / std
            val_x = (val_x - mean) / std
            output["representations"][representation_name][task] = knn_label_consistency(
                train_x,
                train_labels,
                train_rows,
                val_x,
                val_labels,
                val_rows,
                k_values=config["knn_k"],
            )
    train = [row for row in rows if row["split"] == "train"]
    labels = {
        "action_set": lambda row: "|".join(row["valid_actions"]),
        "read_off": lambda row: row["read_off_label"],
        "write_off": lambda row: row["write_off_label"],
    }
    for name, label in labels.items():
        output["entropy"][name] = {
            "H_label_given_layer": _conditional_entropy(train, label, ["target_layer"]),
            "H_label_given_layer_dataset": _conditional_entropy(
                train, label, ["target_layer", "dataset"]
            ),
        }
    return output


def freeze_label_subset(
    rows: list[dict[str, Any]], config: dict[str, Any], analysis_dir: Path
) -> list[dict[str, Any]]:
    manifest_states = load_jsonl(config["data"]["state_manifest"])
    outputs = {
        architecture: {
            row["state_id"]: {
                "predicted_action": architecture_values(row, architecture)["predicted_action"],
                "action_probabilities": probabilities(row, architecture).tolist(),
            }
            for row in rows
        }
        for architecture in ARCHITECTURES
    }
    subset = build_label_incompleteness_subset(
        manifest_states,
        outputs,
        cap_per_architecture_action=int(
            config["label_incompleteness"]["cap_per_architecture_action"]
        ),
        seed=int(config["label_incompleteness"]["seed"]),
    )
    source_rows = load_verified_manifest(
        config["data"]["source_manifest"], config["data"]["source_manifest_sha256"]
    )
    source_by_uid = {row["uid"]: row for row in source_rows}
    expanded = []
    for row in subset:
        candidates = candidate_suffix_routes(
            source_by_uid[row["uid"]],
            row,
            predicted_action=row["predicted_action"],
            max_suffixes=int(config["label_incompleteness"]["max_known_suffixes"]),
            seed=int(config["label_incompleteness"]["seed"]),
        )
        expanded.append({**row, "candidate_routes": candidates})
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in expanded)
    path = analysis_dir / "label_incompleteness_subset.jsonl"
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError("frozen label-incompleteness subset changed")
    else:
        write_frozen(path, text)
    return expanded


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def create_figures(
    figures_dir: Path,
    *,
    gap_rows: list[dict[str, Any]],
    bit_rows: list[dict[str, Any]],
    confusion: dict[str, Any],
    timing_rows: list[dict[str, Any]],
    timing_summary: dict[str, Any],
    layer_only: dict[str, Any],
    when_primary: dict[str, Any],
    shuffle: dict[str, Any],
    knn: dict[str, Any],
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    # 1. Train-to-validation mechanism performance.
    metrics = ["KEEP_vs_DEVIATE_AUROC", "READ_OFF_AUROC", "WRITE_OFF_AUROC", "IGNORE_only_recall"]
    x = np.arange(len(metrics))
    width = 0.2
    for offset, (architecture, split) in enumerate(
        (("polar", "train"), ("polar", "validation"), ("online", "train"), ("online", "validation"))
    ):
        values = [
            next(row[split] for row in gap_rows if row["architecture"] == architecture and row["metric"] == metric)
            for metric in metrics
        ]
        plt.bar(x + (offset - 1.5) * width, values, width, label=f"{architecture}-{split}")
    plt.xticks(x, ["WHEN", "READ_OFF", "WRITE_OFF", "IGNORE"])
    plt.ylim(0, 1)
    plt.ylabel("Metric value")
    plt.legend(fontsize=7)
    _save_figure(figures_dir / "train_validation_mechanism_performance.png")

    # 2. READ/WRITE AUROC by layer.
    for architecture, style in (("polar", "-"), ("online", "--")):
        for bit, color in (("read_off", "tab:blue"), ("write_off", "tab:orange")):
            selected = [
                row for row in bit_rows
                if row["architecture"] == architecture and row["split"] == "validation"
                and row["bit"] == bit and row["scope_type"] == "layer" and math.isfinite(float(row["auroc"]))
            ]
            plt.plot(
                [int(row["scope_value"]) for row in selected],
                [row["auroc"] for row in selected],
                linestyle=style,
                color=color,
                label=f"{architecture}-{bit}",
            )
    plt.ylim(0, 1)
    plt.xlabel("Layer")
    plt.ylabel("Validation AUROC")
    plt.legend(fontsize=7)
    _save_figure(figures_dir / "read_write_auroc_by_layer.png")

    # 3. Validation singleton confusion matrices.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, architecture in zip(axes, ARCHITECTURES):
        matrix = np.asarray(
            [[confusion[architecture]["validation"]["confusion"][target][pred] for pred in FOUR_ACTIONS] for target in FOUR_ACTIONS]
        )
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(architecture)
        axis.set_xticks(range(4), FOUR_ACTIONS, rotation=45, ha="right")
        axis.set_yticks(range(4), FOUR_ACTIONS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Target")
        for i in range(4):
            for j in range(4):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.046)
    _save_figure(figures_dir / "singleton_confusion_matrix.png")

    # 4. First-deviation error histogram.
    bins = np.arange(-29.5, 29.5, 1)
    for architecture in ARCHITECTURES:
        values = [row["error_layers"] for row in timing_rows if row["architecture"] == architecture and row["error_layers"] is not None]
        plt.hist(values, bins=bins, alpha=0.5, label=architecture)
    plt.xlabel("First-deviation layer minus mandatory boundary")
    plt.ylabel("Validation W2C records")
    plt.legend()
    _save_figure(figures_dir / "first_deviation_error_histogram.png")

    # 5. Rescue probability by timing bucket.
    buckets = ["too_early", "within_2_early", "within_1_early", "exact", "within_1_late", "within_2_late", "too_late", "never"]
    for architecture in ARCHITECTURES:
        values = []
        for bucket in buckets:
            selected = [row for row in timing_rows if row["architecture"] == architecture and row["bucket"] == bucket]
            values.append(sum(row["rescued"] for row in selected) / len(selected) if selected else np.nan)
        plt.plot(range(len(buckets)), values, marker="o", label=architecture)
    plt.xticks(range(len(buckets)), buckets, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("W2C rescue probability")
    plt.legend()
    _save_figure(figures_dir / "rescue_probability_by_deviation_error.png")

    # 6. Layer-only versus router WHEN performance.
    labels = ["layer-only", "POLAR", "Online"]
    values = [
        layer_only["when"]["validation"]["auroc"],
        when_primary["polar"]["validation"]["auroc"],
        when_primary["online"]["validation"]["auroc"],
    ]
    plt.bar(labels, values, color=["grey", "tab:blue", "tab:orange"])
    plt.ylim(0, 1)
    plt.ylabel("Validation KEEP/DEVIATE AUROC")
    _save_figure(figures_dir / "layer_only_vs_full_state.png")

    # 7. State-shuffle AUROC effects.
    overall = next(row for row in shuffle["rows"] if row["split"] == "validation" and row["scope_type"] == "overall")
    tasks = ["when", "read_off", "write_off"]
    x = np.arange(len(tasks))
    plt.bar(x - 0.18, [overall[task]["original"]["auroc"] for task in tasks], 0.36, label="original")
    plt.bar(x + 0.18, [overall[task]["shuffled"]["auroc"] for task in tasks], 0.36, label="shuffled")
    plt.xticks(x, tasks)
    plt.ylim(0, 1)
    plt.ylabel("Online validation AUROC")
    plt.legend()
    _save_figure(figures_dir / "online_state_shuffle_effect.png")

    # 8. kNN distance versus label agreement.
    for representation_name in REPRESENTATIONS:
        pairs = knn["representations"][representation_name]["when"]["neighbor_pairs_at_max_k"]
        distances = np.asarray([row["distance"] for row in pairs])
        agreements = np.asarray([row["agreement"] for row in pairs])
        edges = np.quantile(distances, np.linspace(0, 1, 11))
        centers, rates = [], []
        for index in range(10):
            mask = (distances >= edges[index]) & (distances <= edges[index + 1] if index == 9 else distances < edges[index + 1])
            if mask.any():
                centers.append(float(distances[mask].mean()))
                rates.append(float(agreements[mask].mean()))
        plt.plot(centers, rates, marker="o", label=representation_name)
    plt.ylim(0, 1)
    plt.xlabel("Cosine distance to training neighbor")
    plt.ylabel("KEEP/DEVIATE label agreement")
    plt.legend()
    _save_figure(figures_dir / "knn_distance_vs_label_agreement.png")


def render_summary(
    *,
    when: dict[str, Any],
    what: dict[str, Any],
    bits: dict[str, Any],
    confusion: dict[str, Any],
    timing: dict[str, Any],
    layer: dict[str, Any],
    shuffle: dict[str, Any],
    probes: dict[str, Any],
    knn: dict[str, Any],
    label_subset: list[dict[str, Any]],
) -> str:
    lines = [
        "# Four-Action Generalization Diagnostic Summary",
        "",
        "## Main failure decomposition",
        "",
        "| Metric | POLAR Train | POLAR Val | Online Train | Online Val |",
        "|---|---:|---:|---:|---:|",
    ]
    metrics = [
        ("KEEP vs DEVIATE AUROC", lambda arch, split: when[arch][split]["auroc"]),
        ("DEVIATE recall", lambda arch, split: what[arch][split]["deviate_recall"]),
        ("Conditional Valid-Action@1", lambda arch, split: what[arch][split]["conditional_valid_action_at_1"]),
        ("READ_OFF AUROC", lambda arch, split: bits[arch][f"{split}:read_off"]["auroc"]),
        ("WRITE_OFF AUROC", lambda arch, split: bits[arch][f"{split}:write_off"]["auroc"]),
        ("READ_ONLY-only recall", lambda arch, split: confusion[arch][split]["by_class"]["READ_ONLY"]["recall"]),
        ("WRITE_ONLY-only recall", lambda arch, split: confusion[arch][split]["by_class"]["WRITE_ONLY"]["recall"]),
        ("IGNORE-only recall", lambda arch, split: confusion[arch][split]["by_class"]["IGNORE"]["recall"]),
    ]
    for name, getter in metrics:
        lines.append(
            f"| {name} | {getter('polar', 'train'):.6f} | {getter('polar', 'validation'):.6f} | "
            f"{getter('online', 'train'):.6f} | {getter('online', 'validation'):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Timing",
            "",
            "| Metric | POLAR | Online |",
            "|---|---:|---:|",
        ]
    )
    for metric in ("exact", "within_1", "within_2", "too_early", "too_late", "never", "rescue_given_within_2"):
        lines.append(f"| {metric} | {timing['polar'][metric]:.6f} | {timing['online'][metric]:.6f} |")
    shuffle_val = next(row for row in shuffle["rows"] if row["split"] == "validation" and row["scope_type"] == "overall")
    lines.extend(
        [
            "",
            "## Representation usage",
            "",
            f"- Layer-only validation WHEN AUROC: {layer['when']['validation']['auroc']:.6f}.",
            f"- Online joint-shuffle prediction unchanged: {shuffle_val['prediction_unchanged_fraction']:.6f}.",
            f"- Online joint-shuffle WHEN AUROC drop: {shuffle_val['when']['auroc_drop']:.6f}.",
            f"- Online joint-shuffle READ_OFF AUROC drop: {shuffle_val['read_off']['auroc_drop']:.6f}.",
            f"- Online joint-shuffle WRITE_OFF AUROC drop: {shuffle_val['write_off']['auroc_drop']:.6f}.",
            "",
            "## Diagnostic probes and label smoothness",
            "",
        ]
    )
    for representation_name in REPRESENTATIONS:
        when_probe = probes["representations"][representation_name]["when"]["mlp"]["validation"]["auroc"]
        purity = knn["representations"][representation_name]["when"]["by_k"]["10"]["mean_label_purity"]
        lines.append(
            f"- {representation_name}: MLP WHEN AUROC {when_probe:.6f}; k=10 WHEN purity {purity:.6f}."
        )
    lines.extend(
        [
            "",
            "## Bounded label-incompleteness audit",
            "",
            f"- Frozen cached-invalid validation states: {len(label_subset)}.",
            "- Execution results are reported in `label_incompleteness_results.json`.",
            "",
            "## Interpretation boundary",
            "",
            "These measurements apply to the frozen Phase-39 subset, selected",
            "checkpoints, exact state construction, and fixed probes. Probe success",
            "does not authorize a new objective/head, and bounded audit failures do",
            "not prove global action invalidity.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="analysis/4action_generalization_diagnostics/diagnostic_config.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_path = Path(args.config)
    config_sha = file_sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "four_action_generalization_diagnostics_v1":
        raise RuntimeError("analysis requires the frozen diagnostic config")
    if file_sha256(Path(config["source_plan"])) != config["source_plan_sha256"]:
        raise RuntimeError("diagnostic source-plan checksum mismatch")
    rows = load_joined(config, config_sha)
    analysis_dir = Path(config["reporting"]["analysis_dir"])

    when_rows, when_primary = when_table(rows)
    what_rows, what_primary = what_table(rows)
    polar_confusion_rows, polar_confusion = singleton_confusion(rows, "polar")
    online_confusion_rows, online_confusion = singleton_confusion(rows, "online")
    confusion = {"polar": polar_confusion, "online": online_confusion}
    bit_rows, bit_primary = bit_table(rows)
    both_rows, both_primary = both_off_table(rows)
    gap_rows = train_validation_gap(when_primary, what_primary, bit_primary, confusion)
    timing_rows, timing_summary = first_deviation_table(config, rows)
    layer_only = layer_baseline(rows, alpha=float(config["analysis"]["layer_only_alpha"]))
    shuffle = state_shuffle(rows)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA diagnostic probes but CUDA is unavailable")
    random.seed(int(config["probes"]["seed"]))
    np.random.seed(int(config["probes"]["seed"]) % (2**32))
    torch.manual_seed(int(config["probes"]["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=False)
    probes = probe_results(rows, config["probes"], device)
    knn = knn_results(rows, config["probes"])
    label_subset = freeze_label_subset(rows, config, analysis_dir)

    atomic_csv(analysis_dir / "when_keep_vs_deviate.csv", when_rows)
    atomic_csv(analysis_dir / "what_conditional_mechanism.csv", what_rows)
    atomic_csv(analysis_dir / "singleton_confusion_polar.csv", polar_confusion_rows)
    atomic_csv(analysis_dir / "singleton_confusion_online.csv", online_confusion_rows)
    atomic_csv(analysis_dir / "read_write_bit_metrics.csv", bit_rows)
    atomic_csv(analysis_dir / "both_off_error_breakdown.csv", both_rows)
    atomic_csv(analysis_dir / "train_val_gap.csv", gap_rows)
    atomic_csv(analysis_dir / "first_deviation_analysis.csv", timing_rows)
    atomic_json(analysis_dir / "layer_only_baseline.json", layer_only)
    atomic_json(analysis_dir / "state_shuffle_results.json", shuffle)
    atomic_json(analysis_dir / "representation_probe_results.json", probes)
    atomic_json(
        analysis_dir / "knn_label_consistency.json",
        compact_knn_label_consistency(knn),
    )
    create_figures(
        Path(config["reporting"]["figures"]),
        gap_rows=gap_rows,
        bit_rows=bit_rows,
        confusion=confusion,
        timing_rows=timing_rows,
        timing_summary=timing_summary,
        layer_only=layer_only,
        when_primary=when_primary,
        shuffle=shuffle,
        knn=knn,
    )
    Path(config["reporting"]["diagnostic_summary"]).write_text(
        render_summary(
            when=when_primary,
            what=what_primary,
            bits=bit_primary,
            confusion=confusion,
            timing=timing_summary,
            layer=layer_only,
            shuffle=shuffle,
            probes=probes,
            knn=knn,
            label_subset=label_subset,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "four_action_generalization_priority_1_2_complete",
                "states": len(rows),
                "label_audit_states": len(label_subset),
                "polar_validation_when_auroc": when_primary["polar"]["validation"]["auroc"],
                "online_validation_when_auroc": when_primary["online"]["validation"]["auroc"],
                "summary": config["reporting"]["diagnostic_summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
