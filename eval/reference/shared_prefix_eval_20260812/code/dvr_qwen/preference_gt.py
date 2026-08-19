"""Utilities for the 10k correctness-first preference GT dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


NUM_LAYERS = 28
DATASET_VERSION = "correctness_first_preference_gt_v1"
ALLOWED_DATASET_VERSIONS = frozenset(
    {
        DATASET_VERSION,
        "correctness_first_preference_gt_v2",
        "correctness_first_preference_gt_v3",
        "correctness_first_preference_gt_v31",
        "correctness_first_preference_gt_v32",
        "correctness_first_preference_gt_v31_qwen25vl3b",
    }
)
MODEL_RUNTIME_ID = "qwen25vl7b_cc594898_sdpa_greedy_v1"
ALLOWED_MODEL_RUNTIME_IDS = frozenset(
    {
        MODEL_RUNTIME_ID,
        "qwen25vl3b_66285546_sdpa_greedy_v1",
    }
)
CORRECTNESS_THRESHOLDS = {
    "chartqa": 1.0,
    "docvqa": 0.5,
    "gqa": 1.0,
    "textvqa": 0.5,
    "dynamath_generated": 1.0,
    "vinteraction": 1.0,
    "wemath2_sft": 1.0,
}


@dataclass(frozen=True)
class PreferenceGTDatasetPaths:
    root: Path

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    @property
    def sample_targets(self) -> Path:
        return self.root / "sample_targets.jsonl"

    @property
    def training_routes(self) -> Path:
        return self.root / "training_routes.jsonl"

    @property
    def train_pairs(self) -> Path:
        return self.root / "train_preference_pairs.jsonl"

    @property
    def validation_pairs(self) -> Path:
        return self.root / "validation_preference_pairs.jsonl"

    @property
    def excluded_source_score_mismatches(self) -> Path:
        return self.root / "excluded_source_score_mismatches.jsonl"

    @property
    def benchmark_split_summary(self) -> Path:
        return self.root / "benchmark_split_summary.jsonl"

    def require_files(self) -> None:
        missing = [
            str(path)
            for path in [
                self.metadata,
                self.summary,
                self.sample_targets,
                self.training_routes,
                self.train_pairs,
                self.validation_pairs,
                self.excluded_source_score_mismatches,
                self.benchmark_split_summary,
            ]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"missing preference GT files: {missing}")


def iter_jsonl(path: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if limit is not None and emitted >= limit:
                return
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            emitted += 1
            yield row


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_mask_key(mask_key: str, *, num_layers: int = NUM_LAYERS) -> None:
    if len(str(mask_key)) != int(num_layers) or any(bit not in "01" for bit in str(mask_key)):
        raise ValueError(f"mask_key must be a {num_layers}-bit binary string, got {mask_key!r}")


def mask_key_to_list(mask_key: str, *, num_layers: int = NUM_LAYERS) -> list[int]:
    validate_mask_key(mask_key, num_layers=num_layers)
    return [1 if bit == "1" else 0 for bit in str(mask_key)]


def mask_key_to_tensor(mask_key: str, *, num_layers: int = NUM_LAYERS) -> torch.Tensor:
    return torch.tensor(mask_key_to_list(mask_key, num_layers=num_layers), dtype=torch.float32)


def mask_to_key(mask: Iterable[int | bool | float], *, num_layers: int = NUM_LAYERS) -> str:
    values = [int(value) for value in mask]
    if len(values) != int(num_layers) or any(value not in (0, 1) for value in values):
        raise ValueError(f"mask must contain {num_layers} binary values")
    return "".join(str(value) for value in values)


def layers_one_based_from_mask_key(mask_key: str, *, num_layers: int = NUM_LAYERS) -> list[int]:
    return [idx + 1 for idx, value in enumerate(mask_key_to_list(mask_key, num_layers=num_layers)) if value]


def transition_count(mask_key: str, *, num_layers: int = NUM_LAYERS) -> int:
    values = mask_key_to_list(mask_key, num_layers=num_layers)
    return sum(int(left != right) for left, right in zip(values, values[1:]))


def correctness_threshold(benchmark: str) -> float:
    key = str(benchmark).lower()
    if key not in CORRECTNESS_THRESHOLDS:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    return float(CORRECTNESS_THRESHOLDS[key])


def is_correct_score(benchmark: str, score: float) -> bool:
    return float(score) >= correctness_threshold(benchmark)


def row_num_layers(row: dict[str, Any]) -> int:
    num_layers = int(row.get("num_layers", NUM_LAYERS))
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    return num_layers


def validate_dataset_version(row: dict[str, Any], row_id: str) -> None:
    if row.get("dataset_version") not in ALLOWED_DATASET_VERSIONS:
        raise ValueError(f"{row_id}: unexpected dataset_version {row.get('dataset_version')!r}")


def validate_sample_target(row: dict[str, Any]) -> None:
    uid = str(row.get("uid", ""))
    validate_dataset_version(row, uid)
    if row.get("model_runtime_id") not in ALLOWED_MODEL_RUNTIME_IDS:
        raise ValueError(f"{uid}: unexpected model_runtime_id {row.get('model_runtime_id')!r}")
    num_layers = row_num_layers(row)
    benchmark = str(row.get("benchmark", "")).lower()
    if float(row.get("correctness_threshold")) != correctness_threshold(benchmark):
        raise ValueError(f"{uid}: correctness threshold mismatch")
    if row.get("split") not in {"train", "validation"}:
        raise ValueError(f"{uid}: invalid split {row.get('split')!r}")

    cooptimal_ids = list(row.get("cooptimal_route_ids") or [])
    cooptimal_keys = list(row.get("cooptimal_mask_keys") or [])
    if len(cooptimal_ids) != len(cooptimal_keys):
        raise ValueError(f"{uid}: cooptimal route/mask count mismatch")
    if int(row.get("cooptimal_route_count", 0)) != len(cooptimal_ids):
        raise ValueError(f"{uid}: cooptimal_route_count mismatch")
    for mask_key in cooptimal_keys:
        validate_mask_key(str(mask_key), num_layers=num_layers)

    correct_route_count = int(row.get("correct_route_count", 0))
    eligible = bool(row.get("training_eligible"))
    if eligible != (correct_route_count > 0):
        raise ValueError(f"{uid}: training_eligible does not match correct_route_count")
    if eligible and row.get("minimum_correct_budget") is None:
        raise ValueError(f"{uid}: eligible sample has no minimum_correct_budget")
    if not eligible and (cooptimal_ids or cooptimal_keys or row.get("minimum_correct_budget") is not None):
        raise ValueError(f"{uid}: ineligible sample has cooptimal targets")


def validate_route_row(row: dict[str, Any]) -> None:
    route_id = str(row.get("route_id", ""))
    validate_dataset_version(row, route_id)
    if row.get("model_runtime_id") not in ALLOWED_MODEL_RUNTIME_IDS:
        raise ValueError(f"{route_id}: unexpected model_runtime_id {row.get('model_runtime_id')!r}")
    num_layers = row_num_layers(row)
    mask_key = str(row.get("mask_key", ""))
    validate_mask_key(mask_key, num_layers=num_layers)
    mask = [int(value) for value in row.get("visual_on_mask", [])]
    if mask_to_key(mask, num_layers=num_layers) != mask_key:
        raise ValueError(f"{route_id}: visual_on_mask and mask_key differ")
    if int(row.get("num_visual_on_layers")) != mask_key.count("1"):
        raise ValueError(f"{route_id}: num_visual_on_layers does not match mask_key")
    uid = str(row.get("uid", ""))
    benchmark = uid.split(":", 1)[0] if ":" in uid else str(row.get("benchmark", ""))
    if bool(row.get("result_correct")) != is_correct_score(benchmark, float(row.get("score", 0.0))):
        raise ValueError(f"{route_id}: result_correct does not match score threshold")


def validate_preference_pair(row: dict[str, Any], *, expected_split: str | None = None) -> None:
    pair_id = str(row.get("pair_id", ""))
    validate_dataset_version(row, pair_id)
    if row.get("model_runtime_id") not in ALLOWED_MODEL_RUNTIME_IDS:
        raise ValueError(f"{pair_id}: unexpected model_runtime_id {row.get('model_runtime_id')!r}")
    num_layers = row_num_layers(row)
    if expected_split is not None and row.get("split") != expected_split:
        raise ValueError(f"{pair_id}: expected split {expected_split!r}, got {row.get('split')!r}")
    for side in ("chosen", "rejected"):
        mask_key = str(row.get(f"{side}_mask_key", ""))
        validate_mask_key(mask_key, num_layers=num_layers)
        if int(row.get(f"{side}_budget")) != mask_key.count("1"):
            raise ValueError(f"{pair_id}: {side}_budget does not match {side}_mask_key")
    chosen_budget = int(row.get("chosen_budget"))
    rejected_budget = int(row.get("rejected_budget"))
    if int(row.get("budget_delta_rejected_minus_chosen")) != rejected_budget - chosen_budget:
        raise ValueError(f"{pair_id}: budget delta mismatch")

    pair_type = str(row.get("pair_type"))
    chosen_correct = bool(row.get("chosen_correct"))
    rejected_correct = bool(row.get("rejected_correct"))
    if pair_type == "correctness":
        if not (chosen_correct and not rejected_correct):
            raise ValueError(f"{pair_id}: correctness pair must be correct > incorrect")
        if int(row.get("objective_level")) != 1 or float(row.get("recommended_weight")) != 1.0:
            raise ValueError(f"{pair_id}: invalid correctness objective metadata")
    elif pair_type == "efficiency":
        if not (chosen_correct and rejected_correct and chosen_budget < rejected_budget):
            raise ValueError(f"{pair_id}: efficiency pair must be lower-budget correct > higher-budget correct")
        if int(row.get("objective_level")) != 2 or float(row.get("recommended_weight")) != 0.5:
            raise ValueError(f"{pair_id}: invalid efficiency objective metadata")
    else:
        raise ValueError(f"{pair_id}: unknown pair_type {pair_type!r}")


def cooptimal_soft_label(row: dict[str, Any], *, num_layers: int = NUM_LAYERS) -> torch.Tensor:
    """Average the finite-candidate co-optimal masks into a soft layer target."""
    validate_sample_target(row)
    resolved_num_layers = row_num_layers(row) if int(num_layers) == NUM_LAYERS else int(num_layers)
    keys = [str(mask_key) for mask_key in row.get("cooptimal_mask_keys") or []]
    if not keys:
        return torch.zeros(resolved_num_layers, dtype=torch.float32)
    masks = torch.stack(
        [mask_key_to_tensor(mask_key, num_layers=resolved_num_layers) for mask_key in keys], dim=0
    )
    return masks.mean(dim=0).float()


def minimum_budget_representative_mask_key(row: dict[str, Any]) -> str | None:
    """Return a deterministic representative only for baselines that need one mask."""
    validate_sample_target(row)
    keys = [str(mask_key) for mask_key in row.get("cooptimal_mask_keys") or []]
    if not keys:
        return None
    return sorted(keys, key=lambda key: (key.count("1"), key))[0]


def route_score_from_layer_logits(
    layer_logits: torch.Tensor,
    mask_key: str,
    *,
    lambda_on: float = 0.0,
    lambda_transition: float = 0.0,
    normalize_by_on_count: bool = False,
    num_layers: int = NUM_LAYERS,
) -> torch.Tensor:
    """Score a route from per-layer logits without using route correctness labels."""
    logits = torch.as_tensor(layer_logits).float()
    if logits.ndim != 1 or int(logits.numel()) != int(num_layers):
        raise ValueError(f"layer_logits must have shape ({num_layers},), got {tuple(logits.shape)}")
    mask = mask_key_to_tensor(mask_key, num_layers=num_layers).to(device=logits.device)
    selected = (logits * mask).sum()
    on_count = float(mask.sum().item())
    if normalize_by_on_count and on_count > 0.0:
        selected = selected / on_count
    return selected - float(lambda_on) * mask.sum() - float(lambda_transition) * transition_count(mask_key)


def preference_pair_count_by_uid(pair_rows: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in pair_rows:
        counts[str(row["uid"])] += 1
    return counts


def inverse_uid_pair_weights(pair_rows: list[dict[str, Any]]) -> torch.Tensor:
    """Return per-pair weights that make each UID contribute equal total mass."""
    counts = preference_pair_count_by_uid(pair_rows)
    if not pair_rows:
        return torch.empty(0, dtype=torch.float32)
    return torch.tensor([1.0 / float(counts[str(row["uid"])]) for row in pair_rows], dtype=torch.float32)


def summarize_targets(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    eligible = 0
    no_correct = 0
    null_visual_optimal = 0
    by_group: Counter[tuple[str, str, str]] = Counter()
    budgets: list[int] = []
    cooptimal_counts: list[int] = []
    for row in rows:
        validate_sample_target(row)
        total += 1
        key = (str(row["split"]), str(row["benchmark"]), str(row["source_bucket"]))
        by_group[key] += 1
        if bool(row["training_eligible"]):
            eligible += 1
            budgets.append(int(row["minimum_correct_budget"]))
            cooptimal_counts.append(int(row["cooptimal_route_count"]))
        else:
            no_correct += 1
        null_visual_optimal += int(bool(row.get("null_visual_optimal")))
    return {
        "targets": total,
        "eligible": eligible,
        "no_correct_route": no_correct,
        "null_visual_optimal": null_visual_optimal,
        "by_split_benchmark_bucket": {"|".join(key): value for key, value in sorted(by_group.items())},
        "minimum_correct_budget": _numeric_summary(budgets),
        "cooptimal_route_count": _numeric_summary(cooptimal_counts),
    }


def summarize_pairs(rows: Iterable[dict[str, Any]], *, expected_split: str | None = None) -> dict[str, Any]:
    total = 0
    pair_types: Counter[str] = Counter()
    pair_subtypes: Counter[str] = Counter()
    by_group: Counter[tuple[str, str, str]] = Counter()
    per_uid: Counter[str] = Counter()
    budget_delta_by_type: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        validate_preference_pair(row, expected_split=expected_split)
        total += 1
        pair_type = str(row["pair_type"])
        pair_types[pair_type] += 1
        pair_subtypes[f"{pair_type}:{row['pair_subtype']}"] += 1
        by_group[(str(row["split"]), str(row["benchmark"]), str(row["source_bucket"]))] += 1
        per_uid[str(row["uid"])] += 1
        budget_delta_by_type[pair_type].append(int(row["budget_delta_rejected_minus_chosen"]))
    return {
        "pairs": total,
        "pair_type_counts": dict(sorted(pair_types.items())),
        "pair_subtype_counts": dict(sorted(pair_subtypes.items())),
        "by_split_benchmark_bucket": {"|".join(key): value for key, value in sorted(by_group.items())},
        "pairs_per_uid": _numeric_summary(list(per_uid.values())),
        "budget_delta_rejected_minus_chosen": {
            key: _numeric_summary(values) for key, values in sorted(budget_delta_by_type.items())
        },
    }


def summarize_preference_gt_directory(paths: PreferenceGTDatasetPaths) -> dict[str, Any]:
    paths.require_files()
    metadata = read_json(paths.metadata)
    summary = read_json(paths.summary)
    train_pair_summary = summarize_pairs(iter_jsonl(paths.train_pairs), expected_split="train")
    validation_pair_summary = summarize_pairs(iter_jsonl(paths.validation_pairs), expected_split="validation")
    target_summary = summarize_targets(iter_jsonl(paths.sample_targets))
    return {
        "dataset_dir": str(paths.root),
        "dataset_version": summary.get("dataset_version"),
        "model_runtime": metadata.get("model_runtime", {}),
        "recorded_summary": summary,
        "target_summary": target_summary,
        "train_pair_summary": train_pair_summary,
        "validation_pair_summary": validation_pair_summary,
    }


def _numeric_summary(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": sum(numeric) / float(len(numeric)),
    }
