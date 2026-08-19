"""Small preference-routing probes for the correctness-first GT dataset."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from dvr_qwen.preference_gt import NUM_LAYERS, mask_key_to_list, transition_count


BENCHMARK_NAMES = (
    "chartqa",
    "docvqa",
    "gqa",
    "textvqa",
    "dynamath_generated",
    "vinteraction",
    "wemath2_sft",
)
SOURCE_BUCKET_NAMES = ("complete_correct", "complete_wrong")
PAIR_TYPE_NAMES = ("correctness", "efficiency")
PAIR_TYPE_TO_ID = {name: idx for idx, name in enumerate(PAIR_TYPE_NAMES)}
FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "last_token": ("last_token",),
    "global_mean": ("global_mean",),
    "window_mean": ("window_mean",),
    "text_fusion": ("global_mean", "window_mean", "last_token"),
    "visual_summaries": ("visual_mean", "visual_abs_mean"),
    "text_visual_fusion": ("global_mean", "window_mean", "last_token", "visual_mean", "visual_abs_mean"),
}


def mask_key_to_bool_tensor(mask_key: str, *, num_layers: int = NUM_LAYERS) -> torch.Tensor:
    return torch.tensor(mask_key_to_list(mask_key, num_layers=num_layers), dtype=torch.bool)


def mask_keys_to_bool_tensor(mask_keys: Iterable[str], *, num_layers: int = NUM_LAYERS) -> torch.Tensor:
    masks = [mask_key_to_bool_tensor(str(mask_key), num_layers=num_layers) for mask_key in mask_keys]
    if not masks:
        return torch.empty((0, int(num_layers)), dtype=torch.bool)
    return torch.stack(masks, dim=0)


def expand_feature_fields(feature_spec: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(feature_spec, str):
        raw_items = [item.strip() for item in feature_spec.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip() for item in feature_spec if str(item).strip()]
    if not raw_items:
        raise ValueError("feature spec must contain at least one feature")
    fields: list[str] = []
    for item in raw_items:
        fields.extend(FEATURE_ALIASES.get(item, (item,)))
    valid = {"global_mean", "window_mean", "last_token", "visual_mean", "visual_abs_mean"}
    unknown = [field for field in fields if field not in valid]
    if unknown:
        raise ValueError(f"unknown feature fields {unknown}; valid fields are {sorted(valid | set(FEATURE_ALIASES))}")
    return tuple(fields)


def feature_tensor_from_payload(
    payload: dict[str, Any],
    feature_fields: str | Iterable[str],
    *,
    num_layers: int = NUM_LAYERS,
) -> torch.Tensor:
    """Concatenate cached per-layer feature fields into `[L, D]`."""
    fields = expand_feature_fields(feature_fields)
    parts: list[torch.Tensor] = []
    for field in fields:
        if field in {"global_mean", "window_mean", "last_token"}:
            if field not in payload:
                raise KeyError(f"missing feature field {field!r}")
            tensor = torch.as_tensor(payload[field]).detach().cpu()
        elif field in {"visual_mean", "visual_abs_mean"}:
            if "visual_summaries" not in payload:
                raise KeyError(f"missing feature field 'visual_summaries' for {field!r}")
            visual = torch.as_tensor(payload["visual_summaries"]).detach().cpu()
            if visual.ndim != 3 or int(visual.shape[1]) != 2:
                raise ValueError(f"visual_summaries must have shape [L, 2, D], got {tuple(visual.shape)}")
            tensor = visual[:, 0 if field == "visual_mean" else 1, :]
        else:
            raise AssertionError(f"unhandled feature field {field}")
        if tensor.ndim != 2 or int(tensor.shape[0]) != int(num_layers):
            raise ValueError(f"{field} must have shape [{num_layers}, D], got {tuple(tensor.shape)}")
        parts.append(tensor.float())
    return torch.cat(parts, dim=-1)


def transition_counts_for_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim != 2:
        raise ValueError(f"masks must have shape [N, L], got {tuple(masks.shape)}")
    if masks.shape[1] <= 1:
        return torch.zeros(masks.shape[0], dtype=torch.float32, device=masks.device)
    values = masks.to(dtype=torch.bool)
    return (values[:, 1:] != values[:, :-1]).sum(dim=1).to(dtype=torch.float32)


def pair_type_weight_multipliers(
    pair_type_ids: torch.Tensor,
    *,
    correctness_weight: float = 1.0,
    efficiency_weight: float = 1.0,
) -> torch.Tensor:
    """Return per-pair loss multipliers for correctness/efficiency objectives."""
    if correctness_weight < 0.0 or efficiency_weight < 0.0:
        raise ValueError("pair type weights must be non-negative")
    pair_type_ids = torch.as_tensor(pair_type_ids, dtype=torch.long)
    weights = torch.ones(pair_type_ids.shape, dtype=torch.float32, device=pair_type_ids.device)
    weights = torch.where(
        pair_type_ids == PAIR_TYPE_TO_ID["correctness"],
        torch.full_like(weights, float(correctness_weight)),
        weights,
    )
    weights = torch.where(
        pair_type_ids == PAIR_TYPE_TO_ID["efficiency"],
        torch.full_like(weights, float(efficiency_weight)),
        weights,
    )
    return weights


def source_bucket_weight_multipliers(
    source_bucket_ids: torch.Tensor,
    *,
    complete_correct_weight: float = 1.0,
    complete_wrong_weight: float = 1.0,
) -> torch.Tensor:
    """Return per-pair loss multipliers for preservation/rescue source buckets."""
    if complete_correct_weight < 0.0 or complete_wrong_weight < 0.0:
        raise ValueError("source bucket weights must be non-negative")
    source_bucket_ids = torch.as_tensor(source_bucket_ids, dtype=torch.long)
    weights = torch.ones(source_bucket_ids.shape, dtype=torch.float32, device=source_bucket_ids.device)
    weights = torch.where(
        source_bucket_ids == SOURCE_BUCKET_NAMES.index("complete_correct"),
        torch.full_like(weights, float(complete_correct_weight)),
        weights,
    )
    weights = torch.where(
        source_bucket_ids == SOURCE_BUCKET_NAMES.index("complete_wrong"),
        torch.full_like(weights, float(complete_wrong_weight)),
        weights,
    )
    return weights


def route_selection_preservation_utility(
    route_selection_summary: dict[str, Any],
    *,
    complete_wrong_weight: float = 1.0,
    complete_correct_drop_weight: float = 1.0,
    budget_weight: float = 0.0,
    transition_weight: float = 0.0,
    num_layers: int = NUM_LAYERS,
) -> float:
    """Score finite-route selection with rescue, preservation, and sparsity terms.

    complete_wrong contributes rescue rate. complete_correct contributes a drop
    penalty because full context was already correct for that bucket.
    """
    if complete_wrong_weight < 0.0 or complete_correct_drop_weight < 0.0:
        raise ValueError("utility correctness weights must be non-negative")
    if budget_weight < 0.0 or transition_weight < 0.0:
        raise ValueError("utility sparsity weights must be non-negative")
    by_source = route_selection_summary.get("by_source_bucket") or {}
    correct_bucket = by_source.get("complete_correct") or {}
    wrong_bucket = by_source.get("complete_wrong") or {}
    preservation = correct_bucket.get("selected_correct_rate")
    rescue = wrong_bucket.get("selected_correct_rate")
    if preservation is None:
        preservation = route_selection_summary.get("selected_correct_rate", 0.0)
    if rescue is None:
        rescue = route_selection_summary.get("selected_correct_rate", 0.0)
    avg_budget = float(route_selection_summary.get("avg_selected_budget") or 0.0)
    avg_transitions = float(route_selection_summary.get("avg_selected_transitions") or 0.0)
    layer_denom = max(float(num_layers), 1.0)
    transition_denom = max(float(num_layers - 1), 1.0)
    return (
        float(complete_wrong_weight) * float(rescue)
        - float(complete_correct_drop_weight) * (1.0 - float(preservation))
        - float(budget_weight) * (avg_budget / layer_denom)
        - float(transition_weight) * (avg_transitions / transition_denom)
    )


def route_scores_from_layer_scores(
    layer_scores: torch.Tensor,
    masks: torch.Tensor,
    *,
    lambda_on: float = 0.0,
    lambda_transition: float = 0.0,
    normalize_by_on_count: bool = False,
) -> torch.Tensor:
    """Score complete routes from per-layer scores and binary masks."""
    if layer_scores.ndim != 2:
        raise ValueError(f"layer_scores must have shape [N, L], got {tuple(layer_scores.shape)}")
    if masks.shape != layer_scores.shape:
        raise ValueError(f"masks must match layer_scores shape, got {tuple(masks.shape)} vs {tuple(layer_scores.shape)}")
    mask_float = masks.to(device=layer_scores.device, dtype=layer_scores.dtype)
    selected = (layer_scores.float() * mask_float.float()).sum(dim=1)
    on_counts = mask_float.float().sum(dim=1)
    if normalize_by_on_count:
        selected = torch.where(on_counts > 0, selected / on_counts.clamp_min(1.0), selected)
    if lambda_on:
        selected = selected - float(lambda_on) * on_counts.float()
    if lambda_transition:
        selected = selected - float(lambda_transition) * transition_counts_for_masks(masks.to(layer_scores.device))
    return selected


def pairwise_preference_loss_from_scores(
    layer_scores: torch.Tensor,
    chosen_masks: torch.Tensor,
    rejected_masks: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    lambda_on: float = 0.0,
    lambda_transition: float = 0.0,
    normalize_by_on_count: bool = False,
) -> torch.Tensor:
    chosen = route_scores_from_layer_scores(
        layer_scores,
        chosen_masks,
        lambda_on=lambda_on,
        lambda_transition=lambda_transition,
        normalize_by_on_count=normalize_by_on_count,
    )
    rejected = route_scores_from_layer_scores(
        layer_scores,
        rejected_masks,
        lambda_on=lambda_on,
        lambda_transition=lambda_transition,
        normalize_by_on_count=normalize_by_on_count,
    )
    losses = F.softplus(-(chosen - rejected))
    if weights is None:
        return losses.mean()
    weights = torch.as_tensor(weights, dtype=losses.dtype, device=losses.device)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def bernoulli_route_log_probabilities(
    layer_logits: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Return log pi(mask | route-conditioned states) for each batch row."""
    if layer_logits.ndim != 2:
        raise ValueError(f"layer_logits must have shape [N, L], got {tuple(layer_logits.shape)}")
    if masks.shape != layer_logits.shape:
        raise ValueError(f"masks must match layer_logits shape, got {tuple(masks.shape)} vs {tuple(layer_logits.shape)}")
    actions = masks.to(device=layer_logits.device, dtype=torch.bool)
    action_log_probs = torch.where(
        actions,
        F.logsigmoid(layer_logits.float()),
        F.logsigmoid(-layer_logits.float()),
    )
    return action_log_probs.sum(dim=1)


def trajectory_pairwise_preference_loss(
    chosen_layer_logits: torch.Tensor,
    chosen_masks: torch.Tensor,
    rejected_layer_logits: torch.Tensor,
    rejected_masks: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bradley-Terry loss over separately replayed chosen/rejected routes."""
    chosen_log_probs = bernoulli_route_log_probabilities(chosen_layer_logits, chosen_masks)
    rejected_log_probs = bernoulli_route_log_probabilities(rejected_layer_logits, rejected_masks)
    losses = F.softplus(-(chosen_log_probs - rejected_log_probs))
    if weights is None:
        return losses.mean()
    weights = torch.as_tensor(weights, dtype=losses.dtype, device=losses.device)
    if weights.shape != losses.shape:
        raise ValueError(f"weights must have shape {tuple(losses.shape)}, got {tuple(weights.shape)}")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def fixed_reference_route_log_probabilities(
    masks: torch.Tensor,
    *,
    on_probability: float,
) -> torch.Tensor:
    """Score routes under a full-support, content-independent Bernoulli reference."""
    probability = float(on_probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("on_probability must be strictly between 0 and 1")
    actions = masks.to(dtype=torch.bool)
    log_on = torch.log(torch.tensor(probability, dtype=torch.float32, device=actions.device))
    log_off = torch.log(torch.tensor(1.0 - probability, dtype=torch.float32, device=actions.device))
    return torch.where(actions, log_on, log_off).sum(dim=1)


def trajectory_dpo_pair_losses(
    chosen_layer_logits: torch.Tensor,
    chosen_masks: torch.Tensor,
    rejected_layer_logits: torch.Tensor,
    rejected_masks: torch.Tensor,
    *,
    beta: float,
    reference_on_probability: float,
) -> torch.Tensor:
    """Return per-pair DPO losses relative to a conservative stochastic policy."""
    if float(beta) <= 0.0:
        raise ValueError("beta must be positive")
    policy_margin = bernoulli_route_log_probabilities(chosen_layer_logits, chosen_masks) - bernoulli_route_log_probabilities(
        rejected_layer_logits,
        rejected_masks,
    )
    reference_margin = fixed_reference_route_log_probabilities(
        chosen_masks,
        on_probability=reference_on_probability,
    ) - fixed_reference_route_log_probabilities(
        rejected_masks,
        on_probability=reference_on_probability,
    )
    return F.softplus(-float(beta) * (policy_margin - reference_margin))


def pairwise_margins_from_scores(
    layer_scores: torch.Tensor,
    chosen_masks: torch.Tensor,
    rejected_masks: torch.Tensor,
    *,
    lambda_on: float = 0.0,
    lambda_transition: float = 0.0,
    normalize_by_on_count: bool = False,
) -> torch.Tensor:
    chosen = route_scores_from_layer_scores(
        layer_scores,
        chosen_masks,
        lambda_on=lambda_on,
        lambda_transition=lambda_transition,
        normalize_by_on_count=normalize_by_on_count,
    )
    rejected = route_scores_from_layer_scores(
        layer_scores,
        rejected_masks,
        lambda_on=lambda_on,
        lambda_transition=lambda_transition,
        normalize_by_on_count=normalize_by_on_count,
    )
    return chosen - rejected


@dataclass(frozen=True)
class PairTensorStore:
    uid_indices: torch.Tensor
    chosen_masks: torch.Tensor
    rejected_masks: torch.Tensor
    weights: torch.Tensor
    pair_type_ids: torch.Tensor
    benchmark_ids: torch.Tensor
    source_bucket_ids: torch.Tensor
    chosen_budgets: torch.Tensor
    rejected_budgets: torch.Tensor
    pair_ids: list[str]
    uid_names: list[str]
    benchmark_names: tuple[str, ...] = BENCHMARK_NAMES
    source_bucket_names: tuple[str, ...] = SOURCE_BUCKET_NAMES
    pair_type_names: tuple[str, ...] = PAIR_TYPE_NAMES

    @property
    def num_pairs(self) -> int:
        return int(self.uid_indices.numel())

    @property
    def unique_uid_indices(self) -> torch.Tensor:
        return torch.unique(self.uid_indices.cpu()).to(dtype=torch.long)

    def subset(self, pair_indices: torch.Tensor) -> "PairTensorStore":
        pair_indices = torch.as_tensor(pair_indices, dtype=torch.long)
        return PairTensorStore(
            uid_indices=self.uid_indices[pair_indices],
            chosen_masks=self.chosen_masks[pair_indices],
            rejected_masks=self.rejected_masks[pair_indices],
            weights=self.weights[pair_indices],
            pair_type_ids=self.pair_type_ids[pair_indices],
            benchmark_ids=self.benchmark_ids[pair_indices],
            source_bucket_ids=self.source_bucket_ids[pair_indices],
            chosen_budgets=self.chosen_budgets[pair_indices],
            rejected_budgets=self.rejected_budgets[pair_indices],
            pair_ids=[self.pair_ids[int(idx)] for idx in pair_indices.tolist()],
            uid_names=self.uid_names,
            benchmark_names=self.benchmark_names,
            source_bucket_names=self.source_bucket_names,
            pair_type_names=self.pair_type_names,
        )

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        *,
        uid_to_index: dict[str, int],
        num_layers: int = NUM_LAYERS,
    ) -> "PairTensorStore":
        benchmark_to_id = {name: idx for idx, name in enumerate(BENCHMARK_NAMES)}
        source_bucket_to_id = {name: idx for idx, name in enumerate(SOURCE_BUCKET_NAMES)}
        pair_type_to_id = {name: idx for idx, name in enumerate(PAIR_TYPE_NAMES)}
        uid_names = [""] * len(uid_to_index)
        for uid, idx in uid_to_index.items():
            uid_names[int(idx)] = uid

        uid_indices: list[int] = []
        chosen_masks: list[str] = []
        rejected_masks: list[str] = []
        weights: list[float] = []
        pair_type_ids: list[int] = []
        benchmark_ids: list[int] = []
        source_bucket_ids: list[int] = []
        chosen_budgets: list[int] = []
        rejected_budgets: list[int] = []
        pair_ids: list[str] = []
        for row in rows:
            uid = str(row["uid"])
            if uid not in uid_to_index:
                continue
            benchmark = str(row["benchmark"]).lower()
            source_bucket = str(row["source_bucket"])
            pair_type = str(row["pair_type"])
            uid_indices.append(int(uid_to_index[uid]))
            chosen_masks.append(str(row["chosen_mask_key"]))
            rejected_masks.append(str(row["rejected_mask_key"]))
            weights.append(float(row.get("recommended_weight", 1.0)))
            pair_type_ids.append(pair_type_to_id[pair_type])
            benchmark_ids.append(benchmark_to_id[benchmark])
            source_bucket_ids.append(source_bucket_to_id[source_bucket])
            chosen_budgets.append(int(row["chosen_budget"]))
            rejected_budgets.append(int(row["rejected_budget"]))
            pair_ids.append(str(row["pair_id"]))

        return cls(
            uid_indices=torch.tensor(uid_indices, dtype=torch.long),
            chosen_masks=mask_keys_to_bool_tensor(chosen_masks, num_layers=num_layers),
            rejected_masks=mask_keys_to_bool_tensor(rejected_masks, num_layers=num_layers),
            weights=torch.tensor(weights, dtype=torch.float32),
            pair_type_ids=torch.tensor(pair_type_ids, dtype=torch.long),
            benchmark_ids=torch.tensor(benchmark_ids, dtype=torch.long),
            source_bucket_ids=torch.tensor(source_bucket_ids, dtype=torch.long),
            chosen_budgets=torch.tensor(chosen_budgets, dtype=torch.long),
            rejected_budgets=torch.tensor(rejected_budgets, dtype=torch.long),
            pair_ids=pair_ids,
            uid_names=uid_names,
        )


class UIDBalancedPairSampler:
    """Sample UIDs approximately uniformly, then one preference pair per UID."""

    def __init__(self, store: PairTensorStore, *, seed: int = 0) -> None:
        if store.num_pairs <= 0:
            raise ValueError("cannot sample from an empty PairTensorStore")
        self.store = store
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        grouped: dict[int, list[int]] = defaultdict(list)
        for pair_idx, uid_idx in enumerate(store.uid_indices.tolist()):
            grouped[int(uid_idx)].append(int(pair_idx))
        self.uid_indices = sorted(grouped)
        self.pair_indices_by_uid = {
            uid_idx: torch.tensor(indices, dtype=torch.long) for uid_idx, indices in grouped.items()
        }

    def sample_pair_indices(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        uid_positions = torch.randint(
            low=0,
            high=len(self.uid_indices),
            size=(int(batch_size),),
            generator=self.generator,
        )
        sampled: list[int] = []
        for position in uid_positions.tolist():
            uid_idx = self.uid_indices[int(position)]
            candidates = self.pair_indices_by_uid[uid_idx]
            local_idx = int(torch.randint(0, int(candidates.numel()), (1,), generator=self.generator).item())
            sampled.append(int(candidates[local_idx].item()))
        return torch.tensor(sampled, dtype=torch.long)


class GlobalPriorRouter(torch.nn.Module):
    def __init__(self, num_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        self.layer_scores = torch.nn.Parameter(torch.zeros(int(num_layers), dtype=torch.float32))

    def forward(
        self,
        features: torch.Tensor | None,
        benchmark_ids: torch.Tensor,
        uid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(benchmark_ids.numel())
        return self.layer_scores.unsqueeze(0).expand(batch_size, -1)


class BenchmarkPriorRouter(torch.nn.Module):
    def __init__(self, num_benchmarks: int = len(BENCHMARK_NAMES), num_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        self.layer_scores = torch.nn.Embedding(int(num_benchmarks), int(num_layers))
        torch.nn.init.zeros_(self.layer_scores.weight)

    def forward(
        self,
        features: torch.Tensor | None,
        benchmark_ids: torch.Tensor,
        uid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.layer_scores(benchmark_ids.long())


class LayerFeatureLinearRouter(torch.nn.Module):
    def __init__(self, input_dim: int, num_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        self.scorer = torch.nn.Linear(int(input_dim), 1)
        self.layer_bias = torch.nn.Parameter(torch.zeros(int(num_layers), dtype=torch.float32))

    def forward(
        self,
        features: torch.Tensor | None,
        benchmark_ids: torch.Tensor,
        uid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is None:
            raise ValueError("LayerFeatureLinearRouter requires features")
        return self.scorer(features.float()).squeeze(-1) + self.layer_bias.unsqueeze(0)


class LayerFeatureLinearBenchmarkRouter(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_benchmarks: int = len(BENCHMARK_NAMES),
        num_layers: int = NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.scorer = torch.nn.Linear(int(input_dim), 1)
        self.layer_scores = torch.nn.Embedding(int(num_benchmarks), int(num_layers))
        torch.nn.init.zeros_(self.layer_scores.weight)

    def forward(
        self,
        features: torch.Tensor | None,
        benchmark_ids: torch.Tensor,
        uid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is None:
            raise ValueError("LayerFeatureLinearBenchmarkRouter requires features")
        return self.scorer(features.float()).squeeze(-1) + self.layer_scores(benchmark_ids.long())


class LayerFeatureMLPRouter(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1, num_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        self.scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(int(input_dim)),
            torch.nn.Linear(int(input_dim), int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), 1),
        )
        self.layer_bias = torch.nn.Parameter(torch.zeros(int(num_layers), dtype=torch.float32))

    def forward(
        self,
        features: torch.Tensor | None,
        benchmark_ids: torch.Tensor,
        uid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is None:
            raise ValueError("LayerFeatureMLPRouter requires features")
        return self.scorer(features.float()).squeeze(-1) + self.layer_bias.unsqueeze(0)


class LayerFeatureMLPBenchmarkRouter(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_benchmarks: int = len(BENCHMARK_NAMES),
        num_layers: int = NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(int(input_dim)),
            torch.nn.Linear(int(input_dim), int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), 1),
        )
        self.layer_scores = torch.nn.Embedding(int(num_benchmarks), int(num_layers))
        torch.nn.init.zeros_(self.layer_scores.weight)

    def forward(
        self,
        features: torch.Tensor | None,
        benchmark_ids: torch.Tensor,
        uid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is None:
            raise ValueError("LayerFeatureMLPBenchmarkRouter requires features")
        return self.scorer(features.float()).squeeze(-1) + self.layer_scores(benchmark_ids.long())


def pairwise_accuracy_summary(
    *,
    margins: torch.Tensor,
    store: PairTensorStore,
) -> dict[str, Any]:
    if int(margins.numel()) != store.num_pairs:
        raise ValueError(f"margin count {margins.numel()} != pair count {store.num_pairs}")
    margins_cpu = margins.detach().float().cpu()
    correct = margins_cpu > 0.0

    def summarize(mask: torch.Tensor) -> dict[str, Any]:
        if int(mask.sum().item()) == 0:
            return {"pairs": 0, "accuracy": None, "uid_balanced_accuracy": None, "mean_margin": None}
        selected_correct = correct[mask]
        selected_margins = margins_cpu[mask]
        selected_uid_indices = store.uid_indices[mask]
        selected_uid_stats: dict[int, list[float]] = defaultdict(list)
        for uid_idx, is_correct in zip(selected_uid_indices.tolist(), selected_correct.tolist(), strict=False):
            selected_uid_stats[int(uid_idx)].append(float(is_correct))
        selected_uid_accuracy = [
            sum(values) / float(len(values)) for values in selected_uid_stats.values() if values
        ]
        return {
            "pairs": int(mask.sum().item()),
            "accuracy": float(selected_correct.float().mean().item()),
            "uid_balanced_accuracy": (
                sum(selected_uid_accuracy) / float(len(selected_uid_accuracy)) if selected_uid_accuracy else None
            ),
            "mean_margin": float(selected_margins.mean().item()),
        }

    by_pair_type = {
        name: summarize(store.pair_type_ids == idx) for idx, name in enumerate(store.pair_type_names)
    }
    by_benchmark = {
        name: summarize(store.benchmark_ids == idx) for idx, name in enumerate(store.benchmark_names)
    }
    by_source_bucket = {
        name: summarize(store.source_bucket_ids == idx) for idx, name in enumerate(store.source_bucket_names)
    }

    uid_stats: dict[int, list[float]] = defaultdict(list)
    for uid_idx, is_correct in zip(store.uid_indices.tolist(), correct.tolist(), strict=False):
        uid_stats[int(uid_idx)].append(float(is_correct))
    uid_acc = [sum(values) / float(len(values)) for values in uid_stats.values() if values]

    return {
        "pairs": store.num_pairs,
        "accuracy": float(correct.float().mean().item()) if store.num_pairs else None,
        "mean_margin": float(margins_cpu.mean().item()) if store.num_pairs else None,
        "uid_count": len(uid_acc),
        "uid_balanced_accuracy": sum(uid_acc) / float(len(uid_acc)) if uid_acc else None,
        "by_pair_type": by_pair_type,
        "by_benchmark": by_benchmark,
        "by_source_bucket": by_source_bucket,
    }


def make_router(model_name: str, *, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> torch.nn.Module:
    if model_name == "global_prior":
        return GlobalPriorRouter()
    if model_name == "benchmark_prior":
        return BenchmarkPriorRouter()
    if model_name in {"linear_last", "linear_feature"}:
        return LayerFeatureLinearRouter(input_dim=input_dim)
    if model_name in {"mlp_last", "mlp_feature"}:
        return LayerFeatureMLPRouter(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_name in {"linear_last_benchmark", "linear_feature_benchmark"}:
        return LayerFeatureLinearBenchmarkRouter(input_dim=input_dim)
    if model_name in {"mlp_last_benchmark", "mlp_feature_benchmark"}:
        return LayerFeatureMLPBenchmarkRouter(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"unknown probe model {model_name!r}")


def route_budget(mask_key: str) -> int:
    return str(mask_key).count("1")


def route_transition_count(mask_key: str) -> int:
    return transition_count(mask_key)
