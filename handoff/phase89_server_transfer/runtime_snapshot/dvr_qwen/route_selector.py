"""Cache-only complete-route selector utilities for Phase 5B."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


NUM_LAYERS = 28
BENCHMARKS = ("gqa", "textvqa", "chartqa", "docvqa")

DISALLOWED_FEATURE_NAMES = (
    "candidate_score",
    "trace_exact_positive",
    "nearest_trace_positive_hamming",
    "nearest_positive_trace_hamming",
    "nearest_positive_trace_label",
    "live_success",
    "live_target_positive",
    "task_score",
    "task_correctness",
    "answer_correctness",
    "ground_truth_answer",
    "gt_answer",
    "answer",
    "candidate_answer",
    "full_answer",
    "oracle_answer",
    "oracle_selected_candidate_id",
    "posthoc_route_execution_metric",
    "full_score",
    "oracle_score",
    "target_score",
    "live_full_preserving",
    "delta_score",
    "delta_q",
    "fix",
    "improve",
    "preserve",
    "safe_switch",
    "cost_only_preserve",
    "full_qwen_wrong",
    "full_wrong_improvement",
    "full_wrong_fix",
    "full_correct_preserved",
    "full_correct_regression",
    "candidate_reaches_target",
    "v3_accuracy_utility",
    "v3_default_utility",
    "v3_1_accuracy_utility",
    "v3_1_balanced_utility",
)


def assert_no_feature_leakage(feature_names: list[str]) -> None:
    leaked: list[tuple[str, str]] = []
    for name in feature_names:
        lower_name = str(name).lower()
        for bad in DISALLOWED_FEATURE_NAMES:
            if bad in lower_name:
                leaked.append((name, bad))
    if leaked:
        details = ", ".join(f"{name} contains {bad}" for name, bad in leaked[:8])
        raise ValueError(f"selector feature schema includes label-derived fields: {details}")


def normalized_layers(layers_one_based: list[int] | tuple[int, ...], *, num_layers: int = NUM_LAYERS) -> list[int]:
    layers = sorted({int(layer) for layer in layers_one_based})
    for layer in layers:
        if layer < 1 or layer > int(num_layers):
            raise ValueError(f"layer {layer} is out of range for {num_layers} layers")
    return layers


def route_mask_from_layers(layers_one_based: list[int] | tuple[int, ...], *, num_layers: int = NUM_LAYERS) -> torch.Tensor:
    mask = torch.zeros(int(num_layers), dtype=torch.float32)
    for layer in normalized_layers(layers_one_based, num_layers=num_layers):
        mask[layer - 1] = 1.0
    return mask


def transition_count(mask: torch.Tensor) -> int:
    route = mask.float().view(-1) > 0.5
    if int(route.numel()) <= 1:
        return 0
    return int((route[1:] != route[:-1]).sum().item())


def longest_false_gap(mask: torch.Tensor) -> int:
    best = 0
    current = 0
    for value in (mask.float().view(-1) <= 0.5).tolist():
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def visual_on_segments(mask: torch.Tensor) -> list[int]:
    segments: list[int] = []
    current = 0
    for value in (mask.float().view(-1) > 0.5).tolist():
        if value:
            current += 1
        elif current:
            segments.append(current)
            current = 0
    if current:
        segments.append(current)
    return segments


def early_mid_late_counts(mask: torch.Tensor) -> tuple[int, int, int]:
    num_layers = int(mask.numel())
    first_cut = num_layers // 3
    second_cut = (2 * num_layers) // 3 + (1 if num_layers % 3 else 0)
    values = mask.float().view(-1)
    early = int(values[:first_cut].sum().item())
    middle = int(values[first_cut:second_cut].sum().item())
    late = int(values[second_cut:].sum().item())
    return early, middle, late


def candidate_on(row: dict[str, Any]) -> int:
    return int(row.get("candidate_num_visual_on_layers", len(row.get("layers_one_based", []))))


def candidate_budget(row: dict[str, Any]) -> int:
    return int(row.get("budget_count", 0))


def candidate_rank(row: dict[str, Any]) -> int:
    value = row.get("decoder_rank")
    return int(value) if value is not None else int(row.get("candidate_index", 999999))


def candidate_decoder_score(row: dict[str, Any]) -> float:
    value = row.get("decoder_score")
    return float(value) if value is not None else 0.0


def group_rows_by_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)
    for sample_id in list(grouped):
        grouped[sample_id] = sorted(
            grouped[sample_id],
            key=lambda item: (candidate_rank(item), int(item.get("candidate_index", 0))),
        )
    return dict(grouped)


def base_row_for_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select a decoder-top1 row from an empty group")
    return min(rows, key=lambda item: (candidate_rank(item), int(item.get("candidate_index", 0))))


def _rank_scale_by_id(rows: list[dict[str, Any]]) -> dict[str, float]:
    scale: dict[str, int] = defaultdict(int)
    for row in rows:
        sample_id = str(row["id"])
        scale[sample_id] = max(scale[sample_id], candidate_rank(row), int(row.get("candidate_index", 0)))
    return {sample_id: max(float(value), 1.0) for sample_id, value in scale.items()}


def _benchmark_one_hot(benchmark: str) -> tuple[list[float], list[str]]:
    values = [0.0 for _ in BENCHMARKS]
    if benchmark in BENCHMARKS:
        values[BENCHMARKS.index(benchmark)] = 1.0
    return values, [f"context_benchmark_{name}" for name in BENCHMARKS]


def _route_feature_values(
    *,
    row: dict[str, Any],
    layer_scores: torch.Tensor,
    rank_scale: float,
    sample_scalars: torch.Tensor | None,
    num_layers: int,
) -> tuple[torch.Tensor, list[str]]:
    layers = normalized_layers([int(layer) for layer in row.get("layers_one_based", [])], num_layers=num_layers)
    route = route_mask_from_layers(layers, num_layers=num_layers)
    layer_scores = layer_scores.float().view(-1)
    if int(layer_scores.numel()) != int(num_layers):
        raise ValueError(f"layer score vector must have {num_layers} values, got {layer_scores.numel()}")

    on_count = float(candidate_on(row))
    budget = float(candidate_budget(row))
    transitions = float(transition_count(route))
    first_on = float(layers[0]) if layers else 0.0
    last_on = float(layers[-1]) if layers else 0.0
    early, middle, late = early_mid_late_counts(route)
    segments = visual_on_segments(route)
    selected_scores = layer_scores[route.bool()]
    unselected_scores = layer_scores[~route.bool()]
    log_on = F.logsigmoid(layer_scores)
    log_off = F.logsigmoid(-layer_scores)
    route_logprob = torch.where(route.bool(), log_on, log_off)
    benchmark_values, benchmark_names = _benchmark_one_hot(str(row.get("benchmark", "unknown")))

    scalar_values = [
        on_count / float(num_layers),
        budget / float(num_layers),
        (on_count - budget) / float(num_layers),
        abs(on_count - budget) / float(num_layers),
        transitions / max(float(num_layers - 1), 1.0),
        first_on / float(num_layers),
        last_on / float(num_layers),
        float(early) / float(num_layers),
        float(middle) / float(num_layers),
        float(late) / float(num_layers),
        float(longest_false_gap(route)) / float(num_layers),
        float(len(segments)) / float(num_layers),
        float(max(segments) if segments else 0) / float(num_layers),
        float(sum(segments) / len(segments) if segments else 0.0) / float(num_layers),
        float(route[0].item()) if num_layers >= 1 else 0.0,
        float(route[1].item()) if num_layers >= 2 else 0.0,
        float(candidate_rank(row)) / rank_scale,
        float(int(row.get("candidate_index", candidate_rank(row)))) / rank_scale,
        candidate_decoder_score(row) / float(num_layers),
        float(route_logprob.sum().item()) / float(num_layers),
        float(selected_scores.sum().item()) / float(num_layers) if selected_scores.numel() else 0.0,
        float(unselected_scores.sum().item()) / float(num_layers) if unselected_scores.numel() else 0.0,
        float(selected_scores.mean().item()) if selected_scores.numel() else 0.0,
        float(unselected_scores.mean().item()) if unselected_scores.numel() else 0.0,
    ]
    scalar_names = [
        "route_on_count_norm",
        "route_budget_count_norm",
        "route_budget_slack_norm",
        "route_abs_budget_slack_norm",
        "route_transition_count_norm",
        "route_first_on_norm",
        "route_last_on_norm",
        "route_early_on_norm",
        "route_middle_on_norm",
        "route_late_on_norm",
        "route_longest_text_gap_norm",
        "route_segment_count_norm",
        "route_max_segment_len_norm",
        "route_mean_segment_len_norm",
        "route_anchor_layer_1",
        "route_anchor_layer_2",
        "proposal_decoder_rank_norm",
        "proposal_candidate_index_norm",
        "proposal_decoder_logprob_scaled",
        "proposal_route_logprob_mean",
        "proposal_selected_layer_score_sum_norm",
        "proposal_unselected_layer_score_sum_norm",
        "proposal_selected_layer_score_mean",
        "proposal_unselected_layer_score_mean",
    ]

    parts = [
        torch.tensor(benchmark_values, dtype=torch.float32),
        route.float(),
        layer_scores.float(),
        route.float() * layer_scores.float(),
        (1.0 - route.float()) * layer_scores.float(),
        route_logprob.float(),
        torch.tensor(scalar_values, dtype=torch.float32),
    ]
    names = [
        *benchmark_names,
        *[f"route_mask_layer_{idx + 1}" for idx in range(num_layers)],
        *[f"proposal_layer_score_{idx + 1}" for idx in range(num_layers)],
        *[f"proposal_selected_layer_score_{idx + 1}" for idx in range(num_layers)],
        *[f"proposal_unselected_layer_score_{idx + 1}" for idx in range(num_layers)],
        *[f"proposal_layer_logprob_contribution_{idx + 1}" for idx in range(num_layers)],
        *scalar_names,
    ]
    if sample_scalars is not None:
        scalars = sample_scalars.float().view(-1)
        parts.append(scalars)
        names.extend([f"context_scalar_{idx}" for idx in range(int(scalars.numel()))])
    return torch.cat(parts), names


def build_route_selector_examples(
    rows: list[dict[str, Any]],
    layer_scores_by_id: dict[str, torch.Tensor],
    *,
    sample_scalars_by_id: dict[str, torch.Tensor] | None = None,
    num_layers: int = NUM_LAYERS,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[str]]:
    if not rows:
        raise ValueError("no route candidate rows were provided")
    grouped = group_rows_by_id(rows)
    base_by_id = {sample_id: base_row_for_group(group) for sample_id, group in grouped.items()}
    rank_scale = _rank_scale_by_id(rows)
    features: list[torch.Tensor] = []
    labels: list[float] = []
    metadata: list[dict[str, Any]] = []
    feature_names: list[str] | None = None

    for row in rows:
        sample_id = str(row["id"])
        if sample_id not in layer_scores_by_id:
            raise ValueError(f"missing layer scores for {sample_id}")
        sample_scalars = None
        if sample_scalars_by_id is not None:
            if sample_id not in sample_scalars_by_id:
                raise ValueError(f"missing sample scalar features for {sample_id}")
            sample_scalars = sample_scalars_by_id[sample_id]
        feature, names = _route_feature_values(
            row=row,
            layer_scores=layer_scores_by_id[sample_id],
            rank_scale=rank_scale[sample_id],
            sample_scalars=sample_scalars,
            num_layers=num_layers,
        )
        if feature_names is None:
            feature_names = names
            assert_no_feature_leakage(feature_names)
        elif feature_names != names:
            raise ValueError("inconsistent route selector feature schema")

        base = base_by_id[sample_id]
        live_success = bool(row.get("live_target_positive", False))
        base_live_success = bool(base.get("live_target_positive", False))
        layers = normalized_layers([int(layer) for layer in row.get("layers_one_based", [])], num_layers=num_layers)
        route = route_mask_from_layers(layers, num_layers=num_layers)
        is_default = row is base or (
            candidate_rank(row) == candidate_rank(base)
            and int(row.get("candidate_index", 0)) == int(base.get("candidate_index", 0))
        )
        features.append(feature)
        labels.append(1.0 if live_success else 0.0)
        metadata.append(
            {
                "id": sample_id,
                "benchmark": row.get("benchmark"),
                "label": "positive" if live_success else "negative",
                "live_success": live_success,
                "base_live_success": base_live_success,
                "is_default": bool(is_default),
                "is_fix": bool(live_success and not base_live_success),
                "is_regression": bool(base_live_success and not live_success),
                "layers_one_based": layers,
                "on_count": candidate_on(row),
                "candidate_num_visual_on_layers": candidate_on(row),
                "base_on_count": candidate_on(base),
                "budget_count": candidate_budget(row),
                "transition_count": transition_count(route),
                "decoder_rank": candidate_rank(row),
                "candidate_index": int(row.get("candidate_index", 0)),
                "decoder_score": candidate_decoder_score(row),
                "candidate_score": float(row.get("candidate_score", 0.0)),
                "sources": list(row.get("sources", [])),
            }
        )

    if feature_names is None:
        raise ValueError("no route selector feature names were built")
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32), metadata, feature_names


def candidate_utilities(
    labels: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    lambda_on: float,
    lambda_regression: float,
    lambda_transition: float,
    lambda_negative_switch: float = 0.0,
    lambda_below_budget_negative: float = 0.0,
) -> torch.Tensor:
    if int(labels.numel()) != len(metadata):
        raise ValueError(f"labels length {labels.numel()} != metadata length {len(metadata)}")
    on_counts = torch.tensor([float(item.get("on_count", 0.0)) for item in metadata], dtype=torch.float32)
    regressions = torch.tensor([float(bool(item.get("is_regression", False))) for item in metadata], dtype=torch.float32)
    transitions = torch.tensor([float(item.get("transition_count", 0.0)) for item in metadata], dtype=torch.float32)
    negative_switches = torch.tensor(
        [
            float((not bool(item.get("live_success", False))) and (not bool(item.get("is_default", False))))
            for item in metadata
        ],
        dtype=torch.float32,
    )
    below_budget_negative_switches = torch.tensor(
        [
            float(
                (not bool(item.get("live_success", False)))
                and (not bool(item.get("is_default", False)))
                and float(item.get("on_count", 0.0)) < float(item.get("budget_count", 0.0))
            )
            for item in metadata
        ],
        dtype=torch.float32,
    )
    return (
        labels.float()
        - float(lambda_on) * on_counts
        - float(lambda_regression) * regressions
        - float(lambda_transition) * transitions
        - float(lambda_negative_switch) * negative_switches
        - float(lambda_below_budget_negative) * below_budget_negative_switches
    )


def group_indices_from_metadata(metadata: list[dict[str, Any]]) -> list[torch.Tensor]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(metadata):
        grouped[str(item["id"])].append(idx)
    return [torch.tensor(indices, dtype=torch.long) for _, indices in sorted(grouped.items())]


class RouteVerifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).squeeze(-1)


def listwise_utility_loss(scores: torch.Tensor, utilities: torch.Tensor, groups: list[torch.Tensor]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for group in groups:
        group_scores = scores[group]
        group_utilities = utilities[group]
        target = int(torch.argmax(group_utilities).item())
        losses.append(F.cross_entropy(group_scores.view(1, -1), torch.tensor([target], device=scores.device)))
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def _pair_weight(better: dict[str, Any], worse: dict[str, Any]) -> float:
    if bool(worse.get("is_regression", False)):
        return 4.0
    if bool(better.get("is_fix", False)):
        return 2.0
    same_success = bool(better.get("live_success", False)) == bool(worse.get("live_success", False))
    if same_success and float(better.get("on_count", 0.0)) < float(worse.get("on_count", 0.0)):
        return 1.0
    return 0.5


def _hard_negative_multiplier(
    item: dict[str, Any],
    *,
    hard_negative_weight: float,
    below_budget_negative_weight: float,
) -> float:
    if bool(item.get("live_success", False)) or bool(item.get("is_default", False)):
        return 1.0
    weight = float(hard_negative_weight)
    if float(item.get("on_count", 0.0)) < float(item.get("budget_count", 0.0)):
        weight *= float(below_budget_negative_weight)
    return weight


def pair_weight_matrices(
    metadata: list[dict[str, Any]],
    groups: list[torch.Tensor],
    *,
    hard_negative_weight: float = 1.0,
    below_budget_negative_weight: float = 1.0,
) -> list[torch.Tensor]:
    matrices: list[torch.Tensor] = []
    for group in groups:
        group_indices = group.tolist()
        weights = torch.empty((len(group_indices), len(group_indices)), dtype=torch.float32)
        for better_idx, global_better in enumerate(group_indices):
            better_meta = metadata[int(global_better)]
            for worse_idx, global_worse in enumerate(group_indices):
                worse_meta = metadata[int(global_worse)]
                weights[better_idx, worse_idx] = _pair_weight(better_meta, worse_meta) * _hard_negative_multiplier(
                    worse_meta,
                    hard_negative_weight=float(hard_negative_weight),
                    below_budget_negative_weight=float(below_budget_negative_weight),
                )
        matrices.append(weights)
    return matrices


def pairwise_utility_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    metadata: list[dict[str, Any]],
    groups: list[torch.Tensor],
    *,
    pair_weights: list[torch.Tensor] | None = None,
    min_delta: float = 1e-6,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    if pair_weights is not None and len(pair_weights) != len(groups):
        raise ValueError("pair_weights must have the same length as groups")
    for group_idx, group in enumerate(groups):
        group_scores = scores[group]
        group_utilities = utilities[group]
        diff_util = group_utilities.view(-1, 1) - group_utilities.view(1, -1)
        mask = diff_util > float(min_delta)
        if not bool(mask.any().item()):
            continue
        score_diff = group_scores.view(-1, 1) - group_scores.view(1, -1)
        if pair_weights is None:
            weights = pair_weight_matrices(metadata, [group])[0].to(device=scores.device)
        else:
            weights = pair_weights[group_idx].to(device=scores.device)
        losses.append((F.softplus(-score_diff[mask]) * weights[mask]).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def combined_utility_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    metadata: list[dict[str, Any]],
    groups: list[torch.Tensor],
    *,
    pair_weights: list[torch.Tensor] | None = None,
    alpha_listwise: float = 0.5,
) -> torch.Tensor:
    return pairwise_utility_loss(
        scores,
        utilities,
        metadata,
        groups,
        pair_weights=pair_weights,
    ) + float(alpha_listwise) * listwise_utility_loss(
        scores,
        utilities,
        groups,
    )


def standardize_features(
    train_features: torch.Tensor,
    val_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_features.float().mean(dim=0)
    scale = train_features.float().std(dim=0).clamp_min(1e-6)
    return (train_features - mean) / scale, (val_features - mean) / scale, mean, scale


def _checkpoint_policy(checkpoint: dict[str, Any]) -> dict[str, Any]:
    summary = dict(checkpoint.get("summary", {}))
    selected_model = dict(summary.get("selected_model", {}))
    policy = selected_model.get("train_selected_policy") or summary.get("selected_policy")
    if not isinstance(policy, dict):
        raise ValueError("route selector checkpoint is missing the selected inference policy")
    return policy


def _checkpoint_hidden_dim(checkpoint: dict[str, Any]) -> int:
    summary = dict(checkpoint.get("summary", {}))
    training_config = dict(summary.get("training_config", {}))
    if "hidden_dim" in training_config:
        return int(training_config["hidden_dim"])
    state_dict = checkpoint.get("state_dict", {})
    first_weight = state_dict.get("net.0.weight")
    if first_weight is None:
        raise ValueError("route selector checkpoint is missing net.0.weight")
    return int(first_weight.shape[0])


def select_with_frozen_route_verifier(
    rows: list[dict[str, Any]],
    layer_scores_by_id: dict[str, torch.Tensor],
    checkpoint: dict[str, Any],
    *,
    sample_scalars_by_id: dict[str, torch.Tensor] | None = None,
    num_layers: int = NUM_LAYERS,
    policy_name: str = "route_verifier_frozen_replay",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features, _, metadata, feature_names = build_route_selector_examples(
        rows,
        layer_scores_by_id,
        sample_scalars_by_id=sample_scalars_by_id,
        num_layers=num_layers,
    )
    checkpoint_feature_names = [str(name) for name in checkpoint.get("feature_names", [])]
    if feature_names != checkpoint_feature_names:
        raise ValueError("replay feature schema differs from frozen checkpoint schema")
    assert_no_feature_leakage(feature_names)

    feature_mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32)
    feature_scale = torch.as_tensor(checkpoint["feature_scale"], dtype=torch.float32)
    if tuple(feature_mean.shape) != (int(features.shape[1]),) or tuple(feature_scale.shape) != (int(features.shape[1]),):
        raise ValueError("checkpoint feature_mean/feature_scale shape does not match replay features")
    features = (features.float() - feature_mean) / feature_scale.clamp_min(1e-6)

    model = RouteVerifier(
        int(features.shape[1]),
        hidden_dim=_checkpoint_hidden_dim(checkpoint),
        dropout=float(dict(checkpoint.get("summary", {})).get("training_config", {}).get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        scores = model(features).cpu()

    policy = _checkpoint_policy(checkpoint)
    selected_rows = select_with_policy(
        scores,
        metadata,
        lambda_on=float(policy["infer_lambda_on"]),
        lambda_transition=float(policy["infer_lambda_transition"]),
        margin=float(policy["margin"]),
        success_threshold=policy.get("success_threshold"),
        min_budget_slack=policy.get("min_budget_slack"),
        min_on_delta_vs_default=policy.get("min_on_delta_vs_default"),
        max_candidate_rank=policy.get("max_candidate_rank"),
        policy_name=policy_name,
    )
    return selected_rows, evaluate_selected_rows(selected_rows)


def _train_route_verifier_on_utilities(
    train_features: torch.Tensor,
    train_utilities: torch.Tensor,
    train_metadata: list[dict[str, Any]],
    *,
    hidden_dim: int,
    steps: int,
    groups_per_step: int,
    lr: float,
    weight_decay: float,
    seed: int,
    alpha_listwise: float = 0.5,
    dropout: float = 0.1,
    hard_negative_weight: float = 1.0,
    below_budget_negative_weight: float = 1.0,
    objective: str,
    extra_summary: dict[str, Any],
) -> tuple[RouteVerifier, dict[str, Any]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if groups_per_step <= 0:
        raise ValueError("groups_per_step must be positive")
    utilities = train_utilities.float().view(-1)
    if int(train_features.shape[0]) != int(utilities.numel()):
        raise ValueError(f"features rows {train_features.shape[0]} != utilities length {utilities.numel()}")
    if int(utilities.numel()) != len(train_metadata):
        raise ValueError(f"utilities length {utilities.numel()} != metadata length {len(train_metadata)}")
    torch.manual_seed(int(seed))
    model = RouteVerifier(int(train_features.shape[1]), hidden_dim=int(hidden_dim), dropout=float(dropout))
    groups = group_indices_from_metadata(train_metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    all_pair_weights = pair_weight_matrices(
        train_metadata,
        groups,
        hard_negative_weight=float(hard_negative_weight),
        below_budget_negative_weight=float(below_budget_negative_weight),
    )
    model.train()
    with torch.no_grad():
        initial_loss = float(
            combined_utility_loss(
                model(train_features),
                utilities,
                train_metadata,
                groups,
                pair_weights=all_pair_weights,
                alpha_listwise=float(alpha_listwise),
            ).item()
        )
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(int(steps)):
        if groups_per_step >= len(groups):
            selected_groups = groups
            selected_pair_weights = all_pair_weights
        else:
            group_ids = torch.randint(0, len(groups), (int(groups_per_step),), generator=generator)
            selected_indices = [int(idx.item()) for idx in group_ids]
            selected_groups = [groups[idx] for idx in selected_indices]
            selected_pair_weights = [all_pair_weights[idx] for idx in selected_indices]
        optimizer.zero_grad(set_to_none=True)
        loss = combined_utility_loss(
            model(train_features),
            utilities,
            train_metadata,
            selected_groups,
            pair_weights=selected_pair_weights,
            alpha_listwise=float(alpha_listwise),
        )
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = float(
            combined_utility_loss(
                model(train_features),
                utilities,
                train_metadata,
                groups,
                pair_weights=all_pair_weights,
                alpha_listwise=float(alpha_listwise),
            ).item()
        )
    return model, {
        "objective": str(objective),
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "num_train_groups": len(groups),
        "num_train_examples": int(utilities.numel()),
        "hidden_dim": int(hidden_dim),
        "steps": int(steps),
        "groups_per_step": int(groups_per_step),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "dropout": float(dropout),
        "alpha_listwise": float(alpha_listwise),
        "hard_negative_weight": float(hard_negative_weight),
        "below_budget_negative_weight": float(below_budget_negative_weight),
        **extra_summary,
    }


def train_route_verifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_metadata: list[dict[str, Any]],
    *,
    hidden_dim: int,
    steps: int,
    groups_per_step: int,
    lr: float,
    weight_decay: float,
    seed: int,
    lambda_on: float,
    lambda_regression: float,
    lambda_transition: float,
    lambda_negative_switch: float = 0.0,
    lambda_below_budget_negative: float = 0.0,
    alpha_listwise: float = 0.5,
    dropout: float = 0.1,
    hard_negative_weight: float = 1.0,
    below_budget_negative_weight: float = 1.0,
) -> tuple[RouteVerifier, dict[str, Any]]:
    utilities = candidate_utilities(
        train_labels,
        train_metadata,
        lambda_on=float(lambda_on),
        lambda_regression=float(lambda_regression),
        lambda_transition=float(lambda_transition),
        lambda_negative_switch=float(lambda_negative_switch),
        lambda_below_budget_negative=float(lambda_below_budget_negative),
    )
    return _train_route_verifier_on_utilities(
        train_features,
        utilities,
        train_metadata,
        hidden_dim=hidden_dim,
        steps=steps,
        groups_per_step=groups_per_step,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        alpha_listwise=alpha_listwise,
        dropout=dropout,
        hard_negative_weight=hard_negative_weight,
        below_budget_negative_weight=below_budget_negative_weight,
        objective="utility_pairwise_plus_listwise",
        extra_summary={
            "num_positive_train_examples": int(train_labels.float().sum().item()),
            "lambda_on": float(lambda_on),
            "lambda_regression": float(lambda_regression),
            "lambda_transition": float(lambda_transition),
            "lambda_negative_switch": float(lambda_negative_switch),
            "lambda_below_budget_negative": float(lambda_below_budget_negative),
        },
    )


def train_route_verifier_from_utilities(
    train_features: torch.Tensor,
    train_utilities: torch.Tensor,
    train_metadata: list[dict[str, Any]],
    *,
    hidden_dim: int,
    steps: int,
    groups_per_step: int,
    lr: float,
    weight_decay: float,
    seed: int,
    alpha_listwise: float = 0.5,
    dropout: float = 0.1,
    hard_negative_weight: float = 1.0,
    below_budget_negative_weight: float = 1.0,
) -> tuple[RouteVerifier, dict[str, Any]]:
    utilities = train_utilities.float().view(-1)
    return _train_route_verifier_on_utilities(
        train_features,
        utilities,
        train_metadata,
        hidden_dim=hidden_dim,
        steps=steps,
        groups_per_step=groups_per_step,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        alpha_listwise=alpha_listwise,
        dropout=dropout,
        hard_negative_weight=hard_negative_weight,
        below_budget_negative_weight=below_budget_negative_weight,
        objective="precomputed_utility_pairwise_plus_listwise",
        extra_summary={
            "num_positive_train_examples": int((utilities > 0.0).sum().item()),
            "train_utility_mean": float(utilities.mean().item()) if int(utilities.numel()) else 0.0,
            "train_utility_min": float(utilities.min().item()) if int(utilities.numel()) else 0.0,
            "train_utility_max": float(utilities.max().item()) if int(utilities.numel()) else 0.0,
        },
    )


def _default_metadata(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    default = [item for item in candidates if bool(item.get("is_default", False))]
    if default:
        return min(default, key=lambda item: (int(item.get("decoder_rank", 0)), int(item.get("candidate_index", 0))))
    return min(candidates, key=lambda item: (int(item.get("decoder_rank", 999999)), int(item.get("candidate_index", 0))))


def _selected_row(
    *,
    policy: str,
    selected: dict[str, Any],
    default: dict[str, Any],
    selector_logit: float | None = None,
    final_score: float | None = None,
    default_final_score: float | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    live_success = bool(selected.get("live_success", False))
    base_success = bool(default.get("live_success", False))
    on_count = int(selected.get("on_count", selected.get("candidate_num_visual_on_layers", 0)))
    base_on = int(default.get("on_count", default.get("candidate_num_visual_on_layers", 0)))
    return {
        "id": selected["id"],
        "benchmark": selected.get("benchmark"),
        "policy": policy,
        "selected_is_default": bool(selected is default or selected.get("is_default", False)),
        "fallback_used": bool(fallback_used),
        "live_target_positive": live_success,
        "candidate_score": float(selected.get("candidate_score", 0.0)),
        "candidate_num_visual_on_layers": on_count,
        "budget_count": int(selected.get("budget_count", 0)),
        "transition_count": int(selected.get("transition_count", 0)),
        "decoder_rank": int(selected.get("decoder_rank", 0)),
        "candidate_index": int(selected.get("candidate_index", 0)),
        "decoder_score": float(selected.get("decoder_score", 0.0)),
        "layers_one_based": list(selected.get("layers_one_based", [])),
        "base_live_target_positive": base_success,
        "base_num_visual_on_layers": base_on,
        "fix": bool(live_success and not base_success),
        "regression": bool(base_success and not live_success),
        "harmful_densification": bool((not live_success) and on_count > base_on),
        "selector_logit": None if selector_logit is None else float(selector_logit),
        "final_score": None if final_score is None else float(final_score),
        "default_final_score": None if default_final_score is None else float(default_final_score),
    }


def select_with_policy(
    scores: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    lambda_on: float,
    lambda_transition: float,
    margin: float,
    success_threshold: float | None = None,
    max_on: int | None = None,
    min_budget_slack: int | None = None,
    min_on_delta_vs_default: int | None = None,
    max_candidate_rank: int | None = None,
    policy_name: str = "route_verifier",
) -> list[dict[str, Any]]:
    if int(scores.numel()) != len(metadata):
        raise ValueError(f"scores length {scores.numel()} != metadata length {len(metadata)}")
    grouped: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for score, item in zip(scores.float().tolist(), metadata, strict=True):
        grouped[str(item["id"])].append((float(score), item))

    selected_rows: list[dict[str, Any]] = []
    for sample_id, items in sorted(grouped.items()):
        candidates = [item for _, item in items]
        default = _default_metadata(candidates)
        default_score = next(float(score) for score, item in items if item is default)
        default_final = default_score - float(lambda_on) * float(default.get("on_count", 0.0)) - float(
            lambda_transition
        ) * float(default.get("transition_count", 0.0))
        adjusted = []
        for score, item in items:
            if item is not default:
                if min_budget_slack is not None and int(item.get("on_count", 0)) - int(
                    item.get("budget_count", 0)
                ) < int(min_budget_slack):
                    continue
                if min_on_delta_vs_default is not None and int(item.get("on_count", 0)) - int(
                    default.get("on_count", 0)
                ) < int(min_on_delta_vs_default):
                    continue
                if max_candidate_rank is not None and int(item.get("decoder_rank", 999999)) > int(max_candidate_rank):
                    continue
            adjusted.append(
                (
                    float(score)
                    - float(lambda_on) * float(item.get("on_count", 0.0))
                    - float(lambda_transition) * float(item.get("transition_count", 0.0)),
                    float(score),
                    item,
                )
            )
        best_final, best_score, best = max(
            adjusted,
            key=lambda triple: (
                triple[0],
                -float(triple[2].get("on_count", 0.0)),
                -int(triple[2].get("decoder_rank", 999999)),
            ),
        )
        should_switch = best is not default and (best_final - default_final) >= float(margin)
        if success_threshold is not None:
            should_switch = should_switch and torch.sigmoid(torch.tensor(best_score)).item() >= float(success_threshold)
        if max_on is not None:
            should_switch = should_switch and int(best.get("on_count", 0)) <= int(max_on)
        final = best if should_switch else default
        final_score = best_final if should_switch else default_final
        final_logit = best_score if should_switch else default_score
        selected_rows.append(
            _selected_row(
                policy=policy_name,
                selected=final,
                default=default,
                selector_logit=final_logit,
                final_score=final_score,
                default_final_score=default_final,
                fallback_used=not should_switch,
            )
        )
    return selected_rows


def select_decoder_top1(metadata: list[dict[str, Any]], *, policy_name: str = "decoder_top1") -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in metadata:
        grouped[str(item["id"])].append(item)
    rows: list[dict[str, Any]] = []
    for _, candidates in sorted(grouped.items()):
        default = _default_metadata(candidates)
        rows.append(_selected_row(policy=policy_name, selected=default, default=default))
    return rows


def select_budget_or_above_then_score(
    metadata: list[dict[str, Any]],
    *,
    policy_name: str = "budget_or_above_then_score",
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in metadata:
        grouped[str(item["id"])].append(item)
    rows: list[dict[str, Any]] = []
    for _, candidates in sorted(grouped.items()):
        default = _default_metadata(candidates)
        eligible = [item for item in candidates if int(item.get("on_count", 0)) >= int(item.get("budget_count", 0))]
        if not eligible:
            eligible = candidates
        selected = max(
            eligible,
            key=lambda item: (
                float(item.get("decoder_score", 0.0)),
                -int(item.get("decoder_rank", 999999)),
                -int(item.get("on_count", 0)),
            ),
        )
        rows.append(_selected_row(policy=policy_name, selected=selected, default=default))
    return rows


def select_live_oracle(metadata: list[dict[str, Any]], *, policy_name: str = "live_oracle") -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in metadata:
        grouped[str(item["id"])].append(item)
    rows: list[dict[str, Any]] = []
    for _, candidates in sorted(grouped.items()):
        default = _default_metadata(candidates)
        selected = max(
            candidates,
            key=lambda item: (
                float(bool(item.get("live_success", False))),
                float(item.get("candidate_score", 0.0)),
                -int(item.get("on_count", 0)),
                -int(item.get("decoder_rank", 999999)),
            ),
        )
        rows.append(_selected_row(policy=policy_name, selected=selected, default=default))
    return rows


def evaluate_selected_rows(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_benchmark: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "rows": 0.0,
            "positive": 0.0,
            "score_sum": 0.0,
            "on_sum": 0.0,
            "transition_sum": 0.0,
            "fixes": 0.0,
            "regressions": 0.0,
            "harmful_densification": 0.0,
        }
    )
    for row in selected_rows:
        benchmark = str(row.get("benchmark", "unknown"))
        stats = by_benchmark[benchmark]
        stats["rows"] += 1.0
        stats["positive"] += float(bool(row.get("live_target_positive", False)))
        stats["score_sum"] += float(row.get("candidate_score", 0.0))
        stats["on_sum"] += float(row.get("candidate_num_visual_on_layers", 0.0))
        stats["transition_sum"] += float(row.get("transition_count", 0.0))
        stats["fixes"] += float(bool(row.get("fix", False)))
        stats["regressions"] += float(bool(row.get("regression", False)))
        stats["harmful_densification"] += float(bool(row.get("harmful_densification", False)))

    rows = len(selected_rows)
    positives = sum(1 for row in selected_rows if bool(row.get("live_target_positive", False)))
    score_sum = sum(float(row.get("candidate_score", 0.0)) for row in selected_rows)
    on_sum = sum(float(row.get("candidate_num_visual_on_layers", 0.0)) for row in selected_rows)
    transition_sum = sum(float(row.get("transition_count", 0.0)) for row in selected_rows)
    fixes = sum(1 for row in selected_rows if bool(row.get("fix", False)))
    regressions = sum(1 for row in selected_rows if bool(row.get("regression", False)))
    harmful_densification = sum(1 for row in selected_rows if bool(row.get("harmful_densification", False)))
    fallback_count = sum(1 for row in selected_rows if bool(row.get("fallback_used", False)))
    return {
        "num_rows": rows,
        "selected_positive_rows": positives,
        "selected_positive_rate": positives / rows if rows else 0.0,
        "avg_candidate_score": score_sum / rows if rows else 0.0,
        "avg_selected_on": on_sum / rows if rows else 0.0,
        "avg_transition_count": transition_sum / rows if rows else 0.0,
        "fixes": fixes,
        "regressions": regressions,
        "fix_minus_regression": fixes - regressions,
        "harmful_densification": harmful_densification,
        "fallback_rows": fallback_count,
        "fallback_rate": fallback_count / rows if rows else 0.0,
        "by_benchmark": {
            benchmark: {
                "num_rows": int(stats["rows"]),
                "selected_positive_rows": int(stats["positive"]),
                "selected_positive_rate": stats["positive"] / stats["rows"] if stats["rows"] else 0.0,
                "avg_candidate_score": stats["score_sum"] / stats["rows"] if stats["rows"] else 0.0,
                "avg_selected_on": stats["on_sum"] / stats["rows"] if stats["rows"] else 0.0,
                "avg_transition_count": stats["transition_sum"] / stats["rows"] if stats["rows"] else 0.0,
                "fixes": int(stats["fixes"]),
                "regressions": int(stats["regressions"]),
                "harmful_densification": int(stats["harmful_densification"]),
            }
            for benchmark, stats in sorted(by_benchmark.items())
        },
    }


def candidate_set_summary(labels: torch.Tensor, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    groups = group_indices_from_metadata(metadata)
    positive_groups = {
        str(item["id"])
        for item, label in zip(metadata, labels.float().tolist(), strict=True)
        if float(label) > 0.5
    }
    return {
        "num_groups": len(groups),
        "num_positive_groups": len(positive_groups),
        "positive_group_rate": len(positive_groups) / len(groups) if groups else 0.0,
        "num_candidates": int(labels.numel()),
        "num_positive_candidates": int(labels.sum().item()),
        "positive_candidate_rate": float(labels.float().mean().item()) if labels.numel() else 0.0,
        "avg_candidate_on": (
            sum(float(item.get("on_count", 0.0)) for item in metadata) / len(metadata) if metadata else 0.0
        ),
    }
