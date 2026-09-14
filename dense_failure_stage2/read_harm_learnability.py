"""Pure contracts for the READ-harm structure and learnability phase."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .predictability_learnability import SummaryScalarPredictor, robust_target_scale


FEATURE_GROUP_ORDER = ("F_ALL", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
MODEL_ORDER = ("mlp", "linear")


def validate_attention_reconstruction(
    *, max_abs: float, mean_abs: float, cosine: float,
    max_abs_tolerance: float, mean_abs_tolerance: float, min_cosine: float,
) -> bool:
    """Apply all frozen operation-fidelity bounds without fallback."""

    values = (max_abs, mean_abs, cosine, max_abs_tolerance, mean_abs_tolerance, min_cosine)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    return (
        float(max_abs) <= float(max_abs_tolerance)
        and float(mean_abs) <= float(mean_abs_tolerance)
        and float(cosine) >= float(min_cosine)
    )


def classify_read_behavior(full_correct: bool, write_only_correct: bool) -> str:
    """Classify the frozen FULL/WRITE_ONLY behavioral pair."""

    pair = bool(full_correct), bool(write_only_correct)
    return {
        (False, True): "read_harmful_flip",
        (True, False): "read_beneficial_flip",
        (False, False): "stable_wrong",
        (True, True): "stable_correct",
    }[pair]


def harm_sign(value: float) -> str:
    if float(value) > 0.0:
        return "harmful"
    if float(value) < 0.0:
        return "beneficial"
    return "zero"


def adjacent_transition_counts(values: Sequence[float] | np.ndarray) -> dict[tuple[str, str], int]:
    signs = [harm_sign(float(value)) for value in values]
    return dict(Counter(zip(signs[:-1], signs[1:])))


def contiguous_run_lengths(
    values: Sequence[float] | np.ndarray, *, sign: str
) -> list[int]:
    if sign not in {"harmful", "beneficial", "zero"}:
        raise ValueError("sign must be harmful, beneficial, or zero")
    runs: list[int] = []
    current = 0
    for value in values:
        if harm_sign(float(value)) == sign:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.dot(left, right))
    denominator = _norm(left) * _norm(right)
    return 0.0 if denominator == 0.0 else numerator / denominator


def summarize_read_update(
    pre: np.ndarray,
    off: np.ndarray,
    full: np.ndarray,
    token_deltas: np.ndarray,
    *,
    top_k: int,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Return preregistered F1-F3 scalars from the immediate READ update."""

    pre = np.asarray(pre, dtype=np.float64)
    off = np.asarray(off, dtype=np.float64)
    full = np.asarray(full, dtype=np.float64)
    token_deltas = np.asarray(token_deltas, dtype=np.float64)
    if pre.ndim != 1 or off.shape != pre.shape or full.shape != pre.shape:
        raise ValueError("pre/OFF/FULL vectors must be aligned rank-one arrays")
    if token_deltas.ndim != 2 or token_deltas.shape[1] != pre.shape[0] or len(token_deltas) < 1:
        raise ValueError("token deltas must be a nonempty token-by-hidden matrix")
    delta = full - off
    delta_norm = _norm(delta)
    magnitudes = np.linalg.norm(token_deltas, axis=1)
    total = float(magnitudes.sum())
    if total <= epsilon:
        shares = np.zeros_like(magnitudes)
        entropy = 0.0
    else:
        shares = magnitudes / total
        positive = shares > 0.0
        entropy = float(-(shares[positive] * np.log(shares[positive])).sum())
    ordered = np.sort(shares)[::-1]
    return {
        "f1_read_update_norm": delta_norm,
        "f1_update_over_pre": delta_norm / max(_norm(pre), epsilon),
        "f1_update_over_off": delta_norm / max(_norm(off), epsilon),
        "f2_cos_update_pre": _cosine(delta, pre),
        "f2_cos_update_off": _cosine(delta, off),
        "f2_cos_update_full": _cosine(delta, full),
        "f3_max_over_mean_update": float(magnitudes.max() / max(float(magnitudes.mean()), epsilon)),
        "f3_top1_update_share": float(ordered[:1].sum()),
        "f3_topk_update_share": float(ordered[: max(1, int(top_k))].sum()),
        "f3_update_entropy": entropy,
    }


def _tokens_for_mass(weights: torch.Tensor, threshold: float) -> float:
    ordered = torch.sort(weights.float(), descending=True).values
    cumulative = ordered.cumsum(dim=0)
    reached = torch.nonzero(cumulative >= float(threshold), as_tuple=False)
    return float(len(ordered) if len(reached) == 0 else int(reached[0, 0]) + 1)


def summarize_read_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    visual_mask: torch.Tensor,
    output_projection_weight: torch.Tensor,
    residual: torch.Tensor,
    visual_positions: torch.Tensor,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Summarize final-query READ mechanics from projected, rotary q/k/v.

    ``query`` is ``[heads, head_dim]`` and repeated ``keys``/``values`` are
    ``[heads, sequence, head_dim]``. The visual probabilities retain their
    mass under the full softmax; entropy and spatial summaries condition on
    the visual mass to describe which visual rows receive it.
    """

    if query.ndim != 2 or keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("q/k/v shapes must be [heads,dim] and [heads,tokens,dim]")
    if keys.shape[0] != query.shape[0] or keys.shape[2] != query.shape[1]:
        raise ValueError("q/k/v head dimensions differ")
    visual_mask = visual_mask.to(device=keys.device, dtype=torch.bool)
    if visual_mask.ndim != 1 or len(visual_mask) != keys.shape[1] or not bool(visual_mask.any()):
        raise ValueError("visual mask must select at least one key")
    if bool(visual_mask.all()):
        raise ValueError("READ comparison requires at least one nonvisual key")
    heads, _, head_dim = keys.shape
    if output_projection_weight.shape != (heads * head_dim, heads * head_dim):
        raise ValueError("output projection shape differs from concatenated heads")
    if residual.numel() != heads * head_dim:
        raise ValueError("residual width differs from attention output")

    query_f = query.float()
    keys_f = keys.float()
    values_f = values.float()
    logits = torch.einsum("hd,htd->ht", query_f, keys_f) / math.sqrt(head_dim)
    full_prob = torch.softmax(logits, dim=-1)
    text_logits = logits.masked_fill(visual_mask.unsqueeze(0), -torch.inf)
    text_prob = torch.softmax(text_logits, dim=-1)
    visual_prob = full_prob[:, visual_mask]
    visual_mass = visual_prob.sum(dim=-1)
    conditional = visual_prob / visual_mass.clamp_min(float(epsilon)).unsqueeze(-1)
    entropy_by_head = -(conditional * conditional.clamp_min(float(epsilon)).log()).sum(dim=-1)
    top_values = torch.sort(visual_prob, dim=-1, descending=True).values
    top_k = min(5, top_values.shape[-1])

    visual_values = values_f[:, visual_mask]
    visual_value = torch.einsum("hv,hvd->hd", visual_prob, visual_values)
    full_output = torch.einsum("ht,htd->hd", full_prob, values_f)
    text_output = torch.einsum("ht,htd->hd", text_prob, values_f)
    head_delta = full_output - text_output
    projected_delta = F.linear(
        head_delta.reshape(-1), output_projection_weight.float()
    )
    head_delta_norm = torch.linalg.vector_norm(head_delta, dim=-1)

    visual_logits = logits[:, visual_mask]
    sorted_logits, sorted_indices = torch.sort(visual_logits, dim=-1, descending=True)
    compat_k = min(5, sorted_logits.shape[-1])
    top_key_indices = sorted_indices[:, 0]
    top_keys = visual_values.new_empty((heads, head_dim))
    visual_keys = keys_f[:, visual_mask]
    for head in range(heads):
        top_keys[head] = visual_keys[head, top_key_indices[head]]
    cosine = F.cosine_similarity(query_f, top_keys, dim=-1)

    aggregate_conditional = conditional.mean(dim=0)
    aggregate_conditional = aggregate_conditional / aggregate_conditional.sum().clamp_min(float(epsilon))
    indices = torch.linspace(
        0.0, 1.0, aggregate_conditional.numel(), device=aggregate_conditional.device
    )
    index_center = (aggregate_conditional * indices).sum()
    index_spread = torch.sqrt(
        (aggregate_conditional * (indices - index_center).square()).sum().clamp_min(0.0)
    )
    visual_positions = visual_positions.to(device=keys.device, dtype=torch.float32)
    if visual_positions.shape != (int(visual_mask.sum()), 2):
        raise ValueError("visual positions must be [visual_tokens,2]")
    normalized_positions = []
    for axis in range(2):
        coordinate = visual_positions[:, axis]
        span = coordinate.max() - coordinate.min()
        normalized_positions.append(
            torch.zeros_like(coordinate) if float(span) == 0.0 else (coordinate - coordinate.min()) / span
        )
    spatial = torch.stack(normalized_positions, dim=-1)
    center = (aggregate_conditional[:, None] * spatial).sum(dim=0)
    spatial_spread = torch.sqrt(
        (aggregate_conditional[:, None] * (spatial - center).square()).sum().clamp_min(0.0)
    )

    fifth = sorted_logits[:, compat_k - 1]
    second = sorted_logits[:, min(1, sorted_logits.shape[-1] - 1)]
    return {
        "f4_total_visual_attention_mass": float(visual_mass.mean()),
        "f4_visual_attention_entropy": float(entropy_by_head.mean()),
        "f4_top1_attention_weight": float(top_values[:, 0].mean()),
        "f4_top5_attention_mass": float(top_values[:, :top_k].sum(dim=-1).mean()),
        "f4_effective_visual_tokens": float(torch.exp(entropy_by_head).mean()),
        "f4_head_entropy_mean": float(entropy_by_head.mean()),
        "f4_head_entropy_std": float(entropy_by_head.std(unbiased=False)),
        "f5_max_qk": float(sorted_logits[:, 0].mean()),
        "f5_mean_topk_qk": float(sorted_logits[:, :compat_k].mean()),
        "f5_top1_top2_gap": float((sorted_logits[:, 0] - second).mean()),
        "f5_top1_top5_gap": float((sorted_logits[:, 0] - fifth).mean()),
        "f5_cosine_top_visual_key": float(cosine.mean()),
        "f6_read_output_norm": float(torch.linalg.vector_norm(projected_delta)),
        "f6_visual_value_aggregation_norm": float(torch.linalg.vector_norm(visual_value)),
        "f6_read_output_over_residual": float(
            torch.linalg.vector_norm(projected_delta)
            / torch.linalg.vector_norm(residual.float()).clamp_min(float(epsilon))
        ),
        "f6_head_output_norm_mean": float(head_delta_norm.mean()),
        "f6_head_output_norm_std": float(head_delta_norm.std(unbiased=False)),
        "f7_tokens_for_50pct": _tokens_for_mass(aggregate_conditional, 0.5),
        "f7_tokens_for_80pct": _tokens_for_mass(aggregate_conditional, 0.8),
        "f7_attention_index_spread": float(index_spread),
        "f7_attention_center_h": float(center[0]),
        "f7_attention_center_w": float(center[1]),
        "f7_attention_spatial_spread": float(spatial_spread),
        "f7_attention_spatial_concentration": float(1.0 / (1.0 + spatial_spread)),
    }


def exact_nuisance_matches(
    rows: Sequence[Mapping[str, Any]],
    *,
    treated: str,
    control: str,
    cell_key: str,
    seed: int,
) -> list[dict[str, str]]:
    """Make deterministic one-to-one matches strictly within frozen cells."""

    by_cell: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {treated: [], control: []}
    )
    for row in rows:
        cohort = str(row["cohort"])
        if cohort in {treated, control}:
            by_cell[str(row[cell_key])][cohort].append(row)
    output: list[dict[str, str]] = []
    for cell, cohorts in sorted(by_cell.items()):
        ordered: dict[str, list[Mapping[str, Any]]] = {}
        for cohort, values in cohorts.items():
            ordered[cohort] = sorted(
                values,
                key=lambda row: sha256(
                    f"{int(seed)}|{cell}|{cohort}|{row['state_id']}".encode()
                ).hexdigest(),
            )
        for treated_row, control_row in zip(ordered[treated], ordered[control]):
            output.append(
                {
                    "cell": cell,
                    "treated_cohort": treated,
                    "control_cohort": control,
                    "treated_state_id": str(treated_row["state_id"]),
                    "control_state_id": str(control_row["state_id"]),
                }
            )
    return output


def select_dense_winner(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the routed feature/model prospectively by dense OOF performance."""

    if not candidates:
        raise ValueError("winner selection requires candidates")
    group_rank = {name: index for index, name in enumerate(FEATURE_GROUP_ORDER)}
    model_rank = {name: index for index, name in enumerate(MODEL_ORDER)}
    for row in candidates:
        if str(row["feature_group"]) not in group_rank or str(row["model"]) not in model_rank:
            raise ValueError("winner candidate is outside the preregistered ladder")
        if not math.isfinite(float(row["spearman"])) or not math.isfinite(float(row["harmful_auroc"])):
            raise ValueError("winner candidate metrics must be finite")
    winner = min(
        candidates,
        key=lambda row: (
            -float(row["spearman"]),
            -float(row["harmful_auroc"]),
            group_rank[str(row["feature_group"])],
            model_rank[str(row["model"])],
        ),
    )
    return dict(winner)


def validate_feature_census(
    expected_state_ids: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    expected = [str(value) for value in expected_state_ids]
    observed = [str(row["state_id"]) for row in rows]
    if len(expected) != len(set(expected)):
        raise ValueError("expected state IDs contain duplicates")
    if len(observed) != len(set(observed)):
        raise ValueError("feature rows contain duplicate state IDs")
    if set(expected) != set(observed):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(f"feature census differs: missing={missing[:3]} extra={extra[:3]}")
    for row in rows:
        features = row.get("features")
        if not isinstance(features, Mapping) or not features:
            raise ValueError("feature row has no feature mapping")
        if not all(math.isfinite(float(value)) for value in features.values()):
            raise ValueError("feature row contains a non-finite value")


def fit_predict_fold(
    features: torch.Tensor,
    target: Sequence[float] | np.ndarray,
    *,
    fit_indices: Sequence[int] | np.ndarray,
    calibration_indices: Sequence[int] | np.ndarray,
    test_indices: Sequence[int] | np.ndarray,
    fit_weights: Sequence[float] | np.ndarray,
    calibration_weights: Sequence[float] | np.ndarray,
    kind: str,
    spec: Mapping[str, Any],
    hidden_size: int,
    dropout: float,
    target_scale_floor: float,
    gradient_clip_norm: float,
    seed: int,
    device: torch.device,
    classification: bool,
) -> dict[str, Any]:
    """Fit one fold with frozen normalization and early stopping."""

    if features.ndim != 2 or len(features) < 1:
        raise ValueError("features must be a nonempty matrix")
    target_array = np.asarray(target, dtype=np.float64)
    fit = np.asarray(fit_indices, dtype=np.int64)
    calibration = np.asarray(calibration_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if len(target_array) != len(features) or min(len(fit), len(calibration), len(test)) < 1:
        raise ValueError("target and fold roles must be nonempty and aligned")
    fit_weight = np.asarray(fit_weights, dtype=np.float64)
    calibration_weight = np.asarray(calibration_weights, dtype=np.float64)
    if len(fit_weight) != len(fit) or len(calibration_weight) != len(calibration):
        raise ValueError("role weights differ from role indices")
    if not np.isfinite(target_array).all() or not np.isfinite(fit_weight).all() or not np.isfinite(calibration_weight).all():
        raise ValueError("training inputs contain non-finite values")

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    fit_tensor = features.index_select(0, torch.as_tensor(fit, dtype=torch.long)).float()
    mean = fit_tensor.mean(dim=0)
    std = fit_tensor.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    del fit_tensor
    scale = None if classification else robust_target_scale(
        target_array[fit], floor=float(target_scale_floor)
    )
    train_target = target_array if classification else scale.transform(target_array)
    model = SummaryScalarPredictor(
        kind=kind,
        input_size=int(features.shape[1]),
        hidden_size=int(hidden_size),
        dropout=float(dropout),
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history = []

    def forward_indices(indices: np.ndarray) -> torch.Tensor:
        index = torch.as_tensor(indices, dtype=torch.long)
        values = features.index_select(0, index).to(device=device, dtype=torch.float32)
        values = (values - mean.to(device)) / std.to(device)
        return model(values)

    batch_size = int(spec["batch_size"])
    for epoch in range(int(spec["maximum_epochs"])):
        model.train()
        order = torch.randperm(len(fit), generator=generator).numpy()
        for start in range(0, len(order), batch_size):
            local = order[start : start + batch_size]
            indices = fit[local]
            weight = torch.as_tensor(fit_weight[local], dtype=torch.float32, device=device)
            truth = torch.as_tensor(train_target[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_indices(indices)
            loss_values = (
                F.binary_cross_entropy_with_logits(prediction, truth, reduction="none")
                if classification
                else F.huber_loss(prediction, truth, delta=1.0, reduction="none")
            )
            loss = (loss_values * weight).sum() / weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = forward_indices(calibration)
            truth = torch.as_tensor(train_target[calibration], dtype=torch.float32, device=device)
            weight = torch.as_tensor(calibration_weight, dtype=torch.float32, device=device)
            losses = (
                F.binary_cross_entropy_with_logits(prediction, truth, reduction="none")
                if classification
                else F.huber_loss(prediction, truth, delta=1.0, reduction="none")
            )
            calibration_loss = float(((losses * weight).sum() / weight.sum()).cpu())
        history.append({"epoch": epoch, "calibration_loss": calibration_loss})
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= int(spec["minimum_epochs"]) and stale >= int(spec["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("fold training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(test), batch_size):
            outputs.append(forward_indices(test[start : start + batch_size]).float().cpu())
    raw = torch.cat(outputs).numpy().astype(np.float64)
    prediction = 1.0 / (1.0 + np.exp(-raw)) if classification else scale.inverse(raw)
    return {
        "test_indices": test,
        "test_prediction": prediction,
        "best_epoch": int(best_epoch),
        "best_calibration_loss": float(best_loss),
        "history": history,
        "standardizer_mean": mean,
        "standardizer_std": std,
    }
