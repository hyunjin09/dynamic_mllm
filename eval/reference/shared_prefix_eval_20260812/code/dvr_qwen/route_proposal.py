"""Cache-only route proposal utilities for Phase 5B."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


IGNORE_INDEX = -100
DEFAULT_NUM_LAYERS = 28
BENCHMARK_TO_ID = {"gqa": 0, "textvqa": 1, "chartqa": 2, "docvqa": 3}


def sequence_one_based_to_actions(sequence: list[int], *, num_layers: int = DEFAULT_NUM_LAYERS) -> list[int]:
    """Convert dataset tokens 1..L, L+1(STOP) into action indices 0..L."""
    stop_token = num_layers + 1
    actions: list[int] = []
    for token in sequence:
        value = int(token)
        if 1 <= value <= num_layers:
            actions.append(value - 1)
        elif value == stop_token:
            actions.append(num_layers)
        else:
            raise ValueError(f"route token {value} is outside 1..{stop_token}")
    return actions


def route_mask_from_actions(actions: list[int], *, num_layers: int = DEFAULT_NUM_LAYERS) -> torch.Tensor:
    mask = torch.zeros(num_layers, dtype=torch.bool)
    for action in actions:
        value = int(action)
        if value in (IGNORE_INDEX, num_layers):
            continue
        if value < 0 or value >= num_layers:
            raise ValueError(f"route action {value} is outside 0..{num_layers}")
        mask[value] = True
    return mask


def route_mask_from_layers_one_based(
    layers_one_based: list[int],
    *,
    num_layers: int = DEFAULT_NUM_LAYERS,
) -> torch.Tensor:
    return route_mask_from_actions(
        sequence_one_based_to_actions([*layers_one_based, num_layers + 1], num_layers=num_layers),
        num_layers=num_layers,
    )


def benchmark_id(name: str) -> int:
    try:
        return BENCHMARK_TO_ID[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark {name!r}") from exc


def benchmark_ids(names: list[str]) -> torch.Tensor:
    return torch.tensor([benchmark_id(name) for name in names], dtype=torch.long)


def make_route_examples(
    rows: list[dict[str, Any]],
    *,
    num_layers: int = DEFAULT_NUM_LAYERS,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    stop_token = num_layers + 1
    for row in rows:
        for route in row.get("positive_routes", []):
            sequence = route.get("target_sequence")
            if sequence is None:
                sequence = [*route.get("layers_one_based", []), stop_token]
            actions = sequence_one_based_to_actions([int(item) for item in sequence], num_layers=num_layers)
            budget_count = int(route.get("num_visual_on_layers", len(route_mask_from_actions(actions, num_layers=num_layers).nonzero())))
            examples.append(
                {
                    "id": row["id"],
                    "benchmark": row["benchmark"],
                    "target_actions": actions,
                    "budget_count": budget_count,
                    "min_positive_on": int(row.get("min_positive_on", budget_count)),
                }
            )
    return examples


def collate_route_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty route-example batch")
    max_len = max(len(item["target_actions"]) for item in examples)
    targets = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long)
    for row_idx, item in enumerate(examples):
        actions = torch.tensor(item["target_actions"], dtype=torch.long)
        targets[row_idx, : actions.numel()] = actions
    return {
        "ids": [item["id"] for item in examples],
        "benchmarks": [item["benchmark"] for item in examples],
        "benchmark_ids": benchmark_ids([item["benchmark"] for item in examples]),
        "target_actions": targets,
        "budget_counts": torch.tensor([int(item["budget_count"]) for item in examples], dtype=torch.long),
        "min_positive_on": torch.tensor([int(item["min_positive_on"]) for item in examples], dtype=torch.long),
    }


@dataclass
class _Beam:
    score: float
    actions: tuple[int, ...]
    prev_action: int
    hidden: torch.Tensor
    stopped: bool


class BudgetConditionedRouteSetDecoder(nn.Module):
    """Autoregressive decoder that proposes complete VISUAL_ON layer sets."""

    def __init__(
        self,
        *,
        d_model: int,
        num_layers: int = DEFAULT_NUM_LAYERS,
        hidden_dim: int = 256,
        num_benchmarks: int = 4,
        use_layer_scores: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.use_layer_scores = bool(use_layer_scores)
        self.stop_action = self.num_layers
        self.start_action = self.num_layers + 1
        self.num_actions = self.num_layers + 1

        self.context_proj = nn.Sequential(
            nn.LayerNorm(self.d_model * 4 + 2),
            nn.Linear(self.d_model * 4 + 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.benchmark_emb = nn.Embedding(num_benchmarks, self.hidden_dim)
        self.action_emb = nn.Embedding(self.num_layers + 2, self.hidden_dim)
        self.step_emb = nn.Embedding(self.num_layers + 1, self.hidden_dim)
        self.decoder_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.num_actions)
        if self.use_layer_scores:
            self.layer_emb = nn.Embedding(self.num_layers, self.hidden_dim)
            self.layer_score_proj = nn.Linear(1, self.hidden_dim)
            self.layer_query = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.layer_score_bias = nn.Parameter(torch.tensor(1.0))
            self.stop_output = nn.Linear(self.hidden_dim, 1)

    def encode_context(
        self,
        batch: dict[str, Any],
        benchmark_ids_tensor: torch.Tensor,
        budget_counts: torch.Tensor,
        *,
        max_visual_tokens: float | None = None,
    ) -> torch.Tensor:
        global_mean = batch["global_mean"].float().mean(dim=1)
        window_mean = batch["window_mean"].float().mean(dim=1)
        last_token = batch["last_token"].float().mean(dim=1)
        if "visual_summaries" in batch:
            visual_mean = batch["visual_summaries"].float().mean(dim=(1, 2))
        else:
            visual_mean = torch.zeros_like(global_mean)

        visual_tokens = batch.get("num_visual_tokens")
        if visual_tokens is None:
            visual_tokens = torch.zeros(global_mean.shape[0], device=global_mean.device)
        visual_tokens = visual_tokens.to(device=global_mean.device, dtype=torch.float32)
        if max_visual_tokens is None:
            max_visual_tokens = float(visual_tokens.max().clamp_min(1.0).item())
        visual_norm = (visual_tokens / max(float(max_visual_tokens), 1.0)).unsqueeze(-1)
        budget_norm = budget_counts.to(device=global_mean.device, dtype=torch.float32).unsqueeze(-1) / float(
            self.num_layers
        )

        context = torch.cat([global_mean, window_mean, last_token, visual_mean, visual_norm, budget_norm], dim=-1)
        benchmark = self.benchmark_emb(benchmark_ids_tensor.to(device=global_mean.device, dtype=torch.long))
        return torch.tanh(self.context_proj(context) + benchmark)

    def _layer_memory(self, batch: dict[str, Any], *, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if not self.use_layer_scores:
            return None
        layer_memory = self.layer_emb.weight.unsqueeze(0).expand(batch_size, -1, -1)
        layer_scores = batch.get("layer_scores")
        if layer_scores is None:
            layer_scores = torch.zeros(batch_size, self.num_layers, device=device)
        layer_scores = layer_scores.to(device=device, dtype=torch.float32)
        if tuple(layer_scores.shape) != (batch_size, self.num_layers):
            raise ValueError(f"layer_scores must have shape ({batch_size}, {self.num_layers}), got {tuple(layer_scores.shape)}")
        return layer_memory + self.layer_score_proj(layer_scores.unsqueeze(-1))

    def _decode_step(
        self,
        hidden: torch.Tensor,
        prev_action: torch.Tensor,
        step_idx: int,
        layer_memory: torch.Tensor | None = None,
        layer_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step = torch.full_like(prev_action, min(int(step_idx), self.num_layers))
        decoder_input = self.action_emb(prev_action) + self.step_emb(step)
        next_hidden = self.decoder_cell(decoder_input, hidden)
        if layer_memory is None:
            return self.output(next_hidden), next_hidden
        query = self.layer_query(next_hidden)
        layer_logits = torch.einsum("bld,bd->bl", layer_memory, query) / math.sqrt(float(self.hidden_dim))
        if layer_scores is not None:
            layer_logits = layer_logits + self.layer_score_bias * layer_scores.to(
                device=layer_logits.device,
                dtype=layer_logits.dtype,
            )
        stop_logit = self.stop_output(next_hidden)
        return torch.cat([layer_logits, stop_logit], dim=-1), next_hidden

    def teacher_forced_logits(
        self,
        batch: dict[str, Any],
        benchmark_ids_tensor: torch.Tensor,
        budget_counts: torch.Tensor,
        target_actions: torch.Tensor,
        *,
        max_visual_tokens: float | None = None,
    ) -> torch.Tensor:
        hidden = self.encode_context(
            batch,
            benchmark_ids_tensor,
            budget_counts,
            max_visual_tokens=max_visual_tokens,
        )
        batch_size, seq_len = target_actions.shape
        layer_memory = self._layer_memory(batch, batch_size=batch_size, device=hidden.device)
        layer_scores = batch.get("layer_scores")
        if layer_scores is not None:
            layer_scores = layer_scores.to(device=hidden.device, dtype=torch.float32)
        prev = torch.full(
            (batch_size,),
            self.start_action,
            device=target_actions.device,
            dtype=torch.long,
        )
        logits = []
        for step_idx in range(seq_len):
            step_logits, hidden = self._decode_step(hidden, prev, step_idx, layer_memory, layer_scores)
            logits.append(step_logits)
            current = target_actions[:, step_idx]
            prev = torch.where(current >= 0, current, torch.full_like(current, self.stop_action))
        return torch.stack(logits, dim=1)

    def sequence_cross_entropy(self, logits: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_actions.to(device=logits.device).reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    @torch.no_grad()
    def generate_candidates(
        self,
        batch: dict[str, Any],
        benchmark_ids_tensor: torch.Tensor,
        budget_counts: torch.Tensor,
        *,
        beam_size: int = 32,
        max_candidates: int = 128,
        min_budget_slack: int = 0,
        max_budget_slack: int = 1,
        enforce_sorted: bool = True,
        max_visual_tokens: float | None = None,
    ) -> list[list[dict[str, Any]]]:
        self.eval()
        initial_hidden = self.encode_context(
            batch,
            benchmark_ids_tensor,
            budget_counts,
            max_visual_tokens=max_visual_tokens,
        )
        layer_memory = self._layer_memory(batch, batch_size=initial_hidden.shape[0], device=initial_hidden.device)
        layer_scores = batch.get("layer_scores")
        if layer_scores is not None:
            layer_scores = layer_scores.to(device=initial_hidden.device, dtype=torch.float32)
        outputs: list[list[dict[str, Any]]] = []
        for row_idx in range(initial_hidden.shape[0]):
            target_budget = int(budget_counts[row_idx].item())
            min_layers = max(0, target_budget - int(min_budget_slack))
            max_layers = min(self.num_layers, target_budget + int(max_budget_slack))
            beams = [
                _Beam(
                    score=0.0,
                    actions=(),
                    prev_action=self.start_action,
                    hidden=initial_hidden[row_idx],
                    stopped=False,
                )
            ]
            completed: list[_Beam] = []
            for step_idx in range(self.num_layers + 1):
                expanded: list[_Beam] = []
                for beam in beams:
                    if beam.stopped:
                        completed.append(beam)
                        continue
                    prev = torch.tensor([beam.prev_action], device=beam.hidden.device, dtype=torch.long)
                    row_layer_memory = None if layer_memory is None else layer_memory[row_idx : row_idx + 1]
                    row_layer_scores = None if layer_scores is None else layer_scores[row_idx : row_idx + 1]
                    logits, hidden = self._decode_step(
                        beam.hidden.unsqueeze(0),
                        prev,
                        step_idx,
                        row_layer_memory,
                        row_layer_scores,
                    )
                    masked_logits = logits[0].clone()
                    selected = set(beam.actions)
                    for action in selected:
                        masked_logits[action] = -torch.inf
                    if enforce_sorted and selected:
                        last_action = max(selected)
                        masked_logits[: last_action + 1] = -torch.inf
                    if len(selected) >= max_layers:
                        masked_logits[: self.num_layers] = -torch.inf
                    if len(selected) < min_layers:
                        masked_logits[self.stop_action] = -torch.inf
                    if not torch.isfinite(masked_logits).any():
                        masked_logits[self.stop_action] = logits[0, self.stop_action]

                    log_probs = F.log_softmax(masked_logits, dim=-1)
                    top_k = min(int(beam_size), int(torch.isfinite(log_probs).sum().item()))
                    values, actions = torch.topk(log_probs, k=max(top_k, 1))
                    for value, action_tensor in zip(values.tolist(), actions.tolist()):
                        action = int(action_tensor)
                        score = float(beam.score + value)
                        if action == self.stop_action:
                            expanded.append(
                                _Beam(score=score, actions=beam.actions, prev_action=action, hidden=hidden[0], stopped=True)
                            )
                        else:
                            expanded.append(
                                _Beam(
                                    score=score,
                                    actions=(*beam.actions, action),
                                    prev_action=action,
                                    hidden=hidden[0],
                                    stopped=False,
                                )
                            )
                completed.extend([beam for beam in expanded if beam.stopped])
                beams = sorted([beam for beam in expanded if not beam.stopped], key=lambda item: item.score, reverse=True)[
                    : int(beam_size)
                ]
                if not beams:
                    break
            completed.extend(beams)
            candidates = _unique_candidate_dicts(completed, num_layers=self.num_layers, max_candidates=max_candidates)
            budget_filtered = [
                item for item in candidates if min_layers <= int(item["num_visual_on_layers"]) <= max_layers
            ]
            outputs.append(budget_filtered or candidates)
        return outputs


def _unique_candidate_dicts(
    beams: list[_Beam],
    *,
    num_layers: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    best_by_mask: dict[tuple[bool, ...], dict[str, Any]] = {}
    for beam in beams:
        mask = route_mask_from_actions(list(beam.actions), num_layers=num_layers)
        key = tuple(bool(item) for item in mask.tolist())
        layers = [idx + 1 for idx, enabled in enumerate(key) if enabled]
        row = {
            "visual_on_mask": list(key),
            "layers_one_based": layers,
            "num_visual_on_layers": len(layers),
            "score": float(beam.score),
        }
        current = best_by_mask.get(key)
        if current is None or row["score"] > current["score"]:
            best_by_mask[key] = row
    return sorted(best_by_mask.values(), key=lambda item: item["score"], reverse=True)[: int(max_candidates)]


def candidate_coverage_rows(
    generated: list[list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    *,
    num_layers: int = DEFAULT_NUM_LAYERS,
) -> list[dict[str, Any]]:
    if len(generated) != len(rows):
        raise ValueError(f"generated candidate rows {len(generated)} != proposal rows {len(rows)}")
    out: list[dict[str, Any]] = []
    for candidates, row in zip(generated, rows):
        candidate_masks = [
            torch.tensor(item["visual_on_mask"], dtype=torch.bool)
            for item in candidates
            if len(item["visual_on_mask"]) == num_layers
        ]
        positive_masks = [
            route_mask_from_layers_one_based(route.get("layers_one_based", []), num_layers=num_layers)
            for route in row.get("positive_routes", [])
        ]
        min_hamming: int | None = None
        covered = False
        selected_hamming: int | None = None
        selected_covered = False
        selected_mask = candidate_masks[0] if candidate_masks else None
        if selected_mask is not None:
            for positive in positive_masks:
                distance = int(torch.logical_xor(positive, selected_mask).sum().item())
                selected_hamming = distance if selected_hamming is None else min(selected_hamming, distance)
                selected_covered = selected_covered or distance == 0
        for positive in positive_masks:
            for candidate in candidate_masks:
                distance = int(torch.logical_xor(positive, candidate).sum().item())
                min_hamming = distance if min_hamming is None else min(min_hamming, distance)
                covered = covered or distance == 0
        avg_generated_on = (
            sum(float(item["num_visual_on_layers"]) for item in candidates) / len(candidates) if candidates else 0.0
        )
        out.append(
            {
                "id": row["id"],
                "benchmark": row.get("benchmark"),
                "covered": bool(covered),
                "within_hamming2": bool(min_hamming is not None and min_hamming <= 2),
                "selected_covered": bool(selected_covered),
                "selected_within_hamming2": bool(selected_hamming is not None and selected_hamming <= 2),
                "min_hamming_to_positive": min_hamming,
                "selected_hamming_to_positive": selected_hamming,
                "num_candidates": len(candidates),
                "avg_generated_on": avg_generated_on,
                "selected_generated_on": int(candidates[0]["num_visual_on_layers"]) if candidates else 0,
                "min_positive_on": int(row.get("min_positive_on", 0)),
                "num_positive_routes": len(positive_masks),
            }
        )
    return out


def summarize_candidate_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = list(rows)
    for row in rows:
        groups[str(row.get("benchmark", "unknown"))].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(group)
        covered = sum(1 for row in group if row["covered"])
        within2 = sum(1 for row in group if row["within_hamming2"])
        selected = sum(1 for row in group if row["selected_covered"])
        selected_within2 = sum(1 for row in group if row["selected_within_hamming2"])
        hamming = [float(row["min_hamming_to_positive"]) for row in group if row["min_hamming_to_positive"] is not None]
        selected_hamming = [
            float(row["selected_hamming_to_positive"])
            for row in group
            if row["selected_hamming_to_positive"] is not None
        ]
        return {
            "num_rows": n,
            "covered_rows": covered,
            "coverage_rate": covered / n if n else 0.0,
            "within_hamming2_rows": within2,
            "within_hamming2_rate": within2 / n if n else 0.0,
            "selected_covered_rows": selected,
            "selected_coverage_rate": selected / n if n else 0.0,
            "selected_within_hamming2_rows": selected_within2,
            "selected_within_hamming2_rate": selected_within2 / n if n else 0.0,
            "avg_min_hamming_to_positive": sum(hamming) / len(hamming) if hamming else None,
            "avg_selected_hamming_to_positive": (
                sum(selected_hamming) / len(selected_hamming) if selected_hamming else None
            ),
            "avg_num_candidates": sum(float(row["num_candidates"]) for row in group) / n if n else 0.0,
            "avg_generated_on": sum(float(row["avg_generated_on"]) for row in group) / n if n else 0.0,
            "avg_selected_generated_on": (
                sum(float(row["selected_generated_on"]) for row in group) / n if n else 0.0
            ),
            "avg_min_positive_on": sum(float(row["min_positive_on"]) for row in group) / n if n else 0.0,
        }

    overall = summarize(groups.pop("overall"))
    overall["by_benchmark"] = {name: summarize(group) for name, group in sorted(groups.items())}
    return overall
