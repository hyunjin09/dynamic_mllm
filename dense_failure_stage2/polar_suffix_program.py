"""Contracts and model for P90-triggered complete suffix-program prediction."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


ACTION_NAMES = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_NAMES)}
BOS_INDEX = len(ACTION_NAMES)


def canonical_suffix(
    actions: Sequence[str], *, trigger_layer: int, total_layers: int = 28
) -> list[str]:
    """Validate a complete P90-aligned route and return its post-trigger suffix."""
    normalized = [str(action).upper() for action in actions]
    if len(normalized) != int(total_layers):
        raise ValueError(f"route must contain exactly {int(total_layers)} actions")
    if any(action not in ACTION_TO_INDEX for action in normalized):
        raise ValueError("route contains an unsupported action")
    trigger = int(trigger_layer)
    if trigger < 0 or trigger >= int(total_layers):
        raise ValueError("trigger layer is outside the route")
    if any(action != "FULL" for action in normalized[:trigger]):
        raise ValueError("route contains a non-FULL pre-trigger action")
    return normalized[trigger:]


def assign_group_disjoint_dev(
    rows: Sequence[Mapping[str, Any]], *, dev_fraction: float, seed: int
) -> dict[str, str]:
    """Deterministically split image groups within composite data strata."""
    if not 0.0 < float(dev_fraction) < 1.0:
        raise ValueError("dev_fraction must be strictly between zero and one")
    uid_metadata: dict[str, tuple[str, tuple[str, str, str]]] = {}
    for row in rows:
        uid = str(row["uid"])
        value = (
            str(row["image_group_id"]),
            (str(row["dataset"]), str(row["source_regime"]), str(row["dense_outcome"])),
        )
        if uid in uid_metadata and uid_metadata[uid] != value:
            raise ValueError(f"inconsistent split metadata for UID {uid}")
        uid_metadata[uid] = value
    if len(uid_metadata) < 2:
        raise ValueError("at least two UIDs are required for a split")

    group_cells: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for group, cell in uid_metadata.values():
        group_cells[group].add(cell)
    signature_groups: dict[tuple[tuple[str, str, str], ...], list[str]] = defaultdict(list)
    for group, cells in group_cells.items():
        signature_groups[tuple(sorted(cells))].append(group)

    dev_groups: set[str] = set()
    for signature, groups in sorted(signature_groups.items()):
        ranked = sorted(
            groups,
            key=lambda group: sha256(f"{int(seed)}:{group}".encode()).hexdigest(),
        )
        if len(ranked) == 1:
            count = 0
        else:
            count = min(len(ranked) - 1, max(1, round(len(ranked) * float(dev_fraction))))
        dev_groups.update(ranked[:count])

    if not dev_groups:
        ranked = sorted(
            group_cells,
            key=lambda group: sha256(f"{int(seed)}:{group}".encode()).hexdigest(),
        )
        dev_groups.add(ranked[0])
    if dev_groups == set(group_cells):
        dev_groups.remove(max(dev_groups))
    return {
        uid: ("dev" if group in dev_groups else "train")
        for uid, (group, _cell) in sorted(uid_metadata.items())
    }


def hierarchical_program_weights(
    rows: Sequence[Mapping[str, Any]], *, minimum_cell_uids: int = 5
) -> list[float]:
    """Balance cells and UIDs while assigning each W program a 1/K share."""
    if not rows or int(minimum_cell_uids) < 1:
        raise ValueError("rows and a positive minimum_cell_uids are required")
    uid_cell: dict[str, tuple[str, str, str]] = {}
    program_count = Counter()
    cell_uids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        uid = str(row["uid"])
        cell = (str(row["dataset"]), str(row["source_regime"]), str(row["dense_outcome"]))
        if uid in uid_cell and uid_cell[uid] != cell:
            raise ValueError(f"UID {uid} appears in multiple weighting cells")
        uid_cell[uid] = cell
        program_count[uid] += 1
        cell_uids[cell].add(uid)

    raw_cell_mass = {
        cell: min(1.0, len(uids) / float(minimum_cell_uids))
        for cell, uids in cell_uids.items()
    }
    total_mass = sum(raw_cell_mass.values())
    weights = []
    for row in rows:
        uid = str(row["uid"])
        cell = uid_cell[uid]
        cell_mass = raw_cell_mass[cell] / total_mass
        weights.append(cell_mass / len(cell_uids[cell]) / program_count[uid])
    normalizer = sum(weights)
    return [weight / normalizer for weight in weights]


def _provenance_name(row: Mapping[str, Any]) -> str:
    origin = str(row.get("route_origin", ""))
    route_type = str(row.get("route_type", ""))
    if route_type == "preservation_full" or origin == "preservation_full":
        return "preservation"
    if origin == "existing_single":
        return "single"
    if origin == "existing_mcts":
        return "original_mcts"
    if origin in {"new_single", "new_mcts"}:
        return "robust_search"
    raise ValueError(f"unsupported route provenance: {route_type}/{origin}")


def build_program_corpus(
    base_routes: Sequence[Mapping[str, Any]],
    completeness_replays: Sequence[Mapping[str, Any]],
    work_rows: Sequence[Mapping[str, Any]],
    route_store_rows: Sequence[Mapping[str, Any]],
    *,
    total_layers: int = 28,
) -> list[dict[str, Any]]:
    """Build the conservative P90 corpus from replay-bound complete programs."""
    work = {str(row["uid"]): row for row in work_rows}
    if len(work) != len(work_rows):
        raise ValueError("work manifest contains duplicate UIDs")
    route_store = {
        (str(row["uid"]), str(row["route_id"])): row for row in route_store_rows
    }
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    def add(
        *,
        uid: str,
        actions: Sequence[str],
        provenance: str,
        source_reference: Mapping[str, Any],
        expected_token_ids: Sequence[int],
    ) -> None:
        if uid not in work:
            raise ValueError(f"program UID is absent from the P90 work manifest: {uid}")
        work_row = work[uid]
        trigger = work_row.get("triggers", {}).get("P90")
        if trigger is None:
            raise ValueError(f"program UID has no P90 trigger: {uid}")
        suffix = canonical_suffix(
            actions, trigger_layer=int(trigger), total_layers=int(total_layers)
        )
        route_key = "|".join(map(str, actions))
        key = (uid, route_key)
        tokens = [int(token) for token in expected_token_ids]
        if not tokens:
            raise ValueError(f"program lacks expected generated tokens: {uid}")
        if key in merged:
            row = merged[key]
            if row["expected_generated_token_ids"] != tokens:
                raise ValueError(f"duplicate program has conflicting generated tokens: {uid}")
            row["provenance"] = sorted(set(row["provenance"]) | {provenance})
            row["source_references"].append(dict(source_reference))
            return
        dense_correct = bool(work_row["dense_output"]["current_dense_correct"])
        sample = dict(work_row["sample"])
        program_id = sha256(f"{uid}:{route_key}".encode()).hexdigest()[:24]
        merged[key] = {
            "schema_version": "polar_suffix_program_corpus_v1",
            "program_id": program_id,
            "uid": uid,
            "dataset": str(work_row["dataset"]),
            "source_regime": str(work_row["source_regime"]),
            "dense_outcome": "C" if dense_correct else "W",
            "image_group_id": str(sample["image_group_id"]),
            "trigger_layer": int(trigger),
            "full_actions": list(map(str, actions)),
            "suffix_actions": suffix,
            "suffix_action_indices": [ACTION_TO_INDEX[action] for action in suffix],
            "dense_generated_token_ids": [
                int(token) for token in work_row["dense_output"]["generated_token_ids"]
            ],
            "expected_generated_token_ids": tokens,
            "provenance": [provenance],
            "source_references": [dict(source_reference)],
            "sample": sample,
        }

    for source in base_routes:
        uid = str(source["uid"])
        if uid not in work:
            raise ValueError(f"base route UID is absent from the work manifest: {uid}")
        dense_correct = bool(work[uid]["dense_output"]["current_dense_correct"])
        actions = [str(action) for action in source["actions"]]
        if dense_correct and any(action != "FULL" for action in actions):
            continue
        if not bool(source.get("exact_replay_valid")) or not bool(
            source.get("final_lmms_correct")
        ):
            raise ValueError(f"base route is not replay-valid and correct: {source.get('route_id')}")
        stored = route_store.get((uid, str(source["route_id"])))
        if stored is None:
            if dense_correct and all(action == "FULL" for action in actions):
                expected = work[uid]["dense_output"]["generated_token_ids"]
            else:
                raise ValueError(f"base route is absent from the global route store: {source['route_id']}")
        else:
            expected = stored["generated_token_ids"]
        add(
            uid=uid,
            actions=actions,
            provenance=_provenance_name(source),
            source_reference={
                "kind": "phase65",
                "route_id": str(source["route_id"]),
                "route_origin": str(source["route_origin"]),
            },
            expected_token_ids=expected,
        )

    for source in completeness_replays:
        if str(source.get("kind")) != "new_discovery":
            continue
        uid = str(source["uid"])
        if uid not in work:
            raise ValueError(f"completeness UID is absent from the work manifest: {uid}")
        if bool(work[uid]["dense_output"]["current_dense_correct"]):
            continue
        if not bool(source.get("correct")) or not bool(source.get("exact_token_parity")):
            raise ValueError(f"completeness discovery is not replay-valid: {source.get('state_id')}")
        actions = str(source["route_key"]).split("|")
        add(
            uid=uid,
            actions=actions,
            provenance="completeness_audit",
            source_reference={
                "kind": "phase73",
                "state_id": str(source["state_id"]),
                "search_stage": str(source["search_stage"]),
            },
            expected_token_ids=source["generated_token_ids"],
        )

    output = sorted(merged.values(), key=lambda row: (row["uid"], row["program_id"]))
    c_rows = [row for row in output if row["dense_outcome"] == "C"]
    if any(any(action != "FULL" for action in row["full_actions"]) for row in c_rows):
        raise AssertionError("Dense-C corpus contains a non-FULL target")
    return output


class PolarSuffixProgramPredictor(nn.Module):
    """Stage2-A trigger encoder plus a causal complete-program decoder."""

    def __init__(
        self,
        *,
        hidden_size: int,
        router_size: int = 256,
        num_heads: int = 4,
        decoder_layers: int = 2,
        feedforward_size: int = 1024,
        dropout: float = 0.1,
        total_layers: int = 28,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or router_size < 1 or total_layers < 1:
            raise ValueError("model dimensions must be positive")
        if router_size % num_heads:
            raise ValueError("router_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.router_size = int(router_size)
        self.total_layers = int(total_layers)

        self.text_query_projection = nn.Linear(hidden_size, router_size)
        self.visual_projection = nn.Linear(hidden_size, router_size)
        self.read_attention = nn.MultiheadAttention(
            router_size, num_heads, dropout=dropout, batch_first=True
        )
        self.write_attention = nn.MultiheadAttention(
            router_size, num_heads, dropout=dropout, batch_first=True
        )
        self.write_query = nn.Parameter(torch.empty(1, 1, router_size))
        nn.init.normal_(self.write_query, std=router_size**-0.5)
        self.context_projection = nn.Linear(2 * router_size, router_size)
        self.context_activation = nn.GELU()
        self.context_norm = nn.LayerNorm(router_size)

        self.absolute_layer_embedding = nn.Embedding(total_layers, router_size)
        self.relative_position_embedding = nn.Embedding(total_layers, router_size)
        self.previous_action_embedding = nn.Embedding(len(ACTION_NAMES) + 1, router_size)
        layer = nn.TransformerDecoderLayer(
            d_model=router_size,
            nhead=num_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.program_decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.output_head = nn.Linear(router_size, len(ACTION_NAMES))

    def compute_branches(
        self,
        text_states: torch.Tensor,
        visual_states: torch.Tensor,
        text_mask: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_states.ndim != 3 or visual_states.ndim != 3:
            raise ValueError("token states must have [batch, tokens, hidden] shape")
        if text_mask.shape != text_states.shape[:2] or visual_mask.shape != visual_states.shape[:2]:
            raise ValueError("token masks do not match token states")
        text_mask = text_mask.bool()
        visual_mask = visual_mask.bool()
        if not bool(text_mask.any(dim=1).all()) or not bool(visual_mask.any(dim=1).all()):
            raise ValueError("every row requires text and visual tokens")
        dtype = self.text_query_projection.weight.dtype
        text_states = text_states.to(dtype=dtype)
        visual_states = visual_states.to(dtype=dtype)
        last = text_mask.long().sum(dim=1) - 1
        batch = text_states.shape[0]
        query = self.text_query_projection(
            text_states[torch.arange(batch, device=text_states.device), last]
        ).unsqueeze(1)
        visual = self.visual_projection(visual_states)
        padding = ~visual_mask
        z_read, _ = self.read_attention(
            query, visual, visual, key_padding_mask=padding, need_weights=False
        )
        z_write, _ = self.write_attention(
            self.write_query.expand(batch, -1, -1),
            visual,
            visual,
            key_padding_mask=padding,
            need_weights=False,
        )
        return z_read[:, 0], z_write[:, 0]

    def compute_context(
        self,
        text_states: torch.Tensor,
        visual_states: torch.Tensor,
        text_mask: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        z_read, z_write = self.compute_branches(
            text_states, visual_states, text_mask, visual_mask
        )
        return self.context_norm(
            self.context_activation(self.context_projection(torch.cat((z_read, z_write), dim=-1)))
        )

    def decode_teacher_forced(
        self,
        context: torch.Tensor,
        trigger_layers: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2 or target_actions.ndim != 2:
            raise ValueError("context and targets must be batched")
        batch, length = target_actions.shape
        if context.shape[0] != batch or trigger_layers.shape != (batch,):
            raise ValueError("teacher-forcing batch dimensions differ")
        if length < 1 or length > self.total_layers:
            raise ValueError("invalid program length")
        device = context.device
        targets = target_actions.to(device=device, dtype=torch.long)
        trigger = trigger_layers.to(device=device, dtype=torch.long)
        relative = torch.arange(length, device=device).unsqueeze(0).expand(batch, -1)
        absolute = (trigger.unsqueeze(1) + relative).clamp(max=self.total_layers - 1)
        previous = torch.full((batch, length), BOS_INDEX, dtype=torch.long, device=device)
        if length > 1:
            previous[:, 1:] = targets[:, :-1].clamp(min=0)
        decoder_input = (
            self.absolute_layer_embedding(absolute)
            + self.relative_position_embedding(relative)
            + self.previous_action_embedding(previous)
        )
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )
        padding = targets.eq(-100)
        decoded = self.program_decoder(
            decoder_input,
            context.unsqueeze(1),
            tgt_mask=causal,
            tgt_key_padding_mask=padding,
        )
        return self.output_head(decoded)

    @torch.no_grad()
    def beam_decode(
        self, context: torch.Tensor, *, trigger_layer: int, beam_width: int = 8
    ) -> list[dict[str, Any]]:
        if context.shape != (1, self.router_size):
            raise ValueError("beam decoding requires one trigger context")
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        suffix_length = self.total_layers - int(trigger_layer)
        if suffix_length < 1:
            raise ValueError("trigger layer is outside the model")
        beams: list[tuple[list[int], float]] = [([], 0.0)]
        device = context.device
        for _step in range(suffix_length):
            expanded: list[tuple[list[int], float]] = []
            for prefix, score in beams:
                placeholder = torch.tensor(
                    [prefix + [0]], dtype=torch.long, device=device
                )
                logits = self.decode_teacher_forced(
                    context,
                    torch.tensor([int(trigger_layer)], dtype=torch.long, device=device),
                    placeholder,
                )[0, -1]
                log_probabilities = F.log_softmax(logits.float(), dim=-1)
                values, indices = torch.topk(log_probabilities, k=min(beam_width, len(ACTION_NAMES)))
                for value, index in zip(values.tolist(), indices.tolist()):
                    expanded.append((prefix + [int(index)], score + float(value)))
            expanded.sort(key=lambda item: (-item[1], item[0]))
            beams = expanded[:beam_width]
        return [
            {
                "action_indices": actions,
                "actions": [ACTION_NAMES[index] for index in actions],
                "score": score,
            }
            for actions, score in beams
        ]

    @torch.no_grad()
    def greedy_decode(self, context: torch.Tensor, *, trigger_layer: int) -> list[int]:
        prefix: list[int] = []
        device = context.device
        for _step in range(self.total_layers - int(trigger_layer)):
            target = torch.tensor([prefix + [0]], dtype=torch.long, device=device)
            logits = self.decode_teacher_forced(
                context,
                torch.tensor([int(trigger_layer)], dtype=torch.long, device=device),
                target,
            )
            prefix.append(int(logits[0, -1].argmax()))
        return prefix


def initialize_from_stage2a(
    model: PolarSuffixProgramPredictor, state_dict: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Copy every compatible Stage2-A branch/context tensor and verify equality."""
    mapping = {
        "text_query_projection.weight": "text_query_projection.weight",
        "text_query_projection.bias": "text_query_projection.bias",
        "visual_projection.weight": "visual_projection.weight",
        "visual_projection.bias": "visual_projection.bias",
        "read_attention.in_proj_weight": "read_attention.in_proj_weight",
        "read_attention.in_proj_bias": "read_attention.in_proj_bias",
        "read_attention.out_proj.weight": "read_attention.out_proj.weight",
        "read_attention.out_proj.bias": "read_attention.out_proj.bias",
        "write_attention.in_proj_weight": "write_attention.in_proj_weight",
        "write_attention.in_proj_bias": "write_attention.in_proj_bias",
        "write_attention.out_proj.weight": "write_attention.out_proj.weight",
        "write_attention.out_proj.bias": "write_attention.out_proj.bias",
        "write_query": "write_query",
        "context_projection.weight": "action_head.0.weight",
        "context_projection.bias": "action_head.0.bias",
        "context_norm.weight": "action_head.2.weight",
        "context_norm.bias": "action_head.2.bias",
    }
    own = model.state_dict()
    exact: dict[str, bool] = {}
    with torch.no_grad():
        for destination, source in mapping.items():
            if source not in state_dict or own[destination].shape != state_dict[source].shape:
                raise ValueError(f"incompatible Stage2-A initialization tensor: {source}")
            own[destination].copy_(state_dict[source].to(dtype=own[destination].dtype))
            exact[destination] = torch.equal(own[destination].cpu(), state_dict[source].cpu())
    model.load_state_dict(own)
    return {"copied_tensors": exact, "all_exact": all(exact.values())}


def weighted_program_loss(
    logits: torch.Tensor, target_actions: torch.Tensor, program_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted mean of length-normalized teacher-forced program NLLs."""
    if logits.shape[:2] != target_actions.shape or logits.shape[-1] != len(ACTION_NAMES):
        raise ValueError("logits and target shapes differ")
    if program_weights.shape != (target_actions.shape[0],):
        raise ValueError("program_weights must contain one value per program")
    losses = F.cross_entropy(
        logits.transpose(1, 2), target_actions.long(), reduction="none", ignore_index=-100
    )
    valid = target_actions.ne(-100)
    lengths = valid.sum(dim=1)
    if not bool((lengths > 0).all()):
        raise ValueError("every program requires at least one target action")
    per_program = (losses * valid).sum(dim=1) / lengths
    weights = program_weights.to(device=logits.device, dtype=per_program.dtype)
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("program weights must be finite and positive")
    return (per_program * weights).sum() / weights.sum(), per_program
