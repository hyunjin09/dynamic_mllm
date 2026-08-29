"""Prefix-trie supervision and deterministic balanced sampling."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import random
from typing import Any, Iterable, Sequence

import torch


class PrefixTrie:
    """Map every exact valid route prefix to all valid outgoing actions."""

    def __init__(self, routes: Iterable[Sequence[int]], *, num_actions: int = 4) -> None:
        normalized = [tuple(int(action) for action in route) for route in routes]
        if not normalized:
            raise ValueError("prefix trie requires at least one route")
        width = len(normalized[0])
        if width < 1 or any(len(route) != width for route in normalized):
            raise ValueError("all prefix-trie routes must have one common positive width")
        if len(set(normalized)) != len(normalized):
            raise ValueError("prefix-trie routes must be unique")
        if any(action < 0 or action >= num_actions for route in normalized for action in route):
            raise ValueError(f"prefix-trie actions must lie in [0, {num_actions - 1}]")
        outgoing: dict[tuple[int, ...], set[int]] = defaultdict(set)
        for route in normalized:
            for layer, action in enumerate(route):
                outgoing[route[:layer]].add(action)
        self.routes = tuple(normalized)
        self.num_layers = width
        self.num_actions = int(num_actions)
        self._outgoing = {prefix: frozenset(actions) for prefix, actions in outgoing.items()}

    @property
    def node_count(self) -> int:
        return len(self._outgoing)

    @property
    def action_sets(self) -> tuple[frozenset[int], ...]:
        """Return immutable valid-action sets for audit-only aggregation."""

        return tuple(self._outgoing.values())

    def valid_actions(self, prefix: Sequence[int]) -> frozenset[int]:
        key = tuple(int(action) for action in prefix)
        try:
            return self._outgoing[key]
        except KeyError as exc:
            raise KeyError(f"prefix is absent from the valid-route trie: {key}") from exc

    def valid_action_masks_for_route(self, route: Sequence[int]) -> torch.BoolTensor:
        normalized = tuple(int(action) for action in route)
        if normalized not in self.routes:
            raise ValueError("teacher-forced route is absent from the prefix trie")
        masks = torch.zeros(self.num_layers, self.num_actions, dtype=torch.bool)
        for layer in range(self.num_layers):
            for action in self.valid_actions(normalized[:layer]):
                masks[layer, action] = True
        return masks


def set_valued_action_loss(
    logits: torch.Tensor,
    valid_actions: torch.BoolTensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return ``-log sum(valid softmax mass)`` for every routed state."""

    if logits.shape != valid_actions.shape or logits.ndim < 2:
        raise ValueError("logits and valid-actions mask must share shape [..., actions]")
    if valid_actions.dtype != torch.bool:
        raise TypeError("valid-actions mask must be boolean")
    if not bool(valid_actions.any(dim=-1).all().item()):
        raise ValueError("every routed state must have at least one valid action")
    stable_logits = logits.float()
    masked = stable_logits.masked_fill(
        ~valid_actions.to(logits.device), float("-inf")
    )
    losses = torch.logsumexp(stable_logits, dim=-1) - torch.logsumexp(masked, dim=-1)
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError("reduction must be none, mean, or sum")


def _seed_value(*parts: object) -> int:
    return int.from_bytes(sha256(":".join(str(part) for part in parts).encode()).digest()[:8])


def balanced_epoch_indices(
    rows: Sequence[dict[str, Any]],
    *,
    samples_per_epoch: int,
    seed: int,
    epoch: int,
    world_size: int,
) -> list[int]:
    """Sample equal counts from every dataset × W2C/C2C cell."""

    datasets = ("gqa", "chartqa", "textvqa")
    route_types = ("W2C", "C2C")
    cell_count = len(datasets) * len(route_types)
    if samples_per_epoch < 1 or samples_per_epoch % cell_count:
        raise ValueError("samples_per_epoch must be positive and divisible by six")
    if world_size < 1 or samples_per_epoch % world_size:
        raise ValueError("samples_per_epoch must be divisible by world_size")
    pools: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pools[(str(row.get("route_type")), str(row.get("dataset")))].append(index)
    expected = {(route_type, dataset) for route_type in route_types for dataset in datasets}
    if set(pools) != expected or any(not pools[cell] for cell in expected):
        raise ValueError("balanced sampling requires every W2C/C2C × dataset pool")
    per_cell = samples_per_epoch // cell_count
    selected: list[int] = []
    for route_type, dataset in sorted(expected):
        pool = sorted(pools[(route_type, dataset)], key=lambda index: str(rows[index]["uid"]))
        rng = random.Random(_seed_value(seed, epoch, route_type, dataset))
        current: list[int] = []
        while len(current) < per_cell:
            cycle = list(pool)
            rng.shuffle(cycle)
            current.extend(cycle)
        selected.extend(current[:per_cell])
    random.Random(_seed_value(seed, epoch, "global")).shuffle(selected)
    return selected


def guaranteed_boundary_epoch_schedule(
    rows: Sequence[dict[str, Any]],
    *,
    samples_per_epoch: int,
    seed: int,
    epochs: int,
    world_size: int,
) -> list[list[dict[str, Any]]]:
    """Preserve balanced sampling while marking one exact boundary visit per W2C.

    The ordinary balanced sampler remains the source of every visit. If its
    finite schedule happens to omit a W2C row, one repeated W2C visit from the
    same dataset cell is replaced deterministically. Exactly the first scheduled
    visit for each W2C UID is then marked for mandatory-boundary supervision;
    every later visit keeps ordinary valid-route sampling.
    """

    if epochs < 1:
        raise ValueError("epochs must be positive")
    schedule = [
        [
            {"row_index": int(index), "mandatory_boundary": False}
            for index in balanced_epoch_indices(
                rows,
                samples_per_epoch=samples_per_epoch,
                seed=seed,
                epoch=epoch,
                world_size=world_size,
            )
        ]
        for epoch in range(1, epochs + 1)
    ]
    w2c_by_dataset: dict[str, list[int]] = defaultdict(list)
    visit_positions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("route_type") == "W2C":
            w2c_by_dataset[str(row.get("dataset"))].append(index)
    for epoch_index, visits in enumerate(schedule):
        for position, visit in enumerate(visits):
            row = rows[int(visit["row_index"])]
            if row.get("route_type") == "W2C":
                visit_positions[str(row["uid"])].append((epoch_index, position))

    for dataset, pool in sorted(w2c_by_dataset.items()):
        missing = sorted(
            (index for index in pool if not visit_positions[str(rows[index]["uid"])]),
            key=lambda index: _seed_value(seed, "boundary-missing", rows[index]["uid"]),
        )
        for missing_index in missing:
            candidates = []
            for uid, positions in visit_positions.items():
                if len(positions) <= 1:
                    continue
                candidate_row = rows[int(schedule[positions[-1][0]][positions[-1][1]]["row_index"])]
                if candidate_row.get("route_type") == "W2C" and str(
                    candidate_row.get("dataset")
                ) == dataset:
                    candidates.append((positions[-1], uid))
            if not candidates:
                raise ValueError(
                    f"schedule has insufficient W2C visits to cover dataset {dataset}"
                )
            (epoch_index, position), replaced_uid = max(candidates)
            visit_positions[replaced_uid].remove((epoch_index, position))
            schedule[epoch_index][position]["row_index"] = int(missing_index)
            visit_positions[str(rows[missing_index]["uid"])].append(
                (epoch_index, position)
            )

    marked: set[str] = set()
    for visits in schedule:
        for visit in visits:
            row = rows[int(visit["row_index"])]
            uid = str(row["uid"])
            if row.get("route_type") == "W2C" and uid not in marked:
                visit["mandatory_boundary"] = True
                marked.add(uid)
    expected = {
        str(row["uid"]) for row in rows if row.get("route_type") == "W2C"
    }
    if marked != expected:
        raise RuntimeError("mandatory-boundary schedule did not cover every W2C UID")
    return schedule


def deterministic_route_index(*, uid: str, route_count: int, seed: int, epoch: int) -> int:
    if route_count < 1:
        raise ValueError("route_count must be positive")
    return _seed_value(seed, epoch, uid, "teacher_route") % route_count
