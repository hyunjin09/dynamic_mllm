"""Pure contracts for trigger-conditioned four-action corrective search."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import math
from typing import Any, Callable, Mapping, Sequence


ACTIONS = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
NON_FULL_ACTIONS = ACTIONS[1:]
DEPTH_BINS = ("L0", "L1-8", "L9-18", "L19-27")


def trigger_depth_bin(layer: int) -> str:
    value = int(layer)
    if value == 0:
        return "L0"
    if 1 <= value <= 8:
        return "L1-8"
    if 9 <= value <= 18:
        return "L9-18"
    if 19 <= value <= 27:
        return "L19-27"
    raise ValueError("trigger layer must lie in 0..27")


def _stable_digest(*parts: object) -> str:
    return sha256(":".join(str(part) for part in parts).encode()).hexdigest()


def select_pilot_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    allocation: Mapping[str, Mapping[str, int]],
    seed: int,
) -> list[dict[str, Any]]:
    """Select the prospectively allocated pilot with unique UIDs/image groups."""

    seen_uids: set[str] = set()
    for row in rows:
        uid = str(row["uid"])
        if uid in seen_uids:
            raise ValueError(f"duplicate candidate UID: {uid}")
        seen_uids.add(uid)
        if row.get("split") != "train" or not bool(row.get("dense_wrong")):
            raise ValueError("pilot candidates must be triggered train Dense-W rows")

    selected: list[dict[str, Any]] = []
    selected_groups: set[str] = set()
    for dataset, by_depth in allocation.items():
        for depth_bin, count in by_depth.items():
            if depth_bin not in DEPTH_BINS or int(count) < 0:
                raise ValueError("invalid pilot allocation")
            cell = [
                dict(row)
                for row in rows
                if str(row["dataset"]) == dataset
                and trigger_depth_bin(int(row["first_trigger_layer"])) == depth_bin
            ]
            cell.sort(key=lambda row: (_stable_digest(seed, row["uid"]), str(row["uid"])))
            chosen = []
            for row in cell:
                group = str(row["group_id"])
                if group in selected_groups:
                    continue
                chosen.append({**row, "trigger_depth_bin": depth_bin})
                selected_groups.add(group)
                if len(chosen) == int(count):
                    break
            if len(chosen) != int(count):
                raise ValueError(
                    f"insufficient distinct image groups for {dataset}/{depth_bin}: "
                    f"{len(chosen)} < {count}"
                )
            selected.extend(chosen)
    selected.sort(key=lambda row: str(row["uid"]))
    if len({str(row["uid"]) for row in selected}) != len(selected):
        raise RuntimeError("selected pilot contains duplicate UIDs")
    return selected


def all_single_routes(*, start_layer: int, layer_count: int = 28) -> list[dict[str, Any]]:
    """Enumerate every one-layer non-FULL suffix intervention."""

    start = int(start_layer)
    if not 0 <= start < layer_count:
        raise ValueError("start_layer must lie inside the decoder")
    routes = []
    for layer in range(start, layer_count):
        for action in NON_FULL_ACTIONS:
            actions = ["FULL"] * layer_count
            actions[layer] = action
            routes.append(
                {
                    "candidate_index": len(routes),
                    "search_stage": "single",
                    "actions": actions,
                    "route_key": "|".join(actions),
                    "changed_layers": [layer],
                    "changed_actions": [action],
                }
            )
    return routes


def _ordered_actions(seed: int, uid: str, prefix: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            ACTIONS,
            key=lambda action: (_stable_digest(seed, uid, "expand", *prefix, action), action),
        )
    )


def _complete_cardinality_rollout(
    *,
    uid: str,
    seed: int,
    iteration: int,
    prefix: tuple[str, ...],
    horizon: int,
    requested_non_full: int,
) -> tuple[tuple[str, ...], int]:
    """Complete a tree prefix to a deterministic fixed-cardinality suffix."""

    if len(prefix) > horizon:
        raise ValueError("tree prefix exceeds the suffix horizon")
    fixed_non_full = sum(action != "FULL" for action in prefix)
    remaining = horizon - len(prefix)
    target = min(int(requested_non_full), horizon)
    target = max(fixed_non_full, min(target, fixed_non_full + remaining))
    additions = target - fixed_non_full
    positions = list(range(len(prefix), horizon))
    positions.sort(
        key=lambda position: (
            _stable_digest(seed, uid, iteration, "position", *prefix, position),
            position,
        )
    )
    chosen = set(positions[:additions])
    result = list(prefix) + ["FULL"] * remaining
    for position in chosen:
        result[position] = min(
            NON_FULL_ACTIONS,
            key=lambda action: (
                _stable_digest(seed, uid, iteration, "action", position, action, *prefix),
                action,
            ),
        )
    return tuple(result), target


def run_sequential_mcts(
    *,
    uid: str,
    start_layer: int,
    seed: int,
    max_iterations: int,
    extra_iterations_after_success: int,
    evaluate: Callable[[Sequence[str]], bool],
    exploration_constant: float = math.sqrt(2.0),
    rollout_cardinalities: Sequence[int] = (2, 3, 4),
    retain_successes: int = 8,
    layer_count: int = 28,
) -> dict[str, Any]:
    """Run deterministic prefix-tree UCB1 search with binary terminal reward.

    Tree nodes represent actual ordered action prefixes. Every terminal route is
    evaluated at most once; repeated terminal routes remain valid MCTS
    iterations and reuse the previously observed binary reward.
    """

    start = int(start_layer)
    horizon = layer_count - start
    if not uid or not 0 <= start < layer_count:
        raise ValueError("invalid MCTS identity/start layer")
    if max_iterations < 1 or extra_iterations_after_success < 0:
        raise ValueError("invalid MCTS budget")
    cardinalities = tuple(int(value) for value in rollout_cardinalities)
    if not cardinalities or any(value < 2 for value in cardinalities):
        raise ValueError("rollout cardinalities must contain values >=2")

    visits: Counter[tuple[str, ...]] = Counter()
    value_sum: Counter[tuple[str, ...]] = Counter()
    expanded_children: dict[tuple[str, ...], set[str]] = {}
    terminal_cache: dict[tuple[str, ...], int] = {}
    successful: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    first_success: int | None = None

    for iteration in range(1, max_iterations + 1):
        prefix: tuple[str, ...] = ()
        path = [prefix]
        while len(prefix) < horizon:
            used = expanded_children.setdefault(prefix, set())
            ordered = _ordered_actions(seed, uid, prefix)
            untried = [action for action in ordered if action not in used]
            if untried:
                action = untried[0]
                used.add(action)
                prefix = (*prefix, action)
                path.append(prefix)
                break
            parent_visits = max(visits[prefix], 1)

            def ucb(action: str) -> tuple[float, str]:
                child = (*prefix, action)
                child_visits = visits[child]
                if child_visits == 0:
                    return (float("inf"), _stable_digest(seed, uid, "ucb", *child))
                mean = value_sum[child] / child_visits
                bonus = exploration_constant * math.sqrt(math.log(parent_visits) / child_visits)
                return (mean + bonus, _stable_digest(seed, uid, "ucb", *child))

            action = max(ordered, key=ucb)
            prefix = (*prefix, action)
            path.append(prefix)

        requested = cardinalities[(iteration - 1) % len(cardinalities)]
        suffix, target = _complete_cardinality_rollout(
            uid=uid,
            seed=seed,
            iteration=iteration,
            prefix=prefix,
            horizon=horizon,
            requested_non_full=requested,
        )
        full_actions = ("FULL",) * start + suffix
        cache_hit = suffix in terminal_cache
        if cache_hit:
            reward = terminal_cache[suffix]
        else:
            observed = evaluate(full_actions)
            if not isinstance(observed, bool):
                raise TypeError("MCTS evaluator must return a binary bool reward")
            reward = int(observed)
            terminal_cache[suffix] = reward

        for node in path:
            visits[node] += 1
            value_sum[node] += reward
        route_key = "|".join(full_actions)
        row = {
            "iteration": iteration,
            "actions": list(full_actions),
            "route_key": route_key,
            "reward": reward,
            "terminal_cache_hit": cache_hit,
            "tree_prefix_length": len(prefix),
            "rollout_requested_non_full": requested,
            "rollout_target_non_full": target,
            "non_full_count": sum(action != "FULL" for action in full_actions),
        }
        rows.append(row)
        if reward:
            successful.setdefault(route_key, row)
            if first_success is None:
                first_success = iteration
        if first_success is not None and iteration >= first_success + extra_iterations_after_success:
            break

    retained = sorted(
        successful.values(),
        key=lambda row: (int(row["non_full_count"]), str(row["route_key"])),
    )[:retain_successes]
    return {
        "iterations": len(rows),
        "first_success_iteration": first_success,
        "unique_terminal_routes": len(terminal_cache),
        "search_rows": rows,
        "successful_routes": retained,
        "fixable_at_100": first_success is not None and first_success <= 100,
        "fixable_at_200": first_success is not None and first_success <= 200,
        "fixable_at_300": first_success is not None and first_success <= 300,
    }


def classify_outcome(
    *,
    single_successes: Sequence[Mapping[str, Any]],
    mcts_result: Mapping[str, Any] | None,
) -> str:
    if single_successes:
        return "SINGLE-FIXABLE"
    if mcts_result is not None and mcts_result.get("first_success_iteration") is not None:
        return "MCTS-FIXABLE"
    return "UNRESOLVED"


def pilot_cell_weights(population: Mapping[tuple[str, str], int]) -> dict[tuple[str, str], float]:
    total = sum(int(value) for value in population.values())
    if total < 1 or any(int(value) < 0 for value in population.values()):
        raise ValueError("population cell counts must be nonnegative and nonempty")
    return {cell: int(value) / total for cell, value in population.items()}


def validate_complete_results(
    expected_uids: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    expected = Counter(str(uid) for uid in expected_uids)
    actual = Counter(str(row["uid"]) for row in rows)
    if expected != actual:
        raise ValueError(
            f"result coverage mismatch: missing={list((expected - actual).elements())[:5]} "
            f"extra={list((actual - expected).elements())[:5]}"
        )
    if any(not bool(row.get("passed")) for row in rows):
        raise ValueError("result set contains failed records")
