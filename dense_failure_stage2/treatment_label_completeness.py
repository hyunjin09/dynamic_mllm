"""Pure contracts for the bounded Stage-2 treatment-label completeness audit."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence


ACTIONS = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
AUDITED_KEEP = "AUDITED_KEEP"
AUDITED_INTERVENE = "AUDITED_INTERVENE"
AUDITED_MIXED = "AUDITED_MIXED"
SAMPLING_DIMENSIONS = (
    "dataset",
    "source_regime",
    "layer_bin",
    "route_source_signature",
)


def _digest(seed: int, *parts: object) -> str:
    return sha256(":".join((str(seed), *(str(part) for part in parts))).encode()).hexdigest()


def _hamilton_targets(rows: Sequence[Mapping[str, Any]], field: str, total: int) -> dict[str, int]:
    counts = Counter(str(row[field]) for row in rows)
    if not counts:
        raise ValueError(f"sampling dimension is empty: {field}")
    raw = {key: total * value / len(rows) for key, value in counts.items()}
    targets = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(targets.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - targets[key]), key))
    for key in order[:remaining]:
        targets[key] += 1
    return targets


def deterministic_state_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Mapping[str, int],
    seed: int,
    uid_cap: int,
) -> list[dict[str, Any]]:
    """Select fixed label quotas while matching four outcome-blind marginals.

    Label strata are processed from least to most UID capacity.  Within each
    label, a deterministic greedy rule fills Hamilton-apportioned marginal
    targets for dataset, source, layer bin, and route-source signature.  No
    model score, correctness outcome, or feature value participates.
    """

    if uid_cap < 1 or not targets or any(int(value) < 0 for value in targets.values()):
        raise ValueError("invalid sampling targets/UID cap")
    state_ids = [str(row["state_id"]) for row in rows]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("candidate state IDs are duplicated")
    supported = set(targets)
    candidates = [dict(row) for row in rows if str(row["state_label"]) in supported]
    by_label = {
        label: [row for row in candidates if str(row["state_label"]) == label]
        for label in supported
    }
    for label, target in targets.items():
        if len(by_label[label]) < int(target):
            raise ValueError(f"insufficient states for label {label}")
        if len({str(row["uid"]) for row in by_label[label]}) * uid_cap < int(target):
            raise ValueError(f"uid cap makes target impossible for label {label}")
        for row in by_label[label]:
            missing = [field for field in SAMPLING_DIMENSIONS if field not in row]
            if missing:
                raise ValueError(f"sampling metadata missing for {row['state_id']}: {missing}")

    label_order = sorted(
        targets,
        key=lambda label: (
            len({str(row["uid"]) for row in by_label[label]}) * uid_cap / max(int(targets[label]), 1),
            str(label),
        ),
    )
    uid_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for label in label_order:
        target = int(targets[label])
        pool = by_label[label]
        marginal_targets = {
            field: _hamilton_targets(pool, field, target) for field in SAMPLING_DIMENSIONS
        }
        marginal_counts = {field: Counter() for field in SAMPLING_DIMENSIONS}
        chosen_ids: set[str] = set()
        while len(chosen_ids) < target:
            eligible = [
                row
                for row in pool
                if str(row["state_id"]) not in chosen_ids
                and uid_counts[str(row["uid"])] < uid_cap
            ]
            if not eligible:
                raise ValueError(f"uid cap prevents completing target for label {label}")

            def priority(row: Mapping[str, Any]) -> tuple[float, int, str, str]:
                benefit = 0.0
                exact_deficits = 0
                for field in SAMPLING_DIMENSIONS:
                    key = str(row[field])
                    deficit = marginal_targets[field][key] - marginal_counts[field][key]
                    if deficit > 0:
                        benefit += deficit / max(marginal_targets[field][key], 1)
                        exact_deficits += 1
                return (
                    -benefit,
                    -exact_deficits,
                    _digest(seed, label, row["state_id"]),
                    str(row["state_id"]),
                )

            chosen = min(eligible, key=priority)
            chosen_ids.add(str(chosen["state_id"]))
            uid_counts[str(chosen["uid"])] += 1
            for field in SAMPLING_DIMENSIONS:
                marginal_counts[field][str(chosen[field])] += 1
            selected.append(chosen)

    selected.sort(key=lambda row: (str(row["state_label"]), str(row["state_id"])))
    observed = Counter(str(row["state_label"]) for row in selected)
    if observed != Counter({str(key): int(value) for key, value in targets.items()}):
        raise RuntimeError(f"selected label counts differ from targets: {observed}")
    if uid_counts and max(uid_counts.values()) > uid_cap:
        raise RuntimeError("selected manifest exceeds UID cap")
    return selected


def unobserved_actions(observed: Sequence[str]) -> tuple[str, ...]:
    normalized = {str(action) for action in observed}
    unsupported = normalized.difference(ACTIONS)
    if unsupported:
        raise ValueError(f"unsupported actions: {sorted(unsupported)}")
    if not normalized:
        raise ValueError("observed action set cannot be empty")
    return tuple(action for action in ACTIONS if action not in normalized)


def classify_audited_actions(actions: Sequence[str]) -> str:
    normalized = {str(action) for action in actions}
    unsupported = normalized.difference(ACTIONS)
    if unsupported:
        raise ValueError(f"unsupported actions: {sorted(unsupported)}")
    if not normalized:
        raise ValueError("audited action set cannot be empty")
    if normalized == {"FULL"}:
        return AUDITED_KEEP
    if "FULL" not in normalized:
        return AUDITED_INTERVENE
    return AUDITED_MIXED


def validate_complete_state_audits(
    expected_state_ids: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    expected = Counter(map(str, expected_state_ids))
    actual = Counter(str(row["state_id"]) for row in rows)
    duplicates = sum(max(count - 1, 0) for count in actual.values())
    missing = sum((expected - actual).values())
    extra = sum((actual - expected).values())
    if expected != actual:
        raise ValueError(
            f"audit coverage mismatch: missing={missing} extra={extra} duplicates={duplicates}"
        )
    quarantined = sum(bool(row.get("quarantined")) for row in rows)
    failed = sum(not bool(row.get("passed")) for row in rows)
    if quarantined or failed:
        raise ValueError(f"audit contains quarantined/failed rows: {quarantined}/{failed}")
    return {
        "expected": len(expected_state_ids),
        "completed": len(rows),
        "duplicates": duplicates,
        "missing": missing,
        "quarantined": quarantined,
    }


def mcts_budget_saturation(discovery_iterations: Sequence[int]) -> dict[str, Any]:
    """Apply the frozen late-discovery rule to successful MCTS branches.

    At least ten discoveries are needed.  Saturation means no more than 10% of
    discoveries first appear in iterations 151--200, equivalently at least 90%
    of the observed MCTS discoveries have appeared by iteration 150.
    """

    values = [int(value) for value in discovery_iterations]
    if any(value < 1 or value > 200 for value in values):
        raise ValueError("MCTS discovery iterations must lie in 1..200")
    total = len(values)
    late = sum(value > 150 for value in values)
    fraction = late / total if total else 0.0
    if total < 10:
        return {
            "discoveries": total,
            "discoveries_by_150": total - late,
            "discoveries_151_200": late,
            "late_151_200_fraction": fraction,
            "saturated": None,
            "reason": "fewer_than_10_mcts_discoveries",
        }
    return {
        "discoveries": total,
        "discoveries_by_150": total - late,
        "discoveries_151_200": late,
        "late_151_200_fraction": fraction,
        "saturated": fraction <= 0.10,
        "reason": "prospective_late_discovery_rule",
    }
