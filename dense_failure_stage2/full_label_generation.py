"""Pure integrity and aggregation contracts for full corrective-label generation."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence


FIXABILITY_CLASSES = ("SINGLE_FIXABLE", "MCTS_ONLY_FIXABLE", "UNRESOLVED")
ROUTE_SOURCES = ("preservation_full", "single", "mcts")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    """Verify every hash-bound artifact and reject incomplete source runs."""

    if not bool(manifest.get("passed")):
        raise RuntimeError("source artifact manifest is not passing")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("source artifact manifest has no files")
    for relative, expected in files.items():
        path = root / str(relative)
        if not path.is_file():
            raise RuntimeError(f"source artifact is missing: {relative}")
        actual = file_sha256(path)
        if actual != str(expected):
            raise RuntimeError(
                f"source artifact hash mismatch: {relative}: {actual} != {expected}"
            )


def assign_workers(
    rows: Sequence[Mapping[str, Any]], *, world_size: int
) -> list[dict[str, Any]]:
    """Greedily balance frozen work by estimated terminal evaluations."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    seen: set[str] = set()
    for row in rows:
        uid = str(row["uid"])
        if uid in seen:
            raise ValueError(f"duplicate UID in assignment source: {uid}")
        seen.add(uid)
        if int(row["estimated_terminal_evaluations"]) < 0:
            raise ValueError("estimated terminal evaluations must be nonnegative")
    loads = [0] * world_size
    assigned: list[dict[str, Any]] = []
    ordered = sorted(
        rows,
        key=lambda row: (-int(row["estimated_terminal_evaluations"]), str(row["uid"])),
    )
    for source in ordered:
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        cost = int(source["estimated_terminal_evaluations"])
        loads[rank] += cost
        assigned.append({**source, "worker_rank": rank})
    return sorted(assigned, key=lambda row: str(row["uid"]))


def _first_success_by_route(search_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    first: dict[str, int] = {}
    for row in search_rows:
        if int(row.get("reward", 0)) != 1:
            continue
        key = str(row["route_key"])
        iteration = int(row["iteration"])
        first[key] = min(first.get(key, iteration), iteration)
    return first


def cap_pilot_record(
    sample: Mapping[str, Any], *, cap: int, retain_successes: int
) -> dict[str, Any]:
    """Reinterpret a Phase-55 record under a smaller frozen MCTS cap.

    Exact route tensors are reusable only when their route first succeeded at or
    before ``cap``. Single-search results are cap-independent.
    """

    if cap < 1 or retain_successes < 1:
        raise ValueError("cap and retention count must be positive")
    single_routes = [dict(row) for row in sample.get("successful_single_routes", [])]
    if single_routes:
        if sample.get("mcts_result") is not None:
            raise ValueError("single-fixable pilot record unexpectedly ran MCTS")
        return {
            "uid": str(sample["uid"]),
            "outcome_class": "SINGLE_FIXABLE",
            "first_success_iteration": None,
            "mcts_iterations": 0,
            "mcts_unique_terminal_routes": 0,
            "retained_successful_routes": single_routes,
            "eligible_route_keys": sorted(str(row["route_key"]) for row in single_routes),
            "excluded_post_cap_route_keys": [],
        }

    mcts = sample.get("mcts_result")
    if mcts is None:
        return {
            "uid": str(sample["uid"]),
            "outcome_class": "UNRESOLVED",
            "first_success_iteration": None,
            "mcts_iterations": 0,
            "mcts_unique_terminal_routes": 0,
            "retained_successful_routes": [],
            "eligible_route_keys": [],
            "excluded_post_cap_route_keys": [],
        }

    all_search_rows = [dict(row) for row in mcts.get("search_rows", [])]
    capped_rows = [row for row in all_search_rows if int(row["iteration"]) <= cap]
    first_by_route = _first_success_by_route(capped_rows)
    first_success = min(first_by_route.values(), default=None)
    successful_search_rows: dict[str, dict[str, Any]] = {}
    for row in capped_rows:
        if int(row.get("reward", 0)) != 1:
            continue
        successful_search_rows.setdefault(str(row["route_key"]), row)
    canonical_keys = [
        key
        for key, _row in sorted(
            successful_search_rows.items(),
            key=lambda item: (int(item[1]["non_full_count"]), item[0]),
        )[:retain_successes]
    ]
    retained_lookup = {
        str(row["route_key"]): dict(row)
        for row in sample.get("retained_successful_routes", [])
    }
    missing = sorted(set(canonical_keys) - set(retained_lookup))
    if missing:
        raise ValueError(
            "cap-eligible successful route lacks a replay-valid pilot state: "
            + ", ".join(missing[:3])
        )
    eligible = []
    for key in canonical_keys:
        discovery = first_by_route[key]
        route = {
            **retained_lookup[key],
            "discovery_iteration": discovery,
        }
        eligible.append(route)
    eligible.sort(key=lambda row: (int(row["non_full_count"]), str(row["route_key"])))
    retained = eligible
    all_success = _first_success_by_route(all_search_rows)
    excluded = sorted(set(all_success) - {str(row["route_key"]) for row in retained})
    outcome = "MCTS_ONLY_FIXABLE" if first_success is not None else "UNRESOLVED"
    return {
        "uid": str(sample["uid"]),
        "outcome_class": outcome,
        "first_success_iteration": first_success,
        "mcts_iterations": len(capped_rows),
        "mcts_unique_terminal_routes": len({str(row["route_key"]) for row in capped_rows}),
        "retained_successful_routes": retained,
        "eligible_route_keys": [str(row["route_key"]) for row in retained],
        "excluded_post_cap_route_keys": excluded,
    }


def choose_preferred_route(
    routes: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Choose the diagnostic route without discarding any alternatives."""

    if not routes:
        return None
    if any(not bool(row.get("correct", row.get("final_lmms_correct", False))) for row in routes):
        raise ValueError("preferred-route candidates must all be correct")
    return dict(
        min(
            routes,
            key=lambda row: (
                int(row["non_full_count"]),
                int(row.get("discovery_iteration", row.get("discovery_order", 10**9))),
                str(row["route_key"]),
            ),
        )
    )


def validate_population_completion(
    expected_wrong_uids: Sequence[str],
    expected_correct_uids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require exactly one passing result for every frozen W and C UID."""

    expected = Counter(map(str, (*expected_wrong_uids, *expected_correct_uids)))
    actual = Counter(str(row["uid"]) for row in rows)
    if expected != actual:
        raise ValueError(
            f"population coverage mismatch: missing={list((expected - actual).elements())[:5]} "
            f"extra={list((actual - expected).elements())[:5]}"
        )
    wrong = set(map(str, expected_wrong_uids))
    correct = set(map(str, expected_correct_uids))
    if wrong & correct:
        raise ValueError("frozen wrong/correct populations overlap")
    for row in rows:
        uid = str(row["uid"])
        expected_type = (
            "triggered_wrong" if uid in wrong else "triggered_correct_preservation"
        )
        if row.get("task_type") != expected_type:
            raise ValueError(f"result task type mismatch: {uid}")
        if not bool(row.get("passed")):
            raise ValueError(f"failed record in completed population: {uid}")


def build_corpus_rows(
    routes: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Partition replay-valid route supervision without collapsing provenance."""

    mapping = {"preservation_full": "A", "single": "B", "mcts": "C"}
    corpora: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for source in routes:
        route_source = str(source.get("route_source"))
        if route_source not in mapping:
            raise ValueError(f"unsupported route source: {route_source}")
        if not bool(source.get("final_lmms_correct")) or not bool(
            source.get("replay_token_parity")
        ):
            raise ValueError("corpus route is not a correct replay")
        corpora[mapping[route_source]].append(dict(source))
    for rows in corpora.values():
        rows.sort(key=lambda row: (str(row["uid"]), str(row.get("route_id", ""))))
    return corpora
