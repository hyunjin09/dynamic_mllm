"""Pure manifest and accounting helpers for the frozen program beam audit."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence


ACTION_NAMES = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")


def expected_beam_cardinality(
    trigger_layer: int,
    *,
    total_layers: int = 28,
    beam_width: int = 8,
    action_count: int = 4,
) -> int:
    """Return beam width capped by the number of possible complete suffixes."""
    suffix_length = int(total_layers) - int(trigger_layer)
    if suffix_length < 1:
        raise ValueError("trigger layer leaves no suffix decision")
    if beam_width < 1 or action_count < 1:
        raise ValueError("beam width and action count must be positive")
    return min(int(beam_width), int(action_count) ** suffix_length)


def _program_id(uid: str, trigger_layer: int, actions: Sequence[str]) -> str:
    payload = {
        "uid": str(uid),
        "trigger_layer": int(trigger_layer),
        "actions": list(actions),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate_fields(
    row: Mapping[str, Any], beam: Mapping[str, Any], rank: int
) -> dict[str, Any]:
    uid = str(row["uid"])
    trigger = int(row["trigger_layer"])
    actions = [str(action) for action in beam["actions"]]
    if len(actions) != 28 - trigger:
        raise ValueError(f"suffix length differs for {uid} rank {rank}")
    if any(action not in ACTION_NAMES for action in actions):
        raise ValueError(f"unsupported action for {uid} rank {rank}")
    indices = [int(value) for value in beam["action_indices"]]
    expected_indices = [ACTION_NAMES.index(action) for action in actions]
    if indices != expected_indices:
        raise ValueError(f"action indices differ for {uid} rank {rank}")
    non_full_positions = [index for index, action in enumerate(actions) if action != "FULL"]
    counts = Counter(actions)
    return {
        "uid": uid,
        "sample_id": str(row["sample_id"]),
        "benchmark": str(row["benchmark"]),
        "benchmark_family": str(row["benchmark_family"]),
        "image_group_id": str(row["image_group_id"]),
        "dense_correct": bool(row["dense_correct"]),
        "trigger_layer": trigger,
        "rank": int(rank),
        "program_id": _program_id(uid, trigger, actions),
        "action_indices": indices,
        "actions": actions,
        "sequence_score": float(beam["score"]),
        "is_all_full": not non_full_positions,
        "non_full_count": len(non_full_positions),
        "non_full_fraction": len(non_full_positions) / len(actions),
        "first_non_full_layer": (
            trigger + non_full_positions[0] if non_full_positions else None
        ),
        "first_non_full_delay": non_full_positions[0] if non_full_positions else None,
        "read_only_count": counts["READ_ONLY"],
        "write_only_count": counts["WRITE_ONLY"],
        "ignore_count": counts["IGNORE"],
    }


def freeze_beam_manifests(
    rows: Sequence[Mapping[str, Any]],
    *,
    beam_width: int = 8,
    total_layers: int = 28,
    require_complete_cardinality: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Validate frozen beam rows and create ranked and deduplicated manifests."""
    ranked: list[dict[str, Any]] = []
    unique: list[dict[str, Any]] = []
    duplicate_count = 0
    late_count = 0
    seen_uids: set[str] = set()
    for row in sorted(rows, key=lambda item: str(item["uid"])):
        uid = str(row["uid"])
        if uid in seen_uids:
            raise ValueError(f"duplicate triggered UID: {uid}")
        seen_uids.add(uid)
        if not bool(row.get("triggered")):
            raise ValueError(f"untriggered row in beam manifest: {uid}")
        trigger = int(row["trigger_layer"])
        expected = expected_beam_cardinality(
            trigger, total_layers=total_layers, beam_width=beam_width
        )
        beams = list(row["beam_rows"])
        if require_complete_cardinality and len(beams) != expected:
            raise ValueError(
                f"beam cardinality differs for {uid}: {len(beams)} != {expected}"
            )
        if not beams:
            raise ValueError(f"empty beam for {uid}")
        if trigger == total_layers - 1:
            late_count += 1
        scores = [float(beam["score"]) for beam in beams]
        if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
            raise ValueError(f"beam score order differs for {uid}")
        if list(beams[0]["actions"]) != list(row["program_suffix_actions"]):
            raise ValueError(f"stored top-1 program differs for {uid}")
        by_program: dict[str, dict[str, Any]] = {}
        for rank, beam in enumerate(beams, 1):
            candidate = _candidate_fields(row, beam, rank)
            candidate["available_beam_size"] = len(beams)
            ranked.append(candidate)
            program_id = candidate["program_id"]
            if program_id in by_program:
                by_program[program_id]["source_ranks"].append(rank)
                duplicate_count += 1
            else:
                execution = dict(candidate)
                execution.pop("rank")
                execution["source_ranks"] = [rank]
                by_program[program_id] = execution
                unique.append(execution)
    return ranked, unique, {
        "triggered_samples": len(seen_uids),
        "ranked_entries": len(ranked),
        "unique_programs": len(unique),
        "duplicate_ranked_entries": duplicate_count,
        "late_trigger_exhaustive_four": late_count,
    }


def compare_top1_replay(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    score_abs_tolerance: float,
) -> dict[str, bool]:
    """Compare live top-1 replay to Phase 74 without brittle float equality."""
    expected_beam = expected["beam_rows"][0]
    actual_beam = actual["beam_rows"][0]
    return {
        "trigger_layer": actual["trigger_layer"] == expected["trigger_layer"],
        "suffix_actions": list(actual["program_suffix_actions"])
        == list(expected["program_suffix_actions"]),
        "beam_top1_actions": list(actual_beam["actions"])
        == list(expected_beam["actions"]),
        "beam_top1_score": math.isclose(
            float(actual_beam["score"]),
            float(expected_beam["score"]),
            rel_tol=0.0,
            abs_tol=float(score_abs_tolerance),
        ),
        "generated_token_ids": list(actual["program_generated_token_ids"])
        == list(expected["program_generated_token_ids"]),
        "generated_answer": str(actual["program_generated_answer"])
        == str(expected["program_generated_answer"]),
        "evaluator_score": float(actual["program_score"])
        == float(expected["program_score"]),
        "correctness": bool(actual["program_correct"])
        == bool(expected["program_correct"]),
    }


def mean_pairwise_hamming(programs: Sequence[Sequence[str]]) -> float:
    """Mean action-position Hamming distance over all unordered pairs."""
    programs = [list(program) for program in programs]
    if len(programs) < 2:
        return 0.0
    lengths = {len(program) for program in programs}
    if len(lengths) != 1:
        raise ValueError("Hamming distance requires equal-length programs")
    distances = []
    for left in range(len(programs)):
        for right in range(left + 1, len(programs)):
            distances.append(
                sum(a != b for a, b in zip(programs[left], programs[right]))
            )
    return sum(distances) / len(distances)


def summarize_oracle_sample(
    source: Mapping[str, Any],
    ranked_rows: Sequence[Mapping[str, Any]],
    result_by_program: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the fixed per-sample ranking/generation decomposition."""
    uid = str(source["uid"])
    ranked = sorted(
        (row for row in ranked_rows if str(row["uid"]) == uid),
        key=lambda row: int(row["rank"]),
    )
    if not ranked:
        raise ValueError(f"no ranked programs for {uid}")
    missing = [
        row["program_id"] for row in ranked if row["program_id"] not in result_by_program
    ]
    if missing:
        raise ValueError(f"missing executed programs for {uid}: {missing[:3]}")
    correct_ranks = [
        int(row["rank"])
        for row in ranked
        if bool(result_by_program[row["program_id"]]["correct"])
    ]
    first_correct = min(correct_ranks) if correct_ranks else None
    top1_correct = 1 in correct_ranks
    dense_correct = bool(source["dense_correct"])
    all_full = next((row for row in ranked if bool(row["is_all_full"])), None)
    all_full_rank = int(all_full["rank"]) if all_full is not None else None
    all_full_score = float(all_full["sequence_score"]) if all_full is not None else None
    unique_programs: dict[str, list[str]] = {}
    for row in ranked:
        unique_programs.setdefault(str(row["program_id"]), list(row["actions"]))
    distinct_non_full_positions = {
        index
        for actions in unique_programs.values()
        for index, action in enumerate(actions)
        if action != "FULL"
    }
    output = {
        "uid": uid,
        "benchmark": str(source["benchmark"]),
        "benchmark_family": str(source["benchmark_family"]),
        "dense_correct": dense_correct,
        "trigger_layer": int(source["trigger_layer"]),
        "available_beam_size": len(ranked),
        "unique_programs": len(unique_programs),
        "mean_pairwise_hamming": mean_pairwise_hamming(list(unique_programs.values())),
        "distinct_first_actions": len({actions[0] for actions in unique_programs.values()}),
        "distinct_non_full_positions": len(distinct_non_full_positions),
        "first_correct_rank": first_correct,
        "top1_correct": top1_correct,
        "all_full_in_beam": all_full is not None,
        "all_full_rank": all_full_rank,
        "all_full_score": all_full_score,
        "rescue_at_1": (first_correct is not None and first_correct <= 1),
        "rescue_at_2": (first_correct is not None and first_correct <= 2),
        "rescue_at_4": (first_correct is not None and first_correct <= 4),
        "rescue_at_8": (first_correct is not None and first_correct <= 8),
        "w_failure_class": None,
        "c_failure_class": None,
        "c_all_full_class": None,
    }
    if dense_correct:
        if top1_correct:
            output["c_failure_class"] = "TOP1_SUCCESS_C"
        elif correct_ranks:
            output["c_failure_class"] = "RANKING_FAILURE_C"
        else:
            output["c_failure_class"] = "GENERATION_FAILURE_C"
        if not top1_correct:
            output["c_all_full_class"] = (
                "C1_ALL_FULL_RANKED_BELOW_WRONG_TOP1"
                if all_full is not None
                else "C2_ALL_FULL_ABSENT"
            )
    else:
        if top1_correct:
            output["w_failure_class"] = "TOP1_SUCCESS_W"
        elif correct_ranks:
            output["w_failure_class"] = "RANKING_FAILURE_W"
        else:
            output["w_failure_class"] = "GENERATION_FAILURE_W"
    return output

