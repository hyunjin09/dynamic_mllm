"""Pure selection and trigger-map helpers for the Stage-2 scale-up phase."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

from dense_failure_stage2.corrective_search import trigger_depth_bin


def image_extension_for_format(image_format: str) -> str:
    value = str(image_format).strip().lower()
    if value in {"jpeg", "jpg", "mpo"}:
        return ".jpg"
    if value == "png":
        return ".png"
    raise ValueError(f"unsupported embedded image format: {value}")


def _length_bin(text: str) -> str:
    count = len(str(text).split())
    if count <= 5:
        return "q1_5"
    if count <= 10:
        return "q6_10"
    return "q11_plus"


def question_length_stratum(question: str) -> str:
    return _length_bin(question)


def _count_bin(count: int, *, prefix: str) -> str:
    if count <= 0:
        return f"{prefix}0"
    if count <= 5:
        return f"{prefix}1_5"
    return f"{prefix}6_plus"


def chartqa_stratum(annotation_source: str, question: str) -> str:
    source = str(annotation_source).strip().lower()
    if source not in {"human", "augmented"}:
        raise ValueError(f"unsupported ChartQA annotation source: {annotation_source}")
    return f"{source}|{_length_bin(question)}"


def textvqa_stratum(question: str, ocr_tokens: Sequence[str]) -> str:
    return f"{_length_bin(question)}|{_count_bin(len(ocr_tokens), prefix='ocr')}"


def most_common_answer(answers: Sequence[str]) -> str:
    if not answers:
        raise ValueError("TextVQA answers must be nonempty")
    counts: Counter[str] = Counter()
    first: dict[str, tuple[int, str]] = {}
    for index, value in enumerate(answers):
        text = str(value).strip()
        key = text.casefold()
        counts[key] += 1
        first.setdefault(key, (index, text))
    winner = min(counts, key=lambda key: (-counts[key], first[key][0]))
    return first[winner][1]


def select_legacy_source_rows(
    portable_rows: Sequence[Mapping[str, Any]],
    current_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    current_uids = [str(row["uid"]) for row in current_candidates]
    if len(current_uids) != len(set(current_uids)):
        raise ValueError("current candidate UIDs are duplicated")
    by_uid: dict[str, Mapping[str, Any]] = {}
    for row in portable_rows:
        uid = str(row["uid"])
        if uid in by_uid:
            raise ValueError(f"portable source UID is duplicated: {uid}")
        by_uid[uid] = row
    if not set(current_uids) <= set(by_uid):
        raise ValueError("portable source/current candidate UID coverage differs")
    return [dict(by_uid[uid]) for uid in sorted(current_uids)]


def _stratum_targets(reference: Mapping[str, int], target: int) -> dict[str, int]:
    counts = {str(key): int(value) for key, value in reference.items() if int(value) > 0}
    total = sum(counts.values())
    if target < 1 or total < 1:
        raise ValueError("target and reference strata must be nonempty")
    exact = {key: target * value / total for key, value in counts.items()}
    allocated = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = target - sum(allocated.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocated[key]), key))
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def _rank(seed: int, row: Mapping[str, Any]) -> tuple[str, str]:
    key = str(row["native_row_key"])
    digest = sha256(f"stage2-scale:{seed}:{key}".encode()).hexdigest()
    return digest, key


def select_metadata_stratified(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: int,
    reference_strata: Mapping[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one row per image with outcome-blind stratum quotas and backfill."""

    if len({str(row["native_row_key"]) for row in rows}) != len(rows):
        raise ValueError("source rows contain duplicate native row keys")
    targets = _stratum_targets(reference_strata, target)
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row["source_stratum"])].append(row)
    for values in by_stratum.values():
        values.sort(key=lambda row: _rank(seed, row))

    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    shortfall = 0
    for stratum in sorted(targets):
        taken = 0
        for row in by_stratum.get(stratum, []):
            group = str(row["image_group_id"])
            if group in used_groups:
                continue
            selected.append(dict(row))
            used_groups.add(group)
            taken += 1
            if taken == targets[stratum]:
                break
        shortfall += targets[stratum] - taken

    if shortfall:
        remaining = sorted(
            (
                row
                for row in rows
                if str(row["image_group_id"]) not in used_groups
            ),
            key=lambda row: _rank(seed + 1_000_003, row),
        )
        for row in remaining:
            group = str(row["image_group_id"])
            if group in used_groups:
                continue
            selected.append(dict(row))
            used_groups.add(group)
            shortfall -= 1
            if shortfall == 0:
                break
    if shortfall:
        raise ValueError(f"insufficient unique image groups for target: missing={shortfall}")
    selected.sort(key=lambda row: _rank(seed, row))
    actual = Counter(str(row["source_stratum"]) for row in selected)
    backfilled = sum(max(0, actual[key] - targets.get(key, 0)) for key in actual)
    return selected, {
        "target": target,
        "selected": len(selected),
        "reference_strata": dict(sorted((str(key), int(value)) for key, value in reference_strata.items())),
        "target_strata": dict(sorted(targets.items())),
        "selected_strata": dict(sorted(actual.items())),
        "backfilled": backfilled,
        "unique_image_groups": len(used_groups),
        "seed": int(seed),
    }


def validate_frozen_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_counts: Mapping[str, int],
    legacy_uids: set[str],
    legacy_groups: set[str],
) -> dict[str, Any]:
    if Counter(str(row["dataset"]) for row in rows) != Counter(
        {str(key): int(value) for key, value in expected_counts.items()}
    ):
        raise ValueError("candidate dataset counts differ from the frozen quotas")
    uids = [str(row["uid"]) for row in rows]
    groups = [str(row["image_group_id"]) for row in rows]
    native_keys = [str(row["native_row_key"]) for row in rows]
    if len(uids) != len(set(uids)):
        raise ValueError("duplicate UID in frozen candidates")
    if len(native_keys) != len(set(native_keys)):
        raise ValueError("duplicate native row key in frozen candidates")
    if len(groups) != len(set(groups)):
        raise ValueError("duplicate image group in frozen candidates")
    uid_overlap = set(uids) & set(legacy_uids)
    if uid_overlap:
        raise ValueError(f"legacy UID overlap: {sorted(uid_overlap)[:3]}")
    group_overlap = set(groups) & set(legacy_groups)
    if group_overlap:
        raise ValueError(f"legacy image-group overlap: {sorted(group_overlap)[:3]}")
    return {
        "records": len(rows),
        "dataset_counts": dict(Counter(str(row["dataset"]) for row in rows)),
        "unique_uids": len(set(uids)),
        "unique_native_row_keys": len(set(native_keys)),
        "unique_image_groups": len(set(groups)),
        "legacy_uid_overlap": 0,
        "legacy_image_group_overlap": 0,
        "passed": True,
    }


def build_new_trigger_map(
    candidates: Sequence[Mapping[str, Any]],
    dense_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the frozen strict global threshold to new dense feature scores."""

    if not math.isfinite(float(threshold)):
        raise ValueError("trigger threshold must be finite")
    dense = {str(row["uid"]): row for row in dense_rows}
    scores = {str(row["uid"]): row for row in score_rows}
    expected = {str(row["uid"]) for row in candidates}
    if len(dense) != len(dense_rows) or len(scores) != len(score_rows):
        raise ValueError("duplicate UID in dense or score rows")
    if set(dense) != expected or set(scores) != expected:
        raise ValueError("candidate/dense/score UID coverage differs")

    trigger_map: list[dict[str, Any]] = []
    for candidate in candidates:
        uid = str(candidate["uid"])
        dense_row = dense[uid]
        score = scores[uid]
        if (
            str(candidate["dataset"]) != str(dense_row["dataset"])
            or str(candidate["image_group_id"]) != str(dense_row["image_group_id"])
        ):
            raise ValueError(f"candidate/dense provenance differs: {uid}")
        values = [float(score[f"p_{layer}"]) for layer in range(28)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite Stage-1 score: {uid}")
        crossings = [layer for layer, value in enumerate(values) if value > threshold]
        wrong = bool(dense_row["current_dense_wrong"])
        row = {
            "schema_version": "stage2_data_scale_trigger_map_v1",
            "uid": uid,
            "dataset": str(candidate["dataset"]),
            "split": "train_scaleup",
            "group_id": str(candidate["image_group_id"]),
            "image_group_id": str(candidate["image_group_id"]),
            "dense_wrong": wrong,
            "dense_correct": not wrong,
            "triggered": bool(crossings),
            "first_trigger_layer": crossings[0] if crossings else None,
            "score_at_trigger": values[crossings[0]] if crossings else None,
            "threshold": float(threshold),
            "comparison": "strict_greater_than",
        }
        row.update({f"p_{layer}": values[layer] for layer in range(28)})
        trigger_map.append(row)
    triggered_wrong = [row for row in trigger_map if row["triggered"] and row["dense_wrong"]]
    triggered_correct = [row for row in trigger_map if row["triggered"] and not row["dense_wrong"]]
    return trigger_map, triggered_wrong, triggered_correct


def build_search_work(
    candidates: Sequence[Mapping[str, Any]],
    dense_rows: Sequence[Mapping[str, Any]],
    trigger_rows: Sequence[Mapping[str, Any]],
    *,
    maximum_iterations: int,
) -> list[dict[str, Any]]:
    """Join the frozen triggered population to native samples and dense outputs."""

    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    candidate_by_uid = {str(row["uid"]): row for row in candidates}
    dense_by_uid = {str(row["uid"]): row for row in dense_rows}
    if len(candidate_by_uid) != len(candidates) or len(dense_by_uid) != len(dense_rows):
        raise ValueError("candidate or dense rows contain duplicate UIDs")
    if any(not bool(row.get("triggered")) for row in trigger_rows):
        raise ValueError("search rows must all be triggered")
    trigger_uids = [str(row["uid"]) for row in trigger_rows]
    if len(trigger_uids) != len(set(trigger_uids)):
        raise ValueError("trigger rows contain duplicate UIDs")

    sample_keys = (
        "uid",
        "sample_id",
        "dataset",
        "prompt",
        "question",
        "answer",
        "all_answer_norms",
        "local_image_path",
        "image_content_sha256",
        "image_group_id",
        "max_new_tokens",
    )
    output: list[dict[str, Any]] = []
    for trigger in trigger_rows:
        uid = str(trigger["uid"])
        if uid not in candidate_by_uid or uid not in dense_by_uid:
            raise ValueError(f"trigger UID is absent from candidate/dense rows: {uid}")
        candidate = candidate_by_uid[uid]
        dense = dense_by_uid[uid]
        expected = (
            str(candidate["dataset"]),
            str(candidate["image_group_id"]),
            str(candidate["image_content_sha256"]),
        )
        actual = (
            str(dense["dataset"]),
            str(dense["image_group_id"]),
            str(dense["image_content_sha256"]),
        )
        trigger_identity = (
            str(trigger["dataset"]),
            str(trigger["image_group_id"]),
        )
        if actual != expected or trigger_identity != expected[:2]:
            raise ValueError(f"candidate/dense/trigger provenance differs: {uid}")
        wrong = bool(dense["current_dense_wrong"])
        if bool(trigger["dense_wrong"]) != wrong:
            raise ValueError(f"trigger/dense correctness differs: {uid}")
        start = int(trigger["first_trigger_layer"])
        if start not in range(28):
            raise ValueError(f"invalid trigger layer: {uid}")
        sample = {key: candidate[key] for key in sample_keys}
        output.append(
            {
                **dict(trigger),
                "task_type": (
                    "triggered_wrong"
                    if wrong
                    else "triggered_correct_preservation"
                ),
                "trigger_depth_bin": trigger_depth_bin(start),
                "sample": sample,
                "dense_output": dict(dense),
                "estimated_terminal_evaluations": (
                    3 * (28 - start) + maximum_iterations if wrong else 2
                ),
            }
        )
    return sorted(output, key=lambda row: str(row["uid"]))
