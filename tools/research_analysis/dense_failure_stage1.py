"""Pure utilities for current-runtime dense Stage-1 label regeneration."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


STAGE1_DATASETS = ("gqa", "chartqa", "textvqa")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(value: Mapping) -> str:
    """Hash a JSON contract while excluding its self-referential ID."""

    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _current_image_path(row: Mapping, image_root: Path) -> Path:
    bucket = str(row["source_bucket"])
    if bucket == "complete_correct":
        physical_bucket = "correct"
    elif bucket == "complete_wrong":
        physical_bucket = "wrong"
    else:
        raise ValueError(f"unsupported historical bucket: {bucket!r}")
    suffix = Path(str(row["local_image_path"])).suffix
    if not suffix:
        raise ValueError(f"image suffix is unavailable for {row['uid']}")
    return (
        image_root
        / str(row["benchmark"])
        / f"{physical_bucket}__{row['sample_id']}{suffix}"
    ).resolve()


def build_candidate_manifest(
    portable_rows: Iterable[Mapping],
    *,
    image_root: Path,
    expected_counts: Mapping[str, int],
) -> tuple[list[dict], dict]:
    """Build and fail-closed validate the dense-only candidate population."""

    selected = [
        dict(row)
        for row in portable_rows
        if str(row.get("benchmark")) in expected_counts
    ]
    counts = Counter(str(row["benchmark"]) for row in selected)
    if counts != Counter({str(key): int(value) for key, value in expected_counts.items()}):
        raise ValueError(
            f"candidate dataset counts differ: actual={dict(counts)} "
            f"expected={dict(expected_counts)}"
        )

    seen_uids: set[str] = set()
    manifest: list[dict] = []
    for source_index, row in enumerate(selected):
        uid = str(row["uid"])
        if uid in seen_uids:
            raise ValueError(f"duplicate candidate UID: {uid}")
        seen_uids.add(uid)
        for field in (
            "sample_id",
            "benchmark",
            "question",
            "prompt",
            "answer",
            "metric_name",
            "correctness_threshold",
            "max_new_tokens",
            "source_bucket",
        ):
            if row.get(field) is None or (isinstance(row.get(field), str) and not row[field]):
                raise ValueError(f"{uid} has empty required field {field}")

        image_path = _current_image_path(row, image_root)
        if not image_path.is_file():
            raise ValueError(f"candidate image is missing: {uid}: {image_path}")
        if not image_path.stat().st_size:
            raise ValueError(f"candidate image is empty: {uid}: {image_path}")
        actual_sha256 = _file_sha256(image_path)
        frozen_sha256 = row.get("image_content_sha256")
        if not isinstance(frozen_sha256, str) or len(frozen_sha256) != 64:
            raise ValueError(f"{uid} has no valid frozen image content SHA-256")
        try:
            int(frozen_sha256, 16)
        except ValueError as exc:
            raise ValueError(f"{uid} has no valid frozen image content SHA-256") from exc
        if actual_sha256 != frozen_sha256:
            raise ValueError(
                f"candidate image content SHA-256 differs for {uid}: "
                f"actual={actual_sha256} frozen={frozen_sha256}"
            )

        historical_bucket = {
            "complete_correct": "correct",
            "complete_wrong": "wrong",
        }[str(row["source_bucket"])]
        manifest.append(
            {
                "schema_version": "current_dense_candidate_v1",
                "uid": uid,
                "sample_id": str(row["sample_id"]),
                "dataset": str(row["benchmark"]),
                "benchmark": str(row["benchmark"]),
                "question": str(row["question"]),
                "prompt": str(row["prompt"]),
                "answer": str(row["answer"]),
                "all_answer_norms": row.get("all_answer_norms"),
                "metric_name": str(row["metric_name"]),
                "correctness_threshold": float(row["correctness_threshold"]),
                "max_new_tokens": int(row["max_new_tokens"]),
                "max_image_tokens": None,
                "local_image_path": str(image_path),
                "image_identifier": str(row.get("source_asset_id") or actual_sha256),
                "image_content_sha256": actual_sha256,
                "image_group_id": f"sha256:{actual_sha256}",
                "image_size_bytes": int(image_path.stat().st_size),
                "historical_bucket": historical_bucket,
                "source_manifest": row.get("source_manifest"),
                "source_manifest_index": source_index,
            }
        )

    manifest.sort(key=lambda row: row["uid"])
    group_counts = Counter(row["image_group_id"] for row in manifest)
    audit = {
        "records": len(manifest),
        "unique_uids": len(seen_uids),
        "dataset_counts": dict(sorted(counts.items())),
        "historical_bucket_counts": {
            f"{dataset}/{bucket}": value
            for (dataset, bucket), value in sorted(
                Counter(
                    (row["dataset"], row["historical_bucket"])
                    for row in manifest
                ).items()
            )
        },
        "unique_image_groups": len(group_counts),
        "multi_question_image_groups": sum(value > 1 for value in group_counts.values()),
        "max_records_per_image_group": max(group_counts.values(), default=0),
        "unresolved_images": 0,
        "content_hash_mismatches": 0,
    }
    return manifest, audit


def validate_candidate_execution_contract(
    rows: Sequence[Mapping],
    *,
    evaluator_specs: Mapping[str, Mapping],
    max_new_tokens: int,
) -> None:
    """Reject evaluator fallback or per-row generation-policy drift."""

    if not rows:
        raise ValueError("candidate execution manifest is empty")
    supported = set(evaluator_specs)
    for row in rows:
        uid = str(row.get("uid", "<missing>"))
        dataset = str(row.get("dataset", row.get("benchmark", "")))
        if dataset not in supported:
            raise ValueError(f"{uid} has unsupported dataset/evaluator contract: {dataset!r}")
        expected = evaluator_specs[dataset]
        actual_pair = (
            row.get("metric_name"),
            float(row.get("correctness_threshold", -1.0)),
        )
        expected_pair = (
            expected["metric_name"],
            float(expected["correctness_threshold"]),
        )
        if actual_pair != expected_pair:
            raise ValueError(
                f"{uid} evaluator contract differs: actual={actual_pair} expected={expected_pair}"
            )
        if int(row.get("max_new_tokens", -1)) != int(max_new_tokens):
            raise ValueError(
                f"{uid} max_new_tokens differs: actual={row.get('max_new_tokens')} "
                f"expected={max_new_tokens}"
            )
        if row.get("max_image_tokens") is not None:
            raise ValueError(f"{uid} sets a forbidden custom max_image_tokens")


def _stable_order_key(uid: str, seed: int) -> tuple[str, str]:
    return sha256(f"{seed}\0{uid}".encode("utf-8")).hexdigest(), uid


def select_dense_smoke(
    rows: Sequence[Mapping],
    *,
    seed: int,
    per_cell: int = 4,
) -> list[dict]:
    """Select distinct-image smoke rows across dataset/historical cells."""

    chosen: list[dict] = []
    used_groups: set[str] = set()
    for dataset in STAGE1_DATASETS:
        for bucket in ("correct", "wrong"):
            candidates = sorted(
                (
                    dict(row)
                    for row in rows
                    if row["dataset"] == dataset
                    and row["historical_bucket"] == bucket
                ),
                key=lambda row: _stable_order_key(str(row["uid"]), seed),
            )
            cell: list[dict] = []
            for row in candidates:
                group = str(row["image_group_id"])
                if group in used_groups:
                    continue
                cell.append(row)
                used_groups.add(group)
                if len(cell) == per_cell:
                    break
            if len(cell) != per_cell:
                raise ValueError(
                    f"insufficient distinct image groups for smoke cell "
                    f"{dataset}/{bucket}: {len(cell)} < {per_cell}"
                )
            chosen.extend(cell)
    return chosen


@dataclass(frozen=True)
class TokenPositions:
    visual: tuple[int, ...]
    user_text: tuple[int, ...]
    final_user_token: int


def locate_user_and_visual_tokens(
    input_ids: Sequence[int],
    *,
    image_token_id: int,
    vision_end_token_id: int,
    im_end_token_id: int,
) -> TokenPositions:
    """Locate the expanded visual span and literal user prompt token span."""

    ids = [int(value) for value in input_ids]
    visual = tuple(index for index, value in enumerate(ids) if value == image_token_id)
    if not visual:
        raise ValueError("processor input contains no visual image tokens")
    vision_ends = [index for index, value in enumerate(ids) if value == vision_end_token_id]
    if not vision_ends:
        raise ValueError("processor input contains no vision-end token")
    start = vision_ends[-1] + 1
    try:
        stop = next(index for index in range(start, len(ids)) if ids[index] == im_end_token_id)
    except StopIteration as exc:
        raise ValueError("processor input contains no user-message end token") from exc
    user_text = tuple(range(start, stop))
    if not user_text:
        raise ValueError("processor input has an empty user text span")
    return TokenPositions(
        visual=visual,
        user_text=user_text,
        final_user_token=user_text[-1],
    )


def _split_cost(
    *,
    split: str,
    group_rows: Sequence[Mapping],
    assigned_totals: Mapping[str, int],
    assigned_cells: Mapping[str, Counter],
    targets: Mapping[str, int],
    desired_cells: Mapping[str, Mapping[tuple[str, bool], float]],
) -> float:
    target = max(int(targets[split]), 1)
    next_total = int(assigned_totals[split]) + len(group_rows)
    total_cost = ((next_total - target) / target) ** 2
    if next_total > target:
        total_cost += 10.0 * ((next_total - target) / target) ** 2
    cell_delta = Counter(
        (str(row["dataset"]), bool(row["current_dense_wrong"]))
        for row in group_rows
    )
    cell_cost = 0.0
    for cell, desired in desired_cells[split].items():
        next_value = assigned_cells[split][cell] + cell_delta[cell]
        cell_cost += ((next_value - desired) / max(desired, 1.0)) ** 2
    return total_cost + cell_cost / max(len(desired_cells[split]), 1)


def build_image_group_disjoint_split(
    rows: Sequence[Mapping],
    *,
    targets: Mapping[str, int],
    seed: int,
) -> list[dict]:
    """Greedily assign whole image groups while preserving cells and sizes."""

    if set(targets) != {"train", "val", "test"}:
        raise ValueError("split targets must contain train, val, and test")
    uids = [str(row["uid"]) for row in rows]
    if len(uids) != len(set(uids)):
        raise ValueError("cannot split duplicate UIDs")
    if sum(int(value) for value in targets.values()) != len(rows):
        raise ValueError("split targets must sum to the record count")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_group_id"])].append(dict(row))
    population_cells = Counter(
        (str(row["dataset"]), bool(row["current_dense_wrong"]))
        for row in rows
    )
    desired_cells = {
        split: {
            cell: count * int(target) / len(rows)
            for cell, count in population_cells.items()
        }
        for split, target in targets.items()
    }
    assigned_totals = Counter({split: 0 for split in targets})
    assigned_cells = {split: Counter() for split in targets}
    group_assignments: dict[str, str] = {}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            _stable_order_key(item[0], seed),
        ),
    )
    split_order = {"train": 0, "val": 1, "test": 2}
    for group_id, group_rows in ordered_groups:
        group_cells = Counter(
            (str(row["dataset"]), bool(row["current_dense_wrong"]))
            for row in group_rows
        )
        feasible = [
            split
            for split in ("train", "val", "test")
            if assigned_totals[split] + len(group_rows) <= int(targets[split])
        ]
        candidate_splits = feasible or ["train", "val", "test"]
        candidates = []
        for split in candidate_splits:
            target = max(int(targets[split]), 1)
            remaining_fraction = (
                int(targets[split]) - assigned_totals[split]
            ) / target
            cell_pressures = []
            for cell, amount in group_cells.items():
                desired = desired_cells[split][cell]
                remaining_cell = desired - assigned_cells[split][cell]
                cell_pressures.extend([remaining_cell / max(desired, 1.0)] * amount)
            cell_pressure = sum(cell_pressures) / max(len(cell_pressures), 1)
            overshoot = max(
                0,
                assigned_totals[split] + len(group_rows) - int(targets[split]),
            )
            score = remaining_fraction + 0.25 * cell_pressure - 100.0 * overshoot
            candidates.append((-score, split_order[split], split))
        _, _, selected = min(candidates)
        group_assignments[group_id] = selected
        assigned_totals[selected] += len(group_rows)
        assigned_cells[selected].update(
            (str(row["dataset"]), bool(row["current_dense_wrong"]))
            for row in group_rows
        )

    output = [
        {**dict(row), "split": group_assignments[str(row["image_group_id"])]}
        for row in rows
    ]
    output.sort(key=lambda row: row["uid"])
    return output
