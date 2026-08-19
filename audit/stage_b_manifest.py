from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


def _selection_hash(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def select_balanced_unique_assets(
    rows_by_cell: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
    quota_per_cell: int,
    seed: int,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select deterministic cell-balanced samples with no repeated image asset."""
    if quota_per_cell < 1:
        raise ValueError("quota_per_cell must be positive")
    excluded_ids = excluded_ids or set()
    used_assets: set[str] = set()
    used_ids: set[str] = set()
    selected: list[dict[str, Any]] = []

    for cell in sorted(rows_by_cell):
        candidates = []
        for row in rows_by_cell[cell]:
            sample_id = str(row["id"])
            if sample_id in excluded_ids:
                continue
            candidate = dict(row)
            candidate["selection_hash"] = _selection_hash(seed, sample_id)
            candidates.append(candidate)
        candidates.sort(key=lambda row: (row["selection_hash"], str(row["id"])))

        cell_selected = 0
        for candidate_rank, row in enumerate(candidates, start=1):
            sample_id = str(row["id"])
            if sample_id in used_ids:
                continue
            asset_key = str(
                row.get("source_asset_id") or row.get("local_image_path") or row["id"]
            )
            if asset_key in used_assets:
                continue
            row["selection_asset_key"] = asset_key
            row["selection_rank_in_cell"] = candidate_rank
            row["selection_cell"] = f"{cell[0]}:{cell[1]}"
            selected.append(row)
            used_assets.add(asset_key)
            used_ids.add(sample_id)
            cell_selected += 1
            if cell_selected == quota_per_cell:
                break
        if cell_selected != quota_per_cell:
            raise ValueError(
                f"Could not fill quota for {cell}: selected {cell_selected}/{quota_per_cell}"
            )

    return selected
