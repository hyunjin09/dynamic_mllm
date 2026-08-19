from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _index_unique(rows: Sequence[dict[str, Any]], role: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = str(row.get("uid", ""))
        if not uid:
            raise ValueError(f"{role} row has no UID")
        if uid in indexed:
            raise ValueError(f"duplicate {role} UID: {uid}")
        indexed[uid] = row
    return indexed


def align_manifest_policy_rows(
    manifest_rows: Sequence[dict[str, Any]],
    policy_rows: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Validate and align an exact policy replay to manifest order."""

    manifest_by_uid = _index_unique(manifest_rows, "manifest")
    policy_by_uid = _index_unique(policy_rows, "policy")
    if set(manifest_by_uid) != set(policy_by_uid):
        missing_policy = sorted(set(manifest_by_uid) - set(policy_by_uid))[:3]
        missing_manifest = sorted(set(policy_by_uid) - set(manifest_by_uid))[:3]
        raise ValueError(
            "manifest/policy UID sets differ: "
            f"missing_policy={missing_policy}, missing_manifest={missing_manifest}"
        )
    aligned = []
    for manifest_row in manifest_rows:
        uid = str(manifest_row["uid"])
        policy_row = policy_by_uid[uid]
        for field in ("benchmark", "metric_name", "correctness_threshold"):
            if str(manifest_row.get(field)) != str(policy_row.get(field)):
                raise ValueError(
                    f"{field} mismatch for {uid}: "
                    f"manifest={manifest_row.get(field)!r}, policy={policy_row.get(field)!r}"
                )
        for field in (
            "baseline_correct",
            "router_correct",
            "selected_num_visual_on_layers",
        ):
            if field not in policy_row:
                raise ValueError(f"policy row {uid} is missing {field}")
        aligned.append((manifest_row, policy_row))
    return aligned
