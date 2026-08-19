from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from nulls.structured_read import (
    DonorMetadata,
    donor_matching_ratio,
    real_donor_candidates,
    select_real_donors,
)


def _target_metadata(row: Mapping[str, Any]) -> DonorMetadata:
    geometry = row["target_geometry"]
    return DonorMetadata(
        sample_id=str(row["id"]),
        image_id=str(row["image_id"]),
        residual_norm=float(geometry["residual_norm"]),
        postvisual_rows=int(geometry["postvisual_rows"]),
        visual_tokens=int(geometry["image_tokens"]),
        prompt_tokens=int(geometry["prompt_tokens"]),
    )


def build_amended_match_rows(
    audit: Mapping[str, Any],
    donors: Sequence[DonorMetadata],
    tie_seeds: Mapping[str, int],
    draws: int,
    expected_target_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_caliper = float(audit["original_caliper"])
    amended_caliper = float(audit["c_star"])
    if amended_caliper != 1.5833333333333333 and expected_target_count == 800:
        raise ValueError("The approved Stage C amended caliper must be exact 19/12")
    if draws != 8:
        raise ValueError("The frozen real-residual donor count must remain eight")
    target_rows = list(audit["targets"])
    if len(target_rows) != expected_target_count:
        raise ValueError("Coverage audit target count differs from the frozen manifest")
    if set(tie_seeds) != {str(row["id"]) for row in target_rows}:
        raise ValueError("Tie seeds do not cover the exact audited target set")

    output: list[dict[str, Any]] = []
    original_supported = 0
    changed_supported = 0
    widened_ids: list[str] = []
    for audit_row in target_rows:
        target = _target_metadata(audit_row)
        seed = int(tie_seeds[target.sample_id])
        amended_candidates = real_donor_candidates(
            target, donors, seed=seed, matching_ratio_cap=amended_caliper
        )
        amended_selected = select_real_donors(
            target,
            donors,
            draws=draws,
            seed=seed,
            matching_ratio_cap=amended_caliper,
        )
        audited_ids = [
            str(item["sample_id"]) for item in audit_row["nearest_eight_donors"]
        ]
        amended_ids = [donor.sample_id for donor in amended_selected]
        if amended_ids != audited_ids:
            raise RuntimeError(f"Amended donor ordering differs from audit for {target.sample_id}")

        original_candidates = real_donor_candidates(
            target, donors, seed=seed, matching_ratio_cap=original_caliper
        )
        supported = len(original_candidates) >= draws
        unchanged = None
        if supported:
            original_supported += 1
            original_selected = select_real_donors(
                target,
                donors,
                draws=draws,
                seed=seed,
                matching_ratio_cap=original_caliper,
            )
            unchanged = [donor.sample_id for donor in original_selected] == amended_ids
            if not unchanged:
                changed_supported += 1
        else:
            widened_ids.append(target.sample_id)

        output.append(
            {
                "schema_version": "stage_c_real_residual_match_index_v2",
                "id": target.sample_id,
                "image_id": target.image_id,
                "manifest_record_sha256": audit_row["manifest_record_sha256"],
                "layer": int(audit_row["layer"]),
                "hook": audit_row["hook"],
                "tie_break_seed": seed,
                "original_caliper": original_caliper,
                "amended_caliper": amended_caliper,
                "original_eligible_donor_count": len(original_candidates),
                "amended_eligible_donor_count": len(amended_candidates),
                "original_caliper_supplies_eight": supported,
                "selection_unchanged_from_original": unchanged,
                "target_geometry": audit_row["target_geometry"],
                "selected_donors": [
                    {
                        "rank": rank,
                        "sample_id": donor.sample_id,
                        "image_id": donor.image_id,
                        "matching_distance": donor_matching_ratio(target, donor),
                        "enters_due_to_amendment": donor_matching_ratio(target, donor)
                        > original_caliper,
                    }
                    for rank, donor in enumerate(amended_selected, start=1)
                ],
                "outcome_used_for_selection": False,
            }
        )

    summary = {
        "schema_version": "stage_c_real_residual_match_index_summary_v2",
        "target_count": len(output),
        "donor_count_per_target": draws,
        "original_caliper": original_caliper,
        "amended_caliper": amended_caliper,
        "original_supported_target_count": original_supported,
        "amended_supported_target_count": len(output),
        "wider_caliper_target_ids": widened_ids,
        "selection_changed_for_original_supported_count": changed_supported,
        "likelihood_or_intervention_outcome_used": False,
    }
    return output, summary


def validate_frozen_match_row(
    target: DonorMetadata,
    donors: Sequence[DonorMetadata],
    match_row: Mapping[str, Any],
    draws: int,
    seed: int,
    amended_caliper: float,
    norm_relative_tolerance: float = 1e-5,
) -> list[DonorMetadata]:
    if str(match_row["id"]) != target.sample_id or str(match_row["image_id"]) != target.image_id:
        raise RuntimeError("Frozen match row target identity changed")
    if int(match_row["tie_break_seed"]) != int(seed):
        raise RuntimeError("Frozen match row tie seed changed")
    if float(match_row["amended_caliper"]) != float(amended_caliper):
        raise RuntimeError("Frozen match row caliper changed")
    geometry = match_row["target_geometry"]
    if (
        int(geometry["postvisual_rows"]) != target.postvisual_rows
        or int(geometry["image_tokens"]) != target.visual_tokens
        or int(geometry["prompt_tokens"]) != target.prompt_tokens
    ):
        raise RuntimeError("Frozen match row target geometry changed")
    frozen_norm = float(geometry["residual_norm"])
    if not math.isclose(
        frozen_norm,
        target.residual_norm,
        rel_tol=norm_relative_tolerance,
        abs_tol=1e-8,
    ):
        raise RuntimeError("Frozen match row target norm changed")
    selected = select_real_donors(
        target,
        donors,
        draws=draws,
        seed=seed,
        matching_ratio_cap=amended_caliper,
    )
    frozen_ids = [str(row["sample_id"]) for row in match_row["selected_donors"]]
    if [donor.sample_id for donor in selected] != frozen_ids:
        raise RuntimeError("Live donor selection differs from the frozen amended index")
    for donor, frozen in zip(selected, match_row["selected_donors"]):
        if not math.isclose(
            donor_matching_ratio(target, donor),
            float(frozen["matching_distance"]),
            rel_tol=norm_relative_tolerance,
            abs_tol=1e-8,
        ):
            raise RuntimeError("Live donor distance differs from the frozen amended index")
    return selected
