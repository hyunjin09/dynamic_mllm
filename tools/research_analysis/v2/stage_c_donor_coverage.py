from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from nulls.structured_read import (
    DonorMetadata,
    donor_matching_ratio,
    real_donor_candidates,
)


def _ratio(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        return math.inf
    return max(left / right, right / left)


def target_coverage(
    target: DonorMetadata,
    donors: Sequence[DonorMetadata],
    seed: int,
    original_caliper: float,
) -> dict[str, Any]:
    eligible = [
        donor
        for donor in donors
        if donor.sample_id != target.sample_id and donor.image_id != target.image_id
    ]
    if len(eligible) < 9:
        raise ValueError(f"Only {len(eligible)} eligible donors for {target.sample_id}")
    maximum_distance = max(donor_matching_ratio(target, donor) for donor in eligible)
    ordered = real_donor_candidates(
        target,
        donors,
        seed=seed,
        matching_ratio_cap=max(1.0, maximum_distance),
    )
    if len(ordered) != len(eligible):
        raise RuntimeError("Finite all-donor cap did not retain the complete eligible pool")
    distances = [donor_matching_ratio(target, donor) for donor in ordered]

    def donor_payload(donor: DonorMetadata) -> dict[str, Any]:
        return {
            "sample_id": donor.sample_id,
            "image_id": donor.image_id,
            "composite_matching_distance": donor_matching_ratio(target, donor),
            "residual_norm_ratio": _ratio(target.residual_norm, donor.residual_norm),
            "postvisual_row_ratio": _ratio(target.postvisual_rows, donor.postvisual_rows),
            "image_token_ratio": _ratio(target.visual_tokens, donor.visual_tokens),
            "donor_residual_norm": donor.residual_norm,
            "donor_postvisual_rows": donor.postvisual_rows,
            "donor_image_tokens": donor.visual_tokens,
            "donor_prompt_tokens": donor.prompt_tokens,
        }

    return {
        "id": target.sample_id,
        "image_id": target.image_id,
        "target_geometry": {
            "residual_norm": target.residual_norm,
            "postvisual_rows": target.postvisual_rows,
            "image_tokens": target.visual_tokens,
            "prompt_tokens": target.prompt_tokens,
        },
        "eligible_donor_pool_count": len(ordered),
        "donors_within_original_caliper": sum(
            distance <= original_caliper for distance in distances
        ),
        "nearest_distances": {
            "rank_1": distances[0],
            "rank_7": distances[6],
            "rank_8": distances[7],
            "rank_9": distances[8],
        },
        "nearest_eight_donors": [donor_payload(donor) for donor in ordered[:8]],
        "original_caliper": original_caliper,
        "original_caliper_supplies_eight": distances[7] <= original_caliper,
        "likelihood_or_behavior_loaded": False,
    }


def coverage_summary(
    target_rows: Sequence[dict[str, Any]],
    calipers: Sequence[float],
    quantiles: Sequence[float],
) -> dict[str, Any]:
    if not target_rows:
        raise ValueError("Coverage summary requires at least one target")
    eighth = np.asarray(
        [row["nearest_distances"]["rank_8"] for row in target_rows],
        dtype=np.float64,
    )
    c_star = float(eighth.max())
    c_star_targets = [
        row["id"]
        for row in target_rows
        if float(row["nearest_distances"]["rank_8"]) == c_star
    ]
    unique_calipers = sorted({float(value) for value in calipers} | {c_star})
    return {
        "target_count": len(target_rows),
        "c_star": c_star,
        "c_star_target_ids": c_star_targets,
        "eighth_nearest_distance_distribution": {
            "minimum": float(eighth.min()),
            "maximum": c_star,
            "mean": float(eighth.mean()),
            "standard_deviation": float(eighth.std(ddof=1)),
            "quantiles": {
                str(float(quantile)): float(np.quantile(eighth, quantile))
                for quantile in quantiles
            },
        },
        "targets_supported_by_caliper": {
            str(caliper): int(np.sum(eighth <= caliper)) for caliper in unique_calipers
        },
    }
