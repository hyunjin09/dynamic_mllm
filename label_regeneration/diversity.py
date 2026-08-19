"""P6 route-diversity diagnostics for complete binary visual masks."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable


NUM_LAYERS = 28
DATASETS = ("gqa", "textvqa", "chartqa")
STATUSES = ("correct", "wrong")


def _linear_quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[int | float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "minimum": min(values) if values else None,
        "p10": _linear_quantile(values, 0.10),
        "p25": _linear_quantile(values, 0.25),
        "median": _linear_quantile(values, 0.50),
        "p75": _linear_quantile(values, 0.75),
        "p90": _linear_quantile(values, 0.90),
        "p95": _linear_quantile(values, 0.95),
        "p99": _linear_quantile(values, 0.99),
        "maximum": max(values) if values else None,
    }


def histogram_stats(histogram: list[int]) -> dict[str, Any]:
    count = sum(histogram)
    if not count:
        return distribution([])
    weighted_sum = sum(value * frequency for value, frequency in enumerate(histogram))

    def quantile(probability: float) -> float:
        # Exact type-7 quantile without materializing pairwise-distance values.
        position = (count - 1) * probability
        lower_rank = math.floor(position)
        upper_rank = math.ceil(position)

        def value_at(rank: int) -> int:
            cumulative = 0
            for value, frequency in enumerate(histogram):
                cumulative += frequency
                if rank < cumulative:
                    return value
            raise AssertionError("histogram rank out of range")

        lower = value_at(lower_rank)
        upper = value_at(upper_rank)
        return lower + (upper - lower) * (position - lower_rank)

    nonzero = [value for value, frequency in enumerate(histogram) if frequency]
    return {
        "count": count,
        "mean": weighted_sum / count,
        "minimum": float(nonzero[0]),
        "p10": quantile(0.10),
        "p25": quantile(0.25),
        "median": quantile(0.50),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "maximum": float(nonzero[-1]),
    }


def mask_run_lengths(mask: Iterable[int]) -> list[tuple[int, int]]:
    mask = tuple(int(bit) for bit in mask)
    if len(mask) != NUM_LAYERS or any(bit not in (0, 1) for bit in mask):
        raise ValueError("expected a complete 28-bit binary mask")
    runs: list[tuple[int, int]] = []
    action = mask[0]
    length = 1
    for bit in mask[1:]:
        if bit == action:
            length += 1
        else:
            runs.append((action, length))
            action = bit
            length = 1
    runs.append((action, length))
    return runs


def _mask_integer(mask: Iterable[int]) -> int:
    value = 0
    for bit in mask:
        value = (value << 1) | int(bit)
    return value


def _empty_histogram() -> list[int]:
    return [0] * (NUM_LAYERS + 1)


def summarize_record_diversity(record: dict[str, Any]) -> dict[str, Any]:
    sample = record["sample"]
    candidate_by_id = {
        candidate["route_id"]: candidate for candidate in record.get("candidate_executions", [])
    }
    valid = [candidate_by_id[route_id] for route_id in record["successful_route_ids"]]
    best_id = record.get("best_sparse_success_route_id")
    reference = candidate_by_id[best_id] if best_id is not None else None
    reference_mask = tuple(reference["visual_on_mask"]) if reference is not None else None
    reference_integer = _mask_integer(reference_mask) if reference_mask is not None else None

    on_hist = _empty_histogram()
    transition_hist = _empty_histogram()
    hamming_all_on_hist = _empty_histogram()
    hamming_min_hist = _empty_histogram()
    segment_count_hist = _empty_histogram()
    all_segment_length_hist = _empty_histogram()
    on_segment_length_hist = _empty_histogram()
    off_segment_length_hist = _empty_histogram()
    integers: list[int] = []

    for route in valid:
        mask = tuple(int(bit) for bit in route["visual_on_mask"])
        on_count = sum(mask)
        transitions = sum(mask[index] != mask[index - 1] for index in range(1, NUM_LAYERS))
        hamming_all_on = NUM_LAYERS - on_count
        mask_integer = _mask_integer(mask)
        hamming_minimum = (
            (mask_integer ^ reference_integer).bit_count() if reference_integer is not None else 0
        )
        runs = mask_run_lengths(mask)
        on_hist[on_count] += 1
        transition_hist[transitions] += 1
        hamming_all_on_hist[hamming_all_on] += 1
        hamming_min_hist[hamming_minimum] += 1
        segment_count_hist[len(runs)] += 1
        for action, length in runs:
            all_segment_length_hist[length] += 1
            (on_segment_length_hist if action else off_segment_length_hist)[length] += 1
        integers.append(mask_integer)

    pairwise_hist = _empty_histogram()
    for right_index, right in enumerate(integers):
        for left in integers[:right_index]:
            pairwise_hist[(left ^ right).bit_count()] += 1

    on_stats = histogram_stats(on_hist)
    transition_stats = histogram_stats(transition_hist)
    hamming_min_stats = histogram_stats(hamming_min_hist)
    pairwise_stats = histogram_stats(pairwise_hist)
    segment_length_stats = histogram_stats(all_segment_length_hist)
    return {
        "schema_version": "label_regeneration_per_sample_diversity_p6_v1",
        "uid": sample["uid"],
        "dataset": sample["benchmark"],
        "sample_id": sample["sample_id"],
        "image_group_id": sample["image_group_id"],
        "current_all_on_status": sample["current_all_on_status"],
        "valid_route_count": len(valid),
        "minimum_visual_on_valid_route": (
            int(reference["num_visual_on_layers"]) if reference is not None else None
        ),
        "minimum_route_id": best_id,
        "on_count_histogram": on_hist,
        "transition_count_histogram": transition_hist,
        "hamming_to_all_on_histogram": hamming_all_on_hist,
        "hamming_to_minimum_histogram": hamming_min_hist,
        "segment_count_histogram": segment_count_hist,
        "all_segment_length_histogram": all_segment_length_hist,
        "on_segment_length_histogram": on_segment_length_hist,
        "off_segment_length_histogram": off_segment_length_hist,
        "pairwise_hamming_histogram": pairwise_hist,
        "mean_visual_on_count": on_stats["mean"],
        "mean_transition_count": transition_stats["mean"],
        "mean_hamming_to_minimum": hamming_min_stats["mean"],
        "mean_pairwise_hamming": pairwise_stats["mean"],
        "mean_segment_length": segment_length_stats["mean"],
        "pairwise_hamming": pairwise_stats,
    }


def _sum_histograms(rows: list[dict[str, Any]], field: str) -> list[int]:
    output = _empty_histogram()
    for row in rows:
        for value, frequency in enumerate(row[field]):
            output[value] += frequency
    return output


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    on_hist = _sum_histograms(rows, "on_count_histogram")
    transition_hist = _sum_histograms(rows, "transition_count_histogram")
    hamming_all_on_hist = _sum_histograms(rows, "hamming_to_all_on_histogram")
    hamming_min_hist = _sum_histograms(rows, "hamming_to_minimum_histogram")
    segment_count_hist = _sum_histograms(rows, "segment_count_histogram")
    all_segment_hist = _sum_histograms(rows, "all_segment_length_histogram")
    on_segment_hist = _sum_histograms(rows, "on_segment_length_histogram")
    off_segment_hist = _sum_histograms(rows, "off_segment_length_histogram")
    pairwise_hist = _sum_histograms(rows, "pairwise_hamming_histogram")
    positive = [row for row in rows if row["valid_route_count"] > 0]
    paired = [row for row in rows if row["pairwise_hamming"]["count"] > 0]
    valid_masks = sum(on_hist)
    return {
        "samples": len(rows),
        "samples_with_valid_routes": len(positive),
        "zero_valid_samples": len(rows) - len(positive),
        "valid_masks": valid_masks,
        "route_weighted": {
            "visual_on_count": histogram_stats(on_hist),
            "visual_off_count_hamming_to_all_on": histogram_stats(hamming_all_on_hist),
            "transition_count": histogram_stats(transition_hist),
            "segment_count": histogram_stats(segment_count_hist),
            "hamming_to_minimum_route": histogram_stats(hamming_min_hist),
            "all_segment_length": histogram_stats(all_segment_hist),
            "on_segment_length": histogram_stats(on_segment_hist),
            "off_segment_length": histogram_stats(off_segment_hist),
            "pairwise_hamming": histogram_stats(pairwise_hist),
        },
        "sample_balanced": {
            "mean_visual_on_count": distribution(
                row["mean_visual_on_count"] for row in positive
            ),
            "mean_transition_count": distribution(
                row["mean_transition_count"] for row in positive
            ),
            "mean_hamming_to_minimum": distribution(
                row["mean_hamming_to_minimum"] for row in positive
            ),
            "mean_pairwise_hamming": distribution(
                row["mean_pairwise_hamming"] for row in paired
            ),
            "mean_segment_length": distribution(
                row["mean_segment_length"] for row in positive
            ),
        },
        "structural_frequencies": {
            "all_off_masks": on_hist[0],
            "all_on_masks": on_hist[NUM_LAYERS],
            "transition_0": transition_hist[0],
            "transition_le_1": sum(transition_hist[:2]),
            "transition_le_3": sum(transition_hist[:4]),
            "transition_le_7": sum(transition_hist[:8]),
            "transition_ge_14": sum(transition_hist[14:]),
            "transition_le_3_fraction": (
                sum(transition_hist[:4]) / valid_masks if valid_masks else None
            ),
            "transition_ge_14_fraction": (
                sum(transition_hist[14:]) / valid_masks if valid_masks else None
            ),
        },
        "histograms": {
            "visual_on_count": on_hist,
            "transition_count": transition_hist,
            "hamming_to_all_on": hamming_all_on_hist,
            "hamming_to_minimum_route": hamming_min_hist,
            "segment_count": segment_count_hist,
            "all_segment_length": all_segment_hist,
            "on_segment_length": on_segment_hist,
            "off_segment_length": off_segment_hist,
            "pairwise_hamming": pairwise_hist,
        },
    }


def aggregate_diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = sorted({row["dataset"] for row in rows})
    return {
        "schema_version": "label_regeneration_route_diversity_p6_v1",
        "scope": {
            "included_datasets": list(DATASETS),
            "excluded_datasets": ["wemath2pro"],
            "valid_masks_only": True,
            "raw_cache_capped": False,
            "stages_included": ["P6"],
            "stages_excluded": ["P7", "P8", "P9", "P10"],
        },
        "definitions": {
            "minimum_route": "fewest visual-ON layers, then lexicographically smallest complete mask",
            "segment": "maximal contiguous run of identical ON or OFF actions across 28 layers",
            "pairwise_hamming": "all exact unordered pairs of valid masks within the same sample",
            "sample_balanced": "one per-sample mean contributes per positive sample",
            "quantiles": "linear/type-7 empirical quantiles",
        },
        "overall": _aggregate(rows),
        "by_dataset": {
            dataset: _aggregate([row for row in rows if row["dataset"] == dataset])
            for dataset in datasets
        },
        "by_dataset_and_current_status": {
            dataset: {
                status: _aggregate(
                    [
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and row["current_all_on_status"] == status
                    ]
                )
                for status in STATUSES
            }
            for dataset in datasets
        },
    }
