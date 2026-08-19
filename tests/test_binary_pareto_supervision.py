"""Contracts for Pareto-efficient binary-route supervision."""

import pytest

from label_regeneration.pareto_supervision import build_pareto_record, filter_pareto_routes


def route(mask: str, score: float) -> dict:
    return {
        "key": mask,
        "mask": [int(bit) for bit in mask],
        "score": score,
        "num_visual_on_layers": mask.count("1"),
    }


def test_filter_removes_equal_score_routes_with_higher_visual_cost():
    routes = [route("1111", 1.0), route("1100", 1.0), route("0000", 1.0)]

    result = filter_pareto_routes(routes, expected_width=4)

    assert [item["key"] for item in result.retained] == ["0000"]
    assert result.removed_witnesses == {
        "1111": "0000",
        "1100": "0000",
    }


def test_filter_preserves_score_cost_tradeoffs():
    routes = [route("1111", 1.0), route("1100", 0.8), route("0000", 0.6)]

    result = filter_pareto_routes(routes, expected_width=4)

    assert [item["key"] for item in result.retained] == ["1111", "1100", "0000"]
    assert result.removed_witnesses == {}


def test_filter_uses_a_retained_dominance_witness_deterministically():
    routes = [route("1111", 0.8), route("1000", 0.8), route("0000", 1.0)]

    result = filter_pareto_routes(routes, expected_width=4)

    assert [item["key"] for item in result.retained] == ["0000"]
    assert set(result.removed_witnesses.values()) == {"0000"}


def test_filter_rejects_duplicate_or_malformed_routes():
    with pytest.raises(ValueError, match="duplicate"):
        filter_pareto_routes([route("1010", 1.0), route("1010", 1.0)], expected_width=4)

    malformed = route("1010", 1.0)
    malformed["num_visual_on_layers"] = 3
    with pytest.raises(ValueError, match="ON count"):
        filter_pareto_routes([malformed], expected_width=4)


def test_record_builder_preserves_population_and_assigns_group_taxonomy():
    source = {
        "uid": "gqa:1",
        "benchmark": "gqa",
        "split": "validation",
        "split_group": "gqa:image-1",
        "current_all_on_status": "correct",
        "selected_valid_route_count": 2,
        "valid_routes": [route("1111", 1.0), route("0011", 1.0)],
    }

    record, witnesses = build_pareto_record(source, expected_width=4)

    assert record["original_selected_valid_route_count"] == 2
    assert record["pareto_efficient_route_count"] == 1
    assert record["supervision_group"] == "B"
    assert record["original_valid_mask_keys"] == ["1111", "0011"]
    assert [item["key"] for item in record["valid_routes"]] == ["0011"]
    assert witnesses == {"1111": "0011"}


def test_record_builder_retains_zero_positive_rows_for_population_accounting():
    source = {
        "uid": "chartqa:1",
        "benchmark": "chartqa",
        "split": "train",
        "split_group": "chartqa:image-1",
        "current_all_on_status": "wrong",
        "selected_valid_route_count": 0,
        "valid_routes": [],
    }

    record, witnesses = build_pareto_record(source, expected_width=4)

    assert record["pareto_efficient_route_count"] == 0
    assert record["supervision_group"] == "D"
    assert record["valid_routes"] == []
    assert witnesses == {}
