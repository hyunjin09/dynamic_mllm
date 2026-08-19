"""Contracts for absolute VISUAL_ON-cap supervision."""

from label_regeneration.cap_supervision import build_cap_record, filter_routes_by_cap


def _route(mask: str) -> dict:
    return {
        "key": mask,
        "mask": [int(bit) for bit in mask],
        "num_visual_on_layers": mask.count("1"),
        "score": 1.0,
    }


def _row(routes: list[dict]) -> dict:
    return {
        "uid": "gqa:sample",
        "benchmark": "gqa",
        "split": "train",
        "split_group": "gqa:image",
        "current_all_on_status": "correct",
        "selected_valid_route_count": len(routes),
        "valid_routes": routes,
    }


def test_cap_filter_preserves_parent_order_and_uses_only_on_count():
    routes = [_route("1111"), _route("1100"), _route("0000"), _route("1010")]

    filtered = filter_routes_by_cap(routes, cap=2, expected_width=4)

    assert [route["key"] for route in filtered] == ["1100", "0000", "1010"]


def test_cap_record_keeps_only_common_eligible_inputs_without_fallback():
    source = _row([_route("1111"), _route("1110")])

    record = build_cap_record(source, cap=3, common_eligible=False, expected_width=4)

    assert record["valid_routes"] == []
    assert record["cap_surviving_route_count_native"] == 1
    assert record["supervision_route_count"] == 0
    assert record["original_valid_mask_keys"] == ["1111", "1110"]


def test_cap_record_retains_every_surviving_route_for_common_input():
    source = _row([_route("1111"), _route("1110"), _route("0010")])

    record = build_cap_record(source, cap=3, common_eligible=True, expected_width=4)

    assert [route["key"] for route in record["valid_routes"]] == ["1110", "0010"]
    assert record["supervision_route_count"] == 2
    assert record["supervision_group"] == "B"


def test_cap_record_marks_full_wrong_correction_group():
    source = _row([_route("0010")])
    source["current_all_on_status"] = "wrong"

    record = build_cap_record(source, cap=3, common_eligible=True, expected_width=4)

    assert record["supervision_group"] == "A"
