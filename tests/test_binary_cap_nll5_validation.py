"""Prospective contracts for CAP26/CAP24 executed-validation selection."""

from experiments.evaluate_binary_cap_validation_epochs import select_epoch
from experiments.prepare_binary_cap_nll5_supervision import build_matched_records


def _epoch(epoch: int, accuracy: float, mean_on: float, nll: float) -> dict:
    return {
        "epoch": epoch,
        "summary": {
            "overall": {
                "predicted_mask_accuracy": accuracy,
                "average_visual_on_layers": mean_on,
            }
        },
        "validation_objective_loss": nll,
    }


def _route(mask: str) -> dict:
    return {
        "key": mask,
        "mask": [int(bit) for bit in mask],
        "num_visual_on_layers": mask.count("1"),
    }


def _row(uid: str, routes: list[dict]) -> dict:
    return {
        "uid": uid,
        "benchmark": "gqa",
        "split": "train",
        "split_group": f"image:{uid}",
        "current_all_on_status": "correct",
        "selected_valid_route_count": len(routes),
        "valid_routes": routes,
    }


def test_executed_accuracy_is_primary_epoch_selector():
    selected = select_epoch([
        _epoch(1, 0.70, 8.0, 1.0),
        _epoch(2, 0.71, 27.0, 2.0),
    ])
    assert selected["epoch"] == 2


def test_epoch_selector_uses_frozen_tie_breakers_in_order():
    selected = select_epoch([
        _epoch(1, 0.70, 12.0, 0.9),
        _epoch(2, 0.70, 10.0, 1.1),
        _epoch(3, 0.70, 10.0, 0.8),
        _epoch(4, 0.70, 10.0, 0.8),
    ])
    assert selected["epoch"] == 3


def test_cap24_common_population_is_shared_without_fallback():
    rows = [
        _row("a", [_route("1111"), _route("1100")]),
        _row("b", [_route("1111")]),
    ]

    cap26, common26 = build_matched_records(rows, cap=3, common_cap=2, width=4)
    cap24, common24 = build_matched_records(rows, cap=2, common_cap=2, width=4)

    assert common26 == common24 == {"a"}
    assert [route["key"] for route in cap26[0]["valid_routes"]] == ["1100"]
    assert [route["key"] for route in cap24[0]["valid_routes"]] == ["1100"]
    assert cap26[1]["valid_routes"] == cap24[1]["valid_routes"] == []
