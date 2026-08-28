from __future__ import annotations

from experiments.record_route_conditioned_compute_estimate import compute_full_estimate


def test_compute_full_estimate_uses_selected_valid_cell_throughput_and_eight_gpus():
    anchor = {
        "validated_anchor_count": 100,
        "expected_new_cells_3k": 3600,
        "anchor_manifest_sha256": "anchor-hash",
    }
    pilot = {
        "selected_configuration": "two_replicas",
        "selected_replicas_per_gpu": 2,
        "configurations": [
            {
                "name": "one_replica",
                "useful_new_cells_per_second": 1.0,
            },
            {
                "name": "two_replicas",
                "useful_new_cells_per_second": 2.0,
            },
        ],
    }

    estimate = compute_full_estimate(anchor, pilot)

    assert estimate["expected_wall_seconds"] == 1800.0
    assert estimate["expected_wall_hours"] == 0.5
    assert estimate["expected_gpu_hours"] == 4.0
    assert estimate["selected_replicas_per_gpu"] == 2
    assert estimate["gpu_count"] == 8
