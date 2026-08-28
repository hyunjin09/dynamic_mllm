#!/usr/bin/env python3
"""Replace the provisional route-throughput estimate after the GPU smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def calibrated_estimate(smoke: dict[str, Any]) -> dict[str, Any]:
    if smoke.get("passed") is not True:
        raise ValueError("compute calibration requires a passed smoke")
    elapsed = float(smoke["measured_smoke_body_seconds"])
    equivalents = int(smoke["qwen_route_equivalents_per_rank"])
    if elapsed <= 0 or equivalents <= 0:
        raise ValueError("invalid smoke timing calibration")
    seconds_per_route_gpu = elapsed / equivalents
    training_routes = 61_440
    validation_routes = 2 * 866 * 10
    external_routes = 2 * 14_960

    def workload(routes: int) -> dict[str, float | int]:
        wall_hours = routes * seconds_per_route_gpu / 8 / 3600
        return {
            "route_equivalents": routes,
            "estimated_wall_hours_on_8_gpus": wall_hours,
            "estimated_allocated_gpu_hours": 8 * wall_hours,
        }

    return {
        "schema_version": "four_action_online_router_compute_estimate_v1",
        "basis": "passed 8-GPU smoke body; model-load overhead excluded",
        "calibration": {
            "elapsed_seconds": elapsed,
            "route_equivalents_per_rank": equivalents,
            "seconds_per_route_equivalent_per_gpu": seconds_per_route_gpu,
        },
        "training_teacher_replay": workload(training_routes),
        "epoch_validation_teacher_plus_routed": workload(validation_routes),
        "external_routed_plus_unified_full": workload(external_routes),
        "combined": workload(training_routes + validation_routes + external_routes),
        "caveats": [
            "External generation length and visual-token geometry may differ from smoke.",
            "Training has router backward/optimizer overhead absent from a route-equivalent calibration.",
            "The smoke body excludes eight-way model loading and final CPU bootstrap analysis.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrated_estimate(json.loads(args.smoke.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
