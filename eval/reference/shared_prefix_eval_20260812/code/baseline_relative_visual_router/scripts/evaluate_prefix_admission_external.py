#!/usr/bin/env python3
"""Evaluate a calibration-selected shared-prefix gate on external task outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "scripts"))

from analysis_outputs.dense_prefill_hierarchical_gate import binary_metrics
from baseline_relative_visual_router.admission import summarize_admission
from baseline_relative_visual_router.input_admission import (
    compose_admission_score,
    load_prefix_feature_cache,
    prefix_feature_matrix,
)
from train_input_actual_policy_admission import (
    bootstrap,
    decision_summary,
    labels_for,
    matched_random_control,
    member_probabilities,
    policy_rows,
)
from train_prefix_actual_policy_admission import calibration_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-checkpoint", type=Path, required=True)
    parser.add_argument("--external-feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=5807)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--random-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def admission_from_candidate(
    features: torch.Tensor,
    states: dict[str, dict[str, list[dict[str, torch.Tensor]]]],
    candidate: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    members: dict[str, dict[str, np.ndarray]] = {"harm": {}, "rescue": {}}
    for target in ("harm", "rescue"):
        for architecture in {candidate[f"{target}_architecture"]}:
            members[target][architecture] = member_probabilities(
                features, states[target][architecture], architecture, device
            )
    harm = members["harm"][candidate["harm_architecture"]]
    rescue = members["rescue"][candidate["rescue_architecture"]]
    effective = compose_admission_score(
        str(candidate["score_mode"]),
        harm,
        rescue,
        harm_beta=float(candidate["harm_beta"]),
        harm_threshold=float(candidate["harm_threshold"]),
        utility_beta=float(candidate["utility_beta"]),
        rescue_weight=float(candidate["rescue_weight"]),
    )
    admission = effective <= float(candidate["accuracy_point"]["threshold"])
    return admission, members


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.selection_checkpoint, map_location="cpu", weights_only=False)
    prefix = int(checkpoint["selected_accuracy_prefix_layers"])
    candidate = checkpoint["selected_accuracy_candidate"]
    states = checkpoint["prefix_states"][prefix]
    tensors, metadata = load_prefix_feature_cache(
        args.external_feature_root / f"prefix_{prefix:02d}",
        expected_prefix_layers=prefix,
    )
    if len(metadata) != args.expected_count:
        raise RuntimeError(f"expected {args.expected_count} external rows, found {len(metadata)}")
    features = prefix_feature_matrix(tensors)
    device = torch.device(args.device)
    admission, members = admission_from_candidate(
        features, states, candidate, device=device
    )
    rows = policy_rows(metadata)
    baseline_score = np.ones(len(rows), dtype=np.float64)
    router_score = np.zeros(len(rows), dtype=np.float64)
    oracle_score = np.asarray(
        [int(row["baseline_correct"] and not row["router_correct"]) for row in rows],
        dtype=np.float64,
    )
    result = {
        "all_on": summarize_admission(rows, baseline_score, threshold=0.5),
        "ungated_hybrid": summarize_admission(rows, router_score, threshold=0.5),
        "oracle_admission": summarize_admission(rows, oracle_score, threshold=0.5),
        "learned_admission": decision_summary(rows, admission),
    }
    result["learned_admission"].update(
        bootstrap(rows, admission, args.bootstrap_repetitions, args.seed)
    )
    result["matched_random_admission"] = matched_random_control(
        rows, admission, args.random_repetitions, args.seed + 1
    )
    result["by_benchmark"] = {}
    benchmarks = np.asarray([str(row["benchmark"]) for row in rows], dtype=object)
    for benchmark in sorted(set(benchmarks)):
        keep = np.flatnonzero(benchmarks == benchmark)
        subset = [rows[int(index)] for index in keep]
        result["by_benchmark"][benchmark] = {
            "all_on": decision_summary(subset, np.zeros(len(subset), dtype=bool)),
            "ungated_hybrid": decision_summary(subset, np.ones(len(subset), dtype=bool)),
            "learned_admission": decision_summary(subset, admission[keep]),
            "oracle_admission": summarize_admission(
                subset, oracle_score[keep], threshold=0.5
            ),
        }
    target_metrics = {}
    for target in ("harm", "rescue"):
        architecture = candidate[f"{target}_architecture"]
        scores = members[target][architecture].mean(0)
        labels = labels_for(metadata, target)
        target_metrics[target] = {
            "architecture": architecture,
            **binary_metrics(labels, scores),
            **calibration_quality(labels, scores),
        }
    summary = {
        "schema_version": "shared_dense_prefix_external_evaluation_v1",
        "selected_prefix_layers": prefix,
        "candidate": candidate,
        "external_counts": {
            "n": len(metadata),
            "outcomes": dict(Counter(str(row["outcome"]) for row in metadata)),
        },
        "target_metrics": target_metrics,
        "external_test": result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "external_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(metadata):
            handle.write(
                json.dumps(
                    {
                        "uid": str(row["uid"]),
                        "benchmark": str(row["benchmark"]),
                        "outcome": str(row["outcome"]),
                        "baseline_correct": bool(row["baseline_correct"]),
                        "router_correct": bool(row["router_correct"]),
                        "selected_num_visual_on_layers": int(row["selected_num_visual_on_layers"]),
                        "use_sparse_hybrid": bool(admission[index]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
