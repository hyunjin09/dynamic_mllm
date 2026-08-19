#!/usr/bin/env python3
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

from analysis_outputs.dense_prefill_hierarchical_gate import (  # noqa: E402
    FeatureScaler,
    load_feature_parts,
    transform_features,
)
from baseline_relative_visual_router.admission import (  # noqa: E402
    calibrate_threshold,
    outcome_label,
    summarize_admission,
)
from baseline_relative_visual_router.scripts_runtime import ensemble_probabilities  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--random-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def confidence_controls(
    calibration_rows: list[dict[str, Any]],
    calibration_confidence: np.ndarray,
    test_rows: list[dict[str, Any]],
    test_confidence: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    definitions = {
        "top1_logprob": (-calibration_confidence[:, 0], -test_confidence[:, 0]),
        "top1_top2_margin": (-calibration_confidence[:, 1], -test_confidence[:, 1]),
        "entropy": (calibration_confidence[:, 2], test_confidence[:, 2]),
    }
    candidates = []
    for name, (calibration_score, test_score) in definitions.items():
        selected, _ = calibrate_threshold(
            calibration_rows, calibration_score, epsilon=epsilon
        )
        result = summarize_admission(
            test_rows, test_score, threshold=float(selected["threshold"])
        )
        candidates.append(
            {"name": name, "calibration": selected, "locked_test": result}
        )
    return max(
        candidates,
        key=lambda row: (
            row["calibration"]["route_sensitive_layer_saving_fraction"],
            row["calibration"]["selected_accuracy"],
        ),
    ) | {"all_candidates": candidates}


def matched_stratified_random(
    rows: list[dict[str, Any]],
    learned_admission: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    baseline = np.asarray([bool(row["baseline_correct"]) for row in rows], dtype=np.int8)
    routed = np.asarray([bool(row["router_correct"]) for row in rows], dtype=np.int8)
    budgets = np.asarray(
        [int(row["selected_num_visual_on_layers"]) for row in rows], dtype=np.float64
    )
    benchmarks = np.asarray([str(row["benchmark"]) for row in rows], dtype=object)
    groups = []
    for benchmark in sorted(set(benchmarks)):
        indices = np.flatnonzero(benchmarks == benchmark)
        groups.append((indices, int(learned_admission[indices].sum())))
    deltas = np.empty(repetitions, dtype=np.float64)
    means = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        admitted = np.zeros(len(rows), dtype=bool)
        for indices, count in groups:
            if count:
                admitted[rng.choice(indices, size=count, replace=False)] = True
        selected = np.where(admitted, routed, baseline)
        selected_budget = np.where(admitted, budgets, 28.0)
        deltas[repetition] = float((selected - baseline).mean())
        means[repetition] = float(selected_budget.mean())
    return {
        "repetitions": repetitions,
        "benchmark_matched": True,
        "route_count": int(learned_admission.sum()),
        "accuracy_delta_mean": float(deltas.mean()),
        "accuracy_delta_95_interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "mean_visual_on_layers_mean": float(means.mean()),
        "mean_visual_on_layers_95_interval": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    summary = json.loads((args.experiment_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        args.experiment_dir / "admission_gate.pt", map_location="cpu", weights_only=False
    )
    tensors, metadata = load_feature_parts(Path(summary["feature_dir"]))
    policy_rows = read_jsonl(Path(summary["policy_rows"]))
    by_uid = {str(row["uid"]): row for row in policy_rows}
    ordered_rows = [by_uid[str(meta["uid"])] for meta in metadata]
    uid_to_index = {str(meta["uid"]): index for index, meta in enumerate(metadata)}
    split_indices = {
        split: np.asarray(
            [uid_to_index[uid] for uid in summary["split_uids"][split]], dtype=np.int64
        )
        for split in ("train", "calibration", "test")
    }
    scaler = FeatureScaler.from_state_dict(checkpoint["scaler"])
    features = transform_features(tensors, scaler)
    architecture = str(summary["selected_candidate"]["architecture"])
    beta = float(summary["selected_candidate"]["uncertainty_beta"])
    scores = ensemble_probabilities(
        features,
        checkpoint["models"][architecture],
        architecture,
        beta,
        torch.device(args.device),
    )
    calibration_indices = split_indices["calibration"]
    test_indices = split_indices["test"]
    calibration_rows = [ordered_rows[int(index)] for index in calibration_indices]
    test_rows = [ordered_rows[int(index)] for index in test_indices]
    threshold = float(summary["selected_candidate"]["calibration"]["threshold"])
    test_scores = scores[test_indices]
    learned_admission = test_scores <= threshold

    per_benchmark = {}
    test_benchmarks = np.asarray(
        [str(metadata[int(index)]["benchmark"]) for index in test_indices], dtype=object
    )
    for benchmark in sorted(set(test_benchmarks)):
        keep = np.flatnonzero(test_benchmarks == benchmark)
        rows = [test_rows[int(index)] for index in keep]
        benchmark_scores = test_scores[keep]
        oracle_scores = np.asarray(
            [int(outcome_label(row) == "harm") for row in rows], dtype=np.float64
        )
        per_benchmark[benchmark] = {
            "learned": summarize_admission(rows, benchmark_scores, threshold=threshold),
            "oracle": summarize_admission(rows, oracle_scores, threshold=0.5),
        }

    controls = confidence_controls(
        calibration_rows,
        tensors["confidence"][calibration_indices].numpy(),
        test_rows,
        tensors["confidence"][test_indices].numpy(),
        float(summary["epsilon"]),
    )
    random_control = matched_stratified_random(
        test_rows, learned_admission, args.random_repetitions, args.seed
    )
    output = {
        "schema_version": "actual_policy_admission_controls_v1",
        "selected_gate": {
            "architecture": architecture,
            "uncertainty_beta": beta,
            "threshold": threshold,
        },
        "per_benchmark": per_benchmark,
        "confidence_only_control": controls,
        "matched_random_control": random_control,
        "test_outcomes": dict(Counter(outcome_label(row) for row in test_rows)),
    }
    (args.experiment_dir / "controls.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.experiment_dir / "locked_test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for local_index, global_index in enumerate(test_indices):
            row = test_rows[local_index]
            record = {
                "uid": str(metadata[int(global_index)]["uid"]),
                "benchmark": str(metadata[int(global_index)]["benchmark"]),
                "outcome": outcome_label(row),
                "harm_score": float(test_scores[local_index]),
                "admit_sparse": bool(learned_admission[local_index]),
                "baseline_correct": bool(row["baseline_correct"]),
                "router_correct": bool(row["router_correct"]),
                "selected_num_visual_on_layers": int(row["selected_num_visual_on_layers"]),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
