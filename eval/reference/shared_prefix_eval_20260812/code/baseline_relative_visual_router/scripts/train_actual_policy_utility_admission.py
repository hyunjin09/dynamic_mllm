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
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from analysis_outputs.dense_prefill_hierarchical_gate import (  # noqa: E402
    binary_metrics,
    fit_scaler,
    load_feature_parts,
    transform_features,
)
from baseline_relative_visual_router.admission import (  # noqa: E402
    calibrate_threshold,
    outcome_label,
    summarize_admission,
)
from baseline_relative_visual_router.utility import conservative_utility_score  # noqa: E402
from train_actual_policy_admission import (  # noqa: E402
    ARCHITECTURES,
    DEFAULT_FEATURES,
    DEFAULT_ROWS,
    SEEDS,
    bootstrap_gate,
    inner_fit_dev,
    member_probabilities,
    read_jsonl,
    refit_member,
    stratified_three_way_split,
    train_member,
)


RESCUE_WEIGHTS = (0.5, 1.0, 2.0, 4.0)
UNCERTAINTY_BETAS = (0.0, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--policy-rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epsilon", type=float, default=0.002)
    parser.add_argument("--split-seed", type=int, default=20260812)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def feasible(sweep: list[dict[str, Any]], epsilon: float) -> list[dict[str, Any]]:
    return [
        row
        for row in sweep
        if row["accuracy_delta_one_sided_95_lcb"] >= -epsilon
        and row["routed_count"] > 0
    ]


def select_operating_points(
    rows: list[dict[str, Any]], score: np.ndarray, epsilon: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, sweep = calibrate_threshold(rows, score, epsilon=epsilon)
    valid = feasible(sweep, epsilon)
    if not valid:
        raise RuntimeError("no non-trivial feasible utility threshold")
    safety = max(
        valid,
        key=lambda row: (
            row["route_sensitive_layer_saving_fraction"],
            row["selected_accuracy"],
            -row["harm_count"],
        ),
    )
    performance = max(
        valid,
        key=lambda row: (
            row["selected_accuracy"],
            row["route_sensitive_layer_saving_fraction"],
            -row["harm_count"],
        ),
    )
    return safety, performance


def evaluate(
    rows: list[dict[str, Any]], score: np.ndarray, threshold: float, repetitions: int, seed: int
) -> dict[str, Any]:
    result = summarize_admission(rows, score, threshold=threshold)
    result.update(bootstrap_gate(rows, score, threshold, repetitions, seed))
    return result


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    try:
        import setproctitle
        setproctitle.setproctitle("brvr-utility-admission")
    except ImportError:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[utility] loading {args.feature_dir}", flush=True)
    tensors, metadata = load_feature_parts(args.feature_dir)
    by_uid = {str(row["uid"]): row for row in read_jsonl(args.policy_rows)}
    if len(by_uid) != len(metadata) or set(by_uid) != {str(row["uid"]) for row in metadata}:
        raise RuntimeError("feature/policy UID mismatch")
    rows = [by_uid[str(meta["uid"])] for meta in metadata]
    outcomes = [outcome_label(row) for row in rows]
    labels = {
        "harm": np.asarray([int(value == "harm") for value in outcomes], dtype=np.int64),
        "rescue": np.asarray([int(value == "rescue") for value in outcomes], dtype=np.int64),
    }
    splits = stratified_three_way_split(metadata, outcomes, args.split_seed)
    scaler = fit_scaler(tensors, splits["train"])
    features = transform_features(tensors, scaler)
    device = torch.device(args.device)
    states: dict[str, dict[str, list[dict[str, torch.Tensor]]]] = {}
    training: dict[str, Any] = {}
    calibration_members: dict[str, dict[str, np.ndarray]] = {}
    test_members: dict[str, dict[str, np.ndarray]] = {}
    for target, target_labels in labels.items():
        fit_indices, dev_indices = inner_fit_dev(
            splits["train"], target_labels, metadata, args.split_seed + (1 if target == "harm" else 2)
        )
        states[target] = {}
        training[target] = {}
        calibration_members[target] = {}
        test_members[target] = {}
        for architecture in ARCHITECTURES:
            members = []
            histories = []
            for seed in SEEDS:
                print(f"[utility] target={target} architecture={architecture} seed={seed}", flush=True)
                _, best_epoch, history = train_member(
                    features,
                    target_labels,
                    fit_indices,
                    dev_indices,
                    architecture=architecture,
                    seed=seed,
                    args=args,
                )
                members.append(
                    refit_member(
                        features,
                        target_labels,
                        splits["train"],
                        architecture=architecture,
                        seed=seed,
                        epochs=best_epoch,
                        args=args,
                    )
                )
                histories.append(history)
                print(
                    f"[utility] done best_epoch={best_epoch} "
                    f"dev_auprc={history['best_dev_auprc']:.4f}",
                    flush=True,
                )
            states[target][architecture] = members
            training[target][architecture] = histories
            calibration_members[target][architecture] = member_probabilities(
                features[splits["calibration"]], members, architecture, device
            )
            test_members[target][architecture] = member_probabilities(
                features[splits["test"]], members, architecture, device
            )

    calibration_rows = [rows[int(index)] for index in splits["calibration"]]
    test_rows = [rows[int(index)] for index in splits["test"]]
    candidates = []
    for harm_architecture in ARCHITECTURES:
        for rescue_architecture in ARCHITECTURES:
            for beta in UNCERTAINTY_BETAS:
                for rescue_weight in RESCUE_WEIGHTS:
                    score = conservative_utility_score(
                        calibration_members["harm"][harm_architecture],
                        calibration_members["rescue"][rescue_architecture],
                        uncertainty_beta=beta,
                        rescue_weight=rescue_weight,
                    )
                    safety, performance = select_operating_points(
                        calibration_rows, score, args.epsilon
                    )
                    candidates.append(
                        {
                            "harm_architecture": harm_architecture,
                            "rescue_architecture": rescue_architecture,
                            "uncertainty_beta": beta,
                            "rescue_weight": rescue_weight,
                            "calibration_safety": safety,
                            "calibration_performance": performance,
                        }
                    )
    safety_candidate = max(
        candidates,
        key=lambda row: (
            row["calibration_safety"]["route_sensitive_layer_saving_fraction"],
            row["calibration_safety"]["selected_accuracy"],
        ),
    )
    performance_candidate = max(
        candidates,
        key=lambda row: (
            row["calibration_performance"]["selected_accuracy"],
            row["calibration_performance"]["route_sensitive_layer_saving_fraction"],
        ),
    )

    locked_test = {}
    for name, candidate, point_key in (
        ("safety_first", safety_candidate, "calibration_safety"),
        ("performance_first", performance_candidate, "calibration_performance"),
    ):
        score = conservative_utility_score(
            test_members["harm"][candidate["harm_architecture"]],
            test_members["rescue"][candidate["rescue_architecture"]],
            uncertainty_beta=float(candidate["uncertainty_beta"]),
            rescue_weight=float(candidate["rescue_weight"]),
        )
        locked_test[name] = evaluate(
            test_rows,
            score,
            float(candidate[point_key]["threshold"]),
            args.bootstrap_repetitions,
            args.split_seed + (101 if name == "safety_first" else 102),
        )
    oracle_score = np.asarray([int(outcome_label(row) == "harm") for row in test_rows])
    locked_test["oracle"] = summarize_admission(test_rows, oracle_score, threshold=0.5)
    locked_test["target_metrics"] = {
        target: {
            architecture: binary_metrics(
                labels[target][splits["test"]], members.mean(axis=0)
            )
            for architecture, members in test_members[target].items()
        }
        for target in ("harm", "rescue")
    }
    summary = {
        "schema_version": "actual_policy_utility_admission_v1",
        "feature_contract": {
            "route": "all_visual_on",
            "layer": 27,
            "stage": "post_ffn",
            "benchmark_feature_used": False,
            "deployment_status": "two_pass_diagnostic_only",
        },
        "policy": "sw31_bt_leg_s41",
        "feature_dir": str(args.feature_dir),
        "policy_rows": str(args.policy_rows),
        "split_seed": args.split_seed,
        "split_counts": {key: len(value) for key, value in splits.items()},
        "split_uids": {
            key: [metadata[int(index)]["uid"] for index in value] for key, value in splits.items()
        },
        "outcome_counts": dict(Counter(outcomes)),
        "epsilon": args.epsilon,
        "training": training,
        "candidates": candidates,
        "selected_safety_candidate": safety_candidate,
        "selected_performance_candidate": performance_candidate,
        "locked_test": locked_test,
    }
    torch.save(
        {"scaler": scaler.state_dict(), "states": states, "summary_schema": summary["schema_version"]},
        args.output_dir / "utility_admission_gate.pt",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(locked_test, indent=2), flush=True)


if __name__ == "__main__":
    main()
