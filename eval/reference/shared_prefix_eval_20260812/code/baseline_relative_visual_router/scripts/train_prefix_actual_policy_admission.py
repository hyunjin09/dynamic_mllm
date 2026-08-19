#!/usr/bin/env python3
"""Select a shared-prefix depth and actual-policy admission rule on calibration."""

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
from baseline_relative_visual_router.admission import calibrate_threshold, summarize_admission
from baseline_relative_visual_router.input_admission import (
    compose_admission_score,
    fixed_uid_train_calibration_split,
    load_prefix_feature_cache,
    prefix_feature_matrix,
)
from train_input_actual_policy_admission import (
    ARCHITECTURES,
    BETAS,
    RESCUE_WEIGHTS,
    SEEDS,
    fit_member,
    inner_split,
    labels_for,
    member_probabilities,
    policy_rows,
    refit_member,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-layers", default="2,4,8")
    parser.add_argument("--expected-count", type=int, default=22349)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epsilon", type=float, default=0.002)
    parser.add_argument("--split-seed", type=int, default=20260812)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--process-name", default="brvr-prefix-admission")
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("prefix-layers must be a non-empty unique CSV")
    return result


def calibration_quality(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    brier = float(np.mean((scores - labels) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        keep = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]
        )
        if keep.any():
            ece += float(keep.mean()) * abs(float(scores[keep].mean()) - float(labels[keep].mean()))
    return {"brier": brier, "ece_10": float(ece)}


def choose_candidates(
    rows: list[dict[str, Any]],
    members: dict[str, dict[str, np.ndarray]],
    *,
    epsilon: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = []
    for harm_architecture in ARCHITECTURES:
        harm_members = members["harm"][harm_architecture]
        for harm_beta in BETAS:
            harm_score = harm_members.mean(0) + harm_beta * harm_members.std(0)
            safe_point, _ = calibrate_threshold(rows, harm_score, epsilon=epsilon)
            for rescue_architecture in ARCHITECTURES:
                rescue_members = members["rescue"][rescue_architecture]
                for utility_beta in BETAS:
                    for rescue_weight in RESCUE_WEIGHTS:
                        for score_mode in ("harm_only", "utility_only", "hierarchical"):
                            effective = compose_admission_score(
                                score_mode,
                                harm_members,
                                rescue_members,
                                harm_beta=harm_beta,
                                harm_threshold=float(safe_point["threshold"]),
                                utility_beta=utility_beta,
                                rescue_weight=rescue_weight,
                            )
                            efficiency, sweep = calibrate_threshold(
                                rows, effective, epsilon=epsilon
                            )
                            feasible = [
                                row
                                for row in sweep
                                if row["accuracy_delta_one_sided_95_lcb"] >= -epsilon
                                and row["routed_count"] > 0
                            ]
                            if not feasible:
                                continue
                            accuracy = max(
                                feasible,
                                key=lambda row: (
                                    row["selected_accuracy"],
                                    row["route_sensitive_layer_saving_fraction"],
                                ),
                            )
                            candidates.append(
                                {
                                    "score_mode": score_mode,
                                    "harm_architecture": harm_architecture,
                                    "harm_beta": harm_beta,
                                    "harm_threshold": float(safe_point["threshold"]),
                                    "rescue_architecture": rescue_architecture,
                                    "utility_beta": utility_beta,
                                    "rescue_weight": rescue_weight,
                                    "efficiency_point": efficiency,
                                    "accuracy_point": accuracy,
                                }
                            )
    if not candidates:
        raise RuntimeError("no non-empty prefix admission candidate satisfies calibration")
    efficiency = max(
        candidates,
        key=lambda row: (
            row["efficiency_point"]["route_sensitive_layer_saving_fraction"],
            row["efficiency_point"]["selected_accuracy"],
        ),
    )
    accuracy = max(
        candidates,
        key=lambda row: (
            row["accuracy_point"]["selected_accuracy"],
            row["accuracy_point"]["route_sensitive_layer_saving_fraction"],
        ),
    )
    return efficiency, accuracy


def candidate_admission(
    features: torch.Tensor,
    states: dict[str, dict[str, list[dict[str, torch.Tensor]]]],
    candidate: dict[str, Any],
    *,
    device: torch.device,
    point_name: str,
) -> np.ndarray:
    harm = member_probabilities(
        features,
        states["harm"][candidate["harm_architecture"]],
        candidate["harm_architecture"],
        device,
    )
    rescue = member_probabilities(
        features,
        states["rescue"][candidate["rescue_architecture"]],
        candidate["rescue_architecture"],
        device,
    )
    score = compose_admission_score(
        str(candidate["score_mode"]),
        harm,
        rescue,
        harm_beta=float(candidate["harm_beta"]),
        harm_threshold=float(candidate["harm_threshold"]),
        utility_beta=float(candidate["utility_beta"]),
        rescue_weight=float(candidate["rescue_weight"]),
    )
    if point_name not in {"accuracy_point", "efficiency_point"}:
        raise ValueError(f"unknown candidate point: {point_name}")
    return score <= float(candidate[point_name]["threshold"])


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    prefixes = parse_csv_ints(args.prefix_layers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    by_prefix: dict[str, Any] = {}
    selected_states: dict[int, dict[str, dict[str, list[dict[str, torch.Tensor]]]]] = {}
    expected_uids: list[str] | None = None

    for prefix in prefixes:
        tensors, metadata = load_prefix_feature_cache(
            args.feature_root / f"prefix_{prefix:02d}",
            expected_prefix_layers=prefix,
        )
        if len(metadata) != args.expected_count:
            raise RuntimeError(f"K={prefix}: expected {args.expected_count}, found {len(metadata)}")
        uids = [str(row["uid"]) for row in metadata]
        if expected_uids is None:
            expected_uids = uids
        elif uids != expected_uids:
            raise RuntimeError(f"K={prefix}: UID order/set differs from other prefix depths")
        features = prefix_feature_matrix(tensors)
        split = fixed_uid_train_calibration_split(
            metadata, train_fraction=0.8, seed=args.split_seed
        )
        states: dict[str, dict[str, list[dict[str, torch.Tensor]]]] = {}
        training: dict[str, Any] = {}
        calibration_members: dict[str, dict[str, np.ndarray]] = {}
        target_metrics: dict[str, Any] = {}
        for target_index, target in enumerate(("harm", "rescue")):
            labels = labels_for(metadata, target)
            fit_indices, dev_indices = inner_split(
                split["train"], labels, metadata, args.split_seed + target_index + prefix * 10
            )
            states[target] = {}
            training[target] = {}
            calibration_members[target] = {}
            target_metrics[target] = {}
            for architecture in ARCHITECTURES:
                target_states, histories = [], []
                for seed in SEEDS:
                    print(
                        f"[prefix-gate] K={prefix} target={target} architecture={architecture} seed={seed}",
                        flush=True,
                    )
                    _, best_epoch, history = fit_member(
                        features,
                        labels,
                        fit_indices,
                        dev_indices,
                        architecture=architecture,
                        seed=seed,
                        args=args,
                    )
                    target_states.append(
                        refit_member(
                            features,
                            labels,
                            split["train"],
                            architecture=architecture,
                            seed=seed,
                            epochs=best_epoch,
                            args=args,
                        )
                    )
                    histories.append(history)
                states[target][architecture] = target_states
                training[target][architecture] = histories
                members = member_probabilities(
                    features[split["calibration"]], target_states, architecture, device
                )
                calibration_members[target][architecture] = members
                mean_score = members.mean(0)
                target_labels = labels[split["calibration"]]
                target_metrics[target][architecture] = {
                    **binary_metrics(target_labels, mean_score),
                    **calibration_quality(target_labels, mean_score),
                }
        calibration_rows = [
            policy_rows(metadata)[int(index)] for index in split["calibration"]
        ]
        efficiency, accuracy = choose_candidates(
            calibration_rows, calibration_members, epsilon=args.epsilon
        )
        all_rows = policy_rows(metadata)
        baseline_score = np.ones(len(all_rows), dtype=np.float64)
        router_score = np.zeros(len(all_rows), dtype=np.float64)
        oracle_score = np.asarray(
            [int(row["baseline_correct"] and not row["router_correct"]) for row in all_rows],
            dtype=np.float64,
        )
        by_prefix[str(prefix)] = {
            "counts": {
                "n": len(metadata),
                "outcomes": dict(Counter(str(row["outcome"]) for row in metadata)),
                "split": {key: len(value) for key, value in split.items()},
            },
            "actual_policy_full_population": {
                "all_on": summarize_admission(all_rows, baseline_score, threshold=0.5),
                "ungated_hybrid": summarize_admission(all_rows, router_score, threshold=0.5),
                "oracle_admission": summarize_admission(all_rows, oracle_score, threshold=0.5),
            },
            "target_metrics_calibration": target_metrics,
            "selected_efficiency_candidate": efficiency,
            "selected_accuracy_candidate": accuracy,
            "training": training,
        }
        selected_states[prefix] = states
        del features, tensors

    selected_efficiency_prefix = max(
        prefixes,
        key=lambda prefix: (
            by_prefix[str(prefix)]["selected_efficiency_candidate"]["efficiency_point"]
            ["route_sensitive_layer_saving_fraction"],
            by_prefix[str(prefix)]["selected_efficiency_candidate"]["efficiency_point"]
            ["selected_accuracy"],
        ),
    )
    selected_accuracy_prefix = max(
        prefixes,
        key=lambda prefix: (
            by_prefix[str(prefix)]["selected_accuracy_candidate"]["accuracy_point"]
            ["selected_accuracy"],
            by_prefix[str(prefix)]["selected_accuracy_candidate"]["accuracy_point"]
            ["route_sensitive_layer_saving_fraction"],
        ),
    )
    summary = {
        "schema_version": "shared_dense_prefix_admission_selection_v1",
        "policy": "sw31_bt_leg_s41_after_shared_dense_prefix",
        "split_contract": "fixed by UID, benchmark, and all-on correctness; independent of K-specific outcome",
        "epsilon": args.epsilon,
        "feature_contract": {
            "benchmark_feature_used": False,
            "fields": [
                "instruction_mean",
                "instruction_window_mean",
                "instruction_last",
                "visual_mean",
                "visual_mean_abs",
            ],
        },
        "prefixes": by_prefix,
        "selected_efficiency_prefix_layers": selected_efficiency_prefix,
        "selected_accuracy_prefix_layers": selected_accuracy_prefix,
    }
    selected_features, selected_metadata = load_prefix_feature_cache(
        args.feature_root / f"prefix_{selected_accuracy_prefix:02d}",
        expected_prefix_layers=selected_accuracy_prefix,
    )
    selected_split = fixed_uid_train_calibration_split(
        selected_metadata, train_fraction=0.8, seed=args.split_seed
    )
    selected_matrix = prefix_feature_matrix(selected_features)
    split_name = np.full(len(selected_metadata), "train", dtype=object)
    split_name[selected_split["calibration"]] = "calibration"
    accuracy_admission = candidate_admission(
        selected_matrix,
        selected_states[selected_accuracy_prefix],
        by_prefix[str(selected_accuracy_prefix)]["selected_accuracy_candidate"],
        device=device,
        point_name="accuracy_point",
    )
    if selected_efficiency_prefix == selected_accuracy_prefix:
        efficiency_metadata = selected_metadata
        efficiency_matrix = selected_matrix
    else:
        efficiency_features, efficiency_metadata = load_prefix_feature_cache(
            args.feature_root / f"prefix_{selected_efficiency_prefix:02d}",
            expected_prefix_layers=selected_efficiency_prefix,
        )
        efficiency_matrix = prefix_feature_matrix(efficiency_features)
    if [str(row["uid"]) for row in efficiency_metadata] != [
        str(row["uid"]) for row in selected_metadata
    ]:
        raise RuntimeError("selected accuracy/efficiency prefixes have different UIDs")
    efficiency_admission = candidate_admission(
        efficiency_matrix,
        selected_states[selected_efficiency_prefix],
        by_prefix[str(selected_efficiency_prefix)]["selected_efficiency_candidate"],
        device=device,
        point_name="efficiency_point",
    )

    checkpoint = {
        "schema_version": summary["schema_version"],
        "input_size": int(selected_matrix.shape[1]),
        "prefix_states": {
            prefix: selected_states[prefix]
            for prefix in sorted({selected_efficiency_prefix, selected_accuracy_prefix})
        },
        "selected_efficiency_prefix_layers": selected_efficiency_prefix,
        "selected_accuracy_prefix_layers": selected_accuracy_prefix,
        "selected_efficiency_candidate": by_prefix[str(selected_efficiency_prefix)][
            "selected_efficiency_candidate"
        ],
        "selected_accuracy_candidate": by_prefix[str(selected_accuracy_prefix)][
            "selected_accuracy_candidate"
        ],
        "feature_contract": summary["feature_contract"],
    }
    torch.save(checkpoint, args.output_dir / "prefix_admission_selection.pt")
    (args.output_dir / "split_uids.json").write_text(
        json.dumps(
            {
                name: [str(selected_metadata[int(index)]["uid"]) for index in indices]
                for name, indices in selected_split.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "canonical_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for index, row in enumerate(selected_metadata):
            handle.write(
                json.dumps(
                    {
                        "uid": str(row["uid"]),
                        "benchmark": str(row["benchmark"]),
                        "split": str(split_name[index]),
                        "baseline_correct": bool(row["baseline_correct"]),
                        "hybrid_correct": bool(row["router_correct"]),
                        "outcome": str(row["outcome"]),
                        "selected_accuracy_prefix_layers": selected_accuracy_prefix,
                        "selected_accuracy_use_sparse": bool(accuracy_admission[index]),
                        "selected_efficiency_prefix_layers": selected_efficiency_prefix,
                        "selected_efficiency_use_sparse": bool(efficiency_admission[index]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_efficiency_prefix_layers": selected_efficiency_prefix,
                "selected_accuracy_prefix_layers": selected_accuracy_prefix,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
