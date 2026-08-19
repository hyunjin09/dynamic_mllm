#!/usr/bin/env python3
"""Train input-only SW31 treatment admission and evaluate on external tasks."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE / "src"))

from analysis_outputs.dense_prefill_hierarchical_gate import GateHead, binary_metrics
from baseline_relative_visual_router.admission import calibrate_threshold, summarize_admission
from baseline_relative_visual_router.input_admission import (
    input_feature_matrix,
    load_input_feature_cache,
    override_scores_with_safe_admission,
    stratified_train_calibration_split,
)
from baseline_relative_visual_router.utility import conservative_utility_score


ARCHITECTURES = ("linear", "mlp")
SEEDS = (17, 41, 73)
BETAS = (0.0, 1.0, 2.0)
RESCUE_WEIGHTS = (0.5, 1.0, 2.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-feature-dir", type=Path, required=True)
    parser.add_argument("--external-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epsilon", type=float, default=0.002)
    parser.add_argument("--split-seed", type=int, default=20260812)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--random-repetitions", type=int, default=5000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--process-name", default="brvr-input-admission")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def labels_for(metadata: list[dict[str, Any]], target: str) -> np.ndarray:
    return np.asarray([int(str(row["outcome"]) == target) for row in metadata], dtype=np.int64)


def policy_rows(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "baseline_correct": bool(row["baseline_correct"]),
            "router_correct": bool(row["router_correct"]),
            "selected_num_visual_on_layers": int(row["selected_num_visual_on_layers"]),
            "benchmark": str(row["benchmark"]),
        }
        for row in metadata
    ]


def inner_split(
    indices: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, int], list[int]] = {}
    for index in indices:
        key = (str(metadata[int(index)]["benchmark"]), int(labels[int(index)]))
        groups.setdefault(key, []).append(int(index))
    fit: list[int] = []
    dev: list[int] = []
    for key in sorted(groups):
        values = np.asarray(groups[key], dtype=np.int64)
        if len(values) < 2:
            raise ValueError(f"inner stratum {key!r} needs at least two rows")
        rng.shuffle(values)
        count = min(len(values) - 1, max(1, int(round(0.1 * len(values)))))
        dev.extend(values[:count].tolist())
        fit.extend(values[count:].tolist())
    return np.asarray(sorted(fit)), np.asarray(sorted(dev))


def make_model(architecture: str, input_size: int) -> GateHead:
    return GateHead(architecture, input_size=input_size, hidden_size=256)


def fit_member(
    features: torch.Tensor,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    *,
    architecture: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], int, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(args.device)
    model = make_model(architecture, int(features.shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    target = torch.as_tensor(labels[fit_indices], dtype=torch.float32)
    positives = float(target.sum())
    negatives = float(len(target) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / max(positives, 1.0), device=device)
    )
    loader = DataLoader(
        TensorDataset(features[fit_indices], target),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    best_state = None
    best_epoch = 0
    best_auprc = -math.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            score = torch.sigmoid(model(features[dev_indices].to(device))).cpu().numpy()
        metrics = binary_metrics(labels[dev_indices], score)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "dev": metrics})
        if metrics["auprc"] > best_auprc + 1e-5:
            best_auprc = float(metrics["auprc"])
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("no input admission checkpoint was produced")
    return best_state, best_epoch, {
        "best_epoch": best_epoch,
        "best_dev_auprc": best_auprc,
        "history": history,
    }


def refit_member(
    features: torch.Tensor,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    architecture: str,
    seed: int,
    epochs: int,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = make_model(architecture, int(features.shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    target = torch.as_tensor(labels[indices], dtype=torch.float32)
    positives = float(target.sum())
    negatives = float(len(target) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / max(positives, 1.0), device=device)
    )
    loader = DataLoader(
        TensorDataset(features[indices], target),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@torch.inference_mode()
def member_probabilities(
    features: torch.Tensor,
    states: list[dict[str, torch.Tensor]],
    architecture: str,
    device: torch.device,
) -> np.ndarray:
    members = []
    for state in states:
        model = make_model(architecture, int(features.shape[1])).to(device)
        model.load_state_dict(state)
        model.eval()
        output = []
        for start in range(0, len(features), 512):
            output.append(
                torch.sigmoid(model(features[start : start + 512].to(device))).cpu()
            )
        members.append(torch.cat(output).numpy())
    return np.stack(members)


def bootstrap(
    rows: list[dict[str, Any]], admission: np.ndarray, repetitions: int, seed: int
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    score = np.where(admission, 0.0, 1.0)
    deltas, budgets = [], []
    for _ in range(repetitions):
        index = rng.integers(0, len(rows), len(rows))
        sampled = summarize_admission(
            [rows[int(value)] for value in index], score[index], threshold=0.5
        )
        deltas.append(sampled["accuracy_delta"])
        budgets.append(sampled["mean_visual_on_layers"])
    return {
        "accuracy_delta_95_ci": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "mean_visual_on_layers_95_ci": np.quantile(budgets, [0.025, 0.975]).tolist(),
    }


def decision_summary(rows: list[dict[str, Any]], admission: np.ndarray) -> dict[str, Any]:
    return summarize_admission(rows, np.where(admission, 0.0, 1.0), threshold=0.5)


def matched_random_control(
    rows: list[dict[str, Any]], admission: np.ndarray, repetitions: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    benchmarks = np.asarray([str(row["benchmark"]) for row in rows], dtype=object)
    groups = []
    for benchmark in sorted(set(benchmarks)):
        indices = np.flatnonzero(benchmarks == benchmark)
        groups.append((indices, int(admission[indices].sum())))
    deltas = []
    for _ in range(repetitions):
        random_admission = np.zeros(len(rows), dtype=bool)
        for indices, count in groups:
            if count:
                random_admission[rng.choice(indices, count, replace=False)] = True
        deltas.append(decision_summary(rows, random_admission)["accuracy_delta"])
    return {
        "benchmark_matched": True,
        "repetitions": repetitions,
        "accuracy_delta_mean": float(np.mean(deltas)),
        "accuracy_delta_95_interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    result = summary["external_test"]
    lines = [
        "# Input-Only Actual-Policy Admission",
        "",
        "The gate is calibrated on the canonical natural population and evaluated once",
        "on unseen MMStar/MMMU tasks. It uses only pre-language-layer instruction/image",
        "embeddings and no benchmark identity.",
        "",
        "| Policy | Accuracy | Delta | Routed | Harm/Rescue | Mean ON |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("all_on", "ungated_sparse", "oracle", "efficiency_first", "accuracy_first"):
        row = result[name]
        lines.append(
            f"| {name} | {100 * row['selected_accuracy']:.2f}% | "
            f"{100 * row['accuracy_delta']:+.2f}pp | {100 * row['route_fraction']:.2f}% | "
            f"{row['harm_count']}/{row['rescue_count']} | {row['mean_visual_on_layers']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_tensors, train_metadata = load_input_feature_cache(args.train_feature_dir)
    external_tensors, external_metadata = load_input_feature_cache(args.external_feature_dir)
    train_uids = {str(row["uid"]) for row in train_metadata}
    external_uids = {str(row["uid"]) for row in external_metadata}
    if train_uids & external_uids:
        raise RuntimeError("canonical training and external test UIDs overlap")
    features = input_feature_matrix(train_tensors)
    external_features = input_feature_matrix(external_tensors)
    if features.shape[1] != external_features.shape[1]:
        raise RuntimeError("training and external input feature dimensions differ")
    split = stratified_train_calibration_split(
        train_metadata, train_fraction=0.8, seed=args.split_seed
    )
    device = torch.device(args.device)
    states: dict[str, dict[str, list[dict[str, torch.Tensor]]]] = {}
    training: dict[str, Any] = {}
    calibration_members: dict[str, dict[str, np.ndarray]] = {}
    external_members: dict[str, dict[str, np.ndarray]] = {}
    for target_index, target in enumerate(("harm", "rescue")):
        labels = labels_for(train_metadata, target)
        fit_indices, dev_indices = inner_split(
            split["train"], labels, train_metadata, args.split_seed + target_index + 1
        )
        states[target] = {}
        training[target] = {}
        calibration_members[target] = {}
        external_members[target] = {}
        for architecture in ARCHITECTURES:
            target_states, histories = [], []
            for seed in SEEDS:
                print(f"[input-gate] target={target} architecture={architecture} seed={seed}", flush=True)
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
                print(
                    f"[input-gate] done epoch={best_epoch} dev_auprc={history['best_dev_auprc']:.4f}",
                    flush=True,
                )
            states[target][architecture] = target_states
            training[target][architecture] = histories
            calibration_members[target][architecture] = member_probabilities(
                features[split["calibration"]], target_states, architecture, device
            )
            external_members[target][architecture] = member_probabilities(
                external_features, target_states, architecture, device
            )

    calibration_rows = [policy_rows(train_metadata)[int(index)] for index in split["calibration"]]
    external_rows = policy_rows(external_metadata)
    candidates = []
    for harm_architecture in ARCHITECTURES:
        harm_calibration_members = calibration_members["harm"][harm_architecture]
        for harm_beta in BETAS:
            harm_score = harm_calibration_members.mean(0) + harm_beta * harm_calibration_members.std(0)
            safe_point, _ = calibrate_threshold(
                calibration_rows, harm_score, epsilon=args.epsilon
            )
            safe_admission = harm_score <= float(safe_point["threshold"])
            for rescue_architecture in ARCHITECTURES:
                rescue_calibration_members = calibration_members["rescue"][rescue_architecture]
                for utility_beta in BETAS:
                    for rescue_weight in RESCUE_WEIGHTS:
                        utility = conservative_utility_score(
                            harm_calibration_members,
                            rescue_calibration_members,
                            uncertainty_beta=utility_beta,
                            rescue_weight=rescue_weight,
                        )
                        effective = override_scores_with_safe_admission(utility, safe_admission)
                        efficiency, sweep = calibrate_threshold(
                            calibration_rows, effective, epsilon=args.epsilon
                        )
                        feasible = [
                            row
                            for row in sweep
                            if row["accuracy_delta_one_sided_95_lcb"] >= -args.epsilon
                            and row["routed_count"] > 0
                        ]
                        accuracy = max(
                            feasible,
                            key=lambda row: (
                                row["selected_accuracy"],
                                row["route_sensitive_layer_saving_fraction"],
                            ),
                        )
                        candidates.append(
                            {
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
    efficiency_candidate = max(
        candidates,
        key=lambda row: (
            row["efficiency_point"]["route_sensitive_layer_saving_fraction"],
            row["efficiency_point"]["selected_accuracy"],
        ),
    )
    accuracy_candidate = max(
        candidates,
        key=lambda row: (
            row["accuracy_point"]["selected_accuracy"],
            row["accuracy_point"]["route_sensitive_layer_saving_fraction"],
        ),
    )

    def external_admission(candidate: dict[str, Any], point: str) -> np.ndarray:
        harm = external_members["harm"][candidate["harm_architecture"]]
        rescue = external_members["rescue"][candidate["rescue_architecture"]]
        harm_score = harm.mean(0) + float(candidate["harm_beta"]) * harm.std(0)
        safe = harm_score <= float(candidate["harm_threshold"])
        utility = conservative_utility_score(
            harm,
            rescue,
            uncertainty_beta=float(candidate["utility_beta"]),
            rescue_weight=float(candidate["rescue_weight"]),
        )
        effective = override_scores_with_safe_admission(utility, safe)
        return effective <= float(candidate[point]["threshold"])

    efficiency_admission = external_admission(efficiency_candidate, "efficiency_point")
    accuracy_admission = external_admission(accuracy_candidate, "accuracy_point")
    baseline_score = np.ones(len(external_rows), dtype=np.float64)
    router_score = np.zeros(len(external_rows), dtype=np.float64)
    oracle_score = np.asarray(
        [int(row["baseline_correct"] and not row["router_correct"]) for row in external_rows],
        dtype=np.float64,
    )
    external_test = {
        "all_on": summarize_admission(external_rows, baseline_score, threshold=0.5),
        "ungated_sparse": summarize_admission(external_rows, router_score, threshold=0.5),
        "oracle": summarize_admission(external_rows, oracle_score, threshold=0.5),
        "efficiency_first": decision_summary(external_rows, efficiency_admission),
        "accuracy_first": decision_summary(external_rows, accuracy_admission),
    }
    external_test["efficiency_first"].update(
        bootstrap(external_rows, efficiency_admission, args.bootstrap_repetitions, args.split_seed + 10)
    )
    external_test["accuracy_first"].update(
        bootstrap(external_rows, accuracy_admission, args.bootstrap_repetitions, args.split_seed + 11)
    )
    external_test["efficiency_random_control"] = matched_random_control(
        external_rows, efficiency_admission, args.random_repetitions, args.split_seed + 12
    )
    external_test["by_benchmark"] = {}
    benchmarks = np.asarray([str(row["benchmark"]) for row in external_rows], dtype=object)
    for benchmark in sorted(set(benchmarks)):
        keep = np.flatnonzero(benchmarks == benchmark)
        rows = [external_rows[int(index)] for index in keep]
        external_test["by_benchmark"][benchmark] = {
            "all_on": decision_summary(rows, np.zeros(len(rows), dtype=bool)),
            "efficiency_first": decision_summary(rows, efficiency_admission[keep]),
            "accuracy_first": decision_summary(rows, accuracy_admission[keep]),
            "oracle": summarize_admission(rows, oracle_score[keep], threshold=0.5),
        }
    external_test["target_metrics"] = {
        target: {
            architecture: binary_metrics(
                labels_for(external_metadata, target), members.mean(0)
            )
            for architecture, members in external_members[target].items()
        }
        for target in ("harm", "rescue")
    }
    prediction_path = args.output_dir / "external_test_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for index, (metadata, row) in enumerate(zip(external_metadata, external_rows)):
            prediction = {
                "uid": str(metadata["uid"]),
                "benchmark": str(row["benchmark"]),
                "outcome": str(metadata["outcome"]),
                "baseline_correct": bool(row["baseline_correct"]),
                "router_correct": bool(row["router_correct"]),
                "router_visual_on_layers": int(row["selected_num_visual_on_layers"]),
                "efficiency_first_use_sparse": bool(efficiency_admission[index]),
                "accuracy_first_use_sparse": bool(accuracy_admission[index]),
            }
            handle.write(json.dumps(prediction, sort_keys=True) + "\n")
    summary = {
        "schema_version": "input_actual_policy_admission_v1",
        "policy": "sw31_bt_leg_s41",
        "feature_contract": {
            "stage": "pre_language_layer_0",
            "fields": [
                "instruction_mean",
                "instruction_last",
                "visual_mean",
                "visual_mean_abs",
            ],
            "benchmark_feature_used": False,
            "one_pass_deployable": True,
        },
        "train_feature_dir": str(args.train_feature_dir),
        "external_feature_dir": str(args.external_feature_dir),
        "external_prediction_path": str(prediction_path),
        "canonical_counts": {
            "n": len(train_metadata),
            "outcomes": dict(Counter(str(row["outcome"]) for row in train_metadata)),
            "split": {key: len(value) for key, value in split.items()},
        },
        "external_counts": {
            "n": len(external_metadata),
            "outcomes": dict(Counter(str(row["outcome"]) for row in external_metadata)),
        },
        "epsilon": args.epsilon,
        "training": training,
        "selected_efficiency_candidate": efficiency_candidate,
        "selected_accuracy_candidate": accuracy_candidate,
        "external_test": external_test,
    }
    torch.save(
        {
            "schema_version": summary["schema_version"],
            "input_size": int(features.shape[1]),
            "states": states,
            "selected_efficiency_candidate": efficiency_candidate,
            "selected_accuracy_candidate": accuracy_candidate,
            "feature_contract": summary["feature_contract"],
        },
        args.output_dir / "input_admission_gate.pt",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.md", summary)
    print(json.dumps(external_test, indent=2), flush=True)


if __name__ == "__main__":
    main()
