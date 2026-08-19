#!/usr/bin/env python3
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "src"))

from analysis_outputs.dense_prefill_hierarchical_gate import (  # noqa: E402
    GateHead,
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


DEFAULT_FEATURES = Path(
    "/mnt/hyemin/10k_dataset_mask/dense_prefill_hierarchical_gate_v1/features/heldout_legacy"
)
DEFAULT_ROWS = Path(
    "/mnt/hyemin/10k_dataset_mask/heldout_router_generation_eval/"
    "sw31_bt_leg_s41_heldout_plus_v1/merged_final/heldout_generation_rows.jsonl"
)
SEEDS = (17, 41, 73)
ARCHITECTURES = ("linear", "mlp")
UNCERTAINTY_BETAS = (0.0, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--policy-rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epsilon", type=float, default=0.002)
    parser.add_argument("--split-seed", type=int, default=20260812)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stratified_three_way_split(
    metadata: list[dict[str, Any]], outcomes: list[str], seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for index, (meta, outcome) in enumerate(zip(metadata, outcomes)):
        groups.setdefault(f"{meta['benchmark']}:{outcome}", []).append(index)
    splits = {"train": [], "calibration": [], "test": []}
    for key in sorted(groups):
        values = np.asarray(groups[key], dtype=np.int64)
        rng.shuffle(values)
        n = len(values)
        train_end = max(1, int(math.floor(0.60 * n)))
        calibration_end = max(train_end + 1, int(math.floor(0.80 * n)))
        calibration_end = min(calibration_end, n - 1)
        if train_end >= calibration_end or calibration_end >= n:
            raise RuntimeError(f"stratum {key!r} is too small for 60/20/20 splitting")
        splits["train"].extend(values[:train_end].tolist())
        splits["calibration"].extend(values[train_end:calibration_end].tolist())
        splits["test"].extend(values[calibration_end:].tolist())
    return {key: np.asarray(sorted(value), dtype=np.int64) for key, value in splits.items()}


def inner_fit_dev(
    indices: np.ndarray, labels: np.ndarray, metadata: list[dict[str, Any]], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for index in indices:
        groups.setdefault(f"{metadata[index]['benchmark']}:{labels[index]}", []).append(int(index))
    fit, dev = [], []
    for key in sorted(groups):
        values = np.asarray(groups[key], dtype=np.int64)
        rng.shuffle(values)
        dev_n = max(1, int(round(0.10 * len(values))))
        if dev_n >= len(values):
            raise RuntimeError(f"inner stratum {key!r} is too small")
        dev.extend(values[:dev_n].tolist())
        fit.extend(values[dev_n:].tolist())
    return np.asarray(sorted(fit)), np.asarray(sorted(dev))


def train_member(
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
    model = GateHead(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    fit_y = torch.as_tensor(labels[fit_indices], dtype=torch.float32)
    positives = float(fit_y.sum())
    negatives = float(len(fit_y) - positives)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / max(positives, 1.0), device=device)
    )
    loader = DataLoader(
        TensorDataset(features[fit_indices], fit_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    best_state = None
    best_auprc = -math.inf
    best_epoch = 0
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
            dev_score = torch.sigmoid(model(features[dev_indices].to(device))).cpu().numpy()
        metrics = binary_metrics(labels[dev_indices], dev_score)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "dev": metrics})
        if metrics["auprc"] > best_auprc + 1e-5:
            best_auprc = metrics["auprc"]
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    return best_state, best_epoch, {"best_epoch": best_epoch, "best_dev_auprc": best_auprc, "history": history}


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
    model = GateHead(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    target = torch.as_tensor(labels[indices], dtype=torch.float32)
    positives = float(target.sum())
    negatives = float(len(target) - positives)
    loss_fn = nn.BCEWithLogitsLoss(
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
    output = []
    for state in states:
        model = GateHead(architecture).to(device)
        model.load_state_dict(state)
        model.eval()
        batches = []
        for start in range(0, len(features), 512):
            batches.append(torch.sigmoid(model(features[start : start + 512].to(device))).cpu())
        output.append(torch.cat(batches).numpy())
    return np.stack(output)


def ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        keep = (scores >= left) & (scores < right if right < 1.0 else scores <= right)
        if keep.any():
            total += keep.mean() * abs(float(labels[keep].mean()) - float(scores[keep].mean()))
    return float(total)


def bootstrap_gate(
    rows: list[dict[str, Any]], scores: np.ndarray, threshold: float, repetitions: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    delta, budget = [], []
    for _ in range(repetitions):
        index = rng.integers(0, len(rows), size=len(rows))
        sampled_rows = [rows[int(i)] for i in index]
        summary = summarize_admission(sampled_rows, scores[index], threshold=threshold)
        delta.append(summary["accuracy_delta"])
        budget.append(summary["mean_visual_on_layers"])
    return {
        "accuracy_delta_95_ci": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
        "mean_visual_on_layers_95_ci": [float(np.quantile(budget, 0.025)), float(np.quantile(budget, 0.975))],
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    test = summary["locked_test"]
    lines = [
        "# Actual-Policy Conservative Admission Diagnostic",
        "",
        "Frozen treatment: `sw31_bt_leg_s41`. Gate input is all-on L27 post-FFN",
        "instruction mean/last plus answer-start confidence. Benchmark identity is excluded.",
        "Because the feature requires a complete dense prefill, this is a two-pass treatment",
        "predictability diagnostic, not a compute-saving deployment result.",
        "",
        "| Policy | Accuracy | Delta vs all-on | Harm | Rescue | Routed | Mean ON |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all_on", "ungated_sparse", "oracle_admission", "learned_admission"):
        row = test[name]
        lines.append(
            f"| {name} | {100 * row['selected_accuracy']:.2f}% | "
            f"{100 * row['accuracy_delta']:+.2f}pp | {row['harm_count']} | "
            f"{row['rescue_count']} | {100 * row['route_fraction']:.2f}% | "
            f"{row['mean_visual_on_layers']:.2f} |"
        )
    metrics = test["harm_detection"]
    lines += [
        "",
        f"Selected gate: `{summary['selected_candidate']['architecture']}`, "
        f"uncertainty beta `{summary['selected_candidate']['uncertainty_beta']}`.",
        f"Locked-test harm AUROC/AUPRC: {metrics['auroc']:.3f}/{metrics['auprc']:.3f}.",
        "",
        "The locked test is a deterministic 20% split of an already opened canonical",
        "heldout set. It is valid as a method-feasibility diagnostic, not as final test evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    try:
        import setproctitle
        setproctitle.setproctitle("brvr-actual-policy-gate")
    except ImportError:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[admission] loading features from {args.feature_dir}", flush=True)
    tensors, metadata = load_feature_parts(args.feature_dir)
    rows = read_jsonl(args.policy_rows)
    by_uid = {str(row["uid"]): row for row in rows}
    if len(by_uid) != len(rows):
        raise RuntimeError("duplicate policy UID")
    feature_uids = [str(row["uid"]) for row in metadata]
    if set(feature_uids) != set(by_uid):
        raise RuntimeError("feature/policy UID mismatch")
    ordered_rows = [by_uid[uid] for uid in feature_uids]
    outcomes = [outcome_label(row) for row in ordered_rows]
    harm_labels = np.asarray([int(value == "harm") for value in outcomes], dtype=np.int64)
    splits = stratified_three_way_split(metadata, outcomes, args.split_seed)
    print(
        f"[admission] n={len(metadata)} harm={int(harm_labels.sum())} "
        f"splits={{{', '.join(f'{key}:{len(value)}' for key, value in splits.items())}}}",
        flush=True,
    )
    scaler = fit_scaler(tensors, splits["train"])
    features = transform_features(tensors, scaler)
    fit_indices, dev_indices = inner_fit_dev(
        splits["train"], harm_labels, metadata, args.split_seed + 1
    )

    device = torch.device(args.device)
    models: dict[str, list[dict[str, torch.Tensor]]] = {}
    training: dict[str, Any] = {}
    calibration_members: dict[str, np.ndarray] = {}
    test_members: dict[str, np.ndarray] = {}
    for architecture in ARCHITECTURES:
        states = []
        member_summaries = []
        for seed in SEEDS:
            print(f"[admission] train architecture={architecture} seed={seed}", flush=True)
            _, best_epoch, member_summary = train_member(
                features,
                harm_labels,
                fit_indices,
                dev_indices,
                architecture=architecture,
                seed=seed,
                args=args,
            )
            states.append(
                refit_member(
                    features,
                    harm_labels,
                    splits["train"],
                    architecture=architecture,
                    seed=seed,
                    epochs=best_epoch,
                    args=args,
                )
            )
            member_summaries.append(member_summary)
            print(
                f"[admission] done architecture={architecture} seed={seed} "
                f"best_epoch={best_epoch} dev_auprc={member_summary['best_dev_auprc']:.4f}",
                flush=True,
            )
        models[architecture] = states
        training[architecture] = member_summaries
        calibration_members[architecture] = member_probabilities(
            features[splits["calibration"]], states, architecture, device
        )
        test_members[architecture] = member_probabilities(
            features[splits["test"]], states, architecture, device
        )

    calibration_rows = [ordered_rows[int(i)] for i in splits["calibration"]]
    candidates = []
    sweeps = {}
    for architecture in ARCHITECTURES:
        members = calibration_members[architecture]
        for beta in UNCERTAINTY_BETAS:
            score = members.mean(axis=0) + beta * members.std(axis=0)
            selected, sweep = calibrate_threshold(calibration_rows, score, epsilon=args.epsilon)
            candidate_id = f"{architecture}_ucb{beta:g}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "architecture": architecture,
                    "uncertainty_beta": beta,
                    "calibration": selected,
                }
            )
            sweeps[candidate_id] = sweep
    selected_candidate = max(
        candidates,
        key=lambda row: (
            row["calibration"]["route_sensitive_layer_saving_fraction"],
            row["calibration"]["selected_accuracy"],
            -row["calibration"]["harm_count"],
        ),
    )
    architecture = selected_candidate["architecture"]
    beta = float(selected_candidate["uncertainty_beta"])
    test_score = test_members[architecture].mean(axis=0) + beta * test_members[architecture].std(axis=0)
    threshold = float(selected_candidate["calibration"]["threshold"])
    test_rows = [ordered_rows[int(i)] for i in splits["test"]]
    learned = summarize_admission(test_rows, test_score, threshold=threshold)
    all_on = summarize_admission(test_rows, test_score, threshold=float(np.nextafter(test_score.min(), -np.inf)))
    ungated = summarize_admission(test_rows, test_score, threshold=float(np.nextafter(test_score.max(), np.inf)))
    oracle_score = np.asarray([int(outcome_label(row) == "harm") for row in test_rows], dtype=np.float64)
    oracle = summarize_admission(test_rows, oracle_score, threshold=0.5)
    test_labels = harm_labels[splits["test"]]
    harm_metrics = binary_metrics(test_labels, test_score)
    harm_metrics.update(
        {
            "brier": float(np.mean((np.clip(test_score, 0.0, 1.0) - test_labels) ** 2)),
            "ece_10bin": ece(test_labels, np.clip(test_score, 0.0, 1.0)),
        }
    )
    learned.update(
        bootstrap_gate(
            test_rows,
            test_score,
            threshold,
            args.bootstrap_repetitions,
            args.split_seed + 99,
        )
    )

    summary = {
        "schema_version": "actual_policy_admission_diagnostic_v1",
        "feature_contract": {
            "route": "all_visual_on",
            "layer": 27,
            "stage": "post_ffn",
            "fields": ["instruction_mean", "instruction_last", "answer_start_confidence"],
            "benchmark_feature_used": False,
            "deployment_status": "two_pass_diagnostic_only",
        },
        "policy": "sw31_bt_leg_s41",
        "policy_rows": str(args.policy_rows),
        "feature_dir": str(args.feature_dir),
        "split_seed": args.split_seed,
        "split_counts": {key: len(value) for key, value in splits.items()},
        "split_uids": {
            key: [metadata[int(index)]["uid"] for index in value] for key, value in splits.items()
        },
        "outcome_counts": dict(Counter(outcomes)),
        "harm_positive_rate": float(harm_labels.mean()),
        "epsilon": args.epsilon,
        "training": training,
        "calibration_candidates": candidates,
        "selected_candidate": selected_candidate,
        "locked_test": {
            "all_on": all_on,
            "ungated_sparse": ungated,
            "oracle_admission": oracle,
            "learned_admission": learned,
            "harm_detection": harm_metrics,
        },
    }
    checkpoint = {
        "schema_version": 1,
        "scaler": scaler.state_dict(),
        "models": models,
        "selected_candidate": selected_candidate,
        "feature_contract": summary["feature_contract"],
    }
    torch.save(checkpoint, args.output_dir / "admission_gate.pt")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "calibration_sweeps.json").write_text(
        json.dumps(sweeps, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.md", summary)
    print(json.dumps(summary["locked_test"], indent=2))


if __name__ == "__main__":
    main()
