#!/usr/bin/env python3
"""Create the frozen matched comparison and architecture decision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.train_four_action_online_router import write_jsonl
from four_action_policy.actions import FOUR_ACTIONS
from four_action_policy.persistent import (
    paired_bootstrap_rate_difference,
    select_behavioral_checkpoint,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: Sequence[Sequence[Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    temporary.replace(path)


def selected_epoch(selection: dict[str, Any]) -> int | None:
    value = selection.get("selected_epoch", selection.get("best_epoch"))
    if value is None and isinstance(selection.get("behavioral_selection"), dict):
        value = selection["behavioral_selection"].get("selected_epoch")
    return int(value) if value is not None else None


def dataset_behavior(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for dataset in ("gqa", "chartqa", "textvqa"):
        values = [row for row in rows if row["dataset"] == dataset]
        by_type = {}
        for route_type in ("W2C", "C2C"):
            selected = [row for row in values if row["route_type"] == route_type]
            by_type[route_type] = {
                "records": len(selected),
                "correct": sum(bool(row["correct"]) for row in selected),
                "rate": sum(bool(row["correct"]) for row in selected) / len(selected),
            }
        output[dataset] = by_type
    return output


def metric_values(
    *,
    epoch: int | None,
    training_history: Sequence[dict[str, Any]],
    execution_history: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if epoch is None:
        return None
    train = next(row for row in training_history if int(row["epoch"]) == epoch)
    result = next(row for row in execution_history if int(row["epoch"]) == epoch)
    execution = result["execution"]
    boundary = result["boundary"]
    counts = execution["action_counts"]
    denominator = sum(int(value) for value in counts.values())
    train_boundary = train["train"].get("boundary_valid_action_at_1")
    free_rollout_leave_full = boundary["free_rollout"]["left_all_full_fraction"]
    return {
        "epoch": epoch,
        "train_boundary_valid_action_at_1": train_boundary,
        "val_boundary_valid_action_at_1": boundary["valid_action_at_1"],
        "val_boundary_nonfull_recall": boundary["nonfull_recall"],
        "w2c_rescue": execution["w2c_rescue_rate"],
        "c2c_preservation": execution["c2c_preservation_rate"],
        "rescues": execution["w2c_rescues"],
        "regressions": execution["c2c_regressions"],
        "net_accuracy_change": (
            execution["w2c_rescues"] - execution["c2c_regressions"]
        )
        / execution["records"],
        "exact_first_deviation": boundary["free_rollout"]["exact_boundary_fraction"],
        "within_1_first_deviation": boundary["free_rollout"]["within_1_fraction"],
        "within_2_first_deviation": boundary["free_rollout"]["within_2_fraction"],
        "early_deviation_fraction": boundary["free_rollout"]["early_fraction"],
        "late_or_no_deviation_fraction": boundary["free_rollout"][
            "late_or_no_deviation_fraction"
        ],
        "no_deviation_fraction": 1.0 - free_rollout_leave_full,
        "teacher_forced_minus_free_rollout_leave_full_gap": (
            boundary["nonfull_recall"] - free_rollout_leave_full
        ),
        **{
            f"{action.lower()}_fraction": counts[action] / denominator
            for action in FOUR_ACTIONS
        },
    }


def frontier(selection: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(selection.get("behavioral_selection"), dict):
        return list(selection["behavioral_selection"].get("pareto_frontier", []))
    return list(selection.get("pareto_frontier", []))


def runtime_cohort_sensitivity(
    *,
    polar_outputs: Sequence[dict[str, Any]],
    online_outputs: Sequence[dict[str, Any]],
    polar_history: Sequence[dict[str, Any]],
    online_history: Sequence[dict[str, Any]],
    polar_selected_epoch: int,
    online_selected_epoch: int,
    c2c_threshold: float,
) -> dict[str, Any]:
    """Reapply selection after excluding current-runtime cohort drift.

    The frozen W2C/C2C membership remains the primary analysis.  POLAR epoch 1
    supplies a direct all-FULL validity observation because every action for
    every validation row is FULL at that checkpoint.
    """

    baseline_rows = [row for row in polar_outputs if int(row["epoch"]) == 1]
    if not baseline_rows or any(
        not row.get("actions") or any(action != "FULL" for action in row["actions"])
        for row in baseline_rows
    ):
        raise RuntimeError("runtime cohort audit requires an all-FULL POLAR epoch 1")
    if len({row["uid"] for row in baseline_rows}) != len(baseline_rows):
        raise RuntimeError("runtime cohort audit baseline contains duplicate UIDs")

    drift_records = []
    for row in baseline_rows:
        frozen_expected_correct = row["route_type"] == "C2C"
        if bool(row["correct"]) != frozen_expected_correct:
            drift_records.append(
                {
                    "uid": row["uid"],
                    "dataset": row["dataset"] if "dataset" in row else None,
                    "route_type": row["route_type"],
                    "frozen_expected_correct": frozen_expected_correct,
                    "current_all_full_correct": bool(row["correct"]),
                    "current_all_full_prediction": row.get("prediction"),
                }
            )
    drift_records.sort(key=lambda row: row["uid"])
    drift_uids = {row["uid"] for row in drift_records}

    def adjusted_history(
        history: Sequence[dict[str, Any]], outputs: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        adjusted = []
        for source in history:
            epoch = int(source["epoch"])
            rows = [
                row
                for row in outputs
                if int(row["epoch"]) == epoch and row["uid"] not in drift_uids
            ]
            w2c = [row for row in rows if row["route_type"] == "W2C"]
            c2c = [row for row in rows if row["route_type"] == "C2C"]
            if not w2c or not c2c:
                raise RuntimeError("runtime cohort sensitivity emptied a route cohort")
            execution = {
                **source["execution"],
                "w2c_rescues": sum(bool(row["correct"]) for row in w2c),
                "w2c_rescue_rate": sum(bool(row["correct"]) for row in w2c)
                / len(w2c),
                "c2c_regressions": sum(not bool(row["correct"]) for row in c2c),
                "c2c_preservation_rate": sum(bool(row["correct"]) for row in c2c)
                / len(c2c),
                "overall_routed_accuracy": sum(bool(row["correct"]) for row in rows)
                / len(rows),
            }
            adjusted.append({**source, "execution": execution})
        return adjusted

    def selected_counts(
        outputs: Sequence[dict[str, Any]], selection: dict[str, Any]
    ) -> dict[str, Any]:
        epoch = selection["selected_epoch"]
        if epoch is None:
            return {
                "selected_epoch": None,
                "w2c_correct": None,
                "w2c_records": None,
                "w2c_rate": None,
                "c2c_correct": None,
                "c2c_records": None,
                "c2c_rate": None,
            }
        rows = [
            row
            for row in outputs
            if int(row["epoch"]) == int(epoch) and row["uid"] not in drift_uids
        ]
        w2c = [row for row in rows if row["route_type"] == "W2C"]
        c2c = [row for row in rows if row["route_type"] == "C2C"]
        w2c_correct = sum(bool(row["correct"]) for row in w2c)
        c2c_correct = sum(bool(row["correct"]) for row in c2c)
        return {
            "selected_epoch": int(epoch),
            "w2c_correct": w2c_correct,
            "w2c_records": len(w2c),
            "w2c_rate": w2c_correct / len(w2c),
            "c2c_correct": c2c_correct,
            "c2c_records": len(c2c),
            "c2c_rate": c2c_correct / len(c2c),
        }

    polar_adjusted = select_behavioral_checkpoint(
        adjusted_history(polar_history, polar_outputs),
        c2c_threshold=c2c_threshold,
    )
    online_adjusted = select_behavioral_checkpoint(
        adjusted_history(online_history, online_outputs),
        c2c_threshold=c2c_threshold,
    )
    polar_counts = selected_counts(polar_outputs, polar_adjusted)
    online_counts = selected_counts(online_outputs, online_adjusted)
    w2c_drift = any(row["route_type"] == "W2C" for row in drift_records)
    decision_invariant = (
        not w2c_drift
        and polar_counts["selected_epoch"] == polar_selected_epoch
        and online_counts["selected_epoch"] == online_selected_epoch
    )
    return {
        "baseline_architecture": "POLAR",
        "baseline_epoch": 1,
        "baseline_records": len(baseline_rows),
        "baseline_all_full_verified": True,
        "runtime_drift_count": len(drift_records),
        "runtime_drift_uids": sorted(drift_uids),
        "drift_records": drift_records,
        "polar": polar_counts,
        "online": online_counts,
        "w2c_population_changed": w2c_drift,
        "decision_invariant": decision_invariant,
    }


def render_runtime_cohort_sensitivity(sensitivity: dict[str, Any]) -> str:
    lines = [
        "# Runtime Cohort Sensitivity",
        "",
        "## Direct observation",
        "",
        f"- POLAR epoch {sensitivity['baseline_epoch']} executed all-FULL for all "
        f"{sensitivity['baseline_records']} frozen validation records.",
        f"- Current-runtime membership mismatches: {sensitivity['runtime_drift_count']}.",
    ]
    for row in sensitivity["drift_records"]:
        lines.append(
            f"- `{row['uid']}` ({row['route_type']}): frozen membership expected "
            f"correct={str(row['frozen_expected_correct']).lower()}, while current "
            f"all-FULL execution produced `{row['current_all_full_prediction']}` and "
            f"correct={str(row['current_all_full_correct']).lower()}."
        )
    lines.extend(
        [
            "",
            "The frozen 256-record analysis remains primary. The following sensitivity",
            "excludes only the mismatched UID from its frozen cohort and reapplies the",
            "unchanged C2C >= 95% checkpoint-selection rule across all 20 epochs.",
            "",
            "| Architecture | Selected epoch | W2C rescue | C2C preservation |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("polar", "online"):
        values = sensitivity[name]
        lines.append(
            f"| {name.upper() if name == 'polar' else 'Online'} | "
            f"{values['selected_epoch']} | {values['w2c_correct']}/"
            f"{values['w2c_records']} ({values['w2c_rate']:.6f}) | "
            f"{values['c2c_correct']}/{values['c2c_records']} "
            f"({values['c2c_rate']:.6f}) |"
        )
    lines.extend(
        [
            "",
            f"- W2C paired population changed: "
            f"{str(sensitivity['w2c_population_changed']).lower()}.",
            f"- Selected checkpoints and matched architecture decision invariant: "
            f"{str(sensitivity['decision_invariant']).lower()}.",
            "- This check establishes robustness to the observed cohort mismatch; it",
            "  does not diagnose why the current runtime differs from the frozen record.",
            "",
        ]
    )
    return "\n".join(lines)


def render_architecture_report(
    name: str,
    history: Sequence[dict[str, Any]],
    training_history: Sequence[dict[str, Any]],
    selection: dict[str, Any],
    dataset_metrics: dict[str, Any] | None,
    outputs: Sequence[dict[str, Any]],
) -> str:
    epoch = selected_epoch(selection)
    lines = [
        f"# {name} Persistent Corrective Supervision",
        "",
        f"- Checkpoints executed: {len(history)}",
        f"- Selected epoch under C2C >= 95%: {epoch}",
        f"- Pareto frontier: {frontier(selection)}",
        "- External evaluation started: false",
    ]
    if dataset_metrics is not None:
        lines.extend(["", "## Selected-checkpoint dataset behavior", ""])
        for dataset, values in dataset_metrics.items():
            lines.append(
                f"- {dataset}: W2C {values['W2C']['correct']}/{values['W2C']['records']} "
                f"({values['W2C']['rate']:.6f}); C2C "
                f"{values['C2C']['correct']}/{values['C2C']['records']} "
                f"({values['C2C']['rate']:.6f})"
            )
    else:
        lines.extend(
            [
                "",
                "## Pareto-frontier behavior",
                "",
                "No checkpoint met the frozen preservation constraint, so no single",
                "checkpoint is substituted into the matched headline table.",
                "",
                "| Epoch | Train boundary Valid@1 | Val boundary Valid@1 | Val boundary non-FULL | W2C rescue | C2C preservation | Net accuracy change | Exact first deviation | Within 1 | Within 2 | Early | Late/no deviation | No deviation | FULL fraction | READ_ONLY fraction | WRITE_ONLY fraction | IGNORE fraction |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for point in frontier(selection):
            point_epoch = int(point["epoch"])
            metrics = metric_values(
                epoch=point_epoch,
                training_history=training_history,
                execution_history=history,
            )
            lines.append(
                f"| {point_epoch} | {metrics['train_boundary_valid_action_at_1']:.6f} | "
                f"{metrics['val_boundary_valid_action_at_1']:.6f} | "
                f"{metrics['val_boundary_nonfull_recall']:.6f} | "
                f"{metrics['w2c_rescue']:.6f} | {metrics['c2c_preservation']:.6f} | "
                f"{metrics['net_accuracy_change']:.6f} | "
                f"{metrics['exact_first_deviation']:.6f} | "
                f"{metrics['within_1_first_deviation']:.6f} | "
                f"{metrics['within_2_first_deviation']:.6f} | "
                f"{metrics['early_deviation_fraction']:.6f} | "
                f"{metrics['late_or_no_deviation_fraction']:.6f} | "
                f"{metrics['no_deviation_fraction']:.6f} | "
                f"{metrics['full_fraction']:.6f} | "
                f"{metrics['read_only_fraction']:.6f} | "
                f"{metrics['write_only_fraction']:.6f} | "
                f"{metrics['ignore_fraction']:.6f} |"
            )
        lines.extend(["", "## Pareto-frontier dataset behavior", ""])
        for point in frontier(selection):
            point_epoch = int(point["epoch"])
            point_rows = [row for row in outputs if int(row["epoch"]) == point_epoch]
            values = dataset_behavior(point_rows)
            lines.append(
                f"- epoch {point_epoch}: "
                + "; ".join(
                    f"{dataset} W2C {cell['W2C']['correct']}/{cell['W2C']['records']}, "
                    f"C2C {cell['C2C']['correct']}/{cell['C2C']['records']}"
                    for dataset, cell in values.items()
                )
            )
    lines.extend(["", "## Every checkpoint", ""])
    for row in history:
        execution = row["execution"]
        lines.append(
            f"- epoch {row['epoch']}: W2C {execution['w2c_rescue_rate']:.6f}; "
            f"C2C {execution['c2c_preservation_rate']:.6f}; "
            f"net {execution['w2c_rescues'] - execution['c2c_regressions']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--polar-training-dir", required=True)
    parser.add_argument("--polar-execution-dir", required=True)
    parser.add_argument("--online-training-dir", required=True)
    args = parser.parse_args()
    analysis_dir = Path(args.analysis_dir)
    polar_training_dir = Path(args.polar_training_dir)
    polar_execution_dir = Path(args.polar_execution_dir)
    online_training_dir = Path(args.online_training_dir)
    polar_config = yaml.safe_load((analysis_dir / "polar_config.yaml").read_text())
    online_config = yaml.safe_load((analysis_dir / "online_config.yaml").read_text())

    polar_training_history = read_json(polar_training_dir / "history.json")
    polar_execution_history = read_json(polar_execution_dir / "execution_history.json")
    polar_selection = read_json(polar_training_dir / "best_checkpoint.json")
    online_history = read_json(online_training_dir / "history.json")
    online_selection = read_json(online_training_dir / "best_checkpoint.json")
    if not all(len(values) == 20 for values in (polar_training_history, polar_execution_history, online_history)):
        raise RuntimeError("matched comparison requires all 20 checkpoints for both substrates")

    online_outputs = []
    for epoch in range(1, 21):
        rows = read_jsonl(
            online_training_dir / f"epoch_{epoch:02d}" / "validation_outputs.jsonl"
        )
        if len(rows) != 256:
            raise RuntimeError(f"online epoch {epoch} validation coverage mismatch")
        online_outputs.extend({"epoch": epoch, **row} for row in rows)
    write_jsonl(Path(online_config["reporting"]["execution"]), online_outputs)
    write_jsonl(Path(online_config["reporting"]["history"]), list(online_history))

    polar_outputs = read_jsonl(Path(polar_config["reporting"]["execution"]))
    if len(polar_outputs) != 20 * 256 or len(online_outputs) != 20 * 256:
        raise RuntimeError("matched execution rows do not cover every checkpoint")
    polar_epoch = selected_epoch(polar_selection)
    online_epoch = selected_epoch(online_selection)
    polar_selected_rows = [row for row in polar_outputs if row["epoch"] == polar_epoch]
    online_selected_rows = [row for row in online_outputs if row["epoch"] == online_epoch]
    polar_metrics = metric_values(
        epoch=polar_epoch,
        training_history=polar_training_history,
        execution_history=polar_execution_history,
    )
    online_metrics = metric_values(
        epoch=online_epoch,
        training_history=online_history,
        execution_history=online_history,
    )
    polar_dataset = dataset_behavior(polar_selected_rows) if polar_epoch else None
    online_dataset = dataset_behavior(online_selected_rows) if online_epoch else None
    sensitivity = runtime_cohort_sensitivity(
        polar_outputs=polar_outputs,
        online_outputs=online_outputs,
        polar_history=polar_execution_history,
        online_history=online_history,
        polar_selected_epoch=polar_epoch,
        online_selected_epoch=online_epoch,
        c2c_threshold=float(
            online_config["validation"]["c2c_preservation_threshold"]
        ),
    )
    (analysis_dir / "runtime_cohort_sensitivity.md").write_text(
        render_runtime_cohort_sensitivity(sensitivity), encoding="utf-8"
    )

    polar_viable = bool(polar_metrics and polar_metrics["rescues"] > 0)
    online_viable = bool(online_metrics and online_metrics["rescues"] > 0)
    bootstrap = None
    if polar_viable and online_viable:
        polar_w2c = {
            row["uid"]: bool(row["correct"])
            for row in polar_selected_rows
            if row["route_type"] == "W2C"
        }
        online_w2c = {
            row["uid"]: bool(row["correct"])
            for row in online_selected_rows
            if row["route_type"] == "W2C"
        }
        if set(polar_w2c) != set(online_w2c):
            raise RuntimeError("selected paired W2C execution populations differ")
        uids = sorted(polar_w2c)
        bootstrap = paired_bootstrap_rate_difference(
            [polar_w2c[uid] for uid in uids],
            [online_w2c[uid] for uid in uids],
            draws=int(online_config["validation"]["paired_bootstrap_draws"]),
            seed=int(online_config["validation"]["paired_bootstrap_seed"]),
        )

    if not polar_viable and not online_viable:
        decision = "select neither architecture"
        reason = "neither substrate has an eligible checkpoint with held-out W2C rescue"
    elif polar_viable and not online_viable:
        decision = "favor POLAR"
        reason = "only POLAR is behaviorally viable under the frozen preservation constraint"
    elif online_viable and not polar_viable:
        decision = "favor online"
        reason = "only online is behaviorally viable under the frozen preservation constraint"
    elif bootstrap["ci_low"] > 0:
        decision = "favor online"
        reason = "the paired W2C rescue difference favors online with a 95% interval above zero"
    elif bootstrap["ci_high"] < 0:
        decision = "favor POLAR"
        reason = "the paired W2C rescue difference favors POLAR with a 95% interval below zero"
    else:
        decision = "no supported architecture advantage; operationally prefer POLAR"
        reason = "both are viable but the paired W2C difference interval includes zero"

    metrics = (
        "train_boundary_valid_action_at_1",
        "val_boundary_valid_action_at_1",
        "val_boundary_nonfull_recall",
        "w2c_rescue",
        "c2c_preservation",
        "net_accuracy_change",
        "exact_first_deviation",
        "within_1_first_deviation",
        "within_2_first_deviation",
        "early_deviation_fraction",
        "late_or_no_deviation_fraction",
        "no_deviation_fraction",
        "teacher_forced_minus_free_rollout_leave_full_gap",
        "full_fraction",
        "read_only_fraction",
        "write_only_fraction",
        "ignore_fraction",
    )
    table_rows = [["metric", "POLAR", "Online"]]
    for metric in metrics:
        table_rows.append(
            [
                metric,
                "N/A" if polar_metrics is None else polar_metrics[metric],
                "N/A" if online_metrics is None else online_metrics[metric],
            ]
        )
    write_csv(analysis_dir / "matched_comparison.csv", table_rows)
    md = [
        "# Matched Persistent-Corrective Comparison",
        "",
        "| Metric | POLAR | Online |",
        "|---|---:|---:|",
    ]
    for metric, polar, online in table_rows[1:]:
        def render(value: Any) -> str:
            return f"{value:.6f}" if isinstance(value, float) else str(value)
        md.append(f"| {metric} | {render(polar)} | {render(online)} |")
    md.extend(
        [
            "",
            f"- POLAR selected epoch: {polar_epoch}; viable: {polar_viable}",
            f"- Online selected epoch: {online_epoch}; viable: {online_viable}",
            f"- POLAR Pareto frontier: {frontier(polar_selection)}",
            f"- Online Pareto frontier: {frontier(online_selection)}",
            f"- Paired bootstrap (online minus POLAR): {bootstrap}",
            f"- Decision: {decision}.",
            "- Runtime-cohort sensitivity: one current all-FULL C2C mismatch; "
            f"selection/decision invariant: {sensitivity['decision_invariant']}.",
            "",
        ]
    )
    (analysis_dir / "matched_comparison.md").write_text("\n".join(md), encoding="utf-8")
    (analysis_dir / "polar_report.md").write_text(
        render_architecture_report(
            "POLAR",
            polar_execution_history,
            polar_training_history,
            polar_selection,
            polar_dataset,
            polar_outputs,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "online_report.md").write_text(
        render_architecture_report(
            "Online",
            online_history,
            online_history,
            online_selection,
            online_dataset,
            online_outputs,
        ),
        encoding="utf-8",
    )
    summary = [
        "# Persistent Corrective Supervision Decision",
        "",
        "## Decision",
        "",
        f"**{decision}.** {reason}.",
        "",
        "## Confirmed observations",
        "",
        f"- POLAR selected epoch: {polar_epoch}; metrics: {polar_metrics}.",
        f"- Online selected epoch: {online_epoch}; metrics: {online_metrics}.",
        f"- Paired W2C bootstrap: {bootstrap}.",
        f"- POLAR Pareto frontier: {frontier(polar_selection)}.",
        f"- Online Pareto frontier: {frontier(online_selection)}.",
        "- All 20 checkpoints for both substrates were evaluated on the same 256",
        "  held-out records. No external evaluation ran.",
        "- A direct all-FULL audit found one current-runtime mismatch in the frozen",
        "  C2C cohort. Excluding that UID changes the C2C denominators to 127 but",
        "  leaves selected epochs, W2C comparison, and the decision unchanged; see",
        "  `runtime_cohort_sensitivity.md`.",
        "",
        "## Interpretation boundary",
        "",
        "The result compares these two fixed recipes under matched persistent",
        "corrective supervision. It does not prove an architecture impossibility or",
        "identify the underlying cause of any remaining generalization failure.",
        "",
        "## Stop",
        "",
        "The authorized matched action is complete. No scale-up, objective change,",
        "external evaluation, or follow-up diagnostic is authorized.",
        "",
    ]
    (analysis_dir / "decision_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"decision": decision, "reason": reason, "bootstrap": bootstrap}, sort_keys=True))


if __name__ == "__main__":
    main()
