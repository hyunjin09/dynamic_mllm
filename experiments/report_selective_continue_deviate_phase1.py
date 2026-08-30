#!/usr/bin/env python3
"""Validate and report the conditional Phase-1 audit decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.prepare_four_action_collapse import write_frozen
from experiments.train_binary_polar import file_sha256
from four_action_policy.selective_continue_deviate import evaluate_phase1_gate


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _row(label: str, values: Mapping[str, Any]) -> str:
    interval = f"[{_percent(values['ci95_lower'])}, {_percent(values['ci95_upper'])}]"
    return (
        f"| {label} | {values['states']} | {values['rescued']} | "
        f"{_percent(values['rescue_rate'])} | {interval} |"
    )


def render_report(
    config: dict[str, Any], subset_audit: dict[str, Any], result: dict[str, Any]
) -> str:
    rows = [_row("Overall", result["overall"])]
    for key, label in (("chartqa", "ChartQA"), ("gqa", "GQA"), ("textvqa", "TextVQA")):
        rows.append(_row(label, result["by_dataset"][key]))
    for key, label in (("early", "Early"), ("middle", "Middle"), ("late", "Late")):
        rows.append(_row(label, result["by_depth_bin"][key]))
    for key in ("IGNORE", "READ_ONLY", "WRITE_ONLY", "MULTI"):
        rows.append(_row(key, result["by_known_mechanism"][key]))
    histogram = ", ".join(
        f"{suffixes} route(s): {states} states"
        for suffixes, states in sorted(
            result["suffix_count_histogram"].items(), key=lambda item: int(item[0])
        )
    )
    return f"""# WHEN-Label Completeness Report

## Outcome

The prospective Phase-1 label-trust gate **fails**. Forced `FULL` at the
supposed mandatory boundary has a correct bounded continuation for
**{result['overall']['rescued']}/{result['states']} states
({_percent(result['overall']['rescue_rate'])}, 95% UID-bootstrap CI
[{_percent(result['overall']['ci95_lower'])},
{_percent(result['overall']['ci95_upper'])}])**. There are zero unresolved
states. Only {result['status_counts']['FULL-confirmed-invalid']} states remain
trusted under this bounded audit, below the prospectively required 128 clean
validation DEVIATE positives.

Per the frozen protocol, linear/MLP gate training, threshold selection, and
learned-WHEN + oracle-WHAT execution were not started.

## Execution validity

- Cohort: all {result['states']} frozen held-out W2C UIDs; no audit subsampling.
- Live executions: {result['candidate_executions']}/{config['data']['candidate_routes']}.
- Suffix coverage: all {subset_audit['source_compatible_suffixes']} compatible
  frozen source suffixes; {subset_audit['candidate_routes']} complete routes
  after deduplicating {subset_audit['deduplicated_source_routes']} identical routes.
- Statuses: {result['status_counts']['FULL-cache-incomplete']}
  `FULL-cache-incomplete`,
  {result['status_counts']['FULL-confirmed-invalid']}
  `FULL-confirmed-invalid`, 0 unresolved.
- Bootstrap: {config['bootstrap']['draws']:,} UID-group draws with fixed overall
  seed {config['bootstrap']['seed']} (fixed derived seeds for subgroups).
- Base-model revision: `{result['base_model_revision']}`.
- Executor: `{result['executor']}`.
- Audit-subset SHA-256: `{result['subset_sha256']}`.
- Audit-config SHA-256: `{result['config_sha256']}`.

Every `FULL-confirmed-invalid` classification remains bounded: failure of all
known compatible suffixes is not global proof that no unobserved continuation
could work.

## Table 1 — WHEN-label completeness

| Group | States | FULL bounded rescue | Rescue rate | 95% CI |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Suffixes tested per state

{histogram}.

## Interpretation

The incompleteness is not confined to one dataset: bounded rescue is 15/43
ChartQA, 12/43 GQA, and 12/42 TextVQA. It is depth-dependent in this frozen
cohort—25/48 early, 12/43 middle, and 2/37 late—and is largest for multi-valid
boundaries (16/30) and READ_ONLY-valid boundaries (13/33). These subgroup
patterns are descriptive, not causal explanations.

The direct observation is that the existing mandatory-boundary target treats
`FULL` as invalid for 39 states where the unchanged executor finds a correct
route using a frozen compatible suffix. The cause of this cache incompleteness
remains unknown. That observation alone changes the action: a clean binary
CONTINUE/DEVIATE gate must not be trained from these labels under the frozen
minimum-validation contract.

## Conditional stop

The plan's Case A applies. The smallest defensible future action is to
repair/expand route-cache continuation coverage, rebuild WHEN labels, and
re-audit prospectively. That repair is a new research action and was not
executed here. Stage 2 and external evaluation remain out of scope.
"""


def render_decision(result: dict[str, Any], decision: dict[str, Any]) -> str:
    return f"""# Stage-1 Decision Summary

## Decision

**Stop before selective-gate training.** Phase 1 found
{decision['rescued_states']}/{decision['states']} `FULL-cache-incomplete`
mandatory boundaries and {decision['unresolved_states']} unresolved states.
Only {decision['trusted_validation_deviate_positives']} trusted validation
DEVIATE positives remain, versus the frozen requirement of
{decision['required_trusted_validation_positives']}.

## Q1 — Label validity

No. Mandatory-boundary DEVIATE labels are not sufficiently trustworthy under
the expanded audit: bounded FULL rescue is {_percent(result['overall']['rescue_rate'])}
with 95% UID-bootstrap CI [{_percent(result['overall']['ci95_lower'])},
{_percent(result['overall']['ci95_upper'])}].

## Q2 — Signal

Not tested in this phase. The linear and MLP gates were conditionally forbidden
after the Phase-1 failure. The earlier Phase-40 probe is motivation, not a
substitute for the clean gate experiment specified here.

## Q3 — Selectivity

Not tested. No threshold sweep or 99%/98%/95% C2C-preservation operating point
was produced because that would require training on a failed label contract.

## Q4 — Behavioral value

Not tested. Learned-WHEN + oracle-WHAT execution depends on a trained Stage-1
gate and therefore was not run.

## Q5 — Next step

Stage 1 is not good enough to justify READ_OFF/WRITE_OFF/BOTH_OFF Stage-2
training. The smallest defensible next action is a separately authorized
continuation-cache repair, WHEN-label rebuild, and repeat audit. No such repair,
training, Stage 2, or external evaluation was executed.

## Intentionally absent conditional artifacts

`gate_train_manifest.jsonl`, `gate_val_manifest.jsonl`, both gate configs and
histories/results, `threshold_sweep.csv`, `selective_operating_points.json`,
oracle-WHAT outputs, and gate figures do not exist. Their absence is the frozen
Case-A stop behavior, not missing execution.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="analysis/selective_continue_deviate/audit_config.yaml"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "selective_continue_deviate_phase1_v1":
        raise RuntimeError("incompatible Phase-1 protocol")
    if file_sha256(Path(config["source_plan"])) != config["source_plan_sha256"]:
        raise RuntimeError("source-plan checksum mismatch")

    result_path = Path(config["reporting"]["results"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("result/config checksum mismatch")
    if result["subset_sha256"] != config["data"]["audit_subset_sha256"]:
        raise RuntimeError("result/subset checksum mismatch")
    subset_audit_path = Path(config["data"]["audit_subset_audit"])
    if file_sha256(subset_audit_path) != config["data"]["audit_subset_audit_sha256"]:
        raise RuntimeError("subset-audit checksum mismatch")
    subset_audit = json.loads(subset_audit_path.read_text(encoding="utf-8"))

    decision = evaluate_phase1_gate(result, config["phase1_decision"])
    if decision["passed"]:
        raise RuntimeError("Phase-1 passed; Case-B reporting is not implemented here")
    output_dir = Path(config["reporting"]["analysis_dir"])
    decision_path = output_dir / "phase1_decision.json"
    write_frozen(decision_path, json.dumps(decision, indent=2, sort_keys=True) + "\n")
    write_frozen(
        Path(config["reporting"]["report"]),
        render_report(config, subset_audit, result),
    )
    write_frozen(
        Path(config["reporting"]["decision_summary"]),
        render_decision(result, decision),
    )

    checksum_path = output_dir / "artifact_checksums.sha256"
    artifacts = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path != checksum_path
    )
    checksum_text = "".join(
        f"{file_sha256(path)}  {path.relative_to(output_dir)}\n" for path in artifacts
    )
    write_frozen(checksum_path, checksum_text)
    print(
        json.dumps(
            {
                "event": "selective_continue_deviate_phase1_reported",
                "outcome": decision["outcome"],
                "rescued_states": decision["rescued_states"],
                "trusted_validation_deviate_positives": decision[
                    "trusted_validation_deviate_positives"
                ],
                "artifact_checksums": len(artifacts),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
