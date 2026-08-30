#!/usr/bin/env python3
"""Freeze the prospective selective CONTINUE/DEVIATE Phase-1 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.prepare_four_action_collapse import write_frozen
from experiments.train_binary_polar import file_sha256
from four_action_online_router.data import load_jsonl, load_verified_manifest
from four_action_policy.selective_continue_deviate import (
    build_full_insertion_subset,
)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def render_protocol(config: dict[str, Any], audit: dict[str, Any]) -> str:
    decision = config["phase1_decision"]
    return f"""# Selective CONTINUE/DEVIATE Protocol

Frozen before any Phase-1 live execution outcome was observed.

## Authority and fixed inputs

- Source plan: `{config['source_plan']}`
- Source-plan SHA-256: `{config['source_plan_sha256']}`
- Parent online config: `{config['parent_online_config']['path']}`
- Parent config SHA-256: `{config['parent_online_config']['sha256']}`
- Frozen source manifest: `{config['data']['source_manifest']}`
- Frozen boundary manifest: `{config['data']['boundary_manifest']}`
- Audit subset: `{config['data']['audit_subset']}`
- Audit-subset SHA-256: `{config['data']['audit_subset_sha256']}`

No router or base-model parameter, source label, split, executor setting, or
external evaluation contract is changed.

## Phase-1 census

The cohort is the complete held-out W2C split: {audit['states']} unique UIDs,
which lies at the authorized upper bound of 128 and avoids a post-selection
subsample. It contains:

- datasets: {audit['dataset_counts']}
- depth bins: {audit['depth_counts']}
- known mechanisms: {audit['mechanism_counts']}

For each state, replay the exact all-FULL prefix, insert `FULL` at the frozen
mandatory boundary, and retain every suffix named by the frozen boundary route
indices. Identical complete routes are deduplicated while all source-route
provenance is retained. This yields {audit['candidate_routes']} unique live
executions from {audit['source_compatible_suffixes']} compatible source
suffixes ({audit['deduplicated_source_routes']} duplicates removed). No route
cap or outcome-dependent selection is used.

## Classification and uncertainty

- `FULL-cache-incomplete`: at least one tested continuation is correct.
- `FULL-confirmed-invalid`: every bounded tested continuation is incorrect and
  the complete known compatible suffix set was executed.
- `unresolved`: the compatible suffix set or execution coverage is incomplete.

`FULL-confirmed-invalid` is bounded evidence, not global proof of invalidity.
Report overall, dataset, depth, and known-mechanism rescue rates with
{config['bootstrap']['draws']:,} fixed-seed UID-group percentile bootstrap
draws (seed {config['bootstrap']['seed']}).

## Prospective Phase-1 decision

Gate training is admitted only if:

```text
rescued states == 0
unresolved states == 0
trusted validation DEVIATE positives == {decision['required_trusted_validation_positives']}
```

This is not a post-hoc percentage cutoff. The complete census contains exactly
the plan's required minimum of {decision['required_trusted_validation_positives']}
validation DEVIATE positives; one known incomplete or unresolved state makes
that clean held-out contract unattainable without changing the frozen split or
weakening the trusted-label requirement after seeing outcomes.

If the condition fails, write the complete audit report and Stage-1 decision,
then stop without gate training. If it passes, proceed to the plan's frozen
linear/MLP gate and learned-WHEN + oracle-WHAT stages. Never train Stage 2 and
never run external evaluation in this phase.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="plans/selective_continue_deviate_expanded_plan.md"
    )
    parser.add_argument(
        "--online-config",
        default="analysis/persistent_corrective_supervision/online_config.yaml",
    )
    parser.add_argument(
        "--output-dir", default="analysis/selective_continue_deviate"
    )
    parser.add_argument(
        "--raw-output-root",
        default="/mnt/hyemin/qwen_train_eval/outputs/selective_continue_deviate_v1",
    )
    parser.add_argument("--execution-seed", type=int, default=20260830)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()

    plan_path = Path(args.plan)
    online_config_path = Path(args.online_config)
    online_config = yaml.safe_load(online_config_path.read_text(encoding="utf-8"))
    manifest_path = Path(online_config["data"]["manifest"])
    boundary_path = Path(online_config["data"]["boundary_manifest"])
    rows = load_verified_manifest(
        manifest_path, online_config["data"]["manifest_sha256"]
    )
    boundaries = load_jsonl(boundary_path)
    if file_sha256(boundary_path) != online_config["data"]["boundary_manifest_sha256"]:
        raise RuntimeError("boundary-manifest checksum mismatch")

    subset, audit = build_full_insertion_subset(rows, boundaries, split="validation")
    if (
        len(subset) != 128
        or audit["uids"] != 128
        or audit["candidate_routes"] != 252
        or not audit["all_suffix_sets_complete"]
    ):
        raise RuntimeError("Phase-1 audit census differs from the prospective contract")

    output_dir = Path(args.output_dir)
    subset_path = output_dir / "when_full_insertion_subset.json"
    subset_audit_path = output_dir / "when_full_insertion_subset_audit.json"
    config_path = output_dir / "audit_config.yaml"
    protocol_path = output_dir / "protocol.md"
    write_frozen(subset_path, _json(subset))
    audit.update(
        {
            "source_plan": str(plan_path),
            "source_plan_sha256": file_sha256(plan_path),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": file_sha256(manifest_path),
            "boundary_manifest": str(boundary_path),
            "boundary_manifest_sha256": file_sha256(boundary_path),
            "audit_subset": str(subset_path),
            "audit_subset_sha256": file_sha256(subset_path),
        }
    )
    write_frozen(subset_audit_path, _json(audit))

    config = {
        "protocol_version": "selective_continue_deviate_phase1_v1",
        "source_plan": str(plan_path),
        "source_plan_sha256": file_sha256(plan_path),
        "parent_online_config": {
            "path": str(online_config_path),
            "sha256": file_sha256(online_config_path),
        },
        "data": {
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": file_sha256(manifest_path),
            "source_metadata_manifest": online_config["data"]["source_manifest"],
            "source_metadata_manifest_sha256": online_config["data"][
                "source_manifest_sha256"
            ],
            "boundary_manifest": str(boundary_path),
            "boundary_manifest_sha256": file_sha256(boundary_path),
            "audit_subset": str(subset_path),
            "audit_subset_sha256": file_sha256(subset_path),
            "audit_subset_audit": str(subset_audit_path),
            "audit_subset_audit_sha256": file_sha256(subset_audit_path),
            "states": audit["states"],
            "candidate_routes": audit["candidate_routes"],
        },
        "execution": {
            "seed": args.execution_seed,
            "world_size": 4,
            "raw_output_root": str(Path(args.raw_output_root)),
            "executor": "live_unified_four_action_full_insertion_known_suffix_replay",
            "eos_token_ids": online_config["external_evaluation"]["eos_token_ids"],
            "repetition_penalty": online_config["external_evaluation"][
                "repetition_penalty"
            ],
        },
        "bootstrap": {"draws": 10000, "seed": args.bootstrap_seed, "unit": "uid"},
        "phase1_decision": {
            "required_trusted_validation_positives": 128,
            "maximum_rescued_states": 0,
            "maximum_unresolved_states": 0,
            "on_fail": "stop_before_gate_training",
            "on_pass": "proceed_to_linear_and_mlp_gate",
        },
        "reporting": {
            "analysis_dir": str(output_dir),
            "execution_records": str(
                output_dir / "when_full_insertion_executions.jsonl"
            ),
            "results": str(output_dir / "when_label_completeness_results.json"),
            "report": str(output_dir / "when_label_completeness_report.md"),
            "decision_summary": str(output_dir / "stage1_decision_summary.md"),
        },
    }
    write_frozen(config_path, yaml.safe_dump(config, sort_keys=False))
    write_frozen(protocol_path, render_protocol(config, audit))
    print(
        json.dumps(
            {
                "event": "selective_continue_deviate_phase1_prepared",
                "states": audit["states"],
                "candidate_routes": audit["candidate_routes"],
                "subset_sha256": file_sha256(subset_path),
                "config_sha256": file_sha256(config_path),
                "protocol_sha256": file_sha256(protocol_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
