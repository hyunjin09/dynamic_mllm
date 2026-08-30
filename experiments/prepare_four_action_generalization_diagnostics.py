#!/usr/bin/env python3
"""Freeze the matched state population and all Phase-40 diagnostic choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.prepare_four_action_collapse import write_frozen
from experiments.train_binary_polar import file_sha256
from four_action_online_router.data import load_jsonl, load_verified_manifest
from four_action_policy.generalization_diagnostics import build_matched_state_manifest


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def render_protocol(config: dict[str, Any], audit: dict[str, Any]) -> str:
    probe = config["probes"]
    label_audit = config["label_incompleteness"]
    return f"""# Four-Action Generalization Diagnostic Protocol

Frozen before selected-checkpoint diagnostic outcomes were extracted.

## Authority and fixed inputs

- Source plan: `{config['source_plan']}`
- Source-plan SHA-256: `{config['source_plan_sha256']}`
- POLAR checkpoint: epoch {config['checkpoints']['polar']['epoch']}, `{config['checkpoints']['polar']['path']}`
- POLAR checkpoint SHA-256: `{config['checkpoints']['polar']['sha256']}`
- Online checkpoint: epoch {config['checkpoints']['online']['epoch']}, `{config['checkpoints']['online']['path']}`
- Online checkpoint SHA-256: `{config['checkpoints']['online']['sha256']}`
- Frozen source manifest: `{config['data']['source_manifest']}`
- Frozen boundary manifest: `{config['data']['boundary_manifest']}`
- State-manifest SHA-256: `{config['data']['state_manifest_sha256']}`

No router/base-model parameter, label, data split, or executor contract changes.
No external evaluation is admitted.

## State construction

- Mandatory-deviation positives: all {audit['positive_states']} frozen W2C
  mandatory all-FULL-prefix boundaries ({audit['split_counts']['train'] // 2}
  train; {audit['split_counts']['validation'] // 2} validation).
- KEEP_FULL negatives: one W2C correcting-trajectory node with unique valid
  next action `FULL`, matched without replacement at the same split, dataset,
  and exact layer, and required to come from a different UID than its positive.
- Total states: {audit['records']} in {audit['pairs']} exact matched pairs.
- Multi-valid non-FULL nodes remain set-valued. Singleton mechanism analyses do
  not assign them an arbitrary class. READ_OFF/WRITE_OFF analyses include a
  state only when every valid action agrees on that bit.
- Depth bins: layers 0--9 early, 10--18 middle, and 19--27 late.

## Selected-router outputs and shuffle

- Action order: `IGNORE`, `READ_ONLY`, `WRITE_ONLY`, `FULL`.
- WHEN score: `1 - P(FULL)`.
- READ_OFF score: `P(WRITE_ONLY) + P(IGNORE)`.
- WRITE_OFF score: `P(READ_ONLY) + P(IGNORE)`.
- Online state shuffle: jointly replace the text query and visual state by the
  frozen partner in the same split/dataset/layer cell; labels and layer identity
  stay fixed. Seed: {config['extraction']['shuffle_seed']}.
- Four direct torchrun ranks/GPU devices are required for state extraction.

## Metrics and baselines

- Binary threshold: 0.5; report AUROC, AUPRC, accuracy, balanced accuracy,
  precision, recall, F1, FPR, and FNR.
- Layer-only baseline: train-only empirical class probability by exact layer
  with Jeffreys smoothing alpha={config['analysis']['layer_only_alpha']}.
- Probe preprocessing uses training-only mean/standard deviation. Linear and
  one-hidden-layer MLP heads use the same native representation, fixed schedule,
  and no validation-based hyperparameter or checkpoint selection.
- Probe seed/epochs/batch/lr/weight decay: {probe['seed']} / {probe['epochs']} /
  {probe['batch_size']} / {probe['learning_rate']} / {probe['weight_decay']}.
- MLP hidden width: {probe['mlp_hidden_size']}; dropout: {probe['dropout']}.
- kNN uses cosine distance with k={probe['knn_k']}. Candidate pools fall back
  prospectively from exact dataset+layer, to exact layer, to dataset+depth bin,
  then global training states. Preprocessing is training-only.

## Bounded label-incompleteness audit

- Population: validation mandatory-deviation states where a selected router
  predicts non-FULL outside the cached valid-action set.
- Deterministic cap: at most {label_audit['cap_per_architecture_action']} states
  per architecture × predicted action, seed {label_audit['seed']}.
- For each state, execute at most {label_audit['max_known_suffixes']} unique
  routes formed by the exact audited prefix, cached-invalid predicted action,
  and a known compatible correcting-route suffix.
- A correct candidate is positive evidence of label incompleteness. Failure of
  every bounded candidate is recorded as `no_bounded_rescue`, not proof that
  the action is globally invalid.

## Stop

Answer Q1--Q9 and identify the dominant supported failure mode. Do not start
the implied next method, a new full router run, or external evaluation.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="plans/four_action_generalization_diagnostic_plan.md"
    )
    parser.add_argument(
        "--polar-config",
        default="analysis/persistent_corrective_supervision/polar_config.yaml",
    )
    parser.add_argument(
        "--online-config",
        default="analysis/persistent_corrective_supervision/online_config.yaml",
    )
    parser.add_argument(
        "--output-dir", default="analysis/4action_generalization_diagnostics"
    )
    parser.add_argument(
        "--raw-output-root",
        default="/mnt/hyemin/qwen_train_eval/outputs/four_action_generalization_diagnostics_v1",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    plan_path = Path(args.plan)
    polar_config_path = Path(args.polar_config)
    online_config_path = Path(args.online_config)
    plan_sha = file_sha256(plan_path)
    polar_config_sha = file_sha256(polar_config_path)
    online_config_sha = file_sha256(online_config_path)
    polar_config = yaml.safe_load(polar_config_path.read_text(encoding="utf-8"))
    online_config = yaml.safe_load(online_config_path.read_text(encoding="utf-8"))
    if polar_config["data"]["manifest_sha256"] != online_config["data"]["manifest_sha256"]:
        raise RuntimeError("diagnostic inputs do not share one frozen manifest")
    if polar_config["data"]["boundary_manifest_sha256"] != online_config["data"]["boundary_manifest_sha256"]:
        raise RuntimeError("diagnostic inputs do not share one boundary manifest")

    manifest_path = Path(polar_config["data"]["manifest"])
    boundary_path = Path(polar_config["data"]["boundary_manifest"])
    rows = load_verified_manifest(manifest_path, polar_config["data"]["manifest_sha256"])
    boundaries = load_jsonl(boundary_path)
    if file_sha256(boundary_path) != polar_config["data"]["boundary_manifest_sha256"]:
        raise RuntimeError("diagnostic boundary checksum mismatch")

    polar_selection_path = Path(polar_config["reporting"]["output_dir"]) / "best_checkpoint.json"
    online_selection_path = Path(online_config["reporting"]["output_dir"]) / "best_checkpoint.json"
    polar_selection = json.loads(polar_selection_path.read_text(encoding="utf-8"))
    online_selection = json.loads(online_selection_path.read_text(encoding="utf-8"))
    polar_epoch = int(polar_selection.get("selected_epoch", polar_selection["best_epoch"]))
    online_epoch = int(
        online_selection.get(
            "selected_epoch",
            online_selection.get("behavioral_selection", {}).get(
                "selected_epoch", online_selection["best_epoch"]
            ),
        )
    )
    if polar_epoch != 15 or online_epoch != 14:
        raise RuntimeError("diagnostic selected epochs differ from the authorized plan")
    polar_checkpoint = Path(polar_selection["best_checkpoint"])
    online_checkpoint = Path(online_selection["best_checkpoint"])
    if file_sha256(polar_checkpoint) != polar_selection["best_checkpoint_sha256"]:
        raise RuntimeError("selected POLAR checkpoint checksum mismatch")
    if file_sha256(online_checkpoint) != online_selection["best_checkpoint_sha256"]:
        raise RuntimeError("selected online checkpoint checksum mismatch")

    state_rows, audit = build_matched_state_manifest(rows, boundaries, seed=args.seed)
    if not audit["passed"] or audit["records"] != 1280 or audit["pairs"] != 640:
        raise RuntimeError(f"diagnostic state matching failed: {audit}")
    output_dir = Path(args.output_dir)
    state_path = output_dir / "state_manifest.jsonl"
    audit_path = output_dir / "state_manifest_audit.json"
    config_path = output_dir / "diagnostic_config.yaml"
    protocol_path = output_dir / "diagnostic_protocol.md"
    write_frozen(state_path, jsonl_text(state_rows))
    audit.update(
        {
            "source_plan": str(plan_path),
            "source_plan_sha256": plan_sha,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": file_sha256(manifest_path),
            "boundary_manifest": str(boundary_path),
            "boundary_manifest_sha256": file_sha256(boundary_path),
            "state_manifest": str(state_path),
            "state_manifest_sha256": file_sha256(state_path),
        }
    )
    write_frozen(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")

    config = {
        "protocol_version": "four_action_generalization_diagnostics_v1",
        "source_plan": str(plan_path),
        "source_plan_sha256": plan_sha,
        "parent_configs": {
            "polar": {"path": str(polar_config_path), "sha256": polar_config_sha},
            "online": {"path": str(online_config_path), "sha256": online_config_sha},
        },
        "checkpoints": {
            "polar": {
                "epoch": 15,
                "path": str(polar_checkpoint),
                "sha256": file_sha256(polar_checkpoint),
            },
            "online": {
                "epoch": 14,
                "path": str(online_checkpoint),
                "sha256": file_sha256(online_checkpoint),
            },
        },
        "data": {
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": file_sha256(manifest_path),
            "source_metadata_manifest": polar_config["data"]["source_manifest"],
            "source_metadata_manifest_sha256": polar_config["data"]["source_manifest_sha256"],
            "boundary_manifest": str(boundary_path),
            "boundary_manifest_sha256": file_sha256(boundary_path),
            "visual_feature_manifest": polar_config["visual_features"]["manifest"],
            "visual_feature_manifest_sha256": polar_config["visual_features"]["manifest_sha256"],
            "state_manifest": str(state_path),
            "state_manifest_sha256": file_sha256(state_path),
            "state_manifest_audit": str(audit_path),
            "state_manifest_audit_sha256": file_sha256(audit_path),
            "records": audit["records"],
            "pairs": audit["pairs"],
        },
        "action_order": list(polar_config["policy"]["action_order"]),
        "extraction": {
            "seed": args.seed,
            "shuffle_seed": args.seed + 1,
            "world_size": 4,
            "online_shuffle": "joint_text_and_visual_within_split_dataset_layer",
            "raw_output_root": str(Path(args.raw_output_root)),
            "feature_shard_root": str(Path(args.raw_output_root) / "state_features"),
            "dtype": "bfloat16",
        },
        "analysis": {
            "binary_threshold": 0.5,
            "depth_bins": {"early": [0, 9], "middle": [10, 18], "late": [19, 27]},
            "bit_ambiguous_policy": "exclude_from_that_bit_metric_and_report_count",
            "multi_valid_policy": "set_valued_only_no_arbitrary_mechanism_class",
            "layer_only_baseline": "train_empirical_probability_by_exact_layer",
            "layer_only_alpha": 0.5,
        },
        "probes": {
            "seed": args.seed + 2,
            "models": ["linear", "mlp"],
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "mlp_hidden_size": 64,
            "dropout": 0.0,
            "checkpoint_selection": "none_fixed_final_epoch",
            "preprocessing": "training_only_featurewise_mean_std",
            "knn_k": [5, 10, 20],
            "knn_metric": "cosine",
            "knn_pool_fallback": [
                "same_dataset_layer",
                "same_layer",
                "same_dataset_depth_bin",
                "global",
            ],
        },
        "label_incompleteness": {
            "seed": args.seed + 3,
            "cap_per_architecture_action": 12,
            "max_known_suffixes": 8,
            "correctness_interpretation": "positive_proves_cached_incomplete_negative_is_no_bounded_rescue",
        },
        "reporting": {
            "analysis_dir": str(output_dir),
            "state_outputs": str(Path(args.raw_output_root) / "state_outputs.pt"),
            "diagnostic_summary": str(output_dir / "diagnostic_summary.md"),
            "decision_summary": str(output_dir / "decision_summary.md"),
            "figures": str(output_dir / "figures"),
        },
    }
    write_frozen(config_path, yaml.safe_dump(config, sort_keys=False))
    write_frozen(protocol_path, render_protocol(config, audit))
    print(
        json.dumps(
            {
                "event": "four_action_generalization_diagnostic_prepared",
                "passed": audit["passed"],
                "records": audit["records"],
                "pairs": audit["pairs"],
                "state_manifest_sha256": config["data"]["state_manifest_sha256"],
                "config_sha256": file_sha256(config_path),
                "protocol_sha256": file_sha256(protocol_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
