#!/usr/bin/env python3
"""Freeze the W2C WHEN repair protocol, audit, smoke, and full shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.prepare_four_action_collapse import write_frozen
from experiments.train_binary_polar import file_sha256
from four_action_online_router.data import load_jsonl, load_verified_manifest
from four_action_policy.when_repair import (
    assign_cost_balanced_shards,
    build_known_full_candidates,
    local_suffix_search_plan,
    maximal_full_boundary,
    select_repair_smoke,
)


CODE_PATHS = (
    "experiments/run_w2c_when_repair.py",
    "four_action_policy/when_repair.py",
    "experiments/evaluate_four_action_polar_external.py",
    "experiments/train_four_action_online_router.py",
    "four_action_online_router/data.py",
    "label_regeneration/runtime.py",
    "binary_policy/executor/cache.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/masks.py",
    "binary_policy/executor/model.py",
    "reference/dvr_qwen/eval_metrics.py",
)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _percentile(values: Sequence[int], probability: float) -> float:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[index])


def _depth(layer: int) -> str:
    return "early" if layer <= 9 else ("middle" if layer <= 18 else "late")


def _mechanism(actions: Sequence[str]) -> str:
    values = sorted(set(str(value) for value in actions))
    return values[0] if len(values) == 1 else "MULTI"


def render_pre_repair_audit(audit: dict[str, Any]) -> str:
    lines = []
    for split in ("train", "validation"):
        row = audit["by_split"][split]
        lines.append(
            f"| {split.title()} | {row['samples']} | {row['valid_routes']} | "
            f"{row['mean_valid_routes']:.3f} | {row['median_valid_routes']:.1f} | "
            f"{row['single_suffix_samples']} | {row['multi_suffix_samples']} |"
        )
    return f"""# W2C WHEN Repair — Pre-Repair Audit

Frozen before smoke or repair execution.

## Authoritative population

| Split | W2C samples | Valid routes | Mean routes/sample | Median | Single suffix | Multiple suffixes |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

- Total W2C samples: {audit['samples']}.
- Total existing valid routes: {audit['valid_routes']}.
- Overall mean/median valid routes: {audit['mean_valid_routes']:.3f} /
  {audit['median_valid_routes']:.1f}; P95 {audit['p95_valid_routes']:.1f};
  maximum {audit['max_valid_routes']}.
- Compatible known suffixes at old boundaries: {audit['compatible_suffixes']}
  total; mean {audit['mean_compatible_suffixes']:.3f}; median
  {audit['median_compatible_suffixes']:.1f}; maximum
  {audit['max_compatible_suffixes']}.
- Samples with one/multiple compatible suffixes:
  {audit['single_suffix_samples']} / {audit['multi_suffix_samples']}.
- Old boundary depth distribution: {audit['old_boundary_depth_counts']}.
- Dataset counts: {audit['dataset_counts']}.
- Old mechanism counts: {audit['old_mechanism_counts']}.
- Initial known-FULL candidate routes after deduplication:
  {audit['initial_known_candidates']}.
- Initial one-edit suffix population: {audit['initial_one_edit_available']};
  frozen-budget selected maximum {audit['initial_one_edit_selected']}.
- All {audit['source_records_verified']} physical source label records exist and
  match the SHA-256 stored in the authoritative manifest.

These are search-derived caches. Existing valid routes are authoritative input
to repair, not proof that their boundary action sets are complete.
"""


def render_protocol(config: dict[str, Any], audit: dict[str, Any], smoke_audit: dict[str, Any]) -> str:
    return f"""# W2C Route-Cache Repair Protocol

Frozen before any smoke or full repair outcome was observed.

## Authority

- Source plan: `{config['source_plan']}`
- Source-plan SHA-256: `{config['source_plan_sha256']}`
- Executor implementation commit: `{config['executor_contract']['git_commit']}`
- Parent online config: `{config['parent_online_config']['path']}`
- Source W2C manifest: `{config['data']['source_manifest']}`
- Boundary manifest: `{config['data']['boundary_manifest']}`
- Physical source-label root: `{config['data']['source_label_root']}`
- Model revision: `{config['executor_contract']['model_revision']}`
- Existing executor contract: `{config['executor_contract']['parent_executor_contract_sha256']}`
- Action order: `IGNORE`, `READ_ONLY`, `WRITE_ONLY`, `FULL`.
- Evaluator/answer-normalization contract: source `metric_name`, `answer`,
  `all_answer_norms`, and `correctness_threshold`, implemented by the frozen
  evaluator code hashes in the config.

Only the {audit['samples']} W2C records in the authoritative matched population
are repaired ({audit['by_split']['train']['samples']} train,
{audit['by_split']['validation']['samples']} validation). C2C records and all
original cache files remain unchanged.

## Iterative repair

For each sample, compute the maximal all-FULL prefix over the versioned correct
route cache. At its next non-FULL boundary:

1. force FULL and execute every deduplicated same-sample compatible known suffix;
2. if any are correct, add every correct route, record a verified CONTINUE
   state, recompute the maximal prefix, and repeat;
3. if known suffixes are exhausted, construct every one-action mutation strictly
   after the boundary from those FULL-insertion routes;
4. select at most {config['search']['per_state_variant_budget']} candidates with
   deterministic layer-stratified round-robin ordering and stable within-layer
   hashing (seed {config['search']['seed']});
5. add every correct route and repeat, or emit
   `FULL_UNRESCUED_UNDER_BUDGET` when none is correct.

The 96-route budget is fixed because it exhausts every possible one-position
mutation of one 27-layer suffix (maximum 81) and retains limited diversity for
multi-suffix states. It initially selects {audit['initial_one_edit_selected']}
of {audit['initial_one_edit_available']} unique variants and fully exhausts the
one-edit population for {audit['initial_one_edit_fully_covered_samples']}/640
states. No two-edit sweep or four-action MCTS is admitted. The existing MCTS is
binary-only; adapting it would introduce an unvalidated search method.

`FULL_UNRESCUED_UNDER_BUDGET` means only that no rescue was found in this frozen
known-suffix plus one-edit neighborhood. It is not global impossibility.

## Parallelism, caching, and failure handling

- Four direct GPUs, one process per GPU; no Slurm.
- Deterministic cost-balanced static shards from the frozen manifests.
- Atomic per-sample records under `{config['execution']['raw_output_root']}`.
- Cache key: UID + complete 28-action route + model revision + executor/code
  contract. Exact routes execute at most once within a repair record; completed
  smoke samples are reused by the full repair.
- Runtime/evaluator failures and a correct all-FULL route for a W2C sample are
  quarantined as `UNRESOLVED`, never silently labeled.

## Smoke gate

The {smoke_audit['records']}-sample smoke has four records per dataset, six
previously cache-incomplete and six previously bounded-invalid states, six
single- and six multi-suffix states, and all early/middle/late depths. It must
verify all ten checks from the plan, including old-route replay correctness,
iteration, bounded-search ordering, cache deduplication, resume consistency,
and deterministic output. Any quarantine or replay failure stops the full run.

## Post-repair decision

Rebuild `CONTINUE`, `DEVIATE_CANDIDATE`, and `UNRESOLVED` labels without
overwriting the original cache. Re-audit repaired validation candidates under
the identical cached bounded contract with 10,000 UID-group bootstrap draws
(seed {config['post_repair']['bootstrap_seed']}).

Stage-1 gate training is not part of this phase. A future gate is eligible only
if there are at least 512 repaired train and 128 repaired validation DEVIATE
candidates, no major executor inconsistency, and the repaired label audit is
acceptable under this explicitly bounded semantics. Stop after Q1--Q6.

## Independent challenge

One read-only research reviewer ranked this strategy above adapting binary MCTS
or using known suffixes alone (confidence medium). Its strongest objection is
preserved: residual labels establish only local bounded non-rescue, not global
necessity.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="plans/w2c_when_label_repair_plan.md"
    )
    parser.add_argument(
        "--online-config",
        default="analysis/persistent_corrective_supervision/online_config.yaml",
    )
    parser.add_argument(
        "--prior-subset",
        default="analysis/selective_continue_deviate/when_full_insertion_subset.json",
    )
    parser.add_argument(
        "--prior-results",
        default="analysis/selective_continue_deviate/when_label_completeness_results.json",
    )
    parser.add_argument(
        "--source-label-root",
        default="datasets/mcts_labels_4action/sequential_branching_v1/full/records",
    )
    parser.add_argument("--output-dir", default="analysis/w2c_when_repair")
    parser.add_argument(
        "--raw-output-root",
        default="/mnt/hyemin/qwen_train_eval/outputs/w2c_when_repair_v1",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    plan_path = Path(args.plan)
    parent_path = Path(args.online_config)
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    manifest_path = Path(parent["data"]["manifest"])
    boundary_path = Path(parent["data"]["boundary_manifest"])
    rows = load_verified_manifest(manifest_path, parent["data"]["manifest_sha256"])
    boundaries = load_jsonl(boundary_path)
    if file_sha256(boundary_path) != parent["data"]["boundary_manifest_sha256"]:
        raise RuntimeError("boundary-manifest checksum mismatch")
    boundary_by_uid = {str(row["uid"]): row for row in boundaries}
    w2c = [row for row in rows if row["route_type"] == "W2C"]
    if Counter(row["split"] for row in w2c) != {"train": 512, "validation": 128}:
        raise RuntimeError("authoritative W2C split is not 512 train + 128 validation")

    source_label_root = Path(args.source_label_root)
    source_verified = 0
    repair_rows = []
    route_counts = []
    suffix_counts = []
    initial_known = 0
    initial_available = 0
    initial_selected = 0
    fully_covered = 0
    for row in w2c:
        boundary = boundary_by_uid[str(row["uid"])]
        routes = [tuple(str(value) for value in route["actions"]) for route in row["valid_routes"]]
        old_boundary, compatible = maximal_full_boundary(routes)
        if old_boundary != int(boundary["boundary_layer"]):
            raise RuntimeError("recomputed old boundary differs from frozen boundary")
        known = build_known_full_candidates(routes, boundary=old_boundary)
        search = local_suffix_search_plan(
            known,
            boundary=old_boundary,
            uid=str(row["uid"]),
            seed=args.seed,
            budget=96,
            excluded_routes=set(routes) | {
                tuple(candidate["actions"]) for candidate in known
            },
        )
        physical_record = source_label_root / Path(row["source_record_path"]).name
        if not physical_record.is_file() or file_sha256(physical_record) != row["source_record_sha256"]:
            raise RuntimeError(f"missing or mismatched source record: {row['uid']}")
        source_verified += 1
        route_counts.append(len(routes))
        suffix_counts.append(len(compatible))
        initial_known += len(known)
        initial_available += int(search["available_candidates"])
        initial_selected += int(search["selected_candidates"])
        fully_covered += int(search["available_candidates"] <= 96)
        repair_rows.append(
            {
                "uid": str(row["uid"]),
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "dataset": str(row["dataset"]),
                "old_boundary": old_boundary,
                "old_depth_bin": _depth(old_boundary),
                "old_mechanism": _mechanism(boundary["valid_nonfull_actions"]),
                "valid_route_count": len(routes),
                "compatible_suffix_count": len(compatible),
                "initial_known_candidates": len(known),
                "initial_one_edit_available": search["available_candidates"],
                "initial_one_edit_selected": search["selected_candidates"],
                "estimated_cost": len(known) + int(search["selected_candidates"]),
                "boundary_route_keys": list(boundary["boundary_route_keys"]),
                "source_binary_route_ids": list(boundary["source_binary_route_ids"]),
                "source_record": str(physical_record),
                "source_record_sha256": row["source_record_sha256"],
            }
        )

    def split_audit(split: str) -> dict[str, Any]:
        selected = [row for row in repair_rows if row["split"] == split]
        return {
            "samples": len(selected),
            "valid_routes": sum(row["valid_route_count"] for row in selected),
            "mean_valid_routes": statistics.mean(row["valid_route_count"] for row in selected),
            "median_valid_routes": statistics.median(row["valid_route_count"] for row in selected),
            "single_suffix_samples": sum(row["compatible_suffix_count"] == 1 for row in selected),
            "multi_suffix_samples": sum(row["compatible_suffix_count"] > 1 for row in selected),
        }

    audit = {
        "schema_version": "w2c_when_repair_pre_audit_v1",
        "samples": len(repair_rows),
        "valid_routes": sum(route_counts),
        "mean_valid_routes": statistics.mean(route_counts),
        "median_valid_routes": statistics.median(route_counts),
        "p95_valid_routes": _percentile(route_counts, 0.95),
        "max_valid_routes": max(route_counts),
        "compatible_suffixes": sum(suffix_counts),
        "mean_compatible_suffixes": statistics.mean(suffix_counts),
        "median_compatible_suffixes": statistics.median(suffix_counts),
        "max_compatible_suffixes": max(suffix_counts),
        "single_suffix_samples": sum(value == 1 for value in suffix_counts),
        "multi_suffix_samples": sum(value > 1 for value in suffix_counts),
        "by_split": {split: split_audit(split) for split in ("train", "validation")},
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in repair_rows).items())),
        "old_boundary_depth_counts": dict(
            sorted(Counter(row["old_depth_bin"] for row in repair_rows).items())
        ),
        "old_mechanism_counts": dict(
            sorted(Counter(row["old_mechanism"] for row in repair_rows).items())
        ),
        "initial_known_candidates": initial_known,
        "initial_one_edit_available": initial_available,
        "initial_one_edit_selected": initial_selected,
        "initial_one_edit_fully_covered_samples": fully_covered,
        "source_records_verified": source_verified,
    }

    prior_subset_path = Path(args.prior_subset)
    prior_results_path = Path(args.prior_results)
    prior_subset = json.loads(prior_subset_path.read_text(encoding="utf-8"))
    prior_results = json.loads(prior_results_path.read_text(encoding="utf-8"))
    smoke, smoke_audit = select_repair_smoke(
        prior_subset, prior_results["state_results"], seed=args.seed
    )
    repair_by_uid = {row["uid"]: row for row in repair_rows}
    smoke_rows = []
    for state in smoke:
        row = dict(repair_by_uid[state["uid"]])
        row.update(
            {
                "prior_status": state["prior_status"],
                "suffix_class": state["suffix_class"],
                "estimated_cost": row["estimated_cost"] + row["valid_route_count"],
            }
        )
        smoke_rows.append(row)
    smoke_assigned, smoke_shards = assign_cost_balanced_shards(smoke_rows, world_size=4)
    repair_assigned, repair_shards = assign_cost_balanced_shards(repair_rows, world_size=4)

    output_dir = Path(args.output_dir)
    smoke_manifest_path = output_dir / "smoke" / "smoke_manifest.json"
    repair_manifest_path = output_dir / "repair_manifest.json"
    pre_audit_json_path = output_dir / "pre_repair_audit.json"
    pre_audit_md_path = output_dir / "pre_repair_audit.md"
    config_path = output_dir / "repair_config.yaml"
    protocol_path = output_dir / "repair_protocol.md"
    write_frozen(smoke_manifest_path, _json(smoke_assigned))
    write_frozen(repair_manifest_path, _json(repair_assigned))
    audit.update({"smoke_shards": smoke_shards, "repair_shards": repair_shards})
    write_frozen(pre_audit_json_path, _json(audit))
    write_frozen(pre_audit_md_path, render_pre_repair_audit(audit))

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    config = {
        "protocol_version": "w2c_when_repair_v1",
        "source_plan": str(plan_path),
        "source_plan_sha256": file_sha256(plan_path),
        "parent_online_config": {"path": str(parent_path), "sha256": file_sha256(parent_path)},
        "data": {
            "source_label_root": str(source_label_root),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": file_sha256(manifest_path),
            "boundary_manifest": str(boundary_path),
            "boundary_manifest_sha256": file_sha256(boundary_path),
            "source_metadata_manifest": parent["data"]["source_manifest"],
            "source_metadata_manifest_sha256": parent["data"]["source_manifest_sha256"],
            "prior_audit_subset": str(prior_subset_path),
            "prior_audit_subset_sha256": file_sha256(prior_subset_path),
            "prior_audit_results": str(prior_results_path),
            "prior_audit_results_sha256": file_sha256(prior_results_path),
            "smoke_manifest": str(smoke_manifest_path),
            "smoke_manifest_sha256": file_sha256(smoke_manifest_path),
            "repair_manifest": str(repair_manifest_path),
            "repair_manifest_sha256": file_sha256(repair_manifest_path),
            "pre_repair_audit": str(pre_audit_json_path),
            "pre_repair_audit_sha256": file_sha256(pre_audit_json_path),
            "w2c_samples": 640,
            "train_w2c": 512,
            "validation_w2c": 128,
        },
        "executor_contract": {
            "git_commit": git_commit,
            "code_sha256": {path: file_sha256(Path(path)) for path in CODE_PATHS},
            "parent_executor_contract_sha256": parent["executor"]["contract_sha256"],
            "implementation": "unified_complete_four_action_route_v1",
            "model_revision": parent["base_model"]["revision"],
            "model_path": parent["base_model"]["path"],
            "evaluator_contract": "source_metric_answer_norms_threshold_v1",
            "eos_token_ids": parent["external_evaluation"]["eos_token_ids"],
            "repetition_penalty": parent["external_evaluation"]["repetition_penalty"],
        },
        "search": {
            "strategy": "known_suffix_then_stratified_one_edit_suffix_variants",
            "seed": args.seed,
            "per_state_variant_budget": 96,
            "mutation_distance": 1,
            "mutation_positions": "strictly_after_candidate_boundary",
            "selection": "round_robin_by_stable_hashed_layer_then_stable_hashed_route",
            "max_boundary_advances": 28,
            "two_edit_search": False,
            "mcts": False,
        },
        "execution": {
            "seed": args.seed,
            "world_size": 4,
            "workers_per_gpu": 1,
            "scheduler": "direct_no_slurm",
            "raw_output_root": str(Path(args.raw_output_root)),
            "record_cache_key": "uid_route_model_revision_executor_contract",
        },
        "smoke": {
            "records": 12,
            "require_all_old_route_replays_correct": True,
            "require_zero_quarantine": True,
            "require_resume_checksum_stability": True,
            "require_all_plan_checks": True,
        },
        "post_repair": {
            "minimum_train_deviate_candidates": 512,
            "minimum_validation_deviate_candidates": 128,
            "bootstrap_draws": 10000,
            "bootstrap_seed": args.seed + 1,
            "audit_contract": "identical_cached_known_suffix_and_one_edit_budget",
        },
        "reporting": {
            "analysis_dir": str(output_dir),
            "decision_summary": str(output_dir / "decision_summary.md"),
        },
    }
    write_frozen(config_path, yaml.safe_dump(config, sort_keys=False))
    write_frozen(protocol_path, render_protocol(config, audit, smoke_audit))
    print(
        json.dumps(
            {
                "event": "w2c_when_repair_prepared",
                "w2c_samples": len(repair_rows),
                "smoke_samples": len(smoke_assigned),
                "initial_known_candidates": initial_known,
                "initial_one_edit_selected": initial_selected,
                "config_sha256": file_sha256(config_path),
                "protocol_sha256": file_sha256(protocol_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
