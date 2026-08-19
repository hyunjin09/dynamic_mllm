#!/usr/bin/env python3
"""Freeze the P9 provenance/checksum bundle and final label-generation report."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


PRIMARY_PATHS = (
    "plans/dynamic_mllm_label_regeneration_plan.md",
    "outputs/label_regeneration/v1/frozen_execution_contract.json",
    "outputs/label_regeneration/v1/frozen_execution_contract.md",
    "outputs/label_regeneration/v1/source_manifest_v1.jsonl",
    "outputs/label_regeneration/v1/smoke_manifest_v1.jsonl",
    "outputs/label_regeneration/v1/smoke_report_v1.json",
    "outputs/label_regeneration/v1/smoke_report.md",
    "outputs/label_regeneration/v1/p3_resume_amendment_v1.json",
    "outputs/label_regeneration/v1/post_generation/cache_audit_v1.json",
    "outputs/label_regeneration/v1/post_generation/cache_record_index_v1.jsonl",
    "outputs/label_regeneration/v1/post_generation/label_quality_summary_p5_v1.json",
    "outputs/label_regeneration/v1/post_generation/per_sample_route_summary_v1.jsonl",
    "outputs/label_regeneration/v1/post_generation/route_diversity_summary_p6_v1.json",
    "outputs/label_regeneration/v1/post_generation/per_sample_route_diversity_v1.jsonl",
    "outputs/label_regeneration/v1/post_generation/predictor_split_manifest_v1.jsonl",
    "outputs/label_regeneration/v1/post_generation/predictor_split_audit_v1.json",
    "outputs/label_regeneration/v1/post_generation/derived_single_best_manifest_v1.jsonl",
    "outputs/label_regeneration/v1/post_generation/derived_valid_set_manifest_v1.jsonl",
    "outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl",
    "outputs/label_regeneration/v1/post_generation/derived_route_ranking_manifest_v1.jsonl",
    "outputs/label_regeneration/v1/post_generation/derived_polar_segment_manifest_v1.jsonl",
    "outputs/label_regeneration/v1/post_generation/derived_supervision_audit_v1.json",
    "outputs/label_regeneration/v1/post_generation/derived_supervision_verification_v1.json",
    "reports/label_regeneration_p5_summary.md",
    "reports/label_regeneration_p6_route_diversity.md",
    "reports/label_regeneration_p7_predictor_split.md",
    "reports/label_regeneration_p8_derived_supervision.md",
)

CODE_PATHS = (
    "experiments/freeze_label_regeneration.py",
    "experiments/run_label_regeneration.py",
    "binary_policy/executor/cache.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/masks.py",
    "binary_policy/executor/model.py",
    "label_regeneration/data.py",
    "label_regeneration/mcts.py",
    "label_regeneration/runtime.py",
    "label_regeneration/summary.py",
    "label_regeneration/diversity.py",
    "label_regeneration/derived.py",
    "tools/audit_label_regeneration_p4.py",
    "tools/summarize_label_regeneration_p5.py",
    "tools/analyze_label_regeneration_p6.py",
    "tools/freeze_binary_predictor_split.py",
    "tools/build_label_regeneration_p8.py",
    "tools/verify_label_regeneration_p8.py",
    "tools/finalize_label_regeneration_p9.py",
    "configs/binary_polar_loss_comparison_v1.yaml",
)

JOB_PREFIXES = (
    "label_regeneration_p2_",
    "label_regeneration_p3_",
    "label_regeneration_p4_",
    "label_regeneration_p5_",
    "label_regeneration_p6_",
    "label_regeneration_p7_",
    "label_regeneration_p8_",
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sidecar(path: Path) -> str:
    value = digest(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{value}  {path.name}\n", encoding="utf-8")
    return value


def inventory_entry(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise RuntimeError(f"required P9 file is missing: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)}


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def final_report(
    *,
    contract: dict[str, Any],
    smoke: dict[str, Any],
    p4: dict[str, Any],
    p5: dict[str, Any],
    p6: dict[str, Any],
    p7: dict[str, Any],
    p8: dict[str, Any],
    verify: dict[str, Any],
    inventory_sha: str,
    provenance_sha: str,
) -> str:
    overall = p5["overall"]
    div = p6["overall"]
    lines = [
        "# Dynamic MLLM Binary Routing Label Generation Report",
        "",
        "P9 status: **PASS**",
        "",
        "This report closes P0–P9 for the regenerated unrestricted 28-bit visual ON/OFF route cache. "
        "It does not contain predictor-training or external-evaluation results.",
        "",
        "## Why the labels were regenerated",
        "",
        "The prior MCTS cache was not portable to the repaired binary executor: cached outputs and some "
        "previously positive masks failed exact reproduction. The project therefore regenerated every "
        "authoritative route outcome under one pinned Qwen2.5-VL execution contract rather than deleting "
        "only known mismatches or copying historical validity labels.",
        "",
        "## Frozen execution contract",
        "",
        f"- Model: `{contract['model']}` at revision `{contract['model_revision']}`.",
        f"- Executor: `{contract['executor']}`; 28 independent layer actions.",
        f"- ON: {contract['route_semantics']['on']}.",
        f"- OFF: {contract['route_semantics']['off']}.",
        f"- Precision/attention: `{contract['dtype']}` / `{contract['attention_implementation']}`.",
        f"- Image processing: {contract['image_processing']}; custom max-image-token cap: `{contract['custom_max_image_tokens']}`.",
        f"- Generation: deterministic greedy, `max_new_tokens={contract['generation']['max_new_tokens']}`.",
        f"- Environment: Python `{contract['python_version']}`, PyTorch `{contract['packages']['torch']}`, "
        f"Transformers `{contract['packages']['transformers']}`.",
        f"- Frozen contract SHA-256: `{contract['contract_sha256']}`.",
        "- Git revision: unavailable because the workspace was not a Git checkout. The contract and P3 "
        "resume amendment record deterministic source-file hashes instead.",
        "",
        "## Validity and completeness gates",
        "",
        f"- P2 ALL-ON/native generated-token parity: `{smoke['all_on_parity_count']}/{smoke['all_on_required']}`.",
        f"- P2 repeated mixed-route determinism: `{smoke['mixed_determinism_passed']}`.",
        f"- P4 terminal raw records: `{p4['valid_terminal_records']:,}/{p4['expected_records']:,}`.",
        f"- Missing/unexpected/duplicate/invalid/error/temp/zero-byte records: "
        f"`{len(p4['missing_uids'])}/{len(p4['unexpected_uids'])}/{len(p4['duplicate_records'])}/"
        f"{len(p4['invalid_records'])}/{len(p4['error_record_paths'])}/{len(p4['temporary_record_paths'])}/"
        f"{len(p4['zero_byte_record_paths'])}`.",
        f"- P4 raw-record checksum ledger: `{p4['record_index_path']}` "
        f"(`{p4['record_index_sha256']}`).",
        "",
        "## Dataset and current ALL-ON outcomes",
        "",
        "Historical source strata are metadata only; current results below were recomputed by the frozen executor.",
        "",
        "| Dataset | Samples | Current correct | Current wrong | Wrong with correcting route | Fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("gqa", "textvqa", "chartqa"):
        data = p5["by_dataset"][dataset]
        lines.append(
            f"| {dataset.upper()} | {data['samples']:,} | {data['current_all_on']['correct']:,} | "
            f"{data['current_all_on']['wrong']:,} | {data['correction']['recovered']:,} | "
            f"{pct(data['correction']['recovery_fraction'])} |"
        )
    lines += [
        f"| **Total** | **{overall['samples']:,}** | **{overall['current_all_on']['correct']:,}** | "
        f"**{overall['current_all_on']['wrong']:,}** | **{overall['correction']['recovered']:,}** | "
        f"**{pct(overall['correction']['recovery_fraction'])}** |",
        "",
        "Contract drift versus historical metadata:",
        "",
        f"- Stable correct: `{overall['contract_drift']['stable_correct']:,}`; stable wrong: "
        f"`{overall['contract_drift']['stable_wrong']:,}`.",
        f"- Historical correct → current wrong: `{overall['contract_drift']['historical_correct_to_current_wrong']:,}`.",
        f"- Historical wrong → current correct: `{overall['contract_drift']['historical_wrong_to_current_correct']:,}`.",
        "",
        "## Search budgets and label yield",
        "",
        f"- Samples at 200/400/600 MCTS simulations: `{overall['search_budgets']['200']:,}` / "
        f"`{overall['search_budgets']['400']:,}` / `{overall['search_budgets']['600']:,}`.",
        f"- Evaluated routes: `{p8['totals']['evaluated_routes']:,}`; mean/median per sample: "
        f"`{overall['coverage']['evaluated_route_count']['mean']:.2f}` / "
        f"`{overall['coverage']['evaluated_route_count']['median']:.1f}`.",
        f"- Valid routes: `{p8['totals']['raw_valid_routes']:,}`; mean/median per sample: "
        f"`{overall['coverage']['valid_route_count']['mean']:.2f}` / "
        f"`{overall['coverage']['valid_route_count']['median']:.1f}`.",
        f"- Samples with ≥1/≥5/≥10/≥20 valid routes: `{overall['coverage']['with_at_least_1']:,}` / "
        f"`{overall['coverage']['with_at_least_5']:,}` / `{overall['coverage']['with_at_least_10']:,}` / "
        f"`{overall['coverage']['with_at_least_20']:,}`; zero-positive: `{overall['coverage']['zero_valid']:,}`.",
        "",
        "For the 4,045 current-ALL-ON-correct records, the minimum-budget successful route uses "
        f"a mean/median `{overall['preservation']['minimum_visual_on_valid_route']['mean']:.2f}` / "
        f"`{overall['preservation']['minimum_visual_on_valid_route']['median']:.1f}` visual-ON layers. "
        "These are oracle label statistics, not learned-policy or latency results.",
        "",
        "## Route structure",
        "",
        f"- Valid masks analyzed: `{div['valid_masks']:,}` across `{div['samples_with_valid_routes']:,}` samples.",
        f"- Sample-balanced mean transitions: `{div['sample_balanced']['mean_transition_count']['mean']:.2f}`.",
        f"- Sample-balanced mean within-sample pairwise Hamming distance: "
        f"`{div['sample_balanced']['mean_pairwise_hamming']['mean']:.2f}/28`.",
        f"- Exact within-sample unordered mask pairs: `{div['route_weighted']['pairwise_hamming']['count']:,}`.",
        f"- Masks with ≤3 transitions: `{div['structural_frequencies']['transition_le_3']:,}` "
        f"({pct(div['structural_frequencies']['transition_le_3_fraction'])}); with ≥14 transitions: "
        f"`{div['structural_frequencies']['transition_ge_14']:,}` "
        f"({pct(div['structural_frequencies']['transition_ge_14_fraction'])}).",
        "",
        "## Predictor split and P8 supervision",
        "",
        f"- Frozen split: `{p7['totals']['train_records']:,}` train / "
        f"`{p7['totals']['validation_records']:,}` validation; cross-split image groups: "
        f"`{p7['integrity']['cross_split_image_groups']}`.",
        "- Validation historical strata: GQA 250/250, TextVQA 125/125, ChartQA 125/125.",
        f"- Selected max-50 valid routes: `{p8['totals']['selected_valid_routes']:,}`; samples capped: "
        f"`{p8['totals']['capped_samples']:,}`.",
        f"- Shared duplicated-BCE/exact-set-NLL route-set digest: "
        f"`{p8['integrity']['selected_route_set_sha256']}`.",
        f"- Independent P8 verification passed: `{verify['passed']}`; POLAR masks reconstructed: "
        f"`{verify['observed']['polar_routes']:,}`.",
        "",
        "## Operational amendments and failures",
        "",
        "- P3 job 99741 was intentionally stopped after 2,291 atomically published records for a "
        "user-approved 4→8 GPU migration. Job 99758 contributed the remaining 5,709 records. The "
        "resume amendment changed only cross-shard discovery; the scientific contract remained unchanged.",
        "- P8 had three cancelled, unpublished performance-only attempts. The supported bottlenecks were "
        "repeated tuple Hamming work and serialized decoding of the 21 GB trace cache. Exact XOR bit-count "
        "Hamming and bounded process decoding completed the same selection contract. No raw or published "
        "artifact was deleted or altered.",
        "- One proposed P8 CPU request was rejected before submission because 32 GB exceeded the scheduler's "
        "node cap; the streaming job required only 24 GB.",
        "- Scientific failures/incomplete records in the final cache: none.",
        "",
        "## Reproducibility and checksums",
        "",
        f"- Final artifact inventory SHA-256: `{inventory_sha}`.",
        f"- Command/provenance record SHA-256: `{provenance_sha}`.",
        "- Raw records are frozen individually by the P4 record index; copying only the aggregate files "
        "without that index does not reproduce the raw-cache integrity chain.",
        "- Exact P0/P1 reproduction command and every scheduled P2–P8 command are saved in "
        "`p9_run_provenance_v1.json`.",
        "",
        "## Final gate",
        "",
        "P9 passes: the raw route cache, outcomes, diversity summaries, split identities, derived views, "
        "source/runtime provenance, operational amendments, and checksum chain are complete. Predictor "
        "training remains a separate P10 action requiring explicit approval. The first permitted P10 action "
        "is the bounded matched duplicated-BCE versus exact-set-NLL smoke—not full training.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hash-workers", type=int, default=4)
    args = parser.parse_args()
    root = args.project.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output = output.resolve()
    report_path = args.report if args.report.is_absolute() else root / args.report
    report_path = report_path.resolve()
    output.mkdir(parents=True, exist_ok=True)

    contract_path = root / "outputs/label_regeneration/v1/frozen_execution_contract.json"
    smoke_path = root / "outputs/label_regeneration/v1/smoke_report_v1.json"
    p4_path = root / "outputs/label_regeneration/v1/post_generation/cache_audit_v1.json"
    p5_path = root / "outputs/label_regeneration/v1/post_generation/label_quality_summary_p5_v1.json"
    p6_path = root / "outputs/label_regeneration/v1/post_generation/route_diversity_summary_p6_v1.json"
    p7_path = root / "outputs/label_regeneration/v1/post_generation/predictor_split_audit_v1.json"
    p8_path = root / "outputs/label_regeneration/v1/post_generation/derived_supervision_audit_v1.json"
    verify_path = root / "outputs/label_regeneration/v1/post_generation/derived_supervision_verification_v1.json"
    queue_path = root / "state/gpu_experiment_queue.json"
    contract, smoke, p4, p5, p6, p7, p8, verify, queue = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (contract_path, smoke_path, p4_path, p5_path, p6_path, p7_path, p8_path, verify_path, queue_path)
    ]

    if not all((smoke["passed"], p4["passed"], p7["passed"], p8["passed"], verify["passed"])):
        raise RuntimeError("one or more P2/P4/P7/P8 gates did not pass")
    if p4["valid_terminal_records"] != 8000 or p5["overall"]["samples"] != 8000:
        raise RuntimeError("P4/P5 population mismatch")
    if p6["overall"]["valid_masks"] != p8["totals"]["raw_valid_routes"]:
        raise RuntimeError("P6/P8 valid-route count mismatch")
    if verify["generation_audit_sha256"] != digest(p8_path):
        raise RuntimeError("P8 verification is not bound to the published generation audit")

    selected_jobs = [
        {
            key: job.get(key)
            for key in (
                "id", "status", "slurm_job_id", "node", "partition", "gpus", "cpus", "mem",
                "command", "log_path", "result_path", "notes", "created_at", "updated_at"
            )
        }
        for job in queue["jobs"]
        if job["id"].startswith(JOB_PREFIXES)
    ]
    selected_jobs.sort(key=lambda row: (row["created_at"], row["id"]))
    p0_p1_command = (
        ".venv/bin/python experiments/freeze_label_regeneration.py "
        f"--data-root {contract['source_pool']} --model-path {contract['model_path']} "
        f"--revision {contract['model_revision']} --output-root outputs/label_regeneration/v1 "
        "--smoke-seed 20260810"
    )
    provenance = {
        "schema_version": "label_regeneration_p9_run_provenance_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": None,
        "git_revision_reason": "project root was not a Git checkout",
        "p0_p1_reproduction_command": p0_p1_command,
        "frozen_contract_source_hashes": contract["source_code_sha256"],
        "resume_amendment": "outputs/label_regeneration/v1/p3_resume_amendment_v1.json",
        "scheduled_jobs": selected_jobs,
        "final_success_jobs": {
            "P2": "99740", "P3": "99741+99758", "P4": "100342", "P5": "100344",
            "P6": "100345", "P7": "100359", "P8": "100364", "P8_verification": "100365",
        },
        "operational_non_scientific_attempts": [
            "P3 99741 cancelled for user-approved scale migration after atomic publication",
            "P8 100360/100361/100363 cancelled before final publication for supported performance repairs",
            "P8 32G request rejected by scheduler before job creation; rerun used sufficient 24G",
        ],
    }
    provenance_path = output / "p9_run_provenance_v1.json"
    write_json(provenance_path, provenance)
    provenance_sha = sidecar(provenance_path)

    paths = list(PRIMARY_PATHS) + list(CODE_PATHS) + [str(provenance_path.relative_to(root))]
    with ThreadPoolExecutor(max_workers=args.hash_workers) as pool:
        entries = list(pool.map(lambda item: inventory_entry(root, item), paths))
    entries.sort(key=lambda row: row["path"])
    inventory = {
        "schema_version": "label_regeneration_p9_artifact_inventory_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_cache_checksum_representation": {
            "record_count": p4["record_index_rows"],
            "record_index_path": p4["record_index_path"],
            "record_index_sha256": p4["record_index_sha256"],
            "records_verified_by_p4": p4["valid_terminal_records"],
        },
        "files": entries,
    }
    inventory_path = output / "p9_artifact_inventory_v1.json"
    write_json(inventory_path, inventory)
    inventory_sha = sidecar(inventory_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        final_report(
            contract=contract, smoke=smoke, p4=p4, p5=p5, p6=p6, p7=p7,
            p8=p8, verify=verify, inventory_sha=inventory_sha, provenance_sha=provenance_sha,
        ),
        encoding="utf-8",
    )
    report_sha = sidecar(report_path)

    audit = {
        "schema_version": "label_regeneration_p9_final_audit_v1",
        "passed": True,
        "gates": {"P0_P1": True, "P2": True, "P3_P4": True, "P5": True, "P6": True, "P7": True, "P8": True, "P9": True},
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": digest(root / "plans/dynamic_mllm_label_regeneration_plan.md"),
        "raw_cache_records": p4["valid_terminal_records"],
        "raw_cache_record_index_sha256": p4["record_index_sha256"],
        "artifact_inventory_sha256": inventory_sha,
        "run_provenance_sha256": provenance_sha,
        "final_report_sha256": report_sha,
        "final_cache_failures": 0,
        "predictor_training_executed": False,
        "external_evaluation_executed": False,
        "next_gate": "P10_bounded_matched_smoke_requires_explicit_approval",
    }
    audit_path = output / "p9_final_audit_v1.json"
    write_json(audit_path, audit)
    audit_sha = sidecar(audit_path)

    freeze_entries = entries + [
        inventory_entry(root, str(inventory_path.relative_to(root))),
        inventory_entry(root, str(report_path.relative_to(root))),
        inventory_entry(root, str(audit_path.relative_to(root))),
    ]
    checksums_path = output / "P9_SHA256SUMS"
    checksums_path.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in sorted(freeze_entries, key=lambda row: row["path"])),
        encoding="utf-8",
    )
    checksums_sha = sidecar(checksums_path)
    print(json.dumps({
        "passed": True,
        "raw_cache_records": p4["valid_terminal_records"],
        "inventory_files": len(entries),
        "inventory_sha256": inventory_sha,
        "provenance_sha256": provenance_sha,
        "report_sha256": report_sha,
        "audit_sha256": audit_sha,
        "checksums_sha256": checksums_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
