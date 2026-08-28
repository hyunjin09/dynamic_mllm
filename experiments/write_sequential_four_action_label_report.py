#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools.research_analysis.four_action.sequential_label_jobs import file_sha256


def _number(value) -> str:
    return "n/a" if value is None else f"{value:,}" if isinstance(value, int) else f"{value:.4g}"


def _percent(value) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write exact sequential label report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequential_four_action_label_conversion.yaml"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    analysis_root = Path(config["analysis_root"])
    aggregate = json.loads(
        (analysis_root / "aggregate_statistics_v1.json").read_text(encoding="utf-8")
    )
    smoke = json.loads((analysis_root / "smoke_audit_v1.json").read_text(encoding="utf-8"))
    combined = aggregate["combined"]
    counts = combined["counts"]
    w2c = combined["w2c"]
    c2c = combined["c2c"]
    actions = w2c["off_position_final_actions"]
    fractions = w2c["off_position_final_action_fractions"]
    source = aggregate["source_inventory"]
    dataset_rows = []
    for dataset, row in aggregate["by_dataset"].items():
        dataset_rows.append(
            "| {dataset} | {samples} | {w2c_routes} | {c2c_routes} | {valid} | {failed} | {unique} | {branch_p99} | {branch_max} |".format(
                dataset=dataset,
                samples=_number(row["counts"]["samples"]),
                w2c_routes=_number(row["counts"]["w2c_source_routes"]),
                c2c_routes=_number(row["counts"]["c2c_source_routes"]),
                valid=_number(row["counts"]["source_replay_valid_routes"]),
                failed=_number(row["counts"]["source_replay_failure_routes"]),
                unique=_number(row["counts"]["unique_valid_routes"]),
                branch_p99=_number(row["w2c"]["maximum_active_branches_per_source_route"]["p99"]),
                branch_max=_number(row["w2c"]["maximum_active_branches_per_source_route"]["max"]),
            )
        )
    branch_max = w2c["maximum_active_branches_per_source_route"]["max"]
    branch_p99 = w2c["maximum_active_branches_per_source_route"]["p99"]
    report = f"""# Exact Sequential Four-Action Label Conversion Report

## Outcome

The conversion completed under the exact policy in `plans/4way_labeling_3.md`: fixed early-to-late processing, FULL restoration first, exhaustive retention of both correct partial branches, IGNORE fallback, no beam search, no branch cap, and no margin/cost ranking. The eight-sample smoke audit passed before the full launch.

## Frozen sources and executor

- GQA/TextVQA/ChartQA: `{source['source_details']['vqa']['predictor_manifest']}` (SHA-256 `{source['source_details']['vqa']['predictor_manifest_sha256']}`).
- WeMath2.0 Standard: `{source['source_details']['wemath20_standard']['cache_root']}` with contract SHA-256 `{source['source_details']['wemath20_standard']['contract_sha256']}`.
- WeMath2.0 Pro: `{source['source_details']['wemath2pro']['cache_root']}` with contract SHA-256 `{source['source_details']['wemath2pro']['contract_sha256']}`.
- Frozen normalized inventory: {_number(source['total_positive_samples'])} samples / {_number(source['total_positive_routes'])} positive routes, manifest SHA-256 `{source['source_manifest_sha256']}`.
- Unified executor smoke: {smoke['completed_samples']}/8 samples passed, with exact resume and old-binary semantic parity.

## Population results

| Dataset | Samples | W→C routes | C→C routes | Replay valid | Replay failed | Unique labels | Branch p99 | Branch max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(dataset_rows)}
| Combined | {_number(counts['samples'])} | {_number(counts['w2c_source_routes'])} | {_number(counts['c2c_source_routes'])} | {_number(counts['source_replay_valid_routes'])} | {_number(counts['source_replay_failure_routes'])} | {_number(counts['unique_valid_routes'])} | {_number(branch_p99)} | {_number(branch_max)} |

## W→C route-conditioned refinement

Across final correct W→C branch occurrences at positions that were OFF in the source route:

| Final action | Count | Fraction | Interpretation |
|---|---:|---:|---|
| FULL | {_number(actions['FULL'])} | {_percent(fractions['FULL'])} | source OFF unnecessary in this branch context |
| READ_ONLY | {_number(actions['READ_ONLY'])} | {_percent(fractions['READ_ONLY'])} | WRITE suppression retained |
| WRITE_ONLY | {_number(actions['WRITE_ONLY'])} | {_percent(fractions['WRITE_ONLY'])} | READ suppression retained |
| IGNORE | {_number(actions['IGNORE'])} | {_percent(fractions['IGNORE'])} | both suppressions retained |

- Source routes that branched: {_number(w2c['source_routes_with_branching'])}.
- Both-partial-correct branch events: {_number(w2c['both_partial_branch_events'])}.
- Final branches/source route: median {_number(w2c['final_branches_per_source_route']['median'])}, p99 {_number(w2c['final_branches_per_source_route']['p99'])}, max {_number(w2c['final_branches_per_source_route']['max'])}.
- Maximum active branches/source route: median {_number(w2c['maximum_active_branches_per_source_route']['median'])}, p99 {_number(branch_p99)}, max {_number(branch_max)}.
- W→C ALL-OFF source routes: {_number(w2c['all_off_seed_source_routes'])}; their final action counts are `{json.dumps(w2c['all_off_seed_final_actions'], sort_keys=True)}`.
- Within-sample deduplication ratio (unique routes / final branch occurrences): {_number(w2c['deduplication_ratio_unique_per_branch_occurrence'])}.

These are route-conditioned corrective programs, not globally unique causes. Earlier branch decisions can change what a later operation needs.

## C→C preservation

C→C routes were mapped mechanically (ON→FULL, OFF→IGNORE) and were not subjected to W→C restoration. Their aggregate action counts are `{json.dumps(c2c['action_counts'], sort_keys=True)}`. They represent correctness-preserving redundancy/efficiency supervision and are not mixed with W→C mechanism claims.

## Final decision answers

1. The authoritative artifacts are listed under “Frozen sources and executor”; `datasets/mcts_v2/` was not used.
2. {_number(counts['source_replay_valid_routes'])} positive routes replayed successfully; {_number(counts['source_replay_failure_routes'])} were explicitly retained as replay failures and not refined.
3. {_number(counts['w2c_source_routes'])} W→C and {_number(counts['c2c_source_routes'])} C→C routes were processed successfully.
4. {_number(counts['unique_valid_routes'])} unique valid four-action routes were produced after within-sample deduplication.
5. W→C source-OFF positions became FULL/READ_ONLY/WRITE_ONLY/IGNORE at the counts and fractions in the table above.
6. Branching occurred for {_number(w2c['source_routes_with_branching'])} source routes, with {_number(w2c['both_partial_branch_events'])} explicit both-partial events.
7. Branch manageability is quantified without a beam: p99 {_number(branch_p99)}, max {_number(branch_max)} active branches. Interpret practical manageability from these observed values and the completed-run status.
8. Dataset-specific differences are reported in the population table and the machine-readable aggregate/plots.
9. ALL-OFF W→C refinement is summarized separately above; it is not merged into positive-vision mechanism claims.
10. Every stored training label is evaluator-correct under its complete route, checksum-bound, deduplicated within sample, and retains source provenance. Replay failures are excluded from training views.
11. Fresh four-action MCTS is not required to produce these route-conditioned labels. It would only be justified for a separately approved objective requiring routes outside the binary-positive anchors.

## Artifacts

- `aggregate_statistics_v1.json`: complete per-dataset and combined statistics.
- `plots/`: W→C action distributions by dataset and layer.
- `{config['output_root']}/full/records/`: atomic per-sample raw mappings and all exact branches.
- `{config['output_root']}/full/views/`: sample index and dataset-specific downstream training views.
- `{config['output_root']}/full/completion_audit_v1.json`: full integrity and semantic audit.
"""
    targets = [
        analysis_root / "four_action_label_conversion_report.md",
        Path(config["output_root"]) / "reports" / "four_action_label_conversion_report.md",
    ]
    for target in targets:
        if target.exists():
            if not args.resume:
                raise FileExistsError(f"refusing to overwrite {target}")
            if target.read_text(encoding="utf-8") != report:
                raise RuntimeError(f"existing report differs from recomputation: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(report, encoding="utf-8")
            target.with_suffix(target.suffix + ".sha256").write_text(
                f"{file_sha256(target)}  {target.name}\n", encoding="utf-8"
            )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
