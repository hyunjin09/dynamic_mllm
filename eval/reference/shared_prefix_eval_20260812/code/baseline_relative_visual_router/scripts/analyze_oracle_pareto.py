#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from baseline_relative_visual_router.oracle_pareto import (  # noqa: E402
    align_policy_rows,
    benchmark_summaries,
    evaluate_oracle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def actual_policy_summary(aligned: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    baseline = np.asarray([row["baseline_correct"] for row in aligned], dtype=np.int8)
    routed = np.asarray([row["routes"][policy]["correct"] for row in aligned], dtype=np.int8)
    budget = np.asarray([row["routes"][policy]["budget"] for row in aligned], dtype=np.float64)
    return {
        "n": len(aligned),
        "baseline_accuracy": float(baseline.mean()),
        "router_accuracy": float(routed.mean()),
        "accuracy_delta": float((routed - baseline).mean()),
        "rescue_count": int(((baseline == 0) & (routed == 1)).sum()),
        "harm_count": int(((baseline == 1) & (routed == 0)).sum()),
        "mean_visual_on_layers": float(budget.mean()),
        "route_sensitive_layer_saving_fraction": float((28.0 - budget.mean()) / 28.0),
    }


def population_result(
    aligned: list[dict[str, Any]],
    policies: list[str],
    *,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    single = {}
    for offset, policy in enumerate(policies):
        oracle, _ = evaluate_oracle(
            aligned,
            [policy],
            bootstrap_repetitions=repetitions,
            bootstrap_seed=seed + offset,
        )
        single[policy] = {
            "actual": actual_policy_summary(aligned, policy),
            "oracle": oracle,
            "oracle_by_benchmark": benchmark_summaries(aligned, [policy]),
        }
    union, union_rows = evaluate_oracle(
        aligned,
        policies,
        bootstrap_repetitions=repetitions,
        bootstrap_seed=seed + 100,
    )
    return {
        "single_policy": single,
        "proposer_union_oracle": union,
        "proposer_union_oracle_by_benchmark": benchmark_summaries(aligned, policies),
    }, union_rows


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_markdown(output: Path, result: dict[str, Any]) -> None:
    all_result = result["populations"]["all"]
    lines = [
        "# Natural-Distribution Oracle Proposer Pareto Audit",
        "",
        "This is an outcome-aware upper bound. It assumes a free, perfect selector and",
        "does not include the cost of producing or evaluating candidate routes.",
        "`Mean ON` is a route-sensitive layer proxy, not end-to-end FLOPs or latency.",
        "",
        "## Single fixed proposer",
        "",
        "| Policy | Ungated acc | Delta | Mean ON | Rescue/Harm | Oracle acc | Oracle delta | Oracle mean ON | Saving proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, payload in all_result["single_policy"].items():
        actual = payload["actual"]
        oracle = payload["oracle"]
        lines.append(
            f"| `{policy}` | {percent(actual['router_accuracy'])} | "
            f"{100 * actual['accuracy_delta']:+.2f}pp | {actual['mean_visual_on_layers']:.2f} | "
            f"{actual['rescue_count']}/{actual['harm_count']} | {percent(oracle['oracle_accuracy'])} | "
            f"{100 * oracle['accuracy_delta']:+.2f}pp | {oracle['mean_visual_on_layers']:.2f} | "
            f"{percent(oracle['route_sensitive_layer_saving_fraction'])} |"
        )
    union = all_result["proposer_union_oracle"]
    ci = union["accuracy_delta_bootstrap_95_ci"]
    lines += [
        "",
        "## Six-proposer union upper bound",
        "",
        f"- All-on accuracy: {percent(union['baseline_accuracy'])}",
        f"- Oracle accuracy: {percent(union['oracle_accuracy'])}",
        f"- Delta: {100 * union['accuracy_delta']:+.2f}pp "
        f"[95% bootstrap CI {100 * ci['low']:+.2f}, {100 * ci['high']:+.2f}]pp",
        f"- Mean ON layers: {union['mean_visual_on_layers']:.2f}/28",
        f"- Route-sensitive saving proxy: {percent(union['route_sensitive_layer_saving_fraction'])}",
        f"- Rescues / harms: {union['rescue_count']} / {union['harm_count']}",
        "",
        "## Interpretation contract",
        "",
        "A positive point proves route-outcome headroom exists in this natural heldout",
        "population. It does not prove that a deployable gate can identify the point,",
        "or that generating multiple candidates is computationally beneficial.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {name: Path(path) for name, path in config["policies"].items()}
    policy_rows = {name: read_jsonl(path) for name, path in paths.items()}
    aligned = align_policy_rows(policy_rows)
    policies = list(paths)

    populations = {"all": aligned}
    no_seed = [row for row in aligned if row["benchmark"] != "seedbench_lite"]
    if len(no_seed) != len(aligned):
        populations["without_seedbench_lite"] = no_seed

    result: dict[str, Any] = {
        "schema_version": "baseline_relative_oracle_pareto_v1",
        "config": str(args.config.resolve()),
        "population": config["population"],
        "policy_files": {
            name: {"path": str(path), "sha256": sha256(path), "rows": len(policy_rows[name])}
            for name, path in paths.items()
        },
        "alignment_audit": {
            "shared_uids": len(aligned),
            "baseline_mismatches": 0,
            "benchmark_mismatches": 0,
        },
        "populations": {},
    }
    union_rows_by_population = {}
    for offset, (population, rows) in enumerate(populations.items()):
        payload, union_rows = population_result(
            rows,
            policies,
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed + 1000 * offset,
        )
        result["populations"][population] = payload
        union_rows_by_population[population] = union_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "proposer_union_oracle_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in union_rows_by_population["all"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_markdown(args.output_dir / "report.md", result)
    print(json.dumps(result["populations"]["all"]["proposer_union_oracle"], indent=2))


if __name__ == "__main__":
    main()
