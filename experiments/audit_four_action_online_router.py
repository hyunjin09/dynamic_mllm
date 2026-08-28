#!/usr/bin/env python3
"""Freeze the online-router architecture, label, and prefix-trie contract."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.train_binary_polar import file_sha256
from four_action_online_router.model import OnlineFourActionRouter
from four_action_online_router.supervision import PrefixTrie
from four_action_policy.actions import FOUR_ACTIONS, encode_action_route


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("distribution values cannot be empty")

    def quantile(fraction: float) -> int:
        return ordered[round(fraction * (len(ordered) - 1))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "max": ordered[-1],
    }


def build_label_trie_audit(
    rows: Sequence[dict[str, Any]], *, expected_contract: str, num_layers: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError("online-router label population is empty")
    uids = [str(row.get("uid") or "") for row in rows]
    if any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError("label population contains an empty or duplicate UID")
    sample_counts: Counter[tuple[str, str, str]] = Counter()
    action_counts: Counter[str] = Counter()
    per_layer = [Counter() for _ in range(num_layers)]
    route_counts = []
    node_counts = []
    multiplicity: Counter[int] = Counter()
    total_routes = 0
    for row in rows:
        if str(row.get("executor_contract_sha256")) != expected_contract:
            raise ValueError(f"executor contract mismatch for {row['uid']}")
        dataset = str(row.get("dataset"))
        split = str(row.get("split"))
        route_type = str(row.get("route_type"))
        if dataset not in {"gqa", "chartqa", "textvqa"}:
            raise ValueError(f"unexpected training dataset: {dataset}")
        if split not in {"train", "validation"} or route_type not in {"W2C", "C2C"}:
            raise ValueError(f"invalid split/type for {row['uid']}")
        routes = [
            tuple(
                encode_action_route(route["actions"], expected_layers=num_layers).tolist()
            )
            for route in row.get("valid_routes", [])
        ]
        trie = PrefixTrie(routes)
        sample_counts[(split, dataset, route_type)] += 1
        route_counts.append(len(routes))
        node_counts.append(trie.node_count)
        total_routes += len(routes)
        for actions in routes:
            for layer, action_index in enumerate(actions):
                action = FOUR_ACTIONS[action_index]
                action_counts[action] += 1
                per_layer[layer][action] += 1
        for actions in trie.action_sets:
            multiplicity[len(actions)] += 1
    return {
        "samples": len(rows),
        "routes": total_routes,
        "sample_counts": {
            f"{split}/{dataset}/{route_type}": count
            for (split, dataset, route_type), count in sorted(sample_counts.items())
        },
        "action_counts": {action: action_counts[action] for action in FOUR_ACTIONS},
        "action_counts_by_layer": [
            {action: counts[action] for action in FOUR_ACTIONS} for counts in per_layer
        ],
        "routes_per_sample": _distribution(route_counts),
        "trie_nodes": {"total": sum(node_counts), **_distribution(node_counts)},
        "valid_action_multiplicity": {
            {1: "one", 2: "two", 3: "three", 4: "four"}[size]: multiplicity[size]
            for size in range(1, 5)
        },
        "executor_contract_sha256": expected_contract,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("analysis/4action_router")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = Path(config["data"]["manifest"])
    if file_sha256(manifest) != config["data"]["manifest_sha256"]:
        raise RuntimeError("online-router manifest checksum mismatch")
    rows = read_jsonl(manifest)
    audit = build_label_trie_audit(
        rows,
        expected_contract=str(config["executor"]["contract_sha256"]),
        num_layers=int(config["router"]["num_layers"]),
    )
    expected = config["data"]
    if (
        audit["samples"] != int(expected["records"])
        or audit["routes"] != int(expected["valid_routes"])
    ):
        raise RuntimeError("online-router label counts differ from the frozen config")
    architecture = config["router"]
    router = OnlineFourActionRouter(
        hidden_size=int(architecture["hidden_size"]),
        num_layers=int(architecture["num_layers"]),
        d_router=int(architecture["d_router"]),
        num_heads=int(architecture["num_heads"]),
        mlp_hidden_size=int(architecture["mlp_hidden_size"]),
        dropout=float(architecture["dropout"]),
        interaction_scale=float(architecture["interaction_scale"]),
    )
    audit.update(
        {
            "schema_version": "four_action_online_router_label_trie_audit_v1",
            "passed": True,
            "config": str(args.config),
            "config_sha256": file_sha256(args.config),
            "manifest": str(manifest),
            "manifest_sha256": file_sha256(manifest),
            "router_parameters": sum(parameter.numel() for parameter in router.parameters()),
            "training_datasets": ["gqa", "chartqa", "textvqa"],
            "excluded_training_datasets": ["wemath_standard", "wemath_pro"],
        }
    )
    output = args.output_root / "label_and_trie_audit.json"
    write_json(output, audit)
    output.with_suffix(".json.sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    lines = [
        "# Four-Action Online Router Label and Trie Audit",
        "",
        "- Integrity: PASS",
        f"- Samples: {audit['samples']:,}",
        f"- Complete valid routes: {audit['routes']:,}",
        f"- Prefix-trie nodes: {audit['trie_nodes']['total']:,}",
        f"- Router parameters: {audit['router_parameters']:,}",
        "- Training datasets: GQA, ChartQA, TextVQA",
        "- WeMath Standard/Pro: explicitly excluded from this run",
        f"- Executor contract: `{audit['executor_contract_sha256']}`",
        "",
        "Multi-valid outgoing actions are retained at every exact prefix; no sample is expanded in proportion to its route count.",
        "",
    ]
    report = args.output_root / "label_and_trie_audit.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    report.with_suffix(".md.sha256").write_text(
        f"{file_sha256(report)}  {report.name}\n", encoding="utf-8"
    )
    print(json.dumps({"passed": True, "output": str(output), "samples": audit["samples"]}))


if __name__ == "__main__":
    main()
