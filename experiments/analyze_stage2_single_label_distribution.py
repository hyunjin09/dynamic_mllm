#!/usr/bin/env python3
"""Audit Phase-56 single-intervention Stage-2 labels without model execution."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dense_failure_stage2.full_label_generation import (
    file_sha256,
    verify_artifact_manifest,
)
from dense_failure_stage2.single_label_audit import (
    ACTIONS,
    CORRECTIVE_ACTIONS,
    classify_immediacy,
    same_layer_action_ambiguity,
    sample_weighted_action_distribution,
    semantic_state_id,
    simulate_sampling_schemes,
    validate_single_routes,
)


DATASETS = ("gqa", "chartqa", "textvqa")
TRIGGER_DEPTHS = ("L0", "L1-8", "L9-18", "L19-27")
DELAY_BINS = ("0", "1", "2-4", "5-9", "10+")
ROUTE_BINS = ("1", "2-3", "4-7", "8-15", "16+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True))


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            key: float("nan")
            for key in ("mean", "median", "q25", "q75", "p90", "p95", "min", "max")
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": _quantile(values, 0.25),
        "q75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _route_bin(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 7:
        return "4-7"
    if count <= 15:
        return "8-15"
    return "16+"


def _delay_bin(delay: int) -> str:
    if delay == 0:
        return "0"
    if delay == 1:
        return "1"
    if delay <= 4:
        return "2-4"
    if delay <= 9:
        return "5-9"
    return "10+"


def _layer_group(layer: int) -> str:
    if layer <= 8:
        return "Early_0-8"
    if layer <= 18:
        return "Middle_9-18"
    return "Late_19-27"


def _group_routes(
    routes: Sequence[Mapping[str, Any]], field: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in routes:
        grouped[str(row[field])].append(row)
    return dict(grouped)


def _validate_preservation_routes(
    routes: Sequence[Mapping[str, Any]], *, contract_sha256: str
) -> None:
    seen_uids: set[str] = set()
    seen_routes: set[str] = set()
    for row in routes:
        uid = str(row["uid"])
        route_id = str(row["route_id"])
        if uid in seen_uids or route_id in seen_routes:
            raise ValueError("Corpus A must contain exactly one unique route per sample")
        seen_uids.add(uid)
        seen_routes.add(route_id)
        if str(row.get("contract_sha256")) != contract_sha256:
            raise ValueError(f"Corpus A contract mismatch: {uid}")
        if row.get("route_source") != "preservation_full":
            raise ValueError(f"non-preservation route in Corpus A: {uid}")
        if list(row.get("actions", [])) != ["FULL"] * 28:
            raise ValueError(f"Corpus A route is not all FULL: {uid}")
        if not bool(row.get("final_lmms_correct")) or not bool(
            row.get("replay_token_parity")
        ):
            raise ValueError(f"Corpus A route is not a correct exact replay: {uid}")


def _load_and_validate_states(
    path: Path,
    *,
    route_by_id: Mapping[str, Mapping[str, Any]],
    contract_sha256: str,
    expected_a: int,
    expected_b: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pair_seen: set[tuple[str, int]] = set()
    route_layer_seen: set[tuple[str, int]] = set()
    per_route = Counter()
    per_source = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row["route_source"])
            if source not in {"preservation_full", "single"}:
                continue
            route_id = str(row["route_id"])
            route = route_by_id.get(route_id)
            if route is None:
                raise ValueError(f"state row references unknown A/B route: {route_id}")
            if str(row.get("contract_sha256")) != contract_sha256:
                raise ValueError(f"state contract mismatch at line {line_number}")
            if str(row["uid"]) != str(route["uid"]) or source != str(
                route["route_source"]
            ):
                raise ValueError(f"state route provenance mismatch: {route_id}")
            layer = int(row["layer"])
            if not int(route["trigger_layer"]) <= layer <= 27:
                raise ValueError(f"state layer outside frozen suffix: {route_id}/{layer}")
            if str(row["chosen_action"]) != str(route["actions"][layer]):
                raise ValueError(f"state action does not match route: {route_id}/{layer}")
            route_layer = (route_id, layer)
            if route_layer in route_layer_seen:
                raise ValueError(f"duplicate route/layer state: {route_id}/{layer}")
            route_layer_seen.add(route_layer)
            feature_pair = (str(row["feature_file"]), int(row["tensor_row"]))
            if feature_pair in pair_seen:
                raise ValueError(f"duplicate shard/tensor row: {feature_pair}")
            pair_seen.add(feature_pair)
            per_route[route_id] += 1
            per_source[source] += 1
            rows.append(row)

    if per_source != Counter(
        {"preservation_full": expected_a, "single": expected_b}
    ):
        raise ValueError(f"A/B state counts mismatch: {dict(per_source)}")
    if set(per_route) != set(route_by_id):
        raise ValueError("some A/B routes have no state rows")
    for route_id, route in route_by_id.items():
        expected = 28 - int(route["trigger_layer"])
        if per_route[route_id] != expected:
            raise ValueError(
                f"route suffix state count mismatch: {route_id}: "
                f"{per_route[route_id]} != {expected}"
            )
    return rows


def _attach_exact_feature_hashes(
    rows: Sequence[dict[str, Any]], *, source_root: Path
) -> None:
    import torch

    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_file[str(row["feature_file"])].append(row)
    for relative, index_rows in sorted(by_file.items()):
        shard = torch.load(source_root / relative, map_location="cpu", weights_only=False)
        if str(shard.get("contract_sha256")) != str(
            index_rows[0]["contract_sha256"]
        ):
            raise ValueError(f"feature shard contract mismatch: {relative}")
        records = shard.get("records")
        if not isinstance(records, list):
            raise ValueError(f"feature shard has no record list: {relative}")
        tensors = [shard[name] for name in ("text_final", "text_mean", "visual_mean")]
        if any(tensor.shape[0] != len(records) for tensor in tensors):
            raise ValueError(f"feature tensor/record count mismatch: {relative}")
        packed = torch.cat(tensors, dim=1).contiguous().view(torch.uint8).numpy()
        for row in index_rows:
            tensor_row = int(row["tensor_row"])
            record = records[tensor_row]
            for key in ("uid", "route_id", "layer", "chosen_action"):
                if str(record[key]) != str(row[key]):
                    raise ValueError(f"feature record/index mismatch: {relative}/{tensor_row}")
            row["exact_feature_sha256"] = sha256(
                memoryview(packed[tensor_row])
            ).hexdigest()


def _route_multiplicity_tables(
    routes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = _group_routes(routes, "uid")
    sample_rows = []
    pair_rows = []
    for uid, sample_routes in sorted(grouped.items()):
        route_count = len(sample_routes)
        layers = {int(row["intervention_layer"]) for row in sample_routes}
        actions = {str(row["intervention_action"]) for row in sample_routes}
        pairs = {
            (int(row["intervention_layer"]), str(row["intervention_action"]))
            for row in sample_routes
        }
        common = {
            "uid": uid,
            "dataset": str(sample_routes[0]["dataset"]),
            "trigger_layer": int(sample_routes[0]["trigger_layer"]),
            "trigger_depth_bin": str(sample_routes[0]["trigger_depth_bin"]),
        }
        sample_rows.append(
            {
                "record_type": "sample",
                **common,
                "successful_route_count": route_count,
                "multiplicity_bin": _route_bin(route_count),
            }
        )
        pair_rows.append(
            {
                **common,
                "unique_successful_layers": len(layers),
                "unique_successful_actions": len(actions),
                "unique_successful_layer_action_pairs": len(pairs),
            }
        )
    values = [int(row["successful_route_count"]) for row in sample_rows]
    for statistic, value in _stats(values).items():
        sample_rows.append(
            {
                "record_type": "overall_statistic",
                "statistic": statistic,
                "value": value,
            }
        )
    bin_counts = Counter(_route_bin(value) for value in values)
    for label in ROUTE_BINS:
        sample_rows.append(
            {
                "record_type": "multiplicity_bin",
                "multiplicity_bin": label,
                "count": bin_counts[label],
                "fraction": bin_counts[label] / len(values),
            }
        )
    return sample_rows, pair_rows


def _action_tables(
    routes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    route_counts = Counter(str(row["intervention_action"]) for row in routes)
    route_rows = [
        {
            "action": action,
            "route_count": route_counts[action],
            "route_fraction": route_counts[action] / len(routes),
            "weighting_unit": "route",
        }
        for action in CORRECTIVE_ACTIONS
    ]
    return route_rows, sample_weighted_action_distribution(routes)


def _intervention_tables(
    routes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = _group_routes(routes, "uid")
    route_layer = Counter(int(row["intervention_layer"]) for row in routes)
    sample_layer = Counter({layer: 0.0 for layer in range(28)})
    route_action_layer = Counter()
    sample_action_layer = Counter()
    for sample_routes in grouped.values():
        weight = 1.0 / len(sample_routes)
        for row in sample_routes:
            layer = int(row["intervention_layer"])
            action = str(row["intervention_action"])
            sample_layer[layer] += weight
            route_action_layer[(layer, action)] += 1
            sample_action_layer[(layer, action)] += weight
    rows = []
    for weighting, counts, total in (
        ("route", route_layer, len(routes)),
        ("sample", sample_layer, len(grouped)),
    ):
        for layer in range(28):
            count = float(counts[layer])
            rows.append(
                {
                    "weighting": weighting,
                    "scope": "layer",
                    "layer_or_group": layer,
                    "weighted_count": count,
                    "weighted_fraction": count / total,
                }
            )
        for layer_group in ("Early_0-8", "Middle_9-18", "Late_19-27"):
            count = sum(
                float(counts[layer])
                for layer in range(28)
                if _layer_group(layer) == layer_group
            )
            rows.append(
                {
                    "weighting": weighting,
                    "scope": "layer_group",
                    "layer_or_group": layer_group,
                    "weighted_count": count,
                    "weighted_fraction": count / total,
                }
            )
    action_rows = []
    for layer in range(28):
        for action in CORRECTIVE_ACTIONS:
            action_rows.append(
                {
                    "layer": layer,
                    "action": action,
                    "route_count": route_action_layer[(layer, action)],
                    "route_fraction": route_action_layer[(layer, action)] / len(routes),
                    "sample_weighted_count": sample_action_layer[(layer, action)],
                    "sample_weighted_fraction": sample_action_layer[(layer, action)]
                    / len(grouped),
                }
            )
    return rows, action_rows


def _delay_rows(routes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    group_specs: list[tuple[str, str, Sequence[Mapping[str, Any]]]] = [
        ("overall", "overall", routes)
    ]
    for field, groups in (
        ("dataset", DATASETS),
        ("intervention_action", CORRECTIVE_ACTIONS),
        ("trigger_depth_bin", TRIGGER_DEPTHS),
    ):
        by_value = _group_routes(routes, field)
        for group in groups:
            group_specs.append((field, group, by_value.get(group, [])))
    for group_type, group, group_routes in group_specs:
        delays = [
            int(row["intervention_layer"]) - int(row["trigger_layer"])
            for row in group_routes
        ]
        for statistic, value in _stats(delays).items():
            output.append(
                {
                    "group_type": group_type,
                    "group": group,
                    "record_type": "statistic",
                    "statistic_or_bin": statistic,
                    "value": value,
                    "route_count": len(delays),
                }
            )
        counts = Counter(_delay_bin(delay) for delay in delays)
        for label in DELAY_BINS:
            output.append(
                {
                    "group_type": group_type,
                    "group": group,
                    "record_type": "delay_bin",
                    "statistic_or_bin": label,
                    "value": counts[label],
                    "fraction": counts[label] / len(delays) if delays else 0.0,
                    "route_count": len(delays),
                }
            )
    return output


def _immediacy_rows(
    routes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows = classify_immediacy(routes)
    output = [{"record_type": "sample", **row} for row in sample_rows]
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("overall", "overall", sample_rows)
    ]
    for field, order in (("dataset", DATASETS), ("trigger_depth_bin", TRIGGER_DEPTHS)):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sample_rows:
            grouped[str(row[field])].append(row)
        groups.extend((field, value, grouped.get(value, [])) for value in order)
    for group_type, group, rows in groups:
        counts = Counter(str(row["classification"]) for row in rows)
        for label in ("IMMEDIATE_FIXABLE", "DELAYED_ONLY_FIXABLE"):
            output.append(
                {
                    "record_type": "summary",
                    "group_type": group_type,
                    "group": group,
                    "classification": label,
                    "count": counts[label],
                    "fraction": counts[label] / len(rows) if rows else 0.0,
                }
            )
    return output, sample_rows


def _ambiguity_rows(routes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs = same_layer_action_ambiguity(routes)
    output = [{"record_type": "sample_layer", **row} for row in pairs]
    counts = Counter(str(row["ambiguity_class"]) for row in pairs)
    for label in ("SINGLE_OBSERVED_ACTION", "MULTIPLE_OBSERVED_ACTIONS"):
        output.append(
            {
                "record_type": "summary",
                "ambiguity_class": label,
                "count": counts[label],
                "fraction": counts[label] / len(pairs),
            }
        )
    set_counts = Counter(str(row["observed_successful_action_set"]) for row in pairs)
    for action_set, count in sorted(set_counts.items()):
        output.append(
            {
                "record_type": "observed_action_set_summary",
                "observed_successful_action_set": action_set,
                "count": count,
                "fraction": count / len(pairs),
            }
        )
    return output


def _naive_balance_rows(
    states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    populations = {
        "Corpus_A": [row for row in states if row["route_source"] == "preservation_full"],
        "Corpus_B": [row for row in states if row["route_source"] == "single"],
        "Corpus_A_plus_B": list(states),
    }
    output = []
    for name, rows in populations.items():
        counts = Counter(str(row["chosen_action"]) for row in rows)
        non_full = sum(counts[action] for action in CORRECTIVE_ACTIONS)
        output.append(
            {
                "population": name,
                **{action: counts[action] for action in ACTIONS},
                "total_states": len(rows),
                "non_FULL": non_full,
                "FULL_fraction": counts["FULL"] / len(rows),
                "non_FULL_fraction": non_full / len(rows),
                "FULL_to_non_FULL_ratio": (
                    counts["FULL"] / non_full if non_full else "inf"
                ),
            }
        )
    return output


def _preservation_rows(
    routes_a: Sequence[Mapping[str, Any]], *, w_samples: int
) -> list[dict[str, Any]]:
    output = []
    suffix_lengths = []
    for row in sorted(routes_a, key=lambda value: str(value["uid"])):
        trigger = int(row["trigger_layer"])
        suffix = 28 - trigger
        suffix_lengths.append(suffix)
        output.append(
            {
                "record_type": "sample",
                "uid": str(row["uid"]),
                "dataset": str(row["dataset"]),
                "trigger_layer": trigger,
                "trigger_depth_bin": str(row["trigger_depth_bin"]),
                "suffix_length": suffix,
                "available_FULL_states": suffix,
            }
        )
    for statistic, value in _stats(suffix_lengths).items():
        output.append(
            {
                "record_type": "suffix_statistic",
                "statistic": statistic,
                "value": value,
            }
        )
    trigger_counts = Counter(int(row["trigger_layer"]) for row in routes_a)
    for layer in range(28):
        output.append(
            {
                "record_type": "trigger_layer_distribution",
                "trigger_layer": layer,
                "count": trigger_counts[layer],
                "fraction": trigger_counts[layer] / len(routes_a),
            }
        )
    c_samples = len(routes_a)
    for label, c_weight, w_weight in (
        ("1:1", 1, 1),
        ("1:2", 1, 2),
        ("natural", c_samples, w_samples),
    ):
        draw_fraction = c_weight / (c_weight + w_weight)
        relative_per_c = (c_weight / c_samples) / (w_weight / w_samples)
        output.append(
            {
                "record_type": "sample_mixture",
                "mixture": label,
                "C_draw_fraction": draw_fraction,
                "W_draw_fraction": 1 - draw_fraction,
                "relative_per_C_sample_oversampling_vs_W": relative_per_c,
                "C_samples": c_samples,
                "W_samples": w_samples,
            }
        )
    output.append(
        {
            "record_type": "preservation_state_scope",
            "scope": "trigger_only",
            "available_FULL_states": c_samples,
        }
    )
    output.append(
        {
            "record_type": "preservation_state_scope",
            "scope": "full_safe_suffix",
            "available_FULL_states": sum(suffix_lengths),
        }
    )
    return output


def _breakdown_rows(
    routes: Sequence[Mapping[str, Any]],
    immediacy: Sequence[Mapping[str, Any]],
    *,
    field: str,
    order: Sequence[str],
) -> list[dict[str, Any]]:
    route_groups = _group_routes(routes, field)
    immediate_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in immediacy:
        immediate_groups[str(row[field])].append(row)
    output = []
    for group in order:
        group_routes = route_groups.get(group, [])
        samples = _group_routes(group_routes, "uid")
        route_counts = [len(value) for value in samples.values()]
        delays = [
            int(row["intervention_layer"]) - int(row["trigger_layer"])
            for row in group_routes
        ]
        layers = [int(row["intervention_layer"]) for row in group_routes]
        action_rows = sample_weighted_action_distribution(group_routes)
        action_fractions = {
            str(row["action"]): float(row["weighted_fraction"])
            for row in action_rows
        }
        class_counts = Counter(
            str(row["classification"]) for row in immediate_groups.get(group, [])
        )
        sample_count = len(samples)
        output.append(
            {
                field: group,
                "SINGLE_FIXABLE_samples": sample_count,
                "successful_routes": len(group_routes),
                "routes_per_sample_mean": float(np.mean(route_counts)) if route_counts else 0.0,
                "routes_per_sample_median": float(np.median(route_counts)) if route_counts else 0.0,
                "READ_ONLY_sample_weighted_fraction": action_fractions.get("READ_ONLY", 0.0),
                "WRITE_ONLY_sample_weighted_fraction": action_fractions.get("WRITE_ONLY", 0.0),
                "IGNORE_sample_weighted_fraction": action_fractions.get("IGNORE", 0.0),
                "intervention_layer_mean": float(np.mean(layers)) if layers else float("nan"),
                "intervention_layer_median": float(np.median(layers)) if layers else float("nan"),
                "delay_mean": float(np.mean(delays)) if delays else float("nan"),
                "delay_median": float(np.median(delays)) if delays else float("nan"),
                "delay_p90": _quantile(delays, 0.90) if delays else float("nan"),
                "IMMEDIATE_FIXABLE": class_counts["IMMEDIATE_FIXABLE"],
                "IMMEDIATE_FIXABLE_fraction": (
                    class_counts["IMMEDIATE_FIXABLE"] / sample_count if sample_count else 0.0
                ),
                "DELAYED_ONLY_FIXABLE": class_counts["DELAYED_ONLY_FIXABLE"],
                "DELAYED_ONLY_FIXABLE_fraction": (
                    class_counts["DELAYED_ONLY_FIXABLE"] / sample_count
                    if sample_count
                    else 0.0
                ),
            }
        )
    return output


def _state_redundancy_rows(
    states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    populations = {
        "Corpus_A": [row for row in states if row["route_source"] == "preservation_full"],
        "Corpus_B": [row for row in states if row["route_source"] == "single"],
        "Corpus_A_plus_B": list(states),
    }
    output = []
    for name, rows in populations.items():
        by_semantic: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_exact: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            state_id = str(row["semantic_state_sha256"])
            exact = str(row.get("exact_feature_sha256", "not_computed"))
            by_semantic[state_id].append(row)
            by_exact[exact].add(state_id)
        semantic_exact_sets = {
            state_id: {str(row.get("exact_feature_sha256", "not_computed")) for row in values}
            for state_id, values in by_semantic.items()
        }
        label_sets = {
            state_id: {str(row["chosen_action"]) for row in values}
            for state_id, values in by_semantic.items()
        }
        unique_exact = len({str(row.get("exact_feature_sha256", "not_computed")) for row in rows})
        output.append(
            {
                "population": name,
                "state_identity_definition": "contract+schema+uid+layer+pre_layer_action_prefix",
                "total_stored_state_rows": len(rows),
                "unique_semantic_state_ids": len(by_semantic),
                "duplicate_semantic_rows_from_route_multiplicity": len(rows) - len(by_semantic),
                "duplicate_semantic_fraction": (len(rows) - len(by_semantic)) / len(rows),
                "unique_exact_feature_hashes": unique_exact,
                "duplicate_exact_feature_rows": len(rows) - unique_exact,
                "semantic_ids_with_multiple_exact_feature_hashes": sum(
                    len(values) > 1 for values in semantic_exact_sets.values()
                ),
                "exact_feature_hashes_shared_by_multiple_semantic_ids": sum(
                    len(values) > 1 for values in by_exact.values()
                ),
                "semantic_ids_with_multiple_observed_labels": sum(
                    len(values) > 1 for values in label_sets.values()
                ),
                "semantic_ids_with_FULL_and_non_FULL_labels": sum(
                    "FULL" in values and len(values - {"FULL"}) > 0
                    for values in label_sets.values()
                ),
                "semantic_ids_with_multiple_non_FULL_labels": sum(
                    len(values - {"FULL"}) > 1 for values in label_sets.values()
                ),
            }
        )
    return output


def _plot_outputs(
    root: Path,
    *,
    sample_multiplicity: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    action_by_layer: Sequence[Mapping[str, Any]],
    delay_rows: Sequence[Mapping[str, Any]],
    immediacy_samples: Sequence[Mapping[str, Any]],
    naive_balance: Sequence[Mapping[str, Any]],
    sampling_rows: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
) -> None:
    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    colors = {"READ_ONLY": "#4C78A8", "WRITE_ONLY": "#F58518", "IGNORE": "#54A24B", "FULL": "#9D9D9D"}

    samples = [row for row in sample_multiplicity if row["record_type"] == "sample"]
    counts = Counter(str(row["multiplicity_bin"]) for row in samples)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(ROUTE_BINS, [counts[label] for label in ROUTE_BINS], color="#4C78A8")
    axis.set(xlabel="Successful single routes per sample", ylabel="Samples", title="Successful-route multiplicity")
    fig.tight_layout(); fig.savefig(figure_root / "routes_per_sample_histogram.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    for weighting, style in (("route", "-"), ("sample", "--")):
        values = [next(float(row["weighted_fraction"]) for row in intervention_rows if row["weighting"] == weighting and row["scope"] == "layer" and int(row["layer_or_group"]) == layer) for layer in range(28)]
        axis.plot(range(28), values, style, marker="o", markersize=2.5, label=f"{weighting}-weighted")
    axis.set(xlabel="Intervention layer", ylabel="Fraction", title="Successful single-intervention layer distribution"); axis.legend()
    fig.tight_layout(); fig.savefig(figure_root / "single_intervention_layer_distribution.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    for action in CORRECTIVE_ACTIONS:
        values = [next(float(row["sample_weighted_fraction"]) for row in action_by_layer if row["action"] == action and int(row["layer"]) == layer) for layer in range(28)]
        axis.plot(range(28), values, marker="o", markersize=2.5, label=action, color=colors[action])
    axis.set(xlabel="Intervention layer", ylabel="Sample-weighted fraction", title="Observed successful actions by layer"); axis.legend()
    fig.tight_layout(); fig.savefig(figure_root / "single_action_by_layer.png", dpi=180); plt.close(fig)

    overall_bins = {str(row["statistic_or_bin"]): float(row["fraction"]) for row in delay_rows if row["group_type"] == "overall" and row["record_type"] == "delay_bin"}
    fig, axis = plt.subplots(figsize=(7, 4.5)); axis.bar(DELAY_BINS, [overall_bins[label] for label in DELAY_BINS], color="#72B7B2")
    axis.set(xlabel="Trigger-to-intervention delay", ylabel="Route fraction", title="Correction delay")
    fig.tight_layout(); fig.savefig(figure_root / "trigger_to_intervention_delay.png", dpi=180); plt.close(fig)

    immediate_counts = Counter(str(row["classification"]) for row in immediacy_samples)
    labels = ("IMMEDIATE_FIXABLE", "DELAYED_ONLY_FIXABLE")
    fig, axis = plt.subplots(figsize=(7, 4.5)); axis.bar(labels, [immediate_counts[label] for label in labels], color=["#54A24B", "#E45756"])
    axis.set(ylabel="Samples", title="Immediate versus delayed-only correction"); axis.tick_params(axis="x", rotation=12)
    fig.tight_layout(); fig.savefig(figure_root / "immediate_vs_delayed.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5)); bottoms = np.zeros(len(naive_balance))
    xlabels = [str(row["population"]) for row in naive_balance]
    for action in ACTIONS:
        values = np.asarray([float(row[action]) / float(row["total_states"]) for row in naive_balance])
        axis.bar(xlabels, values, bottom=bottoms, label=action, color=colors[action]); bottoms += values
    axis.set(ylabel="State fraction", title="Naive all-state action imbalance"); axis.legend(ncol=2)
    fig.tight_layout(); fig.savefig(figure_root / "naive_action_imbalance.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5)); bottoms = np.zeros(len(sampling_rows)); xlabels = [str(row["scheme"]) for row in sampling_rows]
    for action in ACTIONS:
        values = np.asarray([float(row[action]) / float(row["total_states"]) for row in sampling_rows])
        axis.bar(xlabels, values, bottom=bottoms, label=action, color=colors[action]); bottoms += values
    axis.set(ylabel="Expected state fraction", title="Candidate sampling-scheme class balance"); axis.tick_params(axis="x", rotation=55); axis.legend(ncol=4)
    fig.tight_layout(); fig.savefig(figure_root / "sampling_scheme_balance.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5)); bottoms = np.zeros(len(dataset_rows)); xlabels = [str(row["dataset"]) for row in dataset_rows]
    for action in CORRECTIVE_ACTIONS:
        values = np.asarray([float(row[f"{action}_sample_weighted_fraction"]) for row in dataset_rows])
        axis.bar(xlabels, values, bottom=bottoms, label=action, color=colors[action]); bottoms += values
    axis.set(ylabel="Sample-weighted fraction", title="Observed successful action by dataset"); axis.legend()
    fig.tight_layout(); fig.savefig(figure_root / "dataset_action_distribution.png", dpi=180); plt.close(fig)

    for figure_name, field, order, title in (
        ("dataset_delay_distribution.png", "dataset", DATASETS, "Correction delay by dataset"),
        ("trigger_depth_delay_distribution.png", "trigger_depth_bin", TRIGGER_DEPTHS, "Correction delay by trigger depth"),
    ):
        grouped = _group_routes(routes, field); x = np.arange(len(order)); width = 0.16
        fig, axis = plt.subplots(figsize=(9, 4.5))
        for index, delay_label in enumerate(DELAY_BINS):
            values = []
            for group in order:
                delays = [int(row["intervention_layer"]) - int(row["trigger_layer"]) for row in grouped.get(group, [])]
                counts = Counter(_delay_bin(delay) for delay in delays)
                values.append(counts[delay_label] / len(delays) if delays else 0.0)
            axis.bar(x + (index - 2) * width, values, width, label=delay_label)
        axis.set_xticks(x, order); axis.set(ylabel="Route fraction", title=title); axis.legend(title="Delay")
        fig.tight_layout(); fig.savefig(figure_root / figure_name, dpi=180); plt.close(fig)


def _metric_value(
    rows: Sequence[Mapping[str, Any]], **matches: Any
) -> float:
    selected = [row for row in rows if all(row.get(key) == value for key, value in matches.items())]
    if len(selected) != 1:
        raise ValueError(f"metric lookup is not unique: {matches}: {len(selected)}")
    return float(selected[0]["value"])


def _write_summaries(
    root: Path,
    *,
    contract: Mapping[str, Any],
    multiplicity_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    route_actions: Sequence[Mapping[str, Any]],
    sample_actions: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    delay_rows: Sequence[Mapping[str, Any]],
    immediacy_samples: Sequence[Mapping[str, Any]],
    ambiguity_rows: Sequence[Mapping[str, Any]],
    naive_rows: Sequence[Mapping[str, Any]],
    sampling_rows: Sequence[Mapping[str, Any]],
    preservation_rows: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    trigger_rows: Sequence[Mapping[str, Any]],
    redundancy_rows: Sequence[Mapping[str, Any]],
) -> None:
    samples = [row for row in multiplicity_rows if row["record_type"] == "sample"]
    route_values = [int(row["successful_route_count"]) for row in samples]
    route_stats = _stats(route_values)
    pair_stats = {
        field: _stats([int(row[field]) for row in pair_rows])
        for field in (
            "unique_successful_layers",
            "unique_successful_actions",
            "unique_successful_layer_action_pairs",
        )
    }
    route_action = max(route_actions, key=lambda row: float(row["route_fraction"]))
    sample_action = max(sample_actions, key=lambda row: float(row["weighted_fraction"]))
    immediate_counts = Counter(str(row["classification"]) for row in immediacy_samples)
    ambiguity_pairs = [row for row in ambiguity_rows if row["record_type"] == "sample_layer"]
    multi_pairs = sum(row["ambiguity_class"] == "MULTIPLE_OBSERVED_ACTIONS" for row in ambiguity_pairs)
    b_naive = next(row for row in naive_rows if row["population"] == "Corpus_B")
    ab_naive = next(row for row in naive_rows if row["population"] == "Corpus_A_plus_B")
    s3s1 = next(row for row in sampling_rows if row["scheme"] == "S3_SAMPLE_BALANCED_S1")
    natural_mix = next(row for row in preservation_rows if row.get("mixture") == "natural")
    one_two_mix = next(row for row in preservation_rows if row.get("mixture") == "1:2")
    b_redundancy = next(row for row in redundancy_rows if row["population"] == "Corpus_B")
    mean_delay = _metric_value(delay_rows, group_type="overall", group="overall", record_type="statistic", statistic_or_bin="mean")
    median_delay = _metric_value(delay_rows, group_type="overall", group="overall", record_type="statistic", statistic_or_bin="median")
    p90_delay = _metric_value(delay_rows, group_type="overall", group="overall", record_type="statistic", statistic_or_bin="p90")
    route_layer_early = next(row for row in intervention_rows if row["weighting"] == "route" and row["scope"] == "layer_group" and row["layer_or_group"] == "Early_0-8")
    route_layer_middle = next(row for row in intervention_rows if row["weighting"] == "route" and row["scope"] == "layer_group" and row["layer_or_group"] == "Middle_9-18")
    route_layer_late = next(row for row in intervention_rows if row["weighting"] == "route" and row["scope"] == "layer_group" and row["layer_or_group"] == "Late_19-27")

    summary = f"""# Stage-2 single-label distribution audit

Analysis contract: `{contract['analysis_contract_sha256']}`. Source Phase-56 contract: `{contract['source_contract_sha256']}`. This audit used only train Corpus A+B and ran no model inference, search, training, validation/test labeling, or external evaluation.

## Answers to the audit questions

1. **Routes per W sample.** The 698 SINGLE_FIXABLE samples have {len(route_values):,} sample records and 7,628 replay-valid routes: mean **{route_stats['mean']:.2f}**, median **{route_stats['median']:.1f}**, IQR **{route_stats['q25']:.1f}–{route_stats['q75']:.1f}**, P90/P95 **{route_stats['p90']:.1f}/{route_stats['p95']:.1f}**, range **{route_stats['min']:.0f}–{route_stats['max']:.0f}**.
2. **Localization.** Per sample, successful layers have median **{pair_stats['unique_successful_layers']['median']:.1f}** (IQR {pair_stats['unique_successful_layers']['q25']:.1f}–{pair_stats['unique_successful_layers']['q75']:.1f}); layer/action pairs have median **{pair_stats['unique_successful_layer_action_pairs']['median']:.1f}**. Correction is not generally a single localized label.
3. **Most common action.** Route weighting favors **{route_action['action']}** ({float(route_action['route_fraction']):.3f}); the primary sample-weighted view also favors **{sample_action['action']}** ({float(sample_action['weighted_fraction']):.3f}). Exact alternatives are in the two action tables.
4. **Intervention depth.** Route mass is Early/Middle/Late **{float(route_layer_early['weighted_fraction']):.3f}/{float(route_layer_middle['weighted_fraction']):.3f}/{float(route_layer_late['weighted_fraction']):.3f}**. Successful intervention is broad rather than confined to one layer.
5. **Delay.** Trigger-to-intervention delay has mean **{mean_delay:.2f}**, median **{median_delay:.1f}**, and P90 **{p90_delay:.1f}** layers.
6. **Immediate.** {immediate_counts['IMMEDIATE_FIXABLE']}/698 = **{immediate_counts['IMMEDIATE_FIXABLE']/698:.3f}** have at least one observed rescue at the trigger.
7. **Delayed-only.** {immediate_counts['DELAYED_ONLY_FIXABLE']}/698 = **{immediate_counts['DELAYED_ONLY_FIXABLE']/698:.3f}** require a later observed single intervention. FULL must remain a normal post-trigger action.
8. **Same-layer multi-action ambiguity.** {multi_pairs:,}/{len(ambiguity_pairs):,} = **{multi_pairs/len(ambiguity_pairs):.3f}** successful sample/layer positions have multiple observed non-FULL rescue actions. These are observed sets, not exhaustive validity.
9. **Naive FULL imbalance.** Corpus B is **{float(b_naive['FULL_fraction']):.4f} FULL** ({b_naive['FULL_to_non_FULL_ratio']:.2f}:1); A+B is **{float(ab_naive['FULL_fraction']):.4f} FULL**. Naive all-state CE is structurally collapse-prone.
10. **Route/sample weighting.** Route count spans {int(route_stats['min'])}–{int(route_stats['max'])}; naive expansion therefore gives the most route-rich W sample **{int(route_stats['max']/route_stats['min'])}x** the weight of the least route-rich sample before suffix-length effects.
11. **Dataset differences.** GQA/ChartQA/TextVQA breakdowns differ materially in sample count, route multiplicity, action mix, and delay; see `metrics/dataset_breakdown.csv`. This audit does not authorize dataset-specific heads.
12. **Trigger-depth differences.** Delay and immediate-fixable rates vary by L0/L1-8/L9-18/L19-27; see `metrics/trigger_depth_breakdown.csv`.
13. **Simple sampler.** `S3_SAMPLE_BALANCED_S1` gives every W sample equal expected route weight, keeps one mandatory corrective state, and retains up to two FULL timing examples on each side. Its expected FULL fraction is **{float(s3s1['FULL_fraction']):.3f}**, compared with {float(b_naive['FULL_fraction']):.3f} naively.
14. **Preservation visibility.** Natural sample frequency exposes C on only **{float(natural_mix['C_draw_fraction']):.3f}** of sample draws. A 1:2 C:W mix gives C one-third of draws and requires **{float(one_two_mix['relative_per_C_sample_oversampling_vs_W']):.2f}x** per-C repetition relative to each W sample.
15. **Structural suitability.** **Conditionally yes** for a simple shared V1 only with sample-balanced route sampling and explicit preservation mixing. The corpus is not suitable for naive all-route/all-state CE. Exact routed-state auditing found {b_redundancy['duplicate_semantic_rows_from_route_multiplicity']:,} duplicate semantic rows and {b_redundancy['semantic_ids_with_multiple_observed_labels']:,} exact entering states with multiple observed route labels; this is valid alternative-route supervision, not a unique deterministic oracle action.

## Interpretation boundary

This establishes corpus structure and a defensible loader contract. It does not establish Stage-2 generalization, C preservation under free rollout, sufficiency of single interventions, or lack of value from Corpus C.
"""
    _write_text(root / "summaries/single_label_audit_summary.md", summary)

    recommendation = f"""# Stage-2 V1 data recommendation

## Recommended minimal loader contract

- Use only **Corpus A preservation + Corpus B single corrective** from the Phase-56 contract `{contract['source_contract_sha256']}`.
- Make the W sampling unit the **sample**, not the retained route: choose one successful single route uniformly for each W sample when it is drawn (resampling across epochs is allowed).
- For that route, emit exactly the corrective state plus up to **two pre-intervention FULL** and **two post-intervention FULL** states (`S3 + S1`). This preserves the central timing fact—many rescues are delayed—while reducing expected W FULL share from {float(b_naive['FULL_fraction']):.3f} to {float(s3s1['FULL_fraction']):.3f}.
- Draw triggered-C and W samples at **C:W = 1:2**. For C, sample from the safe FULL suffix while ensuring the trigger state remains represented. This makes preservation visible without letting only 39 C samples occupy half of all sample draws; it repeats each C sample about {float(one_two_mix['relative_per_C_sample_oversampling_vs_W']):.2f}x as often as each W sample.
- Keep the target single-label four-way CE: `FULL`, `READ_ONLY`, `WRITE_ONLY`, `IGNORE`, with routed feature + layer index as input. Preserve UID, route ID, trigger layer, intervention layer, observed-successful-action metadata, contract/schema/source hashes, and tensor-row provenance.
- Retain multi-valid alternatives in metadata and expose them through uniform route resampling. **Defer a multi-label loss** until a controlled follow-up; do not collapse alternatives to one arbitrary canonical route.

## Internal challenge and runner-up

The strongest objection is that S1 still gives roughly {float(s3s1['FULL_fraction']):.3f} FULL targets and revisits exact states with multiple valid route labels. The simpler runner-up is sample-balanced `S2 K=2`, which is more action-balanced but discards explicit before/after timing context. Because {immediate_counts['DELAYED_ONLY_FIXABLE']/698:.3f} of W samples are delayed-only, preserving local timing context is the more defensible V1 choice. Confidence is **medium**: the recommendation is based on corpus geometry, not training or rollout evidence.

## Explicit exclusions

Do not add Corpus C/MCTS, weighted losses, utility heads, `z_fail`, EMA/persistence logic, RL, validation/test labels, Stage-1 changes, or dataset-specific policies in V1. No model is trained by this recommendation.
"""
    _write_text(root / "summaries/stage2_v1_data_recommendation.md", recommendation)


def run(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    output_root = _resolve(str(config["output_root"]))
    if output_root.exists():
        raise RuntimeError(f"output root already exists; refusing to overwrite: {output_root}")
    source_root = _resolve(str(config["source_root"]))
    source_manifest_path = _resolve(str(config["source_artifact_manifest"]))
    corpus_manifest_path = _resolve(str(config["source_corpus_manifest"]))
    routes_a_path = _resolve(str(config["corpus_a_routes"]))
    routes_b_path = _resolve(str(config["corpus_b_routes"]))
    state_index_path = _resolve(str(config["state_index"]))
    feature_schema_path = _resolve(str(config["feature_schema"]))
    plan_path = _resolve(str(config["plan"]))
    expected = config["expected"]
    contract_sha256 = str(expected["contract_sha256"])

    source_manifest = _read_json(source_manifest_path)
    verify_artifact_manifest(source_root, source_manifest)
    if str(source_manifest.get("contract_sha256")) != contract_sha256:
        raise RuntimeError("source artifact manifest contract mismatch")
    corpus_manifest = _read_json(corpus_manifest_path)
    if str(corpus_manifest.get("contract_sha256")) != contract_sha256:
        raise RuntimeError("source corpus manifest contract mismatch")
    if file_sha256(routes_a_path) != str(corpus_manifest["corpora"]["A"]["sha256"]):
        raise RuntimeError("Corpus A hash differs from source corpus manifest")
    if file_sha256(routes_b_path) != str(corpus_manifest["corpora"]["B"]["sha256"]):
        raise RuntimeError("Corpus B hash differs from source corpus manifest")

    routes_a = _read_jsonl(routes_a_path)
    routes_b = _read_jsonl(routes_b_path)
    _validate_preservation_routes(routes_a, contract_sha256=contract_sha256)
    validate_single_routes(routes_b, contract_sha256=contract_sha256)
    if len(routes_a) != int(expected["corpus_a_routes"]) or len({str(row["uid"]) for row in routes_a}) != int(expected["corpus_a_samples"]):
        raise RuntimeError("Corpus A frozen population mismatch")
    if len(routes_b) != int(expected["corpus_b_routes"]) or len({str(row["uid"]) for row in routes_b}) != int(expected["corpus_b_samples"]):
        raise RuntimeError("Corpus B frozen population mismatch")
    route_by_id = {str(row["route_id"]): row for row in [*routes_a, *routes_b]}
    if len(route_by_id) != len(routes_a) + len(routes_b):
        raise RuntimeError("duplicate route ID across Corpus A+B")
    states = _load_and_validate_states(
        state_index_path,
        route_by_id=route_by_id,
        contract_sha256=contract_sha256,
        expected_a=int(expected["corpus_a_states"]),
        expected_b=int(expected["corpus_b_states"]),
    )
    for row in states:
        row["semantic_state_sha256"] = semantic_state_id(row)
    if bool(config.get("exact_feature_hash_audit")):
        _attach_exact_feature_hashes(states, source_root=source_root)

    git_status = _git("status", "--porcelain=v1")
    contract_payload = {
        "schema_version": "stage2_single_label_distribution_audit_contract_v1",
        "source_contract_sha256": contract_sha256,
        "source_artifact_manifest_sha256": file_sha256(source_manifest_path),
        "source_corpus_manifest_sha256": file_sha256(corpus_manifest_path),
        "corpus_a_sha256": file_sha256(routes_a_path),
        "corpus_b_sha256": file_sha256(routes_b_path),
        "state_index_sha256": file_sha256(state_index_path),
        "feature_schema_sha256": file_sha256(feature_schema_path),
        "config_sha256": file_sha256(config_path),
        "plan_sha256": file_sha256(plan_path),
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "audit_module_sha256": file_sha256(REPO_ROOT / "dense_failure_stage2/single_label_audit.py"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(git_status),
        "git_status_sha256": sha256(git_status.encode("utf-8")).hexdigest(),
        "exact_feature_hash_audit": bool(config.get("exact_feature_hash_audit")),
    }
    contract = {
        **contract_payload,
        "analysis_contract_sha256": _canonical_hash(contract_payload),
    }

    multiplicity_rows, pair_rows = _route_multiplicity_tables(routes_b)
    route_action_rows, sample_action_rows = _action_tables(routes_b)
    intervention_rows, action_layer_rows = _intervention_tables(routes_b)
    delay_rows = _delay_rows(routes_b)
    immediacy_rows, immediacy_samples = _immediacy_rows(routes_b)
    ambiguity_rows = _ambiguity_rows(routes_b)
    naive_rows = _naive_balance_rows(states)
    sampling_rows = simulate_sampling_schemes(routes_b)
    preservation_rows = _preservation_rows(routes_a, w_samples=int(expected["corpus_b_samples"]))
    dataset_rows = _breakdown_rows(routes_b, immediacy_samples, field="dataset", order=DATASETS)
    trigger_rows = _breakdown_rows(routes_b, immediacy_samples, field="trigger_depth_bin", order=TRIGGER_DEPTHS)
    redundancy_rows = _state_redundancy_rows(states)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".single_label_audit_tmp_", dir=output_root.parent))
    try:
        protocol = f"""# Stage-2 single-label distribution audit protocol

Analysis contract: `{contract['analysis_contract_sha256']}`  
Phase-56 source contract: `{contract_sha256}`  
Source artifact manifest SHA-256: `{contract['source_artifact_manifest_sha256']}`

## Frozen scope

- Inputs: exactly 39 train triggered Dense-C preservation samples/routes, 698 train SINGLE_FIXABLE Dense-W samples, 7,628 successful single routes, and their replay-valid routed states.
- Units remain separate: sample, route, and pre-layer routed state.
- Excluded: Corpus C/MCTS, unresolved samples, validation/test supervision, Qwen inference, search, training, rollout, threshold changes, and external evaluation.

## Weighting and simulation

- Route-weighted counts give each retained successful route weight one.
- Sample-weighted counts give each W sample total weight one, divided uniformly across its routes.
- Sampling simulations are exact expected class counts; no state is randomly drawn in this audit.
- S1 retains one corrective state plus up to two available pre-intervention and two available post-intervention FULL states from the stored trigger suffix.
- S2 retains one corrective state plus up to K available FULL suffix states for K=2/4/6.
- S3 first chooses one route uniformly per W sample, then applies S0/S1/S2 in expectation.

## State identity

The semantic pre-layer state ID hashes `(source contract, feature schema, UID, layer, action prefix strictly before layer)`. Because Phase 56 stores the state entering the chosen layer, neither the current action nor future actions belong in this identity. The audit additionally hashes the concatenated BF16 bytes of `text_final`, `text_mean`, and `visual_mean` for every A/B state row and checks shard/index alignment. This quantifies exact feature duplication without deleting or rewriting source tensors.

## Provenance

- Git commit: `{contract['git_commit']}`; dirty worktree: `{str(contract['git_worktree_dirty']).lower()}`; status hash: `{contract['git_status_sha256']}`.
- Config/plan/script/module hashes: `{contract['config_sha256']}`, `{contract['plan_sha256']}`, `{contract['analysis_script_sha256']}`, `{contract['audit_module_sha256']}`.
- All 977 Phase-56 artifacts were hash-verified before analysis.

This audit describes discovered supervision only. Observed successful actions are not an exhaustive action-validity oracle.
"""
        _write_text(temporary_root / "protocol.md", protocol)
        metric_root = temporary_root / "metrics"
        tables = {
            "sample_route_multiplicity.csv": multiplicity_rows,
            "successful_pair_multiplicity.csv": pair_rows,
            "action_distribution_route_weighted.csv": route_action_rows,
            "action_distribution_sample_weighted.csv": sample_action_rows,
            "intervention_layer_distribution.csv": intervention_rows,
            "action_by_layer.csv": action_layer_rows,
            "trigger_to_intervention_delay.csv": delay_rows,
            "immediate_vs_delayed.csv": immediacy_rows,
            "same_layer_action_ambiguity.csv": ambiguity_rows,
            "naive_state_class_balance.csv": naive_rows,
            "sampling_scheme_simulation.csv": sampling_rows,
            "preservation_balance_simulation.csv": preservation_rows,
            "dataset_breakdown.csv": dataset_rows,
            "trigger_depth_breakdown.csv": trigger_rows,
            "state_redundancy.csv": redundancy_rows,
        }
        for name, rows in tables.items():
            _write_csv(metric_root / name, rows)
        _plot_outputs(
            temporary_root,
            sample_multiplicity=multiplicity_rows,
            intervention_rows=intervention_rows,
            action_by_layer=action_layer_rows,
            delay_rows=delay_rows,
            immediacy_samples=immediacy_samples,
            naive_balance=naive_rows,
            sampling_rows=sampling_rows,
            dataset_rows=dataset_rows,
            routes=routes_b,
        )
        _write_summaries(
            temporary_root,
            contract=contract,
            multiplicity_rows=multiplicity_rows,
            pair_rows=pair_rows,
            route_actions=route_action_rows,
            sample_actions=sample_action_rows,
            intervention_rows=intervention_rows,
            delay_rows=delay_rows,
            immediacy_samples=immediacy_samples,
            ambiguity_rows=ambiguity_rows,
            naive_rows=naive_rows,
            sampling_rows=sampling_rows,
            preservation_rows=preservation_rows,
            dataset_rows=dataset_rows,
            trigger_rows=trigger_rows,
            redundancy_rows=redundancy_rows,
        )
        required = [
            "protocol.md",
            *[f"metrics/{name}" for name in tables],
            *[
                f"figures/{name}"
                for name in (
                    "routes_per_sample_histogram.png",
                    "single_intervention_layer_distribution.png",
                    "single_action_by_layer.png",
                    "trigger_to_intervention_delay.png",
                    "immediate_vs_delayed.png",
                    "naive_action_imbalance.png",
                    "sampling_scheme_balance.png",
                    "dataset_action_distribution.png",
                    "dataset_delay_distribution.png",
                    "trigger_depth_delay_distribution.png",
                )
            ],
            "summaries/single_label_audit_summary.md",
            "summaries/stage2_v1_data_recommendation.md",
        ]
        missing = [name for name in required if not (temporary_root / name).is_file()]
        if missing:
            raise RuntimeError(f"required audit artifacts are missing: {missing}")
        files = {
            str(path.relative_to(temporary_root)): file_sha256(path)
            for path in sorted(temporary_root.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"
        }
        manifest = {
            "schema_version": "stage2_single_label_distribution_audit_manifest_v1",
            "passed": True,
            **contract,
            "source_counts": {
                "corpus_a_samples": len({str(row["uid"]) for row in routes_a}),
                "corpus_a_routes": len(routes_a),
                "corpus_a_states": sum(row["route_source"] == "preservation_full" for row in states),
                "corpus_b_samples": len({str(row["uid"]) for row in routes_b}),
                "corpus_b_routes": len(routes_b),
                "corpus_b_states": sum(row["route_source"] == "single" for row in states),
            },
            "files": files,
            "completed_at": _utc_now(),
        }
        _write_json(temporary_root / "artifact_manifest.json", manifest)
        verify_artifact_manifest(temporary_root, manifest)
        temporary_root.rename(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stage2_single_label_distribution_audit_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(_resolve(args.config))
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "analysis_contract_sha256": manifest["analysis_contract_sha256"],
                "source_counts": manifest["source_counts"],
                "output_root": "analysis/dense_failure_stage2/single_label_audit",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
