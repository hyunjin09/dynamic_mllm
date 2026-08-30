"""Pure helpers for the frozen four-action generalization diagnostic."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from four_action_online_router.boundary_probe import binary_auroc
from four_action_policy.actions import FOUR_ACTIONS


def _stable(seed: int, purpose: str, *values: object) -> str:
    payload = ":".join((str(seed), purpose, *(str(value) for value in values)))
    return sha256(payload.encode()).hexdigest()


def bit_requirement_labels(valid_actions: Sequence[str]) -> dict[str, int | None]:
    """Return a bit target only when every valid action agrees on that bit."""

    actions = [str(action) for action in valid_actions]
    if not actions or any(action not in FOUR_ACTIONS for action in actions):
        raise ValueError("valid actions must be a nonempty four-action subset")
    read_off_values = {int(action in {"WRITE_ONLY", "IGNORE"}) for action in actions}
    write_off_values = {int(action in {"READ_ONLY", "IGNORE"}) for action in actions}
    return {
        "read_off": next(iter(read_off_values)) if len(read_off_values) == 1 else None,
        "write_off": (
            next(iter(write_off_values)) if len(write_off_values) == 1 else None
        ),
    }


def _state_record(
    *,
    row: Mapping[str, Any],
    state_kind: str,
    target_layer: int,
    prefix_actions: Sequence[str],
    valid_actions: Sequence[str],
    route_indices: Sequence[int],
    teacher_route_index: int,
) -> dict[str, Any]:
    bits = bit_requirement_labels(valid_actions)
    prefix_key = "|".join(prefix_actions)
    state_id = sha256(
        f"{state_kind}:{row['uid']}:{target_layer}:{prefix_key}".encode()
    ).hexdigest()[:24]
    valid = sorted({str(action) for action in valid_actions}, key=FOUR_ACTIONS.index)
    return {
        "state_id": state_id,
        "uid": str(row["uid"]),
        "split": str(row["split"]),
        "dataset": str(row["dataset"]),
        "state_kind": state_kind,
        "target_layer": int(target_layer),
        "depth_bin": min(2, (3 * int(target_layer)) // 28),
        "prefix_actions": list(prefix_actions),
        "valid_actions": valid,
        "valid_action_mask": [action in valid for action in FOUR_ACTIONS],
        "when_label": int(state_kind == "mandatory_deviation"),
        "singleton": len(valid) == 1,
        "mechanism_class": valid[0] if len(valid) == 1 else "MULTI",
        "read_off_label": bits["read_off"],
        "write_off_label": bits["write_off"],
        "boundary_route_indices": sorted({int(value) for value in route_indices}),
        "teacher_route_index": int(teacher_route_index),
    }


def _full_unique_candidates(
    rows: Sequence[dict[str, Any]], *, seed: int
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    candidates: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("route_type") != "W2C":
            continue
        routes = [tuple(str(action) for action in route["actions"]) for route in row["valid_routes"]]
        outgoing: dict[tuple[str, ...], set[str]] = defaultdict(set)
        owners: dict[tuple[str, ...], set[int]] = defaultdict(set)
        for route_index, route in enumerate(routes):
            for layer, action in enumerate(route):
                prefix = route[:layer]
                outgoing[prefix].add(action)
                owners[prefix].add(route_index)
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for prefix, actions in outgoing.items():
            if actions != {"FULL"}:
                continue
            layer = len(prefix)
            route_indices = sorted(owners[prefix])
            by_layer[layer].append(
                _state_record(
                    row=row,
                    state_kind="full_unique",
                    target_layer=layer,
                    prefix_actions=prefix,
                    valid_actions=["FULL"],
                    route_indices=route_indices,
                    teacher_route_index=min(route_indices),
                )
            )
        for layer, values in by_layer.items():
            # Retain at most one deterministic prefix per UID/layer. This makes
            # exact cell matching sample-balanced instead of route-count weighted.
            chosen = min(
                values,
                key=lambda value: _stable(
                    seed,
                    "full-unique-prefix",
                    value["uid"],
                    layer,
                    "|".join(value["prefix_actions"]),
                ),
            )
            candidates[(str(row["split"]), str(row["dataset"]), layer)].append(chosen)
    return candidates


def build_matched_state_manifest(
    rows: Sequence[dict[str, Any]],
    boundaries: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match each mandatory boundary to one exact-cell FULL-unique W2C node."""

    row_by_uid = {str(row["uid"]): row for row in rows}
    if len(row_by_uid) != len(rows):
        raise ValueError("diagnostic source manifest contains duplicate UIDs")
    boundary_by_uid = {str(row["uid"]): row for row in boundaries}
    expected = {
        uid for uid, row in row_by_uid.items() if row.get("route_type") == "W2C"
    }
    if set(boundary_by_uid) != expected:
        raise ValueError("diagnostic boundaries do not exactly cover W2C records")

    positives: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for uid in sorted(expected):
        row = row_by_uid[uid]
        boundary = boundary_by_uid[uid]
        layer = int(boundary["boundary_layer"])
        if int(boundary["all_full_prefix_length"]) != layer:
            raise ValueError("mandatory boundary prefix length mismatch")
        valid_actions = [str(value) for value in boundary["valid_nonfull_actions"]]
        if not valid_actions or "FULL" in valid_actions:
            raise ValueError("mandatory boundary must exclude FULL")
        route_indices = [int(value) for value in boundary["boundary_route_indices"]]
        teacher_route_index = int(
            boundary.get("teacher_route_index", min(route_indices))
        )
        if teacher_route_index not in route_indices:
            raise ValueError("mandatory boundary teacher route is not eligible")
        positive = _state_record(
            row=row,
            state_kind="mandatory_deviation",
            target_layer=layer,
            prefix_actions=["FULL"] * layer,
            valid_actions=valid_actions,
            route_indices=route_indices,
            teacher_route_index=teacher_route_index,
        )
        positive["source_boundary_layer"] = layer
        positives[(str(row["split"]), str(row["dataset"]), layer)].append(positive)

    negative_pools = _full_unique_candidates(list(row_by_uid.values()), seed=seed)
    records: list[dict[str, Any]] = []
    cell_audit = []
    for cell, cell_positives in sorted(positives.items()):
        ordered_positives = sorted(
            cell_positives,
            key=lambda row: _stable(seed, "positive", *cell, row["uid"]),
        )
        ordered_negatives = sorted(
            negative_pools.get(cell, []),
            key=lambda row: _stable(
                seed,
                "negative",
                *cell,
                row["uid"],
                "|".join(row["prefix_actions"]),
            ),
        )
        used: set[str] = set()
        for positive in ordered_positives:
            available = [
                row
                for row in ordered_negatives
                if row["state_id"] not in used and row["uid"] != positive["uid"]
            ]
            if not available:
                raise RuntimeError(f"no different-UID FULL-unique match for cell {cell}")
            negative = available[0]
            used.add(negative["state_id"])
            source_boundary = boundary_by_uid[negative["uid"]]
            negative = {
                **negative,
                "source_boundary_layer": int(source_boundary["boundary_layer"]),
            }
            pair_id = sha256(
                f"{seed}:{positive['state_id']}:{negative['state_id']}".encode()
            ).hexdigest()[:20]
            records.extend(
                [
                    {**positive, "pair_id": pair_id},
                    {**negative, "pair_id": pair_id},
                ]
            )
        cell_audit.append(
            {
                "split": cell[0],
                "dataset": cell[1],
                "layer": cell[2],
                "positives": len(ordered_positives),
                "negative_candidates": len(ordered_negatives),
                "selected_negatives": len(used),
            }
        )

    # Freeze one joint state shuffle within each split/dataset/layer cell.
    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_cell[(row["split"], row["dataset"], int(row["target_layer"]))].append(row)
    for cell, values in by_cell.items():
        if len(values) < 2:
            raise RuntimeError(f"state-shuffle cell is too small: {cell}")
        ordered = sorted(
            values,
            key=lambda row: _stable(seed, "state-shuffle", *cell, row["state_id"]),
        )
        partners = ordered[1:] + ordered[:1]
        for row, partner in zip(ordered, partners):
            row["shuffle_partner_state_id"] = partner["state_id"]

    records.sort(
        key=lambda row: (
            row["split"],
            row["dataset"],
            int(row["target_layer"]),
            row["pair_id"],
            -int(row["when_label"]),
        )
    )
    identities = [row["state_id"] for row in records]
    pair_counts = Counter(row["pair_id"] for row in records)
    positive_count = sum(int(row["when_label"]) for row in records)
    negative_count = len(records) - positive_count
    audit = {
        "schema_version": "four_action_generalization_state_manifest_audit_v1",
        "passed": (
            bool(records)
            and len(identities) == len(set(identities))
            and set(pair_counts.values()) == {2}
            and positive_count == negative_count == len(expected)
        ),
        "seed": int(seed),
        "records": len(records),
        "pairs": len(pair_counts),
        "positive_states": positive_count,
        "negative_states": negative_count,
        "w2c_uids": len(expected),
        "split_counts": dict(sorted(Counter(row["split"] for row in records).items())),
        "dataset_counts": dict(
            sorted(Counter(row["dataset"] for row in records).items())
        ),
        "singleton_class_counts": dict(
            sorted(
                Counter(
                    row["mechanism_class"]
                    for row in records
                    if row["singleton"]
                ).items()
            )
        ),
        "cells": cell_audit,
    }
    return records, audit


def _binary_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    average_precision = 0.0
    previous_recall = 0.0
    for threshold in sorted(set(scores.tolist()), reverse=True):
        predicted = scores >= threshold
        true_positive = int(np.logical_and(predicted, labels == 1).sum())
        false_positive = int(np.logical_and(predicted, labels == 0).sum())
        recall = true_positive / positives
        precision = true_positive / max(1, true_positive + false_positive)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return float(average_precision)


def binary_metrics(
    labels: Sequence[int], probabilities: Sequence[float]
) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or y.size == 0:
        raise ValueError("binary metrics require equal nonempty vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("binary metrics require finite scores and binary labels")
    predicted = scores >= 0.5
    truth = y == 1
    tp = int(np.logical_and(predicted, truth).sum())
    tn = int(np.logical_and(~predicted, ~truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "records": int(y.size),
        "positives": int(truth.sum()),
        "negatives": int((~truth).sum()),
        "auroc": float(binary_auroc(y, scores)),
        "auprc": _binary_auprc(y, scores),
        "accuracy": float((predicted == truth).mean()),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn) if fp + tn else 0.0),
        "false_negative_rate": float(fn / (fn + tp) if fn + tp else 0.0),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def multiclass_metrics(
    labels: Sequence[Any], predictions: Sequence[Any], *, classes: Sequence[Any]
) -> dict[str, Any]:
    if not labels or len(labels) != len(predictions) or not classes:
        raise ValueError("multiclass metrics require equal nonempty inputs and classes")
    class_values = [str(value) for value in classes]
    truth = [str(value) for value in labels]
    predicted = [str(value) for value in predictions]
    if any(value not in class_values for value in truth + predicted):
        raise ValueError("multiclass values lie outside the declared classes")
    confusion = {
        target: {output: 0 for output in class_values} for target in class_values
    }
    for target, output in zip(truth, predicted):
        confusion[target][output] += 1
    by_class = {}
    for value in class_values:
        tp = confusion[value][value]
        support = sum(confusion[value].values())
        predicted_support = sum(confusion[target][value] for target in class_values)
        precision = tp / predicted_support if predicted_support else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        by_class[value] = {
            "support": support,
            "predicted_support": predicted_support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return {
        "records": len(truth),
        "accuracy": sum(a == b for a, b in zip(truth, predicted)) / len(truth),
        "macro_f1": sum(value["f1"] for value in by_class.values())
        / len(by_class),
        "confusion": confusion,
        "by_class": by_class,
    }


def layer_only_binary_scores(
    train_rows: Sequence[dict[str, Any]],
    target_rows: Sequence[dict[str, Any]],
    *,
    label_key: str,
    alpha: float,
    num_layers: int = 28,
) -> list[float]:
    if alpha <= 0 or num_layers < 1:
        raise ValueError("layer baseline requires positive smoothing and layer count")
    positives = [0] * num_layers
    totals = [0] * num_layers
    for row in train_rows:
        layer = int(row["target_layer"])
        label = row.get(label_key)
        if not 0 <= layer < num_layers or label not in {0, 1}:
            raise ValueError("layer baseline rows require valid layers and binary labels")
        positives[layer] += int(label)
        totals[layer] += 1
    scores = []
    for row in target_rows:
        layer = int(row["target_layer"])
        if not 0 <= layer < num_layers:
            raise ValueError("layer baseline target layer is invalid")
        scores.append((positives[layer] + alpha) / (totals[layer] + 2 * alpha))
    return scores


def _cosine_distances(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    candidate_norms = np.linalg.norm(candidates, axis=1)
    denominator = np.maximum(query_norm * candidate_norms, 1e-12)
    return 1.0 - candidates.dot(query) / denominator


def knn_label_consistency(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence[Any],
    train_metadata: Sequence[dict[str, Any]],
    validation_features: Sequence[Sequence[float]] | np.ndarray,
    validation_labels: Sequence[Any],
    validation_metadata: Sequence[dict[str, Any]],
    *,
    k_values: Sequence[int],
) -> dict[str, Any]:
    train = np.asarray(train_features, dtype=np.float64)
    validation = np.asarray(validation_features, dtype=np.float64)
    labels = np.asarray([str(value) for value in train_labels], dtype=object)
    targets = np.asarray([str(value) for value in validation_labels], dtype=object)
    if (
        train.ndim != 2
        or validation.ndim != 2
        or train.shape[1] != validation.shape[1]
        or len(train) != len(labels)
        or len(train) == 0
        or len(validation) != len(targets)
        or len(validation) == 0
        or len(train_metadata) != len(train)
        or len(validation_metadata) != len(validation)
    ):
        raise ValueError("kNN inputs have incompatible shapes")
    requested = sorted({int(value) for value in k_values})
    if not requested or requested[0] < 1 or requested[-1] > len(train):
        raise ValueError("kNN values are invalid for the training population")
    maximum = requested[-1]
    neighbor_rows = []
    fallback_counts: Counter[str] = Counter()
    for query_index, (query, metadata) in enumerate(
        zip(validation, validation_metadata)
    ):
        scopes = (
            (
                "same_dataset_layer",
                lambda row: row["dataset"] == metadata["dataset"]
                and int(row["target_layer"]) == int(metadata["target_layer"]),
            ),
            (
                "same_layer",
                lambda row: int(row["target_layer"])
                == int(metadata["target_layer"]),
            ),
            (
                "same_dataset_depth_bin",
                lambda row: row["dataset"] == metadata["dataset"]
                and int(row["depth_bin"]) == int(metadata["depth_bin"]),
            ),
            ("global", lambda _row: True),
        )
        pool = []
        scope_name = ""
        for name, predicate in scopes:
            pool = [index for index, row in enumerate(train_metadata) if predicate(row)]
            if len(pool) >= maximum:
                scope_name = name
                break
        if len(pool) < maximum:
            raise RuntimeError("kNN fallback did not supply enough training states")
        fallback_counts[scope_name] += 1
        distances = _cosine_distances(query, train[pool])
        order = np.argsort(distances, kind="mergesort")[:maximum]
        indices = [pool[int(index)] for index in order]
        neighbor_rows.append(
            {
                "query_index": query_index,
                "target": str(targets[query_index]),
                "scope": scope_name,
                "indices": indices,
                "labels": [str(labels[index]) for index in indices],
                "distances": [float(value) for value in distances[order]],
            }
        )
    by_k = {}
    for k in requested:
        correct = 0
        purities = []
        mean_distances = []
        for row in neighbor_rows:
            neighbor_labels = row["labels"][:k]
            counts = Counter(neighbor_labels)
            majority = min(
                (label for label, count in counts.items() if count == max(counts.values())),
                key=str,
            )
            correct += int(majority == row["target"])
            purities.append(
                sum(label == row["target"] for label in neighbor_labels) / k
            )
            mean_distances.append(sum(row["distances"][:k]) / k)
        by_k[str(k)] = {
            "records": len(neighbor_rows),
            "majority_accuracy": correct / len(neighbor_rows),
            "mean_label_purity": float(np.mean(purities)),
            "mean_neighbor_distance": float(np.mean(mean_distances)),
        }
    pair_rows = [
        {"distance": distance, "agreement": int(label == row["target"])}
        for row in neighbor_rows
        for label, distance in zip(row["labels"], row["distances"])
    ]
    return {
        "by_k": by_k,
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "neighbor_pairs_at_max_k": pair_rows,
    }


def first_deviation_bucket(predicted: int | None, boundary: int) -> str:
    if predicted is None:
        return "never"
    delta = int(predicted) - int(boundary)
    if delta == 0:
        return "exact"
    if delta == -1:
        return "within_1_early"
    if delta == 1:
        return "within_1_late"
    if delta == -2:
        return "within_2_early"
    if delta == 2:
        return "within_2_late"
    return "too_early" if delta < 0 else "too_late"


def compact_knn_label_consistency(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove plot-only neighbor pairs from the portable kNN report."""

    representations = {}
    for representation, tasks in result["representations"].items():
        representations[representation] = {}
        for task, values in tasks.items():
            pairs = values.get("neighbor_pairs_at_max_k", [])
            representations[representation][task] = {
                key: value
                for key, value in values.items()
                if key != "neighbor_pairs_at_max_k"
            }
            representations[representation][task][
                "neighbor_pairs_at_max_k_records"
            ] = len(pairs)
    return {
        "contract": result["contract"],
        "representations": representations,
        "entropy": result["entropy"],
    }


def build_label_incompleteness_subset(
    states: Sequence[dict[str, Any]],
    outputs: Mapping[str, Mapping[str, dict[str, Any]]],
    *,
    cap_per_architecture_action: int,
    seed: int,
) -> list[dict[str, Any]]:
    if cap_per_architecture_action < 1:
        raise ValueError("label-audit cap must be positive")
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        if state["split"] != "validation" or state["state_kind"] != "mandatory_deviation":
            continue
        for architecture, by_state in outputs.items():
            prediction = by_state[state["state_id"]]
            action = str(prediction["predicted_action"])
            if action == "FULL" or action in state["valid_actions"]:
                continue
            candidates[(str(architecture), action)].append(
                {
                    **state,
                    "architecture": str(architecture),
                    "predicted_action": action,
                    "action_probabilities": prediction["action_probabilities"],
                }
            )
    selected = []
    for cell, values in sorted(candidates.items()):
        values.sort(
            key=lambda row: _stable(
                seed, "label-audit-subset", cell[0], cell[1], row["state_id"]
            )
        )
        selected.extend(values[:cap_per_architecture_action])
    selected.sort(key=lambda row: (row["architecture"], row["predicted_action"], row["state_id"]))
    return selected


def candidate_suffix_routes(
    manifest_row: dict[str, Any],
    state: dict[str, Any],
    *,
    predicted_action: str,
    max_suffixes: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Replace the boundary action while retaining bounded known suffixes."""

    if max_suffixes < 1 or predicted_action not in FOUR_ACTIONS:
        raise ValueError("suffix cap/action is invalid")
    if predicted_action in state["valid_actions"]:
        raise ValueError("label-incompleteness audit requires a cached-invalid action")
    layer = int(state["target_layer"])
    prefix = tuple(str(value) for value in state["prefix_actions"])
    if len(prefix) != layer or str(manifest_row["uid"]) != str(state["uid"]):
        raise ValueError("audit state prefix or UID is inconsistent")
    route_indices = sorted(
        {int(value) for value in state["boundary_route_indices"]},
        key=lambda index: _stable(
            seed,
            "label-audit-suffix",
            state["state_id"] if "state_id" in state else state["uid"],
            index,
        ),
    )[:max_suffixes]
    output = []
    seen: set[tuple[str, ...]] = set()
    for route_index in route_indices:
        source = manifest_row["valid_routes"][route_index]
        actions = tuple(str(value) for value in source["actions"])
        if actions[:layer] != prefix:
            raise RuntimeError("known suffix route does not reach the audited state")
        candidate = (*prefix, predicted_action, *actions[layer + 1 :])
        if candidate in seen:
            continue
        seen.add(candidate)
        output.append(
            {
                "candidate_index": len(output),
                "source_route_index": route_index,
                "source_route_key": source.get("route_key", f"route_index_{route_index}"),
                "actions": list(candidate),
                "predicted_action_cached_invalid": True,
            }
        )
    if not output:
        raise RuntimeError("label audit did not construct any suffix candidate")
    return output


def summarize_label_incompleteness_audit(
    subset: Sequence[dict[str, Any]],
    executions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize bounded route replays at the audited-state level.

    Multiple known suffixes can be tested for one cached-invalid boundary
    action.  The scientific unit is the state: one successful suffix is
    sufficient positive evidence that its cached action set is incomplete.
    """

    by_state: dict[tuple[str, str], dict[str, Any]] = {}
    expected: set[tuple[str, str, int]] = set()
    for state in subset:
        state_id = str(state["state_id"])
        architecture = str(state["architecture"])
        audit_key = (architecture, state_id)
        if audit_key in by_state:
            raise ValueError("label audit subset contains duplicate architecture/state units")
        by_state[audit_key] = state
        candidates = state.get("candidate_routes")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("label audit state has no candidate routes")
        for candidate in candidates:
            key = (architecture, state_id, int(candidate["candidate_index"]))
            if key in expected:
                raise ValueError("label audit subset contains duplicate candidates")
            expected.add(key)

    observed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for execution in executions:
        key = (
            str(execution["architecture"]),
            str(execution["state_id"]),
            int(execution["candidate_index"]),
        )
        if key in observed:
            raise ValueError("label audit executions contain duplicate candidates")
        observed[key] = execution
    if set(observed) != expected:
        raise ValueError("label audit executions must exactly cover frozen candidates")

    state_results = []
    cell_counts: dict[str, Counter[str]] = defaultdict(Counter)
    action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    architecture_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for architecture, state_id in sorted(by_state):
        state = by_state[(architecture, state_id)]
        candidates = sorted(
            (
                observed[
                    (architecture, state_id, int(candidate["candidate_index"]))
                ]
                for candidate in state["candidate_routes"]
            ),
            key=lambda row: int(row["candidate_index"]),
        )
        successful = [
            int(row["candidate_index"]) for row in candidates if bool(row["correct"])
        ]
        rescued = bool(successful)
        action = str(state["predicted_action"])
        for counter in (
            cell_counts[f"{architecture}:{action}"],
            action_counts[action],
            architecture_counts[architecture],
        ):
            counter["states"] += 1
            counter["rescued"] += int(rescued)
            counter["candidate_executions"] += len(candidates)
        state_results.append(
            {
                "state_id": state_id,
                "uid": str(state["uid"]),
                "architecture": architecture,
                "predicted_action": action,
                "candidate_executions": candidates,
                "successful_candidate_indices": successful,
                "cached_invalid_but_execution_correct": rescued,
                "status": (
                    "cached_invalid_but_execution_correct"
                    if rescued
                    else "no_bounded_rescue"
                ),
            }
        )

    def finalize(values: Mapping[str, Counter[str]]) -> dict[str, Any]:
        return {
            key: {
                **dict(counts),
                "fraction": counts["rescued"] / counts["states"],
            }
            for key, counts in sorted(values.items())
        }

    rescued_states = sum(
        int(row["cached_invalid_but_execution_correct"]) for row in state_results
    )
    return {
        "states": len(state_results),
        "candidate_executions": len(executions),
        "cached_invalid_but_execution_correct_states": rescued_states,
        "cached_invalid_but_execution_correct_fraction": (
            rescued_states / len(state_results) if state_results else float("nan")
        ),
        "no_bounded_rescue_states": len(state_results) - rescued_states,
        "by_architecture_action": finalize(cell_counts),
        "by_architecture": finalize(architecture_counts),
        "by_predicted_action": finalize(action_counts),
        "state_results": state_results,
        "negative_result_interpretation": (
            "no_bounded_rescue_is_not_proof_of_global_action_invalidity"
        ),
    }
