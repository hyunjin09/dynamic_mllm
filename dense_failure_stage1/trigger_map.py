"""Pure helpers for the frozen Shared Random-4 trigger-map audit."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np

from dense_failure_stage1.shared_global_gate import first_trigger_layers


DATASETS = ("gqa", "chartqa", "textvqa")
SPLITS = ("train", "val", "test")
DEPTH_BINS = (("early", 0, 8), ("middle", 9, 18), ("late", 19, 27))


def _score_vector(row: Mapping[str, Any]) -> np.ndarray:
    try:
        values = np.asarray([float(row[f"p_{layer}"]) for layer in range(28)])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"score row is missing a finite 28-layer trajectory: {row.get('uid')}") from error
    if values.shape != (28,) or not np.isfinite(values).all():
        raise ValueError(f"score row is missing a finite 28-layer trajectory: {row.get('uid')}")
    return values.astype(np.float64)


def trigger_depth_bin(layer: int) -> str:
    value = int(layer)
    for name, lower, upper in DEPTH_BINS:
        if lower <= value <= upper:
            return name
    raise ValueError("trigger layer must lie in 0..27")


def build_trigger_map(
    split_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    score_source: str = "unspecified",
) -> list[dict[str, Any]]:
    if not split_rows:
        raise ValueError("trigger map requires a nonempty split")
    split_uids = [str(row["uid"]) for row in split_rows]
    if len(set(split_uids)) != len(split_uids):
        raise ValueError("duplicate UID in split rows")
    score_uids = [str(row["uid"]) for row in score_rows]
    if len(set(score_uids)) != len(score_uids):
        raise ValueError("duplicate UID in score rows")
    if set(split_uids) != set(score_uids):
        raise ValueError("split/score UID coverage differs")
    by_uid = {str(row["uid"]): row for row in score_rows}
    ordered = sorted(split_rows, key=lambda row: str(row["uid"]))
    matrix = np.stack([_score_vector(by_uid[str(row["uid"])]) for row in ordered])
    first = first_trigger_layers(matrix, threshold=float(threshold), window=range(28))
    output = []
    control = f"global_raw_threshold:score>{float(threshold):.17g}:layers=0-27"
    for row, values, first_layer in zip(ordered, matrix, first):
        uid = str(row["uid"])
        wrong = bool(row["current_dense_wrong"])
        correct = bool(row.get("current_dense_correct", not wrong))
        if correct == wrong:
            raise ValueError(f"dense labels are not complementary for {uid}")
        layer = None if int(first_layer) < 0 else int(first_layer)
        record = {
            "schema_version": "stage1_trigger_map_row_v1",
            "uid": uid,
            "dataset": str(row["dataset"]),
            "split": str(row["split"]),
            "group_id": str(row["image_group_id"]),
            "image_group_id": str(row["image_group_id"]),
            "dense_correct": correct,
            "dense_wrong": wrong,
            "triggered": layer is not None,
            "first_trigger_layer": layer,
            "score_at_trigger": None if layer is None else float(values[layer]),
            "threshold_or_control_used": control,
            "threshold": float(threshold),
            "score_source": str(score_source),
        }
        record.update({f"score_l{index}": float(values[index]) for index in range(28)})
        output.append(record)
    validate_trigger_map(output, threshold=threshold, expected_records=len(ordered))
    return output


def validate_trigger_map(
    rows: Sequence[Mapping[str, Any]], *, threshold: float, expected_records: int
) -> None:
    if len(rows) != int(expected_records):
        raise ValueError("trigger-map record count differs")
    uids = [str(row["uid"]) for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate UID in trigger map")
    tau = float(threshold)
    for row in rows:
        values = np.asarray([float(row[f"score_l{layer}"]) for layer in range(28)])
        crossings = np.flatnonzero(values > tau)
        expected = None if len(crossings) == 0 else int(crossings[0])
        observed = row["first_trigger_layer"]
        if observed != expected or bool(row["triggered"]) != (expected is not None):
            raise ValueError(f"first-trigger consistency failure for {row['uid']}")
        expected_score = None if expected is None else float(values[expected])
        if row["score_at_trigger"] != expected_score:
            raise ValueError(f"trigger-score consistency failure for {row['uid']}")


def split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty trigger map")
    correct = [row for row in rows if bool(row["dense_correct"])]
    wrong = [row for row in rows if bool(row["dense_wrong"])]
    correct_trigger = sum(bool(row["triggered"]) for row in correct)
    wrong_trigger = sum(bool(row["triggered"]) for row in wrong)
    triggered = correct_trigger + wrong_trigger
    return {
        "split": str(rows[0]["split"]),
        "records": len(rows),
        "dense_correct": len(correct),
        "dense_wrong": len(wrong),
        "dense_correct_no_trigger": len(correct) - correct_trigger,
        "dense_correct_trigger": correct_trigger,
        "dense_wrong_no_trigger": len(wrong) - wrong_trigger,
        "dense_wrong_trigger": wrong_trigger,
        "triggered": triggered,
        "correct_preservation": (len(correct) - correct_trigger) / len(correct),
        "correct_false_admission_rate": correct_trigger / len(correct),
        "wrong_trigger_recall": wrong_trigger / len(wrong),
        "wrong_miss_rate": (len(wrong) - wrong_trigger) / len(wrong),
        "trigger_precision": wrong_trigger / triggered if triggered else None,
    }


def cumulative_trigger_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    correct = [row for row in rows if bool(row["dense_correct"])]
    wrong = [row for row in rows if bool(row["dense_wrong"])]
    if not correct or not wrong:
        raise ValueError("cumulative curves require both dense classes")
    output = []
    for layer in range(28):
        correct_count = sum(
            row["first_trigger_layer"] is not None and int(row["first_trigger_layer"]) <= layer
            for row in correct
        )
        wrong_count = sum(
            row["first_trigger_layer"] is not None and int(row["first_trigger_layer"]) <= layer
            for row in wrong
        )
        output.append(
            {
                "split": str(rows[0]["split"]),
                "layer": layer,
                "correct_cumulative_count": correct_count,
                "correct_cumulative_trigger": correct_count / len(correct),
                "wrong_cumulative_count": wrong_count,
                "wrong_cumulative_trigger": wrong_count / len(wrong),
            }
        )
    return output


def trigger_by_layer_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for class_name, class_key in (("correct", "dense_correct"), ("wrong", "dense_wrong")):
        class_rows = [row for row in rows if bool(row[class_key])]
        triggered = [row for row in class_rows if bool(row["triggered"])]
        for layer in range(28):
            count = sum(int(row["first_trigger_layer"]) == layer for row in triggered)
            output.append(
                {
                    "split": str(rows[0]["split"]),
                    "dense_class": class_name,
                    "layer": layer,
                    "count": count,
                    "triggered_class_records": len(triggered),
                    "full_class_records": len(class_rows),
                    "fraction_of_triggered_class": count / len(triggered) if triggered else None,
                    "fraction_of_full_class": count / len(class_rows),
                }
            )
    return output


def trigger_by_depth_bin_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for class_name, class_key in (("correct", "dense_correct"), ("wrong", "dense_wrong")):
        class_rows = [row for row in rows if bool(row[class_key])]
        triggered = [row for row in class_rows if bool(row["triggered"])]
        counts = Counter(trigger_depth_bin(int(row["first_trigger_layer"])) for row in triggered)
        for name, lower, upper in DEPTH_BINS:
            count = counts[name]
            output.append(
                {
                    "split": str(rows[0]["split"]),
                    "dense_class": class_name,
                    "depth_bin": name,
                    "layer_start": lower,
                    "layer_end": upper,
                    "count": count,
                    "triggered_class_records": len(triggered),
                    "full_class_records": len(class_rows),
                    "fraction_of_triggered_class": count / len(triggered) if triggered else None,
                    "fraction_of_full_class": count / len(class_rows),
                }
            )
    return output


def trigger_score_stats_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    threshold = float(rows[0]["threshold"])
    for class_name, class_key in (("correct", "dense_correct"), ("wrong", "dense_wrong")):
        values = np.asarray(
            [float(row["score_at_trigger"]) for row in rows if bool(row[class_key]) and bool(row["triggered"])],
            dtype=np.float64,
        )
        if len(values) == 0:
            raise ValueError(f"no triggered {class_name} records for score statistics")
        output.append(
            {
                "split": str(rows[0]["split"]),
                "dense_class": class_name,
                "records": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                "margin_mean": float((values - threshold).mean()),
                "margin_median": float(np.median(values - threshold)),
            }
        )
    return output


def dataset_breakdown_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        selected = [row for row in rows if str(row["dataset"]) == dataset]
        if not selected:
            continue
        summary = split_summary(selected)
        correct_layers = [
            int(row["first_trigger_layer"])
            for row in selected
            if bool(row["dense_correct"]) and row["first_trigger_layer"] is not None
        ]
        wrong_layers = [
            int(row["first_trigger_layer"])
            for row in selected
            if bool(row["dense_wrong"]) and row["first_trigger_layer"] is not None
        ]
        wrong_bins = Counter(trigger_depth_bin(layer) for layer in wrong_layers)
        output.append(
            {
                "split": str(rows[0]["split"]),
                "dataset": dataset,
                **{key: summary[key] for key in (
                    "records", "dense_correct", "dense_wrong", "triggered",
                    "correct_preservation", "wrong_trigger_recall", "trigger_precision",
                )},
                "median_trigger_layer_dense_correct": (
                    float(np.median(correct_layers)) if correct_layers else None
                ),
                "median_trigger_layer_dense_wrong": (
                    float(np.median(wrong_layers)) if wrong_layers else None
                ),
                "dense_wrong_trigger_early_fraction": wrong_bins["early"] / len(wrong_layers) if wrong_layers else None,
                "dense_wrong_trigger_middle_fraction": wrong_bins["middle"] / len(wrong_layers) if wrong_layers else None,
                "dense_wrong_trigger_late_fraction": wrong_bins["late"] / len(wrong_layers) if wrong_layers else None,
            }
        )
    return output


def select_reproducibility_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int, per_cell: int
) -> list[dict[str, Any]]:
    if per_cell < 1:
        raise ValueError("reproducibility rows per cell must be positive")
    selected = []
    for split in ("val", "test"):
        for dataset in DATASETS:
            for wrong in (False, True):
                cell = [
                    dict(row)
                    for row in rows
                    if str(row["split"]) == split
                    and str(row["dataset"]) == dataset
                    and bool(row["current_dense_wrong"]) is wrong
                ]
                cell.sort(
                    key=lambda row: sha256(f"{int(seed)}:{row['uid']}".encode()).hexdigest()
                )
                if len(cell) < per_cell:
                    raise ValueError(f"reproducibility cell is too small: {split}/{dataset}/{wrong}")
                selected.extend(cell[:per_cell])
    return selected
