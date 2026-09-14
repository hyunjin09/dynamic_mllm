import numpy as np
import pytest

from dense_failure_stage1.trigger_map import (
    build_trigger_map,
    cumulative_trigger_rows,
    dataset_breakdown_rows,
    select_reproducibility_rows,
    split_summary,
    trigger_by_depth_bin_rows,
    trigger_by_layer_rows,
    trigger_depth_bin,
    trigger_score_stats_rows,
    validate_trigger_map,
)
from experiments.audit_stage1_trigger_map import (
    canonical_hash,
    compare_prior_summaries,
    frozen_contract_payload,
)


THRESHOLD = 0.8


def _split_row(uid: str, *, wrong: bool, dataset: str = "gqa") -> dict:
    return {
        "uid": uid,
        "dataset": dataset,
        "split": "train",
        "image_group_id": f"group:{uid}",
        "current_dense_wrong": wrong,
        "current_dense_correct": not wrong,
    }


def _score_row(uid: str, values: list[float]) -> dict:
    return {"uid": uid, **{f"p_{layer}": value for layer, value in enumerate(values)}}


def test_build_trigger_map_uses_strict_first_crossing_and_preserves_scores() -> None:
    split_rows = [
        _split_row("correct", wrong=False),
        _split_row("wrong", wrong=True),
    ]
    score_rows = [
        _score_row("correct", [THRESHOLD] + [0.1] * 27),
        _score_row("wrong", [0.1, 0.81, 0.95] + [0.1] * 25),
    ]

    rows = build_trigger_map(split_rows, score_rows, threshold=THRESHOLD)

    assert rows[0]["triggered"] is False
    assert rows[0]["first_trigger_layer"] is None
    assert rows[0]["score_at_trigger"] is None
    assert rows[1]["triggered"] is True
    assert rows[1]["first_trigger_layer"] == 1
    assert rows[1]["score_at_trigger"] == pytest.approx(0.81)
    assert [rows[1][f"score_l{layer}"] for layer in range(28)] == [
        0.1,
        0.81,
        0.95,
        *([0.1] * 25),
    ]


def test_build_trigger_map_rejects_missing_and_duplicate_score_uids() -> None:
    split_rows = [_split_row("a", wrong=False), _split_row("b", wrong=True)]
    score = _score_row("a", [0.1] * 28)

    with pytest.raises(ValueError, match="UID coverage"):
        build_trigger_map(split_rows, [score], threshold=THRESHOLD)
    with pytest.raises(ValueError, match="duplicate"):
        build_trigger_map(split_rows, [score, score], threshold=THRESHOLD)


def test_validate_trigger_map_rejects_inconsistent_trigger_claims() -> None:
    rows = build_trigger_map(
        [_split_row("a", wrong=True)],
        [_score_row("a", [0.1, 0.9] + [0.1] * 26)],
        threshold=THRESHOLD,
    )
    rows[0]["first_trigger_layer"] = 2

    with pytest.raises(ValueError, match="first-trigger"):
        validate_trigger_map(rows, threshold=THRESHOLD, expected_records=1)


def test_split_summary_partitions_all_four_cells() -> None:
    split_rows = [
        _split_row("cn", wrong=False),
        _split_row("ct", wrong=False),
        _split_row("wn", wrong=True),
        _split_row("wt", wrong=True),
    ]
    scores = [
        _score_row("cn", [0.1] * 28),
        _score_row("ct", [0.9] + [0.1] * 27),
        _score_row("wn", [0.1] * 28),
        _score_row("wt", [0.1, 0.9] + [0.1] * 26),
    ]

    summary = split_summary(build_trigger_map(split_rows, scores, threshold=THRESHOLD))

    assert summary["dense_correct_no_trigger"] == 1
    assert summary["dense_correct_trigger"] == 1
    assert summary["dense_wrong_no_trigger"] == 1
    assert summary["dense_wrong_trigger"] == 1
    assert summary["correct_preservation"] == pytest.approx(0.5)
    assert summary["wrong_trigger_recall"] == pytest.approx(0.5)
    assert summary["trigger_precision"] == pytest.approx(0.5)


def test_cumulative_and_dataset_metrics_use_full_class_denominators() -> None:
    split_rows = [
        _split_row("c0", wrong=False, dataset="gqa"),
        _split_row("cn", wrong=False, dataset="gqa"),
        _split_row("w1", wrong=True, dataset="gqa"),
        _split_row("wn", wrong=True, dataset="gqa"),
    ]
    scores = [
        _score_row("c0", [0.9] + [0.1] * 27),
        _score_row("cn", [0.1] * 28),
        _score_row("w1", [0.1, 0.9] + [0.1] * 26),
        _score_row("wn", [0.1] * 28),
    ]
    rows = build_trigger_map(split_rows, scores, threshold=THRESHOLD)

    cumulative = cumulative_trigger_rows(rows)
    assert cumulative[0]["correct_cumulative_trigger"] == pytest.approx(0.5)
    assert cumulative[0]["wrong_cumulative_trigger"] == pytest.approx(0.0)
    assert cumulative[1]["wrong_cumulative_trigger"] == pytest.approx(0.5)
    breakdown = dataset_breakdown_rows(rows)
    assert breakdown[0]["correct_preservation"] == pytest.approx(0.5)
    assert breakdown[0]["wrong_trigger_recall"] == pytest.approx(0.5)
    assert breakdown[0]["trigger_precision"] == pytest.approx(0.5)


def test_layer_depth_and_score_tables_use_triggered_class_denominators() -> None:
    split_rows = [
        _split_row("c0", wrong=False),
        _split_row("c1", wrong=False),
        _split_row("w1", wrong=True),
        _split_row("w9", wrong=True),
        _split_row("wn", wrong=True),
    ]
    scores = [
        _score_row("c0", [0.81] + [0.1] * 27),
        _score_row("c1", [0.1, 0.82] + [0.1] * 26),
        _score_row("w1", [0.1, 0.83] + [0.1] * 26),
        _score_row("w9", [0.1] * 9 + [0.84] + [0.1] * 18),
        _score_row("wn", [0.1] * 28),
    ]
    rows = build_trigger_map(split_rows, scores, threshold=THRESHOLD)

    by_layer = trigger_by_layer_rows(rows)
    wrong_layer_1 = next(
        row for row in by_layer if row["dense_class"] == "wrong" and row["layer"] == 1
    )
    assert wrong_layer_1["count"] == 1
    assert wrong_layer_1["fraction_of_triggered_class"] == pytest.approx(0.5)
    assert wrong_layer_1["fraction_of_full_class"] == pytest.approx(1 / 3)
    by_bin = trigger_by_depth_bin_rows(rows)
    wrong_early = next(
        row for row in by_bin if row["dense_class"] == "wrong" and row["depth_bin"] == "early"
    )
    wrong_middle = next(
        row for row in by_bin if row["dense_class"] == "wrong" and row["depth_bin"] == "middle"
    )
    assert wrong_early["count"] == 1
    assert wrong_middle["count"] == 1
    stats = trigger_score_stats_rows(rows)
    wrong_stats = next(row for row in stats if row["dense_class"] == "wrong")
    assert wrong_stats["records"] == 2
    assert wrong_stats["mean"] == pytest.approx(0.835)
    assert wrong_stats["margin_mean"] == pytest.approx(0.035)


@pytest.mark.parametrize(
    ("layer", "expected"), [(0, "early"), (8, "early"), (9, "middle"), (18, "middle"), (19, "late"), (27, "late")]
)
def test_trigger_depth_bin_boundaries(layer: int, expected: str) -> None:
    assert trigger_depth_bin(layer) == expected


def test_reproducibility_selection_is_deterministic_and_stratified() -> None:
    rows = []
    for split in ("val", "test"):
        for dataset in ("gqa", "chartqa", "textvqa"):
            for wrong in (False, True):
                for index in range(4):
                    row = _split_row(f"{split}:{dataset}:{wrong}:{index}", wrong=wrong, dataset=dataset)
                    row["split"] = split
                    rows.append(row)

    first = select_reproducibility_rows(rows, seed=17, per_cell=2)
    second = select_reproducibility_rows(list(reversed(rows)), seed=17, per_cell=2)

    assert [row["uid"] for row in first] == [row["uid"] for row in second]
    cells = {(row["split"], row["dataset"], row["current_dense_wrong"]) for row in first}
    assert len(first) == 24
    assert len(cells) == 12
    assert np.unique([row["uid"] for row in first]).size == 24


def test_frozen_contract_retains_nested_static_config_for_runtime_validation() -> None:
    config = {
        "schema_version": "stage1_trigger_map_audit_config_v1",
        "sources": {"split_manifest": "split.jsonl"},
        "gate": {"threshold": 0.5},
        "reproducibility": {"absolute_probability_tolerance": 1e-6},
    }

    contract = frozen_contract_payload(config, {"git_commit": "abc"})

    assert contract["schema_version"] == "stage1_trigger_map_audit_frozen_contract_v1"
    assert contract["static_config"] == config
    assert contract["sources"] == config["sources"]
    assert contract["gate"] == config["gate"]
    assert contract["reproducibility"] == config["reproducibility"]
    assert contract["provenance"] == {"git_commit": "abc"}
    assert contract["contract_sha256"] == canonical_hash(contract)


def test_prior_summary_comparison_accepts_selection_json_validation_shape() -> None:
    observed = {
        split: {
            "records": 8,
            "dense_correct": 4,
            "dense_wrong": 4,
            "triggered": 4,
            "dense_correct_trigger": 1,
            "dense_wrong_trigger": 3,
            "correct_preservation": 0.75,
            "wrong_trigger_recall": 0.75,
            "trigger_precision": 0.75,
        }
        for split in ("val", "test")
    }
    validation_point = {
        "records": 8,
        "correct": 4,
        "wrong": 4,
        "triggered": 4,
        "correct_false_triggers": 1,
        "wrong_detected": 3,
        "correct_preservation": 0.75,
        "wrong_detection_recall": 0.75,
        "failure_precision": 0.75,
    }
    test_csv_row = {key: str(value) for key, value in validation_point.items()}

    result = compare_prior_summaries(
        observed, validation_point=validation_point, test_row=test_csv_row
    )

    assert result["passed"] is True
    assert result["exact_checks"] == 18
