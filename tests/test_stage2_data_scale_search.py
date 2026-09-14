from __future__ import annotations

from collections import Counter

import pytest

from dense_failure_stage2.data_scale_search import (
    build_search_work,
    build_new_trigger_map,
    chartqa_stratum,
    image_extension_for_format,
    most_common_answer,
    question_length_stratum,
    select_legacy_source_rows,
    textvqa_stratum,
    select_metadata_stratified,
    validate_frozen_candidates,
)


def _source_row(index: int, *, stratum: str, image: str | None = None) -> dict:
    return {
        "uid": f"gqa:new-{index}",
        "dataset": "gqa",
        "native_row_key": f"train:{index}",
        "native_image_id": image or f"image-{index}",
        "image_group_id": f"source:{image or f'image-{index}'}",
        "source_stratum": stratum,
    }


def test_metadata_stratified_selection_matches_reference_and_uses_unique_groups():
    rows = [
        *[_source_row(index, stratum="verify") for index in range(8)],
        *[_source_row(index + 100, stratum="query") for index in range(8)],
        # A second question on an already represented image must not increase weight.
        _source_row(999, stratum="verify", image="image-0"),
    ]

    selected, audit = select_metadata_stratified(
        rows,
        target=8,
        reference_strata=Counter({"verify": 3, "query": 1}),
        seed=20260901,
    )

    assert Counter(row["source_stratum"] for row in selected) == Counter(
        {"verify": 6, "query": 2}
    )
    assert len({row["native_image_id"] for row in selected}) == 8
    assert audit["selected"] == 8
    assert audit["backfilled"] == 0


def test_metadata_selection_deduplicates_content_groups_not_only_native_ids():
    rows = [_source_row(index, stratum="a") for index in range(4)]
    rows[1]["image_group_id"] = rows[0]["image_group_id"]
    selected, _ = select_metadata_stratified(
        rows,
        target=3,
        reference_strata={"a": 1},
        seed=0,
    )
    assert len({row["image_group_id"] for row in selected}) == 3


def test_metadata_stratified_selection_backfills_without_using_outcomes():
    rows = [
        *[_source_row(index, stratum="rare") for index in range(2)],
        *[_source_row(index + 100, stratum="common") for index in range(10)],
    ]
    for index, row in enumerate(rows):
        row["current_dense_wrong"] = bool(index % 2)

    selected, audit = select_metadata_stratified(
        rows,
        target=6,
        reference_strata=Counter({"rare": 3, "common": 1}),
        seed=7,
    )

    assert len(selected) == 6
    assert Counter(row["source_stratum"] for row in selected) == Counter(
        {"rare": 2, "common": 4}
    )
    assert audit["backfilled"] == 2


def test_frozen_candidate_validation_rejects_legacy_or_internal_leakage():
    rows = [_source_row(1, stratum="a"), _source_row(2, stratum="a")]
    with pytest.raises(ValueError, match="legacy UID overlap"):
        validate_frozen_candidates(
            rows,
            expected_counts={"gqa": 2},
            legacy_uids={rows[0]["uid"]},
            legacy_groups=set(),
        )

    duplicated = [rows[0], {**rows[1], "image_group_id": rows[0]["image_group_id"]}]
    with pytest.raises(ValueError, match="duplicate image group"):
        validate_frozen_candidates(
            duplicated,
            expected_counts={"gqa": 2},
            legacy_uids=set(),
            legacy_groups=set(),
        )


def test_trigger_map_uses_strict_global_threshold_and_dense_current_labels():
    candidates = [
        {"uid": "gqa:a", "dataset": "gqa", "image_group_id": "sha:a"},
        {"uid": "gqa:b", "dataset": "gqa", "image_group_id": "sha:b"},
    ]
    dense = [
        {"uid": "gqa:a", "dataset": "gqa", "image_group_id": "sha:a", "current_dense_wrong": True},
        {"uid": "gqa:b", "dataset": "gqa", "image_group_id": "sha:b", "current_dense_wrong": False},
    ]
    scores = [
        {"uid": "gqa:a", **{f"p_{layer}": 0.5 for layer in range(28)}},
        {
            "uid": "gqa:b",
            **{f"p_{layer}": (0.9 if layer == 3 else 0.1) for layer in range(28)},
        },
    ]

    trigger_map, triggered_wrong, triggered_correct = build_new_trigger_map(
        candidates, dense, scores, threshold=0.5
    )

    assert trigger_map[0]["triggered"] is False  # equality is not a crossing
    assert trigger_map[0]["dense_wrong"] is True
    assert trigger_map[1]["triggered"] is True
    assert trigger_map[1]["first_trigger_layer"] == 3
    assert triggered_wrong == []
    assert [row["uid"] for row in triggered_correct] == ["gqa:b"]


def test_source_metadata_strata_and_textvqa_reference_answer_are_deterministic():
    assert chartqa_stratum("human", "How many bars are shown in this chart?") == "human|q6_10"
    assert chartqa_stratum("augmented", "Value?") == "augmented|q1_5"
    assert question_length_stratum("What is written on the sign?") == "q6_10"
    assert textvqa_stratum("What text is printed here?", ["A", "B", "C"]) == "q1_5|ocr1_5"
    assert most_common_answer(["nokia", "Nokia", "toshiba", "nokia"]) == "nokia"
    # Stable first occurrence wins an exact-count tie.
    assert most_common_answer(["beta", "alpha"]) == "beta"


def test_legacy_source_join_filters_the_larger_portable_pool_to_current_uids():
    current = [{"uid": "a"}, {"uid": "b"}]
    portable = [{"uid": "extra"}, {"uid": "b"}, {"uid": "a"}]
    assert [row["uid"] for row in select_legacy_source_rows(portable, current)] == ["a", "b"]
    with pytest.raises(ValueError, match="coverage"):
        select_legacy_source_rows([{"uid": "a"}], current)


def test_mpo_source_images_use_the_jpeg_family_extension():
    assert image_extension_for_format("JPEG") == ".jpg"
    assert image_extension_for_format("MPO") == ".jpg"
    assert image_extension_for_format("PNG") == ".png"
    with pytest.raises(ValueError, match="unsupported"):
        image_extension_for_format("TIFF")


def test_search_work_joins_trigger_candidate_and_dense_without_changing_labels():
    candidates = [
        {
            "uid": "gqa:w",
            "dataset": "gqa",
            "image_group_id": "sha:w",
            "image_content_sha256": "w" * 64,
            "local_image_path": "/tmp/w.jpg",
            "sample_id": "w",
            "prompt": "question",
            "question": "question",
            "answer": "answer",
            "all_answer_norms": None,
            "max_new_tokens": 16,
        },
        {
            "uid": "gqa:c",
            "dataset": "gqa",
            "image_group_id": "sha:c",
            "image_content_sha256": "c" * 64,
            "local_image_path": "/tmp/c.jpg",
            "sample_id": "c",
            "prompt": "question",
            "question": "question",
            "answer": "answer",
            "all_answer_norms": None,
            "max_new_tokens": 16,
        },
    ]
    dense = [
        {
            "uid": "gqa:w",
            "dataset": "gqa",
            "image_group_id": "sha:w",
            "image_content_sha256": "w" * 64,
            "current_dense_wrong": True,
        },
        {
            "uid": "gqa:c",
            "dataset": "gqa",
            "image_group_id": "sha:c",
            "image_content_sha256": "c" * 64,
            "current_dense_wrong": False,
        },
    ]
    triggers = [
        {
            "uid": "gqa:w",
            "dataset": "gqa",
            "image_group_id": "sha:w",
            "dense_wrong": True,
            "triggered": True,
            "first_trigger_layer": 7,
        },
        {
            "uid": "gqa:c",
            "dataset": "gqa",
            "image_group_id": "sha:c",
            "dense_wrong": False,
            "triggered": True,
            "first_trigger_layer": 9,
        },
    ]

    work = build_search_work(candidates, dense, triggers, maximum_iterations=200)

    assert [row["task_type"] for row in work] == [
        "triggered_correct_preservation",
        "triggered_wrong",
    ]
    assert work[0]["estimated_terminal_evaluations"] == 2
    assert work[1]["estimated_terminal_evaluations"] == 3 * (28 - 7) + 200
    assert work[1]["sample"]["image_content_sha256"] == "w" * 64


def test_search_work_rejects_nontriggered_or_provenance_mismatched_rows():
    candidate = {
        "uid": "gqa:x",
        "dataset": "gqa",
        "image_group_id": "sha:x",
        "image_content_sha256": "x" * 64,
    }
    dense = {
        "uid": "gqa:x",
        "dataset": "gqa",
        "image_group_id": "sha:x",
        "image_content_sha256": "x" * 64,
        "current_dense_wrong": True,
    }
    trigger = {
        "uid": "gqa:x",
        "dataset": "gqa",
        "image_group_id": "sha:x",
        "dense_wrong": True,
        "triggered": False,
        "first_trigger_layer": None,
    }
    with pytest.raises(ValueError, match="must all be triggered"):
        build_search_work([candidate], [dense], [trigger], maximum_iterations=200)
    with pytest.raises(ValueError, match="provenance"):
        build_search_work(
            [candidate],
            [{**dense, "image_group_id": "sha:other"}],
            [{**trigger, "triggered": True, "first_trigger_layer": 3}],
            maximum_iterations=200,
        )
