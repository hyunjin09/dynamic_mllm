from __future__ import annotations

import json

import pytest

from experiments.build_four_action_polar_manifest import (
    build_manifest_rows,
    remap_image_paths,
)
from experiments.prepare_four_action_polar_c2c_ablation import (
    build_c2c_no_allfull_manifest,
    filter_feature_rows,
)
from tools.research_analysis.four_action.sequential_label_jobs import (
    file_sha256,
    safe_filename,
)


def source_row(tmp_path, *, uid: str, split: str = "train", group: str = "gqa:image"):
    image = tmp_path / f"{uid.replace(':', '_')}.jpg"
    image.write_bytes(b"image")
    return {
        "uid": uid,
        "dataset": "gqa",
        "benchmark": "gqa",
        "sample_id": uid.split(":", 1)[1],
        "question": f"question for {uid}",
        "prompt": f"question for {uid}\nAnswer briefly.",
        "image_path": str(image),
        "image_id": group,
        "image_group_id": group,
        "source_split": split,
    }


def write_record(
    records_root, source, *, correct: bool = True, zero_valid_replay_failure: bool = False
) -> None:
    route = ["FULL"] * 27 + ["READ_ONLY"]
    record = {
        "schema_version": "exact_sequential_four_action_sample_v1",
        "passed": True,
        "uid": source["uid"],
        "dataset": source["dataset"],
        "sample_id": source["sample_id"],
        "image_id": source["image_id"],
        "image_group_id": source["image_group_id"],
        "source_split": source["source_split"],
        "route_type": "W2C",
        "label_semantics": "correcting_w2c",
        "source_positive_route_count": 1,
        "source_route_replay_valid_count": 0 if zero_valid_replay_failure else 1,
        "source_route_replay_failure_count": 1 if zero_valid_replay_failure else 0,
        "execution_contract": {
            "contract_sha256": "executor-contract",
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "layer_count": 28,
        },
        "unique_valid_four_action_routes": [] if zero_valid_replay_failure else [
            {
                "four_action_route": route,
                "route_key": "|".join(route),
                "label_semantics": "correcting_w2c",
                "evaluation": {"correct": correct},
                "source_binary_route_ids": ["route:a"],
            }
        ],
    }
    path = records_root / safe_filename(source["uid"])
    path.write_text(json.dumps(record) + "\n")
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n"
    )


def test_manifest_builder_freezes_valid_routes_and_group_disjoint_splits(tmp_path) -> None:
    records = tmp_path / "records"
    records.mkdir()
    train = source_row(tmp_path, uid="gqa:train", split="train", group="gqa:image-a")
    validation = source_row(
        tmp_path, uid="gqa:validation", split="validation", group="gqa:image-b"
    )
    write_record(records, train)
    write_record(records, validation)

    rows, audit = build_manifest_rows([train, validation], records, layer_count=28)

    assert [row["uid"] for row in rows] == ["gqa:train", "gqa:validation"]
    assert rows[0]["split_group"] == "gqa:image-a"
    assert rows[0]["valid_routes"][0]["actions"][-1] == "READ_ONLY"
    assert len(rows[0]["valid_routes"][0]["actions"]) == 28
    assert audit["passed"] is True
    assert audit["samples"] == 2
    assert audit["routes"] == 2
    assert audit["split_counts"] == {"train": 1, "validation": 1}
    assert audit["group_leakage_count"] == 0


def test_manifest_builder_rejects_incorrect_or_checksum_invalid_labels(tmp_path) -> None:
    records = tmp_path / "records"
    records.mkdir()
    source = source_row(tmp_path, uid="gqa:bad")
    write_record(records, source, correct=False)

    with pytest.raises(ValueError, match="not evaluator-correct"):
        build_manifest_rows([source], records, layer_count=28)

    write_record(records, source, correct=True)
    path = records / safe_filename(source["uid"])
    path.with_suffix(path.suffix + ".sha256").write_text("bad  record.json\n")
    with pytest.raises(ValueError, match="checksum"):
        build_manifest_rows([source], records, layer_count=28)


def test_manifest_builder_rejects_image_group_leakage(tmp_path) -> None:
    records = tmp_path / "records"
    records.mkdir()
    train = source_row(tmp_path, uid="gqa:train", split="train", group="gqa:image")
    validation = source_row(
        tmp_path, uid="gqa:validation", split="validation", group="gqa:image"
    )
    write_record(records, train)
    write_record(records, validation)

    with pytest.raises(ValueError, match="split-group leakage"):
        build_manifest_rows([train, validation], records, layer_count=28)


def test_manifest_builder_accounts_for_zero_valid_current_replay_exclusions(tmp_path) -> None:
    records = tmp_path / "records"
    records.mkdir()
    valid = source_row(tmp_path, uid="gqa:valid", group="gqa:image-a")
    excluded = source_row(tmp_path, uid="gqa:excluded", group="gqa:image-b")
    write_record(records, valid)
    write_record(records, excluded, zero_valid_replay_failure=True)

    rows, audit = build_manifest_rows([valid, excluded], records, layer_count=28)

    assert [row["uid"] for row in rows] == ["gqa:valid"]
    assert audit["source_samples"] == 2
    assert audit["samples"] == 1
    assert audit["zero_valid_route_exclusions"] == 1
    assert audit["zero_valid_route_exclusion_uids"] == ["gqa:excluded"]


def test_manifest_builder_remaps_machine_local_image_prefix() -> None:
    rows = [{"uid": "gqa:one", "image_path": "/old/root/images/one.jpg"}]

    mapped = remap_image_paths(
        rows,
        source_prefix="/old/root",
        target_prefix="/new/root",
    )

    assert mapped[0]["image_path"] == "/new/root/images/one.jpg"
    assert rows[0]["image_path"] == "/old/root/images/one.jpg"


def test_manifest_builder_rejects_partial_or_mismatched_prefix_mapping() -> None:
    rows = [{"uid": "gqa:one", "image_path": "/old/root/images/one.jpg"}]

    with pytest.raises(ValueError, match="requires both"):
        remap_image_paths(rows, source_prefix="/old/root", target_prefix=None)
    with pytest.raises(ValueError, match="does not start"):
        remap_image_paths(
            rows,
            source_prefix="/different/root",
            target_prefix="/new/root",
        )


def test_c2c_no_allfull_ablation_changes_only_train_c2c_and_excludes_empty() -> None:
    full = {"route_key": "full", "actions": ["FULL", "FULL"]}
    read = {"route_key": "read", "actions": ["FULL", "READ_ONLY"]}
    rows = [
        {
            "uid": "train-empty", "split": "train", "route_type": "C2C",
            "dataset": "gqa", "split_group": "a", "valid_routes": [full],
            "valid_route_count": 1,
        },
        {
            "uid": "train-mixed", "split": "train", "route_type": "C2C",
            "dataset": "chartqa", "split_group": "b", "valid_routes": [full, read],
            "valid_route_count": 2,
        },
        {
            "uid": "validation-full", "split": "validation", "route_type": "C2C",
            "dataset": "textvqa", "split_group": "c", "valid_routes": [full],
            "valid_route_count": 1,
        },
        {
            "uid": "train-w2c", "split": "train", "route_type": "W2C",
            "dataset": "gqa", "split_group": "d", "valid_routes": [read],
            "valid_route_count": 1,
        },
    ]

    derived, audit = build_c2c_no_allfull_manifest(rows, num_layers=2)

    assert [row["uid"] for row in derived] == [
        "train-mixed", "validation-full", "train-w2c"
    ]
    assert derived[0]["valid_routes"] == [read]
    assert derived[0]["valid_route_count"] == 1
    assert derived[1]["valid_routes"] == [full]
    assert audit["removed_train_c2c_allfull_routes"] == 2
    assert audit["source_routes"] == 5
    assert audit["routes"] == 3
    assert audit["excluded_train_c2c_records"] == 1
    assert audit["excluded_uids"] == ["train-empty"]
    assert audit["validation_rows_changed"] == 0
    assert audit["w2c_rows_changed"] == 0


def test_c2c_ablation_feature_filter_reuses_only_retained_uid_rows() -> None:
    rows = [
        {"uid": "a", "path": "shared.pt", "sha256": "x"},
        {"uid": "b", "path": "shared.pt", "sha256": "x"},
        {"uid": "c", "path": "other.pt", "sha256": "y"},
    ]

    assert filter_feature_rows(rows, {"a", "c"}) == [rows[0], rows[2]]
    with pytest.raises(RuntimeError, match="coverage"):
        filter_feature_rows(rows, {"a", "missing"})
