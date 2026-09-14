from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dense_failure_stage1.contract import (
    assert_contract_integrity,
    validate_smoke_gate,
)
from dense_failure_stage1.runtime import DenseFeatureCollector
from dense_failure_stage1.runtime import _open_verified_image
from experiments.finalize_current_dense_regeneration import validate_global_completion
from experiments.run_current_dense_regeneration import (
    validate_feature_provenance,
    validate_resume_marker,
)
from tools.research_analysis.dense_failure_stage1 import (
    build_candidate_manifest,
    build_image_group_disjoint_split,
    canonical_payload_sha256,
    locate_user_and_visual_tokens,
    select_dense_smoke,
    validate_candidate_execution_contract,
)


def _portable_row(
    root: Path,
    *,
    uid: str,
    benchmark: str,
    bucket: str,
    sample_id: str,
    image_bytes: bytes,
) -> dict:
    suffix = ".jpg" if benchmark != "chartqa" else ".png"
    bucket_name = "correct" if bucket == "complete_correct" else "wrong"
    image = root / benchmark / f"{bucket_name}__{sample_id}{suffix}"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(image_bytes)
    from hashlib import sha256

    return {
        "uid": uid,
        "sample_id": sample_id,
        "benchmark": benchmark,
        "question": f"question {uid}",
        "prompt": f"question {uid}\nAnswer the question using a single word or phrase.",
        "answer": "answer",
        "all_answer_norms": None,
        "metric_name": "exact_match_ignore_case_punctuation",
        "correctness_threshold": 1.0,
        "max_new_tokens": 16,
        "source_bucket": bucket,
        "source_asset_id": f"asset:{sample_id}",
        "source_full_prediction": "answer" if bucket == "complete_correct" else "other",
        "source_full_score": 1.0 if bucket == "complete_correct" else 0.0,
        "image_content_sha256": sha256(image_bytes).hexdigest(),
        "local_image_path": f"/old/{sample_id}{suffix}",
    }


def test_candidate_manifest_uses_all_rows_and_physical_content_groups(tmp_path):
    image_root = tmp_path / "images"
    rows = [
        _portable_row(
            image_root,
            uid="gqa:a",
            benchmark="gqa",
            bucket="complete_correct",
            sample_id="a",
            image_bytes=b"shared",
        ),
        _portable_row(
            image_root,
            uid="gqa:b",
            benchmark="gqa",
            bucket="complete_wrong",
            sample_id="b",
            image_bytes=b"shared",
        ),
    ]

    manifest, audit = build_candidate_manifest(
        rows,
        image_root=image_root,
        expected_counts={"gqa": 2},
    )

    assert len(manifest) == 2
    assert manifest[0]["historical_bucket"] == "correct"
    assert manifest[1]["historical_bucket"] == "wrong"
    assert manifest[0]["image_group_id"] == manifest[1]["image_group_id"]
    assert manifest[0]["max_image_tokens"] is None
    assert audit["unique_uids"] == 2
    assert audit["unique_image_groups"] == 1


def test_candidate_manifest_fails_closed_on_a_content_mismatch(tmp_path):
    image_root = tmp_path / "images"
    row = _portable_row(
        image_root,
        uid="chartqa:a",
        benchmark="chartqa",
        bucket="complete_correct",
        sample_id="a",
        image_bytes=b"current",
    )
    row["image_content_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="content SHA-256"):
        build_candidate_manifest(
            [row],
            image_root=image_root,
            expected_counts={"chartqa": 1},
        )


def test_candidate_manifest_requires_a_frozen_source_content_hash(tmp_path):
    image_root = tmp_path / "images"
    row = _portable_row(
        image_root,
        uid="gqa:a",
        benchmark="gqa",
        bucket="complete_correct",
        sample_id="a",
        image_bytes=b"current",
    )
    row["image_content_sha256"] = None

    with pytest.raises(ValueError, match="frozen image content SHA-256"):
        build_candidate_manifest(
            [row],
            image_root=image_root,
            expected_counts={"gqa": 1},
        )


def test_candidate_evaluator_and_generation_contract_rejects_fallbacks():
    base = {
        "uid": "gqa:a",
        "dataset": "gqa",
        "metric_name": "exact_match_ignore_case_punctuation",
        "correctness_threshold": 1.0,
        "max_new_tokens": 16,
    }
    evaluators = {
        "gqa": {
            "metric_name": "exact_match_ignore_case_punctuation",
            "correctness_threshold": 1.0,
            "normalization": "gqa_exact_v1",
        }
    }
    validate_candidate_execution_contract(
        [base],
        evaluator_specs=evaluators,
        max_new_tokens=16,
    )
    with pytest.raises(ValueError, match="evaluator contract"):
        validate_candidate_execution_contract(
            [{**base, "metric_name": "unknown_fallback"}],
            evaluator_specs=evaluators,
            max_new_tokens=16,
        )
    with pytest.raises(ValueError, match="max_new_tokens"):
        validate_candidate_execution_contract(
            [{**base, "max_new_tokens": 64}],
            evaluator_specs=evaluators,
            max_new_tokens=16,
        )


def test_smoke_is_deterministic_and_stratified_by_dataset_and_bucket():
    rows = []
    for dataset in ("gqa", "chartqa", "textvqa"):
        for bucket in ("correct", "wrong"):
            for index in range(6):
                rows.append(
                    {
                        "uid": f"{dataset}:{bucket}:{index}",
                        "dataset": dataset,
                        "historical_bucket": bucket,
                        "image_group_id": f"sha256:{dataset}:{bucket}:{index}",
                    }
                )

    first = select_dense_smoke(rows, seed=19, per_cell=4)
    second = select_dense_smoke(rows, seed=19, per_cell=4)

    assert first == second
    assert len(first) == 24
    assert len({row["image_group_id"] for row in first}) == 24
    for dataset in ("gqa", "chartqa", "textvqa"):
        for bucket in ("correct", "wrong"):
            assert sum(
                row["dataset"] == dataset and row["historical_bucket"] == bucket
                for row in first
            ) == 4


def test_token_selector_uses_user_span_and_visual_pad_tokens():
    # <vision_end>=8, <im_end>=9, image token=7. The final user token is 13.
    ids = [1, 7, 7, 8, 10, 11, 12, 13, 9, 2]
    positions = locate_user_and_visual_tokens(
        ids,
        image_token_id=7,
        vision_end_token_id=8,
        im_end_token_id=9,
    )

    assert positions.visual == (1, 2)
    assert positions.user_text == (4, 5, 6, 7)
    assert positions.final_user_token == 7


def test_group_disjoint_split_has_no_uid_or_group_overlap_and_is_repeatable():
    rows = []
    for index in range(100):
        group = f"group:{index // 2}" if index < 10 else f"group:{index}"
        rows.append(
            {
                "uid": f"gqa:{index}",
                "dataset": "gqa" if index < 60 else "chartqa",
                "image_group_id": group,
                "current_dense_correct": index % 3 != 0,
                "current_dense_wrong": index % 3 == 0,
            }
        )

    first = build_image_group_disjoint_split(rows, targets={"train": 80, "val": 10, "test": 10}, seed=5)
    second = build_image_group_disjoint_split(rows, targets={"train": 80, "val": 10, "test": 10}, seed=5)

    assert first == second
    from collections import Counter

    assert Counter(row["split"] for row in first) == {
        "train": 80,
        "val": 10,
        "test": 10,
    }
    uid_sets = {
        split: {row["uid"] for row in first if row["split"] == split}
        for split in ("train", "val", "test")
    }
    group_sets = {
        split: {row["image_group_id"] for row in first if row["split"] == split}
        for split in ("train", "val", "test")
    }
    assert not (uid_sets["train"] & uid_sets["val"])
    assert not (uid_sets["train"] & uid_sets["test"])
    assert not (uid_sets["val"] & uid_sets["test"])
    assert not (group_sets["train"] & group_sets["val"])
    assert not (group_sets["train"] & group_sets["test"])
    assert not (group_sets["val"] & group_sets["test"])


def test_contract_payload_hash_is_canonical_and_excludes_its_own_id():
    left = {"b": 2, "a": 1, "contract_sha256": "old"}
    right = {"a": 1, "b": 2, "contract_sha256": "different"}

    assert canonical_payload_sha256(left) == canonical_payload_sha256(right)


def test_contract_integrity_aborts_on_any_actual_runtime_difference():
    contract = {
        "contract_sha256": "contract",
        "integrity": {
            "model": {"revision": "rev", "resolved_weight_path": "/model", "file_sha256": {"w": "a"}},
            "git": {"commit": "abc", "status_porcelain": []},
            "environment": {"torch": "2.6.0"},
            "candidate_manifest_sha256": "manifest",
            "generation": {"max_new_tokens": 16},
        },
    }
    assert_contract_integrity(contract, json.loads(json.dumps(contract["integrity"])))
    changed = json.loads(json.dumps(contract["integrity"]))
    changed["model"]["file_sha256"]["w"] = "different"
    with pytest.raises(ValueError, match="integrity mismatch"):
        assert_contract_integrity(contract, changed)


def test_smoke_gate_is_required_and_feature_parity_cannot_be_bypassed():
    gate = {
        "schema_version": "current_dense_smoke_gate_v1",
        "contract_id": "contract",
        "candidate_manifest_sha256": "manifest",
        "repeatability_pass": True,
        "evaluator_parity_pass": True,
        "image_sha_pass": True,
        "contract_integrity_pass": True,
        "global_completeness_pass": True,
        "resume_validation_pass": True,
        "feature_provenance_pass": True,
        "hook_token_parity_pass": True,
        "records": 24,
        "duplicate_uids": 0,
        "missing_uids": 0,
    }
    validate_smoke_gate(
        gate,
        contract_id="contract",
        candidate_manifest_sha256="manifest",
        require_features=True,
    )
    with pytest.raises(ValueError, match="hook token parity"):
        validate_smoke_gate(
            {**gate, "hook_token_parity_pass": False},
            contract_id="contract",
            candidate_manifest_sha256="manifest",
            require_features=True,
        )


def test_normal_image_bytes_are_rehashed_immediately_before_open(tmp_path):
    from hashlib import sha256
    from PIL import Image

    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), color="red").save(path)
    prepared_hash = sha256(path.read_bytes()).hexdigest()
    Image.new("RGB", (4, 4), color="blue").save(path)
    with pytest.raises(ValueError, match="image SHA-256"):
        with _open_verified_image(path, prepared_hash):
            pass


def test_resume_rejects_generation_only_batch_when_features_are_required():
    marker = {
        "schema_version": "current_dense_batch_marker_v2",
        "contract_id": "contract",
        "run_id": "run",
        "rank": 0,
        "world_size": 4,
        "feature_shard_size": 64,
        "completion_state": "completed_without_features",
        "extract_features": False,
        "uids": ["gqa:a"],
    }
    with pytest.raises(ValueError, match="feature completion mode"):
        validate_resume_marker(
            marker,
            contract_id="contract",
            run_id="run",
            rank=0,
            world_size=4,
            feature_shard_size=64,
            expected_uids=["gqa:a"],
            require_features=True,
        )


def test_feature_payload_provenance_rejects_incompatible_run():
    expected = {
        "contract_id": "contract",
        "model_revision": "rev",
        "git_commit": "commit",
        "feature_schema_sha256": "schema",
        "source_manifest_sha256": "manifest",
        "run_id": "run-a",
        "generation_sha256": "generation",
    }
    payload = {"provenance": dict(expected)}
    validate_feature_provenance(payload, expected)
    payload["provenance"]["run_id"] = "run-b"
    with pytest.raises(ValueError, match="feature provenance"):
        validate_feature_provenance(payload, expected)


def test_global_completion_rejects_missing_rank_uid_and_duplicates():
    expected = ["a", "b", "c", "d"]
    rows = [{"uid": uid} for uid in expected]
    markers = [
        {"rank": rank, "world_size": 4, "completion_state": "completed_with_features"}
        for rank in range(4)
    ]
    validate_global_completion(
        expected,
        rows,
        markers,
        expected_world_size=4,
        require_features=True,
    )
    with pytest.raises(ValueError, match="rank coverage"):
        validate_global_completion(
            expected,
            rows,
            markers[:3],
            expected_world_size=4,
            require_features=True,
        )
    with pytest.raises(ValueError, match="UID coverage"):
        validate_global_completion(
            expected,
            rows[:-1],
            markers,
            expected_world_size=4,
            require_features=True,
        )
    with pytest.raises(ValueError, match="duplicate"):
        validate_global_completion(
            expected,
            rows + [{"uid": "a"}],
            markers,
            expected_world_size=4,
            require_features=True,
        )


def test_feature_collector_pools_each_layer_without_changing_outputs():
    class AddOne(torch.nn.Module):
        def forward(self, hidden):
            return (hidden + 1, "unchanged metadata")

    layers = torch.nn.ModuleList([AddOne(), AddOne()])
    collector = DenseFeatureCollector(
        layers,
        visual_positions=(1, 2),
        user_text_positions=(3, 4),
        final_user_token_position=4,
    )
    hidden = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
    with collector:
        first = layers[0](hidden)
        second = layers[1](first[0])

    assert torch.equal(first[0], hidden + 1)
    assert first[1] == "unchanged metadata"
    assert torch.equal(second[0], hidden + 2)
    features = collector.stacked()
    assert features["text_final"].shape == (2, 3)
    assert features["text_mean"].shape == (2, 3)
    assert features["visual_mean"].shape == (2, 3)
    assert torch.equal(features["text_final"][0], hidden[0, 4] + 1)
    assert torch.equal(
        features["visual_mean"][1],
        (hidden[0, 1] + hidden[0, 2]) / 2 + 2,
    )
