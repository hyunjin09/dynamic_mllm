from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import aggregate_predictability_stepD_external_transfer as aggregate_stepd
from experiments import run_predictability_stepD_external_transfer as run_stepd

from dense_failure_stage2.predictability_external_transfer import (
    EXTERNAL_BENCHMARK_COUNTS,
    canonical_question_text,
    external_answer_specs,
    freeze_prediction_hashes,
    stage1_transfer_nuisance,
    stage2_transfer_nuisance,
    strict_first_trigger,
    validate_external_population,
    validate_prediction_census,
)


def _row(uid: str, benchmark: str) -> dict:
    family = (
        "mmmu_pro" if benchmark.startswith("mmmu_pro_") else
        "pope" if benchmark.startswith("pope_") else benchmark
    )
    return {
        "uid": uid,
        "benchmark": benchmark,
        "benchmark_family": family,
        "question": "how many objects?",
        "image_group_id": f"image:{uid}",
        "image_content_sha256s": ["a" * 64],
        "prompt_token_count": 21,
        "user_text_token_count": 7,
        "visual_token_count": 13,
    }


def test_strict_first_trigger_uses_strict_greater() -> None:
    assert strict_first_trigger([0.1, 0.5, 0.5001] + [0.0] * 25, 0.5) == 2
    assert strict_first_trigger([0.5, 0.4] + [0.0] * 26, 0.5) is None


def test_external_answer_specs_freeze_benchmark_adapters() -> None:
    assert external_answer_specs({"benchmark": "chartqa", "answer": " 12.5 "}) == [
        {"text": "12.5", "weight": 1.0}
    ]
    text = external_answer_specs(
        {
            "benchmark": "textvqa",
            "answer": "Coca-Cola",
            "all_answer_norms": ["coca cola", "coca cola", "Coke"],
        }
    )
    assert text == [
        {"text": "coca cola", "weight": pytest.approx(2 / 3)},
        {"text": "coke", "weight": pytest.approx(1 / 3)},
    ]
    assert external_answer_specs(
        {"benchmark": "mmmu_pro_standard_test", "answer": "C"}
    ) == [{"text": "C", "weight": 1.0}]
    assert external_answer_specs(
        {"benchmark": "pope_random", "answer": "YES"}
    ) == [{"text": "yes", "weight": 1.0}]
    with pytest.raises(ValueError):
        external_answer_specs({"benchmark": "mmstar", "answer": "A"})


def test_canonical_question_text_uses_phase69_prompt_fallback() -> None:
    assert canonical_question_text({"uid": "a", "question": "  what? ", "prompt": "other"}) == (
        "what?",
        "question",
    )
    assert canonical_question_text({"uid": "b", "prompt": "  visible MMMU prompt "}) == (
        "visible MMMU prompt",
        "prompt",
    )
    with pytest.raises(ValueError):
        canonical_question_text({"uid": "c"})


def test_transfer_nuisance_excludes_dataset_and_source() -> None:
    row = {
        "layer": 4,
        "question": "abc def",
        "visual_token_count": 99,
        "user_text_token_count": 8,
        "prompt_token_count": 120,
        "dataset": "chartqa",
        "source_regime": "historical",
    }
    left = stage1_transfer_nuisance([row])
    row["dataset"] = "unseen"
    row["source_regime"] = "other"
    right = stage1_transfer_nuisance([row])
    assert left.shape == (1, 32)
    np.testing.assert_array_equal(left, right)

    state = {
        **row,
        "trigger_layer": 3,
        "trigger_relative_depth": 1,
        "visual_tokens": 99,
        "text_tokens": 8,
    }
    left2 = stage2_transfer_nuisance([state])
    state["dataset"] = "pope"
    right2 = stage2_transfer_nuisance([state])
    assert left2.shape == (1, 88)
    np.testing.assert_array_equal(left2, right2)


def test_validate_prediction_census_rejects_missing_duplicate_and_bad_contract() -> None:
    expected = ["a", "b"]
    rows = [
        {"state_id": "a", "contract_sha256": "c", "prediction": 0.1},
        {"state_id": "b", "contract_sha256": "c", "prediction": 0.2},
    ]
    validate_prediction_census(expected, rows, contract_sha256="c")
    with pytest.raises(ValueError):
        validate_prediction_census(expected, rows[:1], contract_sha256="c")
    with pytest.raises(ValueError):
        validate_prediction_census(expected, rows + rows[:1], contract_sha256="c")
    bad = [dict(rows[0], contract_sha256="x"), rows[1]]
    with pytest.raises(ValueError):
        validate_prediction_census(expected, bad, contract_sha256="c")

    stage2_rows = [
        {
            "state_id": state_id,
            "contract_sha256": "c",
            "prediction_read_m3_ensemble": 0.1,
            "prediction_write_m3_ensemble": -0.2,
            "read_m3_ensemble": 0.1,
            "write_m3_ensemble": -0.2,
        }
        for state_id in expected
    ]
    validate_prediction_census(expected, stage2_rows, contract_sha256="c")


def test_prediction_hashes_are_bound_and_verified(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"state_id": "a"}) + "\n")
    second.write_text(json.dumps({"state_id": "b"}) + "\n")
    manifest = freeze_prediction_hashes(
        {"first": first, "second": second},
        contract_sha256="contract",
        model_manifest_sha256="models",
    )
    assert manifest["contract_sha256"] == "contract"
    assert set(manifest["files"]) == {"first", "second"}
    first.write_text("changed\n")
    with pytest.raises(ValueError):
        freeze_prediction_hashes(
            {"first": first, "second": second},
            contract_sha256="contract",
            model_manifest_sha256="models",
            expected=manifest,
        )


@pytest.mark.parametrize("writer", [run_stepd.atomic_csv, aggregate_stepd.write_csv])
def test_csv_writers_preserve_heterogeneous_metric_columns(tmp_path: Path, writer) -> None:
    path = tmp_path / "metrics.csv"
    writer(path, [{"task": "classification", "auroc": 0.8}, {"task": "regression", "rmse": 0.2}])
    header, first, second = path.read_text().splitlines()
    assert header == "task,auroc,rmse"
    assert first == "classification,0.8,"
    assert second == "regression,,0.2"


def test_validate_external_population_exact_contract() -> None:
    rows = []
    for benchmark, count in EXTERNAL_BENCHMARK_COUNTS.items():
        rows.extend(_row(f"{benchmark}:{index}", benchmark) for index in range(count))
    assert validate_external_population(rows)["total"] == 19_960
    with pytest.raises(ValueError):
        validate_external_population(rows[:-1])
