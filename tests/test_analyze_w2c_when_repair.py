from pathlib import Path

from experiments.analyze_w2c_when_repair import (
    audit_smoke_records,
    compare_resume_snapshot,
    record_file_sha256s,
)
from four_action_policy.when_repair import repair_w2c_sample


def _iterative_smoke_fixture():
    original = ["IGNORE", *("FULL" for _ in range(27))]
    source = {
        "uid": "gqa:fixture",
        "sample_id": "fixture",
        "split": "validation",
        "dataset": "gqa",
        "route_type": "W2C",
        "valid_routes": [
            {
                "actions": original,
                "route_key": "|".join(original),
                "source_binary_route_ids": ["source:fixture"],
            }
        ],
    }

    def evaluate(actions):
        # The first round is rescued only by a one-edit route at layer 2. The
        # second round is exhausted, giving a final candidate at layer 2.
        correct = actions[:3] == ("FULL", "FULL", "READ_ONLY")
        return {
            "correct": correct,
            "prediction": "yes" if correct else "no",
            "score": float(correct),
            "generated_ids": [1 if correct else 0],
            "execution_source": "fixture",
            "prompt_sha256": "fixture-prompt",
        }

    repair = repair_w2c_sample(source, evaluate, search_budget=96, seed=20260830)
    record = {
        "schema_version": "w2c_when_repair_sample_v1",
        "config_sha256": "fixture-config",
        "mode": "smoke",
        "uid": source["uid"],
        "split": source["split"],
        "dataset": source["dataset"],
        "status": "completed",
        "old_route_replays": [
            {
                "route_index": 0,
                "route_key": source["valid_routes"][0]["route_key"],
                "correct": True,
            }
        ],
        "repair": repair,
    }
    manifest = {
        "uid": source["uid"],
        "dataset": source["dataset"],
        "split": source["split"],
        "valid_route_count": 1,
    }
    return source, record, manifest


def test_smoke_audit_reconstructs_iterative_repair_contract():
    source, record, manifest = _iterative_smoke_fixture()
    audit = audit_smoke_records(
        [record],
        source_by_uid={source["uid"]: source},
        manifest_by_uid={manifest["uid"]: manifest},
        search_budget=96,
        seed=20260830,
    )

    assert audit["all_passed"] is True
    assert audit["checks"]["candidate_boundary_moves_after_rescue"]["observed"] is True
    assert audit["checks"]["iterative_re_evaluation"]["observed"] is True


def test_resume_snapshot_detects_byte_changes(tmp_path: Path):
    record = tmp_path / "record.json"
    record.write_text('{"value":1}\n', encoding="utf-8")
    baseline = record_file_sha256s(tmp_path)
    assert compare_resume_snapshot(baseline, record_file_sha256s(tmp_path))["passed"]

    record.write_text('{"value":2}\n', encoding="utf-8")
    comparison = compare_resume_snapshot(baseline, record_file_sha256s(tmp_path))
    assert comparison["passed"] is False
    assert comparison["changed"] == ["record.json"]
