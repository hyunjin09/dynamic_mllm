"""Pure contracts for predictability Step-D external transfer.

The helpers in this module deliberately contain no model execution.  They keep
the external-label firewall, transfer-safe nuisance features, benchmark q
adapters, and census/hash validation testable without a GPU.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scoring.benchmark_metrics import normalize_textvqa


EXTERNAL_BENCHMARK_COUNTS = {
    "chartqa": 2_500,
    "textvqa": 5_000,
    "mmmu_pro_standard_test": 1_730,
    "mmmu_pro_vision_test": 1_730,
    "pope_adversarial": 3_000,
    "pope_popular": 3_000,
    "pope_random": 3_000,
}


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_family(benchmark: str) -> str:
    value = str(benchmark).lower()
    if value in {"chartqa", "textvqa"}:
        return value
    if value in {"mmmu_pro_standard_test", "mmmu_pro_vision_test"}:
        return "mmmu_pro"
    if value in {"pope_adversarial", "pope_popular", "pope_random"}:
        return "pope"
    raise ValueError(f"benchmark is outside Step-D scope: {benchmark}")


def canonical_question_text(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return the text available to the frozen question-semantic control.

    Phase-69 MMMU-Pro manifests represent their model-visible question as
    ``prompt``/``instruction_text_chunks`` rather than a separate ``question``
    field.  Using ``prompt`` for those rows preserves the text actually shown
    to the model and does not alter dense execution.
    """

    for field in ("question", "prompt"):
        value = str(row.get(field, "")).strip()
        if value:
            return value, field
    raise ValueError(f"external row has no question/prompt text: {row.get('uid')}")


def validate_external_population(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["benchmark"]).lower() for row in rows)
    if counts != Counter(EXTERNAL_BENCHMARK_COUNTS):
        raise ValueError(f"Step-D benchmark population differs: {dict(counts)}")
    uids = [str(row["uid"]) for row in rows]
    if len(uids) != len(set(uids)):
        raise ValueError("Step-D benchmark population has duplicate UIDs")
    required = {"uid", "benchmark", "image_group_id", "image_content_sha256s"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"external row lacks {sorted(missing)}: {row.get('uid')}")
        canonical_question_text(row)
        if benchmark_family(str(row["benchmark"])) != str(row["benchmark_family"]):
            raise ValueError(f"benchmark family differs: {row['uid']}")
        hashes = list(row["image_content_sha256s"])
        if not hashes or any(len(str(value)) != 64 for value in hashes):
            raise ValueError(f"invalid image hash list: {row['uid']}")
    return {
        "total": len(rows),
        "counts": dict(sorted(counts.items())),
        "families": dict(sorted(Counter(str(row["benchmark_family"]) for row in rows).items())),
    }


def strict_first_trigger(scores: Sequence[float], threshold: float) -> int | None:
    if len(scores) != 28 or not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("Stage-1 trigger requires 28 finite scores")
    return next(
        (index for index, value in enumerate(scores) if float(value) > float(threshold)),
        None,
    )


def external_answer_specs(sample: Mapping[str, Any]) -> list[dict[str, float | str]]:
    """Return the pre-registered finite-string q references for one benchmark.

    Discrete correctness remains the task scorer's responsibility.  In
    particular, ChartQA q represents the literal annotation, not its relaxed
    numeric acceptance interval.
    """

    benchmark = str(sample["benchmark"]).lower()
    if benchmark == "chartqa":
        answer = str(sample["answer"]).strip()
        if not answer:
            raise ValueError("ChartQA q reference is empty")
        return [{"text": answer, "weight": 1.0}]
    if benchmark == "textvqa":
        raw = list(sample.get("all_answer_norms") or [sample["answer"]])
        normalized = [normalize_textvqa(str(answer)) for answer in raw]
        counts = Counter(answer for answer in normalized if answer)
        if not counts:
            raise ValueError("TextVQA q references are empty")
        total = sum(counts.values())
        return [
            {"text": answer, "weight": count / total}
            for answer, count in sorted(counts.items())
        ]
    if benchmark in {"mmmu_pro_standard_test", "mmmu_pro_vision_test"}:
        answer = str(sample["answer"]).strip().upper()
        if answer not in set("ABCDEFGHIJ"):
            raise ValueError(f"MMMU-Pro q reference is not one A-J label: {answer!r}")
        return [{"text": answer, "weight": 1.0}]
    if benchmark in {"pope_adversarial", "pope_popular", "pope_random"}:
        answer = str(sample["answer"]).strip().lower()
        if answer not in {"yes", "no"}:
            raise ValueError(f"POPE q reference is not yes/no: {answer!r}")
        return [{"text": answer, "weight": 1.0}]
    raise ValueError(f"unsupported Step-D q adapter: {benchmark}")


def _one_hot(value: int, width: int, name: str) -> list[float]:
    if not 0 <= int(value) < int(width):
        raise ValueError(f"{name} outside [0,{width}): {value}")
    return [float(index == int(value)) for index in range(int(width))]


def stage1_transfer_nuisance(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Transfer-safe Stage-1 M0-X: layer plus universally available structure."""

    output = []
    for row in rows:
        values = _one_hot(int(row["layer"]), 28, "layer")
        values.extend(
            [
                math.log1p(float(row["visual_token_count"])),
                math.log1p(float(row["user_text_token_count"])),
                math.log1p(float(row["prompt_token_count"])),
                math.log1p(len(str(row["question"]))),
            ]
        )
        output.append(values)
    result = np.asarray(output, dtype=np.float32)
    if result.shape != (len(rows), 32) or not np.isfinite(result).all():
        raise ValueError("invalid Stage-1 transfer nuisance matrix")
    return result


def stage2_transfer_nuisance(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Transfer-safe Stage-2 M0-X without source/dataset/outcome identity."""

    output = []
    for row in rows:
        relative = int(row["trigger_relative_depth"])
        values = _one_hot(int(row["layer"]), 28, "layer")
        values += _one_hot(int(row["trigger_layer"]), 28, "trigger_layer")
        values += _one_hot(min(relative, 27), 28, "trigger_relative_depth")
        values.extend(
            [
                math.log1p(float(row["visual_tokens"])),
                math.log1p(float(row["text_tokens"])),
                math.log1p(float(row["user_text_token_count"])),
                math.log1p(float(row["prompt_token_count"])),
            ]
        )
        output.append(values)
    result = np.asarray(output, dtype=np.float32)
    if result.shape != (len(rows), 88) or not np.isfinite(result).all():
        raise ValueError("invalid Stage-2 transfer nuisance matrix")
    return result


def validate_prediction_census(
    expected_state_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    contract_sha256: str,
) -> None:
    expected = [str(value) for value in expected_state_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected prediction census contains duplicates")
    observed = [str(row["state_id"]) for row in rows]
    duplicates = sorted(value for value, count in Counter(observed).items() if count > 1)
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if duplicates or missing or extra:
        raise ValueError(
            f"prediction census differs: duplicates={duplicates[:3]} "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    for row in rows:
        if str(row.get("contract_sha256")) != str(contract_sha256):
            raise ValueError(f"prediction contract differs: {row['state_id']}")
        predictions = [
            float(value) for key, value in row.items()
            if key == "prediction" or key.startswith("prediction_")
        ]
        if not predictions or not all(math.isfinite(value) for value in predictions):
            raise ValueError(f"prediction is missing/non-finite: {row['state_id']}")


def freeze_prediction_hashes(
    files: Mapping[str, str | Path],
    *,
    contract_sha256: str,
    model_manifest_sha256: str,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "predictability_stepD_prediction_hashes_v1",
        "contract_sha256": str(contract_sha256),
        "model_manifest_sha256": str(model_manifest_sha256),
        "files": {
            str(name): {
                "path": str(Path(path)),
                "bytes": Path(path).stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in sorted(files.items())
        },
    }
    if expected is not None and json.dumps(payload, sort_keys=True) != json.dumps(
        dict(expected), sort_keys=True
    ):
        raise ValueError("frozen prediction hashes differ")
    return payload
