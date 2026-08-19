"""Utilities for the minimum counterfactual evidence source inventory."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BENCHMARKS = ("gqa", "textvqa", "chartqa", "docvqa")
DEFAULT_SAMPLING_SEED = 20260619

FORBIDDEN_SAMPLING_FIELD_PARTS = (
    "oracle",
    "gain",
    "delta",
    "target_score",
    "candidate_score",
    "routed_score",
    "routed_answer",
    "visual_on_mask",
    "old_gt_score",
    "all_text_only_score",
    "all_visual_on_score",
    "search",
)

SAMPLING_FIELDS_USED = (
    "sample_id",
    "dataset",
    "source_pool",
    "baseline_score",
    "source_asset_id",
    "visual_token_count",
    "question_length",
    "answer_length",
    "question_subtype",
)

MANIFEST_FIELDS = (
    "sample_id",
    "dataset",
    "cohort",
    "source_asset_id",
    "baseline_score",
    "baseline_correct",
    "visual_token_count",
    "question_length",
    "answer_length",
    "question_subtype",
    "source_pool",
    "source_path",
    "image_path",
    "metadata_status",
    "sampling_seed",
    "oracle_gain_used_for_sampling",
)

WRONG_SOURCE_PRIORITY = {
    "failure_tcd_lmms": 0,
    "failure_no_old_gt": 1,
    "primary_5k": 2,
    "primary_5k_router": 2,
}

CORRECT_SOURCE_PRIORITY = {
    "primary_5k": 0,
    "primary_5k_router": 0,
    "failure_tcd_lmms": 1,
    "failure_no_old_gt": 2,
}


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path | str) -> Iterable[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def canonical_manifest_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row.get(key) for key in MANIFEST_FIELDS} for row in rows]


def canonical_jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_index(path: Path | str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        sample_id = row.get("sample_id")
        if sample_id is not None:
            index[str(sample_id)] = row
    return index


def assert_no_oracle_gain_sampling_fields(field_names: Iterable[str]) -> None:
    disallowed = []
    for field_name in field_names:
        lowered = field_name.lower()
        if any(part in lowered for part in FORBIDDEN_SAMPLING_FIELD_PARTS):
            disallowed.append(field_name)
    if disallowed:
        raise ValueError(f"oracle/search-derived fields are not allowed for MCE sampling: {disallowed}")


def benchmark_from_row(row: dict[str, Any]) -> str:
    value = row.get("benchmark") or row.get("dataset")
    if value:
        return str(value).lower()
    sample_id = str(row.get("sample_id", ""))
    return sample_id.split("_", 1)[0].lower() if "_" in sample_id else "unknown"


def baseline_score_from_row(row: dict[str, Any]) -> float | None:
    for key in ("full_score", "score"):
        if row.get(key) is not None:
            return float(row[key])
    full_qwen = row.get("full_qwen")
    if isinstance(full_qwen, dict) and full_qwen.get("score") is not None:
        return float(full_qwen["score"])
    return None


def visual_token_count_from_row(row: dict[str, Any]) -> int | None:
    for key in ("num_visual_tokens", "visual_token_count"):
        if row.get(key) is not None:
            return int(row[key])
    all_visual_on = row.get("binary_all_visual_on")
    if isinstance(all_visual_on, dict) and all_visual_on.get("visual_tokens") is not None:
        return int(all_visual_on["visual_tokens"])
    return None


def question_text_from_row(row: dict[str, Any]) -> str:
    return str(row.get("question") or row.get("prompt") or "")


def answer_text_from_row(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, list):
        return " ".join(str(item) for item in answer)
    if answer is not None:
        return str(answer)
    answers = row.get("answers")
    if isinstance(answers, list):
        return " ".join(str(item) for item in answers)
    return ""


def token_count(text: str) -> int:
    return len(text.split()) if text else 0


def _asset_from_manifest(benchmark: str, manifest_row: dict[str, Any] | None) -> tuple[str | None, str]:
    if not manifest_row:
        return None, "missing_manifest_row"
    metadata = manifest_row.get("metadata") if isinstance(manifest_row.get("metadata"), dict) else {}

    if benchmark == "gqa" and metadata.get("image_id") is not None:
        return f"gqa:{metadata['image_id']}", "manifest.metadata.image_id"
    if benchmark == "textvqa" and metadata.get("image_id") is not None:
        return f"textvqa:{metadata['image_id']}", "manifest.metadata.image_id"
    if benchmark == "chartqa" and metadata.get("imgname") is not None:
        return f"chartqa:{metadata['imgname']}", "manifest.metadata.imgname"
    if benchmark == "docvqa" and metadata.get("image_filename") is not None:
        return f"docvqa:{metadata['image_filename']}", "manifest.metadata.image_filename"

    for key in ("source_image_path", "original_image_path", "image_path"):
        if manifest_row.get(key):
            return str(manifest_row[key]), f"manifest.{key}"
    return None, "manifest_no_asset_field"


def _subtype_from_manifest(benchmark: str, manifest_row: dict[str, Any] | None) -> str | None:
    if not manifest_row:
        return None
    metadata = manifest_row.get("metadata") if isinstance(manifest_row.get("metadata"), dict) else {}
    if benchmark == "gqa":
        types = metadata.get("types") if isinstance(metadata.get("types"), dict) else {}
        parts = [types.get("structural"), types.get("semantic"), types.get("detailed")]
        return "/".join(str(part) for part in parts if part) or None
    if benchmark == "chartqa" and metadata.get("type") is not None:
        return str(metadata["type"])
    if benchmark == "textvqa" and metadata.get("ocr_token_count") is not None:
        count = int(metadata["ocr_token_count"])
        if count == 0:
            bucket = "0"
        elif count <= 10:
            bucket = "1-10"
        elif count <= 30:
            bucket = "11-30"
        else:
            bucket = "31+"
        return f"ocr_tokens:{bucket}"
    if benchmark == "docvqa" and metadata.get("source") is not None:
        return str(metadata["source"])
    return None


def _fallback_asset_from_row(row: dict[str, Any]) -> tuple[str, str]:
    image_path = row.get("image") or row.get("image_path")
    if image_path:
        return Path(str(image_path)).name, "row.image_basename"
    return str(row.get("sample_id", "missing_sample_id")), "row.sample_id"


def normalize_mce_record(
    row: dict[str, Any],
    *,
    source_pool: str,
    source_path: str,
    manifest_row: dict[str, Any] | None = None,
    correct_threshold: float = 0.8,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or row.get("id") or "")
    if not sample_id:
        raise ValueError("row is missing sample_id/id")
    benchmark = benchmark_from_row(row)
    baseline_score = baseline_score_from_row(row)
    if baseline_score is None:
        raise ValueError(f"row {sample_id} is missing a full-Qwen baseline score")

    source_asset_id, metadata_status = _asset_from_manifest(benchmark, manifest_row)
    if source_asset_id is None:
        source_asset_id, metadata_status = _fallback_asset_from_row(row)

    image_path = str(row.get("image") or row.get("image_path") or "")
    question_text = question_text_from_row(row)
    answer_text = answer_text_from_row(row)

    return {
        "sample_id": sample_id,
        "dataset": benchmark,
        "source_pool": source_pool,
        "source_path": source_path,
        "image_path": image_path,
        "source_asset_id": source_asset_id,
        "metadata_status": metadata_status,
        "baseline_score": baseline_score,
        "baseline_correct": baseline_score >= correct_threshold,
        "visual_token_count": visual_token_count_from_row(row),
        "question_length": token_count(question_text),
        "answer_length": token_count(answer_text),
        "question_subtype": _subtype_from_manifest(benchmark, manifest_row),
        "oracle_gain_used_for_sampling": False,
    }


def normalize_rows(
    rows: Iterable[dict[str, Any]],
    *,
    source_pool: str,
    source_path: str,
    manifest_index: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    index = manifest_index or {}
    return [
        normalize_mce_record(
            row,
            source_pool=source_pool,
            source_path=source_path,
            manifest_row=index.get(str(row.get("sample_id") or row.get("id") or "")),
        )
        for row in rows
    ]


def _stable_float(seed: int, sample_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _dedupe_candidates(candidates: list[dict[str, Any]], *, cohort: str) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in candidates:
        sample_id = str(row["sample_id"])
        previous = best.get(sample_id)
        if previous is None:
            best[sample_id] = row
            continue
        if _source_priority(row, cohort=cohort) < _source_priority(previous, cohort=cohort):
            best[sample_id] = row
    return list(best.values())


def _source_priority(row: dict[str, Any], *, cohort: str) -> int:
    priorities = WRONG_SOURCE_PRIORITY if cohort == "wrong" else CORRECT_SOURCE_PRIORITY
    return priorities.get(str(row.get("source_pool")), 99)


def _candidate_sort_key(row: dict[str, Any], *, cohort: str, seed: int) -> tuple[Any, ...]:
    score = float(row["baseline_score"])
    score_key = score if cohort == "wrong" else -score
    return (
        _source_priority(row, cohort=cohort),
        score_key,
        int(row.get("visual_token_count") or 0),
        _stable_float(seed, str(row["sample_id"])),
        str(row["sample_id"]),
    )


def _strip_for_manifest(row: dict[str, Any], *, cohort: str, seed: int) -> dict[str, Any]:
    stripped = {key: row.get(key) for key in MANIFEST_FIELDS if key not in {"cohort", "sampling_seed"}}
    stripped["cohort"] = cohort
    stripped["sampling_seed"] = seed
    stripped["oracle_gain_used_for_sampling"] = False
    return {key: stripped.get(key) for key in MANIFEST_FIELDS}


def select_dry_run_cohort(
    records: Iterable[dict[str, Any]],
    *,
    per_dataset_per_cohort: int = 25,
    seed: int = DEFAULT_SAMPLING_SEED,
    wrong_threshold: float = 0.2,
    correct_threshold: float = 0.8,
    source_asset_limit: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert_no_oracle_gain_sampling_fields(SAMPLING_FIELDS_USED)
    rows = list(records)
    selected: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    asset_counts: dict[str, int] = defaultdict(int)

    for benchmark in BENCHMARKS:
        bench_rows = [row for row in rows if row.get("dataset") == benchmark]
        for cohort in ("wrong", "correct"):
            if cohort == "wrong":
                candidates = [row for row in bench_rows if float(row["baseline_score"]) <= wrong_threshold]
            else:
                candidates = [row for row in bench_rows if float(row["baseline_score"]) >= correct_threshold]
            candidates = _dedupe_candidates(candidates, cohort=cohort)
            candidates.sort(key=lambda row, c=cohort: _candidate_sort_key(row, cohort=c, seed=seed))

            cohort_rows: list[dict[str, Any]] = []
            for row in candidates:
                asset_id = str(row.get("source_asset_id") or row["sample_id"])
                if asset_counts[asset_id] >= source_asset_limit:
                    continue
                asset_counts[asset_id] += 1
                cohort_rows.append(_strip_for_manifest(row, cohort=cohort, seed=seed))
                if len(cohort_rows) >= per_dataset_per_cohort:
                    break

            selected.extend(cohort_rows)
            if len(cohort_rows) < per_dataset_per_cohort:
                shortfalls.append(
                    {
                        "dataset": benchmark,
                        "cohort": cohort,
                        "target": per_dataset_per_cohort,
                        "selected": len(cohort_rows),
                        "available_candidates": len(candidates),
                    }
                )

    selected.sort(key=lambda row: (row["dataset"], row["cohort"], row["sample_id"]))
    return selected, shortfalls


def validate_frozen_cohort(
    rows: Iterable[dict[str, Any]],
    *,
    per_dataset_per_cohort: int = 25,
    source_asset_limit: int = 2,
    wrong_threshold: float = 0.2,
    correct_threshold: float = 0.8,
) -> dict[str, Any]:
    rows = list(rows)
    errors: list[str] = []
    row_count_target = len(BENCHMARKS) * 2 * per_dataset_per_cohort
    if len(rows) != row_count_target:
        errors.append(f"expected {row_count_target} rows, found {len(rows)}")

    counts_by_dataset: dict[str, Counter[str]] = {dataset: Counter() for dataset in BENCHMARKS}
    source_pool_counts = Counter()
    metadata_status_counts = Counter()
    asset_counts = Counter()
    seen_sample_ids: set[str] = set()
    unexpected_oracle_like_fields: set[str] = set()
    extra_fields: set[str] = set()
    required_fields = set(MANIFEST_FIELDS)

    for idx, row in enumerate(rows):
        row_label = str(row.get("sample_id") or f"row_{idx}")
        missing = required_fields - set(row)
        if missing:
            errors.append(f"{row_label} missing fields {sorted(missing)}")
        extras = set(row) - required_fields
        extra_fields.update(extras)
        for key in extras:
            lowered = key.lower()
            if any(part in lowered for part in FORBIDDEN_SAMPLING_FIELD_PARTS):
                unexpected_oracle_like_fields.add(key)

        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            errors.append(f"row {idx} is missing sample_id")
        elif sample_id in seen_sample_ids:
            errors.append(f"duplicate sample_id {sample_id}")
        seen_sample_ids.add(sample_id)

        dataset = str(row.get("dataset", "")).lower()
        cohort = str(row.get("cohort", "")).lower()
        if dataset not in BENCHMARKS:
            errors.append(f"{row_label} has invalid dataset {dataset!r}")
            continue
        if cohort not in {"wrong", "correct"}:
            errors.append(f"{row_label} has invalid cohort {cohort!r}")
            continue

        counts_by_dataset[dataset][cohort] += 1
        source_pool_counts[str(row.get("source_pool"))] += 1
        metadata_status_counts[str(row.get("metadata_status"))] += 1
        asset_counts[str(row.get("source_asset_id") or sample_id)] += 1

        if row.get("oracle_gain_used_for_sampling") is not False:
            errors.append(f"{row_label} has oracle_gain_used_for_sampling={row.get('oracle_gain_used_for_sampling')!r}")

        try:
            score = float(row.get("baseline_score"))
        except (TypeError, ValueError):
            errors.append(f"{row_label} has invalid baseline_score {row.get('baseline_score')!r}")
            continue

        if cohort == "wrong":
            if score > wrong_threshold:
                errors.append(f"{row_label} wrong cohort has baseline_score {score} > {wrong_threshold}")
            if row.get("baseline_correct") is not False:
                errors.append(f"{row_label} wrong cohort has baseline_correct={row.get('baseline_correct')!r}")
        else:
            if score < correct_threshold:
                errors.append(f"{row_label} correct cohort has baseline_score {score} < {correct_threshold}")
            if row.get("baseline_correct") is not True:
                errors.append(f"{row_label} correct cohort has baseline_correct={row.get('baseline_correct')!r}")

    for dataset in BENCHMARKS:
        for cohort in ("wrong", "correct"):
            count = counts_by_dataset[dataset][cohort]
            if count != per_dataset_per_cohort:
                errors.append(
                    f"{dataset}/{cohort} expected {per_dataset_per_cohort} rows, found {count}"
                )

    over_cap_assets = {asset: count for asset, count in asset_counts.items() if count > source_asset_limit}
    if over_cap_assets:
        errors.append(f"source assets exceed cap {source_asset_limit}: {dict(sorted(over_cap_assets.items()))}")
    if unexpected_oracle_like_fields:
        errors.append(f"unexpected oracle/search-derived manifest fields: {sorted(unexpected_oracle_like_fields)}")

    if errors:
        raise ValueError("; ".join(errors))

    return {
        "row_count": len(rows),
        "counts_by_dataset": {dataset: dict(counts_by_dataset[dataset]) for dataset in BENCHMARKS},
        "source_pool_counts": dict(source_pool_counts),
        "metadata_status_counts": dict(metadata_status_counts),
        "source_assets_unique": len(asset_counts),
        "source_assets_with_multiple_rows": sum(1 for count in asset_counts.values() if count > 1),
        "max_rows_per_source_asset": max(asset_counts.values()) if asset_counts else 0,
        "oracle_gain_used_for_sampling_count": sum(
            1 for row in rows if row.get("oracle_gain_used_for_sampling") is not False
        ),
        "extra_fields": sorted(extra_fields),
    }
