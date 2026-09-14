#!/usr/bin/env python3
"""Freeze and execute the Stage-2 canonical-source data-scale search."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote
from urllib.request import urlopen

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import pyarrow.parquet as pq
import torch

from dense_failure_stage2.data_scale_search import (
    build_search_work,
    build_new_trigger_map,
    chartqa_stratum,
    image_extension_for_format,
    most_common_answer,
    question_length_stratum,
    select_legacy_source_rows,
    select_metadata_stratified,
    validate_frozen_candidates,
)
from dense_failure_stage1.runtime import configure_dense_determinism
from dense_failure_stage2.corrective_search import ACTIONS, trigger_depth_bin
from dense_failure_stage2.full_label_generation import (
    assign_workers,
    verify_artifact_manifest,
)
from experiments.audit_stage1_trigger_map import _build_frozen_model, _score_feature_rows
from experiments.run_full_corrective_label_generation import _feature_schema
import experiments.run_full_corrective_label_generation as phase56_runner
import experiments.run_current_dense_8k as dense_runner
from experiments.run_trigger_conditioned_corrective_search_pilot import (
    _runtime_metadata,
    canonical_hash,
    command_output,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/stage2_data_scale_search_v1.json"
BOUND_CODE_PATHS = (
    "configs/stage2_data_scale_search_v1.json",
    "configs/stage2_data_scale_dense_v1.json",
    "dense_failure_stage2/data_scale_search.py",
    "dense_failure_stage2/full_label_generation.py",
    "dense_failure_stage2/corrective_search.py",
    "dense_failure_stage1/current_dense_8k.py",
    "dense_failure_stage1/lmms_scoring.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage1/shared_global_gate.py",
    "tools/research_analysis/dense_failure_stage1.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "experiments/run_current_dense_8k.py",
    "experiments/audit_stage1_trigger_map.py",
    "experiments/run_trigger_conditioned_corrective_search_pilot.py",
    "experiments/run_full_corrective_label_generation.py",
    "experiments/run_stage2_data_scale_search.py",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_bytes(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ).encode(),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_question(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "stage2_data_scale_search_config_v1":
        raise ValueError("unsupported Stage-2 data-scale config")
    if int(config["world_size"]) != 4:
        raise ValueError("Stage-2 data-scale search requires four workers")
    if config["candidate_targets"] != {"gqa": 2000, "chartqa": 1000, "textvqa": 1000}:
        raise ValueError("candidate quotas differ from the prospective 50/25/25 design")
    if int(config["mcts"]["maximum_iterations"]) != 200:
        raise ValueError("MCTS cap must remain exactly 200")
    if config.get("reporting_rules") != {
        "old_pattern_absolute_fixability_rate_tolerance": 0.10,
        "gqa_material_minimum_new_fixable_bases": 25,
    }:
        raise ValueError("prospective scale-up reporting rules differ")
    return config


def _chartqa_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = config["source_datasets"]["chartqa"]
    root = resolve_path(Path(config["source_root"]) / spec["path"])
    rows: list[dict[str, Any]] = []
    for source in ("human", "augmented"):
        path = root / f"train_{source}.json"
        for index, item in enumerate(read_json(path)):
            image_id = str(item["imgname"])
            question = str(item["query"])
            native_key = f"{source}:{index}:{image_id}"
            rows.append(
                {
                    "uid": f"chartqa:chartqa_scale_{sha256(native_key.encode()).hexdigest()[:20]}",
                    "dataset": "chartqa",
                    "native_row_key": native_key,
                    "native_image_id": image_id,
                    "image_group_id": f"source:chartqa:{image_id}",
                    "source_stratum": chartqa_stratum(source, question),
                    "source_annotation": source,
                    "source_row_index": index,
                    "question": question,
                    "answer": str(item["label"]),
                }
            )
    return rows


def preselect_chartqa(config_path: Path) -> None:
    config = _load_config(config_path)
    output_root = resolve_path(config["output_root"])
    source_rows = _chartqa_rows(config)
    legacy = [
        row
        for row in read_jsonl(resolve_path(config["sources"]["legacy_source_manifest"]))
        if str(row["benchmark"]) == "chartqa"
    ]
    lookup: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in source_rows:
        lookup[(str(row["native_image_id"]), _normalized_question(row["question"]))].append(
            str(row["source_stratum"])
        )
    reference = Counter()
    unmatched = []
    ambiguous = []
    legacy_images = set()
    for row in legacy:
        asset = str(row.get("source_asset_id") or "")
        image_id = asset.split(":", 1)[1] if asset.startswith("chartqa:") else ""
        legacy_images.add(image_id)
        values = sorted(set(lookup.get((image_id, _normalized_question(row["question"])), [])))
        if len(values) == 1:
            reference[values[0]] += 1
        elif values:
            reference[values[0]] += 1
            ambiguous.append(str(row["uid"]))
        else:
            unmatched.append(str(row["uid"]))
    eligible = [row for row in source_rows if row["native_image_id"] not in legacy_images]
    reserve_target = int(config["candidate_targets"]["chartqa"]) + 200
    selected, selection_audit = select_metadata_stratified(
        eligible,
        target=reserve_target,
        reference_strata=reference,
        seed=int(config["seed"]) + 101,
    )
    revision = str(config["source_datasets"]["chartqa"]["revision"])
    for row in selected:
        relative = f"ChartQA Dataset/train/png/{row['native_image_id']}"
        row["source_revision"] = revision
        row["source_relative_image"] = relative
        row["download_url"] = (
            f"https://raw.githubusercontent.com/vis-nlp/ChartQA/{revision}/"
            + quote(relative, safe="/")
        )
    path = output_root / "work/source_preselection/chartqa_download_manifest.jsonl"
    atomic_jsonl(path, selected)
    atomic_json(
        output_root / "work/source_preselection/chartqa_preselection_audit.json",
        {
            "schema_version": "stage2_scale_chartqa_preselection_v1",
            "source_records": len(source_rows),
            "eligible_records": len(eligible),
            "legacy_records": len(legacy),
            "legacy_reference_matched": sum(reference.values()),
            "legacy_reference_unmatched": len(unmatched),
            "legacy_reference_ambiguous": len(ambiguous),
            "reference_strata": dict(sorted(reference.items())),
            "selection": selection_audit,
            "manifest": str(path),
            "prepared_at": utc_now(),
        },
    )
    print(json.dumps({"selected_for_download": len(selected), "manifest": str(path)}))


def _download_image(row: Mapping[str, Any], root: Path) -> dict[str, Any]:
    target = root / str(row["source_relative_image"])
    if target.is_file():
        payload = target.read_bytes()
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        return {"uid": row["uid"], "path": str(target), "bytes": len(payload), "resumed": True}
    with urlopen(str(row["download_url"]), timeout=120) as response:
        payload = response.read()
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    _atomic_bytes(target, payload)
    return {"uid": row["uid"], "path": str(target), "bytes": len(payload), "resumed": False}


def download_chartqa(config_path: Path) -> None:
    config = _load_config(config_path)
    output_root = resolve_path(config["output_root"])
    rows = read_jsonl(output_root / "work/source_preselection/chartqa_download_manifest.jsonl")
    source_root = resolve_path(Path(config["source_root"]) / "chartqa")
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_download_image, row, source_root): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"uid": row["uid"], "error": str(exc)})
    if failures or len(results) != len(rows):
        atomic_jsonl(output_root / "work/source_preselection/chartqa_download_failures.jsonl", failures)
        raise RuntimeError(f"ChartQA image acquisition incomplete: {len(results)}/{len(rows)}")
    atomic_json(
        output_root / "work/source_preselection/chartqa_download_complete.json",
        {
            "passed": True,
            "records": len(results),
            "bytes": sum(int(row["bytes"]) for row in results),
            "resumed": sum(bool(row["resumed"]) for row in results),
            "completed_at": utc_now(),
        },
    )
    print(json.dumps({"passed": True, "downloaded_or_verified": len(results)}))


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_extension(payload: bytes) -> str:
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
        image_format = str(image.format or "").lower()
    return image_extension_for_format(image_format)


def _materialize_bytes(
    payload: bytes, *, dataset: str, image_id: str, root: Path
) -> tuple[Path, str]:
    digest = sha256(payload).hexdigest()
    extension = _image_extension(payload)
    safe_id = sha256(str(image_id).encode()).hexdigest()[:20]
    target = root / dataset / f"{safe_id}{extension}"
    if target.is_file():
        if file_sha256(target) != digest:
            raise RuntimeError(f"existing materialized image differs: {target}")
    else:
        _atomic_bytes(target, payload)
    return target, digest


def _legacy(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    portable = read_jsonl(resolve_path(config["sources"]["legacy_source_manifest"]))
    candidates = read_jsonl(resolve_path(config["sources"]["legacy_candidate_manifest"]))
    if len(candidates) != 8000:
        raise RuntimeError("legacy exclusion population must contain exactly 8,000 current identities")
    source = select_legacy_source_rows(portable, candidates)
    if len(source) != 8000:
        raise RuntimeError("portable source/current candidate join differs from 8,000")
    return source, candidates


def _candidate_row(
    row: Mapping[str, Any],
    *,
    local_image_path: Path,
    image_sha256: str,
    image_identifier: str,
) -> dict[str, Any]:
    dataset = str(row["dataset"])
    question = str(row["question"])
    return {
        "schema_version": "stage2_data_scale_candidate_v1",
        "uid": str(row["uid"]),
        "sample_id": str(row["uid"]).split(":", 1)[1],
        "dataset": dataset,
        "prompt": question + "\nAnswer the question using a single word or phrase.",
        "question": question,
        "answer": str(row["answer"]),
        "all_answer_norms": row.get("all_answer_norms"),
        "local_image_path": str(local_image_path),
        "image_content_sha256": image_sha256,
        "image_group_id": f"sha256:{image_sha256}",
        "image_identifier": image_identifier,
        "image_present_at_preparation": True,
        "max_new_tokens": 16,
        "historical_bucket": "new_canonical_unlabeled",
        "native_row_key": str(row["native_row_key"]),
        "native_image_id": str(row["native_image_id"]),
        "source_stratum": str(row["source_stratum"]),
        "source_revision": str(row["source_revision"]),
        "source_dataset": str(row["source_dataset"]),
        "source_split": "train",
        "source_metadata": dict(row.get("source_metadata", {})),
    }


def _prepare_gqa(
    config: Mapping[str, Any],
    legacy_source: Sequence[Mapping[str, Any]],
    legacy_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = config["source_datasets"]["gqa"]
    source_root = resolve_path(Path(config["source_root"]) / "gqa")
    image_rows: dict[str, dict[str, Any]] = {}
    excluded_hash = 0
    excluded_native = 0
    legacy_images = {
        str(row["source_asset_id"]).split(":", 1)[1]
        for row in legacy_source
        if str(row["benchmark"]) == "gqa" and str(row.get("source_asset_id", "")).startswith("gqa:")
    }
    for relative in spec["image_files"]:
        path = source_root / relative
        table = pq.read_table(path, columns=["id", "image"])
        for index, item in enumerate(table.to_pylist()):
            image_id = str(item["id"])
            payload = item["image"]["bytes"]
            digest = sha256(payload).hexdigest()
            if image_id in legacy_images:
                excluded_native += 1
                continue
            if digest in legacy_hashes:
                excluded_hash += 1
                continue
            if image_id in image_rows:
                raise RuntimeError(f"duplicate GQA image ID across frozen shards: {image_id}")
            image_rows[image_id] = {
                "bytes": payload,
                "sha256": digest,
                "source_image_file": relative,
                "source_image_row": index,
            }

    instruction_path = source_root / spec["instruction_files"][0]
    table = pq.read_table(
        instruction_path,
        columns=["id", "imageId", "question", "answer", "types"],
    )
    legacy_keys = {
        (
            str(row["source_asset_id"]).split(":", 1)[1],
            _normalized_question(row["question"]),
        )
        for row in legacy_source
        if str(row["benchmark"]) == "gqa" and str(row.get("source_asset_id", "")).startswith("gqa:")
    }
    reference = Counter()
    matched_legacy_keys = set()
    source_rows = []
    for index, item in enumerate(table.to_pylist()):
        image_id = str(item["imageId"])
        question = str(item["question"])
        stratum = str((item.get("types") or {}).get("detailed") or "unknown")
        key = (image_id, _normalized_question(question))
        if key in legacy_keys:
            reference[stratum] += 1
            matched_legacy_keys.add(key)
        if image_id not in image_rows:
            continue
        question_id = str(item["id"])
        source_rows.append(
            {
                "uid": f"gqa:gqa_scale_{question_id}",
                "dataset": "gqa",
                "native_row_key": f"train_balanced:{question_id}",
                "native_image_id": image_id,
                "image_group_id": f"sha256:{image_rows[image_id]['sha256']}",
                "source_stratum": stratum,
                "source_revision": spec["revision"],
                "source_dataset": spec["repo"],
                "source_row_index": index,
                "question": question,
                "answer": str(item["answer"]),
                "all_answer_norms": None,
                "source_metadata": {
                    "question_id": question_id,
                    "image_id": image_id,
                    "question_type_detailed": stratum,
                    "instruction_file": spec["instruction_files"][0],
                    "instruction_row": index,
                    "image_file": image_rows[image_id]["source_image_file"],
                    "image_row": image_rows[image_id]["source_image_row"],
                },
            }
        )
    selected, selection = select_metadata_stratified(
        source_rows,
        target=int(config["candidate_targets"]["gqa"]),
        reference_strata=reference,
        seed=int(config["seed"]) + 201,
    )
    materialized_root = resolve_path(config["materialized_image_root"])
    candidates = []
    for row in selected:
        image = image_rows[str(row["native_image_id"])]
        path, digest = _materialize_bytes(
            image["bytes"], dataset="gqa", image_id=str(row["native_image_id"]), root=materialized_root
        )
        if digest != image["sha256"]:
            raise RuntimeError("GQA image hash changed during materialization")
        candidates.append(
            _candidate_row(
                row,
                local_image_path=path,
                image_sha256=digest,
                image_identifier=f"gqa:{row['native_image_id']}",
            )
        )
    return candidates, {
        "source_image_rows": len(image_rows) + excluded_native + excluded_hash,
        "eligible_image_rows": len(image_rows),
        "eligible_question_rows": len(source_rows),
        "excluded_legacy_native_image": excluded_native,
        "excluded_legacy_content_hash": excluded_hash,
        "legacy_reference_keys": len(legacy_keys),
        "legacy_reference_matched": len(matched_legacy_keys),
        "reference_strata": dict(sorted(reference.items())),
        "selection": selection,
    }


def _prepare_chartqa(
    config: Mapping[str, Any],
    legacy_source: Sequence[Mapping[str, Any]],
    legacy_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_root = resolve_path(config["output_root"])
    source_root = resolve_path(Path(config["source_root"]) / "chartqa")
    rows = read_jsonl(output_root / "work/source_preselection/chartqa_download_manifest.jsonl")
    reference = Counter()
    lookup = {
        (str(row["native_image_id"]), _normalized_question(row["question"])): str(row["source_stratum"])
        for row in _chartqa_rows(config)
    }
    for row in legacy_source:
        if str(row["benchmark"]) != "chartqa":
            continue
        asset = str(row.get("source_asset_id") or "")
        image_id = asset.split(":", 1)[1] if asset.startswith("chartqa:") else ""
        value = lookup.get((image_id, _normalized_question(row["question"])))
        if value is not None:
            reference[value] += 1
    eligible = []
    excluded_hash = 0
    for row in rows:
        path = source_root / str(row["source_relative_image"])
        if not path.is_file():
            continue
        digest = file_sha256(path)
        if digest in legacy_hashes:
            excluded_hash += 1
            continue
        eligible.append(
            {
                **row,
                "image_group_id": f"sha256:{digest}",
                "source_dataset": config["source_datasets"]["chartqa"]["repo"],
                "all_answer_norms": None,
                "_local_image_path": str(path),
                "_image_sha256": digest,
                "source_metadata": {
                    "annotation_source": row["source_annotation"],
                    "annotation_row": row["source_row_index"],
                    "image_id": row["native_image_id"],
                },
            }
        )
    selected, selection = select_metadata_stratified(
        eligible,
        target=int(config["candidate_targets"]["chartqa"]),
        reference_strata=reference,
        seed=int(config["seed"]) + 202,
    )
    candidates = [
        _candidate_row(
            row,
            local_image_path=Path(row["_local_image_path"]),
            image_sha256=str(row["_image_sha256"]),
            image_identifier=f"chartqa:{row['native_image_id']}",
        )
        for row in selected
    ]
    return candidates, {
        "download_reserve_rows": len(rows),
        "eligible_rows": len(eligible),
        "excluded_legacy_content_hash": excluded_hash,
        "legacy_reference_matched": sum(reference.values()),
        "reference_strata": dict(sorted(reference.items())),
        "selection": selection,
    }


def _prepare_textvqa(
    config: Mapping[str, Any],
    legacy_source: Sequence[Mapping[str, Any]],
    legacy_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = config["source_datasets"]["textvqa"]
    source_root = resolve_path(Path(config["source_root"]) / "textvqa")
    legacy_images = {
        str(row["source_asset_id"]).split(":", 1)[1]
        for row in legacy_source
        if str(row["benchmark"]) == "textvqa" and str(row.get("source_asset_id", "")).startswith("textvqa:")
    }
    reference = Counter(
        question_length_stratum(str(row["question"]))
        for row in legacy_source
        if str(row["benchmark"]) == "textvqa"
    )
    source_rows = []
    image_payloads: dict[str, bytes] = {}
    excluded_native = 0
    excluded_hash = 0
    for relative in spec["files"]:
        path = source_root / relative
        table = pq.read_table(path)
        for index, item in enumerate(table.to_pylist()):
            image_id = str(item["image_id"])
            if image_id in legacy_images:
                excluded_native += 1
                continue
            payload = item["image"]["bytes"]
            digest = sha256(payload).hexdigest()
            if digest in legacy_hashes:
                excluded_hash += 1
                continue
            question_id = str(item["question_id"])
            question = str(item["question"])
            answers = [str(value) for value in item["answers"]]
            source_rows.append(
                {
                    "uid": f"textvqa:textvqa_scale_train_{question_id}",
                    "dataset": "textvqa",
                    "native_row_key": f"train:{question_id}",
                    "native_image_id": image_id,
                    "image_group_id": f"sha256:{digest}",
                    "source_stratum": question_length_stratum(question),
                    "source_revision": spec["revision"],
                    "source_dataset": spec["repo"],
                    "question": question,
                    "answer": most_common_answer(answers),
                    "all_answer_norms": answers,
                    "source_metadata": {
                        "question_id": int(item["question_id"]),
                        "image_id": image_id,
                        "source_file": relative,
                        "source_row": index,
                        "ocr_count": len(item.get("ocr_tokens") or []),
                    },
                }
            )
            existing = image_payloads.setdefault(image_id, payload)
            if existing != payload:
                raise RuntimeError(f"TextVQA image bytes differ within image ID: {image_id}")
    selected, selection = select_metadata_stratified(
        source_rows,
        target=int(config["candidate_targets"]["textvqa"]),
        reference_strata=reference,
        seed=int(config["seed"]) + 203,
    )
    materialized_root = resolve_path(config["materialized_image_root"])
    candidates = []
    for row in selected:
        payload = image_payloads[str(row["native_image_id"])]
        path, digest = _materialize_bytes(
            payload,
            dataset="textvqa",
            image_id=str(row["native_image_id"]),
            root=materialized_root,
        )
        if f"sha256:{digest}" != row["image_group_id"]:
            raise RuntimeError("TextVQA image hash changed during materialization")
        candidates.append(
            _candidate_row(
                row,
                local_image_path=path,
                image_sha256=digest,
                image_identifier=f"textvqa:{row['native_image_id']}",
            )
        )
    return candidates, {
        "source_rows": len(source_rows) + excluded_native + excluded_hash,
        "eligible_rows": len(source_rows),
        "eligible_image_groups": len(image_payloads),
        "excluded_legacy_native_image": excluded_native,
        "excluded_legacy_content_hash": excluded_hash,
        "reference_strata": dict(sorted(reference.items())),
        "selection": selection,
    }


def prepare_candidates(config_path: Path) -> None:
    config = _load_config(config_path)
    output_root = resolve_path(config["output_root"])
    candidate_path = output_root / "manifests/new_candidate_manifest.jsonl"
    if candidate_path.exists():
        raise RuntimeError(f"candidate manifest is already frozen: {candidate_path}")
    legacy_source, legacy_candidates = _legacy(config)
    legacy_hashes = {str(row["image_content_sha256"]) for row in legacy_candidates}
    gqa, gqa_audit = _prepare_gqa(config, legacy_source, legacy_hashes)
    chartqa, chartqa_audit = _prepare_chartqa(config, legacy_source, legacy_hashes)
    textvqa, textvqa_audit = _prepare_textvqa(config, legacy_source, legacy_hashes)
    candidates = [*gqa, *chartqa, *textvqa]
    candidates.sort(key=lambda row: ("gqa chartqa textvqa".split().index(row["dataset"]), row["uid"]))
    validation = validate_frozen_candidates(
        candidates,
        expected_counts=config["candidate_targets"],
        legacy_uids={str(row["uid"]) for row in legacy_candidates},
        legacy_groups={str(row["image_group_id"]) for row in legacy_candidates},
    )
    for row in candidates:
        path = Path(row["local_image_path"])
        if not path.is_file() or file_sha256(path) != row["image_content_sha256"]:
            raise RuntimeError(f"candidate image is absent or changed: {row['uid']}")
    atomic_jsonl(candidate_path, candidates)
    smoke = []
    for dataset in ("gqa", "chartqa", "textvqa"):
        values = sorted(
            (row for row in candidates if row["dataset"] == dataset),
            key=lambda row: sha256(f"scale-smoke:{config['seed']}:{row['uid']}".encode()).hexdigest(),
        )
        smoke.extend(values[:4])
    atomic_json(
        output_root / "smoke/dense_smoke_manifest.json",
        {"schema_version": "stage2_data_scale_dense_smoke_v1", "records": smoke},
    )
    audit = {
        "schema_version": "stage2_data_scale_candidate_audit_v1",
        "passed": True,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "validation": validation,
        "gqa": gqa_audit,
        "chartqa": chartqa_audit,
        "textvqa": textvqa_audit,
        "legacy_records_excluded": len(legacy_candidates),
        "smoke_records": len(smoke),
        "prepared_at": utc_now(),
    }
    atomic_json(output_root / "manifests/candidate_audit.json", audit)
    print(json.dumps({"passed": True, "candidates": len(candidates), "sha256": audit["candidate_manifest_sha256"]}))


def _source_files(config: Mapping[str, Any]) -> dict[str, Path]:
    source_root = resolve_path(config["source_root"])
    files: dict[str, Path] = {
        f"repo_source:{key}": resolve_path(value)
        for key, value in config["sources"].items()
    }
    for relative in config["source_datasets"]["gqa"]["instruction_files"]:
        files[f"gqa:{relative}"] = source_root / "gqa" / relative
    for relative in config["source_datasets"]["gqa"]["image_files"]:
        files[f"gqa:{relative}"] = source_root / "gqa" / relative
    chart_root = source_root / config["source_datasets"]["chartqa"]["path"]
    for name in config["source_datasets"]["chartqa"]["annotation_files"]:
        files[f"chartqa:{name}"] = chart_root / name
    for relative in config["source_datasets"]["textvqa"]["files"]:
        files[f"textvqa:{relative}"] = source_root / "textvqa" / relative
    output_root = resolve_path(config["output_root"])
    for relative in (
        "manifests/new_candidate_manifest.jsonl",
        "manifests/candidate_audit.json",
        "smoke/dense_smoke_manifest.json",
        "work/source_preselection/chartqa_download_manifest.jsonl",
        "work/source_preselection/chartqa_preselection_audit.json",
        "work/source_preselection/chartqa_download_complete.json",
    ):
        files[f"internal:{relative}"] = output_root / relative
    return files


def freeze_contract(config_path: Path) -> None:
    config = _load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract_path = output_root / "frozen_protocol.json"
    if contract_path.exists():
        raise RuntimeError(f"frozen contract already exists: {contract_path}")
    candidate_path = output_root / "manifests/new_candidate_manifest.jsonl"
    candidate_audit = read_json(output_root / "manifests/candidate_audit.json")
    if (
        candidate_audit.get("passed") is not True
        or candidate_audit["candidate_manifest_sha256"] != file_sha256(candidate_path)
    ):
        raise RuntimeError("candidate freeze is absent or incompatible")
    sources = _source_files(config)
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"frozen sources are missing: {missing}")
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    bound_hashes = {relative: file_sha256(resolve_path(relative)) for relative in BOUND_CODE_PATHS}
    snapshot = resolve_path(config["model"]["snapshot_path"])
    model_files = sorted(path for path in snapshot.iterdir() if path.is_file())
    model_hashes = {path.name: file_sha256(path) for path in model_files}
    phase56 = read_json(resolve_path(config["sources"]["phase56_contract"]))
    if model_hashes != phase56["model_snapshot_sha256"]:
        raise RuntimeError("live model snapshot differs from the replay-validated Phase-56 snapshot")
    contract: dict[str, Any] = {
        "schema_version": "stage2_data_scale_search_contract_v1",
        "static_config": config,
        "git": {
            "commit": command_output(("git", "rev-parse", "HEAD")),
            "branch": command_output(("git", "branch", "--show-current")),
            "worktree_status_at_freeze": command_output(("git", "status", "--short")),
        },
        "runtime": _runtime_metadata(),
        "source_paths": {name: str(path) for name, path in sources.items()},
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "model_snapshot_sha256": model_hashes,
        "candidate_manifest_sha256": source_hashes["internal:manifests/new_candidate_manifest.jsonl"],
        "phase56_contract_sha256": phase56["contract_sha256"],
        "population": candidate_audit["validation"],
        "review_reconciliation": {
            "verdict": "revise",
            "selected": "outcome-blind metadata-stratified canonical-source 2000/1000/1000 pool",
            "claim_boundary": "canonical-source data scale-up, not a perfectly pure scale-only comparison",
            "guards": [
                "immutable source revision/config/split/native key",
                "deterministic source-shard and row selection",
                "SHA-256 image-group exclusion against all 8000 legacy candidates",
                "source-metadata distributions frozen before dense outcomes",
            ],
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(contract_path, contract)
    protocol = f"""# Stage-2 data-scale search protocol

- Contract SHA-256: `{contract['contract_sha256']}`
- Candidate manifest SHA-256: `{contract['candidate_manifest_sha256']}`
- Population: 4,000 unique train candidates (2,000 GQA / 1,000 ChartQA / 1,000 TextVQA), each a unique SHA-256 image group, with zero overlap against all 8,000 legacy candidates.
- Selection was frozen before current dense outcomes, Stage-1 scores, triggers, or fixability. It matches recoverable pre-outcome metadata strata and uses no historical or current model correctness.
- Runtime: frozen native dense Qwen2.5-VL followed by the frozen Shared Random-4 strict global trigger. Triggered Dense-C receives FULL preservation; triggered Dense-W receives exhaustive singles, then cap-200 Phase-56-equivalent MCTS only if single-unresolved.
- Every retained route must reproduce exact tokens and current LMMS correctness. Single, MCTS, preservation, and unresolved provenance remain separate.
- Claim boundary: this is a canonical-source data-scale expansion, not a perfectly pure scale-only population replication.
- Stop after expanded corpora/audits. No router training, Stage-1 retuning, validation/test search, or held-out evaluation.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"]}))


def load_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = _load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if contract.get("contract_sha256") != canonical_hash(contract) or contract["static_config"] != config:
        raise RuntimeError("frozen data-scale contract/config mismatch")
    if command_output(("git", "rev-parse", "HEAD")) != contract["git"]["commit"]:
        raise RuntimeError("git commit differs from frozen contract")
    if command_output(("git", "branch", "--show-current")) != contract["git"]["branch"]:
        raise RuntimeError("git branch differs from frozen contract")
    if command_output(("git", "status", "--short")) != contract["git"]["worktree_status_at_freeze"]:
        raise RuntimeError("worktree status differs from frozen contract")
    if _runtime_metadata() != contract["runtime"]:
        raise RuntimeError("runtime differs from frozen contract")
    for name, expected in contract["source_sha256"].items():
        if file_sha256(Path(contract["source_paths"][name])) != expected:
            raise RuntimeError(f"source hash differs: {name}")
    for relative, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(relative)) != expected:
            raise RuntimeError(f"bound code differs: {relative}")
    if verify_model:
        snapshot = resolve_path(config["model"]["snapshot_path"])
        if {path.name for path in snapshot.iterdir() if path.is_file()} != set(contract["model_snapshot_sha256"]):
            raise RuntimeError("model snapshot inventory differs")
        for name, expected in contract["model_snapshot_sha256"].items():
            if file_sha256(snapshot / name) != expected:
                raise RuntimeError(f"model snapshot differs: {name}")
    return contract, output_root


def bind_dense(config_path: Path) -> None:
    contract, output_root = load_contract(config_path, verify_model=True)
    dense_root = output_root / "dense"
    paths = {
        "dense_outputs": dense_root / "dense_outputs.jsonl",
        "skipped_samples": dense_root / "skipped_samples.jsonl",
        "generation_summary": dense_root / "generation_summary.json",
        "feature_index": dense_root / "features/feature_index.jsonl",
        "feature_integrity": dense_root / "features/feature_integrity_audit.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("dense aggregate is incomplete")
    outputs = read_jsonl(paths["dense_outputs"])
    skips = read_jsonl(paths["skipped_samples"])
    candidates = read_jsonl(output_root / "manifests/new_candidate_manifest.jsonl")
    attempted = Counter(str(row["uid"]) for row in [*outputs, *skips])
    expected = Counter(str(row["uid"]) for row in candidates)
    if attempted != expected or len(outputs) != len({str(row["uid"]) for row in outputs}):
        raise RuntimeError("dense candidate coverage differs")
    integrity = read_json(paths["feature_integrity"])
    if not integrity.get("passed") or int(integrity["records"]) != len(outputs):
        raise RuntimeError("dense feature integrity differs")
    feature_index = read_jsonl(paths["feature_index"])
    if {str(row["uid"]) for row in feature_index} != {str(row["uid"]) for row in outputs}:
        raise RuntimeError("dense feature index/output coverage differs")
    shard_paths = sorted({str(row["shard"]) for row in feature_index})
    payload: dict[str, Any] = {
        "schema_version": "stage2_data_scale_dense_binding_v1",
        "parent_contract_sha256": contract["contract_sha256"],
        "candidate_manifest_sha256": contract["candidate_manifest_sha256"],
        "attempted": len(candidates),
        "completed": len(outputs),
        "skipped": len(skips),
        "artifact_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "feature_shard_sha256": {
            relative: file_sha256(resolve_path(relative)) for relative in shard_paths
        },
        "created_at": utc_now(),
    }
    payload["binding_sha256"] = canonical_hash(
        payload, excluded=("binding_sha256",)
    )
    atomic_json(output_root / "work/dense_binding.json", payload)
    print(json.dumps({"passed": True, "completed": len(outputs), "skipped": len(skips), "binding": payload["binding_sha256"]}))


def dense_execute(config_path: Path, *, mode: str) -> None:
    contract, output_root = load_contract(config_path, verify_model=True)
    if mode not in {"smoke", "full"}:
        raise ValueError("dense execution mode must be smoke or full")
    if mode == "full":
        smoke_summary = read_json(output_root / "dense/smoke/smoke_summary.json")
        if smoke_summary.get("passed") is not True:
            raise RuntimeError("full dense execution requires a passing dense smoke")
    manifest = output_root / (
        "smoke/dense_smoke_manifest.json"
        if mode == "smoke"
        else "manifests/new_candidate_manifest.jsonl"
    )
    dense_config = resolve_path(contract["static_config"]["sources"]["dense_config"])
    args = argparse.Namespace(
        mode=mode,
        config=str(dense_config),
        output_root=str(output_root / "dense"),
        portable_manifest=None,
        image_root=None,
        manifest=str(manifest),
        model_path=contract["static_config"]["model"]["snapshot_path"],
        run_id=f"{contract['static_config']['run_id']}_dense",
        rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 4)),
        batch_size=64,
    )
    dense_runner.worker(args)


def aggregate_dense(config_path: Path, *, mode: str) -> None:
    contract, output_root = load_contract(config_path, verify_model=True)
    if mode not in {"smoke", "full"}:
        raise ValueError("dense aggregation mode must be smoke or full")
    manifest = output_root / (
        "smoke/dense_smoke_manifest.json"
        if mode == "smoke"
        else "manifests/new_candidate_manifest.jsonl"
    )
    dense_config = resolve_path(contract["static_config"]["sources"]["dense_config"])
    args = argparse.Namespace(
        mode="aggregate-smoke" if mode == "smoke" else "aggregate-full",
        config=str(dense_config),
        output_root=str(output_root / "dense"),
        manifest=str(manifest),
    )
    dense_runner.aggregate(args)


def _load_dense_binding(contract: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    binding = read_json(output_root / "work/dense_binding.json")
    if (
        binding.get("binding_sha256")
        != canonical_hash(binding, excluded=("binding_sha256",))
        or binding["parent_contract_sha256"] != contract["contract_sha256"]
    ):
        raise RuntimeError("dense binding is invalid")
    dense_root = output_root / "dense"
    paths = {
        "dense_outputs": dense_root / "dense_outputs.jsonl",
        "skipped_samples": dense_root / "skipped_samples.jsonl",
        "generation_summary": dense_root / "generation_summary.json",
        "feature_index": dense_root / "features/feature_index.jsonl",
        "feature_integrity": dense_root / "features/feature_integrity_audit.json",
    }
    for name, expected in binding["artifact_sha256"].items():
        if file_sha256(paths[name]) != expected:
            raise RuntimeError(f"dense bound artifact differs: {name}")
    for relative, expected in binding["feature_shard_sha256"].items():
        if file_sha256(resolve_path(relative)) != expected:
            raise RuntimeError(f"dense feature shard differs: {relative}")
    return binding


def stage1_worker(config_path: Path, *, rank: int, world_size: int) -> None:
    worker_started = time.monotonic()
    contract, output_root = load_contract(config_path)
    config = contract["static_config"]
    binding = _load_dense_binding(contract, output_root)
    if world_size != 4 or rank not in range(4) or torch.cuda.device_count() != 4:
        raise RuntimeError("Stage-1 scale scoring requires four visible GPU ranks")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    dense_config = read_json(resolve_path(config["sources"]["dense_config"]))
    configure_dense_determinism(int(config["seed"]) + rank, dense_config["backend_settings"])
    dense_rows = {str(row["uid"]): row for row in read_jsonl(output_root / "dense/dense_outputs.jsonl")}
    feature_index = read_jsonl(output_root / "dense/features/feature_index.jsonl")
    shard_names = sorted({str(row["shard"]) for row in feature_index})
    assigned_shards = {name for index, name in enumerate(shard_names) if index % world_size == rank}
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_index:
        if str(row["shard"]) in assigned_shards:
            dense = dense_rows[str(row["uid"])]
            by_shard[str(row["shard"])].append(
                {
                    "uid": row["uid"],
                    "dataset": dense["dataset"],
                    "image_group_id": dense["image_group_id"],
                    "current_dense_wrong": dense["current_dense_wrong"],
                    "feature_row_index": row["row_index"],
                }
            )
    model_contract = {
        "sources": {
            "shared_contract": config["sources"]["shared_contract"],
            "shared_checkpoint": config["sources"]["shared_checkpoint"],
            "shared_normalization": config["sources"]["shared_normalization"],
        }
    }
    model, mean, std = _build_frozen_model(model_contract, device)
    scores = []
    for relative in sorted(by_shard):
        shard = torch.load(resolve_path(relative), map_location="cpu", weights_only=True)
        scores.extend(_score_feature_rows(model, mean, std, shard, by_shard[relative], device=device))
    scores.sort(key=lambda row: str(row["uid"]))
    result_path = output_root / f"work/stage1_scores/rank{rank:02d}.jsonl"
    atomic_jsonl(result_path, scores)
    atomic_json(
        output_root / f"work/stage1_scores/rank{rank:02d}.complete.json",
        {
            "passed": True,
            "parent_contract_sha256": contract["contract_sha256"],
            "dense_binding_sha256": binding["binding_sha256"],
            "rank": rank,
            "world_size": world_size,
            "records": len(scores),
            "elapsed_seconds": time.monotonic() - worker_started,
            "scores_sha256": file_sha256(result_path),
            "completed_at": utc_now(),
        },
    )
    print(json.dumps({"passed": True, "rank": rank, "records": len(scores)}))


def aggregate_trigger_map(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    binding = _load_dense_binding(contract, output_root)
    scores = []
    for rank in range(4):
        result = output_root / f"work/stage1_scores/rank{rank:02d}.jsonl"
        complete = read_json(output_root / f"work/stage1_scores/rank{rank:02d}.complete.json")
        if (
            complete.get("passed") is not True
            or complete["parent_contract_sha256"] != contract["contract_sha256"]
            or complete["dense_binding_sha256"] != binding["binding_sha256"]
            or complete["scores_sha256"] != file_sha256(result)
        ):
            raise RuntimeError(f"Stage-1 score rank is incomplete: {rank}")
        scores.extend(read_jsonl(result))
    dense = read_jsonl(output_root / "dense/dense_outputs.jsonl")
    dense_uids = {str(row["uid"]) for row in dense}
    candidates = [
        row
        for row in read_jsonl(output_root / "manifests/new_candidate_manifest.jsonl")
        if str(row["uid"]) in dense_uids
    ]
    trigger_map, triggered_wrong, triggered_correct = build_new_trigger_map(
        candidates, dense, scores, threshold=float(config["gate"]["threshold"])
    )
    atomic_jsonl(output_root / "manifests/new_dense_results.jsonl", dense)
    atomic_jsonl(output_root / "manifests/new_trigger_map.jsonl", trigger_map)
    atomic_jsonl(output_root / "manifests/new_triggered_wrong.jsonl", triggered_wrong)
    atomic_jsonl(output_root / "manifests/new_triggered_correct.jsonl", triggered_correct)
    partitions = Counter(
        ("W" if row["dense_wrong"] else "C", "trigger" if row["triggered"] else "no_trigger")
        for row in trigger_map
    )
    payload: dict[str, Any] = {
        "schema_version": "stage2_data_scale_trigger_binding_v1",
        "parent_contract_sha256": contract["contract_sha256"],
        "dense_binding_sha256": binding["binding_sha256"],
        "scores": len(scores),
        "triggered_wrong": len(triggered_wrong),
        "triggered_correct": len(triggered_correct),
        "partitions": {f"{label}_{status}": value for (label, status), value in sorted(partitions.items())},
        "artifact_sha256": {
            relative: file_sha256(output_root / relative)
            for relative in (
                "manifests/new_dense_results.jsonl",
                "manifests/new_trigger_map.jsonl",
                "manifests/new_triggered_wrong.jsonl",
                "manifests/new_triggered_correct.jsonl",
            )
        },
        "created_at": utc_now(),
    }
    payload["binding_sha256"] = canonical_hash(
        payload, excluded=("binding_sha256",)
    )
    atomic_json(output_root / "work/trigger_binding.json", payload)
    print(json.dumps({"passed": True, "scores": len(scores), "triggered_wrong": len(triggered_wrong), "triggered_correct": len(triggered_correct)}))


def _load_trigger_binding(
    contract: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    binding = read_json(output_root / "work/trigger_binding.json")
    if (
        binding.get("binding_sha256")
        != canonical_hash(binding, excluded=("binding_sha256",))
        or binding["parent_contract_sha256"] != contract["contract_sha256"]
    ):
        raise RuntimeError("trigger binding is invalid")
    for relative, expected in binding["artifact_sha256"].items():
        if file_sha256(output_root / relative) != expected:
            raise RuntimeError(f"trigger-bound artifact differs: {relative}")
    return binding


def _select_search_smoke(
    rows: Sequence[Mapping[str, Any]], *, task_type: str, count: int, seed: int
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["task_type"] == task_type]
    ordered = sorted(
        eligible,
        key=lambda row: (
            sha256(f"stage2-scale-search-smoke:{seed}:{task_type}:{row['uid']}".encode()).hexdigest(),
            str(row["uid"]),
        ),
    )
    if len(ordered) < count:
        raise RuntimeError(f"insufficient {task_type} rows for the frozen smoke")
    return [dict(row) for row in ordered[:count]]


def prepare_search(config_path: Path) -> None:
    parent, output_root = load_contract(config_path, verify_model=True)
    dense_binding = _load_dense_binding(parent, output_root)
    trigger_binding = _load_trigger_binding(parent, output_root)
    search_contract_path = output_root / "work/search_contract.json"
    if search_contract_path.exists():
        raise RuntimeError(f"search contract already exists: {search_contract_path}")
    config = parent["static_config"]
    candidates = read_jsonl(output_root / "manifests/new_candidate_manifest.jsonl")
    dense = read_jsonl(output_root / "manifests/new_dense_results.jsonl")
    triggers = [
        row
        for row in read_jsonl(output_root / "manifests/new_trigger_map.jsonl")
        if bool(row["triggered"])
    ]
    work = build_search_work(
        candidates,
        dense,
        triggers,
        maximum_iterations=int(config["mcts"]["maximum_iterations"]),
    )
    assigned = assign_workers(work, world_size=int(config["world_size"]))
    smoke_wrong = _select_search_smoke(
        work,
        task_type="triggered_wrong",
        count=int(config["smoke"]["wrong_records"]),
        seed=int(config["seed"]),
    )
    smoke_correct = _select_search_smoke(
        work,
        task_type="triggered_correct_preservation",
        count=int(config["smoke"]["preservation_records"]),
        seed=int(config["seed"]),
    )
    smoke: list[dict[str, Any]] = []
    for rank, (wrong, correct) in enumerate(
        zip(smoke_wrong, smoke_correct, strict=True)
    ):
        smoke.extend(({**wrong, "worker_rank": rank}, {**correct, "worker_rank": rank}))
    smoke.sort(key=lambda row: str(row["uid"]))
    schema = _feature_schema(config)
    phase56 = read_json(resolve_path(config["sources"]["phase56_contract"]))
    if schema["feature_schema_sha256"] != phase56["feature_schema_sha256"]:
        raise RuntimeError("new routed-state schema differs from the frozen Phase-56 corpus")
    atomic_jsonl(output_root / "work/manifests/full_work_manifest.jsonl", assigned)
    atomic_jsonl(output_root / "work/manifests/smoke_manifest.jsonl", smoke)
    atomic_json(output_root / "states/feature_schema.json", schema)
    internal = {
        relative: file_sha256(output_root / relative)
        for relative in (
            "work/manifests/full_work_manifest.jsonl",
            "work/manifests/smoke_manifest.jsonl",
            "states/feature_schema.json",
        )
    }
    search_contract: dict[str, Any] = {
        "schema_version": "stage2_data_scale_corrective_search_contract_v1",
        "static_config": config,
        "parent_contract_sha256": parent["contract_sha256"],
        "dense_binding_sha256": dense_binding["binding_sha256"],
        "trigger_binding_sha256": trigger_binding["binding_sha256"],
        "git": parent["git"],
        "runtime": parent["runtime"],
        "bound_code_sha256": parent["bound_code_sha256"],
        "model_snapshot_sha256": parent["model_snapshot_sha256"],
        "internal_manifest_sha256": internal,
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "population": {
            "triggered_wrong": sum(row["task_type"] == "triggered_wrong" for row in work),
            "triggered_correct_preservation": sum(
                row["task_type"] == "triggered_correct_preservation" for row in work
            ),
            "full_work": len(work),
            "smoke": len(smoke),
        },
        "source_phase": 59,
        "candidate_pool": "new_canonical_scaleup_v1",
        "search_budget": {
            "single": "exhaustive from trigger through L27",
            "mcts_max_iterations": 200,
            "mcts_only_after_single_unresolved": True,
        },
        "created_at": utc_now(),
    }
    search_contract["contract_sha256"] = canonical_hash(search_contract)
    atomic_json(search_contract_path, search_contract)
    atomic_json(
        output_root / "work/imported/complete.json",
        {
            "passed": True,
            "contract_sha256": search_contract["contract_sha256"],
            "records": 0,
            "purpose": "compatibility marker; scale-up executes every row fresh",
            "completed_at": utc_now(),
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "search_contract_sha256": search_contract["contract_sha256"],
                **search_contract["population"],
            }
        )
    )


def load_search_contract(
    config_path: Path, *, verify_model: bool = False
) -> tuple[dict[str, Any], Path]:
    parent, output_root = load_contract(config_path, verify_model=verify_model)
    dense_binding = _load_dense_binding(parent, output_root)
    trigger_binding = _load_trigger_binding(parent, output_root)
    contract = read_json(output_root / "work/search_contract.json")
    if (
        contract.get("contract_sha256") != canonical_hash(contract)
        or contract["static_config"] != parent["static_config"]
        or contract["parent_contract_sha256"] != parent["contract_sha256"]
        or contract["dense_binding_sha256"] != dense_binding["binding_sha256"]
        or contract["trigger_binding_sha256"] != trigger_binding["binding_sha256"]
    ):
        raise RuntimeError("corrective-search contract is invalid")
    for relative, expected in contract["internal_manifest_sha256"].items():
        if file_sha256(output_root / relative) != expected:
            raise RuntimeError(f"search manifest differs: {relative}")
    schema = read_json(output_root / "states/feature_schema.json")
    if (
        schema.get("feature_schema_sha256") != contract["feature_schema_sha256"]
        or canonical_hash(schema, excluded=("feature_schema_sha256",))
        != contract["feature_schema_sha256"]
    ):
        raise RuntimeError("search feature schema differs")
    return contract, output_root


def _install_phase56_contract_adapter() -> None:
    def adapter(
        config_path: Path, *, verify_model_snapshot: bool = False
    ) -> tuple[dict[str, Any], Path]:
        return load_search_contract(
            config_path, verify_model=verify_model_snapshot
        )

    phase56_runner.load_contract = adapter


def search_worker(
    config_path: Path,
    *,
    mode: str,
    rank: int,
    world_size: int,
    resume: bool,
) -> None:
    _install_phase56_contract_adapter()
    phase56_runner.worker(
        config_path,
        mode=mode,
        rank=rank,
        world_size=world_size,
        resume=resume,
    )


def finalize_search_smoke(config_path: Path) -> None:
    _install_phase56_contract_adapter()
    phase56_runner.finalize_smoke(config_path)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _collect_search_results(
    output_root: Path, contract_sha256: str, mode: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in range(4):
        root = output_root / f"work/{mode}/rank{rank:02d}"
        complete = read_json(root / "complete.json")
        if (
            complete.get("passed") is not True
            or complete.get("contract_sha256") != contract_sha256
        ):
            raise RuntimeError(f"search rank {rank} is incomplete")
        failures = sorted(root.glob("failure_*.json"))
        if failures:
            raise RuntimeError(f"search rank {rank} has failure records")
        rows.extend(read_json(path) for path in sorted((root / "samples").glob("*.json")))
    if len(rows) != len({str(row["uid"]) for row in rows}):
        raise RuntimeError("search results contain duplicate UIDs")
    return sorted(rows, key=lambda row: str(row["uid"]))


def _with_corpus_provenance(
    row: Mapping[str, Any],
    *,
    source_phase: int,
    candidate_pool: str,
    artifact_root: str,
) -> dict[str, Any]:
    route_source = str(row["route_source"])
    return {
        **dict(row),
        "source_phase": source_phase,
        "candidate_pool": candidate_pool,
        "search_type": route_source,
        "search_budget": (
            "none; known-safe FULL preservation"
            if route_source == "preservation_full"
            else "exhaustive single"
            if route_source == "single"
            else "single-unresolved then MCTS max_iterations=200"
        ),
        "feature_artifact_root": artifact_root,
    }


def aggregate(config_path: Path) -> None:
    contract, output_root = load_search_contract(config_path, verify_model=True)
    config = contract["static_config"]
    smoke = read_json(output_root / "work/smoke/completion.json")
    if (
        smoke.get("passed") is not True
        or smoke.get("contract_sha256") != contract["contract_sha256"]
    ):
        raise RuntimeError("full aggregation requires a passing search smoke")
    work = read_jsonl(output_root / "work/manifests/full_work_manifest.jsonl")
    results = _collect_search_results(output_root, contract["contract_sha256"], "full")
    phase56_runner.validate_complete_results(
        [str(row["uid"]) for row in work], results
    )
    if Counter(row["task_type"] for row in work) != Counter(
        row["task_type"] for row in results
    ):
        raise RuntimeError("search task populations changed")

    classes = Counter(str(row["outcome_class"]) for row in results)
    valid_classes = {
        "SINGLE_FIXABLE",
        "MCTS_ONLY_FIXABLE",
        "UNRESOLVED",
        "PRESERVATION_FULL",
    }
    if set(classes) - valid_classes:
        raise RuntimeError(f"unexpected search outcome classes: {classes}")
    new_routes = [
        route for result in results for route in result["retained_successful_routes"]
    ]
    for result in results:
        result.update(
            {
                "source_phase": 59,
                "candidate_pool": "new_canonical_scaleup_v1",
                "search_type": (
                    "preservation_full"
                    if result["task_type"] == "triggered_correct_preservation"
                    else "single_then_mcts_if_unresolved"
                ),
                "search_budget": (
                    "none; known-safe FULL preservation"
                    if result["task_type"] == "triggered_correct_preservation"
                    else "exhaustive single then MCTS max_iterations=200"
                ),
                "stage1_trigger_binding_sha256": contract[
                    "trigger_binding_sha256"
                ],
            }
        )
    for route in new_routes:
        route.update(
            {
                "source_phase": 59,
                "candidate_pool": "new_canonical_scaleup_v1",
                "search_budget": (
                    "none; known-safe FULL preservation"
                    if route["route_source"] == "preservation_full"
                    else "exhaustive single"
                    if route["route_source"] == "single"
                    else "single-unresolved then MCTS max_iterations=200"
                ),
                "stage1_trigger_binding_sha256": contract[
                    "trigger_binding_sha256"
                ],
            }
        )
    new_state_rows = [row for result in results for row in result["feature_index"]]
    state_shards: dict[str, str] = {}
    for result in results:
        if not result.get("feature_file"):
            if result["outcome_class"] != "UNRESOLVED":
                raise RuntimeError(f"retained supervision lacks features: {result['uid']}")
            continue
        path = output_root / str(result["feature_file"])
        expected = str(result["feature_file_sha256"])
        if file_sha256(path) != expected:
            raise RuntimeError(f"new state shard hash differs: {result['uid']}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        phase56_runner._validate_saved_shard(path, payload)
        if (
            payload["contract_sha256"] != contract["contract_sha256"]
            or payload["feature_schema_sha256"] != contract["feature_schema_sha256"]
        ):
            raise RuntimeError(f"new state shard provenance differs: {result['uid']}")
        state_shards[str(result["feature_file"])] = expected
    if any(
        not bool(route["final_lmms_correct"])
        or not bool(route["replay_token_parity"])
        or route["contract_sha256"] != contract["contract_sha256"]
        for route in new_routes
    ):
        raise RuntimeError("new retained route is not an exact correct replay")

    single_results = [row for row in results if row["outcome_class"] == "SINGLE_FIXABLE"]
    mcts_results = [row for row in results if row["outcome_class"] == "MCTS_ONLY_FIXABLE"]
    unresolved_results = [row for row in results if row["outcome_class"] == "UNRESOLVED"]
    preservation_results = [row for row in results if row["outcome_class"] == "PRESERVATION_FULL"]
    single_routes = [row for row in new_routes if row["route_source"] == "single"]
    mcts_routes = [row for row in new_routes if row["route_source"] == "mcts"]
    preservation_routes = [
        row for row in new_routes if row["route_source"] == "preservation_full"
    ]
    if (
        len({str(row["uid"]) for row in single_routes}) != len(single_results)
        or len({str(row["uid"]) for row in mcts_routes}) != len(mcts_results)
        or len({str(row["uid"]) for row in preservation_routes})
        != len(preservation_results)
    ):
        raise RuntimeError("new base/route coverage differs")

    atomic_jsonl(output_root / "manifests/new_single_fixable.jsonl", single_results)
    atomic_jsonl(output_root / "manifests/new_mcts_only_fixable.jsonl", mcts_results)
    atomic_jsonl(output_root / "manifests/new_unresolved.jsonl", unresolved_results)
    atomic_jsonl(
        output_root / "manifests/new_preservation_results.jsonl",
        preservation_results,
    )
    atomic_jsonl(output_root / "routes/new_successful_single_routes.jsonl", single_routes)
    atomic_jsonl(output_root / "routes/new_successful_mcts_routes.jsonl", mcts_routes)
    atomic_jsonl(output_root / "states/new_state_index.jsonl", new_state_rows)

    old_root = ROOT / "analysis/dense_failure_stage2/full_corrective_labels"
    old_artifact = read_json(resolve_path(config["sources"]["old_artifact_manifest"]))
    verify_artifact_manifest(old_root, old_artifact)
    old_a = read_jsonl(resolve_path(config["sources"]["old_corpus_A"]))
    old_b = read_jsonl(resolve_path(config["sources"]["old_corpus_B"]))
    old_c = read_jsonl(resolve_path(config["sources"]["old_corpus_C"]))
    old_unresolved = read_jsonl(resolve_path(config["sources"]["old_unresolved"]))
    old_label_uids = {
        str(row["uid"]) for row in [*old_a, *old_b, *old_c, *old_unresolved]
    }
    new_label_uids = {str(row["uid"]) for row in results}
    if old_label_uids & new_label_uids:
        raise RuntimeError("legacy and new Stage-2 label bases overlap")
    old_root_relative = "analysis/dense_failure_stage2/full_corrective_labels"
    new_root_relative = str(output_root.relative_to(ROOT))
    expanded_a = [
        *[
            _with_corpus_provenance(
                row,
                source_phase=56,
                candidate_pool="legacy_current_dense_8k_train",
                artifact_root=old_root_relative,
            )
            for row in old_a
        ],
        *[
            _with_corpus_provenance(
                row,
                source_phase=59,
                candidate_pool="new_canonical_scaleup_v1",
                artifact_root=new_root_relative,
            )
            for row in preservation_routes
        ],
    ]
    expanded_b = [
        *[
            _with_corpus_provenance(
                row,
                source_phase=56,
                candidate_pool="legacy_current_dense_8k_train",
                artifact_root=old_root_relative,
            )
            for row in old_b
        ],
        *[
            _with_corpus_provenance(
                row,
                source_phase=59,
                candidate_pool="new_canonical_scaleup_v1",
                artifact_root=new_root_relative,
            )
            for row in single_routes
        ],
    ]
    expanded_c = [
        *[
            _with_corpus_provenance(
                row,
                source_phase=56,
                candidate_pool="legacy_current_dense_8k_train",
                artifact_root=old_root_relative,
            )
            for row in old_c
        ],
        *[
            _with_corpus_provenance(
                row,
                source_phase=59,
                candidate_pool="new_canonical_scaleup_v1",
                artifact_root=new_root_relative,
            )
            for row in mcts_routes
        ],
    ]
    expanded_unresolved = [
        *[
            {
                **row,
                "source_phase": 56,
                "candidate_pool": "legacy_current_dense_8k_train",
                "search_budget": "single-unresolved then MCTS max_iterations=200",
            }
            for row in old_unresolved
        ],
        *[
            {
                **row,
                "source_phase": 59,
                "candidate_pool": "new_canonical_scaleup_v1",
                "search_budget": "single-unresolved then MCTS max_iterations=200",
            }
            for row in unresolved_results
        ],
    ]
    combined = {
        "A": ("combined_corpora/expanded_corpus_A_preservation.jsonl", expanded_a),
        "B": ("combined_corpora/expanded_corpus_B_single.jsonl", expanded_b),
        "C": ("combined_corpora/expanded_corpus_C_mcts.jsonl", expanded_c),
    }
    for _key, (relative, rows) in combined.items():
        atomic_jsonl(output_root / relative, rows)
    atomic_jsonl(
        output_root / "combined_corpora/expanded_unresolved.jsonl",
        expanded_unresolved,
    )
    corpus_manifest = {
        "schema_version": "stage2_data_scale_expanded_corpus_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "old_contract_sha256": old_artifact["contract_sha256"],
        "feature_schema_sha256": contract["feature_schema_sha256"],
        "corpora": {
            key: {
                "path": relative,
                "sha256": file_sha256(output_root / relative),
                "route_records": len(rows),
                "unique_samples": len({str(row["uid"]) for row in rows}),
                "old_route_records": len({"A": old_a, "B": old_b, "C": old_c}[key]),
                "new_route_records": len(rows)
                - len({"A": old_a, "B": old_b, "C": old_c}[key]),
                "route_source": {
                    "A": "preservation_full",
                    "B": "single",
                    "C": "mcts",
                }[key],
            }
            for key, (relative, rows) in combined.items()
        },
        "unresolved": {
            "path": "combined_corpora/expanded_unresolved.jsonl",
            "sha256": file_sha256(
                output_root / "combined_corpora/expanded_unresolved.jsonl"
            ),
            "records": len(expanded_unresolved),
            "old_records": len(old_unresolved),
            "new_records": len(unresolved_results),
        },
        "sampling_policy": "not selected; no Stage-2 training in this phase",
    }
    atomic_json(output_root / "combined_corpora/corpus_manifest.json", corpus_manifest)

    trigger_map = read_jsonl(output_root / "manifests/new_trigger_map.jsonl")
    old_trigger_map = read_jsonl(resolve_path(config["sources"]["old_trigger_map"]))
    old_counts = Counter(
        (
            "W" if bool(row["dense_wrong"]) else "C",
            "trigger" if bool(row["triggered"]) else "no_trigger",
        )
        for row in old_trigger_map
    )
    if (
        len(old_trigger_map) != 6399
        or old_counts[("W", "trigger")] != 1881
        or old_counts[("C", "trigger")] != 39
    ):
        raise RuntimeError("legacy trigger population differs from Phase 56")
    new_counts = Counter(
        (
            "W" if bool(row["dense_wrong"]) else "C",
            "trigger" if bool(row["triggered"]) else "no_trigger",
        )
        for row in trigger_map
    )

    def yield_row(
        name: str,
        raw: int,
        counts: Counter,
        single: int,
        mcts: int,
        unresolved: int,
    ) -> dict[str, Any]:
        wrong = counts[("W", "trigger")] + counts[("W", "no_trigger")]
        correct = counts[("C", "trigger")] + counts[("C", "no_trigger")]
        tw = counts[("W", "trigger")]
        tc = counts[("C", "trigger")]
        return {
            "pool": name,
            "raw_candidates": raw,
            "dense_wrong": wrong,
            "dense_correct": correct,
            "triggered_wrong": tw,
            "triggered_correct": tc,
            "single_fixable": single,
            "mcts_only_fixable": mcts,
            "total_bounded_fixable": single + mcts,
            "unresolved": unresolved,
            "p_trigger_given_w": _rate(tw, wrong),
            "p_trigger_given_c": _rate(tc, correct),
            "p_single_given_triggered_w": _rate(single, tw),
            "p_mcts_only_given_triggered_w": _rate(mcts, tw),
            "p_bounded_fixable_given_triggered_w": _rate(single + mcts, tw),
        }

    old_yield = yield_row("old", 6399, old_counts, 698, 209, 974)
    new_yield = yield_row(
        "new",
        len(trigger_map),
        new_counts,
        len(single_results),
        len(mcts_results),
        len(unresolved_results),
    )
    combined_counts = old_counts + new_counts
    combined_yield = yield_row(
        "combined",
        6399 + len(trigger_map),
        combined_counts,
        698 + len(single_results),
        209 + len(mcts_results),
        974 + len(unresolved_results),
    )
    phase56_runner.atomic_csv(
        output_root / "metrics/old_vs_new_yield.csv",
        [old_yield, new_yield, combined_yield],
    )
    phase56_runner.atomic_csv(
        output_root / "metrics/scaleup_summary.csv",
        [
            {
                **new_yield,
                "new_preservation_routes": len(preservation_routes),
                "new_single_routes": len(single_routes),
                "new_mcts_routes": len(mcts_routes),
                "expanded_A_routes": len(expanded_a),
                "expanded_A_bases": len({str(row["uid"]) for row in expanded_a}),
                "expanded_B_routes": len(expanded_b),
                "expanded_B_bases": len({str(row["uid"]) for row in expanded_b}),
                "expanded_C_routes": len(expanded_c),
                "expanded_C_bases": len({str(row["uid"]) for row in expanded_c}),
            }
        ],
    )

    result_by_uid = {str(row["uid"]): row for row in results}
    dataset_rows = []
    for dataset in ("gqa", "chartqa", "textvqa"):
        population = [row for row in trigger_map if row["dataset"] == dataset]
        wrong = [row for row in population if row["dense_wrong"]]
        triggered_w = [row for row in wrong if row["triggered"]]
        triggered_c = [
            row for row in population if not row["dense_wrong"] and row["triggered"]
        ]
        outcomes = [result_by_uid[str(row["uid"])] for row in triggered_w]
        outcome_counts = Counter(str(row["outcome_class"]) for row in outcomes)
        dataset_rows.append(
            {
                "dataset": dataset,
                "raw_new_candidates": len(population),
                "dense_wrong": len(wrong),
                "triggered_wrong": len(triggered_w),
                "single_fixable": outcome_counts["SINGLE_FIXABLE"],
                "single_fixable_rate": _rate(
                    outcome_counts["SINGLE_FIXABLE"], len(triggered_w)
                ),
                "mcts_only_fixable": outcome_counts["MCTS_ONLY_FIXABLE"],
                "mcts_only_fixable_rate": _rate(
                    outcome_counts["MCTS_ONLY_FIXABLE"], len(triggered_w)
                ),
                "total_bounded_fixable": outcome_counts["SINGLE_FIXABLE"]
                + outcome_counts["MCTS_ONLY_FIXABLE"],
                "total_bounded_fixable_rate": _rate(
                    outcome_counts["SINGLE_FIXABLE"]
                    + outcome_counts["MCTS_ONLY_FIXABLE"],
                    len(triggered_w),
                ),
                "triggered_correct_preservation": len(triggered_c),
            }
        )
    phase56_runner.atomic_csv(output_root / "metrics/dataset_breakdown.csv", dataset_rows)

    depth_rows = []
    for depth in ("L0", "L1-8", "L9-18", "L19-27"):
        cell = [
            row
            for row in results
            if row["task_type"] == "triggered_wrong"
            and row["trigger_depth_bin"] == depth
        ]
        cell_counts = Counter(str(row["outcome_class"]) for row in cell)
        depth_rows.append(
            {
                "trigger_depth_bin": depth,
                "new_triggered_wrong": len(cell),
                "single_fixable": cell_counts["SINGLE_FIXABLE"],
                "single_fixable_rate": _rate(cell_counts["SINGLE_FIXABLE"], len(cell)),
                "mcts_only_fixable": cell_counts["MCTS_ONLY_FIXABLE"],
                "mcts_only_fixable_rate": _rate(
                    cell_counts["MCTS_ONLY_FIXABLE"], len(cell)
                ),
                "total_bounded_fixable": cell_counts["SINGLE_FIXABLE"]
                + cell_counts["MCTS_ONLY_FIXABLE"],
                "total_bounded_fixable_rate": _rate(
                    cell_counts["SINGLE_FIXABLE"]
                    + cell_counts["MCTS_ONLY_FIXABLE"],
                    len(cell),
                ),
                "single_terminal_routes": sum(
                    int(row["single_routes_evaluated"]) for row in cell
                ),
                "mcts_iterations": sum(int(row["mcts_iterations"]) for row in cell),
                "physical_terminal_routes": sum(
                    int(row["physical_terminal_evaluations"]) for row in cell
                ),
            }
        )
    phase56_runner.atomic_csv(
        output_root / "metrics/trigger_depth_breakdown.csv", depth_rows
    )

    single_action_rows = []
    for action in ("READ_ONLY", "WRITE_ONLY", "IGNORE"):
        routes = [row for row in single_routes if row["intervention_action"] == action]
        delays = [int(row["intervention_layer"]) - int(row["trigger_layer"]) for row in routes]
        single_action_rows.append(
            {
                "action": action,
                "successful_routes": len(routes),
                "unique_samples": len({str(row["uid"]) for row in routes}),
                "fraction_of_successful_single_routes": _rate(len(routes), len(single_routes)),
                "mean_trigger_to_intervention_delay": (
                    sum(delays) / len(delays) if delays else None
                ),
                "median_trigger_to_intervention_delay": (
                    float(phase56_runner.np.median(delays)) if delays else None
                ),
                "routes_per_single_fixable_sample": _rate(
                    len(routes), len(single_results)
                ),
            }
        )
    phase56_runner.atomic_csv(
        output_root / "metrics/single_action_distribution.csv", single_action_rows
    )
    mcts_complexity_rows = []
    for label, predicate in (
        ("1-2", lambda value: value <= 2),
        ("3", lambda value: value == 3),
        ("4+", lambda value: value >= 4),
    ):
        routes = [row for row in mcts_routes if predicate(int(row["non_full_count"]))]
        mcts_complexity_rows.append(
            {
                "non_full_count_bin": label,
                "routes": len(routes),
                "fraction": _rate(len(routes), len(mcts_routes)),
                "unique_samples": len({str(row["uid"]) for row in routes}),
                "mean_first_success_iteration": (
                    sum(int(row["discovery_iteration"]) for row in routes) / len(routes)
                    if routes
                    else None
                ),
                "routes_per_mcts_fixable_sample": _rate(len(routes), len(mcts_results)),
                "read_only_actions": sum(int(row["read_only_count"]) for row in routes),
                "write_only_actions": sum(int(row["write_only_count"]) for row in routes),
                "ignore_actions": sum(int(row["ignore_count"]) for row in routes),
            }
        )
    phase56_runner.atomic_csv(
        output_root / "metrics/mcts_route_complexity.csv", mcts_complexity_rows
    )

    dense_outputs = read_jsonl(output_root / "dense/dense_outputs.jsonl")
    stage1_completions = [
        read_json(output_root / f"work/stage1_scores/rank{rank:02d}.complete.json")
        for rank in range(4)
    ]
    search_completions = [
        read_json(output_root / f"work/full/rank{rank:02d}/complete.json")
        for rank in range(4)
    ]
    single_terminal = sum(int(row["single_routes_evaluated"]) for row in results)
    physical_terminal = sum(int(row["physical_terminal_evaluations"]) for row in results)
    mcts_terminal = max(0, physical_terminal - single_terminal)
    compute_rows = [
        {"metric": "dense_completed_samples", "value": len(dense_outputs), "unit": "samples"},
        {"metric": "dense_sample_gpu_seconds", "value": sum(float(row["elapsed_seconds"]) for row in dense_outputs), "unit": "gpu_seconds"},
        {"metric": "stage1_scoring_gpu_seconds", "value": sum(float(row["elapsed_seconds"]) for row in stage1_completions), "unit": "gpu_seconds"},
        {"metric": "single_search_terminal_routes", "value": single_terminal, "unit": "routes"},
        {"metric": "mcts_iterations", "value": sum(int(row["mcts_iterations"]) for row in results), "unit": "iterations"},
        {"metric": "mcts_physical_terminal_routes", "value": mcts_terminal, "unit": "routes"},
        {"metric": "all_search_physical_terminal_routes", "value": physical_terminal, "unit": "routes"},
        {"metric": "successful_route_replay_validations", "value": len(new_routes), "unit": "routes"},
        {"metric": "search_sample_gpu_seconds", "value": sum(float(row["fresh_elapsed_seconds"]) for row in results), "unit": "gpu_seconds"},
        {"metric": "search_sample_gpu_hours", "value": sum(float(row["fresh_elapsed_seconds"]) for row in results) / 3600, "unit": "gpu_hours"},
        {"metric": "max_search_worker_wall_seconds", "value": max(float(row["worker_elapsed_seconds"]) for row in search_completions), "unit": "seconds"},
    ]
    phase56_runner.atomic_csv(output_root / "metrics/compute_summary.csv", compute_rows)

    phase56_runner._plot_stacked_fixability(
        output_root / "figures/label_yield_old_vs_new.png",
        ["Old", "New"],
        [old_yield["p_single_given_triggered_w"], new_yield["p_single_given_triggered_w"]],
        [old_yield["p_mcts_only_given_triggered_w"], new_yield["p_mcts_only_given_triggered_w"]],
        "Bounded label yield: old versus new",
    )
    phase56_runner._plot_stacked_fixability(
        output_root / "figures/dataset_label_yield.png",
        [row["dataset"] for row in dataset_rows],
        [row["single_fixable_rate"] for row in dataset_rows],
        [row["mcts_only_fixable_rate"] for row in dataset_rows],
        "New bounded label yield by dataset",
    )
    phase56_runner._plot_composition(
        output_root / "figures/fixability_composition.png",
        ["Single", "MCTS only", "Unresolved"],
        [
            _rate(len(single_results), new_counts[("W", "trigger")]),
            _rate(len(mcts_results), new_counts[("W", "trigger")]),
            _rate(len(unresolved_results), new_counts[("W", "trigger")]),
        ],
        "New triggered-W bounded fixability",
    )
    phase56_runner._plot_stacked_fixability(
        output_root / "figures/trigger_depth_yield.png",
        [row["trigger_depth_bin"] for row in depth_rows],
        [row["single_fixable_rate"] for row in depth_rows],
        [row["mcts_only_fixable_rate"] for row in depth_rows],
        "New bounded fixability by trigger depth",
    )
    phase56_runner._plot_composition(
        output_root / "figures/single_action_distribution.png",
        [row["action"] for row in single_action_rows],
        [row["fraction_of_successful_single_routes"] for row in single_action_rows],
        "New successful single actions",
    )
    phase56_runner._plot_composition(
        output_root / "figures/mcts_route_complexity.png",
        [row["non_full_count_bin"] for row in mcts_complexity_rows],
        [row["fraction"] for row in mcts_complexity_rows],
        "New successful MCTS route complexity",
    )

    tolerance = float(
        config["reporting_rules"]["old_pattern_absolute_fixability_rate_tolerance"]
    )
    pattern_delta = (
        new_yield["p_bounded_fixable_given_triggered_w"]
        - old_yield["p_bounded_fixable_given_triggered_w"]
    )
    pattern_reproduced = abs(pattern_delta) <= tolerance
    gqa = next(row for row in dataset_rows if row["dataset"] == "gqa")
    gqa_new_fixable = int(gqa["total_bounded_fixable"])
    gqa_material = gqa_new_fixable >= int(
        config["reporting_rules"]["gqa_material_minimum_new_fixable_bases"]
    )
    summary = f"""# Stage-2 data-scale corrective-search summary

Contract: `{contract['contract_sha256']}`. Candidate selection was frozen before dense outcomes and contains no legacy UID or SHA-256 image-group overlap.

1. **New candidates processed:** {len(trigger_map):,} completed dense/Stage-1 rows from 4,000 frozen candidates; {4000 - len(trigger_map):,} dense skips are retained in the bound dense audit.
2. **New triggered Dense-W:** {new_counts[("W", "trigger")]:,}.
3. **SINGLE_FIXABLE:** {len(single_results):,} bases and {len(single_routes):,} exact-replay routes.
4. **Additional MCTS_ONLY_FIXABLE:** {len(mcts_results):,} bases and {len(mcts_routes):,} retained exact-replay routes.
5. **UNRESOLVED:** {len(unresolved_results):,}; this means unresolved under the frozen bounded search, not unfixable.
6. **New triggered Dense-C preservation:** {len(preservation_results):,} bases/routes.
7. **Expanded corpora:** A={len(expanded_a):,} routes/{len({str(row['uid']) for row in expanded_a}):,} bases; B={len(expanded_b):,} routes/{len({str(row['uid']) for row in expanded_b}):,} bases; C={len(expanded_c):,} routes/{len({str(row['uid']) for row in expanded_c}):,} bases.
8. **Old label-yield pattern:** {'reproduced within the prospectively frozen ±' + f'{tolerance:.2f}' if pattern_reproduced else 'not reproduced within the prospectively frozen tolerance'}; new-minus-old bounded fixability is {pattern_delta:+.4f}. This is a canonical-source expansion, not a pure scale-only replication.
9. **GQA supervision:** {gqa_new_fixable:,} new bounded-fixable GQA bases; {'material by' if gqa_material else 'below'} the prospective ≥{config['reporting_rules']['gqa_material_minimum_new_fixable_bases']} new-base criterion (old GQA bounded-fixable bases: 85).
10. **Trigger-depth variation:** new total bounded-fixability rates span {max(row['total_bounded_fixable_rate'] for row in depth_rows) - min(row['total_bounded_fixable_rate'] for row in depth_rows):.4f}; exact counts/costs are in `metrics/trigger_depth_breakdown.csv`.
11. **Single-search compute:** {single_terminal:,} exhaustive single terminal routes.
12. **Additional MCTS compute:** {sum(int(row['mcts_iterations']) for row in results):,} iterations and {mcts_terminal:,} physical terminal routes; total search sample time was {sum(float(row['fresh_elapsed_seconds']) for row in results) / 3600:.3f} GPU-hours.
13. **Integrity:** PASS. All {len(new_routes):,} retained new routes are current-LMMS-correct exact-token replays; all {len(state_shards):,} new state shards are contract/model/code/schema/source-bound. Old Phase-56 inputs passed their frozen artifact-manifest verification. No validation/test sample was searched.

No Stage-2 model was trained, Stage 1 was not retuned, and no validation/test/external evaluation was run.
"""
    _atomic_bytes(output_root / "summaries/data_scale_search_summary.md", summary.encode())
    healthy = bool(preservation_results and single_results)
    recommendation = f"""# Next training recommendation

Expanded-corpus health for the planned comparison: **{'PASS' if healthy else 'NOT ESTABLISHED'}**.

If separately authorized, keep the frozen V1 architecture, optimizer/loss, Stage-1 gate, validation set, and free-rollout evaluator fixed:

1. **Scaled-Single:** Expanded A + Expanded B; exclude Corpus C.
2. **Scaled-Single + MCTS:** the same setup with Expanded C added.

Use a matched optimizer-update budget to the frozen V1-698 run or report the exact difference. This file is a recommendation only; neither training experiment was launched.
"""
    _atomic_bytes(output_root / "summaries/next_training_recommendation.md", recommendation.encode())

    required = [
        "protocol.md",
        "frozen_protocol.json",
        "work/search_contract.json",
        "work/dense_binding.json",
        "work/trigger_binding.json",
        "manifests/new_candidate_manifest.jsonl",
        "manifests/new_dense_results.jsonl",
        "manifests/new_trigger_map.jsonl",
        "manifests/new_triggered_correct.jsonl",
        "manifests/new_triggered_wrong.jsonl",
        "manifests/new_single_fixable.jsonl",
        "manifests/new_mcts_only_fixable.jsonl",
        "manifests/new_unresolved.jsonl",
        "manifests/new_preservation_results.jsonl",
        "routes/new_successful_single_routes.jsonl",
        "routes/new_successful_mcts_routes.jsonl",
        "states/feature_schema.json",
        "states/new_state_index.jsonl",
        "combined_corpora/expanded_corpus_A_preservation.jsonl",
        "combined_corpora/expanded_corpus_B_single.jsonl",
        "combined_corpora/expanded_corpus_C_mcts.jsonl",
        "combined_corpora/expanded_unresolved.jsonl",
        "combined_corpora/corpus_manifest.json",
        "metrics/scaleup_summary.csv",
        "metrics/old_vs_new_yield.csv",
        "metrics/dataset_breakdown.csv",
        "metrics/trigger_depth_breakdown.csv",
        "metrics/single_action_distribution.csv",
        "metrics/mcts_route_complexity.csv",
        "metrics/compute_summary.csv",
        "figures/label_yield_old_vs_new.png",
        "figures/dataset_label_yield.png",
        "figures/fixability_composition.png",
        "figures/trigger_depth_yield.png",
        "figures/single_action_distribution.png",
        "figures/mcts_route_complexity.png",
        "summaries/data_scale_search_summary.md",
        "summaries/next_training_recommendation.md",
    ]
    missing = [relative for relative in required if not (output_root / relative).is_file()]
    if missing:
        raise RuntimeError(f"required scale-up artifacts are missing: {missing}")
    files = {relative: file_sha256(output_root / relative) for relative in required}
    files.update(state_shards)
    artifact_manifest = {
        "schema_version": "stage2_data_scale_search_artifact_manifest_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "candidate_records": 4000,
        "dense_completed": len(trigger_map),
        "outcome_counts": dict(classes),
        "files": dict(sorted(files.items())),
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "artifact_manifest.json", artifact_manifest)
    verify_artifact_manifest(output_root, artifact_manifest)
    print(
        json.dumps(
            {
                "passed": True,
                "contract_sha256": contract["contract_sha256"],
                "outcome_counts": dict(classes),
                "new_routes": len(new_routes),
                "expanded_route_records": {
                    "A": len(expanded_a),
                    "B": len(expanded_b),
                    "C": len(expanded_c),
                },
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "preselect-chartqa",
            "download-chartqa",
            "prepare-candidates",
            "freeze-contract",
            "dense-worker",
            "aggregate-dense",
            "bind-dense",
            "stage1-worker",
            "aggregate-trigger-map",
            "prepare-search",
            "search-worker",
            "finalize-search-smoke",
            "aggregate",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--mode", choices=("smoke", "full"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preselect-chartqa":
        preselect_chartqa(args.config)
    elif args.command == "download-chartqa":
        download_chartqa(args.config)
    elif args.command == "prepare-candidates":
        prepare_candidates(args.config)
    elif args.command == "freeze-contract":
        freeze_contract(args.config)
    elif args.command == "dense-worker":
        if args.mode is None:
            raise SystemExit("dense-worker requires --mode")
        dense_execute(args.config, mode=args.mode)
    elif args.command == "aggregate-dense":
        if args.mode is None:
            raise SystemExit("aggregate-dense requires --mode")
        aggregate_dense(args.config, mode=args.mode)
    elif args.command == "bind-dense":
        bind_dense(args.config)
    elif args.command == "stage1-worker":
        if args.rank is None:
            raise SystemExit("stage1-worker requires --rank")
        stage1_worker(args.config, rank=args.rank, world_size=args.world_size)
    elif args.command == "aggregate-trigger-map":
        aggregate_trigger_map(args.config)
    elif args.command == "prepare-search":
        prepare_search(args.config)
    elif args.command == "search-worker":
        if args.rank is None or args.mode is None:
            raise SystemExit("search-worker requires --rank and --mode")
        search_worker(
            args.config,
            mode=args.mode,
            rank=args.rank,
            world_size=args.world_size,
            resume=args.resume,
        )
    elif args.command == "finalize-search-smoke":
        finalize_search_smoke(args.config)
    else:
        aggregate(args.config)


if __name__ == "__main__":
    main()
