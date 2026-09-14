#!/usr/bin/env python3
"""Prepare, freeze, smoke, and run Predictability Step-C generalization fits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dense_failure_stage2.predictability_generalization import (  # noqa: E402
    assign_clusters_to_folds,
    cosine_nearest,
    equal_frequency_bins,
    label_blind_inner_roles,
    last_token_pool,
    normalize_question,
    spherical_kmeans,
)
from experiments import run_predictability_stepB_learnability as stepb  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/predictability_stepC_generalization_v1.json"
BOUND_CODE = (
    "configs/predictability_stepC_generalization_v1.json",
    "dense_failure_stage2/predictability_generalization.py",
    "experiments/run_predictability_stepC_generalization.py",
    "experiments/aggregate_predictability_stepC_generalization.py",
    "experiments/run_predictability_stepB_learnability.py",
    "dense_failure_stage2/predictability_learnability.py",
    "dense_failure_stage1/shared_global_gate.py",
    "dense_failure_stage2/v1_router.py",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(value: str | Path) -> dict[str, Any]:
    with resolve_path(value).open() as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {value}")
    return result


def read_jsonl(value: str | Path) -> list[dict[str, Any]]:
    rows = []
    with resolve_path(value).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected JSON object at {value}:{line_number}")
                rows.append(row)
    return rows


def file_sha256(value: str | Path) -> str:
    from hashlib import sha256

    digest = sha256()
    with resolve_path(value).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    from hashlib import sha256

    payload = {
        key: item for key, item in value.items()
        if key not in {"contract_sha256", "artifact_manifest_sha256"}
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic(path, "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows).encode())


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".npy") as handle:
        temporary = Path(handle.name)
    np.save(temporary, value, allow_pickle=False)
    os.replace(temporary, path)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def git_state() -> dict[str, str]:
    def output(*arguments: str) -> str:
        return subprocess.run(
            arguments, cwd=REPO_ROOT, check=True, text=True, capture_output=True
        ).stdout.strip()

    return {
        "commit": output("git", "rev-parse", "HEAD"),
        "branch": output("git", "branch", "--show-current"),
        "worktree_status": output("git", "status", "--short"),
    }


def runtime_state() -> dict[str, Any]:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def load_config(value: str | Path) -> dict[str, Any]:
    config = read_json(value)
    if config.get("schema_version") != "predictability_stepC_generalization_config_v1":
        raise ValueError("unsupported Step-C config schema")
    if int(config["world_size"]) != 4:
        raise ValueError("Step-C requires four direct GPU workers")
    return config


def _verify_parent_artifact(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("artifact_manifest_sha256") != canonical_hash(manifest):
        raise RuntimeError(f"parent artifact manifest self-hash differs: {manifest_path}")
    for relative, digest in manifest["files"].items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"parent artifact differs: {path}")
    return manifest


def _source_population(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    stepb_contract = read_json(config["parent"]["stepB_contract"])
    source = stepb_contract["static_config"]["sources"]["internal_samples"]
    rows = read_jsonl(source)
    output = []
    seen = set()
    for row in rows:
        uid = str(row["uid"])
        if uid in seen:
            raise RuntimeError(f"duplicate source UID: {uid}")
        seen.add(uid)
        question = normalize_question(str(row["question"]))
        if not question:
            raise RuntimeError(f"empty question after normalization: {uid}")
        output.append(
            {
                "uid": uid,
                "image_group_id": str(row["image_group_id"]),
                "dataset": str(row["dataset"]),
                "source_regime": str(row["source_regime"]),
                "p90_triggered": bool(row["p90"]["triggered"]),
                "question": question,
            }
        )
    if len(output) != 10_399:
        raise RuntimeError(f"internal population differs: {len(output)}")
    return sorted(output, key=lambda row: str(row["uid"]))


def prepare_encoder(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    external_root.mkdir(parents=True, exist_ok=True)
    step_a_root = resolve_path(config["parent"]["stepA_contract"]).parent
    step_b_root = resolve_path(config["parent"]["stepB_root"])
    step_a_manifest = _verify_parent_artifact(
        step_a_root, resolve_path(config["parent"]["stepA_artifact_manifest"])
    )
    step_b_manifest = _verify_parent_artifact(
        step_b_root, resolve_path(config["parent"]["stepB_artifact_manifest"])
    )
    step_b_contract = read_json(config["parent"]["stepB_contract"])
    if step_b_contract.get("contract_sha256") != canonical_hash(step_b_contract):
        raise RuntimeError("parent Step-B contract self-hash differs")
    if not step_b_manifest.get("ready_for_step_c"):
        raise RuntimeError("parent Step-B did not authorize Step-C")
    population = _source_population(config)
    snapshot = resolve_path(config["encoder"]["snapshot"])
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    snapshot_hashes = {}
    for name in required:
        path = snapshot / name
        if not path.is_file() or not os.access(path, os.R_OK):
            raise RuntimeError(f"encoder snapshot incomplete: {path}")
        snapshot_hashes[name] = file_sha256(path)
    for optional in ("merges.txt", "vocab.json", "special_tokens_map.json"):
        if (snapshot / optional).is_file():
            snapshot_hashes[optional] = file_sha256(snapshot / optional)
    contract = {
        "schema_version": "predictability_stepC_encoder_contract_v1",
        "created_at": utc_now(),
        "encoder": config["encoder"],
        "snapshot_files_sha256": snapshot_hashes,
        "population_uids": len(population),
        "uid_order_sha256": canonical_hash({"uids": [row["uid"] for row in population]}),
        "parent_stepA_artifact_manifest_sha256": step_a_manifest["artifact_manifest_sha256"],
        "parent_stepB_contract_sha256": step_b_contract["contract_sha256"],
        "parent_stepB_artifact_manifest_sha256": step_b_manifest["artifact_manifest_sha256"],
        "config_sha256": file_sha256(config_path),
        "review_reconciliation": {
            "review_verdict": "revise",
            "selected": "raw_transformers_qwen3_with_one_shared_symmetric_instruction",
            "reason": "respects_instruction_aware_model_card_without_query_document_asymmetry_or_new_dependency",
            "post_result_prompt_comparison": False,
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "question_semantics/encoder_contract.json", contract)
    atomic_jsonl(output_root / "work/question_population.jsonl", population)
    print(json.dumps({"prepared": True, "encoder_contract_sha256": contract["contract_sha256"]}))


def _verify_encoder(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "question_semantics/encoder_contract.json")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise RuntimeError("encoder contract self-hash differs")
    if contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("config changed after encoder freeze")
    snapshot = resolve_path(config["encoder"]["snapshot"])
    for name, digest in contract["snapshot_files_sha256"].items():
        if file_sha256(snapshot / name) != digest:
            raise RuntimeError(f"encoder snapshot changed: {name}")
    rows = read_jsonl(output_root / "work/question_population.jsonl")
    if len(rows) != int(contract["population_uids"]):
        raise RuntimeError("question population changed")
    return config, contract, output_root


def _load_encoder(config: Mapping[str, Any], device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    snapshot = str(resolve_path(config["encoder"]["snapshot"]))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.padding_side = str(config["encoder"]["padding_side"])
    model = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=str(config["encoder"]["attention_implementation"]),
    ).to(device).eval()
    return tokenizer, model


def _encode_questions(
    questions: Sequence[str], config: Mapping[str, Any], tokenizer: Any, model: Any,
    device: torch.device,
) -> np.ndarray:
    encoder = config["encoder"]
    texts = [
        str(encoder["input_template"]).format(
            instruction=encoder["shared_instruction"], question=question
        )
        for question in questions
    ]
    pieces = []
    batch_size = int(encoder["batch_size_per_gpu"])
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokenized = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=bool(encoder["truncation"]),
                max_length=int(encoder["max_length"]),
                return_tensors="pt",
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            hidden = model(**tokenized).last_hidden_state
            pooled = last_token_pool(hidden, tokenized["attention_mask"]).float()
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            pieces.append(pooled.cpu().numpy())
    return np.concatenate(pieces).astype(np.float32, copy=False)


def embed_worker(config_path: Path, rank: int) -> None:
    config, contract, output_root = _verify_encoder(config_path)
    if not 0 <= rank < int(config["world_size"]):
        raise ValueError("invalid embedding worker rank")
    torch.set_num_threads(int(config["execution_runtime"]["torch_intraop_threads_per_worker"]))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tokenizer, model = _load_encoder(config, device)
    rows = read_jsonl(output_root / "work/question_population.jsonl")
    if rank == 0:
        smoke_rows = sorted(
            rows,
            key=lambda row: __import__("hashlib").sha256(
                f"stepc-smoke|{row['uid']}".encode()
            ).hexdigest(),
        )[: int(config["smoke"]["embedding_uids"])]
        first = _encode_questions([row["question"] for row in smoke_rows], config, tokenizer, model, device)
        second = _encode_questions([row["question"] for row in smoke_rows], config, tokenizer, model, device)
        exact = bool(np.array_equal(first, second))
        unit = bool(np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=2e-5))
        if not exact or not unit:
            raise RuntimeError(f"embedding smoke failed: exact={exact}, unit={unit}")
        atomic_json(
            output_root / "validation/embedding_smoke.json",
            {
                "schema_version": "predictability_stepC_embedding_smoke_v1",
                "encoder_contract_sha256": contract["contract_sha256"],
                "uids": [row["uid"] for row in smoke_rows],
                "exact_repeat": exact,
                "unit_norm": unit,
                "maximum_repeat_absolute_difference": float(np.max(np.abs(first - second))),
            },
        )
    indices = np.arange(rank, len(rows), int(config["world_size"]), dtype=np.int64)
    embeddings = _encode_questions(
        [rows[int(index)]["question"] for index in indices], config, tokenizer, model, device
    )
    shard = output_root / f"work/embeddings/rank{rank:02d}.npz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=shard.parent, delete=False, suffix=".npz") as handle:
        temporary = Path(handle.name)
    np.savez(temporary, indices=indices, embeddings=embeddings)
    os.replace(temporary, shard)
    atomic_json(
        output_root / f"work/embeddings/rank{rank:02d}.json",
        {
            "encoder_contract_sha256": contract["contract_sha256"],
            "rank": rank,
            "rows": len(indices),
            "shard": str(shard),
            "shard_sha256": file_sha256(shard),
        },
    )
    print(json.dumps({"rank": rank, "embedded": len(indices)}))


def _regime_definitions(
    population: Sequence[Mapping[str, Any]], cluster_fold_by_uid: Mapping[str, int]
) -> list[dict[str, Any]]:
    uids = {str(row["uid"]) for row in population}
    by_source = defaultdict(set)
    by_dataset = defaultdict(set)
    for row in population:
        by_source[str(row["source_regime"])].add(str(row["uid"]))
        by_dataset[str(row["dataset"])].add(str(row["uid"]))
    regimes = []
    for fold in range(5):
        test = {uid for uid, value in cluster_fold_by_uid.items() if value == fold}
        regimes.append({"regime_id": f"cluster_fold{fold}", "family": "cluster_ood", "train": uids - test, "test": test})
    for train, test in (("historical", "canonical"), ("canonical", "historical")):
        regimes.append({"regime_id": f"source_{train}_to_{test}", "family": "source_global", "train": by_source[train], "test": by_source[test]})
    for dataset in ("gqa", "chartqa", "textvqa"):
        dataset_uids = by_dataset[dataset]
        for train, test in (("historical", "canonical"), ("canonical", "historical")):
            regimes.append(
                {
                    "regime_id": f"source_{dataset}_{train}_to_{test}",
                    "family": "source_within_dataset",
                    "train": dataset_uids & by_source[train],
                    "test": dataset_uids & by_source[test],
                }
            )
    datasets = ("gqa", "chartqa", "textvqa")
    for test in datasets:
        train = set().union(*(by_dataset[value] for value in datasets if value != test))
        regimes.append({"regime_id": f"lodo_to_{test}", "family": "lodo", "train": train, "test": by_dataset[test]})
    for train in datasets:
        for test in datasets:
            if train != test:
                regimes.append({"regime_id": f"dataset_{train}_to_{test}", "family": "pairwise", "train": by_dataset[train], "test": by_dataset[test]})
    return regimes


def _support(
    roles: Sequence[Mapping[str, Any]], data: Mapping[str, Any], target: str,
    config: Mapping[str, Any], regime_id: str, domain: str,
) -> dict[str, Any]:
    role_by_uid = {str(row["uid"]): str(row["role"]) for row in roles}
    target_values = np.asarray(data["targets"][target], dtype=np.float64)
    uid_target: dict[str, list[float]] = defaultdict(list)
    uid_role: dict[str, str] = {}
    for index, row in enumerate(data["rows"]):
        uid = str(row["uid"])
        role = role_by_uid[uid]
        if role != "excluded":
            uid_role[uid] = role
            uid_target[uid].append(float(target_values[index]))
    counts = Counter(uid_role.values())
    thresholds = config["semantic"]["support"]
    reasons = []
    for role, key in (("fit", "minimum_fit_uids"), ("calibration", "minimum_calibration_uids"), ("outer_test", "minimum_test_uids")):
        if counts[role] < int(thresholds[key]):
            reasons.append(f"{role}_uids<{thresholds[key]}")
        if bool(thresholds["require_nonconstant_target"]):
            values = [value for uid, pieces in uid_target.items() if uid_role[uid] == role for value in pieces]
            if len(set(values)) < 2:
                reasons.append(f"{role}_target_constant")
    return {
        "regime_id": regime_id,
        "domain": domain,
        "target": target,
        "fit_uids": counts["fit"],
        "calibration_uids": counts["calibration"],
        "test_uids": counts["outer_test"],
        "supported": not reasons,
        "reasons": ";".join(reasons),
    }


def _tasks(config: Mapping[str, Any], supports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seeds = list(config["training"]["seeds"])
    base_seed = int(config["seed"])
    tasks = []
    fold_by_regime = {row["regime_id"]: int(row["regime_index"]) for row in config["regimes"]}
    for support in supports:
        if not support["supported"]:
            continue
        domain, target, regime = str(support["domain"]), str(support["target"]), str(support["regime_id"])
        fold = fold_by_regime[regime]
        specs = (
            [("m0_nuisance", "nuisance", base_seed, 1), ("m1_linear", "state", base_seed, 3)]
            if domain == "stage1"
            else [("m0_nuisance", "nuisance", base_seed, 1), ("m1_text_visual", "text_visual", base_seed, 3)]
        )
        specs += (
            [("m3_current_head", "state_layer", seed, 12) for seed in seeds]
            if domain == "stage1"
            else [("m3_z_RW", "z_RW", seed, 20) for seed in seeds]
        )
        for model, input_name, seed, cost in specs:
            task_id = f"{regime}__{domain}__{target}__{model}__s{seed}"
            tasks.append(
                {
                    "task_id": task_id,
                    "regime_id": regime,
                    "regime_family": next(row["family"] for row in config["regimes"] if row["regime_id"] == regime),
                    "domain": domain,
                    "fold": fold,
                    "model": model,
                    "input": input_name,
                    "target": target,
                    "seed": int(seed),
                    "estimated_cost": cost,
                }
            )
    loads = [0] * int(config["world_size"])
    for task in sorted(tasks, key=lambda row: (-int(row["estimated_cost"]), str(row["task_id"]))):
        rank = min(range(len(loads)), key=lambda value: (loads[value], value))
        task["worker_rank"] = rank
        loads[rank] += int(task["estimated_cost"])
    return sorted(tasks, key=lambda row: str(row["task_id"]))


def finalize_semantics(config_path: Path) -> None:
    config, encoder_contract, output_root = _verify_encoder(config_path)
    population = read_jsonl(output_root / "work/question_population.jsonl")
    smoke = read_json(output_root / "validation/embedding_smoke.json")
    if smoke.get("encoder_contract_sha256") != encoder_contract["contract_sha256"] or not smoke.get("exact_repeat"):
        raise RuntimeError("embedding smoke missing or failed")
    embeddings = None
    seen = set()
    shard_hashes = {}
    for rank in range(int(config["world_size"])):
        meta_path = output_root / f"work/embeddings/rank{rank:02d}.json"
        meta = read_json(meta_path)
        shard = resolve_path(meta["shard"])
        if meta.get("encoder_contract_sha256") != encoder_contract["contract_sha256"] or file_sha256(shard) != meta.get("shard_sha256"):
            raise RuntimeError(f"embedding shard {rank} provenance differs")
        payload = np.load(shard)
        indices = payload["indices"]
        current = payload["embeddings"]
        if embeddings is None:
            embeddings = np.empty((len(population), current.shape[1]), dtype=np.float32)
        for index, vector in zip(indices, current, strict=True):
            if int(index) in seen:
                raise RuntimeError("duplicate embedding index")
            seen.add(int(index))
            embeddings[int(index)] = vector
        shard_hashes[str(shard)] = meta["shard_sha256"]
    assert embeddings is not None
    if seen != set(range(len(population))) or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=2e-5):
        raise RuntimeError("embedding merge incomplete or not normalized")
    uid_path = output_root / "question_semantics/uid_question_embeddings.npy"
    atomic_npy(uid_path, embeddings)
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(population):
        by_group[str(row["image_group_id"])].append(index)
    group_ids = sorted(by_group)
    group_embeddings = np.stack([embeddings[by_group[group]].mean(axis=0) for group in group_ids])
    group_embeddings /= np.linalg.norm(group_embeddings, axis=1, keepdims=True)
    group_path = output_root / "question_semantics/group_question_embeddings.npy"
    atomic_npy(group_path, group_embeddings.astype(np.float32))
    atomic_jsonl(
        output_root / "question_semantics/uid_embedding_index.jsonl",
        ({"row_index": index, **row} for index, row in enumerate(population)),
    )
    atomic_jsonl(
        output_root / "question_semantics/group_embedding_index.jsonl",
        ({"row_index": index, "image_group_id": group} for index, group in enumerate(group_ids)),
    )
    step_b_root = resolve_path(config["parent"]["stepB_root"])
    folds = {str(row["uid"]): int(row["fold"]) for row in read_jsonl(step_b_root / "splits/group_fold_registry.jsonl")}
    uid_index = {str(row["uid"]): index for index, row in enumerate(population)}
    nearest_rows = []
    for fold in range(5):
        query_uids = [uid for uid in uid_index if folds[uid] == fold]
        train_uids = [uid for uid in uid_index if folds[uid] != fold]
        scores, neighbors = cosine_nearest(
            embeddings[[uid_index[uid] for uid in query_uids]],
            embeddings[[uid_index[uid] for uid in train_uids]],
        )
        nearest_rows.extend(
            {
                "uid": uid,
                "outer_fold": fold,
                "nearest_train_uid": train_uids[int(neighbor)],
                "nearest_train_similarity": float(score),
            }
            for uid, score, neighbor in zip(query_uids, scores, neighbors, strict=True)
        )
    nearest_rows.sort(key=lambda row: str(row["uid"]))
    bins = equal_frequency_bins([row["nearest_train_similarity"] for row in nearest_rows], bins=int(config["semantic"]["similarity_bins"]))
    for row, bin_id in zip(nearest_rows, bins, strict=True):
        row["similarity_bin"] = f"Q{int(bin_id)}"
    atomic_jsonl(output_root / "question_semantics/nearest_train_similarity.jsonl", nearest_rows)
    stepb.atomic_csv(
        output_root / "question_semantics/similarity_bins.csv",
        [
            {
                "similarity_bin": f"Q{value}",
                "minimum": min(row["nearest_train_similarity"] for row in nearest_rows if row["similarity_bin"] == f"Q{value}"),
                "maximum": max(row["nearest_train_similarity"] for row in nearest_rows if row["similarity_bin"] == f"Q{value}"),
                "uids": sum(row["similarity_bin"] == f"Q{value}" for row in nearest_rows),
            }
            for value in range(1, int(config["semantic"]["similarity_bins"]) + 1)
        ],
    )
    assignments, _, iterations = spherical_kmeans(
        group_embeddings,
        clusters=int(config["semantic"]["cluster_count"]),
        seed=int(config["seed"]),
        maximum_iterations=int(config["semantic"]["cluster_maximum_iterations"]),
    )
    population_by_uid = {str(row["uid"]): row for row in population}
    group_rows = []
    cluster_balance_rows = []
    for group_index, group in enumerate(group_ids):
        members = [population[index] for index in by_group[group]]
        cluster = int(assignments[group_index])
        group_rows.append({"image_group_id": group, "cluster_id": cluster, "uids": [row["uid"] for row in members]})
        for dataset in ("gqa", "chartqa", "textvqa"):
            local = [row for row in members if row["dataset"] == dataset]
            if local:
                cluster_balance_rows.append(
                    {"cluster_id": cluster, "dataset": dataset, "groups": 1, "p90_groups": int(any(row["p90_triggered"] for row in local))}
                )
    cluster_to_fold = assign_clusters_to_folds(cluster_balance_rows, folds=5, seed=int(config["seed"]))
    cluster_fold_by_uid = {}
    assignment_rows = []
    for group_row in group_rows:
        fold = cluster_to_fold[int(group_row["cluster_id"])]
        for uid in group_row["uids"]:
            cluster_fold_by_uid[str(uid)] = fold
            row = population_by_uid[str(uid)]
            assignment_rows.append(
                {"uid": uid, "image_group_id": group_row["image_group_id"], "cluster_id": group_row["cluster_id"], "cluster_fold": fold, "dataset": row["dataset"], "source_regime": row["source_regime"]}
            )
    atomic_jsonl(output_root / "question_cluster_ood/cluster_assignments.jsonl", sorted(assignment_rows, key=lambda row: row["uid"]))
    atomic_jsonl(output_root / "question_cluster_ood/cluster_fold_registry.jsonl", sorted(assignment_rows, key=lambda row: row["uid"]))
    regimes = _regime_definitions(population, cluster_fold_by_uid)
    augmented = copy.deepcopy(config)
    parent_stepb = read_json(config["parent"]["stepB_contract"])
    for key in ("inputs", "models", "training", "targets"):
        augmented[key] = parent_stepb["static_config"][key]
    augmented["regimes"] = [
        {"regime_index": index, "regime_id": row["regime_id"], "family": row["family"], "train_uids": len(row["train"]), "test_uids": len(row["test"])}
        for index, row in enumerate(regimes)
    ]
    parent_data = {
        domain: stepb._domain_data(parent_stepb, step_b_root, domain)
        for domain in ("stage1", "stage2_dense")
    }
    supports = []
    split_hashes = {}
    for index, regime in enumerate(regimes):
        roles = label_blind_inner_roles(
            population,
            eligible_train_uids=set(regime["train"]),
            test_uids=set(regime["test"]),
            calibration_fraction=float(config["semantic"]["inner_calibration_fraction"]),
            seed=int(config["seed"]) + index,
        )
        path = output_root / f"splits/inner_roles_fold{index}.jsonl"
        atomic_jsonl(path, roles)
        split_hashes[str(path.relative_to(output_root))] = file_sha256(path)
        supports.append(_support(roles, parent_data["stage1"], "dense_wrong", augmented, regime["regime_id"], "stage1"))
        for target in ("read", "write"):
            supports.append(_support(roles, parent_data["stage2_dense"], target, augmented, regime["regime_id"], "stage2_dense"))
    for row in supports:
        row["regime_index"] = next(item["regime_index"] for item in augmented["regimes"] if item["regime_id"] == row["regime_id"])
    stepb.atomic_csv(output_root / "splits/support_audit.csv", supports)
    tasks = _tasks(augmented, supports)
    atomic_jsonl(output_root / "work/training_tasks.jsonl", tasks)
    semantic_paths = (
        "question_semantics/encoder_contract.json",
        "question_semantics/uid_question_embeddings.npy",
        "question_semantics/group_question_embeddings.npy",
        "question_semantics/uid_embedding_index.jsonl",
        "question_semantics/group_embedding_index.jsonl",
        "question_semantics/nearest_train_similarity.jsonl",
        "question_semantics/similarity_bins.csv",
        "question_cluster_ood/cluster_assignments.jsonl",
        "question_cluster_ood/cluster_fold_registry.jsonl",
        "splits/support_audit.csv",
        "work/training_tasks.jsonl",
    )
    protocol = f"""# Predictability Step-C protocol\n\n- Targets and model/optimization contracts: unchanged from Step B `{parent_stepb['contract_sha256']}`.\n- Question encoder: `{config['encoder']['model_name']}` snapshot `{Path(config['encoder']['snapshot']).name}`.\n- Shared instruction: `{config['encoder']['shared_instruction']}`\n- Semantic clusters: deterministic spherical k-means K=100; balancing is label-blind.\n- OOD regimes: 5 cluster folds, bidirectional global/within-dataset source transfer, 3 LODO, 6 pairwise dataset transfers.\n- Controls: frozen M0 nuisance, M1 linear, M3 current/full-state, exact k=5 question kNN.\n- Unsupported cells are frozen before outcomes and are not relaxed.\n"""
    _atomic(output_root / "protocol.md", protocol.encode())
    semantic_hashes = {path: file_sha256(output_root / path) for path in semantic_paths}
    semantic_hashes["protocol.md"] = file_sha256(output_root / "protocol.md")
    semantic_hashes.update(split_hashes)
    contract = {
        "schema_version": "predictability_stepC_generalization_contract_v1",
        "created_at": utc_now(),
        "static_config": augmented,
        "config_sha256": file_sha256(config_path),
        "encoder_contract_sha256": encoder_contract["contract_sha256"],
        "parent_stepA_contract_sha256": read_json(config["parent"]["stepA_contract"])["contract_sha256"],
        "parent_stepA_artifact_manifest_sha256": read_json(config["parent"]["stepA_artifact_manifest"])["artifact_manifest_sha256"],
        "parent_stepB_contract_sha256": parent_stepb["contract_sha256"],
        "parent_stepB_artifact_manifest_sha256": read_json(config["parent"]["stepB_artifact_manifest"])["artifact_manifest_sha256"],
        "bound_code_sha256": {path: file_sha256(path) for path in BOUND_CODE},
        "semantic_and_split_sha256": semantic_hashes,
        "embedding_shard_sha256": shard_hashes,
        "cluster_iterations": iterations,
        "git": git_state(),
        "runtime": runtime_state(),
        "support_summary": {"rows": len(supports), "supported": sum(bool(row["supported"]) for row in supports), "unsupported": sum(not bool(row["supported"]) for row in supports)},
        "training_tasks": len(tasks),
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_contract.json", contract)
    print(json.dumps({"frozen": True, "contract_sha256": contract["contract_sha256"], "tasks": len(tasks), "unsupported_cells": contract["support_summary"]["unsupported"]}))


def verify_contract(config_path: Path) -> tuple[dict[str, Any], Path, Path, dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_output_root"])
    contract = read_json(output_root / "frozen_contract.json")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise RuntimeError("Step-C contract self-hash differs")
    if contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("Step-C config changed after freeze")
    for path, digest in contract["bound_code_sha256"].items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"Step-C bound code changed: {path}")
    for relative, digest in contract["semantic_and_split_sha256"].items():
        if file_sha256(output_root / relative) != digest:
            raise RuntimeError(f"Step-C semantic/split artifact changed: {relative}")
    if git_state() != contract["git"] or runtime_state() != contract["runtime"]:
        raise RuntimeError("Step-C git/runtime state differs from frozen contract")
    stepb_contract = read_json(config["parent"]["stepB_contract"])
    stepb_root = resolve_path(config["parent"]["stepB_root"])
    return contract, output_root, external_root, stepb_contract, stepb_root


def _parent_data(stepb_contract: Mapping[str, Any], stepb_root: Path, domain: str) -> dict[str, Any]:
    return stepb._domain_data(stepb_contract, stepb_root, domain)


def smoke_worker(config_path: Path, rank: int) -> None:
    contract, output_root, external_root, parent, parent_root = verify_contract(config_path)
    config = contract["static_config"]
    stepb.configure_worker_runtime(config)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    smoke_contract = copy.deepcopy(contract)
    for name in ("linear", "stage1_m3", "stage2_m3"):
        smoke_contract["static_config"]["training"][name]["minimum_epochs"] = 1
        smoke_contract["static_config"]["training"][name]["maximum_epochs"] = 1
    all_tasks = read_jsonl(output_root / "work/training_tasks.jsonl")
    models = {0: ("stage1", "m0_nuisance"), 1: ("stage1", "m1_linear"), 2: ("stage1", "m3_current_head"), 3: ("stage2_dense", "m3_z_RW")}
    domain, model = models[rank]
    candidates = [row for row in all_tasks if row["domain"] == domain and row["model"] == model and (domain == "stage1" or row["target"] == "read")]
    task = dict(sorted(candidates, key=lambda row: row["task_id"])[0])
    data = _parent_data(parent, parent_root, domain)
    total = 140 if domain == "stage1" else 80
    subset = stepb._smoke_subset(output_root, data, fold=int(task["fold"]), total=total, minimum_groups_per_role=4)
    result = stepb.train_task(smoke_contract, output_root, external_root / "smoke", task, subset, device=device)
    payload = torch.load(resolve_path(result["checkpoint"]), map_location="cpu", weights_only=False)
    if len(payload["test_prediction"]) == 0:
        raise RuntimeError("Step-C smoke produced no test predictions")
    atomic_json(
        output_root / f"validation/training_smoke_rank{rank:02d}.json",
        {"contract_sha256": contract["contract_sha256"], "rank": rank, "task": task, "checkpoint_sha256": result["checkpoint_sha256"], "test_states": len(payload["test_prediction"]), "passed": True},
    )


def finalize_smoke(config_path: Path) -> None:
    contract, output_root, _, _, _ = verify_contract(config_path)
    rows = [read_json(output_root / f"validation/training_smoke_rank{rank:02d}.json") for rank in range(4)]
    if any(row.get("contract_sha256") != contract["contract_sha256"] or not row.get("passed") for row in rows):
        raise RuntimeError("Step-C training smoke failed")
    report = {"contract_sha256": contract["contract_sha256"], "passed": True, "embedding_smoke": read_json(output_root / "validation/embedding_smoke.json"), "training_tasks": rows}
    atomic_json(output_root / "validation/smoke_report.json", report)
    _atomic(output_root / "validation/smoke_report.md", ("# Step-C smoke\n\n- Exact repeated question embeddings: true\n- Unit-normalized embeddings: true\n- M0/M1/M3 Stage-1 and M3 Stage-2 training paths: passed\n- Group-disjoint role construction: passed\n- Result: PASS\n").encode())


def train_worker(config_path: Path, rank: int, resume: bool) -> None:
    contract, output_root, external_root, parent, parent_root = verify_contract(config_path)
    smoke = read_json(output_root / "validation/smoke_report.json")
    if smoke.get("contract_sha256") != contract["contract_sha256"] or not smoke.get("passed"):
        raise RuntimeError("full Step-C training requires passing smoke")
    config = contract["static_config"]
    stepb.configure_worker_runtime(config)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tasks = [row for row in read_jsonl(output_root / "work/training_tasks.jsonl") if int(row["worker_rank"]) == rank]
    rank_root = output_root / f"work/training/rank{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    data_cache = {}
    completed = 0
    started = time.monotonic()
    for task in tasks:
        result_path = rank_root / f"{task['task_id']}.json"
        if resume and result_path.is_file():
            old = read_json(result_path)
            checkpoint = resolve_path(old.get("checkpoint", ""))
            if old.get("contract_sha256") == contract["contract_sha256"] and old.get("task_sha256") == canonical_hash(task) and checkpoint.is_file() and file_sha256(checkpoint) == old.get("checkpoint_sha256"):
                completed += 1
                continue
        domain = str(task["domain"])
        if domain not in data_cache:
            data_cache[domain] = _parent_data(parent, parent_root, domain)
        result = stepb.train_task(contract, output_root, external_root, task, data_cache[domain], device=device)
        atomic_json(result_path, result)
        completed += 1
        torch.cuda.empty_cache()
        print(json.dumps({"rank": rank, "completed": completed, "assigned": len(tasks), "task_id": task["task_id"], "elapsed_seconds": time.monotonic() - started}), flush=True)
    atomic_json(rank_root / "complete.json", {"contract_sha256": contract["contract_sha256"], "rank": rank, "expected_tasks": len(tasks), "completed_tasks": completed, "elapsed_seconds": time.monotonic() - started, "completed_at": utc_now()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-encoder")
    embed = subparsers.add_parser("embed-worker")
    embed.add_argument("--rank", type=int, required=True)
    subparsers.add_parser("finalize-semantics")
    smoke = subparsers.add_parser("smoke-worker")
    smoke.add_argument("--rank", type=int, required=True)
    subparsers.add_parser("finalize-smoke")
    train = subparsers.add_parser("train-worker")
    train.add_argument("--rank", type=int, required=True)
    train.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "prepare-encoder":
        prepare_encoder(arguments.config)
    elif arguments.command == "embed-worker":
        embed_worker(arguments.config, arguments.rank)
    elif arguments.command == "finalize-semantics":
        finalize_semantics(arguments.config)
    elif arguments.command == "smoke-worker":
        smoke_worker(arguments.config, arguments.rank)
    elif arguments.command == "finalize-smoke":
        finalize_smoke(arguments.config)
    else:
        train_worker(arguments.config, arguments.rank, arguments.resume)


if __name__ == "__main__":
    main()
