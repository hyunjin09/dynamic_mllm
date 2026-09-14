#!/usr/bin/env python3
"""Freeze, extract, and summarize dense first-token logit emergence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRMSNorm

from dense_failure_stage1.logit_emergence import (
    classify_wrong_trajectory,
    first_persistent_layer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("gqa", "chartqa", "textvqa")
LAYERS = 28


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    write_text_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def write_csv_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def contract_hash(contract: dict) -> str:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    return canonical_hash(payload)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def validate_config_inputs(config: dict, *, model_artifacts: str) -> None:
    inputs = config["inputs"]
    for key in ("dense_outputs", "feature_index", "feature_schema"):
        path = resolve_repo_path(inputs[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        expected = inputs[f"{key}_sha256"]
        if actual != expected:
            raise ValueError(f"input hash differs for {path}: {actual} != {expected}")
    model_root = Path(config["model"]["path"])
    artifacts = config["model"]["artifact_sha256"]
    if model_artifacts == "all":
        names = artifacts
    elif model_artifacts == "readout":
        names = (
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        )
    elif model_artifacts == "none":
        names = ()
    else:
        raise ValueError(f"unsupported model-artifact validation scope: {model_artifacts}")
    for name in names:
        path = model_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        expected = artifacts[name]
        if actual != expected:
            raise ValueError(f"model artifact hash differs for {path}: {actual} != {expected}")


def _validate_population(config: dict) -> tuple[list[dict], list[dict]]:
    inputs = config["inputs"]
    outputs = read_jsonl(resolve_repo_path(inputs["dense_outputs"]))
    feature_index = read_jsonl(resolve_repo_path(inputs["feature_index"]))
    expected = int(inputs["expected_records"])
    if len(outputs) != expected or len(feature_index) != expected:
        raise ValueError(
            f"record count differs: outputs={len(outputs)} index={len(feature_index)} "
            f"expected={expected}"
        )
    output_uids = [str(row["uid"]) for row in outputs]
    feature_uids = [str(row["uid"]) for row in feature_index]
    if len(set(output_uids)) != expected or len(set(feature_uids)) != expected:
        raise ValueError("duplicate UID in dense outputs or feature index")
    if set(output_uids) != set(feature_uids):
        raise ValueError("dense output and feature-index UID sets differ")
    counts = Counter(bool(row["current_dense_correct"]) for row in outputs)
    if counts[True] != int(inputs["expected_correct"]):
        raise ValueError("correct sample count differs")
    if counts[False] != int(inputs["expected_wrong"]):
        raise ValueError("wrong sample count differs")
    shards = {str(row["shard"]) for row in feature_index}
    if len(shards) != int(inputs["expected_feature_shards"]):
        raise ValueError(f"feature-shard count differs: {len(shards)}")
    schema = read_json(resolve_repo_path(inputs["feature_schema"]))
    readout = config["readout"]
    if schema.get(readout["feature"]) != readout["feature_semantics"]:
        raise ValueError("feature semantics differ from frozen readout")
    if schema.get("layers") != list(range(LAYERS)):
        raise ValueError("feature layer IDs differ")
    return outputs, feature_index


def freeze(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    validate_config_inputs(config, model_artifacts="all")
    outputs, feature_index = _validate_population(config)
    model_root = Path(config["model"]["path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_root,
        revision=config["model"]["revision"],
        local_files_only=True,
        use_fast=False,
    )
    token_rows = []
    for row in sorted(outputs, key=lambda item: str(item["uid"])):
        gt_ids = tokenizer.encode(
            str(row["gt_answer"]), add_special_tokens=False
        )
        pred_ids = [int(value) for value in row["generated_token_ids"]]
        if not gt_ids:
            raise ValueError(f"GT answer tokenizes empty: {row['uid']}")
        if not pred_ids:
            raise ValueError(f"generated answer has no token: {row['uid']}")
        gt_token = int(gt_ids[0])
        pred_token = int(pred_ids[0])
        token_rows.append(
            {
                "uid": str(row["uid"]),
                "dataset": str(row["dataset"]),
                "current_dense_correct": bool(row["current_dense_correct"]),
                "gt_first_token_id": gt_token,
                "pred_first_token_id": pred_token,
                "first_token_collision": gt_token == pred_token,
                "gt_first_token_text": tokenizer.decode(
                    [gt_token], clean_up_tokenization_spaces=False
                ),
                "pred_first_token_text": tokenizer.decode(
                    [pred_token], clean_up_tokenization_spaces=False
                ),
            }
        )
    output_root = Path(args.output_root)
    token_path = output_root / "token_targets.jsonl"
    write_jsonl_atomic(token_path, token_rows)
    code_paths = [
        Path("plans/dense_answer_logit_emergence_analysis_plan.md"),
        Path("configs/dense_logit_emergence_v1.json"),
        Path("dense_failure_stage1/logit_emergence.py"),
        Path("experiments/analyze_dense_logit_emergence.py"),
        Path("tests/test_dense_logit_emergence.py"),
    ]
    contract = {
        "schema_version": "dense_logit_emergence_contract_v1",
        "frozen_at": utc_now(),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "worktree_status_porcelain_v1": _git("status", "--porcelain=v1").splitlines(),
        },
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model": config["model"],
        "inputs": config["inputs"],
        "input_counts": {
            "records": len(outputs),
            "feature_index_records": len(feature_index),
            "datasets": dict(sorted(Counter(row["dataset"] for row in outputs).items())),
            "correct": sum(bool(row["current_dense_correct"]) for row in outputs),
            "wrong": sum(not bool(row["current_dense_correct"]) for row in outputs),
        },
        "token_targets": {
            "path": str(token_path),
            "sha256": file_sha256(token_path),
            "records": len(token_rows),
            "first_token_collisions": sum(row["first_token_collision"] for row in token_rows),
        },
        "readout": config["readout"],
        "emergence": config["emergence"],
        "bootstrap": config["bootstrap"],
        "population_reporting": config["population_reporting"],
        "execution": config["execution"],
        "code_sha256": {
            str(path): file_sha256(REPO_ROOT / path) for path in code_paths
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_models": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "interpretation_limit": (
            "The frozen text_final state is the final literal user-prompt token, "
            "not the actual assistant-start token. Results are a query-position "
            "logit lens and are not calibrated next-token probabilities."
        ),
    }
    contract["contract_sha256"] = contract_hash(contract)
    write_json_atomic(output_root / "analysis_contract.json", contract)
    print(
        json.dumps(
            {
                "contract_sha256": contract["contract_sha256"],
                "records": len(token_rows),
                "first_token_collisions": contract["token_targets"]["first_token_collisions"],
            },
            sort_keys=True,
        )
    )


def validate_contract(args: argparse.Namespace, *, verify_model: bool) -> tuple[dict, dict]:
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    contract = read_json(Path(args.output_root) / "analysis_contract.json")
    if contract_hash(contract) != contract.get("contract_sha256"):
        raise ValueError("analysis contract self-hash differs")
    if file_sha256(config_path) != contract.get("config_sha256"):
        raise ValueError("config differs from frozen contract")
    token_spec = contract["token_targets"]
    if file_sha256(Path(token_spec["path"])) != token_spec["sha256"]:
        raise ValueError("token-target manifest differs from frozen contract")
    validate_config_inputs(
        config, model_artifacts="readout" if verify_model else "none"
    )
    return config, contract


def _load_readout(config: dict, device: torch.device):
    model = config["model"]
    model_root = Path(model["path"])
    index = read_json(model_root / "model.safetensors.index.json")["weight_map"]
    norm_key = "model.norm.weight"
    head_key = "lm_head.weight"
    with safe_open(
        model_root / index[norm_key], framework="pt", device="cpu"
    ) as handle:
        norm_weight = handle.get_tensor(norm_key)
    with safe_open(
        model_root / index[head_key], framework="pt", device="cpu"
    ) as handle:
        head_weight = handle.get_tensor(head_key)
    hidden_size = int(model["hidden_size"])
    vocab_size = int(model["vocab_size"])
    if tuple(norm_weight.shape) != (hidden_size,):
        raise ValueError(f"unexpected final norm shape: {norm_weight.shape}")
    if tuple(head_weight.shape) != (vocab_size, hidden_size):
        raise ValueError(f"unexpected LM-head shape: {head_weight.shape}")
    norm = Qwen2_5_VLRMSNorm(hidden_size, eps=1e-6).to(
        device=device, dtype=torch.bfloat16
    )
    head = torch.nn.Linear(
        hidden_size,
        vocab_size,
        bias=False,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        norm.weight.copy_(norm_weight.to(device=device, dtype=torch.bfloat16))
        head.weight.copy_(head_weight.to(device=device, dtype=torch.bfloat16))
    norm.eval()
    head.eval()
    del norm_weight, head_weight
    return norm, head


@torch.inference_mode()
def _score_hidden_states(
    hidden: torch.Tensor,
    gt_tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    *,
    norm: torch.nn.Module,
    head: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (LAYERS, 3584):
        raise ValueError(f"unexpected hidden-state shape: {hidden.shape}")
    records = int(hidden.shape[0])
    flat = hidden.reshape(records * LAYERS, hidden.shape[-1])
    gt_flat = gt_tokens.repeat_interleave(LAYERS)
    pred_flat = pred_tokens.repeat_interleave(LAYERS)
    gt_scores = []
    pred_scores = []
    competitor_scores = []
    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        states = flat[start:stop].to(device=device, dtype=torch.bfloat16)
        gt = gt_flat[start:stop].to(device)
        pred = pred_flat[start:stop].to(device)
        logits = head(norm(states))
        gt_scores.append(logits.gather(1, gt[:, None]).squeeze(1).float().cpu())
        pred_scores.append(logits.gather(1, pred[:, None]).squeeze(1).float().cpu())
        values, ids = logits.topk(k=2, dim=1)
        competitor_scores.append(
            torch.where(ids[:, 0] == gt, values[:, 1], values[:, 0]).float().cpu()
        )
        del states, logits, values, ids
    shape = (records, LAYERS)
    return (
        torch.cat(gt_scores).reshape(shape),
        torch.cat(pred_scores).reshape(shape),
        torch.cat(competitor_scores).reshape(shape),
    )


def extract(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config, contract = validate_contract(args, verify_model=True)
    expected_world = int(config["execution"]["world_size"])
    if world_size != expected_world or rank not in range(expected_world):
        raise ValueError(f"expected {expected_world} workers; rank={rank} world={world_size}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    targets = {
        row["uid"]: row for row in read_jsonl(Path(contract["token_targets"]["path"]))
    }
    index_rows = read_jsonl(resolve_repo_path(config["inputs"]["feature_index"]))
    rows_by_shard: dict[str, list[dict]] = defaultdict(list)
    for row in index_rows:
        rows_by_shard[str(row["shard"])].append(row)
    shard_paths = sorted(rows_by_shard)
    assigned = shard_paths[rank::world_size]
    norm, head = _load_readout(config, device)
    output_paths = []
    output_root = Path(args.output_root) / "logit_shards"
    for shard_number, source_value in enumerate(assigned):
        source_path = resolve_repo_path(source_value)
        output_path = output_root / f"{source_path.stem}.logits.pt"
        source_rows = sorted(rows_by_shard[source_value], key=lambda row: int(row["row_index"]))
        expected_uids = [str(row["uid"]) for row in source_rows]
        source_hash = file_sha256(source_path)
        if output_path.exists():
            existing = torch.load(output_path, map_location="cpu", weights_only=True)
            if (
                existing.get("contract_sha256") == contract["contract_sha256"]
                and existing.get("source_feature_sha256") == source_hash
                and list(existing.get("uids", [])) == expected_uids
            ):
                output_paths.append(str(output_path))
                print(json.dumps({"rank": rank, "resume": source_path.name}), flush=True)
                continue
            raise ValueError(f"incompatible existing logit shard: {output_path}")
        payload = torch.load(source_path, map_location="cpu", weights_only=True)
        source_uids = [str(uid) for uid in payload["uids"]]
        if source_uids != expected_uids:
            raise ValueError(f"source shard/index UID order differs: {source_path}")
        hidden = payload[config["readout"]["feature"]]
        gt_tokens = torch.tensor(
            [int(targets[uid]["gt_first_token_id"]) for uid in source_uids],
            dtype=torch.long,
        )
        pred_tokens = torch.tensor(
            [int(targets[uid]["pred_first_token_id"]) for uid in source_uids],
            dtype=torch.long,
        )
        gt_logits, pred_logits, competitor_logits = _score_hidden_states(
            hidden,
            gt_tokens,
            pred_tokens,
            norm=norm,
            head=head,
            device=device,
            batch_size=int(config["execution"]["state_batch_size"]),
        )
        result = {
            "schema_version": "dense_logit_emergence_shard_v1",
            "contract_sha256": contract["contract_sha256"],
            "model_revision": config["model"]["revision"],
            "rank": rank,
            "source_feature_shard": str(source_path),
            "source_feature_sha256": source_hash,
            "uids": source_uids,
            "datasets": [str(targets[uid]["dataset"]) for uid in source_uids],
            "current_dense_correct": torch.tensor(
                [bool(targets[uid]["current_dense_correct"]) for uid in source_uids],
                dtype=torch.bool,
            ),
            "gt_token_ids": gt_tokens,
            "pred_token_ids": pred_tokens,
            "gt_logits": gt_logits,
            "pred_logits": pred_logits,
            "competitor_logits": competitor_logits,
            "records": len(source_uids),
            "layers": list(range(LAYERS)),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
        torch.save(result, temporary)
        os.replace(temporary, output_path)
        output_paths.append(str(output_path))
        print(
            json.dumps(
                {
                    "rank": rank,
                    "shard": shard_number + 1,
                    "assigned_shards": len(assigned),
                    "records": len(source_uids),
                }
            ),
            flush=True,
        )
    marker = {
        "schema_version": "dense_logit_emergence_rank_v1",
        "contract_sha256": contract["contract_sha256"],
        "rank": rank,
        "world_size": world_size,
        "source_shards": assigned,
        "output_shards": output_paths,
        "records": sum(len(rows_by_shard[path]) for path in assigned),
        "completed_at": utc_now(),
    }
    write_json_atomic(Path(args.output_root) / "workers" / f"rank{rank:03d}.json", marker)
    print(json.dumps({"rank": rank, "status": "complete", "records": marker["records"]}), flush=True)


def bootstrap_mean_ci(
    values: np.ndarray, *, resamples: int, seed: int, confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or not len(values):
        raise ValueError("bootstrap values must be nonempty [samples,layers]")
    rng = np.random.default_rng(seed)
    estimates = np.empty((resamples, values.shape[1]), dtype=np.float64)
    chunk = 50
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        np.quantile(estimates, alpha, axis=0),
        np.quantile(estimates, 1.0 - alpha, axis=0),
    )


def _summary_rows(
    gt: np.ndarray,
    other: np.ndarray,
    margin: np.ndarray,
    *,
    other_name: str,
    config: dict,
    seed_offset: int,
) -> list[dict]:
    bootstrap = config["bootstrap"]
    low, high = bootstrap_mean_ci(
        margin,
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]) + seed_offset,
        confidence=float(bootstrap["confidence_level"]),
    )
    rows = []
    for layer in range(LAYERS):
        rows.append(
            {
                "layer": layer,
                "n": len(gt),
                "mean_gt_logit": float(gt[:, layer].mean()),
                "median_gt_logit": float(np.median(gt[:, layer])),
                f"mean_{other_name}_logit": float(other[:, layer].mean()),
                f"median_{other_name}_logit": float(np.median(other[:, layer])),
                "mean_margin": float(margin[:, layer].mean()),
                "median_margin": float(np.median(margin[:, layer])),
                "mean_margin_ci_low": float(low[layer]),
                "mean_margin_ci_high": float(high[layer]),
            }
        )
    return rows


def _coverage_layer(emergence: list[int | None], fraction: float) -> int | None:
    total = len(emergence)
    for layer in range(LAYERS):
        if sum(value is not None and value <= layer for value in emergence) / total >= fraction:
            return layer
    return None


def _format_layer(value: int | None) -> str:
    return "not reached" if value is None else str(value)


def _plot_results(
    output_root: Path,
    correct_rows: list[dict],
    wrong_rows: list[dict],
    correct_emergence: list[int | None],
    wrong_emergence: list[int | None],
) -> None:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    layers = np.arange(LAYERS)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figure_root / name, dpi=180)
        plt.close()

    plt.figure(figsize=(8, 4.8))
    correct_mean = np.asarray([row["mean_margin"] for row in correct_rows])
    correct_median = np.asarray([row["median_margin"] for row in correct_rows])
    low = np.asarray([row["mean_margin_ci_low"] for row in correct_rows])
    high = np.asarray([row["mean_margin_ci_high"] for row in correct_rows])
    plt.plot(layers, correct_mean, label="Mean GT − strongest non-GT")
    plt.plot(layers, correct_median, label="Median", linestyle="--")
    plt.fill_between(layers, low, high, alpha=0.2, label="95% bootstrap CI")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    plt.xlabel("Decoder layer")
    plt.ylabel("Raw-logit margin")
    plt.title("Dense-correct first-token margin")
    plt.legend(fontsize=8)
    save("correct_margin_by_layer.png")

    plt.figure(figsize=(8, 4.8))
    plt.plot(layers, [row["mean_gt_logit"] for row in wrong_rows], label="Mean GT logit")
    plt.plot(
        layers,
        [row["mean_eventual_wrong_logit"] for row in wrong_rows],
        label="Mean eventual-predicted logit",
    )
    plt.plot(
        layers,
        [row["median_gt_logit"] for row in wrong_rows],
        label="Median GT logit",
        linestyle="--",
    )
    plt.plot(
        layers,
        [row["median_eventual_wrong_logit"] for row in wrong_rows],
        label="Median eventual-predicted logit",
        linestyle="--",
    )
    plt.xlabel("Decoder layer")
    plt.ylabel("Raw logit")
    plt.title("Dense-wrong GT and eventual-answer logits")
    plt.legend(fontsize=8)
    save("wrong_gt_vs_pred_by_layer.png")

    plt.figure(figsize=(8, 4.8))
    wrong_mean = np.asarray([row["mean_margin"] for row in wrong_rows])
    wrong_median = np.asarray([row["median_margin"] for row in wrong_rows])
    low = np.asarray([row["mean_margin_ci_low"] for row in wrong_rows])
    high = np.asarray([row["mean_margin_ci_high"] for row in wrong_rows])
    plt.plot(layers, wrong_mean, label="Mean GT − eventual answer")
    plt.plot(layers, wrong_median, label="Median", linestyle="--")
    plt.fill_between(layers, low, high, alpha=0.2, label="95% bootstrap CI")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axhline(-1.0, color="gray", linewidth=0.8, linestyle=":")
    plt.xlabel("Decoder layer")
    plt.ylabel("Raw-logit margin")
    plt.title("Dense-wrong GT-minus-eventual-answer margin")
    plt.legend(fontsize=8)
    save("wrong_margin_by_layer.png")

    plt.figure(figsize=(8, 4.8))
    bins = np.arange(-0.5, LAYERS + 0.5, 1)
    correct_defined = [value for value in correct_emergence if value is not None]
    wrong_defined = [value for value in wrong_emergence if value is not None]
    plt.hist(
        correct_defined,
        bins=bins,
        alpha=0.6,
        label=f"Correct (defined {len(correct_defined)}/{len(correct_emergence)})",
    )
    plt.hist(
        wrong_defined,
        bins=bins,
        alpha=0.6,
        label=f"Wrong (defined {len(wrong_defined)}/{len(wrong_emergence)})",
    )
    plt.xlabel("First persistent δ-emergence layer")
    plt.ylabel("Samples")
    plt.title("First-token preference emergence")
    plt.legend(fontsize=8)
    save("emergence_layer_histogram.png")


def aggregate(args: argparse.Namespace) -> None:
    config, contract = validate_contract(args, verify_model=False)
    output_root = Path(args.output_root)
    markers = [
        read_json(output_root / "workers" / f"rank{rank:03d}.json")
        for rank in range(int(config["execution"]["world_size"]))
    ]
    if any(marker["contract_sha256"] != contract["contract_sha256"] for marker in markers):
        raise ValueError("rank marker contract differs")
    output_paths = [path for marker in markers for path in marker["output_shards"]]
    if len(output_paths) != int(config["inputs"]["expected_feature_shards"]):
        raise ValueError(f"logit-shard coverage differs: {len(output_paths)}")
    all_uids: list[str] = []
    all_datasets: list[str] = []
    correct_parts = []
    gt_token_parts = []
    pred_token_parts = []
    gt_parts = []
    pred_parts = []
    competitor_parts = []
    shard_audit = []
    for path_value in sorted(output_paths):
        path = Path(path_value)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["contract_sha256"] != contract["contract_sha256"]:
            raise ValueError(f"logit shard contract differs: {path}")
        source = Path(payload["source_feature_shard"])
        if file_sha256(source) != payload["source_feature_sha256"]:
            raise ValueError(f"source feature shard changed: {source}")
        uids = [str(uid) for uid in payload["uids"]]
        records = len(uids)
        for key in ("gt_logits", "pred_logits", "competitor_logits"):
            tensor = payload[key]
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != (records, LAYERS):
                raise ValueError(f"invalid {key} in {path}: {tensor.shape}/{tensor.dtype}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"non-finite {key} in {path}")
        all_uids.extend(uids)
        all_datasets.extend(str(value) for value in payload["datasets"])
        correct_parts.append(payload["current_dense_correct"])
        gt_token_parts.append(payload["gt_token_ids"])
        pred_token_parts.append(payload["pred_token_ids"])
        gt_parts.append(payload["gt_logits"])
        pred_parts.append(payload["pred_logits"])
        competitor_parts.append(payload["competitor_logits"])
        shard_audit.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "source_feature_shard": str(source),
                "source_feature_sha256": payload["source_feature_sha256"],
                "records": records,
            }
        )
    expected_uids = {
        row["uid"] for row in read_jsonl(Path(contract["token_targets"]["path"]))
    }
    if len(all_uids) != len(set(all_uids)):
        raise ValueError("duplicate UID across logit shards")
    if set(all_uids) != expected_uids:
        raise ValueError("logit-shard UID coverage differs from frozen population")
    order = np.argsort(np.asarray(all_uids, dtype=object))
    uids = np.asarray(all_uids, dtype=object)[order]
    datasets = np.asarray(all_datasets, dtype=object)[order]
    correct = torch.cat(correct_parts).numpy()[order]
    gt_tokens = torch.cat(gt_token_parts).numpy()[order]
    pred_tokens = torch.cat(pred_token_parts).numpy()[order]
    gt = torch.cat(gt_parts).numpy()[order]
    pred = torch.cat(pred_parts).numpy()[order]
    competitor = torch.cat(competitor_parts).numpy()[order]
    wrong = ~correct
    correct_margin = gt - competitor
    wrong_margin = gt - pred

    correct_rows = _summary_rows(
        gt[correct],
        competitor[correct],
        correct_margin[correct],
        other_name="strongest_non_gt",
        config=config,
        seed_offset=1,
    )
    wrong_rows = _summary_rows(
        gt[wrong],
        pred[wrong],
        wrong_margin[wrong],
        other_name="eventual_wrong",
        config=config,
        seed_offset=2,
    )
    correct_fields = list(correct_rows[0])
    wrong_fields = list(wrong_rows[0])
    write_csv_atomic(output_root / "layerwise_correct_summary.csv", correct_rows, correct_fields)
    write_csv_atomic(output_root / "layerwise_wrong_summary.csv", wrong_rows, wrong_fields)

    delta = float(config["emergence"]["delta_raw_logit"])
    persistence = int(config["emergence"]["persistent_layers"])
    early_cutoff = int(config["emergence"]["early_cutoff_layer"])
    sample_rows = []
    taxonomy_rows = []
    correct_emergence: list[int | None] = []
    wrong_emergence: list[int | None] = []
    for index, uid in enumerate(uids):
        if bool(correct[index]):
            margin = correct_margin[index]
            emergence = first_persistent_layer(
                margin, threshold=delta, direction="greater", consecutive=persistence
            )
            zero = first_persistent_layer(
                margin, threshold=0.0, direction="greater", consecutive=persistence
            )
            correct_emergence.append(emergence)
            taxonomy = "not_applicable"
            gt_preference = emergence
        else:
            margin = wrong_margin[index]
            emergence = first_persistent_layer(
                margin, threshold=-delta, direction="less", consecutive=persistence
            )
            zero = first_persistent_layer(
                margin, threshold=0.0, direction="less", consecutive=persistence
            )
            wrong_emergence.append(emergence)
            gt_preference = first_persistent_layer(
                margin, threshold=delta, direction="greater", consecutive=persistence
            )
            taxonomy = classify_wrong_trajectory(
                margin,
                gt_token_id=int(gt_tokens[index]),
                pred_token_id=int(pred_tokens[index]),
                delta=delta,
                consecutive=persistence,
                early_cutoff=early_cutoff,
            )
            taxonomy_rows.append(
                {
                    "uid": uid,
                    "dataset": datasets[index],
                    "taxonomy": taxonomy,
                    "wrong_emergence_layer_delta": emergence,
                    "wrong_emergence_layer_zero": zero,
                    "prior_gt_preference_layer_delta": gt_preference,
                    "gt_first_token_id": int(gt_tokens[index]),
                    "pred_first_token_id": int(pred_tokens[index]),
                    "first_token_collision": int(gt_tokens[index] == pred_tokens[index]),
                    "final_margin": float(margin[-1]),
                }
            )
        sample_rows.append(
            {
                "uid": uid,
                "dataset": datasets[index],
                "current_dense_correct": bool(correct[index]),
                "gt_first_token_id": int(gt_tokens[index]),
                "pred_first_token_id": int(pred_tokens[index]),
                "first_token_collision": int(gt_tokens[index] == pred_tokens[index]),
                "emergence_layer_delta": emergence,
                "emergence_layer_zero": zero,
                "gt_preference_layer_delta": gt_preference,
                "wrong_trajectory_taxonomy": taxonomy,
                "layer0_margin": float(margin[0]),
                "layer27_margin": float(margin[-1]),
                "minimum_margin": float(margin.min()),
                "maximum_margin": float(margin.max()),
            }
        )
    write_csv_atomic(
        output_root / "sample_emergence_layers.csv",
        sample_rows,
        list(sample_rows[0]),
    )
    write_csv_atomic(
        output_root / "wrong_trajectory_taxonomy.csv",
        taxonomy_rows,
        list(taxonomy_rows[0]),
    )

    dataset_rows = []
    seed_offset = 100
    for dataset in DATASETS:
        for outcome, mask in (
            ("correct", (datasets == dataset) & correct),
            ("wrong", (datasets == dataset) & wrong),
        ):
            margin_values = correct_margin[mask] if outcome == "correct" else wrong_margin[mask]
            other_values = competitor[mask] if outcome == "correct" else pred[mask]
            other_name = "strongest_non_gt" if outcome == "correct" else "eventual_wrong"
            rows = _summary_rows(
                gt[mask],
                other_values,
                margin_values,
                other_name=other_name,
                config=config,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            for row in rows:
                dataset_rows.append({"dataset": dataset, "outcome": outcome, **row})
    dataset_fields = sorted({key for row in dataset_rows for key in row})
    dataset_fields = ["dataset", "outcome", "layer", "n"] + [
        key for key in dataset_fields if key not in {"dataset", "outcome", "layer", "n"}
    ]
    write_csv_atomic(output_root / "dataset_breakdown.csv", dataset_rows, dataset_fields)

    taxonomy_counts = Counter(row["taxonomy"] for row in taxonomy_rows)
    taxonomy_by_dataset = {
        dataset: dict(
            sorted(Counter(row["taxonomy"] for row in taxonomy_rows if row["dataset"] == dataset).items())
        )
        for dataset in DATASETS
    }
    coverage_levels = [float(value) for value in config["population_reporting"]["coverage_levels"]]
    selected_layers = [int(value) for value in config["population_reporting"]["selected_curve_layers"]]
    correct_zero = [row["emergence_layer_zero"] for row in sample_rows if row["current_dense_correct"]]
    wrong_zero = [row["emergence_layer_zero"] for row in sample_rows if not row["current_dense_correct"]]
    summary = {
        "schema_version": "dense_logit_emergence_summary_v1",
        "contract_sha256": contract["contract_sha256"],
        "records": len(uids),
        "correct": int(correct.sum()),
        "wrong": int(wrong.sum()),
        "logit_shards": len(output_paths),
        "uids_unique": len(set(uids)),
        "uid_coverage_complete": set(uids) == expected_uids,
        "finite_logits": True,
        "first_token_collisions": {
            "overall": int((gt_tokens == pred_tokens).sum()),
            "correct": int(((gt_tokens == pred_tokens) & correct).sum()),
            "wrong": int(((gt_tokens == pred_tokens) & wrong).sum()),
        },
        "delta": delta,
        "persistent_layers": persistence,
        "correct_emergence_defined": sum(value is not None for value in correct_emergence),
        "wrong_emergence_defined": sum(value is not None for value in wrong_emergence),
        "correct_zero_crossing_defined": sum(value is not None for value in correct_zero),
        "wrong_zero_crossing_defined": sum(value is not None for value in wrong_zero),
        "coverage_layers": {
            "correct_delta": {str(level): _coverage_layer(correct_emergence, level) for level in coverage_levels},
            "wrong_delta": {str(level): _coverage_layer(wrong_emergence, level) for level in coverage_levels},
            "correct_zero": {str(level): _coverage_layer(correct_zero, level) for level in coverage_levels},
            "wrong_zero": {str(level): _coverage_layer(wrong_zero, level) for level in coverage_levels},
        },
        "wrong_taxonomy": dict(sorted(taxonomy_counts.items())),
        "wrong_taxonomy_by_dataset": taxonomy_by_dataset,
        "selected_layer_curves": {
            "correct": {str(layer): correct_rows[layer] for layer in selected_layers},
            "wrong": {str(layer): wrong_rows[layer] for layer in selected_layers},
        },
        "shard_audit": shard_audit,
    }
    write_json_atomic(output_root / "analysis_results.json", summary)
    _plot_results(output_root, correct_rows, wrong_rows, correct_emergence, wrong_emergence)

    def pct(count: int, total: int) -> str:
        return f"{count / total:.1%}" if total else "n/a"

    taxonomy_lines = [
        f"- `{name}`: {count:,} ({pct(count, int(wrong.sum()))})"
        for name, count in sorted(taxonomy_counts.items())
    ]
    dataset_taxonomy_lines = []
    for dataset in DATASETS:
        total = sum(taxonomy_by_dataset[dataset].values())
        values = ", ".join(
            f"{name}={count} ({pct(count, total)})"
            for name, count in sorted(taxonomy_by_dataset[dataset].items())
        )
        dataset_taxonomy_lines.append(f"- {dataset.upper()}: {values}")
    curve_lines = []
    for layer in selected_layers:
        curve_lines.append(
            f"- Layer {layer}: correct mean/median margin "
            f"{correct_rows[layer]['mean_margin']:.3f}/{correct_rows[layer]['median_margin']:.3f}; "
            f"wrong mean/median GT-minus-answer margin "
            f"{wrong_rows[layer]['mean_margin']:.3f}/{wrong_rows[layer]['median_margin']:.3f}."
        )
    report = "\n".join(
        [
            "# Dense answer-logit emergence analysis",
            "",
            f"- Frozen contract: `{contract['contract_sha256']}`",
            f"- Complete population: {len(uids):,}/{config['inputs']['expected_records']:,} unique UIDs; all logits finite.",
            f"- Outcomes: {int(correct.sum()):,} current-dense correct and {int(wrong.sum()):,} current-dense wrong.",
            f"- Readout: `{config['readout']['operation']}` applied to `{config['readout']['feature']}` at layers 0–27.",
            f"- Emergence: strict raw-logit delta {delta:g} for {persistence} consecutive layers; early cutoff layer {early_cutoff}.",
            "",
            "## Population curves",
            "",
            *curve_lines,
            "",
            "## Emergence coverage",
            "",
            f"- Correct delta emergence is defined for {sum(value is not None for value in correct_emergence):,}/{len(correct_emergence):,} ({pct(sum(value is not None for value in correct_emergence), len(correct_emergence))}).",
            f"- Wrong delta emergence is defined for {sum(value is not None for value in wrong_emergence):,}/{len(wrong_emergence):,} ({pct(sum(value is not None for value in wrong_emergence), len(wrong_emergence))}).",
            f"- Correct 50%/75% delta-coverage layers: {_format_layer(_coverage_layer(correct_emergence, 0.5))} / {_format_layer(_coverage_layer(correct_emergence, 0.75))}.",
            f"- Wrong 50%/75% delta-coverage layers: {_format_layer(_coverage_layer(wrong_emergence, 0.5))} / {_format_layer(_coverage_layer(wrong_emergence, 0.75))}.",
            f"- Correct 50%/75% zero-crossing layers: {_format_layer(_coverage_layer(correct_zero, 0.5))} / {_format_layer(_coverage_layer(correct_zero, 0.75))}.",
            f"- Wrong 50%/75% zero-crossing layers: {_format_layer(_coverage_layer(wrong_zero, 0.5))} / {_format_layer(_coverage_layer(wrong_zero, 0.75))}.",
            "",
            "## Wrong-sample taxonomy",
            "",
            *taxonomy_lines,
            "",
            "Per dataset:",
            "",
            *dataset_taxonomy_lines,
            "",
            "## First-token reference audit",
            "",
            f"- GT and generated first tokens coincide for {int((gt_tokens == pred_tokens).sum()):,}/{len(uids):,} samples: {int(((gt_tokens == pred_tokens) & correct).sum()):,} correct and {int(((gt_tokens == pred_tokens) & wrong).sum()):,} wrong.",
            "- TextVQA uses the frozen canonical `gt_answer` first token for this lens; its other LMMS references determine correctness but are not outcome-selected for the logit target.",
            "",
            "## Answers to the plan questions",
            "",
            "1. Early-layer answer-token margins are reported above, but this analysis cannot establish that the full hidden state is uninformative; it only measures a fixed final-head readout.",
            "2. The prospective 50%/75% persistent-emergence layers above locate population-scale first-token preference when those coverage levels are reached.",
            "3. The frozen taxonomy counts quantify early-wrong, progressive-wrong, answer-erosion, and ambiguous trajectories without post-hoc tuning.",
            "4. A supervision start range is defensible only where both outcome groups have substantial persistent coverage; this report deliberately does not freeze a final training range.",
            "5. The next separately authorized Stage-1 experiment can compare all-layer, post-emergence, and informative-range random-k supervision using this evidence; no predictor was trained here.",
            "",
            "## Interpretation boundary",
            "",
            contract["interpretation_limit"],
            "Raw logits are not probabilities. Correct-sample semantic equivalence and TextVQA multi-reference scoring can also differ from the canonical GT first token used by this diagnostic.",
            "",
            "## Stop status",
            "",
            "The planned first-token analysis is complete. Teacher-forced sequence analysis, Stage-1 training, W→C repair, routing, and external evaluation were not run.",
        ]
    ) + "\n"
    write_text_atomic(output_root / "analysis_summary.md", report)
    print(
        json.dumps(
            {
                "records": len(uids),
                "correct_emergence_defined": summary["correct_emergence_defined"],
                "wrong_emergence_defined": summary["wrong_emergence_defined"],
                "wrong_taxonomy": summary["wrong_taxonomy"],
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "extract", "aggregate"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "freeze":
        freeze(args)
    elif args.mode == "extract":
        extract(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
