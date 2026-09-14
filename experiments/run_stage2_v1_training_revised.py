#!/usr/bin/env python3
"""Execute the frozen 698-first Stage-2 V1 training and validation action."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binary_policy.executor.four_action import (  # noqa: E402
    capture_four_action_route,
    capture_online_four_action_route,
    greedy_generate_from_cached_prompt,
)
from binary_policy.executor.inputs import BinaryInputs, build_binary_inputs  # noqa: E402
from binary_policy.executor.model import BinaryQwen25VL  # noqa: E402
from dense_failure_stage1.lmms_scoring import (  # noqa: E402
    lmms_eval_source_metadata,
    score_lmms_sample,
)
from dense_failure_stage1.runtime import build_dense_inputs, configure_dense_determinism  # noqa: E402
from dense_failure_stage2.full_label_generation import verify_artifact_manifest  # noqa: E402
from dense_failure_stage2.v1_router import (  # noqa: E402
    ACTION_NAMES,
    ACTION_TO_INDEX,
    SharedReadWriteRouter,
    build_epoch_draws,
    sample_c_layers,
    sample_w_layers,
    summarize_action_predictions,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage2_v1_training_revised_698_first.json"
BOUND_CODE = (
    "configs/stage2_v1_training_revised_698_first.json",
    "dense_failure_stage2/v1_router.py",
    "experiments/run_stage2_v1_training_revised.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage1/lmms_scoring.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    allowed = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            rows.append(value)
    return rows


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
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows).encode(),
    )


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames=None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot infer columns for empty CSV: {path}")
        fieldnames = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "stage2_v1_training_revised_config_v1":
        raise ValueError("unsupported Stage-2 V1 config")
    if int(config["world_size"]) != 4:
        raise ValueError("the frozen V1 contract requires four ranks")
    if tuple(config["router"]["action_order"]) != ACTION_NAMES:
        raise ValueError("router action order differs from the executor contract")
    training = config["training"]
    if int(training["global_sample_draws_per_epoch"]) != 1048:
        raise ValueError("unexpected global draws per epoch")
    if int(training["total_global_optimizer_updates"]) != (
        int(training["epochs"]) * int(training["global_optimizer_updates_per_epoch"])
    ):
        raise ValueError("training update budget is internally inconsistent")
    return config


def _hash_rank(seed: int, value: str) -> str:
    return sha256(f"{seed}:{value}".encode()).hexdigest()


def _routes_by_uid(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["uid"])].append(dict(row))
    for routes in output.values():
        routes.sort(key=lambda row: str(row["route_id"]))
    return dict(output)


def _select_overfit_cohort(
    routes_by_uid: Mapping[str, Sequence[Mapping[str, Any]]],
    c_uids: Sequence[str],
    *,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for action in ACTION_NAMES[1:]:
        candidates = []
        for uid, routes in routes_by_uid.items():
            matching = [row for row in routes if row["intervention_action"] == action]
            if matching and uid not in selected:
                candidates.append((uid, matching))
        candidates.sort(key=lambda item: _hash_rank(seed, f"{action}:{item[0]}"))
        if len(candidates) < 16:
            raise RuntimeError(f"cannot select 16 overfit UIDs for {action}")
        for uid, routes in candidates[:16]:
            route = min(routes, key=lambda row: _hash_rank(seed, str(row["route_id"])))
            selected[uid] = [dict(route)]
    if len(selected) != 48:
        raise RuntimeError("overfit W cohort is not 48 unique UIDs")
    selected_c = sorted(map(str, c_uids), key=lambda uid: _hash_rank(seed, f"C:{uid}"))[:24]
    if len(selected_c) != 24:
        raise RuntimeError("overfit C cohort is incomplete")
    return selected, selected_c


def _make_schedule(
    *,
    routes_by_uid: Mapping[str, Sequence[Mapping[str, Any]]],
    c_routes_by_uid: Mapping[str, Mapping[str, Any]],
    epochs: int,
    c_draws: int,
    seed: int,
    name: str,
    world_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        draws = build_epoch_draws(
            routes_by_uid,
            sorted(c_routes_by_uid),
            c_draws=int(c_draws),
            seed=int(seed),
            epoch=epoch,
        )
        if len(draws) % world_size:
            raise RuntimeError(f"{name} epoch is not divisible across ranks")
        for draw_index, draw in enumerate(draws):
            local_rng = random.Random(
                int(_hash_rank(seed, f"{name}:{epoch}:{draw_index}:{draw['uid']}")[:16], 16)
            )
            if draw["kind"] == "W":
                route = dict(draw["route"])
                layers = sample_w_layers(
                    route["actions"], trigger_layer=int(route["trigger_layer"]), rng=local_rng
                )
                source = route
            else:
                route = dict(c_routes_by_uid[draw["uid"]])
                layers = sample_c_layers(
                    trigger_layer=int(route["trigger_layer"]),
                    num_layers=len(route["actions"]),
                    rng=local_rng,
                )
                source = route
            actions = [str(source["actions"][layer]) for layer in layers]
            rows.append(
                {
                    "schema_version": "stage2_v1_frozen_draw_v1",
                    "schedule": name,
                    "epoch": epoch,
                    "draw_index": draw_index,
                    "worker_rank": draw_index % world_size,
                    "global_update": epoch * (len(draws) // world_size) + draw_index // world_size + 1,
                    "kind": draw["kind"],
                    "uid": draw["uid"],
                    "dataset": source["dataset"],
                    "route_id": source["route_id"],
                    "route_key": source["route_key"],
                    "trigger_layer": int(source["trigger_layer"]),
                    "intervention_layer": source.get("intervention_layer"),
                    "intervention_action": source.get("intervention_action"),
                    "actions": list(source["actions"]),
                    "selected_layers": layers,
                    "selected_actions": actions,
                    "selected_action_indices": [ACTION_TO_INDEX[action] for action in actions],
                }
            )
    return rows


def _sample_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    sample = row.get("sample", row)
    required = (
        "uid",
        "dataset",
        "prompt",
        "answer",
        "local_image_path",
        "image_content_sha256",
        "max_new_tokens",
    )
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"sample {row.get('uid')} lacks fields: {missing}")
    return {key: sample.get(key) for key in (
        "uid", "sample_id", "dataset", "prompt", "question", "answer",
        "all_answer_norms", "local_image_path", "image_content_sha256",
        "image_group_id", "max_new_tokens",
    )}


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")

    sources = {name: resolve_path(path) for name, path in config["sources"].items()}
    phase56_contract = read_json(sources["phase56_contract"])
    if canonical_hash(phase56_contract) != phase56_contract.get("contract_sha256"):
        raise RuntimeError("Phase-56 contract hash is invalid")
    phase56_manifest = read_json(sources["phase56_artifact_manifest"])
    verify_artifact_manifest(sources["phase56_artifact_manifest"].parent, phase56_manifest)
    if phase56_manifest["contract_sha256"] != phase56_contract["contract_sha256"]:
        raise RuntimeError("Phase-56 artifact/contract IDs differ")
    corpus_manifest = read_json(sources["corpus_manifest"])
    if corpus_manifest["contract_sha256"] != phase56_contract["contract_sha256"]:
        raise RuntimeError("corpus manifest is not bound to Phase 56")

    corpus_a = read_jsonl(sources["corpus_a"])
    corpus_b = read_jsonl(sources["corpus_b"])
    if len(corpus_a) != 39 or len({row["uid"] for row in corpus_a}) != 39:
        raise RuntimeError("Corpus A differs from the 39-sample contract")
    routes_by_uid = _routes_by_uid(corpus_b)
    if len(corpus_b) != 7628 or len(routes_by_uid) != 698:
        raise RuntimeError("Corpus B differs from the 698/7,628 contract")
    if any(row.get("route_source") != "single" for row in corpus_b):
        raise RuntimeError("Corpus B contains a non-single route")
    if any(row.get("route_source") != "preservation_full" for row in corpus_a):
        raise RuntimeError("Corpus A contains a non-preservation route")
    c_routes = {str(row["uid"]): dict(row) for row in corpus_a}

    wrong_manifest = {
        str(row["uid"]): row for row in read_jsonl(sources["full_triggered_wrong_manifest"])
    }
    work_manifest = {
        str(row["uid"]): row for row in read_jsonl(sources["train_work_manifest"])
    }
    train_samples = []
    for uid in sorted(routes_by_uid):
        if uid not in wrong_manifest:
            raise RuntimeError(f"Corpus-B sample metadata missing: {uid}")
        train_samples.append({"uid": uid, "kind": "W", "sample": _sample_payload(wrong_manifest[uid])})
    for uid in sorted(c_routes):
        if uid not in work_manifest:
            raise RuntimeError(f"Corpus-A sample metadata missing: {uid}")
        train_samples.append({"uid": uid, "kind": "C", "sample": _sample_payload(work_manifest[uid])})

    candidates = {row["uid"]: row for row in read_jsonl(sources["candidate_manifest"])}
    dense = {row["uid"]: row for row in read_jsonl(sources["dense_outputs"])}
    trigger_map = sorted(read_jsonl(sources["trigger_map_val"]), key=lambda row: row["uid"])
    if len(trigger_map) != 800 or len({row["uid"] for row in trigger_map}) != 800:
        raise RuntimeError("validation trigger map is not 800 unique UIDs")
    validation = []
    for row in trigger_map:
        uid = row["uid"]
        if uid not in candidates or uid not in dense:
            raise RuntimeError(f"validation sample source missing: {uid}")
        if bool(row["dense_wrong"]) != bool(dense[uid]["current_dense_wrong"]):
            raise RuntimeError(f"validation dense label mismatch: {uid}")
        validation.append(
            {
                "uid": uid,
                "dataset": row["dataset"],
                "triggered": bool(row["triggered"]),
                "trigger_layer": row["first_trigger_layer"],
                "dense_wrong": bool(row["dense_wrong"]),
                "sample": _sample_payload(candidates[uid]),
                "dense_output": dense[uid],
            }
        )
    triggered = [row for row in validation if row["triggered"]]
    if len(triggered) != 235 or Counter(row["dense_wrong"] for row in triggered) != Counter({True: 212, False: 23}):
        raise RuntimeError("validation trigger population differs from the frozen 235-row map")

    overfit_routes, overfit_c = _select_overfit_cohort(
        routes_by_uid, sorted(c_routes), seed=int(config["seed"])
    )
    overfit_schedule = _make_schedule(
        routes_by_uid=overfit_routes,
        c_routes_by_uid={uid: c_routes[uid] for uid in overfit_c},
        epochs=int(config["overfit"]["epochs"]),
        c_draws=24,
        seed=int(config["seed"]) + 1000,
        name="overfit",
        world_size=int(config["world_size"]),
    )
    full_schedule = _make_schedule(
        routes_by_uid=routes_by_uid,
        c_routes_by_uid=c_routes,
        epochs=int(config["training"]["epochs"]),
        c_draws=int(config["sampling"]["c_draws_per_epoch"]),
        seed=int(config["seed"]),
        name="full",
        world_size=int(config["world_size"]),
    )
    if len(full_schedule) != 12576 or full_schedule[-1]["global_update"] != 3144:
        raise RuntimeError("full training schedule differs from the frozen update budget")

    model = config["model"]
    if (
        model["revision"] != phase56_contract["static_config"]["model"]["revision"]
        or resolve_path(model["snapshot_path"])
        != resolve_path(phase56_contract["static_config"]["model"]["snapshot_path"])
    ):
        raise RuntimeError("model path/revision differs from Phase 56")
    snapshot = resolve_path(model["snapshot_path"])
    phase56_model_hashes = phase56_contract["model_snapshot_sha256"]
    actual_names = sorted(path.name for path in snapshot.iterdir() if path.is_file())
    if actual_names != sorted(phase56_model_hashes):
        raise RuntimeError("model snapshot inventory differs from Phase 56")
    for name, expected in phase56_model_hashes.items():
        if file_sha256(snapshot / name) != expected:
            raise RuntimeError(f"model snapshot hash mismatch: {name}")

    output_root.mkdir(parents=True)
    atomic_jsonl(output_root / "work/train_samples.jsonl", train_samples)
    atomic_jsonl(output_root / "work/validation_manifest.jsonl", validation)
    atomic_jsonl(output_root / "work/overfit_schedule.jsonl", overfit_schedule)
    atomic_jsonl(output_root / "work/full_schedule.jsonl", full_schedule)
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    bound_hashes = {relative: file_sha256(resolve_path(relative)) for relative in BOUND_CODE}
    internal_hashes = {
        relative: file_sha256(output_root / relative)
        for relative in (
            "work/train_samples.jsonl",
            "work/validation_manifest.jsonl",
            "work/overfit_schedule.jsonl",
            "work/full_schedule.jsonl",
        )
    }
    contract: dict[str, Any] = {
        "schema_version": "stage2_v1_training_revised_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "git": {
            "commit": command_output(("git", "rev-parse", "HEAD")),
            "branch": command_output(("git", "branch", "--show-current")),
            "worktree_status_at_freeze": command_output(("git", "status", "--short")),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda_runtime": torch.version.cuda,
            "lmms_eval": lmms_eval_source_metadata(),
            "execution": "direct_four_gpu_shared_with_user_authorized_low_load_jobs",
        },
        "phase56_contract_sha256": phase56_contract["contract_sha256"],
        "phase56_artifact_manifest_sha256": file_sha256(sources["phase56_artifact_manifest"]),
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "internal_manifest_sha256": internal_hashes,
        "model_snapshot_sha256": phase56_model_hashes,
        "population": {
            "corpus_a_uids": 39,
            "corpus_b_uids": 698,
            "corpus_b_routes": 7628,
            "validation_uids": 800,
            "validation_triggered": 235,
            "validation_triggered_wrong": 212,
            "validation_triggered_correct": 23,
        },
        "selection": {
            "prospective_update_budget": 3144,
            "checkpoint": "final global update only",
            "validation_openings": 1,
            "training_diagnostics_may_change_selection": False,
        },
        "review_reconciliation": {
            "reviewer_verdict": "revise",
            "accepted_revision": "exact online full-token replay plus prospective updates/seed/final-update selection",
            "rejected_representation": "Phase-56 pooled visual_mean as a one-token attention source",
            "required_discriminator": "one-rank worst-case replay/router-backward memory smoke",
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    _atomic_bytes(
        output_root / "config/stage2_v1_config.yaml",
        (json.dumps({"model": model, "router": config["router"], "training": config["training"]}, indent=2) + "\n").encode(),
    )
    _atomic_bytes(
        output_root / "config/sampler_config.yaml",
        (json.dumps({"sampling": config["sampling"], "seed": config["seed"], "schedule_sha256": internal_hashes["work/full_schedule.jsonl"]}, indent=2) + "\n").encode(),
    )
    protocol = f"""# Stage-2 V1 revised frozen protocol

- Contract: `{contract['contract_sha256']}`
- Phase-56 source: `{phase56_contract['contract_sha256']}`
- Model: `{model['name']}` at `{model['revision']}`
- Trainable component: one shared READ/WRITE router only; no layer identity or Stage-1 latent input.
- State source: exact current routed token sequences replayed online from the selected successful route.
- Data: Corpus A (39 C) + Corpus B (698 W); Corpus C is excluded.
- Sampler: each W UID once/epoch, one uniform successful route, positive-anchored Random-4; 350 C draws/epoch.
- Optimization: 12 epochs, 3,144 global updates, AdamW 3e-4, constant schedule, plain CE.
- Selection: only the final global-update checkpoint; training diagnostics cannot alter it.
- Validation: one frozen 800-sample validation rollout; 235 Stage-1-triggered samples execute Stage-2.
- Stop: after V1 validation/diagnosis; no V1.5 or test execution.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"], "full_schedule_rows": len(full_schedule)}))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("frozen Stage-2 contract hash mismatch")
    if contract["static_config"] != config:
        raise RuntimeError("active config differs from frozen Stage-2 contract")
    for name, expected in contract["source_sha256"].items():
        if file_sha256(resolve_path(config["sources"][name])) != expected:
            raise RuntimeError(f"source hash mismatch: {name}")
    for relative, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(relative)) != expected:
            raise RuntimeError(f"bound-code hash mismatch: {relative}")
    for relative, expected in contract["internal_manifest_sha256"].items():
        if file_sha256(output_root / relative) != expected:
            raise RuntimeError(f"internal manifest hash mismatch: {relative}")
    if verify_model:
        snapshot = resolve_path(config["model"]["snapshot_path"])
        for name, expected in contract["model_snapshot_sha256"].items():
            if file_sha256(snapshot / name) != expected:
                raise RuntimeError(f"model snapshot hash mismatch: {name}")
    return contract, output_root


def _load_model(config: Mapping[str, Any], device: torch.device):
    model_config = config["model"]
    snapshot = str(resolve_path(model_config["snapshot_path"]))
    processor = AutoProcessor.from_pretrained(
        snapshot,
        revision=model_config["revision"],
        local_files_only=True,
        use_fast=False,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        snapshot,
        revision=model_config["revision"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_config["attention_implementation"],
        device_map={"": str(device)},
    ).eval()
    base.requires_grad_(False)
    return processor, base, BinaryQwen25VL(base)


def _router(config: Mapping[str, Any], device: torch.device) -> SharedReadWriteRouter:
    router_config = config["router"]
    return SharedReadWriteRouter(
        hidden_size=int(config["model"]["hidden_size"]),
        router_size=int(router_config["router_size"]),
        num_heads=int(router_config["num_heads"]),
        dropout=float(router_config["dropout"]),
    ).to(device)


def _binary_to(meta: BinaryInputs, device: torch.device | str) -> BinaryInputs:
    values = {}
    for name in meta.__dataclass_fields__:
        value = getattr(meta, name)
        values[name] = value.to(device) if torch.is_tensor(value) else value
    return BinaryInputs(**values)


def _prepare_binary(processor, wrapped, sample: Mapping[str, Any], device: torch.device) -> BinaryInputs:
    inputs, _metadata = build_dense_inputs(processor, dict(sample), device)
    return build_binary_inputs(wrapped, inputs)


def _decode(processor, generated: torch.Tensor) -> tuple[list[int], str]:
    ids = generated[0].detach().cpu().tolist()
    text = processor.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return ids, text


def _generate(processor, wrapped, output, input_ids: torch.Tensor, sample: Mapping[str, Any]):
    if output.cache is None:
        raise RuntimeError("generation requires a complete routed prompt cache")
    generated = greedy_generate_from_cached_prompt(
        wrapped,
        output.prompt_logits,
        output.inputs,
        output.cache,
        input_ids,
        max_new_tokens=int(sample["max_new_tokens"]),
    ).generated_ids
    ids, text = _decode(processor, generated)
    score = score_lmms_sample(
        dataset=str(sample["dataset"]),
        prediction=text,
        answer=str(sample["answer"]),
        answers=sample.get("all_answer_norms"),
        uid=str(sample["uid"]),
    )
    return ids, text, score


def _physical_device_index(local_index: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return int(local_index)
    devices = [value.strip() for value in visible.split(",") if value.strip()]
    if local_index < 0 or local_index >= len(devices):
        raise ValueError("local CUDA device is outside CUDA_VISIBLE_DEVICES")
    if not devices[local_index].isdigit():
        raise ValueError("GPU UUID CUDA_VISIBLE_DEVICES entries are not supported by this audit")
    return int(devices[local_index])


def _nvidia_memory(local_index: int) -> dict[str, int]:
    physical_index = _physical_device_index(local_index)
    output = command_output(
        (
            "nvidia-smi",
            f"--id={physical_index}",
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    values = [int(value.strip()) for value in output.split(",")]
    result = dict(zip(("total_mib", "used_mib", "free_mib", "utilization_percent"), values))
    result["physical_index"] = physical_index
    return result


def _stack_route_states(output, layers: Sequence[int], device: torch.device):
    text = torch.cat(
        [output.pre_layer_states[int(layer)][0].detach().clone() for layer in layers], dim=0
    ).to(device)
    visual = torch.cat(
        [output.pre_layer_states[int(layer)][1].detach().clone() for layer in layers], dim=0
    ).to(device)
    text_mask = output.inputs.text_valid_mask.detach().clone().expand(len(layers), -1).to(device)
    visual_mask = output.inputs.visual_valid_mask.detach().clone().expand(len(layers), -1).to(device)
    return text, visual, text_mask, visual_mask


def implementation_smoke(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    before = _nvidia_memory(device_index)
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    processor, base, wrapped = _load_model(config, device)
    after_model = _nvidia_memory(device_index)
    train_samples = {row["uid"]: row["sample"] for row in read_jsonl(output_root / "work/train_samples.jsonl")}
    schedule = read_jsonl(output_root / "work/full_schedule.jsonl")
    dense_outputs = {
        row["uid"]: row
        for row in read_jsonl(resolve_path(config["sources"]["dense_outputs"]))
    }
    w_uids = {row["uid"] for row in schedule if row["kind"] == "W" and row["epoch"] == 0}
    worst_uid = max(w_uids, key=lambda uid: int(dense_outputs[uid]["prompt_token_count"]))
    row = next(row for row in schedule if row["uid"] == worst_uid and row["epoch"] == 0)
    sample = train_samples[worst_uid]
    inputs, _metadata = build_dense_inputs(processor, sample, device)
    meta = build_binary_inputs(wrapped, inputs)

    dense_route = ["FULL"] * int(config["model"]["decoder_layers"])
    dense_output = capture_four_action_route(
        wrapped, {}, dense_route, prepared_inputs=meta, use_cache=True, native_full_rows=True
    )
    dense_ids, _dense_text, _score = _generate(
        processor, wrapped, dense_output, inputs["input_ids"], sample
    )
    expected_dense_ids = list(dense_outputs[worst_uid]["generated_token_ids"])
    native_token_parity = dense_ids == expected_dense_ids
    del dense_output

    route_output = capture_four_action_route(
        wrapped,
        {},
        row["actions"],
        prepared_inputs=meta,
        use_cache=False,
        native_full_rows=True,
    )
    router = _router(config, device).train()
    text, visual, text_mask, visual_mask = _stack_route_states(
        route_output, row["selected_layers"], device
    )
    target = torch.tensor(row["selected_action_indices"], dtype=torch.long, device=device)
    logits = router(text, visual, text_mask=text_mask, visual_mask=visual_mask)
    loss = F.cross_entropy(logits, target)
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("implementation smoke produced non-finite router output")
    loss.backward()
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device) / 2**20)
    peak_reserved = int(torch.cuda.max_memory_reserved(device) / 2**20)
    after_backward = _nvidia_memory(device_index)
    elapsed = time.monotonic() - started
    router_grad_ok = all(parameter.grad is not None for parameter in router.parameters())
    backbone_grad_none = all(parameter.grad is None for parameter in base.parameters())
    headroom = before["free_mib"] - peak_reserved
    passed = all(
        (
            native_token_parity,
            tuple(logits.shape) == (len(row["selected_layers"]), 4),
            router_grad_ok,
            backbone_grad_none,
            headroom >= 2048,
        )
    )
    report = {
        "schema_version": "stage2_v1_implementation_smoke_v1",
        "passed": passed,
        "contract_sha256": contract["contract_sha256"],
        "local_device_index": device_index,
        "physical_device_index": _physical_device_index(device_index),
        "worst_uid": worst_uid,
        "prompt_tokens": dense_outputs[worst_uid]["prompt_token_count"],
        "visual_tokens": dense_outputs[worst_uid]["visual_token_count"],
        "selected_layers": row["selected_layers"],
        "selected_actions": row["selected_actions"],
        "logit_shape": list(logits.shape),
        "loss": float(loss.item()),
        "native_all_full_token_parity": native_token_parity,
        "router_gradients_present": router_grad_ok,
        "backbone_gradients_all_none": backbone_grad_none,
        "stage1_trainable_parameters": 0,
        "gpu_before": before,
        "gpu_after_model_load": after_model,
        "gpu_after_backward": after_backward,
        "process_peak_allocated_mib": peak_allocated,
        "process_peak_reserved_mib": peak_reserved,
        "conservative_headroom_mib": headroom,
        "elapsed_seconds": elapsed,
    }
    atomic_json(output_root / "smoke/implementation_smoke.json", report)
    markdown = f"""# Implementation smoke

- Passed: **{passed}**
- Contract: `{contract['contract_sha256']}`
- Worst prompt: `{worst_uid}` ({report['prompt_tokens']} prompt / {report['visual_tokens']} visual tokens)
- Native all-FULL token parity: {native_token_parity}
- Router logits: {list(logits.shape)}; router gradients: {router_grad_ok}
- Frozen Qwen gradients all absent: {backbone_grad_none}; no Stage-1 module was loaded.
- Process peak allocated/reserved: {peak_allocated}/{peak_reserved} MiB
- Conservative shared-GPU headroom after this peak: {headroom} MiB
- Wall time: {elapsed:.2f} s
"""
    _atomic_bytes(output_root / "smoke/implementation_smoke.md", markdown.encode())
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise RuntimeError("implementation/memory smoke failed")


def _load_training_sources(output_root: Path):
    samples = {row["uid"]: row["sample"] for row in read_jsonl(output_root / "work/train_samples.jsonl")}
    return samples


def _metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    confusion = confusion.detach().cpu().long()
    targets, predictions = [], []
    for target in range(4):
        for prediction in range(4):
            count = int(confusion[target, prediction])
            targets.extend([target] * count)
            predictions.extend([prediction] * count)
    return summarize_action_predictions(targets=targets, predictions=predictions)


def train_worker(config_path: Path, mode: str) -> None:
    if mode not in {"overfit", "full"}:
        raise ValueError("training mode must be overfit or full")
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    implementation = read_json(output_root / "smoke/implementation_smoke.json")
    if not implementation.get("passed"):
        raise RuntimeError("implementation smoke is not passing")
    if mode == "full":
        overfit_gate = read_json(output_root / "smoke/overfit_gate.json")
        if not overfit_gate.get("passed"):
            raise RuntimeError("overfit gate is not passing")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("training must use exactly four distributed ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    configure_dense_determinism(int(config["seed"]) + rank, config["backend_settings"])
    dist.init_process_group(backend="nccl")
    processor, base, wrapped = _load_model(config, device)

    torch.manual_seed(int(config["seed"]) + (0 if mode == "full" else 5000))
    router = _router(config, device)
    ddp = DistributedDataParallel(router, device_ids=[local_rank], output_device=local_rank)
    settings = config["training"] if mode == "full" else config["overfit"]
    optimizer = torch.optim.AdamW(
        ddp.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    schedule_all = read_jsonl(output_root / f"work/{mode}_schedule.jsonl")
    schedule = [row for row in schedule_all if int(row["worker_rank"]) == rank]
    expected_updates = (
        int(config["training"]["total_global_optimizer_updates"])
        if mode == "full"
        else int(config["overfit"]["epochs"]) * 18
    )
    if len(schedule) != expected_updates:
        raise RuntimeError(f"rank {rank} has {len(schedule)} rather than {expected_updates} draws")
    samples = _load_training_sources(output_root)
    cpu_input_cache: dict[str, BinaryInputs] = {}
    log_path = output_root / ("training/train_log.jsonl" if mode == "full" else "smoke/overfit_train_log.jsonl")
    if rank == 0 and log_path.exists():
        raise RuntimeError(f"refusing to overwrite existing training log: {log_path}")
    dist.barrier()

    epochs = int(settings["epochs"])
    updates_per_rank_epoch = len(schedule) // epochs
    all_epoch_rows = []
    started = time.monotonic()
    for epoch in range(epochs):
        ddp.train()
        local_loss = torch.zeros(2, dtype=torch.float64, device=device)
        local_confusion = torch.zeros(4, 4, dtype=torch.long, device=device)
        epoch_rows = schedule[
            epoch * updates_per_rank_epoch : (epoch + 1) * updates_per_rank_epoch
        ]
        for row in epoch_rows:
            uid = str(row["uid"])
            if uid not in cpu_input_cache:
                cpu_input_cache[uid] = _binary_to(
                    _prepare_binary(processor, wrapped, samples[uid], device), "cpu"
                )
            meta = _binary_to(cpu_input_cache[uid], device)
            with torch.inference_mode():
                output = capture_four_action_route(
                    wrapped,
                    {},
                    row["actions"],
                    prepared_inputs=meta,
                    use_cache=False,
                    native_full_rows=True,
                )
            text, visual, text_mask, visual_mask = _stack_route_states(
                output, row["selected_layers"], device
            )
            targets = torch.tensor(
                row["selected_action_indices"], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(text).all() or not torch.isfinite(visual).all():
                atomic_json(output_root / f"work/{mode}_rank{rank:02d}.failure.json", {
                    "stage": "routed_states", "uid": uid, "row": row,
                    "contract_sha256": contract["contract_sha256"],
                })
                raise RuntimeError(f"non-finite routed state for {uid}")
            logits = ddp(
                text,
                visual,
                text_mask=text_mask,
                visual_mask=visual_mask,
            )
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                atomic_json(output_root / f"work/{mode}_rank{rank:02d}.failure.json", {
                    "stage": "forward_or_loss", "uid": uid, "row": row,
                    "logits_finite": bool(torch.isfinite(logits).all()),
                    "loss": float(loss.item()),
                    "contract_sha256": contract["contract_sha256"],
                })
                raise RuntimeError(f"non-finite router forward/loss for {uid}")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in ddp.parameters()
            ):
                atomic_json(output_root / f"work/{mode}_rank{rank:02d}.failure.json", {
                    "stage": "backward", "uid": uid, "row": row,
                    "contract_sha256": contract["contract_sha256"],
                })
                raise RuntimeError(f"non-finite router gradient for {uid}")
            torch.nn.utils.clip_grad_norm_(
                ddp.parameters(),
                float(settings["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in ddp.parameters()):
                atomic_json(output_root / f"work/{mode}_rank{rank:02d}.failure.json", {
                    "stage": "optimizer", "uid": uid, "row": row,
                    "contract_sha256": contract["contract_sha256"],
                })
                raise RuntimeError(f"non-finite router parameter after {uid}")
            predictions = logits.detach().argmax(dim=-1)
            local_loss[0] += float(loss.item())
            local_loss[1] += 1
            for target, prediction in zip(targets, predictions):
                local_confusion[int(target), int(prediction)] += 1
            del output, text, visual, text_mask, visual_mask, logits, loss, targets

        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_confusion, op=dist.ReduceOp.SUM)
        metrics = _metrics_from_confusion(local_confusion)
        epoch_row = {
            "schema_version": "stage2_v1_training_epoch_v1",
            "contract_sha256": contract["contract_sha256"],
            "mode": mode,
            "epoch": epoch + 1,
            "global_update": (epoch + 1) * updates_per_rank_epoch,
            "loss": float((local_loss[0] / local_loss[1]).item()),
            "accuracy": metrics["accuracy"],
            "non_full_recall": metrics["non_full_recall"],
            "recall": metrics["recall"],
            "predicted_distribution": metrics["predicted_distribution"],
            "target_distribution": metrics["target_distribution"],
            "confusion": local_confusion.cpu().tolist(),
            "elapsed_seconds": time.monotonic() - started,
        }
        all_epoch_rows.append(epoch_row)
        if rank == 0:
            append_jsonl(log_path, epoch_row)
            print(json.dumps(epoch_row, sort_keys=True), flush=True)

    if not all(parameter.grad is None for parameter in base.parameters()):
        raise RuntimeError("frozen Qwen unexpectedly acquired gradients")
    dist.barrier()
    if rank == 0:
        state = {
            "schema_version": "stage2_v1_router_checkpoint_v1",
            "contract_sha256": contract["contract_sha256"],
            "mode": mode,
            "global_update": expected_updates,
            "router_config": config["router"],
            "state_dict": {key: value.detach().cpu() for key, value in ddp.module.state_dict().items()},
        }
        checkpoint = output_root / (
            "training/final_checkpoint.pt" if mode == "full" else "smoke/overfit_checkpoint.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(f".{checkpoint.name}.tmp.{os.getpid()}")
        torch.save(state, temporary)
        os.replace(temporary, checkpoint)
        checkpoint_sha = file_sha256(checkpoint)
        final_metrics = all_epoch_rows[-1]
        if mode == "overfit":
            first_loss = float(all_epoch_rows[0]["loss"])
            final_loss = float(final_metrics["loss"])
            reduction = (first_loss - final_loss) / first_loss
            supported = [
                action for action in ACTION_NAMES[1:]
                if final_metrics["target_distribution"][action] > 0
            ]
            gate_config = config["overfit"]
            predicted_non_full = 1.0 - final_metrics["predicted_distribution"]["FULL"]
            checks = {
                "loss_reduction": reduction >= float(gate_config["minimum_loss_reduction"]),
                "full_recall": final_metrics["recall"]["FULL"] >= float(gate_config["minimum_full_recall"]),
                "non_full_recall": final_metrics["non_full_recall"] >= float(gate_config["minimum_non_full_recall"]),
                "supported_action_recall": all(
                    final_metrics["recall"][action] >= float(gate_config["minimum_supported_action_recall"])
                    for action in supported
                ),
                "predicted_non_full_fraction": predicted_non_full >= float(gate_config["minimum_predicted_non_full_fraction"]),
            }
            gate = {
                "schema_version": "stage2_v1_overfit_gate_v1",
                "passed": all(checks.values()),
                "contract_sha256": contract["contract_sha256"],
                "checks": checks,
                "first_loss": first_loss,
                "final_loss": final_loss,
                "loss_reduction": reduction,
                "final_metrics": final_metrics,
                "checkpoint_sha256": checkpoint_sha,
            }
            atomic_json(output_root / "smoke/overfit_gate.json", gate)
            metric_rows = []
            distribution_rows = []
            for item in all_epoch_rows:
                metric_rows.append({
                    "epoch": item["epoch"], "loss": item["loss"],
                    "accuracy": item["accuracy"], "full_recall": item["recall"]["FULL"],
                    "non_full_recall": item["non_full_recall"],
                    "read_only_recall": item["recall"]["READ_ONLY"],
                    "write_only_recall": item["recall"]["WRITE_ONLY"],
                    "ignore_recall": item["recall"]["IGNORE"],
                })
                distribution_rows.append({"epoch": item["epoch"], **item["predicted_distribution"]})
            atomic_csv(output_root / "smoke/overfit_metrics.csv", metric_rows)
            atomic_csv(output_root / "smoke/overfit_action_distribution.csv", distribution_rows)
            print(json.dumps(gate, sort_keys=True), flush=True)
        else:
            manifest = {
                "schema_version": "stage2_v1_checkpoint_manifest_v1",
                "contract_sha256": contract["contract_sha256"],
                "checkpoints": [{
                    "path": "training/final_checkpoint.pt",
                    "sha256": checkpoint_sha,
                    "global_update": expected_updates,
                    "selection_eligible": True,
                }],
                "epoch_checkpoints": [],
            }
            atomic_json(output_root / "training/checkpoint_manifest.json", manifest)
            atomic_json(output_root / "training/selected_checkpoint.json", {
                "schema_version": "stage2_v1_selected_checkpoint_v1",
                "contract_sha256": contract["contract_sha256"],
                "selection_rule": "final_global_update_only",
                "path": "training/final_checkpoint.pt",
                "sha256": checkpoint_sha,
                "global_update": expected_updates,
            })
    dist.barrier()
    completion = output_root / f"work/{mode}_rank{rank:02d}.complete.json"
    atomic_json(completion, {
        "passed": True, "rank": rank, "updates": len(schedule),
        "contract_sha256": contract["contract_sha256"], "completed_at": utc_now(),
    })
    dist.destroy_process_group()


def rollout_worker(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    selected = read_json(output_root / "training/selected_checkpoint.json")
    checkpoint_path = output_root / selected["path"]
    if file_sha256(checkpoint_path) != selected["sha256"]:
        raise RuntimeError("selected Stage-2 checkpoint hash mismatch")
    for rank_index in range(int(config["world_size"])):
        complete = output_root / f"work/full_rank{rank_index:02d}.complete.json"
        if not complete.is_file() or not read_json(complete).get("passed"):
            raise RuntimeError("full training rank completion is incomplete")

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("free rollout must use exactly four ranks")
    worker_result = output_root / f"work/rollout_rank{rank:02d}.jsonl"
    worker_complete = output_root / f"work/rollout_rank{rank:02d}.complete.json"
    if worker_result.exists() or worker_complete.exists():
        raise RuntimeError(f"refusing to overwrite rollout rank {rank}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    configure_dense_determinism(int(config["seed"]) + 9000 + rank, config["backend_settings"])
    processor, _base, wrapped = _load_model(config, device)
    router = _router(config, device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["contract_sha256"] != contract["contract_sha256"]:
        raise RuntimeError("checkpoint contract differs from active contract")
    router.load_state_dict(checkpoint["state_dict"], strict=True)

    validation = read_jsonl(output_root / "work/validation_manifest.jsonl")
    triggered = sorted([row for row in validation if row["triggered"]], key=lambda row: row["uid"])
    assigned = [row for index, row in enumerate(triggered) if index % world_size == rank]
    started = time.monotonic()
    count = 0
    for row in assigned:
        sample = row["sample"]
        inputs, _metadata = build_dense_inputs(processor, sample, device)
        meta = build_binary_inputs(wrapped, inputs)
        trigger = int(row["trigger_layer"])
        chosen: list[dict[str, Any]] = []

        def selector(layer_index, text_states, visual_states, current_meta):
            if layer_index < trigger:
                chosen.append({"layer": layer_index, "action": "FULL", "active": False})
                return "FULL"
            text = text_states.detach().clone()
            visual = visual_states.detach().clone()
            text_mask = current_meta.text_valid_mask.detach().clone()
            visual_mask = current_meta.visual_valid_mask.detach().clone()
            logits = router(
                text,
                visual,
                text_mask=text_mask,
                visual_mask=visual_mask,
            )
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"non-finite router logits during rollout: {row['uid']} L{layer_index}")
            probabilities = logits.float().softmax(dim=-1)[0]
            index = int(probabilities.argmax().item())
            action = ACTION_NAMES[index]
            chosen.append({
                "layer": layer_index,
                "action": action,
                "active": True,
                "probabilities": {name: float(probabilities[position].item()) for position, name in enumerate(ACTION_NAMES)},
            })
            return action

        output = capture_online_four_action_route(
            wrapped,
            {},
            selector,
            prepared_inputs=meta,
            use_cache=True,
            native_full_rows=True,
        )
        ids, text, score = _generate(
            processor, wrapped, output, inputs["input_ids"], sample
        )
        active_actions = [item["action"] for item in chosen if item["active"]]
        non_full_layers = [
            int(item["layer"])
            for item in chosen
            if item["active"] and item["action"] != "FULL"
        ]
        result = {
            "schema_version": "stage2_v1_free_rollout_row_v1",
            "contract_sha256": contract["contract_sha256"],
            "checkpoint_sha256": selected["sha256"],
            "uid": row["uid"],
            "dataset": row["dataset"],
            "triggered": True,
            "trigger_layer": trigger,
            "dense_correct": not bool(row["dense_wrong"]),
            "dense_wrong": bool(row["dense_wrong"]),
            "dense_generated_answer": row["dense_output"]["generated_answer"],
            "dense_generated_token_ids": row["dense_output"]["generated_token_ids"],
            "routed_generated_answer": text,
            "routed_generated_token_ids": ids,
            "lmms_metric": score.metric_name,
            "lmms_score": score.raw_score,
            "routed_correct": score.correct,
            "routed_wrong": not score.correct,
            "transition": ("W" if row["dense_wrong"] else "C") + "→" + ("C" if score.correct else "W"),
            "actions": [item["action"] for item in chosen],
            "action_rows": chosen,
            "post_trigger_action_counts": dict(Counter(active_actions)),
            "post_trigger_actions": len(active_actions),
            "non_full_count": len(non_full_layers),
            "any_non_full": bool(non_full_layers),
            "first_non_full_layer": non_full_layers[0] if non_full_layers else None,
            "trigger_to_first_non_full_delay": non_full_layers[0] - trigger if non_full_layers else None,
            "full_fraction_after_trigger": active_actions.count("FULL") / len(active_actions),
            "worker_rank": rank,
        }
        append_jsonl(worker_result, result)
        count += 1
        if count % 10 == 0:
            print(json.dumps({"rank": rank, "completed": count, "assigned": len(assigned), "elapsed_seconds": time.monotonic() - started}), flush=True)
        del output, meta, inputs
    atomic_json(worker_complete, {
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_sha256": selected["sha256"],
        "rank": rank,
        "expected": len(assigned),
        "completed": count,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": utc_now(),
    })
    print(json.dumps(read_json(worker_complete), sort_keys=True), flush=True)


def _plot_training(output_root: Path, train_log: Sequence[Mapping[str, Any]]) -> None:
    epochs = [row["epoch"] for row in train_log]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [row["loss"] for row in train_log], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Plain CE loss")
    plt.tight_layout()
    plt.savefig(output_root / "figures/training_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    for action in ACTION_NAMES:
        plt.plot(epochs, [row["recall"][action] for row in train_log], marker="o", label=action)
    plt.xlabel("Epoch")
    plt.ylabel("Teacher-forced recall (train diagnostic)")
    plt.ylim(0, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_root / "figures/action_recall.png", dpi=160)
    plt.close()

    final_distribution = train_log[-1]["predicted_distribution"]
    plt.figure(figsize=(6, 4))
    plt.bar(ACTION_NAMES, [final_distribution[action] for action in ACTION_NAMES])
    plt.xticks(rotation=20)
    plt.ylabel("Predicted fraction")
    plt.tight_layout()
    plt.savefig(output_root / "figures/predicted_action_distribution.png", dpi=160)
    plt.close()


def finalize(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    selected = read_json(output_root / "training/selected_checkpoint.json")
    if file_sha256(output_root / selected["path"]) != selected["sha256"]:
        raise RuntimeError("selected checkpoint changed before aggregation")
    triggered_expected = [
        row for row in read_jsonl(output_root / "work/validation_manifest.jsonl") if row["triggered"]
    ]
    worker_rows = []
    for rank in range(int(config["world_size"])):
        complete_path = output_root / f"work/rollout_rank{rank:02d}.complete.json"
        if not complete_path.is_file():
            raise RuntimeError(f"rollout rank {rank} completion marker is missing")
        complete = read_json(complete_path)
        rows = read_jsonl(output_root / f"work/rollout_rank{rank:02d}.jsonl")
        if not complete.get("passed") or len(rows) != int(complete["expected"]):
            raise RuntimeError(f"rollout rank {rank} is partial")
        worker_rows.extend(rows)
    expected_uids = Counter(row["uid"] for row in triggered_expected)
    actual_uids = Counter(row["uid"] for row in worker_rows)
    if expected_uids != actual_uids or len(worker_rows) != 235:
        raise RuntimeError("free-rollout global UID coverage is incomplete or duplicated")
    routed = {row["uid"]: row for row in worker_rows}
    validation = read_jsonl(output_root / "work/validation_manifest.jsonl")
    per_sample = []
    for row in validation:
        if row["triggered"]:
            per_sample.append(routed[row["uid"]])
        else:
            correct = not bool(row["dense_wrong"])
            per_sample.append({
                "schema_version": "stage2_v1_free_rollout_row_v1",
                "contract_sha256": contract["contract_sha256"],
                "checkpoint_sha256": selected["sha256"],
                "uid": row["uid"],
                "dataset": row["dataset"],
                "triggered": False,
                "trigger_layer": None,
                "dense_correct": correct,
                "dense_wrong": not correct,
                "dense_generated_answer": row["dense_output"]["generated_answer"],
                "dense_generated_token_ids": row["dense_output"]["generated_token_ids"],
                "routed_generated_answer": row["dense_output"]["generated_answer"],
                "routed_generated_token_ids": row["dense_output"]["generated_token_ids"],
                "lmms_metric": row["dense_output"]["lmms_eval_metric"],
                "lmms_score": row["dense_output"]["lmms_eval_per_sample_score"],
                "routed_correct": correct,
                "routed_wrong": not correct,
                "transition": ("C→C" if correct else "W→W"),
                "actions": ["FULL"] * 28,
                "action_rows": [],
                "post_trigger_action_counts": {},
                "post_trigger_actions": 0,
                "non_full_count": 0,
                "any_non_full": False,
                "first_non_full_layer": None,
                "trigger_to_first_non_full_delay": None,
                "full_fraction_after_trigger": None,
                "worker_rank": None,
            })
    per_sample.sort(key=lambda row: row["uid"])
    atomic_jsonl(output_root / "free_rollout/per_sample_results.jsonl", per_sample)

    transitions = Counter(row["transition"] for row in per_sample)
    dense_correct = sum(bool(row["dense_correct"]) for row in per_sample)
    routed_correct = sum(bool(row["routed_correct"]) for row in per_sample)
    triggered_rows = [row for row in per_sample if row["triggered"]]
    post_counts = Counter()
    for row in triggered_rows:
        post_counts.update(row["post_trigger_action_counts"])
    total_post = sum(post_counts.values())
    any_nonfull = sum(row["any_non_full"] for row in triggered_rows)
    delays = [row["trigger_to_first_non_full_delay"] for row in triggered_rows if row["any_non_full"]]
    overall = {
        "schema_version": "stage2_v1_free_rollout_metrics_v1",
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_sha256": selected["sha256"],
        "validation_samples": len(per_sample),
        "triggered_samples": len(triggered_rows),
        "dense_correct": dense_correct,
        "routed_correct": routed_correct,
        "dense_accuracy": dense_correct / len(per_sample),
        "stage1_stage2_accuracy": routed_correct / len(per_sample),
        "delta_accuracy": (routed_correct - dense_correct) / len(per_sample),
        "transitions": {name: transitions[name] for name in ("W→C", "W→W", "C→C", "C→W")},
        "net_corrections": transitions["W→C"] - transitions["C→W"],
        "w_to_c_rescue_rate": transitions["W→C"] / (transitions["W→C"] + transitions["W→W"]),
        "c_to_c_preservation_rate": transitions["C→C"] / (transitions["C→C"] + transitions["C→W"]),
        "triggered_any_non_full_fraction": any_nonfull / len(triggered_rows),
        "mean_non_full_actions_per_triggered": sum(row["non_full_count"] for row in triggered_rows) / len(triggered_rows),
        "post_trigger_action_counts": dict(post_counts),
        "post_trigger_action_distribution": {action: post_counts[action] / total_post for action in ACTION_NAMES},
        "mean_trigger_to_first_non_full_delay": (sum(delays) / len(delays) if delays else None),
    }
    atomic_json(output_root / "free_rollout/overall_metrics.json", overall)
    atomic_csv(output_root / "free_rollout/transition_counts.csv", [
        {"transition": name, "count": transitions[name]} for name in ("W→C", "W→W", "C→C", "C→W")
    ])
    atomic_csv(output_root / "free_rollout/action_behavior.csv", [
        {
            "triggered_samples": len(triggered_rows),
            "any_non_full_samples": any_nonfull,
            "any_non_full_fraction": any_nonfull / len(triggered_rows),
            "mean_non_full_count": overall["mean_non_full_actions_per_triggered"],
            "mean_delay": overall["mean_trigger_to_first_non_full_delay"],
            **{f"{action.lower()}_count": post_counts[action] for action in ACTION_NAMES},
        }
    ])

    dataset_rows = []
    for dataset in ("gqa", "chartqa", "textvqa"):
        cell = [row for row in per_sample if row["dataset"] == dataset]
        cell_triggered = [row for row in cell if row["triggered"]]
        cell_transitions = Counter(row["transition"] for row in cell)
        cell_post = Counter()
        for row in cell_triggered:
            cell_post.update(row["post_trigger_action_counts"])
        dataset_rows.append({
            "dataset": dataset,
            "samples": len(cell),
            "triggered": len(cell_triggered),
            "dense_correct": sum(row["dense_correct"] for row in cell),
            "routed_correct": sum(row["routed_correct"] for row in cell),
            "dense_accuracy": sum(row["dense_correct"] for row in cell) / len(cell),
            "routed_accuracy": sum(row["routed_correct"] for row in cell) / len(cell),
            "delta_accuracy": (sum(row["routed_correct"] for row in cell) - sum(row["dense_correct"] for row in cell)) / len(cell),
            "w_to_c": cell_transitions["W→C"],
            "w_to_w": cell_transitions["W→W"],
            "c_to_c": cell_transitions["C→C"],
            "c_to_w": cell_transitions["C→W"],
            "triggered_non_full_frequency": (
                sum(row["any_non_full"] for row in cell_triggered) / len(cell_triggered)
                if cell_triggered else 0.0
            ),
            "post_trigger_full_fraction": (
                cell_post["FULL"] / sum(cell_post.values()) if cell_post else 1.0
            ),
        })
    atomic_csv(output_root / "free_rollout/dataset_breakdown.csv", dataset_rows)

    train_log = read_jsonl(output_root / "training/train_log.jsonl")
    if len(train_log) != int(config["training"]["epochs"]):
        raise RuntimeError("training log does not contain the frozen epoch count")
    final_train = train_log[-1]
    action_metric_rows = [
        {
            "scope": "final_epoch_frozen_train_diagnostic",
            "action": action,
            "recall": final_train["recall"][action],
            "target_fraction": final_train["target_distribution"][action],
            "predicted_fraction": final_train["predicted_distribution"][action],
        }
        for action in ACTION_NAMES
    ]
    atomic_csv(output_root / "teacher_forced/action_metrics.csv", action_metric_rows)
    confusion_rows = []
    for target_index, target in enumerate(ACTION_NAMES):
        for prediction_index, prediction in enumerate(ACTION_NAMES):
            confusion_rows.append({
                "target": target,
                "prediction": prediction,
                "count": final_train["confusion"][target_index][prediction_index],
            })
    atomic_csv(output_root / "teacher_forced/confusion_matrix.csv", confusion_rows)
    atomic_csv(output_root / "teacher_forced/predicted_action_distribution.csv", [
        {"action": action, "fraction": final_train["predicted_distribution"][action]}
        for action in ACTION_NAMES
    ])

    immediate = [
        row for row in triggered_rows
        if row["any_non_full"] and row["first_non_full_layer"] == row["trigger_layer"]
    ]
    nonfull_counts = Counter({action: post_counts[action] for action in ACTION_NAMES[1:]})
    nonfull_total = sum(nonfull_counts.values())
    dominant_nonfull_share = max(nonfull_counts.values(), default=0) / nonfull_total if nonfull_total else 0.0
    collapse = {
        "schema_version": "stage2_v1_collapse_check_v1",
        "full_collapse": overall["triggered_any_non_full_fraction"] < 0.01,
        "full_collapse_rule": "triggered any-non-FULL fraction < 0.01",
        "immediate_intervention_collapse": (
            len(immediate) / any_nonfull > 0.9 if any_nonfull else False
        ),
        "immediate_intervention_rule": ">90% of intervened samples first act at trigger",
        "immediate_intervention_fraction": len(immediate) / any_nonfull if any_nonfull else None,
        "single_action_collapse": dominant_nonfull_share > 0.95,
        "single_action_rule": "one non-FULL action exceeds 95% of interventions",
        "dominant_nonfull_share": dominant_nonfull_share,
        "teacher_forced_non_full_recall": final_train["non_full_recall"],
        "free_rollout_any_non_full_fraction": overall["triggered_any_non_full_fraction"],
        "teacher_forcing_free_rollout_gap_note": "teacher-forced metric is a final-epoch train diagnostic, not held-out action generalization",
    }
    atomic_json(output_root / "diagnostics/collapse_check.json", collapse)

    overfit = read_json(output_root / "smoke/overfit_gate.json")
    if not overfit["passed"]:
        diagnosis = "architecture_or_implementation_underfit"
    elif collapse["full_collapse"] or collapse["single_action_collapse"]:
        diagnosis = "free_rollout_policy_collapse_or_exposure_shift"
    elif final_train["non_full_recall"] >= 0.4 and overall["net_corrections"] <= 0:
        diagnosis = "free_rollout_exposure_shift_or_data_diversity; not distinguishable from this V1 alone"
    elif overall["net_corrections"] > 0:
        diagnosis = "positive_net_correction"
    else:
        diagnosis = "mixed_or_inconclusive"
    v15_justified = (
        overfit["passed"]
        and not collapse["full_collapse"]
        and overall["net_corrections"] <= 0
        and diagnosis != "architecture_or_implementation_underfit"
    )
    assessment = f"""# V1 data-sufficiency assessment

- Overfit smoke passed: **{overfit['passed']}**.
- Final-epoch teacher-forced non-FULL recall (train diagnostic): {final_train['non_full_recall']:.4f}.
- Free rollout W→C / C→W / net: {transitions['W→C']} / {transitions['C→W']} / {overall['net_corrections']}.
- Triggered samples taking any non-FULL action: {overall['triggered_any_non_full_fraction']:.4f}.
- Diagnosis: **{diagnosis}**.
- Is adding the 209 MCTS-only W samples justified as the next candidate? **{v15_justified}**.

This action does not authorize V1.5. The training diagnostic is not a held-out action-label evaluation, so data diversity and exposure shift remain confounded when rollout fails.
"""
    _atomic_bytes(output_root / "diagnostics/data_sufficiency_assessment.md", assessment.encode())

    output_root.joinpath("figures").mkdir(parents=True, exist_ok=True)
    _plot_training(output_root, train_log)
    plt.figure(figsize=(6, 4))
    names = ["W→C", "W→W", "C→C", "C→W"]
    plt.bar(names, [transitions[name] for name in names])
    plt.ylabel("Validation samples")
    plt.tight_layout()
    plt.savefig(output_root / "figures/transition_counts.png", dpi=160)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.bar([row["dataset"] for row in dataset_rows], [row["delta_accuracy"] for row in dataset_rows])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Accuracy change")
    plt.tight_layout()
    plt.savefig(output_root / "figures/dataset_accuracy_change.png", dpi=160)
    plt.close()
    plt.figure(figsize=(7, 4))
    bins = list(range(0, 29))
    plt.hist(delays, bins=bins, align="left")
    plt.xlabel("Trigger-to-first-non-FULL delay")
    plt.ylabel("Triggered samples")
    plt.tight_layout()
    plt.savefig(output_root / "figures/trigger_to_first_nonfull_delay.png", dpi=160)
    plt.close()

    summary = f"""# Stage-2 V1 revised summary

The frozen 698-first Stage-2 V1 completed one final-update-selected training run and one 800-sample validation rollout.

| Metric | Value |
|---|---:|
| Dense validation accuracy | {overall['dense_accuracy']:.6f} |
| Stage-1 + Stage-2 accuracy | {overall['stage1_stage2_accuracy']:.6f} |
| Accuracy change | {overall['delta_accuracy']:+.6f} |
| W→C | {transitions['W→C']} |
| C→W | {transitions['C→W']} |
| Net corrections | {overall['net_corrections']} |
| C→C preservation | {overall['c_to_c_preservation_rate']:.6f} |
| Triggered any non-FULL | {overall['triggered_any_non_full_fraction']:.6f} |

Diagnosis: **{diagnosis}**. No V1.5, test-set evaluation, or added complexity was executed.
"""
    _atomic_bytes(output_root / "summaries/stage2_v1_summary.md", summary.encode())
    decision = f"""# Stage-2 V1 decision

1. Implementation smoke passed: **{read_json(output_root / 'smoke/implementation_smoke.json')['passed']}**.
2. Small-overfit smoke passed: **{overfit['passed']}**.
3. Avoided FULL collapse: **{not collapse['full_collapse']}**.
4. Learned all supported non-FULL actions at the frozen overfit gate: **{overfit['checks']['supported_action_recall']}**.
5. Validation W→C / C→W: **{transitions['W→C']} / {transitions['C→W']}**.
6. C→C preservation: **{overall['c_to_c_preservation_rate']:.6f}** ({transitions['C→C']} preserved).
7. Net validation accuracy change: **{overall['delta_accuracy']:+.6f}** ({overall['net_corrections']:+d} samples).
8. Mean trigger-to-first-non-FULL delay: **{overall['mean_trigger_to_first_non_full_delay']}**; post-trigger FULL fraction: **{overall['post_trigger_action_distribution']['FULL']:.6f}**.
9. Teacher-forced-to-rollout transfer: final train non-FULL recall **{final_train['non_full_recall']:.6f}** versus triggered any-non-FULL **{overall['triggered_any_non_full_fraction']:.6f}**. This is descriptive because no clean held-out action-label set exists.
10. Dominant failure diagnosis: **{diagnosis}**.
11. Is the 698-sample V1 sufficient to continue the method direction? **{overall['net_corrections'] > 0}**.
12. Is V1.5 with 209 MCTS-only W justified as a candidate? **{v15_justified}**; it was not run.
"""
    _atomic_bytes(output_root / "summaries/stage2_v1_decision.md", decision.encode())

    required = [
        "protocol.md", "config/stage2_v1_config.yaml", "config/sampler_config.yaml",
        "smoke/implementation_smoke.md", "smoke/overfit_metrics.csv",
        "smoke/overfit_action_distribution.csv", "training/train_log.jsonl",
        "training/checkpoint_manifest.json", "training/selected_checkpoint.json",
        "teacher_forced/action_metrics.csv", "teacher_forced/confusion_matrix.csv",
        "teacher_forced/predicted_action_distribution.csv",
        "free_rollout/per_sample_results.jsonl", "free_rollout/overall_metrics.json",
        "free_rollout/transition_counts.csv", "free_rollout/action_behavior.csv",
        "free_rollout/dataset_breakdown.csv", "diagnostics/data_sufficiency_assessment.md",
        "diagnostics/collapse_check.json", "figures/training_loss.png",
        "figures/action_recall.png", "figures/predicted_action_distribution.png",
        "figures/transition_counts.png", "figures/dataset_accuracy_change.png",
        "figures/trigger_to_first_nonfull_delay.png", "summaries/stage2_v1_summary.md",
        "summaries/stage2_v1_decision.md",
    ]
    missing = [relative for relative in required if not (output_root / relative).is_file()]
    if missing:
        raise RuntimeError(f"required Stage-2 outputs missing: {missing}")
    files = {
        str(path.relative_to(output_root)): file_sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    artifact = {
        "schema_version": "stage2_v1_training_revised_artifact_manifest_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "selected_checkpoint_sha256": selected["sha256"],
        "completed_at": utc_now(),
        "files": files,
    }
    atomic_json(output_root / "artifact_manifest.json", artifact)
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"], **overall}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "implementation-smoke", "train", "rollout", "finalize"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("overfit", "full"))
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    if args.command == "prepare":
        prepare(config_path)
    elif args.command == "implementation-smoke":
        implementation_smoke(config_path, args.device_index)
    elif args.command == "train":
        if args.mode is None:
            parser.error("train requires --mode")
        train_worker(config_path, args.mode)
    elif args.command == "rollout":
        rollout_worker(config_path)
    else:
        finalize(config_path)


if __name__ == "__main__":
    main()
