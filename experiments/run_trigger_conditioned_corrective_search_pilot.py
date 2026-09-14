#!/usr/bin/env python3
"""Run the frozen Phase-55 trigger-conditioned corrective-search pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

from binary_policy.executor import (  # noqa: E402
    BinaryQwen25VL,
    capture_four_action_route,
    capture_four_action_suffix_from_full_baseline,
    capture_full_baseline,
    greedy_generate_from_cached_prompt,
)
from binary_policy.executor.inputs import build_binary_inputs  # noqa: E402
from dense_failure_stage1.lmms_scoring import (  # noqa: E402
    lmms_eval_source_metadata,
    score_lmms_sample,
)
from dense_failure_stage1.runtime import (  # noqa: E402
    build_dense_inputs,
    configure_dense_determinism,
    token_positions,
)
from dense_failure_stage2.corrective_search import (  # noqa: E402
    ACTIONS,
    all_single_routes,
    classify_outcome,
    pilot_cell_weights,
    run_sequential_mcts,
    select_pilot_manifest,
    trigger_depth_bin,
    validate_complete_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/trigger_conditioned_corrective_search_pilot_v1.json"
DATASETS = ("gqa", "chartqa", "textvqa")
DEPTH_BINS = ("L0", "L1-8", "L9-18", "L19-27")
BOUND_CODE_PATHS = (
    "configs/trigger_conditioned_corrective_search_pilot_v1.json",
    "dense_failure_stage2/corrective_search.py",
    "dense_failure_stage1/lmms_scoring.py",
    "dense_failure_stage1/runtime.py",
    "tools/research_analysis/dense_failure_stage1.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "experiments/run_trigger_conditioned_corrective_search_pilot.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any], *, excluded: Sequence[str] = ("contract_sha256",)) -> str:
    payload = {key: item for key, item in value.items() if key not in excluded}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
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


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    allowed = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def load_static(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "trigger_conditioned_corrective_search_pilot_config_v1":
        raise ValueError("unsupported corrective-search config")
    if int(config["world_size"]) != 4 or tuple(config["mcts"]["actions"]) != ACTIONS:
        raise ValueError("world size/action order differs from the prospective contract")
    if config["mcts"]["rollout_cardinalities"] != [2, 3, 4]:
        raise ValueError("rollout cardinalities differ from the reviewed contract")
    if int(config["mcts"]["maximum_iterations"]) != 300:
        raise ValueError("MCTS cap differs from the plan")
    if int(config["mcts"]["extra_iterations_after_first_success"]) != 25:
        raise ValueError("post-success budget differs from the plan")
    if float(config["mcts"]["material_gain_absolute_rate"]) != 0.01:
        raise ValueError("200-to-300 materiality rule differs from the prospective contract")
    allocation = config["pilot_allocation"]
    if set(allocation) != set(DATASETS) or sum(
        int(count) for cells in allocation.values() for count in cells.values()
    ) != 120:
        raise ValueError("pilot allocation must freeze exactly 120 rows over three datasets")
    return config


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "cuda_runtime": torch.version.cuda,
        "lmms_eval": lmms_eval_source_metadata(),
    }


def _assign_workers(rows: Sequence[Mapping[str, Any]], world_size: int) -> list[dict[str, Any]]:
    loads = [0] * world_size
    ordered = sorted(
        rows,
        key=lambda row: (-int(row["estimated_terminal_evaluations"]), str(row["uid"])),
    )
    assigned = []
    for source in ordered:
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        loads[rank] += int(source["estimated_terminal_evaluations"])
        assigned.append({**source, "worker_rank": rank})
    return sorted(assigned, key=lambda row: str(row["uid"]))


def _feature_schema(config: Mapping[str, Any]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "schema_version": "stage2_routed_training_state_schema_v1",
        "state_timing": "pre_layer_entering_chosen_action",
        "trajectory_semantics": "actual_successful_four_action_route",
        "layers": list(range(28)),
        "hidden_size": int(config["features"]["hidden_size"]),
        "tensor_dtype": config["features"]["dtype"],
        "features": {
            "text_final": "last literal user instruction/query token in compact text stream",
            "text_mean": "mean of literal user instruction/question token rows",
            "visual_mean": "mean of valid visual-token rows",
        },
        "label": "chosen action at the same layer on an LMMS-correct successful route",
        "forbidden_inputs": ["dataset_id", "ground_truth", "W_to_C", "historical_bucket"],
    }
    schema["feature_schema_sha256"] = canonical_hash(schema, excluded=("feature_schema_sha256",))
    return schema


def _join_candidates(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    sources = config["sources"]
    triggers = read_jsonl(resolve_path(sources["triggered_wrong_manifest"]))
    candidates = {row["uid"]: row for row in read_jsonl(resolve_path(sources["candidate_manifest"]))}
    dense = {row["uid"]: row for row in read_jsonl(resolve_path(sources["dense_outputs"]))}
    if len(triggers) != 1881 or len({row["uid"] for row in triggers}) != 1881:
        raise RuntimeError("Phase-54 triggered-W source is not the frozen 1,881-UID population")
    joined = []
    population = Counter()
    for trigger in triggers:
        uid = str(trigger["uid"])
        candidate, output = candidates[uid], dense[uid]
        if (
            trigger.get("split") != "train"
            or not bool(trigger.get("dense_wrong"))
            or not bool(output.get("current_dense_wrong"))
            or str(trigger["dataset"]) != str(candidate["dataset"])
            or str(trigger["group_id"]) != str(candidate["image_group_id"])
        ):
            raise RuntimeError(f"Phase-54/candidate/current-label mismatch: {uid}")
        layer = int(trigger["first_trigger_layer"])
        depth_bin = trigger_depth_bin(layer)
        population[(str(trigger["dataset"]), depth_bin)] += 1
        joined.append(
            {
                **trigger,
                "trigger_depth_bin": depth_bin,
                "sample": {
                    key: candidate[key]
                    for key in (
                        "uid",
                        "sample_id",
                        "dataset",
                        "prompt",
                        "question",
                        "answer",
                        "all_answer_norms",
                        "local_image_path",
                        "image_content_sha256",
                        "image_group_id",
                        "max_new_tokens",
                    )
                },
                "dense_output": output,
                "estimated_terminal_evaluations": 3 * (28 - layer) + 300,
            }
        )
    return joined, population


def prepare(config_path: Path) -> None:
    config = load_static(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    joined, population = _join_candidates(config)
    selected = select_pilot_manifest(
        joined,
        allocation=config["pilot_allocation"],
        seed=int(config["seed"]),
    )
    selected = _assign_workers(selected, int(config["world_size"]))
    smoke = []
    for dataset in DATASETS:
        for depth_bin in DEPTH_BINS:
            cell = [
                row for row in selected
                if row["dataset"] == dataset and row["trigger_depth_bin"] == depth_bin
            ]
            cell.sort(key=lambda row: (sha256(f"smoke:{config['seed']}:{row['uid']}".encode()).hexdigest(), row["uid"]))
            smoke.append(dict(cell[0]))
    smoke = _assign_workers(smoke, int(config["world_size"]))

    schema = _feature_schema(config)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_root / "pilot_manifest.jsonl", selected)
    atomic_jsonl(output_root / "smoke/smoke_manifest.jsonl", smoke)
    atomic_json(output_root / "future_stage2_labels/feature_schema.json", schema)

    source_hashes = {
        name: file_sha256(resolve_path(path)) for name, path in config["sources"].items()
    }
    bound_hashes = {path: file_sha256(resolve_path(path)) for path in BOUND_CODE_PATHS}
    phase53 = read_json(resolve_path(config["sources"]["phase53_contract"]))
    snapshot = resolve_path(config["model"]["snapshot_path"])
    model_files = sorted(path for path in snapshot.iterdir() if path.is_file())
    model_hashes = {path.name: file_sha256(path) for path in model_files}
    if model_hashes != phase53["model_snapshot_sha256"]:
        raise RuntimeError("live model snapshot differs from the Phase-53 parity-verified snapshot")
    contract: dict[str, Any] = {
        "schema_version": "trigger_conditioned_corrective_search_pilot_contract_v1",
        "static_config": config,
        "git": {
            "commit": command_output(("git", "rev-parse", "HEAD")),
            "branch": command_output(("git", "branch", "--show-current")),
            "worktree_status_at_freeze": command_output(("git", "status", "--short")),
        },
        "runtime": _runtime_metadata(),
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "model_snapshot_sha256": model_hashes,
        "phase54_contract_sha256": read_json(resolve_path(config["sources"]["phase54_contract"]))["contract_sha256"],
        "phase53_contract_sha256": phase53["contract_sha256"],
        "pilot_manifest_sha256": file_sha256(output_root / "pilot_manifest.jsonl"),
        "smoke_manifest_sha256": file_sha256(output_root / "smoke/smoke_manifest.jsonl"),
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "source_population_cells": {
            f"{dataset}/{depth_bin}": population[(dataset, depth_bin)]
            for dataset in DATASETS for depth_bin in DEPTH_BINS
        },
        "review_reconciliation": {
            "verdict": "revise",
            "change": "replace i.i.d. per-layer rollout with deterministic 2/3/4 total-non-FULL cardinality strata",
            "reporting": "include both balanced-pilot and Phase-54 cell-weighted estimates",
            "remaining_limitation": "bounded low-cardinality rollout may miss routes requiring more interventions",
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    protocol = f"""# Trigger-conditioned corrective-search pilot protocol

- Contract SHA-256: `{contract['contract_sha256']}`
- Source cohort: exactly the Phase-54 train triggered Dense-W manifest (1,881 rows); 120 image-group-unique rows are prospectively frozen for this pilot.
- Diagnostic allocation: 40 per dataset. GQA depth cells L0/L1-8/L9-18/L19-27 = 3/12/12/13; ChartQA and TextVQA = 10/10/10/10 each.
- Primary results include both unweighted pilot estimates and cell-weighted estimates against the full 1,881-row source population.
- Execution begins from the exact native dense FULL state entering the frozen first trigger layer. Layers before the trigger remain FULL.
- Phase A exhausts all three non-FULL single-layer interventions at every layer from trigger through 27. Every correct single route is retained. Single-fixable rows do not enter MCTS.
- Phase B is ordered prefix-tree UCB1 MCTS with actions FULL/READ_ONLY/WRITE_ONLY/IGNORE, binary current LMMS correctness reward, maximum 300 iterations, and checkpoints at 100/200/300.
- An absolute Fixable@300 minus Fixable@200 gain above 0.01 is prospectively defined as material for choosing 300 rather than 200 iterations.
- Rollouts deterministically cycle through 2, 3, and 4 total non-FULL interventions, conditional on the already selected tree prefix. This removes a suffix-length-dependent intervention prior.
- After first success, search continues exactly 25 additional iterations, capped at 300; up to 8 MCTS successes are retained by fewer interventions then stable route key.
- Successful routes are replayed exactly, token/correctness parity is required, and compact pre-layer states entering each chosen suffix action are stored for future Stage-2 training.
- All outputs are bounded-search lower estimates. No Stage-2 training, threshold change, validation/test search, triggered-C search, or external evaluation is authorized.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    atomic_json(
        output_root / "preparation_audit.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "source_records": len(joined),
            "pilot_records": len(selected),
            "smoke_records": len(smoke),
            "unique_pilot_uids": len({row["uid"] for row in selected}),
            "unique_pilot_image_groups": len({row["group_id"] for row in selected}),
            "source_population_cells": contract["source_population_cells"],
        },
    )
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"], "pilot_records": len(selected)}))


def load_contract(
    config_path: Path, *, verify_model_snapshot: bool = False
) -> tuple[dict[str, Any], Path]:
    config = load_static(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256") or contract["static_config"] != config:
        raise RuntimeError("frozen contract/config mismatch")
    if command_output(("git", "rev-parse", "HEAD")) != contract["git"]["commit"]:
        raise RuntimeError("git commit differs from the frozen contract")
    if command_output(("git", "branch", "--show-current")) != contract["git"]["branch"]:
        raise RuntimeError("git branch differs from the frozen contract")
    if command_output(("git", "status", "--short")) != contract["git"]["worktree_status_at_freeze"]:
        raise RuntimeError("worktree status differs from the frozen contract")
    if _runtime_metadata() != contract["runtime"]:
        raise RuntimeError("runtime/environment differs from the frozen contract")
    for name, expected in contract["source_sha256"].items():
        if file_sha256(resolve_path(config["sources"][name])) != expected:
            raise RuntimeError(f"source hash mismatch: {name}")
    for relative, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(relative)) != expected:
            raise RuntimeError(f"bound-code hash mismatch: {relative}")
    if file_sha256(output_root / "pilot_manifest.jsonl") != contract["pilot_manifest_sha256"]:
        raise RuntimeError("pilot manifest hash mismatch")
    if file_sha256(output_root / "smoke/smoke_manifest.jsonl") != contract["smoke_manifest_sha256"]:
        raise RuntimeError("smoke manifest hash mismatch")
    schema = read_json(output_root / "future_stage2_labels/feature_schema.json")
    if (
        schema.get("feature_schema_sha256") != contract["feature_schema_sha256"]
        or canonical_hash(schema, excluded=("feature_schema_sha256",))
        != contract["feature_schema_sha256"]
    ):
        raise RuntimeError("feature schema hash mismatch")
    if verify_model_snapshot:
        snapshot = resolve_path(config["model"]["snapshot_path"])
        actual_files = sorted(path.name for path in snapshot.iterdir() if path.is_file())
        if actual_files != sorted(contract["model_snapshot_sha256"]):
            raise RuntimeError("model snapshot inventory mismatch")
        for name, expected in contract["model_snapshot_sha256"].items():
            if file_sha256(snapshot / name) != expected:
                raise RuntimeError(f"model snapshot hash mismatch: {name}")
    return contract, output_root


def _load_model(config: Mapping[str, Any], device: torch.device):
    model = config["model"]
    snapshot = str(resolve_path(model["snapshot_path"]))
    processor = AutoProcessor.from_pretrained(
        snapshot, revision=model["revision"], local_files_only=True, use_fast=False
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        snapshot,
        revision=model["revision"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=model["attention_implementation"],
        device_map={"": str(device)},
    ).eval()
    base.requires_grad_(False)
    return processor, base, BinaryQwen25VL(base)


def _decode(processor, generated: torch.Tensor) -> tuple[list[int], str]:
    ids = generated[0].detach().cpu().tolist()
    text = processor.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return ids, text


def _generate_output(processor, wrapped, output, inputs, sample) -> dict[str, Any]:
    if output.cache is None:
        raise RuntimeError("cached prompt is required for greedy route generation")
    generated = greedy_generate_from_cached_prompt(
        wrapped,
        output.prompt_logits,
        output.inputs,
        output.cache,
        inputs["input_ids"],
        max_new_tokens=int(sample["max_new_tokens"]),
    ).generated_ids
    ids, text = _decode(processor, generated)
    score = score_lmms_sample(
        dataset=sample["dataset"],
        prediction=text,
        answer=sample["answer"],
        answers=sample.get("all_answer_norms"),
        uid=sample["uid"],
    )
    return {
        "generated_ids": ids,
        "generated_answer": text,
        "lmms_metric": score.metric_name,
        "lmms_score": score.raw_score,
        "correctness_threshold": score.correctness_threshold,
        "correct": score.correct,
    }


def _safe_uid(uid: str) -> str:
    return sha256(uid.encode()).hexdigest()[:24]


def _worker_root(output_root: Path, mode: str, rank: int) -> Path:
    return output_root / f"work/{mode}/rank{rank:02d}"


def _pool_route_states(output, processor, base, inputs, start_layer: int):
    positions = token_positions(processor, base, inputs["input_ids"])
    meta = output.inputs
    valid_text = meta.text_valid_mask[0].detach().cpu().bool()
    compact_indices = meta.text_indices[0].detach().cpu()[valid_text].tolist()
    full_to_compact = {int(full): index for index, full in enumerate(compact_indices)}
    try:
        user_rows = [full_to_compact[int(index)] for index in positions.user_text]
        final_row = full_to_compact[int(positions.final_user_token)]
    except KeyError as exc:
        raise RuntimeError("literal user token is absent from the routed text stream") from exc
    visual_valid = meta.visual_valid_mask[0].to(output.pre_layer_states[0][1].device)
    if not bool(visual_valid.any().item()):
        raise RuntimeError("routed state contains no valid visual rows")
    pooled = {"text_final": [], "text_mean": [], "visual_mean": []}
    metadata = []
    for layer in range(int(start_layer), 28):
        text_states, visual_states = output.pre_layer_states[layer]
        pooled["text_final"].append(text_states[0, final_row].detach().cpu().to(torch.bfloat16))
        pooled["text_mean"].append(
            text_states[0, user_rows].mean(dim=0).detach().cpu().to(torch.bfloat16)
        )
        pooled["visual_mean"].append(
            visual_states[0, visual_valid].mean(dim=0).detach().cpu().to(torch.bfloat16)
        )
        metadata.append({"layer": layer, "action": output.layer_actions[layer]})
    return {key: torch.stack(values) for key, values in pooled.items()}, metadata


def _save_route_features(
    *,
    routes: Sequence[Mapping[str, Any]],
    mode: str,
    manifest_row: Mapping[str, Any],
    baseline,
    wrapped,
    processor,
    base,
    inputs,
    contract: Mapping[str, Any],
    output_root: Path,
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    if not routes:
        return None, None, []
    start = int(manifest_row["first_trigger_layer"])
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    index_rows = []
    tensor_row = 0
    for route in routes:
        actions = tuple(route["actions"])
        replay = capture_four_action_suffix_from_full_baseline(
            wrapped, baseline, start, actions[start:]
        )
        replay_state = _generate_output(
            processor, wrapped, replay, inputs, manifest_row["sample"]
        )
        parity = {
            "generated_ids_equal": replay_state["generated_ids"] == route["generated_ids"],
            "answer_equal": replay_state["generated_answer"] == route["generated_answer"],
            "lmms_score_equal": replay_state["lmms_score"] == route["lmms_score"],
            "correctness_equal": replay_state["correct"] == route["correct"],
        }
        if not all(parity.values()):
            raise RuntimeError(f"successful-route replay parity failed: {parity}")
        if mode == "pilot" and not replay_state["correct"]:
            raise RuntimeError("non-correct route entered successful-route replay")
        pooled, metadata = _pool_route_states(replay, processor, base, inputs, start)
        route_id = sha256(str(route["route_key"]).encode()).hexdigest()[:20]
        for key, values in pooled.items():
            tensors[key].append(values)
        for offset, item in enumerate(metadata):
            index_rows.append(
                {
                    "schema_version": "stage2_routed_training_state_index_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "uid": manifest_row["uid"],
                    "dataset": manifest_row["dataset"],
                    "trigger_layer": start,
                    "trigger_depth_bin": manifest_row["trigger_depth_bin"],
                    "route_id": route_id,
                    "route_key": route["route_key"],
                    "route_source": route["search_stage"],
                    "route_non_full_count": sum(action != "FULL" for action in actions),
                    "layer": item["layer"],
                    "chosen_action": item["action"],
                    "tensor_row": tensor_row + offset,
                    "final_lmms_correct": bool(replay_state["correct"]),
                    "replay_token_parity": True,
                }
            )
        tensor_row += len(metadata)
    packed = {key: torch.cat(values, dim=0) for key, values in tensors.items()}
    expected_rows = len(index_rows)
    if set(packed) != {"text_final", "text_mean", "visual_mean"} or any(
        tensor.shape != (expected_rows, int(contract["static_config"]["features"]["hidden_size"]))
        for tensor in packed.values()
    ):
        raise RuntimeError("routed feature tensor shape/schema mismatch")
    uid_key = _safe_uid(str(manifest_row["uid"]))
    if mode == "pilot":
        relative = Path(f"future_stage2_labels/shards/{uid_key}.pt")
    else:
        relative = Path(f"smoke/features/{uid_key}.pt")
    path = output_root / relative
    payload = {
        "schema_version": "stage2_routed_training_state_shard_v1",
        "contract_sha256": contract["contract_sha256"],
        "model_revision": contract["static_config"]["model"]["revision"],
        "code_commit": contract["git"]["commit"],
        "feature_schema_sha256": contract["feature_schema_sha256"],
        "source_manifest_sha256": contract["pilot_manifest_sha256"],
        "split_run_id": contract["static_config"]["run_id"],
        "uid": manifest_row["uid"],
        "mode": mode,
        "records": index_rows,
        **packed,
    }
    atomic_torch(path, payload)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "contract_sha256",
        "model_revision",
        "code_commit",
        "feature_schema_sha256",
        "source_manifest_sha256",
        "split_run_id",
        "uid",
    ):
        if saved.get(key) != payload[key]:
            raise RuntimeError(f"saved feature provenance mismatch: {key}")
    shard_hash = file_sha256(path)
    return str(relative), shard_hash, index_rows


def worker(
    config_path: Path, *, mode: str, rank: int, world_size: int, resume: bool
) -> None:
    contract, output_root = load_contract(config_path, verify_model_snapshot=True)
    config = contract["static_config"]
    if mode not in {"smoke", "pilot"}:
        raise ValueError("worker mode must be smoke or pilot")
    if world_size != 4 or rank not in range(world_size) or torch.cuda.device_count() != 4:
        raise RuntimeError("corrective-search workers require four visible GPUs/processes")
    if mode == "pilot" and not (output_root / "smoke/smoke_completion.json").exists():
        raise RuntimeError("pilot execution requires a passing smoke")
    manifest_path = output_root / (
        "smoke/smoke_manifest.jsonl" if mode == "smoke" else "pilot_manifest.jsonl"
    )
    rows = [row for row in read_jsonl(manifest_path) if int(row["worker_rank"]) == rank]
    worker_root = _worker_root(output_root, mode, rank)
    complete_path = worker_root / "complete.json"
    if complete_path.exists():
        raise RuntimeError(f"worker already complete: {complete_path}")
    sample_dir = worker_root / "samples"
    existing = list(sample_dir.glob("*.json")) if sample_dir.exists() else []
    if existing and not resume:
        raise FileExistsError(f"worker outputs exist; use --resume: rank {rank}")
    completed: dict[str, dict[str, Any]] = {}
    for path in existing:
        row = read_json(path)
        if row.get("contract_sha256") != contract["contract_sha256"] or not row.get("passed"):
            raise RuntimeError(f"incompatible resume record: {path}")
        uid = str(row["uid"])
        if uid in completed:
            raise RuntimeError(f"duplicate resume UID: {uid}")
        feature_file = row.get("feature_file")
        if feature_file:
            feature_path = output_root / feature_file
            if not feature_path.is_file() or file_sha256(feature_path) != row["feature_file_sha256"]:
                raise RuntimeError(f"resume feature shard is missing/incompatible: {uid}")
        completed[uid] = row

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    configure_dense_determinism(
        int(config["seed"]) + rank,
        read_json(resolve_path(config["sources"]["dense_config"]))["backend_settings"],
    )
    processor, base, wrapped = _load_model(config, device)
    native_full_rows = bool(config["execution"]["native_full_row_dispatch"])
    failures = 0
    for index, manifest_row in enumerate(rows):
        uid = str(manifest_row["uid"])
        if uid in completed:
            continue
        sample_path = sample_dir / f"{_safe_uid(uid)}.json"
        started = time.monotonic()
        try:
            sample = manifest_row["sample"]
            inputs, input_metadata = build_dense_inputs(processor, sample, device)
            if input_metadata["consumed_image_sha256"] != sample["image_content_sha256"]:
                raise RuntimeError("consumed image hash differs from the pilot manifest")
            prepared = build_binary_inputs(wrapped, inputs)
            baseline = capture_full_baseline(
                wrapped,
                inputs,
                prepared_inputs=prepared,
                use_cache=True,
                native_causal=native_full_rows,
            )
            baseline_state = _generate_output(processor, wrapped, baseline, inputs, sample)
            expected = manifest_row["dense_output"]
            baseline_checks = {
                "generated_ids_match_current_dense": baseline_state["generated_ids"]
                == expected["generated_token_ids"],
                "answer_match_current_dense": baseline_state["generated_answer"]
                == expected["generated_answer"],
                "lmms_score_match_current_dense": baseline_state["lmms_score"]
                == float(expected["lmms_eval_per_sample_score"]),
                "correctness_match_current_dense": baseline_state["correct"]
                == bool(expected["current_dense_correct"]),
                "prompt_sha256_match_current_dense": input_metadata["literal_prompt_sha256"]
                == expected["literal_prompt_sha256"],
                "image_sha256_match_manifest": input_metadata["consumed_image_sha256"]
                == sample["image_content_sha256"],
            }
            if not all(baseline_checks.values()) or baseline_state["correct"]:
                raise RuntimeError(f"native FULL/current Dense-W parity failed: {baseline_checks}")

            start = int(manifest_row["first_trigger_layer"])
            full_actions = ["FULL"] * 28
            full_key = "|".join(full_actions)
            control = {
                "schema_version": "corrective_search_route_v1",
                "contract_sha256": contract["contract_sha256"],
                "uid": uid,
                "dataset": sample["dataset"],
                "trigger_layer": start,
                "trigger_depth_bin": manifest_row["trigger_depth_bin"],
                "search_stage": "all_full_control",
                "actions": full_actions,
                "route_key": full_key,
                "changed_layers": [],
                "changed_actions": [],
                "non_full_count": 0,
                **baseline_state,
                "physical_terminal_evaluation": False,
            }
            route_cache: dict[str, dict[str, Any]] = {full_key: control}

            def evaluate_actions(actions: Sequence[str], search_stage: str) -> tuple[dict[str, Any], bool]:
                route_key = "|".join(actions)
                if route_key in route_cache:
                    return route_cache[route_key], False
                route_started = time.monotonic()
                output = capture_four_action_suffix_from_full_baseline(
                    wrapped, baseline, start, tuple(actions[start:])
                )
                state = _generate_output(processor, wrapped, output, inputs, sample)
                changed = [layer for layer, action in enumerate(actions) if action != "FULL"]
                row = {
                    "schema_version": "corrective_search_route_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "uid": uid,
                    "dataset": sample["dataset"],
                    "trigger_layer": start,
                    "trigger_depth_bin": manifest_row["trigger_depth_bin"],
                    "search_stage": search_stage,
                    "actions": list(actions),
                    "route_key": route_key,
                    "changed_layers": changed,
                    "changed_actions": [actions[layer] for layer in changed],
                    "non_full_count": len(changed),
                    **state,
                    "elapsed_seconds": time.monotonic() - route_started,
                    "prompt_decoder_layers": 28 - start,
                    "physical_terminal_evaluation": True,
                }
                route_cache[route_key] = row
                return row, True

            singles = all_single_routes(start_layer=start)
            suffix_complete_parity = None
            if mode == "smoke":
                single_executions = []
                for candidate in singles[: int(config["smoke"]["single_routes_per_sample"])]:
                    route, _ = evaluate_actions(candidate["actions"], "smoke_single")
                    single_executions.append(route)
                parity_route = single_executions[0]
                complete = capture_four_action_route(
                    wrapped,
                    inputs,
                    parity_route["actions"],
                    prepared_inputs=prepared,
                    use_cache=True,
                    native_full_rows=native_full_rows,
                )
                complete_state = _generate_output(processor, wrapped, complete, inputs, sample)
                suffix_complete_parity = {
                    "generated_ids_equal": complete_state["generated_ids"]
                    == parity_route["generated_ids"],
                    "answer_equal": complete_state["generated_answer"]
                    == parity_route["generated_answer"],
                    "lmms_score_equal": complete_state["lmms_score"] == parity_route["lmms_score"],
                    "correctness_equal": complete_state["correct"] == parity_route["correct"],
                }
                if not all(suffix_complete_parity.values()):
                    raise RuntimeError(f"cached suffix/full route parity failed: {suffix_complete_parity}")
                mcts_result = run_sequential_mcts(
                    uid=uid,
                    start_layer=start,
                    seed=int(config["seed"]),
                    max_iterations=int(config["smoke"]["mcts_iterations_per_sample"]),
                    extra_iterations_after_success=int(config["smoke"]["mcts_iterations_per_sample"]),
                    evaluate=lambda actions: evaluate_actions(actions, "smoke_mcts")[0]["correct"],
                    exploration_constant=float(config["mcts"]["exploration_constant"]),
                    rollout_cardinalities=config["mcts"]["rollout_cardinalities"],
                    retain_successes=int(config["mcts"]["retained_successful_routes"]),
                )
                replay_routes = [parity_route]
                outcome_class = "SMOKE"
                successful_singles = [row for row in single_executions if row["correct"]]
            else:
                single_executions = [control]
                successful_singles = []
                for candidate in singles:
                    route, _ = evaluate_actions(candidate["actions"], "single")
                    single_executions.append(route)
                    if route["correct"]:
                        successful_singles.append(route)
                mcts_result = None
                if not successful_singles:
                    mcts_result = run_sequential_mcts(
                        uid=uid,
                        start_layer=start,
                        seed=int(config["seed"]),
                        max_iterations=int(config["mcts"]["maximum_iterations"]),
                        extra_iterations_after_success=int(
                            config["mcts"]["extra_iterations_after_first_success"]
                        ),
                        evaluate=lambda actions: evaluate_actions(actions, "mcts")[0]["correct"],
                        exploration_constant=float(config["mcts"]["exploration_constant"]),
                        rollout_cardinalities=config["mcts"]["rollout_cardinalities"],
                        retain_successes=int(config["mcts"]["retained_successful_routes"]),
                    )
                    for search_row in mcts_result["search_rows"]:
                        route = route_cache[search_row["route_key"]]
                        search_row.update(
                            {
                                "uid": uid,
                                "dataset": sample["dataset"],
                                "trigger_layer": start,
                                "trigger_depth_bin": manifest_row["trigger_depth_bin"],
                                "contract_sha256": contract["contract_sha256"],
                                "generated_ids": route["generated_ids"],
                                "generated_answer": route["generated_answer"],
                                "lmms_metric": route["lmms_metric"],
                                "lmms_score": route["lmms_score"],
                                "correct": route["correct"],
                            }
                        )
                outcome_class = classify_outcome(
                    single_successes=successful_singles, mcts_result=mcts_result
                )
                if successful_singles:
                    replay_routes = sorted(
                        successful_singles,
                        key=lambda row: (row["non_full_count"], row["route_key"]),
                    )
                elif mcts_result is not None:
                    replay_routes = [
                        route_cache[row["route_key"]]
                        for row in mcts_result["successful_routes"]
                    ]
                else:
                    replay_routes = []

            feature_file, feature_hash, feature_index = _save_route_features(
                routes=replay_routes,
                mode=mode,
                manifest_row=manifest_row,
                baseline=baseline,
                wrapped=wrapped,
                processor=processor,
                base=base,
                inputs=inputs,
                contract=contract,
                output_root=output_root,
            )
            final = {
                "schema_version": "corrective_search_sample_result_v1",
                "contract_sha256": contract["contract_sha256"],
                "uid": uid,
                "dataset": sample["dataset"],
                "image_group_id": sample["image_group_id"],
                "trigger_layer": start,
                "trigger_depth_bin": manifest_row["trigger_depth_bin"],
                "worker_rank": rank,
                "mode": mode,
                "baseline_state": baseline_state,
                "baseline_checks": baseline_checks,
                "suffix_complete_route_parity": suffix_complete_parity,
                "single_executions": single_executions,
                "successful_single_routes": successful_singles,
                "mcts_result": mcts_result,
                "outcome_class": outcome_class,
                "feature_file": feature_file,
                "feature_file_sha256": feature_hash,
                "feature_index": feature_index,
                "retained_successful_routes": replay_routes,
                "unique_physical_route_evaluations": sum(
                    bool(row.get("physical_terminal_evaluation")) for row in route_cache.values()
                ),
                "replayed_successful_routes": len(replay_routes),
                "elapsed_seconds": time.monotonic() - started,
                "passed": True,
            }
            atomic_json(sample_path, final)
            completed[uid] = final
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "rank": rank,
                        "sample": index + 1,
                        "total": len(rows),
                        "uid": uid,
                        "class": outcome_class,
                        "physical_routes": final["unique_physical_route_evaluations"],
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            failures += 1
            atomic_json(
                worker_root / f"failure_{_safe_uid(uid)}.json",
                {
                    "contract_sha256": contract["contract_sha256"],
                    "uid": uid,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "worker_rank": rank,
                    "created_at": utc_now(),
                },
            )
            break
        finally:
            torch.cuda.empty_cache()
    if failures:
        raise RuntimeError(f"worker {rank} failed")
    expected = [str(row["uid"]) for row in rows]
    final_rows = list(completed.values())
    validate_complete_results(expected, final_rows)
    atomic_json(
        complete_path,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "mode": mode,
            "rank": rank,
            "records": len(final_rows),
            "completed_at": utc_now(),
        },
    )


def _collect_mode(
    output_root: Path, mode: str, contract_sha256: str
) -> list[dict[str, Any]]:
    rows = []
    for rank in range(4):
        root = _worker_root(output_root, mode, rank)
        complete = read_json(root / "complete.json")
        if not complete.get("passed") or complete.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"invalid {mode} completion for rank {rank}")
        failures = list(root.glob("failure_*.json"))
        if failures:
            raise RuntimeError(f"{mode} rank {rank} contains failure records")
        rows.extend(read_json(path) for path in sorted((root / "samples").glob("*.json")))
    if len({row["uid"] for row in rows}) != len(rows):
        raise RuntimeError(f"{mode} outputs contain duplicate UIDs")
    return sorted(rows, key=lambda row: str(row["uid"]))


def finalize_smoke(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    samples = _collect_mode(output_root, "smoke", contract["contract_sha256"])
    manifest = read_jsonl(output_root / "smoke/smoke_manifest.jsonl")
    validate_complete_results([row["uid"] for row in manifest], samples)
    if len(samples) != 12:
        raise RuntimeError("smoke does not contain the frozen 12 rows")
    baseline_pass = all(all(row["baseline_checks"].values()) for row in samples)
    suffix_pass = all(
        row["suffix_complete_route_parity"]
        and all(row["suffix_complete_route_parity"].values())
        for row in samples
    )
    feature_pass = True
    state_rows = 0
    for row in samples:
        path = output_root / row["feature_file"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        feature_pass &= (
            file_sha256(path) == row["feature_file_sha256"]
            and payload["contract_sha256"] == contract["contract_sha256"]
            and payload["feature_schema_sha256"] == contract["feature_schema_sha256"]
            and all(item["replay_token_parity"] for item in payload["records"])
        )
        state_rows += len(payload["records"])
    if not baseline_pass or not suffix_pass or not feature_pass:
        raise RuntimeError("corrective-search smoke parity/provenance gate failed")
    report = f"""# Corrective-search pilot smoke report

- Contract: `{contract['contract_sha256']}`
- Frozen smoke samples: {len(samples)}/12, one per dataset×trigger-depth cell.
- Native current-dense versus unified FULL token/answer/LMMS parity: **PASS** (12/12).
- Cached-prefix suffix versus complete four-action route token/answer/LMMS parity: **PASS** (12/12).
- Exact successful-route replay and compact pre-layer state extraction: **PASS** ({state_rows} state rows).
- Image SHA-256 verification, contract/source/code/runtime hashes, unique UID coverage, and four-rank global completion: **PASS**.
- Smoke actions only exercised implementation validity; they are not included in pilot scientific metrics.
"""
    _atomic_bytes(output_root / "smoke/smoke_report.md", report.encode())
    atomic_json(
        output_root / "smoke/smoke_completion.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "samples": len(samples),
            "state_rows": state_rows,
            "completed_at": utc_now(),
        },
    )
    print(json.dumps({"passed": True, "samples": len(samples), "state_rows": state_rows}))


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else float("nan")


def _weighted_rate(
    rows: Sequence[Mapping[str, Any]], key: str, weights: Mapping[tuple[str, str], float]
) -> float:
    value = 0.0
    for cell, weight in weights.items():
        cell_rows = [
            row for row in rows
            if (str(row["dataset"]), str(row["trigger_depth_bin"])) == cell
        ]
        if not cell_rows:
            raise RuntimeError(f"pilot has no records in population cell {cell}")
        value += weight * _rate(cell_rows, key)
    return value


def _primary_successful_route(sample: Mapping[str, Any]) -> Mapping[str, Any] | None:
    routes = sample["retained_successful_routes"]
    if not routes:
        return None
    return min(
        routes,
        key=lambda row: (
            int(row["non_full_count"]),
            tuple(int(layer) for layer in row["changed_layers"]),
            str(row["route_key"]),
        ),
    )


def _plot_bar(path: Path, labels: Sequence[str], values: Sequence[float], *, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    bars = axis.bar(labels, values, color="#4472C4")
    axis.set_ylim(0, max(1.0, max(values, default=0.0) * 1.15))
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    config = contract["static_config"]
    if not read_json(output_root / "smoke/smoke_completion.json").get("passed"):
        raise RuntimeError("pilot aggregation requires a passing smoke")
    samples = _collect_mode(output_root, "pilot", contract["contract_sha256"])
    manifest = read_jsonl(output_root / "pilot_manifest.jsonl")
    validate_complete_results([row["uid"] for row in manifest], samples)
    if len(samples) != 120:
        raise RuntimeError("pilot completion differs from the frozen 120 UIDs")

    population = {
        tuple(key.split("/", 1)): int(value)
        for key, value in contract["source_population_cells"].items()
    }
    weights = pilot_cell_weights(population)
    summary_rows = []
    all_single_rows = []
    successful_single_rows = []
    all_mcts_rows = []
    successful_mcts_rows = []
    mcts_summary_rows = []
    future_index_rows = []
    successful_route_manifest = []
    for sample in samples:
        mcts = sample["mcts_result"]
        single_success = sample["outcome_class"] == "SINGLE-FIXABLE"
        mcts_success = sample["outcome_class"] == "MCTS-FIXABLE"
        first_success = None if mcts is None else mcts["first_success_iteration"]
        row = {
            "uid": sample["uid"],
            "dataset": sample["dataset"],
            "trigger_layer": sample["trigger_layer"],
            "trigger_depth_bin": sample["trigger_depth_bin"],
            "outcome_class": sample["outcome_class"],
            "single_fixable": single_success,
            "mcts_fixable": mcts_success,
            "total_fixable": single_success or mcts_success,
            "fixable_at_100": single_success or (first_success is not None and first_success <= 100),
            "fixable_at_200": single_success or (first_success is not None and first_success <= 200),
            "fixable_at_300": single_success or (first_success is not None and first_success <= 300),
            "single_routes_evaluated": len(sample["single_executions"]) - 1,
            "successful_single_routes": len(sample["successful_single_routes"]),
            "mcts_ran": mcts is not None,
            "mcts_iterations": 0 if mcts is None else mcts["iterations"],
            "mcts_unique_terminal_routes": 0 if mcts is None else mcts["unique_terminal_routes"],
            "mcts_first_success_iteration": first_success,
            "physical_terminal_evaluations": sample["unique_physical_route_evaluations"],
            "replayed_successful_routes": sample["replayed_successful_routes"],
            "elapsed_seconds": sample["elapsed_seconds"],
            "passed": sample["passed"],
        }
        summary_rows.append(row)
        all_single_rows.extend(sample["single_executions"])
        successful_single_rows.extend(sample["successful_single_routes"])
        if mcts is not None:
            all_mcts_rows.extend(mcts["search_rows"])
            successful_mcts_rows.extend(
                route for route in sample["retained_successful_routes"]
                if route["search_stage"] == "mcts"
            )
        mcts_summary_rows.append(
            {
                "uid": sample["uid"],
                "dataset": sample["dataset"],
                "trigger_layer": sample["trigger_layer"],
                "trigger_depth_bin": sample["trigger_depth_bin"],
                "outcome_class": sample["outcome_class"],
                "mcts_ran": mcts is not None,
                "iterations": 0 if mcts is None else mcts["iterations"],
                "unique_terminal_routes": 0 if mcts is None else mcts["unique_terminal_routes"],
                "first_success_iteration": first_success,
                "fixable_at_100": row["fixable_at_100"],
                "fixable_at_200": row["fixable_at_200"],
                "fixable_at_300": row["fixable_at_300"],
                "retained_mcts_successes": 0 if not mcts_success else len(sample["retained_successful_routes"]),
            }
        )
        feature_file = sample.get("feature_file")
        if feature_file:
            feature_path = output_root / feature_file
            payload = torch.load(feature_path, map_location="cpu", weights_only=False)
            required = {
                "contract_sha256": contract["contract_sha256"],
                "model_revision": config["model"]["revision"],
                "code_commit": contract["git"]["commit"],
                "feature_schema_sha256": contract["feature_schema_sha256"],
                "source_manifest_sha256": contract["pilot_manifest_sha256"],
                "split_run_id": config["run_id"],
                "uid": sample["uid"],
            }
            if any(payload.get(key) != value for key, value in required.items()):
                raise RuntimeError(f"feature shard provenance mismatch: {sample['uid']}")
            actual_hash = file_sha256(feature_path)
            if actual_hash != sample["feature_file_sha256"]:
                raise RuntimeError(f"feature shard hash mismatch: {sample['uid']}")
            for item in payload["records"]:
                future_index_rows.append(
                    {
                        **item,
                        "feature_file": feature_file,
                        "feature_file_sha256": actual_hash,
                        "feature_schema_sha256": contract["feature_schema_sha256"],
                        "source_manifest_sha256": contract["pilot_manifest_sha256"],
                        "split_run_id": config["run_id"],
                    }
                )
            by_route = defaultdict(list)
            for item in payload["records"]:
                by_route[item["route_id"]].append(item)
            route_lookup = {
                sha256(str(route["route_key"]).encode()).hexdigest()[:20]: route
                for route in sample["retained_successful_routes"]
            }
            for route_id, items in sorted(by_route.items()):
                route = route_lookup[route_id]
                successful_route_manifest.append(
                    {
                        "schema_version": "stage2_successful_route_manifest_v1",
                        "contract_sha256": contract["contract_sha256"],
                        "uid": sample["uid"],
                        "dataset": sample["dataset"],
                        "trigger_layer": sample["trigger_layer"],
                        "trigger_depth_bin": sample["trigger_depth_bin"],
                        "route_id": route_id,
                        "route_source": route["search_stage"],
                        "route_key": route["route_key"],
                        "actions": route["actions"],
                        "non_full_count": route["non_full_count"],
                        "generated_answer": route["generated_answer"],
                        "lmms_metric": route["lmms_metric"],
                        "lmms_score": route["lmms_score"],
                        "final_lmms_correct": route["correct"],
                        "feature_file": feature_file,
                        "feature_file_sha256": actual_hash,
                        "feature_rows": len(items),
                        "replay_token_parity": all(item["replay_token_parity"] for item in items),
                    }
                )

    atomic_jsonl(output_root / "single_search/execution_rows.jsonl", all_single_rows)
    atomic_jsonl(output_root / "single_search/successful_routes.jsonl", successful_single_rows)
    atomic_csv(output_root / "single_search/summary.csv", summary_rows)
    atomic_jsonl(output_root / "mcts/search_rows.jsonl", all_mcts_rows)
    atomic_jsonl(output_root / "mcts/successful_routes.jsonl", successful_mcts_rows)
    atomic_csv(output_root / "mcts/per_sample_summary.csv", mcts_summary_rows)
    atomic_jsonl(output_root / "future_stage2_labels/routed_training_states.jsonl", future_index_rows)
    atomic_jsonl(output_root / "future_stage2_labels/successful_route_manifest.jsonl", successful_route_manifest)

    overall_metrics = (
        ("single_fixable", "single_fixable"),
        ("mcts_only_fixable", "mcts_fixable"),
        ("total_bounded_fixable", "total_fixable"),
        ("fixable_at_100", "fixable_at_100"),
        ("fixable_at_200", "fixable_at_200"),
        ("fixable_at_300", "fixable_at_300"),
    )
    overall_rows = [
        {
            "metric": name,
            "pilot_count": sum(bool(row[key]) for row in summary_rows),
            "pilot_total": len(summary_rows),
            "balanced_pilot_rate": _rate(summary_rows, key),
            "phase54_cell_weighted_rate": _weighted_rate(summary_rows, key, weights),
        }
        for name, key in overall_metrics
    ]
    atomic_csv(output_root / "metrics/overall_correctability.csv", overall_rows)
    budget_rows = []
    previous = None
    for budget in (100, 200, 300):
        key = f"fixable_at_{budget}"
        rate = _rate(summary_rows, key)
        weighted = _weighted_rate(summary_rows, key, weights)
        budget_rows.append(
            {
                "mcts_budget": budget,
                "pilot_fixable": sum(bool(row[key]) for row in summary_rows),
                "pilot_total": len(summary_rows),
                "balanced_pilot_rate": rate,
                "phase54_cell_weighted_rate": weighted,
                "gain_from_previous_budget": None if previous is None else rate - previous,
            }
        )
        previous = rate
    atomic_csv(output_root / "metrics/budget_saturation.csv", budget_rows)

    def breakdown(field: str, values: Sequence[str]) -> list[dict[str, Any]]:
        output = []
        for value in values:
            cell = [row for row in summary_rows if str(row[field]) == value]
            output.append(
                {
                    field: value,
                    "records": len(cell),
                    "single_fixable": sum(row["single_fixable"] for row in cell),
                    "mcts_only_fixable": sum(row["mcts_fixable"] for row in cell),
                    "total_fixable": sum(row["total_fixable"] for row in cell),
                    "single_rate": _rate(cell, "single_fixable"),
                    "mcts_only_rate": _rate(cell, "mcts_fixable"),
                    "total_rate": _rate(cell, "total_fixable"),
                    "fixable_at_100": _rate(cell, "fixable_at_100"),
                    "fixable_at_200": _rate(cell, "fixable_at_200"),
                    "fixable_at_300": _rate(cell, "fixable_at_300"),
                    "median_first_success_iteration": (
                        float(np.median([row["mcts_first_success_iteration"] for row in cell if row["mcts_first_success_iteration"] is not None]))
                        if any(row["mcts_first_success_iteration"] is not None for row in cell)
                        else None
                    ),
                }
            )
        return output

    depth_rows = breakdown("trigger_depth_bin", DEPTH_BINS)
    dataset_rows = breakdown("dataset", DATASETS)
    atomic_csv(output_root / "metrics/trigger_depth_breakdown.csv", depth_rows)
    atomic_csv(output_root / "metrics/dataset_breakdown.csv", dataset_rows)

    primary_routes = [
        route for sample in samples if (route := _primary_successful_route(sample)) is not None
    ]
    complexity_counts = Counter(int(route["non_full_count"]) for route in primary_routes)
    complexity_rows = [
        {
            "non_full_interventions": count,
            "fixable_samples": value,
            "fraction_of_fixable_samples": value / len(primary_routes),
        }
        for count, value in sorted(complexity_counts.items())
    ]
    if not complexity_rows:
        complexity_rows = [{"non_full_interventions": 0, "fixable_samples": 0, "fraction_of_fixable_samples": 0.0}]
    atomic_csv(output_root / "metrics/route_complexity.csv", complexity_rows)

    first_action_sets = []
    first_action_counts = Counter()
    for sample in samples:
        routes = sample["retained_successful_routes"]
        if not routes:
            continue
        layer = int(sample["trigger_layer"])
        observed = {route["actions"][layer] for route in routes}
        first_action_sets.append(observed)
        first_action_counts.update(observed)
    first_action_rows = [
        {
            "trigger_action": action,
            "fixable_samples_with_action": first_action_counts[action],
            "fraction_of_fixable_samples": (
                first_action_counts[action] / len(first_action_sets) if first_action_sets else 0.0
            ),
        }
        for action in ACTIONS
    ]
    first_action_rows.append(
        {
            "trigger_action": "MULTIPLE_OBSERVED_ACTIONS",
            "fixable_samples_with_action": sum(len(actions) > 1 for actions in first_action_sets),
            "fraction_of_fixable_samples": (
                sum(len(actions) > 1 for actions in first_action_sets) / len(first_action_sets)
                if first_action_sets
                else 0.0
            ),
        }
    )
    atomic_csv(output_root / "metrics/first_action_distribution.csv", first_action_rows)

    weighted_projected_gpu_seconds = 0.0
    weighted_projected_terminals = 0.0
    for cell, count in population.items():
        cell_rows = [
            row for row in summary_rows
            if (row["dataset"], row["trigger_depth_bin"]) == cell
        ]
        weighted_projected_gpu_seconds += float(np.mean([row["elapsed_seconds"] for row in cell_rows])) * count
        weighted_projected_terminals += float(np.mean([row["physical_terminal_evaluations"] for row in cell_rows])) * count
    compute_rows = [
        {"metric": "pilot_samples", "value": len(samples), "unit": "samples"},
        {"metric": "single_terminal_evaluations", "value": sum(row["single_routes_evaluated"] for row in summary_rows), "unit": "routes"},
        {"metric": "mcts_iterations", "value": sum(row["mcts_iterations"] for row in summary_rows), "unit": "iterations"},
        {"metric": "unique_physical_terminal_evaluations", "value": sum(row["physical_terminal_evaluations"] for row in summary_rows), "unit": "routes"},
        {"metric": "successful_route_replays", "value": sum(row["replayed_successful_routes"] for row in summary_rows), "unit": "routes"},
        {"metric": "summed_gpu_seconds", "value": sum(row["elapsed_seconds"] for row in summary_rows), "unit": "gpu_seconds"},
        {"metric": "mean_gpu_seconds_per_sample", "value": float(np.mean([row["elapsed_seconds"] for row in summary_rows])), "unit": "gpu_seconds"},
        {"metric": "median_gpu_seconds_per_sample", "value": float(np.median([row["elapsed_seconds"] for row in summary_rows])), "unit": "gpu_seconds"},
        {"metric": "projected_1881_terminal_evaluations_cell_weighted", "value": weighted_projected_terminals, "unit": "routes"},
        {"metric": "projected_1881_gpu_hours_cell_weighted", "value": weighted_projected_gpu_seconds / 3600.0, "unit": "gpu_hours"},
        {"metric": "projected_1881_wall_hours_at_four_equal_gpus", "value": weighted_projected_gpu_seconds / 3600.0 / 4.0, "unit": "hours"},
    ]
    atomic_csv(output_root / "metrics/compute_summary.csv", compute_rows)

    _plot_bar(
        output_root / "figures/fixability_vs_mcts_budget.png",
        ["100", "200", "300"],
        [row["balanced_pilot_rate"] for row in budget_rows],
        title="Bounded fixability versus MCTS budget",
        ylabel="Pilot fixable fraction",
    )
    _plot_bar(
        output_root / "figures/single_vs_mcts_rescue.png",
        ["Single", "MCTS only", "Total"],
        [_rate(summary_rows, "single_fixable"), _rate(summary_rows, "mcts_fixable"), _rate(summary_rows, "total_fixable")],
        title="Single versus sequential rescue",
        ylabel="Pilot fraction",
    )
    _plot_bar(
        output_root / "figures/correctability_by_trigger_depth.png",
        [row["trigger_depth_bin"] for row in depth_rows],
        [row["total_rate"] for row in depth_rows],
        title="Bounded correctability by trigger depth",
        ylabel="Correctable fraction",
    )
    _plot_bar(
        output_root / "figures/correctability_by_dataset.png",
        [row["dataset"] for row in dataset_rows],
        [row["total_rate"] for row in dataset_rows],
        title="Bounded correctability by dataset",
        ylabel="Correctable fraction",
    )
    _plot_bar(
        output_root / "figures/intervention_count_distribution.png",
        [str(row["non_full_interventions"]) for row in complexity_rows],
        [row["fraction_of_fixable_samples"] for row in complexity_rows],
        title="Primary successful-route complexity",
        ylabel="Fraction of fixable samples",
    )

    single_rate = _rate(summary_rows, "single_fixable")
    mcts_rate = _rate(summary_rows, "mcts_fixable")
    total_rate = _rate(summary_rows, "total_fixable")
    fix100, fix200, fix300 = (_rate(summary_rows, f"fixable_at_{budget}") for budget in (100, 200, 300))
    gain_200_300 = fix300 - fix200
    material_gain = float(config["mcts"]["material_gain_absolute_rate"])
    cap_recommendation = 200 if gain_200_300 <= material_gain else 300
    nonfull_values = [int(route["non_full_count"]) for route in primary_routes]
    multi_actions = sum(len(actions) > 1 for actions in first_action_sets)
    dataset_span = max(row["total_rate"] for row in dataset_rows) - min(row["total_rate"] for row in dataset_rows)
    depth_span = max(row["total_rate"] for row in depth_rows) - min(row["total_rate"] for row in depth_rows)
    sequential_justified = mcts_rate > 0
    decision = f"""# Trigger-conditioned corrective-search pilot decision summary

Contract: `{contract['contract_sha256']}`. All rates below are bounded-search lower estimates under the frozen 120-sample diagnostic allocation; the balanced-pilot rate is primary for diagnosis, and the Phase-54 cell-weighted estimate is reported in `metrics/overall_correctability.csv` for population projection.

1. **Single-intervention rescue:** {sum(row['single_fixable'] for row in summary_rows)}/{len(samples)} = **{single_rate:.4f}**.
2. **Additional MCTS-only rescue:** {sum(row['mcts_fixable'] for row in summary_rows)}/{len(samples)} = **{mcts_rate:.4f}**.
3. **Total bounded correctability:** {sum(row['total_fixable'] for row in summary_rows)}/{len(samples)} = **{total_rate:.4f}**.
4. **Budget saturation:** Fixable@100 = **{fix100:.4f}**, @200 = **{fix200:.4f}**, and @300 = **{fix300:.4f}**.
5. **200 versus 300:** gain from 200→300 is **{gain_200_300:.4f}**. Using the prospectively frozen **{material_gain:.4f} absolute-rate** materiality yardstick, the evidence selects a cap of **{cap_recommendation}** iterations for any separately authorized full label phase.
6. **Successful-route complexity:** among one post-hoc preferred route per fixable sample, median non-FULL count is **{float(np.median(nonfull_values)) if nonfull_values else float('nan'):.2f}** (IQR {float(np.quantile(nonfull_values, 0.25)) if nonfull_values else float('nan'):.2f}–{float(np.quantile(nonfull_values, 0.75)) if nonfull_values else float('nan'):.2f}).
7. **Multiple successful first actions:** observed for **{multi_actions}/{len(first_action_sets)}** fixable samples ({multi_actions / len(first_action_sets) if first_action_sets else float('nan'):.4f}). This is discovery support, not proof that unobserved actions fail.
8. **Trigger-depth variation:** the max–min total-correctability span across the four bins is **{depth_span:.4f}**; exact rates are in `metrics/trigger_depth_breakdown.csv`.
9. **Dataset variation:** the max–min total-correctability span across GQA/ChartQA/TextVQA is **{dataset_span:.4f}**; exact rates are in `metrics/dataset_breakdown.csv`.
10. **Projected full-cohort cost:** approximately **{weighted_projected_terminals:.0f} terminal routes**, **{weighted_projected_gpu_seconds / 3600.0:.2f} GPU-hours**, or **{weighted_projected_gpu_seconds / 3600.0 / 4.0:.2f} wall-hours** at four equally utilized GPUs, based on dataset×depth-cell means.
11. **Sequential Stage-2 justification:** **{'yes, within this bounded pilot' if sequential_justified else 'not established'}**. MCTS-only rescue {'was observed beyond exhaustive singles' if sequential_justified else 'was not observed beyond exhaustive singles'}; this says whether multi-layer search added discovered labels, not whether a learned policy will generalize.
12. **Recommended MCTS cap:** **{cap_recommendation} iterations/sample**, conditional on separate authorization for full corrective-label generation.

## Execution and label integrity

- All 120 frozen UIDs completed exactly once across four ranks; there were no failed or partial sample records.
- Every sample reproduced its current dense generated tokens, answer, prompt hash, and LMMS correctness under native all-FULL execution before search.
- Layers before each Phase-54 trigger remained FULL. All suffix rewards were binary current LMMS-Eval correctness.
- All {len(successful_route_manifest)} retained successful routes were exactly replayed; {len(future_index_rows)} compact pre-layer action-state rows were stored with contract/model/code/schema/source/run provenance.
- Single-fixable samples were excluded from MCTS. The 39 triggered Dense-C rows, validation/test cohorts, remaining train candidates, Stage-2 training, and external evaluation were not executed.

## Stop decision

Phase 55 stops here. The cap recommendation and whether to scale label generation are conclusions for user review, not authorization to search the remaining 1,761 rows or train Stage 2.
"""
    _atomic_bytes(output_root / "decision_summary.md", decision.encode())

    required = [
        "protocol.md",
        "pilot_manifest.jsonl",
        "single_search/execution_rows.jsonl",
        "single_search/successful_routes.jsonl",
        "single_search/summary.csv",
        "mcts/search_rows.jsonl",
        "mcts/successful_routes.jsonl",
        "mcts/per_sample_summary.csv",
        "metrics/overall_correctability.csv",
        "metrics/budget_saturation.csv",
        "metrics/trigger_depth_breakdown.csv",
        "metrics/dataset_breakdown.csv",
        "metrics/route_complexity.csv",
        "metrics/first_action_distribution.csv",
        "metrics/compute_summary.csv",
        "future_stage2_labels/routed_training_states.jsonl",
        "future_stage2_labels/successful_route_manifest.jsonl",
        "figures/fixability_vs_mcts_budget.png",
        "figures/single_vs_mcts_rescue.png",
        "figures/correctability_by_trigger_depth.png",
        "figures/correctability_by_dataset.png",
        "figures/intervention_count_distribution.png",
        "decision_summary.md",
    ]
    missing = [relative for relative in required if not (output_root / relative).is_file()]
    if missing:
        raise RuntimeError(f"required pilot artifacts are missing: {missing}")
    artifact_paths = sorted(
        path for path in output_root.rglob("*")
        if path.is_file()
        and "work" not in path.relative_to(output_root).parts
        and path.name != "artifact_manifest.json"
    )
    artifact_manifest = {
        "schema_version": "corrective_search_pilot_artifact_manifest_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "pilot_records": len(samples),
        "outcome_counts": dict(Counter(sample["outcome_class"] for sample in samples)),
        "successful_route_records": len(successful_route_manifest),
        "routed_training_state_records": len(future_index_rows),
        "files": {
            str(path.relative_to(output_root)): file_sha256(path) for path in artifact_paths
        },
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "artifact_manifest.json", artifact_manifest)
    print(
        json.dumps(
            {
                "passed": True,
                "contract_sha256": contract["contract_sha256"],
                "outcome_counts": artifact_manifest["outcome_counts"],
                "fixable_at_100": fix100,
                "fixable_at_200": fix200,
                "fixable_at_300": fix300,
                "recommended_cap": cap_recommendation,
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "worker", "finalize-smoke", "aggregate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "pilot"))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "worker":
        if args.mode is None or args.rank is None:
            raise SystemExit("worker requires --mode and --rank")
        worker(
            args.config,
            mode=args.mode,
            rank=args.rank,
            world_size=args.world_size,
            resume=args.resume,
        )
    elif args.command == "finalize-smoke":
        finalize_smoke(args.config)
    else:
        aggregate(args.config)


if __name__ == "__main__":
    main()
