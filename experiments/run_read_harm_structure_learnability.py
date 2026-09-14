#!/usr/bin/env python3
"""Prepare and extract the frozen READ-harm structure/learnability corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binary_policy.executor import capture_four_action_route, one_step_four_action_from_baseline  # noqa: E402
from binary_policy.executor.inputs import build_binary_inputs, resolve_decoder, scatter_streams  # noqa: E402
from dense_failure_stage1.runtime import build_dense_inputs, configure_dense_determinism  # noqa: E402
from dense_failure_stage2.closed_loop_trajectory_set import compact_router_state  # noqa: E402
from dense_failure_stage2.counterfactual_identifiability import tensor_sha256  # noqa: E402
from dense_failure_stage2.read_harm_learnability import (  # noqa: E402
    classify_read_behavior,
    summarize_read_attention,
    summarize_read_update,
    validate_attention_reconstruction,
    validate_feature_census,
)
from experiments.run_counterfactual_effect_identifiability import (  # noqa: E402
    _baseline_for_routed_state,
    _canonical_post,
    _internal_index,
    _load_programs,
    _select_smoke,
)
from experiments.run_predictability_stepA_measurement import _cached_state  # noqa: E402
from experiments.run_stage2_v1_training_revised import _load_model  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/read_harm_structure_learnability_v1.json"
DEFAULT_FEATURE_GROUPS = json.loads(DEFAULT_CONFIG.read_text())["features"]["groups"]
ALLOWED_ROOTS = (ROOT.resolve(), Path("/mnt/hyemin").resolve())
BOUND_CODE = (
    "configs/read_harm_structure_learnability_v1.json",
    "plans/read_harm_structure_learnability_plan.md",
    "dense_failure_stage2/read_harm_learnability.py",
    "experiments/run_read_harm_structure_learnability.py",
    "experiments/analyze_read_harm_structure_learnability.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "dense_failure_stage2/closed_loop_trajectory_set.py",
    "dense_failure_stage2/counterfactual_identifiability.py",
    "dense_failure_stage2/predictability_learnability.py",
    "dense_failure_stage1/runtime.py",
    "experiments/run_counterfactual_effect_identifiability.py",
    "experiments/run_predictability_stepA_measurement.py",
    "experiments/run_stage2_v1_training_revised.py",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def file_sha256(value: str | Path) -> str:
    digest = sha256()
    with resolve_path(value).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key not in {"contract_sha256", "artifact_manifest_sha256"}}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(value: str | Path) -> dict[str, Any]:
    result = json.loads(resolve_path(value).read_text())
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {value}")
    return result


def read_jsonl(value: str | Path) -> list[dict[str, Any]]:
    rows = []
    with resolve_path(value).open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {value}:{number}")
            rows.append(row)
    return rows


def read_csv(value: str | Path) -> list[dict[str, str]]:
    with resolve_path(value).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic(path, "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows).encode())


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import io

    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic(path, stream.getvalue().encode())


def command_output(args: Sequence[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed {args}: {result.stderr.strip()}")
    return result.stdout.strip()


def git_state() -> dict[str, str]:
    return {
        "commit": command_output(("git", "rev-parse", "HEAD")),
        "branch": command_output(("git", "branch", "--show-current")),
        "worktree_status": command_output(("git", "status", "--short")),
    }


def runtime_state() -> dict[str, Any]:
    packages = ("torch", "transformers", "numpy", "lmms-eval", "qwen-vl-utils", "Pillow", "av")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return {
        "python": sys.version.split()[0],
        "packages": versions,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def model_file_hashes(snapshot: Path) -> dict[str, str]:
    names = (
        "config.json", "generation_config.json", "preprocessor_config.json", "chat_template.json",
        "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
        "model.safetensors.index.json", "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors", "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors", "model-00005-of-00005.safetensors",
    )
    output = {}
    for name in names:
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError(f"model snapshot file missing: {path}")
        output[name] = file_sha256(path)
    return output


def _index_unique(rows: Sequence[Mapping[str, Any]], key: str, name: str) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        value = str(row[key])
        if value in output:
            raise RuntimeError(f"duplicate {name}: {value}")
        output[value] = dict(row)
    return output


def _state_population(config: Mapping[str, Any], domain: str) -> list[dict[str, Any]]:
    states = read_jsonl(config["sources"][f"{domain}_states"])
    utilities = _index_unique(read_csv(config["sources"][f"{domain}_utilities"]), "state_id", f"{domain} utility")
    output = []
    for state in states:
        state_id = str(state["state_id"])
        utility = utilities.get(state_id)
        if utility is None:
            raise RuntimeError(f"state lacks frozen utility: {state_id}")
        q_full = float(utility["q_full"])
        q_wo = float(utility["q_write_only"])
        parent = float(utility["u_read_w1"])
        if not math.isclose(parent, q_full - q_wo, abs_tol=1e-9):
            raise RuntimeError(f"parent READ utility algebra differs: {state_id}")
        h_r = q_wo - q_full
        full_correct = str(utility["c_full"]).lower() == "true"
        wo_correct = str(utility["c_write_only"]).lower() == "true"
        output.append(
            {
                **state,
                "schema_version": "read_harm_state_v1",
                "h_r": h_r,
                "q_full": q_full,
                "q_write_only": q_wo,
                "full_correct": full_correct,
                "write_only_correct": wo_correct,
                "cohort": classify_read_behavior(full_correct, wo_correct),
                "read_sign": "harmful" if h_r > 0.0 else "beneficial" if h_r < 0.0 else "zero",
            }
        )
    output.sort(key=lambda row: (str(row["uid"]), int(row["layer"]), str(row["state_id"])))
    return output


def _schedule(rows: Sequence[Mapping[str, Any]], world_size: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["uid"])].append(row)
    load = [0] * int(world_size)
    output = []
    for uid, values in sorted(grouped.items(), key=lambda item: (-len(item[1]), sha256(item[0].encode()).hexdigest())):
        rank = min(range(int(world_size)), key=lambda value: (load[value], value))
        anchors = len({str(row.get("anchor_program_id", "dense")) for row in values})
        cost = 28 * anchors + len(values)
        load[rank] += cost
        output.append({
            "uid": uid,
            "worker_rank": rank,
            "estimated_layer_calls": cost,
            "state_ids": [str(row["state_id"]) for row in values],
        })
    return output


def prepare(config_path: Path) -> None:
    config = read_json(config_path)
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_work_root"])
    external_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "population", "structure", "features", "matched_mechanism", "learnability",
        "generalization", "routed_secondary", "statistics", "figures", "summaries", "validation",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    work = output_root / "work"
    if work.exists() and not work.is_symlink():
        raise RuntimeError("existing work path is not a symlink")
    if not work.exists():
        work.symlink_to(external_root, target_is_directory=True)
    if work.resolve() != external_root:
        raise RuntimeError("external work binding differs")

    dense = _state_population(config, "dense")
    routed = _state_population(config, "routed")
    expected = config["population"]
    if len(dense) != int(expected["dense_states"]) or len({row["uid"] for row in dense}) != int(expected["dense_uids"]):
        raise RuntimeError("dense population differs")
    if len({row["image_group_id"] for row in dense}) != int(expected["dense_image_groups"]):
        raise RuntimeError("dense image-group population differs")
    if len(routed) != int(expected["routed_states"]) or len({row["uid"] for row in routed}) != int(expected["routed_uids"]):
        raise RuntimeError("routed population differs")
    atomic_jsonl(output_root / "population/dense_read_state_manifest.jsonl", dense)
    atomic_jsonl(output_root / "population/routed_read_state_manifest.jsonl", routed)
    registry = read_jsonl(config["split"]["registry"])
    triggered = {str(row["uid"]) for row in dense}
    inherited = [row for row in registry if str(row["uid"]) in triggered]
    if len(inherited) != int(expected["dense_uids"]):
        raise RuntimeError("inherited fold registry lacks dense UIDs")
    atomic_jsonl(output_root / "population/group_registry.jsonl", inherited)
    counts = Counter(
        (domain, str(row["dataset"]), str(row["source_regime"]), str(row["cohort"]), str(row["read_sign"]))
        for domain, rows in (("dense", dense), ("routed", routed)) for row in rows
    )
    atomic_csv(output_root / "population/cohort_counts.csv", [
        {"domain": key[0], "dataset": key[1], "source_regime": key[2], "cohort": key[3], "read_sign": key[4], "states": value}
        for key, value in sorted(counts.items())
    ])
    atomic_jsonl(output_root / "work/dense_schedule.jsonl", _schedule(dense, int(config["world_size"])))
    atomic_jsonl(output_root / "work/routed_schedule.jsonl", _schedule(routed, int(config["world_size"])))

    source_paths = {
        **{f"source:{key}": value for key, value in config["sources"].items()},
        "split:registry": config["split"]["registry"],
        "generalization:similarity": config["generalization"]["similarity"],
        "generalization:cluster_registry": config["generalization"]["cluster_registry"],
    }
    for fold in range(int(config["split"]["folds"])):
        source_paths[f"split:inner_roles:{fold}"] = str(config["split"]["inner_roles_pattern"]).format(fold=fold)
    contract = {
        "schema_version": "read_harm_structure_learnability_contract_v1",
        "static_config": config,
        "config_sha256": file_sha256(config_path),
        "source_sha256": {key: file_sha256(path) for key, path in source_paths.items()},
        "bound_code_sha256": {path: file_sha256(path) for path in BOUND_CODE},
        "prepared_sha256": {
            str(path.relative_to(output_root)): file_sha256(path)
            for path in (
                output_root / "population/dense_read_state_manifest.jsonl",
                output_root / "population/routed_read_state_manifest.jsonl",
                output_root / "population/cohort_counts.csv",
                output_root / "population/group_registry.jsonl",
                output_root / "work/dense_schedule.jsonl",
                output_root / "work/routed_schedule.jsonl",
            )
        },
        "model_snapshot_sha256": model_file_hashes(resolve_path(config["model"]["snapshot_path"])),
        "git": git_state(),
        "runtime": runtime_state(),
        "review": {
            "verdict": "revise_then_proceed",
            "operation_level_validation_added": True,
            "routed_winner_rule_frozen": config["winner_selection"],
        },
        "population": {
            "dense_states": len(dense), "dense_uids": len({row["uid"] for row in dense}),
            "dense_image_groups": len({row["image_group_id"] for row in dense}),
            "routed_states": len(routed), "routed_uids": len({row["uid"] for row in routed}),
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_contract.json", contract)
    feature_groups = config["features"]["groups"]
    atomic_json(output_root / "features/feature_group_manifest.json", {
        "contract_sha256": contract["contract_sha256"],
        "groups": feature_groups,
        "F_ALL": [name for group in config["features"]["all_group_order"] for name in feature_groups[group]],
    })
    protocol = f"""# READ-harm structure and learnability protocol\n\n- Contract: `{contract['contract_sha256']}`\n- Target: `H_R = q_WRITE_ONLY - q_FULL`; positive means READ is harmful.\n- Primary: all {len(dense):,} dense states / {len({row['uid'] for row in dense}):,} UIDs.\n- Secondary: all {len(routed):,} exact routed states / {len({row['uid'] for row in routed}):,} UIDs.\n- READ pair only: FULL versus WRITE_ONLY. No WRITE study or deployment router.\n- F4-F7 must pass operation-level SDPA reconstruction validation before extraction.\n- The routed feature/model is selected by frozen dense OOF Spearman, harmful-AUROC, then fixed group/model tie-breaks.\n"""
    _atomic(output_root / "protocol.md", protocol.encode())
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"], **contract["population"]}, sort_keys=True))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = read_json(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_contract.json")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise RuntimeError("READ contract hash differs")
    if contract.get("config_sha256") != file_sha256(config_path) or contract.get("static_config") != config:
        raise RuntimeError("READ config differs from frozen contract")
    for key, expected in contract["source_sha256"].items():
        if key.startswith("source:"):
            path = config["sources"][key.split(":", 1)[1]]
        elif key == "split:registry":
            path = config["split"]["registry"]
        elif key.startswith("split:inner_roles:"):
            path = str(config["split"]["inner_roles_pattern"]).format(fold=int(key.rsplit(":", 1)[1]))
        elif key == "generalization:similarity":
            path = config["generalization"]["similarity"]
        elif key == "generalization:cluster_registry":
            path = config["generalization"]["cluster_registry"]
        else:
            raise RuntimeError(f"unknown frozen source: {key}")
        if file_sha256(path) != expected:
            raise RuntimeError(f"frozen source differs: {key}")
    for path, expected in contract["bound_code_sha256"].items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"bound code differs: {path}")
    for relative, expected in contract["prepared_sha256"].items():
        if file_sha256(output_root / relative) != expected:
            raise RuntimeError(f"prepared artifact differs: {relative}")
    if git_state() != contract["git"] or runtime_state() != contract["runtime"]:
        raise RuntimeError("git or runtime differs from frozen READ contract")
    work = output_root / "work"
    if not work.is_symlink() or work.resolve() != resolve_path(config["external_work_root"]):
        raise RuntimeError("external work binding differs")
    if verify_model and model_file_hashes(resolve_path(config["model"]["snapshot_path"])) != contract["model_snapshot_sha256"]:
        raise RuntimeError("model snapshot differs")
    return contract, output_root


def _load_phase82_text_map(config: Mapping[str, Any], domain: str) -> tuple[Mapping[str, Any], np.memmap]:
    layout = read_json(config["sources"][f"phase82_{domain}_text_cache"])
    spec = layout["files"]["post_text"]
    mmap = np.memmap(resolve_path(spec["path"]), dtype=np.uint16, mode="r", shape=tuple(spec["shape"]))
    return layout, mmap


def _post_text_from_phase82(source: np.memmap, row: Mapping[str, Any], action_index: int) -> torch.Tensor:
    start, count = int(row["text_offset"]), int(row["text_tokens"])
    array = np.array(source[start : start + count, int(action_index), :], copy=True)
    return torch.from_numpy(array).view(torch.bfloat16).unsqueeze(0)


def _projected_qkv(layer, full_norm: torch.Tensor, position_embeddings) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention = layer.self_attn
    batch, tokens, _ = full_norm.shape
    q = attention.q_proj(full_norm).view(batch, tokens, attention.num_heads, attention.head_dim).transpose(1, 2)
    k = attention.k_proj(full_norm).view(batch, tokens, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    v = attention.v_proj(full_norm).view(batch, tokens, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attention.config.rope_parameters["mrope_section"])
    return q, repeat_kv(k, attention.num_key_value_groups), repeat_kv(v, attention.num_key_value_groups)


@torch.inference_mode()
def _read_features(
    wrapped,
    baseline,
    row: Mapping[str, Any],
    phase82_text: np.memmap,
    *,
    groups: set[str],
    operation_validation: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    layer_index = int(row["layer"])
    decoder = resolve_decoder(wrapped)
    layer = decoder.layers[layer_index]
    meta = baseline.inputs
    pre_text, pre_visual = baseline.pre_layer_states[layer_index]
    validation: dict[str, Any] = {}
    features: dict[str, float] = {}

    needs_update = bool(groups.intersection({"F1", "F2", "F3"}))
    if needs_update:
        full_output = one_step_four_action_from_baseline(wrapped, baseline, layer_index, "FULL")
        off_output = one_step_four_action_from_baseline(wrapped, baseline, layer_index, "WRITE_ONLY")
        if (full_output.execution.read_on, full_output.execution.write_on) != (True, True):
            raise RuntimeError("FULL action bits differ")
        if (off_output.execution.read_on, off_output.execution.write_on) != (False, True):
            raise RuntimeError("WRITE_ONLY action bits differ")
        compact_full = compact_router_state(full_output.post_text_state, full_output.post_visual_state, meta.text_valid_mask, meta.visual_valid_mask)
        compact_off = compact_router_state(off_output.post_text_state, off_output.post_visual_state, meta.text_valid_mask, meta.visual_valid_mask)
        expected_full = _post_text_from_phase82(phase82_text, row, 0)
        expected_off = _post_text_from_phase82(phase82_text, row, 1)
        compact_exact = torch.equal(compact_full["text_states"].cpu(), expected_full) and torch.equal(compact_off["text_states"].cpu(), expected_off)
        if not compact_exact:
            raise RuntimeError(f"Phase-82 compact post-text differs: {row['state_id']}")
        text_valid = meta.text_valid_mask[0].bool()
        text_positions = meta.text_indices[0].to(pre_text.device)
        visual_positions = meta.visual_indices[0][meta.visual_valid_mask[0]].to(pre_text.device)
        post_visual_boundary = int(visual_positions.max().item())
        scope = text_valid & (text_positions > post_visual_boundary)
        if not bool(scope.any()):
            raise RuntimeError(f"no post-visual text/control rows: {row['state_id']}")
        update = summarize_read_update(
            pre_text[0, text_valid][-1].float().cpu().numpy(),
            off_output.post_text_state[0, text_valid][-1].float().cpu().numpy(),
            full_output.post_text_state[0, text_valid][-1].float().cpu().numpy(),
            (full_output.post_text_state[0, scope] - off_output.post_text_state[0, scope]).float().cpu().numpy(),
            top_k=5,
        )
        features.update({key: value for key, value in update.items() if key[:2].upper() in groups})
        validation.update({
            "phase82_compact_post_text_exact": compact_exact,
            "post_visual_text_rows": int(scope.sum()),
            "full_post_text_sha256": tensor_sha256(compact_full["text_states"]),
            "write_only_post_text_sha256": tensor_sha256(compact_off["text_states"]),
        })

    attention_groups = groups.intersection({"F4", "F5", "F6", "F7"})
    if attention_groups:
        full = scatter_streams(pre_text, pre_visual, meta)
        normed = layer.input_layernorm(full)
        position_embeddings = decoder.rotary_emb(normed, meta.full_position_ids.to(normed.device))
        q, k, v = _projected_qkv(layer, normed, position_embeddings)
        text_valid_count = int(meta.text_valid_mask[0].sum())
        query_full_index = int(meta.text_indices[0, text_valid_count - 1])
        visual_full_mask = torch.zeros(full.shape[1], dtype=torch.bool, device=full.device)
        valid_visual_indices = meta.visual_indices[0, meta.visual_valid_mask[0]].to(full.device)
        visual_full_mask[valid_visual_indices] = True
        visual_positions = meta.visual_position_ids[:, 0, meta.visual_valid_mask[0]].T[:, 1:3]
        attention = summarize_read_attention(
            q[0, :, query_full_index], k[0], v[0],
            visual_mask=visual_full_mask,
            output_projection_weight=layer.self_attn.o_proj.weight,
            residual=full[0, query_full_index],
            visual_positions=visual_positions,
        )
        features.update({key: value for key, value in attention.items() if key[:2].upper() in groups})
        validation.update({"query_full_index": query_full_index, "visual_tokens": int(visual_full_mask.sum()), "full_tokens": int(full.shape[1])})

        if operation_validation:
            actual_full = layer.self_attn(
                hidden_states=normed,
                attention_mask=None,
                position_embeddings=position_embeddings,
                use_cache=False,
            )[0][0, query_full_index].float()
            # Match the executor's full-query causal SDPA shape. A one-query
            # reconstruction is mathematically equivalent for the final row,
            # but selects a different fused kernel and introduces avoidable
            # BF16 accumulation differences on long multimodal prompts.
            manual_head = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            manual_all = layer.self_attn.o_proj(manual_head.transpose(1, 2).reshape(1, full.shape[1], -1))
            manual_full = manual_all[0, query_full_index].float()
            error = (actual_full - manual_full).abs()
            cosine = F.cosine_similarity(actual_full, manual_full, dim=0)
            validation.update({
                "sdpa_reconstruction_max_abs": float(error.max()),
                "sdpa_reconstruction_mean_abs": float(error.mean()),
                "sdpa_reconstruction_cosine": float(cosine),
            })
    if set(features) != {name for group in groups for name in DEFAULT_FEATURE_GROUPS[group]}:
        raise RuntimeError(f"feature schema differs for {row['state_id']}")
    if not all(math.isfinite(float(value)) for value in features.values()):
        raise RuntimeError(f"non-finite READ feature: {row['state_id']}")
    return features, validation


def _baseline_groups(wrapped, prepared, rows: Sequence[Mapping[str, Any]], domain: str, programs):
    if domain == "dense":
        yield "dense", capture_four_action_route(
            wrapped, {}, ["FULL"] * 28, prepared_inputs=prepared, use_cache=True, native_full_rows=True
        ), list(rows)
        return
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["anchor_program_id"])].append(row)
    for anchor, values in sorted(grouped.items()):
        yield anchor, _baseline_for_routed_state(wrapped, prepared, values[0], programs), values


def _selected_groups(config: Mapping[str, Any], output_root: Path, domain: str) -> set[str]:
    if domain == "dense":
        return set(config["features"]["all_group_order"])
    winner = read_json(output_root / "learnability/dense_winner.json")
    group = str(winner["feature_group"])
    return set(config["features"]["all_group_order"]) if group == "F_ALL" else {group}


def smoke(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    torch.cuda.set_device(int(device_index))
    device = torch.device(f"cuda:{int(device_index)}")
    processor, _base, wrapped = _load_model(config, device)
    internal = _internal_index(config)
    programs = _load_programs(config)
    report_rows = []
    for domain in ("dense", "routed"):
        rows = read_jsonl(output_root / f"population/{domain}_read_state_manifest.jsonl")
        selected = _select_smoke(rows, int(config["validation"][f"{domain}_states"]), require_layer27=bool(config["validation"]["require_layer_27"]))
        _, phase82_text = _load_phase82_text_map(config, domain)
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[str(row["uid"])].append(row)
        for uid, uid_rows in sorted(grouped.items()):
            sample = internal[uid]["sample"]
            inputs, metadata = build_dense_inputs(processor, sample, device)
            if metadata["consumed_image_sha256"] != sample["image_content_sha256"]:
                raise RuntimeError(f"image hash differs: {uid}")
            prepared = build_binary_inputs(wrapped, inputs)
            for anchor, baseline, baseline_rows in _baseline_groups(wrapped, prepared, uid_rows, domain, programs):
                for row in baseline_rows:
                    pre_text, pre_visual = baseline.pre_layer_states[int(row["layer"])]
                    live = _cached_state(pre_text, pre_visual, prepared.text_valid_mask, prepared.visual_valid_mask)
                    if live["state_sha256"] != str(row["pre_state_sha256"]):
                        raise RuntimeError(f"smoke pre-state differs: {row['state_id']}")
                    before = [one_step_four_action_from_baseline(wrapped, baseline, int(row["layer"]), action) for action in ("FULL", "WRITE_ONLY")]
                    first_features, first_validation = _read_features(
                        wrapped, baseline, row, phase82_text,
                        groups=set(config["features"]["all_group_order"]), operation_validation=True,
                    )
                    second_features, second_validation = _read_features(
                        wrapped, baseline, row, phase82_text,
                        groups=set(config["features"]["all_group_order"]), operation_validation=True,
                    )
                    after = [one_step_four_action_from_baseline(wrapped, baseline, int(row["layer"]), action) for action in ("FULL", "WRITE_ONLY")]
                    branch_exact = all(
                        torch.equal(left.post_text_state, right.post_text_state)
                        and torch.equal(left.post_visual_state, right.post_visual_state)
                        for left, right in zip(before, after)
                    )
                    threshold = config["validation"]
                    operation_valid = validate_attention_reconstruction(
                        max_abs=float(first_validation["sdpa_reconstruction_max_abs"]),
                        mean_abs=float(first_validation["sdpa_reconstruction_mean_abs"]),
                        cosine=float(first_validation["sdpa_reconstruction_cosine"]),
                        max_abs_tolerance=float(threshold["sdpa_reconstruction_max_abs_tolerance"]),
                        mean_abs_tolerance=float(threshold["sdpa_reconstruction_mean_abs_tolerance"]),
                        min_cosine=float(threshold["sdpa_reconstruction_min_cosine"]),
                    )
                    row_report = {
                        "contract_sha256": contract["contract_sha256"], "domain": domain,
                        "state_id": row["state_id"], "uid": uid, "dataset": row["dataset"],
                        "dense_wrong": row["dense_wrong"], "layer": row["layer"], "anchor": anchor,
                        "image_sha256_exact": True, "pre_state_exact": True,
                        "phase82_compact_post_text_exact": bool(first_validation["phase82_compact_post_text_exact"]),
                        "exact_feature_repeat": first_features == second_features and first_validation == second_validation,
                        "branch_exact_after_feature_extraction": branch_exact,
                        "operation_reconstruction_valid": operation_valid,
                        "operation_validation": first_validation,
                    }
                    row_report["passed"] = all(
                        bool(row_report[key]) for key in (
                            "image_sha256_exact", "pre_state_exact", "phase82_compact_post_text_exact",
                            "exact_feature_repeat", "branch_exact_after_feature_extraction", "operation_reconstruction_valid",
                        )
                    )
                    if not row_report["passed"]:
                        print(json.dumps({"smoke_failure": row_report}, sort_keys=True), flush=True)
                        raise RuntimeError(f"READ operation smoke failed: {row['state_id']}")
                    report_rows.append(row_report)
                del baseline
            del prepared, inputs
            torch.cuda.empty_cache()
    atomic_jsonl(output_root / "validation/operation_smoke_rows.jsonl", report_rows)
    atomic_json(output_root / "validation/operation_smoke_report.json", {
        "contract_sha256": contract["contract_sha256"],
        "rows": len(report_rows), "dense_rows": sum(row["domain"] == "dense" for row in report_rows),
        "routed_rows": sum(row["domain"] == "routed" for row in report_rows),
        "maximum_sdpa_reconstruction_error": max(row["operation_validation"]["sdpa_reconstruction_max_abs"] for row in report_rows),
        "maximum_sdpa_reconstruction_mean_error": max(row["operation_validation"]["sdpa_reconstruction_mean_abs"] for row in report_rows),
        "minimum_sdpa_reconstruction_cosine": min(row["operation_validation"]["sdpa_reconstruction_cosine"] for row in report_rows),
        "passed": all(row["passed"] for row in report_rows),
    })
    print(json.dumps({"smoke_passed": True, "rows": len(report_rows)}, sort_keys=True))


def _require_smoke(contract: Mapping[str, Any], output_root: Path) -> None:
    report = read_json(output_root / "validation/operation_smoke_report.json")
    if report.get("contract_sha256") != contract["contract_sha256"] or not bool(report.get("passed")):
        raise RuntimeError("operation smoke is absent or failed")


def extract_worker(config_path: Path, domain: str, rank: int, *, resume: bool) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    _require_smoke(contract, output_root)
    config = contract["static_config"]
    if rank < 0 or rank >= int(config["world_size"]):
        raise ValueError("rank outside frozen world size")
    groups = _selected_groups(config, output_root, domain)
    configure_dense_determinism(int(config["seed"]) + int(rank), config["backend_settings"])
    torch.cuda.set_device(int(rank))
    device = torch.device(f"cuda:{int(rank)}")
    processor, _base, wrapped = _load_model(config, device)
    internal = _internal_index(config)
    programs = _load_programs(config)
    rows = read_jsonl(output_root / f"population/{domain}_read_state_manifest.jsonl")
    rows_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_uid[str(row["uid"])].append(row)
    schedule = [row for row in read_jsonl(output_root / f"work/{domain}_schedule.jsonl") if int(row["worker_rank"]) == int(rank)]
    _, phase82_text = _load_phase82_text_map(config, domain)
    rank_root = output_root / f"work/extraction/{domain}/rank{int(rank):02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    started = time.monotonic()
    for item in schedule:
        uid = str(item["uid"])
        result_path = rank_root / f"{sha256(uid.encode()).hexdigest()[:24]}.json"
        if resume and result_path.is_file():
            old = read_json(result_path)
            if old.get("contract_sha256") == contract["contract_sha256"] and old.get("state_ids") == item["state_ids"] and old.get("feature_groups") == sorted(groups):
                completed += 1
                continue
        sample = internal[uid]["sample"]
        inputs, metadata = build_dense_inputs(processor, sample, device)
        if metadata["consumed_image_sha256"] != sample["image_content_sha256"]:
            raise RuntimeError(f"image hash differs: {uid}")
        prepared = build_binary_inputs(wrapped, inputs)
        feature_rows = []
        for anchor, baseline, baseline_rows in _baseline_groups(wrapped, prepared, rows_by_uid[uid], domain, programs):
            for row in baseline_rows:
                pre_text, pre_visual = baseline.pre_layer_states[int(row["layer"])]
                live = _cached_state(pre_text, pre_visual, prepared.text_valid_mask, prepared.visual_valid_mask)
                if live["state_sha256"] != str(row["pre_state_sha256"]):
                    raise RuntimeError(f"live pre-state differs: {row['state_id']}")
                features, validation = _read_features(
                    wrapped, baseline, row, phase82_text, groups=groups, operation_validation=False
                )
                feature_rows.append({
                    "schema_version": "read_operation_features_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "domain": domain, "state_id": row["state_id"], "uid": uid,
                    "dataset": row["dataset"], "source_regime": row["source_regime"],
                    "image_group_id": row["image_group_id"], "layer": row["layer"],
                    "trigger_layer": row["trigger_layer"], "trigger_relative_depth": row["trigger_relative_depth"],
                    "dense_correct": row["dense_correct"], "dense_wrong": row["dense_wrong"],
                    "h_r": row["h_r"], "cohort": row["cohort"], "read_sign": row["read_sign"],
                    "text_token_count": int(prepared.text_valid_mask.sum()),
                    "visual_token_count": int(prepared.visual_valid_mask.sum()),
                    "feature_groups": sorted(groups), "features": features,
                    "pre_state_sha256": live["state_sha256"],
                    "phase82_compact_post_text_exact": validation.get("phase82_compact_post_text_exact"),
                    "anchor_program_id": None if domain == "dense" else anchor,
                })
            del baseline
        observed = [str(row["state_id"]) for row in feature_rows]
        if len(observed) != len(set(observed)) or sorted(observed) != sorted(item["state_ids"]):
            raise RuntimeError(f"UID feature completion differs: {uid}")
        atomic_json(result_path, {
            "schema_version": "read_operation_uid_features_v1",
            "contract_sha256": contract["contract_sha256"], "domain": domain,
            "uid": uid, "state_ids": item["state_ids"], "feature_groups": sorted(groups),
            "image_sha256": metadata["consumed_image_sha256"], "feature_rows": feature_rows,
        })
        completed += 1
        del prepared, inputs, feature_rows
        torch.cuda.empty_cache()
        if completed % 5 == 0 or completed == len(schedule):
            print(json.dumps({"domain": domain, "rank": rank, "completed_uids": completed, "assigned_uids": len(schedule), "elapsed_seconds": time.monotonic() - started}), flush=True)
    atomic_json(rank_root / "complete.json", {
        "contract_sha256": contract["contract_sha256"], "domain": domain, "rank": rank,
        "feature_groups": sorted(groups), "expected_uids": len(schedule), "completed_uids": completed,
        "elapsed_seconds": time.monotonic() - started,
    })


def finalize_extraction(config_path: Path, domain: str) -> None:
    contract, output_root = verify_contract(config_path)
    _require_smoke(contract, output_root)
    config = contract["static_config"]
    expected_rows = read_jsonl(output_root / f"population/{domain}_read_state_manifest.jsonl")
    expected_ids = [str(row["state_id"]) for row in expected_rows]
    groups = _selected_groups(config, output_root, domain)
    uid_results = []
    for rank in range(int(config["world_size"])):
        rank_root = output_root / f"work/extraction/{domain}/rank{rank:02d}"
        complete = read_json(rank_root / "complete.json")
        if complete.get("contract_sha256") != contract["contract_sha256"] or complete.get("completed_uids") != complete.get("expected_uids") or complete.get("feature_groups") != sorted(groups):
            raise RuntimeError(f"partial or incompatible extraction rank: {domain}/{rank}")
        uid_results.extend(read_json(path) for path in sorted(rank_root.glob("*.json")) if path.name != "complete.json")
    observed_uids = [str(row["uid"]) for row in uid_results]
    expected_uids = sorted({str(row["uid"]) for row in expected_rows})
    if len(observed_uids) != len(set(observed_uids)) or sorted(observed_uids) != expected_uids:
        raise RuntimeError(f"global UID feature completion differs: {domain}")
    feature_rows = [row for result in uid_results for row in result["feature_rows"]]
    validate_feature_census(expected_ids, feature_rows)
    destination = output_root / ("features/read_operation_features.jsonl" if domain == "dense" else "routed_secondary/read_operation_features.jsonl")
    atomic_jsonl(destination, sorted(feature_rows, key=lambda row: str(row["state_id"])))
    atomic_json(output_root / ("features/dense_feature_completion.json" if domain == "dense" else "routed_secondary/feature_completion.json"), {
        "contract_sha256": contract["contract_sha256"], "domain": domain,
        "states": len(feature_rows), "uids": len(expected_uids), "feature_groups": sorted(groups),
        "feature_file": str(destination.relative_to(output_root)), "feature_file_sha256": file_sha256(destination),
        "complete": True,
    })
    print(json.dumps({"domain": domain, "states": len(feature_rows), "uids": len(expected_uids), "feature_groups": sorted(groups), "complete": True}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "extract", "finalize-extraction"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain", choices=("dense", "routed"))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "smoke":
        smoke(args.config, args.device)
    elif args.command == "extract":
        if args.domain is None:
            raise SystemExit("--domain is required")
        extract_worker(args.config, args.domain, args.rank, resume=args.resume)
    else:
        if args.domain is None:
            raise SystemExit("--domain is required")
        finalize_extraction(args.config, args.domain)


if __name__ == "__main__":
    main()
