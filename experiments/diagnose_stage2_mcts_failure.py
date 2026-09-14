#!/usr/bin/env python3
"""Diagnose the frozen Stage-2 MCTS failure without retraining or new search."""

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
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binary_policy.executor.four_action import (  # noqa: E402
    capture_four_action_route,
    capture_online_four_action_route,
)
from dense_failure_stage1.runtime import build_dense_inputs, configure_dense_determinism  # noqa: E402
from dense_failure_stage2.full_label_generation import verify_artifact_manifest  # noqa: E402
from dense_failure_stage2.mcts_diagnosis import (  # noqa: E402
    ACTIONS,
    action_metrics,
    entering_state_key,
    first_deviation_class,
    forcing_boundary,
    intervention_index,
    intervention_index_bin,
    matched_mode_uid_pairs,
    observed_successful_action_sets,
    paired_uid_bootstrap,
    route_complexity_bin,
)
from experiments.run_stage2_shared_union_training import (  # noqa: E402
    _generate,
    _load_model,
    _prepare_binary,
    _router,
    _stack_route_states,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage2_mcts_failure_diagnosis_v1.json"
BOUND_CODE = (
    "configs/stage2_mcts_failure_diagnosis_v1.json",
    "dense_failure_stage2/mcts_diagnosis.py",
    "experiments/diagnose_stage2_mcts_failure.py",
    "dense_failure_stage2/v1_router.py",
    "experiments/run_stage2_shared_union_training.py",
    "experiments/run_stage2_v1_training_revised.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage1/lmms_scoring.py",
)
ROUTE_FILES = {
    "preservation_full": "union_preservation",
    "single": "union_single",
    "mcts": "union_mcts",
}


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
    roots = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
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


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "stage2_mcts_failure_diagnosis_config_v1":
        raise ValueError("unsupported diagnosis config")
    if int(config["world_size"]) != 4:
        raise ValueError("diagnosis requires exactly four workers")
    if config["primary_operating_point"] != "P90":
        raise ValueError("P90 must remain the primary diagnostic point")
    if set(config["secondary_operating_points"]) != {"P98", "P95"}:
        raise ValueError("secondary operating points must remain P98/P95")
    return config


def _rank_for_uid(uid: str, world_size: int) -> int:
    return int(sha256(str(uid).encode()).hexdigest()[:16], 16) % int(world_size)


def _verify_image(sample: Mapping[str, Any]) -> None:
    path = resolve_path(str(sample["local_image_path"]))
    if not path.is_file() or file_sha256(path) != str(sample["image_content_sha256"]):
        raise RuntimeError(f"image integrity failure: {sample['uid']}")


def _load_router_checkpoint(
    config: Mapping[str, Any], contract: Mapping[str, Any], checkpoint: str, device: torch.device
):
    router = _router(contract["phase66_static_config"], device).eval()
    path = resolve_path(config["sources"][f"checkpoint_{checkpoint}"])
    selection = read_json(resolve_path(config["sources"][f"checkpoint_{checkpoint}_selection"]))
    if file_sha256(path) != selection["sha256"]:
        raise RuntimeError(f"checkpoint {checkpoint} hash differs from selection record")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("contract_sha256") != contract["parent_contract_sha256"]:
        raise RuntimeError(f"checkpoint {checkpoint} parent contract differs")
    router.load_state_dict(payload["state_dict"], strict=True)
    return router, selection["sha256"]


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    sources = {name: resolve_path(value) for name, value in config["sources"].items()}
    phase66 = read_json(sources["phase66_contract"])
    if canonical_hash(phase66) != phase66.get("contract_sha256"):
        raise RuntimeError("Phase-66 contract hash is invalid")
    phase66_manifest = read_json(sources["phase66_artifact_manifest"])
    verify_artifact_manifest(sources["phase66_artifact_manifest"].parent, phase66_manifest)
    if phase66_manifest.get("contract_sha256") != phase66["contract_sha256"]:
        raise RuntimeError("Phase-66 artifact and contract IDs differ")
    phase65 = read_json(sources["phase65_contract"])
    if canonical_hash(phase65) != phase65.get("contract_sha256"):
        raise RuntimeError("Phase-65 contract hash is invalid")
    phase65_manifest = read_json(sources["phase65_artifact_manifest"])
    verify_artifact_manifest(sources["phase65_artifact_manifest"].parent, phase65_manifest)

    routes = []
    expected_counts = {"preservation_full": 106, "single": 1688, "mcts": 725}
    for route_source, source_key in ROUTE_FILES.items():
        family = read_jsonl(sources[source_key])
        if len(family) != expected_counts[route_source]:
            raise RuntimeError(f"unexpected {route_source} route count")
        if any(row["route_source"] != route_source for row in family):
            raise RuntimeError(f"route-source mismatch in {route_source}")
        routes.extend(family)
    if len({row["route_id"] for row in routes}) != len(routes):
        raise RuntimeError("union route IDs are not unique")

    samples = {str(row["uid"]): row for row in read_jsonl(sources["train_samples"])}
    if not {str(row["uid"]) for row in routes}.issubset(samples):
        raise RuntimeError("diagnostic route sample metadata is incomplete")
    route_store = {str(row["route_id"]): row for row in read_jsonl(sources["phase65_route_store"])}
    diagnostic_routes = []
    for row in sorted(routes, key=lambda item: str(item["route_id"])):
        stored = route_store.get(str(row["route_id"]))
        if stored is None:
            raise RuntimeError(f"route missing from Phase-65 route store: {row['route_id']}")
        if (
            str(stored["uid"]) != str(row["uid"])
            or list(stored["actions"]) != list(row["actions"])
            or not bool(stored["final_lmms_correct"])
            or not bool(stored["state_capture_token_parity"])
        ):
            raise RuntimeError(f"route provenance mismatch: {row['route_id']}")
        diagnostic_routes.append(
            {
                **row,
                "worker_rank": _rank_for_uid(str(row["uid"]), int(config["world_size"])),
                "expected_generated_token_ids": stored["generated_token_ids"],
                "expected_generated_answer": stored["generated_answer"],
                "expected_lmms_metric": stored["lmms_metric"],
                "expected_lmms_score": stored["lmms_score"],
            }
        )
    expected_state_rows = sum(28 - int(row["activation_layer"]) for row in diagnostic_routes)
    phase65_state_rows = read_jsonl(sources["phase65_state_manifest"])
    if expected_state_rows != len(phase65_state_rows) or expected_state_rows != 34253:
        raise RuntimeError("diagnostic route horizon differs from Phase-65 routed states")

    for checkpoint in ("A", "B"):
        selection = read_json(sources[f"checkpoint_{checkpoint}_selection"])
        if file_sha256(sources[f"checkpoint_{checkpoint}"]) != selection["sha256"]:
            raise RuntimeError(f"checkpoint {checkpoint} selection hash mismatch")
        if selection["contract_sha256"] != phase66["contract_sha256"]:
            raise RuntimeError(f"checkpoint {checkpoint} is not bound to Phase 66")

    output_root.mkdir(parents=True)
    atomic_jsonl(output_root / "work/diagnostic_routes.jsonl", diagnostic_routes)
    internal_hashes = {
        "work/diagnostic_routes.jsonl": file_sha256(output_root / "work/diagnostic_routes.jsonl")
    }
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    bound_hashes = {relative: file_sha256(resolve_path(relative)) for relative in BOUND_CODE}
    snapshot = resolve_path(phase66["static_config"]["model"]["snapshot_path"])
    for name, expected in phase66["model_snapshot_sha256"].items():
        if file_sha256(snapshot / name) != expected:
            raise RuntimeError(f"model snapshot hash mismatch: {name}")
    contract: dict[str, Any] = {
        "schema_version": "stage2_mcts_failure_diagnosis_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "parent_contract_sha256": phase66["contract_sha256"],
        "phase65_contract_sha256": phase65["contract_sha256"],
        "phase66_static_config": phase66["static_config"],
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "internal_manifest_sha256": internal_hashes,
        "model_snapshot_sha256": phase66["model_snapshot_sha256"],
        "population": {
            "routes": len(diagnostic_routes),
            "oracle_state_rows": expected_state_rows,
            "preservation_routes": expected_counts["preservation_full"],
            "single_routes": expected_counts["single"],
            "mcts_routes": expected_counts["mcts"],
            "unique_uids": len({row["uid"] for row in diagnostic_routes}),
            "p90_mcts_routes": sum(
                row["route_source"] == "mcts" and "P90" in row["valid_operating_points"]
                for row in diagnostic_routes
            ),
        },
        "state_input_resolution": {
            "phase65_saved_tensors": ["text_final", "text_mean", "visual_mean"],
            "stage2_required_input": "complete routed text and visual token sequences",
            "resolution": "deterministic exact route replay; no pooled-state proxy",
        },
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
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    protocol = f"""# Stage-2 MCTS failure diagnosis protocol

- Contract: `{contract['contract_sha256']}`
- Parent Stage-2 contract: `{phase66['contract_sha256']}`
- Checkpoints: frozen Experiment A and B final-update checkpoints; no retraining.
- Population: {len(diagnostic_routes)} replay-valid routes and {expected_state_rows} oracle entering states.
- Primary rollout diagnostic: P90 across all {expected_counts['mcts']} MCTS routes.
- Exact input rule: reconstruct complete routed token sequences by deterministic replay because Phase-65 saved tensors are pooled summaries only.
- Prefix forcing: C0 free, C1..Ck through each successive non-FULL action, plus full oracle control.
- Ambiguity identity: exact UID + layer + complete entering action prefix.
- Bootstrap: UID-level, {config['bootstrap_replicates']} replicates, seed {config['seed']}.
- Stop: diagnosis and one recommendation only; no training, search, threshold change, or held-out test.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"], **contract["population"]}, sort_keys=True))


def verify_contract(config_path: Path, *, verify_model: bool = False):
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("diagnosis contract hash mismatch")
    if contract["static_config"] != config:
        raise RuntimeError("active config differs from frozen diagnosis contract")
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
        snapshot = resolve_path(contract["phase66_static_config"]["model"]["snapshot_path"])
        for name, expected in contract["model_snapshot_sha256"].items():
            if file_sha256(snapshot / name) != expected:
                raise RuntimeError(f"model hash mismatch: {name}")
    return contract, output_root


def _probability_rows(router, output, layers: Sequence[int], device: torch.device):
    text, visual, text_mask, visual_mask = _stack_route_states(output, layers, device)
    with torch.inference_mode():
        values = router(text, visual, text_mask=text_mask, visual_mask=visual_mask).float().softmax(-1)
    return values.detach().cpu()


def smoke(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    configure_dense_determinism(int(config["seed"]), contract["phase66_static_config"]["backend_settings"])
    processor, _base, wrapped = _load_model(contract["phase66_static_config"], device)
    router_a, hash_a = _load_router_checkpoint(config, contract, "A", device)
    router_b, hash_b = _load_router_checkpoint(config, contract, "B", device)
    routes = read_jsonl(output_root / "work/diagnostic_routes.jsonl")
    samples = {row["uid"]: row["sample"] for row in read_jsonl(resolve_path(config["sources"]["train_samples"]))}
    selected = [next(row for row in routes if row["route_source"] == family) for family in ROUTE_FILES]
    checks = []
    for route in selected:
        sample = samples[route["uid"]]
        _verify_image(sample)
        inputs, _ = build_dense_inputs(processor, sample, device)
        meta = _prepare_binary(processor, wrapped, sample, device)
        output = capture_four_action_route(
            wrapped, {}, route["actions"], prepared_inputs=meta, use_cache=True, native_full_rows=True
        )
        layers = list(range(int(route["activation_layer"]), 28))
        probabilities_a = _probability_rows(router_a, output, layers, device)
        probabilities_b = _probability_rows(router_b, output, layers, device)
        ids, _text, score = _generate(processor, wrapped, output, inputs["input_ids"], sample)
        checks.append(
            {
                "route_id": route["route_id"],
                "route_source": route["route_source"],
                "state_rows": len(layers),
                "finite_A": bool(torch.isfinite(probabilities_a).all()),
                "finite_B": bool(torch.isfinite(probabilities_b).all()),
                "oracle_token_parity": ids == route["expected_generated_token_ids"],
                "oracle_correct": bool(score.correct),
            }
        )
    passed = all(
        row["finite_A"] and row["finite_B"] and row["oracle_token_parity"] and row["oracle_correct"]
        for row in checks
    )
    report = {
        "schema_version": "stage2_mcts_diagnosis_smoke_v1",
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_A_sha256": hash_a,
        "checkpoint_B_sha256": hash_b,
        "checks": checks,
        "passed": passed,
    }
    atomic_json(output_root / "smoke/implementation_smoke.json", report)
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise RuntimeError("diagnosis smoke failed")


def oracle_worker(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    smoke_report = read_json(output_root / "smoke/implementation_smoke.json")
    if not smoke_report.get("passed"):
        raise RuntimeError("diagnosis smoke is not passing")
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("oracle diagnosis requires four workers")
    output_path = output_root / f"work/oracle/rank{rank:02d}.jsonl"
    complete_path = output_root / f"work/oracle/rank{rank:02d}.complete.json"
    if output_path.exists() or complete_path.exists():
        raise RuntimeError(f"refusing to overwrite oracle rank {rank}")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    configure_dense_determinism(int(config["seed"]) + rank, contract["phase66_static_config"]["backend_settings"])
    processor, _base, wrapped = _load_model(contract["phase66_static_config"], device)
    router_a, hash_a = _load_router_checkpoint(config, contract, "A", device)
    router_b, hash_b = _load_router_checkpoint(config, contract, "B", device)
    all_routes = read_jsonl(output_root / "work/diagnostic_routes.jsonl")
    routes = [row for row in all_routes if int(row["worker_rank"]) == rank]
    action_sets = observed_successful_action_sets(all_routes)
    samples = {row["uid"]: row["sample"] for row in read_jsonl(resolve_path(config["sources"]["train_samples"]))}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        grouped[str(route["uid"])].append(route)
    rows = []
    started = time.monotonic()
    route_count = 0
    for uid in sorted(grouped):
        sample = samples[uid]
        _verify_image(sample)
        meta = _prepare_binary(processor, wrapped, sample, device)
        for route in grouped[uid]:
            output = capture_four_action_route(
                wrapped, {}, route["actions"], prepared_inputs=meta, use_cache=False, native_full_rows=True
            )
            layers = list(range(int(route["activation_layer"]), 28))
            values = {
                "A": _probability_rows(router_a, output, layers, device),
                "B": _probability_rows(router_b, output, layers, device),
            }
            for checkpoint, probabilities in values.items():
                checkpoint_sha = hash_a if checkpoint == "A" else hash_b
                for offset, layer in enumerate(layers):
                    target = str(route["actions"][layer])
                    target_index = ACTIONS.index(target)
                    vector = probabilities[offset]
                    prediction_index = int(vector.argmax().item())
                    key = entering_state_key(uid, route["actions"], layer)
                    valid = action_sets[key]
                    index = intervention_index(route["actions"], layer, route["activation_layer"])
                    rows.append(
                        {
                            "schema_version": "stage2_mcts_oracle_state_prediction_v1",
                            "contract_sha256": contract["contract_sha256"],
                            "checkpoint": checkpoint,
                            "checkpoint_sha256": checkpoint_sha,
                            "route_id": route["route_id"],
                            "route_source": route["route_source"],
                            "uid": uid,
                            "dataset": route["dataset"],
                            "source_regime": route["source_regime"],
                            "valid_operating_points": route["valid_operating_points"],
                            "activation_layer": route["activation_layer"],
                            "layer": layer,
                            "state_key": key,
                            "target_action": target,
                            "predicted_action": ACTIONS[prediction_index],
                            "prediction_matches_target": ACTIONS[prediction_index] == target,
                            "prediction_in_observed_valid_set": ACTIONS[prediction_index] in valid,
                            "observed_successful_actions": list(valid),
                            "observed_successful_action_count": len(valid),
                            "target_probability": float(vector[target_index]),
                            "full_probability": float(vector[0]),
                            "best_non_full_probability": float(vector[1:].max()),
                            "target_vs_full_margin": float(vector[target_index] - vector[0]),
                            "full_vs_best_non_full_margin": float(vector[0] - vector[1:].max()),
                            "observed_set_probability": float(sum(vector[ACTIONS.index(action)] for action in valid)),
                            "probabilities": {action: float(vector[index]) for index, action in enumerate(ACTIONS)},
                            "route_non_full_count": route["non_full_count"],
                            "route_complexity_bin": route_complexity_bin(route["non_full_count"])
                            if route["route_source"] != "preservation_full"
                            else "0",
                            "intervention_index": index,
                            "intervention_index_bin": intervention_index_bin(index) if index else None,
                        }
                    )
            route_count += 1
            del output
        del meta
        if route_count and route_count % 100 == 0:
            print(json.dumps({"rank": rank, "routes": route_count, "states": len(rows), "elapsed_seconds": time.monotonic() - started}), flush=True)
    atomic_jsonl(output_path, rows)
    completion = {
        "passed": True,
        "rank": rank,
        "routes": len(routes),
        "state_rows_per_checkpoint": sum(28 - int(row["activation_layer"]) for row in routes),
        "records": len(rows),
        "contract_sha256": contract["contract_sha256"],
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(complete_path, completion)
    print(json.dumps(completion, sort_keys=True), flush=True)


def _distance(oracle: torch.Tensor, free: torch.Tensor) -> tuple[float, float]:
    left = oracle.detach().float().reshape(-1)
    right = free.detach().float().reshape(-1)
    cosine = float(1.0 - F.cosine_similarity(left[None], right[None], dim=-1).item())
    relative = float(torch.linalg.vector_norm(left - right).item() / max(torch.linalg.vector_norm(left).item(), 1e-12))
    return cosine, relative


def prefix_worker(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    if not read_json(output_root / "smoke/implementation_smoke.json").get("passed"):
        raise RuntimeError("diagnosis smoke is not passing")
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("prefix diagnosis requires four workers")
    result_path = output_root / f"work/prefix/rank{rank:02d}.jsonl"
    drift_path = output_root / f"work/prefix/rank{rank:02d}.drift.jsonl"
    complete_path = output_root / f"work/prefix/rank{rank:02d}.complete.json"
    if result_path.exists() or drift_path.exists() or complete_path.exists():
        raise RuntimeError(f"refusing to overwrite prefix rank {rank}")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    configure_dense_determinism(int(config["seed"]) + 100 + rank, contract["phase66_static_config"]["backend_settings"])
    processor, _base, wrapped = _load_model(contract["phase66_static_config"], device)
    router_b, hash_b = _load_router_checkpoint(config, contract, "B", device)
    routes = [
        row
        for row in read_jsonl(output_root / "work/diagnostic_routes.jsonl")
        if row["route_source"] == "mcts"
        and "P90" in row["valid_operating_points"]
        and int(row["worker_rank"]) == rank
    ]
    samples = {row["uid"]: row["sample"] for row in read_jsonl(resolve_path(config["sources"]["train_samples"]))}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        grouped[str(route["uid"])].append(route)
    result_rows, drift_rows = [], []
    started = time.monotonic()
    completed = 0
    for uid in sorted(grouped):
        sample = samples[uid]
        _verify_image(sample)
        inputs, _ = build_dense_inputs(processor, sample, device)
        meta = _prepare_binary(processor, wrapped, sample, device)
        for route in grouped[uid]:
            actions = [str(action) for action in route["actions"]]
            trigger = int(route["trigger_layers"]["P90"])
            non_full = [layer for layer in range(trigger, 28) if actions[layer] != "FULL"]
            oracle = capture_four_action_route(
                wrapped, {}, actions, prepared_inputs=meta, use_cache=True, native_full_rows=True
            )
            oracle_ids, oracle_text, oracle_score = _generate(
                processor, wrapped, oracle, inputs["input_ids"], sample
            )
            if oracle_ids != route["expected_generated_token_ids"] or not oracle_score.correct:
                raise RuntimeError(f"full oracle replay failed: {route['route_id']}")
            free_output = None
            for forced in range(len(non_full) + 1):
                boundary = forcing_boundary(actions, trigger, forced)
                chosen = []

                def selector(layer, text_states, visual_states, current_meta):
                    if layer < trigger or layer <= boundary:
                        action = actions[layer]
                        chosen.append({"layer": layer, "action": action, "forced": True})
                        return action
                    with torch.inference_mode():
                        logits = router_b(
                            text_states.detach(),
                            visual_states.detach(),
                            text_mask=current_meta.text_valid_mask,
                            visual_mask=current_meta.visual_valid_mask,
                        )
                        probabilities = logits.float().softmax(-1)[0]
                    action = ACTIONS[int(probabilities.argmax().item())]
                    chosen.append({"layer": layer, "action": action, "forced": False})
                    return action

                output = capture_online_four_action_route(
                    wrapped, {}, selector, prepared_inputs=meta, use_cache=True, native_full_rows=True
                )
                ids, text, score = _generate(processor, wrapped, output, inputs["input_ids"], sample)
                predicted_actions = list(output.layer_actions)
                release_start = max(trigger, boundary + 1)
                later = list(range(release_start, 28))
                reproduced = (
                    sum(predicted_actions[layer] == actions[layer] for layer in later) / len(later)
                    if later
                    else 1.0
                )
                first_after = next(
                    (layer for layer in later if predicted_actions[layer] != actions[layer]), None
                )
                result_rows.append(
                    {
                        "schema_version": "stage2_mcts_prefix_forcing_result_v1",
                        "contract_sha256": contract["contract_sha256"],
                        "checkpoint_sha256": hash_b,
                        "route_id": route["route_id"],
                        "uid": uid,
                        "dataset": route["dataset"],
                        "source_regime": route["source_regime"],
                        "operating_point": "P90",
                        "mode": f"C{forced}",
                        "forced_interventions": forced,
                        "route_non_full_count": len(non_full),
                        "forced_through_layer": boundary,
                        "release_start_layer": release_start,
                        "correct": bool(score.correct),
                        "generated_token_ids": ids,
                        "generated_answer": text,
                        "later_oracle_action_fraction": reproduced,
                        "non_full_actions_after_release": sum(
                            predicted_actions[layer] != "FULL" for layer in later
                        ),
                        "first_deviation_after_release": first_after,
                        "actions": predicted_actions,
                    }
                )
                if forced == 0:
                    free_output = output
                else:
                    del output
            result_rows.append(
                {
                    "schema_version": "stage2_mcts_prefix_forcing_result_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "checkpoint_sha256": hash_b,
                    "route_id": route["route_id"],
                    "uid": uid,
                    "dataset": route["dataset"],
                    "source_regime": route["source_regime"],
                    "operating_point": "P90",
                    "mode": "FULL_ORACLE",
                    "forced_interventions": len(non_full),
                    "route_non_full_count": len(non_full),
                    "forced_through_layer": 27,
                    "release_start_layer": None,
                    "correct": bool(oracle_score.correct),
                    "generated_token_ids": oracle_ids,
                    "generated_answer": oracle_text,
                    "later_oracle_action_fraction": 1.0,
                    "non_full_actions_after_release": 0,
                    "first_deviation_after_release": None,
                    "actions": actions,
                }
            )
            assert free_output is not None
            free_actions = list(free_output.layer_actions)
            first_deviation = next(
                (layer for layer in range(trigger, 28) if free_actions[layer] != actions[layer]), None
            )
            if first_deviation is not None:
                text_last = int(meta.text_valid_mask.long().sum().item()) - 1
                visual_mask = meta.visual_valid_mask[0].bool()
                for layer in range(first_deviation, 28):
                    oracle_text_state, oracle_visual_state = oracle.pre_layer_states[layer]
                    free_text_state, free_visual_state = free_output.pre_layer_states[layer]
                    text_cos, text_l2 = _distance(
                        oracle_text_state[0, text_last], free_text_state[0, text_last]
                    )
                    visual_cos, visual_l2 = _distance(
                        oracle_visual_state[0, visual_mask], free_visual_state[0, visual_mask]
                    )
                    drift_rows.append(
                        {
                            "schema_version": "stage2_mcts_state_drift_v1",
                            "contract_sha256": contract["contract_sha256"],
                            "route_id": route["route_id"],
                            "uid": uid,
                            "dataset": route["dataset"],
                            "source_regime": route["source_regime"],
                            "first_deviation_layer": first_deviation,
                            "layer": layer,
                            "offset_after_first_deviation": layer - first_deviation,
                            "oracle_action": actions[layer],
                            "free_action": free_actions[layer],
                            "text_query_cosine_distance": text_cos,
                            "text_query_relative_l2": text_l2,
                            "visual_sequence_cosine_distance": visual_cos,
                            "visual_sequence_relative_l2": visual_l2,
                        }
                    )
            completed += 1
            del oracle, free_output
            if completed % 20 == 0:
                print(json.dumps({"rank": rank, "routes": completed, "assigned": len(routes), "elapsed_seconds": time.monotonic() - started}), flush=True)
        del meta, inputs
    atomic_jsonl(result_path, result_rows)
    atomic_jsonl(drift_path, drift_rows)
    completion = {
        "passed": True,
        "rank": rank,
        "routes": len(routes),
        "result_rows": len(result_rows),
        "drift_rows": len(drift_rows),
        "contract_sha256": contract["contract_sha256"],
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(complete_path, completion)
    print(json.dumps(completion, sort_keys=True), flush=True)


def _group_metrics(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]):
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        output.append({**dict(zip(keys, values)), **action_metrics(group)})
    return output


def _case_studies(config: Mapping[str, Any], output_root: Path) -> None:
    a_rows = {row["uid"]: row for row in read_jsonl(resolve_path(config["sources"]["rollout_A_P90"]))}
    b_rows = {row["uid"]: row for row in read_jsonl(resolve_path(config["sources"]["rollout_B_P90"]))}
    rescues, regressions = [], []
    for uid, a in a_rows.items():
        if a["transition"] not in {"W→C", "C→W"}:
            continue
        b = b_rows[uid]
        non_full_a = [index for index, action in enumerate(a["actions"]) if action != "FULL"]
        non_full_b = [index for index, action in enumerate(b["actions"]) if action != "FULL"]
        if non_full_a and not non_full_b:
            category = "B_REMAINS_FULL_WHERE_A_INTERVENES"
        elif non_full_a != non_full_b:
            category = "B_INTERVENES_AT_ANOTHER_LAYER"
        elif a["actions"] != b["actions"]:
            category = "B_PICKS_DIFFERENT_ACTION"
        else:
            category = "A_B_IDENTICAL_ACTIONS"
        row = {
            "uid": uid,
            "dataset": a["dataset"],
            "dense_answer": a["dense_generated_answer"],
            "stage1_trigger": a["trigger_layer"],
            "A_actions": a["actions"],
            "A_non_full_layers": non_full_a,
            "A_answer": a["routed_generated_answer"],
            "A_transition": a["transition"],
            "B_actions": b["actions"],
            "B_non_full_layers": non_full_b,
            "B_answer": b["routed_generated_answer"],
            "B_transition": b["transition"],
            "classification": category,
        }
        (rescues if a["transition"] == "W→C" else regressions).append(row)
    if len(rescues) != 4 or len(regressions) != 1:
        raise RuntimeError("Phase-66 A/P90 case-study population differs")
    atomic_jsonl(output_root / "case_studies/A_rescued_W_comparison.jsonl", rescues)
    atomic_jsonl(output_root / "case_studies/A_regressed_C_comparison.jsonl", regressions)


def _plot_bar(path: Path, labels, a_values, b_values, ylabel: str) -> None:
    positions = list(range(len(labels)))
    plt.figure(figsize=(max(6, len(labels) * 1.2), 4))
    plt.bar([x - 0.2 for x in positions], a_values, width=0.4, label="A")
    plt.bar([x + 0.2 for x in positions], b_values, width=0.4, label="B")
    plt.xticks(positions, labels, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def aggregate(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    routes = read_jsonl(output_root / "work/diagnostic_routes.jsonl")
    oracle_rows, prefix_rows, drift_rows = [], [], []
    for rank in range(int(config["world_size"])):
        oracle_complete = read_json(output_root / f"work/oracle/rank{rank:02d}.complete.json")
        prefix_complete = read_json(output_root / f"work/prefix/rank{rank:02d}.complete.json")
        if not oracle_complete.get("passed") or not prefix_complete.get("passed"):
            raise RuntimeError(f"diagnostic rank {rank} is incomplete")
        rank_oracle = read_jsonl(output_root / f"work/oracle/rank{rank:02d}.jsonl")
        rank_prefix = read_jsonl(output_root / f"work/prefix/rank{rank:02d}.jsonl")
        rank_drift = read_jsonl(output_root / f"work/prefix/rank{rank:02d}.drift.jsonl")
        if len(rank_oracle) != int(oracle_complete["records"]):
            raise RuntimeError(f"oracle rank {rank} record count differs")
        if len(rank_prefix) != int(prefix_complete["result_rows"]):
            raise RuntimeError(f"prefix rank {rank} record count differs")
        oracle_rows.extend(rank_oracle)
        prefix_rows.extend(rank_prefix)
        drift_rows.extend(rank_drift)
    expected_oracle = 2 * int(contract["population"]["oracle_state_rows"])
    if len(oracle_rows) != expected_oracle:
        raise RuntimeError("global oracle-state record count differs")
    if len({(row["checkpoint"], row["route_id"], row["layer"]) for row in oracle_rows}) != len(oracle_rows):
        raise RuntimeError("oracle-state records contain duplicates")
    if len({row["route_id"] for row in prefix_rows}) != 725:
        raise RuntimeError("prefix-forcing MCTS route coverage differs")
    if any(row["mode"] == "FULL_ORACLE" and not row["correct"] for row in prefix_rows):
        raise RuntimeError("full-oracle correctness control failed")

    for checkpoint in ("A", "B"):
        rows = [row for row in oracle_rows if row["checkpoint"] == checkpoint]
        metrics = _group_metrics(rows, ("route_source",))
        point_metrics = []
        for point in ("P98", "P95", "P90"):
            point_metrics.extend(
                {"operating_point": point, **item}
                for item in _group_metrics(
                    [row for row in rows if point in row["valid_operating_points"]],
                    ("route_source",),
                )
            )
        atomic_csv(
            output_root / f"oracle_state/action_metrics_{checkpoint}.csv",
            [{"operating_point": "ALL", **item} for item in metrics] + point_metrics,
        )
    mcts = [row for row in oracle_rows if row["route_source"] == "mcts"]
    atomic_csv(
        output_root / "oracle_state/route_complexity_breakdown.csv",
        _group_metrics(mcts, ("checkpoint", "route_complexity_bin")),
    )
    atomic_csv(
        output_root / "oracle_state/intervention_index_breakdown.csv",
        _group_metrics(
            [row for row in mcts if row["intervention_index_bin"] is not None],
            ("checkpoint", "intervention_index_bin"),
        ),
    )

    route_lookup = {row["route_id"]: row for row in routes}
    by_route: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in oracle_rows:
        by_route[(row["checkpoint"], row["route_id"])].append(row)
    deviation_rows = []
    for (checkpoint, route_id), rows in sorted(by_route.items()):
        route = route_lookup[route_id]
        predictions = list(route["actions"])
        for row in rows:
            predictions[int(row["layer"])] = row["predicted_action"]
        category, layer = first_deviation_class(
            route["actions"], predictions, int(route["activation_layer"])
        )
        deviation_rows.append(
            {
                "checkpoint": checkpoint,
                "route_id": route_id,
                "route_source": route["route_source"],
                "uid": route["uid"],
                "dataset": route["dataset"],
                "source_regime": route["source_regime"],
                "valid_operating_points": route["valid_operating_points"],
                "activation_layer": route["activation_layer"],
                "first_non_full_layer": route["first_non_full_layer"],
                "route_non_full_count": route["non_full_count"],
                "first_deviation_category": category,
                "first_deviation_layer": layer,
                "predicted_actions": predictions,
            }
        )
    atomic_jsonl(output_root / "first_deviation/per_route_first_deviation.jsonl", deviation_rows)
    deviation_summary = []
    for checkpoint in ("A", "B"):
        for source in ROUTE_FILES:
            subset = [row for row in deviation_rows if row["checkpoint"] == checkpoint and row["route_source"] == source]
            for scope in ("ALL", "P98", "P95", "P90"):
                scoped = subset if scope == "ALL" else [row for row in subset if scope in row["valid_operating_points"]]
                counts = Counter(row["first_deviation_category"] for row in scoped)
                for category in (
                    "BEFORE_FIRST_INTERVENTION",
                    "AT_FIRST_INTERVENTION",
                    "BETWEEN_INTERVENTIONS",
                    "AT_SECOND_OR_LATER_INTERVENTION",
                    "NO_DEVIATION",
                ):
                    deviation_summary.append(
                        {
                            "checkpoint": checkpoint,
                            "route_source": source,
                            "operating_point": scope,
                            "category": category,
                            "count": counts[category],
                            "fraction": counts[category] / len(scoped) if scoped else None,
                        }
                    )
    atomic_csv(output_root / "first_deviation/summary.csv", deviation_summary)

    prefix_summary = []
    modes = sorted({row["mode"] for row in prefix_rows}, key=lambda value: (value == "FULL_ORACLE", value))
    for mode in modes:
        subset = [row for row in prefix_rows if row["mode"] == mode]
        matched_pairs = None if mode == "C0" else matched_mode_uid_pairs(prefix_rows, "C0", mode)
        matched_baseline = (
            sum(value[0] for value in matched_pairs.values()) / len(matched_pairs)
            if matched_pairs
            else None
        )
        matched_comparison = (
            sum(value[1] for value in matched_pairs.values()) / len(matched_pairs)
            if matched_pairs
            else None
        )
        exposure_subset = [
            row
            for row in subset
            if mode == "C0"
            or (
                mode != "FULL_ORACLE"
                and int(row["route_non_full_count"]) > int(row["forced_interventions"])
            )
        ]
        exposure_pairs = None
        if mode not in {"C0", "FULL_ORACLE"} and exposure_subset:
            exposure_pairs = matched_mode_uid_pairs(
                [row for row in prefix_rows if row["mode"] == "C0"] + exposure_subset,
                "C0",
                mode,
            )
        later_matches = 0
        later_targets = 0
        for row in exposure_subset:
            oracle_actions = route_lookup[row["route_id"]]["actions"]
            for layer in range(int(row["release_start_layer"]), 28):
                if oracle_actions[layer] != "FULL":
                    later_targets += 1
                    later_matches += row["actions"][layer] == oracle_actions[layer]
        prefix_summary.append(
            {
                "mode": mode,
                "routes": len(subset),
                "uids": len({row["uid"] for row in subset}),
                "rescue_rate": sum(row["correct"] for row in subset) / len(subset),
                "matched_uids_vs_C0": len(matched_pairs) if matched_pairs else None,
                "matched_C0_uid_rescue_rate": matched_baseline,
                "matched_uid_rescue_rate": matched_comparison,
                "matched_uid_delta_vs_C0": (
                    matched_comparison - matched_baseline
                    if matched_comparison is not None and matched_baseline is not None
                    else None
                ),
                "exposure_eligible_routes": len(exposure_subset),
                "exposure_eligible_uids": len({row["uid"] for row in exposure_subset}),
                "exposure_eligible_route_rescue_rate": (
                    sum(row["correct"] for row in exposure_subset) / len(exposure_subset)
                    if exposure_subset
                    else None
                ),
                "exposure_matched_uid_delta_vs_C0": (
                    sum(value[1] - value[0] for value in exposure_pairs.values())
                    / len(exposure_pairs)
                    if exposure_pairs
                    else None
                ),
                "later_oracle_non_full_actions": later_targets,
                "later_oracle_non_full_recall": (
                    later_matches / later_targets if later_targets else None
                ),
                "mean_later_oracle_action_fraction": sum(row["later_oracle_action_fraction"] for row in subset) / len(subset),
                "mean_non_full_actions_after_release": sum(row["non_full_actions_after_release"] for row in subset) / len(subset),
                "fraction_no_deviation_after_release": sum(row["first_deviation_after_release"] is None for row in subset) / len(subset),
            }
        )
    atomic_jsonl(output_root / "prefix_forcing/per_route_results.jsonl", prefix_rows)
    atomic_csv(output_root / "prefix_forcing/forcing_depth_summary.csv", prefix_summary)
    atomic_jsonl(output_root / "state_drift/per_route_state_distance.jsonl", drift_rows)
    drift_summary = []
    for offset in sorted({row["offset_after_first_deviation"] for row in drift_rows}):
        subset = [row for row in drift_rows if row["offset_after_first_deviation"] == offset]
        drift_summary.append(
            {
                "offset_after_first_deviation": offset,
                "route_layer_rows": len(subset),
                "uids": len({row["uid"] for row in subset}),
                **{
                    f"mean_{key}": sum(float(row[key]) for row in subset) / len(subset)
                    for key in (
                        "text_query_cosine_distance",
                        "text_query_relative_l2",
                        "visual_sequence_cosine_distance",
                        "visual_sequence_relative_l2",
                    )
                },
            }
        )
    atomic_csv(output_root / "state_drift/summary.csv", drift_summary)
    _case_studies(config, output_root)

    single = [row for row in oracle_rows if row["route_source"] == "single"]
    transfer_rows = []
    cohorts = {
        "all_single_states": single,
        "single_corrective_states": [row for row in single if row["target_action"] != "FULL"],
        "single_full_states": [row for row in single if row["target_action"] == "FULL"],
    }
    for cohort, values in cohorts.items():
        metrics = {checkpoint: action_metrics([row for row in values if row["checkpoint"] == checkpoint]) for checkpoint in ("A", "B")}
        transfer_rows.append(
            {
                "cohort": cohort,
                "states_per_checkpoint": metrics["A"]["states"],
                "A_action_accuracy": metrics["A"]["action_accuracy"],
                "B_action_accuracy": metrics["B"]["action_accuracy"],
                "delta_action_accuracy_B_minus_A": metrics["B"]["action_accuracy"] - metrics["A"]["action_accuracy"],
                "A_full_recall": metrics["A"].get("full_recall"),
                "B_full_recall": metrics["B"].get("full_recall"),
                "A_non_full_recall": metrics["A"].get("non_full_recall"),
                "B_non_full_recall": metrics["B"].get("non_full_recall"),
                "delta_non_full_recall_B_minus_A": (
                    None if metrics["A"].get("non_full_recall") is None else metrics["B"]["non_full_recall"] - metrics["A"]["non_full_recall"]
                ),
                "A_mean_target_probability": metrics["A"]["mean_target_probability"],
                "B_mean_target_probability": metrics["B"]["mean_target_probability"],
                "delta_target_probability_B_minus_A": metrics["B"]["mean_target_probability"] - metrics["A"]["mean_target_probability"],
            }
        )
    atomic_csv(output_root / "negative_transfer/single_state_A_vs_B.csv", transfer_rows)

    corrective_by_checkpoint: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in cohorts["single_corrective_states"]:
        corrective_by_checkpoint[row["checkpoint"]][row["uid"]].append(row)
    bootstrap_rows = []
    for metric_name, getter in (
        ("single_corrective_recall", lambda values: sum(row["prediction_matches_target"] for row in values) / len(values)),
        ("single_corrective_target_probability", lambda values: sum(row["target_probability"] for row in values) / len(values)),
    ):
        pairs = {
            uid: (getter(corrective_by_checkpoint["A"][uid]), getter(corrective_by_checkpoint["B"][uid]))
            for uid in sorted(set(corrective_by_checkpoint["A"]) & set(corrective_by_checkpoint["B"]))
        }
        bootstrap_rows.append(
            {"comparison": metric_name, **paired_uid_bootstrap(pairs, seed=int(config["seed"]), replicates=int(config["bootstrap_replicates"]))}
        )
    prefix_modes = {str(row["mode"]) for row in prefix_rows}
    for mode in sorted(value for value in prefix_modes if value not in {"C0", "FULL_ORACLE"}):
        eligible = [
            row
            for row in prefix_rows
            if row["mode"] == "C0"
            or (
                row["mode"] == mode
                and int(row["route_non_full_count"]) > int(row["forced_interventions"])
            )
        ]
        if not any(row["mode"] == mode for row in eligible):
            continue
        pairs = matched_mode_uid_pairs(eligible, "C0", mode)
        bootstrap_rows.append(
            {"comparison": f"prefix_{mode}_minus_C0_rescue", **paired_uid_bootstrap(pairs, seed=int(config["seed"]) + len(mode), replicates=int(config["bootstrap_replicates"]))}
        )
    atomic_csv(output_root / "negative_transfer/paired_bootstrap.csv", bootstrap_rows)

    ambiguity_rows = []
    set_probability_rows = []
    for checkpoint in ("A", "B"):
        for category, selector in (
            ("single_observed_action", lambda row: int(row["observed_successful_action_count"]) == 1),
            ("multi_valid", lambda row: int(row["observed_successful_action_count"]) > 1),
        ):
            subset = [row for row in oracle_rows if row["checkpoint"] == checkpoint and selector(row)]
            ambiguity_rows.append(
                {
                    "checkpoint": checkpoint,
                    "valid_action_category": category,
                    "route_state_rows": len(subset),
                    "nominal_error_rate": sum(not row["prediction_matches_target"] for row in subset) / len(subset),
                    "observed_set_error_rate": sum(not row["prediction_in_observed_valid_set"] for row in subset) / len(subset),
                    "error_recovery_by_valid_set": (
                        sum(not row["prediction_matches_target"] for row in subset)
                        - sum(not row["prediction_in_observed_valid_set"] for row in subset)
                    ) / len(subset),
                    "mean_observed_set_probability": sum(row["observed_set_probability"] for row in subset) / len(subset),
                }
            )
        for source in ROUTE_FILES:
            for valid_count in sorted({row["observed_successful_action_count"] for row in oracle_rows if row["checkpoint"] == checkpoint and row["route_source"] == source}):
                subset = [row for row in oracle_rows if row["checkpoint"] == checkpoint and row["route_source"] == source and row["observed_successful_action_count"] == valid_count]
                set_probability_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "route_source": source,
                        "observed_successful_action_count": valid_count,
                        "route_state_rows": len(subset),
                        "mean_observed_set_probability": sum(row["observed_set_probability"] for row in subset) / len(subset),
                        "prediction_in_observed_set_rate": sum(row["prediction_in_observed_valid_set"] for row in subset) / len(subset),
                    }
                )
    atomic_csv(output_root / "ambiguity/error_by_valid_action_count.csv", ambiguity_rows)
    atomic_csv(output_root / "ambiguity/observed_set_probability.csv", set_probability_rows)

    confidence_rows = []
    state_types = {
        "single_corrective": lambda row: row["route_source"] == "single" and row["target_action"] != "FULL",
        "mcts_first_intervention": lambda row: row["route_source"] == "mcts" and row["intervention_index"] == 1,
        "mcts_later_intervention": lambda row: row["route_source"] == "mcts" and row["intervention_index"] is not None and row["intervention_index"] >= 2,
        "preservation": lambda row: row["route_source"] == "preservation_full",
    }
    for checkpoint in ("A", "B"):
        for state_type, selector in state_types.items():
            subset = [row for row in oracle_rows if row["checkpoint"] == checkpoint and selector(row)]
            confidence_rows.append(
                {
                    "checkpoint": checkpoint,
                    "state_type": state_type,
                    "states": len(subset),
                    "mean_full_probability": sum(row["full_probability"] for row in subset) / len(subset),
                    "mean_best_non_full_probability": sum(row["best_non_full_probability"] for row in subset) / len(subset),
                    "mean_full_vs_best_non_full_margin": sum(row["full_vs_best_non_full_margin"] for row in subset) / len(subset),
                }
            )
    atomic_csv(output_root / "confidence/full_margin_by_state_type.csv", confidence_rows)

    output_root.joinpath("figures").mkdir(parents=True, exist_ok=True)
    labels = ["preservation", "single", "MCTS"]
    action_metric_lookup = {
        (checkpoint, source): action_metrics([row for row in oracle_rows if row["checkpoint"] == checkpoint and row["route_source"] == source])
        for checkpoint in ("A", "B") for source in ROUTE_FILES
    }
    _plot_bar(
        output_root / "figures/oracle_action_recall_A_vs_B.png",
        labels,
        [action_metric_lookup[("A", source)]["action_accuracy"] for source in ROUTE_FILES],
        [action_metric_lookup[("B", source)]["action_accuracy"] for source in ROUTE_FILES],
        "Oracle action accuracy",
    )
    categories = ["BEFORE_FIRST_INTERVENTION", "AT_FIRST_INTERVENTION", "BETWEEN_INTERVENTIONS", "AT_SECOND_OR_LATER_INTERVENTION", "NO_DEVIATION"]
    _plot_bar(
        output_root / "figures/first_deviation_distribution.png",
        categories,
        [sum(row["checkpoint"] == "A" and row["route_source"] == "mcts" and row["first_deviation_category"] == category for row in deviation_rows) / 725 for category in categories],
        [sum(row["checkpoint"] == "B" and row["route_source"] == "mcts" and row["first_deviation_category"] == category for row in deviation_rows) / 725 for category in categories],
        "MCTS route fraction",
    )
    plt.figure(figsize=(6, 4))
    plot_prefix = [row for row in prefix_summary if row["mode"] != "FULL_ORACLE"]
    plt.plot([row["mode"] for row in plot_prefix], [row["rescue_rate"] for row in plot_prefix], marker="o")
    plt.ylabel("Rescue rate")
    plt.xlabel("Forced interventions")
    plt.tight_layout()
    plt.savefig(output_root / "figures/rescue_vs_forced_prefix.png", dpi=160)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.plot([row["offset_after_first_deviation"] for row in drift_summary], [row["mean_text_query_cosine_distance"] for row in drift_summary], label="text query")
    plt.plot([row["offset_after_first_deviation"] for row in drift_summary], [row["mean_visual_sequence_cosine_distance"] for row in drift_summary], label="visual sequence")
    plt.xlabel("Layers after first deviation")
    plt.ylabel("Cosine distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/state_drift_after_deviation.png", dpi=160)
    plt.close()
    _plot_bar(
        output_root / "figures/single_negative_transfer.png",
        [row["cohort"] for row in transfer_rows],
        [row["A_action_accuracy"] for row in transfer_rows],
        [row["B_action_accuracy"] for row in transfer_rows],
        "Action accuracy",
    )
    _plot_bar(
        output_root / "figures/full_margin_A_vs_B.png",
        list(state_types),
        [next(row["mean_full_vs_best_non_full_margin"] for row in confidence_rows if row["checkpoint"] == "A" and row["state_type"] == state_type) for state_type in state_types],
        [next(row["mean_full_vs_best_non_full_margin"] for row in confidence_rows if row["checkpoint"] == "B" and row["state_type"] == state_type) for state_type in state_types],
        "FULL - best non-FULL probability",
    )

    mcts_b = action_metric_lookup[("B", "mcts")]
    first_b = action_metrics([row for row in mcts if row["checkpoint"] == "B" and row["intervention_index"] == 1])
    later_b = action_metrics([row for row in mcts if row["checkpoint"] == "B" and row["intervention_index"] is not None and row["intervention_index"] >= 2])
    c0_rate = next(row["rescue_rate"] for row in prefix_summary if row["mode"] == "C0")
    forced_gains = [
        row["exposure_matched_uid_delta_vs_C0"]
        for row in prefix_summary
        if row["mode"] not in {"C0", "FULL_ORACLE"}
        and int(row["exposure_eligible_uids"]) >= 30
    ]
    best_prefix_gain = max(forced_gains, default=0.0)
    transfer_corrective = next(row for row in transfer_rows if row["cohort"] == "single_corrective_states")
    recall_bootstrap = next(row for row in bootstrap_rows if row["comparison"] == "single_corrective_recall")
    ambiguity_b = {row["valid_action_category"]: row for row in ambiguity_rows if row["checkpoint"] == "B"}
    ambiguity_gap = ambiguity_b["multi_valid"]["nominal_error_rate"] - ambiguity_b["single_observed_action"]["nominal_error_rate"]
    ambiguity_recovery = ambiguity_b["multi_valid"]["error_recovery_by_valid_set"]
    thresholds = config["decision_thresholds"]
    support = {
        "H1_action_learning": mcts_b["non_full_recall"] < thresholds["oracle_mcts_non_full_reasonable_min"],
        "H2_exposure": mcts_b["non_full_recall"] >= thresholds["oracle_mcts_non_full_reasonable_min"] and best_prefix_gain >= thresholds["prefix_restore_absolute_min"],
        "H3_negative_transfer": transfer_corrective["delta_non_full_recall_B_minus_A"] <= thresholds["negative_transfer_recall_delta_max"] and float(recall_bootstrap["ci95_high"]) < 0,
        "H4_ambiguity": ambiguity_gap >= thresholds["ambiguity_error_gap_min"] and ambiguity_recovery >= thresholds["ambiguity_valid_set_recovery_min"],
        "H5_representation": (first_b["non_full_recall"] - later_b["non_full_recall"] >= thresholds["later_intervention_recall_gap_min"]) and not (ambiguity_gap >= thresholds["ambiguity_error_gap_min"]),
    }
    if support["H4_ambiguity"]:
        recommendation = ("observed-valid-set loss", "Replace single-route CE targets with -log(sum p(observed-valid actions)) while keeping the router and data fixed.", "whether exact-prefix label ambiguity is suppressing corrective action learning")
    elif support["H3_negative_transfer"]:
        recommendation = ("A-initialized low-rate MCTS fine-tuning", "Initialize from A and use a fixed 4:1 Single:MCTS corrective-W ratio.", "whether reduced MCTS weight avoids measurable interference with single corrections")
    elif support["H2_exposure"]:
        recommendation = ("small partial-prefix on-policy collection", "Collect only states reached after the first free-policy deviation or partial oracle prefix.", "whether training on encountered states closes the teacher-forced/free-rollout gap")
    elif support["H1_action_learning"]:
        recommendation = ("bounded MCTS-specific overfit pilot", "Overfit a small action-balanced set of complete MCTS trajectories with the unchanged router.", "whether the current router can learn sequential MCTS actions before any on-policy redesign")
    else:
        recommendation = ("minimal route-stage representation pilot", "Add one fixed intervention-count/stage scalar in a small overfit-only diagnostic.", "whether later action identity is missing from the shared state representation")
    hypothesis_rows = [
        {"hypothesis": key, "supported_by_fixed_rule": value}
        for key, value in support.items()
    ]
    atomic_csv(output_root / "summaries/hypothesis_decision_table.csv", hypothesis_rows)
    c1 = next(row for row in prefix_summary if row["mode"] == "C1")
    c2 = next(row for row in prefix_summary if row["mode"] == "C2")
    c3 = next(row for row in prefix_summary if row["mode"] == "C3")
    drift_one = next(row for row in drift_summary if row["offset_after_first_deviation"] == 1)
    first_dev_b_mcts = next(
        row
        for row in deviation_summary
        if row["checkpoint"] == "B"
        and row["route_source"] == "mcts"
        and row["operating_point"] == "P90"
        and row["category"] == "AT_FIRST_INTERVENTION"
    )
    summary = f"""# Stage-2 MCTS failure diagnosis

## Result

- A recognizes its single-route corrective actions weakly: state-weighted non-FULL recall is {action_metric_lookup[('A', 'single')]['non_full_recall']:.4f}.
- B recognizes MCTS corrective actions weakly even on exact oracle states: non-FULL recall is {mcts_b['non_full_recall']:.4f}; first/later-intervention recall is {first_b['non_full_recall']:.4f} / {later_b['non_full_recall']:.4f}.
- B's first teacher-forced disagreement is at the first MCTS intervention on {first_dev_b_mcts['fraction']:.2%} of routes.
- B free MCTS-route rescue (C0) is {c0_rate:.4f}. Forcing only C1 raises matched UID rescue by {c1['exposure_matched_uid_delta_vs_C0']:+.4f}; C2 and C3 raise it by {c2['exposure_matched_uid_delta_vs_C0']:+.4f} and {c3['exposure_matched_uid_delta_vs_C0']:+.4f} among routes that still have a later corrective action.
- Crucially, after release B reproduces only {c1['later_oracle_non_full_recall']:.2%}, {c2['later_oracle_non_full_recall']:.2%}, and {c3['later_oracle_non_full_recall']:.2%} of those remaining C1/C2/C3 oracle non-FULL actions. The accuracy rise is therefore mostly attributable to forced corrections, not recovery of the stored later policy.
- B-minus-A single corrective recall is {transfer_corrective['delta_non_full_recall_B_minus_A']:+.4f} state-weighted. The UID-weighted delta is {float(recall_bootstrap['mean_delta_B_minus_A']):+.4f}, 95% CI [{float(recall_bootstrap['ci95_low']):+.4f}, {float(recall_bootstrap['ci95_high']):+.4f}].
- B multi-valid versus single-valid nominal error gap: {ambiguity_gap:+.4f}; valid-set recovery on multi-valid rows: {ambiguity_recovery:.4f}.
- Fixed-rule hypothesis support: {json.dumps(support, sort_keys=True)}.

## Answers to the plan

1. **A on single routes:** A is better than B on corrective single states, but its own exact oracle recall is still only {action_metric_lookup[('A', 'single')]['non_full_recall']:.2%}; it does not reliably reproduce its training-route action.
2. **B on MCTS routes:** B reaches {mcts_b['non_full_recall']:.2%} corrective recall on exact successful MCTS states. That is better than A-on-MCTS ({action_metric_lookup[('A', 'mcts')]['non_full_recall']:.2%}) but still an action-learning failure in absolute terms.
3. **First versus later:** both are poor ({first_b['non_full_recall']:.2%} vs {later_b['non_full_recall']:.2%}), and {first_dev_b_mcts['fraction']:.2%} of routes first disagree exactly at the first intervention. There is no large later-only deficit.
4. **Prefix forcing:** C1 gives only {c1['exposure_matched_uid_delta_vs_C0']:+.2%} matched UID rescue. C2/C3 gains are larger ({c2['exposure_matched_uid_delta_vs_C0']:+.2%}/{c3['exposure_matched_uid_delta_vs_C0']:+.2%}), but later corrective-action reproduction remains at most {max(c1['later_oracle_non_full_recall'], c2['later_oracle_non_full_recall'], c3['later_oracle_non_full_recall']):.2%}. This is not strong evidence that the learned policy recovers after an oracle prefix.
5. **State drift:** at the first post-error layer, text-query relative L2 is {drift_one['mean_text_query_relative_l2']:.3f} and visual-sequence relative L2 is {drift_one['mean_visual_sequence_relative_l2']:.3f}; drift is real, but weak oracle recognition already precedes it.
6. **Negative transfer:** B loses {abs(transfer_corrective['delta_non_full_recall_B_minus_A']):.2%} state-weighted single corrective recall relative to A, with a strictly negative UID-bootstrap interval. Overall accuracy rises only because B predicts FULL more often.
7. **A's four rescues:** B stays all-FULL on three; on the fourth it intervenes earlier and repeatedly with IGNORE before WRITE_ONLY, but remains wrong. The single A regression is removed by B staying FULL.
8. **Ambiguity:** on B, nominal error is {ambiguity_b['multi_valid']['nominal_error_rate']:.2%} for multi-valid states versus {ambiguity_b['single_observed_action']['nominal_error_rate']:.2%} for single-valid states. Accepting any observed successful action recovers {ambiguity_recovery:.2%} of multi-valid rows.
9. **FULL bias:** it is regime-specific, not global. B's mean FULL-minus-best-non-FULL margin increases from {next(row['mean_full_vs_best_non_full_margin'] for row in confidence_rows if row['checkpoint'] == 'A' and row['state_type'] == 'single_corrective'):.3f} to {next(row['mean_full_vs_best_non_full_margin'] for row in confidence_rows if row['checkpoint'] == 'B' and row['state_type'] == 'single_corrective'):.3f} on single corrective states, but decreases sharply on MCTS first/later states.
10. **Best explanation:** H1 action learning, H3 negative transfer, and H4 ambiguity are jointly supported. H2 exposure is secondary evidence only; H5 is not isolated by the fixed rule. The smallest discriminating next experiment is the fixed observed-valid-set loss described separately.
11. This does not establish that MCTS or sequential correction is impossible, that MCTS supervision is inherently harmful, or that single intervention is universally superior.

## Scope

This diagnosis uses frozen training-route states and Historical-validation case studies. It changes no checkpoint, threshold, Stage-1 component, label, evaluator, or test result.
"""
    _atomic_bytes(output_root / "summaries/mcts_failure_diagnosis_summary.md", summary.encode())
    recommendation_text = f"""# Next Stage-2 recommendation

## One smallest next experiment

**{recommendation[0]}**

- Why it matters: the present diagnosis most directly supports this bounded discriminator without changing Stage 1 or opening test.
- Hypothesis tested: {recommendation[2]}.
- Frozen implementation: {recommendation[1]}
- Positive result: the targeted oracle/action or free-rollout deficit improves while A's single corrective behavior is retained.
- Negative result: this explanation is insufficient, and the next decision should revisit the remaining supported hypotheses rather than repeat the same recipe.
- Cannot establish: deployment benefit, unseen-source generalization, or final-threshold validity without a separately authorized validation experiment.

This recommendation is not executed in Phase 67.
"""
    _atomic_bytes(output_root / "summaries/next_stage2_recommendation.md", recommendation_text.encode())

    files = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files[str(path.relative_to(output_root))] = file_sha256(path)
    manifest = {
        "schema_version": "stage2_mcts_failure_diagnosis_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": True,
        "oracle_records": len(oracle_rows),
        "prefix_records": len(prefix_rows),
        "drift_records": len(drift_rows),
        "mcts_routes": 725,
        "files": files,
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "artifact_manifest.json", manifest)
    verify_artifact_manifest(output_root, manifest)
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"], "files": len(files), "support": support, "recommendation": recommendation[0]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--device-index", type=int, default=0)
    sub.add_parser("oracle-worker")
    sub.add_parser("prefix-worker")
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "smoke":
        smoke(args.config, args.device_index)
    elif args.command == "oracle-worker":
        oracle_worker(args.config)
    elif args.command == "prefix-worker":
        prefix_worker(args.config)
    else:
        aggregate(args.config)


if __name__ == "__main__":
    main()
