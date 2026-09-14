#!/usr/bin/env python3
"""Run frozen Stage-1 gate to four-action treatment-correctability analysis."""

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
from dense_failure_stage1.gate_winner import apply_control  # noqa: E402
from dense_failure_stage1.lmms_scoring import (  # noqa: E402
    lmms_eval_source_metadata,
    score_lmms_sample,
)
from dense_failure_stage1.runtime import (  # noqa: E402
    build_dense_inputs,
    configure_dense_determinism,
)
from dense_failure_stage1.shared_global_gate import SharedFailurePredictor  # noqa: E402
from dense_failure_stage1.treatment_correctability import (  # noqa: E402
    GATES,
    bounded_route_panel,
    summarize_regime,
    trigger_manifest_rows,
    validate_execution_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage1_gate_treatment_correctability_v1.json"
BOUND_CODE_PATHS = (
    "configs/stage1_gate_treatment_correctability_v1.json",
    "dense_failure_stage1/treatment_correctability.py",
    "dense_failure_stage1/gate_winner.py",
    "dense_failure_stage1/lmms_scoring.py",
    "dense_failure_stage1/runtime.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "experiments/analyze_stage1_gate_treatment_correctability.py",
)
DATASETS = ("gqa", "chartqa", "textvqa")


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


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def resolve_path(value: str) -> Path:
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
    if config.get("schema_version") != "stage1_gate_treatment_correctability_config_v1":
        raise ValueError("unsupported treatment-correctability config")
    if tuple(config["gates"]) != GATES or int(config["world_size"]) != 4:
        raise ValueError("gate order/world size differs from the frozen plan")
    search = config["search"]
    if search["actions"] != ["READ_ONLY", "WRITE_ONLY", "IGNORE"]:
        raise ValueError("treatment action order differs")
    if int(search["seeded_distinct_layer_pair_panel"]) != 12:
        raise ValueError("pair-panel size differs from the reviewed contract")
    return config


def _score_rows(rows: Sequence[Mapping[str, Any]], *, variant: str | None = None):
    if variant is not None:
        rows = [row for row in rows if row.get("variant") == variant]
    rows = sorted(rows, key=lambda row: str(row["uid"]))
    if len(rows) != 800 or len({str(row["uid"]) for row in rows}) != 800:
        raise ValueError("gate score split must contain 800 unique UIDs")
    values = np.asarray(
        [[float(row[f"p_{layer}"]) for layer in range(28)] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("gate scores contain nonfinite values")
    return list(rows), values


def _load_split_sources(config: Mapping[str, Any], split: str):
    sources = config["sources"]
    split_name = "val" if split == "val" else "test"
    split_rows = sorted(
        [row for row in read_jsonl(resolve_path(sources["split_manifest"])) if row["split"] == split_name],
        key=lambda row: str(row["uid"]),
    )
    candidates = {row["uid"]: row for row in read_jsonl(resolve_path(sources["candidate_manifest"]))}
    dense = {row["uid"]: row for row in read_jsonl(resolve_path(sources["dense_outputs"]))}
    if len(split_rows) != 800:
        raise ValueError(f"{split} split does not contain 800 records")
    records = []
    for split_row in split_rows:
        uid = split_row["uid"]
        candidate, output = candidates[uid], dense[uid]
        if bool(output["current_dense_wrong"]) != bool(split_row["current_dense_wrong"]):
            raise ValueError(f"dense label mismatch for {uid}")
        records.append({**candidate, **split_row, "dense_output": output})
    independent_name = f"independent_{'validation' if split == 'val' else 'test'}_scores"
    shared_name = f"shared_{'validation' if split == 'val' else 'test'}_scores"
    independent_rows, independent_scores = _score_rows(read_jsonl(resolve_path(sources[independent_name])))
    shared_rows, shared_scores = _score_rows(
        read_jsonl(resolve_path(sources[shared_name])), variant="state_layer_random4" if split == "test" else None
    )
    identities = [row["uid"] for row in records]
    if identities != [row["uid"] for row in independent_rows] or identities != [row["uid"] for row in shared_rows]:
        raise RuntimeError("split/gate score UIDs do not align")
    return records, independent_scores, shared_scores


def _assign_workers(rows: list[dict[str, Any]], world_size: int) -> list[dict[str, Any]]:
    loads = [0] * world_size
    ordered = sorted(rows, key=lambda row: (-int(row["estimated_route_evaluations"]), row["uid"]))
    assigned = []
    for row in ordered:
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        loads[rank] += int(row["estimated_route_evaluations"])
        assigned.append({**row, "worker_rank": rank})
    return sorted(assigned, key=lambda row: row["uid"])


def _execution_manifest(
    records: Sequence[Mapping[str, Any]], trigger_sets: Mapping[str, Sequence[Mapping[str, Any]]], *, split: str, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_gate = {gate: {row["uid"]: row for row in rows} for gate, rows in trigger_sets.items()}
    pair_count = int(config["search"]["seeded_distinct_layer_pair_panel"])
    seed = int(config["seed"])
    output = []
    for record in records:
        regimes = []
        if bool(record["current_dense_wrong"]):
            regimes.append({"gate": "all_wrong_replay", "trigger_layer": None, "treatment_start_layer": 0})
        for gate in GATES:
            trigger = by_gate[gate][record["uid"]]
            if trigger["triggered"]:
                regimes.append(
                    {
                        "gate": gate,
                        "trigger_layer": trigger["trigger_layer"],
                        "treatment_start_layer": trigger["treatment_start_layer"],
                        "risk_at_trigger": trigger["risk_at_trigger"],
                        "trigger_depth_bin": trigger["trigger_depth_bin"],
                    }
                )
        if not regimes:
            continue
        starts = sorted({int(regime["treatment_start_layer"]) for regime in regimes})
        cost = sum(
            len(
                bounded_route_panel(
                    uid=record["uid"],
                    start_layer=start,
                    seed=seed,
                    pair_panel_size=pair_count,
                )
            )
            for start in starts
        )
        output.append(
            {
                "uid": record["uid"],
                "split": split,
                "dataset": record["dataset"],
                "current_dense_wrong": bool(record["current_dense_wrong"]),
                "image_group_id": record["image_group_id"],
                "regimes": regimes,
                "estimated_route_evaluations": cost,
                "sample": {
                    key: record[key]
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
                "dense_output": record["dense_output"],
            }
        )
    return _assign_workers(output, int(config["world_size"]))


def _smoke_manifest(rows: Sequence[Mapping[str, Any]], *, records: int, seed: int) -> list[dict[str, Any]]:
    selected = []
    for dataset in DATASETS:
        for wrong in (False, True):
            cell = [row for row in rows if row["dataset"] == dataset and bool(row["current_dense_wrong"]) is wrong]
            cell.sort(key=lambda row: sha256(f"{seed}:{row['uid']}".encode()).hexdigest())
            selected.extend(cell[:2])
    if len(selected) != records or len({row["uid"] for row in selected}) != records:
        raise RuntimeError("could not construct the frozen 12-record smoke")
    return _assign_workers([dict(row) for row in selected], 4)


def prepare(config_path: Path) -> None:
    config = load_static(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    selection_manifest = read_json(resolve_path(config["sources"]["gate_selection_manifest"]))
    selection = read_json(resolve_path(config["sources"]["gate_selection"]))
    if not selection_manifest.get("passed") or selection["test_evaluated"] is not False:
        raise RuntimeError("Phase-52 frozen selection is invalid")
    controls = selection["candidate_controls"]
    trigger_outputs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    execution_outputs = {}
    for split in ("val", "test"):
        records, independent_scores, shared_scores = _load_split_sources(config, split)
        first = {
            "shared_random4": apply_control("shared_random4", shared_scores, controls["shared_random4"]),
            "independent_sequential": apply_control("independent_sequential", independent_scores, controls["independent_sequential"]),
            "fixed_l27": apply_control("shared_fixed_l27", shared_scores, controls["shared_fixed_l27"]),
        }
        score_map = {
            "shared_random4": shared_scores,
            "independent_sequential": independent_scores,
            "fixed_l27": shared_scores,
        }
        triggers = {}
        for gate in GATES:
            triggers[gate] = trigger_manifest_rows(records, score_map[gate], first[gate], gate=gate, split=split)
            trigger_outputs[(split, gate)] = triggers[gate]
        execution_outputs[split] = _execution_manifest(records, triggers, split=split, config=config)
    smoke = _smoke_manifest(execution_outputs["val"], records=int(config["smoke"]["records"]), seed=int(config["seed"]))

    source_hashes = {name: file_sha256(resolve_path(path)) for name, path in config["sources"].items()}
    bound_hashes = {path: file_sha256(resolve_path(path)) for path in BOUND_CODE_PATHS}
    snapshot = resolve_path(config["model"]["snapshot_path"])
    model_files = sorted(path for path in snapshot.iterdir() if path.is_file())
    model_hashes = {path.name: file_sha256(path) for path in model_files}
    lmms = lmms_eval_source_metadata()
    contract: dict[str, Any] = {
        "schema_version": "stage1_gate_treatment_correctability_contract_v1",
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
            "lmms_eval": lmms,
            "cuda_runtime": torch.version.cuda,
            "gpu_count_required": 4,
        },
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "model_snapshot_sha256": model_hashes,
        "phase52_contract_sha256": selection["contract_sha256"],
        "gate_controls": {
            "shared_random4": controls["shared_random4"],
            "independent_sequential": controls["independent_sequential"],
            "fixed_l27": controls["shared_fixed_l27"],
        },
        "manifest_sha256": {
            f"gate_trigger_manifests/{gate}.jsonl": sha256(
                "".join(json.dumps(row, sort_keys=True) + "\n" for split in ("val", "test") for row in trigger_outputs[(split, gate)]).encode()
            ).hexdigest()
            for gate in GATES
        },
        "execution_manifest_sha256": {
            split: sha256("".join(json.dumps(row, sort_keys=True) + "\n" for row in execution_outputs[split]).encode()).hexdigest()
            for split in ("val", "test")
        },
        "smoke_manifest_sha256": sha256("".join(json.dumps(row, sort_keys=True) + "\n" for row in smoke).encode()).hexdigest(),
        "review_reconciliation": "fixed 12-pair panel replaces fill-to-96 sampling to avoid depth-dependent pair effort",
    }
    contract["contract_sha256"] = canonical_hash(contract)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "frozen_protocol.json", contract)
    for gate in GATES:
        atomic_jsonl(output_root / f"gate_trigger_manifests/{gate}.jsonl", [row for split in ("val", "test") for row in trigger_outputs[(split, gate)]])
    for split in ("val", "test"):
        atomic_jsonl(output_root / f"treatment_search/{split}_execution_manifest.jsonl", execution_outputs[split])
    atomic_jsonl(output_root / "smoke/smoke_manifest.jsonl", smoke)
    protocol = f"""# Stage-1 gate treatment-correctability protocol

- Contract SHA-256: `{contract['contract_sha256']}`
- Model revision: `{config['model']['revision']}`
- Gates: shared Random-4 sequential, independent sequential, and fixed-L27 detect-and-replay at Phase-52 selected controls only.
- Dynamic treatment reuses the exact native all-FULL prefix and may change only layers from the first trigger through 27.
- Every unpadded full-row call uses native maskless causal dispatch, including the exact native all-FULL prefix and routed suffixes; compacted READ-off calls retain explicit masks.
- Fixed-L27 treatment performs a second pass from layer 0.
- Search is outcome-independent: all permitted one-layer READ_ONLY/WRITE_ONLY/IGNORE routes plus exactly 12 seeded distinct-layer pairs, or every pair if fewer than 12 exist.
- Search stops at the first current LMMS-Eval-correct route. Rates are bounded-search lower estimates, not exhaustive four-action oracle rates.
- Enrichment uses the same layer-0 replay panel on every dense-wrong sample, then compares triggered subsets with all wrong. Dynamic deployment correctability is reported separately.
- Native dense token parity and cached-suffix versus complete-route token parity must pass the 12-record smoke.
- Validation is completed before test treatment execution. Test cannot change the validation-selected dynamic substrate.
- No historical cached route correctness is authoritative.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    atomic_json(
        output_root / "preparation_audit.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "trigger_counts": {
                split: {
                    gate: Counter(row["cohort"] for row in trigger_outputs[(split, gate)])
                    for gate in GATES
                }
                for split in ("val", "test")
            },
            "execution_samples": {split: len(rows) for split, rows in execution_outputs.items()},
            "smoke_samples": len(smoke),
        },
    )
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"], "execution_samples": {key: len(value) for key, value in execution_outputs.items()}}))


def load_contract(
    config_path: Path, *, verify_model_snapshot: bool = False
) -> tuple[dict[str, Any], Path]:
    config = load_static(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256") or contract["static_config"] != config:
        raise RuntimeError("frozen contract/config mismatch")
    for name, expected in contract["source_sha256"].items():
        if file_sha256(resolve_path(config["sources"][name])) != expected:
            raise RuntimeError(f"source hash mismatch: {name}")
    for relative, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(relative)) != expected:
            raise RuntimeError(f"bound-code hash mismatch: {relative}")
    if verify_model_snapshot:
        snapshot = resolve_path(config["model"]["snapshot_path"])
        actual_files = sorted(path.name for path in snapshot.iterdir() if path.is_file())
        if actual_files != sorted(contract["model_snapshot_sha256"]):
            raise RuntimeError("model snapshot file inventory mismatch")
        for name, expected in contract["model_snapshot_sha256"].items():
            if file_sha256(snapshot / name) != expected:
                raise RuntimeError(f"model snapshot hash mismatch: {name}")
    return contract, output_root


def _load_model(config: Mapping[str, Any], device: torch.device):
    model = config["model"]
    snapshot = str(resolve_path(model["snapshot_path"]))
    processor = AutoProcessor.from_pretrained(snapshot, revision=model["revision"], local_files_only=True, use_fast=False)
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
    text = processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return ids, text


def _generate_output(processor, wrapped, output, inputs, sample):
    assert output.cache is not None
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


def _worker_paths(output_root: Path, mode: str, rank: int) -> dict[str, Path]:
    root = output_root / f"treatment_search/{mode}_workers"
    return {
        "results": root / f"rank{rank:02d}.results.jsonl",
        "routes": root / f"rank{rank:02d}.routes.jsonl",
        "failures": root / f"rank{rank:02d}.failures.jsonl",
        "complete": root / f"rank{rank:02d}.complete.json",
    }


def _run_regime(
    *,
    panel: Sequence[Mapping[str, Any]],
    evaluate,
    smoke_limit: int | None,
) -> dict[str, Any]:
    candidates = list(panel[:smoke_limit] if smoke_limit is not None else panel)
    evaluated = []
    success = None
    physical = 0
    started = time.monotonic()
    for candidate in candidates:
        route, is_new = evaluate(candidate)
        physical += int(is_new)
        evaluated.append(route["route_key"])
        if route["correct"]:
            success = route
            break
    return {
        "passed": True,
        "correctable": success is not None,
        "single_intervention_success": success is not None and success["search_stage"] == "immediate_single",
        "successful_route_key": None if success is None else success["route_key"],
        "successful_search_stage": None if success is None else success["search_stage"],
        "evaluated_route_keys": evaluated,
        "route_evaluations": len(evaluated),
        "physical_new_route_evaluations": physical,
        "route_panel_size": len(candidates),
        "elapsed_seconds": time.monotonic() - started,
    }


def worker(config_path: Path, *, mode: str, rank: int, world_size: int, resume: bool) -> None:
    contract, output_root = load_contract(config_path, verify_model_snapshot=True)
    config = contract["static_config"]
    if world_size != 4 or rank not in range(world_size) or torch.cuda.device_count() != 4:
        raise RuntimeError("treatment workers require exactly four visible GPUs/processes")
    if mode in {"val", "test"} and not (output_root / "smoke/smoke_completion.json").exists():
        raise RuntimeError("full treatment execution requires a passing smoke")
    if mode == "test" and not (output_root / "treatment_search/validation_completion.json").exists():
        raise RuntimeError("test treatment execution requires finalized validation")
    manifest_path = output_root / ("smoke/smoke_manifest.jsonl" if mode == "smoke" else f"treatment_search/{mode}_execution_manifest.jsonl")
    rows = [row for row in read_jsonl(manifest_path) if int(row["worker_rank"]) == rank]
    paths = _worker_paths(output_root, mode, rank)
    if paths["complete"].exists():
        raise RuntimeError(f"worker already complete: {paths['complete']}")
    if any(path.exists() for key, path in paths.items() if key != "complete") and not resume:
        raise FileExistsError(f"worker outputs exist; use --resume: rank {rank}")
    completed = {row["uid"] for row in read_jsonl(paths["results"])} if paths["results"].exists() else set()
    route_cache: dict[tuple[str, str], dict[str, Any]] = {}
    if paths["routes"].exists():
        for row in read_jsonl(paths["routes"]):
            if row.get("contract_sha256") != contract["contract_sha256"]:
                raise RuntimeError("resume route-cache contract mismatch")
            route_cache[(row["uid"], row["route_key"])] = row
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    configure_dense_determinism(
        int(config["seed"]) + rank,
        read_json(resolve_path(config["sources"]["dense_config"]))["backend_settings"],
    )
    processor, base, wrapped = _load_model(config, device)
    pair_count = int(config["search"]["seeded_distinct_layer_pair_panel"])
    failures = 0
    for index, manifest_row in enumerate(rows):
        if manifest_row["uid"] in completed:
            continue
        sample = manifest_row["sample"]
        started = time.monotonic()
        try:
            inputs, input_metadata = build_dense_inputs(processor, sample, device)
            prepared = build_binary_inputs(wrapped, inputs)
            native_full_rows = bool(config["execution"]["native_full_row_dispatch"])
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
                "generated_ids_match_current_dense": baseline_state["generated_ids"] == expected["generated_token_ids"],
                "answer_match_current_dense": baseline_state["generated_answer"] == expected["generated_answer"],
                "correctness_match_current_dense": baseline_state["correct"] == bool(expected["current_dense_correct"]),
                "prompt_sha256_match_current_dense": input_metadata["literal_prompt_sha256"] == expected["literal_prompt_sha256"],
            }
            if not all(baseline_checks.values()):
                raise RuntimeError(f"unified FULL/current-dense parity failed: {baseline_checks}")
            suffix_parity = []

            def evaluate(candidate: Mapping[str, Any], start: int):
                key = (sample["uid"], candidate["route_key"])
                if key in route_cache:
                    return route_cache[key], False
                route_started = time.monotonic()
                actions = tuple(candidate["actions"])
                output = capture_four_action_suffix_from_full_baseline(wrapped, baseline, start, actions[start:])
                state = _generate_output(processor, wrapped, output, inputs, sample)
                row = {
                    "schema_version": "stage1_treatment_route_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "uid": sample["uid"],
                    "split": manifest_row["split"],
                    "dataset": sample["dataset"],
                    "current_dense_wrong": manifest_row["current_dense_wrong"],
                    "physical_start_layer": start,
                    **{key: candidate[key] for key in ("candidate_index", "search_stage", "actions", "route_key", "changed_layers", "changed_actions")},
                    **state,
                    "elapsed_seconds": time.monotonic() - route_started,
                    "prompt_decoder_layers": 28 - start,
                    "worker_rank": rank,
                }
                route_cache[key] = row
                append_jsonl(paths["routes"], row)
                return row, True

            regime_results = []
            smoke_limit = int(config["smoke"]["routes_per_regime"]) if mode == "smoke" else None
            for regime in manifest_row["regimes"]:
                start = int(regime["treatment_start_layer"])
                panel = bounded_route_panel(uid=sample["uid"], start_layer=start, seed=int(config["seed"]), pair_panel_size=pair_count)
                if mode == "smoke" and panel:
                    candidate = panel[0]
                    suffix_row, _ = evaluate(candidate, start)
                    complete = capture_four_action_route(
                        wrapped,
                        inputs,
                        candidate["actions"],
                        prepared_inputs=prepared,
                        use_cache=True,
                        native_full_rows=native_full_rows,
                    )
                    complete_state = _generate_output(processor, wrapped, complete, inputs, sample)
                    parity = {
                        "start_layer": start,
                        "route_key": candidate["route_key"],
                        "generated_ids_equal": complete_state["generated_ids"] == suffix_row["generated_ids"],
                        "answer_equal": complete_state["generated_answer"] == suffix_row["generated_answer"],
                        "correctness_equal": complete_state["correct"] == suffix_row["correct"],
                    }
                    if not all(value for key, value in parity.items() if key.endswith("equal")):
                        raise RuntimeError(f"suffix/complete-route parity failed: {parity}")
                    suffix_parity.append(parity)
                result = _run_regime(panel=panel, evaluate=lambda candidate, start=start: evaluate(candidate, start), smoke_limit=smoke_limit)
                equivalent = 1.0 + result["route_evaluations"] * (28 - start) / 28.0
                regime_results.append(
                    {
                        "schema_version": "stage1_treatment_regime_result_v1",
                        "contract_sha256": contract["contract_sha256"],
                        "uid": sample["uid"],
                        "split": manifest_row["split"],
                        "dataset": sample["dataset"],
                        "current_dense_wrong": manifest_row["current_dense_wrong"],
                        **regime,
                        **result,
                        "equivalent_prompt_passes": equivalent,
                    }
                )
            final = {
                "schema_version": "stage1_treatment_sample_result_v1",
                "contract_sha256": contract["contract_sha256"],
                "uid": sample["uid"],
                "split": manifest_row["split"],
                "dataset": sample["dataset"],
                "current_dense_wrong": manifest_row["current_dense_wrong"],
                "image_group_id": sample["image_group_id"],
                "baseline_state": baseline_state,
                "baseline_checks": baseline_checks,
                "suffix_complete_route_parity": suffix_parity,
                "regime_results": regime_results,
                "sample_elapsed_seconds": time.monotonic() - started,
                "worker_rank": rank,
                "passed": True,
            }
            append_jsonl(paths["results"], final)
            print(json.dumps({"mode": mode, "rank": rank, "sample": index + 1, "total": len(rows), "uid": sample["uid"], "regimes": len(regime_results)}), flush=True)
        except Exception as exc:
            failures += 1
            append_jsonl(paths["failures"], {"contract_sha256": contract["contract_sha256"], "uid": sample["uid"], "error": str(exc), "traceback": traceback.format_exc(), "worker_rank": rank})
            break
        finally:
            torch.cuda.empty_cache()
    if failures:
        raise RuntimeError(f"worker {rank} failed")
    final_rows = read_jsonl(paths["results"]) if paths["results"].exists() else []
    expected_uids = {row["uid"] for row in rows}
    if {row["uid"] for row in final_rows} != expected_uids or len(final_rows) != len(expected_uids):
        raise RuntimeError("worker completion coverage is incomplete")
    atomic_json(paths["complete"], {"passed": True, "contract_sha256": contract["contract_sha256"], "mode": mode, "rank": rank, "records": len(final_rows), "route_records": len(read_jsonl(paths["routes"])) if paths["routes"].exists() else 0})


def _collect_mode(output_root: Path, mode: str, contract_sha256: str):
    sample_rows, route_rows = [], []
    for rank in range(4):
        paths = _worker_paths(output_root, mode, rank)
        complete = read_json(paths["complete"])
        if not complete.get("passed") or complete.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"invalid {mode} completion for rank {rank}")
        if paths["failures"].exists() and read_jsonl(paths["failures"]):
            raise RuntimeError(f"{mode} rank {rank} contains failures")
        sample_rows.extend(read_jsonl(paths["results"]) if paths["results"].exists() else [])
        route_rows.extend(read_jsonl(paths["routes"]) if paths["routes"].exists() else [])
    if len({row["uid"] for row in sample_rows}) != len(sample_rows):
        raise RuntimeError(f"{mode} sample outputs contain duplicate UIDs")
    route_keys = [(row["uid"], row["route_key"]) for row in route_rows]
    if len(route_keys) != len(set(route_keys)):
        raise RuntimeError(f"{mode} route outputs contain duplicates")
    return sample_rows, route_rows


def _flatten_regimes(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(regime) for sample in sample_rows for regime in sample["regime_results"]]


def _expected_regime_keys(manifest: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    return [
        (str(row["split"]), str(regime["gate"]), str(row["uid"]))
        for row in manifest
        for regime in row["regimes"]
    ]


def finalize_smoke(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    samples, routes = _collect_mode(output_root, "smoke", contract["contract_sha256"])
    manifest = read_jsonl(output_root / "smoke/smoke_manifest.jsonl")
    if len(samples) != 12 or {row["uid"] for row in samples} != {row["uid"] for row in manifest}:
        raise RuntimeError("smoke sample coverage differs")
    regimes = _flatten_regimes(samples)
    validate_execution_coverage(_expected_regime_keys(manifest), regimes)
    baseline_pass = all(all(row["baseline_checks"].values()) for row in samples)
    parity_rows = [item for row in samples for item in row["suffix_complete_route_parity"]]
    suffix_pass = bool(parity_rows) and all(
        item["generated_ids_equal"] and item["answer_equal"] and item["correctness_equal"]
        for item in parity_rows
    )
    if not baseline_pass or not suffix_pass:
        raise RuntimeError("smoke parity gate failed")
    report = f"""# Treatment-correctability smoke report

- Contract: `{contract['contract_sha256']}`
- Samples: {len(samples)}/12
- Regime records: {len(regimes)}
- Unique treatment routes executed: {len(routes)}
- Current native dense versus unified-FULL exact token parity: **PASS** ({len(samples)}/{len(samples)})
- Cached-prefix suffix versus complete-route exact token parity: **PASS** ({len(parity_rows)}/{len(parity_rows)})
- LMMS correctness parity on both checks: **PASS**
- Missing/duplicate/failed sample or regime records: **0**
"""
    _atomic_bytes(output_root / "smoke/smoke_report.md", report.encode())
    atomic_json(
        output_root / "smoke/smoke_completion.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "samples": len(samples),
            "regimes": len(regimes),
            "routes": len(routes),
            "baseline_token_parity": True,
            "suffix_complete_route_token_parity": True,
            "completed_at": utc_now(),
        },
    )
    print(json.dumps({"passed": True, "samples": len(samples), "routes": len(routes)}))


def finalize_validation(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    samples, routes = _collect_mode(output_root, "val", contract["contract_sha256"])
    manifest = read_jsonl(output_root / "treatment_search/val_execution_manifest.jsonl")
    if {row["uid"] for row in samples} != {row["uid"] for row in manifest}:
        raise RuntimeError("validation sample coverage differs")
    regimes = _flatten_regimes(samples)
    validate_execution_coverage(_expected_regime_keys(manifest), regimes)
    atomic_json(
        output_root / "treatment_search/validation_completion.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "samples": len(samples),
            "regimes": len(regimes),
            "unique_routes": len(routes),
            "test_treatment_results_read": False,
            "completed_at": utc_now(),
        },
    )
    print(json.dumps({"passed": True, "samples": len(samples), "regimes": len(regimes), "routes": len(routes)}))


def _stage1_metrics(trigger_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wrong = [row for row in trigger_rows if row["current_dense_wrong"]]
    correct = [row for row in trigger_rows if not row["current_dense_wrong"]]
    wrong_triggered = sum(bool(row["triggered"]) for row in wrong)
    correct_triggered = sum(bool(row["triggered"]) for row in correct)
    layers = [int(row["trigger_layer"]) for row in trigger_rows if row["triggered"]]
    return {
        "records": len(trigger_rows),
        "dense_wrong": len(wrong),
        "dense_correct": len(correct),
        "triggered_wrong": wrong_triggered,
        "triggered_correct": correct_triggered,
        "pre_treatment_preservation": 1.0 - correct_triggered / len(correct),
        "wrong_trigger_recall": wrong_triggered / len(wrong),
        "trigger_precision": wrong_triggered / max(wrong_triggered + correct_triggered, 1),
        "median_trigger_layer": float(np.median(layers)) if layers else None,
    }


def _group_summary(rows: Sequence[Mapping[str, Any]], total_wrong: int) -> dict[str, Any]:
    return summarize_regime(rows, total_dense_wrong=total_wrong)


def _select_dynamic(validation_rows: Sequence[Mapping[str, Any]]) -> str:
    candidates = [row for row in validation_rows if row["gate"] in {"shared_random4", "independent_sequential"}]
    if len(candidates) != 2:
        raise RuntimeError("dynamic selection requires exactly two validation candidates")
    return max(
        candidates,
        key=lambda row: (
            float(row["population_oracle_rescue"]),
            float(row["triggered_wrong_correctability"]),
            -1.0 if row["triggered_correct_preservability"] is None else float(row["triggered_correct_preservability"]),
            -float(row["mean_equivalent_prompt_passes"]),
            int(row["gate"] == "shared_random4"),
        ),
    )["gate"]


def _write_stage2_manifest(
    output_root: Path,
    contract: Mapping[str, Any],
    selected_dynamic: str,
    regimes: Sequence[Mapping[str, Any]],
    route_rows: Sequence[Mapping[str, Any]],
) -> int:
    dynamic = [row for row in regimes if row["gate"] == selected_dynamic]
    if not any(row["current_dense_wrong"] and row["correctable"] for row in dynamic):
        atomic_jsonl(output_root / "stage2_future_manifest.jsonl", [])
        return 0
    feature_index = {
        row["uid"]: row
        for row in read_jsonl(resolve_path(contract["static_config"]["sources"]["feature_index"]))
    }
    routes = {(row["uid"], row["route_key"]): row for row in route_rows}
    ordered = sorted(dynamic, key=lambda value: (value["split"], value["uid"]))
    z_rows: dict[tuple[str, str], int] = {}
    if selected_dynamic == "shared_random4":
        source_config = contract["static_config"]["sources"]
        normalization = torch.load(
            resolve_path(source_config["shared_normalization"]),
            map_location="cpu",
            weights_only=True,
        )
        checkpoint = torch.load(
            resolve_path(source_config["shared_checkpoint"]),
            map_location="cpu",
            weights_only=True,
        )
        predictor = SharedFailurePredictor(
            variant="state_layer_random4",
            input_size=10752,
            projection_size=256,
            layer_embedding_size=32,
            hidden_size=256,
        )
        predictor.load_state_dict(checkpoint["model_state_dict"])
        predictor.eval()
        shard_cache: dict[str, dict[str, Any]] = {}
        vectors = []
        for row in ordered:
            feature = feature_index[row["uid"]]
            shard_name = feature["shard"]
            if shard_name not in shard_cache:
                shard_cache[shard_name] = torch.load(
                    resolve_path(shard_name), map_location="cpu", weights_only=True
                )
            payload = shard_cache[shard_name]
            layer = int(row["trigger_layer"])
            position = int(feature["row_index"])
            state = torch.cat(
                [payload[name][position, layer].float() for name in ("text_final", "text_mean", "visual_mean")]
            )
            state = (state - normalization["mean"].float()) / normalization["std"].float()
            layer_tensor = torch.tensor([layer], dtype=torch.long)
            with torch.inference_mode():
                pieces = [predictor.projection(state[None])]
                pieces.append(predictor.layer_embedding(layer_tensor))
                hidden = predictor.head[1](predictor.head[0](torch.cat(pieces, dim=-1)))[0]
            z_rows[(row["split"], row["uid"])] = len(vectors)
            vectors.append(hidden.to(torch.bfloat16))
        atomic_torch(
            output_root / "stage2_future_features.pt",
            {
                "schema_version": "stage2_future_failure_representations_v1",
                "contract_sha256": contract["contract_sha256"],
                "gate": selected_dynamic,
                "definition": "shared_random4_penultimate_256d_post_GELU_representation",
                "uids": [row["uid"] for row in ordered],
                "splits": [row["split"] for row in ordered],
                "trigger_layers": [int(row["trigger_layer"]) for row in ordered],
                "z_fail": torch.stack(vectors),
            },
        )

    output = []
    for row in ordered:
        layer = int(row["trigger_layer"])
        feature = feature_index[row["uid"]]
        evaluated = [routes[(row["uid"], key)] for key in row["evaluated_route_keys"]]
        output.append(
            {
                "schema_version": "stage2_future_triggered_state_v1",
                "contract_sha256": contract["contract_sha256"],
                "uid": row["uid"],
                "split": row["split"],
                "dataset": row["dataset"],
                "gate": selected_dynamic,
                "trigger_layer": layer,
                "p_wrong": row.get("risk_at_trigger"),
                "h_l": {
                    "feature_shard": feature["shard"],
                    "feature_row_index": int(feature["row_index"]),
                    "layer_index": layer,
                    "components": ["text_final", "text_mean", "visual_mean"],
                    "representation": "exact_reference_to_frozen_current_dense_compact_hidden_state",
                },
                "z_fail": (
                    {
                        "definition": "independent_linear_probe_penultimate_is_the_normalized_h_l_input",
                        "available": True,
                        "materialized": False,
                        "reconstruction": "apply the bound Phase-48 layer-specific normalization to h_l",
                    }
                    if selected_dynamic == "independent_sequential"
                    else {
                        "definition": "shared_random4_penultimate_256d_post_GELU_representation",
                        "available": True,
                        "materialized": True,
                        "feature_file": "analysis/dense_failure_stage1/treatment_correctability/stage2_future_features.pt",
                        "row_index": z_rows[(row["split"], row["uid"])],
                    }
                ),
                "successful_treatment_routes": [
                    {"route_key": item["route_key"], "actions": item["actions"], "generated_answer": item["generated_answer"], "lmms_score": item["lmms_score"]}
                    for item in evaluated
                    if item["correct"]
                ],
                "failed_treatment_routes": [
                    {"route_key": item["route_key"], "actions": item["actions"], "generated_answer": item["generated_answer"], "lmms_score": item["lmms_score"]}
                    for item in evaluated
                    if not item["correct"]
                ],
            }
        )
    atomic_jsonl(output_root / "stage2_future_manifest.jsonl", output)
    return len(output)


def _plot_metrics(output_root: Path, gate_rows, enrichment_rows, depth_rows, dataset_rows, compute_rows):
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    validation = [row for row in gate_rows if row["split"] == "val"]
    names = [row["gate"] for row in validation]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, [row["triggered_wrong_correctability"] for row in validation], 0.36, label="Conditional correctability")
    ax.bar(x + 0.18, [row["population_oracle_rescue"] for row in validation], 0.36, label="Population rescue")
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "gate_correctability_comparison.png", dpi=180)
    plt.close(fig)

    val_enrichment = [row for row in enrichment_rows if row["split"] == "val"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([row["gate"] for row in val_enrichment], [row["correctability_enrichment"] for row in val_enrichment])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Full-replay correctability enrichment")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figure_root / "correctability_enrichment.png", dpi=180)
    plt.close(fig)

    val_depth = [row for row in depth_rows if row["split"] == "val"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    bins = ("early", "middle", "late")
    for offset, gate in zip((-width / 2, width / 2), ("shared_random4", "independent_sequential")):
        mapping = {row["trigger_depth_bin"]: row for row in val_depth if row["gate"] == gate}
        ax.bar(np.arange(3) + offset, [mapping.get(name, {}).get("triggered_wrong_correctability", 0) for name in bins], width, label=gate)
    ax.set_xticks(np.arange(3), bins)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Triggered-wrong correctability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "correctability_by_trigger_depth.png", dpi=180)
    plt.close(fig)

    val_dataset = [row for row in dataset_rows if row["split"] == "val"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        rows = [row for row in val_dataset if row["dataset"] == dataset]
        axis.bar([row["gate"] for row in rows], [row["population_oracle_rescue"] for row in rows])
        axis.set_title(dataset)
        axis.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Population oracle rescue")
    fig.tight_layout()
    fig.savefig(figure_root / "dataset_treatment_potential.png", dpi=180)
    plt.close(fig)

    val_compute = [row for row in compute_rows if row["split"] == "val"]
    gate_lookup = {row["gate"]: row for row in validation}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in val_compute:
        ax.scatter(row["mean_equivalent_prompt_passes"], gate_lookup[row["gate"]]["population_oracle_rescue"], s=65)
        ax.annotate(row["gate"], (row["mean_equivalent_prompt_passes"], gate_lookup[row["gate"]]["population_oracle_rescue"]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Mean equivalent prompt passes", ylabel="Population oracle rescue")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "oracle_rescue_vs_compute.png", dpi=180)
    plt.close(fig)


def finalize(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    validation_completion = read_json(output_root / "treatment_search/validation_completion.json")
    if not validation_completion.get("passed"):
        raise RuntimeError("validation is not finalized")
    all_samples, all_routes = [], []
    for mode in ("val", "test"):
        samples, routes = _collect_mode(output_root, mode, contract["contract_sha256"])
        manifest = read_jsonl(output_root / f"treatment_search/{mode}_execution_manifest.jsonl")
        regimes = _flatten_regimes(samples)
        validate_execution_coverage(_expected_regime_keys(manifest), regimes)
        all_samples.extend(samples)
        all_routes.extend(routes)
    all_regimes = _flatten_regimes(all_samples)
    atomic_jsonl(output_root / "treatment_search/execution_rows.jsonl", all_regimes)
    atomic_jsonl(output_root / "treatment_search/route_cache.jsonl", all_routes)

    trigger_all = {
        gate: read_jsonl(output_root / f"gate_trigger_manifests/{gate}.jsonl")
        for gate in GATES
    }
    gate_metrics_rows, depth_rows, dataset_rows, enrichment_rows, compute_rows = [], [], [], [], []
    for split in ("val", "test"):
        split_regimes = [row for row in all_regimes if row["split"] == split]
        replay = {row["uid"]: row for row in split_regimes if row["gate"] == "all_wrong_replay"}
        if len(replay) != 400:
            raise RuntimeError(f"{split} all-wrong replay baseline is incomplete")
        replay_rate = sum(row["correctable"] for row in replay.values()) / len(replay)
        for gate in GATES:
            triggers = [row for row in trigger_all[gate] if row["split"] == split]
            treatment = [row for row in split_regimes if row["gate"] == gate]
            stage1 = _stage1_metrics(triggers)
            treatment_summary = _group_summary(treatment, total_wrong=400)
            mean_equivalent = float(np.mean([row["equivalent_prompt_passes"] for row in treatment]))
            gate_metrics_rows.append({"split": split, "gate": gate, **stage1, **treatment_summary, "mean_equivalent_prompt_passes": mean_equivalent})
            compute_rows.append(
                {
                    "split": split,
                    "gate": gate,
                    "triggered_samples": len(treatment),
                    "mean_route_evaluations": float(np.mean([row["route_evaluations"] for row in treatment])),
                    "median_route_evaluations": float(np.median([row["route_evaluations"] for row in treatment])),
                    "mean_equivalent_prompt_passes": mean_equivalent,
                    "total_logical_route_evaluations": sum(int(row["route_evaluations"]) for row in treatment),
                    "total_physical_new_route_evaluations": sum(int(row["physical_new_route_evaluations"]) for row in treatment),
                    "mean_search_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in treatment])),
                }
            )
            triggered_wrong_uids = {row["uid"] for row in triggers if row["triggered"] and row["current_dense_wrong"]}
            triggered_replay_rate = sum(replay[uid]["correctable"] for uid in triggered_wrong_uids) / len(triggered_wrong_uids)
            enrichment_rows.append(
                {
                    "split": split,
                    "gate": gate,
                    "triggered_wrong": len(triggered_wrong_uids),
                    "full_wrong_population": len(replay),
                    "full_replay_correctability_triggered_wrong": triggered_replay_rate,
                    "full_replay_correctability_all_wrong": replay_rate,
                    "correctability_enrichment": triggered_replay_rate / replay_rate if replay_rate > 0 else None,
                    "dynamic_deployment_correctability": treatment_summary["triggered_wrong_correctability"],
                }
            )
            for dataset in DATASETS:
                dataset_triggers = [row for row in triggers if row["dataset"] == dataset]
                dataset_treatment = [row for row in treatment if row["dataset"] == dataset]
                total_wrong = sum(row["current_dense_wrong"] for row in dataset_triggers)
                dataset_rows.append({"split": split, "gate": gate, "dataset": dataset, **_stage1_metrics(dataset_triggers), **_group_summary(dataset_treatment, total_wrong=total_wrong)})
            if gate != "fixed_l27":
                for bin_name in ("early", "middle", "late"):
                    subset = [row for row in treatment if row.get("trigger_depth_bin") == bin_name]
                    if subset:
                        depth_rows.append({"split": split, "gate": gate, "trigger_depth_bin": bin_name, **_group_summary(subset, total_wrong=400)})

    atomic_csv(output_root / "metrics/gate_correctability.csv", gate_metrics_rows)
    atomic_csv(output_root / "metrics/trigger_depth_correctability.csv", depth_rows)
    atomic_csv(output_root / "metrics/dataset_breakdown.csv", dataset_rows)
    atomic_csv(output_root / "metrics/enrichment.csv", enrichment_rows)
    atomic_csv(output_root / "metrics/compute_comparison.csv", compute_rows)
    selected_dynamic = _select_dynamic([row for row in gate_metrics_rows if row["split"] == "val"])
    stage2_records = _write_stage2_manifest(output_root, contract, selected_dynamic, all_regimes, all_routes)
    _plot_metrics(output_root, gate_metrics_rows, enrichment_rows, depth_rows, dataset_rows, compute_rows)

    gate_lookup = {(row["split"], row["gate"]): row for row in gate_metrics_rows}
    enrichment_lookup = {(row["split"], row["gate"]): row for row in enrichment_rows}
    validation_table = "\n".join(
        f"| {gate} | {gate_lookup[('val', gate)]['pre_treatment_preservation']:.4f} | {gate_lookup[('val', gate)]['wrong_trigger_recall']:.4f} | {gate_lookup[('val', gate)]['trigger_precision']:.4f} | {gate_lookup[('val', gate)]['triggered_wrong_correctability']:.4f} | {gate_lookup[('val', gate)]['population_oracle_rescue']:.4f} | {gate_lookup[('val', gate)]['median_trigger_layer']:.1f} |"
        for gate in GATES
    )
    test_table = "\n".join(
        f"| {gate} | {gate_lookup[('test', gate)]['pre_treatment_preservation']:.4f} | {gate_lookup[('test', gate)]['wrong_trigger_recall']:.4f} | {gate_lookup[('test', gate)]['trigger_precision']:.4f} | {gate_lookup[('test', gate)]['triggered_wrong_correctability']:.4f} | {gate_lookup[('test', gate)]['population_oracle_rescue']:.4f} | {gate_lookup[('test', gate)]['median_trigger_layer']:.1f} |"
        for gate in GATES
    )
    selected_val = gate_lookup[("val", selected_dynamic)]
    fixed_val = gate_lookup[("val", "fixed_l27")]
    selected_enrichment = enrichment_lookup[("val", selected_dynamic)]
    depth_val = [row for row in depth_rows if row["split"] == "val" and row["gate"] == selected_dynamic]
    depth_text = ", ".join(f"{row['trigger_depth_bin']}={row['triggered_wrong_correctability']:.4f} (n={row['triggered_wrong']})" for row in depth_val)
    proceed = selected_val["population_oracle_rescue"] > 0 and selected_enrichment["correctability_enrichment"] > 1.0
    summary = f"""# Stage-1 gate treatment-correctability decision

Contract: `{contract['contract_sha256']}`

All rates below are current-runtime LMMS-Eval results under the fixed all-single plus 12-pair bounded search. They are lower estimates, not exhaustive four-action oracle rates.

## Validation

| Gate | Preservation | Wrong trigger recall | Trigger precision | Triggered-wrong correctability | Population oracle rescue | Median trigger |
|---|---:|---:|---:|---:|---:|---:|
{validation_table}

## Test confirmation

| Gate | Preservation | Wrong trigger recall | Trigger precision | Triggered-wrong correctability | Population oracle rescue | Median trigger |
|---|---:|---:|---:|---:|---:|---:|
{test_table}

## Answers

1. Triggered dense-wrong correctability is reported for every gate in the tables; exact numerators are in `metrics/gate_correctability.csv`.
2. The validation-selected dynamic gate `{selected_dynamic}` has full-replay correctability enrichment {selected_enrichment['correctability_enrichment']:.4f} versus all validation wrong samples.
3. Its validation correctability by trigger depth is: {depth_text or 'no dynamic depth rows'}.
4. The validation-frozen better dynamic treatment substrate is `{selected_dynamic}` under population rescue, conditional correctability, preservability, compute, then the predeclared tie rule.
5. `{selected_dynamic}` validation population rescue is {selected_val['population_oracle_rescue']:.4f} versus fixed-L27 replay {fixed_val['population_oracle_rescue']:.4f}; compute is reported separately in `metrics/compute_comparison.csv`.
6. Triggered-correct preservability is {selected_val['triggered_correct_preservability']:.4f} for `{selected_dynamic}` and {fixed_val['triggered_correct_preservability']:.4f} for fixed L27 on validation.
7. Evidence to proceed to a learned Stage-2 action head: **{'YES, provisionally' if proceed else 'NO'}**. This rule requires nonzero population rescue and enrichment above 1 for the selected dynamic gate; it does not claim statistical significance or exhaustive oracle coverage.

The Stage-2 future manifest contains {stage2_records} triggered states. No Stage-2 model was trained.
"""
    _atomic_bytes(output_root / "decision_summary.md", summary.encode())
    atomic_json(
        output_root / "treatment_search/search_summary.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "sample_records": len(all_samples),
            "regime_records": len(all_regimes),
            "unique_route_records": len(all_routes),
            "selected_dynamic_gate": selected_dynamic,
            "stage2_future_records": stage2_records,
            "bounded_lower_bound_not_exhaustive_oracle": True,
            "completed_at": utc_now(),
        },
    )
    required = [
        "protocol.md", "frozen_protocol.json", "preparation_audit.json",
        *[f"gate_trigger_manifests/{gate}.jsonl" for gate in GATES],
        "treatment_search/execution_rows.jsonl", "treatment_search/route_cache.jsonl", "treatment_search/search_summary.json",
        "metrics/gate_correctability.csv", "metrics/trigger_depth_correctability.csv", "metrics/dataset_breakdown.csv", "metrics/enrichment.csv", "metrics/compute_comparison.csv",
        "stage2_future_manifest.jsonl",
        "figures/gate_correctability_comparison.png", "figures/correctability_enrichment.png", "figures/correctability_by_trigger_depth.png", "figures/dataset_treatment_potential.png", "figures/oracle_rescue_vs_compute.png",
        "decision_summary.md",
    ]
    if (output_root / "stage2_future_features.pt").exists():
        required.append("stage2_future_features.pt")
    atomic_json(output_root / "artifact_manifest.json", {"schema_version": "stage1_gate_treatment_correctability_artifact_manifest_v1", "passed": True, "contract_sha256": contract["contract_sha256"], "selected_dynamic_gate": selected_dynamic, "required_file_count": len(required), "required_files": {relative: file_sha256(output_root / relative) for relative in required}})
    print(json.dumps({"passed": True, "selected_dynamic_gate": selected_dynamic, "stage2_future_records": stage2_records}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "worker", "finalize-smoke", "finalize-validation", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "val", "test"))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "prepare":
        prepare(args.config.resolve())
    elif args.command == "worker":
        if args.mode is None or args.rank is None:
            raise SystemExit("worker requires --mode and --rank")
        worker(args.config.resolve(), mode=args.mode, rank=args.rank, world_size=args.world_size, resume=args.resume)
    elif args.command == "finalize-smoke":
        finalize_smoke(args.config.resolve())
    elif args.command == "finalize-validation":
        finalize_validation(args.config.resolve())
    else:
        finalize(args.config.resolve())
