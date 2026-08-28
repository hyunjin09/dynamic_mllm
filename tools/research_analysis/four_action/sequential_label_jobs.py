from __future__ import annotations

from collections.abc import Iterable, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any


DATASETS = ("gqa", "textvqa", "chartqa", "wemath20_standard", "wemath2pro")
SEQUENTIAL_CONVERSION_CODE_PATHS = (
    "binary_policy/actions.py",
    "binary_policy/executor/__init__.py",
    "binary_policy/executor/cache.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/masks.py",
    "binary_policy/executor/model.py",
    "label_regeneration/runtime.py",
    "reference/dvr_qwen/eval_metrics.py",
    "tools/research_analysis/four_action/label_runtime.py",
    "tools/research_analysis/four_action/sequential_label_conversion.py",
    "tools/research_analysis/four_action/sequential_label_jobs.py",
    "tools/research_analysis/four_action/targets.py",
    "experiments/run_sequential_four_action_label_conversion.py",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_filename(uid: str) -> str:
    readable = str(uid).replace(":", "__").replace("/", "_")
    return f"{readable}_{sha256(str(uid).encode()).hexdigest()[:10]}.json"


def mode_topology(mode: str) -> dict[str, int]:
    if mode == "smoke":
        return {"gpu_count": 8, "worker_count": 8, "workers_per_gpu": 1}
    if mode == "full":
        return {"gpu_count": 8, "worker_count": 16, "workers_per_gpu": 2}
    raise ValueError(f"unsupported conversion mode: {mode}")


def build_sequential_execution_contract(
    *,
    project_root: Path,
    config_path: Path,
    manifest_path: Path,
    config: dict[str, Any],
    mode: str,
    git_commit: str,
    torch_version: str,
    transformers_version: str,
) -> dict[str, Any]:
    """Bind records to the exact unbounded sequential-branching implementation."""
    root = Path(project_root).resolve()
    code_hashes = {}
    for relative in SEQUENTIAL_CONVERSION_CODE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        code_hashes[relative] = file_sha256(path)
    model = config["model"]
    snapshot = Path(model["snapshot_path"])
    if not snapshot.is_absolute():
        snapshot = root / snapshot
    contract = {
        "schema_version": "exact_sequential_four_action_execution_contract_v1",
        "git_commit": str(git_commit),
        "source_manifest": str(Path(manifest_path).resolve()),
        "source_manifest_sha256": file_sha256(Path(manifest_path)),
        "config": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(Path(config_path)),
        "model_snapshot_path": str(snapshot.resolve()),
        "model_revision": str(model["revision"]),
        "attention_implementation": str(model["attention_implementation"]),
        "seed": int(config["seed"]),
        "layer_count": int(config["layer_count"]),
        "processing_order": str(config["processing_order"]),
        "search_policy": "exact_sequential_verified_branching",
        "retention_policy": "all_evaluator_correct_branches",
        "worker_topology": mode_topology(mode),
        "torch_version": str(torch_version),
        "transformers_version": str(transformers_version),
        "code_sha256": code_hashes,
    }
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = sha256(encoded.encode()).hexdigest()
    return contract


def select_sequential_smoke(
    rows: Iterable[dict[str, Any]],
    smoke_uids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve and validate the deliberately frozen eight-sample smoke set."""
    requested = tuple(str(uid) for uid in smoke_uids)
    if len(requested) != 8 or len(set(requested)) != 8:
        raise ValueError("smoke selection must declare exactly eight unique UIDs")
    by_uid = {str(row["uid"]): row for row in rows}
    missing = [uid for uid in requested if uid not in by_uid]
    if missing:
        raise ValueError(f"smoke UIDs missing from source manifest: {missing}")
    selected = [by_uid[uid] for uid in requested]
    w2c = [row for row in selected if row.get("source_current_all_on_status") == "wrong"]
    c2c = [row for row in selected if row.get("source_current_all_on_status") == "correct"]
    all_off_w2c = [
        row
        for row in w2c
        if any(bool(route.get("source_all_off")) for route in row["source_positive_routes"])
    ]
    multi = [row for row in selected if int(row["source_positive_route_count"]) > 1]
    datasets = sorted({str(row["dataset"]) for row in selected})
    if set(datasets) != set(DATASETS) or not w2c or not c2c or not all_off_w2c or not multi:
        raise ValueError("smoke selection lacks required dataset/semantic/route coverage")
    return selected, {
        "datasets": datasets,
        "w2c_samples": len(w2c),
        "c2c_samples": len(c2c),
        "all_off_w2c_samples": len(all_off_w2c),
        "multi_source_route_samples": len(multi),
        "minimum_source_off_count": min(
            int(route["source_off_count"])
            for row in selected
            for route in row["source_positive_routes"]
        ),
        "maximum_source_off_count": max(
            int(route["source_off_count"])
            for row in selected
            for route in row["source_positive_routes"]
        ),
    }


class SequentialAtomicSampleQueue:
    """Claim pending samples exactly once in a launch-scoped shared queue."""

    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        claim_root: Path,
        completed_uids: set[str],
        claimant: str,
    ) -> None:
        self._rows = sorted(
            rows,
            key=lambda row: (-int(row["estimated_conversion_cost"]), str(row["uid"])),
        )
        self._claim_root = Path(claim_root)
        self._claim_root.mkdir(parents=True, exist_ok=True)
        self._completed_uids = set(completed_uids)
        self._claimant = str(claimant)
        self._cursor = 0

    def claim_next(self) -> dict[str, Any] | None:
        while self._cursor < len(self._rows):
            row = self._rows[self._cursor]
            self._cursor += 1
            uid = str(row["uid"])
            if uid in self._completed_uids:
                continue
            path = self._claim_root / f"{sha256(uid.encode()).hexdigest()}.json"
            try:
                with path.open("x", encoding="utf-8") as handle:
                    json.dump({"uid": uid, "claimant": self._claimant}, handle, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError:
                continue
            return row
        return None
