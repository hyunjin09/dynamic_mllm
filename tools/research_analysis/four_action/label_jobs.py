from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any


DATASETS = ("gqa", "textvqa", "chartqa", "wemath20_standard", "wemath2pro")

# The completed portion of the five-dataset pilot measured a 17.8x median
# seconds-per-static-cost difference between current W2C and C2C samples.  This
# rounded multiplier controls queue order only; it never changes conversion
# semantics or excludes a sample.
PILOT_W2C_COST_MULTIPLIER = 18
CONVERSION_CODE_PATHS = (
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
    "tools/research_analysis/four_action/label_conversion.py",
    "tools/research_analysis/four_action/label_jobs.py",
    "tools/research_analysis/four_action/label_runtime.py",
    "tools/research_analysis/four_action/targets.py",
    "experiments/run_four_action_label_conversion.py",
)


def safe_filename(uid: str) -> str:
    readable = uid.replace(":", "__").replace("/", "_")
    return f"{readable}_{sha256(uid.encode()).hexdigest()[:10]}.json"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_conversion_execution_contract(
    *,
    project_root: Path,
    config_path: Path,
    manifest_path: Path,
    config: dict[str, Any],
    git_commit: str,
    torch_version: str,
    transformers_version: str,
) -> dict[str, Any]:
    """Bind full-label records to their exact model, inputs, and loaded code."""
    root = Path(project_root).resolve()
    code_hashes = {}
    for relative in CONVERSION_CODE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        code_hashes[relative] = _file_sha256(path)
    model = config["model"]
    snapshot = Path(model["snapshot_path"])
    if not snapshot.is_absolute():
        snapshot = root / snapshot
    contract = {
        "schema_version": "four_action_label_conversion_execution_contract_v1",
        "git_commit": str(git_commit),
        "source_manifest": str(Path(manifest_path).resolve()),
        "source_manifest_sha256": _file_sha256(Path(manifest_path)),
        "config": str(Path(config_path).resolve()),
        "config_sha256": _file_sha256(Path(config_path)),
        "model_snapshot_path": str(snapshot.resolve()),
        "model_revision": str(model["revision"]),
        "attention_implementation": str(model["attention_implementation"]),
        "seed": int(config["seed"]),
        "beam_width": int(config["beam_width"]),
        "layer_count": int(config["layer_count"]),
        "torch_version": str(torch_version),
        "transformers_version": str(transformers_version),
        "code_sha256": code_hashes,
    }
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = sha256(encoded.encode()).hexdigest()
    return contract


def empirical_full_run_cost(row: dict[str, Any]) -> int:
    """Return a pilot-calibrated ordering cost for the full conversion queue."""
    cost = int(row["estimated_conversion_cost"])
    if row.get("source_current_all_on_status") == "wrong":
        return PILOT_W2C_COST_MULTIPLIER * cost
    return cost


class AtomicSampleQueue:
    """Claim pending samples once across workers sharing one filesystem.

    Claims are scoped to one launch by ``claim_root``. A later resume launch
    uses a fresh root and may therefore retry samples that did not reach their
    atomic completed record in the earlier launch.
    """

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
            key=lambda row: (-empirical_full_run_cost(row), str(row["uid"])),
        )
        self._claim_root = Path(claim_root)
        self._claim_root.mkdir(parents=True, exist_ok=True)
        self._completed_uids = set(completed_uids)
        self._claimant = str(claimant)
        self._cursor = 0

    @staticmethod
    def _claim_name(uid: str) -> str:
        return f"{sha256(uid.encode()).hexdigest()}.json"

    def claim_next(self) -> dict[str, Any] | None:
        while self._cursor < len(self._rows):
            row = self._rows[self._cursor]
            self._cursor += 1
            uid = str(row["uid"])
            if uid in self._completed_uids:
                continue
            path = self._claim_root / self._claim_name(uid)
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


def balanced_worker_rows(
    rows: Iterable[dict[str, Any]],
    rank: int,
    worker_count: int,
) -> list[dict[str, Any]]:
    """Assign samples by deterministic longest-processing-time bin packing."""
    if worker_count < 1 or not 0 <= rank < worker_count:
        raise ValueError("rank must be inside a positive worker count")
    bins: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    costs = [0] * worker_count
    ordered = sorted(
        rows,
        key=lambda row: (-int(row["estimated_conversion_cost"]), str(row["uid"])),
    )
    for row in ordered:
        target = min(range(worker_count), key=lambda index: (costs[index], index))
        bins[target].append(row)
        costs[target] += int(row["estimated_conversion_cost"])
    return bins[rank]


def _all_off_w2c_proxy(row: dict[str, Any]) -> bool:
    return row.get("source_current_all_on_status") == "wrong" and any(
        bool(route["source_all_off"]) for route in row["source_positive_routes"]
    )


def _minimum_off(row: dict[str, Any]) -> int:
    return min(int(route["source_off_count"]) for route in row["source_positive_routes"])


def _maximum_off(row: dict[str, Any]) -> int:
    return max(int(route["source_off_count"]) for route in row["source_positive_routes"])


def _pick_first(
    population: Sequence[dict[str, Any]],
    selected: list[dict[str, Any]],
    predicate,
    key,
) -> None:
    seen = {str(row["uid"]) for row in selected}
    candidates = [row for row in population if str(row["uid"]) not in seen and predicate(row)]
    if candidates:
        selected.append(min(candidates, key=key))


def _dataset_pilot(population: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    if len(population) < quota:
        raise ValueError("dataset population is smaller than its pilot quota")
    selected: list[dict[str, Any]] = []
    uid_key = lambda row: str(row["uid"])
    priorities = (
        (_all_off_w2c_proxy, uid_key),
        (lambda row: row.get("source_current_all_on_status") == "wrong", uid_key),
        (lambda row: row.get("source_current_all_on_status") == "correct", uid_key),
        (lambda row: int(row["source_positive_route_count"]) > 1, uid_key),
        (lambda row: True, lambda row: (_minimum_off(row), str(row["uid"]))),
        (lambda row: True, lambda row: (-_maximum_off(row), str(row["uid"]))),
        (
            lambda row: True,
            lambda row: (-int(row["source_positive_route_count"]), str(row["uid"])),
        ),
        (
            lambda row: True,
            lambda row: (-int(row["estimated_conversion_cost"]), str(row["uid"])),
        ),
    )
    for predicate, key in priorities:
        if len(selected) >= quota:
            break
        _pick_first(population, selected, predicate, key)
    if len(selected) < quota:
        seen = {str(row["uid"]) for row in selected}
        remaining = sorted(
            (row for row in population if str(row["uid"]) not in seen),
            key=lambda row: (int(row["estimated_conversion_cost"]), str(row["uid"])),
        )
        needed = quota - len(selected)
        if needed == 1:
            chosen = [remaining[len(remaining) // 2]]
        else:
            indices = [round(index * (len(remaining) - 1) / (needed - 1)) for index in range(needed)]
            chosen = [remaining[index] for index in indices]
        selected.extend(chosen)
    return selected


def select_conversion_pilot(
    rows: Iterable[dict[str, Any]],
    *,
    total: int = 56,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select a deterministic five-dataset semantic/cost-stratified pilot."""
    if total < len(DATASETS):
        raise ValueError("pilot must include every dataset")
    population = list(rows)
    base, remainder = divmod(total, len(DATASETS))
    selected = []
    for index, dataset in enumerate(DATASETS):
        current = [row for row in population if row["dataset"] == dataset]
        selected.extend(_dataset_pilot(current, base + (index < remainder)))
    if len({str(row["uid"]) for row in selected}) != len(selected):
        raise RuntimeError("pilot selection produced duplicate UIDs")
    selected.sort(key=lambda row: (DATASETS.index(row["dataset"]), str(row["uid"])))
    coverage = Counter()
    for row in selected:
        coverage["source_w2c_proxy"] += row.get("source_current_all_on_status") == "wrong"
        coverage["source_c2c_proxy"] += row.get("source_current_all_on_status") == "correct"
        coverage["all_off_w2c_proxy"] += _all_off_w2c_proxy(row)
        coverage["multi_route_samples"] += int(row["source_positive_route_count"]) > 1
    return selected, dict(coverage)
