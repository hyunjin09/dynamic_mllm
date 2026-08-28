"""Frozen data contract for the prospective four-action external evaluation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from .actions import FOUR_ACTIONS, normalize_action


ACTIVE_BENCHMARKS = (
    "chartqa",
    "mmmu_pro_standard_test",
    "mmmu_pro_vision_test",
    "pope_adversarial",
    "pope_popular",
    "pope_random",
)
EXPECTED_COUNTS = {
    "chartqa": 2500,
    "mmmu_pro_standard_test": 1730,
    "mmmu_pro_vision_test": 1730,
    "pope_adversarial": 3000,
    "pope_popular": 3000,
    "pope_random": 3000,
}
TOTAL_RECORDS = 14960
CORE_BENCHMARKS = {"chartqa"}
MC_BENCHMARKS = {"mmmu_pro_standard_test", "mmmu_pro_vision_test"}
POPE_BENCHMARKS = {"pope_adversarial", "pope_popular", "pope_random"}
OPTION_SUFFIX = re.compile(r"\s*Answer with the option letter only\.\s*$")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def active_benchmark(name: str) -> bool:
    return str(name).lower() in ACTIVE_BENCHMARKS


def predictor_text(row: dict[str, Any]) -> str:
    benchmark = str(row["benchmark"]).lower()
    if benchmark in CORE_BENCHMARKS or benchmark in POPE_BENCHMARKS:
        text = str(row.get("question") or "").strip()
    elif benchmark in MC_BENCHMARKS:
        text = OPTION_SUFFIX.sub(
            "", "".join(str(chunk) for chunk in row.get("instruction_text_chunks") or [])
        ).strip()
    else:
        raise ValueError(f"inactive benchmark: {benchmark}")
    if not text:
        raise ValueError(f"predictor input is empty for {row.get('uid')}")
    return text


def cluster_key(row: dict[str, Any]) -> str:
    hashes = row.get("image_content_sha256s")
    if hashes:
        return "|".join(str(value) for value in hashes)
    value = row.get("image_content_sha256")
    if not value:
        raise ValueError(f"record lacks an image hash: {row.get('uid')}")
    return str(value)


def load_active_rows(data_root: str | Path) -> list[dict[str, Any]]:
    root = Path(data_root)
    sources = (
        root / "heldout_lmms_recommended_v1/samples.jsonl",
        root / "heldout_mmstar_mmmu_final_v2/samples.jsonl",
        root / "heldout_pope_v1/samples.jsonl",
    )
    rows = []
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"external evaluation source is missing: {source}")
        for row in read_jsonl(source):
            if not active_benchmark(row["benchmark"]):
                continue
            current = dict(row)
            current["data_root"] = str(source.parent)
            current["cluster_key"] = cluster_key(row)
            current["predictor_text"] = predictor_text(row)
            rows.append(current)
    counts = Counter(str(row["benchmark"]).lower() for row in rows)
    if counts != Counter(EXPECTED_COUNTS) or len(rows) != TOTAL_RECORDS:
        raise RuntimeError(f"external evaluation population mismatch: {counts}")
    uids = [str(row["uid"]) for row in rows]
    if len(uids) != len(set(uids)):
        raise RuntimeError("external evaluation UIDs are not unique")
    return rows


def select_shard(
    rows: list[dict[str, Any]], *, num_shards: int, shard_index: int
) -> list[dict[str, Any]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def action_statistics(actions: list[str] | tuple[str, ...]) -> dict[str, Any]:
    normalized = [normalize_action(action) for action in actions]
    counts = Counter(normalized)
    return {
        "route_key": "|".join(normalized),
        "action_counts": {action: counts[action] for action in FOUR_ACTIONS},
        "non_full_layers": len(normalized) - counts["FULL"],
        "read_enabled_layers": counts["READ_ONLY"] + counts["FULL"],
        "write_enabled_layers": counts["WRITE_ONLY"] + counts["FULL"],
        "transition_count": sum(
            left != right for left, right in zip(normalized, normalized[1:])
        ),
    }
