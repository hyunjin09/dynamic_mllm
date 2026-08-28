#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.research_analysis.four_action.cohort import build_rows, summarize


DEFAULT_LABEL_ROOT = Path("datasets/mcts_labels/gqa_textvqa_chartqa_v1")
DEFAULT_IMAGE_ROOT = Path(
    "datasets/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713/images"
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze four-action primary/control cohorts.")
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/4action_answer_alignment/cohort"))
    args = parser.parse_args()

    post = args.label_root / "post_generation"
    summaries = read_jsonl(post / "per_sample_route_summary_v1.jsonl")
    sources = {row["uid"]: row for row in read_jsonl(args.label_root / "source_manifest_v1.jsonl")}
    indices = {row["uid"]: row for row in read_jsonl(post / "cache_record_index_v1.jsonl")}
    rows, taxonomy = build_rows(summaries, sources, indices, args.image_root)
    summary = summarize(rows, taxonomy)
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    write_once(args.output_dir / "cohort_manifest_v1.jsonl", manifest_text)
    write_once(args.output_dir / "cohort_summary_v1.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in ("emitted_rows", "primary_rows", "taxonomy")}, indent=2))


if __name__ == "__main__":
    main()
