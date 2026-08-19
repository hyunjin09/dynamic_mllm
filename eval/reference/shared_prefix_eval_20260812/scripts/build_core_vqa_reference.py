#!/usr/bin/env python3
"""Extract the portable ChartQA/TextVQA/DocVQA reference evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


BENCHMARKS = ("chartqa", "textvqa", "docvqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--source-manifest-jsonl", type=Path, default=None)
    parser.add_argument("--source-rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-output-jsonl", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_mean(
    rows: list[dict[str, Any]],
    getter: Callable[[dict[str, Any]], float],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    import numpy as np

    values = np.asarray([float(getter(row)) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    # Chunking caps temporary memory for the 12,849-sample micro-average.
    for start in range(0, repetitions, 100):
        count = min(100, repetitions - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        draws.extend(values[indices].mean(axis=1).tolist())
    draws.sort()
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "ci_low": percentile(draws, 0.025),
        "ci_high": percentile(draws, 0.975),
    }


def outcome(row: dict[str, Any]) -> str:
    baseline = bool(row["baseline_correct"])
    router = bool(row["router_correct"])
    if baseline and router:
        return "preserve"
    if baseline and not router:
        return "harm"
    if not baseline and router:
        return "rescue"
    return "unsolved"


def summarize(
    rows: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    groups = {"all": rows}
    groups.update({benchmark: [row for row in rows if row["benchmark"] == benchmark] for benchmark in BENCHMARKS})
    metrics: dict[str, Callable[[dict[str, Any]], float]] = {
        "baseline_score": lambda row: float(row["baseline_score"]),
        "baseline_correct_rate": lambda row: float(bool(row["baseline_correct"])),
        "router_score": lambda row: float(row["router_score"]),
        "router_correct_rate": lambda row: float(bool(row["router_correct"])),
        "paired_correct_delta": lambda row: float(bool(row["router_correct"]))
        - float(bool(row["baseline_correct"])),
        "avg_selected_layers": lambda row: float(row["selected_num_visual_on_layers"]),
    }
    summary: dict[str, Any] = {}
    for group_index, (name, group) in enumerate(groups.items()):
        group_summary: dict[str, Any] = {
            "samples": len(group),
            "unique_masks": len({str(row["selected_mask_key"]) for row in group}),
            "outcomes": dict(Counter(outcome(row) for row in group)),
        }
        for metric_index, (metric_name, getter) in enumerate(metrics.items()):
            group_summary[metric_name] = bootstrap_mean(
                group,
                getter,
                repetitions=repetitions,
                seed=seed + group_index * 100 + metric_index,
            )
        summary[name] = group_summary
    return summary


def format_ci(metric: dict[str, Any], *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    return (
        f"{float(metric['mean']) * scale:.2f}"
        f" [{float(metric['ci_low']) * scale:.2f}, {float(metric['ci_high']) * scale:.2f}]"
    )


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Core VQA Reference Evaluation",
        "",
        "Frozen Qwen2.5-VL-7B all-on and SW31 online-router generation on the project prompt.",
        "Correct rate thresholds are 1.0 for ChartQA and 0.5 for TextVQA/DocVQA.",
        "",
        "| group | n | all-on correct % [95% CI] | router correct % [95% CI] | delta pp [95% CI] | all-on score | router score | avg on layers | unique masks | harm | rescue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all",) + BENCHMARKS:
        group = summary[name]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(group["samples"]),
                    format_ci(group["baseline_correct_rate"], percent=True),
                    format_ci(group["router_correct_rate"], percent=True),
                    format_ci(group["paired_correct_delta"], percent=True),
                    f"{group['baseline_score']['mean']:.4f}",
                    f"{group['router_score']['mean']:.4f}",
                    f"{group['avg_selected_layers']['mean']:.2f}",
                    str(group["unique_masks"]),
                    str(group["outcomes"].get("harm", 0)),
                    str(group["outcomes"].get("rescue", 0)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "`score` and thresholded `correct` are both reported because TextVQA and DocVQA scores are fractional.",
            "The combined row is a micro-average over 12,849 samples, not a benchmark macro-average.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    manifest = read_jsonl(args.manifest_jsonl)
    expected = {str(row["uid"]): row for row in manifest}
    if len(manifest) != 12849 or len(expected) != len(manifest):
        raise ValueError("core VQA manifest must contain 12,849 unique UIDs")

    if args.source_manifest_jsonl is not None:
        source_manifest_rows = [
            row
            for row in read_jsonl(args.source_manifest_jsonl)
            if str(row.get("benchmark")) in BENCHMARKS
        ]
        source_manifest = {str(row["uid"]): row for row in source_manifest_rows}
        alignment_fields = (
            "sample_id",
            "benchmark",
            "source_dataset",
            "source_split",
            "question",
            "prompt",
            "answer",
            "all_answer_norms",
            "metric_name",
            "correctness_threshold",
            "max_new_tokens",
            "max_pixels",
            "max_image_tokens",
            "image_content_sha256",
        )
        if set(source_manifest) != set(expected):
            raise ValueError("source and portable core VQA manifest UID sets differ")
        for uid, row in expected.items():
            for field in alignment_fields:
                if row.get(field) != source_manifest[uid].get(field):
                    raise ValueError(f"source manifest {field} mismatch for {uid}")
        write_json(
            args.output_dir / "source_manifest_alignment.json",
            {
                "matched": True,
                "rows": len(expected),
                "fields": list(alignment_fields),
                "portable_manifest_sha256": sha256(args.manifest_jsonl),
                "source_manifest_sha256": sha256(args.source_manifest_jsonl),
            },
        )

    source_rows = read_jsonl(args.source_rows_jsonl)
    rows = [row for row in source_rows if str(row.get("benchmark")) in BENCHMARKS]
    indexed = {str(row["uid"]): row for row in rows}
    if len(rows) != 12849 or len(indexed) != len(rows) or set(indexed) != set(expected):
        raise ValueError("source generation rows do not match the core VQA manifest")

    ordered = [indexed[str(row["uid"])] for row in manifest]
    for row in ordered:
        manifest_row = expected[str(row["uid"])]
        for field in ("benchmark", "metric_name", "correctness_threshold"):
            if row.get(field) != manifest_row.get(field):
                raise ValueError(f"{field} mismatch for {row['uid']}")

    baseline_rows = [
        {
            "uid": row["uid"],
            "sample_id": row["sample_id"],
            "benchmark": row["benchmark"],
            "metric_name": row["metric_name"],
            "correctness_threshold": row["correctness_threshold"],
            "baseline_prediction": row["baseline_prediction"],
            "baseline_score": row["baseline_score"],
            "baseline_correct": row["baseline_correct"],
        }
        for row in ordered
    ]
    write_jsonl(args.baseline_output_jsonl, baseline_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_rows_path = args.output_dir / "heldout_generation_rows.jsonl"
    write_jsonl(reference_rows_path, ordered)
    summary = summarize(
        ordered,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    payload = {
        "evaluation_version": "core_vqa_reference_eval_v1",
        "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "router_checkpoint_sha256": "6ecf2f9119b78d5d11c969b4602b93cecc59d27aab43440abacb84421c4af255",
        "manifest_sha256": sha256(args.manifest_jsonl),
        "source_rows_sha256": sha256(args.source_rows_jsonl),
        "reference_rows_sha256": sha256(reference_rows_path),
        "baseline_rows_sha256": sha256(args.baseline_output_jsonl),
        "summary": summary,
    }
    write_json(args.output_dir / "summary.json", payload)
    (args.output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
