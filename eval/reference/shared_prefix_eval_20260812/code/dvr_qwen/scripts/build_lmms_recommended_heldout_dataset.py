#!/usr/bin/env python3
"""Build the heldout lmms-eval comparison manifest for DVR router evaluation.

This artifact is intentionally sample-only: it fixes images, prompts, answer
references, metrics, and runtime policy, but it does not include model outputs,
route scores, or pseudo-labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path("/mnt/hyemin/10k_dataset_mask/heldout_lmms_recommended_v1")
DEFAULT_DATASETS_CACHE = Path("/mnt/hyemin/dataset/datasets")
DEFAULT_HF_HOME = Path("/mnt/hyemin/dataset")

SPLIT_SPECS = {
    "chartqa": {
        "dataset_path": "lmms-lab/ChartQA",
        "dataset_name": None,
        "split": "test",
        "task_yaml": "lmms_eval/tasks/chartqa/chartqa.yaml",
        "metric_name": "relaxed_accuracy",
        "correctness_threshold": 1.0,
        "max_new_tokens": 16,
        "max_pixels": None,
        "max_image_tokens": 0,
        "lmms_default_post_prompt": "\nAnswer the question with a single word.",
        "lmms_qwen_vl_post_prompt": " Answer:",
        "image_extension": "png",
        "image_save_format": "PNG",
    },
    "textvqa": {
        "dataset_path": "lmms-lab/textvqa",
        "dataset_name": None,
        "split": "validation",
        "task_yaml": "lmms_eval/tasks/textvqa/textvqa_val.yaml",
        "metric_name": "textvqa_evalai_consensus",
        "correctness_threshold": 0.5,
        "max_new_tokens": 16,
        "max_pixels": None,
        "max_image_tokens": 0,
        "lmms_default_post_prompt": "\nAnswer the question using a single word or phrase.",
        "lmms_qwen_vl_post_prompt": " Answer:",
        "image_extension": "jpg",
        "image_save_format": "JPEG",
    },
    "docvqa": {
        "dataset_path": "lmms-lab/DocVQA",
        "dataset_name": "DocVQA",
        "split": "validation",
        "task_yaml": "lmms_eval/tasks/docvqa/docvqa_val.yaml",
        "metric_name": "anls",
        "correctness_threshold": 0.5,
        "max_new_tokens": 32,
        "max_pixels": 802816,
        "max_image_tokens": 1024,
        "lmms_default_post_prompt": "\nAnswer the question using a single word or phrase.",
        "lmms_qwen_vl_post_prompt": " Answer:",
        "image_extension": "png",
        "image_save_format": "PNG",
    },
    "pope_adversarial": {
        "dataset_path": "lmms-lab/POPE",
        "dataset_name": "Full",
        "split": "adversarial",
        "task_yaml": "lmms_eval/tasks/pope/pope_adv.yaml",
        "metric_name": "pope_yes_no_accuracy",
        "correctness_threshold": 1.0,
        "max_new_tokens": 128,
        "max_pixels": None,
        "max_image_tokens": 0,
        "lmms_default_post_prompt": "\nAnswer the question using a single word or phrase.",
        "lmms_qwen_vl_post_prompt": "\nAnswer the question using a single word or phrase.",
        "image_extension": "jpg",
        "image_save_format": "JPEG",
    },
    "pope_popular": {
        "dataset_path": "lmms-lab/POPE",
        "dataset_name": "Full",
        "split": "popular",
        "task_yaml": "lmms_eval/tasks/pope/pope_pop.yaml",
        "metric_name": "pope_yes_no_accuracy",
        "correctness_threshold": 1.0,
        "max_new_tokens": 128,
        "max_pixels": None,
        "max_image_tokens": 0,
        "lmms_default_post_prompt": "\nAnswer the question using a single word or phrase.",
        "lmms_qwen_vl_post_prompt": "\nAnswer the question using a single word or phrase.",
        "image_extension": "jpg",
        "image_save_format": "JPEG",
    },
    "pope_random": {
        "dataset_path": "lmms-lab/POPE",
        "dataset_name": "Full",
        "split": "random",
        "task_yaml": "lmms_eval/tasks/pope/pope_random.yaml",
        "metric_name": "pope_yes_no_accuracy",
        "correctness_threshold": 1.0,
        "max_new_tokens": 128,
        "max_pixels": None,
        "max_image_tokens": 0,
        "lmms_default_post_prompt": "\nAnswer the question using a single word or phrase.",
        "lmms_qwen_vl_post_prompt": "\nAnswer the question using a single word or phrase.",
        "image_extension": "jpg",
        "image_save_format": "JPEG",
    },
    "seedbench": {
        "dataset_path": "lmms-lab/SEED-Bench",
        "dataset_name": None,
        "split": "test",
        "task_yaml": "lmms_eval/tasks/seedbench/seedbench.yaml",
        "metric_name": "seed_choice_accuracy",
        "correctness_threshold": 1.0,
        "max_new_tokens": 16,
        "max_pixels": None,
        "max_image_tokens": 0,
        "lmms_default_post_prompt": "Answer with the option's letter from the given choices directly.",
        "lmms_qwen_vl_post_prompt": "Answer with the option's letter from the given choices directly.",
        "image_extension": "jpg",
        "image_save_format": "JPEG",
    },
    "seedbench_lite": {
        "dataset_path": "lmms-lab/LMMs-Eval-Lite",
        "dataset_name": "seedbench",
        "split": "lite",
        "task_yaml": "lmms_eval/tasks/seedbench/seedbench_lite.yaml",
        "metric_name": "seed_choice_accuracy",
        "correctness_threshold": 1.0,
        "max_new_tokens": 16,
        "max_pixels": None,
        "max_image_tokens": 0,
        "lmms_default_post_prompt": "Answer with the option's letter from the given choices directly.",
        "lmms_qwen_vl_post_prompt": "Answer with the option's letter from the given choices directly.",
        "image_extension": "jpg",
        "image_save_format": "JPEG",
    },
}

PROJECT_POST_PROMPT = "\nAnswer the question using a single word or phrase."
SEED_POST_PROMPT = "Answer with the option's letter from the given choices directly."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-version", default="heldout_lmms_recommended_v1")
    parser.add_argument("--datasets-cache", type=Path, default=DEFAULT_DATASETS_CACHE)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--benchmarks",
        default="chartqa,textvqa,docvqa",
        help="Comma-separated subset of available split specs.",
    )
    parser.add_argument("--limit-per-benchmark", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None and str(item) != ""]
    text = str(value)
    return [text] if text != "" else []


def primary_answer(answers: list[str]) -> str:
    if not answers:
        return ""
    counts = Counter(answers)
    return sorted(counts.items(), key=lambda item: (-item[1], answers.index(item[0])))[0][0]


def save_rgb_image(image: Any, path: Path, save_format: str, overwrite: bool) -> tuple[str, tuple[int, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or overwrite:
        rgb = image.convert("RGB")
        if save_format == "JPEG":
            rgb.save(path, format="JPEG", quality=95)
        elif save_format == "PNG":
            rgb.save(path, format="PNG", compress_level=1)
        else:
            raise ValueError(f"unsupported image save format: {save_format}")
    return sha256_file(path), tuple(image.size)


def safe_id(value: Any) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def sample_id_for(benchmark: str, doc: dict[str, Any], index: int) -> str:
    if benchmark == "chartqa":
        return f"chartqa_test_{index:06d}"
    if benchmark == "textvqa":
        return f"textvqa_val_{int(doc['question_id']):012d}"
    if benchmark == "docvqa":
        return f"docvqa_val_{int(doc['questionId']):012d}"
    if benchmark.startswith("pope_"):
        return f"{benchmark}_{int(doc['id']):06d}"
    if benchmark == "seedbench":
        return f"seedbench_test_{safe_id(doc['question_id'])}"
    if benchmark == "seedbench_lite":
        return f"seedbench_lite_{safe_id(doc['question_id'])}"
    raise ValueError(f"unknown benchmark: {benchmark}")


def seed_prompt(doc: dict[str, Any]) -> str:
    question = str(doc["question"])
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"
    return f"{question}\n{SEED_POST_PROMPT}"


def row_for_doc(
    benchmark: str,
    spec: dict[str, Any],
    doc: dict[str, Any],
    index: int,
    image_path: Path,
    image_sha256: str,
    image_size: tuple[int, int],
    out_dir: Path,
    dataset_version: str,
) -> dict[str, Any]:
    if benchmark == "chartqa":
        question = str(doc["question"])
        answers = normalize_answers(doc["answer"])
        extra = {"chartqa_type": doc.get("type")}
    elif benchmark == "textvqa":
        question = str(doc["question"])
        answers = normalize_answers(doc.get("answers"))
        extra = {
            "image_id": doc.get("image_id"),
            "question_id": doc.get("question_id"),
            "set_name": doc.get("set_name"),
            "ocr_tokens": doc.get("ocr_tokens") or [],
            "question_tokens": doc.get("question_tokens") or [],
            "image_classes": doc.get("image_classes") or [],
        }
    elif benchmark == "docvqa":
        question = str(doc["question"])
        answers = normalize_answers(doc.get("answers"))
        extra = {
            "question_id": doc.get("questionId"),
            "question_types": doc.get("question_types") or [],
            "doc_id": doc.get("docId"),
            "ucsf_document_id": doc.get("ucsf_document_id"),
            "ucsf_document_page_no": doc.get("ucsf_document_page_no"),
            "data_split": doc.get("data_split"),
        }
    elif benchmark.startswith("pope_"):
        question = str(doc["question"]).strip()
        answers = normalize_answers(doc.get("answer"))
        extra = {
            "pope_category": doc.get("category"),
            "question_id": doc.get("question_id"),
            "image_source": doc.get("image_source"),
        }
    elif benchmark in {"seedbench", "seedbench_lite"}:
        question = str(doc["question"])
        answers = normalize_answers(doc.get("answer"))
        extra = {
            "question_id": doc.get("question_id"),
            "question_type_id": doc.get("question_type_id"),
            "data_id": doc.get("data_id"),
            "data_type": doc.get("data_type"),
            "segment": doc.get("segment") or [],
            "choices": {
                "A": doc.get("choice_a"),
                "B": doc.get("choice_b"),
                "C": doc.get("choice_c"),
                "D": doc.get("choice_d"),
            },
        }
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")

    sid = sample_id_for(benchmark, doc, index)
    image_paths = image_path if isinstance(image_path, list) else [image_path]
    image_sha256s = image_sha256 if isinstance(image_sha256, list) else [image_sha256]
    image_sizes = image_size if isinstance(image_size, list) else [image_size]
    rel_images = [path.relative_to(out_dir).as_posix() for path in image_paths]
    if benchmark in {"seedbench", "seedbench_lite"}:
        prompt = seed_prompt(doc)
        lmms_default_prompt = prompt
        lmms_qwen_vl_prompt = prompt
    else:
        prompt = f"{question}{PROJECT_POST_PROMPT}"
        lmms_default_prompt = f"{question}{spec['lmms_default_post_prompt']}"
        lmms_qwen_vl_prompt = f"{question}{spec['lmms_qwen_vl_post_prompt']}"
    row = {
        "uid": f"{benchmark}:{sid}",
        "sample_id": sid,
        "benchmark": benchmark,
        "dataset_role": "heldout_recommended_comparison",
        "dataset_version": dataset_version,
        "source_dataset": spec["dataset_path"],
        "source_dataset_name": spec["dataset_name"],
        "source_split": spec["split"],
        "source_index": index,
        "task_yaml": spec["task_yaml"],
        "question": question,
        "prompt": prompt,
        "prompt_policy": "project_dvr_default_single_word_or_phrase",
        "lmms_eval_default_prompt": lmms_default_prompt,
        "lmms_eval_qwen_vl_prompt": lmms_qwen_vl_prompt,
        "answer": primary_answer(answers),
        "all_answer_norms": answers,
        "metric_name": spec["metric_name"],
        "correctness_threshold": spec["correctness_threshold"],
        "max_new_tokens": spec["max_new_tokens"],
        "max_pixels": spec["max_pixels"],
        "max_image_tokens": spec["max_image_tokens"],
        "image_path": str(image_paths[0]),
        "image_paths": [str(path) for path in image_paths],
        "image_relpath": rel_images[0],
        "image_relpaths": rel_images,
        "image_content_sha256": image_sha256s[0],
        "image_content_sha256s": image_sha256s,
        "image_width": int(image_sizes[0][0]),
        "image_height": int(image_sizes[0][1]),
        "image_sizes": [{"width": int(size[0]), "height": int(size[1])} for size in image_sizes],
        "image_count": len(image_paths),
        "route_or_score_status": "not_evaluated",
        "has_answer": bool(answers),
    }
    row.update(extra)
    return row


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    selected = [item.strip().lower() for item in args.benchmarks.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(SPLIT_SPECS))
    if unknown:
        raise ValueError(f"unknown benchmarks: {unknown}")

    rows: list[dict[str, Any]] = []
    for benchmark in selected:
        spec = SPLIT_SPECS[benchmark]
        kwargs = {
            "path": spec["dataset_path"],
            "split": spec["split"],
            "cache_dir": str(args.datasets_cache),
        }
        if spec["dataset_name"] is not None:
            kwargs["name"] = spec["dataset_name"]
        dataset = load_dataset(**kwargs)
        limit = len(dataset) if args.limit_per_benchmark <= 0 else min(args.limit_per_benchmark, len(dataset))
        print(f"[{benchmark}] split={spec['split']} rows={len(dataset)} writing={limit}", flush=True)
        for index in range(limit):
            doc = dataset[index]
            sid = sample_id_for(benchmark, doc, index)
            source_images = doc["image"] if isinstance(doc["image"], list) else [doc["image"]]
            image_paths = []
            image_sha256s = []
            image_sizes = []
            for image_index, image in enumerate(source_images):
                suffix = "" if len(source_images) == 1 else f"__img{image_index:02d}"
                image_path = args.out_dir / "images" / benchmark / f"{sid}{suffix}.{spec['image_extension']}"
                image_sha256, image_size = save_rgb_image(image, image_path, spec["image_save_format"], args.overwrite)
                image_paths.append(image_path)
                image_sha256s.append(image_sha256)
                image_sizes.append(image_size)
            rows.append(
                row_for_doc(
                    benchmark,
                    spec,
                    doc,
                    index,
                    image_paths,
                    image_sha256s,
                    image_sizes,
                    args.out_dir,
                    args.dataset_version,
                )
            )
            if (index + 1) % 500 == 0:
                print(f"[{benchmark}] {index + 1}/{limit}", flush=True)
    return rows


def summary_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_benchmark: dict[str, dict[str, Any]] = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        bench_rows = [row for row in rows if row["benchmark"] == benchmark]
        answer_counts = [len(row["all_answer_norms"]) for row in bench_rows]
        by_benchmark[benchmark] = {
            "count": len(bench_rows),
            "source_dataset": bench_rows[0]["source_dataset"],
            "source_dataset_name": bench_rows[0]["source_dataset_name"],
            "source_split": bench_rows[0]["source_split"],
            "metric_name": bench_rows[0]["metric_name"],
            "correctness_threshold": bench_rows[0]["correctness_threshold"],
            "max_new_tokens": bench_rows[0]["max_new_tokens"],
            "max_pixels": bench_rows[0]["max_pixels"],
            "max_image_tokens": bench_rows[0]["max_image_tokens"],
            "missing_answer_count": sum(1 for row in bench_rows if not row["has_answer"]),
            "min_answer_refs": min(answer_counts) if answer_counts else 0,
            "max_answer_refs": max(answer_counts) if answer_counts else 0,
            "min_image_count": min(int(row["image_count"]) for row in bench_rows),
            "max_image_count": max(int(row["image_count"]) for row in bench_rows),
            "multi_image_rows": sum(1 for row in bench_rows if int(row["image_count"]) > 1),
        }
    return {
        "dataset_version": rows[0]["dataset_version"] if rows else "unknown",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_count": len(rows),
        "benchmarks": by_benchmark,
        "prompt_policy": {
            "prompt": "project DVR prompt: question + '\\nAnswer the question using a single word or phrase.'",
            "lmms_eval_default_prompt": "stored separately for exact lmms-eval default reproduction",
            "lmms_eval_qwen_vl_prompt": "stored separately for lmms-eval qwen_vl prompt reproduction",
        },
    }


def schema() -> dict[str, Any]:
    return {
        "primary_file": "samples.jsonl",
        "row_keys": {
            "uid": "Stable unique sample key: '<benchmark>:<sample_id>'.",
            "sample_id": "Stable local sample id derived from source split and source question id/index.",
            "benchmark": "Dataset/split key such as chartqa, textvqa, docvqa, pope_adversarial, pope_popular, pope_random, or seedbench.",
            "prompt": "Project DVR prompt used for router/binary generation unless overridden.",
            "lmms_eval_default_prompt": "Prompt generated by lmms-eval default kwargs.",
            "lmms_eval_qwen_vl_prompt": "Prompt generated by lmms-eval qwen_vl kwargs.",
            "answer": "Primary fallback answer string.",
            "all_answer_norms": "All available answer references for consensus/ANLS, or singleton answer refs for exact-choice tasks.",
            "metric_name": "Metric name accepted by dvr_qwen.eval_metrics.score_prediction.",
            "correctness_threshold": "Score threshold used to convert score into binary correctness.",
            "image_path": "Absolute saved first image path.",
            "image_paths": "Absolute saved image paths. This is authoritative for multi-image SEED rows.",
            "image_relpath": "Path relative to this dataset directory.",
            "image_relpaths": "Image paths relative to this dataset directory.",
            "route_or_score_status": "not_evaluated until a model/route generation pass fills outputs.",
        },
        "not_included": [
            "No model predictions.",
            "No all-on/all-off scores.",
            "No greedy or MCTS mask labels.",
            "No train/validation preference pairs.",
        ],
    }


def readme_text(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['dataset_version']} - Final Ver",
        "",
        "This directory fixes the recommended locally scorable comparison splits for the router/Qwen evaluation stage.",
        "",
        "## Splits",
        "",
        "| Benchmark | Source | Split | Count | Metric | Correct threshold | Runtime notes |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for benchmark, item in summary["benchmarks"].items():
        source_name = item["source_dataset"]
        if item["source_dataset_name"]:
            source_name += f"/{item['source_dataset_name']}"
        runtime = f"max_new_tokens={item['max_new_tokens']}, max_pixels={item['max_pixels']}, max_image_tokens={item['max_image_tokens']}"
        lines.append(
            f"| {benchmark} | `{source_name}` | `{item['source_split']}` | {item['count']} | "
            f"`{item['metric_name']}` | {item['correctness_threshold']} | {runtime} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `samples.jsonl`: authoritative heldout sample manifest.",
            "- `images/`: RGB image copies of the source images used by the manifest.",
            "- `summary.json`: counts and scoring/runtime policy.",
            "- `schema.json`: row field definitions.",
            "- `metadata.json`: build environment and source cache metadata.",
            "- `checksums.sha256`: checksums for manifest files.",
            "",
            "## Important Policy",
            "",
            "This is not a preference-GT dataset and does not contain route labels or model scores. "
            "Run baseline/all-off/router generation separately and write prediction files that join on `uid`.",
            "",
            "The `prompt` field follows the project DVR training/evaluation policy. Exact lmms-eval prompts are also stored as "
            "`lmms_eval_default_prompt` and `lmms_eval_qwen_vl_prompt` so prompt policy can be audited explicitly.",
            "",
            "GQA is intentionally absent because the local cache does not contain the recommended `testdev` split.",
            "",
        ]
    )
    if "seedbench_lite" in summary["benchmarks"] and "seedbench" not in summary["benchmarks"]:
        lines.extend(
            [
                "## SEED-Bench Full Test Status",
                "",
                "`seedbench_lite` is included as the locally completed SEED-family comparison split. "
                "The full `lmms-lab/SEED-Bench` `test` split is not included here unless the `seedbench` benchmark key "
                "appears in the split table above.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.datasets_cache))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args)
    if not rows:
        raise RuntimeError("no rows were written")

    uids = [row["uid"] for row in rows]
    if len(uids) != len(set(uids)):
        duplicates = [uid for uid, count in Counter(uids).items() if count > 1]
        raise RuntimeError(f"duplicate uids: {duplicates[:10]}")

    missing_by_benchmark = defaultdict(int)
    for row in rows:
        if not row["has_answer"]:
            missing_by_benchmark[row["benchmark"]] += 1
    if missing_by_benchmark:
        raise RuntimeError(f"unexpected missing answers: {dict(missing_by_benchmark)}")

    write_jsonl(args.out_dir / "samples.jsonl", rows)
    summary = summary_for(rows)
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "schema.json", schema())
    write_json(
        args.out_dir / "metadata.json",
        {
            "dataset_version": args.dataset_version,
            "builder_script": str(Path(__file__).resolve()),
            "project_root": str(ROOT),
            "datasets_cache": str(args.datasets_cache),
            "hf_home": str(args.hf_home),
            "python_env_note": "built with project dvr_qwen/.venv",
            "split_specs": SPLIT_SPECS,
            "limit_per_benchmark": args.limit_per_benchmark,
            "overwrite": args.overwrite,
        },
    )
    (args.out_dir / "README_final_ver.md").write_text(readme_text(summary), encoding="utf-8")

    checksum_targets = ["samples.jsonl", "summary.json", "schema.json", "metadata.json", "README_final_ver.md"]
    checksum_lines = [f"{sha256_file(args.out_dir / name)}  {name}" for name in checksum_targets]
    (args.out_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    write_json(
        args.out_dir / "manifest_fingerprint.json",
        {
            "samples_jsonl_sha256": sha256_file(args.out_dir / "samples.jsonl"),
            "uid_sequence_sha256": sha256_text("\n".join(uids) + "\n"),
            "total_count": len(rows),
            "counts_by_benchmark": dict(Counter(row["benchmark"] for row in rows)),
        },
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
