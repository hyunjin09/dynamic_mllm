#!/usr/bin/env python3
"""Build the heldout manifest for MMStar, MMMU, and MMMU-Pro.

The builder reads the already cached Arrow files directly.  This avoids
Hugging Face cache lock creation on the evaluation server and makes the
source snapshot used by the manifest explicit.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, Image as HFImage, concatenate_datasets


DEFAULT_CACHE = Path("/mnt/hyemin/dataset/datasets")
DEFAULT_OUT = Path("/mnt/hyemin/10k_dataset_mask/heldout_mmstar_mmmu_v1")
MMMU_CACHE_REV = "364f2e2eb107b36e07ff4c5a15f5947a759cef47"
MMMU_PRO_CACHE_REV = "1ba55708b8588a8f9b180b8fec9e6435c88ce363"
MMSTAR_CACHE_REV = "bc98d668301da7b14f648724866e57302778ab27"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_arrow_files(files: list[Path]) -> Dataset:
    if not files:
        raise FileNotFoundError("no cached Arrow files found")
    dataset = concatenate_datasets([Dataset.from_file(str(path)) for path in files])
    for name, feature in dataset.features.items():
        if isinstance(feature, HFImage):
            dataset = dataset.cast_column(name, HFImage(decode=False))
    return dataset


def source_files(cache: Path) -> dict[str, list[Path]]:
    root = cache
    return {
        "mmstar_val": [
            root / "Lin-Chen___mm_star" / "val" / "0.0.0" / MMSTAR_CACHE_REV / "mm_star-val.arrow",
        ],
        "mmmu_val": [
            root / "lmms-lab___mmmu" / "default" / "0.0.0" / MMMU_CACHE_REV / "mmmu-validation.arrow",
        ],
        "mmmu_pro_standard_test": sorted(
            (root / "MMMU___mmmu_pro" / "standard (10 options)" / "0.0.0" / MMMU_PRO_CACHE_REV).glob("*.arrow")
        ),
        "mmmu_pro_vision_test": sorted(
            (root / "MMMU___mmmu_pro" / "vision" / "0.0.0" / MMMU_PRO_CACHE_REV).glob("*.arrow")
        ),
    }


def options(value: Any) -> list[str]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"options is not a list: {value!r}")
    return [str(item) for item in parsed]


def option_text(values: list[str]) -> str:
    return "\n".join(f"{chr(ord('A') + index)}. {value}" for index, value in enumerate(values))


def image_value(doc: dict[str, Any], key: str) -> Any:
    value = doc.get(key)
    if value is None:
        return None
    return value


def image_list(benchmark: str, doc: dict[str, Any]) -> list[Any]:
    if benchmark == "mmstar_val":
        return [doc["image"]]
    if benchmark == "mmmu_pro_vision_test":
        return [doc["image"]]
    return [image_value(doc, f"image_{index}") for index in range(1, 8) if image_value(doc, f"image_{index}") is not None]


def prompt_for(benchmark: str, doc: dict[str, Any]) -> tuple[str, list[str]]:
    if benchmark == "mmstar_val":
        prompt = f"{doc['question']}\nAnswer with the option letter only."
        return prompt, [str(doc["question"])]
    choice_text = option_text(options(doc["options"]))
    if benchmark == "mmmu_pro_vision_test":
        prompt = f"Question: analyze the image and answer the associated question.\nOptions:\n{choice_text}\nAnswer with the option letter only."
        return prompt, [prompt]
    question = str(doc["question"])
    prompt = f"{question}\n{choice_text}\nAnswer with the option letter only."
    chunks = [chunk for chunk in re.split(r"<image(?: \d+)?>", question) if chunk.strip()]
    chunks.append(choice_text + "\nAnswer with the option letter only.")
    return prompt, chunks


def sample_id(benchmark: str, doc: dict[str, Any], index: int) -> str:
    if benchmark == "mmstar_val":
        return f"mmstar_val_{int(doc['index']):06d}"
    return f"{benchmark}_{str(doc['id']).replace('/', '_')}"


def source_info(benchmark: str) -> dict[str, Any]:
    specs = {
        "mmstar_val": {
            "source_dataset": "Lin-Chen/MMStar", "source_dataset_name": None, "source_split": "val",
            "task_yaml": "lmms_eval/tasks/mmstar/mmstar_qwen.yaml", "metric_name": "mmstar_choice_accuracy",
            "max_new_tokens": 16, "max_pixels": None, "max_image_tokens": 0, "image_extension": "png",
        },
        "mmmu_val": {
            "source_dataset": "lmms-lab/MMMU", "source_dataset_name": None, "source_split": "validation",
            "task_yaml": "lmms_eval/tasks/mmmu/mmmu_val.yaml", "metric_name": "mmmu_acc",
            "max_new_tokens": 16, "max_pixels": 802816, "max_image_tokens": 1024, "image_extension": "png",
        },
        "mmmu_pro_standard_test": {
            "source_dataset": "MMMU/MMMU_Pro", "source_dataset_name": "standard (10 options)", "source_split": "test",
            "task_yaml": "lmms_eval/tasks/mmmu_pro/mmmu_pro_standard.yaml", "metric_name": "mmmu_acc",
            "max_new_tokens": 16, "max_pixels": 802816, "max_image_tokens": 1024, "image_extension": "png",
        },
        "mmmu_pro_vision_test": {
            "source_dataset": "MMMU/MMMU_Pro", "source_dataset_name": "vision", "source_split": "test",
            "task_yaml": "lmms_eval/tasks/mmmu_pro/mmmu_pro_vision.yaml", "metric_name": "mmmu_acc",
            "max_new_tokens": 16, "max_pixels": 802816, "max_image_tokens": 1024, "image_extension": "png",
        },
    }
    return specs[benchmark]


def build_rows(name: str, dataset: Dataset, out_dir: Path, overwrite: bool) -> list[dict[str, Any]]:
    spec = source_info(name)
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        doc = dataset[index]
        sid = sample_id(name, doc, index)
        paths: list[Path] = []
        checksums: list[str] = []
        sizes: list[dict[str, int]] = []
        for image_index, image in enumerate(image_list(name, doc)):
            suffix = "" if len(image_list(name, doc)) == 1 else f"__img{image_index:02d}"
            path = out_dir / "images" / name / f"{sid}{suffix}.{spec['image_extension']}"
            path.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not path.exists():
                if isinstance(image, dict) and image.get("bytes"):
                    path.write_bytes(image["bytes"])
                else:
                    image.convert("RGB").save(path, format="PNG")
            paths.append(path)
            checksums.append(sha256(path))
            if isinstance(image, dict) and image.get("bytes"):
                from PIL import Image
                with Image.open(io.BytesIO(image["bytes"])) as decoded:
                    size = decoded.size
            else:
                size = image.size
            sizes.append({"width": int(size[0]), "height": int(size[1])})
        prompt, instruction_chunks = prompt_for(name, doc)
        answer = str(doc.get("answer", ""))
        row = {
            "uid": f"{name}:{sid}", "sample_id": sid, "benchmark": name,
            "dataset_role": "heldout_external_multimodal_comparison", "dataset_version": "heldout_mmstar_mmmu_v1",
            **spec, "source_index": index, "prompt": prompt, "instruction_text_chunks": instruction_chunks,
            "answer": answer, "all_answer_norms": [answer], "correctness_threshold": 1.0,
            "has_answer": bool(answer and answer != "?"), "image_path": str(paths[0]),
            "image_paths": [str(path) for path in paths],
            "image_relpaths": [path.relative_to(out_dir).as_posix() for path in paths],
            "image_content_sha256s": checksums, "image_sizes": sizes, "image_count": len(paths),
            "route_or_score_status": "not_evaluated",
        }
        if name == "mmstar_val":
            row.update({"category": doc.get("category"), "l2_category": doc.get("l2_category"), "source_index": int(doc["index"])})
        else:
            row.update({"source_id": doc.get("id"), "subject": doc.get("subject", doc.get("subfield")), "options": options(doc["options"])})
        rows.append(row)
    return rows


def main() -> None:
    parsed = args()
    source_paths = source_files(parsed.datasets_cache)
    parsed.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for name, paths in source_paths.items():
        dataset = load_arrow_files(paths)
        if name == "mmmu_val":
            dataset = dataset.filter(lambda row: row.get("question_type") == "multiple-choice")
        rows = build_rows(name, dataset, parsed.out_dir, parsed.overwrite)
        print(f"[{name}] {len(rows)} rows", flush=True)
        all_rows.extend(rows)
    if len({row["uid"] for row in all_rows}) != len(all_rows):
        raise RuntimeError("duplicate UID in heldout manifest")
    samples = parsed.out_dir / "samples.jsonl"
    with samples.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    counts = Counter(row["benchmark"] for row in all_rows)
    summary = {
        "dataset_version": "heldout_mmstar_mmmu_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_count": len(all_rows), "counts_by_benchmark": dict(counts),
        "answer_available_by_benchmark": {name: sum(bool(row["has_answer"]) for row in all_rows if row["benchmark"] == name) for name in counts},
        "source_cache_revisions": {"MMStar": MMSTAR_CACHE_REV, "MMMU": MMMU_CACHE_REV, "MMMU_Pro": MMMU_PRO_CACHE_REV},
        "scoring": "Exact option-letter correctness; MMMU/MMMU-Pro and MMStar use the first standalone option letter in the generation.",
    }
    write_json(parsed.out_dir / "summary.json", summary)
    write_json(parsed.out_dir / "schema.json", {"primary_file": "samples.jsonl", "image_paths_authoritative": True, "instruction_text_chunks": "Text spans used to build the instruction-token mask for interleaved MMMU prompts."})
    write_json(parsed.out_dir / "metadata.json", {"builder": str(Path(__file__).resolve()), "datasets_cache": str(parsed.datasets_cache), "sources": {name: source_info(name) for name in source_paths}, "source_arrow_files": {name: [str(path) for path in paths] for name, paths in source_paths.items()}})
    readme = [
        "# heldout_mmstar_mmmu_v1 - Final Ver", "", "Four scored external heldout splits for paired all-on/router evaluation.", "",
        "| benchmark | source | split | rows | answer rows | options |", "|---|---|---:|---:|---:|---:|",
    ]
    for name, count in counts.items():
        sample = next(row for row in all_rows if row["benchmark"] == name)
        option_count = len(sample.get("options", [])) if sample.get("options") else (4 if name in {"mmstar_val", "mmmu_val"} else 10)
        answers = sum(bool(row["has_answer"]) for row in all_rows if row["benchmark"] == name)
        readme.append(f"| {name} | `{sample['source_dataset']}/{sample.get('source_dataset_name') or ''}` | `{sample['source_split']}` | {count} | {answers} | {option_count} |")
    readme += ["", "`samples.jsonl` is authoritative. Images are copied under `images/`; no model outputs or route labels are included.", ""]
    (parsed.out_dir / "README_final_ver.md").write_text("\n".join(readme), encoding="utf-8")
    targets = ["samples.jsonl", "summary.json", "schema.json", "metadata.json", "README_final_ver.md"]
    (parsed.out_dir / "checksums.sha256").write_text("\n".join(f"{sha256(parsed.out_dir / name)}  {name}" for name in targets) + "\n", encoding="utf-8")
    write_json(parsed.out_dir / "manifest_fingerprint.json", {"total_count": len(all_rows), "counts_by_benchmark": dict(counts), "samples_sha256": sha256(samples)})
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
