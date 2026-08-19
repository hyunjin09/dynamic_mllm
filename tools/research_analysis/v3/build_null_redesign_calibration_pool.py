from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.ipc as ipc
from PIL import Image


GQA_ROOT = Path("/data/dataset/huggingface/datasets/lmms-lab___gqa")
GQA_VERSION = Path(
    "0.0.0/a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8"
)
GQA_TRAIN = GQA_ROOT / "train_balanced_instructions" / GQA_VERSION / "gqa-train.arrow"
GQA_VALIDATION = GQA_ROOT / "val_balanced_instructions" / GQA_VERSION / "gqa-val.arrow"
TEXTVQA_ROOT = Path(
    "/data/dataset/huggingface/datasets/lmms-lab___textvqa/default/0.0.0/"
    "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
)
VG_ROOTS = (Path("/data/dataset/VG/VG_100K"), Path("/data/dataset/VG/VG_100K_2"))
IMAGE_OUTPUT_ROOT = Path(
    "/data/dataset/dynamic_mllm/v3_null_redesign/calibration_images_v1"
)
PROMPT_SUFFIX = "Answer the question using a single word or phrase."
SELECTION_SEED = 2026080701
PER_DATASET = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build answer-free v3 null calibration pool.")
    parser.add_argument(
        "--output",
        default="outputs/v3_null_redesign/calibration_pool_manifest.json",
    )
    parser.add_argument(
        "--jsonl",
        default="data_manifests/v3_null_redesign_calibration_2000_v1.jsonl",
    )
    parser.add_argument("--per-dataset", type=int, default=PER_DATASET)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    return parser.parse_args()


def arrow_rows(paths: Iterable[Path], columns: list[str]) -> Iterable[dict[str, Any]]:
    for path in sorted(paths):
        with path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            for batch in reader:
                yield from batch.select(columns).to_pylist()


def canonical_image_hash_bytes(image_bytes: bytes) -> str:
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{rgb.width}:{rgb.height}:".encode("ascii"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def canonical_image_hash_path(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{rgb.width}:{rgb.height}:".encode("ascii"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(seed: int, dataset: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{dataset}:{record_id}".encode()).hexdigest()


def read_jsonl_selected(path: Path, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                source = json.loads(line)
                rows.append({key: source.get(key) for key in keys})
    return rows


def resolve_gqa_image(image_id: str) -> Path | None:
    for root in VG_ROOTS:
        path = root / f"{image_id}.jpg"
        if path.is_file():
            return path
    return None


def inspected_exclusions() -> dict[str, set[str]]:
    result = {
        "gqa_record_ids": set(),
        "gqa_image_ids": set(),
        "textvqa_record_ids": set(),
        "textvqa_image_ids": set(),
        "textvqa_image_hashes": set(),
    }
    stage_a = read_jsonl_selected(
        Path("outputs/stage_a/stage_a_samples.jsonl"),
        ("id", "benchmark", "dataset", "source_asset_id", "local_image_path"),
    )
    stage_b = read_jsonl_selected(
        Path("data_manifests/stage_b_discovery_candidates_400.jsonl"),
        ("id", "benchmark", "dataset", "source_asset_id", "local_image_path"),
    )
    stage_c = read_jsonl_selected(
        Path("outputs/stage_c/manifest/stage_c_manifest_v1.jsonl"),
        ("id", "image_id", "image_sha256"),
    )
    for row in stage_a + stage_b:
        dataset = row.get("benchmark") or row.get("dataset")
        record_id = str(row.get("id") or "")
        asset = str(row.get("source_asset_id") or "")
        if dataset == "gqa":
            result["gqa_record_ids"].add(record_id)
            if asset:
                result["gqa_image_ids"].add(asset.split(":", 1)[-1])
        elif dataset == "textvqa":
            result["textvqa_record_ids"].add(record_id)
            if asset:
                result["textvqa_image_ids"].add(asset.split(":", 1)[-1])
            local_path = row.get("local_image_path")
            if local_path and Path(str(local_path)).is_file():
                result["textvqa_image_hashes"].add(
                    canonical_image_hash_path(Path(str(local_path)))
                )
    for row in stage_c:
        result["textvqa_record_ids"].add(str(row.get("id") or ""))
        result["textvqa_image_ids"].add(str(row.get("image_id") or ""))
        result["textvqa_image_hashes"].add(str(row.get("image_sha256") or ""))
    return result


def validation_universe() -> tuple[set[str], set[str], set[str]]:
    gqa_image_ids = {
        str(row["imageId"])
        for row in arrow_rows([GQA_VALIDATION], ["imageId"])
        if row.get("imageId")
    }
    text_ids: set[str] = set()
    text_hashes: set[str] = set()
    for row in arrow_rows(TEXTVQA_ROOT.glob("textvqa-validation-*.arrow"), ["image_id", "image"]):
        text_ids.add(str(row["image_id"]))
        text_hashes.add(canonical_image_hash_bytes(row["image"]["bytes"]))
    return gqa_image_ids, text_ids, text_hashes


def select_gqa(
    excluded: dict[str, set[str]], validation_images: set[str], count: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_image: dict[str, dict[str, Any]] = {}
    counters = Counter()
    for row in arrow_rows([GQA_TRAIN], ["id", "imageId", "question"]):
        source_id = str(row.get("id") or "")
        image_id = str(row.get("imageId") or "")
        question = str(row.get("question") or "").strip()
        record_id = f"gqa:gqa_train_{source_id}"
        if not question:
            counters["missing_question"] += 1
            continue
        if image_id in validation_images:
            counters["validation_image_overlap"] += 1
            continue
        if image_id in excluded["gqa_image_ids"] or record_id in excluded["gqa_record_ids"]:
            counters["inspected_overlap"] += 1
            continue
        image_path = resolve_gqa_image(image_id)
        if image_path is None:
            counters["image_unavailable"] += 1
            continue
        candidate = {
            "dataset": "gqa",
            "id": record_id,
            "source_id": source_id,
            "source_split": "train_balanced_instructions",
            "image_id": f"gqa:{image_id}",
            "source_image_id": image_id,
            "question": question,
            "prompt": f"{question}\n{PROMPT_SUFFIX}",
            "local_image_path": str(image_path),
            "selection_hash": rank(seed, "gqa", record_id),
        }
        previous = by_image.get(image_id)
        if previous is None or candidate["selection_hash"] < previous["selection_hash"]:
            by_image[image_id] = candidate
    selected = sorted(by_image.values(), key=lambda row: row["selection_hash"])[:count]
    if len(selected) != count:
        raise RuntimeError(f"GQA has only {len(selected)} eligible unique images")
    for row in selected:
        path = Path(row["local_image_path"])
        with Image.open(path) as image:
            image.verify()
        row["image_file_sha256"] = sha256_file(path)
        row["image_canonical_sha256"] = canonical_image_hash_path(path)
    counters["eligible_unique_images"] = len(by_image)
    counters["selected_unique_images"] = len(selected)
    return selected, dict(sorted(counters.items()))


def textvqa_candidates(
    excluded: dict[str, set[str]], validation_ids: set[str], validation_hashes: set[str], seed: int
) -> tuple[dict[str, dict[str, Any]], Counter]:
    by_image: dict[str, dict[str, Any]] = {}
    known_hashes: dict[str, str] = {}
    counters = Counter()
    paths = list(TEXTVQA_ROOT.glob("textvqa-train-*.arrow"))
    for row in arrow_rows(paths, ["image_id", "question_id", "question", "image"]):
        image_id = str(row["image_id"])
        question_id = int(row["question_id"])
        question = str(row.get("question") or "").strip()
        record_id = f"textvqa:textvqa_train_{question_id}"
        if not question:
            counters["missing_question"] += 1
            continue
        image_hash = known_hashes.get(image_id)
        if image_hash is None:
            image_hash = canonical_image_hash_bytes(row["image"]["bytes"])
            known_hashes[image_id] = image_hash
        if image_id in validation_ids or image_hash in validation_hashes:
            counters["validation_image_overlap"] += 1
            continue
        if (
            image_id in excluded["textvqa_image_ids"]
            or image_hash in excluded["textvqa_image_hashes"]
            or record_id in excluded["textvqa_record_ids"]
        ):
            counters["inspected_overlap"] += 1
            continue
        candidate = {
            "dataset": "textvqa",
            "id": record_id,
            "source_id": question_id,
            "source_split": "train",
            "image_id": f"textvqa:{image_id}",
            "source_image_id": image_id,
            "image_canonical_sha256": image_hash,
            "question": question,
            "prompt": f"{question}\n{PROMPT_SUFFIX}",
            "selection_hash": rank(seed, "textvqa", record_id),
        }
        previous = by_image.get(image_id)
        if previous is None or candidate["selection_hash"] < previous["selection_hash"]:
            by_image[image_id] = candidate
    counters["eligible_unique_images"] = len(by_image)
    return by_image, counters


def materialize_textvqa(selected: list[dict[str, Any]]) -> None:
    selected_by_image = {row["source_image_id"]: row for row in selected}
    IMAGE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    remaining = set(selected_by_image)
    for source in arrow_rows(TEXTVQA_ROOT.glob("textvqa-train-*.arrow"), ["image_id", "image"]):
        image_id = str(source["image_id"])
        if image_id not in remaining:
            continue
        image_bytes = source["image"]["bytes"]
        source_path = str(source["image"].get("path") or "")
        suffix = Path(source_path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        path = IMAGE_OUTPUT_ROOT / f"textvqa_{image_id}{suffix}"
        if path.exists():
            if sha256_file(path) != hashlib.sha256(image_bytes).hexdigest():
                raise RuntimeError(f"Existing materialized image differs: {path}")
        else:
            path.write_bytes(image_bytes)
        row = selected_by_image[image_id]
        row["local_image_path"] = str(path)
        row["image_file_sha256"] = sha256_file(path)
        if canonical_image_hash_path(path) != row["image_canonical_sha256"]:
            raise RuntimeError(f"Canonical image mismatch for {image_id}")
        remaining.remove(image_id)
        if not remaining:
            break
    if remaining:
        raise RuntimeError(f"Could not materialize {len(remaining)} selected TextVQA images")


def validate_records(records: list[dict[str, Any]], count: int) -> dict[str, Any]:
    datasets = Counter(row["dataset"] for row in records)
    image_ids = [row["image_id"] for row in records]
    record_ids = [row["id"] for row in records]
    forbidden = {"answer", "answers", "accepted_answers", "reference_answer"}
    forbidden_found = sorted({key for row in records for key in row if key in forbidden})
    if datasets != Counter({"gqa": count, "textvqa": count}):
        raise RuntimeError(f"Unexpected dataset counts: {datasets}")
    if len(set(image_ids)) != len(image_ids) or len(set(record_ids)) != len(record_ids):
        raise RuntimeError("Calibration record/image IDs are not unique")
    if forbidden_found:
        raise RuntimeError(f"Answer fields leaked into manifest: {forbidden_found}")
    for row in records:
        path = Path(row["local_image_path"])
        if not path.is_file() or not row["prompt"].endswith(PROMPT_SUFFIX):
            raise RuntimeError(f"Invalid calibration record: {row['id']}")
    return {
        "record_count": len(records),
        "unique_image_count": len(set(image_ids)),
        "dataset_counts": dict(sorted(datasets.items())),
        "answer_fields_present": forbidden_found,
    }


def execute(args: argparse.Namespace) -> None:
    if args.per_dataset < 1:
        raise ValueError("--per-dataset must be positive")
    excluded = inspected_exclusions()
    gqa_validation, text_validation_ids, text_validation_hashes = validation_universe()
    gqa, gqa_counts = select_gqa(excluded, gqa_validation, args.per_dataset, args.seed)
    text_candidates, text_counts = textvqa_candidates(
        excluded, text_validation_ids, text_validation_hashes, args.seed
    )
    textvqa = sorted(text_candidates.values(), key=lambda row: row["selection_hash"])[
        : args.per_dataset
    ]
    if len(textvqa) != args.per_dataset:
        raise RuntimeError(f"TextVQA has only {len(textvqa)} eligible unique images")
    materialize_textvqa(textvqa)
    text_counts["selected_unique_images"] = len(textvqa)
    records = sorted(gqa + textvqa, key=lambda row: (row["dataset"], row["selection_hash"]))
    validation = validate_records(records, args.per_dataset)

    jsonl_path = Path(args.jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    source_paths = [GQA_TRAIN, GQA_VALIDATION, *sorted(TEXTVQA_ROOT.glob("textvqa-train-*.arrow")), *sorted(TEXTVQA_ROOT.glob("textvqa-validation-*.arrow"))]
    payload = {
        "schema_version": "v3_null_redesign_calibration_pool_v1",
        "purpose": "independent answer-free READ/WRITE residual geometry calibration",
        "outcome_blind": True,
        "reference_answers_loaded_into_scoring": False,
        "terminal_action_values_computed": False,
        "selection": {
            "seed": args.seed,
            "target_per_dataset": args.per_dataset,
            "rule": "one minimum SHA256-ranked train question per image, then minimum ranked unique images",
            "rank_input": "SHA256(seed:dataset:record_id)",
        },
        "independence": {
            "source_splits": {"gqa": "train_balanced_instructions", "textvqa": "train"},
            "entire_validation_universe_reserved": True,
            "excluded_sets": [
                "v2/v3 Stage B discovery",
                "v2 Stage C",
                "v3 candidate confirmation pools",
                "reserved Stage C2 pools",
            ],
            "gqa_validation_image_count": len(gqa_validation),
            "textvqa_validation_image_id_count": len(text_validation_ids),
            "textvqa_validation_image_hash_count": len(text_validation_hashes),
        },
        "audit": {
            "gqa": gqa_counts,
            "textvqa": dict(sorted(text_counts.items())),
            "validation": validation,
        },
        "source_files": {str(path): sha256_file(path) for path in source_paths},
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": sha256_file(jsonl_path),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "jsonl": str(jsonl_path), **validation}, sort_keys=True))


if __name__ == "__main__":
    execute(parse_args())
