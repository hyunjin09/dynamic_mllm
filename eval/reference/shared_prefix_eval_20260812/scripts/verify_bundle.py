#!/usr/bin/env python3
"""Validate the portable shared-prefix evaluation bundle."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_EXPECTED_COUNTS = {
    "mmstar_val": 1500,
    "mmmu_val": 847,
    "mmmu_pro_standard_test": 1730,
    "mmmu_pro_vision_test": 1730,
}
CORE_EXPECTED_COUNTS = {
    "chartqa": 2500,
    "textvqa": 5000,
    "docvqa": 5349,
}
POPE_EXPECTED_COUNTS = {
    "pope_adversarial": 3000,
    "pope_popular": 3000,
    "pope_random": 3000,
}
EXPECTED_SHA256 = {
    "checkpoints/sw31/router_epoch_001.pt": "6ecf2f9119b78d5d11c969b4602b93cecc59d27aab43440abacb84421c4af255",
    "checkpoints/prefix_admission/prefix_admission_selection.pt": "3a1385420bc9abae7b596ecf967a34ea0a9ce86c636a99140c6be2c25ed07ef4",
    "data/heldout_mmstar_mmmu_final_v2/samples.jsonl": "717af22ceb0438ff609af8df5193646b48e7ae59e175683c0952d3be4d1409e2",
    "baseline/all_on_generation_rows.jsonl": "385eb60598ca294af326e66e8bc6c9f8f3845650797fed333a756390d916449b",
    "data/heldout_lmms_recommended_v1/samples.jsonl": "0ad5718d3ade02b7d812b023772adfdbe8bc811cd72e6970ce4f8773a75b34cc",
    "baseline/core_vqa_all_on_generation_rows.jsonl": "a52997cc4947d3ba932127c3fb88182461554ea6dfee64e129e0df41d8f66b8b",
    "results/reference_core_vqa/heldout_generation_rows.jsonl": "ded7085387ac8358d33351ec02ba0e15d71b9f9d3bb472b3421ca2654a533f05",
    "data/heldout_pope_v1/samples.jsonl": "1a39be02d9d031715080908c284f6444b3bf3772e7e3a162c642caf7eb30695c",
    "baseline/pope_all_on_generation_rows.jsonl": "b859d7851cd3eef2b738764f18ae82a9938eafa9408a83e7e3a7f028788b9562",
    "results/reference_pope/heldout_generation_rows.jsonl": "238db9a7b0205319c686870b10cdc22eaa4a636f8b27c76b24b8102019de0c97",
}
EXPECTED_MODEL_BLOBS = {
    "model-00001-of-00005.safetensors": "e97b877e47fde53a6c6e77aafb36e58e91ee9d95c4a3eeac6f1b5c0e6a1c986e",
    "model-00002-of-00005.safetensors": "a9a300a43b4724eee2abe7c18ceb26768d0ab011eb0cad19d9bfd2476a24d024",
    "model-00003-of-00005.safetensors": "111223d173e00bbee81cba1216fad28668df3476706b7fd26f4d5b50f8b3a507",
    "model-00004-of-00005.safetensors": "ef47f634fa57d46ee134edcc09f34085a47da1e16c12a2abe0d67118be6d72ed",
    "model-00005-of-00005.safetensors": "0c859795ad3a627a9b95bcb762e059d5b768a4a36fdd4affeff269d93fdecc67",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_option(text: str) -> str:
    match = re.search(r"\b([A-J])\b", str(text).upper())
    return match.group(1) if match else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-images", action="store_true")
    parser.add_argument("--full-model", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing required file: {relative}")
        elif sha256(path) != expected:
            failures.append(f"sha256 mismatch: {relative}")

    manifest_path = ROOT / "data/heldout_mmstar_mmmu_final_v2/samples.jsonl"
    rows = read_jsonl(manifest_path)
    uids = [str(row["uid"]) for row in rows]
    if len(rows) != 5807 or len(set(uids)) != 5807:
        failures.append(f"manifest UID contract failed: rows={len(rows)} unique={len(set(uids))}")
    counts = Counter(str(row["benchmark"]) for row in rows)
    if dict(counts) != EXTERNAL_EXPECTED_COUNTS:
        failures.append(f"benchmark counts differ: {dict(counts)}")
    if any(float(row["correctness_threshold"]) != 1.0 for row in rows):
        failures.append("not every correctness threshold is 1.0")
    if any(str(row["metric_name"]) not in {"mmstar_choice_accuracy", "mmmu_acc"} for row in rows):
        failures.append("unexpected metric_name in manifest")

    referenced_images = 0
    for row in rows:
        relpaths = row.get("image_relpaths") or []
        checksums = row.get("image_content_sha256s") or []
        if len(relpaths) != int(row["image_count"]) or len(checksums) != len(relpaths):
            failures.append(f"image metadata count mismatch: {row['uid']}")
            continue
        referenced_images += len(relpaths)
        for relative, expected in zip(relpaths, checksums):
            image = ROOT / "data/heldout_mmstar_mmmu_final_v2" / relative
            if not image.is_file():
                failures.append(f"missing image: {relative}")
            elif args.full_images and sha256(image) != expected:
                failures.append(f"image sha256 mismatch: {relative}")
    if referenced_images != 6173:
        failures.append(f"expected 6173 image references, found {referenced_images}")

    baseline_rows = read_jsonl(ROOT / "baseline/all_on_generation_rows.jsonl")
    baseline = {str(row["uid"]): row for row in baseline_rows}
    if len(baseline_rows) != 5807 or len(baseline) != 5807 or set(baseline) != set(uids):
        failures.append("baseline cache is not a one-to-one match with the manifest")
    else:
        for row in rows:
            cached = baseline[str(row["uid"])]
            score = float(first_option(cached["baseline_prediction"]) == str(row["answer"]).strip().upper())
            correct = score >= float(row["correctness_threshold"])
            if score != float(cached["baseline_score"]) or correct != bool(cached["baseline_correct"]):
                failures.append(f"baseline rescoring mismatch: {row['uid']}")
                if len(failures) >= 20:
                    break

    core_manifest_path = ROOT / "data/heldout_lmms_recommended_v1/samples.jsonl"
    core_rows = read_jsonl(core_manifest_path)
    core_uids = [str(row["uid"]) for row in core_rows]
    core_counts = Counter(str(row["benchmark"]) for row in core_rows)
    if len(core_rows) != 12849 or len(set(core_uids)) != 12849:
        failures.append(
            f"core VQA UID contract failed: rows={len(core_rows)} unique={len(set(core_uids))}"
        )
    if dict(core_counts) != CORE_EXPECTED_COUNTS:
        failures.append(f"core VQA benchmark counts differ: {dict(core_counts)}")
    expected_metrics = {
        "chartqa": ("relaxed_accuracy", 1.0),
        "textvqa": ("textvqa_evalai_consensus", 0.5),
        "docvqa": ("anls", 0.5),
    }
    core_image_refs = 0
    for row in core_rows:
        expected_metric, expected_threshold = expected_metrics[str(row["benchmark"])]
        if str(row["metric_name"]) != expected_metric:
            failures.append(f"core VQA metric mismatch: {row['uid']}")
        if float(row["correctness_threshold"]) != expected_threshold:
            failures.append(f"core VQA threshold mismatch: {row['uid']}")
        relpaths = row.get("image_relpaths") or [row.get("image_relpath")]
        checksums = row.get("image_content_sha256s") or [row.get("image_content_sha256")]
        if len(relpaths) != len(checksums) or not relpaths:
            failures.append(f"core VQA image metadata mismatch: {row['uid']}")
            continue
        core_image_refs += len(relpaths)
        for relative, expected in zip(relpaths, checksums):
            image = ROOT / "data/heldout_lmms_recommended_v1" / str(relative)
            if not image.is_file():
                failures.append(f"missing core VQA image: {relative}")
            elif args.full_images and sha256(image) != str(expected):
                failures.append(f"core VQA image sha256 mismatch: {relative}")
    if core_image_refs != 12849:
        failures.append(f"expected 12849 core VQA image references, found {core_image_refs}")

    core_baseline_rows = read_jsonl(ROOT / "baseline/core_vqa_all_on_generation_rows.jsonl")
    core_baseline = {str(row["uid"]): row for row in core_baseline_rows}
    if (
        len(core_baseline_rows) != 12849
        or len(core_baseline) != 12849
        or set(core_baseline) != set(core_uids)
    ):
        failures.append("core VQA baseline cache is not a one-to-one manifest match")
    else:
        sys.path.insert(0, str(ROOT / "code"))
        from dvr_qwen.eval_metrics import score_prediction

        for row in core_rows:
            cached = core_baseline[str(row["uid"])]
            score = float(
                score_prediction(
                    str(row["metric_name"]),
                    str(cached["baseline_prediction"]),
                    row.get("answer"),
                    row.get("all_answer_norms"),
                )
            )
            correct = score >= float(row["correctness_threshold"])
            if abs(score - float(cached["baseline_score"])) > 1e-12 or correct != bool(
                cached["baseline_correct"]
            ):
                failures.append(f"core VQA baseline rescoring mismatch: {row['uid']}")
                if len(failures) >= 20:
                    break

    core_reference_rows = read_jsonl(
        ROOT / "results/reference_core_vqa/heldout_generation_rows.jsonl"
    )
    alignment = json.loads(
        (ROOT / "results/reference_core_vqa/source_manifest_alignment.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        not bool(alignment.get("matched"))
        or int(alignment.get("rows", 0)) != 12849
        or str(alignment.get("portable_manifest_sha256"))
        != EXPECTED_SHA256["data/heldout_lmms_recommended_v1/samples.jsonl"]
    ):
        failures.append("core VQA source-manifest alignment audit failed")
    core_reference = {str(row["uid"]): row for row in core_reference_rows}
    if len(core_reference_rows) != 12849 or set(core_reference) != set(core_uids):
        failures.append("core VQA reference rows are not a one-to-one manifest match")
    elif any(len(row.get("selected_visual_on_mask") or []) != 28 for row in core_reference_rows):
        failures.append("core VQA reference contains a non-28-layer route")
    else:
        for row in core_rows:
            reference = core_reference[str(row["uid"])]
            router_score = float(
                score_prediction(
                    str(row["metric_name"]),
                    str(reference["router_prediction"]),
                    row.get("answer"),
                    row.get("all_answer_norms"),
                )
            )
            router_correct = router_score >= float(row["correctness_threshold"])
            if abs(router_score - float(reference["router_score"])) > 1e-12 or router_correct != bool(
                reference["router_correct"]
            ):
                failures.append(f"core VQA router rescoring mismatch: {row['uid']}")
                if len(failures) >= 20:
                    break

    pope_rows = read_jsonl(ROOT / "data/heldout_pope_v1/samples.jsonl")
    pope_uids = [str(row["uid"]) for row in pope_rows]
    pope_counts = Counter(str(row["benchmark"]) for row in pope_rows)
    if len(pope_rows) != 9000 or len(set(pope_uids)) != 9000:
        failures.append(f"POPE UID contract failed: rows={len(pope_rows)} unique={len(set(pope_uids))}")
    if dict(pope_counts) != POPE_EXPECTED_COUNTS:
        failures.append(f"POPE split counts differ: {dict(pope_counts)}")
    pope_image_refs = 0
    pope_image_hashes: set[str] = set()
    for row in pope_rows:
        if str(row.get("metric_name")) != "pope_yes_no_accuracy":
            failures.append(f"POPE metric mismatch: {row['uid']}")
        if float(row.get("correctness_threshold", -1)) != 1.0:
            failures.append(f"POPE threshold mismatch: {row['uid']}")
        if int(row.get("max_new_tokens", 0)) != 128:
            failures.append(f"POPE max_new_tokens mismatch: {row['uid']}")
        relpaths = row.get("image_relpaths") or [row.get("image_relpath")]
        checksums = row.get("image_content_sha256s") or [row.get("image_content_sha256")]
        if len(relpaths) != len(checksums) or not relpaths:
            failures.append(f"POPE image metadata mismatch: {row['uid']}")
            continue
        pope_image_refs += len(relpaths)
        for relative, expected in zip(relpaths, checksums):
            pope_image_hashes.add(str(expected))
            image = ROOT / "data/heldout_pope_v1" / str(relative)
            if not image.is_file():
                failures.append(f"missing POPE image: {relative}")
            elif args.full_images and sha256(image) != str(expected):
                failures.append(f"POPE image sha256 mismatch: {relative}")
    if pope_image_refs != 9000 or len(pope_image_hashes) != 500:
        failures.append(
            f"POPE image contract failed: refs={pope_image_refs}, unique_hashes={len(pope_image_hashes)}"
        )

    pope_baseline_rows = read_jsonl(ROOT / "baseline/pope_all_on_generation_rows.jsonl")
    pope_baseline = {str(row["uid"]): row for row in pope_baseline_rows}
    pope_reference_rows = read_jsonl(ROOT / "results/reference_pope/heldout_generation_rows.jsonl")
    pope_reference = {str(row["uid"]): row for row in pope_reference_rows}
    if len(pope_baseline_rows) != 9000 or set(pope_baseline) != set(pope_uids):
        failures.append("POPE baseline cache is not a one-to-one manifest match")
    if len(pope_reference_rows) != 9000 or set(pope_reference) != set(pope_uids):
        failures.append("POPE reference rows are not a one-to-one manifest match")
    elif any(len(row.get("selected_visual_on_mask") or []) != 28 for row in pope_reference_rows):
        failures.append("POPE reference contains a non-28-layer route")
    else:
        sys.path.insert(0, str(ROOT / "code"))
        from dvr_qwen.eval_metrics import score_prediction

        for row in pope_rows:
            baseline = pope_baseline[str(row["uid"])]
            reference = pope_reference[str(row["uid"])]
            for prefix, prediction, stored_score, stored_correct in (
                (
                    "baseline",
                    baseline["baseline_prediction"],
                    baseline["baseline_score"],
                    baseline["baseline_correct"],
                ),
                (
                    "router",
                    reference["router_prediction"],
                    reference["router_score"],
                    reference["router_correct"],
                ),
            ):
                score = float(
                    score_prediction(
                        "pope_yes_no_accuracy",
                        str(prediction),
                        row.get("answer"),
                        row.get("all_answer_norms"),
                    )
                )
                correct = score >= 1.0
                if score != float(stored_score) or correct != bool(stored_correct):
                    failures.append(f"POPE {prefix} rescoring mismatch: {row['uid']}")
                    break
            if len(failures) >= 20:
                break

    model_dir = ROOT / "model/Qwen2.5-VL-7B-Instruct_cc594898137f460bfe9f0759e9844b3ce807cfb5"
    for name, expected in EXPECTED_MODEL_BLOBS.items():
        path = model_dir / name
        if not path.is_file():
            failures.append(f"missing model shard: {name}")
        elif args.full_model and sha256(path) != expected:
            failures.append(f"model shard sha256 mismatch: {name}")

    if failures:
        print("BUNDLE VALIDATION FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(
        "BUNDLE VALIDATION PASSED: "
        f"external=5807 UIDs/{referenced_images} image refs, "
        f"core_vqa=12849 UIDs/{core_image_refs} image refs, "
        f"pope=9000 UIDs/{pope_image_refs} image refs, "
        f"external_counts={dict(counts)}, core_counts={dict(core_counts)}, "
        f"pope_counts={dict(pope_counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
