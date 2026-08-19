from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.ipc as ipc

from scoring.benchmark_metrics import normalize_textvqa


PREVIEW = Path("outputs/v3_preflight/candidate_pool_audit.json")
GQA_ARROW = Path(
    "/data/dataset/huggingface/datasets/lmms-lab___gqa/val_balanced_instructions/"
    "0.0.0/a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8/gqa-val.arrow"
)
TEXTVQA_ROOT = Path(
    "/data/dataset/huggingface/datasets/lmms-lab___textvqa/default/0.0.0/"
    "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
)
GQA_SCENES = Path("/data/dataset/GQA/sceneGraphs_v1.1/val_sceneGraphs.json")
TEXTOCR = Path("/data/dataset/TextOCR/annotations_v0.1/TextOCR_0.1_val.json")
OUTPUT = Path("outputs/v3_preflight/grounding_eligibility_audit_v1.json")
MINIMUM_SUBSET = 100


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrow_rows(paths: Iterable[Path], columns: list[str]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("rb") as handle:
            for batch in ipc.open_stream(handle):
                for row in batch.select(columns).to_pylist():
                    yield row


def box_iou(first: list[float], second: list[float]) -> float:
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def matched_control(
    target: list[float],
    candidates: list[tuple[str, list[float], str]],
    width: float,
    height: float,
) -> dict[str, Any] | None:
    tx, ty, tw, th = target
    if min(tw, th) <= 0:
        return None
    target_area = tw * th
    rows = []
    for identifier, box, label in candidates:
        x, y, w, h = box
        if min(w, h) <= 0:
            continue
        area_ratio = max(target_area / (w * h), (w * h) / target_area)
        if area_ratio > 1.25 or box_iou(target, box) > 0.05:
            continue
        cx, cy = x + w / 2, y + h / 2
        control = [cx - tw / 2, cy - th / 2, tw, th]
        if control[0] < 0 or control[1] < 0 or control[0] + tw > width or control[1] + th > height:
            continue
        if box_iou(target, control) > 0.05:
            continue
        aspect_ratio = max((tw / th) / (w / h), (w / h) / (tw / th))
        rows.append((math.log(area_ratio) ** 2 + math.log(aspect_ratio) ** 2, identifier, control, label, box))
    if not rows:
        return None
    rows.sort(key=lambda item: (item[0], item[1]))
    score, identifier, control, label, source_box = rows[0]
    return {
        "source_region_id": identifier,
        "source_region_label": label,
        "source_region_box": source_box,
        "equal_area_control_box": control,
        "match_score": score,
    }


def audit_gqa(preview_ids: set[str]) -> dict[str, Any]:
    scenes = json.loads(GQA_SCENES.read_text(encoding="utf-8"))
    wanted = {item.removeprefix("gqa:gqa_val_") for item in preview_ids}
    reasons = Counter()
    eligible = []
    rows = arrow_rows(
        [GQA_ARROW],
        ["id", "imageId", "question", "answer", "annotations", "semantic", "semanticStr"],
    )
    for row in rows:
        if str(row["id"]) not in wanted:
            continue
        object_ids = set()
        for values in row["annotations"].values():
            for item in values:
                # The Arrow adapter stores the original annotation-map key
                # (a token span such as "2" or "1:3") in objectId and the
                # referenced GQA scene-graph object identifier in value.
                if item.get("value") and str(item["value"]).isdigit():
                    object_ids.add(str(item["value"]))
        for step in row.get("semantic") or []:
            object_ids.update(re.findall(r"\((\d+)\)", str(step.get("argument", ""))))
        if len(object_ids) != 1:
            reasons["not_exactly_one_referenced_object"] += 1
            continue
        image_id = str(row["imageId"])
        scene = scenes.get(image_id)
        target_id = next(iter(object_ids))
        if scene is None or target_id not in scene.get("objects", {}):
            reasons["referenced_object_missing_from_scene_graph"] += 1
            continue
        target_object = scene["objects"][target_id]
        target = [float(target_object[key]) for key in ("x", "y", "w", "h")]
        candidates = [
            (
                str(identifier),
                [float(value[key]) for key in ("x", "y", "w", "h")],
                str(value.get("name", "")),
            )
            for identifier, value in scene["objects"].items()
            if str(identifier) != target_id
        ]
        control = matched_control(
            target, candidates, float(scene["width"]), float(scene["height"])
        )
        if control is None:
            reasons["no_geometrically_matched_nontarget"] += 1
            continue
        eligible.append(
            {
                "id": f"gqa:gqa_val_{row['id']}",
                "image_id": image_id,
                "target_object_id": target_id,
                "target_label": target_object.get("name"),
                "target_box": target,
                "control": control,
            }
        )
    reasons["eligible"] = len(eligible)
    return {
        "candidate_count": len(preview_ids),
        "counts": dict(reasons),
        "eligible_count": len(eligible),
        "minimum_required": MINIMUM_SUBSET,
        "sufficient": len(eligible) >= MINIMUM_SUBSET,
        "eligible": eligible,
    }


def audit_textvqa(preview_ids: set[str]) -> dict[str, Any]:
    textocr = json.loads(TEXTOCR.read_text(encoding="utf-8"))
    wanted = {int(item.removeprefix("textvqa:textvqa_validation_")) for item in preview_ids}
    reasons = Counter()
    eligible = []
    paths = sorted(TEXTVQA_ROOT.glob("textvqa-validation-*.arrow"))
    rows = arrow_rows(paths, ["image_id", "question_id", "question", "answers"])
    for row in rows:
        question_id = int(row["question_id"])
        if question_id not in wanted:
            continue
        image_id = str(row["image_id"])
        image = textocr["imgs"].get(image_id)
        if image is None:
            reasons["image_missing_from_textocr"] += 1
            continue
        counts = Counter(normalize_textvqa(answer) for answer in row["answers"])
        accepted = {answer for answer, count in counts.items() if answer and count >= 3}
        if not accepted:
            reasons["no_frequency_three_normalized_answer"] += 1
            continue
        annotation_ids = textocr["imgToAnns"].get(image_id, [])
        matches = [
            identifier
            for identifier in annotation_ids
            if normalize_textvqa(textocr["anns"][identifier]["utf8_string"]) in accepted
        ]
        if len(matches) != 1:
            reasons["not_exactly_one_matching_ocr_box"] += 1
            continue
        target_id = matches[0]
        target_ann = textocr["anns"][target_id]
        target = [float(item) for item in target_ann["bbox"]]
        candidates = [
            (
                identifier,
                [float(item) for item in textocr["anns"][identifier]["bbox"]],
                str(textocr["anns"][identifier]["utf8_string"]),
            )
            for identifier in annotation_ids
            if identifier != target_id
            and normalize_textvqa(textocr["anns"][identifier]["utf8_string"]) not in accepted
        ]
        control = matched_control(
            target, candidates, float(image["width"]), float(image["height"])
        )
        if control is None:
            reasons["no_geometrically_matched_nontarget"] += 1
            continue
        eligible.append(
            {
                "id": f"textvqa:textvqa_validation_{question_id}",
                "image_id": image_id,
                "normalized_accepted_answers_frequency_at_least_three": sorted(accepted),
                "target_ocr_id": target_id,
                "target_text": target_ann["utf8_string"],
                "target_box": target,
                "control": control,
            }
        )
    reasons["eligible"] = len(eligible)
    return {
        "candidate_count": len(preview_ids),
        "counts": dict(reasons),
        "eligible_count": len(eligible),
        "minimum_required": MINIMUM_SUBSET,
        "sufficient": len(eligible) >= MINIMUM_SUBSET,
        "eligible": eligible,
    }


def main() -> None:
    preview = json.loads(PREVIEW.read_text(encoding="utf-8"))[
        "proposed_manifest_identity_preview"
    ]
    payload = {
        "schema_version": "v3_grounding_eligibility_audit_v1",
        "outcome_blind": True,
        "terminal_action_values_loaded_or_computed": False,
        "not_a_frozen_stage_c_manifest": True,
        "rules": {
            "minimum_subset_per_dataset": MINIMUM_SUBSET,
            "max_source_area_ratio": 1.25,
            "max_target_overlap_iou": 0.05,
            "control_region": "target-sized rectangle centered on the matched non-target annotation",
        },
        "sources": {
            str(PREVIEW): sha256(PREVIEW),
            str(GQA_ARROW): sha256(GQA_ARROW),
            str(GQA_SCENES): sha256(GQA_SCENES),
            str(TEXTOCR): sha256(TEXTOCR),
            **{str(path): sha256(path) for path in sorted(TEXTVQA_ROOT.glob("textvqa-validation-*.arrow"))},
        },
        "gqa": audit_gqa({row["id"] for row in preview["gqa"]}),
        "textvqa": audit_textvqa({row["id"] for row in preview["textvqa"]}),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256(OUTPUT),
                "gqa_eligible": payload["gqa"]["eligible_count"],
                "textvqa_eligible": payload["textvqa"]["eligible_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
