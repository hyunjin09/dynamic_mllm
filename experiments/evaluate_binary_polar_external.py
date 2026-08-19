#!/usr/bin/env python3
"""Evaluate frozen full10 static binary predictors on the reference bundle."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import yaml

from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from experiments.train_binary_polar import file_sha256


ACTIVE_BENCHMARKS = (
    "chartqa",
    "textvqa",
    "mmstar_val",
    "mmmu_val",
    "mmmu_pro_standard_test",
    "mmmu_pro_vision_test",
    "pope_adversarial",
    "pope_popular",
    "pope_random",
)
CORE_BENCHMARKS = {"chartqa", "textvqa"}
MC_BENCHMARKS = {
    "mmstar_val",
    "mmmu_val",
    "mmmu_pro_standard_test",
    "mmmu_pro_vision_test",
}
POPE_BENCHMARKS = {"pope_adversarial", "pope_popular", "pope_random"}
OPTION_SUFFIX = re.compile(r"\s*Answer with the option letter only\.\s*$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def active_benchmark(name: str) -> bool:
    return str(name).lower() in ACTIVE_BENCHMARKS


def predictor_text(row: dict[str, Any]) -> str:
    benchmark = str(row["benchmark"]).lower()
    if benchmark in CORE_BENCHMARKS or benchmark in POPE_BENCHMARKS:
        value = str(row.get("question") or "").strip()
    elif benchmark in MC_BENCHMARKS:
        chunks = row.get("instruction_text_chunks") or []
        value = OPTION_SUFFIX.sub("", "".join(str(chunk) for chunk in chunks)).strip()
    else:
        raise ValueError(f"inactive benchmark: {benchmark}")
    if not value:
        raise ValueError(f"predictor input is empty for {row.get('uid')}")
    return value


def select_shard(
    rows: list[dict[str, Any]], *, num_shards: int, shard_index: int
) -> list[dict[str, Any]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def mask_statistics(mask: list[int]) -> dict[str, Any]:
    values = [int(value) for value in mask]
    if not values or any(value not in (0, 1) for value in values):
        raise ValueError("mask must be a nonempty binary sequence")
    return {
        "mask_key": "".join(map(str, values)),
        "num_visual_on_layers": sum(values),
        "transition_count": sum(left != right for left, right in zip(values, values[1:])),
    }


def cluster_key(row: dict[str, Any]) -> str:
    hashes = row.get("image_content_sha256s")
    if hashes:
        return "|".join(str(value) for value in hashes)
    value = row.get("image_content_sha256")
    if not value:
        raise ValueError(f"record lacks an image hash: {row.get('uid')}")
    return str(value)


def load_active_rows(bundle: Path) -> list[dict[str, Any]]:
    sources = (
        bundle / "data/heldout_lmms_recommended_v1/samples.jsonl",
        bundle / "data/heldout_mmstar_mmmu_final_v2/samples.jsonl",
        bundle / "data/heldout_pope_v1/samples.jsonl",
    )
    rows = []
    for source in sources:
        data_root = source.parent
        for row in read_jsonl(source):
            if active_benchmark(row["benchmark"]):
                current = dict(row)
                current["data_root"] = str(data_root)
                current["cluster_key"] = cluster_key(row)
                current["predictor_text"] = predictor_text(row)
                rows.append(current)
    counts = Counter(str(row["benchmark"]).lower() for row in rows)
    expected = {
        "chartqa": 2500,
        "textvqa": 5000,
        "mmstar_val": 1500,
        "mmmu_val": 847,
        "mmmu_pro_standard_test": 1730,
        "mmmu_pro_vision_test": 1730,
        "pope_adversarial": 3000,
        "pope_popular": 3000,
        "pope_random": 3000,
    }
    if counts != Counter(expected) or len(rows) != 22307:
        raise RuntimeError(f"active evaluation population mismatch: {counts}")
    uids = [str(row["uid"]) for row in rows]
    if len(uids) != len(set(uids)):
        raise RuntimeError("active evaluation UIDs are not unique")
    return rows


def baseline_index(bundle: Path) -> dict[str, dict[str, Any]]:
    paths = (
        bundle / "baseline/core_vqa_all_on_generation_rows.jsonl",
        bundle / "baseline/all_on_generation_rows.jsonl",
        bundle / "baseline/pope_all_on_generation_rows.jsonl",
    )
    result = {}
    for path in paths:
        for row in read_jsonl(path):
            uid = str(row["uid"])
            if uid in result:
                raise RuntimeError(f"duplicate baseline UID: {uid}")
            result[uid] = row
    return result


def checkpoint_architecture(config: dict[str, Any], *, image: bool) -> dict[str, Any]:
    return {
        "num_layers": int(config["policy"]["num_layers"]),
        "d_model": int(config["predictor"]["d_model"]),
        "num_heads": int(config["predictor"]["num_heads"]),
        "num_layer_blocks": int(config["predictor"]["num_layer_blocks"]),
        "dropout": float(config["predictor"]["dropout"]),
        **(
            {"image_dim": int(config["visual_features"]["feature_width"])}
            if image
            else {}
        ),
    }


def native_generation_inputs(
    inputs: dict[str, Any], *, device: torch.device | None = None
) -> dict[str, Any]:
    excluded = {"instruction_token_mask"}
    return {
        key: value.to(device) if device is not None and torch.is_tensor(value) else value
        for key, value in inputs.items()
        if key not in excluded
    }


def decode_generated(processor: Any, token_ids: torch.Tensor) -> str:
    return processor.batch_decode(
        token_ids.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def score_prediction(metric_name: str, prediction: str, answer: Any, norms: Any) -> float:
    from dvr_qwen.eval_metrics import score_prediction as bundle_score

    return float(bundle_score(metric_name, prediction, answer, norms))


def choose_preflight_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    chosen = []
    for benchmark in ACTIVE_BENCHMARKS:
        candidates = [row for row in rows if row["benchmark"] == benchmark]
        candidates.sort(
            key=lambda row: sha256(f"{seed}:external-preflight:{row['uid']}".encode()).hexdigest()
        )
        chosen.append(candidates[0])
    return chosen


def load_predictors(
    config: dict[str, Any],
    question_checkpoint: Path,
    image_question_checkpoint: Path,
    device: torch.device,
    modalities: tuple[str, ...] = ("question", "image_question"),
):
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_path, padding_side="left", local_files_only=True
    )
    encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device).eval()
    common = {"input_dim": encoder.output_dim}
    predictors = {}
    if "question" in modalities:
        question = BinaryPolarBackbone(
            **common, **checkpoint_architecture(config, image=False)
        ).to(device).eval()
        payload = torch.load(question_checkpoint, map_location="cpu", weights_only=False)
        question.load_state_dict(payload["predictor"], strict=True)
        predictors["question"] = question
    if "image_question" in modalities:
        image_question = BinaryPolarBackbone(
            **common, **checkpoint_architecture(config, image=True)
        ).to(device).eval()
        payload = torch.load(
            image_question_checkpoint, map_location="cpu", weights_only=False
        )
        image_question.load_state_dict(payload["predictor"], strict=True)
        predictors["image_question"] = image_question
    return tokenizer, encoder, predictors


@torch.inference_mode()
def predict_masks(
    *,
    text: str,
    visual_rows: torch.Tensor,
    tokenizer: Any,
    encoder: FrozenHFTokenEncoder,
    predictors: dict[str, BinaryPolarBackbone],
    max_question_tokens: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_question_tokens,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention = encoded["attention_mask"].to(device)
    token_features = encoder(input_ids, attention)
    image = visual_rows.to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    image_attention = torch.ones(image.shape[:2], dtype=torch.bool, device=device)
    logits_by_name = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if "question" in predictors:
            logits_by_name["question"] = predictors["question"](token_features, attention)
        if "image_question" in predictors:
            logits_by_name["image_question"] = predictors["image_question"](
                token_features, attention, image, image_attention
            )
    output = {}
    for name, logits in logits_by_name.items():
        mask = (logits[0] >= 0).to(torch.int64).cpu().tolist()
        output[name] = {"mask": mask, "logits": logits[0].float().cpu().tolist()}
    return output


@torch.inference_mode()
def execute_mask(
    *, model: Any, processor: Any, inputs: dict[str, Any], prepared: Any,
    mask: list[int], row: dict[str, Any], eos_token_ids: list[int],
    repetition_penalty: float,
) -> dict[str, Any]:
    from dvr_qwen.binary_generate import binary_dvrc_greedy_generate

    device = prepared.text_states.device
    route = torch.tensor([mask], dtype=torch.bool, device=device)
    output = binary_dvrc_greedy_generate(
        model,
        inputs,
        visual_on_mask=route,
        max_new_tokens=int(row["max_new_tokens"]),
        eos_token_ids=eos_token_ids,
        stop_on_eos=True,
        repetition_penalty=repetition_penalty,
        prepared_binary_inputs=prepared,
    )
    prediction = decode_generated(processor, output.generated_ids)
    score = score_prediction(
        str(row["metric_name"]), prediction, row.get("answer"), row.get("all_answer_norms")
    )
    threshold = float(row["correctness_threshold"])
    return {
        "generated_ids": output.generated_ids.detach().cpu().view(-1).tolist(),
        "prediction": prediction,
        "score": score,
        "correct": bool(score >= threshold),
        "execution_source": "live_binary_executor",
    }


def process_row(
    *, row: dict[str, Any], baseline: dict[str, Any], model: Any, processor: Any,
    tokenizer: Any, encoder: Any, predictors: dict[str, Any],
    max_question_tokens: int, device: torch.device, eos_token_ids: list[int],
    repetition_penalty: float,
) -> dict[str, Any]:
    from dvr_qwen.binary_generate import prepare_binary_dvrc_inputs
    from dvr_qwen.scripts.cache_preference_gt_router_features import build_processor_inputs

    inputs = build_processor_inputs(processor, row, data_root=Path(row["data_root"]))
    prepared = prepare_binary_dvrc_inputs(model, inputs)
    visual = prepared.visual_states[0, prepared.visual_valid_mask[0]]
    predictions = predict_masks(
        text=row["predictor_text"],
        visual_rows=visual,
        tokenizer=tokenizer,
        encoder=encoder,
        predictors=predictors,
        max_question_tokens=max_question_tokens,
        device=device,
    )
    executions = {}
    execution_cache = {}
    for modality in predictors:
        mask = predictions[modality]["mask"]
        key = "".join(map(str, mask))
        if key not in execution_cache:
            execution_cache[key] = execute_mask(
                model=model,
                processor=processor,
                inputs=inputs,
                prepared=prepared,
                mask=mask,
                row=row,
                eos_token_ids=eos_token_ids,
                repetition_penalty=repetition_penalty,
            )
        executions[modality] = {
            **predictions[modality],
            **mask_statistics(mask),
            **execution_cache[key],
        }
    all_on_key = "1" * 28
    if all_on_key in execution_cache:
        live_baseline = execution_cache[all_on_key]
        baseline_source = "predicted_all_on_live_execution"
    else:
        live_baseline = execute_mask(
            model=model,
            processor=processor,
            inputs=inputs,
            prepared=prepared,
            mask=[1] * 28,
            row=row,
            eos_token_ids=eos_token_ids,
            repetition_penalty=repetition_penalty,
        )
        baseline_source = "separate_live_all_on_execution"
    return {
        "uid": row["uid"],
        "sample_id": row["sample_id"],
        "benchmark": row["benchmark"],
        "suite": (
            "core_vqa" if row["benchmark"] in CORE_BENCHMARKS
            else "external_multiple_choice" if row["benchmark"] in MC_BENCHMARKS
            else "pope"
        ),
        "cluster_key": row["cluster_key"],
        "metric_name": row["metric_name"],
        "correctness_threshold": float(row["correctness_threshold"]),
        "baseline_prediction": live_baseline["prediction"],
        "baseline_score": live_baseline["score"],
        "baseline_correct": live_baseline["correct"],
        "baseline_generated_ids": live_baseline["generated_ids"],
        "baseline_source": baseline_source,
        "reference_cache_prediction": str(baseline["baseline_prediction"]),
        "reference_cache_score": float(baseline["baseline_score"]),
        "reference_cache_correct": bool(baseline["baseline_correct"]),
        "reference_cache_exact_match": bool(
            live_baseline["prediction"] == str(baseline["baseline_prediction"])
            and live_baseline["score"] == float(baseline["baseline_score"])
            and live_baseline["correct"] == bool(baseline["baseline_correct"])
        ),
        "predictor_text_sha256": sha256(row["predictor_text"].encode()).hexdigest(),
        "visual_tokens": int(visual.shape[0]),
        **executions,
    }


def preflight_native_parity(
    *, row: dict[str, Any], baseline: dict[str, Any], model: Any, processor: Any,
    eos_token_ids: list[int], repetition_penalty: float,
) -> dict[str, Any]:
    from dvr_qwen.binary_generate import prepare_binary_dvrc_inputs
    from dvr_qwen.scripts.cache_preference_gt_router_features import build_processor_inputs

    inputs = build_processor_inputs(processor, row, data_root=Path(row["data_root"]))
    prepared = prepare_binary_dvrc_inputs(model, inputs)
    all_on = [1] * 28
    binary = execute_mask(
        model=model, processor=processor, inputs=inputs, prepared=prepared,
        mask=all_on, row=row, eos_token_ids=eos_token_ids,
        repetition_penalty=repetition_penalty,
    )
    native_inputs = native_generation_inputs(
        inputs, device=next(model.base_model.parameters()).device
    )
    model.base_model.rope_deltas = None
    native_output = model.base_model.generate(
        **native_inputs,
        max_new_tokens=int(row["max_new_tokens"]),
        do_sample=False,
        use_cache=True,
        eos_token_id=eos_token_ids,
        repetition_penalty=repetition_penalty,
    )
    native_ids = native_output[:, inputs["input_ids"].shape[1] :]
    native_prediction = decode_generated(processor, native_ids)
    native_score = score_prediction(
        str(row["metric_name"]), native_prediction, row.get("answer"), row.get("all_answer_norms")
    )
    return {
        "uid": row["uid"],
        "benchmark": row["benchmark"],
        "binary_native_token_parity": binary["generated_ids"] == native_ids.cpu().view(-1).tolist(),
        "binary_prediction": binary["prediction"],
        "native_prediction": native_prediction,
        "cache_prediction": str(baseline["baseline_prediction"]),
        "binary_cache_prediction_parity": binary["prediction"] == str(baseline["baseline_prediction"]),
        "binary_cache_score_parity": math.isclose(binary["score"], float(baseline["baseline_score"]), abs_tol=1e-12),
        "binary_native_score_parity": math.isclose(binary["score"], native_score, abs_tol=1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--question-checkpoint", required=True)
    parser.add_argument("--image-question-checkpoint", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument(
        "--modality",
        choices=("both", "question", "image_question"),
        default="both",
    )
    parser.add_argument("--preflight-path")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    bundle_code = bundle / "code"
    sys.path.insert(0, str(bundle_code))
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    question_checkpoint = Path(args.question_checkpoint)
    image_question_checkpoint = Path(args.image_question_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_active_rows(bundle)
    baselines = baseline_index(bundle)
    if any(row["uid"] not in baselines for row in rows):
        raise RuntimeError("baseline cache does not cover active evaluation")
    if args.mode == "preflight":
        rows = choose_preflight_rows(rows, args.seed)
    else:
        rows = select_shard(rows, num_shards=args.num_shards, shard_index=args.shard_index)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from types import SimpleNamespace
    from dvr_qwen.scripts.cache_preference_gt_router_features import load_model_and_processor

    model_path = bundle / "model/Qwen2.5-VL-7B-Instruct_cc594898137f460bfe9f0759e9844b3ce807cfb5"
    load_args = SimpleNamespace(
        model_source=model_path,
        hf_hub_cache=bundle / "model",
        processor_use_fast="false",
        device_map="auto",
        first_gpu_max_memory_gb=40,
        other_gpu_max_memory_gb=40,
        cpu_max_memory_gb=0,
        attn_implementation="sdpa",
    )
    model, processor = load_model_and_processor(load_args)
    device = next(model.parameters()).device
    modalities = (
        ("question", "image_question")
        if args.modality == "both"
        else (args.modality,)
    )
    tokenizer, encoder, predictors = load_predictors(
        config,
        question_checkpoint,
        image_question_checkpoint,
        device,
        modalities=modalities,
    )
    eos_token_ids = [151645]
    repetition_penalty = 1.05

    if args.mode == "preflight":
        parity = []
        repeated = []
        for row in tqdm(rows, desc="external preflight", unit="sample"):
            parity.append(
                preflight_native_parity(
                    row=row,
                    baseline=baselines[row["uid"]],
                    model=model,
                    processor=processor,
                    eos_token_ids=eos_token_ids,
                    repetition_penalty=repetition_penalty,
                )
            )
            first = process_row(
                row=row, baseline=baselines[row["uid"]], model=model, processor=processor,
                tokenizer=tokenizer, encoder=encoder, predictors=predictors,
                max_question_tokens=int(config["data"]["max_question_tokens"]),
                device=device, eos_token_ids=eos_token_ids,
                repetition_penalty=repetition_penalty,
            )
            if row["benchmark"] in {"chartqa", "mmstar_val", "pope_adversarial"}:
                second = process_row(
                    row=row, baseline=baselines[row["uid"]], model=model, processor=processor,
                    tokenizer=tokenizer, encoder=encoder, predictors=predictors,
                    max_question_tokens=int(config["data"]["max_question_tokens"]),
                    device=device, eos_token_ids=eos_token_ids,
                    repetition_penalty=repetition_penalty,
                )
                repeated.append(
                    {
                        "uid": row["uid"],
                        **{
                            f"{modality}_exact": first[modality] == second[modality]
                            for modality in modalities
                        },
                    }
                )
        passed = all(
            item["binary_native_token_parity"]
            and item["binary_cache_prediction_parity"]
            and item["binary_cache_score_parity"]
            and item["binary_native_score_parity"]
            for item in parity
        ) and all(
            all(item[f"{modality}_exact"] for modality in modalities)
            for item in repeated
        )
        payload = {
            "schema_version": "binary_polar_external_preflight_v1",
            "passed": passed,
            "active_records": 22307,
            "checkpoints": {
                "question": {"path": str(question_checkpoint), "sha256": file_sha256(question_checkpoint)},
                "image_question": {"path": str(image_question_checkpoint), "sha256": file_sha256(image_question_checkpoint)},
            },
            "generation": {"eos_token_ids": eos_token_ids, "repetition_penalty": repetition_penalty},
            "parity": parity,
            "repeated_predictions": repeated,
        }
        path = output_dir / "preflight_v1.json"
        write_json(path, payload)
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
        )
        if not passed:
            raise RuntimeError("external evaluation preflight failed")
        return

    if len(modalities) != 1:
        raise RuntimeError(
            "full evaluation requires one --modality per GPU job; use question or image_question"
        )

    preflight = (
        Path(args.preflight_path)
        if args.preflight_path is not None
        else output_dir / "preflight_v1.json"
    )
    if not preflight.exists() or not json.loads(preflight.read_text())["passed"]:
        raise RuntimeError("full evaluation requires passed preflight_v1.json")
    shard_dir = output_dir / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    if shard_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite external shard: {shard_dir}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = shard_dir / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"external shard is already complete: {shard_dir}")
    existing_parts = sorted(shard_dir.glob("part_*.jsonl"))
    completed_uids = {
        str(row["uid"])
        for path in existing_parts
        for row in read_jsonl(path)
    }
    if len(completed_uids) != sum(len(read_jsonl(path)) for path in existing_parts):
        raise RuntimeError(f"duplicate UIDs in resumable shard: {shard_dir}")
    expected_uids = {str(row["uid"]) for row in rows}
    if not completed_uids <= expected_uids:
        raise RuntimeError(f"resumable shard contains unexpected UIDs: {shard_dir}")
    rows = [row for row in rows if str(row["uid"]) not in completed_uids]
    buffer = []
    part = len(existing_parts)
    completed = len(completed_uids)
    started = time.time()
    progress = tqdm(rows, desc=f"external shard {args.shard_index}", unit="sample")
    for row in progress:
        result = process_row(
            row=row, baseline=baselines[row["uid"]], model=model, processor=processor,
            tokenizer=tokenizer, encoder=encoder, predictors=predictors,
            max_question_tokens=int(config["data"]["max_question_tokens"]),
            device=device, eos_token_ids=eos_token_ids,
            repetition_penalty=repetition_penalty,
        )
        buffer.append(result)
        completed += 1
        if len(buffer) >= args.chunk_size:
            write_jsonl(shard_dir / f"part_{part:05d}.jsonl", buffer)
            buffer.clear()
            part += 1
        progress.set_postfix(
            **{name: result[name]["num_visual_on_layers"] for name in modalities}
        )
    if buffer:
        write_jsonl(shard_dir / f"part_{part:05d}.jsonl", buffer)
    metadata = {
        "schema_version": "binary_polar_external_shard_v1",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "records": completed,
        "resumed_records": len(completed_uids),
        "elapsed_seconds": time.time() - started,
        "config_sha256": file_sha256(config_path),
        "question_checkpoint_sha256": file_sha256(question_checkpoint),
        "image_question_checkpoint_sha256": file_sha256(image_question_checkpoint),
        "source_sha256": file_sha256(Path(__file__)),
        "modalities": list(modalities),
    }
    write_json(shard_dir / "metadata.json", metadata)


if __name__ == "__main__":
    main()
