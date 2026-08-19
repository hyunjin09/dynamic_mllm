#!/usr/bin/env python3
"""Validate offline admission composition against the one-pass runtime branch."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE / "src"))
os.environ.setdefault("HF_HOME", "/mnt/hyemin/models")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/hyemin/models/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/hyemin/models/hub")

import torch

from baseline_relative_visual_router.input_admission import load_prefix_feature_cache
from baseline_relative_visual_router.input_features import align_manifest_policy_rows
from baseline_relative_visual_router.prefix_runtime import PrefixAdmissionRuntime
from dvr_qwen.binary_generate import binary_dvrc_router_greedy_generate
from dvr_qwen.eval_metrics import score_prediction
from dvr_qwen.scripts.cache_preference_gt_router_features import (
    build_processor_inputs,
    ensure_min_free_cuda_memory,
    load_model_and_processor,
)
from dvr_qwen.scripts.evaluate_heldout_online_visual_router_generation import (
    decode_generated,
    load_router_checkpoint,
)


DEFAULT_MODEL = Path(
    "/mnt/hyemin/models/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/"
    "snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
DEFAULT_ROUTER = Path(
    "/mnt/hyemin/10k_dataset_mask/online_visual_router_preference_runs/"
    "sw31_bt_leg_s41/router_epoch_001.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-rows-jsonl", type=Path, required=True)
    parser.add_argument("--hybrid-feature-root", type=Path, required=True)
    parser.add_argument("--selection-checkpoint", type=Path, required=True)
    parser.add_argument("--external-predictions-jsonl", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--samples-per-benchmark", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hf-hub-cache", type=Path, default=Path("/mnt/hyemin/models/hub"))
    parser.add_argument("--processor-use-fast", choices=["true", "false"], default="false")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", choices=["auto", "cpu", "none"], default="auto")
    parser.add_argument("--first-gpu-max-memory-gb", type=int, default=20)
    parser.add_argument("--other-gpu-max-memory-gb", type=int, default=20)
    parser.add_argument("--cpu-max-memory-gb", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--process-name", default="brvr-prefix-runtime-audit")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fixed_benchmark_decision_subset(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    offline_decisions: dict[str, bool],
    per_benchmark: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for manifest, baseline in rows:
        groups.setdefault(str(manifest["benchmark"]), []).append((manifest, baseline))
    result = []
    for benchmark in sorted(groups):
        values = sorted(groups[benchmark], key=lambda pair: str(pair[0]["uid"]))
        sparse = [pair for pair in values if offline_decisions[str(pair[0]["uid"])]]
        all_on = [pair for pair in values if not offline_decisions[str(pair[0]["uid"])]]
        half = per_benchmark // 2
        chosen = sparse[:half] + all_on[: per_benchmark - half]
        chosen_uids = {str(pair[0]["uid"]) for pair in chosen}
        if len(chosen) < per_benchmark:
            chosen.extend(
                pair
                for pair in values
                if str(pair[0]["uid"]) not in chosen_uids
            )
            chosen = chosen[:per_benchmark]
        result.extend(chosen)
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    gate = PrefixAdmissionRuntime(
        args.selection_checkpoint, selection="accuracy", device="cuda"
    )
    _, hybrid_rows = load_prefix_feature_cache(
        args.hybrid_feature_root / f"prefix_{gate.prefix_layers:02d}",
        expected_prefix_layers=gate.prefix_layers,
    )
    hybrids = {str(row["uid"]): row for row in hybrid_rows}
    aligned = align_manifest_policy_rows(
        read_jsonl(args.manifest_jsonl), read_jsonl(args.baseline_rows_jsonl)
    )
    offline_decisions = {
        str(row["uid"]): bool(row["use_sparse_hybrid"])
        for row in read_jsonl(args.external_predictions_jsonl)
    }
    if set(offline_decisions) != {str(pair[0]["uid"]) for pair in aligned}:
        raise RuntimeError("external prediction and manifest UID sets differ")
    selected = fixed_benchmark_decision_subset(
        aligned, offline_decisions, args.samples_per_benchmark
    )
    ensure_min_free_cuda_memory(args.min_free_gb, device_map=args.device_map)
    model, processor = load_model_and_processor(args)
    router_device = next(model.parameters()).device
    router, runtime, _ = load_router_checkpoint(
        args.checkpoint,
        allow_initial=False,
        threshold_override=None,
        device=router_device,
    )
    mismatch = Counter()
    decisions = Counter()
    rows = []
    for manifest, baseline in selected:
        uid = str(manifest["uid"])
        hybrid = hybrids[uid]
        inputs = build_processor_inputs(processor, manifest, data_root=args.data_root)
        output = binary_dvrc_router_greedy_generate(
            model,
            inputs,
            visual_on_router=router,
            max_new_tokens=int(manifest.get("max_new_tokens") or 16),
            eos_token_ids=[151643, 151645],
            stop_on_eos=True,
            repetition_penalty=1.05,
            visual_summary_mode=str(runtime["visual_summary_mode"]),
            text_summary_mode=str(runtime["text_summary_mode"]),
            router_threshold=float(runtime["router_threshold"]),
            return_route_logits=True,
            forced_visual_on_prefix_layers=gate.prefix_layers,
            capture_prefix_gate_features=True,
            prefix_admission_callback=gate,
        )
        use_sparse = bool(output.state.prefix_admission_used_sparse)
        decisions["sparse" if use_sparse else "all_on"] += 1
        prediction = decode_generated(processor, output.generated_ids)
        score = float(
            score_prediction(
                str(manifest["metric_name"]),
                prediction,
                manifest.get("answer"),
                manifest.get("all_answer_norms"),
            )
        )
        correct = bool(score >= float(manifest["correctness_threshold"]))
        mask = output.state.route_binary.detach().cpu().view(-1).to(torch.int64).tolist()
        expected = hybrid if use_sparse else baseline
        expected_prediction = str(
            expected["router_prediction"] if use_sparse else expected["baseline_prediction"]
        )
        expected_score = float(
            expected["router_score"] if use_sparse else expected["baseline_score"]
        )
        expected_correct = bool(
            expected["router_correct"] if use_sparse else expected["baseline_correct"]
        )
        expected_mask = (
            list(hybrid["selected_visual_on_mask"]) if use_sparse else [1] * len(mask)
        )
        checks = {
            "admission": use_sparse == offline_decisions[uid],
            "prediction": prediction == expected_prediction,
            "score": abs(score - expected_score) <= 1e-12,
            "correct": correct == expected_correct,
            "mask": mask == expected_mask,
        }
        for name, passed in checks.items():
            mismatch[name] += int(not passed)
        rows.append({"uid": uid, "benchmark": manifest["benchmark"], "use_sparse": use_sparse, **checks})
        print(f"[prefix-runtime-audit] {len(rows)}/{len(selected)} {uid} sparse={use_sparse}", flush=True)
    summary = {
        "schema_version": "shared_prefix_runtime_equivalence_v1",
        "n": len(rows),
        "prefix_layers": gate.prefix_layers,
        "decisions": dict(decisions),
        "mismatches": dict(mismatch),
        "passed": not any(mismatch.values()),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("n", "prefix_layers", "decisions", "mismatches", "passed")}, indent=2))
    if not summary["passed"]:
        raise RuntimeError("one-pass prefix runtime does not match offline policy composition")


if __name__ == "__main__":
    main()
