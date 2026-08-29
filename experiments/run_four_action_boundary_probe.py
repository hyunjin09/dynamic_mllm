#!/usr/bin/env python3
"""Extract unified-FULL states and run the matched boundary probes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn import functional as F
import yaml

from binary_policy.executor import capture_four_action_route
from experiments.train_binary_polar import file_sha256
from experiments.train_four_action_online_router import prepare_sample, write_json
from four_action_online_router.boundary_probe import (
    BoundaryProbe,
    binary_classification_metrics,
    paired_uid_bootstrap_auc_difference,
)
from four_action_online_router.data import (
    load_jsonl,
    load_source_metadata,
    load_verified_manifest,
)
from four_action_online_router.runtime import select_last_text_state
from label_regeneration.runtime import configure_determinism, load_frozen_model


def distributed_context(expected_world_size: int) -> tuple[int, int, int, torch.device]:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_JOB_ID")
    if any(not os.environ.get(name) for name in required):
        raise RuntimeError("boundary probe requires torchrun inside Slurm")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != expected_world_size or not torch.cuda.is_available():
        raise RuntimeError(f"boundary probe requires exactly {expected_world_size} GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return rank, world_size, local_rank, device


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_probe_manifest(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("boundary-probe manifest checksum mismatch")
    rows = load_jsonl(path)
    identities = [(str(row["pair_id"]), int(row["label"])) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("boundary-probe manifest has duplicate pair/label records")
    if any(int(row["target_layer"]) >= 27 for row in rows):
        raise RuntimeError("boundary-probe manifest includes an unmatched terminal layer")
    return rows


def _mean_valid_visual(
    visual_states: torch.Tensor, visual_valid_mask: torch.BoolTensor
) -> torch.Tensor:
    mask = visual_valid_mask.to(visual_states.device)
    selected = visual_states[mask]
    if selected.numel() == 0:
        raise RuntimeError("boundary probe encountered an empty visual state")
    return selected.mean(dim=0)


def _feature_record(
    row: dict[str, Any],
    *,
    upfront_text: torch.Tensor,
    upfront_visual: torch.Tensor,
    online_text: torch.Tensor,
    online_visual: torch.Tensor,
) -> dict[str, Any]:
    metadata = {
        key: row[key]
        for key in (
            "pair_id",
            "uid",
            "split",
            "dataset",
            "target_layer",
            "source_boundary_layer",
            "label",
            "class_name",
        )
    }
    return {
        **metadata,
        "upfront_text": upfront_text.detach().to(device="cpu", dtype=torch.bfloat16),
        "upfront_visual": upfront_visual.detach().to(device="cpu", dtype=torch.bfloat16),
        "online_text": online_text.detach().to(device="cpu", dtype=torch.bfloat16),
        "online_visual": online_visual.detach().to(device="cpu", dtype=torch.bfloat16),
    }


def extract_rank_features(
    *,
    config: dict[str, Any],
    config_sha256: str,
    probe_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    shard_path: Path,
    rank: int,
    world_size: int,
    local_rank: int,
    device: torch.device,
) -> None:
    records_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in probe_rows:
        records_by_uid[str(row["uid"])].append(row)
    assigned_uids = sorted(records_by_uid)[rank::world_size]
    source_by_uid = {str(row["uid"]): row for row in source_rows}
    processor, base_model, wrapped_model, _ = load_frozen_model(
        config["base_model"]["path"], config["base_model"]["revision"], local_rank
    )
    base_model.requires_grad_(False).eval()
    output: list[dict[str, Any]] = []
    started = time.time()
    for position, uid in enumerate(assigned_uids, start=1):
        source_row = source_by_uid[uid]
        _sample, inputs, _metadata, prepared = prepare_sample(
            processor, wrapped_model, source_row, sources[uid], device
        )
        full = capture_four_action_route(
            wrapped_model,
            inputs,
            ("FULL",) * 28,
            prepared_inputs=prepared,
            use_cache=False,
        )
        layer_zero_text, layer_zero_visual = full.pre_layer_states[0]
        upfront_text = select_last_text_state(
            layer_zero_text, prepared.text_valid_mask.to(layer_zero_text.device)
        )[0]
        upfront_visual = _mean_valid_visual(
            layer_zero_visual, prepared.visual_valid_mask
        )
        for row in records_by_uid[uid]:
            target_layer = int(row["target_layer"])
            online_text_states, online_visual_states = full.pre_layer_states[target_layer]
            online_text = select_last_text_state(
                online_text_states,
                prepared.text_valid_mask.to(online_text_states.device),
            )[0]
            online_visual = _mean_valid_visual(
                online_visual_states, prepared.visual_valid_mask
            )
            output.append(
                _feature_record(
                    row,
                    upfront_text=upfront_text,
                    upfront_visual=upfront_visual,
                    online_text=online_text,
                    online_visual=online_visual,
                )
            )
        if position <= 3 or position % 25 == 0:
            print(
                json.dumps(
                    {
                        "event": "boundary_probe_feature",
                        "rank": rank,
                        "completed_uids": position,
                        "total_uids": len(assigned_uids),
                        "records": len(output),
                        "elapsed_seconds": time.time() - started,
                        "uid": uid,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del full, prepared, inputs
    atomic_torch_save(
        shard_path,
        {
            "schema_version": "four_action_boundary_probe_feature_shard_v1",
            "config_sha256": config_sha256,
            "rank": rank,
            "world_size": world_size,
            "uids": assigned_uids,
            "records": output,
        },
    )


def validate_feature_shard(
    path: Path, *, config_sha256: str, rank: int, world_size: int
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != "four_action_boundary_probe_feature_shard_v1"
        or payload.get("config_sha256") != config_sha256
        or int(payload.get("rank", -1)) != rank
        or int(payload.get("world_size", -1)) != world_size
    ):
        raise RuntimeError(f"incompatible boundary-probe feature shard: {path}")
    return payload


def load_all_features(
    root: Path,
    *,
    config_sha256: str,
    world_size: int,
    probe_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    seen_uids = set()
    for rank in range(world_size):
        payload = validate_feature_shard(
            root / f"features_rank_{rank:02d}.pt",
            config_sha256=config_sha256,
            rank=rank,
            world_size=world_size,
        )
        if seen_uids.intersection(payload["uids"]):
            raise RuntimeError("feature shards overlap in UID coverage")
        seen_uids.update(payload["uids"])
        records.extend(payload["records"])
    expected = {(str(row["pair_id"]), int(row["label"])) for row in probe_rows}
    actual = {(str(row["pair_id"]), int(row["label"])) for row in records}
    if actual != expected or len(records) != len(probe_rows):
        raise RuntimeError("feature shards do not exactly cover the probe manifest")
    records.sort(key=lambda row: (str(row["pair_id"]), int(row["label"])))
    widths = {
        int(row[key].numel())
        for row in records
        for key in ("upfront_text", "upfront_visual", "online_text", "online_visual")
    }
    if widths != {3584}:
        raise RuntimeError(f"unexpected boundary-probe feature widths: {widths}")
    return records


def _stack_split(
    records: list[dict[str, Any]], *, split: str, representation: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    selected = [row for row in records if row["split"] == split]
    text = torch.stack([row[f"{representation}_text"].float() for row in selected])
    visual = torch.stack([row[f"{representation}_visual"].float() for row in selected])
    layers = torch.tensor([int(row["target_layer"]) for row in selected], dtype=torch.long)
    labels = torch.tensor([float(row["label"]) for row in selected], dtype=torch.float32)
    return text, visual, layers, labels, selected


@torch.inference_mode()
def predict_probabilities(
    model: BoundaryProbe,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]],
    *,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    model.eval()
    text, visual, layers, _labels, _rows = tensors
    probabilities = []
    for start in range(0, len(text), batch_size):
        stop = start + batch_size
        logits = model(
            text[start:stop].to(device),
            visual[start:stop].to(device),
            layers[start:stop].to(device),
        )
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return [float(value) for value in probabilities]


def train_probe(
    *,
    representation: str,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    config_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = config["training"]
    probe = config["probe"]
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = BoundaryProbe(
        hidden_width=int(config["features"]["hidden_width"]),
        num_layers=28,
        branch_width=int(probe["branch_width"]),
        layer_embedding_width=int(probe["layer_embedding_width"]),
        classifier_hidden_width=int(probe["classifier_hidden_width"]),
        dropout=float(probe["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    train = _stack_split(records, split="train", representation=representation)
    validation = _stack_split(records, split="validation", representation=representation)
    batch_size = int(training["batch_size"])
    history = []
    best_key: tuple[float, float, float, int] | None = None
    best_state = None
    best_epoch = 0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(train[0]), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                train[0][indices].to(device),
                train[1][indices].to(device),
                train[2][indices].to(device),
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, train[3][indices].to(device)
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(indices)
        probabilities = predict_probabilities(
            model, validation, batch_size=batch_size, device=device
        )
        labels = [int(value) for value in validation[3].tolist()]
        metrics = binary_classification_metrics(labels, probabilities)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train[0]),
            "validation": metrics,
        }
        history.append(row)
        key = (metrics["auroc"], metrics["accuracy"], metrics["f1"], -epoch)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        print(
            json.dumps(
                {"event": "boundary_probe_epoch", "representation": representation, **row},
                sort_keys=True,
            ),
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("probe training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    probabilities = predict_probabilities(
        model, validation, batch_size=batch_size, device=device
    )
    labels = [int(value) for value in validation[3].tolist()]
    metrics = binary_classification_metrics(labels, probabilities)
    checkpoint_path = output_dir / f"{representation}_best.pt"
    atomic_torch_save(
        checkpoint_path,
        {
            "schema_version": "four_action_boundary_probe_checkpoint_v1",
            "representation": representation,
            "config_sha256": config_sha256,
            "epoch": best_epoch,
            "model": best_state,
            "validation": metrics,
        },
    )
    predictions = [
        {
            "pair_id": row["pair_id"],
            "uid": row["uid"],
            "dataset": row["dataset"],
            "target_layer": int(row["target_layer"]),
            "source_boundary_layer": int(row["source_boundary_layer"]),
            "label": int(row["label"]),
            "probability": probability,
        }
        for row, probability in zip(validation[4], probabilities)
    ]
    summary = {
        "representation": representation,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "validation": metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "history": history,
    }
    return summary, predictions


def render_report(summary: dict[str, Any]) -> str:
    upfront = summary["probes"]["upfront"]
    online = summary["probes"]["online"]
    bootstrap = summary["paired_online_minus_upfront_auroc"]
    answer = "YES" if summary["online_advantage"] else "NO"
    return "\n".join(
        [
            "# Upfront vs Online Mandatory-Boundary Probe",
            "",
            "## Frozen comparison",
            "",
            f"- Config SHA-256: `{summary['config_sha256']}`",
            f"- Probe records: {summary['records']} ({summary['train_records']} train, {summary['validation_records']} validation)",
            f"- Unique feature UIDs: {summary['feature_uids']}",
            f"- Matching: `{summary['matching']}`",
            f"- Model parameters per probe: {upfront['parameter_count']:,}",
            "- Upfront state: unified-FULL pre-layer-0 final text/control row plus mean visual row, with target-layer identity.",
            "- Online state: unified-FULL pre-target-layer final text/control row plus mean visual row, with target-layer identity.",
            "",
            "## Validation results",
            "",
            "| Representation | Best epoch | AUROC | Accuracy | F1 |",
            "|---|---:|---:|---:|---:|",
            f"| Upfront | {upfront['best_epoch']} | {upfront['validation']['auroc']:.6f} | {upfront['validation']['accuracy']:.6f} | {upfront['validation']['f1']:.6f} |",
            f"| Online | {online['best_epoch']} | {online['validation']['auroc']:.6f} | {online['validation']['accuracy']:.6f} | {online['validation']['f1']:.6f} |",
            "",
            "## Paired primary comparison",
            "",
            f"- Online-minus-upfront AUROC: {bootstrap['point_estimate']:.6f}",
            f"- UID-group bootstrap 95% CI: [{bootstrap['ci_low']:.6f}, {bootstrap['ci_high']:.6f}]",
            f"- Valid bootstrap draws: {bootstrap['valid_draws']}/{bootstrap['requested_draws']}",
            f"- Frozen decision rule: lower 95% CI > 0.",
            f"- Are mandatory deviations more predictable from current routed state? **{answer}**",
            "",
            "This probe is representational evidence. It does not by itself establish that a free-running router will use the signal correctly.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "upfront_vs_online_boundary_probe_v1":
        raise RuntimeError("unexpected boundary-probe protocol")
    if file_sha256(Path(config["source_plan"])) != config["source_plan_sha256"]:
        raise RuntimeError("boundary-probe plan checksum mismatch")
    if file_sha256(Path(config["source_config"])) != config["source_config_sha256"]:
        raise RuntimeError("boundary-probe source-config checksum mismatch")
    config_sha256 = file_sha256(config_path)
    rank, world_size, local_rank, device = distributed_context(args.world_size)
    try:
        configure_determinism(int(config["training"]["seed"]))
        probe_rows = load_probe_manifest(
            Path(config["data"]["manifest"]), config["data"]["manifest_sha256"]
        )
        source_config = yaml.safe_load(
            Path(config["source_config"]).read_text(encoding="utf-8")
        )
        source_rows = load_verified_manifest(
            source_config["data"]["manifest"],
            source_config["data"]["manifest_sha256"],
        )
        required_uids = {str(row["uid"]) for row in probe_rows}
        source_rows = [row for row in source_rows if str(row["uid"]) in required_uids]
        if {str(row["uid"]) for row in source_rows} != required_uids:
            raise RuntimeError("probe UIDs are not covered by the source manifest")
        sources = load_source_metadata(
            source_config["data"]["source_manifest"],
            source_config["data"]["source_manifest_sha256"],
            required_uids,
        )
        feature_root = Path(config["reporting"]["feature_root"])
        shard_path = feature_root / f"features_rank_{rank:02d}.pt"
        if shard_path.is_file():
            validate_feature_shard(
                shard_path,
                config_sha256=config_sha256,
                rank=rank,
                world_size=world_size,
            )
            print(
                json.dumps(
                    {"event": "boundary_probe_feature_resume", "rank": rank, "path": str(shard_path)},
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            extract_rank_features(
                config=config,
                config_sha256=config_sha256,
                probe_rows=probe_rows,
                source_rows=source_rows,
                sources=sources,
                shard_path=shard_path,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                device=device,
            )
        dist.barrier()
        if rank == 0:
            records = load_all_features(
                feature_root,
                config_sha256=config_sha256,
                world_size=world_size,
                probe_rows=probe_rows,
            )
            output_dir = Path(config["reporting"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            probe_summaries = {}
            predictions = {}
            for representation in ("upfront", "online"):
                probe_summaries[representation], predictions[representation] = train_probe(
                    representation=representation,
                    records=records,
                    config=config,
                    device=device,
                    output_dir=output_dir,
                    config_sha256=config_sha256,
                )
            upfront_by_key = {
                (row["pair_id"], row["label"]): row for row in predictions["upfront"]
            }
            online_by_key = {
                (row["pair_id"], row["label"]): row for row in predictions["online"]
            }
            if upfront_by_key.keys() != online_by_key.keys():
                raise RuntimeError("upfront and online validation predictions are not paired")
            keys = sorted(upfront_by_key)
            labels = [int(upfront_by_key[key]["label"]) for key in keys]
            upfront_probabilities = [
                float(upfront_by_key[key]["probability"]) for key in keys
            ]
            online_probabilities = [
                float(online_by_key[key]["probability"]) for key in keys
            ]
            uid_groups = [str(upfront_by_key[key]["uid"]) for key in keys]
            bootstrap = paired_uid_bootstrap_auc_difference(
                labels,
                upfront_probabilities,
                online_probabilities,
                uid_groups,
                draws=int(config["analysis"]["bootstrap_draws"]),
                seed=int(config["training"]["seed"]) + 1,
            )
            paired_rows = []
            for key in keys:
                base = upfront_by_key[key]
                paired_rows.append(
                    {
                        **{field: base[field] for field in (
                            "pair_id", "uid", "dataset", "target_layer",
                            "source_boundary_layer", "label",
                        )},
                        "upfront_probability": upfront_by_key[key]["probability"],
                        "online_probability": online_by_key[key]["probability"],
                    }
                )
            write_json(output_dir / "validation_predictions.json", paired_rows)
            summary = {
                "schema_version": "upfront_vs_online_boundary_probe_summary_v1",
                "passed": True,
                "config": str(config_path),
                "config_sha256": config_sha256,
                "source_plan_sha256": config["source_plan_sha256"],
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "records": len(records),
                "train_records": sum(row["split"] == "train" for row in records),
                "validation_records": sum(row["split"] == "validation" for row in records),
                "feature_uids": len({str(row["uid"]) for row in records}),
                "matching": config["data"]["matching"],
                "probes": probe_summaries,
                "paired_online_minus_upfront_auroc": bootstrap,
                "online_advantage_rule": config["analysis"]["online_advantage_rule"],
                "online_advantage": bool(bootstrap["ci_low"] > 0.0),
            }
            write_json(output_dir / "summary.json", summary)
            write_json(Path(config["reporting"]["summary"]), summary)
            report_path = Path(config["reporting"]["report"])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(render_report(summary), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "event": "boundary_probe_complete",
                        "upfront_auroc": probe_summaries["upfront"]["validation"]["auroc"],
                        "online_auroc": probe_summaries["online"]["validation"]["auroc"],
                        "difference": bootstrap["point_estimate"],
                        "ci_low": bootstrap["ci_low"],
                        "ci_high": bootstrap["ci_high"],
                        "online_advantage": summary["online_advantage"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
