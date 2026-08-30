#!/usr/bin/env python3
"""Four-GPU synthetic gradient smoke for both persistent-supervision paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from four_action_online_router.model import OnlineFourActionRouter
from four_action_online_router.supervision import set_valued_action_loss
from four_action_policy.losses import exact_valid_set_nll
from four_action_policy.persistent import persistent_boundary_loss
from four_action_policy.predictor import FourActionPolarBackbone


def _finite_nonzero_gradients(module: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(value).all().item()) for value in gradients) and any(
        bool((value != 0).any().item()) for value in gradients
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != args.world_size or world_size != 4 or not torch.cuda.is_available():
        raise RuntimeError("persistent smoke requires direct torchrun on exactly four GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        torch.manual_seed(20260830 + rank)
        boundary_layers = torch.tensor([2, -1], device=device)
        boundary_valid = torch.tensor(
            [[False, True, True, False], [False, False, False, False]], device=device
        )
        boundary_present = torch.tensor([True, False], device=device)

        polar = FourActionPolarBackbone(
            num_layers=4,
            input_dim=16,
            image_dim=16,
            d_model=16,
            num_heads=4,
            num_layer_blocks=1,
            dropout=0.0,
        ).to(device)
        polar_logits = polar(
            torch.randn(2, 3, 16, device=device),
            torch.ones(2, 3, dtype=torch.bool, device=device),
            torch.randn(2, 5, 16, device=device),
            torch.ones(2, 5, dtype=torch.bool, device=device),
        )
        routes = torch.tensor(
            [[[3, 3, 1, 3]], [[3, 3, 3, 3]]], dtype=torch.long, device=device
        )
        route_mask = torch.ones(2, 1, dtype=torch.bool, device=device)
        weights = torch.ones(2, 1, device=device)
        polar_base = exact_valid_set_nll(
            polar_logits, routes, valid_mask=route_mask, route_weights=weights
        )
        polar_boundary = persistent_boundary_loss(
            polar_logits,
            boundary_layers=boundary_layers,
            valid_actions=boundary_valid,
            present=boundary_present,
        )
        (polar_base + polar_boundary).backward()

        online = OnlineFourActionRouter(
            hidden_size=16,
            num_layers=4,
            d_router=16,
            num_heads=4,
            mlp_hidden_size=32,
            dropout=0.0,
            interaction_scale=0.1,
        ).to(device)
        text = torch.randn(4, 16, device=device)
        visual = torch.randn(4, 5, 16, device=device)
        visual_mask = torch.ones(4, 5, dtype=torch.bool, device=device)
        online_logits = torch.stack(
            [online(text[layer : layer + 1], visual[layer : layer + 1], visual_mask[layer : layer + 1], layer)[0] for layer in range(4)]
        )
        online_masks = torch.tensor(
            [
                [False, False, False, True],
                [False, False, False, True],
                [False, True, True, False],
                [False, False, False, True],
            ],
            device=device,
        )
        online_base = set_valued_action_loss(online_logits, online_masks)
        online_boundary = set_valued_action_loss(
            online_logits[2:3], boundary_valid[0:1]
        )
        (online_base + online_boundary).backward()

        local = {
            "rank": rank,
            "device": str(device),
            "semantic_boundary_mask": boundary_valid[0].tolist(),
            "polar_base_loss_finite": bool(torch.isfinite(polar_base).item()),
            "polar_boundary_loss_finite": bool(torch.isfinite(polar_boundary).item()),
            "polar_gradients_finite_nonzero": _finite_nonzero_gradients(polar),
            "online_base_loss_finite": bool(torch.isfinite(online_base).item()),
            "online_boundary_loss_finite": bool(torch.isfinite(online_boundary).item()),
            "online_gradients_finite_nonzero": _finite_nonzero_gradients(online),
            "c2c_has_boundary_term": bool(boundary_present[1].item()),
        }
        gathered: list[dict | None] = [None] * world_size
        dist.all_gather_object(gathered, local)
        if rank == 0:
            records = [value for value in gathered if value is not None]
            passed = (
                len(records) == 4
                and {row["device"] for row in records}
                == {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}
                and all(
                    row["semantic_boundary_mask"] == [False, True, True, False]
                    and row["polar_base_loss_finite"]
                    and row["polar_boundary_loss_finite"]
                    and row["polar_gradients_finite_nonzero"]
                    and row["online_base_loss_finite"]
                    and row["online_boundary_loss_finite"]
                    and row["online_gradients_finite_nonzero"]
                    and not row["c2c_has_boundary_term"]
                    for row in records
                )
            )
            payload = {
                "schema_version": "persistent_corrective_four_gpu_smoke_v1",
                "passed": passed,
                "world_size": world_size,
                "boundary_lambda": 1.0,
                "records": records,
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(payload, sort_keys=True))
            if not passed:
                raise RuntimeError("persistent four-GPU smoke failed")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
