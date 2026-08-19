from __future__ import annotations

import numpy as np
import torch

from analysis_outputs.dense_prefill_hierarchical_gate import GateHead


@torch.inference_mode()
def ensemble_probabilities(
    features: torch.Tensor,
    states: list[dict[str, torch.Tensor]],
    architecture: str,
    uncertainty_beta: float,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    members = []
    for state in states:
        model = GateHead(architecture).to(device)
        model.load_state_dict(state)
        model.eval()
        output = []
        for start in range(0, len(features), batch_size):
            logits = model(features[start : start + batch_size].to(device))
            output.append(torch.sigmoid(logits).cpu())
        members.append(torch.cat(output).numpy())
    stacked = np.stack(members)
    return stacked.mean(axis=0) + float(uncertainty_beta) * stacked.std(axis=0)
