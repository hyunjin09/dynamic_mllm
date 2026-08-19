from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_outputs.dense_prefill_hierarchical_gate import GateHead
from baseline_relative_visual_router.input_admission import (
    compose_admission_score,
    prefix_feature_matrix,
)


class PrefixAdmissionRuntime:
    """Frozen ensemble callback for one-pass shared-prefix routing."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        selection: str = "accuracy",
        device: str | torch.device = "cpu",
    ) -> None:
        if selection not in {"accuracy", "efficiency"}:
            raise ValueError("selection must be 'accuracy' or 'efficiency'")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != "shared_dense_prefix_admission_selection_v1":
            raise RuntimeError("shared-prefix admission checkpoint schema mismatch")
        self.device = torch.device(device)
        self.selection = selection
        self.prefix_layers = int(payload[f"selected_{selection}_prefix_layers"])
        self.candidate = payload[f"selected_{selection}_candidate"]
        self.point_name = f"{selection}_point"
        self.input_size = int(payload["input_size"])
        states = payload["prefix_states"][self.prefix_layers]
        self.models: dict[str, list[GateHead]] = {}
        for target in ("harm", "rescue"):
            architecture = str(self.candidate[f"{target}_architecture"])
            target_models = []
            for state in states[target][architecture]:
                model = GateHead(architecture, input_size=self.input_size, hidden_size=256)
                model.load_state_dict(state)
                model.to(self.device).eval()
                target_models.append(model)
            self.models[target] = target_models
        self.last_decision: dict[str, float | bool] | None = None

    @torch.inference_mode()
    def _members(self, target: str, features: torch.Tensor) -> np.ndarray:
        values = [
            torch.sigmoid(model(features.to(self.device))).cpu().numpy()
            for model in self.models[target]
        ]
        return np.stack(values)

    @torch.inference_mode()
    def __call__(self, features: dict[str, torch.Tensor]) -> bool:
        matrix = prefix_feature_matrix(features)
        if tuple(matrix.shape) != (1, self.input_size):
            raise RuntimeError(
                f"expected one prefix feature of width {self.input_size}, got {tuple(matrix.shape)}"
            )
        harm = self._members("harm", matrix)
        rescue = self._members("rescue", matrix)
        score = compose_admission_score(
            str(self.candidate["score_mode"]),
            harm,
            rescue,
            harm_beta=float(self.candidate["harm_beta"]),
            harm_threshold=float(self.candidate["harm_threshold"]),
            utility_beta=float(self.candidate["utility_beta"]),
            rescue_weight=float(self.candidate["rescue_weight"]),
        )
        threshold = float(self.candidate[self.point_name]["threshold"])
        use_sparse = bool(score[0] <= threshold)
        self.last_decision = {
            "harm_probability": float(harm.mean(0)[0]),
            "rescue_probability": float(rescue.mean(0)[0]),
            "admission_score": float(score[0]),
            "threshold": threshold,
            "use_sparse": use_sparse,
        }
        return use_sparse

