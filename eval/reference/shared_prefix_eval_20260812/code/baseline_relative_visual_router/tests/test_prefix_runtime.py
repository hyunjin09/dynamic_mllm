from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from analysis_outputs.dense_prefill_hierarchical_gate import GateHead
from baseline_relative_visual_router.prefix_runtime import PrefixAdmissionRuntime


class PrefixAdmissionRuntimeTest(unittest.TestCase):
    def test_checkpoint_callback_uses_calibrated_point(self) -> None:
        with TemporaryDirectory() as directory:
            input_size = 15
            states = {}
            for target in ("harm", "rescue"):
                model = GateHead("linear", input_size=input_size, hidden_size=256)
                for parameter in model.parameters():
                    parameter.data.zero_()
                states[target] = {"linear": [model.state_dict()]}
            candidate = {
                "score_mode": "harm_only",
                "harm_architecture": "linear",
                "rescue_architecture": "linear",
                "harm_beta": 0.0,
                "harm_threshold": 0.6,
                "utility_beta": 0.0,
                "rescue_weight": 1.0,
                "accuracy_point": {"threshold": 0.5},
                "efficiency_point": {"threshold": 0.4},
            }
            path = Path(directory) / "gate.pt"
            torch.save(
                {
                    "schema_version": "shared_dense_prefix_admission_selection_v1",
                    "input_size": input_size,
                    "selected_accuracy_prefix_layers": 2,
                    "selected_efficiency_prefix_layers": 2,
                    "selected_accuracy_candidate": candidate,
                    "selected_efficiency_candidate": candidate,
                    "prefix_states": {2: states},
                },
                path,
            )
            runtime = PrefixAdmissionRuntime(path, selection="accuracy")
            features = {
                "instruction_mean": torch.ones(1, 3),
                "instruction_window_mean": torch.ones(1, 3),
                "instruction_last": torch.ones(1, 3),
                "visual_summaries": torch.ones(1, 2, 3),
            }
            self.assertTrue(runtime(features))
            self.assertEqual(runtime.prefix_layers, 2)
            self.assertAlmostEqual(float(runtime.last_decision["harm_probability"]), 0.5)


if __name__ == "__main__":
    unittest.main()
