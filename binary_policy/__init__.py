"""Binary visual-token routing for frozen Qwen2.5-VL models.

The package intentionally separates three concerns:

* :mod:`binary_policy.executor` implements the layer-local VISUAL_ON/OFF
  counterfactual;
* :mod:`binary_policy.labels` adapts the existing MCTS records without
  regenerating labels;
* :mod:`binary_policy.predictor` contains the lightweight POLAR-style policy
  model.  Importing this package never loads a foundation model.
"""

from .actions import NUM_QWEN_LAYERS, mask_key, normalize_visual_on_mask
from .predictor import BinaryPolarBackbone, SegmentedBinaryPolarBackbone

__all__ = [
    "BinaryPolarBackbone",
    "SegmentedBinaryPolarBackbone",
    "NUM_QWEN_LAYERS",
    "mask_key",
    "normalize_visual_on_mask",
]
