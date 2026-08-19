"""Static binary visual-token executor.

VISUAL_OFF compacts the layer input to text/control rows and carries visual
rows unchanged. VISUAL_ON scatters both streams into native sequence order and
runs the unmodified decoder layer.
"""

from .generation import binary_greedy_generate, binary_prefill, binary_route_forward
from .model import BinaryQwen25VL

__all__ = ["BinaryQwen25VL", "binary_greedy_generate", "binary_prefill", "binary_route_forward"]
