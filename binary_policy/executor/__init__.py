"""Static binary visual-token executor.

VISUAL_OFF compacts the layer input to text/control rows and carries visual
rows unchanged. VISUAL_ON scatters both streams into native sequence order and
runs the unmodified decoder layer.
"""

from .generation import binary_greedy_generate, binary_prefill, binary_route_forward
from .four_action import (
    FOUR_ACTIONS,
    capture_four_action_route,
    capture_full_baseline,
    capture_online_four_action_route,
    capture_route_baseline,
    four_action_layer,
    greedy_generate_from_cached_prompt,
    greedy_generate_from_local_forward,
    full_baseline_post_layer_text_states,
    layerwise_token_scores_from_cached_prompt,
    local_four_action_forward,
    route_conditioned_four_action_forward,
    score_token_ids_from_cached_prompt,
    score_token_ids_from_local_forward,
    unified_target_four_action_layer,
)
from .model import BinaryQwen25VL

__all__ = [
    "BinaryQwen25VL",
    "FOUR_ACTIONS",
    "binary_greedy_generate",
    "binary_prefill",
    "binary_route_forward",
    "capture_four_action_route",
    "capture_full_baseline",
    "capture_online_four_action_route",
    "capture_route_baseline",
    "four_action_layer",
    "greedy_generate_from_cached_prompt",
    "greedy_generate_from_local_forward",
    "full_baseline_post_layer_text_states",
    "layerwise_token_scores_from_cached_prompt",
    "local_four_action_forward",
    "route_conditioned_four_action_forward",
    "score_token_ids_from_cached_prompt",
    "score_token_ids_from_local_forward",
    "unified_target_four_action_layer",
]
