"""Load only the immutable packaged executor in the existing project environment."""
import sys
from common import PACKAGE


def runtime_imports():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PACKAGE / '07_RESUME_TOOLS'))
    import torch
    from transformers import AutoProcessor
    from dvr_qwen.modeling_dvr_qwen2_5_vl import DVRQwen2_5_VLForConditionalGeneration
    from dvr_qwen.binary_generate import binary_dvrc_greedy_generate, prepare_binary_dvrc_inputs
    from dvr_qwen.eval_metrics import score_prediction
    from dvr_qwen.scripts.cache_preference_gt_router_features import build_processor_inputs
    return torch, AutoProcessor, DVRQwen2_5_VLForConditionalGeneration, binary_dvrc_greedy_generate, prepare_binary_dvrc_inputs, score_prediction, build_processor_inputs
