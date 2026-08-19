from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.research_analysis.v3.confirmation_preflight import right_pad_prompt_inputs
from experiments.stage_a_validity import max_abs_difference, prepare_prompt, set_determinism
from experiments.stage_b_reference_likelihood import capture_prompt_with_cache
from experiments.v3_confirmation_preflight import (
    BASE_SEED,
    LAYER_SET,
    gqa_query_pair,
    load_model,
    write_json,
)


def rms_relative(first: torch.Tensor, second: torch.Tensor) -> float:
    difference = (first.float() - second.float()).square().mean().sqrt()
    scale = torch.stack(
        [first.float().square().mean().sqrt(), second.float().square().mean().sqrt()]
    ).mean()
    return float((difference / scale.clamp_min(1e-12)).item())


def execute() -> None:
    set_determinism(BASE_SEED)
    model, processor, model_config = load_model()
    first, second = gqa_query_pair()
    device = torch.device("cuda")
    prepared = [prepare_prompt(processor, record, device)[1] for record in (first, second)]
    original_lengths = [int(item["input_ids"].shape[1]) for item in prepared]
    common_length = max(original_lengths)
    padded = [
        right_pad_prompt_inputs(item, common_length, processor.tokenizer.pad_token_id)
        for item in prepared
    ]
    captured = []
    with torch.inference_mode():
        for inputs in padded:
            visual = inputs["input_ids"] == model.config.image_token_id
            indices = torch.where(visual[0])[0]
            outputs, contexts = capture_prompt_with_cache(model, inputs, LAYER_SET)
            captured.append(
                {
                    layer: {
                        "pre": contexts[layer].pre_layer_state[0, indices].detach().cpu(),
                        "full": contexts[layer].full_layer_output[0, indices].detach().cpu(),
                    }
                    for layer in LAYER_SET
                }
            )
            del outputs, contexts
    rows = []
    for layer in LAYER_SET:
        first_pre = captured[0][layer]["pre"]
        second_pre = captured[1][layer]["pre"]
        first_full = captured[0][layer]["full"]
        second_full = captured[1][layer]["full"]
        first_write = first_full.float() - first_pre.float()
        second_write = second_full.float() - second_pre.float()
        rows.append(
            {
                "layer": layer,
                "pre_max_abs": max_abs_difference(first_pre, second_pre),
                "pre_rms_relative": rms_relative(first_pre, second_pre),
                "post_max_abs": max_abs_difference(first_full, second_full),
                "post_rms_relative": rms_relative(first_full, second_full),
                "write_max_abs": max_abs_difference(first_write, second_write),
                "write_rms_relative": rms_relative(first_write, second_write),
            }
        )
    exact = all(
        row["pre_max_abs"] == row["post_max_abs"] == row["write_max_abs"] == 0.0
        for row in rows
    )
    payload = {
        "schema_version": "v3_query_invariance_equal_length_diagnostic_v1",
        "outcome_blind": True,
        "heldout_terminal_action_values_computed_or_inspected": False,
        "diagnostic_question": "Does equal total prompt length remove the observed same-image visual-state numerical divergence?",
        "sample_ids": [first["id"], second["id"]],
        "original_lengths": original_lengths,
        "common_padded_length": common_length,
        "right_padding_only": True,
        "model": model_config,
        "layers": rows,
        "exact_visual_state_and_write_equality": exact,
    }
    output = Path("outputs/v3_preflight/query_invariance_equal_length_diagnostic.json")
    write_json(output, payload)
    print(json.dumps({"output": str(output), "exact": exact}, indent=2))


if __name__ == "__main__":
    execute()
