from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from interventions.read_path import ReadDecomposition, ReadInterventionCache, ReadPathController


@dataclass
class LayerContext:
    layer_index: int
    pre_layer_state: torch.Tensor
    full_layer_output: torch.Tensor
    layer_kwargs: dict[str, Any]


@dataclass
class StateResult:
    name: str
    target_output: torch.Tensor
    logits: torch.Tensor
    read_decomposition: ReadDecomposition | None
    read_hook_identity_max_abs: float
    write_hook_identity_max_abs: float
    injected_prestate_max_abs: float


class LayerCapture:
    def __init__(self, layer, layer_index: int):
        self.layer = layer
        self.layer_index = layer_index
        self.pre_layer_state = None
        self.full_layer_output = None
        self.layer_kwargs = None
        self._pre_handle = None
        self._post_handle = None

    def __enter__(self):
        self._pre_handle = self.layer.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        self._post_handle = self.layer.register_forward_hook(self._post_hook, with_kwargs=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._pre_handle.remove()
        self._post_handle.remove()

    def _pre_hook(self, module, args, kwargs):
        hidden_states = args[0] if args else kwargs["hidden_states"]
        self.pre_layer_state = hidden_states.detach().clone()
        self.layer_kwargs = {
            key: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
            for key, value in kwargs.items()
        }

    def _post_hook(self, module, args, kwargs, output):
        self.full_layer_output = output[0].detach().clone()

    def context(self) -> LayerContext:
        if self.pre_layer_state is None or self.full_layer_output is None or self.layer_kwargs is None:
            raise RuntimeError("Target layer was not captured")
        return LayerContext(
            layer_index=self.layer_index,
            pre_layer_state=self.pre_layer_state,
            full_layer_output=self.full_layer_output,
            layer_kwargs=self.layer_kwargs,
        )


def _write_state(
    target_output: torch.Tensor,
    context: LayerContext,
    visual_token_mask: torch.Tensor,
    mode: str,
    replacement_delta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    if mode not in {"full", "off", "reconstruct", "replace", "subtract"}:
        raise ValueError(f"Unsupported WRITE mode: {mode}")
    if mode in {"replace", "subtract"} and replacement_delta is None:
        raise ValueError(f"WRITE {mode} mode requires a replacement delta")
    if replacement_delta is not None and replacement_delta.shape != target_output.shape:
        raise ValueError("Replacement WRITE delta must match the layer output shape")
    visual_rows = visual_token_mask.unsqueeze(-1)
    write_off = torch.where(visual_rows, context.pre_layer_state, target_output)
    delta_write = torch.where(
        visual_rows,
        context.full_layer_output.float() - context.pre_layer_state.float(),
        torch.zeros_like(target_output, dtype=torch.float32),
    )
    reconstructed = torch.where(
        visual_rows,
        (write_off.float() + delta_write).to(target_output.dtype),
        target_output,
    )
    identity = float(
        (reconstructed[visual_rows.expand_as(reconstructed)].float() - context.full_layer_output[visual_rows.expand_as(context.full_layer_output)].float())
        .abs()
        .max()
        .item()
    )
    if mode == "off":
        return write_off, identity
    if mode == "replace":
        assert replacement_delta is not None
        replaced = torch.where(
            visual_rows,
            (context.pre_layer_state.float() + replacement_delta.float()).to(
                target_output.dtype
            ),
            target_output,
        )
        return replaced, identity
    if mode == "subtract":
        assert replacement_delta is not None
        subtracted = torch.where(
            visual_rows,
            (target_output.float() - replacement_delta.float()).to(target_output.dtype),
            target_output,
        )
        return subtracted, identity
    if mode == "reconstruct":
        return reconstructed, identity
    return target_output, identity


def run_cached_state(
    causal_lm,
    context: LayerContext,
    visual_token_mask: torch.Tensor,
    name: str,
    read_mode: str,
    write_mode: str,
    read_cache: ReadInterventionCache | None = None,
    read_replacement_delta: torch.Tensor | None = None,
    write_replacement_delta: torch.Tensor | None = None,
) -> StateResult:
    decoder = causal_lm.model
    layer = decoder.layers[context.layer_index]
    injected = context.pre_layer_state.detach().clone()
    injected_error = float((injected.float() - context.pre_layer_state.float()).abs().max().item())
    kwargs = dict(context.layer_kwargs)
    kwargs["past_key_value"] = None
    kwargs["use_cache"] = False
    kwargs["output_attentions"] = False

    with ReadPathController(
        layer.self_attn,
        visual_token_mask,
        read_mode,
        read_cache,
        replacement_delta=read_replacement_delta,
    ) as controller:
        target_output = layer(injected, **kwargs)[0]
    target_output, write_identity = _write_state(
        target_output,
        context,
        visual_token_mask,
        write_mode,
        write_replacement_delta,
    )

    hidden_states = target_output
    for suffix_layer in decoder.layers[context.layer_index + 1 :]:
        hidden_states = suffix_layer(hidden_states, **kwargs)[0]
    hidden_states = decoder.norm(hidden_states)
    logits = causal_lm.lm_head(hidden_states)
    if controller.hook_identity_max_abs is None:
        raise RuntimeError("READ controller did not record its identity")
    return StateResult(
        name=name,
        target_output=target_output,
        logits=logits,
        read_decomposition=controller.decomposition,
        read_hook_identity_max_abs=controller.hook_identity_max_abs,
        write_hook_identity_max_abs=write_identity,
        injected_prestate_max_abs=injected_error,
    )


FOUR_STATES = {
    "IGNORE": ("off", "off"),
    "READ_ONLY": ("full", "off"),
    "WRITE_ONLY": ("off", "full"),
    "FULL": ("full", "full"),
}
