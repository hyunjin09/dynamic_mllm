"""Custom generation for binary contextualized DVR-C."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from dvr_qwen.binary_layer import (
    BinaryDVRCLayerStats,
    BinaryDVRCStaticInputCache,
    forward_decode_text_layer,
    forward_text_only_layer,
    forward_visual_on_layer,
    normalize_visual_on_mask,
)
from dvr_qwen.cache import DVRCache
from dvr_qwen.fallback_gate import initial_input_gate_features
from dvr_qwen.generate import (
    _eos_token_set,
    apply_repetition_penalty,
    generation_repetition_penalty,
    generation_token_history,
    next_decode_position_ids,
)
from dvr_qwen.modeling_dvr_qwen2_5_vl import (
    DVRQwen2_5_VLForConditionalGeneration,
    qwen_num_hidden_layers,
    qwen_text_model,
)
from dvr_qwen.router_features import summarize_text_states, summarize_visual_states
from dvr_qwen.routing import BinaryVisualOnRouter, InputFallbackGate
from dvr_qwen.split_scatter import BinaryDVRCIndexCache, BinaryDVRCInputs, build_binary_dvrc_inputs

_STATIC_ROUTE_MODE = "static_route"
_ROUTER_TEACHER_FORCING_MODE = "router_teacher_forcing"
_ROUTER_ONLINE_TEACHER_FORCING_MODE = "router_online_teacher_forcing"
_ROUTER_ONLINE_MODE = "router_online"
_ROUTE_MODES = {
    _STATIC_ROUTE_MODE,
    _ROUTER_TEACHER_FORCING_MODE,
    _ROUTER_ONLINE_TEACHER_FORCING_MODE,
    _ROUTER_ONLINE_MODE,
}
_ROUTER_ROUTE_MODES = {
    _ROUTER_TEACHER_FORCING_MODE,
    _ROUTER_ONLINE_TEACHER_FORCING_MODE,
    _ROUTER_ONLINE_MODE,
}
_ONLINE_ROUTE_MODES = {_ROUTER_ONLINE_TEACHER_FORCING_MODE, _ROUTER_ONLINE_MODE}
_TEACHER_ROUTE_MODES = {
    _STATIC_ROUTE_MODE,
    _ROUTER_TEACHER_FORCING_MODE,
    _ROUTER_ONLINE_TEACHER_FORCING_MODE,
}


@dataclass
class BinaryDVRCPrefillState:
    last_hidden_state: torch.Tensor
    cache: DVRCache | None
    binary_inputs: BinaryDVRCInputs
    route_binary: torch.Tensor
    prefill_layer_stats: list[BinaryDVRCLayerStats]
    route_logits: torch.Tensor | None = None
    router_feature_batch: dict[str, torch.Tensor] | None = None
    fallback_gate_logit: torch.Tensor | None = None
    fallback_used_sparse_router: bool | None = None


@dataclass
class BinaryDVRCGenerationOutput:
    generated_ids: torch.Tensor
    state: BinaryDVRCPrefillState
    prefill_logits: torch.Tensor
    decode_layer_stats: list[list[BinaryDVRCLayerStats]] = field(default_factory=list)


def _lm_logits(model: DVRQwen2_5_VLForConditionalGeneration, hidden_states: torch.Tensor) -> torch.Tensor:
    lm_device = next(model.lm_head.parameters()).device
    return model.lm_head(hidden_states.to(lm_device))


def logits_from_binary_prefill(
    model: DVRQwen2_5_VLForConditionalGeneration,
    state: BinaryDVRCPrefillState,
) -> torch.Tensor:
    last_idx = state.binary_inputs.text_valid_mask.long().sum(dim=1) - 1
    batch_idx = torch.arange(state.last_hidden_state.shape[0], device=state.last_hidden_state.device)
    last_hidden = state.last_hidden_state[
        batch_idx,
        last_idx.to(state.last_hidden_state.device),
    ]
    return _lm_logits(model, last_hidden[:, None, :])[:, -1, :]


def _previous_gates_from_route(route: torch.Tensor) -> torch.Tensor:
    prev = torch.zeros_like(route, dtype=torch.long)
    prev[:, 1:] = route[:, :-1].to(dtype=torch.long)
    return prev


def _router_parameter_device(router: torch.nn.Module) -> torch.device:
    return next(router.parameters()).device


def _compute_teacher_forced_route_logits(
    router: BinaryVisualOnRouter,
    feature_batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    router_device = _router_parameter_device(router)
    return router(
        feature_batch["global_mean"].to(device=router_device, dtype=torch.float32),
        feature_batch["window_mean"].to(device=router_device, dtype=torch.float32),
        feature_batch["last_token"].to(device=router_device, dtype=torch.float32),
        feature_batch["layer_idx"].to(device=router_device, dtype=torch.long),
        feature_batch["prev_gates"].to(device=router_device, dtype=torch.long),
        scalar_features=(
            feature_batch["scalar_features"].to(device=router_device, dtype=torch.float32)
            if "scalar_features" in feature_batch
            else None
        ),
        visual_summaries=(
            feature_batch["visual_summaries"].to(device=router_device, dtype=torch.float32)
            if "visual_summaries" in feature_batch
            else None
        ),
    )


def _compute_online_layer_route_logit(
    router: torch.nn.Module,
    current_summary: dict[str, torch.Tensor],
    *,
    layer_idx: int,
    prev_gate: int,
    scalar_features: torch.Tensor | None,
    visual_summary: torch.Tensor | None,
    text_states: torch.Tensor | None = None,
    text_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    router_device = _router_parameter_device(router)
    forward_layer_states = getattr(router, "forward_layer_states", None)
    if callable(forward_layer_states):
        if text_states is None or text_valid_mask is None:
            raise ValueError("state-aware router requires text states and a valid-token mask")
        return forward_layer_states(
            text_states.detach().to(device=router_device),
            text_valid_mask.to(device=router_device, dtype=torch.bool),
            layer_idx=layer_idx,
        )
    return router.forward_layer(
        current_summary["global_mean"].to(device=router_device, dtype=torch.float32),
        current_summary["window_mean"].to(device=router_device, dtype=torch.float32),
        current_summary["last_token"].to(device=router_device, dtype=torch.float32),
        layer_idx=layer_idx,
        prev_gate=prev_gate,
        scalar_features=(
            scalar_features.to(device=router_device, dtype=torch.float32)
            if scalar_features is not None
            else None
        ),
        visual_summaries=(
            visual_summary.to(device=router_device, dtype=torch.float32)
            if visual_summary is not None
            else None
        ),
    )


def _build_teacher_forced_router_feature_batch(
    summaries: dict[str, list[torch.Tensor]],
    route: torch.Tensor,
    *,
    scalar_features: torch.Tensor | None,
    visual_summaries: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch_size, num_layers = route.shape
    feature_batch = {
        key: torch.stack(values, dim=1).to(dtype=torch.float32).cpu()
        for key, values in summaries.items()
    }
    feature_batch["layer_idx"] = (
        torch.arange(num_layers, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).cpu()
    )
    feature_batch["prev_gates"] = _previous_gates_from_route(route).to(dtype=torch.long).cpu()
    if visual_summaries:
        feature_batch["visual_summaries"] = torch.stack(visual_summaries, dim=1).to(dtype=torch.float32).cpu()
    if scalar_features is not None:
        feature_batch["scalar_features"] = torch.as_tensor(scalar_features, dtype=torch.float32).cpu()
    return feature_batch


def prepare_binary_dvrc_inputs(
    model: DVRQwen2_5_VLForConditionalGeneration,
    inputs: Mapping[str, Any],
) -> BinaryDVRCInputs:
    """Build immutable multimodal inputs once for reuse across route evaluations."""

    return build_binary_dvrc_inputs(
        model.base_model,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        mm_token_type_ids=inputs["mm_token_type_ids"],
        pixel_values=inputs.get("pixel_values"),
        pixel_values_videos=inputs.get("pixel_values_videos"),
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        second_per_grid_ts=inputs.get("second_per_grid_ts"),
        instruction_token_mask=inputs.get("instruction_token_mask"),
    )


def binary_dvrc_prefill(
    model: DVRQwen2_5_VLForConditionalGeneration,
    inputs: Mapping[str, Any],
    visual_on_mask: torch.Tensor | None,
    use_cache: bool = True,
    use_static_input_cache: bool = True,
    use_index_cache: bool = False,
    route_mode: str = "static_route",
    visual_on_router: BinaryVisualOnRouter | None = None,
    return_route_logits: bool = False,
    scalar_features: torch.Tensor | None = None,
    visual_summary_mode: str = "none",
    text_summary_mode: str = "all_text",
    router_threshold: float = 0.0,
    prepared_binary_inputs: BinaryDVRCInputs | None = None,
) -> BinaryDVRCPrefillState:
    if route_mode not in _ROUTE_MODES:
        raise ValueError(f"route_mode must be one of {sorted(_ROUTE_MODES)}, got {route_mode!r}")
    if visual_summary_mode not in {"none", "mean_abs"}:
        raise ValueError("visual_summary_mode must be 'none' or 'mean_abs'")
    if text_summary_mode not in {"all_text", "instruction_only"}:
        raise ValueError("text_summary_mode must be 'all_text' or 'instruction_only'")
    if route_mode in _TEACHER_ROUTE_MODES and visual_on_mask is None:
        raise ValueError(f"visual_on_mask is required for route_mode={route_mode!r}")
    if route_mode == _STATIC_ROUTE_MODE:
        if return_route_logits:
            raise ValueError("static_route execution does not produce router logits")
        if visual_on_router is not None:
            raise ValueError("visual_on_router is only used with router route modes")
        if scalar_features is not None:
            raise ValueError("scalar_features are only used with router route modes")
        if visual_summary_mode != "none":
            raise ValueError("visual_summary_mode is only used with router route modes")
    elif route_mode in _ROUTER_ROUTE_MODES and visual_on_router is None:
        raise ValueError(f"visual_on_router is required for route_mode={route_mode!r}")
    if route_mode == _ROUTER_ONLINE_MODE and visual_on_mask is not None:
        raise ValueError("router_online mode predicts its own route; pass visual_on_mask=None")
    if not isinstance(router_threshold, (int, float)):
        raise TypeError("router_threshold must be a numeric logit threshold")
    online_router_requires_grad = torch.is_grad_enabled() and route_mode == _ROUTER_ONLINE_TEACHER_FORCING_MODE

    with torch.no_grad():
        binary_inputs = prepared_binary_inputs or prepare_binary_dvrc_inputs(model, inputs)
        batch_size = binary_inputs.text_states.shape[0]
        if batch_size != 1:
            raise NotImplementedError("binary DVR-C validation currently supports batch size 1")

        text_model = qwen_text_model(model)
        num_layers = qwen_num_hidden_layers(model.config)
        if route_mode in _TEACHER_ROUTE_MODES:
            route = normalize_visual_on_mask(
                visual_on_mask,
                batch_size=batch_size,
                num_layers=num_layers,
                device=binary_inputs.text_states.device,
            )
        else:
            route = torch.zeros(
                batch_size,
                num_layers,
                dtype=torch.bool,
                device=binary_inputs.text_states.device,
            )
        cache = DVRCache(num_layers=num_layers) if use_cache else None
        static_input_cache = None
        if use_static_input_cache:
            static_input_cache = BinaryDVRCStaticInputCache(
                index_cache=BinaryDVRCIndexCache() if use_index_cache else None
            )
        layer_stats: list[BinaryDVRCLayerStats] = []
        text_states = binary_inputs.text_states
        visual_states = binary_inputs.visual_states
        summaries: dict[str, list[torch.Tensor]] = {
            "global_mean": [],
            "window_mean": [],
            "last_token": [],
        }
        visual_summaries: list[torch.Tensor] = []
        online_route_logits: list[torch.Tensor] = []
        collect_router_features = (
            route_mode in {
                _ROUTER_TEACHER_FORCING_MODE,
                _ROUTER_ONLINE_TEACHER_FORCING_MODE,
                _ROUTER_ONLINE_MODE,
            }
            and return_route_logits
        )
        summary_valid_mask = binary_inputs.text_valid_mask
        if text_summary_mode == "instruction_only":
            if binary_inputs.instruction_valid_mask is None:
                raise ValueError("instruction_only text summaries require instruction_token_mask")
            summary_valid_mask = binary_inputs.instruction_valid_mask

        for layer_idx, layer in enumerate(text_model.layers):
            current_summary = None
            current_visual_summary = None
            needs_layer_summary = route_mode in _ONLINE_ROUTE_MODES or collect_router_features
            if needs_layer_summary:
                current = summarize_text_states(
                    text_states.detach(),
                    summary_valid_mask.to(text_states.device),
                )
                current_summary = current
                if visual_summary_mode == "mean_abs":
                    current_visual_summary = summarize_visual_states(
                        visual_states.detach(),
                        binary_inputs.visual_valid_mask.to(visual_states.device),
                    )
                if collect_router_features:
                    for key, value in current.items():
                        summaries[key].append(value.detach().float().cpu())
                    if current_visual_summary is not None:
                        visual_summaries.append(current_visual_summary.detach().float().cpu())

            if route_mode in _ONLINE_ROUTE_MODES:
                assert visual_on_router is not None
                assert current_summary is not None
                prev_gate = 0 if layer_idx == 0 else int(route[0, layer_idx - 1].item())
                grad_context = torch.enable_grad() if online_router_requires_grad else torch.no_grad()
                with grad_context:
                    layer_logit = _compute_online_layer_route_logit(
                        visual_on_router,
                        current_summary,
                        layer_idx=layer_idx,
                        prev_gate=prev_gate,
                        scalar_features=scalar_features,
                        visual_summary=current_visual_summary,
                        text_states=text_states,
                        text_valid_mask=summary_valid_mask,
                    )
                if return_route_logits:
                    online_route_logits.append(layer_logit)
                if route_mode == _ROUTER_ONLINE_MODE:
                    route[0, layer_idx] = bool((layer_logit.detach()[0] > float(router_threshold)).item())

            if bool(route[0, layer_idx].item()):
                text_states, visual_states, stats = forward_visual_on_layer(
                    text_model,
                    layer,
                    text_states,
                    visual_states,
                    binary_inputs,
                    layer_idx=layer_idx,
                    cache=cache,
                    use_cache=use_cache,
                    static_input_cache=static_input_cache,
                )
            else:
                text_states, visual_states, stats = forward_text_only_layer(
                    text_model,
                    layer,
                    text_states,
                    visual_states,
                    binary_inputs,
                    layer_idx=layer_idx,
                    cache=cache,
                    use_cache=use_cache,
                    static_input_cache=static_input_cache,
                )
            layer_stats.append(stats)

        norm_device = next(text_model.norm.parameters()).device
        text_states = text_model.norm(text_states.to(norm_device))
    route_logits = None
    router_feature_batch = None
    if route_mode == _ROUTER_TEACHER_FORCING_MODE and return_route_logits:
        assert visual_on_router is not None
        router_feature_batch = _build_teacher_forced_router_feature_batch(
            summaries,
            route.detach().cpu(),
            scalar_features=scalar_features,
            visual_summaries=visual_summaries,
        )
        route_logits = _compute_teacher_forced_route_logits(visual_on_router, router_feature_batch)
    elif route_mode in _ONLINE_ROUTE_MODES and return_route_logits:
        if len(online_route_logits) != route.shape[1]:
            raise AssertionError(
                f"expected {route.shape[1]} online route logits, got {len(online_route_logits)}"
            )
        router_feature_batch = _build_teacher_forced_router_feature_batch(
            summaries,
            route.detach().cpu(),
            scalar_features=scalar_features,
            visual_summaries=visual_summaries,
        )
        route_logits = torch.stack(online_route_logits, dim=1)
    return BinaryDVRCPrefillState(
        last_hidden_state=text_states,
        cache=cache,
        binary_inputs=binary_inputs,
        route_binary=route,
        prefill_layer_stats=layer_stats,
        route_logits=route_logits,
        router_feature_batch=router_feature_batch,
    )


@torch.inference_mode()
def binary_dvrc_decode_one_step(
    model: DVRQwen2_5_VLForConditionalGeneration,
    token_id: torch.Tensor,
    state: BinaryDVRCPrefillState,
    generated_step: int,
) -> tuple[torch.Tensor, list[BinaryDVRCLayerStats]]:
    if state.cache is None:
        raise ValueError("binary_dvrc_decode_one_step requires a cached prefill state")

    text_model = qwen_text_model(model)
    embed_device = next(model.model.get_input_embeddings().parameters()).device
    hidden_states = model.model.get_input_embeddings()(token_id.to(embed_device).view(-1, 1))
    position_ids = next_decode_position_ids(state.binary_inputs.dvr_inputs, generated_step)
    layer_stats: list[BinaryDVRCLayerStats] = []

    for layer_idx, layer in enumerate(text_model.layers):
        hidden_states, stats = forward_decode_text_layer(
            text_model,
            layer,
            hidden_states,
            position_ids,
            layer_idx=layer_idx,
            cache=state.cache,
        )
        stats.visual_on = bool(state.route_binary[0, layer_idx].item())
        layer_stats.append(stats)

    norm_device = next(text_model.norm.parameters()).device
    hidden_states = text_model.norm(hidden_states.to(norm_device))
    state.last_hidden_state = hidden_states
    logits = _lm_logits(model, hidden_states)[:, -1, :]
    return logits, layer_stats


def _continue_binary_dvrc_greedy_generate(
    model: DVRQwen2_5_VLForConditionalGeneration,
    inputs: Mapping[str, Any],
    state: BinaryDVRCPrefillState,
    *,
    max_new_tokens: int,
    eos_token_ids: int | list[int] | None = None,
    stop_on_eos: bool = True,
    repetition_penalty: float | None = None,
) -> BinaryDVRCGenerationOutput:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    logits = logits_from_binary_prefill(model, state)
    prefill_logits = logits.detach().cpu()
    eos_ids = _eos_token_set(model, eos_token_ids)
    penalty = generation_repetition_penalty(model, repetition_penalty)
    generated: list[torch.Tensor] = []
    decode_stats: list[list[BinaryDVRCLayerStats]] = []

    for step in range(max_new_tokens):
        token_history = generation_token_history(
            inputs["input_ids"],
            generated,
            device=logits.device,
        )
        processed_logits = apply_repetition_penalty(logits, token_history, penalty)
        next_token = torch.argmax(processed_logits, dim=-1)
        generated.append(next_token[:, None])
        logits, step_stats = binary_dvrc_decode_one_step(
            model,
            next_token,
            state,
            generated_step=step,
        )
        decode_stats.append(step_stats)
        if stop_on_eos and eos_ids and int(next_token[0].item()) in eos_ids:
            break

    if generated:
        generated_ids = torch.cat(generated, dim=1)
    else:
        generated_ids = torch.empty(
            state.route_binary.shape[0],
            0,
            dtype=torch.long,
            device=state.route_binary.device,
        )
    return BinaryDVRCGenerationOutput(
        generated_ids=generated_ids,
        state=state,
        prefill_logits=prefill_logits,
        decode_layer_stats=decode_stats,
    )


@torch.inference_mode()
def binary_dvrc_greedy_generate(
    model: DVRQwen2_5_VLForConditionalGeneration,
    inputs: Mapping[str, Any],
    visual_on_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: int | list[int] | None = None,
    stop_on_eos: bool = True,
    repetition_penalty: float | None = None,
    use_static_input_cache: bool = True,
    use_index_cache: bool = False,
    prepared_binary_inputs: BinaryDVRCInputs | None = None,
) -> BinaryDVRCGenerationOutput:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    state = binary_dvrc_prefill(
        model,
        inputs,
        visual_on_mask=visual_on_mask,
        use_cache=True,
        use_static_input_cache=use_static_input_cache,
        use_index_cache=use_index_cache,
        prepared_binary_inputs=prepared_binary_inputs,
    )
    return _continue_binary_dvrc_greedy_generate(
        model,
        inputs,
        state,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        stop_on_eos=stop_on_eos,
        repetition_penalty=repetition_penalty,
    )


@torch.inference_mode()
def binary_dvrc_router_greedy_generate(
    model: DVRQwen2_5_VLForConditionalGeneration,
    inputs: Mapping[str, Any],
    visual_on_router: BinaryVisualOnRouter,
    max_new_tokens: int,
    eos_token_ids: int | list[int] | None = None,
    stop_on_eos: bool = True,
    repetition_penalty: float | None = None,
    use_static_input_cache: bool = True,
    use_index_cache: bool = False,
    scalar_features: torch.Tensor | None = None,
    visual_summary_mode: str = "none",
    text_summary_mode: str = "all_text",
    router_threshold: float = 0.0,
    return_route_logits: bool = True,
    prepared_binary_inputs: BinaryDVRCInputs | None = None,
) -> BinaryDVRCGenerationOutput:
    """Generate with live layer-wise router decisions during prefill.

    The router observes each layer-entry hidden summary, selects VISUAL_ON when
    its logit exceeds ``router_threshold``, and the selected route is immediately
    used to run that Qwen layer.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    state = binary_dvrc_prefill(
        model,
        inputs,
        visual_on_mask=None,
        use_cache=True,
        use_static_input_cache=use_static_input_cache,
        use_index_cache=use_index_cache,
        route_mode=_ROUTER_ONLINE_MODE,
        visual_on_router=visual_on_router,
        return_route_logits=return_route_logits,
        scalar_features=scalar_features,
        visual_summary_mode=visual_summary_mode,
        text_summary_mode=text_summary_mode,
        router_threshold=router_threshold,
        prepared_binary_inputs=prepared_binary_inputs,
    )
    return _continue_binary_dvrc_greedy_generate(
        model,
        inputs,
        state,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        stop_on_eos=stop_on_eos,
        repetition_penalty=repetition_penalty,
    )


@torch.inference_mode()
def binary_dvrc_input_fallback_router_greedy_generate(
    model: DVRQwen2_5_VLForConditionalGeneration,
    inputs: Mapping[str, Any],
    visual_on_router: BinaryVisualOnRouter,
    input_fallback_gate: InputFallbackGate,
    max_new_tokens: int,
    *,
    fallback_threshold: float = 0.0,
    eos_token_ids: int | list[int] | None = None,
    stop_on_eos: bool = True,
    repetition_penalty: float | None = None,
    use_static_input_cache: bool = True,
    use_index_cache: bool = False,
    scalar_features: torch.Tensor | None = None,
    visual_summary_mode: str = "none",
    text_summary_mode: str = "all_text",
    router_threshold: float = 0.0,
    gate_text_summary_mode: str = "instruction_only",
    gate_visual_summary_count: int = 2,
    return_route_logits: bool = True,
    prepared_binary_inputs: BinaryDVRCInputs | None = None,
) -> BinaryDVRCGenerationOutput:
    """Generate with a conservative input-only all-on fallback.

    The fallback gate is queried once from pre-language-layer states.  When its
    logit does not exceed ``fallback_threshold``, the function executes the
    all-on route exactly through :func:`binary_dvrc_greedy_generate`; otherwise
    it delegates to the existing causal per-layer sparse router.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not isinstance(fallback_threshold, (int, float)):
        raise TypeError("fallback_threshold must be a numeric logit threshold")
    prepared = prepared_binary_inputs or prepare_binary_dvrc_inputs(model, inputs)
    gate_features = initial_input_gate_features(
        prepared,
        text_summary_mode=gate_text_summary_mode,
        visual_summary_count=gate_visual_summary_count,
    )
    gate_device = _router_parameter_device(input_fallback_gate)
    gate_logit = input_fallback_gate(
        gate_features["instruction_mean"].to(device=gate_device, dtype=torch.float32),
        gate_features["instruction_last"].to(device=gate_device, dtype=torch.float32),
        gate_features.get("visual_summaries", None).to(device=gate_device, dtype=torch.float32)
        if "visual_summaries" in gate_features
        else None,
    )
    if gate_logit.numel() != 1:
        raise ValueError("input fallback generation currently requires batch size 1")
    use_sparse_router = bool((gate_logit[0] > float(fallback_threshold)).item())
    if use_sparse_router:
        output = binary_dvrc_router_greedy_generate(
            model,
            inputs,
            visual_on_router=visual_on_router,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            stop_on_eos=stop_on_eos,
            repetition_penalty=repetition_penalty,
            use_static_input_cache=use_static_input_cache,
            use_index_cache=use_index_cache,
            scalar_features=scalar_features,
            visual_summary_mode=visual_summary_mode,
            text_summary_mode=text_summary_mode,
            router_threshold=router_threshold,
            return_route_logits=return_route_logits,
            prepared_binary_inputs=prepared,
        )
    else:
        output = binary_dvrc_greedy_generate(
            model,
            inputs,
            visual_on_mask=torch.ones(
                (1, qwen_num_hidden_layers(model.config)),
                dtype=torch.bool,
                device=prepared.text_states.device,
            ),
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            stop_on_eos=stop_on_eos,
            repetition_penalty=repetition_penalty,
            use_static_input_cache=use_static_input_cache,
            use_index_cache=use_index_cache,
            prepared_binary_inputs=prepared,
        )
    output.state.fallback_gate_logit = gate_logit.detach().float().cpu()
    output.state.fallback_used_sparse_router = use_sparse_router
    return output
