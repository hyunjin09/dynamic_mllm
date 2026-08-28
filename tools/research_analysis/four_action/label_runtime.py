from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from binary_policy.executor import (
    binary_greedy_generate,
    capture_four_action_route,
    greedy_generate_from_cached_prompt,
    score_token_ids_from_cached_prompt,
)
from binary_policy.executor.inputs import build_binary_inputs
from label_regeneration.runtime import (
    build_native_processor_inputs,
    score_prediction_with_timeout,
)
from tools.research_analysis.four_action.targets import (
    AnswerTarget,
    accepted_answer_targets,
    full_wrong_target,
)


@dataclass(frozen=True)
class CurrentFullState:
    evaluation: dict[str, Any]
    correct_targets: tuple[AnswerTarget, ...]
    wrong_target: AnswerTarget | None


class FourActionSampleRuntime:
    """Preprocess one sample once and evaluate complete four-action routes."""

    def __init__(
        self,
        *,
        processor,
        model,
        sample: dict[str, Any],
        device: torch.device,
        scoring_timeout_seconds: float = 5.0,
    ):
        self.processor = processor
        self.model = model
        self.sample = dict(sample)
        self.device = device
        self.scoring_timeout_seconds = float(scoring_timeout_seconds)
        if self.scoring_timeout_seconds <= 0:
            raise ValueError("scoring_timeout_seconds must be positive")
        portable_sample = {**self.sample, "local_image_path": self.sample["image_path"]}
        self.inputs, self.input_metadata = build_native_processor_inputs(
            processor,
            portable_sample,
            device,
        )
        self.prepared = build_binary_inputs(model, self.inputs)
        self._prediction_scores: dict[str, tuple[float, bool, bool]] = {}
        self._full: CurrentFullState | None = None

    @property
    def geometry(self) -> dict[str, int]:
        return {
            "text_tokens": int(self.prepared.text_valid_mask.sum().item()),
            "visual_tokens": int(self.prepared.visual_valid_mask.sum().item()),
            "full_prompt_tokens": int(self.prepared.full_attention_mask.sum().item()),
        }

    def _decode(self, generated_ids: torch.Tensor) -> str:
        return self.processor.batch_decode(
            generated_ids.detach().cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _correctness(self, prediction: str) -> tuple[float, bool, bool]:
        if prediction not in self._prediction_scores:
            score, timed_out = score_prediction_with_timeout(
                self.sample["metric_name"],
                prediction,
                self.sample["answer"],
                self.sample.get("all_answer_norms"),
                timeout_seconds=self.scoring_timeout_seconds,
            )
            self._prediction_scores[prediction] = (
                float(score),
                bool(score >= float(self.sample["correctness_threshold"])),
                bool(timed_out),
            )
        return self._prediction_scores[prediction]

    def _token_ids(self, text: str) -> torch.Tensor:
        ids = self.processor.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0].to(self.device)
        if ids.numel() < 1:
            raise ValueError(f"answer target tokenized to empty content: {text!r}")
        return ids

    def _target_score(self, output, target: AnswerTarget) -> dict[str, Any]:
        assert output.cache is not None
        score = score_token_ids_from_cached_prompt(
            self.model,
            output.prompt_logits,
            output.inputs,
            output.cache,
            self._token_ids(target.text),
        )
        return {
            "text": target.text,
            "evaluator_score": target.evaluator_score,
            "token_ids": score.token_ids,
            "token_logprobs": score.token_logprobs,
            "sequence_logprob": score.sequence_logprob,
            "mean_logprob": score.mean_logprob,
        }

    def _score_output(
        self,
        output,
        *,
        correct_targets: tuple[AnswerTarget, ...] | None,
        wrong_target: AnswerTarget | None,
    ) -> dict[str, Any]:
        assert output.cache is not None
        generation = greedy_generate_from_cached_prompt(
            self.model,
            output.prompt_logits,
            output.inputs,
            output.cache,
            self.inputs["input_ids"],
            max_new_tokens=int(self.sample["max_new_tokens"]),
        )
        prediction = self._decode(generation.generated_ids)
        correctness_score, correct, timed_out = self._correctness(prediction)
        state = {
            "generated_answer": prediction,
            "generated_ids": generation.generated_ids[0].detach().cpu().tolist(),
            "correctness_score": correctness_score,
            "correct": correct,
            "scoring_timed_out": timed_out,
            "cache_lengths": output.cache.lengths(),
        }
        if correct_targets is not None:
            state.update(self._alignment_state(output, correct_targets, wrong_target))
        del generation
        return state

    def _alignment_state(
        self,
        output,
        correct_targets: tuple[AnswerTarget, ...],
        wrong_target: AnswerTarget | None,
    ) -> dict[str, Any]:
        correct_rows = [self._target_score(output, target) for target in correct_targets]
        selected_correct = max(
            correct_rows,
            key=lambda row: (row["mean_logprob"], row["sequence_logprob"], row["text"]),
        )
        wrong_row = None if wrong_target is None else self._target_score(output, wrong_target)
        correct_value = float(selected_correct["mean_logprob"])
        wrong_value = None if wrong_row is None else float(wrong_row["mean_logprob"])
        return {
            "S_correct": correct_value,
            "S_full_wrong": wrong_value,
            "answer_alignment_margin": (
                correct_value if wrong_value is None else correct_value - wrong_value
            ),
            "score_quantity": (
                "S_correct" if wrong_value is None else "S_correct_minus_S_full_wrong"
            ),
            "correct_target_scores": correct_rows,
            "selected_correct_target": selected_correct,
            "full_wrong_target_score": wrong_row,
        }

    @torch.inference_mode()
    def initialize_full(self) -> CurrentFullState:
        if self._full is not None:
            return self._full
        actions = ("FULL",) * len(self.model.decoder.layers)
        output = capture_four_action_route(
            self.model,
            self.inputs,
            actions,
            prepared_inputs=self.prepared,
            use_cache=True,
        )
        generation_only = self._score_output(
            output,
            correct_targets=None,
            wrong_target=None,
        )
        target_record = {
            **self.sample,
            "full_prediction": generation_only["generated_answer"],
            "full_correct": generation_only["correct"],
        }
        correct_targets = tuple(accepted_answer_targets(target_record))
        wrong_target = None if generation_only["correct"] else full_wrong_target(target_record)
        state = generation_only
        state.update(self._alignment_state(output, correct_targets, wrong_target))
        self._full = CurrentFullState(state, correct_targets, wrong_target)
        del output
        return self._full

    @torch.inference_mode()
    def evaluate(self, route: tuple[str, ...]) -> dict[str, Any]:
        full = self.initialize_full()
        if all(action == "FULL" for action in route):
            return dict(full.evaluation)
        return self.evaluate_uncached(route)

    @torch.inference_mode()
    def evaluate_uncached(self, route: tuple[str, ...]) -> dict[str, Any]:
        """Execute a complete route even when an identical state was seen before.

        This is reserved for prospective within-unified repeatability controls.
        Normal conversion continues to use the sample-local route cache.
        """
        full = self.initialize_full()
        output = capture_four_action_route(
            self.model,
            self.inputs,
            route,
            prepared_inputs=self.prepared,
            use_cache=True,
        )
        state = self._score_output(
            output,
            correct_targets=full.correct_targets,
            wrong_target=full.wrong_target,
        )
        del output
        return state

    @torch.inference_mode()
    def evaluate_old_binary(self, mask: tuple[int, ...]) -> dict[str, Any]:
        output = binary_greedy_generate(
            self.model,
            self.inputs,
            mask,
            max_new_tokens=int(self.sample["max_new_tokens"]),
            prepared_inputs=self.prepared,
        )
        prediction = self._decode(output.generated_ids)
        score, correct, timed_out = self._correctness(prediction)
        state = {
            "generated_answer": prediction,
            "generated_ids": output.generated_ids[0].detach().cpu().tolist(),
            "correctness_score": score,
            "correct": correct,
            "scoring_timed_out": timed_out,
        }
        del output
        return state
