from __future__ import annotations

import math
import unittest

import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from experiments.stage_b_reference_likelihood import apply_repetition_penalty
from scoring.reference_likelihood import (
    AcceptedAnswer,
    AnswerTokenScore,
    accepted_answers,
    aggregate_accepted_scores,
    factorial_effects,
    score_answer_token_logits,
    weighted_logsumexp,
)


class AcceptedAnswerTests(unittest.TestCase):
    def test_gqa_uses_one_normalized_canonical_answer(self) -> None:
        answers = accepted_answers(
            {"benchmark": "gqa", "answer": "  Blue. ", "all_answer_norms": None}
        )

        self.assertEqual([(row.text, row.weight) for row in answers], [("blue", 1.0)])

    def test_textvqa_aggregates_normalized_annotation_frequency(self) -> None:
        answers = accepted_answers(
            {
                "benchmark": "textvqa",
                "answer": "Messbecher",
                "all_answer_norms": ["Messbecher ", "messbecher", "measuring glass"],
            }
        )

        self.assertEqual([row.text for row in answers], ["messbecher", "measuring glass"])
        self.assertAlmostEqual(answers[0].weight, 2 / 3)
        self.assertAlmostEqual(answers[1].weight, 1 / 3)
        self.assertAlmostEqual(sum(row.weight for row in answers), 1.0)


class ReferenceScoreTests(unittest.TestCase):
    def test_cached_greedy_repetition_penalty_matches_transformers(self) -> None:
        input_ids = torch.tensor([[0, 1, 0]])
        scores = torch.tensor([[2.0, -2.0, 1.0, -1.0]])
        expected = RepetitionPenaltyLogitsProcessor(2.0)(input_ids, scores.clone())
        actual = apply_repetition_penalty(scores, input_ids, 2.0)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_weighted_logsumexp_is_stable(self) -> None:
        value = weighted_logsumexp([-1000.0, -1001.0], [0.75, 0.25])
        expected = -1000.0 + math.log(0.75 + 0.25 * math.exp(-1.0))
        self.assertAlmostEqual(value, expected)

    def test_answer_span_uses_prompt_last_logit_then_continuation_logits(self) -> None:
        prompt_next = torch.tensor([0.0, 4.0, 0.0])
        continuation = torch.tensor([[0.0, 0.0, 3.0], [2.0, 0.0, 0.0]])
        result = score_answer_token_logits(
            prompt_next_logits=prompt_next,
            continuation_logits=continuation,
            answer_ids=torch.tensor([1, 2, 0]),
        )

        expected = [
            torch.log_softmax(prompt_next.float(), dim=-1)[1].item(),
            torch.log_softmax(continuation[0].float(), dim=-1)[2].item(),
            torch.log_softmax(continuation[1].float(), dim=-1)[0].item(),
        ]
        self.assertEqual(result.token_ids, [1, 2, 0])
        self.assertEqual(len(result.token_logprobs), 3)
        for actual, wanted in zip(result.token_logprobs, expected):
            self.assertAlmostEqual(actual, wanted)
        self.assertAlmostEqual(result.sequence_logprob, sum(expected))
        self.assertAlmostEqual(result.mean_logprob, sum(expected) / 3)

    def test_factorial_effects_keep_conditional_effects_and_interaction(self) -> None:
        effects = factorial_effects(
            {"IGNORE": -5.0, "READ_ONLY": -4.0, "WRITE_ONLY": -6.0, "FULL": -3.0}
        )

        self.assertEqual(effects["read_w0"], 1.0)
        self.assertEqual(effects["read_w1"], 3.0)
        self.assertEqual(effects["write_r0"], -1.0)
        self.assertEqual(effects["write_r1"], 1.0)
        self.assertEqual(effects["interaction"], 2.0)

    def test_accepted_answer_aggregation_applies_same_weights_to_sum_and_mean(self) -> None:
        answers = [AcceptedAnswer("a", 0.75), AcceptedAnswer("b", 0.25)]
        scores = [
            AnswerTokenScore([1], [-2.0], -2.0, -2.0),
            AnswerTokenScore([2, 3], [-1.0, -2.0], -3.0, -1.5),
        ]

        aggregate = aggregate_accepted_scores(answers, scores)

        self.assertAlmostEqual(
            aggregate["sequence_logprob"], weighted_logsumexp([-2.0, -3.0], [0.75, 0.25])
        )
        self.assertAlmostEqual(
            aggregate["mean_logprob"], weighted_logsumexp([-2.0, -1.5], [0.75, 0.25])
        )


if __name__ == "__main__":
    unittest.main()
