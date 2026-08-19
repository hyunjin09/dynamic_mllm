from __future__ import annotations

import unittest

from scoring.contextual_reference_likelihood import contextual_continuation


class CharacterTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return type("Tokens", (), {"input_ids": [ord(char) for char in text]})()

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)


class BoundaryChangingTokenizer(CharacterTokenizer):
    def __call__(self, text: str, add_special_tokens: bool = False):
        tokens = super().__call__(text, add_special_tokens=add_special_tokens)
        if text.endswith("Answer:"):
            tokens.input_ids[-1] = 999
        return tokens


class ContextualContinuationTests(unittest.TestCase):
    def test_moves_boundary_space_into_contextual_target(self) -> None:
        continuation = contextual_continuation(
            CharacterTokenizer(),
            prompt_text="assistant\nAnswer:",
            continuation_text=" blue",
            expected_literal_text="assistant\nAnswer: blue",
        )

        self.assertEqual(continuation.target_text, " blue")
        self.assertEqual(
            "".join(chr(token_id) for token_id in continuation.target_token_ids),
            " blue",
        )
        self.assertEqual(continuation.target_token_count, 5)
        self.assertTrue(continuation.prompt_is_combined_prefix)
        self.assertTrue(continuation.decoded_text_exact)
        self.assertEqual(continuation.prompt_positions_contributing_to_score, 0)

    def test_rejects_contextual_tokenization_when_prompt_boundary_changes(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a token prefix"):
            contextual_continuation(
                BoundaryChangingTokenizer(),
                prompt_text="Answer:",
                continuation_text=" blue",
                expected_literal_text="Answer: blue",
            )

    def test_rejects_empty_contextual_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            contextual_continuation(
                CharacterTokenizer(),
                prompt_text="Answer:",
                continuation_text="",
                expected_literal_text="Answer:",
            )


if __name__ == "__main__":
    unittest.main()
