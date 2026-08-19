# Frozen Stage C Accepted-Reference Scoring Specification

Status: frozen before any Stage C intervention score was computed.

## Runtime and Prompt

- Model/processor snapshot: Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Transformers: `4.51.3`; processor `Qwen2_5_VLProcessor`; slow tokenizer
  `Qwen2Tokenizer` (`use_fast=False`).
- Chat-template SHA-256:
  `a0bc6f6fc7a29a80017a433e8f03a1cc1236e838a944a2d034295a60c4f2fddb`.
- User text is exactly:

  ```text
  {question}
  Answer the question using a single word or phrase.
  ```

- `apply_chat_template(..., tokenize=False, add_generation_prompt=True)` is
  used with one image item followed by that text item.
- The answer prefix is the empty string: the first answer token immediately
  follows the rendered assistant-generation prompt.
- The same rendered prompt, tokenizer, image processing, accepted-answer set,
  and weights are used in every state. Target answers are never placed in the
  intervention prompt.

## Accepted Answers

`scoring.benchmark_metrics.normalize_textvqa` is the frozen official
EvalAI/VQA-style normalizer. Its containing source file has SHA-256
`bab93464c86e3ee485d32ef0e943f8d5b55b6160365ed9cfa570d0ac29d86975`.

For each record:

1. normalize all ten human answers;
2. discard normalized empty strings;
3. aggregate duplicate normalized strings with a `Counter`;
4. assign each unique string its annotation-frequency weight
   `count / total_nonempty_count`;
5. require every weight to be positive and the weights to sum to one within
   absolute tolerance `1e-9`.

No alias is selected by model score, state, or outcome.

## Answer-Token Scores

For accepted answer `a=(y_1,...,y_T)`:

```text
S_i(a,state) = sum_t log p(y_t | image, prompt, y_<t, state)
m_i(a,state) = S_i(a,state) / T
```

Only the `T` answer tokens contribute. Prompt positions, image tokens, and EOS
do not contribute to either score. EOS-inclusive scoring is not used in the
primary or secondary Stage C analysis.

The accepted-reference scores are:

```text
sequence_positive_i(state)
  = logsumexp_a(log(w_a) + S_i(a,state))

mean_positive_i(state)
  = logsumexp_a(log(w_a) + m_i(a,state))
```

`mean_positive` is primary. `sequence_positive` is secondary. The scoring
implementation is `scoring/reference_likelihood.py`, SHA-256
`654860d1874d226369b7bffa50802ff83a8d0da211d12fbc67a586f3c3d0976e`.

The frozen primary sample statistic remains:

```text
U_i = mean_positive_i(FULL) - mean_positive_i(WRITE_ONLY)
```

## Span and Leakage Gate

For every accepted answer, all of the following are required before selection:

- standalone tokenization is nonempty;
- tokenizing `prompt_text + answer` preserves the complete prompt-token prefix;
- the remaining suffix IDs equal the standalone answer IDs exactly;
- zero prompt positions contribute to answer likelihood;
- prompt length is at most 4,861 tokens;
- the image-token range is nonempty and contiguous.

An answer string or token may naturally occur inside the question; that is not
target leakage because no prompt position is scored as the target. The leakage
rule concerns target placement and score-span alignment, not lexical overlap
with a legitimate question.

## Frozen Source and Smoke Evidence

- Config: `configs/stage_c_entry.yaml`.
- Manifest formatting/audit driver:
  `experiments/prepare_stage_c_manifest.py`.
- Unit conformance: `tests/test_reference_likelihood.py` and
  `tests/test_stage_c_manifest.py`.
- The entry-gate report records the final manifest-wide span audit and pinned
  processor smoke result. No Stage C likelihood contrast is computed during
  that smoke validation.

## Frozen Secondary Robustness and Wrong-Answer Controls

These checks are secondary and cannot replace or rescue the primary endpoint.

- Accepted-answer aggregation robustness: recompute `mean_positive` with
  equal weight on each unique normalized accepted answer. Report its endpoint
  and clustered 95% CI alongside the annotation-frequency primary result.
- Answer-prefix robustness: append the literal unscored assistant prefix
  `Answer: ` after the rendered generation prompt, then retokenize and verify
  the same exact answer-span rules. Report the frequency-weighted endpoint and
  clustered 95% CI. The prompt without a prefix remains primary.
- A robustness check is directionally consistent only when its point estimate
  is at or below zero. Robustness failure weakens interpretation and cannot be
  repaired by choosing a different prefix or aggregation after results are
  visible.
- Frozen FULL-wrong-answer contrast: deterministically generate and freeze the
  original FULL greedy answer before inspecting intervention contrasts. For
  records on which that answer is strictly wrong, score that exact normalized
  answer under both FULL and WRITE_ONLY. At each state form accepted-reference
  `mean_positive` minus the frozen-wrong-answer mean log-likelihood, and report
  the FULL-minus-WRITE_ONLY change in this margin with image-clustered
  uncertainty. Do not regenerate or select a wrong answer per state.
