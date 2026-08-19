# Stage B Reference-Likelihood Implementation Report

Status: validity passed and the corrected 400-record discovery sweep completed
with zero exclusions. This report describes the frozen implementation; the
scientific interpretation is in `reports/stage_b_conclusion.md`.

## Scoring Procedure

For every accepted answer, the scorer starts from a state-specific prompt cache.
The last prompt logit scores the first answer token; the remaining answer tokens
except the last are teacher-forced through the unchanged model and their logits
score the next tokens. Float32 log-softmax values are gathered only at answer
token IDs. Their sum is the primary within-sample score and the sum divided by
answer-token count is the cross-sample robustness score. No prompt, image, or
EOS token contributes.

The intervention hook is active only while constructing the prompt cache. It is
removed before teacher-forced answer tokens and greedy continuation enter the
model.

## Prompt and Target Formatting

- Input prompt: the unchanged manifest prompt, containing the open-ended
  question followed by `Answer the question using a single word or phrase.`
- Chat formatting: the pinned Qwen processor template with one image followed
  by the prompt and `add_generation_prompt=True`.
- Answer prefix: none beyond the template's assistant-generation prefix.
- Target: normalized canonical GQA answer or each normalized accepted TextVQA
  answer, tokenized with `add_special_tokens=False`.
- Boundary audit: the prompt token IDs must be an exact prefix of prompt plus
  answer, and standalone answer IDs must equal that suffix. Any failure is a
  predefined technical exclusion evaluated before intervention outcomes.

## Accepted-Answer Aggregation

GQA uses one canonical lowercased, punctuation-insensitive reference with
weight one.

TextVQA applies the official EvalAI/VQA answer normalization, counts duplicate
normalized human annotations, and divides by the total accepted annotations to
obtain weights summing to one. For either sequence or mean answer score (s_a),
the aggregate is computed stably as

\[
\max_a(s_a+\log w_a) +
\log\sum_a \exp(s_a+\log w_a-\max_b(s_b+\log w_b)).
\]

The normalized answers, weights, token IDs, token log-probabilities, sequence
scores, and mean scores are retained separately in each sample record.

## Intervention Hooks

- READ: a pre/post hook on `decoder.layers[l].self_attn`. With the original
  fixed attention weights, the projected visual-value contribution is separated
  from the projected non-visual contribution. READ OFF subtracts only the
  visual-value path on non-visual query rows. Non-visual attention paths and
  the original softmax normalization are preserved.
- WRITE: the output of `decoder.layers[l]`, after that layer's residual and MLP.
  WRITE OFF restores visual rows to the cached pre-layer hidden state while
  retaining the current layer output on every non-visual row.
- State start: every factorial branch clones the same captured pre-layer state
  and the same unmodified prefix-layer KV cache. The target and suffix caches
  are rebuilt for that state, after which the suffix and answer continuation
  are unchanged.

## Pinned Runtime and Greedy Evaluation

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`, revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Decoder: Transformers stock eager; vision encoder: SDPA; model dtype:
  bfloat16; score log-softmax: float32.
- Processor/tokenizer: pinned `Qwen2_5_VLProcessor` / `Qwen2Tokenizer`, slow
  image processor explicitly retained.
- Greedy decoding: `do_sample=False`, `max_new_tokens=32`, both pinned EOS IDs,
  and the pinned repetition penalty `1.05`. Cached greedy applies the same
  prompt-plus-generated-token penalty as Transformers generation.
- Full serialized runtime, chat template, and generation config:
  `outputs/stage_b_validity_v4/runtime.json`.

## FULL Parity and No-Op Noise

The corrected validity probe used one frozen GQA and one frozen TextVQA record
at all eight layers. All seven gates passed:

- instrumented FULL prompt logits, sequence scores, and mean scores matched the
  unmodified path within tolerance, and generated token IDs matched;
- repeated scores for all four states were exactly equal;
- READ and WRITE save-and-reinsert identities matched through the hook and
  unchanged suffix;
- cached greedy token IDs matched standard Transformers `generate` after the
  pinned repetition penalty repair;
- token layouts and target spans passed on both datasets.

Across 114 no-op comparisons, sequence and mean absolute-difference p99 were
both zero. The prospectively declared rule
`max(predeclared floor, empirical no-op p99)` therefore froze
`epsilon_sequence=1e-5` and `epsilon_mean=1e-6` before the corrected full sweep.
Evidence is in `outputs/stage_b_validity_v4/`.

## Superseded Artifacts

`outputs/stage_b_attempt_01_raw_greedy/` contains 109 partial records from a
stopped attempt whose teacher-forced likelihoods were valid but whose secondary
greedy decoder omitted the pinned repetition penalty. Those records are not
used in any Stage B analysis. Earlier validity attempts remain preserved and
are likewise superseded by v4.
