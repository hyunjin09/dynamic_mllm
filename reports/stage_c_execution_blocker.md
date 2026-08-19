# Stage C Frozen-Sweep Execution Blocker

## Status

**STOPPED outcome-blind before the full sweep.** No Stage C reference score,
intervention effect, structured-null effect, greedy outcome, or primary
endpoint was computed or inspected.

The source-plan and manifest checksums still match their frozen values:

- source plan: `d476736dde6d5d7d44ab3e18794ebc3c4e988d703829657271bc285ecd5171d1`
- manifest: `e3e9e08329fa626bc75706fba6623357f9ca05140bae1f138c98b9cd26e45357`

## Exact failed condition

The frozen answer-prefix robustness check requires appending the literal
unscored prefix `Answer: ` and then satisfying the same exact token-span rules
as the primary empty-prefix score. Manifest-wide preflight found that 1,834 of
2,279 accepted-answer components fail the required identity:

```text
tokenize(prefixed_prompt + answer)[:len(tokenize(prefixed_prompt))]
    == tokenize(prefixed_prompt)
```

The preflight therefore stopped before launching any of the eight Stage C GPU
shards. There is no `outputs/stage_c/stage_c_results_v1.jsonl`.

## Observation and diagnosis

- Confirmed observation: 1,834 components fail both prompt-prefix preservation
  and standalone-answer/suffix identity; zero components have empty answer
  spans.
- Diagnosis: **supported**. The literal trailing space in `Answer: ` merges at
  the tokenizer boundary with many following answers. Consequently, the
  tokenized prefixed prompt alone is not a prefix of the tokenized combined
  text. This is a target-boundary issue in the frozen robustness procedure,
  not evidence about the Stage C effect.
- Ruled out: empty accepted answers, modification of the 800-record manifest,
  a failure of the already frozen empty-prefix primary spans, and any
  outcome-dependent exclusion.

Concrete diagnostic evidence is in
`outputs/stage_c/preflight/stage_c_prefix_span_diagnostic_v1.json`; the direct
exception is in `outputs/stage_c/failures/stage_c_preflight_failure.json`.

## Why execution cannot continue silently

Continuing requires changing one frozen secondary scoring detail. The valid
implementation choices have different target-token semantics:

1. Contextual-suffix scoring: tokenize the complete
   `prefixed_prompt + answer`, derive the answer span from the combined token
   sequence, and score those contextual target IDs. This preserves the literal
   text but relaxes standalone-answer token-ID identity.
2. Token-concatenation scoring: concatenate separately tokenized prefix and
   standalone answer IDs. This preserves the existing standalone target IDs
   but is not the tokenizer output for the literal combined text.
3. Remove the answer-prefix robustness check. This leaves the primary endpoint
   unchanged but drops a required frozen secondary analysis.

The smallest scientifically coherent resolution is option 1, limited to the
secondary prefix-robustness analysis. It does not alter the primary endpoint,
manifest, nulls, or bootstrap, but it does amend the frozen span rule and
therefore requires explicit approval.

No choice was made after inspecting outcomes. The source plan's scientific
stop condition has not been reached; this is a pre-execution protocol blocker.
