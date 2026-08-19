# Stage C Prefix-Only Contextual Tokenization Amendment v1

Status: **approved by the user and frozen outcome-blind on 2026-08-05**.

This amendment changes only the secondary answer-prefix robustness condition.
It does not change the source plan, 800-record manifest, primary empty-prefix
accepted-reference score, primary `FULL - WRITE_ONLY` endpoint, structured
nulls, seeds, bootstrap procedure, thresholds, or success criteria.

## Literal Text and Boundary

For accepted normalized answer `a`, the exact secondary literal continuation is:

```text
Answer: <a>
```

The stable token boundary is frozen as:

```text
prompt suffix: "Answer:"
target text:   " " + a
```

Thus the concatenated Unicode text is exactly the originally frozen
`"Answer: " + a`. Target IDs are the suffix of tokenizing the complete literal
`prompt_text + "Answer: " + a` after the token IDs of
`prompt_text + "Answer:"`. Standalone answer token IDs are not reused.

Every component must pass all of these gates before the full sweep:

1. prompt text plus target text equals the exact frozen literal text;
2. prompt token IDs are an exact prefix of combined-text token IDs;
3. the contextual target suffix is nonempty;
4. decoding the combined IDs with special tokens retained and cleanup disabled
   exactly reproduces the literal text;
5. only contextual suffix IDs contribute to the score and zero prompt
   positions contribute;
6. the answer text is absent from the constructed prompt suffix—the lexical
   answer may naturally occur in the original question, as in the primary
   leakage rule;
7. repeated deterministic scoring of every component is numerically identical
   within the unchanged frozen score tolerance.

The contextual target IDs for an accepted answer are derived once and used
identically for FULL and WRITE_ONLY within that prefix condition.

## Secondary Analysis

For each sample, aggregate accepted answers with the same frozen annotation
weights and compute within-prefix contrasts only:

```text
U_prefix_mean = mean_positive_prefix(FULL)
                - mean_positive_prefix(WRITE_ONLY)

U_prefix_seq  = sequence_positive_prefix(FULL)
                - sequence_positive_prefix(WRITE_ONLY)
```

Because target segmentation can differ from the primary empty-prefix
condition, raw per-token likelihood levels are not compared across prefixes.
Report:

- the prefix contrast and its frozen image-clustered confidence interval;
- sign agreement between `U_prefix_mean` and the primary `U_mean`;
- the paired sample effect difference `U_prefix_mean - U_mean` and clustered
  confidence interval;
- the analogous sequence-level results as secondary checks.

This robustness condition cannot replace or rescue the primary endpoint and
does not alter Outcome A/B/C.

## Fail-Closed Rule

If any component fails exact literal reconstruction, token-prefix alignment,
nonempty target span, prompt masking/leakage, or deterministic score
reproduction, stop outcome-blind. Do not filter components, replace samples,
change the literal prefix, or inspect the Stage C primary endpoint.
