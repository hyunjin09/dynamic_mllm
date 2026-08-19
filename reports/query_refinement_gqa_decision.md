# Query-Conditioned Visual Refinement Decision

## Decision

Stop the frozen query-refinement direction under this protocol.

The technical premise passed strongly: identical existing visual evidence,
exact unconditioned reconstruction, matched replay compute, unchanged text
state/suffix, valid question-only masks, and deterministic scoring. The
scientific success criteria did not.

- Layer 4 showed a small positive conditioning raw mean (`0.01918`, CI
  `[0.00062, 0.04182]`), but the median (`0.000021`) and 20%-trimmed mean
  (`0.00212`) were near zero, the mean did not reach `0.05`, and target versus
  paired-other question was uncertain.
- Layer 12 was negative, including a target-versus-other CI entirely below
  zero.
- Layer 20 was near zero and uncertain.
- Zero of three anchors passed; two were required.
- Target replay had a net one correctness regression versus baseline at every
  anchor.

Internal challenge: the strongest case against stopping is the positive
layer-4 conditioning CI. It does not reverse the decision because the paired
wrong-question contrast—the direct semantic-specificity control—crossed zero,
the robust central summaries were two orders of magnitude smaller than the
practical threshold, the top 5% materially drove the mean, and selecting layer
4 or relation-like categories now would violate the frozen anti-search rule.

This is a negative frozen-model falsification result, not evidence that all
query-conditioned visual architectures are impossible. It does rule out
proceeding to TextVQA replication, training, deeper replay, or confirmation on
the strength of this operator.

STOP_QUERY_REFINEMENT_DIRECTION
