# Prospective Binary-Route Architecture Pivot Proposal

## Status

Proposal only. P11 did not authorize implementation, training, or evaluation of
a new route representation.

## Evidence motivating the proposal

Under exact valid-set NLL, aligned questions receive substantially better
valid-set probability mass than deterministic within-dataset shuffled questions
(`14.8699` versus `15.5089` set-NLL). Yet the selected direct factorized head
decodes ALL-ON for 98% of route-validation inputs and 95% of the bounded
execution set. Its three non-FULL execution masks are uncached and remain
wrong. This separates probability-level input signal from coherent complete-mask
decoding.

The result does not prove that factorization is the unique cause. The matched
bias heads may be underoptimized and P11 is only a two-epoch smoke. It does,
however, make a structured-head comparison more defensible than full-scaling
the current almost-constant decoder.

## Proposed controlled next experiment

If explicitly approved, compare the unchanged P11 exact-set model against one
POLAR-compatible structured mask representation derived from the same frozen
28-bit masks. Keep the following fixed:

- regenerated GQA/TextVQA/ChartQA labels and image-group split;
- max-50 selected valid masks per input;
- frozen Qwen3 question encoder and POLAR layer encoder;
- optimizer, initialization seed, training budget, checkpoint rule, and actual
  executor;
- valid-set supervision weights and complete-mask evaluation.

Change only the route head/decoding representation so it can express coherent
cross-layer patterns. The primary candidate is the already derived canonical
POLAR segment representation, mapped losslessly back to a 28-bit mask before
execution. No beam search, compute penalty, RL, new data, or encoder change
should be added in the first comparison.

## Admission checks before any training

1. Prove lossless mask ↔ segment round-trip on every selected supervision mask.
2. Match the direct head's training identities, weights, and budget exactly.
3. Define one deterministic top-1 decoder and freeze it before outcomes.
4. Reuse the global/dataset priors and within-dataset shuffle diagnostics.
5. Execute every selected mask, including uncached masks.

## Success and kill criteria

Success requires more than lower set-NLL: aligned input must beat shuffled
input, decoded masks must be materially nonconstant, and bounded execution must
show useful non-FULL selection without unacceptable FULL-correct regressions.

Kill the pivot if the structured head also collapses to a constant, if decoded
diversity does not improve actual execution, or if gains require changing the
encoder/data/compute objective simultaneously. In those cases the current
evidence would favor insufficient learnable routing headroom rather than a
factorization-only bottleneck.
