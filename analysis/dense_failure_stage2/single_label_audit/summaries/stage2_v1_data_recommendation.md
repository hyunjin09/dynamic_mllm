# Stage-2 V1 data recommendation

## Recommended minimal loader contract

- Use only **Corpus A preservation + Corpus B single corrective** from the Phase-56 contract `6489a0b39af3efb22bcb527d3b43c474dc7bcb01cc453445aab45f420eb9905b`.
- Make the W sampling unit the **sample**, not the retained route: choose one successful single route uniformly for each W sample when it is drawn (resampling across epochs is allowed).
- For that route, emit exactly the corrective state plus up to **two pre-intervention FULL** and **two post-intervention FULL** states (`S3 + S1`). This preserves the central timing fact—many rescues are delayed—while reducing expected W FULL share from 0.962 to 0.785.
- Draw triggered-C and W samples at **C:W = 1:2**. For C, sample from the safe FULL suffix while ensuring the trigger state remains represented. This makes preservation visible without letting only 39 C samples occupy half of all sample draws; it repeats each C sample about 8.95x as often as each W sample.
- Keep the target single-label four-way CE: `FULL`, `READ_ONLY`, `WRITE_ONLY`, `IGNORE`, with routed feature + layer index as input. Preserve UID, route ID, trigger layer, intervention layer, observed-successful-action metadata, contract/schema/source hashes, and tensor-row provenance.
- Retain multi-valid alternatives in metadata and expose them through uniform route resampling. **Defer a multi-label loss** until a controlled follow-up; do not collapse alternatives to one arbitrary canonical route.

## Internal challenge and runner-up

The strongest objection is that S1 still gives roughly 0.785 FULL targets and revisits exact states with multiple valid route labels. The simpler runner-up is sample-balanced `S2 K=2`, which is more action-balanced but discards explicit before/after timing context. Because 0.703 of W samples are delayed-only, preserving local timing context is the more defensible V1 choice. Confidence is **medium**: the recommendation is based on corpus geometry, not training or rollout evidence.

## Explicit exclusions

Do not add Corpus C/MCTS, weighted losses, utility heads, `z_fail`, EMA/persistence logic, RL, validation/test labels, Stage-1 changes, or dataset-specific policies in V1. No model is trained by this recommendation.
