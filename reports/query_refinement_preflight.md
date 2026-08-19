# Query-Conditioned Refinement Preflight

## Decision

The frozen 12-image/24-question technical preflight **passed** at all three
anchors `[4, 12, 20]`. The authorized 100-image GQA discovery may proceed
without changing the operator, manifest, layers, masks, tolerances, or success
criteria.

## Operator validated

The replay runs the native frozen decoder layer once from captured `H_l`, with
the original token order and MRoPE positions. It discards replay text outputs,
retains only post-layer visual rows, substitutes them into the target's native
`H_(l+1)`, and resumes the unchanged suffix at `l+1`.

- `UNCONDITIONED_REPLAY` uses the native causal mask.
- `TARGET_QUERY_REPLAY` adds only visual-query edges to the target literal
  question's minimal contextual token cover.
- `OTHER_QUERY_REPLAY` uses the paired same-image question states and analogous
  question edges, then inserts the result into the target state.

The pinned slow tokenizer and the fast offset tokenizer produced identical
prompt token IDs. Processor image-placeholder expansion reproduced actual
input IDs exactly. Boundary coverage outside each literal question was empty
or a newline only; no answer or padding token was exposed.

## Gate results

All frozen checks passed:

- baseline repeated prompt logits: exact;
- same-image common-padded visual `H_l` and `H_(l+1)`: bitwise exact;
- unconditioned replay versus native visual output: bitwise exact;
- unconditioned unchanged-suffix logits and accepted-answer scores versus
  baseline: exact;
- deterministic repeated replay scores: exact;
- all replay states and scores: finite;
- B/C/D replay tensor shapes and analytic FLOPs: identical within every
  sample/layer;
- unchanged visual token counts; no answer/padding leakage.

Prospectively frozen activation-plausibility gates also passed. Across replay
rows, the replay/native visual RMS ratio was `0.98643` to `1.00110`, minimum
cosine similarity was `0.99725`, and the largest replay-to-native difference
was `0.30176` times the native visual-WRITE RMS. These are validity diagnostics,
not answer-outcome effects.

## Integrity boundary

The preflight did not aggregate, inspect, serialize, or interpret any absolute
scientific likelihood or conditioning contrast. Its controls contain only
identity, repeatability, finite-state, activation-geometry, leakage, and compute
diagnostics.

The first Slurm attempt failed during CUDA initialization on `node03`, before
sample loading. A single execution-environment repair used the known-working
`node02` configuration; the scientific implementation and frozen protocol were
unchanged.

Evidence:

- `outputs/query_refinement/preflight_v1/summary.json`
- `outputs/query_refinement/preflight_v1/controls.json`
- `outputs/query_refinement/preflight_v1/runtime.json`
- `runs/query_refinement_preflight_v1/retry2.log`
