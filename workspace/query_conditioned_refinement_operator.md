# Frozen Query-Conditioned Visual Refinement Operator

## Status and scope

This is the prospective operator for the authorized, bounded GQA discovery.
The pinned Qwen2.5-VL-7B-Instruct model remains frozen. The operator neither
re-encodes the image nor changes, selects, adds, or removes visual tokens. It
does not reopen the v2-v4 local READ/WRITE routing program.

The frozen anchors are decoder layers `[4, 12, 20]`: coarse early/middle/late
nonterminal locations on the previously validated sparse grid. They were
selected from architecture and suffix coverage before any new outcome was
computed, not from prior layerwise effect maxima.

## Exact causal boundary

For a target layer `l`, the ordinary dense prompt pass captures:

- pre-layer state `H_l`;
- native post-layer state `H_(l+1)`;
- additive 4-D causal/padding mask;
- the original Qwen multimodal RoPE embeddings and positions;
- the dense prompt K/V cache.

All same-image question pairs use the approved common right-padding rule. The
visual-token indices, positions, tensor shapes, `H_l` visual rows, and native
visual WRITE are therefore identical across the pair.

One replay call runs the unmodified frozen decoder layer `l` from a copied
`H_l`. It preserves the stock sequence order, pre-RMSNorm, Q/K/V and output
projections, GQA expansion, MRoPE, attention softmax, residual path,
post-attention RMSNorm, MLP, and second residual. Only the replay attention mask
may add visual-query edges to question-token keys. All replay text outputs are
discarded. Only post-layer visual rows are retained as `V'_(l+1)`.

The retained rows replace visual rows in the target question's native
`H_(l+1)`; all target text/control rows remain exactly native. The prompt cache
through layer `l` is the target's unchanged dense cache: changing attention
edges changes visual query outputs but not the pre-attention K/V written to
that layer's cache. Execution resumes at layer `l+1` and uses the unchanged
dense suffix and accepted-answer scorer.

```text
same image encoding and common-padded dense prefix -> H_l
                         |
                         +-> one frozen layer-l replay
                               (retain visual output rows only)
                         |
native target H_(l+1) -- replace visual rows with V'_(l+1)
                         |
                  unchanged layers l+1..27
                         |
                 identical answer scoring
```

This output-boundary design was selected after a research-control review. It
avoids the more off-manifold alternative of applying layer `l` sequentially
twice. Its limitation is explicit: unconditioned replay is a redundant native
reconstruction, so the experiment tests the added question-to-visual edge, not
generic sequential depth.

## Frozen variants

- `BASELINE`: original dense frozen prompt and suffix.
- `UNCONDITIONED_REPLAY`: replay with the original causal mask. Visual replay
  must equal native `H_(l+1)` visual rows bitwise and reproduce baseline suffix
  logits/scores.
- `TARGET_QUERY_REPLAY`: replay from the target question's `H_l`, adding edges
  only from visual query rows to the minimal contextual token cover of the
  target literal question.
- `OTHER_QUERY_REPLAY`: replay from the paired same-image question's `H_l`,
  adding the analogous edges to that other literal question. Its refined visual
  rows are inserted into the target question's native `H_(l+1)`.

The minimal contextual cover is derived with offset mapping, but the fast and
pinned slow tokenizers must produce identical IDs and the processor's single
image placeholder expansion must reproduce the actual prompt IDs exactly. Any
characters covered outside the literal question must be whitespace only. No
instruction, assistant-prefix, answer, or padding row is deliberately exposed.

## Equal-compute accounting

`UNCONDITIONED_REPLAY`, `TARGET_QUERY_REPLAY`, and `OTHER_QUERY_REPLAY` all run
one full frozen layer call at the identical common-padded shape. Different
masks do not alter dense eager matrix dimensions. They therefore have identical
Q/K/V, attention, output-projection, MLP, normalization, and residual compute.
The exact per-pair MAC/FLOP formula and dimensions are serialized for audit.
Only visual rows are retained, but discarded row computation is intentionally
not optimized away in this experiment.

`BASELINE` has no replay call and is secondary for causal interpretation.

## Mandatory preflight gates

On 12 outcome-blind geometry-covering manifest images, all anchors must pass:

- original baseline and accepted-answer scoring parity;
- exact same-image visual input and position identity;
- exact `UNCONDITIONED_REPLAY` visual reconstruction;
- baseline-equivalent unconditioned suffix logits/scores within frozen
  tolerances;
- valid literal-question span, no answer/padding visibility, unchanged token
  counts, and no answer-score leakage;
- finite replay states and scores;
- deterministic repeated scores within `1e-5`;
- identical replay FLOPs for all three replay branches;
- replay/native visual RMS ratio in `[0.5, 2.0]`, cosine at least `0.5`, and
  replay-to-native difference no more than four times the native visual WRITE
  RMS.

These activation gates are prospective severe-artifact checks, not outcome
selection rules. Any failure stops before the 100-image scientific sweep.

## Frozen discovery decision rule

Per anchor, the primary contrasts use per-token accepted-reference likelihood:

`Delta_condition = TARGET_QUERY_REPLAY - UNCONDITIONED_REPLAY`

`Delta_target = TARGET_QUERY_REPLAY - OTHER_QUERY_REPLAY`.

An anchor supports each contrast only if its image-clustered 95% bootstrap CI
for the mean lies above zero, the mean reaches `0.05` nats/token, the median and
20% trimmed mean are positive, and more than `55%` of samples are positive.
The direction proceeds only if both contrasts pass at at least two of the three
preselected anchors, no preflight gate fails, and target replay has no net
correctness regression versus baseline or either replay control at those
anchors. A one-anchor or raw-mean-only result is a kill outcome.

These are frozen-model discovery criteria. Passing would not establish
efficiency, routing, deployability, a harmful operation, or generality.
