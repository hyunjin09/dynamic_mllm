# v3 Query-Invariance Validation

## Computational causal graph

```text
same image + identical system/control prefix
        |
        v
identical vision encoder output and visual token rows H_0[V]
        |
        v
for decoder layer l:
  RMSNorm(H_l) -> Q_l,K_l,V_l
  causal attention: visual query i has zero weight for every later question key j>i
  -> visual attention output
  -> row-wise residual/MLP updates
  -> H_{l+1}[V]
        |
        +--> WRITE_l = H_{l+1}[V] - H_l[V]

text READ_l uses Q_l[text] against K_l[V],V_l[V]
computed from the pre-WRITE H_l state
```

The pinned processor orders system/control tokens, vision-start, contiguous
visual rows, vision-end, question/instruction, and assistant prefix. The
decoder mask is causal. In exact arithmetic, induction over layers therefore
gives identical visual rows and current-layer WRITE for the same image and
preceding prefix, independent of later question content. READ at layer `l`
consumes pre-WRITE visual K/V because all projections are computed from the
same normalized `H_l` before `H_{l+1}` exists.

Evidence: `outputs/stage_a/architecture_causal_graph.md`,
`outputs/stage_a/token_layout.json`, and `interventions/read_path.py`.

## Numerical sanity check

The outcome-blind check used one reserved GQA image with two different
questions and inspected visual states only. It did not score an answer or any
terminal action value.

Under ordinary single-record execution, prompt lengths were `281` and `273`.
Pre-layer visual rows at layer 0 were exactly equal, but BF16 stock-eager
execution produced a post-layer visual maximum difference of `0.0625` at layer
0. The difference accumulated to post-layer `14.0` and WRITE-residual `10.5`
at layer 24. Causal future-attention mass remains zero; the observation is a
shape-dependent finite-precision execution effect, not evidence that future
question tokens are causally visible.

The single allowed diagnostic right-padded both prompts to the common length
`281` without changing their non-padding tokens. With identical execution
shape, pre-layer visual rows, post-layer visual rows, and WRITE residuals were
bitwise identical (`max_abs = 0`, relative RMS `= 0`) at every layer in
`[0,4,8,12,16,20,24]`.

Evidence:

- `outputs/v3_preflight/null_preflight_manifest.json`
- `outputs/v3_preflight/query_invariance_equal_length_diagnostic.json`

## Decision consequence

The formal causal argument is supported, but the unpadded pinned runtime does
not provide numerical query invariance across unequal prompt lengths. A future
Stage C2 would need a prospective fixed-shape/right-padding rule for every
within-image question group. That rule was not part of the inherited protocol
and is not silently adopted here. It requires explicit approval and a small
scoring/parity preflight before Stage C2.
