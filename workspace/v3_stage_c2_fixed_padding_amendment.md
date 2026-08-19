# Prospective v3 Stage C2 Common-Padding Amendment

Status: frozen prospectively for a future Stage C2 preflight; Stage C2 is not
authorized or executed by this amendment. This rule is separate from the main
v3 Stage C scoring protocol.

## Reason

With the pinned BF16 stock-eager decoder, the same image run in prompts of
length 281 and 273 had identical layer-0 pre-layer visual states but
shape-dependent numerical divergence in the post-layer visual state and WRITE.
The layer-0 maximum absolute difference was `0.0625` and accumulated to a
post-layer difference of `14.0` and WRITE difference of `10.5` at layer 24.
Causal future-token attention mass remained zero. A single outcome-blind
diagnostic right-padded both inputs to 281 and restored bitwise equality of
pre-layer visual states, post-layer visual states, and WRITE at all seven
validated layers.

Evidence: `workspace/v3_query_invariance_validation.md` and
`outputs/v3_preflight/query_invariance_equal_length_diagnostic.json`.

## Frozen group rule

For every future same-image question group:

1. Construct each literal prompt independently with the pinned processor,
   tokenizer, and chat template. Do not edit, truncate, or normalize its
   non-padding token sequence.
2. Let the common prompt length be the maximum unpadded prompt length within
   that image group. Append only the pinned tokenizer's padding token on the
   right of shorter prompts until all group members have that length.
3. Keep the visual prefix and its image-token positions identical across group
   members. No padding may be inserted before or inside the visual prefix.
4. Set attention-mask entries for appended padding to zero. Padding positions
   may contribute neither attention context nor answer scoring.
5. Append/align teacher-forced answer targets only after the common padded
   prompt boundary. Mask every prompt, image, and padding position from the
   answer-token loss. The answer text must not appear in the prompt.
6. Execute every member of the group with the identical tensor shape, decoder
   backend, precision, and layer hooks. The literal decoded non-padding prompt
   must equal the unamended prompt exactly.

## Entry checks for a future Stage C2

Before any Stage C2 terminal action value is opened, verify for every group:

- equal tensor shapes and identical visual-token positions;
- unchanged literal non-padding prompts;
- zero attention-mask values on all right-padding positions;
- nonempty, correctly masked answer spans and no answer leakage;
- exact equality of visual pre-layer state and current-layer WRITE for every
  question at every frozen layer;
- instrumented-FULL parity and deterministic score reproduction under the
  common padded layout.

Any failure stops Stage C2 before outcome inspection. Common padding does not
modify the main Stage C endpoint, its prompt/scoring rules, its nulls, or its
manifest construction.
