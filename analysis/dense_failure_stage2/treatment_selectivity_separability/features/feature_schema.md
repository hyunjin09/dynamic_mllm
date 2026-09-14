# Frozen representation schema

- Exact state timing: immediately before the indexed decoder layer under the complete routed prefix.
- `z_R`: 256-wide FP32 output of Experiment-A `read_attention` at its single text-query position.
- `z_W`: 256-wide FP32 output of Experiment-A `write_attention` at its learned write-query position.
- `action_logits`: four FP32 logits in `[FULL, READ_ONLY, WRITE_ONLY, IGNORE]` order from the unchanged Experiment-A action head over `[z_R; z_W]`.
- `nonfull_full_margin`: `max(action_logits[1:]) - action_logits[0]`.
- No layer, dataset, source, Stage-1, visual-count, label, or route-result feature is concatenated to the learned probe inputs.
- Branch extraction mirrors the frozen router forward and is accepted only after exact-logit parity.
