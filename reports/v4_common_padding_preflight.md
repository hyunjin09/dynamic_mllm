# v4 GQA Common-Padding Entry Gate

**Decision:** `PASS`

The outcome-blind entry gate used the 12 image groups frozen in
`outputs/v4_discovery/manifest/v4_gqa_discovery_manifest_v1.jsonl` (24 natural
questions) and the pinned seven-layer grid `[0,4,8,12,16,20,24]`. No terminal
four-action value was retained or aggregated during this gate.

## Frozen data and layout

- Discovery manifest: 240 questions from 120 unique GQA images, exactly two
  questions per image.
- Manifest SHA-256:
  `089b55c2e704c47d8c2cf9d516821d3f02dabf7e4dac47727f7a41153b267164`.
- Selection: 60 image pairs with resolved disjoint scene-object evidence and
  60 metadata-matched comparison pairs.
- Inspected-data/calibration overlap: zero by record and image identifiers.
- Common right-padding changed no non-padding token and was masked from both
  attention and accepted-answer scoring.

## Gate evidence

All required checks passed:

- visual-token indices, multimodal positions, and common tensor shapes were
  identical within every same-image pair;
- pre-layer visual rows, post-layer visual rows, and visual WRITE residuals
  were bitwise identical (`max_abs = 0`) at all seven layers;
- visual-query attention to later question or padding keys was exactly zero;
- instrumented `FULL` matched the unmodified common-padded path;
- READ and WRITE save/reinsert reconstruction identities passed;
- every one of the four actions executed deterministically with finite scores;
- accepted-answer spans matched the frozen IDs, were nonempty, and included no
  prompt or padding positions.

Across 1,176 identity/control records, the 99th-percentile absolute score
difference was zero for sequence and per-token scoring. The prospectively
frozen floors therefore remain:

- per-token epsilon: `1e-6` nats/token;
- sequence epsilon: `1e-5` nats.

Machine-readable evidence is in
`outputs/v4_discovery/preflight/v4_common_padding_preflight_v1.json` and
`outputs/v4_discovery/preflight/v4_common_padding_preflight_controls_v1.json`.

## Infrastructure note

The first submission on node03 failed before CUDA initialization and before any
model forward or score. An unchanged retry on node02 completed the gate. This
was an execution-environment failure, not a failed architectural or numerical
validity condition; the original failure record is preserved under
`runs/v4_preflight_20260807/failure_attempt1.json`.
