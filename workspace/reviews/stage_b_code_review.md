# Stage B Engineering Review

Date: 2026-08-04  
Scope: reference scoring, official evaluators, prompt-cache intervention path,
greedy continuation, result schema, aggregate/bootstrap analysis, and final
artifacts.

## Outcome

No unresolved blocking implementation defect was found in the finalized Stage B
path.

## Material Findings Resolved Before Final Sweep

1. The initial TextVQA normalizer was simpler than the official EvalAI/VQA
   procedure. It was replaced, unit-tested for number/contraction/punctuation
   behavior, and the validity gate was rerun.
2. The first cached greedy decoder omitted the pinned repetition penalty `1.05`.
   The partial attempt was stopped and isolated; the corrected implementation
   matches Transformers' processor exactly in a unit test and matches standard
   `generate` on both validity datasets.
3. One corrected validity launch had a prompt-ID call-site wiring error. It
   failed before scoring a sample, was repaired, and the subsequent v4 gate
   passed.

## Verification

- 27 unit tests pass.
- Source modules compile successfully.
- Candidate manifest: 400 records, SHA-256
  `de4dada63a677172bb4eafaaba2787ea338750dd8c4e8fb5eeaa5cc820211ed7`.
- Final result: 400 records, SHA-256
  `411cb24899973ad19d7c3186bfa9f88ae59f0a1ca09366b889b57053c32ba4d4`.
- Technical exclusions: 0.
- FULL sequence parity maximum absolute difference: 0.
- FULL generated-token parity: exact at every layer.
- Aggregate output includes nine analysis CSV tables, two relabel tables, one integrity
  manifest, and 15 nonempty SVG plots.
- No compute job remains queued or running.
- Approved source-plan SHA-256 remains
  `d476736dde6d5d7d44ab3e18794ebc3c4e988d703829657271bc285ecd5171d1`.

## Residual Risks (Scientific, Not Engineering Blockers)

- Numerical epsilon is only a noise floor and is not a practical effect cutoff.
- Discovery effects are heterogeneous and selection-adjustment is deferred to a
  held-out structured-null protocol.
- A reference-likelihood Stage C requires explicit amendment of the source-plan
  multiple-choice controls and P0 gate.
