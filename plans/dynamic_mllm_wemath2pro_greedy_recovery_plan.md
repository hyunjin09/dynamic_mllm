# WeMath2.0-Pro Greedy Recovery Label-Extraction Plan

Status: active execution as of 2026-08-18. The user authorized G0--G5 with
four-way GPU parallelism: two GPUs on node06 and two GPUs on node07.

## Objective

Use the supplied frozen greedy Phase-1/Phase-2 search geometry to seek new
valid 28-bit binary visual ON/OFF routes for the WeMath2.0-Pro samples where
the completed hard-cap-400 MCTS search found no valid route.

This is conditional recovery of MCTS failures, not a new dataset-wide yield
estimate and not a replacement for the preserved MCTS cache.

## Frozen recovery population

Select exactly the completed analysis Group D:

```text
current ALL-ON wrong
AND zero valid routes in the completed cap-400 MCTS cache
```

- Records: 2,278.
- Unique image groups: 1,104.
- Source: `outputs/wemath2pro_mcts_label_analysis_v1/`.
- Preserve the eight prospectively technical-invalid source rows outside this
  population.
- Do not expand to current-FULL-correct records or records already having a
  valid MCTS route.

## Frozen execution contract

- Model: Qwen2.5-VL-7B-Instruct snapshot
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Environment: project `.venv`, Transformers 5.3.0.
- Executor: current verified binary visual-routing executor.
- Route: unrestricted 28-bit mask; `1=VISUAL_ON`, `0=TEXT_ONLY`.
- Processing: native/default Qwen image processing, no custom visual-token cap.
- Generation: deterministic greedy, maximum 96 new tokens, frozen direct-answer
  prompt.
- Scoring: frozen WeMath MathRuler accuracy, threshold 1.0, bounded scorer
  timeout; timeout is recorded and scored invalid.
- Existing compatible MCTS mask payloads may be reused only by exact mask and
  must retain their provenance. New masks must be executed under this contract.
- Never mutate or overwrite the completed MCTS cache or the supplied reference
  package.

## Search semantics

The immutable algorithmic reference is
`search/greedy_phase1_phase2_reproduction/`. Its old runtime wrapper is not a
drop-in executor for this project.

### Phase 1

For every recovery record:

1. Evaluate/reuse ALL-ON and ALL-OFF anchors.
2. Run ten 28-step greedy removal traces:
   - early-to-late;
   - late-to-early;
   - center-out;
   - outside-in;
   - six UID-conditioned random orders using seeds 20260714–20260719.
3. At each step, turn the next currently ON layer OFF and accept the candidate
   exactly when:

   ```text
   candidate_score + 1e-9 >= max(ALL_ON_score, current_score)
   ```

4. Save every accepted and rejected request, generated token sequence,
   decoded prediction, raw score, validity, mask, origin, and cache-reuse flag.

Phase 1 permits at most 282 unique requests per sample including anchors,
subject to exact-mask reuse and deduplication.

### Phase 2

After Phase 1 is complete and aggregated:

- choose up to three diverse low-ON successful Phase-1 final masks per sample;
- generate the frozen budget-stratified random masks;
- generate same-budget swaps, add-one/remove-one neighbors, and pairwise
  unions/intersections from successful bases;
- preserve Phase-2 seed 20260720 and the reference request counts;
- save the frozen request manifest before executing new Phase-2 masks.

If no successful Phase-1 final mask exists anywhere, the global budget center
is undefined: stop rather than inventing one. Samples without a successful
base receive only the reference six budget-stratified random requests once the
global center is valid.

## Ordered stages and gates

| Stage | Action | Gate / stop rule |
|---|---|---|
| G0 | Materialize and checksum the 2,278-record recovery manifest. | Exactly 2,278 UIDs and 1,104 image groups; every record is current-FULL-wrong with zero valid cap-400 MCTS routes; source/MCTS checksums pass. |
| G1 | Implement a WeMath adapter around the current executor while preserving the frozen search core as reference-only. | Deterministic unit tests reproduce all ten orders, acceptance, Phase-2 proposal rules, exact deduplication, and provenance serialization. |
| G2 | Run a five-record technical gate only after separate execution approval. | 5/5 native/binary ALL-ON token parity; cached mixed-mask parity; one new mixed mask repeats exact tokens/score; native processing and atomic output pass. |
| G3 | Run Phase 1 over all 2,278 records. | Exactly 2,278 terminal records and 22,780 final traces; zero unresolved errors; resume accepts only atomic contract-valid outputs. |
| G4 | Aggregate Phase 1 and freeze Phase-2 requests. | Phase-1 counts/checksums pass; a global successful-mask budget center exists; every request is frozen before outcomes. |
| G5 | Execute Phase 2, merge exact-mask origins, and audit. | All UIDs reconcile; no cache mutation; exact route/payload linkage; final audit and checksums pass. |

## Output policy

Use a new recovery-specific output root. Preserve separately:

- frozen recovery manifest and checksums;
- Phase-1 per-sample atomic outputs and traces;
- Phase-1 aggregate and successful-mask budget statistics;
- frozen Phase-2 request manifest;
- Phase-2 per-sample outputs;
- combined exact-mask index retaining all search origins;
- completeness/provenance/checksum audit;
- conditional recovery summary.

The raw cache retains every positive and negative evaluated mask. A separate
deterministic diverse supervision view retains at most 50 valid masks per
image-question pair; samples with fewer than 50 retain every valid mask.

Primary descriptive result, after execution approval, is the number and
fraction of the 2,278 MCTS-failure records with at least one newly discovered
valid route. Report Phase-1 recovery and incremental Phase-2 recovery
separately. Retain zero-recovery records.

## Non-goals

- Do not rerun or extend MCTS.
- Do not use POLAR segment-constrained search.
- Do not alter route semantics, scorer, prompt, processor, snapshot, or timeout.
- Do not run the old four-benchmark package unchanged or downgrade the project
  environment to its Transformers 4.57.1 runtime.
- Do not train a router, predictor, probe, controller, or base model.
- Do not treat conditional recovery as dataset-wide prevalence.
- Do not discard negative routes or retain only successful routes.

## Current authorization boundary

The user explicitly authorized the complete bounded search. G0--G2 must pass
before Phase 1. Phase 1 runs as four deterministic global shards split across
two separate two-GPU Slurm jobs on node06 and node07. Phase 2 may start only
after the complete Phase-1 aggregate freezes a single global budget center and
the Phase-2 request manifest. Predictor training remains unauthorized.
