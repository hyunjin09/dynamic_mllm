# Implementation Plan: Exact Sequential Four-Action Label Conversion

## Overview

Execute `plans/4way_labeling_3.md` over the frozen five-dataset binary-route
inventory. Reuse the unified executor, evaluator, sample runtime, source
manifest, and dynamic worker queue, but replace the superseded purification and
beam policy with fixed early-to-late sequential branching that retains every
evaluator-correct W2C branch. Preserve C2C routes mechanically as FULL/IGNORE,
run an isolated 8-sample smoke on eight workers, and gate the resumable
16-worker full conversion on that smoke.

## Architecture Decisions

- Keep the frozen `source_inventory_v1` manifest byte-identical; do not rerun
  MCTS or rebuild the source population.
- Add a contract-isolated sequential conversion core and runner instead of
  changing historical beam-conversion artifacts in place.
- For each W2C source route, process only its original OFF layers once in
  ascending layer order. FULL short-circuits partial tests when correct;
  otherwise retain every correct READ_ONLY/WRITE_ONLY branch or the known-
  correct IGNORE branch when neither partial action works.
- Never rank, prune, cap, or combine branches without complete-route execution.
  Branch counts are evidence; a real explosion is a stop condition, not a
  reason to introduce an unapproved beam.
- For C2C, emit the replay-valid mechanical FULL/IGNORE source route unchanged.
- Reuse one model per worker, one sample-local exact-route cache, and the
  launch-scoped dynamic queue. Use 8 workers/8 GPUs for smoke and 16 workers/8
  GPUs for full.
- Write only under `datasets/mcts_labels_4action/sequential_branching_v1/` and
  `analysis/4action_sequential_label_conversion/`.

## Task List

### Phase 1: Evidence and contract isolation

- [x] Record the failed beam-pilot evidence and cancel jobs 1609/1610.
- [ ] Freeze the 8-sample smoke manifest and new sequential execution contract.

### Phase 2: Exact branching core

- [ ] Add RED tests for FULL restoration, one-sided partial restoration,
  IGNORE fallback, both-partial branching, later-layer branch context, C2C
  preservation, caching, and no beam/pruning fields.
- [ ] Implement the minimum sequential branching core and make focused tests
  green.
- [ ] Add sample runner output, atomic resume, topology-specific worker checks,
  deduplication, and source provenance.

### Checkpoint: CPU/synthetic gate

- [ ] Focused sequential-label tests pass.
- [ ] Complete active `tests/` suite passes.
- [ ] Static audit finds no beam, margin/cost selection, branch cap, or MCTS in
  the new conversion contract.

### Phase 3: Real-data smoke

- [ ] Run exactly 8 deliberate samples on 8 GPUs with one process/GPU.
- [ ] Rerun with `--resume` and prove completed outputs are unchanged.
- [ ] Audit executor parity, replay, branch truth-table paths, joint
  correctness, cache reuse, deduplication, worker isolation, and checksums.

### Phase 4: Full conversion and reporting

- [ ] After the smoke passes, launch all 12,278 samples with 16 workers on all
  8 GPUs using the dynamic queue.
- [ ] Complete exact accounting for 545,531 source routes, merge/deduplicate,
  build raw/unique/training/provenance views, and validate every final route.
- [ ] Produce per-dataset and combined W2C/C2C analyses, branch statistics,
  ALL-OFF analysis, throughput/compute logs, checksums, and the final report.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Correct branching grows exponentially | A sample can become intractable | Measure every layer/source route; stop only a demonstrated pathological case and preserve evidence before proposing a cap |
| Historical beam outputs contaminate labels | Invalid plan implementation | New output root, schema, code hashes, config hash, and smoke manifest |
| Static smoke imbalance | Idle GPUs and misleading ETA | Exactly one selected sample per worker; full mode uses the dynamic shared queue |
| Oversized transferred images fail locally | Missing positive samples | Preserve the tested content-hash-gated Pillow retry in the execution contract |
| Dirty research worktree | Accidental unrelated edits | Surgical new files and explicit diffs; no broad commit or cleanup |

## Open Questions

- None before the smoke. A demonstrated branch explosion is the only condition
  that would require a new bounded-policy decision.
