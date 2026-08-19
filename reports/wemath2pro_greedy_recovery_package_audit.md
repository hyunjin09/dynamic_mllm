# WeMath2.0-Pro Greedy Phase-1/Phase-2 Recovery Audit

## Decision

The package in `search/greedy_phase1_phase2_reproduction/` is intact and its
search algorithm is a scientifically reasonable **complement** to the completed
unrestricted graph-MCTS cache. It is not directly runnable for WeMath2.0-Pro.
The frozen package targets a different 10,000-record, four-benchmark manifest,
an older project layout, Transformers 4.57.1, benchmark-specific image caps,
and scoring/import modules that are not the current verified WeMath execution
path.

The recommended approach is to preserve the package unchanged as the
algorithmic reference and implement a small WeMath-specific adapter around the
current verified binary executor, native image processing, and MathRuler
scorer. No search was run in this audit.

## Package integrity

`reference/CHECKSUMS.sha256` passes for every frozen file: manifest, config,
both collectors, both aggregators, final auditor, launch scripts, README,
requirements, and runtime record. The core reproduction package was not
modified.

The package reproduces a prior search over:

- 10,000 samples: 4,000 GQA and 2,000 each ChartQA, TextVQA, and DocVQA;
- Qwen2.5-VL-7B-Instruct snapshot
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
- 28 binary visual ON/OFF layer actions;
- SDPA, BF16, slow processor, greedy generation;
- a canonical Transformers 4.57.1 environment.

The copied `paths.current_server.env` points to another server and is not an
active configuration for this repository.

## Exact search semantics

### Phase 1: ten greedy removal traces

Every sample starts from ALL-ON. ALL-ON and ALL-OFF anchors are evaluated, then
ten independent greedy traces are run:

1. early to late;
2. late to early;
3. center out;
4. outside in;
5. six UID-conditioned random orders using seeds 20260714–20260719.

Within each trace, the algorithm tries to turn one currently ON layer OFF in
the frozen order. A candidate is accepted when

```text
candidate_score + 1e-9 >= max(ALL_ON_score, current_score).
```

Accepted and rejected candidates are both retained. Each trace makes exactly
28 requests. Including two anchors, Phase 1 has at most 282 unique candidate
executions per sample, with fewer when traces revisit the same mask.

For the WeMath recovery population, ALL-ON score is already known to be zero.
MathRuler validity is binary at threshold 1.0. Therefore zero-score removals
are accepted until a correct route is found; after a score-1 route is found,
only later removals that preserve score 1 are accepted. A discovered correct
intermediate route consequently yields a correct final route under this binary
scoring contract.

### Phase 2: random/local expansion

Phase 2 is not a second greedy pass. It first chooses up to three diverse,
low-ON successful Phase-1 final masks. It then proposes:

- six budget-stratified random masks at benchmark budget center −2, center,
  and center +2, with two draws per budget;
- up to four same-budget ON/OFF swaps per successful base;
- up to four add-one and four remove-one neighbors per base;
- unions and intersections of successful base pairs.

With three bases, this is at most 48 deduplicated requests per sample before
reuse against Phase 1. If a sample has no successful Phase-1 base, it receives
only the six budget-stratified random requests. The benchmark budget center is
computed from successful Phase-1 final masks.

Combined, the reference procedure offers at most 330 candidate requests per
sample, including anchors. Its value is not a larger raw budget than the
completed 400-simulation MCTS. Its value is the different geometry: correlated
nested removals, local neighbors, and recombinations rather than graph-MCTS
rollouts.

## Proposed WeMath recovery population

Use exactly the completed analysis Group D:

```text
current ALL-ON wrong
AND no valid route among the completed cap-400 MCTS evaluations.
```

This yields:

| Quantity | Value |
|---|---:|
| Recovery samples | 2,278 |
| Unique image groups | 1,104 |
| Existing MCTS evaluations per normal sample | 402 unique masks |
| Mean visual tokens | 2,422.49 |
| Median visual tokens | 850.5 |
| P90 visual tokens | 8,239 |
| Maximum visual tokens | 11,235 |

Difficulty counts are: base 207, x 340, xy 356, xyz 357, xz 348, y 217,
yz 227, and z 226. Selection is intentionally outcome-dependent on the old
MCTS search because the scientific purpose is recovery of its explicitly
failed cases. Any success rate must therefore be described as conditional
recovery, not dataset-wide prevalence.

## Why the frozen package cannot be run unchanged

1. **Preflight is hard-coded to the old 10K population.** It requires exact
   GQA/ChartQA/TextVQA/DocVQA cell counts and the old manifest semantic hash.
2. **The project imports differ.** The collectors require
   `analysis_outputs/harmful_validation_common.py` and
   `analysis_outputs/run_harmful_interventions.py`; those are not the active
   current-repository executor modules.
3. **The runtime differs.** The frozen reproduction expects Transformers
   4.57.1, whereas the verified WeMath executor and MCTS labels use the project
   Transformers 5.3.0 environment. Downgrading would break the intended
   executor contract.
4. **The preprocessing policy differs.** The old package permits per-row image
   caps. WeMath must preserve native/default Qwen processing and no custom
   visual-token cap.
5. **The scorer differs.** WeMath requires the frozen
   `wemath2pro_mathruler_accuracy` path, threshold 1.0, and the existing bounded
   scorer timeout behavior.
6. **The manifest schema differs.** The old collector expects `data_split`,
   `source_bucket`, `source_full_score`, and `source_full_prediction`. The
   authoritative current anchors live in the completed WeMath MCTS records.
7. **The final auditor expects the old corpus totals.** A new recovery-specific
   completeness and provenance audit is required.

These are implementation-contract differences, not evidence against the
greedy search algorithm.

## Proposed protocol, not yet authorized for execution

### G0 — Freeze a recovery manifest

- Select the exact 2,278 Group-D UIDs from the checksum-bound WeMath analysis.
- Join each UID to its frozen source row and completed MCTS record.
- Preserve image hash/group, question, prompt, answer, difficulty, token counts,
  current ALL-ON prediction/IDs/score, and the raw-record checksum.
- Record that this is a conditional MCTS-failure recovery set.
- Do not inspect greedy outcomes while constructing it.

Acceptance: exactly 2,278 UIDs, 1,104 image groups, no positive MCTS route,
no duplicate UID, and every MCTS/source checksum passes.

### G1 — Implement a minimal current-runtime adapter

- Copy only the frozen Phase-1 order/acceptance and Phase-2 candidate-generation
  semantics into a new WeMath-specific module; do not edit the reproduction
  package.
- Use `label_regeneration.runtime.RouteEvaluator`, the current
  `binary_policy.executor`, native Qwen preprocessing, 96-token greedy
  generation, and the frozen MathRuler scorer timeout.
- Preseed the per-sample mask cache from the completed compatible MCTS record.
  Exact mask matches may reuse their frozen tokens/prediction/score and must be
  marked `mcts_cache_reuse`; genuinely new masks must be executed and marked
  `greedy_phase1` or `greedy_phase2`.
- Write to a new output root. Never mutate or overwrite the MCTS cache.

Acceptance: deterministic order and Phase-2 synthetic tests match the frozen
reference algorithm; reused masks reproduce their full cached payload; new
records preserve the verified 28-bit semantics.

### G2 — Minimal execution preflight

- Select five deterministic recovery samples before outcomes are opened.
- Require current adapter ALL-ON to equal native Qwen and the cached MCTS
  ALL-ON generated IDs, prediction, and score on 5/5.
- Require one cached mixed mask per sample to match its MCTS tokens and score.
- Require one newly executed mixed mask to reproduce identically twice.
- Verify native image processing, token counts, scorer timeout, and atomic
  serialization.

Stop on any mismatch. Do not weaken parity or switch runtime.

### G3 — Run Phase 1 only

- Run all ten frozen greedy orders on all 2,278 recovery records.
- Retain every candidate request, acceptance decision, cache-reuse origin,
  generated token sequence, prediction, score, and validity.
- Resume only atomic, contract-valid records under a fixed shard count.

Acceptance: exactly 2,278 terminal Phase-1 records, 22,780 final traces,
zero unresolved errors, and exact candidate/trace linkage.

### G4 — Freeze the Phase-2 expansion

- Aggregate Phase 1 and compute the WeMath budget center exactly as the
  reference does, from successful unique Phase-1 final masks.
- If Phase 1 produces no successful final mask anywhere, stop: the reference
  Phase-2 budget center is undefined and changing it requires approval.
- Otherwise freeze the budget statistics, Phase-2 seed 20260720, two random
  draws per budget, four local draws per operation, and the per-sample request
  manifest before Phase-2 execution.

### G5 — Run Phase 2 and finalize

- Execute the frozen requests with the same cache-reuse rules.
- Combine MCTS, greedy Phase 1, and greedy Phase 2 by exact 28-bit mask while
  preserving every search origin.
- Audit all 2,278 UIDs, route uniqueness, score/validity linkage, current
  contract, and checksums.
- Report Phase-1-only and incremental Phase-2 recovery separately.

Primary descriptive outcome: number and fraction of the 2,278 MCTS-failure
samples for which at least one correct route is newly found. Secondary outputs
include new valid routes per recovered sample, minimum ON count, transitions,
Hamming distance from FULL, difficulty strata, and overlap/reuse with MCTS.
Do not start predictor training.

## Candidate comparison and recommendation

| Candidate | What it resolves | Cost | Main weakness | Decision |
|---|---|---:|---|---|
| Run frozen package unchanged | Literal old reproduction | high | Violates current WeMath runtime/data/scorer contract | reject |
| Port Phase 1 only | Whether systematic nested removal recovers MCTS failures | high | Does not test the documented local/random expansion | viable first execution stage |
| Port frozen Phase 1+2 with a Phase-1 gate | Full complementary search under current contract | high | Phase 2 is weak for samples without a successful base | recommended protocol |
| Extend MCTS again | More of the prior search family | high | Prior >400 extension had low yield and user requested greedy detour | reject |

The provisional recommendation is the faithful current-runtime Phase-1+2
adapter, executed sequentially with an explicit Phase-1 aggregation gate. The
strongest objection is that Phase 1 accepts every score-zero removal on these
binary-scored failures, which can make many traces collapse toward ALL-OFF;
Phase 2 then gives a still-unsuccessful sample only six random masks. Recovery
yield may therefore be low despite the substantial inference cost. The method
remains defensible because its nested/local route geometry is genuinely
different from the completed MCTS rollouts and the stop rule prevents an
undefined Phase-2 adaptation.

Confidence in the implementation recommendation is high. Confidence that the
search will recover many additional samples is low-to-medium and must be
measured rather than assumed. No independent reviewer is required because the
unchanged-package option clearly violates the verified execution contract and
the remaining candidate ranking is not narrow.

## Authorization boundary

This audit authorizes no implementation run, GPU job, manifest freeze, or
search. The smallest next action, after explicit approval, is **G0–G2 only**:
freeze the recovery manifest, implement the adapter, and pass the five-sample
parity/determinism preflight. Full Phase 1 must not begin before that gate.

