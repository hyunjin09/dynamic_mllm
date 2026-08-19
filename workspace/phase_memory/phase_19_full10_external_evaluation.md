# Phase 19: Full10 Best-Checkpoint External Evaluation

## Current Objective

Evaluate the validation-selected Question-only and Image+Question full10
checkpoints on the frozen, expanded external bundle using their direct static
28-bit masks and the verified binary executor.

## Active Constraints

- Use Question epoch 2 and Image+Question epoch 4, selected before external
  evaluation from the same frozen internal-validation criterion.
- Evaluate exactly 22,307 records: ChartQA test, TextVQA validation, MMStar
  validation, MMMU validation MC, MMMU-Pro standard/vision test, and all three
  POPE splits. DocVQA remains excluded.
- Preserve bundle prompts, benchmark scorers, deterministic greedy generation,
  EOS 151645, repetition penalty 1.05, and a static predicted mask from layer 0.
- Report core VQA, multiple choice, and POPE separately. Report full POPE and
  the frozen 8,982-record image-disjoint sensitivity.
- Require stratified ALL-ON native/binary/cache parity and deterministic mask
  replay before any full evaluation.
- Do not use node04.

## Current State

- Done: checkpoint selection, expanded-suite overlap audit, adapter tests,
  preflights, both 22,307-record executions, integrity merge, clustered
  analysis, report, and checksums.
- In progress: none.
- Blocked: none.
- Most recent useful observation: Image+Question selected ALL-ON on all 22,307
  records. Question-only selected non-ALL-ON on 44 records, but none changed
  prediction, benchmark score, or correctness from current live ALL-ON.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Question epoch 2 is validation-selected best Hit@1 | `outputs/binary_polar/full10/question_v1/history.json` | Freezes the Question checkpoint outcome-blind to external results | confirmed |
| Image+Question epoch 4 is validation-selected best Hit@1 | `outputs/binary_polar/full10/image_question_v1/history.json` | Freezes the multimodal checkpoint outcome-blind to external results | confirmed |
| Active external suite is 22,307 records and DocVQA is excluded | `reports/binary_router_expanded_eval_suite_audit.md` | Fixes population and reporting strata | confirmed |
| Static evaluator contract tests pass | `tests/test_binary_polar_external_eval.py` | Validates text construction, benchmark inclusion, sharding, and mask summaries | confirmed |
| Joint and separate predictor preflights pass | `outputs/binary_polar/external_eval/full10_best_v1/*preflight_v1.json` | Validates inference and scoring before the full runs | confirmed |
| Frozen POPE sensitivity removes exactly 18 records on one shared image | `outputs/binary_polar/external_eval/full10_best_v1/evaluation_contract_v1.json` | Prespecifies official and strict image-disjoint POPE reports | confirmed |
| Both predictors completed exactly 22,307 unique UIDs and output hashes pass | `outputs/binary_polar/external_eval/full10_best_v1/analysis_manifest_v1.json` | Establishes final evaluation integrity | confirmed |
| Neither checkpoint changes external benchmark behavior | `reports/binary_polar_full10_external_eval.md` | Shows predicted routes reproduce current live FULL scores and correctness | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| None in this phase | N/A | unknown | N/A | Run only the frozen preflight next | Do not launch full shards before parity passes |
| Historical cache shortcut after 9-row parity | 7/192 initial ChartQA rows had different current live ALL-ON predictions; one changed correctness | supported cache non-equivalence under the current complete population; exact upstream cause unknown | `outputs/binary_polar/external_eval/full10_best_v1/*/shard_000_of_001/pre_live_baseline_repair/` | Use current live ALL-ON as the scientific baseline for every sample; retain cache only as historical audit data | Do not substitute cached baseline output globally |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Frozen expanded-suite evaluation | Explicitly authorized and checkpoints are already frozen | Measures cross-task behavior and compute of both predictors | high | completed |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: complete; no additional evaluation action is
  authorized.
- Relevant memory item used: the expanded-suite audit requires stopping cached
  baseline use if parity fails.
- Confirmed observation: 7/192 initial ChartQA rows disagree between current
  live ALL-ON and the historical cache in both independent jobs; the two live
  jobs agree with one another.
- Confirmed observation / unverified interpretation: both checkpoints collapse
  essentially or exactly to FULL externally; this is evidence against the
  frozen direct-factorized setup, not proof that routing is impossible.
- Diagnosis: supported historical-cache non-equivalence for the current run;
  evidence is preserved in the partial-repair backups and ledgers. The exact
  cause of cache drift remains unknown and is not needed for the baseline rule.
- Viable alternatives considered: use the historical cache (rejected by direct
  mismatch); stop the evaluation (unnecessary because current live ALL-ON is
  available); use current live ALL-ON per sample (chosen).
- Chosen action: stop after the completed authorized evaluation and preserve
  the negative result.
- Strongest objection: non-ALL-ON predictions cost an extra dense execution;
  this is required for valid paired comparison and is expected to be rare from
  frozen validation behavior.
- How this differs from failed attempts: this is the first external direct-head
  adapter execution; it does not reuse the historical SW31 router path.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to evaluate both best checkpoints.
- Stop condition: reached; exact population, source, checkpoint, and checksum
  gates passed.

## Latest Research-Action Result

- Action taken: completed both full external runs and the frozen 5,000-draw
  image-clustered aggregation.
- Result: integrity PASS. Question-only was ALL-ON for 22,263/22,307 and its 44
  non-ALL-ON routes changed no prediction, score, or correctness. Image+Question
  was ALL-ON for 22,307/22,307. Every reported benchmark therefore matched
  current live FULL behavior.
- Evidence saved: `outputs/binary_polar/external_eval/full10_best_v1/` and
  `reports/binary_polar_full10_external_eval.md`.
- Failure or issue: historical cached FULL differed from current live FULL on
  485 records, including 168 correctness labels; it is audit-only.
- Lesson learned: the frozen direct-head checkpoints provide no meaningful
  external behavioral improvement or visual-layer compute reduction.
- Next implication: stop. A different formulation is a separate research
  decision.
