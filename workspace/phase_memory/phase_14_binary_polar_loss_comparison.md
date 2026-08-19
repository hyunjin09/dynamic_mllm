# Phase 14: Binary POLAR Loss Comparison Memory

## Current Objective

Evaluate the controlled loss-only comparison between POLAR-style duplicated
valid-route BCE and grouped exact one-of-valid-set NLL for the same direct
28-bit predictor, while keeping full training separately gated.

## Active Constraints

- Keep the frozen Qwen3 question encoder, POLAR-style predictor, 28-bit head,
  regenerated MCTS masks, splits, selected-route cap, optimizer, schedule,
  initialization, decoding, and evaluation identical across objectives.
- Primary route weights are equal within input; no compute-aware weighting.
- The factorized binary head remains unchanged.
- Both objectives use the identical deterministic diverse maximum of 50 valid
  masks per positive input. Raw MCTS routes remain untruncated.
- Actual execution, not cached Hit@1, is the later behavioral gate.
- The regenerated-label P9 manifest is frozen. The bounded smoke is complete;
  full predictor training still requires a separate explicit decision.

## Current State

- Done: loss/data/trainer objective switch; deterministic tests A-E; matched
  initialization hashing; route-cap, duplicate, and split-leakage guards;
  checksum-bound sanity artifact and implementation audit; P4 strict 8K
  GQA/TextVQA/ChartQA cache reconciliation; P5 per-sample/outcome summaries;
  P6 full-cache route-diversity analysis; P7 split-design/external-overlap
  audits; exact P7 split manifest and checksums; P8 matched derived supervision
  manifests and independent streaming verification; P10 matched smoke and
  constant-policy challenge audit.
- In progress: none.
- Blocked: full matched training and external evaluation await explicit user
  direction after the weak constant-policy smoke signal.
- Most recent useful observation: exact set-NLL beats duplicated BCE in the
  frozen smoke, but constant ALL-ON reproduces its Hit@1 and all 18 executed
  predictions, so no conditional routing behavior has yet been demonstrated.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Single-route formula and padded-route errors are `0.0` | `outputs/binary_polar/preflight/loss_comparison_sanity_v1.json` | Validates exact complete-mask likelihood and padding | confirmed |
| Exact set-NLL fell `2.7723 -> 0.7028` and chose `1100` | same | Shows the implemented objective can concentrate on one valid mode | confirmed, synthetic only |
| Duplicated BCE converged to `p_l=0.5` | same | Confirms the intended marginal behavior of duplicated contradictory labels | confirmed, synthetic only |
| Frozen encoder received zero gradients; predictor gradients finite `21/21` | same | Validates the trainable-parameter boundary | confirmed |
| Ten new and sixteen existing focused tests pass | direct test commands; implementation report | Guards objective, collation, executor, and microbatch behavior | confirmed |
| P4 reconciles 8,000/8,000 original-pool records | `outputs/label_regeneration/v1/post_generation/cache_audit_v1.json` | Opens P5 without any label rerun | confirmed |
| P5 current ALL-ON is 4,045 correct / 3,955 wrong; 2,872 current-wrong samples have a correction | `outputs/label_regeneration/v1/post_generation/label_quality_summary_p5_v1.json` | Establishes regenerated-label coverage before structural analysis and splitting | confirmed, discovery-label diagnostic |
| P6 exact full-cache diversity is high: sample-balanced pairwise Hamming 13.36/28 and transitions 13.20 | `outputs/label_regeneration/v1/post_generation/route_diversity_summary_p6_v1.json` | Shows later supervision must preserve non-contiguous multi-mask structure | confirmed, label-geometry diagnostic |
| External bundle has zero MCTS overlap across IDs, exact image hashes, normalized text/prompts, and image-question pairs | `outputs/label_regeneration/v1/post_generation/external_eval_overlap_split_audit_v1.json` | Permits a disjoint transfer evaluation but does not make the already-inspected bundle an untouched test | confirmed |
| Exact 6,500/1,000/500 internal grouped split is feasible | `outputs/label_regeneration/v1/post_generation/predictor_split_design_audit_v1.json` | Earlier feasibility result; the role decision was superseded after the bundle added core VQA and POPE | confirmed, superseded design |
| Expanded bundle passes full image verification; the active 22,307 records exclude DocVQA; core and MC are MCTS-image disjoint, POPE has one shared image/18 rows | `outputs/label_regeneration/v1/post_generation/eval_suite_overlap_audit_v1.json` | Adds task-native core VQA and POPE evaluation and changes the optimal internal split role | confirmed, outcome-blind audit plus user exclusion |
| POLAR defaults to at most 50 valid paths per sample; 3,616 regenerated samples have more than 50 valid masks | `reference/polar/PoLar/polar/data.py`, `outputs/label_regeneration/v1/post_generation/per_sample_route_summary_v1.jsonl` | Fixes the matched primary supervision budget before P8 | confirmed |
| P7 freezes 7,000/1,000 with zero image leakage and exact historical validation strata | `outputs/label_regeneration/v1/post_generation/predictor_split_audit_v1.json` | Fixes all internal identities before any derived view or training | confirmed, outcome-blind |
| P8 derives 237,802 selected valid routes, 2,642,998 ranking routes, and exact POLAR reconstructions | `outputs/label_regeneration/v1/post_generation/derived_supervision_audit_v1.json`, `derived_supervision_verification_v1.json` | Opens final P9 packaging without training | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Directly duplicate tokenized inputs for every route | Up to 32x redundant frozen-encoder rows and mismatched effective input batching | supported by collator shape inspection | implementation review | Encode each unique input once, then index frozen features into distinct predictor route rows | Do not average routes or change the BCE target |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Matched bounded smoke: duplicated BCE vs exact set-NLL | Directly tests optimization and early validation behavior | Opens or rejects full training | medium | awaits approval |
| Full matched training | Required held-out comparison | Main empirical decision | high | unauthorized |
| Structured POLAR segmentation | Fallback if route structure limits factorized head | Representation bottleneck | medium | explicitly deferred |

## Next-Step Decision

- Deliberation mode: fast
- Active objective and bottleneck: P9 is frozen; the only remaining boundary is
  authorization for the bounded matched smoke.
- Relevant memory item used: old executor-domain labels were invalid, so only
  the regenerated cache may supervise this comparison.
- Confirmed observation: the loss/data implementation passes all bounded
  deterministic checks.
- Unverified interpretation: exact set-NLL will optimize and generalize better
  than duplicated BCE under a shared predictor.
- Diagnosis: no research failure diagnosis; real training has not run.
- Chosen action: on explicit approval, execute only the bounded matched P10
  smoke; do not proceed directly to full training.
- Supervision amendment: P8 must derive a deterministic diverse maximum of 50
  valid masks per positive input and reuse exactly that set for both losses.
- How this differs from failed attempts: this compares complete-mask set
  likelihood directly and treats marginal coverage only as a label-structure
  diagnostic.
- Automatic execution authorized: no further action after this P8 boundary.
- Authorization basis: the user explicitly authorized P8 as the next step.
- Stop condition: reached; P9 passed, while training and evaluation were not
  executed.

### P10 pre-training readiness boundary (2026-08-12)

- Action taken: audited and repaired the complete loss/data/training/checkpoint
  and actual-execution paths without training a predictor.
- Confirmed math: exact set-NLL uses complete 28-bit Bernoulli log probability,
  normalized equal route weights, masked padded routes, and stable `logsumexp`.
  Duplicated BCE gives every original input total weight one while retaining
  independent per-route predictor/BCE rows.
- Confirmed data: 8,000 unique rows, exact 7K/1K image-group split, 6,043
  positive train, 874 positive validation, 1,083 zero-positive evaluation-only
  rows, no duplicate/malformed masks, equal weights, and max-50 compliance.
- Repairs: checksum-bound every gate/manifest; added a dedicated seeded
  DataLoader generator; froze a common Hit@1-first checkpoint rule; refused
  output overwrite; froze 300/150 balanced smoke identities; added execution
  of every predicted mask including uncached masks; reported raw-cache and
  selected-set Hit@1 separately; enforced smoke-only versus full-mode authority.
- Real runtime evidence: one GQA, TextVQA, and ChartQA row produced Qwen3 BF16
  features of shape `[3,11,1024]`. Duplicated BCE loss was `0.67772764`; exact
  set-NLL loss was `18.23254204` (raw magnitudes are not comparable). All 33
  predictor gradient tensors were finite for both objectives, initialization
  hashes matched, the encoder had no gradients, and no optimizer step ran.
- Important boundary: the predictor remains question-only to match released
  POLAR. Image conditioning would be an architecture change, not the approved
  loss-only comparison. The factorized head still does not explicitly model
  cross-layer dependencies.
- Evidence: `outputs/binary_polar/preflight/p10_readiness_gate_v1.json`,
  `p10_training_readiness_v1.json`, `p10_real_encoder_preflight_v1.json`,
  `p10_smoke_manifest_v1.json`, and
  `reports/binary_polar_p10_readiness_final.md`.
- Next implication: the bounded matched smoke is technically ready but remains
  unexecuted. Full training and external evaluation remain closed.

## Latest Research-Action Result

### P10 next-action selection (2026-08-12)

- Deliberation mode: STANDARD.
- Active objective and bottleneck: compare duplicated-route BCE against exact
  valid-set NLL while changing only supervision/loss formulation; the remaining
  bottleneck is empirical smoke evidence, not implementation readiness.
- Confirmed observation / unverified interpretation: the checksum-bound static
  audit and real BF16 no-update runtime preflight pass; whether exact set-NLL
  yields a plausible validation or execution advantage remains unverified.
- Diagnosis: no failure diagnosis; training has not started.
- Viable alternatives considered: bounded matched smoke or premature full
  training. The full run remains explicitly gated and would spend substantially
  more compute without first testing the training/evaluation path end to end.
- Chosen action and strongest objection: run only the frozen 300-train,
  150-validation, two-epoch matched smoke with 18 executed masks per objective.
  Its sample size is too small for a scientific conclusion, but the action is
  intentionally an admission gate rather than the final comparison.
- Authorization and stop condition: await explicit user approval; after the
  smoke, stop to interpret the frozen admission criteria. Do not automatically
  begin full training or external evaluation.

- Action taken: derived and independently verified all P8 supervision views
  from the checksum-bound raw cache and frozen P7 assignments.
- Result: pass. P8 retains 8,000 sample rows, 6,917 positive samples, 1,083
  zero-positive samples, all 2,642,998 evaluated routes for ranking, and
  237,802 selected valid routes after applying the max-50 cap to 3,616 samples.
- Matched comparison invariant: duplicated BCE and exact set-NLL use identical
  per-sample mask lists and equal weights; shared route-set SHA-256 is
  `eafd8bb9dd66b2a800850e6f8e778eb68ad493c24c2861b921e91bb387c2bc0b`.
- Evidence saved:
  `outputs/label_regeneration/v1/post_generation/derived_supervision_audit_v1.json`,
  `derived_supervision_verification_v1.json`, and
  `reports/label_regeneration_p8_derived_supervision.md`.
- Failure or issue: initial CPU attempts were too slow. The supported cause was
  repeated Python tuple Hamming work plus serialized 21 GB JSON decoding. The
  implementation was repaired with mathematically identical XOR bit counts,
  single-pass reads, and bounded process decoding; only unpublished temporary
  outputs were removed between attempts.
- Independent checks: artifact checksums, route cap/equal weights, split-group
  isolation, ranking positive/negative counts, and all 237,802 canonical POLAR
  reconstructions pass.
- Boundary note: P9 was the next gate after this P8 result and is now completed
  in the entry below; real predictor training remains unopened.

### P9 prerequisite result (2026-08-12)

- P9 status: PASS. The regenerated 8K cache, P7 split, P8 max-50 matched route
  sets, provenance, report, and checksum chain are now frozen.
- Integrity evidence:
  `outputs/label_regeneration/v1/post_generation/p9_final_audit_v1.json` and
  `P9_SHA256SUMS`; independent verification passed all 53 entries.
- Research boundary: this removes the data-integrity blocker but does not
  authorize training. The next permitted action, if explicitly approved, is
  only the bounded duplicated-BCE versus exact-set-NLL smoke.

### P10 bounded matched smoke result (2026-08-13)

- Deliberation mode: STANDARD with one independent review because the possible
  next action is the expensive full matched run.
- Active objective and bottleneck: determine whether exact valid-set NLL gives
  enough coherent held-out signal over duplicated BCE to justify full training.
- Confirmed observation: both matched 300/150, two-epoch runs completed from
  the same initialization. Exact set-NLL improved selected-epoch validation
  Hit@1 from `0.1333` to `0.5733`, nearest-valid Hamming from `9.4733` to
  `3.7267`, and 18-record execution accuracy from `0.2222` to `0.5000`.
  However, duplicated BCE decoded ALL-OFF for all 18 executions and exact
  set-NLL decoded ALL-ON for all 18; exact set-NLL merely matched the FULL
  baseline and reduced no visual layers.
- Diagnosis: supported smoke constant-policy collapse. On the same 150 route
  sets, constant ALL-OFF exactly matches BCE Hit@1/Hamming, and constant ALL-ON
  exactly matches exact-set Hit@1 while slightly improving its Hamming. Evidence:
  `outputs/binary_polar/p10_smoke/constant_policy_audit_v1.json`.
- Viable alternatives considered: full matched training to test whether scale
  breaks the constant mode; stop/defer because the smoke has no learned routing
  evidence. No loss, weighting, or architecture redesign is authorized.
- Chosen action and strongest objection: stop after the authorized smoke and
  request an explicit decision on full training. The literal admission rule
  versus duplicated BCE passes, but the strongest objection is that its entire
  observed execution advantage is baseline-preserving ALL-ON rather than useful
  route selection.
- How this differs from failed attempts: the completed result follows a tested
  BF16 validation-autocast repair and refreshed checksum-bound readiness gate;
  it is not an executor or dtype failure.
- Authorization and stop condition: the smoke action is complete. Full training
  and external evaluation were not executed.
- Evidence: `reports/binary_polar_p10_smoke_results.md`, both `*_v2/history.json`
  files, both `*_execution_v2.json` files, and
  `outputs/binary_polar/preflight/repair_v2/p10_readiness_gate_v2.json`.
