# Phase 18: Full10 POLAR-Matched Direct Predictor

## Current Objective

Run the full frozen Question-only and Image+Question direct exact-valid-set
predictors for ten epochs under `plans/full_train.md`, preserving every epoch
and the complete optimization trajectory.

## Active Constraints

- Use exactly 6,043 positive train and 874 positive validation inputs from the
  frozen image-group-disjoint GQA/TextVQA/ChartQA split.
- Preserve the deterministic diverse max-50 valid sets, P11/P13 ALL-ON weight
  0.3, direct 28-bit head, exact valid-set NLL, frozen Qwen3 question encoder,
  and P13 unpooled projected visual rows.
- Train 10 epochs with AdamW, learning rate 5e-4, cosine schedule, 10 warmup
  optimizer steps, physical/effective batch 128 if the prospective 128-record
  longest-image memory preflight passes, and matched seed/initialization.
- Save and validate every epoch. Do not early-stop for scientific performance.
- Run Question and Image+Question concurrently on one GPU each after the full
  visual cache passes.
- Preserve P10-P13 artifacts. Do not use node04.

## Current State

- Done: full visual cache, full10 preflight, both ten-epoch trainings, all 20
  checkpoints, selected-checkpoint conditioning diagnostics, and four frozen
  60-record actual executions.
- Result: longer training produces nonconstant masks at later epochs, but the
  best-Hit@1 checkpoints still match constant ALL-ON and later diversity lowers
  validation route quality.
- Stop: direct predictor is not admitted to external evaluation; no new
  objective is authorized.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Positive population is 6,043 train / 874 validation | `outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl` | Defines the full objective population | confirmed |
| P13 selected Image+Question is 150/150 ALL-ON after two epochs | `reports/binary_polar_p13_results.md` | Motivates testing optimization budget without method changes | confirmed |
| Native visual rows pass leakage/repeat checks | `outputs/binary_polar/p13/visual_features_v1/cache_audit_v1.json` | Validates the feature definition to extend | confirmed |
| Both full10 runs completed 10 epochs / 480 steps | `outputs/binary_polar/full10/*_v1/training_summary.json` | Establishes complete approved optimization budget | confirmed |
| Best Hit@1 equals constant ALL-ON at 58.12% | `outputs/binary_polar/full10/*_v1/history.json` | Longer training does not improve selected routes | confirmed |
| Best-checkpoint executions are both 50%, W→C=0, C→W=0 | `outputs/binary_polar/full10/execution_*best_hit_at_1_v1.json` | No selected-policy behavioral gain | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| P11/P13 two-epoch direct predictors | probability signal but near/exact ALL-ON decode | optimization-budget limitation remains untested | P11/P13 reports | Run the explicitly approved complete 10-epoch trajectory | Do not early-stop or retune |
| Initial full10 execution serialization | all 60 forwards completed, then shared summary raised `KeyError: raw_cached_valid_set_size` | supported row-schema omission | `runs/binary_polar/full10_exec_*_10076*.log` | Added cache-size fields at row construction and a regression test; unchanged reruns reproduced results | Do not bypass the shared summary contract |

## Open Candidates

None. The user explicitly selected the matched Question-only and
Image+Question full10 runs.

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: determine whether the complete full10
  trajectory admits the direct predictor to external evaluation.
- Confirmed observation / unverified interpretation: aligned inputs improve
  set-NLL, but selected complete masks do not beat constant ALL-ON; this is a
  supported failure for the frozen direct factorized setup, not proof that no
  route predictor can work.
- Diagnosis: supported persistent probability-to-decoding failure; evidence in
  `reports/binary_polar_full10_polar_matched_results.md`.
- Viable alternatives considered: external evaluation of the selected direct
  predictor; post-hoc use of epoch 10; stop and require separate approval for a
  different formulation. The first two lack validation admission.
- Chosen action and strongest objection: stop the direct path without external
  evaluation. Two uncached epoch-10 ChartQA routes correct FULL, but that
  checkpoint has worse 874-record route metrics and was not selected.
- How this differs from failed attempts: the full population and complete
  ten-epoch POLAR-scale schedule now rule out insufficient epoch budget as the
  narrow explanation for selected-checkpoint collapse.
- Authorization and stop condition: full10 is complete; do not execute another
  objective or external evaluation without explicit approval.

## Latest Research-Action Result

- Action taken: completed matched Question-only and Image+Question full10
  training, trajectory validation, conditioning diagnostics, and frozen-60
  execution.
- Result: direct factorized predictor not admitted; selected checkpoints match
  constant ALL-ON and later diversity is lower quality.
- Evidence saved: `outputs/binary_polar/full10/` and
  `reports/binary_polar_full10_polar_matched_results.md`.
- Next implication: stop. A different formulation is a new research decision.
