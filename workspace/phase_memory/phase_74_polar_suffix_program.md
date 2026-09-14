# Phase 74: POLAR-Style Suffix Program Memory

## Current Objective
Train one P90-triggered autoregressive complete suffix-program predictor on every eligible replay-valid training route, then evaluate it on the full established four-benchmark protocol.

## Active Constraints
- Qwen2.5-VL-7B, robust ALL-source Stage-1 P90, LMMS scoring, four-action semantics, and pre-trigger FULL execution are frozen.
- Dense-C receives only the all-FULL suffix; Dense-W retains all unique correct complete programs with 1/K route weight.
- External evaluation is exactly ChartQA, TextVQA, MMMU-Pro Standard/Vision, and POPE adversarial/popular/random (19,960 records).
- Beam width 8 is primary; greedy is diagnostic. No external-set tuning or architecture sweep.
- All four GPUs are available through direct execution; no Slurm.

## Current State
- Done: plan/protocol audit, corpus construction, full exact replay/cache, group-disjoint epoch selection, full-corpus refit, smoke validation, and the complete 19,960-row paired evaluation.
- In progress: none.
- Blocked: none.
- Most recent useful observation: complete-program prediction reduced C-to-W regressions from Sequential-A's 19 to 8 while retaining 3 W-to-C rescues, but remained 5 correct answers below Dense overall.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Sequential-A full evaluation had 3 W→C and 19 C→W | `analysis/dense_failure_stage2/full_benchmark_eval/` | Complete-program prediction must improve rescue-regression balance, not local imitation only | confirmed |
| Phase-73 invalidated 99.2% of KEEP and 54.2% of INTERVENE labels | `analysis/dense_failure_stage2/treatment_label_completeness/` | Local labels erase extensive valid suffix structure | confirmed |
| 4,948 eligible P90 programs from 569 UIDs | Phase-74 read-only corpus census | Full training population and weighting denominator are known | confirmed |
| Stage2-A compatible branch/context tensors are present and hash-bound | `analysis/dense_failure_stage2/shared_union_training/experiment_A_single/training/final_checkpoint.pt` | Required initialization is implementable exactly | confirmed |
| Every one of 4,948 programs replayed LMMS-correct with exact cached/live trigger-state parity | `analysis/dense_failure_stage2/polar_suffix_program/work/replay_completion.json` | The training corpus and feature cache are execution-valid | confirmed |
| Program full eval: 3 W-to-C, 8 C-to-W, net -5; Sequential-A: 3/19/-16 | `analysis/dense_failure_stage2/polar_suffix_program/metrics/full_benchmark_summary.csv` | Program supervision improves preservation but not treatment transfer enough to beat Dense | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Sequential local Stage2-A deployment | Net -16 on full evaluation | supported: local action imitation does not transfer adequately | Phase-69 paired results | Preserve complete program structure in the one authorized experiment | another unmodified local-action refit |
| Observed-valid-set local loss | Better oracle agreement without free-run rescue | supported | Phase-68 summary | Local ambiguity is not solved by another local target formulation | collapse programs back to local valid sets |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Complete autoregressive suffix program | Retains joint MCTS trajectory structure | Whether trajectory supervision improves end-to-end Net | high | completed; not a deployment winner |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: determine whether complete-program supervision solves the Stage-2 treatment generalization bottleneck.
- Relevant memory item used: Phase-73 local action ambiguity and Phase-69 negative sequential deployment.
- Confirmed observation: the full eligible corpus is reconstructable and P90 aligned.
- Unverified interpretation: coherent complete-program decoding will transfer better than local decisions.
- Diagnosis: supported
- Evidence path if diagnosis is not unknown: `analysis/dense_failure_stage2/treatment_label_completeness/` and `analysis/dense_failure_stage2/full_benchmark_eval/`
- Viable alternatives considered: live Qwen recomputation during every optimizer step versus one exact raw-trigger-state cache.
- Chosen action: completed exact replay/cache, group-disjoint epoch selection, full-corpus refit, and complete four-family evaluation.
- Strongest objection: cached states could silently differ from the live trigger representation; all-UID tensor/branch/context/initial-logit parity is therefore mandatory.
- How this differs from failed attempts: the supervision and decoding unit is the entire post-trigger trajectory, not an independent local action.
- Automatic execution authorized: completed
- Authorization basis: user explicitly requested execution of `plans/polar_style_post_trigger_suffix_program_predictor_plan.md`.
- Stop condition: full corpus/replay, internal dev and 100% refit, all 19,960 paired evaluations, required metrics/diagnostics, and one recommendation are complete; or a validity gate fails.

## Latest Research-Action Result
- Action taken: trained the frozen autoregressive post-trigger suffix-program predictor on 4,948 verified programs from 569 UIDs and evaluated it against Dense and Sequential-A on all 19,960 established external records.
- Result: internal-dev beam-8 exact match was 24/115 (20.87%). The full refit selected two epochs. External Program accuracy was 0.780311 versus Dense 0.780561 and Sequential-A 0.779760; Program W-to-C/C-to-W/net was 3/8/-5 versus Sequential-A 3/19/-16. Program supervision therefore preserved 11 additional Dense-C answers relative to Sequential-A but added no rescues.
- Evidence saved: `analysis/dense_failure_stage2/polar_suffix_program/`; frozen contract `0c1696ed...597a49`; checkpoint `1621851b...f0564`; all 65 declared artifact hashes pass.
- Failure or issue: one aggregation attempt failed because the empty `figures/` directory was absent; creating that output directory and rerunning the unchanged hash-bound aggregation completed successfully. No scientific computation was changed or repeated.
- Lesson learned: coherent complete-program supervision is substantially more conservative than the local Sequential-A policy, but the dominant unresolved bottleneck is transfer of corrective treatment rather than preservation alone.
- Next implication: stop. Keep program supervision only as a conservative formulation; no follow-on experiment is authorized.
