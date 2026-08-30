# Phase 42: W2C WHEN-Label Repair Memory

## Current Objective
Repair the authoritative 512-train/128-validation W2C continuation cache with
one identical iterative algorithm, then rebuild and audit the binary WHEN
labels without training a gate or starting another experiment.

## Active Constraints
- Follow `plans/w2c_when_label_repair_plan.md` (SHA-256
  `19d750c7acca5caaf37a85438f432e566dd980cbc29ddb1e6cf7d3c8e0c23e88`).
- Preserve every original correct route; write new records and artifacts only.
- At each current maximal all-FULL prefix, first execute every deduplicated
  known correct-route suffix after forcing FULL at the boundary.
- Only after known suffixes are exhausted, run the prospectively frozen,
  deterministic, layer-stratified one-edit continuation search with at most 96
  candidates per boundary and fixed seed `20260830`.
- A non-rescue supports only `FULL_UNRESCUED_UNDER_BUDGET`, not global
  invalidity or necessity of DEVIATE.
- Apply the same algorithm and budget to train and validation. Use all four
  local RTX 6000 Ada GPUs through direct `torchrun`; this server has no Slurm.
- Pass a 12-sample smoke gate before the complete 640-sample repair. Per-sample
  runtime failures are quarantined rather than relabeled.
- Do not train a router/gate, run Stage 2, run external evaluation, add a
  dataset, or select another research action.

## Current State
- Done: audited the authoritative Phase-39 matched manifest and recovered
  exactly 512 train plus 128 validation W2C samples spanning GQA, ChartQA, and
  TextVQA, with 16,848 original correct routes.
- Done: implemented and unit-tested the pure iterative repair engine, stable
  smoke selection, cost-balanced four-rank sharding, and a resumable direct
  four-GPU live executor with atomic per-sample records.
- Done: froze the repair manifest/protocol/config and ran the 12-sample smoke
  on all four GPUs. The repair trace and resume checks pass with zero
  quarantine, but exact old-route replay fails.
- Stopped: the complete 640-sample repair, repaired-label build, post-repair
  audit, and every downstream training/evaluation step were not started.
- Blocked: the exact source executor implementation is absent from the current
  repository state; proceeding under the current executor would violate the
  prospectively frozen old-route replay gate.
- Most recent useful observation: 37/312 original cached-correct routes replay
  incorrectly under the current runtime, affecting 10/12 smoke samples across
  all datasets. Four representative failures repeat exactly twice on the
  current runtime, and 0/4 match the original cached generated IDs.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Forced FULL had a compatible correct continuation for 39/128 validation W2C boundaries | `analysis/selective_continue_deviate/when_label_completeness_results.json` | Directly establishes WHEN-label cache incompleteness | confirmed |
| All 252 Phase-41 audit routes completed with zero unresolved states | `analysis/selective_continue_deviate/when_full_insertion_executions.jsonl` | Separates the scientific failure from a runtime failure | confirmed |
| Authoritative repair population is 512 train and 128 validation W2C samples | `analysis/persistent_corrective_supervision/training_manifest.jsonl`; `analysis/persistent_corrective_supervision/boundary_manifest.jsonl` | Fixes the population and minimum downstream candidate counts | confirmed |
| The physical source records and model remain available, but current executor code hashes differ from the label records | `analysis/w2c_when_repair/smoke/replay_failure_diagnostic.json` | Exact source-route validity cannot be assumed under this runtime | confirmed |
| One-base one-edit suffix neighborhoods have at most 81 variants | deterministic pre-repair enumeration from the frozen routes | A cap of 96 exhausts a single-base neighborhood and permits limited multi-suffix coverage | confirmed |
| 37/312 original cached-correct routes replay incorrectly across 10/12 smoke samples | `analysis/w2c_when_repair/smoke/smoke_executions.jsonl`; `smoke_gate.json` | Fails the prospective smoke gate and forbids the full repair | confirmed |
| Four representative current-runtime failures reproduce exactly twice; none matches original cached tokens | `analysis/w2c_when_repair/smoke/replay_failure_diagnostic.json`; external raw diagnostic records | Supports reproducible contract drift rather than transient nondeterminism | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Phase-38 four-action isolation | Both router families remained near-all-FULL with zero held-out W2C rescue | supported WHEN generalization failure; exact cause unknown | `analysis/4action_collapse/decision_summary.md` | Repair the target before another router run | Another unchanged four-action training run |
| Phase-41 selective-gate prerequisite audit | 39/128 mandatory boundaries accepted FULL under a known continuation | supported cache incompleteness; discovery omission cause unknown | `analysis/selective_continue_deviate/stage1_decision_summary.md` | Iteratively augment the correct-route cache and recompute boundaries | Train a gate on the known-incomplete DEVIATE labels |
| Phase-42 exact old-route smoke replay | 37/312 cached positives are incorrect under the current runtime | supported reproducible runtime/cache drift; root cause unknown | `analysis/w2c_when_repair/smoke/smoke_report.md`; `replay_failure_diagnostic.json` | Recover the source execution contract before population repair | Drop failed routes silently or continue under a mixed contract |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Known suffixes, then deterministic one-edit suffix repair capped at 96 per boundary | Directly reuses verified continuations, then expands locally with explicit bounded semantics | Whether enough clean WHEN candidates remain after iterative repair | high | stopped at failed smoke prerequisite |
| Adapt binary GraphMCTS to four actions | Broader continuation search could find multi-edit rescues | Whether remaining states survive a substantially wider search | high | rejected for this action; larger semantic and implementation change |
| Known-suffix-only repair | Minimal and directly reproduces Phase-41 rescue mechanism | Repairs proven omissions but cannot discriminate nearby unseen continuations | medium | rejected as underpowered |
| Recover the exact source executor and rerun only the smoke replay gate | Restores the contract that defined cached correctness | Whether repair can start from a valid route population | medium | smallest future action; requires source artifact/authorization |
| Rebuild all W2C correct routes under the current executor | Establishes a self-consistent new cache when the old executor is unavailable | A new current-runtime supervision population | high | alternative new plan; not authorized |

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: rebuild trustworthy W2C WHEN labels; the
  exact old-route replay prerequisite fails under the current executor.
- Relevant memory item used: Phase 41 found 39/128 compatible known-suffix
  rescues and stopped gate training prospectively.
- Confirmed observation: 37/312 cached positives fail current execution, and
  four representative failures are deterministic within the current runtime.
- Unverified interpretation: which code/environment difference causes the
  original-versus-current token changes.
- Diagnosis: supported reproducible runtime/cache drift; root cause unknown.
- Evidence path if diagnosis is not unknown:
  `analysis/w2c_when_repair/smoke/replay_failure_diagnostic.json`.
- Viable alternatives considered: recover the source executor and repeat only
  smoke; rebuild the W2C correct-route population under the current executor;
  or ignore/drop replay failures and continue.
- Chosen action: apply the frozen smoke stop. Do not run the full repair.
- Strongest objection: the current executor is internally deterministic and
  could define a new cache by dropping old failures. That changes the source
  supervision contract after observing the smoke and requires a new plan.
- How this differs from failed attempts: it preserves the failed prerequisite
  rather than mixing route-validity contracts within one repaired cache.
- Automatic execution authorized: no further action.
- Authorization basis: explicit user request to read and perform
  `plans/w2c_when_label_repair_plan.md`.
- Independent review: required because the plan left the bounded continuation
  strategy unspecified and the full live run is expensive. One read-only
  `research_reviewer` ranked the chosen action above GraphMCTS adaptation and
  known-suffix-only repair with medium confidence; the ranking was stable and
  the strongest objection was the same bounded-validity limitation recorded
  above.
- Stop condition: satisfied early because exact old-route replay failed. The
  640-sample repair and all downstream work remain unexecuted.

## Latest Research-Action Result
- Action taken: froze the 640-sample contract and 12-sample cohort, executed
  the smoke on four GPUs, verified byte-stable resume and all repair trace
  invariants, then ran one bounded four-GPU replay-stability diagnostic.
- Result: smoke **FAIL**. There were 37 incorrect old-route replays among 312,
  across 10/12 samples; all other smoke checks passed and zero samples were
  quarantined.
- Evidence saved: `analysis/w2c_when_repair/decision_summary.md`, preflight
  artifacts, `smoke/smoke_executions.jsonl`, `smoke/smoke_gate.json`,
  `smoke/smoke_report.md`, and `smoke/replay_failure_diagnostic.json`; raw
  records under `/mnt/hyemin/qwen_train_eval/outputs/w2c_when_repair_v1/`.
- Failure or issue: the original label-record executor hashes differ from the
  current code; the exact original executor artifact is unavailable here.
- Lesson learned: transferred correct-route labels are not portable across a
  changed four-action execution implementation even when model revision,
  evaluator, prompt hashes, and high-level contract ID match.
- Next implication: recover the exact source executor and rerun only the smoke
  gate, or authorize a new current-runtime cache-regeneration plan.
