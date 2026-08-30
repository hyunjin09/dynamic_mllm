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
- Apply the same algorithm and budget to train and validation. The original
  Phase-42 smoke ran on four RTX 6000 Ada GPUs through direct `torchrun` on a
  server without Slurm. This H100 server instead requires Slurm for any future
  GPU replay; that machine difference is now part of the parity question.
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
- Stopped: the exact contract-bound historical source has now been
  cryptographically reconstructed in a separate ignored tree, but end-to-end
  parity remains untested because the audit-only H100 replay was canceled
  before allocation at the user's instruction. Proceeding with label repair is
  still unauthorized.
- Most recent useful observation: all 16/16 historical source hashes and the
  frozen YAML hash match the label contract. The historical fixed-route
  executor is scientifically valid and source-equivalent to the current fixed
  path, apart from later unused API additions and one no-op device placement
  change. The leading H100-versus-RTX numerical-path explanation remains an
  inference, not a verified cause.

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
| Historical dirty-worktree source reconstructs to all 16/16 contract hashes and implements the intended READ/WRITE truth table | `analysis/executor_provenance_audit/executor_provenance_audit.md`; `historical_vs_current_executor_diff.md` | Rules out a semantic change in the fixed-route executor source as the observed replay cause | confirmed |
| Recovered-source H100 replay job 1763 was canceled before allocation at user instruction | `analysis/executor_provenance_audit/replay_parity_report.md`; Slurm accounting | Leaves cached-token parity and the hardware/kernel explanation unresolved | confirmed |

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
| Run the recovered exact source executor on the existing smoke replay gate | Tests the final unverified execution-parity component | Whether repair can start from a valid route population | low-to-medium | source recovered; GPU replay explicitly deferred |
| Rebuild all W2C correct routes under the current executor | Establishes a self-consistent new cache when the old executor is unavailable | A new current-runtime supervision population | high | alternative new plan; not authorized |

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: rebuild trustworthy W2C WHEN labels; exact
  cached-token replay under the recovered historical runtime is still
  unverified.
- Relevant memory item used: 37/312 cached positives fail on RTX 6000 Ada even
  though the current fixed-route implementation is deterministic.
- Confirmed observation: the historical dirty-worktree source is exactly
  reconstructed and its action semantics are valid; the current fixed-route
  path has no semantic source difference that explains the mismatch.
- Unverified interpretation: H100-versus-Ada BF16 SDPA/GEMM behavior is the
  leading explanation for the historical-versus-current token divergence.
- Diagnosis: root cause unknown; hardware/kernel numerical-path drift is
  suspected, not supported, because the discriminating replay did not run.
- Chosen action: stop after the provenance audit. Do not regenerate labels,
  modify either executor, or resume W2C repair.
- Automatic execution authorized: none.
- Authorization basis: explicit executor-provenance audit request followed by
  explicit instruction to report without running the GPU job.
- Independent review: one required read-only reviewer confirmed exact source
  recovery and valid semantics, and recommended a fail-closed end-to-end
  classification unless the H100 replay reaches 312/312 exact parity.
- Stop condition: satisfied. Audit job 1763 was canceled before allocation and
  consumed zero GPU time.

## Latest Research-Action Result
- Action taken: audited label contracts, Git history and objects, dirty-tree
  hashes, launch wrappers, import resolution, executor semantics, generation
  settings, and Phase-42 replay evidence; reconstructed the historical source
  separately and prepared—but did not execute—the exact 12-sample replay.
- Result: all 16/16 source hashes and the config hash match. FULL, READ_ONLY,
  WRITE_ONLY, and IGNORE implement the intended state-transition truth table,
  with text READ using pre-layer visual K/V. Exact cached-token parity is not
  measured because pending job 1763 was canceled at user instruction.
- Evidence saved: `analysis/executor_provenance_audit/executor_provenance_audit.md`,
  `historical_vs_current_executor_diff.md`, and `replay_parity_report.md`.
- Failure or issue: the complete original numerical execution environment was
  not frozen, and the distinguishing H100 replay remains unexecuted.
- Next implication: no W2C repair or label regeneration is authorized. If
  later requested, run only the existing 12-sample/312-route recovered-source
  H100 replay before reconsidering label portability.
