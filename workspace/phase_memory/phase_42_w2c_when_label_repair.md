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
- In progress: freeze the repair manifest, protocol, config, pre-repair audit,
  and 12-sample smoke cohort before any live model execution.
- Blocked: none.
- Most recent useful observation: the frozen initial-boundary search space has
  1,269 compatible known suffixes plus 55,293 possible one-edit variants; the
  cap selects at most 34,941 variants and completely enumerates the one-edit
  neighborhood for 470/640 initial states.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Forced FULL had a compatible correct continuation for 39/128 validation W2C boundaries | `analysis/selective_continue_deviate/when_label_completeness_results.json` | Directly establishes WHEN-label cache incompleteness | confirmed |
| All 252 Phase-41 audit routes completed with zero unresolved states | `analysis/selective_continue_deviate/when_full_insertion_executions.jsonl` | Separates the scientific failure from a runtime failure | confirmed |
| Authoritative repair population is 512 train and 128 validation W2C samples | `analysis/persistent_corrective_supervision/training_manifest.jsonl`; `analysis/persistent_corrective_supervision/boundary_manifest.jsonl` | Fixes the population and minimum downstream candidate counts | confirmed |
| The physical source records and exact evaluator/model contract remain available on this server | `/mnt/hyemin/qwen_train_eval/datasets/mcts_labels_4action/sequential_branching_v1/full/records`; Phase-39 manifest | Makes a full live repair feasible without reconstructing transferred assets | confirmed |
| One-base one-edit suffix neighborhoods have at most 81 variants | deterministic pre-repair enumeration from the frozen routes | A cap of 96 exhausts a single-base neighborhood and permits limited multi-suffix coverage | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Phase-38 four-action isolation | Both router families remained near-all-FULL with zero held-out W2C rescue | supported WHEN generalization failure; exact cause unknown | `analysis/4action_collapse/decision_summary.md` | Repair the target before another router run | Another unchanged four-action training run |
| Phase-41 selective-gate prerequisite audit | 39/128 mandatory boundaries accepted FULL under a known continuation | supported cache incompleteness; discovery omission cause unknown | `analysis/selective_continue_deviate/stage1_decision_summary.md` | Iteratively augment the correct-route cache and recompute boundaries | Train a gate on the known-incomplete DEVIATE labels |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Known suffixes, then deterministic one-edit suffix repair capped at 96 per boundary | Directly reuses verified continuations, then expands locally with explicit bounded semantics | Whether enough clean WHEN candidates remain after iterative repair | high | selected and implementing |
| Adapt binary GraphMCTS to four actions | Broader continuation search could find multi-edit rescues | Whether remaining states survive a substantially wider search | high | rejected for this action; larger semantic and implementation change |
| Known-suffix-only repair | Minimal and directly reproduces Phase-41 rescue mechanism | Repairs proven omissions but cannot discriminate nearby unseen continuations | medium | rejected as underpowered |

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: rebuild trustworthy W2C WHEN labels; the
  cache contains proven omissions and the broader validity of FULL is unknown.
- Relevant memory item used: Phase 41 found 39/128 compatible known-suffix
  rescues and stopped gate training prospectively.
- Confirmed observation: some mandatory boundaries move when a valid FULL
  continuation is added; the same sample can therefore require repeated repair.
- Unverified interpretation: how many additional valid FULL continuations a
  broader multi-edit or MCTS search would find after this bounded action.
- Diagnosis: supported cache incompleteness; discovery mechanism unknown.
- Evidence path if diagnosis is not unknown:
  `analysis/selective_continue_deviate/when_label_completeness_results.json`.
- Viable alternatives considered: reviewed one-edit bounded expansion, a
  four-action GraphMCTS adaptation, and known-suffix-only repair.
- Chosen action: known suffixes first, followed only when needed by at most 96
  deterministic layer-stratified one-edit suffix candidates at each new
  boundary; add all correct routes and iterate to convergence or bounded
  non-rescue.
- Strongest objection: one-edit non-rescue cannot establish that FULL is truly
  invalid. The output label and report retain this limitation explicitly.
- How this differs from failed attempts: it changes the supervision cache and
  recomputes boundary targets; it does not retrain a collapsed router.
- Automatic execution authorized: yes.
- Authorization basis: explicit user request to read and perform
  `plans/w2c_when_label_repair_plan.md`.
- Independent review: required because the plan left the bounded continuation
  strategy unspecified and the full live run is expensive. One read-only
  `research_reviewer` ranked the chosen action above GraphMCTS adaptation and
  known-suffix-only repair with medium confidence; the ranking was stable and
  the strongest objection was the same bounded-validity limitation recorded
  above.
- Stop condition: complete the repair and post-repair audit, answer Q1-Q6, then
  stop before any gate/router training or follow-on experiment.

## Latest Research-Action Result
- Action taken: implementation and pre-execution audit only; live smoke is not
  yet started.
- Result: pure repair tests pass and the four-GPU executor compiles.
- Evidence saved: `four_action_policy/when_repair.py`,
  `experiments/prepare_w2c_when_repair.py`,
  `experiments/run_w2c_when_repair.py`, and
  `tests/test_w2c_when_repair.py`.
- Failure or issue: none material. Test discovery requires `PYTHONPATH=.`
  because the repository package is not installed into `.venv`.
- Lesson learned: live execution must remain resumable at sample granularity
  because iterative route costs vary substantially across W2C samples.
- Next implication: freeze all manifests/hashes and run the smoke gate.
