# Phase 26: Binary CAP26/CAP24 NLL5 Executed Validation

## Current Objective

Train matched CAP26 and CAP24 Image+Question binary predictors for five epochs
with exact valid-set NLL, select checkpoints by actual validation execution
accuracy, and run the unchanged external evaluation.

## Active Constraints

- Frozen GQA/TextVQA/ChartQA max-50 MCTS supervision; no new search.
- Common CAP24-eligible population: 6,007 train / 872 validation.
- Exact-set-NLL, direct factorized 28-bit head, five epochs.
- Actual route-conditioned validation generation is the primary checkpoint
  selector; cached Hit@1 is diagnostic only.
- Two concurrent one-GPU pipelines on node02; node03 is prohibited.
- No architecture change or follow-on experiment.

## Current State

- Done: bounded protocol and matched population chosen.
- Done: implementation, 20 focused tests, checksum-frozen manifests/configs,
  and static readiness PASS.
- In progress: jobs 102961 (CAP26) and 102960 (CAP24) are submitted to node02
  and pending GPU availability.
- Blocked: none.
- Most recent useful observation: CAP26 and CAP24 have identical eligible UID
  coverage, so the comparison can use all 6,879 common positive records.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| CAP26 and CAP24 eligibility both equal 6,007 train / 872 validation | frozen parent-manifest geometry check | Removes sample-composition confounding | confirmed |
| CAP22 BCE selection tied at zero cap-valid Hit@1 and selected an ALL-ON checkpoint | `outputs/binary_cap_sweep_v1/cap22/history.json` | Supports replacing cache membership as primary selector | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Cache-Hit checkpoint selection | Incomplete route caches can leave all epochs tied or near zero | supported proxy-selection limitation | CAP sweep histories | Select prospectively by executed validation behavior | Do not use external test outcomes for selection |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| CAP26 exact NLL | Retains moderately dense valid alternatives | Whether a permissive cap avoids sparse collapse | high | testing |
| CAP24 exact NLL | Removes more dense routes under the same inputs | Accuracy/compute effect of the cap under coherent NLL | high | testing |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: implement and execute the explicitly bounded matched comparison.
- Confirmed observation: both caps support exactly the same train/validation UIDs.
- Unverified interpretation: exact-set-NLL plus executed-accuracy checkpoint selection may avoid the BCE cap failures.
- Diagnosis: unknown
- Viable alternatives considered: cache-Hit selection was rejected prospectively because it is an incomplete-cache proxy; executed accuracy directly measures the target behavior.
- Chosen action: two five-epoch exact-NLL pipelines with five-checkpoint executed validation and unchanged external evaluation.
- Strongest objection: executing 4,360 validation routes per cap adds substantial runtime, but it is bounded and prevents proxy-based checkpoint selection.
- How this differs from failed attempts: changes objective to exact set NLL and freezes behavioral validation selection before outcomes.
- Automatic execution authorized: yes
- Authorization basis: explicit user request on 2026-08-20.
- Stop condition: both frozen pipelines complete, or a technical integrity failure occurs.

## Latest Research-Action Result

- Action taken: implemented and froze the two train-to-validation-to-external
  pipelines, then submitted one GPU per cap.
- Result: readiness PASS; Slurm jobs 102961/102960 are pending on node02.
- Evidence saved: `outputs/binary_cap_nll5_v1/audits/training_readiness_v1.json`,
  configs under `configs/binary_cap{26,24}_nll5_execval_v1.yaml`, and focused tests.
- Failure or issue: the first local geometry freeze was stopped and archived
  before completion because it was too heavy for the login node. The unchanged
  preparation then passed on node05. Its chained config step safely stopped on
  already-created checksum-valid configs; a separate static readiness audit
  verified both configs and manifests exactly.
- Lesson learned: cap geometry preparation belongs on CPU Slurm; idempotency
  guards prevented silent overwrite during recovery.
- Next implication: when node02 releases two GPUs, both jobs can start without
  further scientific or implementation changes.
