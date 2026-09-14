# Phase 43: Stage-1 Dense-Failure Label Audit Memory

## Current Objective
Audit and freeze the 8,000 dense all-on correctness labels for future Stage-1
failure prediction, stopping before hidden-state extraction or training.

## Active Constraints
- Follow `plans/stage1_dense_failure_8k_label_audit_plan.md` (SHA-256
  `1e645d2f59df580606be067812d78d4e2c3e19e8973ce78387e9cc727fcdcc6d`).
- Use only dense all-on replay; do not invoke four-action routes.
- Do not train, extract the full hidden-state population, resume W2C repair, or
  freeze a split from incomplete source metadata.
- Use four direct GPUs for the replay only after its complete 8K source gate passes.

## Current State
- Done: audited all 8,000 physical images, the 6,917-row positive-route source
  derivative, current-label count reconstruction, and exact image-content groups.
- In progress: none.
- Blocked: the six 8K source JSONLs and canonical regenerated-label bundle are absent.
- Most recent useful observation: images are complete, but 1,083 zero-positive
  source rows and every original A6000 dense prediction/token record required
  for parity are unavailable.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| 8,000 images are present with historical 4K/4K buckets | `analysis/dense_failure_stage1/8k_label_audit/population_audit.md` | Image payload is not the blocker | confirmed |
| Available source derivative contains only 6,917 positive-route rows | `analysis/dense_failure_stage1/8k_label_audit/source_inventory.md` | It excludes all 1,083 zero-positive samples | confirmed |
| Reconstructed current labels are 4,045 correct / 3,955 wrong | `population_counts.csv`; Phase-12 P5 record | Separates historical balance from the Stage-1 target | supported reconstruction |
| Original dense predictions are missing from the derivative | `source_inventory.md` | Prevents token/answer parity | confirmed |
| Physical images form 7,477 content groups | `duplicate_group_audit.md` | Requires group-disjoint splitting | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Freeze 128 replay rows from transferred assets | Complete 8K manifest and 1,083 rows are absent | supported incomplete transfer for this plan | `source_inventory.md` | Restore metadata before replay | Do not substitute a biased 6,917-only cohort |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Restore metadata-complete canonical 8K bundle | Directly supplies labels, prompts, answers, groups, and cached outputs | Enables valid 128-row replay | low | required |
| Replay only 6,917 positive-route rows | Those rows have prompts and labels | Partial compatibility only | medium | rejected as nonrepresentative |
| Rebuild all 8K labels now | Would create one current Ada contract | New label-generation action; 1,083 prompts absent | high | not authorized / infeasible |

## Next-Step Decision
- Deliberation mode: standard.
- Active objective and bottleneck: establish label authority; complete 8K source metadata is missing.
- Relevant memory item used: Phase 12 distinguishes historical buckets from fresh current-dense authority.
- Confirmed observation: all images exist, but the authoritative label/contract artifacts do not.
- Unverified interpretation: none needed for the stop decision.
- Diagnosis: supported incomplete transferred asset set for this plan.
- Evidence path if diagnosis is not unknown: `analysis/dense_failure_stage1/8k_label_audit/source_inventory.md`.
- Viable alternatives considered: restore metadata; biased partial replay; full regeneration.
- Chosen action: record `EXECUTION_CONTRACT_UNRESOLVED` and stop before GPU replay.
- Strongest objection: a 6,917-only replay could still measure some drift, but cannot validate the specified 8K target and cannot compare cached dense tokens.
- How this differs from failed attempts: no scientific inference is made from a derivative that omits an entire outcome-defined cohort.
- Automatic execution authorized: no further action.
- Authorization basis: explicit plan execution; the plan's source-validity and stop gates fail.
- Stop condition: satisfied before model load or GPU execution.

## Latest Research-Action Result
- Action taken: completed the source/count/balance/duplicate feasibility audit and evaluated replay readiness.
- Result: `EXECUTION_CONTRACT_UNRESOLVED`; no split or replay subset was fabricated.
- Evidence saved: `analysis/dense_failure_stage1/8k_label_audit/`.
- Failure or issue: canonical 8K metadata and dense outputs were not transferred.
- Lesson learned: complete images plus positive-route derivatives are insufficient for dense-failure label authority.
- Next implication: transfer the minimal canonical metadata bundle, then run only the 128-record four-GPU dense replay.

