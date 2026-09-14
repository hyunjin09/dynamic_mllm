# Phase 44: Current Dense Stage-1 Regeneration Memory

## Current Objective
Regenerate authoritative dense all-on correctness for the recovered VQA candidate
population under one frozen current runtime, optionally extracting parity-safe
28-layer pooled states, then freeze a leakage-safe Stage-1 split.

## Active Constraints
- The target is current dense all-on final correctness only; historical buckets
  are selection metadata and must not overwrite regenerated outcomes.
- Use native Qwen2.5-VL dense generation with all 28 decoder layers fully on;
  no sparse, binary-route, or four-action intervention is permitted.
- Gate the full run on 24-record current-runtime repeatability and exact token
  parity with passive feature extraction.
- Use four direct GPUs in parallel; this server has no Slurm.
- Stop after regeneration, label/group audit, and split creation. Do not train a
  Stage-1 predictor or resume route-repair work.
- Attempt all 8,000 recovered candidates. If a genuinely unusable record remains,
  report and exclude it rather than fabricating data, and preserve population
  proportions in the final split.

## Current State
- Done: found the tracked portable source manifest containing all 8,000 requested
  GQA/ChartQA/TextVQA rows and excluded its 2,000 DocVQA rows.
- Done: verified exact annotation-field agreement on all 6,917 rows overlapping
  the transferred positive-route derivative.
- In progress: freeze the reconstructed population and current dense execution
  contract, then run smoke gates.
- Blocked: none.
- Most recent useful observation: the previous Phase-43 metadata blocker was a
  search-location issue, not a missing candidate population; the portable
  reproduction manifest contains the needed sample identities and annotations.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Portable manifest has 4,000 GQA, 2,000 ChartQA, 2,000 TextVQA rows | `search/greedy_phase1_phase2_reproduction/manifests/all_samples.jsonl` | Restores the complete candidate annotations without positive-route selection | confirmed |
| All required semantic fields agree for 6,917 transferred overlaps | preparatory overlap audit, to be frozen under the Phase-44 output root | Validates portable annotations against independent transferred rows | confirmed |
| Physical image tree has all 8,000 historical-bucket filenames | `datasets/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713/images/` | Supports current dense execution | confirmed |
| Image content yields 7,477 groups | Phase-43 duplicate audit; recomputed in Phase 44 | Requires group-disjoint split | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Treat only transferred positive-route inventory as candidate metadata | 1,083 rows appeared unavailable | supported incomplete derivative, but not incomplete repository handoff | Phase-43 source audit plus portable 10K manifest | Search repository reproduction bundles before declaring candidate identity unavailable | Do not restrict Stage-1 candidates by route positivity |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Native dense regeneration without features | Smallest authoritative label path | Current correctness labels | high | fallback if feature parity fails |
| Native dense regeneration with passive pooled-state hooks | Avoids a second 8K pass | Labels plus future predictor inputs | high | pending smoke parity |

## Next-Step Decision
- Deliberation mode: standard
- Active objective and bottleneck: establish one self-consistent current dense
  target population; the remaining gate is runtime and feature repeatability.
- Relevant memory item used: Phase 43 established that route-positive derivatives
  are not a valid population authority.
- Confirmed observation: the complete 8K annotation population is recoverable
  independently of route labels.
- Unverified interpretation: passive hooks will preserve exact tokens on this
  Ada runtime.
- Diagnosis: unknown until the smoke comparison.
- Evidence path if diagnosis is not unknown: none.
- Viable alternatives considered: labels only; labels plus passive features.
- Chosen action: freeze the dense contract and run the 24-record repeatability
  and feature-parity smoke, enabling full-run features only on exact token parity.
- Strongest objection: feature collection adds runtime and memory cost; the
  parity gate and label-only fallback protect the primary label objective.
- How this differs from failed attempts: current labels are regenerated from the
  full sample population and do not depend on historical route availability or
  output parity.
- Automatic execution authorized: yes.
- Authorization basis: explicit user specification of the new Stage-1 data
  contract and four-GPU execution.
- Stop condition: after full valid-population generation, audits, feature freeze
  if parity-safe, and image-group-disjoint split creation.

## Latest Research-Action Result
- Action taken: in progress.
- Result: pending.
- Evidence saved: `analysis/dense_failure_stage1/current_dense_regeneration/`.
- Failure or issue: none currently.
- Lesson learned: pending.
- Next implication: pending smoke gates.
