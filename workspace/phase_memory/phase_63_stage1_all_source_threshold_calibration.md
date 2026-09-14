# Phase 63: Stage-1 ALL-Source Robust Threshold Calibration Memory

## Current Objective
Calibrate one shared global any-layer threshold for the frozen Phase-62 ALL-source
Shared Random-4 head, prioritizing Dense-C preservation across Historical and
Canonical source regimes.

## Active Constraints
- Follow `plans/stage1_all_source_robust_threshold_calibration_plan.md` as one bounded action.
- Freeze the Phase-62 head ensemble, old normalization, score trajectories, feature definition, and strict `score > tau` rule.
- Select only on Historical validation plus Canonical OOF trajectories; keep Historical test out of threshold selection.
- Primary rule: maximize pooled wrong recall subject to at least 98% C preservation in both calibration sources.
- Do not regenerate a trigger map, touch Stage 2, rerun search, retrain a head, or run external evaluation.

## Current State
- Done: Read the plan and Phase-62 decision state.
- Done: Confirmed Phase 62 produced 4,000 Canonical OOF and 1,600 Historical held-out score trajectories under contract `08dbf146...90a4fa`.
- Done: Verified all 85 Phase-62 artifact hashes, five main checkpoint hashes, the normalization hash, score schemas/model IDs, and all population identities.
- Done: Froze calibration contract `30288d8a...262a38` before computing outcomes.
- Done: Ran the full breakpoint sweep, five cross-fit checks, 99/98/95 references, 20,000 group-bootstrap replicates, dataset/source and trigger-depth audits, and the old-threshold comparison.
- Done/stopped: Decision A freezes `tau=0.9711347410314399` and gate hash `d3b019b3...cae4f`; 33 artifact hashes and 24 focused/inherited tests pass. No downstream action ran.
- In progress: none.
- Blocked: none.
- Most recent useful observation: the 98% gate preserves Historical/Canonical C at 0.9875/0.9808 and recalls W at 0.0650/0.1240, but Canonical TextVQA recalls 0/19 W.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| ALL-source Historical/Canonical AUROC is 0.8394/0.7689 | `analysis/dense_failure_stage1/all_source_robustness/main_all/metrics/source_summary.csv` | Supports calibrating the frozen shared head on the declared mixture | confirmed |
| Canonical ChartQA AUROC is 0.5826; Canonical TextVQA has 19 W | `analysis/dense_failure_stage1/all_source_robustness/main_all/metrics/dataset_source_breakdown.csv` | Requires dataset-cell and uncertainty audits before freeze | confirmed |
| Historical validation/test are separate 800-record balanced populations | Phase-62 ensemble score manifest | Allows threshold selection without Historical test leakage | confirmed |
| Primary `tau=0.9711347410` gives worst-source C preservation 0.9808 and pooled W recall 0.1054 | `metrics/reference_operating_points.csv` | Passes the fixed conservative selection rule with nontrivial detection | confirmed |
| Cross-fit threshold range is 0.00361 and minimum held-out Canonical-fold C preservation is 0.9728 | `crossfit/threshold_stability.csv` | Passes the prospective stability gate | confirmed |
| No supported cell is catastrophic; Canonical ChartQA C preservation is 0.9842 | `metrics/dataset_source_breakdown.csv` | Clears the dataset/source freeze gate | confirmed |
| Untouched Historical test C preservation/W recall is 0.9975/0.0650 | `metrics/source_breakdown.csv` | Provides an independent Historical check without selection leakage | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Old Historical-only global gate on new canonical data | Excessive canonical Dense-C triggering | supported source-regime boundary mismatch | Phase-60/61 reports | Require worst-source preservation rather than pooled-only calibration | Do not reuse the old threshold for compatibility |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| 98% worst-source operating point | Prospective primary rule balances conservative admission and W detection | Whether a robust in-scope gate can be frozen | low | completed / frozen |
| 99% reference point | More conservative safety reference | Cost of stronger C preservation | low | completed / reference only |
| 95% reference point | More permissive diagnostic only | Recall available if 98% is impractical | low | completed / diagnostic only |

## Next-Step Decision
- Deliberation mode: standard
- Active objective and bottleneck: select a conservative in-scope threshold without assuming source robustness implies safe dataset-cell behavior.
- Relevant memory item used: Phase 62 repaired aggregate source transfer but retained weak Canonical ChartQA and sparse Canonical TextVQA-W support.
- Confirmed observation: complete out-of-fold/held-out trajectories already exist, so no new training or GPU inference is needed.
- Unverified interpretation: useful W recall may survive the 98% worst-source preservation constraint.
- Diagnosis: supported source-regime repair with unresolved operating-point viability.
- Evidence path if diagnosis is not unknown: `analysis/dense_failure_stage1/all_source_robustness/`.
- Viable alternatives considered: 98%, 99%, and 95% preservation operating points under the same frozen head.
- Chosen action: run the prescribed any-layer sweep, five canonical cross-fit checks, fixed-point audits, and bootstrap uncertainty; freeze only if Decision A conditions pass.
- Strongest objection: aggregate source constraints may hide a catastrophic Canonical ChartQA cell even when both source-level preservation rates exceed 98%.
- How this differs from failed attempts: selection is source-worst-case, uses canonical OOF trajectories, and keeps Historical test outside calibration.
- Automatic execution authorized: yes
- Authorization basis: the user explicitly requested this plan.
- Stop condition: after threshold sweep, stability/reference audits, Decision A/B/C, and optional gate freeze; no trigger-map regeneration.

## Latest Research-Action Result
- Action taken: calibrated the exact any-layer gate over Historical validation plus Canonical OOF scores, evaluated cross-fold stability and dataset/source safety, and checked the frozen point on untouched Historical test.
- Result: Decision A. Freeze strict `score > 0.9711347410314399`; worst-source C preservation is 0.9808 and pooled W recall is 0.1054.
- Evidence saved: `analysis/dense_failure_stage1/all_source_threshold_calibration/`.
- Failure or issue: no execution failure. Canonical TextVQA has 0/19 detected W, and canonical calibration remains OOF rather than a pristine external test.
- Lesson learned: conservative worst-source calibration removes the ALL-head L0 canonical false triggers seen at the old threshold and retains modest W opportunities without a catastrophic supported cell.
- Next implication: await explicit authorization for a new robust trigger-map and Stage-2 label compatibility audit; do not reuse old trigger samples or layers automatically.
