# Phase 34: Three-Action Answer-Aligned Label Conversion Memory

## Current Objective

Execute `plans/4way_labeling_fix.md`: reuse every authoritative positive binary MCTS route across five datasets and refine its OFF positions into answer-aligned READ_OFF, WRITE_OFF, or BOTH_OFF supervision without rerunning MCTS.

## Active Constraints

- Reuse the validated unified executor; FULL is a cached reference, not a fourth exploration action.
- Use the frozen 12,278-sample / 545,531-route source inventory and current unified runtime semantics.
- Positive routes must be evaluator-correct; continuous support guides screening/ranking but cannot admit a wrong route.
- Freeze a within-unified repeatability epsilon before full aggregate analysis; native-vs-unified drift is not the threshold.
- Evaluate every intervention in its source-route context and validate every combined final route jointly.
- Keep W2C hard/soft correction and C2C compensated-alignment labels semantically separate.
- Use all eight H100s with 16 workers after a new-semantics five-dataset pilot passes.
- Preserve the checksum-gated oversized-image compatibility repair in `label_regeneration/runtime.py`.
- Write to a new contract-isolated three-action output root; do not mix old four-action outputs or overwrite source labels.

## Current State

- Superseded on 2026-08-25 by the user-approved exact sequential four-action
  conversion in `plans/4way_labeling_3.md`; active state moved to Phase 35.
- Job 1609 was canceled after 24/56 pilot samples because its immutable partial
  evidence already failed the prospective beam-stability gate; dependent job
  1610 was canceled before execution.
- Done: the modified plan, Phase 33 evidence, source inventory, old pilot audit, and promoted route-context lesson were reviewed.
- Done: one required read-only independent review returned `stable` for a clean new-root implementation.
- In progress: final contract and 404/404 CPU gates pass; job 1609 is executing calibration/pilot on all eight H100s, and fail-closed full job 1610 is queued with `afterok:1609`.
- Blocked: no infrastructure blocker. Full conversion remains scientifically gated on the new-semantics pilot audit.
- Most recent useful observation: calibration completed 56/56 with zero failures, froze within-unified epsilon `1e-6` from 224 differences, and the pilot started all 16 intended workers.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Frozen authority contains 12,278 samples and 545,531 positive routes | `datasets/mcts_labels_4action/source_inventory_v1/`; `analysis/4action_label_conversion/implementation_audit.md` | Reusable input boundary; no MCTS or source reselection is needed | confirmed |
| Route-conditioned necessity differs strongly from FULL-context local effects | `analysis/4action_route_conditioned/aggregate_summary.json`; `workspace/decision_log.md` 2026-08-24 lesson | Requires route-context screening and joint refinement | confirmed |
| Existing executor/source/concurrency pilot passed 56/56 | `analysis/4action_label_conversion/pilot_audit_v1.json` | Reusable foundation, but not a new-semantics gate | confirmed, limited |
| Oversized frozen images are valid under exact native preprocessing | `analysis/4action_label_conversion/experiment_log.md`; `tests/test_label_regeneration.py` | The checksum-gated compatibility repair must remain in every new run | confirmed |
| Independent review ranks clean three-action implementation above retrofit/continuation | research-control reviewer result in the 2026-08-25 session | Supports contract isolation and a fresh pilot | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Treat every W2C OFF retained by binary MCTS as corrective | Earlier plan retained some suppressions based on correctness-only purification | supported objective mismatch under the new approved criterion | `plans/4way_labeling_fix.md`; route-conditioned redundancy evidence | Screen both discrete necessity and continuous alignment contribution | Do not call tolerated OFF answer-unaligned |
| Treat C2C routes as mechanical efficiency labels | Correctness-preserving OFF does not establish improved answer alignment | supported scientific-criterion change | `plans/4way_labeling_fix.md` sections 12--17 | Require correctness plus support gain above frozen epsilon | Do not mix efficiency metadata with C2C alignment supervision |
| Open all frozen images under the local Pillow default | Valid oversized Standard image raised `DecompressionBombError` | supported machine/Pillow threshold mismatch | job 1604 failure and exact replay evidence | Retain hash-gated scoped retry | Do not remove or bypass the content-hash gate |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Clean three-action implementation and new-root pilot/full run | Exactly matches the approved objective and isolates score semantics/contracts | Produces answer-aligned W2C/C2C labels | high | selected |
| Retrofit the old four-action output/schema | Reuses more reporting code | Reduces implementation churn | medium | rejected: semantic contamination risk |
| Continue pending old job 1605 | Existing implementation is tested | Produces old correctness/efficiency labels | high compute | rejected: obsolete objective |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: implement the approved three-suppression conversion; the key validity bottleneck is prospectively validating score repeatability, reference handling, and bounded-beam stability.
- Relevant memory item used: binary route effects are trajectory-context dependent, and transferred current-runtime predicates must be recomputed.
- Confirmed observation: old executor/source/concurrency gates pass, but old C2C mechanics and correctness-only W2C purification do not satisfy the new answer-alignment definition.
- Unverified interpretation: normalized teacher-forced support will separate meaningful operations above a stable within-unified epsilon, and beam width 8 will be stable enough for canonical selection.
- Diagnosis: supported objective/contract change; evidence is `plans/4way_labeling_fix.md` and the reviewer decision packet.
- Viable alternatives considered: clean new-root implementation; retrofit old schema/output; continue old job 1605.
- Chosen action: cancel zero-progress job 1605, implement the three-suppression converter and score/noise gates in a new output contract, run CPU/synthetic tests, then run a fresh five-dataset pilot. Freeze epsilon from repeated identical unified routes and compare beam 8 with one wider bounded beam before automatically launching the full conversion.
- Strongest objection: score/reference instability could turn numerical variation into supervision; the prospective repeatability/reference/beam gates directly test this before full scale.
- How this differs from failed attempts: it does not reuse pre-fix output semantics, does not treat every binary OFF as causal, and does not use native/unified drift as an effect threshold.
- Automatic execution authorized: yes
- Authorization basis: the user's explicit request to perform `plans/4way_labeling_fix.md`, cancel the pending old run, and relaunch the modified run.
- Stop condition: unresolved executor parity, evaluator/reference-policy defect, unstable within-unified scoring/epsilon, or materially unstable beam classification at the pilot gate.

## Latest Research-Action Result

- Superseding result: `analysis/three_action_answer_aligned_label_conversion/early_stop_audit.md`
  records 322/1,417 beam canonical mismatches and 167/1,417 Jaccard failures.
  Executor parity/correctness remained clean, so this is preserved as a
  beam-policy failure rather than a model/executor failure.
- Action taken: implemented and reviewed the complete clean three-action contract, passed 404/404 tests, canceled superseded zero-runtime jobs 1605--1607, and launched calibration/pilot job 1609 plus fail-closed dependent full job 1610.
- Result: calibration contract `2cbee2ba...` is frozen; 56/56 calibration records passed with epsilon `1e-6`, and the real modified-label pilot is running on all 8 GPUs/16 workers.
- Evidence saved: `analysis/three_action_answer_aligned_label_conversion/`, `tasks/plan.md`, `tasks/todo.md`, new tests/code/config, and Slurm accounting for jobs 1605--1607.
- Failure or issue: no scientific/runtime failure. Jobs 1606 and 1607 were canceled before start to correct, respectively, shell control flow and a missing explicit joint-composition analysis control.
- Lesson learned: semantic executor parity from the old pilot is reusable, but label-validity evidence is not transferable across the objective change; scheduler wrappers with background telemetry must be reviewed as shell control flow.
- Next implication: monitor job 1609. Job 1610 will automatically run the formal pilot audit and compute estimate, then launch the full conversion only if every gate passes.
