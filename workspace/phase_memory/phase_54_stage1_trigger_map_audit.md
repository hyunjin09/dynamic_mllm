# Phase 54: Stage-1 Trigger Map Audit Memory

## Current Objective
Freeze the existing Shared Random-4 Stage-1 dynamic gate and construct a complete per-sample trigger map over the current 7,999-sample dense FULL split without new Qwen inference or corrective search.

## Active Constraints
- Follow `plans/stage1_trigger_map_audit_plan.md` and stop after trigger-map construction and descriptive diagnostics.
- Keep the Shared Random-4 checkpoint, global strict-crossing threshold, current LMMS labels, and Phase-48 image-group-disjoint split unchanged.
- Reuse saved validation/test score trajectories; recompute only missing train gate scores from frozen dense feature shards.
- Use four direct GPUs for the shard-local gate scoring workers while they remain idle; no Slurm exists on this server.
- Do not run four-action search, generate Stage-2 action labels, train Stage 2, alter thresholds, or add persistence/EMA logic.

## Current State
- Done: Read the plan, Phase-53 memory, workflow state, promoted Phase-51/52/53 evidence, source contracts, trigger sources, feature schema, and live compute state.
- Done: Confirmed 7,999 split rows (6,399 train / 800 validation / 800 test), saved 28-layer validation/test Shared Random-4 scores, a 1,600-row Phase-53 trigger manifest, and all frozen checkpoint/normalization files.
- Done: Froze final contract `d9dd591abcc3d56fb1caf56f57815e8a5749ebe631896f095a8118e3ea5653a7`; four direct GPU workers reconstructed all 6,399 train trajectories from verified stored features.
- Done: Completed 7,999 unique trigger rows, the six triggered C/W handoff manifests, all required metrics/figures, and a 41-file SHA-256 artifact audit.
- Done: The deterministic 24-record score check passed at maximum absolute error `2.81e-7` under the prospectively frozen `1e-6` tolerance, with all first-trigger decisions identical.
- Stopped: No Qwen forward, four-action search, Stage-2 labeling/training, threshold change, persistence/EMA, W-to-C repair, or external evaluation ran.
- Blocked: No.
- Most recent useful observation: the train handoff contains 1,881 triggered Dense-W and 39 triggered Dense-C samples; validation/test detection remains strongly task-dependent, with GQA wrong recall `0.180/0.185` versus ChartQA `0.880/0.800` and TextVQA `0.880/0.870`.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Frozen Shared Random-4 threshold is `0.9253839280601031` with strict `p > tau` over layers 0-27 | `analysis/dense_failure_stage1/gate_winner_selection/selected_winner.json`; `dense_failure_stage1/shared_global_gate.py` | Defines the trigger map without new selection | confirmed |
| Checkpoint and normalization SHA-256 are `1aeeaa27...1211` and `ce4b63ab...aed3` | `analysis/dense_failure_stage1/shared_global_gate/` | Binds train-score reconstruction to the selected gate | confirmed |
| Validation/test scores cover 800 unique UIDs each; train scores are absent | `analysis/dense_failure_stage1/shared_global_gate/evaluation/` | Selects reuse-first reconstruction route | confirmed |
| Phase-48 split has 6,399/800/800 rows and zero image-group overlap | `analysis/dense_failure_stage1/layerwise_failure_probe/` | Preserves population and future usage separation | confirmed |
| Final split preservation/recall/precision are train `0.9878/0.5878/0.9797`, val `0.9425/0.5300/0.9021`, test `0.9400/0.5100/0.8947` | `analysis/dense_failure_stage1/trigger_map/metrics/split_summary.csv` | Quantifies Stage-1 admission behavior and future train workload | confirmed |
| All 1,600 Phase-53 trigger rows and all 18 Phase-52 aggregate fields match exactly | `analysis/dense_failure_stage1/trigger_map/consistency_audit.json` | Shows the map preserves the frozen gate semantics | confirmed |
| All 41 final artifact hashes match and 36 focused/regression tests pass | `analysis/dense_failure_stage1/trigger_map/artifact_manifest.json`; test output | Establishes implementation and artifact completeness | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| First contract freeze | Protocol rendering stopped with missing `static_config` | supported: contract-construction schema defect | `analysis/dense_failure_stage1/trigger_map_invalid_contract_shape_20260831/INVALID_RUN.md` | Bind both the exact nested static snapshot and projected runtime fields; regression-tested before refreeze | Do not use the invalid metadata-only contract |
| First aggregate attempt | Prior validation comparison raised missing `records` | supported: wrong Phase-52 validation source schema | `analysis/dense_failure_stage1/trigger_map_invalid_prior_source_20260831/INVALID_RUN.md` | Use the frozen validation operating point in `selected_winner.json`; keep test comparison from its test CSV | Do not use the invalid worker outputs across a changed code contract |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Reuse val/test scores and shard-stream train features | All required inputs are frozen and hash-bound | Complete 7,999-row map without Qwen inference | low | completed |
| Full MLLM rerun | Could recreate features | Only needed if stored feature provenance fails | high | rejected |

## Next-Step Decision
- Deliberation mode: fast
- Active objective and bottleneck: fill the missing train Shared Random-4 score trajectories while preserving the frozen gate and proving consistency with prior validation/test outputs.
- Relevant memory item used: Phase 53 selected Shared Random-4 for dynamic treatment and froze its validation/test trigger cohorts; Phase 51 already saved its checkpoint and score trajectories.
- Confirmed observation: Existing artifacts fully cover validation/test, and frozen features cover every train UID.
- Unverified interpretation: None needed; this is deterministic reconstruction.
- Diagnosis: unknown
- Viable alternatives considered: reuse existing scores/features versus rerun Qwen; only the former is necessary.
- Chosen action: score train shards with the frozen lightweight gate on four direct GPUs, reproduce a deterministic 24-row saved-score subset, construct all manifests/metrics/figures, and require exact prior aggregate consistency.
- How this differs from failed attempts: no scientific method change, new search, label regeneration, or Qwen forward pass occurs.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to read and perform `plans/stage1_trigger_map_audit_plan.md`.
- Stop condition: required trigger maps, train candidate manifests, metrics, figures, decision summary, hashes, and compact state update are complete and audited.

## Latest Research-Action Result
- Action taken: Reused the saved validation/test Shared Random-4 trajectories and recomputed the missing train trajectories from hash-verified stored dense features with four shard-local GPU workers; then built and audited the complete trigger map.
- Result: Train has 3,160 C/no-trigger, 39 C/trigger, 1,319 W/no-trigger, and 1,881 W/trigger. Validation has 377/23/188/212 and test has 376/24/196/204. Dense-W triggers are early (layers 0-8) for 69.1%/69.8%/74.0% of triggered W on train/val/test, but this is descriptive only. GQA wrong recall remains much lower than ChartQA/TextVQA.
- Evidence saved: `analysis/dense_failure_stage1/trigger_map/decision_summary.md`, all `manifests/`, `metrics/`, five required figures, `consistency_audit.json`, and `artifact_manifest.json`.
- Failure or issue: Two implementation-only attempts stopped fail-closed and were quarantined: one malformed contract snapshot and one wrong prior-validation source schema. Neither produced accepted final maps or scientific results.
- Lesson learned: The frozen Shared Random-4 map is reproducible from stored features and strongly enriches triggered train samples for dense failure, but coverage is task-dependent and 1,319 train Dense-W samples remain outside any future Stage-2 opportunity under this architecture.
- Next implication: Any separately authorized triggered-W corrective suffix search should use exactly `manifests/train_triggered_wrong.jsonl` (1,881 rows) for label generation and `manifests/train_triggered_correct.jsonl` (39 rows) for FULL-suffix preservation supervision; validation/test manifests must remain non-training cohorts.
