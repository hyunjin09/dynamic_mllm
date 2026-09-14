# Phase 69: Full-Benchmark End-to-End Evaluation Memory

## Current Objective
Evaluate the one frozen method candidate—robust ALL-source Stage-1 Shared Random-4 at strict P90 followed by Phase-66 Stage-2 Experiment A—against paired dense all-FULL inference on the complete established ChartQA, TextVQA, MMMU-Pro, and POPE populations.

## Active Constraints
- Follow `plans/full_benchmark_end_to_end_evaluation_plan.md` as one bounded research action.
- Evaluate ChartQA; TextVQA validation; MMMU-Pro Standard and Vision test; and POPE adversarial, popular, and random. Exclude GQA, DocVQA, MMStar, and base MMMU.
- Freeze Stage-1 weights/normalization, strict P90 threshold `0.9061332901863008`, Stage-2 Experiment A checkpoint, executor semantics, generation settings, and existing LMMS-Eval-compatible task scorers.
- Every UID receives a paired dense all-FULL and routed result. Do not filter on correctness, trigger status, fixability, or previous search membership.
- Use direct execution on this non-Slurm server and all four GPUs when ready.
- Stop after full paired analysis, one method-level decision, and one unexecuted next-improvement recommendation.

## Current State
- Done: Completed the exact 19,960-UID paired evaluation and all required aggregation under final contract `63379eeff80fd5b046cb980cdfccea7fab232e15eb5b14924a24b60393327e83`. All 36 final artifact hashes pass.
- In progress: None; Phase 69 is complete and stopped.
- Blocked: No.
- Most recent useful observation: pooled W→C/C→W/net is `3/19/-16`, delta accuracy -0.000802 with paired-bootstrap 95% CI [-0.001253,-0.000351]. The exact frozen candidate is regression-dominated.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase-66 A yielded Historical-800 P90 W→C/C→W/net = 4/1/+3 | `analysis/dense_failure_stage2/shared_union_training/` | Justifies the frozen candidate and this external scale test | confirmed |
| The Stage-1 deployment object is the five-checkpoint probability mean with strict P90 | `analysis/dense_failure_stage1/all_source_threshold_calibration/inputs/head_manifest.json`; `configs/stage2_shared_union_threshold_comparison_v1.json` | Defines online admission exactly | confirmed |
| Stage-2 A checkpoint SHA-256 is `48536bfbf6ebf62071898aea91952b6fe6a3f13b5723b09f6d814eb0ad9bbc7f` | `analysis/dense_failure_stage2/shared_union_training/experiment_A_single/training/selected_checkpoint.json` | Binds the treatment policy | confirmed |
| Frozen external manifests total 19,960 rows: 2,500 ChartQA, 5,000 TextVQA, 1,730+1,730 MMMU-Pro, 3,000×3 POPE | `/mnt/hyemin/qwen_train_eval/datasets/eval/data/*/samples.jsonl` | Defines the full paired population | confirmed |
| TextVQA evaluation images were deliberately omitted from transfer | `workspace/dataset_inventory.md`; physical path audit | Requires exact recovery before launch | confirmed |
| Exact reference population and input/scoring method are frozen in Phase 69 | `analysis/dense_failure_stage2/full_benchmark_eval/frozen_protocol.json`; `eval/reference/shared_prefix_eval_20260812/` | Prevents sample/prompt/evaluator drift | confirmed |
| Seven-variant paired parity smoke passed under final contract | `analysis/dense_failure_stage2/full_benchmark_eval/smoke/smoke_report.json` | Validates deterministic paired execution and exact hooked-versus-hook-free native dense equivalence, including trailing-vision MMMU-Pro geometry | confirmed |
| Independent review's only objection was the lack of a launch-specific non-FULL integration branch | reviewer result plus `analysis/dense_failure_stage2/full_benchmark_eval/smoke/nonfull_action_integration.json` | A label-blind forced READ_ONLY/WRITE_ONLY/IGNORE execution passed exact action trace, repeat-token/evaluator, and image-hash checks; launch concern resolved without changing the frozen policy | confirmed |
| Full paired external result is regression-dominated | `analysis/dense_failure_stage2/full_benchmark_eval/metrics/benchmark_summary.csv` | Pooled W→C/C→W/net `3/19/-16`; family nets ChartQA -2, TextVQA -10, MMMU-Pro -4, POPE 0 | confirmed |
| Stage-1 admission and Stage-2 intervention remain sparse and task-dependent | `analysis/dense_failure_stage2/full_benchmark_eval/metrics/stage1_admission.csv`; `analysis/dense_failure_stage2/full_benchmark_eval/metrics/stage2_action_behavior.csv` | 901 triggers, 211 with any non-FULL; POPE has zero triggers | confirmed |
| Current dense accuracies are close to but not identical to the rounded original-server report | `analysis/dense_failure_stage2/full_benchmark_eval/metrics/reference_dense_parity.csv` | Exact samples/method were preserved, but current-runtime results differ by -0.04 to +0.284 percentage points across variants; labels were not overwritten | confirmed observation; diagnosis unknown |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Broad `/mnt/hyemin` search | Timed out without useful output | supported | command result in active turn | Use only targeted allowed-root checks | Broad recursive storage scans |
| First Phase-69 smoke launch | Stopped before its first sample with `KeyError: router` | supported: the reused Stage-2 loader expects the historical `router` key while the frozen evaluation config names the identical block `stage2` | smoke traceback plus `experiments/run_stage2_v1_training_revised.py::_router` | Pass a shallow compatibility view with `router = stage2`; re-freeze because the evaluator is contract-bound | Do not change the scientific Stage-2 config or checkpoint |
| First full four-GPU launch | Rank 1 and rank 3 stopped after 1,024 durable rows each with `processor input has an empty user text span`; remaining ranks were stopped and no aggregate was emitted | supported: affected MMMU-Pro rows declare more images than prompt placeholders, so the exact reference builder can render a final vision block immediately before message end; the legacy locator assumes text follows the last vision block | rank logs; targeted 16-row reproduction identified `...Pharmacy_22` and `...Pharmacy_4` | Pool Stage-1 text features from the exact `instruction_token_mask` already emitted by the frozen reference builder; add a trailing-vision regression smoke, re-freeze, and restart | Do not skip these UIDs or change the reference prompt builder/population |
| Trailing-vision smoke after locator repair | Paired repeats passed, but custom unified all-FULL decoded `A` while reference native dense decoded `A.` on the targeted row; scorer/correctness agreed | supported: rare multi-image prompt geometry can produce a small cached-decode numerical difference even with all actions FULL | `full_benchmark_eval_failed_20260904_native_parity1/smoke/native_full_parity.jsonl` | Make reference-native `model.generate` the authoritative dense path and collect Stage-1 features with passive native layer hooks; exact no-trigger/all-FULL policy paths return that native result, while the custom executor runs only for an actual non-FULL intervention | Do not use custom unified generation as the dense baseline on external inputs |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Execute the frozen paired full evaluation | Explicitly authorized and directly tests method-level net correction | Whether validation net gain transfers to four external families | high | completed: regression-dominated |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: completed; the full-scale evidence shows the frozen candidate is regression-dominated.
- Relevant memory item used: Phase 66 retained Experiment A at P90 because it alone produced a positive Historical-800 rollout signal; Phase 62 warns that Stage-1 remains benchmark-OOD limited.
- Confirmed observation: all 19,960 paired rows are complete; pooled W→C/C→W/net is 3/19/-16, with TextVQA the dominant source of both rescue and regression and zero POPE triggers.
- Unverified interpretation: whether a preservation-focused Stage-2 calibration can retain the three rescues while removing most regressions.
- Diagnosis: the current candidate is regression-dominated; the causal source of individual regressions remains unknown.
- Viable alternatives considered: none—the user prospectively fixed the exact evaluation action and candidate.
- Chosen action: stop after the completed evaluation. Recommend exactly one unexecuted direction: conservative action-selection calibration focused on preservation.
- Strongest objection: the Stage-1 gate is only calibrated on GQA/ChartQA/TextVQA mixtures and may admit failures poorly on MMMU-Pro/POPE, so trigger diagnostics and per-family preservation must be reported without post-hoc threshold selection.
- How this differs from failed attempts: this is the first full external paired evaluation; no threshold, model, checkpoint, or training change is introduced.
- Automatic execution authorized: no further action.
- Authorization basis: the user explicitly requested execution of the named plan.
- Stop condition: full paired results, required summaries/figures, one method-level decision, and one unexecuted improvement recommendation are saved.

## Latest Research-Action Result
- Action taken: exact reference-population paired native-dense versus frozen Stage-1 P90→Stage-2 A evaluation on four direct GPUs.
- Result: Decision D, regression-dominated. ChartQA W→C/C→W/net 1/3/-2; TextVQA 2/12/-10; MMMU-Pro 0/4/-4; POPE 0/0/0; overall 3/19/-16. Dense/routed accuracy is 0.780561/0.779760.
- Evidence saved: `analysis/dense_failure_stage2/full_benchmark_eval/`, including all paired rows, metrics, bootstrap intervals, figures, summaries, and a 36-file verified artifact manifest.
- Failure or issue: implementation preflights exposed and repaired a historical config-key mismatch, trailing-vision Stage-1 position assumption, and rare custom all-FULL punctuation drift. Failed contracts/runs remain quarantined and none contributed to final results. Original-server rounded dense accuracies are close but not exact under the current runtime; diagnosis unknown.
- Lesson learned: the Historical-800 +3 signal does not transfer to this full four-family population for the exact frozen candidate. External Stage-1 admission is sparse/task-dependent, and current non-FULL actions regress more correct answers than they rescue.
- Next implication: do not promote this candidate. If separately authorized, test one conservative action-selection calibration focused on preservation; do not retune Stage 1, reopen search, or retrain automatically.
