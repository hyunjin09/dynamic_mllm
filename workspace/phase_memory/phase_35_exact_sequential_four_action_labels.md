# Phase 35: Exact Sequential Four-Action Labels Memory

## Current Objective

Execute `plans/4way_labeling_3.md`: convert every replay-valid positive binary
route into evaluator-correct four-action supervision using exact fixed-order
sequential branching for W2C and mechanical FULL/IGNORE preservation for C2C.

## Active Constraints

- Reuse the frozen 12,278-sample / 545,531-route source inventory and validated
  unified executor; do not rerun MCTS.
- Process each original W2C OFF layer once, early-to-late, in the current branch
  context; retain all correct branches and never beam-rank or prune them.
- Preserve C2C routes mechanically and keep their interpretation separate.
- Use a new schema/output contract and preserve historical beam outputs only as
  provenance.
- Run exactly 8 smoke samples with 8 workers/8 GPUs before a 16-worker/8-GPU
  full conversion.
- Preserve atomic resume, exact-route caching, dynamic full scheduling, source
  provenance, and the hash-gated oversized-image repair.

## Current State

- Done: the replacement plan, Phase 34 evidence, prior four-action audit, and
  independent review were read; jobs 1609/1610 were canceled.
- Done: the exact core, isolated execution contract, full lifecycle tooling,
  and 432/432 active tests pass after the dataset-priority launch helper and
  isolated topology provenance checks were
  added outside the scientific execution contract.
- Done: job 1611 completed `0:0`; all semantic, parity, checksum, branch,
  worker-layout, and exact-resume smoke gates passed.
- Done: resumed job 1628 was cleanly canceled at the user's request after the
  committed count remained 262 with zero failures; its 16 uncommitted WeMath
  samples will be reclaimed later.
- Done: jobs 1629/1630 were canceled for a user-requested three-replica trial
  after job 1629 preserved 33 new VQA records with zero failures. Isolated job
  1631 loaded 24 workers successfully but achieved only 0.990x matched
  estimated-cost throughput versus job 1629, so it was rejected and canceled.
- Done: user-requested one-replica job 1634 was stopped before its 551-second
  gate. Its partial 440-second result was 2.011x baseline estimated-cost
  throughput with zero failures, but remains incomplete evidence; four records
  are isolated and do not enter accepted labels.
- Done: repeated one-replica job 1638 completed the matched 551-second gate
  with five samples / 4,282 estimated-cost units versus five / 4,192 for the
  16-worker baseline (1.021x). It failed the prospective 1.10x keep threshold
  with zero failures and was rejected; its seven eventual records remain
  isolated.
- Done: job 1641 completed `0:0` in 17:32:42 and produced exact complete
  coverage for GQA (3,386), TextVQA (1,746), and ChartQA (1,785), with zero
  failures.
- Paused: at the user's request, job 1642 was cleanly canceled after 10:39:05.
  The accepted output contains 7,998 atomic checksum-backed records: all 6,917
  VQA records plus 742 WeMath2.0 Standard and 339 WeMath2.0 Pro records. There
  are zero failure, temporary, or zero-byte record files. The 16 interrupted
  samples will be reclaimed from their sample boundaries on resume.
- Blocked: no infrastructure or semantic blocker.
- Most recent useful observation: the canceled beam pilot had 322 canonical
  mismatches and 167 Jaccard failures among 1,417 completed route comparisons,
  while executor/parity/correctness checks remained clean.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Frozen authority has 12,278 samples and 545,531 positive routes | `datasets/mcts_labels_4action/source_inventory_v1/` | Fixes complete conversion scope without MCTS | confirmed |
| Unified arbitrary-route executor and source replay already passed real-data validation | `analysis/4action_label_conversion/implementation_audit.md`; `pilot_audit_v1.json` | Executor/runtime/queue may be reused | confirmed |
| Existing converter uses purification and beam-pruned single-route selection | `tools/research_analysis/four_action/label_conversion.py` | Its conversion core does not implement the new plan | confirmed |
| Beam-8/16 label choices were materially unstable | `analysis/three_action_answer_aligned_label_conversion/early_stop_audit.md` | Rules out continuing the superseded beam conversion | confirmed |
| Independent review recommends reusing infrastructure but replacing the conversion-policy core | 2026-08-25 `labeling3_review` result | Prevents semantic contamination from the old converter | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Beam-8 answer-aligned three-suppression refinement | 322/1,417 canonical mismatches and 167/1,417 Jaccard failures against beam 16 | supported beam-policy instability | `early_stop_audit.md` | Use the approved exact sequential branch policy | Do not relax the prospective gate or launch beam-8 full conversion |
| Reuse the old four-action converter unchanged | Static audit shows two-order purification, margin/cost selection, and beam pruning | supported plan incompatibility | `label_conversion.py`; independent review | Reuse only executor/runtime/scheduling foundations | Do not relabel old beam outputs as sequential labels |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| New exact branching core over existing runtime/queue | Matches the approved plan with minimal executor risk | Produces required W2C/C2C labels | high | selected |
| Retrofit the three-action beam converter | Reuses recent orchestration | Would retain score/beam concepts excluded by the plan | medium | rejected |
| Reuse old beam converter unchanged | Prior pilot passed its own contract | Does not implement exact all-branch refinement | low | rejected |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: complete exact sequential conversion over
  the frozen five-dataset authority; the bottleneck is long-running GPU
  inference rather than executor correctness.
- Relevant memory item used: route effects are context-dependent and locally
  valid actions must be jointly executed.
- Confirmed observation: executor semantics remain valid, but both historical
  conversion cores use policies excluded by the new plan.
- Unverified interpretation: branch counts will remain tractable without a cap.
- Diagnosis: supported beam-policy/contract mismatch.
- Evidence path if diagnosis is not unknown:
  `analysis/three_action_answer_aligned_label_conversion/early_stop_audit.md`.
- Viable alternatives considered: new core over reused infrastructure; retrofit
  current three-action converter; reuse historical beam converter.
- Chosen action: preserve all 7,998 accepted-contract atomic completions while
  paused. On explicit user authorization, submit a fresh eight-H100,
  16-worker resume using the unchanged full wrapper and `gpu-large` QOS; it
  must skip completed records, reclaim the 16 interrupted WeMath samples, and
  finalize/analyze/report only after full coverage.
- Strongest objection: all-valid branching may explode on some W2C routes; the
  plan explicitly requires measuring that before proposing a cap.
- How this differs from failed attempts: no purification-order competition,
  score threshold, canonical beam choice, or branch pruning enters labeling.
- Automatic execution authorized: no while paused
- Authorization basis: the user's 2026-08-27 request to pause until a later
  explicit resume instruction.
- Stop condition: semantic/executor inconsistency, non-correct stored route, or
  demonstrated pathological branch explosion requiring an unapproved cap.

## Latest Research-Action Result

- Action taken: completed the VQA-first stage, ran WeMath-last for 10:39:05,
  and cleanly paused it at the user's request.
- Result: job 1641 completed `0:0` with all 6,917 VQA records. Job 1642 was
  canceled cleanly with 1,081/5,361 WeMath records complete, bringing accepted
  coverage to 7,998/12,278. All records have checksum sidecars; there are zero
  failure, temporary, or zero-byte record files.
- Evidence saved: `analysis/4action_sequential_label_conversion/dataset_priority_1629_0.json`,
  contract SHA-256 `d8f524b928fb30ea0bb37c6a9389893adb338d4f91992d85255fdfb9bea283cb`,
  `analysis/4action_sequential_label_conversion/worker_topology_trial_20260826.md`,
  and Slurm accounting for jobs 1628--1642.
- Failure or issue: no executor failure; the superseded beam policy failed its
  own prospective stability criterion.
- Lesson learned: infrastructure validation transfers across label-policy
  changes, but label-validity evidence does not.
- Next implication: wait for explicit resume authorization. A new full-wrapper
  launch will skip 7,998 completions, process the remaining 4,280 WeMath
  samples, and then run finalization, analysis, plotting, and reporting.
