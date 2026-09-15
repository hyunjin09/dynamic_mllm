# Phase 89: 3B greedy route corpus replay audit

## Cross-server Handoff — 2026-09-15
- Latest user request is to push all necessary handoff code/metadata to Git;
  large model/checkpoint/route payloads transfer separately via rsync.
- Exact stopped-file validation:734,481 routed-stream records;1,858complete,
  31partial and8,111without routed checkpoints;0saved errors. Separate Dense
  complete for all10,000; total unique available outcomes742,592. This supersedes
  live progress counters, which lagged the last durable checkpoint writes.
- Transfer entry:`handoff/phase89_server_transfer/README.md`; exact source,
  contracts, metadata restore, environment lock and22,035 external-file hashes.
  External payloads24,788,970,428bytes are not pushed. No restart authorized.
  Final routed interpretation/RS/readiness remains incomplete.

## User Stop — 2026-09-15 11:41:56 KST
- User explicitly requested cancellation of the current run. Replay2994 and
  dependent report2957 were cancelled and Slurm confirms cancellation.
- All saved results, contracts and prior evidence are preserved. Do not restart
  or resubmit without a new user instruction. V2 remains incomplete.
- Evidence: V2concurrency_v1/user_cancellation.json. Running/ETA statements
  below are historical and superseded by this stop.

## Current Objective
Finish `plans/3b_greedy_route_corpus_audit_replay_filtering_plan_v2.md`.
Latest user instruction: try three processes per GPU, monitor a few samples,
and provide a measured ETA. This bounded check is complete;2994 remains running
and the user will notify us when jobs finish. Full V2 reporting remains pending.
No geometry or training.

## Active Constraints
- Canonical3B package and V1 failed-attempt evidence are immutable. V2 writes:
  `analysis/3b_greedy_route_corpus_replay_audit_v2/`.
- Exact model revision66285546d2b821cf421d4f5eb2576359d3770cd3;36 binary layers.
- All old labels preserved. Current dense and route labels alone define cohorts.
- GPU computation only via Slurm;8GPUs for Dense/routed replay. Local project
  .venv unchanged. No downloads, source rewrites, geometry or finetuning.
- Frozen diagnostic, repair and execution contracts remain unchanged; any
  future repair requires explicit preserved provenance, not bypassed hashes.
- Phase85 stopped. Phase88 source completion/final interpretation still separate.

## Current State
- Source verified:84 files,10,000 prepared shards,10,000 images;10,000 samples,
 9,273 image groups,3,907,717 unique routes; one dense anchor per sample.
- Repaired Gate A2942 PASSED32/32; failing-anchor trace has no numerical
  divergence over72 recorded layer boundaries. Eight-worker preflights passed.
- Dense2954 COMPLETED00:48:25 KST,16m19s,10,000/10,000 saved.
- CPU dense certification2955 COMPLETED01:27:43 KST,1m06s. Current5702C/4298W;
  old→current5607C→C/125C→W/95W→C/4173W→W (220 changes,2.2%).
- Routed2994 RUNNING since2026-09-15 11:23:14 KST,24workers/8H100/96CPU/
  384G/7days. Previous2956 was deliberately cancelled for this transition after
  report2957 was retargeted; preserved720,540 records and1,834 complete samples.
- Concurrent Gate A PASSED32/32 plus4 sparse probes twice across24workers,
  exactly matching previous sparse outputs. Saved-sample pilot PASSED24/24,
  all9,323 routed outputs exact,270.1s wall,34.52routes/s. Historical same-sample
  inference-time comparison suggests1.90x throughput; pilot is ChartQA-only.
  Production resumed: at11:36:42 KST,8,414 new routes and8new complete samples;
  independent identity/hash/dense-reuse checks passed for all8. Aggregate
  checkpoint counters728,986/3,907,717. No worker errors, checkpoint ages≤29s.
  Observed GPU utilization96–99%, memory25–26GB/GPU,3processes on everyGPU.
  Early DocVQA throughput17.53routes/s over433.8s; pilot34.52routes/s. These
  imply25.6–50.4hours remaining at constant rates. Planning range25–55hours
  (Sep16 afternoon–Sep17 evening KST); different sample mixes limit precision.
  CPU allocation also rose32→96, so1.90x is configuration throughput evidence,
  not isolated process-count causality. No matched2/GPU comparison was run.
  Review: V2concurrency_v1/initial_throughput_review.{json,md}.
- Scheduling overlay9a4f24328f8b86d22d6e97d0257a3fe42c3948e4ff23df7e12f90089c1495292
  preserves original inference contract,24disjoint LPT shards and exact original
  source-rank accounting. New records include overlay/worker/GPU provenance.
  Four unit tests and end-to-end complete/partial/missing resume fixture pass.
  Evidence: V2concurrency_v1/{contract.json,submission.json,validation.json,
  job_2994/}; live monitor: monitoring/concurrency_status.py.
- Final CPU report2957 PENDING afterok:2994. Scientific RS/readiness review
  remains pending after its results. User will notify us to resume review.
- Latest machine-readable status: V2 `monitoring/latest.json`; history is
  `monitoring/history.jsonl`. Run `monitoring/status.py` to query Slurm and
  checkpoint progress. It becomes read-only when final reporting starts.
- Execution contract:0e096f78a05a5f139fc9976931c22328493beb9cc966f2b60334ed83f6ccac47.
  Dense manifest:0fe3091b7d2853b7983eda17ee4bcc51e258204c79d24a2a5b4686aa03b5e094.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Source current HF/binary gate32/32 | package `02_GREEDY_SEARCH/vqa_10k/replay_gate_v1/summary.json` | Historical-source matches are not its hard acceptance rule | confirmed |
| Source3B36 layers and Torch2.9.1 | package `final_phase1_phase2/summary.json` | Record runtime differences and actual depth | confirmed |
| Native decode positions may differ despite prefill agreement | `workspace/decision_log.md`, Phase88 lesson | Gate full generated token sequences, not just first logits | promoted lesson |
| Independent census10,000 samples /9,273 image hashes /3,907,717 unique routes | `analysis/3b_greedy_route_corpus_replay_audit/census/global_census.json` | Zero duplicate routes; unique dense anchor for every sample | confirmed |

## Failed Attempts and Lessons
Job2939 failed the fixed gate: `textvqa:textvqa_19417` native HF output22
(score0) versus binary all-on23 (score1). Source gate recorded22 for both.
Other31 anchors matched; all sparse repeat/cross-worker checks passed. No
anchor runtime exceptions. Diagnosis: unknown; runtime version differences
are observations, not established causes. Evidence: `replay_gate/gate_result.json`
and `summaries/job_2939_failure.json` under the audit output root. The gate
correctly prevented full replay. No identical retry or acceptance-rule change.

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Gated full label-only replay | Directly authorized | Current route/dense labels and geometry eligibility | high | selected |
| Capture question-token states during replay | Saves a later pass | Future geometry cache | additional storage/index validation | deferred |
| Stop after gate/pilot | Cheap validity evidence | No full-corpus eligibility | low | insufficient for named plan |

## Next-Step Decision
- USER STOP: no further execution or monitoring. Wait for new instructions;
  do not act on the earlier continuation decisions below.
- Latest outcome: the requested three-process trial and brief monitoring are
  complete. Leave2994 running and2957 dependent; pause assistant polling per
  the user's earlier instruction. On user notification query actual Slurm state,
  inspect failures, and then verify final reports and scientific interpretation.
- Historical decision packet follows; do not re-run the completed pilot.
- Current authorized action: test24 workers on8GPUs and measure throughput.
  STANDARD mode; no method/data/decoding change. Current bottleneck is replay
  wall time; observed one process/GPU,10–22% GPU use and8–9GB/80GB memory.
  Three processes may overlap CPU dispatch gaps; speedup is not established.
  Compared keeping1/GPU versus the user-requested3/GPU. Bounded3/GPU validity
  and throughput check dominates; objection is CPU contention and unequal
  sample cost. Preserve all original contracts and add a hashed scheduler
  overlay; replay saved samples exactly before resuming canonical work.
  Replacement must own disjoint samples, import completed records unchanged,
  and retarget the dependent report before old-job cancellation. No duplicate
  canonical writers. Monitor initial completed samples, then report ETA and
  leave the successful configuration running. No further concurrency sweep.
- Earlier instructions below are historical where superseded by this action.
- Latest steering supersedes continuous active monitoring: pause the assistant
  polling loop now, leave2956/2957 untouched, and wait for user notification.
- On return query actual Slurm state and checkpoint/failure evidence first;
  preserve and diagnose any failure before a targeted repair/resume. Never
  duplicate a running job or infer success from elapsed time.
- After2957 finishes verify complete coverage, all artifacts and source integrity.
- Review exact dense/routed/semantic transitions, current cohorts and V3.1 audit;
  write qualitative RS-A/B/C and READY/NOT READY with explicit practical limits,
  no invented numerical cutoffs. The batch reporter deliberately leaves that
  interpretation pending when drift is nonzero.
- Plan completion remains pending; only active monitoring is paused by user.
- ETA at2026-09-15 11:11 KST: recent1/2/4hour aggregate throughput16.16/16.37/
  18.28 records/s implies49–55hours, but slowest-worker projections imply55–76
  hours until routed completion (Sep17 evening–Sep18 afternoon KST). This is
  an estimate, not a deadline; later sample costs and rank imbalance may change.
  Final CPU report queue wait/runtime and scientific review are additional.
  Evidence: V2monitoring/history.jsonl and current per-rank checkpoint counts.

## Latest Research-Action Result
- CPU census completed: saved1,880,345C /2,027,372W; no unknown labels or
  duplicate routes. Route count mean390.7717, median400, range218–409.
- All10,000 input-image hashes match. All20 listed canonical checksum entries
  and32 sampled raw-to-final lineage files pass. All32 gate-anchor CPU input
  geometries match. Five implementation tests and packaged runtime imports pass.
- Frozen contract: `2e7553f84734b61f097c831d6a2e8e775090c85c576c23ad19334053caabb004`.
- Prospective last-question BF16 storage576,216,317,952 bytes (~536.64GiB).
  Raw visual BF16/FP16 estimate457,995,695,505,408 bytes; no capture authorized
  for raw visuals. Last-question capture deferred pending validated token indexing.
- Submitted jobs2939/2940; details and scheduler snapshots are in
  `analysis/3b_greedy_route_corpus_replay_audit/submission.json` and `logs/`.
- Follow-up queue inquiry: gate FAILED31/32, full replay not started, reporter
  cancelled. Saved accounting and exact mismatch; updated submission status.
  Gate failure remains unresolved; no resubmission during this status check.
  The CPU reporter leaves nonzero-drift RS classification for qualitative review
  and does not launch the next phase.

V2 evidence: `analysis/3b_greedy_route_corpus_replay_audit_v2/submission.json`,
`source_inventory/reuse_verification.json`, and `summaries/`. Diagnostic job2941
will write `parity/job_2941/diagnostic_result.json` and layer/logit comparisons.

Latest follow-up: diagnostic2941 succeeded; source of the first numerical
divergence is supported by an isolated same-Q/K/V SDPA convention comparison.
Repair and gate evidence: V2 contracts/parity_repair_v1.json,
parity/diagnosis_review.md, repair_submission.json. Full replay remains gated.

User explicitly requested continuous monitoring even while pending. Remain active
until V2 finishes or an actual blocker requires input. Dense-first eight-worker
runner and dependent CPU certification, routed replay and final reporting are
prepared; four CPU pipeline tests pass. Submit only after repaired Gate A passes.

Gate2942 COMPLETED at2026-09-15 00:31:34 KST after48s. All32 complete
sequences/answers/scores/labels match. Failing-anchor trace has no numerical
divergence across72 prefill/decode layer boundaries; four sparse probes repeat.
The targeted repair is supported. Proceed with the explicitly authorized
eight-worker preflight, all-sample Dense replay, CPU dense certification,
routed replay and reporting. No gate relaxation or canonical change.

Pipeline submitted2026-09-15 around00:32 KST:2954 Dense/FULL on8GPUs,
2955 CPU dense certification,2956 routes on8GPUs,2957 CPU final reports.
Job2954 started; later stages depend afterok. Four pipeline CPU tests pass.
Execution contract 0e096f78a05a5f139fc9976931c22328493beb9cc966f2b60334ed83f6ccac47.
Continue monitoring, including final qualitative RS/readiness interpretation.

Dense2954 COMPLETED00:48:25 KST in16m19s, all10,000 saved records and8/8
worker completion markers. Eight-worker32-anchor/sparse preflight passed.
CPU certification2955 is pending Resources;2956 routes and2957 reports remain
dependent. Do not infer dense certification or routed completion from elapsed time.

CPU2955 COMPLETED01:27:43 KST in1m06s. Dense manifest hash
0fe3091b7d2853b7983eda17ee4bcc51e258204c79d24a2a5b4686aa03b5e094 verifies. Dense current5702C/4298W; old→current
5607C→C/125C→W/95W→C/4173W→W. Routed2956 pending8GPU resources;
report2957 dependent. Continue monitoring. No final RS/cohort interpretation yet.

Routed2956 started2026-09-15 01:29:56 KST. Its independent eight-worker
preflight passed32/32 anchors and cross-worker sparse repeatability. Durable
route records advance onall8workers; one completed sample checked for exact
record/source hashes, unique coverage and byte-equivalent dense reuse. No
runtime failure markers. Continue monitoring; final2957 is dependent.
