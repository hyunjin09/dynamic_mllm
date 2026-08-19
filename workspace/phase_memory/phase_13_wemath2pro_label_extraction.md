# Phase 13: We-Math2.0-Pro Label Extraction Memory

## Current Objective

Generate unrestricted binary visual-route labels for all 4,552
We-Math2.0-Pro records under a frozen benchmark-specific prompt and evaluator,
without disturbing the active original 8K extraction.

## Active Constraints

- Use node06 with the available 7 GPUs, 84 CPUs, and 197 GiB Slurm memory; apply the
  mandatory node06 NCCL safeguards. Do not use node04.
- Preserve the verified 28-bit ON/OFF executor and unrestricted graph MCTS.
- Use native Qwen image processing and no custom visual-token cap.
- Score direct final answers with the official We-Math MathRuler equivalence
  contract; do not silently fall back to exact string matching.
- Run only a five-record parity/determinism smoke before all-sample extraction.
- No predictor or model training.

## Current State

- Done: official dataset schema and reward implementation inspected; 4,552-row
  Pro split confirmed; benchmark adapter unit tests pass.
- Done: all 4,552 rows and images materialized; eight invalid records preserved
  in the inventory; 4,544 valid MCTS records and five smoke records frozen.
- In progress: hard-cap-400 seven-worker resumable MCTS extraction, Slurm job
  `100407` on node06.
- Blocked: no.
- Most recent useful observation: the self-contained cap-400 audit retained
  640 complete 200/400-simulation records, excluded 529 completed
  600-simulation records, and left 3,904 records to run.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Pro split has 4,552 rows and fields `question`, `image`, `answer`, `difficulty` | local HF dataset cache and card | Defines complete population and adapter | confirmed |
| Official We-Math R1-V reward uses `mathruler.grader.grade_answer` | official `dynamic_scheduling/examples/reward_function/r1v.py` | Freezes equivalence scoring | confirmed |
| Current binary executor supports native processing without visual cap | `label_regeneration/runtime.py` and prior smoke | Preserves route semantics | confirmed |
| Eight Pro rows lack a question and/or answer | `outputs/label_regeneration/wemath2pro_v1/preflight/data_validity_failure_v1.json` | Full-population MCTS cannot assign valid rewards to every row | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Dispatch We-Math through existing generic metric fallback | Equivalent `\\frac{1}{2}` and `0.5` scored unequal | supported missing benchmark adapter | unit-test red result | Add explicit MathRuler dispatch and fail if dependency is absent | Do not use generic exact match |
| Freeze all 4,552 records without a validity audit | Builder stopped at the first empty answer | supported source-data incompleteness | `runs/label_regeneration/wemath2pro_manifest_v1.log` | Require an explicit exclusion/recovery policy | Do not coerce missing labels to empty-string targets |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Direct-answer MathRuler contract | Matches official accuracy logic and bounded generation | Correct route reward | low | selected |
| Five-record parity/determinism smoke | Minimal protection against executor/prompt drift | Authority for full run | medium | pending |
| Five-worker all-sample extraction | User-approved full population | New raw route cache | high | pending |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: complete the 4,544-record hard-cap-400 MCTS
  run without losing terminal predecessor work or accepting post-cap labels.
- Relevant memory item used: only atomic terminal records whose requested and
  completed budgets agree may be resumed.
- Confirmed observation: the cap-400 cache covers the full manifest as 640
  retained plus 3,904 remaining records, and all seven active workers are
  computing under the new contract.
- Unverified interpretation: no worker will encounter another long-lived stall;
  the five-second scorer bound addresses one recurrence class but the old
  stall's exact cause remains unknown.
- Diagnosis: no active failure.
- Viable alternatives considered: reuse 600-simulation records; rerun them
  under the new cap. The latter is selected because post-400 outcomes violate
  the amended label contract.
- Chosen action: let job `100407` complete and monitor atomic publication and
  worker health; do not begin downstream analysis or training.
- Strongest objection: rerunning 529 completed samples costs time, but retaining
  them would contaminate the capped protocol with search outcomes unavailable
  to other samples.
- How this differs from failed attempts: the replacement root is self-contained
  and excludes every above-cap record before worker startup.
- Automatic execution authorized: yes.
- Authorization basis: explicit user request to run all We-Math2.0-Pro samples.
- Stop condition: worker loss, terminal-record contract violation, incomplete
  manifest coverage, OOM, or repeated scoring/executor stall.

## Latest Research-Action Result

- Action taken: inspected the official dataset/reward and implemented the
  benchmark adapter plus manifest builder.
- Result: focused unit tests pass; CPU materialization stopped atomically at
  source index 940 after identifying an empty answer. A bounded full-column
  schema check found eight unique invalid records.
- Evidence saved: `tests/test_label_regeneration.py`,
  `reference/dvr_qwen/eval_metrics.py`, and
  `experiments/prepare_wemath2pro_label_manifest.py`.
- Failure or issue: the all-sample requirement conflicts with missing required
  fields. Diagnosis is supported source-data incompleteness, not executor or
  scorer failure.
- Lesson learned: open-ended math answers require equivalence grading and a
  bounded direct-answer format for scalable route evaluation.
- Next implication: the user approved marking the exact eight records invalid;
  run the smoke and start the full eight-worker sweep only on pass.

### First node06 launch repair

- Direct observation: job `99848` stopped during the first native smoke forward
  before publishing any smoke record or starting MCTS.
- Diagnosis: supported launch-environment omission. Frozen deterministic
  algorithms require `CUBLAS_WORKSPACE_CONFIG=:4096:8` for the observed CuBLAS
  operation.
- Evidence: `runs/label_regeneration/wemath2pro_mcts_v1.log` and
  `outputs/label_regeneration/wemath2pro_v1/launch_amendment_r2.json`.
- Next action: resubmit the identical smoke-to-full command with only the
  required CuBLAS deterministic workspace variable added.

### Passed smoke and launcher repair

- Job `99849` passed exact 5/5 ALL-ON/native generated-token parity and all ten
  repeated mixed-mask token/score checks.
- The post-smoke shell then failed before MCTS because bare `torchrun` was not
  on the plain-shell `PATH`. No raw route record was created.
- A focused diagnostic verified `.venv/bin/torchrun` and
  `torch.distributed.run`. Start the full sweep once with that absolute path;
  retain the already passed contract-bound smoke.
- Evidence: `outputs/label_regeneration/wemath2pro_v1/smoke_report_v1.json`
  and `outputs/label_regeneration/wemath2pro_v1/launch_amendment_r3.json`.

### Full-sweep launch boundary

- Frozen contract SHA-256:
  `96b2c632ebc6e020c607b3d9a0eddd2a29f7aff1912f5219327ae96a507c3a50`.
- Manifest: 4,552 inventory rows, exactly eight technical-invalid rows, and
  4,544 MCTS rows; all sidecar checksums pass.
- Smoke: exact 5/5 binary ALL-ON/native token parity and exact repeated
  token/score equality for both mixed masks on every smoke record.
- Active run: Slurm `99850`, node06, 8 A6000 GPUs, 96 CPUs, 240 GB RAM,
  eight `torchrun` workers, mandatory node06 NCCL settings, and deterministic
  CuBLAS workspace configuration.
- Startup validation: all eight GPUs are active; initial atomically published
  records are contract-bound, terminal, benchmark-tagged `wemath2pro`, and
  contain candidate executions. Zero error records were present.
- Historical implication at launch: job `99850` was initially allowed to
  continue. This was superseded by the scoring-stall repair below.

### Throughput diagnosis

- Compared 80 recent original-cache records with 80 WeMath records on the same
  `NVIDIA RTX A6000` device model.
- WeMath median visual tokens were 9,095 versus 580 (15.7x); mean generated
  tokens per route were 21.2 versus 5.9 (3.6x); mean requested simulations were
  467.5 versus 435.0 (1.07x).
- Median per-worker publication interval was 940.4 seconds versus 141.9
  seconds (6.6x). All eight WeMath GPUs were active at the diagnostic.
- Diagnosis: supported sample-workload bottleneck. Node-specific overhead is
  not ruled out, but the same GPU model plus much larger token workloads is
  sufficient to explain the dominant slowdown.
- Evidence:
  `outputs/label_regeneration/wemath2pro_v1/throughput_diagnostic_v1.json`.

### Scoring-stall repair and resumable restart (2026-08-12)

- Observation: ranks 3 and 6 of job `99850` stopped publishing for about 16.0
  and 18.7 hours. Their GPUs held model memory at 0% utilization while their
  Python workers consumed a CPU core; the other six ranks continued.
- Diagnosis: unknown. Unbounded MathRuler/SymPy equivalence scoring was a
  plausible suspected cause because the installed grader calls
  `sympy.simplify` without a timeout. However, both formerly stalled samples
  completed after restart with zero scoring timeouts, so that explanation was
  not confirmed.
- Repair: normal MathRuler results are unchanged; grading is bounded at five
  seconds, with timeout scored conservatively as incorrect and explicitly
  recorded. Twelve focused tests pass, including a synthetic nonterminating
  scorer and predecessor-contract resume acceptance.
- Preservation: the checksum-bound resume audit accepted 1,156 complete atomic
  records and found 3,388 remaining. Job `99850` was cancelled only after this
  audit passed; no complete record was deleted or overwritten.
- Amended contract: `outputs/label_regeneration/wemath2pro_v1/frozen_execution_contract_v2.json`,
  contract SHA-256
  `fc4a1df38925d20816770b861989b87d119bcdbf13b3bdff26a89b7abc90d485`.
- Active resume: job `100398`, node06, six GPUs, 72 CPUs, 180 GB. All six ranks
  completed a new record with zero errors and predecessor records were skipped.
  The two formerly stalled samples (`wemath2pro:591`, `wemath2pro:676`) each
  completed all 600 simulations with zero scoring-timeout flags.
- Evidence: `outputs/label_regeneration/wemath2pro_v1/resume_compatibility_audit_v2.json`
  and `runs/label_regeneration/wemath2pro_mcts_r5.log`.
- Next implication: monitor job `100398`; do not start training or post-label
  analysis as part of this repair action.

### Hard-cap decision (2026-08-13)

- Observation: in a 1,164-record completed-cache snapshot, 528 samples used the
  600-simulation extension. Only 25 (4.73%) found any valid route after
  simulation 400, yielding 28 new masks; 503/528 found none.
- User decision: remove the extension and cap every sample at 400 simulations.
- Preservation rule: retain only checksum-valid terminal 200/400 records;
  exclude and rerun all predecessor 600-simulation records. Do not reuse partial
  in-memory work from the cancelled process.
- Completed action: the new contract and self-contained capped cache passed
  audit, after which job `100398` was cancelled. Seven-worker job `100407` is
  running on node06 with 84 CPUs and 197 GiB. The memory adjustment from 200G
  is scheduler-only: 200G expands to 204,800 MiB and would exceed node06's
  250,000 MiB total alongside the existing 48,000 MiB allocation.
- Evidence: `outputs/label_regeneration/wemath2pro_v1/extension_yield_snapshot_v1.json`.

### Cap-400 relaunch (2026-08-13)

- Self-contained cache:
  `outputs/label_regeneration/wemath2pro_cap400_v2/`.
- Resume audit: all checks passed; 640 retained records consist of 207 at 200
  simulations and 433 at 400. All 529 records above the cap were excluded;
  retained plus remaining records cover all 4,544 manifest UIDs.
- Active execution contract SHA-256:
  `80c7ea4ca2ca9df091696290dc644a4092508337f89cf85ecc5b849a0f4092c7`.
- Launch history: job `100405` was cancelled while pending because the 200G
  request could not fit; job `100406` exited before model loading due to the
  launch-only flag typo `--model-revision`; it wrote no samples. Job `100407`
  uses the corrected `--revision` flag.
- Runtime check: all seven worker processes loaded the pinned model and all
  seven GPUs were actively computing. The first three newly published records
  each completed 400/400 simulations under the new contract, had no extension
  and no overlap with the retained cache. Zero errors and zero temporary files
  were present. No temporary record files were left by either superseded
  launch.
- Next implication: allow job `100407` to finish; do not start predictor
  training or post-label analysis in this action.
