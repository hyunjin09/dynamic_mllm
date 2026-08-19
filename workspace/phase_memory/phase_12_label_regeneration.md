# Phase 12: Label Regeneration Memory

## Current Objective

Regenerate complete 28-bit visual ON/OFF route labels for the fixed 8K
GQA/TextVQA/ChartQA pool under one frozen, reproducible native-Qwen execution
contract. Preserve enough raw route information for later predictor
comparisons without training any predictor in this phase.

## Active Constraints

- Active source: `plans/dynamic_mllm_label_regeneration_plan.md`, SHA-256
  `634f2736d287c647cda7b21755b2ace753db29316ecc9c51523218b498380918`.
- Use 4,000 GQA, 2,000 TextVQA, and 2,000 ChartQA records; no DocVQA.
- Generate the full raw 8K cache first; freeze the approved image-group-
  disjoint 7K/1K predictor split afterward and before training.
- Use unrestricted layer-wise 28-bit routes and actual route-conditioned greedy
  evaluation; no segment-constrained MCTS.
- Use native Qwen image processing and no custom `max_image_tokens` cap.
- Treat historical buckets and old cache labels as metadata/proposals only.
- Retain every evaluated positive and negative route, including zero-positive
  samples and raw MCTS metadata.
- Cap the primary derived training view at 50 deterministic diverse valid masks
  per positive sample for both matched objectives; never truncate the raw
  cache.
- P9 has passed. Predictor training remains outside this phase and requires a
  separate P10 authorization.
- Use only the 15-record pre-extraction smoke: five records per dataset,
  15/15 ALL-ON/native generated-token parity, and repeated mixed-route
  token/score equality.
- Execute amended P0–P10 in order; stop on any frozen hard-stop condition.

## Current State

- Done: P0-P9, including the complete 8K cache, summaries, route-diversity
  analysis, exact image-group-disjoint 7K/1K predictor split, and all matched
  derived supervision views, provenance, final report, and checksum freeze.
- In progress: none.
- Blocked: no.
- Most recent useful observation: P9 independently verifies the 53-entry
  checksum ledger and freezes the 8,000-record raw-cache integrity chain plus
  all downstream supervision/provenance artifacts.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Old cached labels are executor-contract dependent | `reports/binary_mcts_label_mismatch_analysis.md` | Historical validity cannot be copied into the new cache | confirmed |
| Repaired BP-1 still has two cached-positive to target-invalid fixtures | `outputs/binary_polar/preflight/executor_preflight_v3.json`, `reports/binary_polar_bp1_input_contract_repair.md` | Fresh target-executor scoring is required | confirmed |
| All-ON/native execution itself can achieve exact logit parity | same BP-1 artifacts | Provides a viable invariant for the new Gate A | confirmed |
| Amended plan freezes a minimal smoke followed by immediate 8K extraction | `plans/dynamic_mllm_label_regeneration_plan.md` | Defines the approved phase scope and order | confirmed |
| P0/P1 artifacts reconcile and checksum | `outputs/label_regeneration/v1/` | Establishes the immutable contract and exact smoke/full populations | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Reuse old MCTS cache as exact target labels | Exact cached outputs failed on 4/16 repaired fixtures; two positives became invalid | supported executor-domain label drift | `reports/binary_polar_bp1_input_contract_repair.md` | Regenerate authoritative outcomes under one frozen runtime | Do not delete only known mismatches or copy old validity |
| Reconstruct old runtime contract incrementally | Geometry repaired, but exact output drift remained and source provenance was incomplete | unknown exact residual cause | `reports/binary_mcts_label_mismatch_analysis.md` | Freeze a new reproducible contract rather than chase missing provenance | Do not repeat unchanged provenance reconstruction |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| P0 contract freeze | Mandatory first plan step and prevents another nonportable cache | Exact model/processor/executor/evaluator/runtime authority | low | complete |
| P1 15-record smoke freeze | Makes the minimal parity test deterministic and outcome-blind | Exact smoke population | low | complete |
| P2 minimal smoke | Catches executor-contract mistakes without a large pilot | Authority to start full extraction | medium | complete |
| P3 full extraction | Generates the authoritative unrestricted route cache | 8,000 fresh sample caches | high | complete |
| P9 integrity freeze | Makes the cache and derived views reproducible inputs to P10 | Final report, provenance, and checksum chain | low | complete |

## Next-Step Decision

- Deliberation mode: fast
- Active objective and bottleneck: the label-regeneration phase is complete;
  no label-generation gate remains.
- Relevant memory item used: raw routes must remain untruncated, while both
  matched predictor objectives must consume the identical deterministic
  diverse maximum of 50 valid routes per positive sample.
- Confirmed observation: the P7 manifest freezes all 8,000 identities into
  exactly 7,000 train and 1,000 validation records with zero image leakage.
- Diagnosis: no scientific failure. The first P9 finalizer attempt had a
  supported relative/absolute path bug and was repaired before hashing.
- Chosen action: stop this phase. P10 may begin only on explicit approval.
- Automatic execution authorized: no further action at this boundary.
- Stop condition: reached; P9 passed and predictor training was not executed.

### Minimal-smoke/full-extraction amendment (2026-08-10)

- Deliberation mode: standard
- Active objective and bottleneck: revise the regeneration order without
  executing it; remove validation work that delays extraction but does not
  protect the executor contract.
- Confirmed observation / unverified interpretation: MCTS is independent per
  sample, so post-extraction predictor splitting does not leak across route
  searches; full native-processing feasibility remains untested.
- Diagnosis: supported for the need to retain exact parity/determinism after
  prior cache drift; no failure diagnosis is being made for the amended plan.
- Viable alternatives considered: original split-first/large-pilot order versus
  the user-approved minimal smoke then full extraction. The amendment is fixed.
- Chosen action and strongest objection: freeze the 15-record smoke and move
  splits after extraction; the full run may discover a native-processing
  resource issue that a larger pilot would have measured earlier.
- How this differs from failed attempts: it preserves the exact executor gates
  that detect contract mismatch while removing provenance and pilot analyses
  unrelated to those gates.
- Authorization and stop condition: update plans/memory only; do not start P0,
  select samples, run inference, or launch MCTS.

## Latest Research-Action Result

- Action taken: amended the source and compact plans to a minimal 15-record
  smoke, immediate 8K extraction after passage, and post-extraction predictor
  splitting.
- Result: P0 remains the active but unopened stage; no execution started.
- Evidence saved: `workspace/research_plan.md`, `workspace/workflow_state.md`,
  and this phase memory.
- Failure or issue: none in this planning-only action. P0 will use source hashes
  if no Git commit is available.
- Lesson learned: regenerated labels must bind masks to the complete frozen
  executor, processor, generation, evaluator, and hardware/software contract.
- Next implication: on explicit approval, execute P0, then P1/P2; start P3
  immediately only if the frozen smoke passes.

### P0/P1 execution boundary (2026-08-10)

- Action taken: inspected the preserved MCTS v2 semantics, validated the new
  unrestricted graph-MCTS/runtime implementation with four local tests, and
  froze the execution contract plus source and smoke manifests.
- Result: P0 and P1 pass. The source manifest contains exactly 8,000 records;
  the smoke manifest contains 5/5/5 GQA/TextVQA/ChartQA records; all four
  frozen artifact checksums pass.
- Evidence:
  `outputs/label_regeneration/v1/frozen_execution_contract.json`,
  `outputs/label_regeneration/v1/source_manifest_v1.jsonl`, and
  `outputs/label_regeneration/v1/smoke_manifest_v1.jsonl`.
- Failure or issue: the project root has no Git metadata. Diagnosis is
  supported by `git rev-parse` failure; source-file hashes are frozen as the
  approved fallback. A first checksum verification command used the wrong
  working directory; rerunning from the artifact directory verified every
  sidecar successfully and required no artifact change.
- Next-Step Decision: run only the frozen P2 smoke. If 15/15 parity and all
  three mixed-route repeats pass, launch the four-worker P3 extraction without
  adding another gate.

### P2 execution boundary (2026-08-10)

- Action taken: ran the frozen 15-record smoke as Slurm job `99740` on node07.
- Confirmed observation: exact binary ALL-ON/native generated-token parity was
  15/15; all three frozen mixed masks reproduced identical generated IDs and
  benchmark scores; the report checksum and contract binding pass.
- Evidence: `outputs/label_regeneration/v1/smoke_report_v1.json` and
  `outputs/label_regeneration/v1/smoke_report.md`.
- Diagnosis: no failure. The Transformers temperature warning is non-operative
  because both native and binary paths were deterministic and matched exactly.
- Next implication: launch P3 immediately under the unchanged contract; no
  larger pilot or additional executor gate is warranted.

### P3 launch boundary (2026-08-10)

- Action taken: launched Slurm job `99741` on node07 with four A6000 GPUs, 32
  CPUs, 240 GB RAM, and four `torchrun` workers over the frozen 8K manifest.
- Confirmed observation: all four ranks loaded the pinned snapshot, each owns
  exactly 2,000 rows, initial records contain ALL-ON/ALL-OFF plus 200 MCTS
  simulations, positive and negative routes, current ALL-ON results, token
  geometry, and the frozen contract hash. Early status: 8 completed records,
  zero errors, roughly 72–76 seconds elapsed per rank.
- Evidence: `runs/label_regeneration/p3_mcts_8k.log` and
  `outputs/label_regeneration/v1/raw_route_cache/shard_*/summary.json`.
- Diagnosis: no failure. Full completion remains unverified and is expected to
  be long-running because wrong-root records use 400–600 simulations.
- Next implication: monitor and resume only under the identical contract if
  the allocation fails; completion requires 8,000 verified sample records.

### P3 four-to-eight-worker resume boundary (2026-08-10)

- Action taken: stopped job `99741` after 2,291 atomically published records
  and resumed as job `99758` on node02 with eight NVIDIA RTX A6000 GPUs,
  64 CPUs, 400 GB RAM, and eight `torchrun` workers.
- Confirmed observation: the stopped cache had zero zero-byte records, stale
  temporary files, or error records. A focused test proved that resume can find
  a complete record from a prior shard count. Every new rank validated and
  skipped its inherited records before completing a new record; the first
  post-resume audit found 2,300 unique records, no duplicate filenames, and
  zero errors.
- Operational amendment: `experiments/run_label_regeneration.py` now indexes
  complete records across every `shard_*_of_*` layout. Its hash changed from
  the frozen `d21239...` to `da1577...`; this is disclosed in
  `outputs/label_regeneration/v1/p3_resume_amendment_v1.json`. All other 13
  frozen source hashes match, and route evaluation, scoring, MCTS, per-sample
  seeds, output schema, model, and GPU type are unchanged.
- Diagnosis: no scientific or runtime failure. The only discarded work was the
  unfinished in-memory search for up to four samples at cancellation; no
  partial final record existed.
- Next implication: monitor job `99758` to 8,000 unique terminal records across
  both shard layouts, then run the frozen P3 completion verification. Do not
  enter P4 without a separate authorized action.

### Portable MCTS handoff boundary (2026-08-11)

- Action taken: packaged the verified binary executor, unrestricted graph
  MCTS, current evaluator, portable runner, manifest/smoke/contract/cache-audit
  tooling, pinned requirements, Slurm template, tests, and agent-oriented
  reproduction runbook in
  `handoff/binary_visual_mcts_reproduction_v1/`.
- Confirmed observation: six CPU-only bundle tests pass; the copied
  executor/MCTS/runtime/evaluator files match the active project sources; all
  transfer files pass `BUNDLE_SHA256SUMS` verification.
- Scope: the bundle contains no model weights, dataset, route output, or
  predictor training. A new benchmark must freeze and validate its own prompt,
  manifest, official scoring adapter, and correctness threshold before the
  parity smoke and full search.
- Contract impact: none. Active jobs `99758` and `99850`, their outputs, and
  their frozen execution contracts were not modified or restarted.

### P7 split-freeze boundary (2026-08-12)

- Action taken: froze the exact outcome-blind image-group-disjoint predictor
  split with seed `20260809` after P4-P6 completion.
- Confirmed observation: all 8,000 source UIDs appear exactly once; train has
  7,000 records and validation has 1,000; there are zero cross-split image
  groups. Validation historical strata are GQA 250/250, TextVQA 125/125, and
  ChartQA 125/125.
- Evidence:
  `outputs/label_regeneration/v1/post_generation/predictor_split_manifest_v1.jsonl`,
  `outputs/label_regeneration/v1/post_generation/predictor_split_audit_v1.json`,
  and `reports/label_regeneration_p7_predictor_split.md`.
- Integrity: all sidecar checksums pass; manifest SHA-256 is
  `4d12bf427f08b0cc55d21c82bf7eaac7d19d283dc514ffd4f59894d6faf1bd1a`.
- Next implication: P8 may derive matched supervision views; it must not alter
  the raw cache or the frozen P7 assignments.

### P8 derived-supervision boundary (2026-08-12)

- Action taken: built single-best, diverse max-50 valid-set, matched binary
  predictor, complete positive/negative ranking, and canonical POLAR-segment
  manifests from the unchanged checksum-verified cache.
- Confirmed observation: 6,917 samples have at least one valid route and 1,083
  have none. P8 selected 237,802 of 528,047 raw valid routes and retained all
  2,642,998 evaluated routes in the ranking view; 3,616 samples were capped.
- Integrity: zero selected masks are absent from raw valid sets, all required
  anchors are retained, both objectives have identical route lists and equal
  weights, zero split-group leakage exists, and all 237,802 POLAR targets
  reconstruct their original complete mask.
- Evidence:
  `outputs/label_regeneration/v1/post_generation/derived_supervision_audit_v1.json`,
  `derived_supervision_verification_v1.json`, and
  `reports/label_regeneration_p8_derived_supervision.md`.
- Failure response: two unpublished slow attempts were cancelled. Concrete
  profiling-by-throughput supported repeated tuple Hamming work and serialized
  JSON decoding as the bottlenecks; exact XOR Hamming plus bounded process
  decoding completed the unchanged action. No raw or final artifact was
  removed.
- Boundary note: P9 was the next gate after this P8 result and is now completed
  in the entry below.

### P9 completion boundary (2026-08-12)

- Action taken: froze the final label report, scheduled-command provenance,
  primary/code artifact inventory, raw-cache checksum linkage, and P9 checksum
  ledger.
- Result: PASS. The audit binds 8,000/8,000 raw records, 50 inventory files,
  and a 53-entry checksum ledger; independent `sha256sum -c` verification
  passed 53/53.
- Evidence: `reports/label_generation_report.md`,
  `outputs/label_regeneration/v1/post_generation/p9_final_audit_v1.json`,
  `p9_artifact_inventory_v1.json`, `p9_run_provenance_v1.json`, and
  `P9_SHA256SUMS`.
- Failure response: the first finalizer attempt failed before hashing because a
  relative output path was compared with the absolute project root. The
  diagnosis is supported by the traceback; path normalization fixed it. A
  subsequent review added the finalizer itself to the frozen inventory.
- Next implication: P0-P9 are closed. Do not alter the cache or start training
  automatically; the next eligible action is a separately approved bounded P10
  loss-comparison smoke.
