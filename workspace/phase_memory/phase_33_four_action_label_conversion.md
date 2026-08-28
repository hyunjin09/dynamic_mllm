# Phase 33: Four-Action Label Conversion Memory

## Current Objective

Execute `plans/4way_labeling.md`: convert every authoritative positive binary MCTS route for GQA, TextVQA, ChartQA, WeMath2.0 Standard, and WeMath2.0 Pro into replay-validated four-action supervision without new MCTS.

## Active Constraints

- Use the existing unified four-action executor and current Qwen2.5-VL revision `cc594898...`.
- VQA authority is only `datasets/mcts_labels/gqa_textvqa_chartqa_v1/`; never use `datasets/mcts_v2`.
- Use the current training-authoritative positive route view; do not promote raw negative/evaluated masks.
- W2C labels are corrective and may be purified/refined toward FULL; C2C labels are preserving/efficiency labels and must remain mechanically mapped.
- Every final route must be executed jointly and evaluator-correct.
- Use all eight GPUs and prefer two model replicas per GPU after a clean five-dataset pilot.
- Preserve source labels, splits, provenance, dirty worktree contents, and append-only resumable outputs.

## Current State

- Superseded on 2026-08-25 by the user-approved answer-aligned three-suppression conversion in `plans/4way_labeling_fix.md`; active execution state moves to Phase 34. Job 1605 had zero records and is to be canceled rather than run under obsolete label semantics.
- Done: scientific plan and relevant Phase 12/13/23/32 memories inspected; live environment, cluster, assets, and source roots audited.
- Done: source authority is frozen at 12,278 positive samples and 545,531 positive binary routes; complete-route execution, deterministic conversion logic, resumability, current ALL-OFF stratification, and the 56-sample pilot/full runners are implemented. The active project suite passes 380/380 tests.
- Done: Slurm jobs 1598 and 1599 completed the five-dataset pilot and exact-resume gate. The formal pilot audit passed every semantic, accounting, concurrency, checksum, and worker-health check over all 56 samples.
- In progress: job 1604 exposed a machine-local Pillow image-size limit after 87 records and was canceled rather than allowing a partial-failure finish. Its artifacts are archived as provenance. A checksum-gated oversized-image open now exactly replays the affected frozen image; clean-contract replacement job 1605 is pending for all eight H100s, with scheduler-projected start 2026-08-26 15:57:56 KST.
- Open limitation: the complete Standard source manifest has 5,843 dataset rows while the largest transferred same-contract terminal cache has 5,381 records. The 462 rows without a terminal record have no existing binary label to convert and are disclosed separately rather than inferred negative or regenerated.
- Most recent useful observation: the pilot completed 56/56 samples and 4,026/4,026 source-route decisions with zero worker failures. It produced 27 W2C and 29 C2C records, retained 121 current-runtime replay failures explicitly, and passed exact old-binary semantic parity on all 56 samples. Fixed-bin tail imbalance depressed median allocated-GPU utilization to zero even though active replicas remained healthy, directly validating the dynamic shared queue for full execution.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| VQA frozen training view has 6,917 positive records and 237,802 selected routes | `outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl`; Phase 12 P8 audit | Fixes GQA/TextVQA/ChartQA input authority | confirmed |
| WeMath Pro root has 4,544 terminal records under the completed hard-cap-400 lineage | `datasets/math_labels/wemath20_pro_mcts_max400_v2/` | Fixes Pro raw positive-route source | confirmed |
| Standard project-linked root has 5,302 unique files versus 5,843 expected | `datasets/math_labels/wemath20_standard_mcts_max400_v1/`; frozen contract/audit plus current file census | Prevents a false all-label completion claim | confirmed incomplete |
| External Standard partial root contains the same 5,302 records byte-identically plus 79 later records | `/data/research/datasets/Sparse_Visual_Contextualization/.../wemath20_standard_max400_partial/` | Supplies recoverable same-lineage evidence but remains 462 short | confirmed partial |
| Frozen conversion inventory contains 12,278 positive samples and 545,531 positive routes | `datasets/mcts_labels_4action/source_inventory_v1/` | Fixes the exact all-label conversion population | confirmed |
| Complete heterogeneous four-action routes preserve route-specific caches | `tests/test_four_action_binary_executor.py`; 20 passing executor tests | Enables joint trajectory conversion rather than local-only interventions | confirmed |
| Pilot jobs 1598/1599 used 16 workers on eight GPUs and passed exact resume | `analysis/4action_label_conversion/pilot_audit_v1.json`; Slurm accounting; pilot telemetry | Satisfies the semantic/concurrency gate for full launch | confirmed passed; 56/56 records, zero failures |
| Frozen oversized images are valid and exact native preprocessing is reproducible | `analysis/4action_label_conversion/experiment_log.md`; targeted runtime regression tests | Supports a server-compatibility repair without changing image semantics | confirmed; 16 images / 26 samples, 380/380 tests pass |
| Prior route study validated two replicas/H100 and arbitrary binary route baselines | `analysis/4action_route_conditioned/`; Phase 32 memory | Provides executor/concurrency foundation | confirmed |
| Slurm exposes one server with 192 CPUs, 1,869,259 MiB RAM, and eight H100s | fresh `squeue`, `sinfo`, `scontrol`, and `nvidia-smi` on 2026-08-25 | Fixes current compute/memory topology | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Treat a stale Standard audit as current cache census | Audit reported 3,357 terminal records while 5,302 files now exist | supported stale audit artifact | audit JSON versus direct unique-file census | Rebuild a read-only current census and bind it to file hashes before source selection | Do not use historical audit counts as current completeness |
| Consider combined v31 as missing Standard replacement | Its own input summary says Standard is partial (5,345/5,843) and metadata uses a distinct Torch 2.9.1 lineage | supported contract/source mismatch | combined v31 `input_summary.json` and preference metadata | Exclude it from authoritative conversion input | Do not merge merely available math routes |
| Run full conversion under this server's default Pillow image limit | One checksum-matched 231,050,981-pixel Standard image raised `DecompressionBombError`; 16 frozen images affect 26 samples | supported machine/Pillow threshold mismatch | job 1604 failure log, frozen-manifest header scan, exact processor replay | Verify content hash before a scoped limit-free open; restart under one clean contract | Do not mix job 1604 records with post-fix records |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| VQA max-50 + all raw positive math routes | Exactly follows each source tree's current training authority | Plan-consistent source set | high conversion cost | selected |
| Combined v31 training routes | Already derived and includes both math datasets | Faster source preparation | medium, but changes runtime/source and is partial | rejected |
| All raw VQA and math positive routes | Maximizes route count | Avoids any cap | much higher and violates frozen VQA training authority | rejected |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: start and complete the clean-contract full conversion; the immediate bottleneck is another user's exclusive eight-GPU allocation.
- Relevant memory item used: transferred/current-runtime labels must be replayed, and route effects are trajectory-context dependent.
- Confirmed observation: VQA has an explicit training-authoritative max-50 view; math roots do not, so their positive candidates are the least transformed authoritative views.
- Unverified interpretation: the remaining 462 Standard terminal records may exist on the original server or an untransferred location.
- Diagnosis: supported incomplete transfer or incomplete source cache; exact origin is not yet distinguished and does not change current implementation work.
- Viable alternatives considered: frozen VQA/max-50 plus raw-positive math; external combined preference routes; raw routes for every dataset.
- Chosen action: preserve job 1604 as excluded provenance, relaunch from an empty full-output directory under replacement contract `32f6c3...`, and let job 1605 wait under normal Slurm policy for all eight H100s. Once live, verify all 16 ranks, contract hash, zero failures, and sustained GPU use before continuing exact-resume monitoring.
- Strongest objection: raw math positive sets may make conversion substantially more expensive than a derived cap, but silently imposing a cap before conversion would violate the plan.
- How this differs from failed attempts: it does not treat cached historical correctness as current validity and does not merge incompatible label lineages.
- Automatic execution authorized: yes, by the user's explicit request to perform and finish `plans/4way_labeling.md`.
- Authorization basis: current user goal and the plan's automatic pilot-to-full sequence.
- Stop condition: unresolved executor/evaluator semantic failure. A clean pilot automatically authorizes the resumable full conversion; ordinary current-runtime source replay failures remain explicit label exclusions as required by the plan.

## Latest Research-Action Result

- Action taken: diagnosed job 1604's first worker failure, scanned every frozen positive-label image header, implemented a hash-gated Pillow limit retry, replayed the exact affected sample, passed 380/380 tests, archived the old partial output, and submitted clean job 1605.
- Result: the 19,831 x 11,651 image reproduces the authoritative 16,334-token prompt and `[1,196,332]` image grid without resizing. Job 1605 has 0/12,278 clean records while pending; Slurm projects 2026-08-26 15:57:56 KST because job 1600 currently owns the node.
- Evidence saved: `analysis/4action_label_conversion/implementation_audit.md`, `analysis/4action_label_conversion/experiment_log.md`, `full_pre_image_limit_fix_job1604/`, pre-fix telemetry, regression tests, and Slurm accounting for jobs 1604/1605.
- Failure or issue: this server's Pillow threshold rejected valid frozen assets. The scientific semantics are not unresolved; the remaining issue is compute availability.
- Lesson learned: the live host exposes about 1.78 TiB RAM. The 180G request is
  a measured right-sized allocation (aggregate worker RSS below 50G), not a
  workaround for an undersized host; an earlier 187 GB transcription was
  corrected from fresh `scontrol` evidence.
- Next implication: monitor job 1605's transition to running, then verify its 16-rank startup and replacement contract before collecting clean telemetry. Resume from its atomic records if the seven-day allocation expires; on terminal success, run the full audit before finalization and analysis.
