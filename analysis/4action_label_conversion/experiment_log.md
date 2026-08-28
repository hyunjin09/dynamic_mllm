# Four-Action Label Conversion Experiment Log

## 2026-08-25: Source freeze and implementation

- Frozen source manifest: `datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl` (`a44ca6e8684bc1a559997ce0ea52b2796f3265d19be90e22439c653741f36ed7`).
- Population: 12,278 positive samples and 545,531 positive binary routes.
- Standard source limitation: 5,381/5,843 source rows have transferred terminal records; the other 462 have no binary label record and are not regenerated.
- Focused tests before launch: 41 passed. A later training-view selector test brings the preparation suite to 42 without changing the in-memory pilot converter.
- Pilot: 56 samples, 4,026 source routes, static cost proxy 59,651.
- Frozen concurrency: 16 workers, two independent Qwen replicas per each of eight H100s, one sample per worker at a time.

## 2026-08-25: Pilot job 1598

- Submission: Slurm job `1598`, name `4act-label-pilot-v1`.
- Resources: 8 H100 80GB GPUs, 64 CPUs, 180G RAM, 24-hour limit.
- Live topology: the server exposes 1,869,259 MiB scheduler-visible RAM
  (about 1.78 TiB). The pilot used 180G because measured aggregate RSS for 16
  replicas remained under 50G; the 512G template would fit but was unnecessary.
- Command: `torchrun --standalone --nproc_per_node=16 experiments/run_four_action_label_conversion.py --mode pilot --resume` with deterministic CuBLAS configuration.
- Initial worker gate: all 16 processes started, all eight GPUs reached approximately 99% utilization, and memory stabilized near 33.4--33.9 GiB per GPU for two replicas (about 16.7 GiB per replica).
- First completed samples: exact generated-token, generated-answer, and correctness parity between the old binary executor and unified FULL/IGNORE route execution for the checked source routes; no worker failure artifacts.
- Telemetry: `analysis/4action_label_conversion/pilot_gpu_telemetry.csv` (5-second per-GPU samples while the job runs).
- At 29 completed records, no worker failure was present. Completed timing showed median seconds/static-cost of about 0.87 for W2C versus 0.032 for C2C; a 322-route W2C Standard stress sample completed in 1,813 seconds after 11,843 unique route evaluations.
- Fixed pilot bins exposed the intended load-balancing stress: after some ranks completed, GPUs with two replicas remained at 97--99% utilization while one-replica GPUs fell to roughly 15--31%. Full mode now uses a launch-scoped atomic shared queue, with pending work ordered by the measured W2C/C2C cost ratio. This changes scheduling only, and a new Slurm launch token makes incomplete claims retryable.
- Focused tests after the queue, execution-contract, and joint-composition audit additions: 58 passed.
- Current verification after ALL-OFF stratification, server compatibility,
  and explicit final-decision reporting: active project `tests/` suite 378/378
  passed. A temporary real-schema smoke over
  32 completed pilot records passed both final-view construction (2,364 raw
  conversions, 1,675 unique routes, 970 bounded training routes) and the full
  statistics/plot/report pipeline; temporary artifacts were automatically
  removed.
- Resume-validation job `1599` used `afterok:1598`, all eight GPUs/16 ranks,
  and exited in 20 seconds without overwriting any records.

## 2026-08-25: Pilot completion, audit, and full estimate

- Job `1598` completed in 5:59:48 with exit code 0; job `1599` completed in
  00:00:20 with exit code 0.
- Final pilot output: 56/56 atomic sample records, zero worker failure
  artifacts, 27 W2C and 29 C2C samples, and exact accounting for all 4,026
  source routes.
- Current-runtime replay: 3,905 valid routes and 121 explicit replay failures;
  no failed route was replaced. All 2,310 unique final action routes are jointly
  evaluator-correct.
- Formal audit: `analysis/4action_label_conversion/pilot_audit_v1.json`, passed
  all semantic, checksum, worker, concurrency, resume, and source-accounting
  gates.
- Telemetry: 4,147 five-second snapshots / 33,176 per-GPU observations, peak
  memory 41,979 MiB, mean allocated-GPU utilization 21.51%. The low aggregate
  mean/zero median reflects the deliberate fixed-bin tail after most ranks
  finished, not stalled active workers.
- Measured throughput: 9.33 samples/hour, 0.186 source routes/second, and 48.02
  allocated GPU-hours for the stress-stratified pilot.
- Full projection: `analysis/4action_label_conversion/full_compute_estimate_v1.json`.
  At 16 workers, the central estimate is 111.3 ideal wall-hours or 139.1 wall-hours
  at 80% scheduling efficiency; the p25/p75 stress range is 57.1--375.1 hours.
  Full execution is therefore resumable and may require continuation beyond one
  seven-day Slurm allocation.

## 2026-08-25: Full conversion launch

- Submitted Slurm job `1604`, name `4act-label-full-v1`.
- Frozen command: `torchrun --standalone --nproc_per_node=16 experiments/run_four_action_label_conversion.py --mode full --resume` with deterministic CuBLAS configuration.
- Resources: eight H100 GPUs, 64 CPUs, 180G RAM, `gpu-large` QOS, seven-day
  limit. Full mode uses the launch-scoped atomic shared queue and writes one
  checksum-protected record per completed sample.
- Launch checks: implementation hashes match `implementation_audit.md`; no
  prior full output exists; source manifest remains 12,278 samples / 545,531
  routes; jobs 1598/1599 are completed and the pilot audit passed.
- Initial state: pending with Slurm reason `AssocGrpGRES`. Another user's
  exclusive eight-GPU allocation is live, so the job is queued without
  interference. GPU telemetry will begin only after job 1604 starts.

## 2026-08-25: Full conversion early-run audit

- Job `1604` started at approximately 08:55 KST after the association GPU limit
  cleared. No external job was interrupted.
- At 1:13 elapsed, the full run had 28/12,278 checksum-protected records and
  zero failure artifacts. All 16 ranks were live and the atomic queue contained
  exactly 44 claims: 28 completed plus 16 active.
- All 28 record checksums passed. Every record used execution-contract SHA-256
  `678b20f40c9d4983d100a7f5c4a5a899e5ee84117c4a90b6adcd9f728fb09164`.
- Early semantics: 25 W2C and 3 C2C records; exact accounting for 7,788 source
  routes (7,639 replay-valid and 149 explicit replay failures); 1,290 unique
  final routes, all jointly evaluator-correct; 464 unique routes already
  contained READ_ONLY or WRITE_ONLY.
- Job-scoped telemetry started at 10:08 KST in
  `analysis/4action_label_conversion/full_gpu_telemetry.csv`. Its first 37
  snapshots showed 98.47% mean / 99% median utilization across all eight GPUs,
  per-GPU means of 98.22--99.00%, and 40,199 MiB peak memory. This verifies that
  the dynamic queue keeps two replicas per GPU supplied while enough work
  remains.
- No traceback, CUDA OOM, runtime error, or worker failure appeared in the full
  log or failure directory.
- At the 50-record milestone, all 50 sidecar checksums passed and every record
  still used the single frozen execution-contract hash. Exact accounting covered
  13,310 source routes (13,029 replay-valid plus 281 explicit replay failures),
  yielding 2,044 unique final routes; all were jointly evaluator-correct, and
  1,178 already contained READ_ONLY or WRITE_ONLY. No delayed semantic or
  contract inconsistency appeared.

## 2026-08-25: Job 1604 image-limit failure and contract-clean restart

- Direct observation: job 1604 recorded one worker failure for
  `wemath20_standard:2793`: Pillow raised `DecompressionBombError` for a frozen
  19,831 x 11,651 RGBA image (231,050,981 pixels), above this server's Pillow
  11.1.0 safety threshold. The job kept other ranks live, but would have exited
  nonzero and left 26 affected samples unconvertible.
- Validity check: the frozen image's content SHA-256 matched the source manifest.
  A scan of all 11,678 distinct frozen positive-label images found zero header
  failures, 22 warning-level images, and 16 error-threshold images affecting 26
  samples. The authoritative source record used the failing image at native
  resolution with 16,334 prompt tokens and 16,268 visual tokens.
- Supported diagnosis: this was a machine/Pillow safety-threshold difference,
  not corrupt data, a CUDA failure, or an executor-semantic failure. An exact
  local processor replay after the scoped fix reproduced dimensions
  19,831 x 11,651, 16,334 input tokens, grid `[1, 196, 332]`, and 65,072 pixel
  rows without resizing or capping.
- Repair: oversized-image retry is limited to `DecompressionBombError` and is
  permitted only after the file matches its frozen content SHA-256. The Pillow
  limit is restored immediately after opening. Two targeted regression tests
  pass, and the complete active project suite passes 380/380.
- Contract hygiene: job 1604 was canceled after 07:02:24. Its 87 completed
  records and one failure are preserved under
  `conversion_v1/full_pre_image_limit_fix_job1604/`, and its telemetry is
  preserved as `full_gpu_telemetry_pre_image_limit_fix_job1604.csv`. These
  records are provenance only and will not be mixed with the replacement run.
- Replacement execution-contract SHA-256:
  `32f6c3a5076e65ab15d7432ad296e16053528ebdb96b21fb1c695fda42fc2858`.
- Clean replacement job `1605` (`4act-label-full-v2`) was submitted at
  16:02:59 KST with the same 8-H100, 16-worker, 64-CPU, 180G, seven-day layout.
  It initially remained pending with reason `AssocGrpGRES` while another user's
  exclusive eight-GPU job 1600 occupied the server. Slurm's initial projected
  start was 2026-08-26 15:57:56 KST; this estimate is scheduler-derived and may
  move earlier if that job releases the allocation before its limit.
