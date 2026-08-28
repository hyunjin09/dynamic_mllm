# Route-Conditioned Four-Action Experiment Log

## Execution policy

- GPU inference runs through Slurm and requests all eight H100 GPUs, as the
  user required for this experiment.
- CPU-only manifest building, merging, checksumming, aggregation, plotting, and
  report generation run locally on this server. No new CPU-only Slurm jobs are
  to be submitted unless the user explicitly changes this policy.
- Completed shards are append-only and are never overwritten.

## 2026-08-24 — prerequisites and frozen inputs

- The complete `plans/4way.md` pipeline finished successfully. Final job `1573`
  exited `0:0`; all newly generated checksum sidecars verify.
- Authoritative A+ input: 1,880 samples (GQA 1,222; TextVQA 658) from
  `analysis/4action_answer_alignment/cohort/cohort_manifest_v1.jsonl`.
- The route audit found no malformed cached correcting route. Historical
  nearest-route OFF counts range from 2 to 22 (median 9, mean 9.5271), implying
  53,733 new READ_ONLY/WRITE_ONLY/FULL-restoration cells before fallback.
- Deterministic candidate order is: ascending Hamming distance, descending
  comparable cached evaluator score, ascending route ID, ascending mask key.
  Every candidate must still generate a correct answer under the current
  unified executor; fallback follows the same fixed order.
- Candidate manifest: exactly 1,880 rows at
  `analysis/4action_route_conditioned/anchor_candidates.jsonl`, SHA-256
  `779f4a2cec5f71358e09c7fe25be001b75441e92f56d089fcc9937b6e7b44b30`.

## 2026-08-24 — implementation and verification

- Extended the existing unified four-action executor with an arbitrary binary
  FULL/IGNORE route baseline and one target-layer four-action override. All
  branches retain the same unified attention/mask/kernel machinery.
- Implemented current-runtime anchor validation, deterministic fallback,
  resumable pilot/full decomposition, exact-coverage mergers, GPU monitoring,
  pilot throughput comparison, and final aggregate-analysis preparation.
- Test status at 02:39 KST: 84 focused four-action tests pass; all new route-
  conditioned scripts compile.
- Local CPU analysis uses image-group bootstrap and an equivalent sum/count
  cluster resampler to avoid repeated array concatenation.

## 2026-08-24 — anchor validation launches

- Job `1576` (`four-action-route-anchor-v1-20260824`) requested all eight H100s
  and failed before any scientific result because deterministic PyTorch/CuBLAS
  required `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Two failure rows were preserved;
  no result row was produced. Evidence:
  `logs/slurm/four-action-route-anchor-v1-20260824-1576.log`.
- Resumable replacement job `1578`
  (`four-action-route-anchor-v2-cublas-20260824`) uses the unchanged validator
  with the required CuBLAS workspace setting. At 02:38 KST it was pending on
  `(AssocGrpGRES)` with all eight H100s requested while another user's
  eight-GPU job occupied the node. This is a GPU job, not a CPU job.
- There are no project CPU-only jobs in the live Slurm queue.

## 2026-08-24 — validated anchor freeze

- Job `1578` started at 02:40:45 KST and completed at 02:44:30 KST with exit
  `0:0` after 3m45s. All eight shard runtime contracts and all 1,880 unique
  current-runtime validation rows are present.
- Local merger passed with zero disqualifying failures. It froze 1,804 current-
  correct anchors (GQA 1,170; TextVQA 634) and excluded 76 samples for the
  prespecified reason `no_cached_correcting_route_current_correct`.
- 206 anchors required deterministic fallback beyond the first cached
  candidate, totaling 256 fallback evaluations.
- Production has 17,262 anchor-OFF positions and exactly 51,786 new branches.
  The 32 deterministic work units range from 1,614 to 1,626 expected new cells.
- Frozen anchor JSONL SHA-256:
  `f87911e0710200ed9ed4dd3b5631d4746e532e2807684d1de4cc3ed72cc19823`.
  Frozen anchor Parquet SHA-256:
  `e98242ca893baffddbe26ef38acd15eed8a16764e2b9bc62f9947ac1ba155a2c`.
- Every new anchor/pilot/work-unit checksum sidecar verifies locally.

## 2026-08-24 — pilot queue

- One-replica pilot job `1579` requests all eight H100s, 64 CPUs, and 512 GiB
  for the fixed 56-sample manifest. Its GPU telemetry is collected inside the
  allocation and is append-safe under resume.
- At submission it is pending on `(AssocGrpGRES)` behind another user's
  full-node run. No CPU-only Slurm work was submitted.

## 2026-08-24 — pilot results and full launch

- One-replica pilot job `1579` completed `0:0` in 3m39s. The local merger
  passed exact 56-sample / 523-OFF-position / 2,092-action-row coverage, all
  explicit M00 reproductions, all semantic gates, and zero failures. Useful
  throughput was 8.613831 new cells/s; peak VRAM was 17,475 MiB/H100 and mean
  sampled GPU utilization was 11.37%.
- Two-replica pilot job `1580` completed `0:0` in 2m51s on the identical
  manifest. Its merger produced the exact same taxonomy counts and passed all
  semantic/numerical gates with zero failures. Useful throughput was 12.183885
  new cells/s (1.414456x baseline); peak VRAM was 34,745 MiB/H100 and mean
  sampled GPU utilization was 27.77%.
- Selection: two replicas/GPU, based on valid intervention throughput rather
  than utilization alone.
- Exact full estimate saved before launch: 51,786 new cells, 4,250.37 seconds /
  1.18066 wall-hours, 9.44526 GPU-hours; 1.41679 wall-hours with 20% contingency.
- Full job `1581` started at 03:06:24 KST using all eight H100s and 16 workers.
  Startup checkpoint: 271 unique passing samples / 7,869 cells, zero failures,
  ~34.7 GiB peak VRAM/H100, 71.7% mean and 99% median sampled GPU utilization.

## 2026-08-24 — full completion, analysis, and interpretation

- Full job `1581` completed `0:0` at 03:35:37 KST after 29m13s. The local
  strict merger passed exact coverage for 1,804 samples, 17,262 anchor-OFF
  positions, 51,786 new branches, 69,048 action rows, all 16 workers, and zero
  failures. Measured merged throughput was 30.394689 new cells/s.
- The local image-group-bootstrap analysis completed in 15.2s. Of 17,262 OFF
  positions, 7,880 (45.65%, 95% CI 44.31--46.87%) were individually necessary
  and 9,382 (54.35%) redundant. Among necessary positions: READ-mediated
  20.55%, WRITE-mediated 42.88%, either-removal-sufficient 9.94%, and both-
  required 26.64%.
- READ-mediated positions occur later than WRITE-mediated positions (mean layer
  17.79 vs 9.97; image-group bootstrap difference 7.31--8.31).
- FULL-context discrete local rescue recalled 7.30% of route-necessary
  positions; 7,305/7,880 (92.70%) were missed without the correcting-route
  context. This is context-dependent route evidence, not global harmfulness.
- `research-control` STANDARD review compared stop, bounded joint refinement,
  and direct four-action search/router. The required independent reviewer
  returned `stable`, ranking bounded proposal > stop > direct pivot with high
  confidence. The bounded proposal is documented but explicitly not launched.
- Final raw-table integrity audit recomputed all margins, five factorial
  effects, action semantics, anchor targets, M00 correctness, and taxonomy with
  maximum absolute error 0.0. All 51 pre-audit checksum sidecars passed; the
  audit itself and proposed plan have verified sidecars. The focused test suite
  passes 90 tests.

The approved `plans/4way_2.md` action is complete. No further GPU or CPU job is
authorized. The optional bounded joint-refinement pilot remains a proposal only
and requires an explicit new decision.
