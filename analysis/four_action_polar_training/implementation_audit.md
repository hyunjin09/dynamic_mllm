# Four-Action Image+Question POLAR Implementation Audit

Date: 2026-08-28

## Outcome

The implementation is ready for the separately authorized fresh visual-cache
extraction. Training is deliberately fail-closed until that cache is complete;
no extraction, training, or external generation was launched during
preparation.

## Scientific contract

- Conditioning: Image+Question only.
- Training datasets: GQA, ChartQA, TextVQA.
- Eligible population: 6,811 records, split 5,945 train / 866 validation with
  no image-group leakage.
- Exclusion: 106/6,917 source records have zero replay-valid routes. Their
  identities and reason are preserved in the manifest audit; no label is
  fabricated.
- Supervision: all 248,804 replay-valid complete 28-layer routes, without the
  historical binary max-50 cap.
- Action order: IGNORE, READ_ONLY, WRITE_ONLY, FULL, matching the unified
  executor contract SHA-256
  `d8f524b928fb30ea0bb37c6a9389893adb338d4f91992d85255fdfb9bea283cb`.
- Objectives: duplicated one-hot four-action BCE and exact weighted
  complete-valid-set NLL.
- Inputs: frozen Qwen3-Embedding-0.6B question token rows plus unpooled frozen
  Qwen2.5-VL projected visual rows entering decoder layer 0.

## Matched optimization and selection

Both runs freeze the binary full10 settings: ten epochs, AdamW, learning rate
`5e-4`, weight decay `0.01`, cosine schedule, ten warmup steps, global physical
and effective batch 128, accumulation 1, route microbatch 32, gradient clip
1.0, BF16, deterministic algorithms, seed 20260809, and no early stopping.

Every epoch saves predictor, optimizer, scheduler, config, resolved asset
checksums, step, and full metrics. All 866 validation examples run after every
epoch. Best checkpoint order is validation Hit@1, Hit@5, lower nearest-valid
Hamming, lower objective loss, then earlier epoch. Selection is written before
external outcomes can be evaluated. Epoch-boundary resume verifies every prior
checkpoint checksum and the complete asset contract.

## External evaluation

The evaluator loads exactly 14,960 records: ChartQA 2,500; MMMU-Pro Standard
1,730; MMMU-Pro Vision 1,730; POPE adversarial/popular/random 3,000 each. It
does not load DocVQA, TextVQA, MMStar, or base MMMU. Predicted complete routes
execute through the current four-action executor and are paired with a live
unified-FULL baseline; imported prediction caches do not enter the comparison.

Before a full run, one deterministic row from every split must pass native vs
unified FULL token-sequence, generated-answer, and evaluator-correctness parity,
plus repeated predicted-route/cache determinism. Full evaluation is resumable
and sharded. Integrity merge requires every UID exactly once and reports
per-benchmark and within-suite metrics without a cross-suite pooled score.

## Five-axis review

1. Correctness: action geometry, both loss equations, exact top-k decoding,
   route weighting, gradient normalization, validation aggregation, checkpoint
   order, and external population are covered by focused tests.
2. Provenance/leakage: manifests, audits, source evaluation files, feature
   tensors, checkpoints, preflights, and selections are checksum-bound. Visual
   extraction consumes no answer, correctness, or route-outcome fields.
3. Failure safety: cache writes are atomic per image group; finalization cannot
   publish partial shards; trainers and evaluators refuse overwrite by default
   and have verified resume paths.
4. Efficiency: questions are encoded once per unique input, BCE expands only
   predictor features in route microbatches, repeated image paths load once per
   batch, cache extraction/evaluation can use eight shards, and the two losses
   can share the eight GPUs four/four in one allocation.
5. Portability/operations: tracked configs contain scientific settings, not
   server topology. The eight-GPU Slurm recipe is machine-local and ignored by
   Git. GPU code has not yet received a real-device smoke because this process
   is outside an allocation; that smoke is a required runtime gate, not an
   unresolved code finding.

## Verification evidence

- `PYTHONPATH=/home/hyunjin/projects/dynamic_mllm .venv/bin/pytest -q tests`:
  460 passed.
- All new Python modules compile.
- BCE static preflight: PASS, `ready_for_training: false` only because the
  fresh cache is absent.
- NLL static preflight: same outcome and the same cache contract.
- Live host: one idle Slurm node, eight idle NVIDIA H100 80GB GPUs, no current
  user queue entries.

## Required next runtime gate

Extract the fresh visual cache in eight GPU shards inside one Slurm allocation,
finalize it on CPU, then rerun both preflights with `--require-visual-cache`.
Only after both say `ready_for_training: true` should training start.
