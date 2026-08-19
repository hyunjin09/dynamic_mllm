# We-Math2.0-Pro Binary Route Label Extraction Plan

Status: active supplemental extraction, approved 2026-08-11.

## Objective

Inventory all 4,552 records in `We-Math/We-Math2.0-Pro` and generate
unrestricted 28-bit binary visual ON/OFF MCTS route labels for the 4,544
technically valid records while the original 8K
GQA/TextVQA/ChartQA extraction continues unchanged.

## Frozen execution contract

- Model: Qwen2.5-VL-7B-Instruct snapshot
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Executor: current verified `BinaryQwen25VL` layer-wise visual bypass.
- Route: full unrestricted 28-bit mask; no POLAR segmentation.
- Processing: native/default Qwen image processing; no custom visual-token cap.
- Generation: deterministic greedy, at most 96 new tokens, with a direct-answer
  prompt requiring only `<answer>...</answer>` and no reasoning.
- Scoring: official We-Math R1-V accuracy contract—extract the `<answer>` span
  when present, otherwise use the stripped response, then evaluate with
  `mathruler==0.1.0` `grade_answer`.
- Reward: binary correctness at threshold 1.0; retain raw score.
- MCTS: 200 simulations for current ALL-ON-correct and 400 for current
  ALL-ON-wrong, with a hard per-sample maximum of 400 and no extension.
- Cache: retain every evaluated positive and negative route, anchors, search
  trace, token geometry, masks, outputs, and scores.
- Hardware: node06, 8 A6000 GPUs, 96 CPUs, 240 GB RAM, with the required
  node06 NCCL P2P/CUMEM/IB safeguards.

### Runtime nontermination amendment (2026-08-12)

- Complete records produced under contract
  `96b2c632ebc6e020c607b3d9a0eddd2a29f7aff1912f5219327ae96a507c3a50`
  remain valid and are retained; the resume audit found 1,156 such records.
- The unchanged MathRuler result is used whenever grading completes within five
  seconds. A nonterminating grade is conservatively scored `0.0`/incorrect and
  explicitly marked `scoring_timed_out=true`. Identical decoded predictions
  reuse the first bounded score within a sample.
- Amended contract:
  `fc4a1df38925d20816770b861989b87d119bcdbf13b3bdff26a89b7abc90d485`.
- Because only six node06 GPUs were available at restart, the resumable run uses
  six workers, 72 CPUs, and 180 GB RAM. Shard count changes work assignment
  only; sample-specific seeds, MCTS, route semantics, scoring, and outputs are
  unchanged.

### Hard 400-simulation cap amendment (2026-08-13)

- The user closed the 400-to-600 extension after the completed-cache snapshot
  showed only 25/528 extended samples yielded any correction after simulation
  400, producing 28 valid masks.
- Every sample is now capped at 400 MCTS simulations: current ALL-ON-correct
  samples use 200; current ALL-ON-wrong samples use 400; no extension runs.
- Completed predecessor records with 200 or 400 requested/completed simulations
  may be retained after checksum and contract validation. Every predecessor
  600-simulation record is excluded from reuse and rerun under the 400 cap.
- The capped run uses the seven GPUs currently available on node06, 84 CPUs,
  and 197 GiB of Slurm-requested RAM. The initial 200G request could not
  co-reside with node06's existing 48,000 MiB allocation because Slurm expands
  `200G` to 204,800 MiB; 197G is the maximum whole-GiB request that fits the
  250,000 MiB node. Shard count affects assignment only; sample UID-derived seed,
  executor, score, and unrestricted mask space remain unchanged.
- Do not treat an interrupted in-memory sample as completed. Only atomic records
  with requested simulations equal to completed simulations may be retained.

## Ordered actions and gates

1. Materialize and checksum exactly 4,552 official Pro records and images.
   Preserve eight records with an empty question and/or answer as
   `technical_invalid`; exclude only these eight from MCTS.
2. Freeze the manifest, source/runtime contract, and five-record smoke set.
3. Run a minimal smoke: exact native/binary ALL-ON generated-token parity on
   5/5 records and exact repeated token/score equality for two fixed mixed
   masks per record.
4. If and only if smoke passes, immediately run seven-way sharded MCTS over all
   4,544 technically valid records.
5. Stop the launch action on smoke failure, missing/invalid records, scorer
   failure, nondeterminism, parity failure, or native-processing resource
   failure. Do not weaken the contract.

## Non-goals

- Do not cancel, modify, or merge with the active original 8K extraction.
- Do not train a predictor, router, probe, controller, or base model.
- Do not restrict routes to contiguous segments.
- Do not interpret label yields scientifically in this launch action.
