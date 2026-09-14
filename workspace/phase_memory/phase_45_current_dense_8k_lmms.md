# Phase 45: Current Dense 8K LMMS Dataset Memory

## Current Objective
Create the Stage-1 dataset of current native-dense Qwen2.5-VL layer-wise hidden
states paired with current task-specific LMMS-Eval final correctness.

## Active Constraints
- Use all recovered GQA/ChartQA/TextVQA candidates; historical correct/wrong is
  metadata only and never the target.
- Execute native dense all-on inference at every one of 28 decoder layers; no
  sparse or four-action code path is allowed.
- Score with installed official LMMS-Eval task implementations. Preserve raw
  TextVQA consensus and reuse the repository's existing score >= 0.5 binary
  convention.
- Extract final-query, mean-query, and mean-visual post-layer states in the same
  successful inference. Do not train a predictor.
- Run an 18-sample smoke, then the full population on four direct GPUs. Isolated
  bad samples are logged and skipped rather than aborting the population.
- Stop after dense outputs, feature shards/index, skip log, and summaries.

## Current State
- Done: recovered exactly 8,000 candidate annotations (4,000 GQA, 2,000
  ChartQA, 2,000 TextVQA) and verified all four local GPUs are idle.
- Done: installed `lmms-eval==0.7.3` in the project-local `.venv`; exact GQA,
  ChartQA, and TextVQA scoring imports and focused semantics tests pass.
- Done: the 18/18 four-GPU smoke completed with LMMS labels and complete
  28-layer features.
- Done: all 8,000 candidates were attempted; 7,999 completed with features and
  one ChartQA record was skipped because its transferred image file is absent.
- In progress: none; this authorized action is complete.
- Blocked: none.
- Most recent useful observation: current native-dense generation exactly
  reproduced the stored historical prediction text for all 7,999 executable
  records, yielding no historical-bucket label flips. This is a measured parity
  result, not use of historical correctness in the scoring path.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Full candidate population is 4,000/2,000/2,000 | `search/greedy_phase1_phase2_reproduction/manifests/all_samples.jsonl` | Defines the route-independent population | confirmed |
| Existing binary thresholds are GQA 1.0, ChartQA 1.0, TextVQA 0.5 | same manifest and transferred evaluation contract | Avoids inventing a new fractional-score threshold | confirmed |
| Official LMMS task functions import from version 0.7.3 | project-local `.venv`; focused tests | Satisfies evaluator requirement without custom scoring | confirmed |
| Four RTX 6000 Ada GPUs currently report 0% utilization | fresh `nvidia-smi` | Enables direct four-rank execution | confirmed |
| Full run completed 7,999/8,000 with one missing-image skip | `analysis/dense_failure_stage1/current_dense_8k/generation_summary.json` | Defines the authoritative completed Stage-1 population | confirmed |
| All 7,999 feature records are BF16 [28,3584] with zero non-finite values | `analysis/dense_failure_stage1/current_dense_8k/features/feature_integrity_audit.json` | Validates feature completeness | confirmed |
| Independent rescore has zero score/label mismatches | deterministic post-run LMMS rescore; final JSONLs | Confirms saved labels follow LMMS task functions | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Previous fail-closed 24-sample parity/recheck workflow | Added gates beyond the newly requested simple dataset run | user intentionally replaced that workflow | explicit user instruction | Use only the small functional smoke and skip isolated bad samples | Do not resume the repair/recheck track |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| 18-sample dense+LMMS+feature smoke | Covers every dataset and historical stratum | Functional validity before full run | low | passed |
| Four-GPU full recovered population | Produces the requested Stage-1 dataset | Current labels and layer features | high | completed |

## Next-Step Decision
- Deliberation mode: fast
- Active objective and bottleneck: the data-generation objective is complete;
  the next phase would be split creation and predictor training, which remain
  separately unauthorized.
- Relevant memory item used: Phase 44 proved the complete route-independent
  8K identities are recoverable.
- Confirmed observation: official evaluator semantics and the historical-only
  role of source buckets are now explicit.
- Unverified interpretation: none needed for the completed data contract.
- Diagnosis: one skipped sample is supported as a missing transferred image;
  evidence is `skipped_samples.jsonl`.
- Chosen action: stop after the completed dataset and summaries, as required.
- How this differs from failed attempts: this does not seek historical parity,
  route correctness, hook/no-hook parity, or exact 8K completion.
- Automatic execution authorized: no further research action.
- Authorization basis: explicit user instruction to smoke then immediately run
  the full candidate population.
- Stop condition: met; predictor training has not started.

## Latest Research-Action Result
- Action taken: ran an 18-sample smoke followed by native dense all-on inference,
  LMMS-Eval scoring, and same-pass 28-layer feature extraction for the complete
  8,000-candidate population on four direct GPUs.
- Result: 7,999 completed and one missing-image skip. Current labels are 3,999
  correct / 4,000 wrong: GQA 2,000/2,000, ChartQA 999/1,000, TextVQA
  1,000/1,000. There are 7,476 completed image groups.
- Evidence saved: `analysis/dense_failure_stage1/current_dense_8k/` (4.6 GiB),
  including 128 feature shards, exact output/index manifests, evaluator
  contract, and summaries.
- Failure or issue: one missing ChartQA image:
  `chartqa:chartqa_train_OECD_COUNTRY_PROGRAMMABLE_AID_(CPA)_CYP_ISR_LTU_LVA_MLT_000023_7a9af7720d`.
- Lesson learned: the current normal dense executor is parity-stable with the
  historical stored dense predictions for every executable candidate, despite
  the separate four-action replay mismatch.
- Next implication: treat `dense_outputs.jsonl` and `features/feature_index.jsonl`
  as the authoritative completed population; request separate approval before
  creating a split or training the Stage-1 predictor.
