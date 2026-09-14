# Phase 48: Layer-Wise Dense-Failure Predictability Memory

## Current Objective
Measure when current native-dense LMMS failure becomes linearly predictable
from the frozen 28-layer compact hidden-state summaries, before training the
shared Stage-1 predictor.

## Active Constraints
- Use only the 7,999 completed Phase-45 samples and their current LMMS labels:
  3,999 correct and 4,000 wrong across GQA, ChartQA, and TextVQA.
- Freeze one deterministic image-content-group-disjoint approximately
  6,400/800/800 train/validation/test split before fitting any probe.
- At each layer use the same fixed compact state: concatenated `text_final`,
  `text_mean`, and `visual_mean` vectors from the Phase-45 feature contract.
  Do not add dataset identity, route information, confidence readouts, an MLP,
  a shared predictor, or learnable layer embeddings.
- Fit 28 independent binary linear probes with identical optimization settings.
  Use only train/validation data for fitting, checkpoint selection, thresholds,
  and the descriptive candidate-region decision; evaluate test exactly once
  afterward.
- Use four direct GPUs if all remain idle. Stop after this probe analysis and
  held-out evaluation.

## Current State
- Done: froze contract `3cf49a46...0234cd`, exact image-group-disjoint
  6,399/800/800 split, 28 probe checkpoints, validation-only thresholds and
  candidate region, one held-out test evaluation, five figures, and the final
  artifact manifest.
- In progress: none; the authorized action is complete.
- Blocked: none.
- Most recent useful observation: held-out failure AUROC is already 0.8421 at
  layer 0 and peaks at 0.8992 at layer 21, while explicit answer identity from
  Phase 47 remains a layers-25-27 event.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Current dense labels cover 7,999 executable samples | `analysis/dense_failure_stage1/current_dense_8k/dense_outputs.jsonl` | Defines the only allowed target population | confirmed |
| All 7,999 records have three finite 28-layer feature summaries | `analysis/dense_failure_stage1/current_dense_8k/features/feature_integrity_audit.json` | Makes the planned probe analysis executable without model inference | confirmed |
| Explicit answer identity emerges mainly at layers 25-27 | `analysis/dense_failure_stage1/answer_logit_emergence_v2/analysis_summary.md` | Supplies the fixed reference curve/depth for the main comparison | confirmed |
| `text_final` is not answer-start aligned but remains a valid learned-predictor input | `workspace/decision_log.md` | Prevents invalid logit-lens reuse while preserving the probe input contract | confirmed |
| Failure is linearly accessible from the first probed layer | `analysis/dense_failure_stage1/layerwise_failure_probe/test_metrics.csv` | Rules out a late-only onset for this compact representation | confirmed |
| Fixed validation thresholds retain useful selective recall on test | same `test_metrics.csv` | Establishes operational signal beyond AUROC | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Phase-46 unlearned final-head readout at `text_final` | Read chat structure rather than answer token | supported positional mismatch | `analysis/dense_failure_stage1/logit_emergence/validity_diagnostic.json` | A learned failure probe may use the state, but its curve must be compared with the corrected Phase-47 answer-position result | Do not interpret probe weights/logits as an answer-start vocabulary readout |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Concatenate all three frozen compact summaries per layer | Uses the complete prospectively extracted Stage-1 state without adding a learned module | Earliest linearly accessible failure signal for the intended compact input | medium | completed |
| Probe `text_final` only | Lower-dimensional and directly token-indexed | A narrower query-state diagnostic | low | not selected; discards two frozen state summaries |
| Run a representation ablation over all summaries | Could attribute signal to text or vision | Modality attribution rather than the authorized depth question | high | out of scope |

## Next-Step Decision
- Deliberation mode: standard
- Active objective and bottleneck: determine whether final dense failure is
  linearly accessible before the corrected layers-25-27 answer-emergence range.
- Relevant memory item used: Phase 47 established late explicit answer
  commitment but explicitly did not show that earlier hidden states lack
  correctness information.
- Confirmed observation: the current dataset has 7,999 complete feature/label
  joins and 7,476 image-content groups.
- Confirmed interpretation: the concatenated compact state exposes held-out
  failure information before answer identity becomes explicit; this is learned
  linear accessibility, not a claim that an answer is formed early.
- Diagnosis: not applicable; the planned analysis completed validly.
- Viable alternatives considered: full compact concatenation, `text_final` only,
  and a representation ablation.
- Chosen action: freeze the full three-summary input, image-group split, one
  common regularized linear-probe protocol, validation-only selective
  thresholds, and then execute exactly the requested 28-layer analysis.
- Strongest objection: concatenation is wider than the sample count and does not
  attribute signal by modality; fixed train-only normalization and regularized
  linear fitting address overfitting risk, while attribution remains outside
  this plan.
- How this differs from failed attempts: this learns a held-out binary failure
  decoder from compact states and never applies the LM head at the invalid
  literal-user-token position.
- Automatic execution authorized: yes, one layer-wise linear-probe analysis.
- Authorization basis: explicit user instruction to read and perform
  `plans/layerwise_dense_failure_predictability_plan.md`.
- Stop condition: all required split, checkpoint, validation/test metric,
  figure, and analysis artifacts exist, or a validity failure makes the planned
  result uninterpretable. Met; no shared-predictor training followed.

## Latest Research-Action Result
- Action taken: trained 28 independent regularized linear probes over the
  concatenated 10,752-dimensional compact state using four direct GPUs, froze
  the informative region from validation, and evaluated the untouched test
  split once.
- Result: layer-0 test AUROC/AUPRC are 0.8421/0.8519; best test AUROC is 0.8992
  at layer 21. The full 0-27 stack is informative, with a stronger descriptive
  plateau around layers 16-27. At transferred validation thresholds, the best
  test recalls that still meet 99%/98%/95% correct preservation are
  0.3350/0.4275/0.5225 at layers 14/19/27. Signal begins at layer 0 in every
  dataset, but GQA is weaker (0.6981 at layer 0, peak 0.7713) than ChartQA and
  TextVQA (already about 0.94 at layer 0; peaks above 0.97).
- Evidence saved: `analysis/dense_failure_stage1/layerwise_failure_probe/`,
  especially `analysis_summary.md`, `test_metrics.csv`,
  `dataset_layerwise_metrics.csv`, and `artifact_manifest.json`.
- Failure or issue: none. All 13 required artifacts and 28 checkpoints pass
  SHA-256 verification; the final focused suite passes 21/21 tests.
- Lesson learned: final dense failure is linearly decodable long before explicit
  answer-token commitment for this compact representation. The depth ordering
  is strong, but the probe/answer-readout axes are different and modality
  attribution was not tested.
- Next implication: a later separately authorized training experiment should
  not compare all-layer with “informative-only 0-27,” because those arms are
  identical. The clean comparison is all-layer versus random-k over 0-27; an
  optional predeclared 16-27 arm would be a stronger-plateau efficiency
  sensitivity, not evidence that early layers lack signal.
