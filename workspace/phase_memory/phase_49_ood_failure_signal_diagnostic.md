# Phase 49: OOD First-Failure Signal Diagnostic Memory

## Current Objective
Measure whether the frozen Phase-48 layerwise dense-failure signal transfers across datasets using three leave-one-dataset-out linear-probe experiments, with a native pre-language-decoder control.

## Active Constraints
- Use only the current-runtime 7,999-sample GQA/ChartQA/TextVQA population and LMMS-eval dense correctness labels.
- Train 28 independent linear probes with the Phase-48 model form and optimization settings; do not train a shared predictor or tune on held-out target data.
- Hold the target dataset out from training, checkpoint selection, threshold calibration, and layer selection.
- Calibrate 99%, 98%, and 95% correct-preservation thresholds on source validation only and transfer them unchanged to the full held-out target.
- Reuse the Phase-48 image-group-disjoint split assignments for source train/validation; evaluate OOD on the full held-out dataset.
- Use all four directly available GPUs after live availability verification; this server has no Slurm.
- Stop after the OOD diagnostic and its reports; no Stage-1 shared model, W-to-C work, routing, Stage 2, or external benchmark evaluation.

## Current State
- Done: Completed the frozen 12-row capture smoke, all-7,999 input-control extraction, 87 source-only fits, pre-target source decision freeze, one-pass OOD scoring, in-domain comparison, figures, reports, and integrity audit.
- In progress: None.
- Blocked: No.
- Most recent useful observation: Mid-layer ranking transfer exists on TextVQA and ChartQA, but source-selected depth and conservative calibration do not transfer uniformly; the prospective benchmark-general gate fails.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Current dense population has 7,999 valid samples with 3,999 correct and 4,000 wrong | `analysis/dense_failure_stage1/current_dense_8k/` | Defines the only allowed labels/population | confirmed |
| Phase-48 layer-0 test AUROC was 0.8421 and peak AUROC was 0.8992 at layer 21 | `analysis/dense_failure_stage1/layerwise_failure_probe/` | Motivates testing whether the signal is transferable rather than only in-domain | confirmed |
| Existing saved features are post-decoder-layer states | `dense_failure_stage1/runtime.py`; current feature schema | A separate pre-language-decoder control must be extracted | confirmed |
| Phase-48 split manifest covers all 7,999 UIDs and is image-group-disjoint across its global train/val/test partition | `analysis/dense_failure_stage1/layerwise_failure_probe/split_manifest.jsonl` | Supplies frozen source train/validation membership | confirmed |
| Four GPUs were idle at the latest live check | `nvidia-smi` on 2026-08-31 | Supports direct four-way parallel execution | confirmed |
| Native pre-decoder capture passed 12/12 exact repeats with one pre-hook and zero decoder forwards | `analysis/dense_failure_stage1/ood_signal_diagnostic/smoke/smoke_audit.json` | Validates the input-control boundary | confirmed |
| Input control covers 7,999 unique UIDs in 128 provenance-bound shards | `analysis/dense_failure_stage1/ood_signal_diagnostic/input_features/extraction_audit.json` | Makes the same-population input-vs-hidden comparison complete | confirmed |
| Source-selected OOD AUROC is 0.7236 TextVQA, 0.4502 ChartQA, and 0.6160 GQA | `analysis/dense_failure_stage1/ood_signal_diagnostic/decision_summary.md` | Rejects uniform benchmark-general transfer under the frozen rule | confirmed |
| Layer-21 descriptive OOD AUROC is 0.8046 TextVQA, 0.8018 ChartQA, and 0.5975 GQA | per-run `layerwise_metrics.csv` | Shows nonuniform mid-layer signal rather than total OOD absence | confirmed |
| Source 99%-preservation thresholds yield actual target preservation 0.995/0.305/0.138 | per-run `selective_metrics.csv` | Conservative calibration transfers only to TextVQA | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Treating the last literal user token as the answer-generation position in Phase 46 | That position predicts a chat delimiter, not the first assistant answer token | supported | Phase-47 corrected analysis | Keep claims position-specific; here `text_final` is an input representation, not an answer-start readout | Do not interpret this diagnostic as answer-token emergence |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| First-decoder-layer pre-hook over the native merged prompt representation | Preserves the production tokenizer, processor, visual encoder, and multimodal insertion while stopping before decoder layer 0 | Cleanest native pre-language-decoder control | high | testing |
| Manual reconstruction of embeddings plus visual insertion | Could expose the same conceptual boundary | Alternative if the native hook cannot be validated | high | unchecked |
| Reuse saved layer-0 states as the control | No new extraction needed | Cheap comparison only | low | rejected because it has already passed through a decoder layer |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: Determine whether dense-failure predictability transfers to a held-out dataset and how much decoder processing adds beyond the native pre-language-decoder representation.
- Relevant memory item used: Phase 48 showed strong in-domain predictability from layer 0 onward, but did not distinguish transferable failure signal from dataset-specific regularities.
- Confirmed observation: The saved Phase-48 representation begins after decoder layer 0, while the native runtime exposes a first-layer pre-hook boundary after multimodal prompt construction.
- Unverified interpretation: Strong in-domain early-layer performance may reflect transferable input difficulty, benchmark-specific regularities, or both.
- Diagnosis: unknown
- Viable alternatives considered: Native first-layer pre-hook; manual multimodal sequence reconstruction; reusing post-layer-0 states.
- Chosen action: Run the specified three leave-one-dataset-out independent-probe suites, including a validated native pre-language-decoder control and fixed hidden layers 0/14/21/27.
- Strongest objection: The control is not raw input; it contains learned token embeddings, visual-encoder features, and multimodal insertion. It supports claims only about information beyond the native pre-language-decoder representation.
- How this differs from failed attempts: It does not reinterpret prompt-position logits and does not reuse target data for model or threshold selection.
- Automatic execution authorized: yes
- Authorization basis: The user explicitly requested execution of `plans/ood_first_failure_signal_diagnostic_plan.md`.
- Stop condition: All three OOD experiments, the pre-decoder comparison, Phase-48 in-domain comparison, required figures, and decision summary are complete, or a validity gate fails.

## Latest Research-Action Result
- Action taken: Ran the three frozen leave-one-dataset-out 28-layer linear-probe suites plus the native pre-language-decoder controls and same-UID Phase-48 comparisons.
- Result: All 87 tasks completed. Source validation selected layers 22/26/20 for TextVQA/ChartQA/GQA; full-target AUROC was 0.7236/0.4502/0.6160. The fixed interpretation heuristic does not support a strong benchmark-general computation-dependent claim.
- Evidence saved: `analysis/dense_failure_stage1/ood_signal_diagnostic/` with the frozen protocol, source decisions, all required CSVs/figures/summaries, and a passing artifact manifest.
- Failure or issue: No execution failure. Scientifically, source-optimal depth and source-calibrated preservation thresholds are unstable across target tasks; ChartQA's selected layer is below chance and ChartQA/GQA preservation collapses.
- Lesson learned: In-domain accessibility and target-descriptive mid-layer OOD performance do not guarantee target-blind layer selection or conservative calibration transfer.
- Next implication: Stop. Do not train the shared predictor on a benchmark-general rationale without a new explicit authorization and a prospectively stated deployment-mixture or cross-task robustness objective.
