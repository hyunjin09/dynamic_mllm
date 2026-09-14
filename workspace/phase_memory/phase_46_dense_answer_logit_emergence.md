# Phase 46: Dense Answer-Logit Emergence Memory

## Current Objective
Measure how first-answer-token preference develops across the 28 saved native-
dense decoder states, before any Stage-1 predictor training.

## Active Constraints
- Use only the 7,999 completed current-runtime dense samples and their LMMS-Eval
  labels from Phase 45.
- Apply the frozen Qwen2.5-VL final RMSNorm and LM head to the saved
  `text_final` query-position state; do not fit a learned probe.
- Use canonical `gt_answer` first-token IDs and exact generated first-token IDs.
- Freeze raw-logit delta `1.0`, three-layer persistence, and early cutoff layer
  `4` before inspecting results; also report persistent zero crossings.
- Run the extraction over four direct GPUs and stop after tables, figures, and
  interpretation. Do not train, route, repair W2C, or evaluate externally.

## Current State
- Done: the user-authorized plan and Phase-45 evidence were read completely.
- Done: 128/128 feature shards and 7,999/7,999 UIDs were scored by four direct
  GPUs under frozen contract
  `73daef6c851378784aa498493b152a49ec5fa864c66b53b033f29749e606c99b`.
- Done: required CSVs, four figures, bootstrap bands, sample emergence rows,
  deterministic wrong taxonomy, and interpretation report were written.
- In progress: none; this bounded action is complete and stopped.
- Blocked: none.
- Most recent useful observation: zero correct samples meet either the delta or
  zero-crossing emergence rule because the final-user-token readout predicts
  chat structure; `<|im_end|>` is layer-27 top-1 for 54/54 correct records in
  the one frozen-shard validity diagnostic.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Current dense population is 3,999 correct / 4,000 wrong | `analysis/dense_failure_stage1/current_dense_8k/generation_summary.json` | Defines the label strata | confirmed |
| All 7,999 records have finite 28-layer features | `analysis/dense_failure_stage1/current_dense_8k/features/feature_integrity_audit.json` | Makes complete-population logit analysis possible | confirmed |
| `text_final` is the final literal user-prompt token | `analysis/dense_failure_stage1/current_dense_8k/features/feature_schema.json` | Bounds interpretation to a query-position logit lens | confirmed |
| Complete logit coverage and finite values | `analysis/dense_failure_stage1/logit_emergence/analysis_results.json` | Validates mechanical completion | confirmed |
| Correct delta/zero emergence is 0/3,999 | `analysis/dense_failure_stage1/logit_emergence/layerwise_correct_summary.csv`; `sample_emergence_layers.csv` | Shows intended correct answer emergence is absent | confirmed |
| Layer-27 top-1 is `<|im_end|>` for 54/54 checked correct records | `analysis/dense_failure_stage1/logit_emergence/validity_diagnostic.json` | Supports positional mismatch as the explanation | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Final-head strongest-token comparison at final literal user token | Correct emergence was 0/3,999 and mean layer-27 margin was -29.683 | supported: the state predicts the next chat delimiter, not the answer | `analysis/dense_failure_stage1/logit_emergence/validity_diagnostic.json` | Preserve the exact negative result; do not infer hidden-state uninformative or select a supervision cutoff | Do not retune delta or exclude structural tokens post hoc |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| First-token final-head logit lens | Directly specified by the plan and reuses saved states | Depth of answer preference | medium | completed; positionally invalid for correct strongest-token competition |
| Teacher-forced answer-sequence check | Could test whether a clear first-token pattern generalizes | Multi-token robustness | medium | rejected for this phase because primary pattern was not valid/clear |
| Actual assistant-start state extraction | Aligns the readout with first generated answer token | Whether answer preference truly emerges by depth | high | unchecked; requires separate authorization |

## Next-Step Decision
- Deliberation mode: standard after the surprising zero-correct-emergence result
- Active objective and bottleneck: the computation is complete, but the saved
  position does not support the intended correct answer-vocabulary comparison.
- Relevant memory item used: Phase 45 froze current LMMS outcomes and complete
  dense layer states for 7,999 samples.
- Confirmed observation: all 7,999 samples were scored, correct emergence is
  zero, and wrong delta emergence is 2,215/4,000 (50% coverage at layer 21;
  75% not reached).
- Unverified interpretation: the wrong-only GT-vs-generated first-token
  trajectories may still describe relative query-state answer evidence.
- Diagnosis: supported positional mismatch for the correct strongest-token
  comparator.
- Evidence path if diagnosis is not unknown:
  `analysis/dense_failure_stage1/logit_emergence/validity_diagnostic.json`.
- Viable alternatives considered: post-hoc special-token exclusion, optional
  sequence scoring, or later extraction at the actual assistant-start state.
- Chosen action: preserve and report the primary result unchanged, perform only
  one cheap top-token validity diagnostic, and stop without a follow-up.
- Strongest objection: one diagnostic shard does not estimate the full top-token
  distribution; however, the frozen token-position semantics and unanimous
  layer-27 result are sufficient to invalidate the intended comparator.
- How this differs from failed attempts: it does not reproduce historical
  routes, regenerate labels, or fit a model.
- Automatic execution authorized: yes
- Authorization basis: explicit request to read and perform
  `plans/dense_answer_logit_emergence_analysis_plan.md`.
- Stop condition: met; required outputs and validity interpretation are written,
  and no optional sequence analysis or training started.

## Latest Research-Action Result
- Action taken: applied the exact final norm/head to all 28 saved `text_final`
  states for the full 7,999-record population on four GPUs, then aggregated
  prospective delta/zero emergence and taxonomy metrics.
- Result: complete mechanical coverage, but zero correct emergence. Wrong
  taxonomy is 1,203 early-wrong, 425 progressive-wrong, 587 answer erosion,
  and 1,785 ambiguous; 594 ambiguous records are exact first-token collisions.
- Evidence saved: `analysis/dense_failure_stage1/logit_emergence/`.
- Failure or issue: the Phase-45 saved position precedes `<|im_end|>`, so the
  correct strongest-vocabulary competitor is structurally confounded.
- Lesson learned: do not equate final literal user-token and assistant-start
  readouts; this result does not show that early hidden states are uninformative.
- Next implication: no supervision start layer or training strategy is selected.
  The smallest defensible future action, only if separately authorized, is a
  prospective actual-assistant-start feature extraction smoke before any full
  rerun.
