# Phase 47: Answer-Position Logit Emergence V2 Memory

## Current Objective
Redo the dense answer-logit emergence analysis at the actual final prompt token
that predicts the first assistant answer token, including teacher-forced first
divergence comparisons for wrong answers with shared prefixes.

## Active Constraints
- Use only the 7,999 current native-dense LMMS-labeled samples: 3,999 correct
  and 4,000 wrong across GQA, ChartQA, and TextVQA.
- Reconstruct the exact `add_generation_prompt=true` multimodal prompt and use
  its final non-padding token after the assistant prefix; never reuse Phase-46
  `text_final` trajectories or conclusions.
- Require exact layer-27 top-1 agreement with the stored first generated token
  on a 72-record dataset/outcome-stratified sanity set before the full run.
- Use strict zero-margin crossings sustained for three layers and an early
  cutoff of layer 4, all frozen before population results.
- For wrong shared-prefix cases, teacher-force the common prefix and compare at
  the first divergence. Log genuinely unusable comparisons instead of assigning
  ambiguity by construction.
- Use four direct GPUs. Stop after analysis; do not train Stage 1, resume W→C,
  or run four-action routing.

## Current State
- Done: Phase-46 failure and promoted positional lesson were read.
- Done: the exact prompt suffix was inspected without model execution; it ends
  `<|im_end|>`, newline, `<|im_start|>`, `assistant`, newline. The final newline
  is the candidate state whose logits predict the first answer token.
- Done: corrected implementation, frozen contract, position/collision sanity,
  full 7,999-record extraction, global aggregation, figures, and interpretation.
- Done: the required independent review ranked the true assistant-prefix
  position first but requested two pre-launch revisions: guaranteed
  teacher-forced shared-prefix cases in the sanity gate, and a complete tested
  aggregation path before GPU execution. Both revisions are implemented.
- Blocked: none. Phase stop condition reached.
- Most recent useful observation: a representative exact prompt has length 667;
  the assistant-prefix newline is position 666 while the last literal user
  token precedes `<|im_end|>` at position 661.
- Most recent execution observation: contract `5703b57d...b38ab4de` passed the
  base answer-start gate 72/72 but failed the appended-full-prefill collision
  gate at 9/13. The full run was forbidden and did not start.
- Follow-up focused diagnostic: cached replay under contract
  `1e912c26...ce3548d` exactly reproduced all 13 stored prefix-plus-next-token
  sequences. The repeated 9/13 number was instead raw LM-head top-1 versus the
  generated token after the frozen `repetition_penalty=1.05`; these are
  different score spaces. The final gate now validates processed sequence
  replay and raw-readout/raw-logit parity separately.
- Third gate observation: raw parity passed 13/13 and processed continuation
  replay passed 12/13, with at least four exact cases per dataset. The sole
  unreplayable TextVQA continuation crossed GPU ranks. Per the prospective
  collision rule, such comparisons are logged and excluded, not assigned to
  the ambiguous taxonomy and not allowed to invalidate the answer-start gate.
- First full attempt under `f223f56a...73089a3` was intentionally stopped after
  3,455 successes and one failure. A near-tie GQA sample exposed that applying
  final norm/head after detaching the layer-27 hook can round differently from
  the model's own raw output. Layer 27 is now sourced from native
  `output.logits`; no partial result was aggregated.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase-46 position predicts chat structure | `analysis/dense_failure_stage1/logit_emergence/validity_diagnostic.json` | Prevents reuse of invalid trajectories | confirmed |
| Exact dense prompt includes the assistant generation prefix | `dense_failure_stage1/runtime.py`; processor inspection | Locates the candidate answer-prediction position | confirmed pre-execution |
| Current labels and generated IDs cover 7,999 records | `analysis/dense_failure_stage1/current_dense_8k/dense_outputs.jsonl` | Defines targets and sanity reference | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Phase-46 final-user-token logit lens | 0/3,999 correct emergence; delimiter top-1 at layer 27 | supported positional mismatch | Phase-46 report/diagnostic | Validate layer-27 generated-token parity before any full answer-position analysis | Do not reinterpret or copy Phase-46 curves |
| V2 attempt-01 appended-prefix full prefill | Base 72/72; shared-prefix 9/13 | supported replay-contract mismatch; exact numerical mechanism unresolved | `analysis/dense_failure_stage1/answer_logit_emergence_v2/failed_sanity_attempt_01/` | Replay the native cached greedy path and capture the exact cached call predicting divergence | Do not treat an appended multimodal full prefill as identical to cached generation |
| V2 attempt-02 raw-top1/processed-token gate | Cached prefix plus next token 13/13, but raw top-1 equals generated token only 9/13 | supported score-space mismatch from frozen repetition penalty | `analysis/dense_failure_stage1/answer_logit_emergence_v2/failed_sanity_attempt_02/`; model `generation_config.json` | Gate raw readout against raw model logits and processed replay against stored tokens | Do not require raw logits to reproduce a repetition-penalized continuation token |
| V2 attempt-03 all-replays gate | Base 72/72, raw parity 13/13, stored continuation 12/13 | supported single-sample cross-rank continuation sensitivity | `analysis/dense_failure_stage1/answer_logit_emergence_v2/failed_sanity_attempt_03/` | Apply the frozen per-sample unusable-comparison skip rule after requiring four exact cases per dataset | Do not abort population analysis for one logged unusable collision |
| V2 full attempt-01 detached layer-27 reconstruction | 3,455 successes; `gqa:gqa_ge_10447544` raw top-1 mismatch | supported BF16 reconstruction difference at a near tie | `analysis/dense_failure_stage1/answer_logit_emergence_v2/failed_full_attempt_01/` | Use the model's native raw `output.logits` as authoritative layer 27 | Do not use a separately reconstructed layer-27 vector when exact native logits exist |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Actual assistant-prefix final-token prefill | It is the position used by greedy generation for token one | Correct layer-wise answer evidence | high | selected; sanity-gated |
| Exclude structural tokens at Phase-46 position | Would change the comparator without fixing causal position | Only a post-hoc surrogate | low | rejected |
| Learned correctness probe | Could decode information without answer alignment | Stage-1 prediction, not answer emergence | high | out of scope |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: establish a position-valid answer-logit
  trajectory before using depth evidence to design Stage-1 supervision.
- Relevant memory item used: Phase 46 directly showed final-user-token and
  answer-start positions are not interchangeable.
- Confirmed observation: the assistant generation prompt ends in an assistant-
  prefix newline after the user message delimiter.
- Unverified interpretation: the final prefix position will reproduce stored
  token-one generation across the transferred full population.
- Diagnosis: supported positional mismatch in Phase 46.
- Evidence path if diagnosis is not unknown:
  `analysis/dense_failure_stage1/logit_emergence/validity_diagnostic.json`.
- Viable alternatives considered: actual answer position, post-hoc structural-
  token exclusion, and a learned probe.
- Chosen action: exact answer-position prefill plus a strict 72-record parity
  gate, with collision prefixes replayed through the native cached generation
  path, followed only on success by the authorized full analysis.
- Strongest objection: canonical GT tokens can differ from evaluator-valid
  generated tokens, and collision teacher forcing changes the prompt length;
  both will be audited explicitly and never repaired using observed logits.
- Independent review: `revise` (high confidence). Reconciled by requiring four
  shared-prefix wrong examples per dataset in the sanity set, exact layer-27
  next-token parity after teacher forcing, canonical-GT mismatch counts, and
  completion of the full aggregation/report path before contract freeze.
- How this differs from failed attempts: it reconstructs the complete assistant
  generation prefix and validates token-one parity before population scoring.
- Automatic execution authorized: yes, one corrected analysis only.
- Authorization basis: explicit user instruction in the current turn.
- Stop condition: required v2 tables, figures, trajectory records, and report
  exist, or the position sanity gate fails. No training or routing follows.

## Latest Research-Action Result
- Action taken: extracted and analyzed raw answer-token trajectories from the
  true assistant answer-start position and cached first-divergence positions
  under frozen contract `c9a6d6302779c0dfa07886397106bd23d1d7dbbf323ee9008906b0b524cbdcce`.
- Result: sanity passed 72/72; four GPUs completed 7,999/7,999 with zero
  execution failures. Correct GT first reaches raw top-1 at median layer 26
  (IQR 26–27); 94.95% are top-1 at layer 27. The requested persistent rule is
  defined for 851/3,999 correct records and is right-censored. Wrong raw margins
  remain close to zero with low target ranks through the early/middle stack,
  then separate materially at layers 23–27. The literal fixed crossover median
  is layer 2 but is not a defensible semantic formation depth.
- Evidence saved:
  `analysis/dense_failure_stage1/answer_logit_emergence_v2/analysis_summary.md`,
  `analysis_results.json`, the requested CSV/JSONL outputs, and four figures.
  Failed gates and the interrupted partial full attempt are preserved in named
  subdirectories under the same root; no partial aggregate was promoted.
- Failure or issue: 7/594 shared-prefix wrong comparisons did not replay the
  transferred processed continuation on their assigned GPU and are excluded,
  not classified ambiguous. Canonical GT and generated first tokens differ for
  189 evaluator-correct samples, chiefly TextVQA, and remain explicitly audited.
- Lesson learned: answer-token formation is late under the fixed final-head
  lens; early zero-threshold sign persistence is not equivalent to semantic
  answer presence. Native raw output must be used at layer 27, and generated
  tokens selected after repetition penalty must not be equated with raw top-1.
- Next implication: the next separately authorized decision may compare Stage-1
  supervision strategies using layers 25–27/26–27 as evidence-backed candidate
  regions. Do not train or select the strategy in this phase.
