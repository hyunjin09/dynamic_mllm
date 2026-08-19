# Phase 02: Stage B Discovery Memory

## Current Objective

Complete and report a 400-sample exploratory factorial discovery study on GQA
and TextVQA without using discovery as confirmatory evidence.

## Active Constraints

- Stage A passed only on the stock-eager decoder for prompts up to 4,861 tokens.
- The user approved 400 samples: 100 per GQA/TextVQA × inherited easy/hard cell.
- Inherited easy/hard labels are sampling metadata, not pinned-revision outcomes.
- Stage A samples are excluded and source images must be unique within discovery.
- User-approved Stage B amendment: reference-answer sequence likelihood is the
  primary within-sample diagnostic and per-token mean likelihood is used for
  aggregation. No distractors, multiple-choice conversion, or router training.
- Exact READ/WRITE semantics remain fixed; Stage C/D are not authorized.

## Current State

- Done: Validity, the fixed 400-record/eight-layer sweep, sample-level bootstrap
  analyses, plots, implementation report, and Stage B conclusion.
- In progress: Final documentation and handoff only.
- Blocked: Stage C entry requires user approval of a core metric/control
  amendment; Stage C execution is not authorized.
- Most recent useful observation: The strongest negative candidate is TextVQA
  layer-0 `read_w1`; early WRITE is strongly positive, and GQA layer-27 READ is
  a separate correctness-stratified tentative pattern rather than replication.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Candidate manifest has exactly 100 records per inherited cell and 400 unique effective image assets. | `data_manifests/stage_b_discovery_candidates_400_audit.json` | Implements the user-approved allocation without Stage A overlap. | confirmed |
| Inherited labels drift under the pinned model. | `outputs/stage_a/benchmark_scoring_reproduction.csv` | Requires fresh FULL relabeling before stratified analysis. | confirmed |
| User explicitly approved reference likelihood and prohibited distractors. | Current Stage B task; `workspace/stage_b_protocol.md` | Resolves the metric/data mismatch with a narrower exploratory claim. | confirmed |
| Stock-eager decoder substitutes failed suffix equivalence. | `workspace/decision_log.md` | Stage B must retain the Stage A runtime domain. | confirmed |
| Corrected Stage B reference-scoring and pinned-greedy validity passed all seven checks with zero exclusions. | `outputs/stage_b_validity_v4/stage_b_validity_summary.json` | Authorizes the fixed 400-record discovery sweep. | confirmed |
| Simplified TextVQA normalization was not protocol-conformant; official EvalAI-style normalization was installed and retested before discovery. | `scoring/benchmark_metrics.py`; `outputs/stage_b_validity_v2` | Freezes the accepted-answer estimand before inspecting discovery outcomes. | confirmed |
| All 400 fixed candidates completed with zero exclusions and exact FULL parity. | `outputs/stage_b/analysis_v1/analysis_manifest.json` | Establishes complete Stage B execution without selection leakage. | confirmed |
| TextVQA layer-0 conditional READ is the strongest negative discovery candidate; early WRITE is robustly positive. | `outputs/stage_b/analysis_v1/layer_signed_effects.csv`; `reports/stage_b_conclusion.md` | Selects the smallest plausible Stage C target while preserving exploratory terminology. | confirmed as discovery only |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Initial unique-asset audit | 400 selected IDs reported only 384 source assets because null asset IDs collapsed in the audit. | supported | First generated audit; regression test `test_uses_local_image_path_when_source_asset_id_is_missing` | Persist the effective asset grouping key and fall back to local image path. | Counting null source IDs as one asset or using sample ID before image path. |
| Initial TextVQA scoring implementation | Simplified normalization omitted official number, contraction, and punctuation behavior. | supported | `scoring/benchmark_metrics.py`; superseded `outputs/stage_b_validity/` | Accepted-answer normalization is part of the estimand and must be frozen before outcomes. | Generic punctuation normalization for TextVQA. |
| First partial full sweep | Cached greedy omitted pinned repetition penalty `1.05`; 109 secondary behavioral records were superseded. | supported | `outputs/stage_b_attempt_01_raw_greedy/`; `outputs/stage_b_validity_v4/` | Match the full pinned generation processor, not raw argmax alone. | Using the superseded attempt in any analysis. |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Run two-dataset cached-prompt validity/noise probe | Required before the 400-sample sweep and directly tests the new scoring path. | FULL parity, target alignment, cache semantics, epsilon. | medium | completed; passed |
| Execute fixed 400-sample sweep | Validity, metric, manifest, thresholds, and layer grid are frozen. | Discovery outcomes. | high | completed |
| Approval-gated reference-likelihood Stage C draft | Continues the strongest TextVQA candidate without widening operation/layer search. | Held-out test design. | high | recommended proposal; cannot freeze without approval |
| Preserve original MC Stage C | Preserves source-plan P0 option controls. | Original confirmatory estimand. | very high | infeasible under current open-ended/no-distractor direction without approval |
| Stop without any Stage C draft | Avoids plan conflict. | Authorization ambiguity. | low | valid but underdelivers requested proposal |

## Next-Step Decision

- Deliberation mode: deep with one independent reviewer
- Active objective and bottleneck: Hand off complete Stage B evidence and expose
  the single authorization decision blocking a valid Stage C freeze.
- Relevant memory item used: Stage A requires the faithful stock-eager runtime;
  user requires interventions only on the prompt, with answer tokens excluded
  from the intervention hook.
- Confirmed observation: All 400 records completed; TextVQA layer-0 `read_w1`
  is negative in sequence and mean aggregate CIs, while WRITE is strongly
  positive and no shared negative cross-dataset band appears.
- Unverified interpretation: A reference-likelihood Stage C can replace the
  source plan's MC margin/permutation endpoint without weakening the P0 gate.
- Diagnosis: supported protocol conflict, not an implementation failure.
- Viable alternatives considered: Approval-gated complete Stage C draft; stop
  pending direction; restore original MC Stage C with option-bearing data.
- Chosen action: Deliver the complete reference-likelihood Stage C draft but
  leave it explicitly unfrozen and unexecuted pending user approval.
- Strongest objection: The current task asks for a proposed frozen Stage C
  protocol and might be read as implicit amendment authority. The source-plan
  P0 controls are nevertheless irreconcilable without an explicit metric/control
  decision, so silently freezing would be invalid.
- How this differs from failed attempts: Answer tokens are processed only after
  the intervention hook is removed, using the state-specific prompt cache.
- Automatic execution authorized: no further research action
- Authorization basis: Stage B execution is complete; Stage C is explicitly not
  authorized and its plan conflict requires user input.
- Stop condition: Stop after Stage B handoff until the user approves or declines
  the Stage C reference-likelihood amendment.

## Latest Research-Action Result

- Action taken: Completed the corrected 400-record Stage B sweep and interpreted
  it with sample-level bootstrap plus one conditional independent review.
- Result: Zero exclusions; strongest negative candidate is TextVQA layer-0
  conditional READ; WRITE is predominantly answer-aligned; no harmfulness or
  confirmatory prevalence claim is supported.
- Evidence saved: `outputs/stage_b/`, `outputs/stage_b/analysis_v1/`,
  `reports/stage_b_reference_likelihood_implementation.md`, and
  `reports/stage_b_conclusion.md`.
- Failure or issue: The Stage C reference-likelihood continuation cannot retain
  the unchanged source plan's option/margin/permutation P0 controls.
- Lesson learned: Numerical epsilon is a noise floor, not by itself a practical
  effect threshold; near-zero medians and heavy tails require a separate frozen
  magnitude rule in confirmation.
- Next implication: User decides whether to approve the approval-gated draft in
  `workspace/stage_c_reference_likelihood_proposal.md`; do not execute Stage C.
