# Stage B Discovery Protocol — Frozen Execution Version

Scientific source: `plans/dynamic_mllm_read_write_causal_analysis_plan_v2.md`

Status: complete. Candidate sampling, reference-likelihood metric, sparse layer
grid, technical-invalid rules, numerical thresholds, and strict behavioral
correctness rule were frozen before the corrected 400-record sweep. All records
completed with zero exclusions; results are under `outputs/stage_b/`.

## User-Approved Scope Exception

On 2026-08-04, the user explicitly approved 400 Stage B intervention samples,
expanding the source plan's suggested 100–200 discovery size. This does not
change the claim, stage order, metrics, validity gates, or prohibition on using
discovery as confirmatory prevalence evidence.

## Candidate Allocation

| Benchmark | Inherited easy | Inherited hard | Total |
|---|---:|---:|---:|
| GQA | 100 | 100 | 200 |
| TextVQA | 100 | 100 | 200 |
| Total | 200 | 200 | 400 |

- `complete_correct` is recorded as inherited easy; `complete_wrong` as
  inherited hard.
- Inherited labels are sampling metadata only. The pinned checkpoint must
  generate a fresh deterministic FULL prediction and score before analysis.
- Samples remain selected regardless of relabeling outcome; do not replace
  label flips to manufacture balanced pinned-model strata.
- All 24 requested Stage A records are excluded.
- Selection is SHA-256 hash-ranked with seed `20260804` and requires 400 unique
  effective image assets.
- Candidate manifest: `data_manifests/stage_b_discovery_candidates_400.jsonl`.
- Audit: `data_manifests/stage_b_discovery_candidates_400_audit.json`.

## Runtime-Domain Gate Before Interventions

Reprocess every candidate with the pinned processor and record the actual token
layout. The validated stock-eager Stage A domain currently ends at 4,861 prompt
tokens. A longer record is technically invalid for this sweep and must be
excluded with its exact reason; the fixed selected set is not replenished. The
other frozen technical-invalid rules are an empty normalized accepted set, an
empty answer-token span, or prompt-boundary-dependent answer tokenization. All
rules are evaluated before that sample's READ/WRITE outcomes. Do not use
rejected SDPA or query-chunked decoder runtimes as substitutes.

## Fixed Intervention Semantics

- Frozen Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Stock-eager decoder FULL path; vision encoder SDPA as validated in Stage A.
- Prompt/prefill-only single-layer interventions followed by the unchanged suffix.
- Four states: `IGNORE`, `READ_ONLY`, `WRITE_ONLY`, `FULL`.
- Primary READ and WRITE definitions remain exactly those in the source plan and
  Stage A implementation.

## User-Approved Reference-Likelihood Amendment

For Stage B only, the primary within-sample diagnostic is the sum of teacher-
forced log probabilities over normalized reference-answer tokens. Prompt and
image positions contribute nothing. The per-token mean is the cross-sample and
answer-length robustness diagnostic. No multiple-choice prompt or synthetic
distractor is created.

- GQA uses one canonical normalized reference consistently across all states.
- TextVQA applies the official EvalAI/VQA normalization procedure to every
  human answer, aggregates duplicate normalized strings by annotation
  frequency, and uses stable weighted log-sum-exp with weights summing to one.
  Each accepted-answer score remains in the raw result.
- The prompt template, answer prefix (none beyond the assistant-generation
  prefix), tokenizer, target answer, and target token span are identical across
  states.
- Interventions run during prompt prefill only. Teacher-forced answer tokens and
  greedy continuation use the resulting state-specific prompt cache through the
  unchanged model with no intervention hook active.
- Interpret signed effects only as reference-answer evidence shifts. Stage B
  cannot establish correct-over-alternative preference, harmful participation,
  or prevalence.

## Frozen Layer Grid

`[0, 4, 8, 12, 16, 20, 24, 27]`, fixed before READ/WRITE outcomes.

Strongest limitation: suffix depth differs across layers, especially layer 27;
layer ordering is exploratory and cannot be interpreted as intrinsic operator
harmfulness.

## Passed Validity Gate and Frozen Thresholds

- `outputs/stage_b_validity_v4/stage_b_validity_summary.json` records
  `gate_pass: true` for GQA and TextVQA.
- All 114 no-op comparisons had zero absolute sequence and mean score
  differences.
- Frozen before discovery interpretation:
  `epsilon_sequence = 1e-5`, `epsilon_mean = 1e-6`, selected as
  `max(predeclared floor, empirical absolute no-op p99)`.
- A behavioral branch is called correct only when its official dataset score is
  at least `1.0`; partial TextVQA consensus scores remain numeric but are not
  promoted to the categorical “correct” label.
