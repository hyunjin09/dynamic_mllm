# v2 to v3 Migration Audit

Date: 2026-08-06  
Active plan: `plans/dynamic_mllm_read_write_policy_conditional_plan_v3.md`  
Plan SHA-256: `8612e1b22d76dcfcc8a5f63780493d9d29e1f9c0288d986ee21047890c129605`

## Scope and integrity

This was a static and deterministic artifact audit. It did not load the model,
run an intervention, collect a new sample, open a new held-out endpoint, or
train a router, controller, policy, probe, or model. Plan v3 supersedes v2; the
v2 source plan, results, reports, checksums, and checksum-backed archive were
not deleted or overwritten.

The archived v2 snapshot remains
`archives/stage_b_stage_c_frozen_outcome_b_v1/stage_b_stage_c_artifacts_v1.tar.gz`
with SHA-256
`65ac4d07fcf348a55475266479d3786dcbb9e8914711faa988677fcb7515df92`.

## v3 formal mapping

| v2 state | v3 cell | Executed meaning |
|---|---|---|
| `IGNORE` | `Q_l(0,0)` | READ off, WRITE off |
| `READ_ONLY` | `Q_l(1,0)` | READ on, WRITE off |
| `WRITE_ONLY` | `Q_l(0,1)` | READ off, WRITE on |
| `FULL` | `Q_l(1,1)` | READ on, WRITE on |

Static inspection of `interventions/four_state.py`,
`interventions/prompt_cache.py`, and
`experiments/stage_b_reference_likelihood.py` confirms the required branch:

```text
captured FULL pre-layer state and cloned FULL prefix cache
-> exactly one mapped action at layer l
-> unchanged decoder layers l+1...L and branch-specific dense cache
-> identical accepted-reference scoring and deterministic generation
```

The intervention hook is active only for prompt/prefill construction and is
removed before teacher-forced answer tokens and greedy decoding. Thus the
existing Stage B cell scores have the policy-conditional dense-suffix meaning
required by v3, rather than the value of a future adaptive policy.

## Component audit

| Component | Evidence | Classification under v3 |
|---|---|---|
| Architecture/token causal graph | `outputs/stage_a/architecture_causal_graph.md`, `token_layout.json` | Reusable within the validated runtime domain. The causal mask and zero future attention establish the query-invariance premise structurally. |
| FULL no-op parity | `outputs/stage_a/no_op_parity.csv`, Stage B validity v4 | Reusable; layer/logit/score/generation parity passed. |
| READ intervention | `interventions/read_path.py`, `outputs/stage_a/read_reconstruction.csv` | Reusable. It subtracts the fixed-softmax visual-value path only on nonvisual query rows. The BF16 OFF state uses an explicitly bounded half-ULP adjustment and exact representable add-back. |
| WRITE intervention | `interventions/four_state.py`, `outputs/stage_a/write_reconstruction.csv` | Reusable. WRITE OFF restores pre-layer visual rows while preserving current-layer nonvisual output; reconstruction is exact within tolerance. |
| Four-action determinism | `outputs/stage_a/four_state_stability.csv`, Stage B validity v4 | Reusable; repeated branches are deterministic and start from identical cloned prestates. |
| Activation plausibility | `outputs/stage_a/activation_plausibility.csv` | Reusable. Norm/RMS, cosine, PCA-residual, and nearest-natural distances are recorded. |
| Reference-answer utility | `scoring/reference_likelihood.py`, `scoring/benchmark_metrics.py` | Reusable. GQA canonical and TextVQA frequency-weighted accepted answers retain the validated normalization and answer-only scoring. |
| Stage B four-action values | `outputs/stage_b/stage_b_results_v1.jsonl` | Reusable as discovery-only v3 `Q` values. |
| Stage B conditional effects | Per-layer `sequence_effects` and `mean_effects` | Retain only as derived contrasts; every stored value recomputes from the corresponding four cells with maximum absolute error `0.0`. |
| Stage C primary result | `outputs/stage_c/stage_c_results_v1.jsonl` | Discovery-only under v3. It contains only `FULL` and `WRITE_ONLY` at layer 0 and is not a complete v3 `Q` vector. |
| Stage C structured nulls | `outputs/stage_c/nulls/`, `analysis_v1/` | Engineering/calibration evidence only. They are endpoint-specific and did not receive the complete v3 layer/action search budget. |
| Previously selected Stage C cases | Frozen 800-record manifest and inspected outcomes | Invalid for v3 held-out confirmation or mechanistic replication; preserve without reuse for those roles. |
| Superseded/partial attempts | Explicitly isolated Stage A/B/C attempt directories | Preserve for provenance; invalid as primary analytical inputs. |

## Verified Stage B completeness

- 400 records: 200 GQA and 200 TextVQA.
- Exact layer grid: `[0,4,8,12,16,20,24,27]`.
- 3,200 sample-layer pairs and 12,800 action cells.
- Every sample-layer pair contains all four actions.
- All sequence and per-token scores are finite.
- FULL-to-baseline maximum absolute score difference is `0.0` for both score
  definitions.
- Maximum cached-prestate injection difference is `0.0`.
- Maximum recorded READ and WRITE reconstruction identity differences are
  `5.960464477539063e-08` and `2.9802322387695312e-08` respectively.
- Deterministic greedy outputs and official correctness fields are retained.

This is sufficient to reconstruct both sequence-score and per-token-score
vectors
`[Q(0,0), Q(1,0), Q(0,1), Q(1,1)]` without another model run.

## Preserved v2 interpretation

The v2 Stage C decision remains Outcome B: the TextVQA layer-0
reference-support effect replicated, but actual READ removal did not outperform
either frozen structured residual null. It did not establish a confirmed
answer-misaligned READ effect. All 800 outcomes are inspected, so the manifest
and selected cases are discovery evidence only under v3. The old Stage D stays
closed and cannot be reframed as a v3 mechanism study.

## Missing v3 items

The following are missing but do not invalidate the existing four-cell values:

1. A versioned v3 discovery table containing each `Q` vector, all three
   FULL-relative suppression advantages, `G_l`, the four conditional effects,
   and interaction.
2. Per-cell output KL, final text-state distance, READ/WRITE residual norm, and
   runtime/memory fields. These were not serialized in v2 Stage B.
3. A numerical same-image/different-question visual-state and WRITE-invariance
   sanity check. The causal premise is established structurally, but the v3
   sanity artifact is absent.
4. Exploratory same-image question pairs; v2 Stage B deliberately used 400
   unique effective images.
5. A preliminary structured-null pipeline whose layer/action/control search
   budget exactly matches the eventual real v3 statistic.
6. A v3 confirmatory manifest freezing `L*`, `epsilon`, the sample-level
   statistic, null hierarchy/replicates, robustness controls, same-image rules,
   and success/pivot criteria.
7. A new held-out set with no overlap with any inspected v2 Stage B or Stage C
   record/image. No such set is frozen yet.

## Candidate decision comparison

| Candidate | What it resolves | Cost | Main weakness |
|---|---|---:|---|
| `REUSE_AND_CONFIRM` | Preserves valid Stage A and complete Stage B `Q` cells, then closes only the v3-specific preparation gaps before a new confirmation. | Lowest | Must not be mistaken for immediate Stage C entry. |
| `RERUN_DISCOVERY` | Recollects every missing diagnostic and same-image pair. | High | Repeats 12,800 already valid action cells and violates the bounded-migration intent. |
| `REPAIR_STAGE_A` | Adds the numerical query-invariance sanity test. | Low | No parity/reconstruction/hook failure exists; this overstates one missing v3 artifact as a causal repair. |
| `STOP_AND_REDESIGN` | Responds to fundamental intervention invalidity. | Highest | Direct parity, reconstruction, and dense-suffix evidence rule out its premise. |

The strongest objection to reuse is that missing state/residual diagnostics and
the unmatched v2 null search could change which v3 endpoint is defensible.
That objection blocks immediate confirmation, not reuse of the observed `Q`
cells. The plan's explicit migration rule permits reconstructing the four-cell
vectors from inspected v2 data.

## Minimum next action and experiment

Do not run a new experiment first. The next bounded action should be a
deterministic, versioned reanalysis of the existing 400 Stage B records into
the v3 `Q`, FULL-relative advantage, `G_l`, conditional-effect, and interaction
schema. Treat the v2 Stage C 800 records only as discovery risk evidence, not
as replication.

If that reanalysis identifies a prospectively defensible `L*` and statistic,
the minimum new experiment is a small discovery-only preflight, not a large
sweep: verify same-image visual-WRITE invariance and exercise every selected
real/null action under the identical frozen search budget while collecting the
missing residual/state diagnostics. Only after that preflight passes may a new
nonoverlapping held-out manifest and confirmatory protocol be frozen.

## Decision

REUSE_AND_CONFIRM

