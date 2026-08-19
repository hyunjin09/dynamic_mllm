# Frozen GQA Query-Conditioned Visual Refinement Discovery

## Result in one sentence

The frozen replay operator was technically exact and target-question access
produced a small positive raw mean at layer 4, but target specificity did not
survive the frozen robust criteria at any anchor; effects were near zero in
medians/trimmed means, inconsistent across layers, and behaviorally nonpositive.

## Integrity and execution

- Pinned model: Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16, frozen weights.
- Data: 100 new GQA validation images, exactly two questions per image, 200
  questions total.
- Manifest SHA-256:
  `af5ad2c498cd8ae7064274dd94f7d566559d949af7c5e63c96503a1d38afac70`.
- Overlap: zero images against v2 Stage A, v2/v3 Stage B discovery, v3 geometry
  calibration, and v4 query discovery.
- Anchors: `[4, 12, 20]`, frozen before outcomes.
- Completed: 200/200 records and 600/600 sample-layer matrices; no replacement
  or outcome-dependent exclusion.
- Four result-shard checksums match their completion records.

The 12-image entry preflight passed. Same-image visual states, native visual
outputs, unconditioned replay, suffix logits, and repeated scores were exact at
all anchors. Conditioned replay remained numerically close to native geometry
(minimum cosine `0.99725`). No answer or padding token entered replay.

The first parallel GPU attempts on `node03` failed before model/sample loading
because their Slurm CUDA environment exposed zero usable devices. They wrote no
scientific records. All 200 records were subsequently executed unchanged on
the validated `node02` environment.

## Frozen operator and contrasts

The output-boundary replay runs one native frozen decoder layer from captured
`H_l`, permits visual queries to access either the target or paired-other
literal-question token span, retains only visual post-layer rows, inserts them
into native target `H_(l+1)`, and resumes the unchanged suffix at `l+1`.

Primary per-token likelihood contrasts were:

- `Delta_condition = TARGET_QUERY_REPLAY - UNCONDITIONED_REPLAY`;
- `Delta_target = TARGET_QUERY_REPLAY - OTHER_QUERY_REPLAY`.

The replay variants have identical dense compute and shapes. Unconditioned
replay is an exact native reconstruction, so `UNCONDITIONED_REPLAY - BASELINE`
is exactly zero. This isolates the added question-to-visual edges, but does not
test a second sequential depth step.

## Primary likelihood results

| Layer | Contrast | Mean | Image-clustered 95% CI | Median | 20% trimmed mean | Positive fraction | Fraction >= 0.05 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | target - unconditioned | 0.01918 | [0.00062, 0.04182] | 0.000021 | 0.00212 | 0.540 | 0.165 |
| 4 | target - other question | 0.01096 | [-0.00048, 0.02400] | 0.000018 | 0.00240 | 0.530 | 0.120 |
| 12 | target - unconditioned | -0.01200 | [-0.02654, 0.00349] | -0.000107 | -0.00348 | 0.415 | 0.130 |
| 12 | target - other question | -0.02352 | [-0.03982, -0.00820] | -0.000119 | -0.00510 | 0.460 | 0.105 |
| 20 | target - unconditioned | 0.00255 | [-0.00944, 0.01414] | 0.000035 | 0.00279 | 0.575 | 0.160 |
| 20 | target - other question | -0.00361 | [-0.01531, 0.00788] | 0.000020 | -0.00011 | 0.535 | 0.100 |

No anchor passed either frozen robust contrast rule, and no anchor passed the
joint discovery rule. In particular:

- Layer 4's conditioning mean CI was above zero, but its mean was below the
  frozen `0.05` practical threshold, the positive fraction was below `0.55`,
  and its target-versus-other CI crossed zero.
- Removing the largest 5% of absolute layer-4 conditioning effects reduced the
  mean from `0.01918` to `0.00350`; the 20%-trimmed mean was `0.00212`.
- Layer 12 moved in the wrong direction under both robust summaries, including
  a target-versus-other CI entirely below zero.
- Layer 20 was near zero and uncertain for both contrasts.

The most defensible interpretation is heavy-tailed, layer-dependent sensitivity
to the new attention edges—not robust target-question-specific refinement.

## Generated behavior

Against baseline/unconditioned replay, target replay produced the following
correction/regression counts:

| Layer | Wrong -> correct | Correct -> wrong | Net |
|---:|---:|---:|---:|
| 4 | 1 | 2 | -1 |
| 12 | 1 | 2 | -1 |
| 20 | 0 | 1 | -1 |

Against paired-other-question replay, the corresponding counts were `0/2` at
layer 4, `0/1` at layer 12, and `1/0` at layer 20. These sparse secondary
changes do not support a behavioral improvement and do not rescue the primary
failure.

## Same-image pair dependence and metadata controls

Absolute within-image differences in `Delta_condition` remained sizable in a
descriptive sense (`0.0851`, `0.0864`, and `0.0689` mean at layers 4, 12, and
20). They did not align with semantic evidence separation: Spearman
correlations with scene-object-set distance were `-0.068`, `0.040`, and
`0.034`. Question-length-difference correlations were also small (`0.020`,
`0.113`, `0.196`).

Frozen GQA structural/semantic/detailed categories showed heterogeneous raw
means, but directions changed across anchors and most category medians stayed
near zero. These are post-aggregation descriptive strata, not licensed
endpoints; no category was used to rescue or retarget the experiment. The full
table is in
`outputs/query_refinement/analysis_v1/question_type_summary.json`.

Sample-level correlations with question length, answer length, baseline
difficulty, prompt length, and visual-token count were all modest in magnitude
(absolute Spearman at most about `0.16` for the primary contrasts), providing
no simple format/difficulty explanation and no positive semantic control.

## Interpretation boundary

Supported:

- The frozen model admits a numerically stable post-question visual-query edge
  intervention without new visual evidence or learned parameters.
- Local likelihood responses remain heterogeneous and heavy-tailed.
- At layer 4, target conditioning has a small positive raw mean relative to the
  exact unconditioned reconstruction.

Unsupported:

- robust target-question-specific visual refinement value;
- benefit beyond the wrong same-image question under the frozen criteria;
- correctness improvement;
- routing, acceleration, efficiency, harmfulness, or a semantic mechanism.

The output-boundary operator's exact unconditioned identity is also a scope
limitation: this experiment cleanly tests added question access but cannot rule
on generic sequential extra visual depth. Searching deeper replay, another
layer, or a selected GQA subgroup is explicitly outside the frozen task and is
not justified by these results.

Evidence paths:

- `outputs/query_refinement/query_refinement_q_v1.parquet`
- `outputs/query_refinement/conditioning_contrasts_v1.csv`
- `outputs/query_refinement/analysis_v1/layer_summaries.json`
- `outputs/query_refinement/analysis_v1/behavior_counts.json`
- `outputs/query_refinement/analysis_v1/question_pair_dependence.json`
- `outputs/query_refinement/analysis_manifest.json`
