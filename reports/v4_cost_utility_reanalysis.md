# v4 Query-Conditioned versus Image-Only Cost–Utility Reanalysis

## Scope and integrity

This is a deterministic reanalysis of the completed, inspected GQA discovery
only. It uses all 1,680 complete question-layer four-action Q matrices (120
images, two questions, seven layers) and performs no model inference, sample
collection, answer generation, training, or paraphrase experiment. The frozen
Q-matrix and discovery-manifest checksums were verified before analysis.

The raw prospective grid has 14 compute penalties and 45 exact local-budget
levels. Epsilon ties use the frozen `1e-6` nats/token tolerance and then choose
the lowest-cost action, ordered `IGNORE`, `READ_ONLY`, `WRITE_ONLY`, `FULL`.
Aggregate Pareto curves use linear interpolation between nondominated raw grid
points on a 1,001-point normalized-compute grid. Matched-utility comparisons
are restricted to the image-only frontier's attainable utility range.

## Exact local operation cost

For hidden width `d=3584`, MLP width `m=18944`, visual rows `V`, accessible
prefix rows `P`, and common-padded post-visual query rows `T`, matrix multiply
FLOPs count one multiply-add as two operations:

\[
C_R=2T V d,
\]

\[
C_W=2Vd^2+4d\left(VP+\frac{V(V+1)}{2}\right)+2Vd^2+6Vdm.
\]

`READ` counts the text-query-to-visual value accumulation. Text-to-visual QK
logits remain action-invariant because the validated READ-OFF intervention
preserves the original fixed softmax weights. `WRITE` counts visual-row Q,
causal attention QK/AV, output projection, and gated FFN matrix multiplies.
Visual K/V projection, text/nonvisual computation, and text-to-visual QK are
action-invariant. RMSNorm, softmax, and SiLU scalar operations are excluded
because there is no implementation-independent FLOP convention for them.

| Action | Local visual cost, mean GFLOPs | Median GFLOPs | Increment vs IGNORE, mean | Increment vs FULL, mean |
|---|---:|---:|---:|---:|
| IGNORE | 0 | 0 | 0 | -123.4557 |
| READ_ONLY | 0.05146 | 0.04490 | +0.05146 | -123.4043 |
| WRITE_ONLY | 123.4043 | 107.7925 | +123.4043 | -0.05146 |
| FULL | 123.4557 | 107.8310 | +123.4557 | 0 |

The mean action-invariant local computation is 21.5831 GFLOPs, including
visual K/V, nonvisual linear/attention work, and preserved text-to-visual QK.
READ averages only `0.0419%` of FULL's action-dependent visual cost; WRITE
therefore dominates this local cost model.

These are semantic operation-level costs for a hypothetical row/edge-sparse
implementation. The validated counterfactual runner executes the dense FULL
layer for every branch, so its measured runtime is identical across actions.
Nothing here is a wall-clock or acceleration result.

## Oracle construction

At each image-layer pair, the image-only oracle chooses one action shared by
both questions. The image+query oracle may choose one action per question.
Both use the same costs, grids, epsilon, and tie rule. Exact local-budget rows
enumerate all four shared actions or all 16 question-action pairs subject to
the pair's hard average budget. The aggregate frontier additionally permits
compute allocation across image-layer pairs through its nondominated pooled
curve; this distinction matters because WRITE is nearly all-or-nothing in the
local accounting.

## Unconstrained comparison and conservative-FULL hypothesis

| Quantity | Image-only | Image+query |
|---|---:|---:|
| Mean utility relative to FULL | 0.10547 | 0.11984 |
| Median utility relative to FULL | 0.02181 | 0.03738 |
| 20% trimmed utility relative to FULL | 0.04372 | 0.05769 |
| Mean local visual compute (GFLOPs) | 61.5239 | 63.1171 |
| Mean normalized compute | 0.4964 | 0.5226 |
| IGNORE selection | 25.48% | 25.30% |
| READ_ONLY selection | 24.88% | 22.44% |
| WRITE_ONLY selection | 25.60% | 24.70% |
| FULL selection | 24.05% | 27.56% |

The query-conditioned utility increment is `0.01438` mean, `0.00017` median,
and `0.00173` after 20% trimming. It is not purchased by a cheaper
unconstrained policy: image+query uses 1.5933 GFLOPs more on average, or 2.62%
of pair-specific FULL cost. On the 568 robust-disagreement pairs, it uses 4.31%
more FULL-equivalent compute and selects FULL 27.38% of the time versus 22.01%
for image-only. The hypothesis that image-only matches query-specific utility
mainly by conservatively over-computing is therefore rejected.

## Cost–utility frontiers

Primary pooled mean-utility results are:

| Scope | Integrated utility gap | Maximum utility gain at matched compute | Mean compute saving at matched utility | Utility targets saving at least 10% FULL |
|---|---:|---:|---:|---:|
| All 840 image-layer pairs | 0.01486 | 0.02370 | 1.40% FULL | 3.40% |
| 568 robust non-tie/disagreement pairs | 0.02090 | 0.02298 | 1.47% FULL | 4.10% |

At no normalized-compute point does the all-pair or robust-non-tie mean
utility gain reach the prespecified practical reference of `0.05`
nats/token. The pointwise maximum matched-utility saving is large (38.44% FULL;
40.34% for robust non-ties), but it occurs on only 3.40% (4.10%) of the
attainable utility-target grid. It is an isolated frontier geometry result,
not a sustained saving.

The aggregation sensitivities do not show a stable practical utility gain:

| Scope / aggregation | Integrated gap | Maximum matched-compute gain | Mean matched-utility saving | Targets saving at least 10% FULL |
|---|---:|---:|---:|---:|
| All / mean | 0.01486 | 0.02370 | 1.40% | 3.40% |
| All / median | 0.01447 | 0.01674 | 7.36% | 26.97% |
| All / 20% trimmed mean | 0.01313 | 0.01444 | 3.30% | 9.39% |
| Robust / mean | 0.02090 | 0.02298 | 1.47% | 4.10% |
| Robust / median | 0.02402 | 0.02730 | 6.69% | 25.87% |
| Robust / 20% trimmed mean | 0.01896 | 0.02036 | 3.58% | 12.59% |

The median cost-saving coverage is the strongest objection to calling the
curves nearly identical. However, the all-pair image-only median frontier spans
only `-0.00005` to `0.02181` nats/token, and its maximum matched-compute gap is
`0.01674`. Thus the percentage saving occurs over a narrow, sub-practical
utility range and is not corroborated by the mean or trimmed result.

Exact per-pair budgets expose the discreteness of WRITE. At a 0.50 local FULL
budget, image-only cannot choose a shared WRITE-bearing action and remains at
`-0.04616` mean utility relative to FULL; image+query can assign a costly
action to one question and reaches `0.08865`. The mean gap is `0.13481`, but
the median and 20%-trimmed gaps are `0.02432` and `0.02601`. Once compute can
be distributed across image-layer pairs on the pooled frontier, the maximum
matched-compute mean gain falls to `0.02370`. The hard-budget result is real
for that local constraint, but it does not establish a broad cost advantage.

Query-specific action pairs strictly expand the four-point shared-action
frontier for 92.14% of image-layer pairs and strictly dominate the
epsilon-tied unconstrained shared action for 40.71%. These frequent local
differences establish oracle flexibility, but their aggregate utility scale is
small.

## Layer and predefined-covariate robustness

| Layer | Integrated mean gap | Maximum matched-compute mean gain | Mean matched-utility saving |
|---:|---:|---:|---:|
| 0 | 0.04223 | 0.12743 | 2.50% |
| 4 | 0.01143 | 0.01529 | 2.85% |
| 8 | 0.01961 | 0.02124 | 2.21% |
| 12 | 0.01742 | 0.02138 | 3.34% |
| 16 | 0.01146 | 0.01464 | 2.45% |
| 20 | 0.00962 | 0.01354 | 2.46% |
| 24 | 0.00682 | 0.00960 | 3.07% |

Layer 0 is the only mean frontier with a practical pointwise excursion, over
18.38% of its compute axis. It is heavy-tail sensitive: its median and
20%-trimmed maximum gaps are only `0.01884` and `0.02203`. The other six
layers never approach `0.05` on the mean frontier.

The prospectively frozen descriptive controls do not reveal a stable hidden
regime. Mean maximum matched-compute gains are `0.02344`, `0.02879`, and
`0.05105` for short, medium, and long questions; the long-question threshold
is crossed on only 0.20% of its compute grid. They are approximately `0.02544`
for both answer-length groups and `0.02776`, `0.03134`, and `0.03613` across
easy, medium, and hard FULL-difficulty tertiles. These are descriptive
discovery strata, not selected endpoints.

## Interpretation and decision

The reanalysis supports frequent query-associated oracle action differences,
including exact local-budget advantages caused by the nearly binary WRITE
cost. It does not support the narrower practical proposition posed here:
query conditioning does not provide a sustained material pooled
compute-allocation advantage at matched utility. It also does not support the
proposed explanation that an image-only oracle matches utility by selecting a
more expensive conservative action.

The strongest counterevidence is the median matched-utility saving coverage,
the layer-0 raw mean excursion, and widespread pairwise frontier expansion.
Against that, all-pair and robust matched-compute gains remain below 0.05 under
mean, median, and trimmed aggregation; mean/trimmed matched-utility savings are
small; layer 0 collapses under robust aggregation; and the unconstrained
query-conditioned oracle uses more, not less, compute. An independent
read-only challenge agreed with closure, with medium confidence, while
requiring the median result to remain explicit.

Accordingly, the planned paraphrase experiment is no longer scientifically
justified as a route to a meaningful compute-allocation result. Preserve the
query-dependence finding as inspected discovery evidence and close the dynamic
policy direction for this protocol. This makes no claim about routing,
latency, acceleration, harmfulness, or other models/tasks.

Artifact checksums and full raw/interpolated grids are recorded in
`outputs/v4_discovery/cost_utility_frontier_summary_v1.json`.

STOP_DYNAMIC_POLICY_DIRECTION
