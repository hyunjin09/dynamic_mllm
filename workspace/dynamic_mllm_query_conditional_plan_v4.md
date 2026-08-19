# Analysis-First Research Plan v4
## Query-Conditional Value of Structurally Identical Visual READ/WRITE

**Status:** active strategic plan; planning complete, execution not yet
authorized.  This plan closes the v3 harmfulness-confirmation direction and
does not reopen its failed structured-null gates.

**Primary model:** frozen `Qwen/Qwen2.5-VL-7B-Instruct`, revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, stock-eager decoder in BF16.

## 1. Scientific question and bounded claim

The primary question is:

> Is the policy-conditional value of visual READ/WRITE query-dependent even
> when the underlying visual computation is structurally and numerically
> identical for different questions about the same image?

The strongest claim that v4 may establish is:

> Under the pinned prefix-causal MLLM and common-padded execution, a fixed
> same-image visual state and WRITE can have different downstream four-action
> value patterns for different questions, so one image-only action at a layer
> does not recover the question-conditioned oracle on the studied question
> distribution.

The broader wording “in prefix-causal MLLMs” requires replication beyond one
model. GQA-only discovery or confirmation supports only the pinned-model/GQA
version of the claim.

This is not a harmfulness claim. A negative action contrast is not called
harmful, and a positive oracle gap is not an acceleration result.

## 2. Fixed scope and non-goals

- Use the validated four actions at one layer at a time:
  `IGNORE=(0,0)`, `READ_ONLY=(1,0)`, `WRITE_ONLY=(0,1)`, and `FULL=(1,1)`.
- Every branch starts from the same dense pre-layer state, changes exactly one
  layer, and executes an unchanged dense suffix.
- Use the validated answer-token-only reference-answer likelihood. Per-token
  accepted-reference log-likelihood is primary; sequence likelihood is
  secondary.
- Use the frozen nonterminal sparse layer grid
  `[0,4,8,12,16,20,24]`. Layer 27 is excluded because WRITE is structurally
  silent there in the validated local intervention.
- Use common right-padding within every same-image prompt group.
- Treat the image as the independent statistical unit.
- Do not train or deploy a router, controller, policy, probe, or other model.
- Do not fine-tune the base MLLM, run multi-layer policies, or alter decoding.
- Do not run v3 confirmation, weaken its null gates, enter v3 Stage D, or seek
  a READ-specific harmful mechanism.
- Do not use oracle action counts as FLOPs, latency, or deployable-policy
  evidence.
- Do not use inspected v2 Stage B, v2 Stage C, or v3 discovery outcomes as
  held-out v4 evidence.

## 3. Formal object

For image `I`, question `q`, and layer `l`, define

\[
Q_{I,q,l}(a),\qquad
a\in\{IGNORE,READ\_ONLY,WRITE\_ONLY,FULL\},
\]

using the fixed dense prefix, one local action, dense suffix, and identical
question-specific accepted-reference scoring. Define the FULL-relative action
pattern

\[
V_{I,q,l}(a)=Q_{I,q,l}(a)-Q_{I,q,l}(FULL).
\]

Raw `Q` values are saved, but cross-question analyses use `V`, conditional
contrasts, or within-query regret. This removes the question-specific baseline
score that otherwise makes raw likelihood levels incomparable across different
answers.

The conditional contrasts are

\[
\Delta_R^{w=0}=Q(1,0)-Q(0,0),\quad
\Delta_R^{w=1}=Q(1,1)-Q(0,1),
\]

\[
\Delta_W^{r=0}=Q(0,1)-Q(0,0),\quad
\Delta_W^{r=1}=Q(1,1)-Q(1,0).
\]

## 4. Architectural identity premise

### 4.1 Deterministic causal argument

The pinned prompt order is:

```text
system/control prefix
-> vision-start
-> contiguous visual rows
-> vision-end
-> question/instruction
-> assistant prefix
-> right padding, when required for group shape equality
```

For two questions `q1` and `q2` about the same image, construct the prompts
independently, then right-pad both to the maximum prompt length in their image
group. Non-padding token content is unchanged; padding is masked from attention
and scoring.

Let `H_l^q[V]` be the visual rows entering decoder layer `l`.

1. At `l=0`, the same image, identical preceding control prefix, identical
   visual positions, and common execution shape give identical `H_0[V]`.
2. For a visual query row `i`, the causal mask assigns zero weight to every
   later question or padding key `j>i`. All accessible prefix and visual rows
   are identical across the questions.
3. The visual-row Q/K/V projections, multimodal RoPE coordinates, attention
   output, residual additions, RMSNorm inputs, and row-wise MLP output are
   therefore identical.
4. By induction, `H_l^{q1}[V]=H_l^{q2}[V]` for every decoder layer, and
   \(\Delta W_l=H_{l+1}[V]-H_l[V]\) is identical.
5. Same-layer text READ consumes visual K/V projected from pre-WRITE `H_l`, so
   each question can value the same visual state differently without changing
   that state or WRITE.

The exact computational graph and hook semantics remain those in
`outputs/stage_a/architecture_causal_graph.md`, `interventions/read_path.py`,
and `interventions/four_state.py`.

### 4.2 Why common right-padding is mandatory

Unequal prompt shapes produced BF16 kernel-shape numerical divergence even
though future attention mass was zero. The existing one-image diagnostic found
visual WRITE differences as large as `10.5` by layer 24. Common right-padding
restored bitwise equality (`max_abs=0`) at every frozen layer. This is a
numerical-execution control, not a change to the prompt or causal estimand.

### 4.3 v4 numerical entry gate

Before any v4 terminal action value is inspected, run an outcome-blind check on
12 prospective GQA images spanning the prompt-length and visual-token-count
range, with at least two questions per image. At all seven frozen layers require:

- identical tensor shapes and visual-token positions within each image group;
- unchanged decoded non-padding prompt text;
- padding masked from attention and answer scoring;
- exact equality of pre-layer visual rows, post-layer visual rows, and WRITE;
- zero visual-query attention to later question/padding tokens;
- exact instrumented-FULL parity within the Stage A tolerance;
- correct, nonempty answer spans, no answer leakage, and deterministic scoring.

Any failure stops before scientific outcomes are aggregated. The failure must
be recorded as architecture, layout, instrumentation, or finite-precision
invalidity; it may not be repaired by silently changing the v4 estimand.

## 5. Stage V4-B — minimum discovery experiment

### 5.1 Dataset and sample

The first discovery uses GQA only because it has dense same-image question
groups plus official question types, semantic programs, object annotations,
and `equivalent`/`entailed` metadata. TextVQA is reserved for later replication
rather than adding a weaker semantic-control path to the minimum discriminator.

Freeze, outcome-blind, exactly:

- 120 unique GQA images;
- exactly two primary natural questions per image;
- at least two technically valid questions for every selected image;
- no image or record overlap with v2 Stage B, v2 Stage C, v3 discovery, v3
  calibration, or any future v4 held-out pool;
- deterministic selection and question-pair rules, versioned manifest, and
  SHA-256 checksum before intervention scores are opened.

Candidate data may come from the outcome-blind GQA Stage C2 reserve or the
additional local GQA instruction data under `datasets/datasets`, after complete
overlap and technical-validity audits. Metadata inspection is allowed for
selection; action values, model correctness, and likelihood outcomes are not.

The core sweep has `120 images x 2 questions x 7 layers x 4 actions = 6,720`
branch scores. The 30-image paraphrase arm adds 840 scores, for 7,560 total if
its prospective metadata gate passes. This is the single minimum scientific
experiment.

### 5.2 Semantic-control construction

Before outcomes are opened, stratify the selected image pairs using only
official metadata:

- **Different-evidence pairs:** require nonempty, disjoint GQA semantic-program
  object-ID sets that resolve to distinct scene-graph objects/boxes. Target 60
  images.
- **Matched comparison pairs:** select the remaining 60 images while balancing
  structural/semantic question type, program depth, question length, and
  reference-answer token length.
- **Paraphrase control:** on 30 selected images, add one verified paraphrase of
  one primary question. Prefer a resolvable official `equivalent` question with
  the same image, accepted answer, semantic program target, and question type.
  If official equivalents are insufficient, stop the paraphrase arm for an
  explicit prospective amendment; do not generate or select paraphrases after
  action outcomes are visible.

Different-evidence classification is `unresolved`, not guessed, when object
links are absent or ambiguous. Such records may remain in the core random
discovery set but may not enter the semantic-control comparison.

### 5.3 Noise and ties

Before scientific scoring, freeze

\[
\epsilon=\max(10^{-6},q_{0.99}(|\Delta_{identity}|))
\]

from repeated FULL, hook-enabled FULL, and save/reinsert identity controls under
the common-padded layout. The inherited `0.05` nats/token band is reported as a
secondary practical threshold. An effect may be called query-dependent when it
clears the numerical gate, but “substantially different” is reserved for robust
summaries that also clear `0.05` nats/token.

For each question/layer, retain the epsilon-best set

\[
B_{I,q,l}=\{a:\max_b Q_{I,q,l}(b)-Q_{I,q,l}(a)\le\epsilon\}.
\]

This set-valued rule prevents deterministic tie-breaking from creating false
query disagreement. Exact argmax with the inherited deterministic tie order is
saved only as a secondary audit field.

## 6. Primary discovery quantities

All quantities are computed at each frozen layer, separately by semantic
stratum, and with a prespecified joint summary that averages layer-level values
without maximizing over layers.

### 6.1 Within-image best-action disagreement

For a question pair `(q1,q2)`, robust disagreement is

\[
D_{I,l}=1[B_{I,q1,l}\cap B_{I,q2,l}=\varnothing].
\]

Also report exact-argmax disagreement and the fraction made ambiguous by the
epsilon rule.

### 6.2 Within-image sign reversal

Map each of the four conditional READ/WRITE contrasts to
`positive`, `silent`, or `negative` using `epsilon`. A reversal requires one
question to be strictly positive and the other strictly negative for the same
image, layer, and conditional contrast. Silent-to-signed changes are reported
separately and are not called reversals.

### 6.3 Within-image four-action variance

Use the FULL-relative vector `V`. The primary image/layer dispersion is

\[
H_{I,l}=\frac{1}{4}\sum_a
\operatorname{Var}_{q\in I}[V_{I,q,l}(a)].
\]

Also report the Euclidean distance between paired `V` vectors. Raw-Q variance
is secondary because different reference answers have different baseline
likelihoods.

### 6.4 Cross-query action-transfer regret

For source question `s` and target question `t`, use a conservative tie rule:

\[
R_{s\to t,l}=
\max_a Q_{I,t,l}(a)-\max_{a\in B_{I,s,l}}Q_{I,t,l}(a).
\]

Average both directions within each image. Positive regret means that even the
best transferable member of the source question's epsilon-best set loses value
on the target question.

### 6.5 Image-only versus image+query oracle gap

Using FULL-relative values, define

\[
G^{query}_{I,l}=\frac{1}{|Q_I|}\sum_q\max_a V_{I,q,l}(a)
-\max_a\frac{1}{|Q_I|}\sum_qV_{I,q,l}(a).
\]

This compares a per-question oracle with the best single action for the image
at that layer. It is a descriptive insufficiency measure, not a learned policy,
latency estimate, or acceleration claim.

## 7. Statistical analysis and semantic controls

- Resample and split by image, never by question.
- Report mean, median, 5% and 20% trimmed means, quantiles, and image-bootstrap
  confidence intervals for dispersion, regret, and oracle gap.
- Report layerwise results and the prespecified equal-layer average. Do not pick
  a layer from the largest observed mean and call it confirmatory.
- Compare different-evidence pairs with paraphrase pairs on `V`-vector distance,
  transfer regret, and best-action disagreement.
- A valid semantic pattern is smaller action-pattern distance and regret for
  paraphrases than for different-evidence questions.
- Match or adjust prospectively for GQA structural/semantic type, program
  depth, question length, accepted-answer token length, and answer format.
  FULL likelihood or correctness may be reported descriptively after the core
  analysis but may not determine inclusion or matching.
- Repeat summaries after excluding epsilon ties and after winsorization only as
  sensitivity analyses; never replace the complete image-level primary set.

Sign reversal alone does not establish a semantic mechanism. A stable
paraphrase/different-evidence ordering is required before making a semantic
interpretation.

## 8. Discovery decision gate

Proceed to a held-out v4 confirmation design only if all conditions hold:

1. the common-padding architectural and numerical gate passes;
2. four-action results are complete and FULL parity remains valid;
3. best-action disagreement, transfer regret, and the query-oracle gap persist
   after epsilon ties, medians/trimmed means, and image-level uncertainty;
4. at least one prespecified robust image-level summary of transfer regret or
   query-oracle gap clears the `0.05` nats/token practical band; otherwise the
   result may be query-dependent but does not support the intended
   “substantially different” claim;
5. the result is not dominated by a small number of images or one layer;
6. matched difficulty and answer-format adjustment does not collapse it;
7. paraphrase action patterns are more stable than different-evidence pairs;
8. an image-disjoint, outcome-uninspected confirmation pool remains.

If the values are mainly numerical ties, paraphrases are as unstable as
different-evidence pairs, covariate adjustment removes the effect, or the
common-padding identity fails, stop the v4 direction. Do not search a different
layer grid, action definition, task, or metric to rescue it.

## 9. Prospective held-out confirmation

This stage is designed but not authorized.

- Use new GQA images disjoint from discovery and every inspected v2/v3 set.
- Determine the exact image count prospectively from the discovery image-level
  variance of the equal-layer oracle gap and transfer regret, using robust and
  heavy-tail-aware precision calculations. Target 300--500 images; never resize
  from held-out results.
- Freeze exactly two primary natural questions per image and the same common
  padding, layer grid, actions, dense suffix, scoring, epsilon, and tie rule.
- Freeze one primary image-level endpoint before opening outcomes: the
  equal-layer average of `G_query`. Transfer regret, robust disagreement, and
  sign reversals are secondary/coherence endpoints.
- Give the identity/no-op noise process the same four-action and seven-layer
  opportunity as the real statistic. Confirmation requires the real gap to
  exceed this tie/search-matched numerical control and have an image-bootstrap
  confidence interval above zero.
- Also require nonzero median/trimmed support, no extreme-image dominance, and
  the frozen paraphrase-versus-different-evidence ordering.

The failed v3 residual-null families are not silently repurposed. They were
required to distinguish suppression from generic perturbations for a
harmfulness claim. V4 instead compares exact validated actions across questions
while holding the image computation fixed. Numerical/search controls and
semantic controls remain mandatory, but passing v4 would not rehabilitate the
v3 harmfulness conclusion.

TextVQA remains an optional independent task replication after GQA discovery,
subject to a separate outcome-blind audit proving enough accepted-answer/OCR
grounding and paraphrase controls. It cannot be substituted post hoc if GQA
fails.

## 10. Optional pre-action diagnostic probe

Only after held-out query dependence passes may a separate plan propose a
low-capacity offline diagnostic. No probe is authorized by this plan.

Use image-grouped train/validation/test splits and matched model capacity to
compare:

1. image-only summaries;
2. question-only summaries;
3. additive image plus question summaries;
4. an interaction-aware image-question summary.

Targets may be the four `V` values, epsilon-best action set, or transfer regret.
Features must exist before the layer action. This study tests pre-action
predictability only; it does not execute actions, establish a router, or support
deployment/acceleration claims.

## 11. Reusable evidence and protected artifacts

Reusable without rerunning:

- validated READ/WRITE implementation:
  `interventions/read_path.py`, `interventions/four_state.py`, and
  `interventions/prompt_cache.py`;
- Stage A graph, token layout, parity, deterministic four-state behavior, and
  exact READ/WRITE reconstruction artifacts under `outputs/stage_a/`;
- four-action reference-likelihood scoring and accepted-answer handling from
  Stage B;
- common-padding causal proof and numerical diagnostic in
  `workspace/v3_query_invariance_validation.md`,
  `workspace/v3_stage_c2_fixed_padding_amendment.md`, and
  `outputs/v3_preflight/query_invariance_equal_length_diagnostic.json`;
- outcome-blind prospective multi-question GQA and TextVQA metadata pools in
  `outputs/v3_preflight/stage_c2_reserved_pool_audit.json`;
- local GQA instruction metadata under `datasets/datasets`;
- inspected Stage B four-action results as background motivation only;
- v2 Outcome B and v3 null-redesign failure as negative-result and claim-boundary
  evidence.

All existing v2/v3 artifacts and checksums remain unchanged. Once an image is
used in v4 discovery, semantic-control design, or any outcome inspection, it is
permanently ineligible for v4 held-out confirmation.

## 12. Required v4 artifacts when execution is authorized

The minimum discovery action must create:

- a versioned, checksum-backed GQA image/question manifest;
- a common-padding identity/preflight report;
- complete per-question/layer four-action `Q` and `V` records;
- image-level disagreement, reversal, variance, transfer-regret, and oracle-gap
  tables;
- semantic-control and covariate-sensitivity reports;
- a decision report ending in either proceed-to-confirmation-design or stop.

No discovery experiment is executed by this planning document.
