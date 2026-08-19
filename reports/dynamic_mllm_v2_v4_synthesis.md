# Dynamic MLLM v2–v4 Evidence Synthesis and Strategic Redesign

## Scope and decision boundary

This synthesis uses only preserved v2–v4 reports and artifacts. No experiment,
model inference, route search, null retuning, or training was performed. The
closed decisions remain unchanged:

- v2 Stage C: `Outcome B`;
- v3: `STOP_V3_CONFIRMATION`;
- v4 local policy: `STOP_DYNAMIC_POLICY_DIRECTION`.

The recommendation below is a proposed strategic pivot, not an active plan or
execution authorization. It does not reopen harmfulness, local layer skipping,
or router training.

## 1. Evidence chain

### v2 — Local causal decomposition of READ and WRITE

V2 first established a technically valid intervention: every branch began
from the same cached dense pre-layer state, changed READ and/or WRITE at one
layer, and continued through the unchanged dense suffix. FULL parity,
determinism, token alignment, and READ/WRITE reconstruction passed.

Stage B then measured all four states at eight layers on 200 GQA and 200
TextVQA records using accepted-reference likelihood and greedy behavior. The
main surviving descriptive finding was functional asymmetry. Early layer-0
WRITE was strongly answer-aligned on average in both datasets, whereas READ
was heterogeneous and the strongest negative candidate was TextVQA layer 0
with WRITE enabled.

Stage C froze that one READ contrast on 800 held-out TextVQA images. The mean
reference-support effect replicated (`-0.07294` nats/token, clustered 95% CI
`[-0.14128,-0.01710]`), but the median and trimmed means were approximately
zero, the `Answer:` prefix condition crossed zero, and the real intervention
did not outperform either frozen structured residual-null family. The frozen
decision was Outcome B. The descriptive +10 net greedy-correct count and
positive reference-versus-original-wrong-answer margin did not override the
failed causal-specificity conjunction.

What survived v2 is therefore intervention validity, early-WRITE alignment as
discovery evidence, and one held-out but nonspecific reference-support
contrast. Harmful or answer-misaligned READ did not survive.

### v3 — Complete four-action value and causal specificity

V3 reframed each sample/layer as the full policy-conditional value vector
`[Q(IGNORE), Q(READ_ONLY), Q(WRITE_ONLY), Q(FULL)]`. All 3,200 preserved Stage
B matrices were complete and valid. Their descriptive landscape was
heterogeneous: the sample-layer oracle gained `0.0976` nats/token over FULL,
while fixed per-layer and per-dataset/layer schedules gained only `0.0045` and
`0.0075`. Conditional sign reversals, READ–WRITE interaction, and failures of
independent main-effect action recovery were common, although medians were
small and practical near-ties frequent.

V3 did not fail because a held-out maximum-over-21 statistic was negative. It
stopped outcome-blind because a defensible specificity test could not be
constructed. After 4,000 independent calibration images, the paired donor
null required a global caliper of `3.09375`, and all candidate joint covariance
representations failed either native-shape coverage or the frozen `0.50`
out-of-sample fidelity gate. Weakening those gates would have made the null
easier only after its failure was known. The broader causal suppression
hypothesis is thus technically unresolved, while the planned confirmation
protocol is invalid and closed.

### v4 — Same-image query dependence and practical local-policy headroom

V4 asked a distinct descriptive question. Under common right-padding, two
questions about the same image had bitwise-identical visual states and visual
WRITE at all seven layers; causal masking prevented those visual rows from
seeing the later question. On 120 new GQA images with two questions each,
epsilon-aware action disagreement was frequent (`0.6762`) and bidirectional
action-transfer regret was robust (`0.1121` mean, `0.0634` median). Thus
different questions valued the same query-blind visual computation
differently.

That variation did not translate into a strong local routing opportunity. The
image+query versus image-only oracle gap was only `0.0144` mean and `0.00344`
median. The exact-FLOP frontier found a maximum pooled matched-compute gain of
`0.02370` nats/token and only `1.40%` mean matched-utility FULL-cost saving.
The unconstrained query-conditioned oracle used `2.62%` more compute, rejecting
the explanation that an image-only policy matched utility mainly by
over-computing. Semantic ordering was also mixed, and the prospective
paraphrase gate was unavailable. V4 therefore supported query-associated
action-value variation but not a materially useful query-conditioned local
policy.

## 2. Prior-hypothesis classification

The status labels below apply to the exact studied model, data, metric, and
protocol. “Supported discovery” is descriptive, not confirmatory causality.

| Prior hypothesis | Status | Evidence boundary |
|---|---|---|
| The one-layer four-state intervention has the intended causal graph and dense suffix. | **Supported** | FULL parity, reconstruction, determinism, and token/scoring checks passed. |
| Early visual WRITE is broadly answer-misaligned. | **Unsupported** | Layer-0 WRITE had large positive reference-support effects in both tasks. |
| Early visual WRITE is answer-aligned on average. | **Supported** | Strong two-task discovery result; not a cross-model or behavioral theorem. |
| TextVQA layer-0 READ with WRITE on reduces accepted-reference support. | **Partially supported** | Mean discovery and held-out effects were negative, but median/trimmed/prefix robustness was weak. |
| That TextVQA READ contribution is a confirmed answer-misaligned or harmful mechanism. | **Unsupported** | The real intervention failed both frozen structured-null comparisons. |
| A shared negative READ/WRITE layer band exists across GQA and TextVQA. | **Unsupported** | READ was heterogeneous and WRITE was generally positive early. |
| Greedy wrong-to-correct flips identify a causal correction. | **Unsupported** | Improvements and regressions were sparse/mixed and did not establish mechanism or accuracy gain. |
| Four-action values are heterogeneous across samples, tasks, and layers. | **Supported** | Complete v3 matrices, robust positive trimmed gains, and diverse best actions; discovery only. |
| A fixed global, layer, or dataset-layer schedule explains that landscape. | **Unsupported** | Fixed schedules recovered little of the inspected sample-layer oracle gain. |
| READ and WRITE combine as independent main effects. | **Unsupported** | Sign reversals and independent-action recovery failures were common, with heavy-tailed interaction. |
| The v3 maximum-over-21 suppression statistic is causally specific. | **Technically unresolved** | No valid search-budget-matched structured-null hierarchy could be frozen. |
| Same-image visual state and WRITE are query-invariant in the prefix-causal layout. | **Supported** | Formal mask argument plus bitwise equality under common padding. |
| Local action ranking varies with the question despite identical visual state/WRITE. | **Supported** | Frequent tie-aware disagreement and transfer regret on new GQA discovery images. |
| That variation reflects question semantics rather than format/difficulty. | **Partially supported** | It survives several matched sensitivities, but different-evidence ordering was mixed and paraphrase stability was untested. |
| One image-only local action is substantially insufficient. | **Unsupported** | Direct query-oracle and cost-frontier gaps were small under robust aggregation. |
| A difficult router could unlock large existing local-action headroom. | **Unsupported** | Even the outcome-aware query oracle had little sustained practical advantage. |
| Multi-layer combinations would create useful headroom. | **Technically unresolved** | They were not tested; single-layer data neither prove nor refute them. |
| Reference likelihood alone faithfully measures practical policy utility. | **Partially supported** | Sequence/per-token rankings agree, but prompt sensitivity, heavy tails, null failures, and sparse greedy coherence limit it. |

## 3. Robust observations that survive

| Observation | Safe status | Can motivate a new paper direction? |
|---|---|---|
| Early visual WRITE is strongly answer-aligned. | Robust discovery across GQA/TextVQA; causal universality unproven. | **Yes.** Preserve early global visual consolidation rather than treating all visual compute as skippable. |
| READ/WRITE functional asymmetry. | Strongly supported. WRITE dominates local FLOPs and early reference support; READ is cheap and heterogeneous. | **Yes.** Motivate typed visual-memory operations instead of a single depth gate. |
| Heterogeneous four-action value landscape. | Supported descriptive observation with small medians and heavy tails. | **Yes, cautiously.** It motivates studying missing capabilities, not another oracle router. |
| Sample/query-dependent action ranking. | Supported on inspected data; semantic cause only partial. | **Yes.** It motivates giving the question a causal path into visual computation. |
| Structured-null non-specificity. | Directly supported for the v2 narrow READ endpoint; broader v3 specificity remains unresolved because nulls were invalid. | **Yes, methodologically.** New claims need matched content/compute controls and behavioral coherence. |
| Small practical query-conditioned cost–utility headroom. | Supported for the v4 GQA local action space. | **Yes, negatively.** It rules out better routing over the same actions and motivates changing the action space. |
| Heavy-tailed local intervention effects. | Repeatedly supported across v2–v4. | **Yes.** Make medians, trimmed means, clustered inference, and behavior necessary rather than optional. |

The strongest coherent paper premise is therefore not “some visual layers are
harmful” or “a router can skip them.” It is: **a prefix-causal model performs
important early, query-blind visual consolidation; questions value that state
differently, but selecting among local keep/drop operations has little robust
headroom. The missing capability is query-conditioned visual computation, not
a better selector over the existing action set.**

## 4. Why local dynamic routing failed

The explanations are ranked by existing evidential support; they are not
mutually exclusive.

| Rank | Explanation | Support | Diagnosis |
|---:|---|---|---|
| 1 | **B. The action space has insufficient practical separation.** | Strong | The v4 outcome-aware query oracle, matched-compute frontier, median, and trimmed summaries all show small pooled gains. This directly bounds any router over the same four local actions. |
| 2 | **D. Existing visual WRITE is structurally query-blind.** | Strong architectural fact; causal role suspected | Common-padding proof and numerics show that the question cannot alter visual WRITE. The old policy can only retain/remove the same image-conditioned update; it cannot create query-specific visual evidence. |
| 3 | **E. Reference likelihood overstates local utility variation.** | Moderate, indirect | Mean effects are heavy-tailed and prompt-sensitive, structured nulls absorb the v2 endpoint, and greedy changes are much smaller/mixed. A direct alternative-utility comparison was not performed, so “overstates” remains an interpretation. |
| 4 | **C. Single-layer interventions miss multi-layer interactions.** | Technically unresolved | V3 found within-layer READ–WRITE interaction, but no multi-layer factorial evidence exists. It could increase or cancel headroom. Testing it would remain close to the closed skipping action family. |
| 5 | **A. The router is merely hard while meaningful headroom exists.** | Weak/unsupported | The oracle had access to outcomes and still showed little sustained practical frontier gain. Prediction difficulty cannot explain away a small oracle upper bound. |

Multi-layer interaction is retained as an unresolved scientific limitation, not
promoted to the next project. A new sweep over layer bundles would be another
minor local-suppression variant and would not address the query-blind visual
state or objective-specificity problems.

## 5. Genuinely new candidate directions

### Direction 1 — Post-question visual-token refinement/replay

**Scientific hypothesis.** Replaying a fixed budget of question-relevant,
already encoded visual tokens after the question—so those tokens can causally
integrate question context before answer decoding—will produce a robust,
content-specific gain over geometry- and compute-matched query-blind token
replays.

**Why it follows from the negative results.** V4 proves that current visual
WRITE is query-blind and that choosing keep/drop states is insufficient. This
direction adds the missing causal edge `question -> visual state/update`
without adding new pixels, training a router, or revisiting failed residual
nulls.

**Novelty relative to skipping/routing.** It constructs a new post-question
visual computation. The action is not “run or skip the existing layer”; it is
“contextualize a bounded visual memory with the question.”

**Smallest falsification experiment.** Use a new, image-disjoint set of 100 GQA
images with exactly two questions linked to distinct scene-graph boxes. Keep
the base model frozen. Deterministically pool each target box to a fixed `4x4`
window of 16 already encoded visual tokens and replay that window after the
literal question and before the answer prefix. Do not pass box labels, object
names, or answers to the model. Use identical token count, positions, and
suffix for:

1. own-question target window;
2. the paired other-question target window;
3. geometry-matched non-target window;
4. deterministic random window;
5. uniform whole-image 16-token summary;
6. no-replay baseline (secondary, not compute-matched).

The primary within-image cross-over statistic is

\[
D_I=\tfrac12[(S(q_1,r_1)-S(q_1,r_2))+(S(q_2,r_2)-S(q_2,r_1))],
\]

with image-clustered inference. Accepted-reference likelihood remains a
continuous diagnostic, but success also requires robust median/trimmed support,
target advantage over all matched replay controls, no extreme-image dominance,
and coherent official greedy-correctness changes. This is an oracle-region
capacity test, not a learned selector or deployment result.

**Likely reviewer objection.** Scene-graph boxes are privileged supervision,
and nonstandard token replay may create prompt-format effects. The response is
to interpret success only as existence of capacity headroom, balance the exact
same token windows across the cross-over, and require all matched controls.

**Expected compute cost.** Low: approximately 1,200 frozen teacher-forced and
greedy branches, no second vision encoding, no optimization, and less than one
fifth of the completed v4 core branch count.

**Kill criterion.** Stop this direction if the clustered cross-over CI includes
zero, the 20%-trimmed effect fails the predeclared practical band, target replay
does not beat matched non-target/random/whole-image replay, or likelihood gains
lack correctness coherence. Do not train a selector or adapter after failure.

### Direction 2 — Sparse query-triggered high-resolution visual revisitation

**Scientific hypothesis.** Some questions fail because the prefix visual
encoding lacks localized detail; selectively re-encoding a question-relevant
region at higher resolution will outperform equal-area non-target revisits at
the same extra vision cost.

**Why it follows.** Strong early WRITE suggests the global visual pass is
valuable, while heavy tails suggest a minority of questions may need more
evidence rather than less computation.

**Novelty.** This is active spatial evidence acquisition, not decoder depth
routing. It changes what visual evidence is computed.

**Smallest falsification experiment.** With a frozen base model and oracle GQA
target boxes, append one fixed-resolution target crop after the question and
compare against equal-area non-target, random, and full-image-resized controls.
Use the same cross-over and behavioral coherence rules as Direction 1.

**Likely reviewer objection.** A positive result may reflect privileged crops
or simply more pixels, not query-conditioned reasoning. Equal-cost controls
reduce but do not remove that concern.

**Expected compute cost.** Low-to-medium because every condition requires an
additional vision encoding.

**Kill criterion.** Stop if the target crop does not beat all equal-compute
controls robustly, or if gains are confined to likelihood tails without
correctness coherence.

### Direction 3 — Explicit query-writable visual memory

**Scientific hypothesis.** A small typed memory with separate visual read and
query-conditioned write interfaces will improve evidence use more reliably
than suppressing dense residual paths.

**Why it follows.** READ/WRITE asymmetry is robust, but the current WRITE is
image-only. An explicit memory would preserve the useful early global write
while allowing bounded question-dependent refinement.

**Novelty.** The unit of computation is a persistent cross-modal memory state,
not a decoder layer or skip bit. It can expose interpretable read/write budgets
and multi-turn persistence.

**Smallest falsification experiment.** Freeze the base MLLM and train only a
small low-rank cross-attention memory adapter on image-disjoint GQA training
data. Compare to parameter- and compute-matched query-blind memory, text-only
adapter, and extra-MLP controls.

**Likely reviewer objection.** Any gain could come from extra parameters or
supervision rather than memory semantics, and v2/v3 do not yet justify this
larger method.

**Expected compute cost.** Medium-high: adapter optimization, multiple matched
controls, and held-out evaluation.

**Kill criterion.** Stop if the query-writable adapter does not beat all
capacity-matched controls, or if a positive likelihood effect lacks robust
behavioral and grounding coherence.

## 6. Comparison with the closed local-routing direction

| New direction | New capability absent from four actions | Why v4 does not falsify it | Risk of collapsing into generic dynamic-depth routing |
|---|---|---|---|
| Post-question token refinement/replay | A causal question-to-visual update using a fixed visual-token budget. | V4 only retained/removed query-blind local READ/WRITE; it never created a question-conditioned visual state. | Low if replay is fixed-budget and always executed. High only if later converted prematurely into a trigger/router. |
| High-resolution visual revisitation | Acquisition of new localized pixels after the question identifies the needed evidence. | V4 acted on the existing prefix representation and could not recover discarded spatial detail. | Low; the choice is spatial evidence acquisition, not language-layer depth. |
| Explicit visual memory | A persistent, typed state with separately controlled semantic read and query-conditioned write. | No v4 action altered the memory representation or permitted question-conditioned write-back. | Medium; it must retain explicit memory semantics and matched fixed-depth controls to avoid becoming a generic adapter/depth policy. |

## 7. Recommendation and internal challenge

Recommend **Direction 1: post-question visual-token refinement/replay**. It is
the only candidate that directly tests the missing architectural capability
with a frozen model, fixed compute, no new pixels, and no learned router. It
also fails cheaply: a negative cross-over result would justify stopping before
adapter training or a new architecture.

The strongest objection is that oracle target boxes make the intervention
privileged and the replay layout is not the model's native training
distribution. A preliminary proposal to re-encode crops was revised after an
independent read-only challenge: reusing existing visual tokens avoids
conflating question conditioning with extra pixels or a second vision pass.
The oracle remains only an upper-bound capacity instrument. A positive result
would justify a later, separately approved learnability study; it would not
itself establish a deployable selector.

The strongest case for stopping the entire project is that small v4 oracle
headroom may reflect a generally weak relationship between local reference
likelihood and behavior. Direction 1 is preferred only because it changes the
missing causal capability while retaining a small, behavior-coherent kill
test. If its matched cross-over fails, the correct next decision is project
closure, not Direction 2 or 3 by default.

## 8. Single minimum next experiment

Prospectively freeze the 100-image, two-question GQA visual-token replay test
defined above. Run the frozen base model only; do not train a router, selector,
adapter, or base model. Freeze image IDs, target-to-token mapping, 16-token
budget, controls, prompt layout, likelihood/correctness endpoints, clustered
analysis, practical threshold, and kill rule before opening outcomes.

No execution is authorized by this synthesis.

TEST_QUERY_CONDITIONED_VISUAL_REFINEMENT
