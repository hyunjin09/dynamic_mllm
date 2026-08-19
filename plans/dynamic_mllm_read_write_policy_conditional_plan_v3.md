# Analysis-First Research Plan v3
## Policy-Conditional Counterfactual Value of Visual READ/WRITE in MLLMs

**Status:** This document supersedes the earlier READ/WRITE causal-analysis plans.  
**Primary mode:** frozen-model analysis only.  
**Router status:** **do not train or deploy a router/controller in this phase.**  
**Primary goal:** establish and explain the signed, policy-conditional value of visual participation before attempting dynamic routing.

---

## 0. Execution Contract

The executor must follow these constraints.

1. Keep the base MLLM frozen.
2. Do not train a routing policy, controller, or end-to-end action selector.
3. Do not fine-tune the MLLM.
4. Do not infer independent intrinsic utilities for READ and WRITE.
5. Treat the four READ/WRITE action values under a fixed suffix policy as the primary object.
6. Do not call a computation “harmful” from a top-1 correction alone.
7. Do not use discovery data as held-out prevalence evidence.
8. Match the layer/action search budget of every null to the real intervention.
9. Do not use post-action activations as pre-action predictability features.
10. Stop at every stage gate and write a decision report before continuing.

A small **offline diagnostic probe** is allowed only after the held-out causal claim passes. It is not a deployment router, must not execute actions, and must not support an acceleration claim.

---

## 1. Paper Positioning

Existing MLLM-efficiency work mainly asks whether visual computation changes the output or can be removed while preserving dense-model behavior. This project asks a different question:

> **Does the visual operation change grounded answer evidence in the correct direction?**

The central distinction is between influence and signed value.

- **Answer-aligned action:** increases the preregistered correct-answer utility.
- **Answer-silent action:** changes the utility by less than a frozen tolerance.
- **Answer-misaligned dense participation:** suppressing part of the dense visual path produces a reliably higher utility than `FULL`.
- **FULL-critical participation:** every cheaper alternative is reliably worse than `FULL`.

Core statement:

> **Visual computation is not intrinsically redundant or useful. Its value is the signed, policy-conditional advantage of applying a READ/WRITE action to the current multimodal state.**

The strongest architecture-specific hypothesis is:

> **In a prefix-causal MLLM, visual WRITE can be structurally query-invariant while its downstream advantage reverses sign across questions on the same image.**

This hypothesis may be used only after the architecture audit proves its causal premises.

---

## 2. Scope and Non-Goals

### 2.1 Primary model

Use one frozen primary model: the project’s current Qwen2.5-VL analysis model, unless Stage A proves that the proposed decomposition is invalid.

### 2.2 Primary task families

Use the two already established task families:

1. general visual question answering / reasoning, represented by GQA or the project’s validated equivalent;
2. OCR- or text-rich visual question answering, represented by TextVQA or the project’s validated equivalent.

Use the project’s already validated answer-probability scoring implementation. Do not silently replace its handling of accepted answers, tokenization, or length normalization.

A multiple-choice diagnostic subset may be added for option-order controls, but multiple choice is not required to be the only primary task.

### 2.3 Primary intervention regime

Use a **single-layer intervention during prompt/prefill**, then run an unchanged dense suffix. Multi-layer policies, repeated actions, and per-decoding-step routing are out of scope until the local claim is confirmed.

### 2.4 Explicit non-goals

- no router training;
- no adaptive inference policy;
- no general acceleration claim from oracle intervention counts;
- no broad attention-vs-FFN decomposition in the initial study;
- no claim that hidden-state decodability proves the actual mechanism;
- no attempt to explain every observed correction with every proposed mechanism;
- no long-form-generation paper in this phase.

---

## 3. Formal Object: Four Policy-Conditional Action Values

### 3.1 State and action

At layer `l`, let the pre-layer state be

\[
s_l=(V_l,T_l,h_l),
\]

where:

- `V_l` is the visual-token state;
- `T_l` is the text/control-token state;
- `h_l` contains any fixed metadata required to reproduce the computation, such as masks, positions, and the prior route prefix. In the primary local study, the prefix is always dense/FULL.

The current-layer action is

\[
a_l=(r_l,w_l)\in\{0,1\}^2.
\]

The four actions are:

| `r` | `w` | Name | Meaning |
|---:|---:|---|---|
| 0 | 0 | `IGNORE` | Remove the current path-specific visual READ and freeze the visual block output. |
| 1 | 0 | `READ_ONLY` | Preserve the original visual READ; freeze the visual block output. |
| 0 | 1 | `WRITE_ONLY` | Preserve the original visual block update; remove the current path-specific visual READ. |
| 1 | 1 | `FULL` | Original dense MLLM computation. |

All text self-attention, text FFN, normalization, residual, and control-token computation remain active in all four actions.

### 3.2 Fixed suffix policy

The primary suffix policy is fixed to dense execution:

\[
\pi_{>l}=\pi_{\mathrm{FULL}}.
\]

Every branch must therefore follow this protocol:

```text
FULL prefix through layer l-1
-> one of the four actions at layer l
-> FULL execution at every layer after l
-> identical answer-scoring procedure
```

Do not mix this quantity with the value under a future adaptive router.

### 3.3 Terminal task utility

Let `F_{l:L}` denote the frozen suffix rollout and `M_q` the preregistered terminal answer metric for sample/query `q`.

\[
Q_l^{\mathrm{dense}}(r,w\mid s_l,q)
=
M_q\!\left(
F_{l:L}(s_l,(r,w),\pi_{>l}=\pi_{\mathrm{FULL}})
\right).
\]

Evaluation is deterministic, so an expectation is unnecessary unless stochastic kernels or sampling are explicitly introduced.

#### Short-answer VQA

Use the already validated reference-answer sequence score as the primary continuous metric. Generated-answer correctness is secondary.

#### Multiple choice, when used

For option `y_j`, compute the validated length-normalized content score `s(y_j)`. Record:

\[
M_{\max}=s(y^*)-\max_{j\neq *}s(y_j),
\]

\[
M_{\mathrm{lse}}=s(y^*)-\log\sum_{j\neq *}\exp s(y_j),
\]

and every pairwise margin `s(y*) - s(y_j)`.

Do not interpret a top-1 flip with a near-zero continuous margin as a robust correction.

### 3.4 FULL-relative suppression advantage

The primary advantage is relative to the dense action:

\[
A_l^{\mathrm{sup}}(r,w)
=
Q_l^{\mathrm{dense}}(r,w)-Q_l^{\mathrm{dense}}(1,1).
\]

Define the best suppression gain:

\[
G_l
=
\max_{(r,w)\neq(1,1)}A_l^{\mathrm{sup}}(r,w).
\]

Interpretation after applying a frozen tolerance `epsilon` and uncertainty rule:

- `G_l > epsilon`: candidate answer-misaligned dense visual participation;
- `|G_l| <= epsilon` for at least one cheaper action, with no positive alternative: dense visual participation is locally redundant/silent;
- `G_l < -epsilon`: `FULL` is locally critical.

Do not use “harmful” before held-out confirmation and mechanistic verification.

### 3.5 Derived conditional contrasts

READ and WRITE effects are derived from the four action values, not treated as independent intrinsic utilities.

\[
\Delta_R^{w=0}=Q_l(1,0)-Q_l(0,0),
\]

\[
\Delta_R^{w=1}=Q_l(1,1)-Q_l(0,1),
\]

\[
\Delta_W^{r=0}=Q_l(0,1)-Q_l(0,0),
\]

\[
\Delta_W^{r=1}=Q_l(1,1)-Q_l(1,0).
\]

Interaction:

\[
I_l=Q_l(1,1)-Q_l(1,0)-Q_l(0,1)+Q_l(0,0).
\]

Rules:

- always report all four cells;
- report both conditional READ effects and both conditional WRITE effects;
- do not collapse them into one READ label and one WRITE label when interaction is material;
- a negative `Delta_W^(r=1)` means WRITE is answer-misaligned **under READ ON and the fixed dense suffix**, not universally harmful.

---

## 4. Exact Intervention Definitions

### 4.1 Primary READ counterfactual: path-specific removal

Decompose the text-query attention output into visual-value and non-visual-value paths while preserving the original attention probabilities.

At a validated additive hook:

```text
text_state_full = text_state_no_read + delta_read
```

Primary `READ OFF` removes only `delta_read`. It must not alter the non-visual attention contribution or softmax normalization.

### 4.2 Secondary READ counterfactual: operational masking

Mask text-query-to-visual-key edges and recompute attention. This is an implementation/deployment-style action and may redistribute text-to-text attention.

Report it separately from the path-specific estimand.

### 4.3 Primary WRITE counterfactual: block-output freeze

Let:

```text
delta_write = V_full_block_out - V_pre_layer
```

Primary `WRITE OFF` sets:

```text
V_intervened_block_out = V_pre_layer
```

while preserving the current layer’s text output according to the selected READ state.

At the exact hook:

```text
V_intervened_block_out + delta_write == V_full_block_out
```

must hold within the declared numerical tolerance.

### 4.4 Attention/FFN decomposition

Do not include attention-specific and FFN-specific WRITE actions in the primary four-action sweep. Add this decomposition only if:

1. WRITE effects survive Stage C;
2. the sublayer reconstruction identities are valid;
3. WRITE is selected as the single mechanism focus for Stage D.

---

## 5. Primary Research Questions

### RQ1 — Four-action value landscape

Do `FULL`, `READ_ONLY`, `WRITE_ONLY`, and `IGNORE` have meaningfully different signed terminal values under the same pre-layer state and fixed dense suffix?

### RQ2 — Dense visual participation beyond redundancy

Does suppressing dense visual participation produce a held-out positive `G_l` more often, and by a larger amount, than a search-adjusted structured null?

### RQ3 — READ/WRITE interaction

How often is the sign or magnitude of a READ/WRITE contrast conditional on the other bit? Is a factorized independent interpretation inadequate?

### RQ4 — Query-invariant WRITE, query-dependent value

After verifying the prefix-causal architecture, can the same structural visual WRITE have opposite `Delta_W^(r=1)` signs for different questions on the same image?

### RQ5 — Compact pre-action predictability

Can a compact, low-capacity summary of the pre-action cross-modal state predict the four action values or their ranking better than layer-only, difficulty-only, or unimodal summaries?

This is an optional diagnostic analysis, not router training.

### RQ6 — Mechanism

Which one dominant path best explains verified effects:

- WRITE-mediated erosion of query-relevant visual evidence; or
- READ-mediated injection of a distractor/conflicting visual signal?

Choose one after Stage C. Do not attempt every mechanism in the main paper.

---

## 6. Data Splits and Statistical Units

Use disjoint sets.

### 6.1 Stage A validity set

20–50 samples. May overlap neither confirmatory nor mechanistic replication data.

### 6.2 Stage B discovery set

100–200 samples across both task families. Use a sparse predetermined layer grid, such as `[0,4,8,...]`.

Discovery is allowed to determine:

- fixed confirmatory layer grid/band `L*`;
- terminal metric adapter;
- tolerance `epsilon`;
- search statistic;
- structured-null hierarchy;
- which mechanism family will be considered if confirmation succeeds;
- same-image paired subset construction.

No final prevalence claim is allowed.

### 6.3 Stage C confirmatory set

Use new held-out samples. Target 500+ when affordable; determine the final size from Stage B effect sizes and power analysis.

Before opening results, freeze the analysis manifest.

### 6.4 Stage D explanation/replication set

Use new cases selected by a fixed rule from a held-out pool. Apply the protocol to all eligible cases or a random subset, never only visually attractive examples.

### 6.5 Clustering

- cluster standard errors and bootstrap resampling by sample;
- when multiple questions share an image, use image-level or hierarchical clustering;
- same-image paired analysis must split by image, never by question.

---

## 7. Stage A — Architecture and Intervention Validity

### 7.1 Architecture audit

Document:

- exact system/image/question/answer token order;
- visual-token indices;
- causal attention mask;
- whether visual query rows can see question tokens;
- whether current-layer text READ uses pre-WRITE visual K/V;
- pre-norm/post-norm structure;
- GQA/MQA attention details;
- RoPE/MRoPE and position handling;
- fused-kernel behavior;
- KV-cache behavior.

Produce a computational causal graph, not only a verbal description.

### 7.2 Query-invariance theorem/sanity check

If visual tokens precede the question and the causal mask blocks visual rows from later question tokens, document the deterministic implication:

```text
same image + same preceding prefix
=> identical visual states and visual WRITE across different later questions
```

Numerically verify this only as an implementation sanity check. Do not present equality itself as the discovery.

### 7.3 No-op parity

The instrumented `FULL` path must reproduce the original model for:

- layer states;
- final logits;
- reference-answer scores;
- deterministic generated answers.

### 7.4 Algebraic reconstruction

Validate:

```text
state_no_read + delta_read ~= state_full
V_no_write + delta_write ~= V_full
```

at the intervention hook and after the identical suffix.

### 7.5 Activation plausibility diagnostics

Record:

- token norms and RMSNorm statistics;
- cosine and Euclidean distances to natural states;
- PCA/residual-subspace distances;
- nearest-neighbor distances to same-layer natural states.

These diagnostics limit interpretation; they do not by themselves validate or invalidate a causal intervention.

### Stage A gate

Proceed only if:

- causal graph is explicit;
- no-op parity passes;
- READ and WRITE reconstruction pass;
- all four actions are deterministic and numerically stable;
- baseline scoring is reproduced.

Otherwise stop and redesign the hook.

### Required outputs

- `architecture_causal_graph.md`
- `token_layout.json`
- `baseline_parity.csv`
- `read_reconstruction.csv`
- `write_reconstruction.csv`
- `stage_a_decision.md`

---

## 8. Stage B — Four-Action Discovery

### 8.1 Core sweep

For every discovery sample and every layer in the sparse grid:

1. cache the identical dense pre-layer state;
2. execute all four actions;
3. run the identical dense suffix;
4. record the complete four-cell `Q` vector.

### 8.2 Required records

For each sample–layer–action:

- dataset, image ID, question ID;
- layer and action;
- terminal `Q` value;
- reference-answer score or MC margins;
- top-1/generated correctness;
- strongest distractor, where applicable;
- output KL to FULL;
- final text-state distance;
- READ/WRITE residual norm;
- runtime and memory for analysis accounting only.

### 8.3 Required analyses

- four-cell value heatmaps;
- `A_sup` and `G_l` distributions;
- all four conditional contrasts;
- interaction `I_l`;
- FULL-correct vs FULL-wrong strata;
- task-family differences;
- numerical-noise distribution;
- preliminary structured-null distribution;
- exploratory same-image sign reversals.

### 8.4 Freeze for confirmation

Write `confirmatory_manifest.yaml` containing:

- model checkpoint and code commit;
- exact token/scoring protocol;
- fixed layer set `L*`;
- `epsilon` and tie threshold;
- sample-level primary statistic;
- structured-null hierarchy;
- number of null replicates;
- option/paraphrase controls;
- same-image pair construction;
- success and pivot criteria.

### Stage B gate

Continue only if the four-action landscape contains reproducible heterogeneity not explained entirely by numerical ties or a single global layer schedule.

No router or diagnostic probe may be trained in Stage B.

### Required outputs

- `four_action_discovery.parquet`
- `q_matrix_summary.md`
- `confirmatory_manifest.yaml`
- `stage_b_decision.md`

---

## 9. Stage C — Held-Out Confirmation

### 9.1 Primary sample-level statistic

Use a frozen statistic. A default form is:

\[
S(x)=\max_{l\in L^*,\,a\neq\mathrm{FULL}}
\left[Q_l(a)-Q_l(\mathrm{FULL})\right].
\]

If Stage B chooses an operation-specific statistic, declare it in the manifest. Do not change it after viewing confirmatory results.

### 9.2 Search-adjusted structured null

Every null replicate must receive the same search budget across:

- all layers in `L*`;
- the same action/control families;
- the same maximum/minimum selection rule.

Required null hierarchy:

1. isotropic norm-matched residual;
2. layer-wise covariance/subspace-matched residual;
3. real residual from another sample matched by layer, task family, and norm.

Avoid using visual-token permutation as the universal primary null because it destroys spatial semantics and may be more distribution-shifting than the target intervention.

### 9.3 Primary confirmatory endpoint

Test whether the real sample-level suppression statistic shows an excess over the search-adjusted structured null, with image/sample-clustered uncertainty.

Report separately:

- sample-level excess prevalence;
- layer-level average effects;
- sample–layer local-effect distributions;
- FULL-correct regression;
- FULL-wrong improvement;
- interaction prevalence;
- sensitivity to `epsilon`.

### 9.4 Robustness controls

Use task-appropriate controls.

#### For multiple choice

- option-order permutation;
- A/B/C/D label permutation;
- content scoring rather than label-token scoring;
- `M_max`, `M_lse`, and all pairwise margins.

#### For short-answer VQA

- validated answer variants;
- deterministic choice-free generation;
- prompt paraphrase on a frozen subset;
- reference-answer likelihood and generated correctness reported separately.

### 9.5 Grounding control

Prefer:

1. same-question counterfactual image pairs where answer-determining content changes;
2. target-region intervention with an equal-area matched non-target control;
3. no-image or shuffled-image only as secondary stress tests.

### Stage C gate

The answer-misaligned-participation claim survives only if:

- real effects exceed the search-adjusted structured null;
- effects are not dominated by near-zero ties;
- answer-content controls pass;
- sample-level prevalence and uncertainty are reported;
- visual-content changes modulate the effect in the expected direction.

Otherwise pivot to answer-silent redundancy or stop.

### Required outputs

- `four_action_confirmatory.parquet`
- `search_adjusted_null.parquet`
- `confirmatory_results.md`
- `stage_c_decision.md`

---

## 10. Stage C2 — Same-Image Query-Dependence Test

This is a central confirmatory analysis, not a router experiment.

### 10.1 Preconditions

Use only architectures for which Stage A proves structural query invariance of visual WRITE under the actual token order and mask.

### 10.2 Primary WRITE contrast

Use:

\[
\Delta_W^{r=1}(I,q,l)=Q_l(1,1)-Q_l(1,0).
\]

Positive means WRITE helps under READ ON and the dense suffix. Negative means WRITE reduces terminal utility under the same conditions.

### 10.3 Metrics

For images with multiple questions, report:

- within-image WRITE-sign reversal rate;
- within-image variance of `Delta_W^(r=1)`;
- between-image vs within-image variance decomposition;
- baseline-difficulty-matched sign reversal;
- semantic-paraphrase consistency;
- task/category-matched question comparison.

### 10.4 Cross-query action-transfer regret

Let `a*(I,q,l)` be the best of the four local actions for a question. Evaluate the cost of transferring the action from another question on the same image:

\[
R_{\mathrm{transfer}}
=
Q_l(a^*_{q_{target}}\mid q_{target})
-
Q_l(a^*_{q_{source}}\mid q_{target}).
\]

A positive regret supports query-specific action value beyond image-only complexity.

### 10.5 Interpretation limit

Do not claim that the model “knows the correct query-specific WRITE” from sign reversal alone. This stage establishes query-dependent value under a structurally fixed WRITE, not pre-action predictability.

### Required outputs

- `same_image_qvalue_pairs.parquet`
- `same_image_sign_reversal.md`
- `cross_query_transfer_regret.csv`

---

## 11. Stage D1 — Optional Compact Pre-Action Predictability Study

Begin only if Stage C and C2 pass.

This is an offline diagnostic probe, not a routing policy. It must not execute an action or support an efficiency claim.

### 11.1 Targets

Prefer predicting:

```text
[Q(0,0), Q(1,0), Q(0,1), Q(1,1)]
```

or the three FULL-relative suppression advantages.

A pairwise ranking target is also allowed:

```text
Q(a_i) > Q(a_j)
```

Do not train independent READ-harmful and WRITE-harmful classifiers as the primary target.

### 11.2 Pre-action inputs only

Allowed:

- compact visual summary from `V_l`;
- compact question/text summary from `T_l`;
- layer index;
- baseline difficulty/uncertainty features available before the action;
- explicit visual–text interaction features;
- prior route metadata only if the studied prefix includes non-dense actions.

Forbidden:

- current action residuals computed after executing the action;
- post-READ/post-WRITE logits;
- final correctness;
- full future trajectory features.

### 11.3 Matched-capacity baselines

Compare:

1. layer only;
2. difficulty only;
3. visual only;
4. text/query only;
5. additive `[V,T]`;
6. interaction-aware `[V,T,V⊙T,|V-T|]` or an equally compact bilinear alternative.

Match parameter count and optimization budget as closely as practical.

### 11.4 Splits

Use image-grouped train/validation/test splits. If multiple datasets are used, a source-group OOD split is optional but not required for the first paper.

### 11.5 Metrics

Primary:

\[
\mathrm{Regret}=Q(a^*)-Q(\hat a).
\]

Secondary:

- Spearman correlation for each `Q`/advantage;
- pairwise ranking accuracy;
- AUROC for `G_l > epsilon`;
- calibration error;
- high-confidence precision and coverage.

### 11.6 Claim limit

A positive result supports:

> A compact pre-action cross-modal summary contains useful information about policy-conditional action value.

It does not prove that an eventual router uses a specific human-interpretable mechanism such as “visual-memory adequacy” or “textual-belief sufficiency.”

### Required outputs

- `diagnostic_probe_config.yaml`
- `diagnostic_probe_results.csv`
- `preaction_predictability.md`

---

## 12. Stage D2 — One Mechanistic Replication Path

Choose exactly one path after Stage C.

### Option A — WRITE-mediated evidence erosion

Use this path if `READ_ONLY > FULL` effects dominate.

Required analyses:

1. exact WRITE removal and reconstruction-valid add-back;
2. dose response `V_out(alpha)=V_in + alpha*delta_write`;
3. structured residual controls;
4. query-relevant evidence before/after WRITE using a validated probe or localized evidence score;
5. gradient/update analysis as supporting explanation only.

Compare:

\[
\langle\nabla J,\delta_W\rangle,
\quad
\cos(\nabla J,\delta_W),
\quad
\|\delta_W\|,
\quad
\|\nabla J\|.
\]

Ground truth remains the exact branch `Q` value.

### Option B — READ-mediated distractor/conflict injection

Use this path if no-READ actions dominate.

Required analyses:

1. path-specific READ removal and exact add-back;
2. operational masking as a separate implementation comparison;
3. dose response on the exact READ contribution;
4. text-state belief before/after READ;
5. visual evidence or distractor contribution with matched region controls.

### Optional adequacy/sufficiency interventions

Only if scope permits, directly manipulate:

- visual adequacy;
- text-state sufficiency;
- cross-modal conflict.

Without these interventions, describe “adequacy/sufficiency” as an interpretation consistent with the evidence, not as a proven router mechanism.

### Mechanistic replication rule

Apply the mechanism protocol to every case meeting the frozen rule or a random subset. Do not select only the strongest or most interpretable examples.

### Required outputs

- `mechanism_cases_manifest.json`
- `addback_results.csv`
- `dose_response.parquet`
- `mechanism_replication.md`

---

## 13. Terminology and Claim Ladder

Use the following terms in order.

### Tier 0 — Observed counterfactual difference

A four-action `Q` difference exists.

### Tier 1 — Candidate answer-misaligned dense participation

A cheaper action beats `FULL` by more than the discovery threshold.

### Tier 2 — Confirmed answer-misaligned dense participation

The effect replicates on held-out data, beats the search-adjusted structured null, and passes answer-content robustness controls.

### Tier 3 — Mechanistically verified harmful visual contribution

Tier 2 plus reconstruction-valid add-back/rescue, component specificity, and visual-content sensitivity on a held-out mechanism set.

Do not collapse these tiers in reporting.

---

## 14. Success, Kill, and Pivot Criteria

### 14.1 Success for the core causal paper

Required:

1. exact intervention validity;
2. a heterogeneous four-action `Q` landscape;
3. held-out FULL-relative suppression gains beyond search-adjusted structured nulls;
4. sample/image-clustered prevalence with confidence intervals;
5. answer-content and grounding robustness;
6. non-trivial READ/WRITE interaction or a coherent conditional effect;
7. one replicated mechanism path.

Strongly desirable:

- same-image WRITE sign reversal at fixed layers;
- positive cross-query transfer regret;
- compact pre-action interaction features outperform simpler probe baselines;
- one small second-model replication.

### 14.2 Kill or weaken the harmfulness claim if

- reconstruction fails;
- best-of-many real effects are matched by best-of-many structured nulls;
- effects are mostly numerical ties;
- answer-content controls fail;
- visual counterfactuals do not modulate the effect;
- same-image query dependence disappears after difficulty/task matching;
- mechanistic add-back does not replicate on held-out cases.

### 14.3 Pivot paths

- **Answer-silent redundancy:** four-action alternatives preserve `FULL` but rarely improve grounded utility.
- **Interaction analysis only:** four-cell effects are real but do not support a simple READ-vs-WRITE taxonomy.
- **Representation analysis:** state geometry is systematic but terminal causal effects are weak.
- **Stop/redesign:** interventions are invalid or indistinguishable from structured nulls.

Router training remains excluded in every pivot for this project phase.

---

## 15. Planned Main-Paper Story

### Figure 1 — Formal distinction

Unsigned influence versus signed, policy-conditional four-action value.

### Figure 2 — Four-action value landscape

Layer-wise `Q` matrices, FULL-relative suppression gains, and interaction.

### Figure 3 — Held-out confirmation

Sample-level effect prevalence versus search-adjusted structured null.

### Figure 4 — Same image, same WRITE, different value

Paired questions with opposite `Delta_W^(r=1)`, variance decomposition, and action-transfer regret.

### Figure 5 — Compact pre-action predictability

Only if Stage D1 succeeds: matched-capacity probe comparison and decision regret.

### Figure 6 — One verified mechanism

Either WRITE evidence erosion or READ distractor injection, with exact add-back and dose response.

### Main tables

1. intervention validity and reconstruction;
2. held-out four-action effects by task;
3. answer-content/grounding robustness;
4. same-image sign reversal and transfer regret;
5. mechanistic replication rate.

---

## 16. Repository and Artifact Layout

```text
analysis_v3/
  00_architecture_audit.py
  01_validate_full_parity.py
  02_validate_read_write_reconstruction.py
  03_collect_four_action_q.py
  04_build_structured_nulls.py
  05_run_discovery.py
  06_freeze_confirmatory_manifest.py
  07_run_confirmatory.py
  08_same_image_analysis.py
  09_diagnostic_preaction_probe.py
  10_mechanism_write.py
  11_mechanism_read.py
  12_make_figures.py

configs/
  model.yaml
  datasets.yaml
  intervention.yaml
  answer_metric.yaml
  discovery.yaml
  confirmatory_manifest.yaml

artifacts/
  stage_a/
  stage_b/
  stage_c/
  same_image/
  diagnostic_probe/
  mechanism/

reports/
  stage_a_decision.md
  stage_b_decision.md
  stage_c_decision.md
  final_analysis_summary.md
```

Every artifact must store:

- code commit;
- model checkpoint hash;
- config hash;
- dataset IDs;
- layer/action definitions;
- scoring version;
- random seeds;
- creation timestamp.

---

## 17. Migration from Existing v2 Artifacts

Existing four-state results can be reused only as follows.

1. Treat all previously inspected data as discovery data.
2. Reconstruct the four-cell vector for every sample–layer pair:

```text
[Q(0,0), Q(1,0), Q(0,1), Q(1,1)]
```

3. Recompute FULL-relative suppression advantages and interaction.
4. Retain old conditional `U_read` and `U_write` only as derived contrasts.
5. Verify that every branch used an identical dense suffix.
6. Do not use previously selected examples as mechanistic replication evidence.
7. Freeze a new held-out confirmatory manifest before collecting new confirmatory results.
8. Do not resume router training from old labels in this phase.

---

## 18. First Agent Assignment

The first bounded assignment under this plan is:

> Audit the existing Stage A/B artifacts against the v3 definitions. Determine whether the existing four-state runs can be represented as valid dense-suffix `Q_l(r,w)` matrices with exact READ/WRITE reconstruction. Produce a migration report listing reusable artifacts, invalid artifacts, missing tests, and the minimum next experiment. Do not run a new large sweep and do not train any model.

Required first output:

- `reports/v3_migration_audit.md`

The report must end with exactly one decision:

- `REUSE_AND_CONFIRM`
- `REPAIR_STAGE_A`
- `RERUN_DISCOVERY`
- `STOP_AND_REDESIGN`

