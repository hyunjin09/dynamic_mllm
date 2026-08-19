# Analysis-First Research Plan v2
## Confirmatory Causal Analysis of Visual READ/WRITE Contributions in MLLMs

## 0. Project Status and Non-Goals

This project studies whether visual participation inside a frozen decoder-based multimodal large language model (MLLM) is answer-aligned, answer-silent, or answer-misaligned.

This version is intentionally narrow and confirmatory.

### Explicit non-goals

- Do **not** train a router, controller, or policy.
- Do **not** fine-tune the base MLLM.
- Do **not** make a general acceleration claim from intervention counts alone.
- Do **not** equate a `wrong -> correct` top-1 flip with a harmful mechanism.
- Do **not** use qualitative case selection as prevalence evidence.
- Do **not** expand to all proposed mechanisms, models, and generation regimes before the primary claim is confirmed.

---

## 1. Narrow Paper Claim

The project tests the following claim:

> Visual influence is signed. At a fixed pre-layer state, executing a visual READ or WRITE can improve, preserve, or reduce grounded correct-answer evidence. Some visual contributions are therefore answer-silent, while a smaller subset may be locally answer-misaligned.

The initial estimand is deliberately limited:

> **Local total-suffix intervention effect:** the change in answer utility caused by a precisely defined READ or WRITE intervention at layer `l`, followed by the unchanged downstream suffix of the frozen model.

The terminology is staged:

1. **Candidate answer-misaligned effect:** a negative signed effect found during discovery.
2. **Confirmed answer-misaligned effect:** a negative effect replicated on held-out data under a predeclared analysis and a search-adjusted structured null.
3. **Mechanistically verified harmful contribution:** a confirmed effect that additionally passes exact reconstruction/add-back, specificity, and visual-grounding tests.

Do not use the third term before all required evidence is available.

Core slogan:

> **Influence is not utility.**

---

## 2. Scope

### Primary model

Use one frozen primary MLLM: the current Qwen2.5-VL analysis model, unless the architecture audit shows that the proposed decomposition is invalid.

### Primary task families

Use exactly two task families for the main study:

1. **General multiple-choice visual reasoning / VQA**
2. **OCR- or text-rich visual question answering**

A hallucination-sensitive or open-ended task may be used only as a small confirmatory extension after the primary analysis succeeds.

### Primary intervention regime

The main experiment applies a **single-layer intervention during the prompt/prefill forward pass** and then runs the unchanged suffix. Multi-layer windows and autoregressive per-step interventions are out of scope until the local claim is established.

---

## 3. Computational Causal Graph

Before experimentation, document the actual graph for the target architecture.

At layer `l`, let the pre-layer state be:

- `V_l`: visual-token rows
- `T_l`: text/control-token rows
- `H_l = [V_l; T_l]`

The two causal roles are:

1. **READ (`T <- V`)**: visual key/value contributions entering text-query attention outputs.
2. **WRITE (`V -> V'`)**: the net change in visual-token rows produced by the current Transformer block.

Important interpretation:

- A READ intervention estimates the **total downstream effect** of changing the current visual-to-text path.
- A WRITE intervention estimates the **total downstream effect** of changing the current visual memory, including all future READ/WRITE pathways.
- These are not claimed to be isolated intrinsic utilities of an operator independent of the suffix.

### Architecture audit questions

Verify, rather than assume:

- the exact sequence order of system, image, question, and answer tokens;
- whether visual query rows can attend to question tokens;
- whether the model is pre-norm or post-norm;
- whether same-layer text READ uses pre-WRITE visual keys/values;
- how GQA/MQA, RoPE, fused attention kernels, and KV caches affect decomposition;
- whether text states can causally affect future visual WRITE under the actual mask.

Produce `architecture_causal_graph.md` before any large sweep.

---

## 4. Exact Counterfactual Definitions

## 4.1 Path-specific READ

Let the full text attention contribution at layer `l` be decomposed into visual and non-visual value paths.

Define `delta_read_l` at an additive hook point such that:

```text
text_state_full = text_state_no_read + delta_read_l
```

The implementation must preserve all non-visual attention paths and avoid softmax renormalization for the primary causal estimand.

- **Primary READ OFF:** path-specific subtraction of the exact visual value contribution.
- **Secondary operational READ OFF:** mask text-query-to-visual-key edges and recompute attention.

The operational mask is a deployment-style intervention, not the primary path-specific estimand.

## 4.2 Block-output WRITE

Let the full visual block output be `V_l_full_out` and the pre-layer visual state be `V_l`.

Define:

```text
delta_write_l = V_l_full_out - V_l
```

The primary WRITE OFF intervention sets:

```text
V_l_intervened_out = V_l
```

while leaving the current layer's text output unchanged.

At the exact layer-output hook:

```text
V_l_intervened_out + delta_write_l = V_l_full_out
```

must hold numerically.

Attention- and FFN-specific WRITE decomposition is a **secondary mechanism analysis**, not part of the initial factorial sweep.

## 4.3 Four states

At an identical cached pre-layer state, evaluate:

| READ | WRITE | Name | Interpretation |
|---:|---:|---|---|
| 0 | 0 | `IGNORE` | No path-specific visual READ; visual block output is frozen to its input. |
| 1 | 0 | `READ_ONLY` | Text receives the original visual READ; visual block output is frozen. |
| 0 | 1 | `WRITE_ONLY` | Visual rows receive the original block update; the current text visual-read path is removed. |
| 1 | 1 | `FULL` | Original model computation. |

All text self-attention, text FFN, residual, and normalization operations remain active.

---

## 5. Algebraic and Numerical Validity Tests

No causal result is interpretable until the following tests pass.

### 5.1 No-op parity

The instrumented `FULL` path must reproduce the unmodified model within a declared tolerance for:

- layer states;
- final logits;
- option scores;
- generated answer under deterministic decoding.

### 5.2 READ reconstruction identity

At the exact additive hook:

```text
state_no_read + delta_read_l ~= state_full
```

Then run the same suffix and verify that full logits are recovered.

### 5.3 WRITE reconstruction identity

At the layer output:

```text
V_no_write + delta_write_l ~= V_full
```

Then run the same suffix and verify that full logits are recovered.

### 5.4 Activation plausibility diagnostics

Report, but do not overinterpret:

- token norms and RMSNorm statistics;
- cosine distance to natural full states;
- activation PCA/subspace distance;
- nearest-neighbor distance to natural states from the same layer/task.

The intervention need not be a naturally sampled training state, but its interpretation must be limited to the declared counterfactual and supported by structured controls.

### Stop condition

Stop the project if exact reconstruction or no-op parity cannot be established at a clean hook point.

---

## 6. Answer-Utility Metrics

For each answer option `y_j`, compute a teacher-forced, length-normalized content score:

```text
s(y_j) = (1 / |y_j|) * sum_t log p(y_j,t | input, y_j,<t)
```

### Primary metric

```text
M_max = s(y*) - max_{j != *} s(y_j)
```

### Secondary metrics

```text
M_lse = s(y*) - logsumexp({s(y_j): j != *})
M_j   = s(y*) - s(y_j)    for every distractor j
```

Also record:

- top-1 correctness;
- answer entropy;
- the identity of the strongest distractor;
- option-order and label-order variants.

Do not claim a robust effect when only top-1 changes but all margins remain near zero.

---

## 7. Factorial Effects

For each layer `l`, define:

```text
M_l^(r,w) = answer margin after state (r,w) and the unchanged suffix
```

Conditional effects:

```text
U_read^(w)  = M_l^(1,w) - M_l^(0,w)
U_write^(r) = M_l^(r,1) - M_l^(r,0)
```

Interaction:

```text
U_interaction = M^(1,1) - M^(1,0) - M^(0,1) + M^(0,0)
```

Optional summary main effects:

```text
U_read_bar  = 0.5 * [(M^(1,0)-M^(0,0)) + (M^(1,1)-M^(0,1))]
U_write_bar = 0.5 * [(M^(0,1)-M^(0,0)) + (M^(1,1)-M^(1,0))]
```

Never replace the conditional effects and interaction with only the averages. When interaction is large, interpret the four cells directly.

---

## 8. Data Splits and Selection-Bias Control

Use three disjoint sets.

## 8.1 Discovery set

Purpose:

- validate implementation at scale;
- identify a small, fixed candidate layer grid or band;
- estimate numerical noise;
- choose effect thresholds;
- choose whether READ or WRITE is the primary mechanism target.

Rules:

- discovery results are exploratory;
- do not use discovery prevalence as a final claim;
- do not choose qualitative examples as evidence.

Suggested size: 100–200 samples across the two task families.

## 8.2 Confirmatory set

Purpose:

- test a frozen analysis protocol on new data;
- estimate sample-level prevalence and population effects.

Before opening results, freeze:

- candidate layer grid/band;
- primary metric;
- effect threshold `epsilon`;
- structured null construction;
- search statistic;
- success criteria.

Suggested size: determined from discovery effect sizes and a power analysis, with 500+ samples as a practical target when affordable.

## 8.3 Mechanistic replication set

Purpose:

- test add-back, dose response, and grounding on new cases selected by a fixed rule;
- avoid analyzing only the strongest discovery examples.

Apply the mechanism protocol to **all cases meeting the predeclared rule**, or to a random subset of them. Do not choose only visually appealing cases.

Suggested size: 20–50 verified candidates, subject to prevalence.

### Clustering

Bootstrap and uncertainty estimates must cluster at the sample level. If multiple questions share one image, use image-level or hierarchical clustering.

---

## 9. Search-Adjusted Confirmatory Null

Layer search must be matched between the real intervention and every null.

Let `L*` be the fixed confirmatory layer grid. For each sample, the real statistic may search over `L*` only if the null performs the identical search.

Example sample-level statistic:

```text
S_read(x) = min_{l in L*, conditioning state c} U_read(x,l,c)
```

A structured null replicate must:

1. generate a matched null READ/WRITE residual at every layer in `L*`;
2. apply the same scoring procedure;
3. take the same minimum or maximum across layers and states.

Compare the real extreme statistic against the distribution of null extreme statistics. Do not compare a best-of-many real intervention with a single random intervention.

Report separately:

- sample-level prevalence;
- layer-level mean effects;
- sample-layer local-effect distributions.

---

## 10. Structured Null Interventions

Simple isotropic norm-matched noise is insufficient as the only null.

Use a hierarchy of controls.

### Required

1. **Isotropic norm-matched residual**
2. **Layer-wise covariance/subspace-matched residual** sampled from the empirical READ or WRITE residual distribution
3. **Real-residual control** using an actual residual from another sample, matched by layer, norm, task family, and optionally activation similarity

### Optional

- residual direction sampled within the top activation/residual PCA subspace;
- nearest-neighbor real residual with mismatched answer semantics;
- different-component control at the same hook.

Avoid treating visual-token permutation as a universal null: it destroys spatial semantics and may be more out-of-distribution than the target intervention.

All nulls must receive the same layer-search budget as the real intervention.

---

## 11. Stage A — Intervention Validity

### Sample size

20–50 samples are sufficient for implementation validation.

### Required outputs

- `architecture_causal_graph.md`
- `token_layout.json`
- `no_op_parity.csv`
- `read_reconstruction.csv`
- `write_reconstruction.csv`
- activation plausibility diagnostics

### Gate

Proceed only if:

- no-op parity passes;
- READ and WRITE reconstruction identities pass;
- the four states are deterministic and numerically stable;
- full benchmark scoring is reproduced.

---

## 12. Stage B — Discovery

### Design

- 100–200 new samples
- two task families
- a sparse, predetermined layer grid, such as every fourth layer
- all four READ/WRITE states
- primary and secondary answer margins

### Analyses

- four-cell margin distributions;
- conditional READ and WRITE effects;
- interaction;
- full-correct vs full-wrong strata;
- task-family differences;
- numerical-noise and structured-null estimates;
- candidate layer band selection.

### Outputs frozen for confirmation

- `L*`: layer grid/band
- primary operation target: READ, WRITE, or both
- `epsilon`: minimum effect magnitude
- confirmatory search statistic
- null hierarchy
- option-permutation protocol
- sample-level primary endpoint

No final prevalence claim is allowed from this stage.

---

## 13. Stage C — Held-Out Confirmatory Test

### Primary endpoint

On the held-out confirmatory set:

> Does the exact visual READ/WRITE intervention produce an excess prevalence of sample-level negative signed effects beyond the search-adjusted structured null?

Operationalize with the frozen sample-level statistic and threshold from Stage B.

### Required confirmatory controls

1. option-order permutation;
2. A/B/C/D label permutation;
3. content-based option scoring;
4. search-adjusted covariance/subspace-matched null;
5. prompt paraphrase on a predefined subset;
6. sample- or image-clustered confidence intervals.

### Required reporting

- excess prevalence over null;
- effect-size distribution;
- confidence intervals;
- `M_max`, `M_lse`, and pairwise-margin agreement;
- full-correct regression and full-wrong correction separately;
- interaction prevalence;
- sensitivity to the effect threshold.

### Decision gate

Proceed to mechanism claims only if the confirmatory effect:

- exceeds the structured search-adjusted null;
- is not driven only by near-zero margins;
- follows answer content under option/label permutations;
- appears at non-trivial sample-level prevalence with uncertainty reported.

Otherwise pivot to an answer-silent redundancy analysis or stop.

---

## 14. Stage D — Mechanistic Replication

Focus on **one primary mechanism path**, chosen from the discovery and confirmatory results:

- READ-mediated distractor injection, or
- WRITE-mediated evidence erosion.

Do not attempt every proposed mechanism in the main paper.

## 14.1 Exact add-back and rescue

At the validated additive hook:

- remove the exact component;
- add the exact component back;
- compare with structured null residuals;
- verify reconstruction at the hook and at final logits.

Report the proportion of all predeclared cases for which the original margin is restored by at least a fixed fraction.

## 14.2 Dose response

For the exact component `delta`, evaluate:

```text
state(alpha) = state_no_component + alpha * delta
alpha in {0, 0.25, 0.5, 0.75, 1.0, 1.25}
```

Report full curves, not only selected examples.

## 14.3 Visual grounding

Use a minimal, matched grounding protocol:

1. answer-determining counterfactual image pair when available;
2. target-region intervention plus equal-area non-target control;
3. one soft intervention, such as blur or inpainting, to reduce hard-occlusion artifacts;
4. shuffled/no-image only as a secondary stress test.

The strongest evidence is that the signed effect changes with the answer-determining visual content.

## 14.4 Optional secondary decomposition

Only after a WRITE effect is confirmed, decompose:

```text
delta_write = delta_write_attn + downstream-conditioned delta_write_ffn
```

Carefully respect the nonlinear dependence of FFN input on the attention residual. Do not assume independent additive sufficiency without reconstruction tests.

Only after a READ effect is confirmed, analyze head/token contributions or confounder trajectories.

---

## 15. Layer-Position Interpretation

The primary claim concerns the existence and prevalence of local total-suffix effects, not an intrinsic ranking of layer harmfulness.

Do not claim that early layers are more harmful than late layers without controlling for:

- suffix depth;
- number of future READ opportunities;
- task-specific remaining computation.

Optional secondary analyses:

- fixed-horizon state effect;
- effect at the first future READ;
- effect conditioned on remaining READ opportunities.

These are not required for the core paper unless layer ordering becomes a central claim.

---

## 16. Open-Ended Confirmation

Open-ended generation is not part of the primary confirmatory endpoint.

After the MC analysis succeeds, use a small choice-free verification subset with a precisely defined protocol:

- intervention during prompt/prefill only;
- fixed visual KV-cache construction;
- deterministic generation;
- reference-answer teacher-forced likelihood;
- exact match and semantic correctness reported separately.

Do not mix prompt-only and per-decoding-step interventions in the same claim.

---

## 17. Success Criteria

The primary claim is supported only if all P0 criteria pass.

### P0 — Required

1. exact no-op parity and algebraic reconstruction;
2. held-out negative signed effect beyond a search-adjusted structured null;
3. sample-level prevalence with clustered confidence intervals;
4. option/label permutation consistency;
5. effects larger than numerical and near-tie instability;
6. exact add-back/rescue replicated on a held-out mechanism set;
7. visual-content sensitivity under a matched grounding test.

### P1 — Strongly desirable

- coherent dose-response curves;
- agreement across `M_max`, `M_lse`, and pairwise margins;
- mechanism replication in both task families;
- one small second-model replication.

### P2 — Optional

- text-manifold contamination;
- multi-layer windows;
- long-form generation;
- extensive head/token spatial attribution;
- large cross-model evaluation.

---

## 18. Kill and Pivot Criteria

### Stop or weaken the harmfulness claim if:

- exact reconstruction fails;
- real effects do not beat search-adjusted structured nulls;
- effects disappear under option/label permutation;
- corrections are dominated by near-zero margin ties;
- exact component add-back does not restore the original state/logits at the validated hook;
- visual-content counterfactuals do not modulate the effect;
- prevalence is confined to a few cherry-picked cases.

### Pivot options

- **Answer-silent redundancy paper:** if exact interventions are mostly near-zero but stable.
- **Representation analysis:** if geometry changes are systematic but causal answer effects are weak.
- **No-paper / redesign:** if interventions are numerically invalid or indistinguishable from structured nulls.

Router training remains out of scope in all pivots for this project phase.

---

## 19. Main-Paper Experiment Budget

Keep the main paper compact.

### Main text

1. computational graph and exact intervention definitions;
2. validity/reconstruction table;
3. held-out 2x2 READ/WRITE effects;
4. search-adjusted null and sample-level prevalence;
5. option/label robustness;
6. one dominant mechanism with add-back and grounding;
7. one small second-task or second-model replication.

### Appendix

- all secondary metrics;
- activation plausibility diagnostics;
- extra prompts and option permutations;
- secondary operation decomposition;
- open-ended confirmation;
- additional qualitative cases;
- optional text-manifold analysis.

---

## 20. Recommended Repository Layout

```text
project/
  configs/
    model.yaml
    discovery.yaml
    confirmatory.yaml
    mechanism.yaml
  audit/
    architecture_audit.py
    causal_graph.md
    token_layout.py
  interventions/
    read_path.py
    write_block.py
    four_state.py
    reconstruction.py
  scoring/
    option_scores.py
    margins.py
    calibration.py
  nulls/
    isotropic.py
    covariance_matched.py
    real_residual.py
    search_adjusted.py
  experiments/
    stage_a_validity.py
    stage_b_discovery.py
    stage_c_confirmatory.py
    stage_d_mechanism.py
  grounding/
    counterfactual_pairs.py
    matched_region_edit.py
  analysis/
    factorial_effects.py
    prevalence.py
    clustered_bootstrap.py
    dose_response.py
  outputs/
    stage_a/
    stage_b/
    stage_c/
    stage_d/
```

---

## 21. Agent Execution Rules

1. Never train a router or modify model weights.
2. Never use discovery examples as confirmatory prevalence evidence.
3. Never compare a best-of-many real search with a single random null.
4. Never call a top-1 flip a harmful mechanism.
5. Verify exact reconstruction before add-back interpretation.
6. Preserve all text computation; intervene only on the declared visual READ/WRITE paths.
7. Store all option scores and all four factorial cells, not only successful outputs.
8. Report failure cases and unverified flips.
9. Stop at every stage gate before expanding scope.
10. Prefer one replicated causal mechanism over many weak post-hoc explanations.
