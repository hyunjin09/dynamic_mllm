# Dynamic MLLM Experiment History: Plan v2 to the Current Frozen Result

**Coverage:** Plan v2 through the completed frozen-model query-conditioned
visual-refinement experiment

**Model:** `Qwen/Qwen2.5-VL-7B-Instruct`

**Pinned revision:** `cc594898137f460bfe9f0759e9844b3ce807cfb5`

**Precision/runtime:** frozen BF16 model; Transformers stock-eager decoder;
vision SDPA; PyTorch `2.6.0+cu124`; Transformers `4.51.3`

**Current project decision:** `STOP_QUERY_REFINEMENT_DIRECTION`

**Last updated:** 2026-08-07

## 1. How to read this history

This document reconstructs the full experimental chain from the original v2
READ/WRITE causal-analysis plan to the current negative frozen-model refinement
result. It distinguishes four kinds of evidence:

1. **Technical validity evidence:** parity, reconstruction, token-layout,
   deterministic-execution, and architectural checks. These determine whether
   an intervention is interpretable, not whether a scientific hypothesis is
   true.
2. **Inspected discovery evidence:** results used to identify patterns or
   candidate endpoints. These are not held-out prevalence or generalization
   evidence.
3. **Held-out confirmatory evidence:** a prospectively frozen endpoint evaluated
   on uninspected data.
4. **Outcome-blind calibration or deterministic reanalysis:** work that did not
   inspect new answer outcomes. This includes null fitting, donor-coverage
   audits, protocol preflights, and reanalysis of already inspected values.

The project repeatedly found positive-looking raw means or oracle advantages.
Those numbers are preserved here, but they are not promoted beyond the gates
they actually passed. In particular:

- a top-1 correction is not treated as proof of a harmful mechanism;
- a maximum over inspected actions/layers is not treated as a deployable policy;
- a held-out mean effect is not called mechanism-specific when structured nulls
  explain it equally well;
- heavy-tailed raw means are reported alongside medians and trimmed means;
- deterministic reanalyses do not count as new samples or replications.

## 2. Chronological overview

| Phase | Evidence type | Benchmarks/data | New scientific samples | Main result | Decision |
|---|---|---|---:|---|---|
| v2 Stage A | Technical validity | GQA, ChartQA, DocVQA, TextVQA | 23 usable of 24 requested | Exact FULL parity, READ/WRITE reconstruction, and deterministic four-state execution passed | Stage B gate passed |
| v2 Stage B | Inspected discovery | GQA 200; TextVQA 200 | 400 | Early WRITE was strongly answer-aligned; strongest negative READ candidate was TextVQA layer 0 with WRITE on | Freeze narrow Stage C candidate |
| v2 Stage C | Held-out confirmation | TextVQA | 800 unique images | Primary mean replicated, but neither structured-null comparison passed | `Outcome B`; harmful READ path closed |
| v3 migration/reanalysis | Deterministic reuse of v2 Stage B | GQA 200; TextVQA 200 | 0 new | Complete four-action landscape was heterogeneous; fixed schedules recovered little of the inspected oracle gain | `PROCEED_TO_V3_PREFLIGHT` |
| v3 preflight/null repair | Outcome-blind technical/calibration work | Stage B geometry, prospective GQA/TextVQA pools | 0 answer-scored samples | Search-matched null protocol could not yet be frozen; common padding repaired query-invariance numerics | Repair null design |
| v3 independent null redesign | Outcome-blind geometry extraction | 2,000 GQA + 2,000 TextVQA unique images | 4,000 geometry-only; 0 answer outcomes | Empirical donor coverage and covariance fidelity still failed frozen gates | `STOP_V3_CONFIRMATION` |
| v4 same-image discovery | Inspected discovery | GQA | 120 images, 240 questions | Same-image action values differed across questions, but the image+query oracle gap was small | Stop before confirmation; examine cost frontier |
| v4 cost–utility analysis | Deterministic reanalysis of v4 | Same 240 GQA questions | 0 new | Query conditioning had little sustained pooled compute-allocation headroom | `STOP_DYNAMIC_POLICY_DIRECTION` |
| Frozen query-refinement test | Inspected falsification experiment | GQA | 100 images, 200 questions | Replay operator was valid, but target-question specificity failed at all three anchors | `STOP_QUERY_REFINEMENT_DIRECTION` |

The unique scientific-outcome cohorts should not be summed naively because v3
reused v2 Stage B and the v4 cost analysis reused v4 discovery. The main new
outcome-bearing cohorts were: v2 Stage A validity samples, v2 Stage B discovery,
v2 Stage C confirmation, v4 discovery, and query-refinement discovery. The v3
4,000-image pool contained geometry only and deliberately excluded answers and
terminal action values.

---

## 3. Plan v2: signed visual READ/WRITE contributions

### 3.1 Why v2 was undertaken

Plan v2 asked whether visual participation inside a frozen decoder MLLM could
have signed answer value at a fixed pre-layer state. It separated two roles:

- **READ (`T <- V`)**: the visual-value contribution to text-query attention;
- **WRITE (`V -> V'`)**: the current decoder block's update to visual-token
  rows.

At one selected layer, four actions were defined from the same cached dense
pre-layer state:

| Name | READ | WRITE | Meaning |
|---|---:|---:|---|
| `IGNORE` | 0 | 0 | remove the path-specific visual READ and restore visual rows to their pre-layer values |
| `READ_ONLY` | 1 | 0 | preserve text's visual READ but restore pre-layer visual rows |
| `WRITE_ONLY` | 0 | 1 | preserve the visual block update but remove the current text visual-value path |
| `FULL` | 1 | 1 | reproduce the original dense model |

Every branch had to run the unchanged dense suffix. The intended progression
was validity (A), discovery (B), held-out confirmation (C), and mechanistic
verification (D). The plan explicitly prohibited router training, base-model
fine-tuning, broad pre-validity sweeps, and interpreting a wrong-to-correct
top-1 flip as a harmful mechanism.

The plan originally described multiple-choice answer margins. The selected GQA
and TextVQA records were actually open-ended and had no explicit distractor
options. Rather than create synthetic distractors or change datasets, Stage B
was explicitly amended to use teacher-forced accepted-reference likelihood.
This changed the answer-utility diagnostic but did not change the four
counterfactual actions.

### 3.2 Stage A — intervention validity

#### Purpose

Stage A tested whether the causal graph and interventions were real properties
of the pinned model implementation rather than assumptions in the plan. No
large discovery sweep was allowed until this gate passed.

#### Benchmarks and samples

The source pool was the local Qwen2.5-VL `easy_hard_5k` collection. The 24
requested records were balanced as three inherited-correct and three
inherited-wrong examples from each of four benchmarks:

| Benchmark | Requested | Completed | Inherited-correct / inherited-wrong completed |
|---|---:|---:|---:|
| GQA | 6 | 6 | 3 / 3 |
| ChartQA | 6 | 6 | 3 / 3 |
| TextVQA | 6 | 6 | 3 / 3 |
| DocVQA | 6 | 5 | 3 / 2 |
| **Total** | **24** | **23** | **12 / 11** |

The validated layers were `[0, 14, 27]`. One DocVQA record with a 16,314-token
prompt was excluded because faithful stock-eager decoder attention exceeded a
48 GiB RTX A6000. The validated maximum prompt length was 4,861 tokens.

The inherited easy/hard or correct/wrong labels were used only as sampling
metadata. A fresh pinned-model check later showed one GQA label drift (`boy`
versus `boys`), so inherited labels were never treated as analysis outcomes.

#### Architecture facts established

- The language decoder has 28 layers, hidden width 3,584, 28 query heads, four
  key/value heads, and seven query groups per KV head.
- Visual tokens precede question tokens in the prompt.
- The causal mask gives visual query rows zero attention mass to later question
  tokens.
- Same-layer text READ consumes pre-WRITE visual K/V.
- The primary READ intervention subtracts only the fixed-softmax visual-value
  path from nonvisual query rows. It preserves all nonvisual paths and avoids
  softmax-renormalization confounding.
- WRITE OFF restores pre-layer visual rows after the block while retaining the
  current layer's text/control output.
- All four branches clone the same cached pre-layer state and execute the same
  suffix.

#### Numeric validity results

| Check | Result |
|---|---:|
| Instrumented FULL layer max absolute difference | `0.0` |
| Instrumented FULL logit max absolute difference | `0.0` |
| READ reconstruction hook max absolute difference | `1.49e-08` |
| READ reconstructed suffix-logit max absolute difference | `0.0` |
| WRITE reconstruction hook max absolute difference | `0.0` |
| WRITE reconstructed suffix-logit max absolute difference | `0.0` |
| Repeated four-state logit max absolute difference | `0.0` |
| Option-logit and implemented option-score parity | `0.0` |
| Cached prestate injection max absolute difference | `0.0` |
| Maximum visual future-attention mass | `0.0` |
| Stored-evaluator reproduction | 23/23 |
| Instrumented FULL prediction agreement with pinned model | 23/23 |
| Inherited bucket score stability | 22/23 |

The READ OFF state required an exact representable `FULL - OFF` add-back in
BF16. Naively adding an ideal residual to an already rounded OFF state had
produced hook error `0.015625` and suffix-logit error `3.59375`; the validated
implementation bounded the required adjustment by the local half-ULP and
reconstructed exactly through the suffix.

#### Failed technical routes retained as evidence

- Stock eager could not fit the 16,314-token DocVQA case on the A6000.
- SDPA was rejected as a causal-decoder substitute because its eager-reference
  suffix-logit RMS difference was `0.03484`, above the frozen `0.0078125` gate.
- Query-chunked eager was rejected because its stock-eager suffix-logit RMS
  difference was `0.12424` at a 4,793-token boundary case.
- These failures restricted the validated runtime domain; tolerances were not
  relaxed to make an alternate backend pass.

#### Stage A conclusion

All required gates passed on 23 samples, so Stage B was authorized within the
documented stock-eager and prompt-length domain. This established intervention
validity, not a scientific effect.

Primary evidence: `outputs/stage_a/stage_a_summary.json`,
`outputs/stage_a/architecture_causal_graph.md`,
`outputs/stage_a/read_reconstruction.csv`, and
`outputs/stage_a/write_reconstruction.csv`.

### 3.3 Stage B — 400-record reference-likelihood discovery

#### Why this experiment was run

After Stage A passed, Stage B looked for stable signed READ/WRITE effects and a
narrow candidate for held-out confirmation. Because GQA and TextVQA records
were open-ended, no synthetic distractors were constructed. The prompt remained
open-ended, and only accepted-answer tokens contributed to the teacher-forced
score.

#### Benchmarks, selection, and scale

- GQA: 100 inherited-easy/correct plus 100 inherited-hard/wrong records.
- TextVQA: 100 inherited-easy/correct plus 100 inherited-hard/wrong records.
- Total: 400 records and 400 unique effective image assets.
- Layers: `[0,4,8,12,16,20,24,27]`.
- Four actions at every layer.
- Complete scale: 3,200 sample-layer matrices and 12,800 action cells.
- Technical exclusions: zero.

The inherited buckets controlled sampling only. Fresh pinned FULL evaluation
produced:

- GQA: 98 correct and 102 wrong;
- TextVQA: 104 strictly correct and 96 not-strictly-correct;
- six TextVQA records had partial consensus scores, three at `1/3` and three at
  `2/3`, and were retained numerically but not counted as strictly correct.

#### Exact scoring

For accepted answer tokens `y_1...y_T`, the primary within-sample score was

`S(y) = sum_t log p(y_t | image, question, y_<t)`.

The per-token mean `S(y)/T` was the primary cross-sample aggregation metric.
Prompt, image, padding, and EOS tokens did not contribute.

- GQA used one normalized canonical reference.
- TextVQA used official EvalAI/VQA normalization, duplicate-answer frequency
  weights, and weighted log-sum-exp across accepted answers. The same answer
  set and weights were used for all branches.
- The intervention hook was active only during prompt/prefill construction and
  was removed before teacher-forced answer continuation and greedy decoding.
- Greedy decoding was deterministic with `max_new_tokens=32` and repetition
  penalty `1.05`.

The final validity run made 114 no-op comparisons. Both sequence and per-token
absolute-difference p99 values were zero, freezing numerical floors of `1e-5`
nats/sequence and `1e-6` nats/token.

An initial 109-record partial run was discarded from analysis because its
secondary cached greedy implementation omitted the pinned repetition penalty.
The teacher-forced scores in that attempt were not used to avoid mixing
superseded and final outputs.

#### Main numeric findings

**1. Early WRITE was strongly answer-aligned.** At layer 0, enabling WRITE
increased per-token reference support under both READ conditions:

| Dataset | WRITE effect with READ off | WRITE effect with READ on |
|---|---:|---:|
| GQA | `+0.544`, 95% CI `[+0.306,+0.806]` | `+0.599`, CI `[+0.314,+0.943]` |
| TextVQA | `+1.245`, CI `[+0.964,+1.547]` | `+1.219`, CI `[+0.925,+1.523]` |

This directly rejected the exploratory idea that early visual WRITE was
broadly answer-misaligned.

**2. The strongest candidate negative READ effect was TextVQA layer 0 with
WRITE enabled.** The contrast `FULL - WRITE_ONLY` was:

| Metric/stratum | Mean | 95% CI | Additional detail |
|---|---:|---:|---|
| Sequence likelihood, all 200 | `-0.196` | `[-0.374,-0.050]` | negative aggregate |
| Per-token likelihood, all 200 | `-0.052953` | `[-0.116,-0.008]` | candidate Stage C endpoint |
| Per-token, FULL-wrong subset (`n=96`) | `-0.098` | `[-0.233,-0.006]` | 59.4% negative beyond epsilon |

The all-sample per-token distribution was heavy-tailed: SD `0.407473`, median
`-0.000004`, 5% trimmed mean `-0.012113`, and 5th/95th percentiles
`-0.406769 / +0.125852`. The READ effect with WRITE off and the READ-WRITE
interaction had CIs crossing zero, so they were not collapsed into the same
claim.

**3. GQA layer 27 had a weaker correctness-stratified READ candidate.** At the
terminal layer, WRITE and interaction are structurally silent. In the 102
FULL-wrong GQA records, the per-token READ effect was `-0.077`, CI
`[-0.146,-0.004]`, with 54.9% negative. In the 98 FULL-correct records, it was
positive: `+0.022`, CI `[+0.011,+0.036]`. The all-sample mean and the
FULL-wrong sequence CI crossed zero, so this remained tentative.

**4. READ was heterogeneous, not globally negative.** Across-layer READ means
were near zero with CIs spanning zero in each dataset; no common negative layer
band was supported. WRITE across-layer means were positive.

**5. Sequence/per-token agreement was high but imperfect.** At the TextVQA
layer-0 candidate, threshold-label agreement was 89.5% and Pearson `r=0.637`.
At the GQA layer-27 candidate, agreement was 97% and `r=0.924`. Correlations
between answer length and layer-averaged effects were weak (`|r| <= 0.17`).

#### Secondary greedy outcomes

- TextVQA layer-0 READ removal: five strict wrong-to-correct changes and four
  strict correct-to-wrong changes.
- GQA layer-27 READ removal: three corrections and two regressions.
- Many likelihood improvements remained wrong.

These counts were treated as descriptive behavioral outcomes, not causal
corrections or accuracy evidence.

#### Stage B conclusion

Stage B supported early-WRITE alignment and identified TextVQA layer-0
`FULL - WRITE_ONLY` as the narrowest negative READ candidate. It did not
support harmful visual participation, a shared negative band, or confirmatory
prevalence. A new, nonoverlapping held-out test with structured nulls was
required.

Primary evidence: `reports/stage_b_reference_likelihood_implementation.md`,
`reports/stage_b_conclusion.md`, and `outputs/stage_b/analysis_v1/`.

### 3.4 Stage C — held-out TextVQA confirmation

#### Why this experiment was run

Stage C froze the single strongest Stage B candidate instead of searching new
datasets, layers, operations, or strata. The primary endpoint was TextVQA,
layer 0, conditional READ with WRITE on:

`U_i = mean_logprob_i(FULL) - mean_logprob_i(WRITE_ONLY)`.

Negative `U` means that enabling the current layer's READ reduced support for
the accepted reference. The primary success gate was an image-clustered 95%
bootstrap CI for the held-out mean entirely below zero. The practical threshold
`-0.05` nats/token was secondary.

#### Data and power

- 800 held-out TextVQA records.
- 800 unique images.
- No Stage B record or image overlap.
- Manifest frozen before intervention outcomes.
- 10,000 image-clustered bootstrap draws.

Power planning used the 200 Stage B TextVQA records. Based on the observed
magnitude `0.052953`, estimated sample sizes were 465 for 80% power, 623 for
90%, and 770 for 95%; estimated power at 800 was 95.7%. Under a conservative
`0.05` magnitude, the corresponding sizes were 522, 698, and 864, with 93.5%
power at 800. Because the discovery distribution was heavy-tailed, these were
planning calculations rather than guarantees.

#### Frozen controls and technical amendments

- Exact FULL/no-op parity and READ reconstruction were retained.
- A covariance/subspace-matched READ-residual null and a same-layer
  cross-sample real-residual null were frozen.
- Both nulls used the same effect orientation as the real intervention.
- The real intervention had to outperform both null families before the term
  “confirmed answer-misaligned READ effect” was allowed.
- A frozen wrong-answer contrast compared the accepted reference with the
  original FULL greedy wrong string.
- TextVQA accepted answers retained official frequency-weighted aggregation.

The original real-donor caliper `1.5` covered 798/800 targets. An explicitly
outcome-blind geometry audit found the exact minimum complete-coverage caliper
`19/12 = 1.5833333333333333`. Only two targets required the wider caliper; the
nearest eight donors were unchanged for the other 798. The weakest target,
`textvqa_validation_36174`, had only three donors under 1.5 and needed five
additional donors at the amended boundary. The amended all-800 comparison was
primary, and the original-caliper-supported 798 subset was prespecified as a
secondary sensitivity.

The secondary `Answer:` prefix condition required contextual continuation
tokenization so the exact literal `Answer: <accepted answer>` was reproduced.
This amendment affected only prefix robustness; it did not change the primary
endpoint.

#### Primary held-out result

| Quantity | Result |
|---|---:|
| Mean `U` | `-0.07294332` nats/token |
| Standard deviation | `0.88503527` |
| Median | `+0.00000097` |
| 5% trimmed mean | `-0.00281866` |
| 20% trimmed mean | `-0.00011059` |
| Image-clustered 95% CI | `[-0.14127645,-0.01710262]` |
| Fraction `U < 0` | `0.49625` |
| Fraction `U <= -0.05` | `0.13000`, CI `[0.10625,0.15375]` |

The frozen primary mean gate passed. This was legitimately described as a
**held-out reference-support replication**. The near-zero median and trimmed
means showed that the result was highly heavy-tailed rather than a typical
sample-wide shift.

#### Structured-null specificity

| Frozen comparison, real minus null | Mean | Image-clustered 95% CI | Gate |
|---|---:|---:|---|
| Covariance/subspace null | `-0.00757412` | `[-0.02833798,+0.01255147]` | Fail |
| Cross-sample real-residual null, all 800 | `+0.00168244` | `[-0.01228950,+0.01741056]` | Fail |
| Original-caliper sensitivity, 798 | `+0.00182635` | `[-0.01203445,+0.01741453]` | Fail |

Because both required paired CIs crossed zero, the actual READ-removal effect
was not distinguishable from structured residual perturbations. The wider
caliper's two extra targets did not explain the failure because the 798-target
sensitivity also failed.

#### Secondary and robustness results

- Sequence-sum effect: mean `-0.13544451`, CI
  `[-0.27602624,-0.01992088]`; directional agreement with the per-token
  endpoint 87.375%, Pearson `0.95644`, Spearman `0.84661`.
- Uniform rather than frequency-weighted accepted-answer aggregation: mean
  `-0.06833159`, CI `[-0.13572732,-0.01386681]`; primary sign agreement 91.875%.
- Contextual `Answer:` prefix: mean `-0.00485841`, CI
  `[-0.02332004,+0.01396219]`; sign agreement only 57.25%. The paired
  prefix-minus-primary effect was `+0.06808491`, CI
  `[+0.01305588,+0.13494431]`. This was a material prompt-sensitivity warning.
- FULL-wrong reference-versus-original-wrong-answer contrast (`n=189`): mean
  `Delta C=+0.41465425`, median `+0.01744709`, CI
  `[+0.17302903,+0.69316965]`; 56.085% above zero, CI
  `[49.206%,62.963%]`.
- Strict greedy transitions: 22 FULL-wrong to WRITE_ONLY-correct, 12
  FULL-correct to WRITE_ONLY-wrong, 48 wrong-to-different-wrong, 599 unchanged
  correct, and 119 unchanged wrong. The descriptive net was +10 correct over
  800 records.

The positive wrong-answer margin shift and +10 net correctness count were
preserved, but neither was allowed to override the failed structured-null
specificity gate.

#### Stage C conclusion

The exact frozen decision was **Outcome B**:

> The reference-support effect replicated, but it was not distinguishable from
> the frozen structured intervention nulls.

Therefore v2 established neither a confirmed answer-misaligned READ effect nor
a harmful visual participation mechanism. The previously planned Stage D
add-back, dose-response, and grounded-mechanism study was cancelled rather than
used to rescue the result.

Primary evidence: `reports/stage_c_results.md`,
`reports/stage_c_conclusion.md`, and
`reports/stage_c_frozen_outcome_b_closure.md`.

---

## 4. Plan v3: the complete four-action policy-conditional landscape

### 4.1 Why the project moved to v3

V2 focused on conditional READ or WRITE contrasts. V3 reframed each complete
sample-layer factorial as the value vector

`[Q(IGNORE), Q(READ_ONLY), Q(WRITE_ONLY), Q(FULL)]`.

The purpose was to ask whether the whole action landscape contained meaningful
heterogeneity that a fixed global or per-layer action schedule could not
explain. It did not reopen the v2 harmful-READ claim. All inspected v2 Stage B
and Stage C data became discovery evidence under v3.

### 4.2 Migration audit — no new inference

The audit verified that all 400 v2 Stage B records supplied complete v3
matrices:

- 200 GQA and 200 TextVQA records;
- eight layers `[0,4,8,12,16,20,24,27]`;
- 3,200 sample-layer matrices;
- 12,800 finite action cells;
- no missing or imputed values;
- FULL score parity exactly `0.0`;
- cached-prestate injection difference `0.0`;
- maximum recorded READ and WRITE reconstruction discrepancies `5.96e-08` and
  `2.98e-08`.

This was a deterministic artifact audit, not a new experiment. The migration
decision was `REUSE_AND_CONFIRM`, meaning “reuse the valid discovery matrices
and build a new confirmation protocol,” not “treat them as confirmation.”

### 4.3 v3 Stage B reanalysis of the 400 existing records

#### Quantities computed

For every sample-layer matrix, v3 computed the three non-FULL advantages,
best suppression gain

`G_l = max_{a != FULL}[Q_l(a)-Q_l(FULL)]`,

the best action, conditional READ and WRITE effects, and READ-WRITE interaction.
Per-token accepted-reference likelihood remained primary. Exact ties used a
deterministic audit order, while numerical near-ties used `1e-6` nats/token and
a secondary practical band of `0.05`.

#### Four-action heterogeneity

At layers 0–24, exact-epsilon multi-action ties were at most 2%. Practical
near-ties were much more common: 44–55% of GQA and 60–70% of TextVQA
sample-layer pairs had `G <= 0.05`. Even so, `G > 0.05` occurred in 37.2% of
all GQA pairs and 25.6% of TextVQA pairs.

The distribution was strongly heavy-tailed:

- GQA layerwise mean `G`: `0.076–0.165`; medians `0.004–0.017` at layers
  0–24; 20% trimmed means `0.028–0.045`.
- TextVQA layerwise mean `G`: `0.040–0.146`; medians `0.001–0.007`; 20%
  trimmed means `0.013–0.021`.
- Layer 27 was mainly answer-silent/redundant because WRITE is structurally
  silent: 65.5% of GQA and 62.0% of TextVQA were within the no-op threshold.

FULL-wrong strata contained most of the discovery oracle gain. Averaging
layerwise summaries, GQA FULL-wrong `G` mean/median/trimmed-20 was
`0.191/0.098/0.117`, versus `0.013/0.001/0.005` for FULL-correct. TextVQA
was `0.108/0.035/0.045` versus `0.014/0.0004/0.002`.

At TextVQA layer 0, exact best actions were:

| Action | Count out of 200 |
|---|---:|
| `FULL` | 86 |
| `WRITE_ONLY` | 85 |
| `READ_ONLY` | 19 |
| `IGNORE` | 10 |

`WRITE_ONLY - FULL` had mean `+0.05295`, median `+0.0000035`, and 20%-trimmed
mean `+0.00565`. Only 10 of the 85 WRITE_ONLY wins had a WRITE-off action
within 0.05, showing that the v2 contrast was one edge inside a broader
WRITE-on-dominated landscape rather than generic equivalence among all
non-FULL actions.

#### Fixed-policy test

All policies were fitted and evaluated on the same inspected discovery set, so
these values are optimistic descriptions rather than held-out policy estimates.

| Policy | Mean utility vs FULL | 95% CI | Oracle-action match | Mean regret to sample-layer oracle |
|---|---:|---:|---:|---:|
| One global action (`FULL`) | `0` | — | 0.340 | `0.0976` |
| One action per layer | `+0.0045` | `[-0.0014,+0.0105]` | 0.348 | `0.0931` |
| One action per dataset and layer | `+0.0075` | `[-0.0005,+0.0159]` | 0.333 | `0.0902` |
| Always FULL | `0` | — | 0.340 | `0.0976` |
| Sample-layer oracle | `+0.0976` | `[+0.0814,+0.1139]` | 1.000 | `0` |

The best per-layer schedule chose WRITE_ONLY at layers 0, 20, and 24 and FULL
elsewhere. Best actions varied across layers for 198/200 GQA and 197/200
TextVQA samples. The dataset/layer schedule closed only `0.0075` of the
`0.0976` inspected oracle gap.

Behaviorally, the per-dataset/layer schedule produced 14 FULL-correct
regression pairs and 29 FULL-wrong improvement pairs. The sample-layer oracle
produced eight regression pairs and 109 improvement pairs. These are repeated
sample-layer observations, not sample-level accuracy estimates.

#### READ-WRITE interaction

Across all layers:

| Quantity | GQA | TextVQA |
|---|---:|---:|
| READ sign reversed between WRITE states | 26.6% | 29.1% |
| WRITE sign reversed between READ states | 24.7% | 15.4% |
| Independent main effects missed exact best action | 22.9% | 22.6% |
| Interaction magnitude exceeded `0.05` | 40.5% | 27.9% |
| Interaction exceeded numerical no-op floor | about 87% | about 87% |

Medians were usually near zero, so interaction was material for a minority and
heavy-tailed rather than a uniform global effect.

Sequence/per-token `G` agreement was high for GQA (Pearson `0.944`, Spearman
`0.994`, sign label `0.978`) and lower but still substantial for TextVQA
(Pearson `0.786`, Spearman `0.867`, sign label `0.890`).

#### Same-image feasibility

The 400 Stage B records used 400 unique effective image assets, so no
same-image query analysis was possible in that dataset. Prospective metadata
contained 9,800 GQA images with at least two eligible questions and 1,243
TextVQA two-question images after excluding the 800 inspected Stage C images.

#### v3 discovery conclusion

The four-action landscape was genuinely heterogeneous beyond exact numerical
ties and was poorly approximated by fixed schedules. However, the v2 Outcome B
warned that a maximum over 21 layer/action opportunities could also select
generic perturbation effects. V3 therefore required every structured null to
receive the same seven-layer by three-action search budget. The decision was
`PROCEED_TO_V3_PREFLIGHT`, not confirmation.

Primary evidence: `reports/v3_stage_b_reanalysis.md` and
`reports/v3_stage_b_decision.md`.

### 4.4 v3 held-out preflight — no held-out outcomes opened

The proposed confirmatory statistic was

`S_real(x) = max over layers [0,4,8,12,16,20,24] and three non-FULL actions of Q(a)-Q(FULL)`.

Layer 27 was excluded because of structural WRITE silence. The proposed design
used 1,600 unique images, balanced 800 GQA and 800 TextVQA, with a 10,000-draw
stratified image bootstrap. Candidate-pool audit found 10,234 eligible GQA
images and 2,362 eligible TextVQA images; 800 multi-question image groups per
dataset were reserved separately for a future same-image analysis.

The data split, real four-action mechanics, FULL parity, accepted-answer
scoring, and serialization passed. Three unresolved gates prevented freezing
the final manifest:

1. the complete paired READ/WRITE real-donor index and outcome-blind caliper
   were not frozen;
2. null draw count and joint covariance seed stability were not frozen;
3. a sufficient visual-grounding control was not frozen.

The query-invariance audit also exposed a numerical issue. Causally, visual
rows cannot attend to later questions, but unequal BF16 prompt shapes produced
different visual WRITE values that accumulated across layers. Common
right-padding from length 273 to 281 restored exact equality at all seven
layers. This led to a separate prospective padding amendment; it did not alter
answer content or the main v3 endpoint.

The preflight decision was `REPAIR_V3_NULL_DESIGN`. No held-out terminal action
value was computed or inspected.

### 4.5 v3 null repair using the inspected Stage B geometry

#### Calibration scale

The repair extracted exact paired READ/WRITE residual geometry from the 400
inspected Stage B records at seven layers:

- 2,800 complete sample-layer residual pairs;
- 200 records per dataset;
- maximum reconstruction error `5.96e-08`;
- no answer likelihood, generated correctness, or action value used in fitting.

Fourteen path-specific PCA models and joint standardized-score covariance
models were fitted with five-fold geometry-only cross-validation. Geometry
precision froze four isotropic draws, 16 joint-covariance draws, and eight real
donors, conditional on the null families passing.

Official-annotation grounding eligibility identified 123 GQA and 130 TextVQA
records meeting the prospective unambiguous-target/matched-control rules, but
no grounding intervention was executed.

#### Failed gates

**Joint covariance null:** after decoding, native-row remapping, and exact
row-norm matching, final-native subspace relative error was `0.634965` at GQA
layer 16 and `0.531477` at GQA layer 24, above the frozen `0.50` gate. Joint
coordinate covariance, norms, conditioning, finiteness, and deterministic
serialization otherwise passed.

**Paired real-residual donor null:** all GQA layers and TextVQA layer 4 needed
an eighth-donor caliper above the prospective tight/local rule. The worst cap
was `2.187192` for GQA sample `gqa_gh_02672825` at layer 16. Sixty-four
target-layer rows (59 GQA, five TextVQA) exceeded `1.5`. READ norm dominated 29
weak pairs and row/image-token geometry dominated 23, so the failure was not a
single removable outlier.

The decision was `PIVOT_BEFORE_CONFIRMATION`: either authorize an independent,
larger geometry-only calibration redesign or close v3. Thresholds and calipers
were not weakened.

### 4.6 Independent calibration-pool null redesign

#### Why it was run

One bounded outcome-blind attempt tested whether a larger, image-disjoint donor
pool and better native-row representations could repair both null families.
The real statistic, seven layers, three actions, maximum-over-21 rule, donor
count eight, matching distance, exact norm matching, and final-native fidelity
threshold `0.50` all remained fixed.

#### Data and extraction

- Initial pool: 1,000 unique GQA and 1,000 unique TextVQA train images.
- Prospective enlargement: 2,000 unique images per dataset, 4,000 total.
- Geometry extracted at seven layers: 28,000 paired READ/WRITE residuals.
- Maximum reconstruction error: `5.96e-08`.
- No answer fields in the manifest; no answer likelihood, generated answer,
  correctness, or four-action Q value was computed.

#### Empirical donor result

At 1,000 images per dataset, the minimum global eighth-neighbor caliper was
`2.625`, with 58 of 14,000 target-layer rows above `1.5`. After enlargement:

| Dataset | Eighth-neighbor median range | 99th-percentile range | Worst distance | Rows above 1.5 |
|---|---:|---:|---:|---:|
| GQA | `1.0734–1.1036` | `1.2583–1.3991` | `3.09375` | 38 |
| TextVQA | `1.0651–1.1005` | `1.1829–1.2962` | `1.6419` | 3 |

Bulk matching improved, but rare GQA shapes made the exact complete-coverage
caliper worse. A global `3.09375` cap was not a local repair under the frozen
`1.5/1.6` rule.

#### Covariance-representation result

Three prospectively specified representations were tested:

| Representation | Outcome |
|---|---|
| A: fixed 32-row path PCAs | Failed native WRITE fidelity in multiple strata; joint covariance Monte Carlo error `0.1578–0.1689` exceeded `0.15` |
| B: exact native-row strata | Failed coverage; sufficiently populated shapes covered only 48.2% of GQA and 50.5% of TextVQA |
| C: native-row distribution | Failed out-of-sample native geometry fidelity even at rank cap 1,024 |

For representation C at rank 1,024, remaining errors included TextVQA layer-4
WRITE `0.6367`, TextVQA layer-8 READ/WRITE `0.5589/0.5601`, and GQA layer-8
READ/WRITE `0.5461/0.5227`, all above `0.50`.

#### v3 final conclusion

Neither keeping both nulls nor promoting the empirical donor null alone was
defensible: the donor null itself lacked complete tight matching. The gates
were not relaxed after failure. The exact decision was:

`STOP_V3_CONFIRMATION`.

This does **not** show that the maximum-over-21 scientific effect is zero. It
shows that the planned causal-specificity claim could not be tested with a
valid frozen null hierarchy. V3 ended as a protocol-validity negative result,
with the broader causal claim technically unresolved.

Primary evidence: `reports/v3_null_repair_report.md` and
`reports/v3_null_redesign_v2.md`.

---

## 5. Plan v4: same-image query-conditioned value

### 5.1 Why v4 was scientifically distinct

V3 asked whether a best-of-21 suppression statistic was unusually beneficial
relative to generic structured perturbations. V4 dropped the harmfulness and
specificity claim. It instead asked whether two questions about the same image
assign different values to the same exact visual state and visual WRITE.

This was motivated by two established facts:

1. visual tokens precede question tokens, so visual query rows cannot causally
   consume the later question; and
2. common padding made same-image visual states and WRITE bitwise identical.

Thus any within-image difference in downstream action value could be attributed
to how the later text computation used the same visual state, without claiming
that a negative action was harmful.

### 5.2 v4 common-padding entry gate

The outcome-blind preflight used 12 of the frozen discovery images, 24 natural
questions, and all seven layers `[0,4,8,12,16,20,24]`. It retained no terminal
four-action values.

Across 1,176 identity/control comparisons:

- pre-layer visual states, post-layer visual states, and WRITE residuals were
  bitwise identical within every same-image pair (`max_abs=0`);
- visual-query attention to question or padding keys was exactly zero;
- common padding changed no nonpadding token and was masked from attention and
  scoring;
- instrumented FULL, READ reconstruction, WRITE reconstruction, answer spans,
  and deterministic four-action execution passed;
- sequence and per-token identity-score p99 were both zero, preserving
  `1e-5` and `1e-6` noise floors.

The first Slurm submission failed before CUDA initialization on one node. An
unchanged retry completed on the validated node, so this was recorded as an
environment failure rather than a scientific failure.

### 5.3 v4 GQA same-image discovery

#### Data and execution

- 120 new, image-disjoint GQA images.
- Exactly two natural questions per image: 240 questions.
- 60 pairs had resolved disjoint scene-object evidence; 60 were
  metadata-matched comparison pairs.
- Seven layers `[0,4,8,12,16,20,24]`.
- Four actions at one layer at a time.
- 1,680 complete question-layer Q matrices.
- 6,720 finite action scores.
- No sample replacement or outcome-dependent exclusion.
- Per-token accepted-reference likelihood was primary; the image was the
  bootstrap unit; 10,000 bootstrap draws were used.

The planned paraphrase arm was not run: no official GQA equivalent pair met the
prospective same-answer, question-type, and semantic-target requirements. No
generated paraphrase was introduced after outcomes.

#### Primary results

The joint row was an equal-layer average, not a maximum over layers.

| Equal-layer image-level quantity | Mean and 95% CI | Median | 20% trimmed mean |
|---|---:|---:|---:|
| Robust epsilon-set best-action disagreement | `0.6762 [0.6405,0.7119]` | `0.7143` | `0.6905` |
| Bidirectional cross-query action-transfer regret | `0.1121 [0.0870,0.1401]` | `0.0634` | `0.0711` |
| Image+query minus image-only oracle gap | `0.0144 [0.0104,0.0188]` | `0.00344` | `0.00644` |
| FULL-relative four-action vector distance | `0.6405 [0.5244,0.7722]` | `0.4244` | `0.4775` |
| Four-action variance | `0.2353 [0.1136,0.3892]` | `0.0165` | `0.0300` |

The epsilon-tie ambiguity rate was only `0.0262`; exact-argmax disagreement was
`0.6833`. Thus frequent action disagreement was not caused mainly by numerical
ties.

Conditional sign-reversal rates were also high: `0.4762` and `0.4583` for READ
under WRITE off/on, and `0.4440` and `0.4726` for WRITE under READ off/on.
These were descriptive interaction patterns, not causal semantic mechanisms.

#### Layerwise results

| Layer | Robust disagreement | Transfer regret | Query-oracle gap |
|---:|---:|---:|---:|
| 0 | 0.5667 | 0.2399 | 0.0260 |
| 4 | 0.6917 | 0.0877 | 0.0106 |
| 8 | 0.7250 | 0.1221 | 0.0203 |
| 12 | 0.7333 | 0.1158 | 0.0171 |
| 16 | 0.6750 | 0.0911 | 0.0109 |
| 20 | 0.6750 | 0.0776 | 0.00921 |
| 24 | 0.6667 | 0.0502 | 0.00651 |

No action dominated globally; layer-specific exact-best frequencies ranged
from 13.75% to 37.50% across actions.

#### Robustness and semantic controls

- On 108 images with no epsilon ambiguity at any layer, transfer-regret
  mean/median/trimmed-20 was `0.1137/0.0679/0.0767`.
- On 98 answer-length-matched images, it was `0.1035/0.0573/0.0652`.
- On the strict 40-image question-type, program-depth, answer-format, and
  answer-length matched subset, it was `0.0952/0.0440/0.0511`.
- Removing the largest 5% left mean transfer regret `0.0854`; the top 5% still
  contributed 27.6% of total regret.
- Only 10% of images had an equal-layer image+query oracle gap at least `0.05`.
- Sequence/per-token Pearson correlations were `0.933` for transfer regret and
  `0.934` for the query-oracle gap.

Different-evidence minus matched-comparison estimates were mixed:

| Quantity | Difference and 95% CI | Covariate-adjusted coefficient |
|---|---:|---:|
| Robust disagreement | `+0.0667` (approximately `[0,+0.1310]`) | `+0.0494` |
| Vector distance | `+0.1192 [-0.1245,+0.3754]` | `+0.0226` |
| Transfer regret | `+0.00335 [-0.0483,+0.0557]` | `-0.0353` |
| Query-oracle gap | `-0.00792 [-0.0162,-0.00007]` | `-0.00636` |

The semantic ordering therefore did not support the stronger claim that
questions requiring different visual evidence necessarily produced a larger
practical image-only insufficiency gap.

#### Discovery interpretation

Supported: question-associated four-action variation existed under identical
same-image visual computation. Not supported: a semantic mechanism, large
image-only-policy insufficiency, deployable routing, acceleration, or accuracy
improvement. Because the direct query-oracle gap was small and the paraphrase
control was missing, the immediate decision was `STOP_BEFORE_V4_CONFIRMATION`.

Primary evidence: `reports/v4_common_padding_preflight.md`,
`reports/v4_discovery_results.md`, and `reports/v4_discovery_decision.md`.

### 5.4 v4 query-conditioned versus image-only cost–utility frontier

#### Why this deterministic analysis was run

Frequent action disagreement and transfer regret did not necessarily imply a
useful compute-allocation advantage. The same 1,680 Q matrices were therefore
reanalyzed with exact local operation FLOPs. No model inference, new sample, or
paraphrase experiment was run.

The analysis compared an image-only oracle, which had to choose one shared
action for both questions, with an image+query oracle, which could choose a
different action for each question under the same compute price or budget.

#### Local action costs

With hidden width 3,584 and MLP width 18,944, the mean local action-dependent
visual costs were:

| Action | Mean GFLOPs | Median GFLOPs | Mean increment vs IGNORE | Mean difference vs FULL |
|---|---:|---:|---:|---:|
| `IGNORE` | 0 | 0 | 0 | `-123.4557` |
| `READ_ONLY` | `0.05146` | `0.04490` | `+0.05146` | `-123.4043` |
| `WRITE_ONLY` | `123.4043` | `107.7925` | `+123.4043` | `-0.05146` |
| `FULL` | `123.4557` | `107.8310` | `+123.4557` | 0 |

Mean action-invariant local computation was `21.5831` GFLOPs. READ was only
`0.0419%` of FULL's action-dependent visual cost; WRITE dominated. These were
operation-level FLOPs for a hypothetical sparse implementation, not measured
wall-clock acceleration. The actual intervention runner executed dense layers
for every branch.

#### Unconstrained oracle comparison

| Quantity | Image-only oracle | Image+query oracle |
|---|---:|---:|
| Mean utility vs FULL | `0.10547` | `0.11984` |
| Median utility vs FULL | `0.02181` | `0.03738` |
| 20% trimmed utility vs FULL | `0.04372` | `0.05769` |
| Mean local visual compute | `61.5239` GFLOPs | `63.1171` GFLOPs |
| Mean normalized compute | `0.4964` | `0.5226` |
| FULL selection | 24.05% | 27.56% |

The query-conditioned mean utility increment was `0.01438`, median `0.00017`,
and 20%-trimmed `0.00173`. It used `1.5933` GFLOPs more, equal to 2.62% of
pair-specific FULL cost. On 568 robust-disagreement pairs it used 4.31% more
FULL-equivalent compute. This rejected the “image-only matches utility by
conservatively over-computing” explanation.

#### Pooled frontier results

| Scope | Integrated mean utility gap | Max utility gain at matched compute | Mean compute saving at matched utility | Utility targets saving >=10% FULL |
|---|---:|---:|---:|---:|
| All 840 image-layer pairs | `0.01486` | `0.02370` | 1.40% FULL | 3.40% |
| 568 robust non-tie/disagreement pairs | `0.02090` | `0.02298` | 1.47% FULL | 4.10% |

At no pooled normalized-compute point did the all-pair or robust mean gain reach
`0.05` nats/token. The largest pointwise matched-utility saving was 38.44% FULL,
but it occurred over only 3.40% of the attainable utility-target grid.

Aggregation sensitivities also remained below a practical utility gain:

| Scope/aggregation | Integrated gap | Maximum matched-compute gain | Mean matched-utility saving |
|---|---:|---:|---:|
| All / mean | `0.01486` | `0.02370` | 1.40% |
| All / median | `0.01447` | `0.01674` | 7.36% |
| All / 20% trimmed | `0.01313` | `0.01444` | 3.30% |
| Robust / mean | `0.02090` | `0.02298` | 1.47% |
| Robust / median | `0.02402` | `0.02730` | 6.69% |
| Robust / 20% trimmed | `0.01896` | `0.02036` | 3.58% |

At an exact local 0.50-FULL budget, discreteness produced a larger advantage:
image-only utility was `-0.04616`, image+query was `+0.08865`, and the mean gap
was `0.13481`. But its median and trimmed gaps were only `0.02432` and
`0.02601`, and the pooled maximum fell back to `0.02370` when compute could be
allocated across image-layer pairs.

Layer 0 was the only frontier with a raw practical pointwise excursion: maximum
mean gain `0.12743` over 18.38% of its compute axis. Its median and trimmed
maximum gains were only `0.01884` and `0.02203`, so it was not used to reopen a
layer search.

#### v4 local-policy conclusion

Query-specific action pairs frequently expanded the local frontier—92.14% of
image-layer pairs—and strictly dominated the epsilon-tied unconstrained shared
action in 40.71%. Nevertheless, their aggregate practical headroom was small.
The exact decision was:

`STOP_DYNAMIC_POLICY_DIRECTION`.

This closed routing over the existing four local actions. It did not deny the
descriptive same-image query dependence or make a claim about latency or other
models.

Primary evidence: `reports/v4_cost_utility_reanalysis.md`.

---

## 6. Strategic synthesis after v4

No experiment was run in this phase. The synthesis classified the surviving
evidence and asked why local routing had failed.

The explanations were ranked:

1. **Strongest support:** the existing four-action space had insufficient
   practical separation. Even an outcome-aware query oracle showed little
   pooled frontier headroom.
2. **Strong architectural fact:** current visual WRITE was query-blind. The old
   actions could retain or remove a visual update but could not make it consume
   the question.
3. **Moderate indirect support:** reference likelihood overstated practical
   variation, given heavy tails, prompt sensitivity, null non-specificity, and
   sparse greedy coherence.
4. **Technically unresolved:** single-layer interventions might miss multi-layer
   interactions, but testing more skip combinations would remain close to the
   closed local-routing direction.
5. **Weak/unsupported:** the router was merely difficult while large action
   headroom existed. The oracle bound argued against this explanation.

Three new capability directions were compared: post-question visual-token
refinement, high-resolution visual revisitation, and explicit query-writable
visual memory. The first was selected because it could add a missing
question-to-visual causal edge with frozen weights, existing visual evidence,
and no router training. The strategic decision was
`TEST_QUERY_CONDITIONED_VISUAL_REFINEMENT`.

Primary evidence: `reports/dynamic_mllm_v2_v4_synthesis.md` and
`workspace/dynamic_mllm_next_direction.md`.

---

## 7. Current experiment: frozen query-conditioned visual refinement

### 7.1 Why this experiment was run

The closed v4 action space only selected among query-blind visual operations.
The new hypothesis was that already encoded visual tokens might benefit from
one fixed-budget opportunity to consume the question. The experiment was
designed to separate target-question conditioning from an unconditioned replay
with identical dense computation and from conditioning on the wrong question
about the same image.

It did not re-encode or crop the image, use an object detector or OCR system,
add visual tokens, train a module, or alter model weights.

### 7.2 Frozen operator

At decoder layer `l`, the dense pass captured pre-layer state `H_l` and native
post-layer state `H_(l+1)`. One frozen native layer call was replayed:

- `BASELINE`: original dense model.
- `UNCONDITIONED_REPLAY`: native causal mask; the replayed visual output had to
  reconstruct native visual `H_(l+1)` exactly.
- `TARGET_QUERY_REPLAY`: add attention edges only from visual query rows to the
  literal target-question token span.
- `OTHER_QUERY_REPLAY`: use the paired same-image question states and expose
  the analogous other-question span, then insert its refined visual rows into
  the target state.

Only replay visual outputs were retained. Native target text/control rows were
unchanged, and the original dense suffix resumed at `l+1`. All replay variants
used identical token counts, tensor shapes, frozen layer weights, MRoPE
positions, residual/normalization paths, and dense FLOPs. No answer, padding,
assistant-prefix, or instruction token was deliberately exposed.

This output-boundary construction has an important limitation:
`UNCONDITIONED_REPLAY` is an exact reconstruction of the native layer, not a
second sequential visual-depth step. The experiment cleanly tests adding
question-to-visual attention edges while holding computation matched, but does
not answer whether generic additional sequential visual depth helps.

The anchors `[4,12,20]` were frozen as early/middle/late nonterminal locations
before outcomes. No broad layer or replay-depth search was performed.

### 7.3 Data and validity preflight

- Dataset: GQA validation.
- 100 new unique images.
- Exactly two questions per image: 200 questions.
- Zero image overlap with v2 Stage A, v2/v3 Stage B, v3 geometry calibration,
  and v4 discovery.
- 200/200 records and 600/600 sample-layer matrices completed.
- No replacement or outcome-dependent exclusion.

The technical preflight used 12 images and 24 questions. It passed at all three
anchors:

| Check | Result |
|---|---:|
| Same-image common-padded visual `H_l` and `H_(l+1)` identity | exact, max difference `0` |
| Unconditioned replay vs native visual output | exact |
| Unconditioned suffix logits and accepted-answer scores vs baseline | exact |
| Repeated replay scores | exact |
| B/C/D replay tensor shapes and analytic FLOPs | identical |
| Answer/padding leakage | none |
| Replay/native visual RMS ratio | `0.98643–1.00110` |
| Minimum replay/native cosine | `0.99725` |
| Largest replay/native difference relative to native WRITE RMS | `0.30176` |

The first GPU submissions on one node failed before model/sample loading
because CUDA exposed zero usable devices. They produced no scientific records;
the unchanged protocol completed on the validated node.

### 7.4 Frozen scientific contrasts and success rule

The primary continuous utility was per-token accepted-reference likelihood.

- Conditioning value: `Delta_condition = TARGET - UNCONDITIONED`.
- Target specificity: `Delta_target = TARGET - OTHER_QUERY`.
- `TARGET - BASELINE` was secondary; because UNCONDITIONED exactly reproduced
  BASELINE, it was numerically identical to the conditioning contrast.

For each anchor, each primary contrast required all of:

- mean at least `0.05` nats/token;
- image-clustered 95% mean CI above zero;
- positive median and 20% trimmed mean;
- positive fraction greater than 55%;
- positive mean after removing the largest 5% of absolute effects.

Both contrasts, plus no net correctness regression, had to pass at at least two
of the three anchors.

### 7.5 Likelihood results

| Layer | Contrast | Mean | Image-clustered 95% CI | Median | 20% trimmed | Positive fraction | Fraction >= 0.05 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | target - unconditioned | `+0.01918` | `[+0.00062,+0.04182]` | `+0.000021` | `+0.00212` | 0.540 | 0.165 |
| 4 | target - other question | `+0.01096` | `[-0.00048,+0.02400]` | `+0.000018` | `+0.00240` | 0.530 | 0.120 |
| 12 | target - unconditioned | `-0.01200` | `[-0.02654,+0.00349]` | `-0.000107` | `-0.00348` | 0.415 | 0.130 |
| 12 | target - other question | `-0.02352` | `[-0.03982,-0.00820]` | `-0.000119` | `-0.00510` | 0.460 | 0.105 |
| 20 | target - unconditioned | `+0.00255` | `[-0.00944,+0.01414]` | `+0.000035` | `+0.00279` | 0.575 | 0.160 |
| 20 | target - other question | `-0.00361` | `[-0.01531,+0.00788]` | `+0.000020` | `-0.00011` | 0.535 | 0.100 |

Layer 4 was the only positive conditioning mean with a CI above zero, but it
failed four critical robustness/specificity checks:

- mean `0.01918` was below the frozen `0.05` practical threshold;
- positive fraction 54.0% was below 55%;
- median and trimmed mean were close to zero;
- target-versus-other-question CI crossed zero.

Removing the largest 5% of absolute layer-4 conditioning effects reduced its
mean from `0.01918` to `0.00350`. Layer 12 moved in the wrong direction and had
a target-versus-other CI wholly below zero. Layer 20 was near zero and
uncertain. Zero anchors passed; the protocol required two.

### 7.6 Generated behavior and pair dependence

Target replay versus baseline/unconditioned replay:

| Layer | Wrong to correct | Correct to wrong | Net |
|---:|---:|---:|---:|
| 4 | 1 | 2 | -1 |
| 12 | 1 | 2 | -1 |
| 20 | 0 | 1 | -1 |

Target versus paired-other-question replay produced correction/regression
counts `0/2` at layer 4, `0/1` at layer 12, and `1/0` at layer 20.

Mean absolute within-image differences in `Delta_condition` were still
descriptively sizable: `0.0851`, `0.0864`, and `0.0689` at layers 4, 12, and
20. They did not align with semantic evidence separation. Spearman correlation
with scene-object-set distance was `-0.068`, `+0.040`, and `+0.034`;
correlation with question-length difference was `+0.020`, `+0.113`, and
`+0.196`. Sample-level correlations with question length, answer length,
baseline difficulty, prompt length, and visual-token count were all modest
(`|Spearman|` at most about `0.16`).

### 7.7 Current conclusion

Supported:

- a mathematically coherent, deterministic, frozen post-question visual-edge
  intervention can be implemented without new visual evidence or learned
  parameters;
- the intervention remains close to native activation geometry;
- local likelihood responses remain heterogeneous and heavy-tailed;
- layer 4 has a small positive raw conditioning mean.

Not supported:

- robust target-question-specific refinement value;
- target-question benefit beyond the paired wrong question;
- correctness improvement;
- proceeding to TextVQA replication;
- routing, acceleration, efficiency, harmfulness, or semantic-mechanism claims.

The exact final decision is:

`STOP_QUERY_REFINEMENT_DIRECTION`.

The positive layer-4 mean was not used to select that layer after the fact,
increase replay depth, search subgroups, or open TextVQA. This negative result
rules out this frozen replay operator under its prespecified three-anchor
protocol; it does not establish that every possible query-conditioned visual
architecture is impossible.

Primary evidence: `reports/query_refinement_preflight.md`,
`reports/query_refinement_gqa_discovery.md`, and
`reports/query_refinement_gqa_decision.md`.

---

## 8. What survived across the full project

### 8.1 Supported technical facts

- The path-specific READ and block-output WRITE interventions are valid at the
  documented hooks and reconstruct through the unchanged suffix.
- Instrumented FULL exactly reproduces the pinned model in the validated
  stock-eager domain.
- Under common right-padding, same-image visual states and visual WRITE are
  bitwise identical across different later questions.
- The frozen model can execute the tested post-question visual-query edge
  intervention stably without changing visual evidence or model weights.

### 8.2 Supported scientific observations, with scope limits

- Early layer-0 visual WRITE was strongly answer-aligned on average in the
  400-record GQA/TextVQA discovery.
- READ and WRITE have different functional and compute roles: early WRITE is
  highly valuable and compute-dominant; READ is cheap and heterogeneous.
- Four-action values and action rankings vary substantially across samples,
  layers, and same-image questions.
- Local intervention effects are repeatedly heavy-tailed: means can appear
  substantive while medians and trimmed means are near zero.
- Same-image question-associated action disagreement is real under the frozen
  numerical definition, but the practical pooled cost–utility gap is small.

### 8.3 Unsupported or closed claims

- A confirmed harmful or answer-misaligned TextVQA layer-0 READ mechanism.
- A shared harmful READ/WRITE layer band across GQA and TextVQA.
- A causally specific v3 maximum-over-21 suppression effect.
- A materially useful query-conditioned policy over the same four local
  actions.
- Robust target-question-specific value from the tested frozen replay operator.
- Accuracy improvement, acceleration, deployable routing, cross-task/model
  generalization, or a semantic causal mechanism.

### 8.4 Technically unresolved questions

- Whether a different, scientifically valid search-budget-matched structured
  null can ever be built for the v3 maximum-over-21 statistic without weakening
  specificity gates.
- Whether genuinely multi-layer, nonlocal interactions have useful headroom;
  these were not tested and are not authorized as an automatic fallback.
- Whether a different trained architecture with explicit query-writable visual
  memory could succeed; no such model was trained.
- Whether generic sequential extra visual depth helps; the tested
  unconditioned replay was an exact native reconstruction, not a second depth
  step.

## 9. Final evidence chain in one paragraph

V2 proved that the local READ/WRITE counterfactuals could be implemented
faithfully, found strong early-WRITE alignment and a negative TextVQA layer-0
READ discovery mean, and then showed on 800 held-out images that the mean
replicated but was not specific relative to structured residual nulls. V3
showed that the complete four-action landscape was heterogeneous and that
fixed schedules captured little of an inspected oracle gain, but its broader
causal confirmation stopped outcome-blind because neither required
search-matched null family passed frozen geometry gates. V4 then established
that different questions assigned different action values to identical
same-image visual states, yet exact cost–utility analysis showed too little
sustained practical headroom for local dynamic routing. Finally, a new frozen
operator added target-question-to-visual attention without new evidence or
training; it was technically valid but failed robust target-specific value at
all three frozen anchors. The current record therefore supports architectural
and descriptive insights about early WRITE, READ/WRITE asymmetry, query-blind
visual state, and heavy-tailed intervention response, while closing the tested
harmfulness, local-routing, and frozen query-refinement claims.

## 10. Canonical evidence index

### v2

- Source plan: `plans/dynamic_mllm_read_write_causal_analysis_plan_v2.md`
- Stage A summary: `outputs/stage_a/stage_a_summary.json`
- Stage A phase memory: `workspace/phase_memory/phase_01_stage_a_validity.md`
- Stage B implementation: `reports/stage_b_reference_likelihood_implementation.md`
- Stage B results/conclusion: `reports/stage_b_conclusion.md`
- Stage B analysis: `outputs/stage_b/analysis_v1/`
- Stage C power analysis: `workspace/stage_c_power_analysis.md`
- Stage C results: `reports/stage_c_results.md`
- Stage C conclusion: `reports/stage_c_conclusion.md`
- Frozen Outcome B closure: `reports/stage_c_frozen_outcome_b_closure.md`

### v3

- Migration audit: `reports/v3_migration_audit.md`
- Four-action reanalysis: `reports/v3_stage_b_reanalysis.md`
- Discovery decision: `reports/v3_stage_b_decision.md`
- Confirmation preflight: `reports/v3_preflight_report.md`
- First null repair: `reports/v3_null_repair_report.md`
- Independent-pool null redesign: `reports/v3_null_redesign_v2.md`

### v4 and current experiment

- v4 strategy transition: `reports/v4_strategy_transition.md`
- v4 active plan: `workspace/dynamic_mllm_query_conditional_plan_v4.md`
- Common-padding preflight: `reports/v4_common_padding_preflight.md`
- Same-image discovery results: `reports/v4_discovery_results.md`
- Same-image discovery decision: `reports/v4_discovery_decision.md`
- Cost–utility reanalysis: `reports/v4_cost_utility_reanalysis.md`
- v2–v4 strategic synthesis: `reports/dynamic_mllm_v2_v4_synthesis.md`
- Frozen refinement operator: `workspace/query_conditioned_refinement_operator.md`
- Refinement preflight: `reports/query_refinement_preflight.md`
- Refinement results: `reports/query_refinement_gqa_discovery.md`
- Refinement decision: `reports/query_refinement_gqa_decision.md`
- Current workflow state: `workspace/workflow_state.md`
