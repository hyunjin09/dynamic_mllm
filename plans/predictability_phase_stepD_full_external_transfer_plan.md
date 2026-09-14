# Predictability & Generalization Phase — Step D
## Full External Transfer on ChartQA, TextVQA, MMMU-Pro, and POPE

## 0. Executive purpose

Steps A–C established the internal picture:

```text
Step A:
    controlled failure / READ / WRITE targets can be measured cleanly

Step B:
    Stage-1 failure is learnable in-domain
    Stage-2 local READ/WRITE utility is barely learnable in-domain

Step C:
    Stage-1 is robust to semantic-question shift
    but strongly source/task-regime specific

    Stage-2 remains weak, with no convincing broad niche
```

Key internal results:

```text
Stage-1 ID M3 AUROC                  = 0.7869
Stage-1 semantic Q1 AUROC            = 0.7828
Stage-1 K=100 cluster-OOD AUROC      = 0.7725
Historical → Canonical AUROC         = 0.4342
Canonical → Historical AUROC         = 0.5563
LODO minimum AUROC                   = 0.5508

Stage-2 READ ID Spearman             = 0.0416
Stage-2 WRITE ID Spearman            = 0.0347
Stage-2 cluster-OOD READ/WRITE rho   = 0.0512 / 0.0300
```

Step D supplies the final point on the generalization ladder:

```text
ID
→ semantic shift
→ source shift
→ dataset shift
→ external benchmark transfer
```

The scientific questions are:

> **Does the Stage-1 failure signal learned from the full internal corpus transfer to the established external benchmarks?**

and:

> **Does current-state READ/WRITE utility remain weak when measured prospectively on the external Stage-2 decision domain?**

This is a predictability transfer study, not a new routing-method evaluation.

No target, architecture, or hyperparameter may be redesigned in Step D.

---

## 1. External benchmark population

Use the exact established full evaluation manifests:

```text
ChartQA    = 2,500
TextVQA    = 5,000
MMMU-Pro   = 3,460
POPE       = 9,000
Total      = 19,960
```

Use the same frozen:

```text
Qwen2.5-VL-7B-Instruct snapshot
prompt templates
generation settings
answer normalization
LMMS-Eval correctness
image/sample manifests
```

as the prior Phase-69/full-routing evaluations wherever applicable.

Do not use a subset.

---

## 2. Two external evaluation domains

### D1 — Stage-1 external failure transfer

Population:

```text
all 19,960 external samples
all 28 dense decoder states
```

Question:

```text
Can a failure predictor trained on the full internal corpus
rank eventual Dense correctness on external benchmarks?
```

### D2 — Stage-2 external local-utility transfer

Primary population:

```text
all external samples that enter the frozen method's Stage-2 decision domain
under the existing robust ALL-source P90 Stage-1 trigger
```

For each such sample:

```text
all dense post-trigger states from L* through layer 27
```

Question:

```text
Can READ/WRITE utility predictors trained on the full internal
S2-Dense corpus predict controlled local utility externally?
```

Do not force Stage-2 evaluation on samples where the frozen robust P90 Stage-1 never triggers.

---

## 3. Separate predictability model from method trigger

For Stage-1 predictability, train a fresh full-internal refit of the frozen Step-B reference models on all internal data.

For the Stage-2 state-domain definition, use the existing frozen robust ALL-source P90 trigger:

```text
tau_P90 = 0.9061332901863008
strict score > tau_P90
```

Reason:

> Step A/B/C Stage-2 utility was defined only after this established handoff, so Step D must preserve the same decision domain.

---

## 4. Freeze Step-D contracts before external scoring

Before inspecting any external metric, freeze:

```text
internal training manifests
full-refit model configs
model seeds
optimization schedules
feature normalization
Stage-1 threshold calibration recipe
question encoder
kNN k
external state-domain rule
Stage-2 utility scoring contract
all metrics
all benchmark aggregation rules
```

Create:

```text
frozen_contract.json
```

No external result may change these choices.

---

# Part I — Full internal refits

## 5. Full-internal Stage-1 refit

Train on all internal Step-A Stage-1 data:

```text
10,399 UIDs
9,982 image groups
291,172 dense layer states
```

Reference models:

```text
M0-X  transfer-safe nuisance baseline
M1    linear hidden-state predictor
M3    current Stage-1 head
```

Use Step-B hyperparameters unchanged.

Do not rerun architecture search.

---

## 6. Transfer-safe Stage-1 nuisance baseline

For Step D define a prospectively frozen external-safe nuisance baseline using only features with meaningful support on every benchmark:

```text
absolute layer
text-token count
visual-token count
question length
other purely structural metadata already available everywhere
```

Do not use:

```text
Historical/Canonical source ID
unseen dataset one-hot weights
Dense correctness
answer labels
```

For ChartQA/TextVQA only, a known-dataset-ID nuisance result may be reported secondarily, but it must not replace the transfer-safe baseline.

---

## 7. Stage-1 seed policy

Use the same Step-B seed policy.

If Step B used three neural seeds:

```text
train 3 full-internal M3 refits
```

Primary external prediction:

```text
pre-registered ensemble across seeds
```

Report individual-seed results and seed SD.

Never choose the best external seed.

---

## 8. Stage-1 internal threshold calibration

External labels may never choose a threshold.

For each full-refit seed, use internal Dense-C only to calibrate thresholds at nominal:

```text
90%
95%
98%
99%
```

C preservation.

Record numerical thresholds before external metric computation.

Label these thresholds:

```text
internal-fit-calibrated
```

Step-B OOF operating points remain the unbiased ID estimate.

The Step-D purpose is to measure external calibration drift without external repair.

---

## 9. Existing robust P90 operating point

Also retain the actual existing method gate:

```text
robust ALL-source Stage-1
P90 threshold
```

Report external:

```text
P(trigger | C)
P(trigger | W)
C preservation
W recall
precision
trigger-layer distribution
```

Require parity with the prior external trigger manifest where contracts match.

This is separate from the new full-refit M3 AUROC analysis.

---

## 10. Full-internal Stage-2 utility refit

Train on all primary internal S2-Dense data:

```text
1,413 P90-triggered UIDs
1,385 image groups
15,185 dense post-trigger states
```

Train separate predictors for:

```text
READ:
U_READ_PRIMARY = q_F - q_WO

WRITE:
U_WRITE_PRIMARY = q_F - q_RO
```

Primary model:

```text
M3 joint router-style [z_R ; z_W]
```

Controls:

```text
M0-X transfer-safe nuisance
M1 simple multimodal linear
question-kNN
```

Reuse Step-B loss, hyperparameters, preprocessing, and seed policy exactly.

---

## 11. No Stage-2 target redesign

Do not switch external evaluation to:

```text
best action
correctness flip only
factorial average only
MCTS route validity
program success
```

Primary targets remain:

```text
READ  = q_F - q_WO
WRITE = q_F - q_RO
```

---

# Part II — External state and label construction

## 12. External Stage-1 dense-state census

For all 19,960 external samples, run the canonical all-FULL trajectory.

For all layers:

```text
l = 0 ... 27
```

store/reference the exact current state needed by M1/M3.

Expected state count:

```text
19,960 × 28 = 558,880
```

Every sample receives:

```text
y_fail_external
=
1 if Dense answer is wrong
0 if Dense answer is correct
```

---

## 13. External Dense baseline parity

Require exact parity with the established external Dense baseline for:

```text
sample manifest
decoded answer
LMMS correctness
```

Per-benchmark aggregate correctness must match the frozen prior evaluation under the same model/prompt contract.

Do not continue with unexplained parity mismatches.

---

## 14. External P90 trigger manifest

Apply the frozen robust ALL-source P90 Stage-1 sequentially to all external Dense trajectories.

Record:

```text
triggered / not triggered
first trigger layer L*
Dense C/W
benchmark
```

Prior reference under the established external evaluation:

```text
total triggers ≈ 901
POPE triggers = 0
```

Exact recomputed parity is authoritative.

---

## 15. External Stage-2 dense-state census

For every P90-triggered external sample, construct each Dense post-trigger pre-action state:

```text
l = L* ... 27
```

All prior layers remain FULL.

This is the external analogue of Step-A S2-Dense.

Do not include routed-policy/MCTS/program states in the primary external utility study.

---

## 16. Freeze Stage-2 predictions before utility labels

Use this order:

```text
1. build external triggered-state feature manifest
2. run frozen full-internal READ/WRITE predictors
3. save all external predictions
4. hash/freeze prediction files
5. only then execute four-branch counterfactuals
6. compute external utility labels and final metrics
```

This prevents any accidental model adaptation to external utility outcomes.

---

## 17. External four-branch utility measurement

For every external dense post-trigger state `s_l`, execute:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

at the current layer.

Then:

```text
layers l+1 ... 27 = FULL
```

Record:

```text
q_F
q_RO
q_WO
q_I

c_F
c_RO
c_WO
c_I
```

No MCTS.
No later router.
No search.

---

## 18. External q-score contract

Use the same conceptual Step-A score:

```text
token-normalized log-likelihood of the evaluator-compatible gold answer
```

Freeze benchmark-specific adapters before execution.

### ChartQA

```text
continuous q uses literal annotation/reference
discrete correctness uses frozen LMMS relaxed numeric semantics
```

Do not try to encode the ±5% acceptance interval as one finite q target.

### TextVQA

Use the frozen acceptable-reference aggregation rule inherited from Step A.

### MMMU-Pro

Freeze one evaluator-consistent gold-answer string/label representation before branch execution.

### POPE

If no P90 triggers occur:

```text
Stage-2 utility transfer = N/A
```

Do not manufacture Stage-2 states.

---

## 19. External utility algebra

Primary:

```text
U_READ  = q_F - q_WO
U_WRITE = q_F - q_RO
```

Also store secondary:

```text
q_RO - q_I
q_WO - q_I
factorial U_READ
factorial U_WRITE
U_INT
correctness flips
local rescue/regression flags
```

Require exact algebra checks from raw q values.

---

## 20. External FULL-branch parity

For every external Stage-2 dense state:

```text
FULL at current layer
+ FULL suffix
```

must reproduce the canonical Dense branch.

Require parity for:

```text
final answer
LMMS correctness
q score within frozen tolerance
```

---

# Part III — External Stage-1 transfer

## 21. Stage-1 primary external metrics

For each benchmark report full-refit M3:

```text
AUROC
AUPRC
```

Also report:

```text
pooled AUROC/AUPRC
macro-average AUROC/AUPRC across the four benchmarks
```

Macro-average is the primary cross-benchmark summary because failure prevalence differs by benchmark.

---

## 22. Stage-1 layerwise external transfer

For each benchmark report:

```text
AUROC by layer 0...27
```

and:

```text
peak external layer
peak AUROC
AUROC at internal peak layer 20
```

Question:

> Does the internal layer-20-ish failure signal survive externally?

Do not choose a benchmark-specific deployment layer after seeing external results.

---

## 23. Stage-1 external calibration drift

Apply the internal-fit-calibrated thresholds.

For each benchmark and nominal internal threshold report:

```text
actual external C preservation
external W recall
precision
median trigger layer
```

Primary nominal points:

```text
95%
98%
99%
```

Do not repair thresholds using external labels.

---

## 24. Stage-1 control models

For each benchmark compare:

```text
M0-X transfer-safe nuisance
M1 linear hidden-state
M3 current head
question-kNN
```

Primary question:

> Does the hidden-state model retain a material external advantage over structural nuisance and semantic-neighbor baselines?

---

## 25. External question-kNN Stage-1 baseline

Use the frozen Step-C question encoder and:

```text
k = 5
```

nearest internal training questions.

Predict:

```text
mean internal eventual-failure label
```

No external labels enter prediction.

---

## 26. External semantic familiarity

For every external UID compute nearest internal question similarity using the frozen Step-C encoder.

Report per benchmark:

```text
mean
median
P10/P90
```

Also map samples into the frozen Step-C similarity edges where possible.

Do not redefine bins from external outcomes.

---

## 27. Stage-1 similarity-conditioned external transfer

Where support permits, report M3 AUROC for:

```text
lower semantic-similarity half
upper semantic-similarity half
```

within each benchmark.

This helps distinguish source shift from semantic unfamiliarity.

---

# Part IV — External Stage-2 utility transfer

## 28. Stage-2 primary metrics are benchmark-specific

Because q contracts and utility prevalence differ across benchmarks, primary results are per benchmark.

For READ and WRITE separately report:

```text
Spearman
Pearson
MAE
harmful AUROC
harmful AUPRC
Precision@top5%
Precision@top10%
Precision@top20%
```

Exact-zero states are neutral for sign classification.

---

## 29. Stage-2 macro summary

For benchmarks with valid Stage-2 support, compute:

```text
macro-average Spearman
macro-average harmful AUROC
```

Do not use raw pooled Spearman as the primary cross-benchmark summary.

Raw pooled metrics may be reported secondarily with a warning.

---

## 30. Stage-2 controls

For each supported benchmark compare:

```text
M0-X transfer-safe nuisance
M1 simple multimodal linear
M3 joint router [z_R;z_W]
question-kNN
```

Question-kNN uses:

```text
k = 5 nearest internal questions
restricted to training states at the same absolute layer
```

exactly as Step C.

---

## 31. Strong behavioral event analysis

Report:

```text
READ harmful flip:
c_F=0, c_WO=1

READ beneficial flip:
c_F=1, c_WO=0

WRITE harmful flip:
c_F=0, c_RO=1

WRITE beneficial flip:
c_F=1, c_RO=0
```

Evaluate whether predicted continuous utility ranks these events correctly.

Never train on external flip labels.

---

## 32. External local-rescue census

Per benchmark report:

```text
# external post-trigger states
# local single-layer rescue states
# local regression states
# all-four-correct
# all-four-wrong
```

This extends the Step-A "not all visual computation is helpful" observation externally, independently of predictability.

Do not equate local rescue existence with deployable routing.

---

## 33. Stage-2 layerwise external analysis

For each supported benchmark report READ/WRITE predictability by:

```text
Early
Middle
Late
```

and exact layer only where support is adequate.

Also report trigger-relative:

```text
at trigger
1-2 layers after trigger
3-5 layers after trigger
6+ layers
```

Avoid overinterpreting isolated small positive cells.

---

## 34. Stage-2 semantic-similarity analysis

For supported external states, inherit each UID's nearest internal question similarity.

Where support permits report READ/WRITE Spearman for:

```text
lower similarity half
upper similarity half
```

This tests whether any external utility niche is limited to familiar questions.

---

# Part V — Final generalization ladder

## 35. Final Stage-1 ladder

Produce:

| Regime | M3 AUROC |
|---|---:|
| Step-B ID | 0.7869 |
| Step-C least-similar Q1 | 0.7828 |
| Step-C cluster OOD | 0.7725 |
| Historical→Canonical | 0.4342 |
| Canonical→Historical | 0.5563 |
| LODO GQA | 0.5508 |
| LODO ChartQA | 0.6043 |
| LODO TextVQA | 0.6006 |
| External ChartQA | |
| External TextVQA | |
| External MMMU-Pro | |
| External POPE | |
| External macro | |

Do not average unlike internal regimes into one artificial overall score.

---

## 36. Final Stage-2 ladders

Create separate READ and WRITE tables.

Reference:

```text
READ ID rho  = 0.0416
WRITE ID rho = 0.0347
```

Add:

```text
cluster OOD
source transfer
LODO
external ChartQA
external TextVQA
external MMMU-Pro
external macro
```

POPE is included only if the Stage-2 decision domain is entered.

---

## 37. Same-family versus new-family transfer

Treat benchmarks descriptively as:

```text
ChartQA:
same named task family

TextVQA:
same named task family

MMMU-Pro:
new reasoning/task family

POPE:
new hallucination/object-presence family
```

Do not claim strict statistical equivalence of these categories from only four benchmarks.

---

## 38. Statistical uncertainty

Use sample/image-group bootstrap.

Stage-1:

```text
AUROC
AUPRC
external-vs-ID AUROC drop
```

Stage-2:

```text
Spearman
harmful AUROC
Precision@top10%
```

Never bootstrap individual layer-state rows independently.

---

## 39. Seed reporting

For full-refit M3:

```text
report each seed
report mean
report SD
```

Primary prediction may use the pre-registered seed ensemble.

Never select the best external seed.

---

## 40. External-label firewall

External labels may be used only to:

```text
construct y_fail
construct q/counterfactual utility
compute final metrics
```

They may not be used to:

```text
choose model
choose layer
choose seed
choose threshold
change normalization
change q aggregation
change k
retrain predictor
drop benchmark
select favorable subset
```

---

# Interpretation categories

## 41. Stage-1 categories

### D1-A — Broad external transfer

```text
ChartQA/TextVQA remain strong
MMMU-Pro/POPE remain materially above controls
macro remains reasonably close to ID
calibration drifts only moderately
```

Interpretation:

> Stage-1 failure information is more transferable than Step-C source-transfer results suggested.

This requires reconciliation with Historical↔Canonical collapse.

### D1-B — Same-family transfer only

```text
ChartQA/TextVQA useful
MMMU-Pro/POPE collapse
```

Interpretation:

> Failure prediction transfers within related task families but not broadly across reasoning/data regimes.

### D1-C — External source-specific collapse

```text
most external AUROCs approach nuisance/chance
calibration drifts strongly
```

Interpretation:

> Stage-1 failure predictability is strongly population/source specific despite semantic-question robustness.

---

## 42. Stage-2 categories

### D2-A — Weak externally everywhere

```text
READ/WRITE Spearman near zero
harmful AUROC near 0.5
top-harmful precision near prevalence
```

Interpretation:

> Current-state local utility is weak both internally and externally.

### D2-B — Same-family utility niche

```text
ChartQA or TextVQA shows reproducible material utility prediction
MMMU-Pro remains weak
```

Interpretation:

> Local READ/WRITE utility may be predictable only under restricted familiar task regimes.

Do not generalize the niche globally.

### D2-C — Unexpected broad external utility signal

```text
multiple external benchmarks are materially stronger than Step-B ID
```

Interpretation:

> First trigger a parity/leakage/distribution audit before making a positive method claim.

---

## 43. Combined final categories

### Combined A

```text
Stage-1 transfers externally
Stage-2 remains weak
```

Conclusion:

> Detecting impending failure is genuinely more transferable than identifying the local visual computation that should be suppressed.

### Combined B

```text
Stage-1 transfers only within related task families
Stage-2 weak
```

Conclusion:

> Failure detection is learnable but task-regime specific; local intervention utility is substantially harder.

### Combined C

```text
Stage-1 broadly collapses
Stage-2 weak
```

Conclusion:

> Oracle intervention opportunities exist, but both failure detection and local intervention selection face serious transfer limits, with Stage-2 already weak in-domain.

---

## 44. What Step D can establish

Step D can establish:

```text
full external transfer of Stage-1 failure predictability
external calibration drift of internally chosen Stage-1 operating points
whether Stage-1 hidden-state prediction beats external structural/question-neighbor controls
external local READ/WRITE utility predictability on the actual P90 Stage-2 decision domain
external existence of local single-layer rescue opportunities
whether the Stage-1/Stage-2 learnability asymmetry persists externally
the final internal→external generalization ladder
```

---

## 45. What Step D cannot establish

Step D does not establish:

```text
deployment routing improvement
causal hidden-state mechanism
that Stage-2 can never work with history/counterfactual features
that a different Stage-1 population construction cannot generalize
that local utility is the only useful Stage-2 target
```

---

## 46. No new routing-method training in Step D

Do not train:

```text
new deployment Stage-1 router
new Stage-2 policy
first-intervention-aware objective
on-policy relabeling
history-conditioned Stage-2
new MCTS/program predictor
```

beyond the pre-registered full-internal predictability refits.

Step D finishes the analysis phase before the next method-design phase begins.

---

# Required artifacts

## 47. Output directory

Use:

```text
analysis/predictability_generalization/stepD_external_transfer/
```

Create:

```text
protocol.md
frozen_contract.json

refit/
    stage1_full_refit_manifest.json
    stage1_seed_checkpoints.json
    stage1_internal_calibration_thresholds.csv
    stage2_read_full_refit_manifest.json
    stage2_write_full_refit_manifest.json
    stage2_seed_checkpoints.json
    refit_training_summary.md

external_manifests/
    benchmark_manifest.jsonl
    dense_state_manifest.jsonl
    p90_trigger_manifest.jsonl
    stage2_dense_state_manifest.jsonl
    parity_report.md

predictions_frozen_before_labels/
    stage1_external_predictions.jsonl
    stage2_read_external_predictions.jsonl
    stage2_write_external_predictions.jsonl
    prediction_hashes.json

stage1/
    overall_metrics.csv
    benchmark_metrics.csv
    layer_metrics.csv
    operating_point_transfer.csv
    robust_p90_external_funnel.csv
    model_control_comparison.csv
    question_knn_metrics.csv
    semantic_similarity_metrics.csv
    seed_metrics.csv

stage2_measurement/
    four_branch_results.jsonl
    utility_labels.csv
    correctness_flip_labels.csv
    utility_distribution_summary.csv
    parity_report.md
    local_rescue_summary.csv

stage2_predictability/
    read_benchmark_metrics.csv
    write_benchmark_metrics.csv
    macro_metrics.csv
    control_comparison.csv
    strong_flip_metrics.csv
    layer_metrics.csv
    trigger_relative_metrics.csv
    semantic_similarity_metrics.csv
    seed_metrics.csv

question_semantics/
    external_question_embeddings.npy
    nearest_internal_similarity.jsonl
    benchmark_similarity_summary.csv

statistics/
    group_bootstrap_ci.csv
    external_vs_internal_degradation.csv

figures/
    stage1_external_benchmark_auroc.png
    stage1_layer_emergence_external.png
    stage1_operating_point_calibration_drift.png
    stage1_generalization_ladder_final.png
    stage1_hidden_vs_knn_vs_nuisance.png
    stage2_read_external_transfer.png
    stage2_write_external_transfer.png
    stage2_generalization_ladder_final.png
    external_semantic_similarity.png
    stage1_stage2_asymmetry_final.png

summaries/
    stepD_external_transfer_summary.md
    predictability_phase_final_summary.md
    next_method_implication.md

artifact_manifest.json
```

---

## 48. Main Stage-1 external table

| Benchmark | N | Dense C | Dense W | M0-X AUROC | M1 AUROC | M3 AUROC | M3 AUPRC | kNN AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartQA | 2500 | | | | | | | |
| TextVQA | 5000 | | | | | | | |
| MMMU-Pro | 3460 | | | | | | | |
| POPE | 9000 | | | | | | | |
| Macro | | | | | | | | |
| Pooled | 19960 | | | | | | | | |

---

## 49. Stage-1 operating-point transfer table

For each benchmark:

| Internal nominal C-preserve | External C-preserve | External W recall | Precision | Median trigger layer |
|---|---:|---:|---:|---:|
| 95% | | | | |
| 98% | | | | |
| 99% | | | | |

Also report the existing robust P90 operating point separately.

---

## 50. Main Stage-2 READ table

| Benchmark | Triggered UIDs | States | Spearman | Harmful AUROC | Harmful AUPRC | Precision@Top10% | M0-X rho | kNN rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartQA | | | | | | | | |
| TextVQA | | | | | | | | |
| MMMU-Pro | | | | | | | | |
| POPE | | | | | | | | |
| Macro | | | | | | | | |

---

## 51. Main Stage-2 WRITE table

Produce the same table for:

```text
q_F - q_RO
```

---

## 52. `stepD_external_transfer_summary.md` must answer

1. Did full-internal Stage-1/Stage-2 refits complete under frozen Step-B contracts?
2. Did external Dense baseline parity pass on all 19,960 samples?
3. How many external samples entered the frozen P90 Stage-2 domain?
4. Did the P90 trigger manifest reproduce prior external behavior?
5. What are external Stage-1 M3 AUROC/AUPRC for ChartQA, TextVQA, MMMU-Pro, and POPE?
6. What is Stage-1 macro AUROC?
7. At which layers does external Stage-1 predictability peak?
8. Does the internal layer-20 signal survive externally?
9. How much do internally calibrated 95/98/99% C-preservation points drift externally?
10. How does the existing robust P90 method operating point behave externally?
11. Does M3 beat the transfer-safe nuisance baseline externally?
12. Does M3 beat question-kNN externally?
13. How semantically similar are external questions to the internal corpus?
14. Is external Stage-1 transfer stronger for semantically familiar questions?
15. How many external Stage-2 dense states were measured?
16. Did external four-branch FULL parity and utility algebra pass?
17. How many external local single-layer rescue/regression opportunities exist?
18. What is external READ utility predictability per supported benchmark?
19. What is external WRITE utility predictability per supported benchmark?
20. Does any Stage-2 benchmark show a reproducible nontrivial utility niche?
21. Does Stage-2 hidden-state prediction beat nuisance/kNN controls externally?
22. Does the Stage-1/Stage-2 asymmetry persist externally?
23. Which Stage-1 category D1-A/B/C is supported?
24. Which Stage-2 category D2-A/B/C is supported?
25. What does Step D not justify concluding?

---

## 53. `predictability_phase_final_summary.md`

Summarize the full A→D phase around:

```text
Observation:
Do harmful/beneficial visual-computation counterfactuals exist?

In-domain learnability:
Can hidden states predict failure?
Can hidden states predict local READ/WRITE utility?

Internal generalization:
Does prediction depend on semantic familiarity?
Does it survive source/task shift?

External transfer:
Does the signal survive the four external benchmarks?
```

Clearly separate:

```text
measurement
learnability
generalization
deployment
```

Do not conflate them.

---

## 54. `next_method_implication.md`

Do not automatically propose a more complicated router.

Based on the final A→D evidence, state exactly one method-level implication.

Examples:

### If Stage-1 transfers but Stage-2 stays weak

```text
Keep a failure-detection handoff,
but abandon current-state local utility classification as the default Stage-2 formulation.

Next investigate richer information:
history,
counterfactual probes,
or search-conditioned planning.
```

### If Stage-1 is source-specific and Stage-2 weak

```text
The next method must address both:
robust failure representation
and
nonlocal intervention selection.
```

### If a narrow Stage-2 niche appears

```text
Characterize and exploit only that restricted regime
before making a general routing claim.
```

For the recommendation include:

```text
what evidence motivates it
what the next experiment should distinguish
what it cannot yet establish
```

---

## 55. Stop rule

STOP after:

```text
1. full-internal Stage-1 refit
2. full-internal Stage-2 READ/WRITE refit
3. all 19,960 external Stage-1 samples evaluated
4. external Dense parity validated
5. frozen robust P90 external trigger manifest built
6. all external P90-triggered dense Stage-2 states measured
7. external predictions frozen before utility labels are scored
8. Stage-1 benchmark/layer/calibration transfer completed
9. Stage-2 READ/WRITE benchmark transfer completed
10. nuisance + question-kNN controls completed
11. semantic-familiarity analysis completed
12. final A→D generalization ladder produced
13. final predictability-phase summary produced
14. exactly one next method implication written
```

Do not begin a new routing-method training phase inside Step D.

---

## 56. Core principle

The project began from:

> **Some visual computation is harmful: suppressing it can convert wrong answers into correct ones.**

The completed internal analysis suggests:

```text
"Will this trajectory fail?"
    is learnable in-domain

"Is this specific READ/WRITE computation harmful right now?"
    is barely learnable from the current-state snapshot
```

Step D tests whether this asymmetry survives the strongest test:

> **full transfer to ChartQA, TextVQA, MMMU-Pro, and POPE with zero external retuning.**
