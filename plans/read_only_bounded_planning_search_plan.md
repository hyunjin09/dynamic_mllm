# READ-Only Bounded Planning / Search Plan
## If READ harm is not locally identifiable, is it still easy to solve as a trajectory-level planning problem?

## 0. Motivation

The READ-only line now has a consistent result:

```text
READ harm exists,
but local identification is weak.
```

Observed evidence:

```text
generic pre-state prediction                  → weak
one-step READ ON/OFF counterfactual delta     → weak
READ-specific attention/update features       → weak
H=1/2/4/8 propagated counterfactual effects   → weak
```

The short-horizon phase ended with `H-READ-D`.

Therefore the next question is not another local-classifier question.

> **If WRITE is always ON and the controller only plans READ ON/OFF, how much search is required to recover correct trajectories?**

This phase characterizes READ as a **binary trajectory-planning problem**.

It is an analysis phase, not yet a deployable planner.

---

# 1. Core scientific outcomes

We want to distinguish:

### P-READ-A — Easy trajectory structure
Small bounded search quickly recovers most READ-only corrections.

Interpretation:
> READ is locally non-identifiable but globally/trajectory structured.

### P-READ-B — Moderate planning problem
A moderate budget is needed, but structured search clearly approaches the READ-only ceiling.

### P-READ-C — Needle-in-a-haystack
Only large budgets work and structured search has little advantage over matched random search.

### READ-only ceiling limited
Even high-budget binary READ search recovers only a small fraction of the previously observed four-action correction opportunity.

This last axis is reported separately from A/B/C because a subset may be easy to plan even if READ alone cannot explain most failures.

---

# 2. Frozen Stage-1 contract

Keep the existing robust P90 Stage-1 unchanged.

For first trigger layer `L*`:

```text
layers < L*:
    FULL

layers L* ... 27:
    READ-only planning domain
```

Do not intervene before the first trigger.

Do not retrain Stage-1 or change the threshold.

---

# 3. Binary READ action space

WRITE is always ON.

The only actions are:

```text
READ_ON  = FULL       = READ1 WRITE1
READ_OFF = WRITE_ONLY = READ0 WRITE1
```

Thus:

```text
a_l ∈ {READ_ON, READ_OFF}
```

for `l = L* ... 27`.

All decisions are closed-loop:

```text
observe actual routed state at l
→ choose READ ON/OFF
→ execute action
→ observe actual routed state at l+1
→ choose next action
```

Never predict the whole suffix open-loop from the trigger state.

---

# 4. Population

Primary:

```text
all P90-triggered Dense-W UIDs
```

Expected from the frozen registry:

```text
1,307 Dense-W UIDs
```

Verify exact population before execution.

Secondary preservation/ranking control:

```text
P90-triggered Dense-C UIDs
```

Expected:

```text
106 UIDs
```

Primary question:

> Among wrong samples that Stage-1 has already triggered on, how easily can READ-only trajectory search discover a correct answer?

---

# 5. Route definition

Represent the post-trigger route as:

```text
1 = READ_ON / FULL
0 = READ_OFF / WRITE_ONLY
```

Example for trigger L20:

```text
1 1 0 1 0 1 1 1
```

Store for every evaluated route:

```text
UID
L*
binary suffix
route hash
# READ_OFF actions
first/last READ_OFF layer
# ON↔OFF switches
q score
generated answer
LMMS correctness
search algorithm
evaluation rank
```

---

# 6. Candidate discovery vs selection

This distinction is mandatory.

For every algorithm and budget `B`, report:

## ANY_CORRECT@B

> Did any of the first `B` unique complete routes evaluated by this algorithm produce a correct answer?

This measures candidate-generation/search difficulty.

## SELECTED_CORRECT@B

Select the route with the highest frozen `q` among evaluated candidates.

> Is the q-selected route correct?

This measures search + ranking.

Define:

```text
ranking gap
=
ANY_CORRECT@B - SELECTED_CORRECT@B
```

For each UID classify outcome as:

```text
SUCCESS:
    selected route is correct

RANKING_FAILURE:
    some evaluated route is correct,
    but q-selected route is wrong

GENERATION_FAILURE:
    no evaluated route is correct
```

---

# 7. Search score

Reuse the frozen teacher-forced gold-answer score:

```text
q(route)
=
token-normalized annotated-answer log-likelihood
```

Search algorithms may use `q` as an **analysis-time oracle heuristic**.

This does not make the planner deployable.

Every route counted for discrete rescue must have actual generated-answer correctness evaluated with the frozen LMMS protocol.

Do not infer correctness from q.

---

# 8. Reuse exact Hamming-1 evidence

Before new search, reconstruct the exact single-READ_OFF census from Step-A.

For each W UID:

```text
Dense:
    all READ_ON

Hamming-1:
    exactly one READ_OFF layer
    all other suffix layers READ_ON
```

These are already represented by the dense READ counterfactual measurements.

Derive:

```text
# W UIDs rescued by at least one single READ_OFF
best single-off route
rescue layer distribution
first-rescue layer relative to trigger
```

Do not recompute except for parity checks.

This is the exact low-complexity baseline.

---

# 9. Optional exact Hamming-2 census

For W UIDs not rescued at Hamming-1, estimate the number of exact distance-2 routes:

```text
C(T, 2)
```

with `T = 28 - L*`.

Before running, write:

```text
projected new candidate count
expected cache hits
estimated GPU hours
```

Run exhaustive Hamming-2 only if:

```text
projected new unique route evaluations <= 250,000
```

Otherwise skip it rather than silently subsampling.

Purpose:

> Determine whether many READ corrections require exactly two coordinated OFF decisions.

---

# 10. Search algorithm suite

## A0 — Dense baseline

```text
all READ_ON
```

## A1 — Exact single-off scan

Reuse the Hamming-1 census.

## A2 — Greedy q-guided closed-loop search

Equivalent to beam width 1.

At each layer:

1. branch retained prefix with READ_ON and READ_OFF;
2. complete each branch with READ_ON for all later layers;
3. evaluate each resulting complete route with q;
4. retain the better-q prefix;
5. continue from the actual routed state.

Every score corresponds to a valid complete route:

```text
current prefix + all-ON suffix
```

## A3 — Beam search

Fixed widths:

```text
K = 2, 4, 8
```

At each layer:

1. expand each beam prefix with ON/OFF;
2. complete each new prefix with all-ON suffix;
3. evaluate q;
4. retain top-K prefixes;
5. continue closed-loop.

Deduplicate identical complete routes.

## A4 — Random-uniform

Sample binary suffixes with:

```text
P(READ_OFF)=0.5
```

## A5 — Random-sparse

Sample:

```text
m ∈ {1,2,3,4}
```

uniformly over valid `m <= T`, then choose `m` OFF positions uniformly.

Use fixed RNG seeds.

## A6 — Binary MCTS reference

Reuse the existing MCTS infrastructure, but restrict actions to:

```text
{FULL, WRITE_ONLY}
```

No READ_ONLY or IGNORE.

Use the same closed-loop transition semantics and q terminal/rollout score.

This is a high-budget reference, not the primary method.

---

# 11. Budget axis

Compare algorithms by:

```text
number of unique complete route evaluations
```

not by beam width or MCTS iteration alone.

Frozen checkpoints:

```text
B = 1, 4, 8, 16, 32, 64, 128
```

Higher-budget reference:

```text
B = 256, 512
```

Report actual unique-route counts per UID.

---

# 12. Adaptive high-budget policy

Run the full W population through B=128.

Then:

```text
B=256:
    continue unresolved UIDs
    + fixed random sample of resolved UIDs for saturation checks

B=512:
    continue still-unresolved UIDs only if
    B128→B256 gains have not saturated
```

Do not mix adaptive and full-population denominators.

High-budget curves must explicitly state their evaluated cohort.

---

# 13. Cache reuse

Canonical route key:

```text
UID + L* + binary action suffix
```

Search existing artifacts from:

```text
Step-A counterfactual branches
prior corrective search
MCTS corpora
completeness audit
READ short-horizon phase
```

Reuse only if all contracts match:

```text
same model/checkpoint
same prompt/input
same trigger
same READ/WRITE semantics
same decoding/evaluation contract
```

Log provenance for every cache hit.

---

# 14. Smoke validation

Before the full run, test:

```text
32 triggered W UIDs
8 triggered C UIDs
```

stratified over:

```text
dataset
trigger depth
suffix length
```

Validate:

```text
route hashing
cache lookup
closed-loop state transitions
q parity
generation parity
LMMS correctness parity
beam traces
random determinism
binary MCTS action restriction
unique-evaluation accounting
```

Smoke results are infrastructure-only, not scientific evidence.

---

# 15. Primary planning curves

For triggered W UIDs, plot for each algorithm:

```text
ANY_CORRECT@B
SELECTED_CORRECT@B
```

against:

```text
unique route evaluations B
```

Include:

```text
greedy
beam2
beam4
beam8
random-uniform
random-sparse
binary MCTS
```

Also show:

```text
single-off baseline
empirical high-budget READ-only ceiling
```

---

# 16. Empirical READ-only ceiling

Define:

```text
READ_BINARY_CEILING
=
fraction of triggered W UIDs
for which at least one correct READ-only route
appears anywhere in the union of all high-budget evaluated candidates
```

This is an empirical lower bound on the true binary ceiling, not exhaustive proof.

For algorithm/budget define:

```text
ceiling_recovery(B)
=
# UIDs rescued by algorithm by B
/
# UIDs rescued anywhere in the high-budget union
```

---

# 17. First-rescue budget

For every rescued W UID record:

```text
first_correct_evaluation_rank
```

Report:

```text
median
p75
p90
CDF
```

per search algorithm.

Question:

> When a READ-only correction exists, how many evaluations are typically needed to encounter it?

---

# 18. Compare to prior four-action opportunity

Only if the exact UID population aligns with the prior robust P90 corrective-search registry, compare:

```text
READ-only empirical high-budget rescue coverage
vs
prior four-action corrective coverage
```

Historical reference to verify from source artifacts:

```text
~463 / 1307 ≈ 35.4%
```

Do not compare if populations differ.

Question:

> What fraction of joint READ/WRITE correction opportunity is recoverable by READ decisions alone?

---

# 19. Route-complexity analysis

For each rescued W UID, characterize successful routes by:

```text
# READ_OFF actions
Hamming distance from Dense
first READ_OFF delay from trigger
last READ_OFF layer
first-to-last OFF span
# ON↔OFF switches
```

Compare:

```text
single-off rescues
beam-found rescues
MCTS-only rescues
```

Question:

> Are correct routes sparse edits or coordinated multi-layer programs?

---

# 20. Solution multiplicity

At matched budget, for every rescued UID report:

```text
# correct evaluated routes
fraction of evaluated routes correct
pairwise Hamming distance among correct routes
# distinct first actions among correct routes
```

Interpretation:

```text
many nearby correct routes
→ broad solution basin

one/few isolated routes
→ needle-like solution
```

---

# 21. Prefix/funnel structure

Among discovered correct routes, compute by trigger-relative depth:

```text
READ_ON/OFF action distribution
prefix entropy
fraction of correct-route pairs sharing the same prefix
```

This is descriptive only.

Question:

> Do successful READ trajectories form a coherent funnel or remain highly multimodal?

---

# 22. q-ranking quality

Among evaluated routes, report:

```text
q(correct) vs q(wrong)
within-UID q AUROC where both exist
rank of best correct route by q
q-selected route correctness
```

This separates:

```text
candidate-generation difficulty
from
candidate-ranking difficulty
```

---

# 23. Dense-C control

For triggered Dense-C UIDs, run search at:

```text
B = 8, 32, 128
```

Report:

```text
whether dense FULL remains selected
C→W regression rate of q-selected route
fraction of searched routes that are wrong
ranking-failure pattern
```

This is a safety/ranking control, not part of the primary W planning-ease category.

---

# 24. Beam-vs-random test

At matched budgets compare:

```text
beam8 ANY_CORRECT@B
vs
random-uniform
vs
random-sparse
```

using paired UID bootstrap.

A trajectory-structure claim requires structured search to beat budget-matched random exploration.

---

# 25. Greedy-vs-beam test

Compare at matched route-evaluation budgets:

```text
greedy
beam2
beam4
beam8
```

Interpretation:

```text
beam >> greedy:
    local q choices are unreliable;
    multi-hypothesis trajectory search matters

beam ≈ greedy:
    search landscape is comparatively simple
```

---

# 26. Search saturation

Report marginal rescue gains:

```text
4→8
8→16
16→32
32→64
64→128
128→256
256→512
```

using:

```text
ANY_CORRECT gain
SELECTED_CORRECT gain
```

This reveals whether useful routes are found early or only after prolonged exploration.

---

# 27. Dataset/source and trigger-depth breakdown

Report primary planning curves by:

```text
Historical GQA
Historical ChartQA
Historical TextVQA
Canonical GQA
Canonical ChartQA
Canonical TextVQA
```

where support is sufficient.

Also report by:

```text
trigger layer bin
suffix length T
```

Question:

> Is planning difficulty mainly driven by a larger binary search space for early triggers?

---

# 28. Local-predictor reference

Do not retrain prior predictors.

Include only as context:

```text
generic pre-state READ predictor
one-step counterfactual predictor
READ-specific local mechanism predictor
H=8 short-horizon predictor
```

Compare conceptually:

```text
local predictability
vs
bounded planning rescue
```

If planning succeeds while local prediction remains weak, the key claim becomes:

> READ correction is trajectory-dependent rather than locally decodable.

---

# 29. Planning-ease decision rules

Use the empirical high-budget binary rescue union as the denominator.

## P-READ-A — Easy trajectory structure

Support if all are true:

```text
1. some structured search with B<=32 recovers >=70%
   of empirical binary rescues;

2. structured search beats both random baselines
   by >=10 percentage points in ceiling recovery
   at a matched small budget;

3. median first-rescue rank <=32.
```

## P-READ-B — Moderate planning

Support if:

```text
1. B<=128 recovers >=70% of empirical binary rescues;
2. P-READ-A is not satisfied;
3. structured search clearly beats random.
```

## P-READ-C — Expensive / needle-like planning

Support if:

```text
1. B=128 recovers <70% of empirical binary rescues
   OR substantial gains continue through B=256/512;

2. structured search has little/inconsistent advantage over random
   OR solution density is very low.
```

---

# 30. READ-only ceiling status

Report separately:

## READ-CEILING-SUFFICIENT

READ-only high-budget search recovers a substantial fraction of the exactly aligned prior four-action correction opportunity.

## READ-CEILING-LIMITED

If exact population alignment exists, use the frozen operational rule:

```text
READ-only rescue coverage
<
50% of prior four-action rescue coverage
```

Then READ alone explains only a minority of the known joint correction opportunity.

Planning-ease category and ceiling status can coexist:

```text
P-READ-A + ceiling-limited
P-READ-C + ceiling-sufficient
...
```

This two-axis result is more informative than forcing one label.

---

# 31. Interpretation matrix

### High ceiling + P-READ-A/B

Conclusion:

> READ failures have trajectory-level structure despite weak local identifiability.

Next method candidate:
bounded READ planner with a learned label-free proposal/value signal.

### High ceiling + P-READ-C

Conclusion:

> READ is trajectory-solvable, but correction requires substantial search.

Next:
learn trajectory proposal/value functions before deployment.

### Low ceiling + P-READ-A/B

Conclusion:

> A subset of failures is cleanly READ-plannable, but broader correction requires WRITE or joint control.

Next:
move to WRITE-only characterization.

### Low ceiling + P-READ-C

Conclusion:

> READ-only control is both incomplete and search-intensive.

Next:
do not spend more effort on a standalone READ router; move to WRITE/joint reasoning.

---

# 32. What this phase can establish

This phase can establish:

```text
how much correction is reachable using only READ decisions
how rapidly bounded search approaches the empirical READ ceiling
whether beam/greedy exploit structure beyond random search
whether successful routes are sparse or highly coordinated
whether solution basins are broad or needle-like
whether q ranking or candidate generation is the dominant bottleneck
```

---

# 33. What it cannot establish

Even a strong positive result does not establish:

```text
deployable planning
label-free route scoring
net compute savings
external benchmark gain
causal mechanism of READ harm
WRITE behavior
joint READ/WRITE optimality
```

Gold-answer q makes this an analysis-time planning study.

---

# 34. No learned planner yet

Do not train:

```text
policy network
value network
trajectory transformer
route proposal model
RL controller
```

First characterize the binary search landscape.

---

# 35. No WRITE actions

WRITE stays ON throughout.

Do not reintroduce the four-action set.

The only actions are:

```text
FULL
WRITE_ONLY
```

---

# 36. No external deployment

Do not run the final four-benchmark routed evaluation in this phase.

External evaluation comes only after a deployable scoring/controller mechanism exists.

---

# 37. Recommended execution order

```text
1. verify exact W/C population
2. reconstruct cached Hamming-1 READ rescue census
3. estimate/run Hamming-2 only under compute cap
4. validate search engine on smoke set
5. run full W population through B=128:
      greedy
      beam2/4/8
      random-uniform
      random-sparse
6. run binary MCTS as higher-budget reference
7. adaptively extend unresolved UIDs to B=256/512 if needed
8. run Dense-C ranking/regression control
9. analyze route complexity, solution multiplicity, ranking failures
10. compare with prior four-action coverage if exactly aligned
11. make one final READ characterization
```

---

# 38. Output directory

Use:

```text
analysis/read_only_bounded_planning/
```

Required artifacts:

```text
protocol.md
frozen_contract.json

population/
    triggered_w_manifest.jsonl
    triggered_c_manifest.jsonl
    population_alignment_report.md
    suffix_length_distribution.csv

cache/
    route_cache_manifest.jsonl
    cache_provenance.csv
    parity_report.md

single_off/
    hamming1_routes.jsonl
    hamming1_uid_summary.csv
    hamming1_rescue_layers.csv

hamming2/
    projected_cost.md
    routes.jsonl
    uid_summary.csv

search/
    greedy_routes.jsonl
    beam2_routes.jsonl
    beam4_routes.jsonl
    beam8_routes.jsonl
    random_uniform_routes.jsonl
    random_sparse_routes.jsonl
    binary_mcts_routes.jsonl

metrics/
    correction_curve_any_correct.csv
    correction_curve_selected_correct.csv
    first_rescue_budget.csv
    ceiling_recovery.csv
    ranking_failure_decomposition.csv
    route_complexity.csv
    solution_multiplicity.csv
    prefix_entropy.csv
    q_ranking_quality.csv
    search_saturation.csv
    dataset_source_breakdown.csv
    trigger_depth_breakdown.csv

controls/
    dense_c_preservation.csv
    random_search_comparison.csv
    greedy_vs_beam.csv
    cache_parity.csv

statistics/
    paired_uid_bootstrap.csv
    seed_metrics.csv

figures/
    read_search_budget_curve_any_correct.png
    read_search_budget_curve_selected.png
    read_ceiling_recovery.png
    read_first_rescue_cdf.png
    read_ranking_vs_generation_failure.png
    read_route_hamming_distance.png
    read_solution_density.png
    read_beam_vs_random.png
    read_search_saturation.png
    read_trigger_depth_difficulty.png

summaries/
    read_bounded_planning_summary.md
    read_problem_characterization.md
    next_research_direction.md

artifact_manifest.json
```

---

# 39. `read_bounded_planning_summary.md` must answer

1. What exact W/C population was used?
2. Is it exactly aligned with the prior P90 correction-search registry?
3. How many W UIDs are rescued by one READ_OFF?
4. If run, how many additional UIDs are rescued by exact Hamming-2?
5. What is ANY_CORRECT@B for every search algorithm?
6. What is SELECTED_CORRECT@B?
7. What fraction is ranking failure vs generation failure?
8. What is the empirical high-budget READ-only ceiling?
9. What fraction of that ceiling is recovered by B=8/16/32/64/128?
10. What is median/p90 first-rescue rank?
11. Does beam beat matched random search?
12. Does beam width beat greedy?
13. Does search saturate early or continue improving through 256/512?
14. How many READ_OFF actions do successful routes use?
15. Where is the first READ_OFF relative to trigger?
16. Are successful routes sparse or dense?
17. Are there many correct routes per UID or isolated solutions?
18. Do correct routes share prefixes/funnels?
19. How well does q rank correct routes?
20. How often does q-selected search regress Dense-C?
21. How does difficulty depend on trigger depth/suffix length?
22. How does it vary by dataset/source?
23. How much aligned four-action correction coverage is recovered by READ-only search?
24. Is planning ease P-READ-A/B/C?
25. Is the READ-only ceiling sufficient or limited?
26. What does this phase not establish?

---

# 40. `read_problem_characterization.md`

Synthesize all READ phases:

```text
A. Existence:
   READ suppression can improve/correct answers.

B. Local structure:
   mostly isolated; no robust harmful span regime.

C. Local mechanism:
   no robust local attention/update signature.

D. Local learnability:
   generic and READ-specific predictors are weak.

E. Short-horizon propagation:
   H<=8 remains weak.

F. Trajectory planning:
   summarize this bounded-search result.
```

End with the strongest supported characterization, such as:

```text
READ is trajectory-structured but locally non-identifiable.

READ is trajectory-solvable but requires expensive search.

READ-only control explains only a limited subset of correctable failures.

READ-only control is both weakly identifiable and weakly searchable.
```

---

# 41. `next_research_direction.md`

Recommend exactly one next move.

If high ceiling + P-READ-A/B:

```text
design a bounded READ planner
with a learned label-free proposal/value signal
```

and explicitly state what could replace gold-answer q.

If high ceiling + P-READ-C:

```text
study trajectory proposal/value learning
before deployment
```

If READ ceiling is limited:

```text
move to WRITE-only characterization
```

using the same principle:

```text
existence is already known;
now test structure, learnability, and planning solvability.
```

Do not return directly to a four-action router.

---

# 42. Stop rule

STOP after:

```text
1. exact population alignment
2. cached single-off census
3. optional Hamming-2 under the frozen compute cap
4. full bounded-search curves through B=128
5. enough high-budget binary search to estimate saturation
6. ANY_CORRECT vs SELECTED_CORRECT decomposition
7. beam-vs-random comparison
8. route-complexity and multiplicity analysis
9. Dense-C ranking/regression control
10. aligned comparison to prior four-action coverage
11. one final READ characterization
12. exactly one next research direction
```

Do not train a learned planner, start WRITE search, or run external deployment inside this phase.

---

# 43. Core principle

The project already knows:

```text
READ harm exists,
but READ harm is not locally identifiable.
```

The final unresolved READ question is:

> **Is the problem easy once treated as a binary trajectory-planning problem, or are successful READ trajectories themselves sparse and search-intensive?**

This phase answers that question while keeping WRITE fixed ON and removing the confound of the original four-action routing space.
