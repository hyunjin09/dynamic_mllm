You are working in the existing `dynamic_mllm` repository.

Your goal is NOT to train a new router or run a new 4-action MCTS search yet.

The immediate scientific goal is to test whether errors that are correctable by binary visual routing can be causally attributed to answer-unaligned visual READ operations, visual WRITE/update operations, or their interaction.

Proceed carefully and comprehensively. Reuse the existing executor, datasets, MCTS caches, evaluation code, and scoring utilities whenever possible. Do not redesign working infrastructure unnecessarily.

============================================================
0. SCIENTIFIC QUESTION
============================================================

We already have binary visual-routing results where:

- FULL visual computation is wrong.
- ALL-OFF visual computation is also wrong.
- But at least one non-ALL-OFF binary visual route produces the correct answer.

Call these samples the primary A+ cohort.

The key question is:

"When a dense pretrained MLLM makes an error that can be corrected by an alternative visual route, which native visual operation is answer-unaligned: visual READ, visual WRITE/update, or their interaction?"

The desired interpretation is:

- READ = text/control tokens directly access visual K/V at this decoder layer.
- WRITE = visual-token rows are updated by this decoder layer.
- These should be independently controllable.

The four actions are therefore:

1. IGNORE
   READ=0, WRITE=0

2. READ_ONLY
   READ=1, WRITE=0

3. WRITE_ONLY
   READ=0, WRITE=1

4. FULL
   READ=1, WRITE=1

This experiment is a LOCAL CAUSAL INTERVENTION experiment, not a 4-action trajectory search.

For each layer l, modify only layer l and keep every other layer in the native FULL configuration.

============================================================
1. FIRST: AUDIT WHETHER 4-ACTION EXECUTION ALREADY EXISTS
============================================================

Before writing any new code, inspect the repository carefully.

Search for any existing implementation related to:

- READ / WRITE decomposition
- visual_read / visual_write
- read_only / write_only
- four-action / 4-state routing
- tri-state routing
- visual attention masking
- visual-row bypass
- BinaryQwen25VL or related executor code
- previous local READ/WRITE intervention experiments

Determine:

1. whether all four actions are already implemented,
2. exactly what each action currently does,
3. whether the implementation is compatible with the current binary executor and KV-cache/generation path,
4. whether it matches the intended semantics above.

Do NOT assume that an older implementation is correct merely because it has similar names.

If a correct implementation already exists:
- reuse it,
- make only minimal changes needed for this experiment.

If it does not exist or is incomplete:
- implement the minimum clean extension required to support the four actions.

Do not create an unnecessary parallel model implementation.

============================================================
2. REQUIRED 4-ACTION SEMANTICS
============================================================

At a given decoder layer l, starting from exactly the same pre-layer hidden state:

FULL (R=1, W=1)
- Native pretrained-model behavior.
- Visual rows execute normally.
- Text/control rows execute normally.
- Text/control rows retain direct access to visual K/V.

IGNORE (R=0, W=0)
- Must match the existing binary VISUAL_OFF semantics at that layer.
- Visual rows bypass the layer and remain unchanged.
- Text/control rows execute without direct visual K/V.

READ_ONLY (R=1, W=0)
- Visual rows bypass the layer and remain unchanged.
- Text/control rows execute normally and may directly read the current visual K/V.
- Thus visual evidence can be accessed, but visual state is not refined at this layer.

WRITE_ONLY (R=0, W=1)
- Visual rows execute/update through the native layer.
- Text/control rows execute without direct visual K/V at this layer.
- Thus the visual state may be refined, but the text stream cannot directly read visual K/V at this layer.

Preserve every other native computation unless the action explicitly changes it.

Be especially careful with:
- causal masks,
- token indexing,
- multimodal token boundaries,
- prefill vs autoregressive decoding,
- KV-cache behavior,
- attention masks,
- residual paths,
- dtype/device consistency.

Do not silently change the definition of READ or WRITE to make implementation easier.

============================================================
3. IMPLEMENTATION VALIDATION BEFORE LARGE RUNS
============================================================

Create explicit tests/sanity checks before running the analysis.

At minimum verify:

A. FULL reproduction
- The 4-action executor with FULL at every layer must reproduce the original model.
- Compare logits and generated outputs.
- Report numerical tolerance.

B. IGNORE reproduction
- A single-layer IGNORE intervention must reproduce the current binary single-layer VISUAL_OFF implementation.

C. READ_ONLY state behavior
- Visual hidden states after the intervened layer should equal their pre-layer states, up to expected numerical identity.
- Text/control states should still be able to depend on visual K/V.

D. WRITE_ONLY state behavior
- Visual rows must actually update.
- Text/control rows must not directly access visual K/V at that layer.

E. Same-input branch consistency
For a local layer-l factorial experiment:
- all four actions must begin from the same pre-layer state,
- only the action at layer l changes,
- all later layers return to FULL/native execution.

F. Generation/KV-cache correctness
Verify that READ gating remains correctly applied during decoding and that WRITE semantics remain consistent with how visual states/KV are constructed during prefill.

G. No training
The base model remains completely frozen.

Run:
1. unit/synthetic checks,
2. a few real GQA/TextVQA examples,
3. a ~5-example smoke test,
4. a ~50-example pilot,

before launching the full analysis.

If any semantic mismatch is found, fix it before proceeding.

============================================================
4. BUILD THE PRIMARY ANALYSIS COHORT
============================================================

Start with:

- GQA
- TextVQA

Use the existing binary MCTS/search cache.

Do NOT rerun MCTS just to select this cohort.

Primary inclusion criteria:

1. FULL model = WRONG.
2. ALL-OFF = WRONG.
3. At least one binary route with positive visual participation is CORRECT.

In other words:

FULL wrong
+
ALL-OFF wrong
+
nonzero-vision correcting route exists.

Exclude:

- FULL-correct samples from the primary A+ analysis.
- FULL-wrong samples rescued by ALL-OFF from the primary A+ analysis.

Important:
ALL-OFF-rescued samples can be retained in metadata for later separate analysis, but they are NOT part of the primary mechanism cohort.

Use ALL eligible primary samples available in the existing matched cache.

Do not arbitrarily force exactly 1000 examples per dataset.

First report the exact counts discovered from the current cache and verify that the taxonomy is consistent with previous cache metadata.

Also store, for every primary sample:

- dataset
- sample ID
- image ID if available
- FULL generated answer
- ground-truth/reference answer(s)
- FULL correctness
- ALL-OFF correctness
- all known correcting binary routes
- number of correcting routes
- minimum Hamming distance from FULL among correcting routes
- nearest-to-FULL correcting route(s)
- minimum ON count among correcting routes
- visual-token count
- any existing MCTS metadata/search budget

Do not filter to Hamming-1 or Hamming-2 samples.
We want the full A+ population.

Hamming distance will be used later as a stratification variable.

============================================================
5. EXHAUSTIVE LOCAL 4-ACTION SWEEP
============================================================

For every primary sample and every decoder layer l = 0,...,27:

Evaluate:

- IGNORE
- READ_ONLY
- WRITE_ONLY
- FULL

However, FULL is identical to the native baseline and should be cached/reused rather than redundantly recomputed 28 times.

Thus the expensive new branches should normally be the three non-FULL actions for each layer.

CRITICAL:

This is not a trajectory experiment.

For layer l:

layers < l : FULL
layer l    : selected one of the four actions
layers > l : FULL

Never combine interventions at multiple layers in the primary factorial sweep.

Save results incrementally and make the run resumable/shardable.

Parallelize safely across GPUs/datasets if appropriate.

============================================================
6. PRIMARY DECISION-ALIGNED SCORE
============================================================

Do not analyze only binary correct/wrong outcomes.

For every sample/action/layer compute a continuous answer-alignment score.

Preferred score:

M = S(correct answer) - S(FULL baseline wrong answer)

where S is a sequence-level model score.

Reuse an existing answer-logit / answer-probability / sequence-score implementation in this repository if one already exists and is appropriate.

If none exists, implement a clearly documented teacher-forced sequence score.

For short-answer tasks:

GQA:
- use the canonical evaluator-valid ground-truth answer.

TextVQA:
- respect its multiple-reference evaluation semantics.
- Do not arbitrarily pick a reference without documenting it.
- Prefer an evaluator-compatible set of accepted references.
- If a scalar sequence score requires choosing among multiple valid references, use a defensible rule such as the highest normalized teacher-forced score among evaluator-valid references, and report this explicitly.

Prefer length-normalized log probability for multi-token strings so that score comparisons are not trivially driven by answer length.

Also save separately:

- S_correct
- S_full_wrong
- margin M
- raw generated output
- evaluator correctness

The continuous margin is the primary causal quantity.
Discrete W/C flips are secondary behavioral evidence.

============================================================
7. FACTORIAL READ / WRITE EFFECTS
============================================================

For each sample-layer pair define:

M00 = margin under IGNORE      (R=0,W=0)
M10 = margin under READ_ONLY   (R=1,W=0)
M01 = margin under WRITE_ONLY  (R=0,W=1)
M11 = margin under FULL        (R=1,W=1)

Compute:

READ effect given WRITE on:
Delta_READ_W1 = M11 - M01

READ effect given WRITE off:
Delta_READ_W0 = M10 - M00

WRITE effect given READ on:
Delta_WRITE_R1 = M11 - M10

WRITE effect given READ off:
Delta_WRITE_R0 = M01 - M00

READ-WRITE interaction:
Interaction = M11 - M10 - M01 + M00

Sign convention:

negative Delta_READ_*:
enabling READ moves the model AWAY from the correct answer.

negative Delta_WRITE_*:
enabling WRITE moves the model AWAY from the correct answer.

These are candidate answer-unaligned operations.

Do not use an arbitrary hard threshold initially.
Report full effect distributions first.

Later report prevalence under several magnitude thresholds derived from the observed scale, not chosen to maximize a result.

============================================================
8. DISCRETE LOCAL RESCUE TAXONOMY
============================================================

Because FULL is wrong in the primary cohort, classify each sample-layer by what happens under the other three actions.

Important cases:

A. READ_ONLY correct, WRITE_ONLY wrong
- Removing WRITE while retaining READ is sufficient for rescue.
- Candidate answer-unaligned WRITE.

B. WRITE_ONLY correct, READ_ONLY wrong
- Removing READ while retaining WRITE is sufficient for rescue.
- Candidate answer-unaligned READ.

C. READ_ONLY correct AND WRITE_ONLY correct
- Removing either component is sufficient.
- Do not automatically call both individually causal; report as an "either-removal-sufficient" class.

D. IGNORE correct, but READ_ONLY and WRITE_ONLY both wrong
- Joint removal is required locally.
- Candidate READ-WRITE interaction / full visual-participation failure.

E. None of the three actions correct
- No single-layer local 4-action intervention is sufficient for final behavioral correction.
- Continuous effects can still be large and scientifically important.

Report:
- per-dataset counts
- percentages
- number of unique samples with at least one local rescue
- number of rescue layers per sample
- depth distribution of rescue layers

Do not discard category E.

============================================================
9. DEPTH-WISE ANALYSIS
============================================================

Analyze all 28 layers.

For GQA and TextVQA separately and jointly report:

- mean/median Delta_READ_W1 by layer
- mean/median Delta_READ_W0 by layer
- mean/median Delta_WRITE_R1 by layer
- mean/median Delta_WRITE_R0 by layer
- interaction by layer
- fraction of negative effects by layer
- fraction of strong-negative effects by layer
- discrete READ-removal rescue frequency by layer
- discrete WRITE-removal rescue frequency by layer
- joint-removal rescue frequency by layer

Use bootstrap confidence intervals over samples.

If samples share the same image, prefer image-group bootstrap or at least report whether grouping materially changes confidence intervals.

Do not infer a mid/late-layer story unless the actual layerwise distributions support it.

============================================================
10. STRATIFY BY BINARY ROUTE DISTANCE
============================================================

For every A+ sample compute the Hamming distance between FULL and its nearest known correcting binary route.

Do NOT use this to select the dataset.

Use it only for analysis.

Suggested strata, adapting bins if counts are sparse:

- distance 1
- distance 2
- distance 3-4
- distance 5-8
- distance >8

Ask:

- Do distance-1/2 samples tend to contain one very strong harmful READ or WRITE operation?
- Do larger-distance samples show several weaker negative operations?
- Does interaction strength increase with required binary route distance?
- Does the number of negative local operations correlate with nearest correcting-route distance?

Report both continuous and discrete effects.

============================================================
11. CONNECT LOCAL 4-ACTION EFFECTS TO EXISTING BINARY ROUTES
============================================================

This is an important analysis.

For each sample, we already know which layers are OFF in its correcting binary routes.

Test whether layers suppressed by successful binary routes are enriched for locally answer-unaligned READ/WRITE effects.

At minimum analyze:

A. Nearest-to-FULL correcting route
Compare:
- layers OFF in the route
vs
- layers ON in the route

in terms of local:
- Delta_READ_W1
- Delta_WRITE_R1
- M00 - M11
- strongest negative component

B. All known correcting routes
For each layer, compute:
- fraction of correcting routes in which that layer is OFF.

Ask whether this OFF-frequency correlates with local harmfulness.

C. Ranking analysis
For each sample:
- rank layers by strongest local harmful READ/WRITE effect.
- test whether binary routes preferentially turn off top-ranked harmful layers.

Useful metrics may include:
- enrichment over same-size random layer sets
- recall@k
- average rank
- within-sample rank correlation
- AUROC/AUPRC only if the target definition is statistically sensible

Do not force a metric if its assumptions are inappropriate.

The scientific question is:

"Are successful binary routes suppressing layers that local causal analysis independently identifies as answer-unaligned?"

============================================================
12. FULL-MODEL ANSWER EROSION ANALYSIS
============================================================

We have previously observed that some wrong samples show substantial support for the correct answer at intermediate layers, followed by a drop in middle/late layers.

Analyze this systematically.

For the FULL model, compute a layerwise answer-alignment trajectory using a consistent logit-lens style readout.

Use:
- the same correct-vs-FULL-wrong target whenever possible,
- final norm + LM head or the repository's existing logit-lens convention,
- teacher-forced answer positions for multi-token answers.

Document the exact readout.

For each sample compute at least:

- maximum intermediate correct-vs-wrong margin
- layer of maximum margin
- final margin
- peak-to-final erosion:
    E = max_l M_l - M_final
- largest negative adjacent-layer change
- layer of largest negative drop

Compare these between:
- A+ primary samples
- controls described later.

Then ask:

1. Are A+ errors frequently cases where the correct answer was internally supported before being eroded later?
2. Does the layer of strongest negative local READ/WRITE effect align with the layer or neighborhood of strongest answer erosion?
3. If a culprit READ/WRITE operation is disabled, is the later erosion reduced?

For alignment analysis use robust quantities such as:
- absolute layer distance between strongest harmful operation and strongest logit drop
- fraction within ±1 layer / ±2 layers
- permutation/random-layer baselines.

Do not overinterpret raw intermediate logit-lens values as causal evidence.
The exact 4-action intervention is primary.
The logit trajectory is supporting temporal evidence.

============================================================
13. TRAJECTORY RESCUE FOR STRONG CULPRITS
============================================================

For samples with strong local causal effects or discrete local rescue:

store/visualize both:

FULL trajectory:
layerwise answer margin

and

single-operation-suppressed trajectory:
same model except the identified local READ or WRITE is removed.

Examples:

- harmful WRITE candidate:
  FULL vs READ_ONLY at culprit layer

- harmful READ candidate:
  FULL vs WRITE_ONLY at culprit layer

Ask:

"Does removing the identified operation prevent the downstream collapse of correct-answer support?"

Quantify this across the population, not only with cherry-picked examples.

============================================================
14. CONTROL COHORTS
============================================================

The primary A+ cohort is selected because a correcting binary route already exists.

Therefore it can establish existence/mechanism, but not unbiased prevalence across all model errors.

Add controls after the primary analysis is validated.

Control 1: FULL-wrong, no correcting route found in the matched binary search
- Use the existing D-like cohort.
- Be precise: "no correction found within the existing search budget," not "no correcting route exists."

Control 2: FULL-correct, ALL-OFF-wrong
- vision-required correct samples.
- This tests whether negative READ/WRITE effects are simply common everywhere or are especially associated with correctable errors.

Use all controls if computationally reasonable, or a deterministic matched sample if needed.
If sampling controls:
- match dataset,
- preferably visual-token-count distribution,
- use a fixed seed,
- record IDs.

Primary comparisons:

A+ vs no-correction-found wrong:
- strength/prevalence of negative READ/WRITE effects

A+ vs FULL-correct vision-required:
- whether strong answer-unaligned operations are associated with errors rather than generic local variability

Do not mix controls into the primary A+ headline before presenting primary results clearly.

============================================================
15. OPTIONAL MECHANISTIC FOLLOW-UP AFTER CAUSAL LOCALIZATION
============================================================

Only after identifying robust harmful READ/WRITE cases, perform representation/attention analysis.

Do NOT use attention maps or visual-token logit lens as primary proof.

They are explanatory follow-ups to exact causal interventions.

----------------------------------------
15A. Harmful WRITE cases
----------------------------------------

For strongest WRITE cases inspect:

- pre-WRITE visual states
- post-WRITE visual states
- WRITE residual:
    Delta V_l = V_after - V_before

Possible analyses:

1. Visual-token logit lens before vs after WRITE
   Look for interpretable concept drift.

Example qualitative pattern:
"car" evidence decreases while an incorrect concept such as "tree" increases.

But explicitly label logit-lens interpretation as diagnostic, not ground truth.

2. Decision-aligned WRITE contribution
If tractable, compute patch/token-level signed contribution:

G_l = gradient of correct-vs-wrong margin w.r.t. post-WRITE visual state

contribution_j =
    <G_l,j, Delta V_l,j>

Interpret:
positive = update locally supports the correct-answer margin
negative = update locally opposes it.

Compare the summed/aggregated signed contribution with the exact WRITE intervention effect.

----------------------------------------
15B. Harmful READ cases
----------------------------------------

For strongest READ cases inspect:

- text-to-visual attention maps
- attended visual regions
- but do NOT equate high attention with causal support

If tractable compute decision-aligned patch-level READ contribution using:
- visual attention weights
- value vectors
- gradient/sensitivity of the correct-vs-wrong answer margin

Produce signed spatial maps:
- regions supporting correct answer
- regions pushing toward the wrong answer

Look for cases such as:
- question refers to left person
- harmful READ emphasizes evidence associated with right person
- exact READ removal improves or corrects the answer.

Again, exact intervention is primary evidence.
Attention is only supporting visualization.

============================================================
16. QUERY-BLIND WRITE ANALYSIS — SECONDARY, NOT ASSUMED
============================================================

Inspect the actual Qwen2.5-VL causal ordering carefully.

If visual rows cannot attend to later question tokens under the native causal layout, document that visual WRITE is query-blind in that precise sense.

Do NOT automatically conclude this is harmful.

As an optional secondary analysis, if same-image multi-question pairs exist, test whether the same visual WRITE can have different decision-aligned effects for different questions.

For example:
same image, same layer WRITE,
Q1: positive WRITE effect
Q2: negative WRITE effect.

This would support query-dependent utility of otherwise query-blind visual refinement.

Only report this if there is enough matched data.

============================================================
17. COMPUTE / EXECUTION PRACTICES
============================================================

This analysis is large.

Requirements:

- smoke-test first
- make every long run resumable
- shard by dataset/sample range
- save intermediate outputs
- never discard completed shards
- reuse FULL baseline outputs
- avoid recomputing image preprocessing/visual embeddings unnecessarily
- batch intervention branches where semantically safe
- preserve exact deterministic evaluation
- log model revision/config/checkpoint
- log git commit and working-tree diff
- use fixed seeds where relevant

Before the full run:
- benchmark throughput on a small shard,
- estimate total GPU-hours,
- record the estimate.

Do not silently reduce the scientific scope because of runtime.

If exhaustive greedy generation for every sample-layer-action is disproportionately expensive:
- exhaustive teacher-forced answer-margin evaluation remains mandatory,
- estimate the cost of exhaustive generation separately,
- preferentially still run full generation if practical,
- otherwise document a principled generation-validation subset based on predeclared criteria rather than cherry-picking.

Do not substitute a cheaper proxy without explicitly documenting it.

============================================================
18. DO NOT DO THESE THINGS
============================================================

For this task, DO NOT:

- run a new 4-action MCTS label search,
- train a new 4-action router,
- train/fine-tune the base MLLM,
- change the existing binary labels,
- Pareto-filter the cohort for this analysis,
- keep only Hamming-1/2 examples,
- cherry-pick only examples that show the desired phenomenon,
- conclude that READ or WRITE is harmful from attention/logit-lens alone,
- claim that a route is impossible merely because MCTS did not find it,
- conflate ALL-OFF with "no image information whatsoever."

Remember:
ALL-OFF means no decoder layer has direct visual access under the project's executor semantics; upstream/image-derived information or other side channels may still exist.

============================================================
19. REQUIRED OUTPUT ARTIFACTS
============================================================

Create a clearly organized output directory, e.g.

analysis/4action_answer_alignment/

with at least:

1. `implementation_audit.md`
   - what 4-action code existed
   - what was reused
   - what was changed
   - exact semantics
   - validation results

2. `cohort_summary.json` or `.csv`
   - counts and IDs for GQA/TextVQA primary/control cohorts

3. per-sample/per-layer structured results
   Prefer Parquet/JSONL with fields including:
   - dataset
   - sample_id
   - image_id
   - layer
   - action
   - S_correct
   - S_full_wrong
   - margin
   - generated_answer
   - correctness
   - M00/M10/M01/M11
   - four conditional main effects
   - interaction
   - binary route metadata
   - nearest correcting route distance

4. aggregate tables
   - per-layer effects
   - rescue taxonomy
   - dataset comparisons
   - Hamming strata
   - route-overlap analysis
   - answer-erosion analysis
   - controls

5. figures
   At minimum:
   - READ effect vs layer
   - WRITE effect vs layer
   - interaction vs layer
   - local rescue prevalence vs layer
   - effect distributions
   - Hamming-distance stratification
   - binary OFF-frequency vs local harmfulness
   - answer erosion curves
   - culprit-layer vs collapse-layer alignment

6. `4action_answer_unaligned_report.md`

The final report should clearly separate:

A. implementation facts
B. empirical observations
C. causal claims directly supported by interventions
D. supporting correlational/representation evidence
E. negative results
F. limitations
G. implications for whether a future 4-action MCTS/router is scientifically justified

============================================================
20. FINAL DECISION CRITERIA
============================================================

At the end, explicitly answer:

1. Do answer-unaligned READ operations exist at nontrivial frequency in A+ samples?
2. Do answer-unaligned WRITE operations exist at nontrivial frequency?
3. Which is stronger/more prevalent, if either?
4. Are there meaningful READ×WRITE interactions?
5. At what depths do these effects occur?
6. Do single-operation removals actually rescue final answers?
7. Do strong negative effects align with the observed erosion of correct-answer support?
8. Do successful binary correcting routes preferentially suppress layers identified as locally harmful?
9. Are these effects enriched in A+ samples relative to appropriate controls?
10. Is the evidence strong enough to justify a full 4-action trajectory search and a 4-action router?

The desired scientific outcome is NOT predetermined.

Possible valid conclusions include:

- harmful WRITE dominates,
- harmful READ dominates,
- both occur with distinct depth patterns,
- joint interaction matters more than either marginal effect,
- local effects exist but do not explain multi-layer correcting routes,
- negative effects are equally common in correct controls and therefore not error-specific,
- the 4-action expansion is not justified.

Report whichever result the data supports.

Do not force the narrative.

============================================================
21. WORKING STYLE
============================================================

Start by auditing the repository and writing a short execution plan based on the actual code and cache structure you find.

Then implement/test only what is necessary.

Do not spend excessive time refactoring unrelated code.

Prefer:
inspect -> minimal implementation -> unit checks -> 5-sample smoke -> ~50-sample pilot -> full primary run -> controls -> mechanism follow-up -> final report.

Keep a running experiment log so that another researcher can reproduce every result.