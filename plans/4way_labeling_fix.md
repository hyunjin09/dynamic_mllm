You are continuing the `dynamic_mllm` project.

Your task is to convert the project's existing authoritative BINARY MCTS route
labels into answer-aligned READ/WRITE route labels.

DO NOT rerun MCTS.

The existing binary MCTS routes already localize useful regions of the route
space. We will reuse those routes and refine their binary visual-OFF operations
into READ/WRITE-level suppressions.

The main conceptual change from the previous plan is:

    We do NOT need to treat FULL as a fourth search action.

For a layer whose native FULL behavior is already known/cached, the three
suppression actions of interest are:

    READ_OFF
        READ=0, WRITE=1
        equivalent to WRITE_ONLY

    WRITE_OFF
        READ=1, WRITE=0
        equivalent to READ_ONLY

    BOTH_OFF
        READ=0, WRITE=0
        equivalent to IGNORE

FULL = READ=1, WRITE=1 is the reference/native state and should be reused from
cache whenever possible.

Thus this experiment should be treated as:

    existing binary route
        ->
    3-action READ/WRITE suppression refinement
        +
    continuous answer-support analysis

rather than a fresh four-action search.

======================================================================
0. DATASETS
======================================================================

Convert all authoritative positive binary route labels for:

    GQA
    TextVQA
    ChartQA
    WeMath2.0 Standard
    WeMath2.0 Pro

Sources:

A. GQA / TextVQA / ChartQA

    datasets/mcts_labels/gqa_textvqa_chartqa_v1/

Use this regenerated canonical collection only.

DO NOT use:

    datasets/mcts_v2/

B. WeMath2.0 Standard / Pro

    datasets/math_labels

Inspect this directory first and identify the authoritative/current MCTS
artifacts for Standard and Pro.

Do not guess file names.

Write the exact selected artifacts, counts, and provenance to an audit before
starting the full conversion.

======================================================================
1. CORE SCIENTIFIC IDEA
======================================================================

The new labels are intended to characterize ANSWER-ALIGNED VISUAL COMPUTATION.

Binary VISUAL_OFF conflates two operations:

    visual READ
        text/control accesses visual K/V

    visual WRITE
        visual representations are updated/refined

We want to determine whether suppressing:

    READ only,
    WRITE only,
    or both

is useful for the answer.

Importantly, usefulness is NOT defined only by whether one intervention
individually flips the final answer.

A visual suppression can be useful even if the model remains wrong after that
single change, as long as it causally moves the model toward the correct answer.

Therefore every candidate must be evaluated using BOTH:

    1. evaluator correctness
    2. continuous correct-answer support / decision margin

Final positive route labels must still be evaluator-correct.

Continuous answer support is used to:
    - identify useful suppressions,
    - distinguish redundant OFF operations,
    - guide joint refinement,
    - rank multiple valid routes.

======================================================================
2. REUSE THE VALIDATED EXECUTOR
======================================================================

Reuse the existing unified READ/WRITE executor already validated in:

    - FULL-context four-action analysis
    - route-conditioned READ/WRITE decomposition

Required semantics:

    FULL
        READ=1 WRITE=1

    READ_OFF / WRITE_ONLY
        READ=0 WRITE=1

    WRITE_OFF / READ_ONLY
        READ=1 WRITE=0

    BOTH_OFF / IGNORE
        READ=0 WRITE=0

Do NOT implement another parallel executor.

Verify on a smoke set that:

    binary ON  == unified FULL
    binary OFF == unified BOTH_OFF/IGNORE

Reuse the current benchmark scorers, preprocessing, generation settings,
answer-scoring utilities, and deterministic execution contract.

======================================================================
3. ANSWER-SUPPORT SCORE
======================================================================

Use a continuous decision-aligned score in addition to evaluator correctness.

For W2C, prefer a fixed correct-vs-original-FULL-wrong score:

    M_W2C(r)
      = S(correct answer | route r)
        - S(original FULL wrong answer | route r)

The original FULL wrong answer must remain fixed across route comparisons.

For multi-token answers, use the repository's existing normalized
teacher-forced sequence score.

For TextVQA/multiple valid references, preserve the evaluator-compatible
reference handling already used in the previous four-action analysis.

For C2C there is no original FULL-wrong answer.

Use:

    A_C2C(r) = normalized support for an evaluator-valid correct answer

as the mandatory continuous score.

If a stable incorrect competitor can be defined without introducing
benchmark-specific artifacts, also save:

    M_C2C(r)
      = S(correct) - S(best fixed incorrect competitor)

but do NOT require such a margin if it is not reliable.

The primary C2C continuous quantity may therefore be called:

    correct-answer support

rather than decision margin.

======================================================================
4. CALIBRATE NUMERICAL NOISE
======================================================================

Do NOT treat a tiny score increase as meaningful.

During the pilot, evaluate repeated identical routes and estimate within-unified
score repeatability.

Define a numerical tolerance epsilon from this empirical repeatability.

For example, use an appropriately conservative high percentile of absolute
repeat differences.

Do not use native-vs-unified executor drift as this threshold.

Save all raw continuous effects regardless of threshold.

Report sensitivity for:

    delta > 0
    delta > epsilon

The threshold must be fixed before the full aggregate analysis.

======================================================================
5. CLASSIFY ROUTES BY CURRENT FULL BEHAVIOR
======================================================================

Replay/recompute current unified FULL correctness for every source sample.

Every replay-valid positive binary route must be tagged as:

    W2C:
        FULL wrong
        source binary route correct

    C2C:
        FULL correct
        source binary route correct

Also record:

    all_off_seed = true/false

Do not infer W2C/C2C only from historical metadata.

Any source positive route that does not replay as correct under the current
unified executor must be recorded and excluded from positive supervision.

Do NOT search for a replacement route.

======================================================================
6. W2C: SCIENTIFIC OBJECTIVE
======================================================================

For W2C:

    FULL = wrong
    source route = correct

The question is:

    "Which visual suppressions move the model toward and ultimately recover
     the correct answer?"

We want to distinguish two kinds of useful suppression:

A. HARD / DISCRETE NECESSITY

    restoring a suppressed operation makes the correcting route wrong.

B. SOFT / ALIGNMENT CONTRIBUTION

    restoring the operation may leave the route correct,
    but significantly decreases correct-answer margin.

Therefore an OFF position must NOT be discarded merely because restoring FULL
does not immediately change correct -> wrong.

If keeping the suppression significantly improves correct-answer margin, it is
still potentially answer-aligned and should remain a refinement candidate.

======================================================================
7. W2C: SCREEN BINARY OFF POSITIONS
======================================================================

For every source W2C binary route:

    ON  -> FULL
    OFF -> BOTH_OFF

The binary source route is already a correct anchor.

For every OFF position l, evaluate a route-conditioned FULL restoration:

    target l = FULL
    all other layers remain exactly as in the current correcting route

Compare against the current BOTH_OFF anchor.

Let:

    M_off  = correct-answer margin with l=BOTH_OFF
    M_full = correct-answer margin with l=FULL

Classify:

--------------------------------------------------
W2C-HARD
--------------------------------------------------

If:

    BOTH_OFF route = correct
    FULL-restored route = wrong

then suppression at l is DISCRETELY NECESSARY in this route context.

Keep l as a 3-action refinement candidate.

--------------------------------------------------
W2C-SOFT
--------------------------------------------------

If:

    BOTH_OFF route = correct
    FULL-restored route = correct
    M_off > M_full + epsilon

then suppression at l is not necessary for discrete correctness,
but it provides a significant positive answer-alignment contribution.

Keep l as a SOFT alignment-improving candidate.

--------------------------------------------------
W2C-REDUNDANT
--------------------------------------------------

If:

    FULL restoration remains correct
    and
    M_off <= M_full + epsilon

then there is no evidence that suppressing this visual operation helps the
answer.

Restore this position to FULL and remove it from corrective supervision.

This is the important difference from the original binary MCTS labels:

    old binary label:
        OFF may remain simply because OFF was tolerated

    new answer-aligned label:
        suppression is retained only if it contributes to correctness
        or meaningfully improves correct-answer support.

======================================================================
8. W2C: 3-ACTION DECOMPOSITION
======================================================================

For every retained W2C-HARD or W2C-SOFT position, decompose the suppression into:

    READ_OFF
    WRITE_OFF
    BOTH_OFF

FULL is NOT a new search action here.

FULL has already been evaluated/cached during screening.

Likewise, BOTH_OFF is normally the cached source/current anchor state.

Therefore the only NEW expensive evaluations per retained position should
normally be:

    READ_OFF
    WRITE_OFF

This is a major compute-saving requirement.

Evaluate these actions in the CURRENT correcting-route context.

Never reset all other layers to FULL.

For each action save:

    evaluator correctness
    correct-answer score
    W2C margin
    delta versus FULL-restored state
    delta versus BOTH_OFF state

======================================================================
9. W2C: INTERPRETATION OF THE 3 ACTIONS
======================================================================

Suppose FULL restoration is harmful/useful suppression was established.

Then:

A. READ_OFF works best / is needed

    READ=0, WRITE=1

Interpretation:
    READ suppression is answer-aligned.

B. WRITE_OFF works best / is needed

    READ=1, WRITE=0

Interpretation:
    WRITE suppression is answer-aligned.

C. BOTH_OFF is needed

Interpretation:
    both visual READ and WRITE need suppression in that context.

D. READ_OFF and WRITE_OFF both preserve correctness/support

Interpretation:
    either component removal may be sufficient;
    do not force a unique culprit label.

Do not choose the final label from discrete correctness alone.

Use the continuous score to distinguish cases such as:

    READ_OFF  -> wrong, margin improves strongly
    WRITE_OFF -> wrong, margin barely changes

because the first action may still be an important partial correction when
combined with other trajectory changes.

======================================================================
10. W2C: JOINT REFINEMENT WITHOUT MCTS
======================================================================

Do NOT combine independently chosen per-layer actions without validation.

Previous experiments showed strong trajectory dependence.

Use the source correcting binary route as a warm start and perform a bounded
joint refinement over only the retained candidate positions.

No MCTS.

Use a deterministic small beam / coordinate-beam refinement.

Search space per candidate layer:

    READ_OFF
    WRITE_OFF
    BOTH_OFF

FULL is allowed only when a candidate is being removed/reverted based on the
screening/refinement logic; it is not treated as a fourth exploration branch.

Suggested initial beam width:

    8

unless pilot results show a clearly better bounded setting.

For partial WRONG candidates:

    use improvement in W2C margin as a search-ranking signal.

A wrong candidate is NOT a positive label, but should remain explorable if it
moves the model strongly toward the correct answer.

For complete CORRECT routes:

    retain them as positive valid routes.

======================================================================
11. W2C: FINAL ROUTE OBJECTIVE
======================================================================

Do not collapse everything into one label too early.

Preserve a valid-set of answer-aligned routes.

For each correct refined route store:

    evaluator correctness
    answer margin
    suppression-component count
    READ_OFF count
    WRITE_OFF count
    BOTH_OFF count

Define suppression cost:

    FULL      = 0
    READ_OFF  = 1
    WRITE_OFF = 1
    BOTH_OFF  = 2

Keep the Pareto frontier over at least:

    lower suppression cost
    higher correct-answer margin

For a canonical W2C route:

    1. must be correct
    2. prefer smaller suppression cost,
       provided answer margin is not meaningfully worse than the best
       available corrective seed
    3. then prefer higher answer margin
    4. deterministic tie-break

Also save separately:

    max_margin_route

because the most answer-aligned route and the minimally intervened route may
not always be identical.

======================================================================
12. C2C: SCIENTIFIC OBJECTIVE
======================================================================

For C2C:

    FULL = correct

We are NOT asking:

    "what computation caused the error?"

There is no final error.

Instead ask:

    "Does the correct model still contain visual operations that weaken the
     correct answer, even though the remaining computation compensates for
     them?"

Thus C2C is used to identify:

    COMPENSATED ANSWER-UNALIGNED VISUAL COMPUTATION.

Conceptually:

    weak/moderate answer-unaligned visual operation
        -> model remains correct (C2C)

    stronger / jointly accumulated answer-unaligned visual operation
        -> model becomes wrong (W2C)

This is a hypothesis to test, not an assumed conclusion.

======================================================================
13. C2C: USE EXISTING BINARY ROUTES AS CANDIDATE LOCALIZATION
======================================================================

Do NOT perform 28 layers x 3 actions for every C2C sample by default.

Use the existing valid binary C2C routes to cheaply identify candidate
positions first.

For each source C2C route:

    source OFF -> BOTH_OFF

The source route is correct.

For each OFF position, compare the BOTH_OFF state with a route-conditioned FULL
reference where that position is restored while all other source-route actions
are unchanged.

This FULL reference is a screening/baseline state, not a fourth search action.

A C2C OFF position is interesting if suppressing visual participation:

    1. preserves correctness
    AND
    2. increases correct-answer support by > epsilon

relative to its FULL-restored route-conditioned reference.

Also retain positions where restoring FULL makes the route wrong, because this
shows strong context dependence even though the native all-FULL model is
globally correct.

Tag these separately.

If:

    BOTH_OFF and FULL-restored are both correct
    and suppression does NOT improve answer support beyond epsilon

then this OFF is only tolerated/redundant from an answer-alignment perspective.

Do not use it as an answer-unaligned label.

======================================================================
14. C2C: OPTIONAL FULL-TRAJECTORY LOGIT SCREEN
======================================================================

The native FULL model is correct.

For analysis, save the layerwise correct-answer support trajectory.

This can identify locations where the correct answer becomes weaker.

However:

    DO NOT treat a layerwise logit/support drop by itself as causal evidence.

Previous experiments showed that the largest logit erosion and strongest local
causal operation did not reliably align layer-by-layer.

Therefore FULL-trajectory logit drops may be used only as:

    candidate-screening / descriptive evidence

not as the label itself.

If desired, add a SECONDARY candidate set consisting of layers with unusually
large negative correct-answer-support changes, then validate those candidates
with the same 3-action intervention.

Keep binary-route-derived and logit-screen-derived candidates separately
tagged.

Do not automatically expand to all 28 layers unless pilot analysis shows that
binary-route localization misses substantial useful C2C operations.

======================================================================
15. C2C: 3-ACTION CAUSAL TEST
======================================================================

For each retained C2C candidate layer evaluate:

    READ_OFF
    WRITE_OFF
    BOTH_OFF

FULL is the cached/reference state and should not be rerun unnecessarily.

Every candidate action must satisfy:

    evaluator remains correct

to become a positive C2C label.

Then measure:

    delta_correct_support
      = A_candidate - A_full_reference

If a reliable decision margin is available, save its delta as well.

A suppression is alignment-improving when:

    evaluator remains correct
    and
    delta_correct_support > epsilon

Classify:

    READ_ALIGNMENT_GAIN
    WRITE_ALIGNMENT_GAIN
    EITHER_ALIGNMENT_GAIN
    BOTH_REQUIRED_FOR_GAIN
    NO_MEANINGFUL_GAIN

Do not call tiny score changes answer-unaligned.

======================================================================
16. C2C: IMPORTANT DIFFERENCE FROM EFFICIENCY LABELS
======================================================================

The new C2C labels are NOT simply:

    "this computation can be removed while staying correct."

That would be a pure efficiency/redundancy objective.

The current scientific objective is stronger:

    "suppressing this visual operation preserves correctness AND improves
     correct-answer support."

Therefore:

    correct + lower compute only
        != answer-aligned C2C label

unless answer support also improves meaningfully.

You may retain pure efficiency metadata separately, but do not mix it with the
answer-alignment supervision.

======================================================================
17. C2C: JOINT VALIDATION
======================================================================

As with W2C, local C2C improvements cannot simply be combined independently.

If:

    L2 READ_OFF improves support
    L8 WRITE_OFF improves support

then:

    L2 READ_OFF + L8 WRITE_OFF

must be executed jointly before becoming a valid multi-layer label.

Use the same bounded beam/coordinate refinement framework.

For C2C partial candidates:

    hard constraint:
        remain evaluator-correct

    ranking:
        larger correct-answer-support improvement
        with lower suppression cost

Store the Pareto frontier over:

    answer support
    suppression cost

======================================================================
18. W2C VS C2C LABEL SEMANTICS
======================================================================

Never conflate them.

Every route must contain:

    route_type:
        W2C or C2C

    label_semantics:

        W2C_HARD_CORRECTIVE
        W2C_SOFT_ALIGNMENT
        C2C_COMPENSATED_ALIGNMENT

Possible layer-level tags:

        HARD_NECESSARY
        SOFT_ALIGNMENT_HELPFUL
        REDUNDANT
        READ_SUPPRESSION
        WRITE_SUPPRESSION
        BOTH_SUPPRESSION
        EITHER_SUPPRESSION

For later training this allows separate weighting of:

    actual error corrections
    vs
    margin-improving but already-correct examples.

======================================================================
19. ALL-OFF W2C
======================================================================

Include W2C samples where ALL-OFF is correct, but tag them:

    all_off_seed = true

Do not merge their mechanism statistics with:

    FULL wrong
    ALL-OFF wrong
    selective nonzero-vision correction

without reporting them separately.

For an ALL-OFF W2C route, use the same logic:

    BOTH_OFF states are the seed
    restore/screen operations toward FULL
    use margin changes
    retain only suppressions that contribute to correctness or answer support
    refine retained positions into READ_OFF / WRITE_OFF / BOTH_OFF

This tests whether apparently global visual suppression can actually be
explained by a smaller selective answer-aligned route.

======================================================================
20. MULTIPLE EXISTING BINARY ROUTES
======================================================================

Process all positive routes in the authoritative conversion view.

Within one sample:

    preprocess image once
    cache FULL once
    share exact-route execution cache across all source labels

If multiple binary routes refine to the same 3-action trajectory:

    deduplicate the final route

but retain:

    source_binary_route_ids = [...]

Do not give duplicate refined routes extra training weight.

======================================================================
21. COMPUTATIONAL PRINCIPLE: 3 ACTIONS, NOT 4
======================================================================

Make this explicit in code and reporting.

At a target layer, the conceptual alternatives of interest are:

    READ_OFF
    WRITE_OFF
    BOTH_OFF

FULL is the reference.

Do not blindly execute all four states every time.

Reuse:

    native/unified FULL cache
    route-conditioned FULL restoration cache
    source BOTH_OFF route cache

In many W2C cases:

    FULL reference = already evaluated during candidate screening
    BOTH_OFF       = already cached from binary route

Therefore only:

    READ_OFF
    WRITE_OFF

require new model execution.

The implementation should exploit this aggressively.

Report:

    theoretical 4-state evaluations avoided
    actual cache hit rate
    number of new forwards per converted label

======================================================================
22. GPU EXECUTION
======================================================================

Use all 8 GPUs.

Preferred layout:

    2 worker processes per GPU
    = 16 total workers

Each process:

    loads one model replica once
    handles ONE SAMPLE AT A TIME
    processes all routes/candidate actions for that sample
    reuses image/model/route caches within the sample
    then moves to the next sample

Preferred:

    GPU0: workers 0,1
    GPU1: workers 2,3
    ...
    GPU7: workers 14,15

Do not batch multiple independent samples inside one process unless needed for a
correctness/performance fix.

Use many small balanced work units or a shared work queue.

Estimate per-sample cost from:

    number of source routes
    number of candidate OFF positions
    expected new READ_OFF/WRITE_OFF evaluations

Do not balance only by raw sample count.

Before full launch, benchmark the 2-process/GPU configuration.

Primary metric:

    valid route/candidate evaluations per second

GPU utilization is a secondary metric.

Use 2 processes/GPU unless it is unstable or causes a severe measured
throughput regression.

======================================================================
23. PILOT
======================================================================

Before full conversion, run a stratified 5-dataset pilot.

Include:

    GQA
    TextVQA
    ChartQA
    WeMath2.0 Standard
    WeMath2.0 Pro

Include examples of:

    W2C hard-necessary
    W2C soft-alignment
    C2C
    ALL-OFF W2C
    short routes
    long routes
    multi-route samples

Approximately 50-100 samples total is sufficient if all execution paths are
covered.

Validate:

    executor parity
    source route replay
    score repeatability / epsilon
    W2C hard/soft screening
    C2C answer-support screening
    READ_OFF semantics
    WRITE_OFF semantics
    BOTH_OFF cache reuse
    FULL cache reuse
    joint validation
    deduplication
    resume behavior
    16-worker execution
    GPU memory/throughput

If anything fails:

    stop
    fix
    rerun pilot

After a clean pilot:

    automatically launch the full conversion.

======================================================================
24. OUTPUT VIEWS
======================================================================

Do not overwrite original labels.

Suggested root:

    datasets/mcts_labels_3action/

or another clearly documented new directory.

Create at least:

1. source_conversion_view

    every source binary label
    ->
    all screening/refinement provenance

2. unique_valid_route_view

    deduplicated correct 3-action/FULL trajectories

3. W2C corrective training view

4. C2C alignment-improving training view

5. combined training manifest

    but with W2C/C2C semantics preserved

6. canonical routes

For every route save:

    28-layer action sequence
    evaluator correctness
    answer score/margin
    suppression cost
    READ_OFF count
    WRITE_OFF count
    BOTH_OFF count
    source binary route IDs
    all_off_seed
    route_type
    label_semantics
    executor/version hashes

======================================================================
25. REQUIRED ANALYSIS: W2C
======================================================================

For each dataset separately and jointly report:

    number of W2C samples/routes

    fraction of binary OFF positions that are:
        HARD necessary
        SOFT alignment-helpful
        REDUNDANT

    among retained positions:
        READ suppression preferred/required
        WRITE suppression preferred/required
        BOTH suppression preferred/required
        either

    discrete correction effect
    continuous margin improvement

    depth distribution

    how many useful suppressions would have been missed by a correctness-only
    criterion

This last quantity is especially important.

Explicitly quantify:

    "OFF does not individually flip correctness but improves answer margin"

because this is a central motivation for using the continuous signal.

======================================================================
26. REQUIRED ANALYSIS: C2C
======================================================================

For each dataset separately and jointly report:

    number of C2C samples/routes

    number of candidate OFF positions

    fraction where suppression:
        preserves correctness but does not improve support
        improves correct support > epsilon

    among alignment-improving positions:
        READ_OFF
        WRITE_OFF
        BOTH_OFF
        either

    mean/median correct-answer-support gain

    depth distribution

    fraction of C2C samples with at least one robust alignment-improving visual
    suppression

    distribution of number of such operations per sample

Compare C2C effect magnitudes against W2C.

This tests the hypothesis that:

    C2C contains weaker/compensated answer-unaligned visual computation,
    while W2C contains stronger or jointly accumulated forms.

Do not state this hypothesis as a conclusion unless supported.

======================================================================
27. IMPORTANT COMPARISON: W2C VS C2C
======================================================================

Directly compare:

    effect magnitude
    READ vs WRITE composition
    depth
    number of affected layers
    BOTH-off prevalence
    joint-dependence
    score improvement

Potentially interesting outcome:

    same kinds of negative visual operations occur in both,
    but effect magnitude / accumulation separates C2C from W2C.

Alternative valid outcomes:

    W2C and C2C have qualitatively different mechanisms.

Report whichever the data supports.

======================================================================
28. FINAL LABEL ACCEPTANCE RULE
======================================================================

A route may enter POSITIVE training supervision only if:

    evaluator correctness = true

Continuous answer support alone is NOT enough.

For W2C:

    wrong partial routes with improved margin
    are search/guidance evidence only.

For C2C:

    a suppression must preserve correctness.

Save wrong-but-margin-improved W2C partial routes separately as:

    corrective_partial_candidates

They may be useful later for ranking or RL, but they are not positive route
labels.

======================================================================
29. DO NOT DO
======================================================================

Do NOT:

    rerun MCTS
    search 4^28
    blindly evaluate FULL as a fourth action everywhere
    use old mcts_v2 labels
    use raw correct-logit changes without evaluator correctness
    call logit-lens drops causal
    treat tiny numerical score improvements as meaningful
    construct final labels by independently combining local actions
    call C2C suppression "error correction"
    call every binary OFF answer-unaligned
    overwrite original labels
    silently subsample source positive labels

======================================================================
30. FINAL REPORT
======================================================================

Produce:

    three_action_answer_aligned_label_conversion_report.md

It must explicitly answer:

1. How many authoritative binary labels from each of the five datasets were
   successfully replayed and converted?

2. How much binary-OFF redundancy was removed?

3. How many W2C suppressions were:
       hard correctness-critical
       soft margin-improving
       redundant?

4. How many correction-supporting operations would have been missed if we used
   only discrete W->C flips?

5. In W2C, how often is READ_OFF, WRITE_OFF, or BOTH_OFF preferred?

6. In C2C, how often can suppression preserve correctness AND increase
   correct-answer support?

7. Are C2C answer-unaligned effects weaker than W2C effects?

8. Are READ and WRITE distributed differently over depth?

9. How much joint trajectory validation changes conclusions from independent
   per-layer refinement?

10. How many final unique valid 3-action/FULL trajectories exist per sample?

11. Are the resulting W2C/C2C valid sets clean enough to train an
    answer-aligned visual trajectory router?

12. Is there any remaining empirical reason to run a fresh MCTS search?

Proceed:

    source audit
    ->
    executor/cache audit
    ->
    score/noise calibration
    ->
    stratified pilot
    ->
    validate 2 processes/GPU x 8 GPUs
    ->
    full W2C/C2C conversion
    ->
    joint refinement
    ->
    deduplication
    ->
    training manifests
    ->
    final analysis/report