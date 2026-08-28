You are continuing the `dynamic_mllm` project.

Your task is to convert the project's existing authoritative BINARY MCTS route
labels into FOUR-ACTION visual-computation labels.

DO NOT rerun MCTS from scratch.

We already spent substantial compute discovering useful binary visual routes.
The goal now is to reuse those routes as seeds and refine them into the
four-action space.

======================================================================
0. HIGH-LEVEL GOAL
======================================================================

The existing binary route has 28 layer actions:

    ON  = native visual participation
    OFF = no direct visual participation at that layer

The new four-action route has:

    FULL
        READ=1, WRITE=1

    READ_ONLY
        READ=1, WRITE=0

    WRITE_ONLY
        READ=0, WRITE=1

    IGNORE
        READ=0, WRITE=0

Binary mapping:

    ON  -> FULL
    OFF -> IGNORE

We already know from the previous route-conditioned analysis that binary OFF
positions frequently contain more structure:

- some only require READ suppression;
- some only require WRITE suppression;
- some require both;
- many binary OFF positions are unnecessary/redundant.

Therefore the new labels should refine the EXISTING binary route rather than
searching the full 4^28 space again.

======================================================================
1. DATASETS TO CONVERT
======================================================================

Convert ALL authoritative label records for the following datasets.

----------------------------------------------------------------------
A. WeMath labels
----------------------------------------------------------------------

Source root:

    datasets/math_labels

This directory contains MCTS labels for:

    - WeMath2.0 Standard
    - WeMath2.0 Pro

First inspect the directory carefully and identify the exact authoritative
label artifacts for both datasets.

Do not guess file names.

Use the current authoritative/final MCTS route labels and preserve all source
metadata/provenance.

----------------------------------------------------------------------
B. GQA / TextVQA / ChartQA
----------------------------------------------------------------------

Use ONLY:

    datasets/mcts_labels/gqa_textvqa_chartqa_v1/

This is the current canonical regenerated label collection.

DO NOT use:

    datasets/mcts_v2/

Do not merge old and new caches.

The historical `mcts_v2` labels are tied to an older execution contract and
some positive masks did not reproduce under the repaired executor.

======================================================================
2. WHAT "CONVERT ALL LABELS" MEANS
======================================================================

Process every sample in the authoritative positive-route supervision set for:

    GQA
    TextVQA
    ChartQA
    WeMath2.0 Standard
    WeMath2.0 Pro

Do not subsample samples.

Do not convert negative evaluated masks into positive labels.

Before starting, audit each source tree and determine which artifact is the
training-authoritative positive-route label view.

If a source contains both:

    raw MCTS evaluated masks
    raw valid routes
    derived selected/max-50 training routes
    predictor manifests

then use the project's CURRENT TRAINING-AUTHORITATIVE POSITIVE ROUTE VIEW as the
conversion input.

Do not blindly convert millions of raw negative/evaluated masks merely because
they are stored in the raw cache.

Do not silently choose between multiple candidate manifests.

Write the exact chosen source artifacts and counts to:

    implementation_audit.md

before the full run.

Within the chosen authoritative input view:

    process ALL samples
    process ALL positive binary route labels
    do not randomly subsample routes

If several source binary routes for the same sample eventually refine to the
same four-action route, deduplicate the final four-action route while retaining
the complete list of source binary-route IDs that mapped to it.

======================================================================
3. FIRST AUDIT THE EXISTING FOUR-ACTION EXECUTOR
======================================================================

Reuse the already validated unified four-action executor from the previous:

    - FULL-context four-action analysis
    - route-conditioned READ/WRITE decomposition

Do NOT create another independent executor unless absolutely necessary.

Verify that it still implements:

    FULL       = READ ON,  WRITE ON
    READ_ONLY  = READ ON,  WRITE OFF
    WRITE_ONLY = READ OFF, WRITE ON
    IGNORE     = READ OFF, WRITE OFF

Verify that arbitrary multi-layer four-action routes can be executed.

Verify that:

    binary ON     == unified FULL
    binary OFF    == unified IGNORE

under the same semantics used in the previous analyses.

Reuse all existing:

    evaluators
    answer scoring
    image preprocessing
    visual-token utilities
    deterministic generation settings
    route execution machinery
    resume/sharding infrastructure

Do not refactor unrelated code.

======================================================================
4. REVALIDATE EVERY SOURCE BINARY ROUTE
======================================================================

Before refining a source binary positive route:

1. Convert it mechanically:

       ON  -> FULL
       OFF -> IGNORE

2. Replay it using the CURRENT unified four-action executor.

3. Verify that it is still evaluator-correct.

Only a currently valid route may be converted into a positive four-action
label.

If a historical/source route no longer reproduces:

    - do not invent a replacement;
    - do not launch new MCTS;
    - record it as source_route_replay_failure;
    - keep complete provenance.

Also recompute the CURRENT unified FULL result for the sample so that route type
is based on the current execution contract, not only historical metadata.

Classify each valid route as:

    W2C:
        FULL wrong
        route correct

    C2C:
        FULL correct
        route correct

Also tag:

    all_off_seed = true/false

where the binary source route is ALL-OFF.

======================================================================
5. IMPORTANT: W2C AND C2C HAVE DIFFERENT SEMANTICS
======================================================================

Do NOT mix their interpretation.

----------------------------------------------------------------------
W2C
----------------------------------------------------------------------

FULL is wrong but the route is correct.

This is a CORRECTIVE / ANSWER-ALIGNMENT label.

Scientific interpretation:

    some modification of the visual execution trajectory repairs an error.

For W2C, we want to identify the MINIMUM visual-operation suppression needed
to preserve the correction.

Therefore W2C routes SHOULD be actively refined from binary OFF into
READ/WRITE-level actions.

----------------------------------------------------------------------
C2C
----------------------------------------------------------------------

FULL is already correct and the binary route remains correct.

This is primarily a CORRECTNESS-PRESERVING / REDUNDANCY / EFFICIENCY label.

Do NOT call its OFF operations "answer-unaligned".

For C2C:

    ON  -> FULL
    OFF -> IGNORE

is already a valid four-action representation of the existing label.

Do NOT perform W2C-style "restore redundant OFF to FULL" purification on C2C,
because FULL is correct by definition and that procedure would trivially
collapse C2C supervision toward FULL.

Preserve C2C routes as efficiency/correctness-preserving labels.

Store:

    label_semantics = corrective_w2c

or:

    label_semantics = preserving_c2c

so the two can never be accidentally conflated during later training or
analysis.

======================================================================
6. W2C CONVERSION: BINARY ROUTE PURIFICATION
======================================================================

For every valid W2C binary source route:

    binary ON  -> FULL
    binary OFF -> IGNORE

The initial converted route is therefore already correct.

Example:

    binary route:

        L2 OFF
        L4 OFF
        L9 OFF
        others ON

    initial four-action anchor:

        L2 IGNORE
        L4 IGNORE
        L9 IGNORE
        others FULL

This is only the starting point.

Previous analysis showed that many binary OFF positions are redundant.

We do NOT want the new corrective four-action labels to inherit unnecessary
suppression.

Therefore perform ROUTE PURIFICATION.

For each IGNORE/OFF position:

    temporarily restore that layer to FULL
    while keeping every other layer at the current anchor route.

If correctness remains:

    permanently restore that position to FULL.

If correctness breaks:

    keep it suppressed for now.

Repeat passes until no remaining suppressed position can individually be
restored to FULL without losing correctness.

This produces a deterministic 1-minimal-ish corrective binary anchor under the
chosen restoration procedure.

Important:

    do NOT test an OFF layer under the all-FULL context.

Always test restoration inside the CURRENT correcting-route context.

Because restoration order can matter:

    use a deterministic documented ordering.

If implementation cost is small, also try both:

    early-to-late
    late-to-early

restoration orders and keep the candidate with:

    1. correctness
    2. fewer suppressed READ/WRITE components
    3. higher answer-alignment margin as tie-breaker

Do not explode into combinatorial enumeration merely to find a globally
minimal binary route.

======================================================================
7. W2C CONVERSION: REFINE NECESSARY IGNORE POSITIONS
======================================================================

After purification, only the remaining suppressed positions should be treated
as candidate answer-unaligned visual computation.

At each remaining IGNORE position, the possible relaxations are:

    IGNORE
        READ=0 WRITE=0

    READ_ONLY
        READ=1 WRITE=0
        only WRITE remains suppressed

    WRITE_ONLY
        READ=0 WRITE=1
        only READ remains suppressed

    FULL
        READ=1 WRITE=1
        no suppression

The goal is NOT to assign each layer independently.

Previous experiments showed strong trajectory dependence.

Therefore use the successful correcting route as the context.

Example:

    purified correct anchor:

        L2 IGNORE
        L4 IGNORE
        all others FULL

When evaluating L2:

        keep L4 IGNORE

and test:

        L2 IGNORE
        L2 READ_ONLY
        L2 WRITE_ONLY
        L2 FULL

When evaluating L4:

        keep L2 at the current correcting-route action.

Never reset all non-target layers to FULL for this refinement.

======================================================================
8. NO NEW MCTS: USE BOUNDED MONOTONE JOINT REFINEMENT
======================================================================

Do NOT run MCTS.

We already have a correct binary anchor.

We only need to relax unnecessary suppression while preserving correctness.

Use a small deterministic bounded search from the correct anchor TOWARD FULL.

State:

    a complete 28-layer four-action route

Initial state:

    purified W2C anchor

Allowed transitions:

    IGNORE -> READ_ONLY
    IGNORE -> WRITE_ONLY
    IGNORE -> FULL

and, when relevant:

    READ_ONLY  -> FULL
    WRITE_ONLY -> FULL

Only original/purified binary-OFF positions are allowed to vary.

All layers that were FULL after purification remain FULL.

A candidate is valid only if it remains evaluator-correct.

Use a small beam / monotone refinement procedure, NOT MCTS.

Recommended initial beam width:

    8

unless an existing repository implementation suggests a better safe bounded
value.

Search objective is lexicographic:

    1. route must be correct
    2. minimize suppression from FULL
    3. maximize answer-alignment margin
    4. deterministic tie-break

Define suppression-component cost:

    FULL       = 0
    READ_ONLY  = 1
    WRITE_ONLY = 1
    IGNORE     = 2

Therefore:

    smaller total suppression cost is preferred.

This implements the intended corrective-label semantics:

    "Keep the model as close to native FULL as possible while suppressing only
     visual operations required to repair the error."

Do NOT optimize primarily for minimum visual compute in W2C.

The purpose of W2C labels is correction of answer-unaligned visual computation,
not aggressive pruning.

======================================================================
9. JOINT VALIDITY IS MANDATORY
======================================================================

Do not construct the final label by independently choosing each layer's best
local action.

Example:

    L2 READ_ONLY individually correct
    L4 WRITE_ONLY individually correct

does NOT imply:

    L2 READ_ONLY + L4 WRITE_ONLY

is jointly correct.

Every final four-action route must be executed as a COMPLETE trajectory and
verified correct.

This is essential because the previous route-conditioned analysis found strong
trajectory dependence.

======================================================================
10. ALL-OFF W2C CASES
======================================================================

Do NOT automatically discard W2C samples whose source binary correcting route is
ALL-OFF.

Include them in conversion.

Tag:

    all_off_seed = true

An ALL-OFF seed simply begins as:

    28 x IGNORE

Then use the same corrective purification/refinement logic:

    restore FULL where possible
    refine remaining IGNORE into READ_ONLY / WRITE_ONLY where possible
    preserve correctness

The objective is to discover whether an apparently ALL-OFF correction can be
represented by a much smaller selective set of READ/WRITE suppressions.

However, keep this group separately identifiable in all analyses because its
scientific interpretation differs from A+ vision-required W2C samples.

Report results separately for:

    W2C with ALL-OFF correct
    W2C with ALL-OFF wrong

Do not merge them when making answer-unaligned mechanism claims.

======================================================================
11. MULTIPLE SOURCE ROUTES PER SAMPLE
======================================================================

A sample may contain many existing valid binary labels.

Process all routes in the chosen authoritative source label view.

However, aggressively reuse computation.

Within one sample:

    - preprocess image once
    - cache unified FULL
    - cache route evaluations by exact 28-layer four-action route
    - share the cache across all source binary routes for that sample
    - never re-evaluate an identical four-action route unnecessarily

After refinement:

    deduplicate identical final four-action routes.

Preserve:

    source_binary_route_ids = [...]

for every deduplicated four-action label.

Therefore many binary routes that collapse to the same refined four-action route
do not receive duplicated training weight.

======================================================================
12. OUTPUT LABEL SETS
======================================================================

For each sample preserve:

    dataset
    sample_id
    image_id if available
    source split
    current unified FULL answer
    FULL correctness
    source binary route ID
    source binary route
    source binary OFF count
    source route correctness
    W2C / C2C
    all_off_seed
    label_semantics

For every final four-action route preserve:

    28-layer action sequence
    FULL count
    READ_ONLY count
    WRITE_ONLY count
    IGNORE count

    read_suppression_count
    write_suppression_count
    total suppression-component cost

    generated answer
    evaluator correctness
    correct-answer score
    answer-alignment margin if defined

    source binary route IDs
    conversion provenance
    executor hash/version

Create at least three views:

1. RAW CONVERSION VIEW
   - mapping from every authoritative source binary label to its final
     four-action conversion

2. UNIQUE VALID FOUR-ACTION VIEW
   - per-sample deduplicated valid four-action routes

3. TRAINING VIEW
   - deterministic bounded valid-set view suitable for later router training
   - if a sample has >50 unique valid four-action routes, create a diverse
     max-50 view using the project's existing diversity-selection philosophy
   - preserve the full unique set separately

Also create:

    canonical_4action_route

per sample.

For W2C canonical route:

    minimum suppression-component cost among correct routes
    then highest answer-alignment margin
    then deterministic tie-break

For C2C:

    preserve the existing efficiency/correctness semantics;
    do not reinterpret it as a corrective canonical route.

======================================================================
13. GPU EXECUTION CONFIGURATION
======================================================================

Use ALL 8 GPUs.

Preferred execution:

    2 independent worker processes per GPU

Therefore:

    8 GPUs x 2 processes/GPU = 16 concurrent worker processes

Each worker process:

    - owns one model replica
    - is pinned to exactly one GPU
    - processes ONE SAMPLE AT A TIME
    - processes all binary labels/refinements for that sample serially
    - then moves to the next sample

Do NOT batch multiple samples together inside one process unless required to fix
a correctness bug.

The intended architecture is approximately:

    GPU 0:
        worker 0
        worker 1

    GPU 1:
        worker 2
        worker 3

    ...

    GPU 7:
        worker 14
        worker 15

Each process should load the model once and reuse it.

Do not reload the model for every sample.

Use the previously validated two-replica-per-GPU infrastructure if compatible.

Before the full run, perform a short pilot to verify:

    - both replicas fit safely in VRAM
    - no OOM
    - no executor semantic change
    - deterministic outputs
    - throughput is acceptable
    - GPU utilization is materially improved

The user's preferred configuration is 2 processes/GPU.

Use it unless it is technically unstable or causes a severe throughput
regression.

If it fails:

    diagnose first,
    fix concurrency/sharding if possible,
    only fall back to 1 process/GPU if 2 processes/GPU cannot be made reliable.

Document any fallback explicitly.

======================================================================
14. LOAD BALANCING
======================================================================

Do NOT statically divide only by sample count if costs differ greatly.

A sample's cost depends on:

    - number of source positive routes
    - number of OFF positions
    - number of unique route states evaluated during refinement

Prefer a shared work queue or many smaller work units so that all 16 workers stay
busy.

If a shared queue is inconvenient:

    generate substantially more than 16 deterministic shards
    and let workers consume shards as they finish.

Estimate sample cost before launch using something like:

    number_of_source_routes
    x average_OFF_count

or a better observed pilot estimate.

The goal is to avoid:

    one GPU finishing early
    while another GPU remains stuck on high-route-count samples.

======================================================================
15. RESUMABILITY AND FAILURE SAFETY
======================================================================

The full conversion may be expensive.

Requirements:

    append-only outputs
    per-worker progress files
    atomic completed-record writes
    resume without recomputing completed samples
    exact sample/route IDs
    no duplicated final labels
    worker/GPU provenance
    deterministic seeds
    checksum or hash sidecars where consistent with project practice

A failed worker must not invalidate completed work from other workers.

Do not overwrite the original binary labels.

======================================================================
16. PILOT
======================================================================

Before full conversion, run a stratified pilot containing examples from ALL five
datasets:

    GQA
    TextVQA
    ChartQA
    WeMath2.0 Standard
    WeMath2.0 Pro

Include:

    W2C
    C2C
    short binary routes
    long binary routes
    ALL-OFF W2C if available
    multi-route samples

Use enough samples to exercise all code paths, approximately 40-80 total.

Validate:

1. binary ON -> FULL parity
2. binary OFF -> IGNORE parity
3. source binary positive route reproduction
4. W2C purification preserves correctness
5. C2C routes are not accidentally collapsed to FULL
6. READ_ONLY / WRITE_ONLY semantics
7. joint refinement correctness
8. cache reuse
9. deduplication
10. resume behavior
11. two processes/GPU execution
12. all-eight-GPU scheduling

If any correctness/semantic issue occurs:

    STOP
    fix
    rerun pilot
    verify cleanly
    then automatically start full conversion.

Do not wait for manual approval after a clean pilot.

======================================================================
17. OUTPUT LOCATION
======================================================================

Do NOT modify the source binary label directories.

Create a separate new root, for example:

    datasets/mcts_labels_4action/

Suggested structure:

    mcts_labels_4action/
        gqa_textvqa_chartqa_v1/
        wemath2_standard/
        wemath2_pro/

If the exact WeMath source naming suggests better matching names, preserve those
names and document them.

Store:

    source manifest
    conversion manifest
    execution contract
    per-sample outputs
    unique route sets
    canonical labels
    training views
    split manifests
    audit reports
    checksums

Preserve original dataset splits.

Do not introduce image leakage between existing train/validation partitions.

======================================================================
18. REQUIRED FINAL REPORT
======================================================================

Create:

    four_action_label_conversion_report.md

and dataset-specific summaries.

For each dataset report:

    - total source samples
    - total source positive binary labels
    - replay-valid labels
    - replay-invalid labels
    - W2C count
    - C2C count
    - ALL-OFF W2C count
    - number of source binary routes converted
    - number of unique final four-action routes
    - deduplication ratio
    - average source OFF count
    - average W2C purified OFF count
    - average final suppression-component cost
    - FULL / READ_ONLY / WRITE_ONLY / IGNORE action frequencies
    - number/fraction of W2C routes using READ_ONLY
    - number/fraction of W2C routes using WRITE_ONLY
    - number/fraction still requiring IGNORE
    - number/fraction refined all the way back to FULL at specific positions
    - distribution over model depth
    - route length / suppression distributions
    - source -> refined answer-margin changes
    - pilot throughput
    - full-run throughput
    - GPU utilization
    - peak memory
    - total GPU-hours
    - all failures/exclusions

Report GQA/TextVQA/ChartQA and WeMath Standard/Pro separately.

Also provide combined summaries, but never hide dataset-specific behavior.

======================================================================
19. IMPORTANT SCIENTIFIC ANALYSES
======================================================================

For W2C labels specifically answer:

1. How much binary OFF redundancy disappeared during purification?

2. Among remaining corrective positions, how often could binary IGNORE be
   relaxed to:

       READ_ONLY
       WRITE_ONLY
       FULL

3. How often was IGNORE still required?

4. Are WRITE suppressions more common than READ suppressions?

5. Are their depth distributions different?

6. How different are GQA, TextVQA, ChartQA, WeMath Standard, and WeMath Pro?

7. Do ALL-OFF-seed W2C samples refine into selective positive-vision routes?

8. How many distinct refined four-action routes exist per sample?

9. How much does joint validation reduce the number of seemingly valid
   independent relaxations?

For C2C labels separately report:

    - route/action distribution
    - efficiency/redundancy structure

but DO NOT call C2C suppression "answer-unaligned".

======================================================================
20. DO NOT DO THESE THINGS
======================================================================

Do NOT:

    - rerun binary MCTS
    - run four-action MCTS
    - search all 4^28 trajectories
    - use datasets/mcts_v2
    - mix historical and regenerated GQA/TextVQA/ChartQA labels
    - overwrite source labels
    - treat C2C OFF as evidence of harmful computation
    - independently combine locally valid READ/WRITE relaxations without
      joint execution verification
    - drop ALL-OFF W2C without recording/converting it
    - silently subsample the authoritative label set
    - silently switch executor/scorer semantics
    - silently fall back from 2 processes/GPU
    - optimize GPU utilization at the expense of correctness

======================================================================
21. FINAL DELIVERABLE / DECISION
======================================================================

At the end, explicitly state:

1. Exact authoritative source artifact used for each of the five datasets.

2. Exact number of binary labels successfully converted.

3. Exact number of unique valid four-action labels produced.

4. Whether the conversion recovered substantial READ_ONLY / WRITE_ONLY structure
   beyond binary FULL/IGNORE.

5. Whether W2C and C2C should remain separate supervision types.

6. Whether the resulting four-action valid sets are sufficiently rich and clean
   for training a four-action trajectory predictor/router.

7. Any remaining reason that a fresh four-action search would still be needed.

The expected default is that fresh MCTS should NOT be necessary unless the
conversion reveals a clear coverage failure.

Proceed:

    audit sources
    ->
    audit executor
    ->
    implement minimal converter
    ->
    5-dataset stratified pilot
    ->
    validate 2 processes/GPU on all 8 GPUs
    ->
    fix if necessary
    ->
    full conversion
    ->
    exact merge/dedup
    ->
    build training views
    ->
    final report