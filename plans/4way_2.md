You are continuing the existing four-action answer-alignment experiment in the
`dynamic_mllm` repository.

Do NOT interrupt or modify the currently running trajectory-rescue experiment
or its dependent final analysis.

Your next task is to run a new experiment:

    Route-Conditioned READ/WRITE Decomposition of Known Binary Correcting Routes

This experiment must begin only after the current run and final analysis finish
successfully.

The purpose is to answer a different and more direct causal question than the
existing FULL-context single-layer sweep.

======================================================================
0. EXECUTION ORDER — DO NOT CHANGE
======================================================================

Current pipeline:

1. Complete the currently running trajectory-rescue job.
2. Complete its validation/merge.
3. Complete the dependent final aggregate analysis/report.
4. Verify that the current experiment completed successfully and that there is
   no disqualifying implementation/numerical-consistency failure.
5. THEN immediately start the route-conditioned decomposition pilot.
6. If the pilot passes all semantic/numerical checks:
      launch the full frozen A+ cohort immediately.
7. If the pilot reveals an implementation or semantic problem:
      fix the problem,
      rerun the pilot,
      verify it,
      THEN launch the full frozen A+ cohort.

Do not wait for manual approval between the successful pilot and the full run
unless a genuinely ambiguous scientific/implementation issue is discovered.

The previously mentioned Slurm jobs were:
- trajectory rescue: 1572
- dependent final analysis: 1573

However, do not rely only on job IDs because jobs may have been resumed or
requeued. Verify completion from the authoritative output artifacts and logs.

======================================================================
1. WHY THIS NEW EXPERIMENT IS NEEDED
======================================================================

The completed/current four-action experiment is a FULL-context local
intervention.

For a target layer l, it evaluates:

    all other layers = FULL
    target layer     = one of:
                       FULL
                       READ_ONLY
                       WRITE_ONLY
                       IGNORE

This measures the local marginal causal effect of READ/WRITE under the dense
FULL model.

That experiment is valuable and must remain unchanged.

However, it does NOT directly explain why a known MULTI-LAYER binary correcting
route succeeds.

Example:

    FULL:
        all layers ON
        -> WRONG

    known binary correcting route:
        layer 2  = OFF
        layer 6  = OFF
        all others = ON
        -> CORRECT

It is entirely possible that:

    only layer 2 OFF -> still wrong
    only layer 6 OFF -> still wrong
    layers 2+6 OFF   -> correct

Therefore a FULL-context single-layer intervention may fail to identify the
mechanism even though the joint correcting route is real.

The new experiment must preserve the context of the ACTUALLY SUCCESSFUL
correcting route and decompose each OFF layer into READ and WRITE.

Scientific question:

    "Within a known successful binary correcting route, why does each visual-OFF
     position need to be suppressed?

     Is suppressing READ sufficient?
     Is suppressing WRITE sufficient?
     Must both be suppressed?
     Or was that OFF position unnecessary/redundant in the correcting route?"

======================================================================
2. DO NOT RUN A NEW SEARCH YET
======================================================================

This is NOT:

- a new binary MCTS search,
- a 4-action MCTS search,
- a new router training run,
- an exhaustive 4^28 trajectory search.

Use the EXISTING frozen binary correcting-route cache.

We are decomposing already-discovered successful routes.

======================================================================
3. USE THE FROZEN PRIMARY A+ COHORT
======================================================================

Use the exact frozen primary A+ sample IDs produced by the completed four-action
experiment.

Do NOT reconstruct the cohort from an older cache if a current authoritative
cohort manifest exists.

The latest known frozen counts are:

    GQA:       1,222
    TextVQA:     658
    Total:     1,880

Definition:

    FULL = wrong
    ALL-OFF = wrong
    at least one non-ALL-OFF binary correcting route exists

These samples are ideal because:

    vision cannot simply be removed globally,
    yet changing the visual execution trajectory can correct the answer.

If the authoritative final cohort manifest after the current run differs
slightly, use the authoritative manifest and document the exact final counts.

======================================================================
4. FIRST AUDIT EXISTING CODE
======================================================================

Before adding code, inspect the repository.

The current unified four-action executor should already implement:

    FULL
        READ=1, WRITE=1

    READ_ONLY
        READ=1, WRITE=0

    WRITE_ONLY
        READ=0, WRITE=1

    IGNORE
        READ=0, WRITE=0

Reuse this implementation.

Do NOT create another independent READ/WRITE executor unless absolutely
necessary.

Determine whether the current code already supports:

    - arbitrary multi-layer binary route context
    - overriding one layer within that route
    - using the unified executor for all four actions
    - extracting answer scores/evaluator correctness
    - resumable/sharded execution

If route-conditioned execution is missing, implement the smallest clean
extension needed.

======================================================================
5. SELECT ONE PRIMARY BINARY CORRECTING ROUTE PER SAMPLE
======================================================================

A sample may have many binary correcting routes.

For the PRIMARY route-conditioned analysis, choose one deterministic anchor
correcting route per sample.

Preferred rule:

1. Consider only known non-ALL-OFF routes marked correct in the existing cache.
2. Prefer the route with minimum Hamming distance from FULL.
   Equivalently, prefer the correcting route with the fewest OFF layers.
3. If several routes tie:
      prefer the route with the strongest valid answer-alignment score if such
      scores are directly comparable;
      otherwise use a deterministic stable tie-break and document it.
4. Record all tied candidates in metadata.

Why minimum Hamming distance from FULL?

Because it provides the smallest known perturbation that corrects the model and
therefore gives the cleanest route-level mechanism analysis.

CRITICAL:

Before using an anchor route, verify that it is still CORRECT under the CURRENT
unified executor/evaluator.

If the selected route fails current-runtime validation:

    try the next nearest known correcting route deterministically.

If no cached correcting route remains correct under the current unified
executor:

    do NOT invent a route,
    do NOT rerun MCTS automatically,
    exclude the sample from this route-conditioned analysis,
    record the reason,
    and report the count.

The original A+ cohort remains unchanged; this only defines
route-conditioned-analyzable samples.

======================================================================
6. ROUTE-CONDITIONED FACTORIAL DECOMPOSITION
======================================================================

Suppose the validated anchor binary route is:

    L2 = OFF
    L6 = OFF
    all other layers = FULL

and this route is CORRECT.

The anchor route therefore corresponds to:

    L2 = BOTH_OFF
    L6 = BOTH_OFF

where:

    BOTH_OFF = READ=0, WRITE=0

For EACH OFF layer independently, keep EVERY OTHER layer exactly as specified
by the correcting anchor route.

Example: analyze L2 while preserving L6=OFF.

Evaluate:

    A. BOTH_OFF
       L2: READ=0, WRITE=0
       L6: BOTH_OFF
       -> anchor baseline, already known correct

    B. READ_OFF
       L2: READ=0, WRITE=1
       L6: BOTH_OFF

       This is identical to WRITE_ONLY at L2.

    C. WRITE_OFF
       L2: READ=1, WRITE=0
       L6: BOTH_OFF

       This is identical to READ_ONLY at L2.

    D. FULL RESTORE
       L2: READ=1, WRITE=1
       L6: BOTH_OFF

Then analyze L6 separately while keeping L2=BOTH_OFF:

    L2: BOTH_OFF

    L6:
        BOTH_OFF
        READ_OFF
        WRITE_OFF
        FULL

IMPORTANT:

This is NOT the same as the previous FULL-context intervention.

Previous experiment:

    target layer changes
    every other layer = FULL

New experiment:

    target layer changes
    every other layer = the known correcting binary route

This distinction is the entire point of the new experiment.

======================================================================
7. THREE NEW EVALUATIONS PER OFF POSITION
======================================================================

The BOTH_OFF state is the anchor correcting route and should normally already
be available/cached.

Therefore, for each OFF position, only three NEW states need to be evaluated:

    READ_OFF   = WRITE_ONLY
    WRITE_OFF  = READ_ONLY
    FULL

Reuse the BOTH_OFF score/output when numerical/execution identity is guaranteed.

If there is any doubt, verify BOTH_OFF reproduction during the pilot.

For an anchor route with K OFF layers, the basic experiment therefore requires
approximately:

    3 * K

new route evaluations per sample,

not 4^K and not 3^K.

Do NOT launch an exhaustive joint combinatorial refinement in this stage.

======================================================================
8. WHAT EACH OUTCOME MEANS
======================================================================

The anchor BOTH_OFF route is correct.

For each OFF layer, classify the route-conditioned relaxation results.

--------------------------------------------------
Case 0: FULL restoration remains CORRECT
--------------------------------------------------

    BOTH_OFF   correct
    FULL       correct

Interpretation:

    This OFF position is not individually necessary in this anchor-route
    context.

Do NOT attribute the correction to harmful READ or harmful WRITE at this layer.

Label:

    REDUNDANT / NON-ESSENTIAL OFF

This result is important because MCTS correcting routes may contain unnecessary
OFF positions.

--------------------------------------------------
Case 1: READ_OFF correct, WRITE_OFF wrong
--------------------------------------------------

Recall:

    READ_OFF = READ=0, WRITE=1

Therefore visual WRITE can be restored, but READ must remain suppressed.

Interpretation:

    READ suppression is sufficient/necessary in this route context.

Candidate:

    READ-mediated correction
    / answer-unaligned READ

--------------------------------------------------
Case 2: WRITE_OFF correct, READ_OFF wrong
--------------------------------------------------

Recall:

    WRITE_OFF = READ=1, WRITE=0

Therefore visual READ can be restored, but WRITE must remain suppressed.

Interpretation:

    WRITE suppression is sufficient/necessary in this route context.

Candidate:

    WRITE-mediated correction
    / answer-unaligned WRITE

--------------------------------------------------
Case 3: READ_OFF correct AND WRITE_OFF correct
--------------------------------------------------

Either suppression alone preserves correction.

Interpretation:

    EITHER-REMOVAL-SUFFICIENT

Do NOT claim a uniquely identified culprit.

--------------------------------------------------
Case 4: READ_OFF wrong AND WRITE_OFF wrong,
        while BOTH_OFF correct
--------------------------------------------------

Neither operation alone can be restored.

Interpretation:

    BOTH READ AND WRITE must remain suppressed at this layer,
    conditioned on the rest of the correcting route.

Candidate:

    joint READ+WRITE requirement / within-layer interaction.

--------------------------------------------------
Case 5: numerical/evaluator inconsistency
--------------------------------------------------

Do not classify.

Record and diagnose.

======================================================================
9. ALSO COMPUTE CONTINUOUS ROUTE-CONDITIONED EFFECTS
======================================================================

Do not analyze only final correct/wrong flips.

Use the same answer-alignment score as the current four-action experiment.

Prefer:

    M = S(correct answer) - S(original FULL-model wrong answer)

Keep the target identity fixed across all route-conditioned states.

For an OFF position under the anchor-route context define:

    M00 = BOTH_OFF
    M10 = WRITE_OFF = READ_ONLY
    M01 = READ_OFF  = WRITE_ONLY
    M11 = FULL

Then compute the complete 2x2 factorial quantities:

    Delta_READ_W0  = M10 - M00
    Delta_WRITE_R0 = M01 - M00

    Delta_READ_W1  = M11 - M01
    Delta_WRITE_R1 = M11 - M10

    Interaction =
        M11 - M10 - M01 + M00

Be explicit about the sign convention.

Because the anchor M00 route is correct, a negative change when restoring an
operation means that restoring that operation moves the route away from the
correct answer.

Also save:

    - generated output
    - evaluator correctness
    - S_correct
    - S_original_FULL_wrong
    - margin
    - anchor route ID/mask
    - target OFF layer
    - target action
    - anchor OFF count K
    - anchor Hamming distance from FULL

======================================================================
10. THIS EXPERIMENT IS CONDITIONAL, NOT GLOBAL CAUSAL ATTRIBUTION
======================================================================

Use precise language.

If READ must remain OFF at layer l while the other anchor-route suppressions are
held fixed, the valid conclusion is:

    "READ suppression at layer l is required/sufficient for preserving this
     correction under the anchor-route context."

Do NOT immediately generalize this to:

    "READ at layer l is globally harmful."

The previous FULL-context experiment addresses local behavior under dense
execution.

This new experiment addresses conditional behavior inside a successful
correcting route.

The combination of both experiments is scientifically useful.

======================================================================
11. PILOT BEFORE FULL A+
======================================================================

After current trajectory-rescue + final analysis complete, run a
route-conditioned pilot first.

Use approximately 48-64 samples total, stratified across:

    - GQA / TextVQA
    - small anchor OFF count
    - medium anchor OFF count
    - larger anchor OFF count
    - different Hamming-distance ranges

Do not select only easy or Hamming-1 samples.

The pilot must verify:

A. Anchor reproduction
- BOTH_OFF must reproduce the validated correcting route.
- Correctness must match.
- Scores must match within expected numerical tolerance.

B. Route context preservation
- all non-target layers must remain exactly at anchor-route actions.

C. Target isolation
- only the target OFF layer is changed among
  BOTH_OFF / READ_OFF / WRITE_OFF / FULL.

D. Four-action semantics
- READ_OFF really means R=0,W=1.
- WRITE_OFF really means R=1,W=0.
- FULL really means R=1,W=1.

E. Scoring
- exact same answer targets are used across all states.

F. Evaluator
- GQA/TextVQA evaluator behavior is deterministic and consistent.

G. Resume/sharding
- no duplicate records after resume.
- completed records are preserved.

H. Throughput
- benchmark wall-clock useful throughput.

If any semantic/numerical issue is found:

    STOP the full launch,
    fix it,
    rerun the pilot,
    verify the fix,
    then continue automatically.

======================================================================
12. GPU UTILIZATION / THROUGHPUT OPTIMIZATION
======================================================================

The current intervention workload has shown low GPU utilization in some runs.

For this new experiment, actively attempt to improve hardware utilization.

However:

    PRIMARY optimization objective:
        completed valid intervention cells / wall-clock second

    SECONDARY diagnostic:
        GPU utilization %

Do not choose a configuration solely because nvidia-smi reports a larger
utilization if useful throughput becomes worse.

During the pilot, benchmark practical concurrency strategies that require
minimal code risk.

Candidates include:

1. one model replica per GPU, current serial execution
2. sample batching if supported safely
3. batching multiple route/action branches where tensor shapes permit
4. two concurrent workers/processes per GPU if VRAM permits
5. more than two concurrent workers only if a short benchmark shows an actual
   throughput gain and memory remains safe

It is acceptable to use multiple processes/model replicas per GPU if they
increase useful throughput for THIS workload.

Do not reject them merely because a previous workload behaved differently.

Likewise, do not keep them merely because GPU utilization looks high.

Measure:

    - completed intervention cells/sec
    - completed samples/sec
    - peak VRAM
    - GPU utilization
    - failure/OOM rate

Choose the fastest stable configuration.

Try to use otherwise-idle GPU capacity when possible.

======================================================================
13. SHARDING / LOAD BALANCING
======================================================================

Do NOT shard only by equal sample count.

The cost of a sample depends strongly on:

    K = number of OFF layers in its anchor correcting route

because the approximate new evaluation count is:

    3K

Therefore estimate cost per sample using at least anchor OFF count K and create
balanced shards by expected INTERVENTION CELL COUNT.

Create more shards than GPUs so that long-tail samples do not leave GPUs idle.

For example, with 8 GPUs, use enough shards/work units to allow dynamic or
near-dynamic load balancing.

If running multiple concurrent workers per GPU is beneficial, distribute shards
accordingly.

Requirements:

- append-only/resumable outputs
- deterministic sample assignment
- no duplicate work
- completed records skipped on resume
- each record includes worker/shard provenance

======================================================================
14. FULL RUN AFTER PILOT
======================================================================

If the pilot passes:

    immediately launch the experiment over every
    route-conditioned-analyzable sample in the frozen A+ cohort.

Expected source cohort is approximately:

    GQA       ~1,222
    TextVQA   ~658

but use the authoritative manifest.

Do not subsample merely to save compute unless there is a hard technical
constraint.

Do not restrict to Hamming-1/2 routes.

Do not restrict to routes with only a few OFF layers.

The purpose is a population-level decomposition of known correcting routes.

======================================================================
15. PRIMARY ANALYSES AFTER FULL RUN
======================================================================

Produce at least the following.

A. OFF-layer necessity

Across anchor correcting routes:

    fraction of OFF positions where FULL restoration:
        - stays correct
        - becomes wrong

This tells us how redundant the cached correcting routes are.

B. Route-conditioned decomposition taxonomy

Among conditionally necessary OFF positions:

    - READ-mediated
    - WRITE-mediated
    - either-removal-sufficient
    - BOTH-required
    - unresolved/inconsistent

Report:

    GQA
    TextVQA
    combined

with bootstrap confidence intervals.

C. Depth distribution

Where do:

    READ-mediated positions occur?
    WRITE-mediated positions occur?
    BOTH-required positions occur?

Plot across all 28 layers.

D. Sample-level structure

For each sample report:

    - anchor OFF count
    - number of essential OFF positions
    - number READ-mediated
    - number WRITE-mediated
    - number BOTH-required
    - number redundant

Ask whether most corrected samples are explained by:

    one dominant operation,
    multiple same-type operations,
    mixed READ/WRITE suppression,
    or joint BOTH suppression.

E. Hamming / route-size stratification

Analyze by anchor distance/OFF count.

Ask:

    - Do short correcting routes have cleaner READ/WRITE attribution?
    - Do longer routes contain more redundant OFF positions?
    - Do larger routes show more BOTH-required or mixed mechanisms?

F. Continuous effects

Report route-conditioned:

    Delta_READ_W0
    Delta_WRITE_R0
    Delta_READ_W1
    Delta_WRITE_R1
    Interaction

by layer and taxonomy.

======================================================================
16. CONNECT TO THE EXISTING FULL-CONTEXT 4-ACTION ANALYSIS
======================================================================

This comparison is scientifically important.

For the same sample/layer, compare:

    Existing experiment:
        READ/WRITE effect under FULL context

versus

    New experiment:
        READ/WRITE effect under successful correcting-route context

Ask:

1. Do layers identified as harmful under FULL context also require suppression
   inside actual correcting routes?

2. Are there layers that look weak under FULL-context single-layer
   intervention but become essential under the joint correcting-route context?

3. Does this explain why many FULL-context single-layer interventions fail to
   produce a final W->C flip even though a multi-layer correcting route exists?

4. How strong are cross-layer/context effects?

This may be one of the most important outcomes of the experiment.

A particularly interesting result would be:

    single-layer FULL-context intervention:
        no final correction

    route-conditioned decomposition:
        suppression of that READ/WRITE is necessary inside a successful
        multi-layer route

This would show that visual-operation effects are trajectory-dependent rather
than independently additive.

======================================================================
17. OPTIONAL SECOND-STAGE JOINT REFINEMENT — DO NOT RUN AUTOMATICALLY
======================================================================

Do NOT immediately enumerate 3^K or 4^K joint READ/WRITE refinements.

First complete the route-conditioned one-position-at-a-time decomposition.

After the final analysis, estimate whether a joint refinement experiment is
scientifically necessary.

Examples that might justify it:

    - many BOTH-required positions
    - strong context dependence
    - many ambiguous "either-removal-sufficient" cases
    - evidence that several OFF positions can jointly be relaxed

If joint refinement appears useful, write a proposed plan and compute-cost
estimate.

Do not launch it without a separate decision.

======================================================================
18. OUTPUT DIRECTORY
======================================================================

Create a separate directory, for example:

    analysis/4action_route_conditioned/

Do not mix raw outputs with the existing FULL-context analysis.

At minimum produce:

1. implementation_audit.md
2. pilot_report.md
3. anchor_route_manifest.parquet/jsonl
4. route_conditioned_cells.parquet
5. shard manifests / completion summaries
6. aggregate_summary.json
7. figures/
8. route_conditioned_decomposition_report.md

The final report must clearly distinguish:

    - previous FULL-context local causal effects
    - new correcting-route-conditioned effects

Never conflate them.

======================================================================
19. REQUIRED FINAL QUESTIONS
======================================================================

Explicitly answer:

1. How many frozen A+ samples had a validated current-runtime correcting anchor
   route?

2. How many OFF positions in correcting routes were actually individually
   necessary?

3. Among necessary OFF positions, how often was correction preserved by:
      READ suppression only?
      WRITE suppression only?
      either?
      both suppression?

4. Are READ-mediated and WRITE-mediated corrections distributed differently
   across model depth?

5. Are long correcting routes genuinely composed of many necessary operations,
   or do they contain substantial redundant OFF positions?

6. How often does FULL-context local harmfulness agree with
   correcting-route-conditioned necessity?

7. How often does the route-conditioned experiment reveal important operations
   that the FULL-context single-layer experiment missed?

8. Does the evidence support the hypothesis that binary routing corrects errors
   by suppressing answer-unaligned READ and/or WRITE operations?

9. Does the evidence justify moving from binary routing to a true four-action
   trajectory search/router?

Report negative results equally clearly.

======================================================================
20. SCIENTIFIC INTERPRETATION BOUNDARY
======================================================================

Do not force the desired story.

Possible valid outcomes include:

- correction is primarily READ-mediated;
- correction is primarily WRITE-mediated;
- different layers/samples require different suppression types;
- both operations must often be suppressed jointly;
- many binary OFF positions are redundant;
- FULL-context and route-conditioned effects agree strongly;
- FULL-context effects fail to predict route-conditioned behavior because
  cross-layer interactions dominate;
- four-action decomposition provides little additional explanatory value.

The experiment is designed to determine which is true.

======================================================================
21. WORKING STYLE
======================================================================

Proceed in this order:

    finish current run
        ->
    finish current final analysis
        ->
    verify current experiment integrity
        ->
    audit route-conditioned implementation needs
        ->
    implement minimal extension
        ->
    small semantic tests
        ->
    stratified route-conditioned pilot
        ->
    throughput/concurrency benchmark
        ->
    fix any issue if needed
        ->
    rerun pilot until clean
        ->
    automatically launch full frozen A+ cohort
        ->
    validate/merge
        ->
    analyze
        ->
    final report

Keep a detailed experiment log throughout.

Do not refactor unrelated infrastructure.
Do not invalidate or overwrite previous experiment outputs.
Reuse the validated unified four-action executor.