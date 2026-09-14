# Program Predictor Beam-Oracle Audit Plan
## Is the remaining Stage-2 bottleneck preservation collapse, ranking, or candidate generation?

## 0. Motivation

The POLAR-style suffix-program predictor improved preservation but not rescue:

```text
Sequential-A:
W→C = 3
C→W = 19
Net = -16

Program predictor:
W→C = 3
C→W = 8
Net = -5
```

Thus trajectory-level supervision helped reduce regressions, but did not increase rescue count.

The next discriminating question is:

> **Does the frozen program predictor already generate useful corrective programs inside its beam, but rank them below top-1?**

This audit separates:

```text
A. preservation-mode collapse
B. ranking failure
C. candidate-generation / representation failure
```

No model retraining is allowed in this phase.

---

## 1. Frozen components

Freeze exactly the checkpoint and contracts from the completed full evaluation:

```text
Qwen2.5-VL-7B-Instruct
robust ALL-source Stage-1
P90 Stage-1 threshold
POLAR-style suffix-program checkpoint
beam-8 decoder
four action semantics
benchmark manifests/prompts/evaluator
```

Do not change:

```text
Stage-1 threshold
beam score
beam width
action semantics
checkpoint
reranking rule
```

---

## 2. Audit population

Use the complete established four-benchmark evaluation:

```text
ChartQA   = 2,500
TextVQA   = 5,000
MMMU-Pro  = 3,460
POPE      = 9,000
Total     = 19,960
```

Candidate execution is needed only for samples where Stage-1 triggers.

Primary audit population:

```text
ALL Stage-1-triggered samples
```

Do not subsample.

---

## 3. Top-1 parity

Before the oracle audit, reproduce the completed program evaluation exactly.

Verify per triggered sample:

```text
trigger layer
top-1 suffix program
top-1 final answer
top-1 correctness
```

The pooled top-1 result must match:

```text
W→C = 3
C→W = 8
Net = -5
```

Do not proceed if parity is broken.

---

## 4. Preservation-collapse audit from existing logs

For triggered Dense-W report:

```text
# triggered W
# top-1 all-FULL
# top-1 any non-FULL
P(all-FULL | triggered W)
P(any non-FULL | triggered W)
mean/median # non-FULL
first non-FULL delay
```

For triggered Dense-C report the same quantities.

Primary question:

> Did the program predictor improve preservation mainly by collapsing toward all-FULL programs?

---

## 5. Recover complete beam-8 candidates

For every triggered sample recover:

```text
tau_1, tau_2, ..., tau_8
```

with frozen sequence scores:

```text
S_1 >= S_2 >= ... >= S_8
```

For every candidate store:

```text
rank
complete suffix action sequence
sequence score
# non-FULL actions
first non-FULL layer
RO/WO/IGNORE counts
```

If the beam was not previously persisted, regenerate it under the frozen decoder contract.

---

## 6. Deduplicate before execution

If multiple beam entries correspond to the same complete suffix program:

```text
execute once
reuse the result for duplicate ranks
```

Still preserve original beam ranks for ranking analysis.

Report:

```text
mean unique programs per beam
duplicate-beam rate
```

---

## 7. Execute every unique beam candidate

For every triggered sample and every unique beam program:

```text
reconstruct exact P90 trigger state
execute candidate suffix from L* to 27
decode final answer
score with the same benchmark evaluator
```

No MCTS.

No additional search.

No oracle modification of the program.

The oracle is used only after execution to ask whether any already-generated candidate was correct.

---

## 8. Dense-W rescue@k

For each triggered Dense-W sample define:

```text
r* = first beam rank whose executed program is correct
```

or:

```text
r* = NONE
```

Compute:

```text
W→C@1
W→C@2
W→C@4
W→C@8
```

where `W→C@k` means at least one correct program exists among ranks `1..k`.

Key quantity:

```text
additional rescue capacity
=
W→C@8 - W→C@1
```

---

## 9. W failure decomposition

Classify every triggered Dense-W sample.

### TOP1_SUCCESS_W

```text
rank-1 program is correct
```

### RANKING_FAILURE_W

```text
rank-1 is wrong
AND
some rank 2-8 candidate is correct
```

### GENERATION_FAILURE_W

```text
all 8 beam candidates are wrong
```

Report counts and rates overall and per benchmark.

This is the main diagnostic.

---

## 10. First-correct-rank analysis

Among W samples with any correct beam candidate, report the first successful rank:

```text
1
2
3
4
5
6
7
8
```

Also report cumulative rescue availability:

```text
@1
@2
@4
@8
```

If many rescues appear at ranks 2-8, ranking is limiting.

If nearly none appear beyond rank 1, generation is limiting.

---

## 11. Score-gap analysis

For every `RANKING_FAILURE_W` compute:

```text
top1_score
best_correct_score
score_gap = top1_score - best_correct_score
```

Report:

```text
median
IQR
benchmark breakdown
```

Interpretation:

```text
small score gap
→ correct programs are nearly tied; ranking objective likely weak

large score gap
→ current model strongly prefers wrong programs; deeper representation/preference mismatch
```

---

## 12. Dense-C preservation decomposition

Execute all beam candidates for triggered Dense-C too.

Classify each top-1 C→W regression.

### RANKING_FAILURE_C

```text
top-1 is wrong
AND
some rank 2-8 candidate is correct
```

### GENERATION_FAILURE_C

```text
all 8 candidates are wrong
```

Because Dense-C is known to be correct under the true all-FULL suffix, separately audit whether all-FULL appears in the beam.

---

## 13. All-FULL inclusion for triggered C

For every triggered Dense-C report:

```text
is all-FULL suffix in beam-8?
if yes, at what rank?
what is its score?
```

For each of the 8 top-1 regressions classify:

### C1

```text
all-FULL is in beam
but ranked below a wrong top-1
```

Interpretation:

```text
clean ranking failure
```

### C2

```text
all-FULL absent from beam
```

Interpretation:

```text
candidate-support / decoding failure
```

This is a particularly strong preservation diagnostic because all-FULL correctness is known.

---

## 14. All-FULL inclusion for triggered W

For triggered Dense-W also record:

```text
all-FULL in beam?
all-FULL rank?
```

Since all-FULL is known wrong for Dense-W, frequent high ranking of all-FULL is direct evidence of preservation-mode collapse.

---

## 15. Beam diversity

For every triggered sample compute:

```text
# unique programs
mean pairwise Hamming distance
# distinct first actions
# distinct non-FULL positions
```

Compare:

```text
triggered W vs triggered C
rescued W vs generation-failure W
```

Interpretation:

```text
low diversity + all wrong
→ mode collapse / weak generation

high diversity + all wrong
→ generator explores alternatives but lacks transferable corrective signal
```

---

## 16. Intervention intensity

For each candidate compute:

```text
# non-FULL actions
fraction non-FULL
first intervention delay
```

For triggered W compare:

```text
top-1 wrong program
best correct beam program
all-wrong beam programs
```

Question:

> Are correct rescue programs systematically more intervention-heavy or earlier than top-1?

Do not tune a new penalty from this audit.

---

## 17. Action-pattern analysis

For successful W rescue candidates report:

```text
first non-FULL action:
READ_ONLY
WRITE_ONLY
IGNORE
```

and total:

```text
RO count
WO count
IGNORE count
```

Also report common short motifs.

This is descriptive only.

---

## 18. Benchmark-wise tables

Dense-W table:

| Benchmark | Triggered W | W→C@1 | W→C@2 | W→C@4 | W→C@8 | RankingFail-W | GenFail-W |
|---|---:|---:|---:|---:|---:|---:|---:|

Dense-C table:

| Benchmark | Triggered C | C→W@1 | RankingFail-C | GenFail-C | all-FULL in beam |
|---|---:|---:|---:|---:|---:|

Run for:

```text
ChartQA
TextVQA
MMMU-Pro
POPE
pooled overall
```

---

## 19. TextVQA-specific question

Top-1 program result:

```text
W→C = 0
C→W = 0
```

If beam-8 contains many W rescues:

```text
ranking / conservatism is the main issue
```

If beam-8 also contains no W rescues:

```text
corrective-program generation does not generalize to TextVQA
```

---

## 20. MMMU-Pro-specific question

Top-1 program result:

```text
W→C = 2
C→W = 2
```

Check whether beam-8 contains substantially more than 2 rescues.

If yes:

```text
latent corrective capability exists but ranking suppresses it
```

If no:

```text
the two rescues are isolated candidate-generation successes
```

---

## 21. POPE

If Stage-1 remains inactive:

```text
no Stage-2 beam audit population exists
```

Record this as expected behavior.

Do not force a Stage-2 analysis onto non-triggered samples.

---

## 22. Primary decision logic

### Case A — Preservation-mode collapse

Evidence:

```text
P(any non-FULL | triggered W) is low
all-FULL frequently ranks first for W
beam diversity is low
```

Next direction:

```text
W corrective-program preference / optimization balance
```

Do not redesign the representation first.

### Case B — Ranking bottleneck

Evidence:

```text
W→C@8 materially > W→C@1
many RANKING_FAILURE_W
correct programs frequently occur at ranks 2-8
```

Next direction:

```text
training-side program preference/ranking objective
```

Do not change Stage-1.

### Case C — Candidate-generation / representation bottleneck

Evidence:

```text
W→C@8 ≈ W→C@1
most triggered W are GENERATION_FAILURE_W
```

Next direction:

```text
minimal trigger-state representation enrichment
or corrective-program generalization experiment
```

### Case D — Mixed

Some additional rescues exist in beam, but most W remain generation failures.

Quantify both and choose the quantitatively dominant bottleneck.

---

## 23. Decision-grade metrics

The final conclusion must be based mainly on:

```text
1. P(any non-FULL | triggered W)
2. W→C@1
3. W→C@8
4. # RANKING_FAILURE_W
5. # GENERATION_FAILURE_W
6. all-FULL inclusion/rank for triggered C
7. C ranking-failure vs candidate-support failure
```

Do not base the decision on internal route exact-match alone.

---

## 24. What this audit can establish

It can distinguish whether the current frozen program predictor is mainly limited by:

```text
preservation collapse
program ranking
candidate generation / representation
or a mixture
```

It can also determine whether top-1 deployment hides useful corrective capacity already present in beam-8.

---

## 25. What this audit cannot establish

Even if beam-8 contains correct programs, do not claim:

```text
a deployable reranker already exists
beam-oracle accuracy is a method result
external benchmark labels can be used at inference
the correct candidate can be selected without new training
```

`W→C@8` is only an oracle diagnostic ceiling inside the current candidate set.

---

## 26. External-label firewall

External labels may only evaluate already-generated beam candidates.

Do not use them to:

```text
fit a reranker
choose a threshold
choose a score penalty
change beam width
select a checkpoint
change action preferences
```

Any later method must be trained/tuned on training-side or development-side supervision and then evaluated prospectively.

---

## 27. Required outputs

Use:

```text
analysis/dense_failure_stage2/program_beam_oracle_audit/
```

Create:

```text
protocol.md

parity/
    top1_parity_report.md
    manifest_hashes.json
    checkpoint_contract.json

candidates/
    triggered_sample_manifest.jsonl
    beam8_ranked_programs.jsonl
    unique_program_execution_manifest.jsonl
    candidate_execution_results.jsonl

metrics/
    top1_intervention_statistics.csv
    beam_oracle_summary.csv
    rescue_at_k.csv
    first_correct_rank.csv
    w_failure_decomposition.csv
    c_failure_decomposition.csv
    score_gap_analysis.csv
    all_full_inclusion.csv
    beam_diversity.csv
    nonfull_intensity.csv
    action_pattern_summary.csv
    benchmark_breakdown.csv

figures/
    rescue_at_k.png
    first_correct_rank.png
    w_ranking_vs_generation_failure.png
    c_ranking_vs_generation_failure.png
    all_full_rank_distribution.png
    beam_diversity_w_vs_c.png
    score_gap_correct_vs_top1.png
    nonfull_intensity_correct_vs_top1.png

summaries/
    program_beam_oracle_audit_summary.md
    next_stage2_recommendation.md

artifact_manifest.json
```

---

## 28. `program_beam_oracle_audit_summary.md` must answer

1. How many Stage-1-triggered samples were audited?
2. How many triggered Dense-W and Dense-C?
3. What is `P(all-FULL | triggered W)`?
4. What is `P(any non-FULL | triggered W)`?
5. What are W→C@1, @2, @4, @8?
6. How many additional W rescues exist at ranks 2-8?
7. What fraction of top-1 W failures are ranking failures?
8. What fraction are generation failures?
9. At what ranks do first correct rescue programs appear?
10. How large are top1-vs-correct score gaps?
11. How diverse are failed-W beams?
12. Are correct rescue programs systematically more intervention-heavy?
13. How often is all-FULL present in triggered-C beams?
14. For the 8 top-1 C→W regressions, how many are ranking failures versus all-FULL-absent failures?
15. Does TextVQA contain latent beam rescue capacity?
16. Does MMMU-Pro contain more than the observed top-1 two rescues?
17. Is the main bottleneck preservation collapse, ranking, generation/representation, or mixed?
18. What exactly should be changed next?
19. What does the oracle audit not justify claiming?

---

## 29. `next_stage2_recommendation.md`

Recommend exactly one next experiment.

If ranking dominates:

```text
training-side program preference/ranking objective
```

If preservation collapse dominates:

```text
rebalance/reformulate W corrective-program preference
without changing Stage-1
```

If generation dominates:

```text
one minimal trigger-state representation enrichment experiment
```

If mixed:

```text
choose the quantitatively dominant failure mode
```

For the recommendation state:

```text
why it is the smallest discriminating next experiment
what a positive result means
what a negative result means
what cannot be concluded
```

---

## 30. Stop rule

STOP after:

```text
1. exact top-1 parity verification
2. full triggered-population beam recovery
3. execution of every unique beam-8 candidate
4. W→C@1/@2/@4/@8 computation
5. W ranking-vs-generation decomposition
6. C preservation/ranking decomposition
7. benchmark-specific analysis
8. exactly one next-stage recommendation
```

Do not retrain or modify Stage-2 in this phase.

---

## 31. Core principle

The program predictor already produced a real improvement:

```text
C→W: 19 → 8
```

but no rescue gain:

```text
W→C: 3 → 3
```

The next question is therefore:

> **Are useful corrective suffix programs already generated but ranked below top-1, or are they absent from the beam altogether?**

That distinction determines whether the next Stage-2 change should target **preference/ranking** or **representation/generation**.
