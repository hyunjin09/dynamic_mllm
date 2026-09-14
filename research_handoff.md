# Dynamic MLLM research handoff

Updated for cross-server Git handoff: 2026-09-14T19:42:52+09:00

**Active frontier: Phase88 is IN PROGRESS, now in random matched-sparsity controls.**
This Git snapshot is sufficient to recover code, plans, memories and key summary
context. It is not a copy of the raw experiment cache and is not a final result.
Read `handoff/phase88_server_transfer/README.md` for exact environment, compressed
metadata restoration and external-data requirements.

At the recorded check, **15,703/19,452 TEST UIDs** had all 20 random schedules
completed. The full Dense baseline, both calibration sweeps, selection/stability,
M1/M2/M3 held-out evaluation and global control are complete. Four existing
workers continue on the source server (supervisor PID2361209, source-specific).
Their current-stage ETA was about 2.8 hours;
this estimate and count age immediately. No worker error was reported.
`handoff/phase88_server_transfer/progress_snapshot.json` is a saved observation,
not live status on another machine. Source live status is
`analysis/benchmark_calibrated_fixed_rw_schedule/work/continuation_status.json`.

The user paused main-agent interpretation/reporting while computation continues,
then requested status checks and this Git transfer. This packaging action does
not stop the source pipeline or launch anything on the destination. The supervisor
will aggregate and stop at `analysis_ready_for_interpretation`; reports, scientific
review and final artifact verification still need completion after user resumption.
Phase85 remains stopped. No experiment beyond the named Phase88 plan is authorized.

## Phase88: exact current state

Core hypothesis: a small labeled calibration set may identify benchmark-specific
fixed READ-OFF/WRITE-OFF layers that generalize to held-out correctness, despite
weak per-instance action predictability. Stage1 is OFF, there is no fitting or
sample-dependent policy, and the selected bit schedules are never retuned on TEST.

| Completed item | Evidence / count |
|---|---|
| Group-disjoint frozen CAL/TEST | 1,019 CAL; 19,452 TEST; zero detected cross-split image/content leakage |
| Regenerated native Dense parity | 20,471/20,471 exact token matches; zero dual-output fallbacks |
| Isolated READ and WRITE sweeps | 28,532 branches per bit; all 1,019 terminal-WRITE controls pass |
| Selection and stability | Nested group prefixes, split halves, 2,000 group bootstraps; all schedules frozen before TEST |
| READ-only / WRITE-only / combined | All 19,452 TEST UIDs complete |
| Global schedule | All 19,452 TEST UIDs complete |
| Random schedules | In progress; 20 prospective draws per family, each evaluated on its full TEST cohort |

Selected zero-indexed READ/WRITE layers: ChartQA12/25, TextVQA11/24,
MMMU-Pro17/14, POPE7/14; global17/19. WRITE27 is a zero-effect control and
excluded from selection. Random draws remain statistically weighted even when
identical routes reuse inference (ChartQA19 unique, others20).

Exact TEST denominators: ChartQA2,500, TextVQA5,000, MMMU-Pro3,204, POPE8,748.
CAL sizes are256/255/256/252, with118/160/127/14 independent groups. The POPE
CAL has only14 independent images. Five TextVQA rows (1CAL/4TEST) have no valid
normalized gold q but remain in correctness. q is secondary; correctness drives
selection. Smaller prefixes preserve whole groups and need not equal nominal N.

Active contract:
`b995483d75b300ef0170db9f06eff2a68801ba4f78fcb922e17951e96ee7492e`.
Frozen split SHA256:
`21dc39f7ca4ac02a47fafda2dd583ed6fb37515a059f8eb74b5547fe42caa512`.
The contract binds source and installed Transformers implementation files.
The scoped native continuation-position repair uses the final prompt position,
not the maximum visual position. All predecessor partial Dense/q records were
excluded from active analysis and retained as failed-attempt provenance.

**Interpretation trap:** only TextVQA WRITE and POPE READ pass both frozen
stability predicates. CAL alone already precludes A/B/C under the prospective
rubric, leaving qualified CAL-D. This does **not** establish no held-out benefit.
Actual held-out efficacy, corrections/regressions, bit interaction, global/random
contrasts and group-bootstrap intervals have not yet been interpreted. Read
`parity/calibration_category_scope_review.md` in the Phase88 analysis directory.
Do not change the rubric after outcomes or accept the reporter's provisional
D-direction text without comparing the actual full-cohort efficacy evidence.

Remaining authorized work: finish random controls; aggregate only complete exact
common TEST cohorts; produce paired group-bootstrap intervals, macro/overfitting/
interaction/control comparisons, cost accounting, ten figures and four summaries;
review conclusions; verify artifact/model/source/raw-record hashes; update compact
state; give exactly one unexecuted recommendation. No optional top2 or new study.

## Start here

This project asks whether a frozen multimodal language model can use its own
hidden trajectory to decide **when visual computation is likely to be harmful
or unnecessary, and which visual operations to retain or suppress**, while
preserving correct dense answers and rescuing some dense failures. The current
backbone is Qwen2.5-VL-7B-Instruct, revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, with 28 decoder layers.

The present result is not a winning router. The defensible state is:

- Dense failure is learnable in-domain, but the tested Stage-1 signal is
  source- and benchmark-sensitive.
- Corrective READ/WRITE effects exist, but the tested Stage-2 local predictors
  do not identify them reliably or transfer into a beneficial deployed policy.
- In particular, **READ harm is real, but it was not locally or
  short-horizon identifiable under the tested information families**.
- Phase85 bounded READ search was authorized and partially executed, then stopped by the user. Its completed-cohort interim evidence is PRELIMINARY and does not establish saturation.
- Phase87 independently tested WRITE: modest structural persistence and verified delayed text divergence, but no useful local/H≤8 harm detector. H8−H0 rho gain +.0561 is statistically supported yet sub-material. The latest scoped closure recommendation supersedes historical next-step recommendations.
- Phase86 tested the exact frozen Stage1 critic on all one-layer READ branches. Absolute failure signal remains moderate, but branch preference is unsupported: BC-C, qualified. Its historical unexecuted recommendation was paired branch-risk calibration/ranking; Phase88 is the current authorized action.

Repository provenance: before this transfer, branch `main` was at
`6c07e0aa1f0f1469c399b0b21caed9fa7f6f3ef2` and most Phases43–88 lived only
in the worktree. The transfer commit includes the accumulated source, tests,
plans and memories, preserving pre-existing edits. Its parent is that historical
commit; use `git log -1` for the new transfer commit ID. Large raw caches, dataset
and model payloads remain outside Git. Available Markdown reports are included;
references from them to raw evidence may require a separate transfer. The source
has four RTX6000Ada GPUs and direct execution, but a new server must establish
its own access and compute policy. Do not replay source PIDs or queues.

## Exact four-action semantics

At the state entering decoder layer `l`, the action is the Cartesian product
of a READ bit and a WRITE bit:

| Action | READ | WRITE | Exact current-executor meaning |
|---|---:|---:|---|
| `FULL` | 1 | 1 | Execute the normal joint multimodal decoder layer. Text/control rows can attend to visual K/V, visual rows are advanced through the layer, and the normal multimodal cache is retained. |
| `READ_ONLY` | 1 | 0 | Use the joint multimodal layer result for text/control rows, but carry the entering visual rows forward unchanged. |
| `WRITE_ONLY` | 0 | 1 | Advance visual rows using the joint full-row call, but advance text/control rows with a separate compact text-only call and a READ-off cache. Both calls start from the same entering state. |
| `IGNORE` | 0 | 0 | Advance only compact text/control rows without visual K/V and carry entering visual rows forward unchanged. |

Thus READ means that the current text/control computation can consume visual
K/V; WRITE means that the current decoder block updates the visual-token rows.
Same-layer READ consumes pre-WRITE visual K/V, so a layer's WRITE can influence
later layers but not that same layer's READ. Normal dense all-on execution is
`FULL` at all 28 layers. The authoritative implementation is
`binary_policy/executor/four_action.py` plus
`binary_policy/executor/layers.py`; do not substitute the older approximate
attention-subtraction intervention.

For the current READ-only target, WRITE is fixed ON and the comparison is:

```text
FULL       = READ ON,  WRITE ON
WRITE_ONLY = READ OFF, WRITE ON
H_R(s_l)   = q_WRITE_ONLY(s_l) - q_FULL(s_l)
```

`H_R > 0` means READ is harmful to the frozen continuous answer-quality score;
`H_R < 0` means READ is beneficial. Here `q` is the frozen teacher-forced,
token-normalized evaluator-compatible gold-answer score after the chosen
current-layer action and an all-FULL suffix. This continuous target is distinct
from LMMS final-generation correctness; controlled correctness flips are saved
as secondary behavioral labels.

## Current two-stage architecture

The two stages were introduced to separate the monolithic router's conflated
questions:

1. **Stage 1 — WHEN / dense-failure risk.** While following the dense all-FULL
   trajectory, predict whether the current dense computation will eventually
   answer incorrectly. The target is current-runtime LMMS correctness only:
   `y=1` for dense wrong and `y=0` for dense correct. It is not W-to-C
   membership, route existence, or treatment utility.
2. **Stage 2 — WHAT / corrective treatment.** After Stage 1 admits a sample,
   choose `FULL`, `READ_ONLY`, `WRITE_ONLY`, or `IGNORE` at the current and/or
   later layers so as to improve the answer while preserving correct dense
   behavior. Its supervision comes from exact counterfactual branches or
   replay-valid corrective programs, not from the Stage-1 failure label.

The frozen robust Stage-1 system is the Shared Random-4 head trained across the
Historical and Canonical GQA/ChartQA/TextVQA sources. At each layer it consumes
the 10,752-vector concatenation of `text_final`, `text_mean`, and
`visual_mean`; `text_final` is the last instruction/query-token state and is
valid as a learned feature, but it is not the assistant answer-start position.
The head uses a 256-dimensional state projection, a 32-dimensional learned
layer embedding, a 256-dimensional GELU hidden layer, and no dropout. The
deployment score is the per-layer probability mean of five group-disjoint fold
checkpoints. A trigger is the first layer whose score is strictly greater than
the chosen global threshold.

Three source-mixture operating points were retained, not one universally
valid threshold:

| Point | Threshold | Worst-source C preservation | Pooled W recall | Role |
|---|---:|---:|---:|---|
| P98 | 0.9711347410314399 | 0.9808 | 0.1054 | conservative frozen default |
| P95 | 0.9448015466742209 | 0.9501 | 0.2179 | middle diagnostic |
| P90 | 0.9061332901863008 | 0.9003 | 0.3572 | permissive point used by later Stage-2 and predictability work |

These guarantees apply only to the observed Historical+Canonical internal
mixture. They are not benchmark-universal.

The principal local Stage-2 implementation, `SharedReadWriteRouter`, observes
the actual routed token states. Its READ pathway projects the last valid text
state as a query and cross-attends to projected visual tokens. Its separate
WRITE pathway uses a learned query to attend to the same projected visual
tokens. The two 256-dimensional summaries are concatenated and passed through
one four-action head. It is depth-agnostic and receives no dataset ID. Complete
suffix-program and closed-loop trajectory-set variants were also tested later.
No Stage-2 checkpoint is currently accepted as a deployment winner.

## Why the project moved to two stages

The original upfront POLAR and online state-conditioned four-action routers
both collapsed toward all-FULL deployment. Phase 38 isolated two leading
explanations:

- The online A1 overfit pilot proved local capacity: boundary action recall was
  0.9583 and W-to-C rescue 0.8958 on the fixed pilot. This was not population
  generalization.
- Giving every W-to-C training sample one guaranteed mandatory-boundary visit
  still produced zero held-out rescue; the selected model made 24,244 FULL and
  only four IGNORE decisions out of 24,248.
- Removing 3,501 exact all-FULL C-to-C routes from POLAR supervision still
  produced all-FULL for 866/866 validation samples and zero W-to-C rescue.
- Matched upfront versus online boundary AUROC was 0.5764 versus 0.5751;
  online-minus-upfront was -0.0013 with 95% CI [-0.0548, 0.0534].

The subsequent mechanism diagnostic made the decomposition unavoidable:
KEEP-versus-DEVIATE AUROC was 0.5429 for POLAR and 0.5078 for online, while a
fresh online-state linear probe reached 0.7380. The learned systems were not
reliably deciding **when** to deviate, and their corrective action caches were
also incomplete. Stage 1 was therefore separated from Stage 2 so failure
admission could be studied independently from action selection.

## Evidence-driven trajectory

Only results that changed the research direction are retained here.

| Direction change | Strongest evidence | Decision it caused |
|---|---|---|
| Replace historical labels with current dense authority | 7,999/8,000 current native-dense runs completed with 3,999 C / 4,000 W and complete 28-layer features; one ChartQA image was missing. | Stage-1 target became current LMMS dense correctness. Historical 4K/4K buckets are metadata only. |
| Do not equate answer emergence with failure predictability | Correct answer token becomes raw top-1 at median L26, but independent failure probes already reach AUROC 0.8421 at L0 and peak at 0.8992 at L21. | Treat learned failure accessibility and answer-token emergence as different questions. |
| Reject a universal Stage-1 claim | Leave-one-dataset-out source-selected AUROC was 0.7236 TextVQA, 0.4502 ChartQA, 0.6160 GQA; transferred 99%-preservation became 0.995/0.305/0.138. | Build source-aware robustness evidence before connecting treatment. |
| Diagnose historical shortcut/source shift | Old head AUROC was 0.886 on old test but 0.4056 on canonical data; unchanged-architecture canonical refit recovered 0.8178, while reverse canonical-to-historical transfer was 0.5511. | Keep the architecture, reject source-specific fitted boundaries, and train the balanced ALL-source head. |
| Freeze only an internal-mixture robust gate | ALL-source Historical/Canonical AUROC was 0.8394/0.7689; LODO worst-source AUROC remained roughly 0.569-0.626. | Retain P98/P95/P90 only for the declared internal mixture. |
| Establish that correctable failures exist | In the old 1,881 triggered-W train cohort, exact search found 698 single-fixable, 209 MCTS-only, and 974 unresolved samples. A bounded held-out treatment audit found about 41% of triggered wrongs correctable and about 21% population rescue potential. | Stage 2 was worth testing, but oracle route yield was not treated as learnability. |
| Reject the first learned local Stage-2 policy as broad | V1 validation gave 5 W-to-C and 0 C-to-W, but 98.397% of post-trigger actions were FULL, 96.3% of interventions occurred immediately, WRITE_ONLY was never chosen, and GQA was never treated. | Expand/audit data and supervision instead of claiming general action learning. |
| Reject “more MCTS labels implies a better router” | Shared-union single-only at P90 gave validation net +3; adding MCTS gave 0 and reduced teacher-forced non-FULL recall from 0.1920 to 0.1687. | Keep label provenance separate and test actual rollout behavior. |
| Reject the frozen two-stage deployment candidate | Across 19,960 external rows, Stage1-P90 plus Stage2-A produced 3 W-to-C, 19 C-to-W, net -16. Treated-W success was 3/119; treated-C regression was 19/92. | Do not deploy or merely widen Stage-1 admission. |
| Reject a scalar abstention repair | Every positive global margin lost rescues before improving net; all five folds chose margin 0. | Do not tune another scalar confidence threshold on this router. |
| Recognize severe local-label incompleteness | Bounded search changed 496/500 old KEEP and 271/500 old INTERVENE labels to MIXED; only 4 clean KEEP and 229 clean INTERVENE remained. | Stop treating one observed action as a unique local target; move to set/program supervision. |
| Reject complete-program supervision as sufficient | Autoregressive suffix programs improved external net from -16 to -5 by reducing regressions, but still gave only 3 rescues. Beam-8 oracle raised rescue from 3 to 33, yet 463/493 top-1 W failures had no correct candidate. | Candidate generation/representation, not ranking alone, dominated. |
| Reject the tested closed-loop trajectory formulation | Exact state feedback plus trajectory-set marginal training gave 0 W-to-C / 8 C-to-W / net -8. Seen first-nonFULL recall was only 2.54%; forcing the first expert correction raised success from 43/463 to 264/463. | Sparse corrective-action learning was a real failure, but later work still had to test whether the target was predictable at all. |
| Move from router recipes to predictability measurement | Predictability Steps A-D found strong in-domain Stage 1 but consistently weak local Stage 2 and weak external transfer. | Measure identifiability before training another router. |
| Narrow to READ-only | One-step READ signal was modestly stronger than WRITE but still weak; READ harm had a large, balanced population and a clean fixed-WRITE contrast. | Analyze READ structure and propagation without four-action or WRITE confounds. |
| Move beyond local and H<=8 representations | Best local READ model was rho 0.0697; exact one-step delta was 0.0773; H8 delta rose only to 0.1097 with non-material, non-monotone gains. | The only remaining recommended discriminator is bounded longer-horizon READ planning/search. |

## Predictability phase A to D

### Step A — measurement construction: supported measurement, no predictability claim

- Internal population: 10,399 UIDs (6,399 Historical-train + 4,000
  Canonical), 9,982 image groups, and 291,172 Stage-1 `(UID, layer)` states.
- Robust P90 triggered 1,413 UIDs, yielding 15,185 primary dense Stage-2
  states. All 60,740 four-action branches completed from identical prestates
  with a common all-FULL suffix.
- Context-averaged main effects had both signs: `U_READ` positive/negative on
  7,962/7,223 states and `U_WRITE` positive/negative on 7,353/6,419, with
  1,291 READ and 981 WRITE strong controlled correctness flips.
- This establishes that counterfactual effects and behavioral opportunities
  exist. It does not establish that they are predictable. The context-averaged
  `U_READ` here is not the later conditional `H_R` target.

### Step B — in-domain learnability: Stage 1 strong, Stage 2 weak

- Five image-group-disjoint folds: Stage-1 nuisance/linear/MLP/current-head
  AUROC = 0.6814/0.7665/0.7874/0.7869; the current head peaked at L20 with
  0.8257.
- Dense Stage-2 READ/WRITE joint-router Spearman = 0.0416/0.0347 and harmful
  AUROC = 0.5193/0.5115. Top-10% harmful precision was essentially prevalence.
- Frozen category: **Case D — Stage-1 strong / Stage-2 weak**.

### Step C — semantic/source generalization: Stage 1 source-specific, Stage 2 weak

- Semantic distance was not the main Stage-1 failure: least/most-similar Q1/Q5
  AUROC was 0.7828/0.8012; K=100 cluster-OOD AUROC was 0.7725 versus ID
  0.7869.
- Source transfer collapsed: Historical-to-Canonical 0.4342 and
  Canonical-to-Historical 0.5563. Dataset LODO AUROC was 0.6043 ChartQA,
  0.5508 GQA, and 0.6006 TextVQA.
- Stage-2 READ/WRITE remained near chance across semantic, source, and dataset
  shifts. Frozen taxonomy: **S1-C source-specific signal / S2-A weak
  everywhere**.

### Step D — external transfer: neither stage supports deployment

- Zero-retuning transfer covered 19,960 rows: ChartQA 2,500, TextVQA 5,000,
  MMMU-Pro 3,460, and POPE 9,000. Dense outputs and trigger traces matched the
  established evaluation oracle.
- Stage-1 M3 AUROC was 0.4998/0.6794/0.5399/0.5582 by family, macro 0.5693.
  P90 triggered 901 UIDs and zero POPE UIDs.
- On 8,442 external post-trigger states, READ Spearman was
  0.0848/0.1095/0.0246 for ChartQA/TextVQA/MMMU-Pro; WRITE was
  0.0335/0.0158/0.0204. POPE was not estimable because it never triggered.
- Frozen categories: **D1-C** for Stage 1 and **D2-A** for Stage 2. This does
  not establish impossibility of redesigned representations or nonlocal
  planning.

## READ-only findings

### 1. READ harm existence — supported

On all 15,185 primary dense post-trigger states, the fixed-WRITE target
`H_R = q_WRITE_ONLY - q_FULL` has 7,285 harmful and 7,900 beneficial states,
with no exact zeros. There are 625 harmful correctness flips and 41 beneficial
flips. Therefore READ harm is a real and common conditional effect. This does
not support a universal “READ is harmful” rule; the sign is nearly balanced
and state-dependent.

### 2. Local structure — mostly isolated, not a broad harmful band

- Harmful adjacent persistence was 0.5174, below the UID-shuffled 97.5th
  percentile 0.5185.
- Among 3,842 harmful spans, mean length was 1.896; 55.78% were singletons and
  only the mean-span component marginally exceeded its null threshold.
- Harm prevalence rose only from 0.4586 at the trigger to 0.4847 at +6 or
  later. Immediate neighbors of strong harmful flips were close to population
  prevalence.
- Frozen category: **R-STRUCT-B — mostly isolated**. Neither a general
  early-READ-harm nor late-READ-harm phase is supported.

### 3. Local mechanism — no robust measured signature

F1-F7 covered update magnitude/direction, token concentration, attention
structure, query-key compatibility, output/value statistics, and visual
concentration. In nuisance-matched harmful-versus-beneficial comparisons, only
1/34 scalar bootstrap intervals excluded zero. That one association was lower
token-index attention spread, not evidence of attending to a semantically
wrong object. The matched linear probe AUROC was 0.4186 and the MLP AUROC
0.3254. Frozen category: **R-MECH-B — no robust local signature**.

### 4. Local learnability — weak and not practically selective

The best prospectively selected local model, F_ALL with a two-layer MLP,
reached Spearman 0.0697 and harmful AUROC 0.5338. It did not beat the one-step
delta on the primary metric, produced top-10% precision 0.5161 versus harmful
prevalence 0.4797, and had zero recall at 90% or 95% precision. Source transfer
and LODO remained weak; the external-transfer gate was false. Frozen category:
**R-LEARN-C — not locally solvable under the tested feature/capacity family**.

### 5. One-step counterfactual identifiability — unsupported

Exact same-prestate branches compared PRE, FULL post-state, WRITE_ONLY
post-state, paired states, their delta, pair+delta, and a token-aware
comparator. Dense READ Spearman was 0.0626/0.0533/0.0552/0.0523/0.0773/0.0568,
with token comparator 0.0694. Delta improved over PRE by only 0.0147 and its
95% group-bootstrap difference CI was [-0.0089, 0.0381]. WRITE was weaker;
its best Spearman was 0.0421. Frozen category: **Case D** for READ, WRITE, and
jointly. Do not build a one-step speculative controller from this evidence.

### 6. Short-horizon propagation — effects grow, identifiability does not

The ON and OFF branches differed only at layer `l`; all later executed layers
were FULL. Native H1/H2/H4/H8 support was 15,185/13,772/11,260/6,916 states;
the main comparison used the common H8 support of 6,916 states, 872 UIDs, and
858 image groups.

- DELTA Spearman H1/H2/H4/H8 = 0.0756/0.0688/0.0646/0.1097.
- Harmful AUROC = 0.5316/0.5287/0.5314/0.5483.
- H8-minus-H1 gain was only +0.0340 Spearman and +0.0167 AUROC; bootstrap
  lower bounds were -0.0009 and -0.0020, below the preregistered materiality
  requirements.
- H8 DELTA top-10% precision was 0.5462 versus prevalence 0.4926, an absolute
  gain of 0.0536, below the required 0.10.
- Median H8/H1 effect norm grew 2.268x, but predictability was non-monotone;
  pair, token, random-pair, permuted-target, and Dense-W controls did not change
  the decision.

Frozen category: **H-READ-D**. Downstream effects can grow in norm without
becoming reliably identifiable from the tested paired H<=8 representations.

### Bottom line on READ

**Supported:** turning READ off while leaving WRITE on often changes answer
quality and sometimes changes final correctness; READ harm is real.

**Unsupported:** a stable early/late harmful region, a robust attention/update
mechanism, a useful local classifier, a one-step effect detector, or an H<=8
short-horizon controller under the tested representations.

**Not yet tested:** whether a bounded sequence of READ ON/OFF choices, evaluated
and planned jointly over a longer trajectory with WRITE always ON, exposes a
solvable planning/search problem.

## Claim ledger

| Claim | Status | Scope/qualification |
|---|---|---|
| Current dense labels and 28-layer features are authoritative for Stage 1. | **Supported** | 7,999 current-runtime LMMS-scored samples; one missing-image skip. |
| Final dense failure is accessible from hidden states before answer-token identity emerges. | **Supported** | Strong in-domain learned decoding; not a causal mechanism claim. |
| The current Stage-1 head is benchmark-general or externally calibrated. | **Unsupported** | Source/LODO transfer and external macro AUROC are weak. |
| The ALL-source gate is calibrated and validated within its declared internal mixture. | **Supported** | P98/P95/P90 were prospectively frozen for Historical+Canonical GQA/ChartQA/TextVQA only. |
| Either POLAR or online four-action routing is the winning architecture. | **Unsupported** | Both collapsed; no population-level architecture advantage was established. |
| Bounded corrective routes exist for many dense failures. | **Supported** | Search, exact replay, and four-branch measurements establish existence. |
| More oracle/MCTS labels make the current Stage-2 router better. | **Unsupported** | MCTS supervision reduced non-FULL recall and removed the only positive validation cell. |
| Existing local READ/WRITE summaries support a reliable treatment gate. | **Unsupported** | Local utility, separability, one-step, transfer, and deployment results are weak/negative. |
| The original observed single-action labels were complete. | **Unsupported** | 99.2% of audited KEEP and 54.2% of INTERVENE labels changed to MIXED. |
| Complete programs, beam reranking, or the tested closed-loop trajectory loss solve Stage 2. | **Unsupported** | Best external nets were -5, beam candidate failure dominated, and closed-loop net was -8. |
| READ harm is caused by attention to the wrong object or by larger updates. | **Unsupported** | Matched mechanism evidence did not establish either explanation. |
| READ harm is locally or through H<=8 identifiable under tested information families. | **Unsupported** | R-LEARN-C, one-step Case D, and H-READ-D. |
| Stage 2 is impossible in principle. | **Unsupported** | Only the frozen targets, representations, capacities, and horizons were tested. |
| Longer-horizon bounded READ planning/search can solve the problem. | **Unresolved; Phase85 stopped** | Selected 612-W interim cohort shows late-budget gains; incomplete population/high-budget work cannot establish saturation. |
| Frozen Stage1 directly ranks one-layer READ alternatives reliably. | **Unsupported; BC-C qualified** | All 15,185 states: balanced flip accuracy 47.77%, discordant AUROC 0.4684; moderate absolute risk signal remains. |
| A comparable WRITE-only planning study would succeed. | **Not yet tested** | Phase 84 explicitly did not study WRITE. |
| A READ-only controller will improve external benchmarks or compute. | **Not yet tested** | No controller or external evaluation exists for the current frontier. |

## Already tried: do not repeat unchanged

- Upfront POLAR with duplicated BCE and exact-set NLL; online routed-state
  classification; single mandatory-boundary exposure; C-to-C all-FULL removal;
  and matched persistent supervision. None selected a final four-action family.
- Independent layer probes, their sequential union, a shared global gate,
  fixed-L27 selection, canonical-only refit, balanced ALL-source refit, and
  P98/P95/P90 calibration. Repeating the historical split or old fitted
  boundary will reproduce source-regime evidence, not robustness.
- Exhaustive single interventions, MCTS@200, route replay, trigger-conditioned
  search, robust missing-label search, and a large-scale label-completeness
  audit. Search yield is oracle evidence, not learned-policy evidence.
- Plain local action CE, single-only versus single+MCTS training, observed-valid
  local losses, a global abstention margin, and lightweight treatment probes.
- Autoregressive complete suffix programs, beam-8 oracle scoring, exact
  closed-loop state feedback with trajectory-set marginal loss, and
  teacher-forced/free-run divergence auditing.
- Generic prestates, individual one-step posts, paired posts, explicit deltas,
  pair+delta, token comparators, F1-F7 READ-operation features, nuisance/matched
  controls, semantic/source/LODO transfer, and exact H1/H2/H4/H8 propagation.
- External evaluation of the frozen P90 + Stage2-A candidate on ChartQA,
  TextVQA, MMMU-Pro, and POPE. Its net was negative; do not run it again merely
  to complete an old queue.

One first-nonFULL auxiliary-CE objective was recommended after the Phase-77
audit but was **not** run. Later A-D predictability and READ analyses made
another immediate router-training recipe less discriminating than testing
nonlocal identifiability first. Do not describe that objective as completed,
and do not elevate it above the current READ-planning frontier without a new
research decision.

## Historical Phase86 frontier (superseded by the authorized Phase88 plan)

Phase86 is complete on 15,185 states / 1,413 UIDs / 1,385 image groups. ON/OFF AUROC is 0.6728/0.5949, but delta-p Spearman is -0.0187 and balanced flip preference is 47.77%. First-trigger offline policy yields 30 W→C and 6 C→W, Net+24; this does not establish branch-ranking quality or controller readiness. Fixed-target cross-scoring leaves AUROC essentially unchanged, so the branch AUROC gap alone does not establish an OFF representation collapse. Category is **BC-C, qualified**, with moderate absolute signal and unknown cause of relative-choice failure.

The exact original Stage1 representation is decoder output indexed l; action l post-state is scored at index l, including valid layer27. Its user-text features cannot be replaced by the compact router's final control token. Original trigger l already used the dense output at l, while StepA intervenes at pre-layer l: the current policy is a retrospective rollback diagnostic, not a forward-only controller result.

Exactly one next-step recommendation, **unexecuted**: paired branch-risk calibration/ranking experiment under separately approved held-out image-group evaluation. No new next-experiment plan has been authorized. Do not resume Phase85 or run a closed-loop controller.

Primary evidence: `analysis/read_counterfactual_stage1_branch_critic/summaries/branch_critic_summary.md`; current memory: `workspace/phase_memory/phase_86_read_branch_critic.md`.

## Historical Phase84 recommendation (superseded by subsequent user plans)

The next research action, if explicitly authorized, should be **one
prospectively frozen bounded trajectory-level READ planning/search audit with
WRITE fixed ON**.

Its action alphabet must be only:

```text
FULL       (READ ON,  WRITE ON)
WRITE_ONLY (READ OFF, WRITE ON)
```

Reuse the exact Phase-78/84 dense post-trigger population and `H_R`/LMMS
contracts. Start from exact shared dense states. Unlike Phase 84, permit a
bounded multi-layer READ schedule over a longer suffix rather than one READ
toggle followed by all-FULL. Freeze the horizon(s), search budget, candidate
construction, saturation rule, objective, and validation split before opening
outcomes. The first goal is measurement: determine whether bounded search adds
replay-valid answer-quality improvement or correctness rescue beyond the
single-toggle/H<=8 baselines, and whether successful plans have stable
trajectory-level structure. Do not train a deployment controller in this first
action.

Why this is discriminating:

- A positive, saturated result would show that the missing information/action
  structure is genuinely nonlocal and would justify a later separately
  authorized planner/policy learnability study.
- Correct plans that exist but cannot be ranked from permitted trajectory
  information would separate oracle plan existence from controller
  identifiability.
- No material improvement over the exact local/H<=8 baselines would rule out
  the bounded READ-only extension and strongly favor stopping this branch.

The strongest objection is cost and interpretation: a bounded-search oracle
can prove existence without proving a learnable or deployable controller.
Therefore the search must be capped, saturation-audited, and stopped after the
measurement result. Do not add READ_ONLY or IGNORE, change WRITE, redesign
Stage 1, train a router, or run external evaluation in the same action.

## Historical planning questions (Phase85 remains incomplete and stopped)

1. **Do useful multi-layer READ schedules exist beyond H<=8?** Material,
   replay-valid gain with budget saturation supports planning; no gain supports
   stopping READ-only work.
2. **Is the gain an oracle-only phenomenon?** If routes exist but simple frozen
   trajectory rankers/generalization checks fail, the next issue is plan
   identifiability rather than action existence. This would not authorize a
   deployed router.
3. **Is success source- or dataset-specific?** Stable image-group-disjoint
   behavior across Historical/Canonical and GQA/ChartQA/TextVQA supports an
   internal method study; concentration in one small cell limits the claim and
   should not be pooled away.
4. **Does correctness improve, or only teacher-forced `q`?** `q` gain without
   LMMS rescue is mechanistic evidence but not a corrective-policy win.
5. **Would a redesigned Stage-1 representation be necessary for external
   deployment?** Existing evidence says yes if the project returns to a
   benchmark-general pipeline, but it is orthogonal to the immediate internal
   READ planning discriminator.
6. **What about WRITE?** WRITE local utility was weakly identifiable and has not
   received the Phase-83/84 depth of analysis. A WRITE-only branch is a
   separate future decision, not part of the next READ action.

## Important artifacts and paths

### Runtime, model, and evaluation

- Machine policy and runtime: `ACCESS_POLICY.md`, `infra/gpu_policy.md`,
  `workspace/env_state.md`.
- Qwen snapshot:
  `eval/reference/shared_prefix_eval_20260812/model/Qwen2.5-VL-7B-Instruct_cc594898137f460bfe9f0759e9844b3ce807cfb5`
  (symlinked into `/mnt/hyemin/qwen_train_eval`).
- Environment: Python 3.12.7, PyTorch 2.6.0+cu124, Transformers 5.3.0,
  qwen-vl-utils 0.0.14, and LMMS-Eval 0.7.3 in project-local `.venv`.
- Reference evaluation contract: `eval/reference/shared_prefix_eval_20260812/EVAL_PROTOCOL.md`,
  `eval/reference/shared_prefix_eval_20260812/README.md`, and
  `eval/reference/shared_prefix_eval_20260812/REFERENCE_RESULT.md`.
- Standing external families: ChartQA; TextVQA; MMMU-Pro Standard and Vision;
  POPE adversarial, popular, and random. Do not silently add DocVQA, MMStar, or
  base MMMU.

### Authoritative dense and Stage-1 data

- `datasets` is a symlink to `/mnt/hyemin/qwen_train_eval/datasets`.
- Current dense outputs: `analysis/dense_failure_stage1/current_dense_8k/dense_outputs.jsonl`.
- Current dense candidate and evaluator contracts:
  `analysis/dense_failure_stage1/current_dense_8k/candidate_manifest.jsonl` and
  `analysis/dense_failure_stage1/current_dense_8k/lmms_eval_contract.md`.
- Current 28-layer features:
  `analysis/dense_failure_stage1/current_dense_8k/features/feature_index.jsonl`,
  `feature_schema.json`, and 128 shards in the same directory.
- Canonical 4K expansion:
  `analysis/dense_failure_stage2/data_scale_search/manifests/new_candidate_manifest.jsonl`.
- Image-group-disjoint fold registry used by later predictability work:
  `analysis/predictability_generalization/stepB_id_learnability/splits/group_fold_registry.jsonl`.

### Frozen Stage-1 gate and tested Stage-2 checkpoints

- Five-head ALL-source gate/checkpoint inventory:
  `analysis/dense_failure_stage1/all_source_threshold_calibration/frozen/robust_stage1_gate.json`.
- Retained P98/P95/P90 thresholds:
  `analysis/dense_failure_stage1/robust_operating_points_and_compatibility/thresholds/named_operating_points.csv`.
- The five Stage-1 checkpoints are under
  `analysis/dense_failure_stage1/all_source_robustness/main_all/training/fold_{0..4}/checkpoint.pt`.
- Local Stage2-A checkpoint that was externally regression-dominated:
  `analysis/dense_failure_stage2/shared_union_training/experiment_A_single/training/selected_checkpoint.json`
  and `final_checkpoint.pt`.
- Complete-program checkpoint manifest:
  `analysis/dense_failure_stage2/polar_suffix_program/training/full_refit_checkpoint_manifest.json`.
- Closed-loop checkpoint manifest:
  `analysis/dense_failure_stage2/closed_loop_trajectory_set/training/full_refit_checkpoint_manifest.json`.

These Stage-2 files are provenance for negative comparisons, not recommended
deployment checkpoints.

### Predictability A-D

- Plans: `plans/predictability_phase_stepA_measurement_construction_plan.md`,
  `plans/predictability_phase_stepB_full_id_learnability_plan.md`,
  `plans/predictability_phase_stepC_semantic_source_generalization_plan.md`,
  `plans/predictability_phase_stepD_full_external_transfer_plan.md`.
- Step-A outcomes and state identities:
  `analysis/predictability_generalization/stepA_measurement/stage2_dense/utility_labels.csv`,
  `exact_state_manifest.jsonl`, and the `states` symlink to
  `/mnt/hyemin/qwen_train_eval/outputs/predictability_stepA_measurement_v1/dense_states`.
- Step-A/B/C/D contracts:
  `analysis/predictability_generalization/stepA_measurement/frozen_contract.json`
  (`6103b9b9...613d`),
  `analysis/predictability_generalization/stepB_id_learnability/frozen_contract.json`
  (`792cc760...ba4`),
  `analysis/predictability_generalization/stepC_generalization/frozen_contract.json`
  (`fafd6442...7ecd`), and
  `analysis/predictability_generalization/stepD_external_transfer/frozen_contract.json`
  (`efa342f0...71cf`).
- Final A-D summary:
  `analysis/predictability_generalization/stepD_external_transfer/summaries/predictability_phase_final_summary.md`.
- Step-C semantic-only encoder, not the MLLM backbone:
  `datasets/eval/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418`.

### READ frontier

- One-step plan and summary: `plans/counterfactual_effect_identifiability_plan.md` and
  `analysis/dense_failure_stage2/counterfactual_effect_identifiability/summaries/counterfactual_effect_identifiability_summary.md`.
- One-step contract: `analysis/dense_failure_stage2/counterfactual_effect_identifiability/frozen_contract.json`
  (`a70e921a...2705`); large cache symlink points to
  `/mnt/hyemin/qwen_train_eval/outputs/counterfactual_effect_identifiability_v1`.
- READ structure plan and summaries: `plans/read_harm_structure_learnability_plan.md` and
  `analysis/read_harm_structure_learnability/summaries/`.
- READ structure contract: `analysis/read_harm_structure_learnability/frozen_contract.json`
  (`eaaef860...e4d6`); large work symlink points to
  `/mnt/hyemin/qwen_train_eval/outputs/read_harm_structure_learnability_v1`.
- Short-horizon plan and summary:
  `plans/read_short_horizon_counterfactual_propagation_plan.md` and
  `analysis/read_harm_short_horizon_propagation/summaries/read_short_horizon_propagation_summary.md`.
- Short-horizon contract: `analysis/read_harm_short_horizon_propagation/frozen_contract.json`
  (`3a0d2251...bb6a`); large cache symlink points to
  `/mnt/hyemin/qwen_train_eval/outputs/read_short_horizon_propagation_v1`.
- Historical READ phase memory:
  `workspace/phase_memory/phase_84_read_short_horizon_propagation.md`.

### Prior route and deployment evidence

- Four-action collapse: `analysis/4action_collapse/decision_summary.md`.
- Robust route store:
  `analysis/dense_failure_stage2/robust_gate_corrective_search/routes/global_route_store.jsonl`.
- Full external two-stage result:
  `analysis/dense_failure_stage2/full_benchmark_eval/metrics/benchmark_summary.csv`.
- Label completeness:
  `analysis/dense_failure_stage2/treatment_label_completeness/`.
- Complete-program and beam audits:
  `analysis/dense_failure_stage2/polar_suffix_program/summaries/polar_suffix_program_full_eval_summary.md`
  and `analysis/dense_failure_stage2/program_beam_oracle_audit/`.
- Closed-loop and teacher-forced/free-run audits:
  `analysis/dense_failure_stage2/closed_loop_trajectory_set/summaries/closed_loop_trajectory_set_full_eval_summary.md`
  and `analysis/dense_failure_stage2/teacher_forced_free_run_audit/summaries/teacher_forced_free_run_audit_summary.md`.

## Recommended reading order for the next agent

First establish/read destination `ACCESS_POLICY.md` and `AGENTS.md`. Then:

1. This handoff, `workspace/workflow_state.md`, and
   `workspace/phase_memory/phase_88_benchmark_fixed_rw_schedule.md`.
2. `handoff/phase88_server_transfer/README.md` and its timestamped
   `progress_snapshot.json`; restore compressed metadata as instructed.
3. In `analysis/benchmark_calibrated_fixed_rw_schedule/`: `protocol.md`,
   `splits/benchmark_support.csv`, `summaries/calibration_summary.md`,
   `parity/decode_position_repair.md`, and
   `parity/calibration_category_scope_review.md`.
4. For immediately preceding scientific context:
   `analysis/write_harm_structure_learnability/summaries/final_write_characterization.md`
   and `read_vs_write_characterization.md` in the same summaries directory.
5. `plans/benchmark_calibrated_fixed_read_write_schedule_plan.md` is the
   current authorized plan. Its optional top2 is not authorized to run.
6. Only on a server with the synchronized live work cache, use
   `.venv/bin/python -m experiments.status_benchmark_fixed_schedule` for current
   execution status. Never treat the saved snapshot as a live queue or launch
   another supervisor while the source run continues.

Open raw artifacts only for necessary verification or complete-cohort reporting.
Do not read every historical plan/cache. The earlier sections preserve the
scientific trajectory; their historical next recommendations do not supersede
Phase88. Phase85 remains PRELIMINARY, incomplete and stopped.
