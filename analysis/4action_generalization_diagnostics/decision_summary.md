# Four-Action Generalization Diagnostic Decision

## Decision

The dominant deployed decision failure is **WHEN**: both selected routers fail
to recognize held-out states where `FULL` must stop. This is a supported
description of the failure, not a complete single-cause diagnosis.

- POLAR validation KEEP-vs-DEVIATE AUROC is 0.542877 and its argmax deviation
  recall is 0.054688.
- Online validation KEEP-vs-DEVIATE AUROC is 0.507751 and its argmax deviation
  recall is 0.148438.
- Both preserve matched `FULL`-unique states well (singleton `FULL` recall
  0.953125 POLAR; 0.976562 online), but do so by remaining overly conservative.
- WHAT also fails once intervention is attempted: validation conditional
  Valid-Action@1 is only 0.285714 for POLAR and 0.526316 for online.

The WHY evidence is substrate-dependent. Frozen online states contain a
transferable WHEN signal that the trained router does not use effectively: a
fresh prespecified linear probe reaches 0.737976 validation AUROC from the
online representation, versus 0.507751 for the trained router. The upfront
representation's probes remain near the POLAR router (0.541290--0.563141), so
the same claim is not supported for POLAR. Neither `z_R` nor `z_W` shows the
intended operation-specific specialization.

The route-label cache is also materially incomplete on the selected
cached-invalid subset: 6/14 states have an execution-correct supposedly
invalid non-`FULL` action under a frozen compatible known suffix. This is
positive evidence of WHAT-label incompleteness, but it is not a population
estimate and does not establish that it causes the WHEN collapse.

No new router, objective, training run, or external evaluation is selected or
authorized by this result.

## Validity and scope

- The analysis used the frozen Phase-39 POLAR epoch-15 and online epoch-14
  checkpoints with verified SHA-256 checksums.
- The state population contains all 640 mandatory W2C boundaries and 640
  different-UID `FULL`-unique W2C trajectory states, matched exactly by split,
  dataset, and layer. Train/validation counts are 1,024/256.
- Multi-valid mechanisms remain set-valued; ambiguous READ/WRITE bit targets
  are excluded only from the affected bit metric.
- Frozen representations and router outputs were extracted with four direct
  GPU ranks. All 1,280 outputs are finite and checksum-bound.
- Probes use train-only normalization, fixed capacity/schedule/seeds, and no
  validation checkpoint selection. Shuffle and kNN contracts were frozen
  before outcomes were observed.
- The label audit used all 14 eligible selected-router validation states, 19
  prespecified known-suffix routes, and four direct GPU ranks. It did not run
  an unbounded continuation search.
- These conclusions are internal to frozen GQA, ChartQA, and TextVQA training
  labels/checkpoints. No external evaluation was run.

## Table 1 — Main failure decomposition

| Metric | POLAR Train | POLAR Val | Online Train | Online Val |
|---|---:|---:|---:|---:|
| KEEP vs DEVIATE AUROC | 0.915585 | 0.542877 | 0.994514 | 0.507751 |
| DEVIATE recall | 0.660156 | 0.054688 | 1.000000 | 0.148438 |
| Conditional Valid-Action@1 | 0.979290 | 0.285714 | 1.000000 | 0.526316 |
| READ_OFF AUROC | 0.930602 | 0.527860 | 0.995946 | 0.536952 |
| WRITE_OFF AUROC | 0.946790 | 0.587545 | 0.999415 | 0.469554 |
| READ_ONLY-only recall | 0.697674 | 0.060606 | 1.000000 | 0.000000 |
| WRITE_ONLY-only recall | 0.687500 | 0.000000 | 1.000000 | 0.156250 |
| IGNORE-only recall | 0.550388 | 0.000000 | 1.000000 | 0.000000 |

## Table 2 — Validation singleton confusion matrices

Rows are the single valid target; columns are the predicted action.

### POLAR

| Target | IGNORE | READ_ONLY | WRITE_ONLY | FULL | Support |
|---|---:|---:|---:|---:|---:|
| IGNORE | 0 | 0 | 1 | 32 | 33 |
| READ_ONLY | 1 | 2 | 0 | 30 | 33 |
| WRITE_ONLY | 2 | 1 | 0 | 29 | 32 |
| FULL | 2 | 0 | 4 | 122 | 128 |

### Online

| Target | IGNORE | READ_ONLY | WRITE_ONLY | FULL | Support |
|---|---:|---:|---:|---:|---:|
| IGNORE | 0 | 2 | 4 | 27 | 33 |
| READ_ONLY | 1 | 0 | 1 | 31 | 33 |
| WRITE_ONLY | 1 | 0 | 5 | 26 | 32 |
| FULL | 1 | 1 | 1 | 125 | 128 |

Train and validation precision/recall/F1 tables are preserved in
`singleton_confusion_polar.csv` and `singleton_confusion_online.csv`.

## Table 3 — First-deviation timing

| Metric | POLAR | Online |
|---|---:|---:|
| Exact boundary | 0.031250 | 0.046875 |
| Within +/-1 layer | 0.085938 | 0.093750 |
| Within +/-2 layers | 0.109375 | 0.171875 |
| Too early | 0.195312 | 0.476562 |
| Too late | 0.156250 | 0.156250 |
| Never deviates | 0.539062 | 0.195312 |
| Rescue given within +/-2 | 0.000000 | 0.045455 |

Near-boundary timing is not sufficient for rescue: none of POLAR's 14 and
only 1/22 of online's within-two-layer cases is correct. POLAR most often never
deviates; online most often deviates too early.

## Table 4 — Representation usage

| Diagnostic | POLAR | Online |
|---|---:|---:|
| Layer-only KEEP/DEVIATE AUROC | 0.500000 | 0.500000 |
| Full-router KEEP/DEVIATE AUROC | 0.542877 | 0.507751 |
| Joint state-shuffle argmax unchanged | N/A | 0.835938 |
| Joint state-shuffle WHEN AUROC drop | N/A | -0.029541 |
| Joint state-shuffle READ_OFF AUROC drop | N/A | -0.013142 |
| Joint state-shuffle WRITE_OFF AUROC drop | N/A | -0.079286 |
| Upfront/online linear WHEN probe AUROC | 0.541290 | 0.737976 |
| `z_R` linear READ_OFF AUROC | N/A | 0.643982 |
| `z_W` linear WRITE_OFF AUROC | N/A | 0.567768 |

A negative shuffle "drop" means the shuffled score was higher. The shuffle
does produce nonzero distributional changes (mean KL 0.377131), so the router
is not literally state-invariant; however, those changes do not carry useful
held-out discrimination under this test.

## Table 5 — Label learnability at k=10

Values are mean validation-neighbor label purity under the frozen kNN fallback
contract.

| Representation | KEEP/DEVIATE | READ_OFF | WRITE_OFF | 3-way mechanism |
|---|---:|---:|---:|---:|
| Upfront | 0.505859 | 0.631739 | 0.638696 | 0.371429 |
| Online | 0.555469 | 0.700870 | 0.673913 | 0.430612 |
| `z_R` | 0.568359 | 0.711739 | 0.672174 | 0.371429 |
| `z_W` | 0.556250 | 0.629130 | 0.648696 | 0.377551 |

The exact valid-action-set entropy is 2.001 bits conditional on layer and 1.828
bits conditional on layer plus dataset. Exact three-way mechanisms therefore
remain weakly smooth in all tested representations; the higher bit purities
partly coexist with class imbalance and do not translate into router recall.

## Bounded label-incompleteness audit

| Group | States | States with bounded execution rescue | Fraction |
|---|---:|---:|---:|
| POLAR | 5 | 3 | 0.600000 |
| Online | 9 | 3 | 0.333333 |
| Predicted IGNORE | 5 | 3 | 0.600000 |
| Predicted READ_ONLY | 3 | 0 | 0.000000 |
| Predicted WRITE_ONLY | 6 | 3 | 0.500000 |
| Overall | 14 | 6 | 0.428571 |

This conditional audit proves that some cached-invalid actions are merely
absent from the discovered valid route set. It does not estimate global label
error, prove that all unrescued actions are invalid, or test whether `FULL`
itself is missing from mandatory-boundary valid sets.

## Q1--Q9

### Q1 — WHEN

Yes. The dominant observed deployment failure is recognizing when `FULL` must
stop. Validation ranking is near chance and false-negative rates are 0.945312
for POLAR and 0.812500 for online, despite strong train discrimination.

### Q2 — READ

No robust generalization is demonstrated for suppressing READ. Validation
READ_OFF AUROC/recall are 0.527860/0.043478 for POLAR and
0.536952/0.159420 for online.

### Q3 — WRITE

No robust generalization is demonstrated for suppressing WRITE. Validation
WRITE_OFF AUROC/recall are 0.587545/0.042857 for POLAR and
0.469554/0.057143 for online. POLAR's weak ranking advantage does not produce
usable recall, and online is below chance in ranking.

### Q4 — BOTH

No. IGNORE-only recall is zero for both. POLAR predicts `FULL` for 32/33 such
states; online predicts `FULL` for 27/33 and only partially suppresses one
operation for 6/33.

### Q5 — PRESERVE

Yes on the matched frozen population, but by an overly conservative policy.
`FULL`-only recall is 0.953125 for POLAR and 0.976562 for online, while the
same routers miss nearly all mandatory departures.

### Q6 — REPRESENTATION

The trained online router does not use sample-specific state in a way that
generalizes under the frozen shuffle test. Joint within-cell shuffling leaves
83.5938% of argmax decisions unchanged and does not reduce any validation bit
AUROC. This supports weak/non-generalizable state use, not literal invariance.
The exact layer-only WHEN baseline is chance because layer was matched, and
the full routers barely exceed it.

### Q7 — OBJECTIVE

For online states, yes: a simple fresh linear probe substantially outperforms
the trained router on WHEN (0.737976 versus 0.507751 AUROC), with smaller gains
on READ_OFF and WRITE_OFF. The supported diagnosis is failure of the trained
head/objective/optimization combination to exploit available state signal; the
diagnostic does not isolate which member of that combination is causal. For
the upfront representation, probes do not materially exceed POLAR, so an
analogous conclusion is unsupported.

### Q8 — LABEL LEARNABILITY

Exact corrective mechanisms are not consistently local in the tested feature
spaces. k=10 three-way mechanism purity is only 0.371--0.431, and action-set
entropy remains high after conditioning on layer and dataset. Bit labels have
more aggregate structure, but that evidence is insufficient to call the exact
four-action mapping smooth or readily learnable.

### Q9 — LABEL COMPLETENESS

Some supposedly invalid actions are merely missing from the route cache:
6/14 frozen selected cached-invalid states have a correct bounded execution.
This is conclusive for those six cases and unresolved for the other eight. It
does not establish a population prevalence.

## Smallest defensible next action — not executed

The next prospective action should be a small, stratified **WHEN-label
completeness audit**, not a new router training run. At frozen mandatory
boundaries, it should insert `FULL` at the audited layer while retaining
compatible known suffixes and test whether those routes remain execution-
correct. This directly tests the currently unverified premise that `FULL` is
truly invalid at a mandatory boundary.

Candidate ranking after one independent read-only challenge:

1. bounded WHEN-label completeness audit;
2. broader route-cache repair/expansion;
3. two-stage CONTINUE-vs-DEVIATE then mechanism objective/head;
4. representation redesign.

The strongest objection to the first action is that all known correcting
routes already depart at the boundary and all-FULL is wrong, so the mandatory
label has stronger structural support than the WHAT labels. It still ranks
first because observed non-`FULL` cache incompleteness makes undiscovered
`FULL` continuations plausible, the audit is cheaper than retraining, and its
outcome decides whether supervision repair or a head/objective intervention is
defensible. The reviewer originally described testing non-`FULL` alternatives;
that would not test WHEN because at least one such action is already valid by
construction. The reconciled audit must test `FULL` insertion instead.

This next action requires a new prospective plan and explicit user approval.
It was not run in this phase.
