# P11 POLAR-Weighted Input-Dependence Diagnostic

## Decision

**Outcome C — input signal exists, but factorized top-1 remains poor.**

The POLAR-compatible ALL-ON weight of `0.3` revealed question-dependent
probability-mass allocation under exact valid-set NLL, but it did not produce a
useful decoded routing policy. The frozen exact checkpoint predicted ALL-ON for
`147/150` route-validation records and `57/60` real-execution records. Its three
non-FULL execution masks were uncached and remained wrong. Full matched
training is **not justified** by P11 and was not started.

This is not Outcome B because exact set-NLL performed materially better with
aligned questions than with within-dataset shuffled questions on the common
set-NLL (`14.8699` versus `15.5089`), while also weakly improving Hit@1 and
Hamming distance. It also outperformed both matched trained bias-only controls.
It is not Outcome A because decoded behavior remained nearly constant and the
bounded execution showed no useful route selection.

An independent read-only research review ranked C over B with medium confidence.
Its strongest objection was that the detected conditioning signal is
likelihood-only and that the two-epoch bias controls may be underoptimized.

## Frozen scope and implementation

- Source plan: `plans/p11.md`, SHA-256
  `18bc8a797149f2076982d8cc68d36e16d189dbb6a73b0efe83290b3db104e44d`.
- Config: `configs/binary_polar_p11_weighted_smoke_v1.yaml`, SHA-256
  `896816c035aa2e2b31bbd29ae849298c70784c75fbf8466c2f1b334214996232`.
- Regenerated labels: GQA, TextVQA, and ChartQA only; image-group-disjoint
  7,000/1,000 split; deterministic diverse maximum of 50 valid masks per
  positive input.
- Matched question smoke: 300 positive training inputs and 150 positive
  validation inputs, 100/50 per dataset, two epochs, seed `20260809`.
- Architecture: frozen Qwen3-Embedding-0.6B question encoder, unchanged
  POLAR-style cross-attention/cross-layer encoder, direct factorized 28-bit
  binary head.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `0.01`, input batch 32,
  BF16 predictor path, unchanged objective-independent checkpoint rule.
- P11 weighting: if ALL-ON and a cheaper selected valid mask coexist, ALL-ON
  receives relative weight `0.3`; every other valid mask receives `1.0`.
  Relative weights are normalized within input for both duplicated BCE and
  exact set-NLL.
- No full training, external evaluation, new MCTS labels, encoder change, head
  redesign, or route-search expansion was performed.

The implementation adds the weighting mode, global/dataset bias-only heads,
checkpoint-level diversity metrics, deterministic within-dataset question
shuffling, and bounded execution. Seventeen focused P11/objective tests pass.

## P11-A: label geometry

The predictor manifest contains 8,000 rows. Of these, 6,917 have at least one
selected valid route and are eligible for positive-route training: 6,043 train
and 874 validation. The remaining 1,083 zero-positive rows remain preserved but
are not inputs to the positive valid-set objective.

### Constant-route coverage

| Split | Group | Positive inputs | ALL-ON coverage | ALL-OFF coverage | ALL-ON + cheaper valid |
|---|---:|---:|---:|---:|---:|
| Train | Overall | 6,043 | 58.53% | 19.39% | 57.93% |
| Train | GQA | 2,957 | 59.11% | 30.10% | 58.98% |
| Train | TextVQA | 1,525 | 59.21% | 5.97% | 58.82% |
| Train | ChartQA | 1,561 | 56.76% | 12.24% | 55.09% |
| Validation | Overall | 874 | 58.12% | 17.39% | 57.89% |
| Validation | GQA | 429 | 58.74% | 27.04% | 58.74% |
| Validation | TextVQA | 221 | 59.28% | 5.88% | 59.28% |
| Validation | ChartQA | 224 | 55.80% | 10.27% | 54.91% |

ALL-ON is the best global constant and the best constant within every dataset.
On validation it covers 508/874 positive inputs. The strongest non-ALL-ON
constant is ALL-OFF, covering 152/874. Thus ALL-ON is 40.73 percentage points,
or 3.34 times, stronger than the best non-ALL-ON constant.

### Valid-set and visual-ON geometry

| Split | Statistic | Mean | Q25 | Median | Q75 | Min–max |
|---|---|---:|---:|---:|---:|---:|
| Train | selected `|V_x|` | 34.37 | 14.00 | 50.00 | 50.00 | 1–50 |
| Train | minimum valid ON count | 8.76 | 7.00 | 10.00 | 12.00 | 0–28 |
| Train | median valid ON count | 15.14 | 14.00 | 15.00 | 16.00 | 0–28 |
| Train | maximum valid ON count | 24.11 | 20.00 | 28.00 | 28.00 | 0–28 |
| Validation | selected `|V_x|` | 34.45 | 14.25 | 50.00 | 50.00 | 1–50 |
| Validation | minimum valid ON count | 8.94 | 7.00 | 10.00 | 12.00 | 0–28 |
| Validation | median valid ON count | 15.19 | 14.00 | 15.00 | 16.00 | 9–28 |
| Validation | maximum valid ON count | 24.22 | 20.00 | 28.00 | 28.00 | 10–28 |

### Common-mask overlap

| Top global masks | Train union coverage | Validation union coverage |
|---:|---:|---:|
| 1 | 58.53% | 58.12% |
| 5 | 61.77% | 61.10% |
| 10 | 61.82% | 61.10% |
| 25 | 61.92% | 61.56% |
| 50 | 62.14% | 62.24% |

After ALL-ON and ALL-OFF, complete masks are almost entirely input-specific:
every remaining validation top-10 mask appears for only one input. The labels
therefore contain two unusually strong global shortcuts while approximately
38% of positive inputs are not covered even by the 50 most common complete
masks. This provides geometric pressure for conditional routing, but the
ALL-ON shortcut dominates the simple valid-set objective.

Raw evidence: `outputs/binary_polar/p11/label_geometry_v1.json`.

## P11-B: input-independent baselines

The four bias heads used the same 300/150 smoke identities, route cap, P11
weights, AdamW settings, two epochs, seed, and checkpoint rule as the question
smoke. The global model has 28 logits; the dataset model has three independent
28-logit rows. They receive no question representation.

| Conditioning | Objective | Epoch | Common set-NLL | Hit@1 | Hit@5 | Nearest Hamming | Unique masks | Mean ON |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Global bias | duplicated BCE | 2 | 19.4018 | 0.00% | 0.00% | 4.88 | 1 | 26.00 |
| Global bias | exact set-NLL | 2 | 19.4018 | 0.00% | 0.00% | 4.88 | 1 | 26.00 |
| Dataset bias | duplicated BCE | 2 | 19.4026 | 0.00% | 0.00% | 6.58 | 3 | 21.67 |
| Dataset bias | exact set-NLL | 2 | 19.4026 | 0.00% | 0.00% | 6.58 | 3 | 21.67 |

These matched-budget learned priors are weaker than the question model, but
their tiny learning rate/two-epoch budget is a real limitation. The stronger
non-trained constant audit remains essential: constant ALL-ON alone gives the
same 57.33% Hit@1 as both selected question checkpoints on the 150-record
smoke.

## P11-C: matched weighted training

### Training curves

| Objective | Epoch | Train loss | Val set-NLL | Hit@1 | Hit@5 | Hamming | Unique | ALL-ON | ALL-OFF | Mean ON | Entropy (nats) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Duplicated BCE | 1 | 0.9010 | 19.2090 | 0.67% | 3.33% | 7.187 | 100 | 0.67% | 0.00% | 19.24 | 4.329 |
| Duplicated BCE | 2* | 0.6936 | 18.5093 | 57.33% | 57.33% | 3.693 | 1 | 100.00% | 0.00% | 28.00 | 0.000 |
| Exact set-NLL | 1* | 15.4847 | 14.8695 | 57.33% | 57.33% | 3.727 | 4 | 98.00% | 0.00% | 27.90 | 0.120 |
| Exact set-NLL | 2 | 14.9048 | 15.3071 | 56.00% | 56.67% | 3.767 | 23 | 84.67% | 0.67% | 25.61 | 0.900 |

`*` marks the checkpoint selected by the unchanged Hit@1-first rule. Exact
set-NLL allocates probability to valid sets much better than duplicated BCE,
but its selected top-1 mask is still ALL-ON for 147/150 inputs. Epoch 2 is more
diverse but was not substituted post hoc because it loses the frozen primary
checkpoint criterion.

## P11-D: direct input dependence

| Objective | Input | Set-NLL | Hit@1 | Hit@5 | Hamming | Unique | ALL-ON | Mean ON |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Duplicated BCE | aligned | 18.5092 | 57.33% | 57.33% | 3.693 | 1 | 100.0% | 28.00 |
| Duplicated BCE | shuffled | 18.5000 | 57.33% | 57.33% | 3.693 | 1 | 100.0% | 28.00 |
| Exact set-NLL | aligned | 14.8699 | 57.33% | 57.33% | 3.733 | 4 | 98.0% | 27.91 |
| Exact set-NLL | shuffled | 15.5089 | 56.67% | 56.67% | 3.753 | 4 | 98.0% | 27.90 |

For exact set-NLL, aligned input improves set-NLL by `0.6390`, Hit@1 by `0.67`
percentage points, and nearest-valid Hamming by `0.020`. This is direct evidence
that question representations alter the learned route distribution in a useful
direction. However, almost none of that signal survives threshold decoding.
Duplicated BCE shows no decoded conditionality and slightly worse aligned
set-NLL.

## P11-E: bounded real execution

The execution manifest was frozen before P11 training outcomes. It contains
exactly 60 validation records: per dataset, 10 current FULL-correct and 10
current FULL-wrong/MCTS-fixable records. Every predicted mask, including
uncached masks, was executed through the frozen Qwen2.5-VL binary executor.

| Strategy | Accuracy | W→C | C→W | Unchanged correct | Unchanged wrong | Mean ON | ALL-ON | Cached Hit@1 | Uncached count / accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL / best global constant | 50.0% | — | — | 30 | 30 | 28.000 | 100.0% | 50.0% | 30 / 0.0% |
| Weighted duplicated BCE | 50.0% | 0 | 0 | 30 | 30 | 28.000 | 100.0% | 50.0% | 30 / 0.0% |
| Weighted exact set-NLL | 50.0% | 0 | 0 | 30 | 30 | 27.817 | 95.0% | 50.0% | 30 / 0.0% |
| Cached MCTS oracle | 100.0% | 30 | 0 | 30 | 0 | route-dependent | — | 100.0% | — |

The three non-FULL exact masks had 27, 25, and 21 visual-ON layers. All were
ChartQA FULL-wrong records, all were absent from the raw cached valid set, and
all remained wrong after fresh execution. Exact set-NLL therefore recovered
none of the 50-point oracle headroom on this deliberately balanced subset.
No latency or acceleration claim is made.

## Gate assessment

| Gate | Result | Reason |
|---|---|---|
| A: non-constant behavior | **Fail** | Selected exact checkpoint is 98% ALL-ON on route validation and 95% ALL-ON in execution. |
| B: beat input-independent baseline | **Partial pass** | Exact question model beats trained global/dataset bias heads, but only matches constant ALL-ON Hit@1. |
| C: aligned beats shuffled | **Pass for probability mass** | Exact aligned set-NLL is lower by 0.6390; decoded Hit@1 gain is only 0.67 points. |
| D: useful actual routing | **Fail** | No corrections, no regressions, 27.817/28 average ON, and all three non-FULL masks remain wrong. |
| E: exact better than duplicated | **Partial pass** | Exact strongly improves set-NLL and has minimal diversity, but execution accuracy and corrections are identical. |

## Interpretation and boundary

Supported: the question-conditioned exact objective learns input-associated
probability structure that is absent from duplicated BCE and the matched bias
controls. Also supported: the current factorized threshold decoder does not
turn that structure into useful complete masks in the bounded experiment.

Not supported: useful learned routing, compute reduction, oracle-gap recovery,
generalization, an accuracy gain, or a claim that exact valid-set NLL solves
cross-layer dependency. The bias-only optimization caveat prevents treating
their poor Hit@1 as standalone proof; the paired aligned/shuffled comparison is
the stronger input-signal evidence.

The result selects Outcome C under the frozen plan. A separate architecture
proposal is recorded at
`workspace/binary_polar_architecture_pivot_proposal.md`. It is a proposal only:
no new head was implemented or trained, and any pivot requires explicit user
approval.

## Evidence index

- Geometry: `outputs/binary_polar/p11/label_geometry_v1.json`
- Frozen identities: `outputs/binary_polar/preflight/p11_smoke_manifest_v1.json`
- Readiness: `outputs/binary_polar/preflight/p11_readiness_gate_v1.json`
- Bias controls: `outputs/binary_polar/p11/bias/`
- Training curves/checkpoints: `outputs/binary_polar/p11/question/`
- Aligned/shuffled diagnostics:
  `outputs/binary_polar/p11/question/*_conditioning_v1.json`
- Actual execution: `outputs/binary_polar/p11/execution/`

**Outcome C — Input signal exists but factorized top-1 remains poor.**
