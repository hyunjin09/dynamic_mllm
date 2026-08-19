# P10 Matched Loss-Comparison Smoke

Date: 2026-08-13

## Scope and integrity

This was the frozen admission smoke for the loss-only comparison between
POLAR-style duplicated valid-route BCE and grouped exact valid-set NLL. It was
not the full predictor experiment.

- Data: regenerated GQA, TextVQA, and ChartQA route labels only.
- Training: 100 positive inputs per dataset, 300 total; two epochs.
- Validation: 50 positive inputs per dataset, 150 total.
- Actual execution: six validation records per dataset, 18 total per
  objective, including uncached predicted masks.
- Supervision: the same deterministic diverse maximum of 50 valid masks per
  input, with equal within-input route weights.
- Predictor: the same question-only frozen-Qwen3 POLAR-style backbone and
  direct factorized 28-bit head.
- Initialization: both objectives used predictor SHA-256
  `3c1689abcc703f0a7ec6ce10236a7a67d99d2dde9e3e59a7cdf49535450284bd`.
- One A6000 was used per objective on node03.
- Full training remained blocked throughout.

The first launch used the intermediate readiness artifact rather than the
final composite gate and stopped before model loading. The next launch exposed
an objective-independent validation bug: the BF16 frozen-encoder output reached
the FP32 predictor without autocast. No checkpoint or execution result was
published. Validation and checkpoint inference were repaired to use the same
BF16 autocast contract as training. Sixteen focused tests passed, including a
new BF16 validation regression test. Refreshed static and real-encoder
preflights pass in
`outputs/binary_polar/preflight/repair_v2/p10_readiness_gate_v2.json`.

## Training and validation curves

The raw loss magnitudes across the two training objectives are not directly
comparable. The common validation set-NLL and route metrics are comparable.

| Objective | Epoch | Train objective loss | Validation set-NLL | Valid-set Hit@1 | Hit@5 | Nearest-valid Hamming |
|---|---:|---:|---:|---:|---:|---:|
| Duplicated BCE | 1 | 0.9004 | 19.7578 | 0.1333 | 0.1333 | 9.4733 |
| Duplicated BCE | 2 | 0.7144 | 19.4773 | 0.0000 | 0.0333 | 9.0533 |
| Exact set-NLL | 1 | 14.7918 | 14.1629 | 0.5733 | 0.5733 | 3.7267 |
| Exact set-NLL | 2 | 14.1175 | 14.5796 | 0.5600 | 0.5733 | 3.7400 |

The common frozen checkpoint rule selected epoch 1 for both objectives. Both
training losses were finite and decreased. Exact set-NLL had substantially
better selected-epoch route membership, common validation likelihood, and
nearest-valid Hamming than duplicated BCE.

## Actual 18-record execution

| Metric | FULL baseline | Duplicated BCE | Exact set-NLL | MCTS oracle |
|---|---:|---:|---:|---:|
| Accuracy | 0.5000 | 0.2222 | 0.5000 | 1.0000 |
| Cached valid-set Hit@1 | — | 0.2222 | 0.5000 | — |
| FULL wrong to predicted correct | — | 1 | 0 | — |
| FULL correct to predicted wrong | — | 6 | 0 | — |
| Unchanged correct | — | 3 | 9 | — |
| Unchanged wrong | — | 8 | 9 | — |
| Average visual-ON layers | 28.0 | 0.0 | 28.0 | — |
| Uncached top-1 predictions | — | 14 | 9 | — |
| Uncached top-1 accuracy | — | 0.0000 | 0.0000 | — |
| Oracle accuracy gap | — | 0.7778 | 0.5000 | 0.0000 |

Every duplicated-BCE top-1 prediction was the 28-bit ALL-OFF mask. Every exact
set-NLL top-1 prediction was ALL-ON. Exact set-NLL therefore avoided the severe
regressions of duplicated BCE and improved both cached route membership and
actual execution accuracy, but its entire smoke execution result retained FULL
compute and exactly matched baseline behavior.

Dataset execution accuracy was:

| Objective | GQA | TextVQA | ChartQA |
|---|---:|---:|---:|
| Duplicated BCE | 0.5000 | 0.0000 | 0.1667 |
| Exact set-NLL | 0.5000 | 0.5000 | 0.5000 |

## Interpretation boundary

Supported smoke observations:

- Exact valid-set NLL provides a plausible route-coherence and execution-level
  advantage over duplicated BCE under the matched two-epoch smoke.
- Duplicated BCE exhibits the expected marginal/hybrid failure in an extreme
  form here: every decoded mask is ALL-OFF and six of nine FULL-correct cases
  regress.
- The factorized exact-set model has not yet demonstrated useful sparse routing;
  its smoke checkpoint chooses ALL-ON for every executed record.

Not supported by this smoke:

- predictor generalization;
- compute reduction at preserved task quality;
- recovery of the MCTS oracle advantage;
- a conclusion that exact set-NLL resolves cross-layer dependence;
- an acceleration claim.

The bounded smoke satisfies the predeclared requirement of a plausible
exact-set-NLL advantage over duplicated BCE, but the ALL-ON collapse is the
strongest objection to spending the full training budget. Full training was
not started in this action.

### Constant-policy challenge

An independent review requested one outcome-preserving discriminator before a
full-training recommendation. A deterministic audit of the same 150 validation
route sets found:

| Policy | Valid-set Hit@1 | Nearest-valid Hamming |
|---|---:|---:|
| Constant ALL-OFF | 0.1333 | 9.4733 |
| Learned duplicated BCE | 0.1333 | 9.4733 |
| Constant ALL-ON | 0.5733 | 3.6933 |
| Learned exact set-NLL | 0.5733 | 3.7267 |

The learned execution masks also equal these constants on all 18 records per
objective. Therefore, the smoke passes the literal predeclared comparison gate
against duplicated BCE, but it does not provide evidence that the predictor
learned question-dependent route selection. The exact-set advantage is
predominantly attributable to the constant ALL-ON prior. This is a supported
diagnosis for the smoke result, not proof that full-data training cannot break
the constant mode.

The full matched run remains the direct way to test whether scale breaks this
mode, but its cost is no longer supported by a strong smoke routing signal.
It requires an explicit user decision; it was not launched automatically.

## Evidence

- `outputs/binary_polar/p10_smoke/duplicated_bce_v2/history.json`
- `outputs/binary_polar/p10_smoke/exact_set_nll_v2/history.json`
- `outputs/binary_polar/p10_smoke/duplicated_bce_execution_v2.json`
- `outputs/binary_polar/p10_smoke/exact_set_nll_execution_v2.json`
- `runs/binary_polar/p10_smoke_duplicated_bce_r3_v2.log`
- `runs/binary_polar/p10_smoke_exact_set_nll_r3_v2.log`
- `outputs/binary_polar/p10_smoke/constant_policy_audit_v1.json`
