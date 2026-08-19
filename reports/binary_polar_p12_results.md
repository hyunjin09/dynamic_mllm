# P12 Structured Binary Program Head — Bounded Result

**Decision: Outcome B — Structured representation also collapses.**

The bounded P12 smoke preserved the probability-level question signal but did
not convert it into nonconstant executable masks. The selected structured
checkpoint decoded ALL-ON for all 150 validation inputs and all 60 frozen
execution inputs. Actual execution therefore reproduced FULL exactly: 50%
accuracy on the deliberately balanced subset, zero corrections, zero
regressions, and zero visual-layer reduction. Full training is not justified
under the frozen P12 decision rule and was not launched.

## 1. Why P12 was run

P11's question-conditioned direct 28-bit Bernoulli head assigned more valid-set
probability to the correct input than to a shuffled question, but its selected
top-1 masks were 98% ALL-ON on validation and 95% ALL-ON in the 60-record
execution subset. P12 isolated one proposed explanation: independent bit
thresholding might be the probability-to-decoding bottleneck.

Only the output representation changed:

| Component | P11 direct head | P12 structured head |
|---|---|---|
| Input | question only | same |
| Frozen encoder | Qwen3-Embedding-0.6B | same |
| Layer encoder | POLAR cross-attention/cross-layer encoder | same |
| Supervision | same selected MCTS valid masks | same |
| Valid-route cap | 50 | 50 |
| ALL-ON relative weight | 0.3 when cheaper valid route coexists | same |
| Training | 300 inputs, 2 epochs, seed 20260809 | same |
| Validation | same 150 inputs | same |
| Execution | same 60 inputs | same |
| Exact-set principle | weighted one-of-valid-set NLL | same |
| Output | 28 Bernoulli bits | 28 boundary logits + 28 binary operation logits |
| Decode | independent bit threshold | maximal-run boundary threshold + operation argmax |

The shared POLAR layer encoder has identical initialization in the P11 and P12
architectures (`0e655f...d769`). Qwen2.5-VL executor semantics did not change.

## 2. Canonical structured representation

Every 28-bit mask was represented by its unique maximal runs. Layer 0 is an
explicit start. A later layer is a start exactly when its bit differs from the
previous layer; the operation label at a start is ON or OFF and non-start
operation labels are ignored.

For route `m`, P12 computes:

```text
log P(r(m)|x)
 = sum over 28 boundary Bernoulli log-probabilities
 + sum over canonical starts of operation categorical log-probabilities.
```

The input loss is the stable weighted `logsumexp` over all selected valid
canonical routes. Padded routes contribute no mass. Top-1 decoding forces the
layer-0 boundary, thresholds other boundary probabilities at 0.5, applies
operation argmax at predicted starts, and expands one complete 28-bit mask.
There is no beam search; consequently Hit@5 is unavailable rather than
silently synthesized.

Full details are frozen in
`workspace/binary_polar_p12_canonical_spec.md`.

## 3. Admission and validity evidence

### Route geometry

The audit covered every one of the 237,802 selected valid-route occurrences
from 6,917 positive GQA, TextVQA, and ChartQA inputs.

| Group | Mean segments | Median | 90th pct. | Maximum |
|---|---:|---:|---:|---:|
| All selected masks | 14.113 | 14 | 18 | 26 |
| Non-ALL-ON masks | 14.340 | 14 | 18 | 26 |
| Minimum-ON mask per input | 11.303 | 13 | 17 | 23 |
| GQA selected masks | 14.047 | 14 | 18 | 26 |
| TextVQA selected masks | 14.237 | 14 | 18 | 25 |
| ChartQA selected masks | 14.142 | 14 | 18 | 25 |

Across all selected masks, mean ON/OFF segment counts were 7.148/6.964. The
median segment length was one layer and the mean was 1.984. Only 2.26%, 2.27%,
2.38%, and 3.65% of masks had at most 2, 4, 6, and 8 segments respectively.

This is important negative geometry evidence: the labels are losslessly
representable as runs, but they are not naturally low-segment programs. The
plan did not make strong compression an admission requirement, so the bounded
smoke proceeded; no segment cap was introduced.

### Round trip and runtime checks

- Exact mask → canonical representation → mask reconstruction:
  237,802/237,802.
- Failures, missing masks, ambiguous canonicalizations: 0/0/0.
- ALL-ON, ALL-OFF, alternating, singleton-set, contradictory-set, padded-set,
  manual probability, and BF16 train/eval tests passed.
- Focused P11/P12 suite: 15/15 tests passed.
- Real Qwen3 BF16 preflight: finite loss and predictor gradients, no frozen
  encoder gradients, exact repeated validation logits.
- Readiness gate explicitly set `ready_for_full_training=false`.
- Node04 was not used.

## 4. Two-epoch matched smoke

The selected checkpoint rule was frozen before training: maximum validation
Hit@1, then the inherited single-candidate hit field, minimum nearest-valid
Hamming, minimum structured set-NLL, then earliest epoch.

| Epoch | Train set-NLL | Validation set-NLL | Hit@1 | Nearest Hamming | Unique masks | ALL-ON | Mean ON | Entropy | Predicted segments |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1* | 19.2863 | 17.4920 | 57.33% | 3.693 | 1 | 100.00% | 28.00 | 0.000 | 1.000 |
| 2 | 16.1434 | 17.7024 | 57.33% | 3.707 | 5 | 96.67% | 27.90 | 0.191 | 1.673 |

Epoch 1 was correctly selected. Epoch 2 cannot be substituted based on its
slightly greater diversity: it had identical Hit@1, worse nearest-valid
Hamming, and worse validation set-NLL. Even descriptively, 145/150 epoch-2
masks remained ALL-ON.

Native structured diagnostics at the selected checkpoint were boundary
accuracy 52.55%, boundary precision 100%, boundary recall 7.00%, and operation
accuracy at ground-truth starts 51.82%. These diagnose the collapse: the model
almost never starts a later segment. They are not promoted over complete-mask
metrics.

### Direct P11 comparison

| Selected checkpoint | Hit@1 | Nearest Hamming | Unique masks | ALL-ON | Mean ON | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| P11 direct exact-set | 57.33% | 3.727 | 4 | 98.00% | 27.90 | 0.120 |
| P12 structured exact-set | 57.33% | 3.693 | 1 | 100.00% | 28.00 | 0.000 |

P12 improves nearest-valid Hamming by only 0.033 layers while worsening the
actual collapse. Raw P11 and P12 set-NLL values are not directly comparable
because they are likelihoods over different output representations; only
within-representation comparisons are interpreted.

## 5. Aligned-versus-shuffled conditioning

| Condition | Structured set-NLL | Hit@1 | Nearest Hamming | Unique masks | ALL-ON | Mean ON |
|---|---:|---:|---:|---:|---:|---:|
| Aligned question | 17.4918 | 57.33% | 3.693 | 1 | 100% | 28.0 |
| Within-dataset shuffled question | 18.1939 | 57.33% | 3.693 | 1 | 100% | 28.0 |

Aligned set-NLL is lower by 0.7021, so Gate A passes: the structured model
preserves probability-level input dependence. But every decoded metric is
identical. This repeats the P11 probability-to-decoding gap rather than fixing
it.

## 6. Frozen 60-record actual execution

The subset has 20 records per dataset: ten FULL-correct and ten FULL-wrong but
MCTS-fixable. Every predicted mask was executed, including masks absent from
the cached valid set.

| Strategy | Accuracy | W→C | C→W | Unchanged correct | Unchanged wrong | Mean ON | ALL-ON | Cached Hit@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL / best global constant | 50.0% | 0 | 0 | 30 | 30 | 28.000 | 100% | 50.0% |
| P11 direct exact-set | 50.0% | 0 | 0 | 30 | 30 | 27.817 | 95% | 50.0% |
| P12 structured exact-set | 50.0% | 0 | 0 | 30 | 30 | 28.000 | 100% | 50.0% |
| Cached MCTS oracle | 100.0% | 30 | 0 | 30 | 0 | — | — | 100% |

P12 selected 60/60 ALL-ON masks. The 30 FULL-wrong cases therefore appear as
uncached predictions and all remain wrong; this was verified by live execution,
not inferred from cache absence. Results are identical within each benchmark:
10/20 correct, no corrections or regressions, and mean 28 ON layers.

## 7. Frozen gates and decision

| Gate | Result | Evidence |
|---|---|---|
| A: probability conditioning survives | Pass | aligned set-NLL 17.4918 < 18.1939 shuffled |
| B: constant mode materially reduced | Fail | selected checkpoint is 150/150 ALL-ON |
| C: decoded/executed quality improves | Fail | Hit@1 unchanged; actual accuracy identical to FULL/P11 |
| D: useful non-FULL execution | Fail | no non-FULL selections; W→C=0 |
| E: regressions controlled | Trivial pass | C→W=0 because policy is FULL |

The internal challenge is that a two-epoch smoke may be insufficient to train
the higher-dimensional structured head. That objection limits the conclusion
to this frozen setup; it does not justify overriding the prospective gate.
The labels also have median 14 segments, so scaling a mismatched contiguous-run
bias would be costly and is not supported by bounded evidence.

This is Outcome B, not Outcome C: the selected structured representation did
not materially increase diversity at all. It preserved question-dependent
probability mass but collapsed more completely than P11 at top-1.

## 8. Conclusion

P12 rules out the simple claim that replacing independent bit thresholding
with this lossless maximal-run boundary/operation head is sufficient to close
the P11 probability-to-decoding gap under matched bounded training. The result
does not prove that every structured model or longer optimization schedule
would fail. It does establish that the approved canonical segment pivot did
not earn full-training escalation.

Do not full-train P12 and do not stack additional route-structure modules.
The next research decision, if requested, should revisit whether the current
question-only features and valid-set supervision contain enough deployable
route-selection signal; that is a separate action and is not executed here.

**Outcome B — Structured representation also collapses.**
