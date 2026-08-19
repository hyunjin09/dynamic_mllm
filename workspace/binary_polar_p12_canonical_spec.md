# P12 Canonical Structured-Mask Specification

Status: frozen for the bounded P12 smoke on 2026-08-13.

## Executable object

The executor still consumes one complete 28-bit mask. `1` means the visual
rows execute the decoder layer; `0` means visual rows bypass it while
text/control rows execute. P12 changes only the predictor output
representation, not executor semantics.

## Canonical representation

For mask `m[0:28]`, define boundary labels

```text
b[0] = 1
b[i] = 1[m[i] != m[i-1]], i > 0
```

At every position with `b[i] = 1`, the operation label is `m[i]`. Operation
labels elsewhere are ignored (`-100`). This is the unique maximal-run
representation: adjacent segments necessarily alternate operations and no
`Kmax` splitting is used.

The inverse starts a segment at every positive boundary, reads the operation
at that start, and fills it through the layer before the next start. It rejects
missing layer-0 boundaries, invalid operations, and adjacent same-operation
segments.

## Structured route probability

The head returns boundary logits `z^b` with shape `[B,28]` and operation
logits `z^o` with shape `[B,28,2]`. For canonical route `r(m)`:

```text
log P(r(m)|x)
 = sum_i log Bernoulli(b_i; sigmoid(z^b_i))
 + sum_{i:b_i=1} log Softmax(z^o_i)[m_i].
```

For selected valid set `V_x`, the loss remains the P11 exact weighted
one-of-valid-set objective:

```text
L(x) = -logsumexp_m[log w_m + log P(r(m)|x)], m in V_x.
```

Weights are normalized within input. ALL-ON has relative weight `0.3` iff a
selected cheaper valid route exists; every other selected valid route has
relative weight `1.0`. Padded routes are excluded before `logsumexp`.

## Frozen top-1 decoder

1. Force `b[0]=1`.
2. At layers 1--27, set a boundary iff its logit is at least zero (probability
   threshold `0.5`).
3. At each predicted start, select operation `argmax`.
4. Fill that operation to the next predicted start.
5. Return the resulting complete 28-bit mask.

No beam search, candidate search, threshold tuning, or canonical-segment cap
is permitted. Consequently P12 has one candidate per input: Hit@5 is not
available and the inherited checkpoint tie-break field is explicitly equal to
Hit@1.

## Admission evidence

`outputs/binary_polar/p12/segment_geometry_v1.json` audited all 237,802 selected
route occurrences across 6,917 positive inputs. It reconstructed all 237,802
exactly, with zero failures and zero ambiguous canonicalizations.

The geometry is not strongly compressed: selected masks have mean/median 14.11
/ 14 segments, and only 3.65% have at most eight segments. This weakens the
segment-inductive-bias motivation but does not violate the prospective P12
admission rule; the bounded two-epoch smoke is retained as the direct test.

