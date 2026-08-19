# P13 Native Visual Feature and Fusion Specification

Status: prospectively frozen before P13 training outcomes.

## Visual feature

P13 uses every native Qwen2.5-VL projected visual-token row that replaces an
`image_token_id` placeholder immediately before decoder layer 0. The tensor is
obtained through the same pinned processor and `get_image_features` path used
by `binary_policy.executor.inputs.build_binary_inputs`.

For each image, the cached tensor is `[V,3584]` BF16, where `V` is the native
processor-dependent visual-token count. No rows are pooled, capped, selected,
or reordered. The feature is computed from the image before any language
decoder layer, route action, answer generation, or benchmark scoring.

This feature is already required by normal Qwen2.5-VL inference. Thus an
integrated router would not require another vision-tower pass; P13's offline
cache construction runs that existing tower once per unique smoke/execution
image. Added predictor computation consists of a trainable `3584 -> 256`
linear projection for each visual row plus the existing layer-query attention
over the additional rows. It is descriptive predictor overhead, not a latency
or acceleration claim.

## Matched architecture

All three variants instantiate the same P11 direct binary backbone with:

```text
question projection: 1024 -> 256
image projection:    3584 -> 256
28 learned layer queries
unchanged 4-head cross-attention
unchanged two-block cross-layer encoder
unchanged one-logit-per-layer direct binary head
```

The visual projection is constructed after every P11-shared parameter. With
seed `20260809`, all parameters shared with the P11 architecture therefore
have bit-identical initialization in all P13 variants.

The modality conditions change only visibility masks:

- `question`: Qwen3 question tokens visible; no visual rows supplied.
- `image`: native visual rows visible; question sequence is empty/masked.
- `image_question`: Qwen3 question tokens and native visual rows visible.

Question and image rows are projected separately and concatenated only before
the existing layer-query cross-attention. Padding rows are masked. No deep
fusion block, new vision tower, pooling, decoder state, answer, correctness,
or route-outcome feature is used.

## Frozen supervision and decode

The selected max-50 valid masks, P11 POLAR-compatible 0.3 ALL-ON weighting,
exact Bernoulli valid-set NLL, independent 0.5 bit threshold, optimizer,
300/150 identities, two epochs, and checkpoint ordering remain unchanged.

## Modality shuffle

Question, image, both-question, and both-image donors are deterministic cyclic
derangements of independent SHA-256 within-dataset orders. Every donor differs
from its target. The mapping is frozen before model outcomes are inspected.

