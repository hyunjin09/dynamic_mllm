# P13 Multimodal Input-Isolation Result

## Decision

**Outcome B — More input information changes probabilities but not routes.**

Native visual features materially changed how the predictor allocated
probability mass to cached valid route sets, but the improvement did not
survive complete-mask decoding. The selected Image+Question checkpoint had
lower aligned exact valid-set NLL than the matched Question-only checkpoint
(`14.4944` versus `14.8699`) and image shuffling worsened its NLL by `0.3804`.
Nevertheless, Image+Question decoded the constant ALL-ON route for all
`150/150` validation records. Its cached Hit@1 (`57.33%`), nearest-valid
Hamming (`3.693`), and mean VISUAL_ON count (`28.0`) exactly matched the
constant ALL-ON baseline.

The prospectively frozen execution-admission gate failed. Therefore the
60-record Qwen route execution was not run, no full training was launched, and
no outcome-dependent epoch or decoder was substituted.

## Question and controlled scope

P13 tested whether P11/P12 failed because the route predictor observed only a
question even though MCTS route labels belong to an image-query pair. It
compared:

1. Question-only;
2. Image-only;
3. Image+Question.

The experiment held fixed the direct factorized 28-bit binary head, exact
one-of-valid-set NLL, GQA/TextVQA/ChartQA regenerated labels, deterministic
maximum-50 valid-route sets, P11 POLAR-compatible ALL-ON weight `0.3`, frozen
image-group split, 300/150 positive smoke identities (100/50 per benchmark),
two epochs, AdamW settings, BF16 path, seed `20260809`, threshold decode, and
Hit@1-first checkpoint rule. It did not regenerate labels, alter MCTS, use the
P12 segmented head, tune a decoder, or train the base MLLM.

- Source plan: `plans/p13.md`, SHA-256
  `366da8d2d41a747d2625b369d4d686a802ad65079f5e63f3123c017659a05419`.
- Config: `configs/binary_polar_p13_multimodal_smoke_v1.yaml`, SHA-256
  `50c20916fa1996b48dd280a8eda82531ded35069d80bf432d433363401b2969b`.
- Frozen execution gate:
  `workspace/binary_polar_p13_execution_admission_gate.md`, SHA-256
  `cc305a481c19fab11e53dd5889dfc72d0aaf45f02cf60fc07bfc045b33c62fcc`.

## Frozen visual feature and cost audit

The feature is the full sequence of projected Qwen2.5-VL visual-token rows
entering decoder layer 0. It is the visual representation the frozen MLLM
already computes after its native vision encoder/projector and before any
route-controlled decoder layer.

| Property | Frozen value |
|---|---|
| Source | projected visual rows entering language decoder layer 0 |
| Shape per image | `[V, 3584]` with native, variable `V` |
| Dtype | BF16 |
| Pooling | none |
| Cached records | 502 record identities, 500 unique image groups |
| Visual-token count | min 48, median 580, mean 577.08, max 1,479 |
| Exact repeat checks | 3/3 |
| External vision model | none |
| Decoder state used | none |
| Answer/route outcome fields used | none |

The representation adds no second vision backbone and is available before
route selection in ordinary inference. P13 does add the lightweight visual
projection and predictor cross-attention over those already-computed rows; it
does not establish wall-clock savings. The unpooled cache is roughly 2 GiB for
the bounded identities and was chosen prospectively to avoid a lossy pooled
feature causing a false negative.

Evidence:
`workspace/binary_polar_p13_feature_and_fusion_spec.md` and
`outputs/binary_polar/p13/visual_features_v1/`.

## Fusion and validity checks

The common predictor projects frozen Qwen3 question-token embeddings from
1,024 to 256 dimensions and projected Qwen visual rows from 3,584 to 256
dimensions. It concatenates the visible token streams, then reuses the same
28 learned layer queries, four-head cross-attention, two-block cross-layer
encoder, and one-logit-per-layer direct head as P11. Padding rows are masked
from attention. A modality mask exposes only question, only image, or both;
there is no additional deep fusion module.

All three models used the same full P13 initialization hash
`5b3b99ad5a5602dddf33fb8893dc12f788ca8e85bc2039e1ee8b61fb62020f28`.
Every P11-shared tensor matched exactly, with shared-state hash
`3c1689abcc703f0a7ec6ce10236a7a67d99d2dde9e3e59a7cdf49535450284bd`.

The technical gate passed:

- feature-cache manifest, tensor shapes, finite values, and all tensor
  checksums verified;
- no answer, generated-output, correctness, or route-outcome field was
  consumed by feature extraction;
- variable native visual rows were padded only in a batch and padding was
  masked;
- real Qwen3 BF16 forwards produced finite losses and predictor gradients for
  all three modalities;
- the frozen Qwen3 encoder received no gradients;
- repeated logits were exact under the deterministic preflight;
- 25 combined P11-P13 focused tests passed;
- the readiness bundle authorizes only this bounded smoke and explicitly sets
  full training to false.

Evidence:
`outputs/binary_polar/preflight/p13_bf16_preflight_v1.json` and
`outputs/binary_polar/preflight/p13_readiness_gate_v2.json`.

## Matched two-epoch training curves

All three variants trained on the same 300 positive inputs and evaluated on
the same 150 positive inputs. `*` marks the checkpoint chosen by the frozen
rule: maximize Hit@1, then Hit@5, minimize nearest-valid Hamming, minimize
set-NLL, then choose the earlier epoch.

| Input | Epoch | Train set-NLL | Val set-NLL | Hit@1 | Hit@5 | Hamming | Unique | ALL-ON | ALL-OFF | Mean ON | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Question | 1* | 15.4847 | 14.8695 | 57.33% | 57.33% | 3.727 | 4 | 98.00% | 0.00% | 27.90 | 0.120 |
| Question | 2 | 14.9048 | 15.3071 | 56.00% | 56.67% | 3.767 | 23 | 84.67% | 0.67% | 25.61 | 0.900 |
| Image | 1* | 15.0707 | 14.6163 | 57.33% | 57.33% | 3.693 | 1 | 100.00% | 0.00% | 28.00 | 0.000 |
| Image | 2 | 14.1085 | 15.6620 | 38.67% | 40.00% | 5.273 | 26 | 45.33% | 27.33% | 15.79 | 1.829 |
| Image+Question | 1* | 15.0438 | 14.4948 | 57.33% | 57.33% | 3.693 | 1 | 100.00% | 0.00% | 28.00 | 0.000 |
| Image+Question | 2 | 14.0008 | 15.5644 | 40.00% | 41.33% | 5.240 | 35 | 48.67% | 24.00% | 16.77 | 1.979 |

The later multimodal epochs are more diverse, but they substantially worsen
the frozen route metrics. Choosing them after observing this result would
violate the matched checkpoint rule. The common diagnostic recomputed NLL over
the full 150-record tensor; BF16 reduction-order differences from the batched
epoch log are at most `0.0004` and do not change any decoded metric or decision.

## Selected-checkpoint route validation

| Input | Set-NLL | Hit@1 | Hit@5 | Nearest Hamming | Unique masks | ALL-ON | ALL-OFF | Mean ON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Question | 14.8699 | 57.33% | 57.33% | 3.733 | 4 | 98.00% | 0.00% | 27.907 |
| Image | 14.6166 | 57.33% | 57.33% | 3.693 | 1 | 100.00% | 0.00% | 28.000 |
| Image+Question | **14.4944** | 57.33% | 57.33% | 3.693 | 1 | 100.00% | 0.00% | 28.000 |

Relative to Question-only, Image+Question improves NLL by `0.3755` and
Hamming by only `0.040`; Hit@1 and Hit@5 do not change. It eliminates rather
than increases the Question-only model's small four-mask diversity.

## Frozen four-way modality shuffle

Every shuffle is a deterministic within-dataset derangement frozen before
training. Scores always use the target sample's unchanged valid set.

| Model | Condition | Set-NLL | Hit@1 | Hamming | Unique | ALL-ON | Mean ON |
|---|---|---:|---:|---:|---:|---:|---:|
| Question | aligned | 14.8699 | 57.33% | 3.733 | 4 | 98.0% | 27.907 |
| Question | question shuffled | 15.4256 | 55.33% | 3.773 | 4 | 98.0% | 27.900 |
| Question | image shuffled | 14.8699 | 57.33% | 3.733 | 4 | 98.0% | 27.907 |
| Question | both shuffled | 15.3763 | 56.00% | 3.760 | 4 | 98.0% | 27.907 |
| Image | aligned | 14.6166 | 57.33% | 3.693 | 1 | 100.0% | 28.000 |
| Image | question shuffled | 14.6166 | 57.33% | 3.693 | 1 | 100.0% | 28.000 |
| Image | image shuffled | 14.9526 | 57.33% | 3.693 | 1 | 100.0% | 28.000 |
| Image | both shuffled | 15.0061 | 57.33% | 3.693 | 1 | 100.0% | 28.000 |
| Image+Question | aligned | **14.4944** | 57.33% | 3.693 | 1 | 100.0% | 28.000 |
| Image+Question | question shuffled | 14.5019 | 57.33% | 3.693 | 1 | 100.0% | 28.000 |
| Image+Question | image shuffled | 14.8748 | 57.33% | 3.693 | 1 | 100.0% | 28.000 |
| Image+Question | both shuffled | 14.9335 | 57.33% | 3.693 | 1 | 100.0% | 28.000 |

For Image+Question, the NLL penalties relative to aligned are `+0.0075` for a
question shuffle, `+0.3804` for an image shuffle, and `+0.4390` when both are
shuffled. Thus the selected model contains clear image-associated probability
signal but little incremental target-question signal once the image is
present. None of the probability changes alter the threshold-decoded mask.

The image-shuffle effect is also benchmark-heterogeneous. For Image+Question,
shuffling the image changes NLL by `-0.2499` on GQA (shuffled is better),
`+0.5672` on TextVQA, and `+0.8239` on ChartQA. The aggregate visual signal is
therefore driven by TextVQA and ChartQA and is not a uniform per-task result.

Evidence:
`outputs/binary_polar/p13/conditioning_diagnostic_v1.json`.

## Constant-baseline comparison

P11 established that ALL-ON is the strongest global constant: it covers
58.12% of all 874 positive validation records, while the strongest non-ALL-ON
constant (ALL-OFF) covers 17.39%. On the frozen 150-record smoke, constant
ALL-ON gives 57.33% Hit@1, nearest-valid Hamming 3.693, one unique mask, and 28
VISUAL_ON layers.

The selected Image-only and Image+Question predictors reproduce those constant
ALL-ON metrics exactly. The lower NLL therefore reflects probability
calibration beneath an unchanged top-1 decision, not useful input-conditioned
routing.

## Prospective execution gate

| Gate condition | Result | Evidence |
|---|---|---|
| Aligned IQ NLL below question-shuffled | Pass | 14.4944 < 14.5019 |
| Aligned IQ NLL below image-shuffled | Pass | 14.4944 < 14.8748 |
| IQ ALL-ON fraction at most 90% | **Fail** | 100% |
| IQ decodes at least 10 unique masks | **Fail** | 1 |
| IQ improves Hit@1 by 0.03 or Hamming by 0.25 vs Question | **Fail** | 0.000 Hit@1; 0.040 Hamming |
| Hit@1 not materially worse | Pass | difference 0.000 |
| Hamming not materially worse | Pass | improves by 0.040 |

Because the conjunctive gate failed, P13 did not load Qwen2.5-VL for the
60-record route execution. Consequently W→C, C→W, uncached-mask accuracy, and
executed compute reduction are deliberately **not measured** in P13. Reusing
P11's ALL-ON execution as if it were a new P13 execution would be invalid.

## Interpretation

Supported:

- native pre-routing visual tokens contain information associated with the
  cached valid-route distribution;
- the image+question model assigns more mass to valid sets than the matched
  question-only model under this bounded smoke;
- almost all of the selected image+question shuffle sensitivity is visual,
  not target-question-specific;
- none of this information changes the selected complete route under the
  frozen decoder/checkpoint rule.

Not supported:

- that missing visual context is the main cause of failed route selection;
- image-query-conditioned routing;
- useful executed masks, task improvement, compute reduction, latency gain, or
  oracle-gap recovery;
- scaling P13 to full training;
- replacing the negative result with the more diverse but objectively worse
  epoch-2 checkpoints.

The most important limitation is the intentionally bounded two-epoch smoke.
It does not prove that all multimodal predictors must fail. It does establish
that the approved native-feature addition does not pass the prospective gate
needed to justify scaling this direct exact-set route-generation pipeline.

The optional next idea in P13—candidate-route validity/utility scoring—is a
strategic objective change. It remains unapproved and was not implemented.

## Evidence index

- Analysis ledger:
  `outputs/binary_polar/p13/analysis_manifest_v1.json`
- Visual cache and modality permutations:
  `outputs/binary_polar/p13/visual_features_v1/`
- Technical preflight:
  `outputs/binary_polar/preflight/p13_bf16_preflight_v1.json`
- Readiness:
  `outputs/binary_polar/preflight/p13_readiness_gate_v2.json`
- Question smoke:
  `outputs/binary_polar/p13/question_v1/`
- Image smoke:
  `outputs/binary_polar/p13/image_v1/`
- Image+Question smoke:
  `outputs/binary_polar/p13/image_question_v1/`
- Four-condition diagnostic:
  `outputs/binary_polar/p13/conditioning_diagnostic_v1.json`

**Outcome B — More input information changes probabilities but not routes.**
