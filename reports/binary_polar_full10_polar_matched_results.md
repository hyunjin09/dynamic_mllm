# Full10 POLAR-Matched Binary Predictor Results

## Executive result

Both authorized ten-epoch trainings completed successfully. The longer
POLAR-style schedule made the direct predictors less constant at later epochs,
and aligned inputs clearly improved probability-level valid-set likelihood.
It did **not**, however, convert that signal into better selected complete
masks.

- The best-Hit@1 Question-only checkpoint (epoch 2) reached `58.12%` cached
  Hit@1, predicted ALL-ON for `99.66%` of validation records, and matched the
  constant ALL-ON Hit@1 baseline.
- The best-Hit@1 Image+Question checkpoint (epoch 4) also reached `58.12%`
  Hit@1 and predicted ALL-ON for `100%` of validation records.
- Later checkpoints produced more masks, but validation Hit@1 fell to `55.03%`
  for Question-only and `55.84%` for Image+Question at epoch 10.
- On the frozen balanced 60-record actual-execution set, both selected
  best-Hit@1 checkpoints achieved `50.0%`, with no corrections or regressions.
- Question-only epoch 10 achieved `53.33%` through two uncached ChartQA
  corrections and no regressions, but this small diagnostic result came from a
  checkpoint with worse full-validation route quality and is not evidence for
  selecting epoch 10 post hoc.

The two-epoch collapse was therefore **not merely an insufficient optimization
budget**. Longer training creates diversity, but that diversity is generally
not useful under the frozen route-quality metrics. The direct factorized
predictor should not proceed to the 22,307-record external evaluation on this
evidence.

## 1. Experimental question and controlled comparison

The experiment tested whether extending the P11/P13 direct exact-valid-set
predictors from two epochs to the released POLAR-style ten-epoch optimization
scale would turn probability-level conditioning into useful sample-specific
binary routes.

Two predictors were trained concurrently on one A6000 GPU each:

1. **Question-only:** frozen Qwen3-Embedding-0.6B question tokens, learned 28
   layer queries, cross-attention, two cross-layer blocks, and a direct
   factorized 28-logit binary head.
2. **Image+Question:** the identical shared predictor plus the unpooled BF16
   projected Qwen2.5-VL visual rows entering decoder layer 0, projected to the
   predictor dimension and exposed alongside question tokens.

Everything other than the presence of visual input was held fixed: data,
selected valid masks, route weighting, predictor shared initialization,
objective, optimizer, schedule, batch size, epochs, seed, decoder, checkpoint
selection, and evaluation code.

## 2. Frozen data and supervision

The run used the regenerated GQA, TextVQA, and ChartQA labels and the frozen
image-group-disjoint split. Only records with at least one discovered valid
route enter exact valid-set training.

| Split | GQA | TextVQA | ChartQA | Total |
|---|---:|---:|---:|---:|
| Train | 2,957 | 1,525 | 1,561 | 6,043 |
| Validation | 429 | 221 | 224 | 874 |

- Source predictor manifest SHA-256:
  `3620a347a3498d16853463a6f9f8b842fecbab7b442cb869f1fb11bc9ab8aa52`
- Maximum selected valid routes per input: `50`.
- Selection: the frozen deterministic diverse max-50 policy; no new MCTS and
  no outcome-dependent resampling.
- Exact set objective:
  `-log sum_m w_m P_theta(m | x)` over complete 28-bit masks.
- Route weight: normally `1.0`; ALL-ON receives `0.3` when it is valid and a
  cheaper selected valid route exists.
- The raw route cache and P10-P13 artifacts were not modified.

The full visual cache covers 6,574 positive image groups and 6,917 positive
records. It stores the unpooled `[visual rows, 3584]` BF16 tensor and contains
no answer, correctness, or route-outcome fields. Cache checks, including
finite values, repeated extraction, and checksums, passed. Feature-manifest
SHA-256:
`fd0e9163ac6ef6eb863e32b65a81720f191fce7200440e6ebe401d1beb7eb60a`.

## 3. Exact training configuration

| Setting | Frozen value |
|---|---|
| Epochs | 10, no early stopping |
| Physical/effective batch | 128 / 128 |
| Optimizer steps | 48 per epoch, 480 total |
| Optimizer | AdamW |
| Learning rate | `5e-4` |
| Weight decay | `0.01` |
| Scheduler | cosine |
| Warmup | 10 optimizer steps |
| Gradient clipping | `1.0` |
| Precision | BF16 |
| Seed | `20260809` |
| Predictor width / heads / blocks | 256 / 4 / 2 |
| Binary decode | logit `>= 0`, independently at 28 layers |
| Frozen text encoder | Qwen3-Embedding-0.6B revision `c54f2e...3418` |
| Base executor | Qwen2.5-VL-7B-Instruct revision `cc5948...cfb5` |
| Software | PyTorch 2.6.0+cu124; Transformers 5.3.0; tqdm 4.70.0 |

Config SHA-256:
`1488f22911e19cfdaa6054065f202a36a103d1a9e85c3a5f3a5cfa3b4be51d5b`.
The required progress bars were enabled for cache verification, every training
epoch, every validation epoch, conditioning diagnostics, and actual execution.

## 4. Initialization and preflight

The shared predictor tensors were initialized identically:

- Shared initialization SHA-256:
  `3c1689abcc703f0a7ec6ce10236a7a67d99d2dde9e3e59a7cdf49535450284bd`.
- Question-only full initialization SHA-256: identical to the shared hash.
- Image+Question full initialization SHA-256:
  `5b3b99ad5a5602dddf33fb8893dc12f788ca8e85bc2039e1ee8b61fb62020f28`,
  differing because it includes the additional visual projection.

The prospective batch-128 preflight passed on the longest cached visual rows:
finite loss, finite predictor gradients, no frozen-encoder gradients, exact
repeated logits, and bounded GPU memory (`1.711 GB` Question-only; `3.791 GB`
Image+Question). No scientific optimizer state from preflight was reused.

## 5. Question-only ten-epoch trajectory

| Ep | Train NLL | Val NLL | Hit@1 | Hit@5 | Ham | Unique | ALL-ON | Mean ON | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.6269 | 14.0344 | 58.12% | 58.12% | 3.780 | 5 | 99.31% | 27.979 | 0.050 |
| 2 | 14.1062 | 13.6981 | 58.12% | 58.12% | 3.779 | 2 | 99.66% | 27.997 | 0.023 |
| 3 | 13.8524 | 14.0141 | 57.55% | 57.67% | 3.834 | 63 | 91.53% | 27.043 | 0.631 |
| 4 | 13.5748 | 13.9383 | 57.89% | 58.01% | 3.776 | 33 | 95.19% | 27.714 | 0.354 |
| 5 | 13.3039 | **13.6579** | 57.67% | 58.01% | 3.789 | 7 | 96.80% | 27.959 | 0.180 |
| 6 | 13.0394 | 14.1264 | 56.86% | 57.09% | 3.882 | 89 | 89.13% | 26.874 | 0.823 |
| 7 | 12.7579 | 14.2510 | 55.61% | 56.41% | 3.896 | 104 | 86.84% | 27.159 | 0.993 |
| 8 | 12.4955 | 14.6183 | 55.26% | 55.95% | 3.954 | 111 | 86.04% | 26.982 | 1.051 |
| 9 | 12.2806 | 15.0683 | 54.23% | 55.61% | 4.003 | 138 | 82.15% | 26.738 | 1.330 |
| 10 | 12.1860 | 15.1255 | 55.03% | 55.95% | 3.970 | 122 | 84.55% | 26.900 | 1.161 |

Training NLL decreases monotonically, while validation NLL bottoms at epoch 5
and worsens afterward. Diversity appears repeatedly from epoch 3 onward, but
Hit@1 and nearest-valid Hamming do not improve with it.

## 6. Image+Question ten-epoch trajectory

| Ep | Train NLL | Val NLL | Hit@1 | Hit@5 | Ham | Unique | ALL-ON | Mean ON | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.2881 | 14.0308 | 58.12% | 58.12% | 3.780 | 1 | 100.00% | 28.000 | 0.000 |
| 2 | 13.7948 | 13.8974 | 56.29% | 56.52% | 4.030 | 64 | 83.87% | 25.602 | 1.043 |
| 3 | 13.5161 | 13.7050 | 58.12% | 58.12% | 3.780 | 1 | 100.00% | 28.000 | 0.000 |
| 4 | 13.3048 | **13.4768** | **58.12%** | 58.12% | 3.780 | 1 | 100.00% | 28.000 | 0.000 |
| 5 | 13.0220 | 13.6632 | 58.01% | 58.01% | 3.767 | 8 | 93.59% | 27.881 | 0.325 |
| 6 | 12.7559 | 13.8786 | 57.78% | 58.01% | 3.771 | 14 | 90.16% | 27.596 | 0.531 |
| 7 | 12.4682 | 13.9379 | 57.89% | 58.01% | **3.763** | 10 | 90.16% | 27.683 | 0.518 |
| 8 | 12.2137 | 14.0935 | 56.41% | 56.86% | 3.816 | 52 | 82.27% | 27.204 | 1.050 |
| 9 | 11.9803 | 14.3850 | 55.61% | 56.52% | 3.894 | 69 | 79.41% | 27.082 | 1.252 |
| 10 | 11.8818 | 14.4603 | 55.84% | 56.64% | 3.873 | 64 | 80.32% | 27.204 | 1.185 |

The model alternates between constant and diverse modes early in training.
Epoch 2 sharply reduces ALL-ON and compute, but route quality worsens. Epoch 4
has the best NLL and ties the maximum Hit@1, yet is exactly constant ALL-ON.
Later diversity again accompanies lower rather than higher Hit@1.

## 7. Dataset-wise validation trajectories

Each cell is `Hit@1 / nearest Hamming / ALL-ON / mean ON`.

### Question-only

| Ep | GQA | TextVQA | ChartQA |
|---:|---|---|---|
| 1 | 58.74% / 3.855 / 100.0% / 28.00 | 59.28% / 3.484 / 100.0% / 28.00 | 55.80% / 3.929 / 97.3% / 27.92 |
| 2 | 58.74% / 3.855 / 100.0% / 28.00 | 59.28% / 3.484 / 100.0% / 28.00 | 55.80% / 3.924 / 98.7% / 27.99 |
| 3 | 58.74% / 3.848 / 99.5% / 27.97 | 59.28% / 3.507 / 99.1% / 27.97 | 53.57% / 4.129 / 68.8% / 24.35 |
| 4 | 58.74% / 3.855 / 100.0% / 28.00 | 59.28% / 3.484 / 100.0% / 28.00 | 54.91% / 3.911 / 81.2% / 26.88 |
| 5 | 58.74% / 3.855 / 100.0% / 28.00 | 59.28% / 3.484 / 100.0% / 28.00 | 54.02% / 3.964 / 87.5% / 27.84 |
| 6 | 57.58% / 3.911 / 97.0% / 27.85 | 58.82% / 3.507 / 96.4% / 27.81 | 53.57% / 4.196 / 67.0% / 24.08 |
| 7 | 56.88% / 3.930 / 95.8% / 27.83 | 58.82% / 3.502 / 95.0% / 27.87 | 50.00% / 4.219 / 61.6% / 25.18 |
| 8 | 55.01% / 4.035 / 93.2% / 27.60 | 59.28% / 3.502 / 92.8% / 27.69 | 51.79% / 4.246 / 65.6% / 25.11 |
| 9 | 53.85% / 4.105 / 88.1% / 27.35 | 58.82% / 3.516 / 89.6% / 27.41 | 50.45% / 4.290 / 63.4% / 24.91 |
| 10 | 54.55% / 4.084 / 91.1% / 27.48 | 58.82% / 3.475 / 91.0% / 27.55 | 52.23% / 4.241 / 65.6% / 25.14 |

### Image+Question

| Ep | GQA | TextVQA | ChartQA |
|---:|---|---|---|
| 1 | 58.74% / 3.855 / 100.0% / 28.00 | 59.28% / 3.484 / 100.0% / 28.00 | 55.80% / 3.929 / 100.0% / 28.00 |
| 2 | 56.88% / 3.939 / 96.7% / 27.75 | 56.11% / 3.964 / 77.8% / 25.42 | 55.36% / 4.268 / 65.2% / 21.68 |
| 3 | 58.74% / 3.855 / 100.0% / 28.00 | 59.28% / 3.484 / 100.0% / 28.00 | 55.80% / 3.929 / 100.0% / 28.00 |
| 4 | 58.74% / 3.855 / 100.0% / 28.00 | 59.28% / 3.484 / 100.0% / 28.00 | 55.80% / 3.929 / 100.0% / 28.00 |
| 5 | 58.74% / 3.855 / 100.0% / 28.00 | 59.28% / 3.480 / 99.5% / 28.00 | 55.36% / 3.879 / 75.4% / 27.54 |
| 6 | 58.74% / 3.855 / 100.0% / 28.00 | 58.37% / 3.471 / 95.0% / 27.95 | 55.36% / 3.906 / 66.5% / 26.48 |
| 7 | 58.74% / 3.855 / 100.0% / 28.00 | 58.82% / 3.457 / 94.6% / 27.94 | 55.36% / 3.888 / 67.0% / 26.83 |
| 8 | 57.11% / 3.914 / 94.2% / 27.77 | 56.11% / 3.538 / 76.9% / 27.53 | 55.36% / 3.902 / 64.7% / 25.80 |
| 9 | 55.48% / 4.028 / 88.6% / 27.41 | 57.01% / 3.561 / 77.4% / 27.52 | 54.46% / 3.964 / 63.8% / 26.01 |
| 10 | 55.94% / 4.021 / 89.5% / 27.46 | 57.01% / 3.543 / 79.2% / 27.58 | 54.46% / 3.915 / 63.8% / 26.34 |

ChartQA is where most nonconstant behavior appears, but its Hit@1 generally
falls when mean ON falls. TextVQA remains the most stable for Question-only;
neither dataset produces a consistent route-quality gain over ALL-ON.

## 8. Checkpoint selection

The frozen selection hierarchy was applied only after epoch 10 completed.

| Run | Best Hit@1 | Best NLL | Lowest-ALL-ON/highest-diversity diagnostic | Final |
|---|---:|---:|---:|---:|
| Question-only | epoch 2 | epoch 5 | epoch 9 | epoch 10 |
| Image+Question | epoch 4 | epoch 4 | epoch 9 | epoch 10 |

Epoch 2 rather than epoch 1 wins the Question-only Hit@1 tie through its
slightly lower nearest-valid Hamming. Image+Question epoch 4 wins its tie
through the best validation NLL. The diversity checkpoint is diagnostic only.

## 9. Aligned-versus-shuffled conditioning

| Run / checkpoint | Condition | NLL | Hit@1 | Hamming | Unique | ALL-ON | Mean ON |
|---|---|---:|---:|---:|---:|---:|---:|
| Q best Hit (e2) | aligned | 13.6981 | 58.12% | 3.779 | 2 | 99.66% | 27.997 |
| Q best Hit (e2) | question shuffled | 16.1331 | 57.78% | 3.784 | 2 | 99.66% | 27.997 |
| Q best NLL (e5) | aligned | 13.6579 | 57.67% | 3.789 | 7 | 96.80% | 27.959 |
| Q best NLL (e5) | question shuffled | 16.8732 | 55.72% | 3.809 | 7 | 96.91% | 27.959 |
| Q final (e10) | aligned | 15.1255 | 55.03% | 3.970 | 122 | 84.55% | 26.900 |
| Q final (e10) | question shuffled | 20.2527 | 48.51% | 4.333 | 122 | 84.44% | 26.904 |
| IQ best Hit/NLL (e4) | aligned | 13.4768 | 58.12% | 3.780 | 1 | 100.00% | 28.000 |
| IQ best Hit/NLL (e4) | question shuffled | 14.3390 | 58.12% | 3.780 | 1 | 100.00% | 28.000 |
| IQ best Hit/NLL (e4) | image shuffled | 15.5730 | 58.12% | 3.780 | 1 | 100.00% | 28.000 |
| IQ best Hit/NLL (e4) | both shuffled | 16.6541 | 58.12% | 3.780 | 1 | 100.00% | 28.000 |
| IQ final (e10) | aligned | 14.4603 | 55.84% | 3.873 | 64 | 80.32% | 27.204 |
| IQ final (e10) | question shuffled | 15.9790 | 54.12% | 3.953 | 66 | 79.86% | 27.122 |
| IQ final (e10) | image shuffled | 18.6881 | 49.43% | 4.096 | 66 | 80.43% | 27.165 |
| IQ final (e10) | both shuffled | 20.2546 | 46.34% | 4.263 | 79 | 79.18% | 27.102 |

This is direct evidence that both question and image inputs affect learned
probability mass. At the selected Image+Question checkpoint, however, every
condition still decodes exactly ALL-ON. Epoch-10 decoded metrics become
input-sensitive, but aligned route quality remains below the constant baseline.

## 10. Constant ALL-ON and model comparison

On all 874 positive validation inputs, constant ALL-ON has `58.12%` cached
Hit@1, nearest-valid Hamming `3.780`, one unique mask, and 28 ON layers.

| Strategy | Checkpoint | Hit@1 | Hamming | Unique | ALL-ON | Mean ON |
|---|---:|---:|---:|---:|---:|---:|
| Constant ALL-ON | — | 58.12% | 3.780 | 1 | 100.00% | 28.000 |
| Question-only best Hit | 2 | 58.12% | 3.779 | 2 | 99.66% | 27.997 |
| Image+Question best Hit | 4 | 58.12% | 3.780 | 1 | 100.00% | 28.000 |
| Question-only final | 10 | 55.03% | 3.970 | 122 | 84.55% | 26.900 |
| Image+Question final | 10 | 55.84% | 3.873 | 64 | 80.32% | 27.204 |
| Cached MCTS oracle | — | 100% by positive-set definition | 0 | input-specific | — | — |

Image+Question improves the best validation NLL over Question-only
(`13.4768` versus `13.6579`) but not selected Hit@1, Hamming, compute, or
diversity. The visual input therefore improves probabilistic fit without
improving the selected complete mask.

## 11. Actual frozen-60 Qwen execution

The execution set contains, per dataset, 10 FULL-correct and 10 FULL-wrong but
MCTS-fixable inputs. All predicted masks were executed even when absent from
the cached route set.

| Predictor | Epoch | Accuracy | W→C | C→W | Unchanged C/W | Mean ON | ALL-ON | Cached Hit@1 | Uncached count / accuracy |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| FULL | — | 50.00% | 0 | 0 | 30 / 30 | 28.000 | 100% | 50.00% | 30 / 0.00% |
| Q best Hit | 2 | 50.00% | 0 | 0 | 30 / 30 | 27.983 | 98.33% | 50.00% | 30 / 0.00% |
| IQ best Hit | 4 | 50.00% | 0 | 0 | 30 / 30 | 28.000 | 100.00% | 50.00% | 30 / 0.00% |
| Q final | 10 | 53.33% | 2 | 0 | 30 / 28 | 26.767 | 81.67% | 48.33% | 31 / 9.68% |
| IQ final | 10 | 50.00% | 0 | 0 | 30 / 30 | 26.983 | 76.67% | 48.33% | 31 / 3.23% |
| Cached MCTS oracle | — | 100.00% | 30 | 0 | 30 / 0 | — | — | 100.00% | — |

The two Question-only epoch-10 corrections are both ChartQA and both use
uncached masks (15 and 17 ON layers). They establish that uncached does not
mean invalid, as intended by the protocol. They do not rescue the checkpoint:
epoch 10 has worse 874-record Hit@1 and Hamming, and the 60-record set is a
small, deliberately balanced diagnostic rather than an independent selection
set.

No latency claim is made. Mean ON is only descriptive visual-layer execution
accounting; the best checkpoints yield essentially zero visual-layer reduction.

## 12. Relation to P11 and P13

- P11's two-epoch selected Question-only model was `98%` ALL-ON on its
  150-record validation smoke and executed at `50%` with no corrections.
  Full10 best-Hit@1 is still `99.66%` ALL-ON on the larger validation set and
  has the same behavioral result.
- P13's two-epoch selected Image+Question model was exactly ALL-ON on 150/150.
  Full10 best-Hit@1 remains exactly ALL-ON on 874/874.
- The full trajectories show that nonconstant modes are reachable by the
  optimizer, ruling out the narrow explanation that the head is incapable of
  emitting diverse masks. Those modes generally have worse cached route
  quality.

## 13. Checkpoint integrity

Every checkpoint includes predictor, optimizer, scheduler, epoch, global step,
config, seed, and a SHA-256 sidecar.

### Question-only

| Ep | Checkpoint SHA-256 |
|---:|---|
| 1 | `ea7363132a5b731a83008330226dbc96f167307d7f9d9e245f149d2aad5a4b43` |
| 2 | `68e0ae230ff2e1c09ba41d138dc414266ac7d5be8899dbb15dcdb6fc58c43ad7` |
| 3 | `ef60d873a0348cfbaa00feacb6e837d0d395dc51db4267f0f4161428fa4979ff` |
| 4 | `972a74e96d03e0b055943df44e2f552727fd0a5966aa1c85ffc64c0ad9545533` |
| 5 | `01e98f149fcb759cde1283beb27a76aa5050b6aed0c87916d89fa58dc372ae5f` |
| 6 | `d88681ee458f26d40510b45e7204e055d9b449dbdee98e2d1bd46d6f25f241a7` |
| 7 | `5d2998e020a9ce908570372d11089452fcc53aa70815c59d1e1e85377afcdb00` |
| 8 | `a3827814df89711795f38bfa2e88f2f7bb2ae45cfd31aa028a0b59b045dcd134` |
| 9 | `a1fd9a4728c7a50c98ed85cea99a42fa7a912247b83295dab69b8a59659b2cf3` |
| 10 | `938468a274c43136ef66d00290e292eabb54f7abd91737a1b6419b0c6251743f` |

### Image+Question

| Ep | Checkpoint SHA-256 |
|---:|---|
| 1 | `ce1fa9ba26b9bf936395ccded206f1d3e715a9569465a7a32bc72d056c3616d6` |
| 2 | `6899f09c3633599e94aa9fb914feec308fb0a6a0afa4550fa43bf63deb4b1e06` |
| 3 | `42da4b1c345bbc6d20f2c8e12b0692efac3c4286363c45b985c65baab40cff8d` |
| 4 | `c91993d6a1343efefb372c44f2c1c1244c060c7f90b5f4f0d3c4276b3a0bf7b5` |
| 5 | `ee28d85228fc8b0854e01f7cfdd64e522564d99a68cd3045e377b809b4ef01d1` |
| 6 | `bc8fc7ee60fae42c9a229b70b5fc8571040682fb499ccc6c972ddba0a179d7a6` |
| 7 | `1d18147ab262252e23d03b86426b4d02a57b1943df06efae08bd2583c2583cb2` |
| 8 | `94eaf14f2d30bc3002e9a5ae4356021caad60b208bbbef9d65fa9c2b37b18a39` |
| 9 | `40d1a283326ba9518f311d432498c8cba112f854641c1fb28fabe7b788c99070` |
| 10 | `a1ec0bc2b7221f2a78b12e116f98f9e642184ee3a169de67e9e3f5d725766920` |

The conditioning artifact and all four execution artifacts have passing
sidecar checksums. A first actual-execution attempt completed all 60 model
forwards but failed during summary serialization because two cache-size audit
fields were absent. The root cause was repaired at the row-construction site,
a regression test was added, 15 focused tests passed, and the unchanged four
executions reproduced deterministically. This did not affect training,
checkpoint selection, masks, generated answers, or scores. The repaired
execution source SHA-256 is
`e87f0118b53b723302f31293f508b31d435e32890c23ea9a1a6309c20a344643`.

## 14. Supported findings and interpretation boundary

Supported:

- Longer training does break strict ALL-ON behavior at some epochs.
- Aligned question and image inputs improve valid-set probability mass relative
  to deterministic group-disjoint shuffles.
- Under the frozen direct factorized head, greater mask diversity does not
  improve full-validation Hit@1 or Hamming.
- Image+Question improves best validation NLL, but not the selected mask or
  actual best-checkpoint execution.
- The best-Hit@1 models remain the constant ALL-ON solution, up to one
  Question-only validation variation.

Not established:

- that every structured mask predictor would fail;
- that the MCTS label cache is complete;
- that the two epoch-10 ChartQA corrections generalize;
- deployable accuracy, compute, or latency improvement;
- that exact valid-set NLL removes cross-layer factorization limitations.

The most defensible diagnosis is **supported persistent probability-to-decoding
failure for this direct factorized setup**, not a general absence of input
signal. The aligned/shuffled NLL gaps support the presence of signal; the
complete trajectories and execution results support its failure to become a
useful top-1 route.

## 15. Decision and recommendation

Do not scale this direct factorized predictor to the external ChartQA/TextVQA,
MMStar/MMMU, and POPE suites. Its internal selected checkpoint does not beat
constant ALL-ON, and selecting a later epoch for two favorable diagnostic
corrections would violate the frozen validation-first rule.

The strongest objection is that actual execution can validate uncached routes
that cached Hit@1 misses, as the two epoch-10 corrections demonstrate. That is
true, but the evidence is only 2/60 from a non-selected checkpoint while the
874-record route metrics degrade. A larger execution sweep would therefore be
an expensive post-hoc rescue rather than the smallest defensible next step.

If the project continues, it requires a separately approved research decision
about a different prediction formulation or candidate-ranking objective. Do
not launch it automatically; P12 already showed that the tested canonical
segment representation is not a sufficient fallback.

**FULL10_COMPLETE_DIRECT_PREDICTOR_NOT_ADMITTED**
