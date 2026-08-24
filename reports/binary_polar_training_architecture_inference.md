# Binary POLAR Training, Architecture, and Inference

**Project:** Dynamic MLLM  
**Model:** Qwen2.5-VL-7B-Instruct  
**Model revision:** `cc594898137f460bfe9f0759e9844b3ce807cfb5`  
**Environment used by the completed full runs:** PyTorch `2.6.0+cu124`, Transformers `5.3.0`, BF16, SDPA  
**Document date:** 2026-08-24

## 1. Executive summary

Our system is a **POLAR-inspired, pre-action, static binary route predictor** for
Qwen2.5-VL. It is not the released tri-state POLAR model unchanged.

For each image-question input, the predictor emits 28 real-valued logits, one
for each Qwen language-model decoder layer:

\[
z(x)=[z_0,\ldots,z_{27}]\in\mathbb{R}^{28}.
\]

Each logit is independently converted into a binary action:

\[
m_l=\mathbf{1}[z_l\ge 0]
    =\mathbf{1}[\sigma(z_l)\ge 0.5].
\]

- `m_l = 1` (`VISUAL_ON`): run decoder layer `l` on the complete ordered
  sequence, including visual and text/control rows.
- `m_l = 0` (`VISUAL_OFF`): remove visual rows from decoder layer `l`, run the
  unchanged Qwen layer on text/control rows only, and carry the visual hidden
  rows through the layer unchanged.

The 28 bits are predicted **once, before the Qwen decoder stack begins**. They
form one static route used throughout prompt prefill and autoregressive
generation. This predictor is therefore not a causal online layer-by-layer
router, even though the execution bundle also contains separate online-router
utilities that are not used by these POLAR-style checkpoints.

The main predictor inputs are:

1. **Question-only:** frozen Qwen3-Embedding-0.6B token embeddings with shape
   `[B, T, 1024]`.
2. **Image+Question:** the same question embeddings plus the unpooled,
   already-encoded Qwen2.5-VL visual rows entering decoder layer 0, with shape
   `[B, V, 3584]` before projection.

Both inputs are projected to width 256. Twenty-eight learned layer embeddings
cross-attend to those input rows, two Transformer blocks communicate across
the 28 layer positions, and a shared linear head produces one ON/OFF logit per
layer.

The frozen base MLLM and frozen Qwen3 question encoder are never trained. Only
the lightweight binary predictor is optimized.

## 2. What is retained from POLAR, and what is different

The project name “binary POLAR” refers to the predictor design and training
comparison inherited from POLAR, not to identical actions or execution.

| Component | Released POLAR | Our binary visual adaptation |
|---|---|---|
| Base problem | Change the execution program of a text LLM | Decide visual participation in each Qwen2.5-VL decoder layer |
| Actions | `SKIP`, `EXECUTE`, `REPEAT` on layer segments | `VISUAL_OFF`, `VISUAL_ON` independently for 28 layers |
| Output representation | Segment-boundary logits plus 3-way operation logits | Direct 28-logit binary mask |
| Supervision object | Valid lists of executed/repeated layer indices | Valid complete 28-bit visual ON/OFF masks |
| Predictor input in the release | Frozen Qwen3 question-token embeddings | Same question-token design; optional native Qwen visual rows added |
| Layer representation | Learned layer queries cross-attend to input tokens, then self-attend across layers | Retained |
| Multi-route baseline | Duplicate an input once per valid program | Duplicate an input once per valid 28-bit mask and apply per-bit BCE |
| Proposed objective | Not applicable | Exact one-of-valid-set NLL over complete masks |
| Decoding | Segment construction and beam over operations | Independent threshold at logit zero; exact factorized top-k only for diagnostics |
| MLLM execution | Skip or repeat whole decoder layers | Text always executes; visual rows are included or bypassed per layer |

Released POLAR produces `seg_logits [B,D]` and `op_logits [B,D,3]`, then
constructs segmented programs with beam search. Our current model deliberately
does neither. It produces only `route_logits [B,28]`.

The phrase **POLAR-style duplicated BCE** in this project means a controlled
binary analogue of POLAR's duplicated-valid-program supervision. It does not
mean that the original segmentation and tri-state POLAR heads are still being
used.

## 3. End-to-end system overview

```text
Raw image + question
        |
        +---------------------------+
        |                           |
        v                           v
Qwen3 question tokenizer      Native Qwen2.5-VL image processing
left-pad, max 512             and frozen vision encoder/merger
        |                           |
        v                           v
[B,T,1024] frozen tokens       [B,V,3584] projected visual rows
        |                           |
        +-------------+-------------+
                      |
                      v
           separate linear projections
                      |
                      v
        predictor memory [B,T(+V),256]
                      |
                      v
       28 learned layer queries [B,28,256]
                      |
                      v
       cross-attention into predictor memory
                      |
                      v
        2 cross-layer Transformer blocks
                      |
                      v
              28 binary logits [B,28]
                      |
                      v
          threshold each logit at zero
                      |
                      v
             static 28-bit Qwen route
                      |
                      v
      route-conditioned Qwen prompt prefill
                      |
                      v
     deterministic autoregressive generation
                      |
                      v
       benchmark-specific answer evaluation
```

There are two distinct frozen encoders in the complete Image+Question system:

- Qwen3-Embedding-0.6B encodes the predictor's question input.
- The Qwen2.5-VL vision stack produces the visual rows that both the predictor
  and the routed Qwen decoder use.

The router does **not** consume answer tokens, reference answers, generated
correctness, MCTS scores, or intermediate routed decoder states.

## 4. Inputs to the predictor

### 4.1 Question input

The question is tokenized directly with the frozen Qwen3-Embedding-0.6B
tokenizer:

- padding side: left;
- maximum length: 512 tokens;
- truncation: enabled;
- output IDs: `[B,T]`;
- attention mask: `[B,T]`;
- frozen token features: `[B,T,1024]`.

The router tokenizer is separate from the Qwen2.5-VL processor used to build
the actual multimodal generation prompt. The two paths represent the same
question semantically, but they do not share token IDs or token boundaries.

During training, the manifest's `question` field is used. During the external
evaluation:

- ChartQA, TextVQA, and POPE use the normalized question string prepared in the
  evaluation bundle;
- multiple-choice suites use their instruction text with the evaluator-added
  “Answer with the option letter only” suffix removed from the router input.

No Qwen chat template is applied by the Qwen3 router tokenizer. The native
Qwen2.5-VL generation prompt is still built with its normal multimodal chat
template.

### 4.2 Image input

The Image+Question predictor receives the **native projected visual rows
entering Qwen decoder layer 0**:

\[
X_I\in\mathbb{R}^{B\times V\times3584}.
\]

These rows are produced by:

1. the pinned Qwen2.5-VL processor with native image sizing;
2. the frozen Qwen2.5-VL vision encoder and visual merger;
3. insertion into the image-token placeholders of the language-model prompt.

They are not pooled, reordered, capped, cropped, or generated by another
vision model. The cache was built with no project-specific
`max_image_tokens` override and no decoder layer execution.

For the completed full10 cache:

| Visual-row property | Value |
|---|---:|
| Width | 3,584 |
| Dtype | BF16 |
| Minimum rows per image | 48 |
| Median rows | 270 |
| Mean rows | 502.07 |
| 90th / 95th percentile | 999 / 999 |
| Maximum rows | 1,798 |

Within a minibatch, native visual tensors are padded to `[B,V_max,3584]`, and
a Boolean `[B,V_max]` mask excludes padded rows from predictor attention.

### 4.3 Modalities actually implemented

The shared code supports three diagnostic modalities:

- `question`: question rows only;
- `image`: visual rows only, with a masked dummy question row;
- `image_question`: concatenated question and visual rows.

The main full training comparison used `question` and `image_question`.
Image-only was a bounded modality-isolation diagnostic, not a main full10
model.

### 4.4 What is not an input

The predictor does not see:

- the ground-truth answer;
- answer aliases;
- whether ALL-ON was correct;
- which route MCTS preferred;
- the number of valid routes;
- route compute or reward fields;
- Qwen decoder states after layer 0;
- generated tokens.

MCTS results are supervision targets only.

## 5. Predictor architecture and dimensions

The canonical architecture is `BinaryPolarBackbone` in
`binary_policy/predictor.py`.

### 5.1 Projection into the common width

Question features are projected as:

\[
[B,T,1024]\xrightarrow{\text{Linear}(1024,256)}[B,T,256].
\]

For Image+Question, visual features are independently projected:

\[
[B,V,3584]\xrightarrow{\text{Linear}(3584,256)}[B,V,256].
\]

The two projected row sets are concatenated along the sequence dimension:

\[
H_{input}=[H_Q;H_I]\in\mathbb{R}^{B\times(T+V)\times256}.
\]

There is no pooling and no learned fusion token. There is also no explicit
question/image type embedding in this predictor. The two sources are
distinguished by their separate learned projection layers and their row
contents.

### 5.2 Learned layer queries

The model contains 28 learned embeddings:

\[
E_L\in\mathbb{R}^{28\times256}.
\]

They are expanded across the batch to `[B,28,256]`. Entry `l` is the learned
query associated with Qwen decoder layer `l`.

### 5.3 Input-to-layer cross-attention

The learned layer embeddings are the queries; the projected question/image
rows are keys and values:

\[
H_{attn}=\operatorname{MHA}(Q=E_L,K=H_{input},V=H_{input}).
\]

Frozen settings:

- model width: 256;
- heads: 4;
- head width: 64;
- dropout: 0.1;
- output: `[B,28,256]`.

The implementation adds the original learned layer embedding back to the
attended representation before cross-layer encoding:

\[
H_0=H_{attn}+E_L.
\]

### 5.4 Cross-layer encoder

`H_0` passes through two standard Transformer encoder blocks operating over
the 28 layer positions:

- input/output: `[B,28,256]`;
- self-attention heads: 4;
- feed-forward width: 1,024;
- dropout: 0.1.

This stage lets the representation for one Qwen layer depend on the
representations of all other layers. Therefore the network can learn shared
cross-layer context before prediction.

However, the **output probability model is still factorized**. Cross-layer
self-attention does not turn the final 28 Bernoulli variables into an
autoregressive or energy-based joint distribution.

### 5.5 Binary route head

One shared linear head is applied to each of the 28 encoded layer positions:

\[
z_l=w^\top h_l+b,
\qquad
z\in\mathbb{R}^{B\times28}.
\]

The head has weight shape `[1,256]` and one bias. It shares parameters across
all layer positions; layer identity is carried by the learned layer embedding
and cross-layer context.

### 5.6 Parameter counts

| Component | Parameters |
|---|---:|
| Question projection | 262,400 |
| Learned layer embeddings | 7,168 |
| Input-to-layer cross-attention | 263,168 |
| Two-block cross-layer encoder | 1,579,520 |
| Binary route head | 257 |
| Additional image projection | 917,760 |
| **Question-only predictor total** | **2,112,513** |
| **Image+Question predictor total** | **3,030,273** |

These are the trainable predictor parameters. They exclude the frozen
Qwen3-Embedding model and frozen Qwen2.5-VL model.

## 6. The binary route and probability model

For a complete mask

\[
m=(m_0,\ldots,m_{27}),\qquad m_l\in\{0,1\},
\]

the head defines 28 Bernoulli probabilities:

\[
p_l=\sigma(z_l).
\]

The probability of the complete mask is:

\[
P_\theta(m\mid x)
=
\prod_{l=0}^{27}
p_l^{m_l}(1-p_l)^{1-m_l}.
\]

Equivalently, the stable log probability is:

\[
\log P_\theta(m\mid x)
=
\sum_l
\left[
m_l\log\sigma(z_l)+(1-m_l)\log\sigma(-z_l)
\right].
\]

The exact MAP mask under this factorized distribution is obtained by choosing
the more probable bit independently at each layer. The implementation uses:

```python
mask = (logits >= 0).to(torch.int64)
```

Thus a logit exactly equal to zero resolves to ON. No route cache, ground-truth
answer, MCTS search, beam reranking, or compute constraint is consulted for the
primary top-1 decision.

The code can enumerate exact top-k masks under the factorized model for cached
Hit@k diagnostics. Primary execution uses only the threshold-decoded top-1
mask.

## 7. How training labels are constructed

### 7.1 Source route labels

The regenerated label pool contains 8,000 image-question records:

| Benchmark | Total | Predictor train split | Predictor validation split |
|---|---:|---:|---:|
| GQA | 4,000 | 3,500 | 500 |
| TextVQA | 2,000 | 1,750 | 250 |
| ChartQA | 2,000 | 1,750 | 250 |
| **Total** | **8,000** | **7,000** | **1,000** |

The split is deterministic and image-group-disjoint. Its historical ALL-ON
correct/wrong strata are balanced by construction. Current correctness was
recomputed under the regenerated executor and is descriptive rather than a
split-selection variable.

MCTS evaluates unrestricted 28-bit masks. A route is positive when executing
that exact mask through the frozen MLLM produces an answer that passes the
benchmark-specific correctness threshold. The raw cache preserves both
positive and negative evaluated masks.

### 7.2 Positive-record population

The predictor objectives require at least one valid mask. Of the 8,000 source
records, 6,917 have one or more discovered valid masks and 1,083 do not.

| Split | GQA | TextVQA | ChartQA | Total positive records |
|---|---:|---:|---:|---:|
| Train | 2,957 | 1,525 | 1,561 | 6,043 |
| Validation | 429 | 221 | 224 | 874 |

Zero-positive records remain in the label cache but do not enter the positive
route objectives. This is important when interpreting validation: cached
valid-set Hit@1 is defined only for the positive subset, whereas actual
execution can and should evaluate predicted masks even when they were not
cached.

### 7.3 Maximum 50 routes per input

The raw MCTS cache is not truncated. A deterministic derived supervision view
selects at most 50 valid masks per input. The frozen policy preserves a
minimum-ON route, useful extreme anchors, and diverse masks chosen with
ON-count stratification, Hamming distance, and transition structure.

If a sample has fewer than 50 valid routes, all of them are retained. The
dataset loader refuses to silently subsample a manifest that exceeds the
frozen cap.

Across the full positive population, the derived view contains 237,802 valid
route occurrences selected from 528,047 raw valid routes.

### 7.4 Route weighting

The canonical full10 matched configuration uses the released-POLAR-style
FULL-path rule:

- if ALL-ON is valid and a cheaper valid mask also exists, ALL-ON receives raw
  weight `0.3`;
- every other selected valid mask receives raw weight `1.0`;
- weights are normalized to sum to one within each input.

The code also supports equal weighting and explicit manifest weights. The
first clean loss sanity comparison used equal weights, while the completed
matched full10 BCE and exact-set-NLL runs both used the same `0.3` rule.

No exponential compute weight, sparsity loss, route-length penalty, RL reward,
or latency objective is part of these two canonical losses.

## 8. Training objective A: POLAR-style duplicated-path BCE

For an input with selected valid set

\[
V_x=\{m_1,\ldots,m_K\},
\]

the collator conceptually creates:

```text
(x, m_1)
(x, m_2)
...
(x, m_K)
```

The frozen Qwen3 question representation is computed once for each unique
input and then indexed for the expanded route rows. Image features are
similarly expanded by index. This avoids repeatedly running the frozen encoder
for the same input.

For each duplicated path, the loss is the mean bitwise BCE:

\[
L_{BCE}(x,m)
=
-\frac{1}{28}\sum_l
\left[m_l\log p_l+(1-m_l)\log(1-p_l)\right].
\]

The per-route values are combined with normalized within-input route weights.
Consequently, an input with 50 routes does not automatically receive 50 times
the total gradient weight of an input with one route.

This objective pushes the 28 probabilities toward weighted bitwise marginals.
When valid routes disagree strongly across layers, its thresholded output can
combine common bits from different valid routes into a complete mask that was
never itself observed.

## 9. Training objective B: exact valid-set NLL

For exact set NLL, all selected masks remain grouped with their source input:

```text
{
  input: x,
  valid_masks: [m_1, ..., m_K],
  valid_mask_weights: [w_1, ..., w_K]
}
```

A minibatch pads this to:

- `valid_masks [B,K_max,28]`;
- `valid_mask [B,K_max]`;
- `route_weights [B,K_max]`.

Padded rows are assigned zero mass by setting their log contribution to
negative infinity before `logsumexp`.

The loss is:

\[
L_{set}(x)
=
-\log\sum_{m\in V_x}w_mP_\theta(m\mid x).
\]

The implementation computes complete-mask log probabilities with
`logsigmoid`, adds `log(w_m)`, and uses `torch.logsumexp`. Duplicate masks are
rejected by the dataset adapter rather than counted multiple times.

This is **one-of-valid-set supervision**. It rewards total probability mass on
the valid complete masks and does not require matching every valid mask at
once.

It is still not a structured joint mask head. Because
`P_theta(m|x)` remains a product of Bernoulli probabilities, exact-set NLL can
prefer a coherent mode but cannot represent arbitrary cross-layer dependency
patterns. A threshold-decoded hybrid mask remains possible.

## 10. What is trained and what remains frozen

### Trainable

- question projection;
- optional image projection;
- 28 learned layer embeddings;
- predictor cross-attention;
- two cross-layer Transformer blocks;
- shared binary route head.

### Frozen

- Qwen3-Embedding-0.6B;
- Qwen2.5-VL vision encoder and merger;
- all 28 Qwen2.5-VL language-model decoder layers;
- Qwen token embeddings, MRoPE machinery, final norm, and LM head.

The Qwen2.5-VL base model is not loaded in the normal predictor training loop.
Its visual rows were cached beforehand, and MCTS already supplied the route
labels. The training loop runs the frozen Qwen3 question encoder under
`torch.no_grad()` and verifies that it receives no gradients.

## 11. Canonical full10 optimization settings

The matched full10 BCE and exact-set-NLL comparisons used:

| Setting | Value |
|---|---:|
| Epochs | 10 |
| Early stopping | None |
| Physical batch size | 128 unique inputs |
| Gradient accumulation | 1 |
| Effective batch size | 128 |
| Optimizer | AdamW |
| Learning rate | `5e-4` |
| Weight decay | `0.01` |
| Scheduler | cosine |
| Warmup | 10 optimizer steps |
| Gradient clip | `1.0` |
| Predictor precision | BF16 autocast |
| Seed | `20260809` |
| Deterministic algorithms | Required |
| Checkpoint frequency | Every epoch |
| Duplicated-BCE route microbatch | 32 route rows |

For duplicated BCE, gradients from route microbatches are accumulated and
then divided by the exact number of unique inputs. Exact set NLL naturally
uses one grouped item per unique input.

Every checkpoint stores predictor, optimizer, scheduler, epoch, global step,
frozen config, and checksums.

## 12. How one predicted bit changes a Qwen layer

Let the ordered prompt contain:

- `S_t` valid text/control rows;
- `S_v` visual rows;
- Qwen hidden width 3,584.

The executor first separates the prompt into:

\[
T_l\in\mathbb{R}^{1\times S_t\times3584},
\qquad
V_l\in\mathbb{R}^{1\times S_v\times3584},
\]

while retaining each row's original sequence index and MRoPE position.

### 12.1 `VISUAL_ON` at layer `l`

1. Scatter `T_l` and `V_l` back into their original full-sequence order.
2. Run the unchanged native Qwen decoder layer on all valid rows.
3. Use the full causal mask and the original full MRoPE position IDs.
4. Split the layer output back into text/control and visual streams.

Conceptually:

\[
(T_{l+1},V_{l+1})
=
\operatorname{split}\left(
F_l(\operatorname{scatter}(T_l,V_l))
\right).
\]

Consequences:

- text queries can attend to the visual K/V rows available at this layer;
- visual rows pass through the layer's attention, residual, normalization, and
  MLP update;
- the layer's prompt K/V cache contains `S_t + S_v` rows.

### 12.2 `VISUAL_OFF` at layer `l`

1. Do not place visual rows in the layer input.
2. Run the unchanged Qwen decoder layer on text/control rows only.
3. Build the text-only causal mask from the rows' **original sequence
   indices**, preserving causal ordering after compaction.
4. Preserve the original text MRoPE position IDs.
5. Return the incoming visual states unchanged.

Conceptually:

\[
T_{l+1}=F_l^{text-only}(T_l),
\qquad
V_{l+1}=V_l.
\]

Consequences:

- text cannot attend to visual K/V at that layer;
- visual hidden states do not receive attention, residual, normalization, or
  MLP updates at that layer;
- the layer's prompt K/V cache contains only `S_t` rows;
- the carried visual rows can re-enter at a later layer whose bit is ON.

`VISUAL_OFF` is therefore a **layer-local removal**, not permanent deletion
from the model. It jointly disables visual READ by text and visual WRITE/update
at that layer. It does not reproduce the earlier v2 four-action decomposition
that controlled READ and WRITE separately.

### 12.3 Example route

Suppose a shortened example route is:

```text
layer:  0 1 2 3 4 5
mask:   1 1 0 0 1 0
```

- Layers 0 and 1 run normally with visual and text/control rows.
- Layers 2 and 3 run only text/control rows; the visual state exiting layer 1
  is carried unchanged across both layers.
- Layer 4 scatters that carried visual state back into its original token
  positions and runs the complete Qwen layer.
- Layer 5 again runs text/control only.

The actual model always uses 28 bits; this six-bit example is only explanatory.

## 13. Prompt prefill and autoregressive decoding

### 13.1 Prompt prefill

The predicted static mask controls the 28 prompt-prefill layer calls. Because
ON and OFF layers admit different rows, the executor uses a route-aware cache
with a separate K/V sequence length for each layer:

- ON-layer cache length: full prompt length;
- OFF-layer cache length: compact text/control prompt length.

This per-layer cache is necessary for a mixed route. A conventional cache that
assumes identical prompt length in every layer would be incorrect.

### 13.2 Generation

After prefill, deterministic greedy generation proceeds one text token at a
time:

1. take the last valid prompt text hidden state;
2. apply final Qwen RMSNorm and LM head;
3. choose `argmax` after the frozen repetition-penalty rule;
4. embed the new token;
5. execute that text token through all 28 Qwen layers;
6. at each layer, attend to that layer's route-specific prefill cache;
7. stop at EOS or the benchmark's maximum generated-token budget.

There are no visual query rows during autoregressive decoding. The original
route still matters because:

- an ON layer's cache includes visual prompt K/V;
- an OFF layer's cache does not.

The generated text token itself executes every decoder layer. We are reducing
visual-row computation and visual access, not skipping text-token layer
execution.

### 13.3 ALL-ON invariant

The route `111...111` must be semantically equivalent to native Qwen. The
validated internal executor uses Qwen's native maskless causal SDPA path for an
unpadded batch-one ALL-ON prompt so that equivalent explicit masks do not
introduce BF16 kernel-path drift. The repaired invariant achieved exact native
logit parity in the executor validation, and regenerated labels were produced
under the verified current contract.

## 14. Inference: from a new sample to a scored answer

The actual static-router evaluation path is:

1. **Build the native MLLM input.** Apply the Qwen2.5-VL multimodal chat
   template and processor to the image and prompt.
2. **Prepare binary streams.** Obtain full embeddings, split visual and
   text/control rows, and retain indices, masks, MRoPE positions, and rope
   deltas.
3. **Build router question features.** Tokenize the predictor question with
   Qwen3-Embedding and obtain `[1,T,1024]` frozen features.
4. **Obtain router visual features if needed.** Reuse the already-encoded
   pre-decoder visual rows `[1,V,3584]` from the same prepared MLLM input.
5. **Predict logits.** Run the Question-only or Image+Question predictor to get
   `[1,28]` logits.
6. **Decode one route.** Set bit `l` to ON iff `z_l >= 0`.
7. **Execute the route.** Run route-conditioned prompt prefill and deterministic
   greedy generation through the frozen Qwen model.
8. **Decode text.** Use the pinned Qwen processor without token cleanup.
9. **Score behavior.** Apply the benchmark-specific evaluator and correctness
   threshold.
10. **Execute live ALL-ON baseline.** Reuse it if the predicted route itself is
    ALL-ON; otherwise run a paired live ALL-ON generation under the same
    runtime.

All predicted masks are executed, including masks not present in the cached
MCTS valid set. An uncached mask is “unknown to the cache,” not automatically
invalid.

## 15. Checkpoint selection and evaluation

### 15.1 Offline route metrics

For each positive validation sample, the evaluator reports:

- cached Valid-Set Hit@1;
- exact factorized Hit@5;
- Hamming distance to the nearest selected valid mask;
- ON-count error relative to the nearest selected valid mask;
- mean predicted ON layers;
- ALL-ON and ALL-OFF fractions;
- number and entropy of unique predicted masks;
- exact-set NLL and duplicated-BCE diagnostics.

The original full10 runs selected the checkpoint by maximum validation cached
Hit@1, then smaller nearest-valid Hamming, smaller objective loss, and earlier
epoch. Best validation-loss and final checkpoints were also retained.

Later CAP24/CAP26 exact-set-NLL runs used a prospectively changed rule: execute
every epoch on all 872 internal validation records and select maximum actual
validation accuracy, then lower mean ON, lower validation NLL, and earlier
epoch. Therefore checkpoint selection must always be read from the relevant
frozen config rather than assumed to be universal.

### 15.2 Actual execution metrics

The behavioral evaluation reports:

- routed benchmark accuracy/score;
- current live ALL-ON accuracy/score;
- ALL-ON wrong to routed correct (`rescue`);
- ALL-ON correct to routed wrong (`harm` or regression count in reports);
- unchanged correct and unchanged wrong;
- average and distribution of VISUAL_ON layers;
- distinct predicted masks;
- cached-set membership for diagnostics.

Mean ON layers is a local visual-computation proxy. It is not wall-clock
latency, because the current executor performs dynamic split/scatter operations
and has not been optimized as a production sparse kernel.

### 15.3 External suite used by the completed evaluations

The frozen no-DocVQA external suite contained 22,307 records:

| Benchmark | Records |
|---|---:|
| ChartQA | 2,500 |
| TextVQA | 5,000 |
| MMStar validation | 1,500 |
| MMMU validation | 847 |
| MMMU-Pro standard test | 1,730 |
| MMMU-Pro vision test | 1,730 |
| POPE adversarial | 3,000 |
| POPE popular | 3,000 |
| POPE random | 3,000 |

The predictor never uses external answers during route selection. External
outcomes were opened only after checkpoint selection.

## 16. Important implementation boundaries

### 16.1 Static predictor versus online router utilities

The shared evaluation bundle contains functions for live layer-wise routers
that consume intermediate Qwen states. Those are separate experiments. The
Question-only and Image+Question POLAR-style checkpoints documented here use
`static_route` execution: all 28 decisions are known before layer 0.

### 16.2 Factorized output versus cross-layer feature processing

The two Transformer blocks allow each layer's hidden feature to depend on
other layers. Nevertheless, the loss assigns complete-mask probabilities using
a product of 28 Bernoulli terms, and top-1 is decoded independently. The model
does not have:

- an autoregressive mask decoder;
- a CRF;
- an energy model;
- an explicit route codebook;
- POLAR segment constraints;
- post-decoding projection onto the valid set.

### 16.3 OFF is not whole-layer skipping

At an OFF layer, text/control tokens still run self-attention, residual paths,
normalization, and MLP. Only visual rows are excluded and bypassed. Therefore
“18 ON layers” means visual rows participate in 18 decoder layers; it does not
mean the language model executes only 18 total layers.

### 16.4 Image features do not add privileged evidence

The Image+Question predictor uses the same native visual rows already required
by Qwen. It does not use crops, OCR annotations, object detections, answer
regions, or an external image encoder.

### 16.5 Cache membership is not behavioral correctness

MCTS enumerates only a tiny part of the `2^28` route space. A predicted route
outside the selected or raw cached valid set must be executed before it can be
called correct or incorrect.

## 17. Invariant and variant components across our experiments

The following remained invariant across the main BCE/NLL and later supervision
filtering studies:

- Qwen2.5-VL base model and snapshot;
- 28-bit ON/OFF semantics;
- frozen Qwen3 question encoder;
- predictor width, heads, blocks, and direct binary head;
- logit-zero top-1 decoding;
- benchmark scorers;
- image-group split discipline;
- live route execution.

The following have varied under explicit plans:

- Question-only versus Image+Question input;
- duplicated BCE versus exact valid-set NLL;
- full max-50 route sets versus Pareto-filtered route sets;
- maximum-ON supervision caps such as 18, 20, 22, 24, and 26;
- cached-Hit@1 versus actual-executed-accuracy checkpoint selection in later
  cap experiments.

These variants should not be conflated with an architecture change unless the
predictor config itself changes.

## 18. Minimal pseudocode

### 18.1 Predictor

```python
# Frozen question encoder
question_h = qwen3_embedding(input_ids, attention_mask)  # [B,T,1024]

# Trainable projections
question_h = question_projection(question_h)             # [B,T,256]
memory = question_h

if image_question:
    image_h = image_projection(visual_rows)               # [B,V,256]
    memory = concat([question_h, image_h], dim=1)          # [B,T+V,256]

# Trainable POLAR-style layer representation
layer_q = learned_layer_embedding(arange(28))             # [28,256]
layer_q = expand_batch(layer_q)                           # [B,28,256]
attended = cross_attention(layer_q, memory, memory)       # [B,28,256]
layer_h = cross_layer_transformer(attended + layer_q)     # [B,28,256]
logits = shared_linear(layer_h).squeeze(-1)               # [B,28]

# Static MAP route under the factorized head
mask = (logits >= 0)                                      # [B,28]
```

### 18.2 Route-conditioned Qwen prefill

```python
text_states, visual_states, metadata = split_native_prompt(inputs)

for layer_idx, qwen_layer in enumerate(qwen_layers):
    if mask[layer_idx]:
        full = scatter_original_order(text_states, visual_states, metadata)
        full = qwen_layer(full, full_causal_mask, full_mrope_positions)
        text_states, visual_states = split_streams(full, metadata)
    else:
        text_states = qwen_layer(
            text_states,
            text_only_causal_mask_using_original_indices,
            original_text_mrope_positions,
        )
        visual_states = visual_states  # exact bypass
```

### 18.3 Exact valid-set NLL

```python
log_p_on = F.logsigmoid(logits)[:, None, :]      # [B,1,28]
log_p_off = F.logsigmoid(-logits)[:, None, :]    # [B,1,28]

route_log_prob = (
    valid_masks * log_p_on
    + (1.0 - valid_masks) * log_p_off
).sum(dim=-1)                                    # [B,K]

weighted = route_log_prob + log(normalized_route_weights)
weighted = weighted.masked_fill(~valid_route_slots, -inf)
loss = -torch.logsumexp(weighted, dim=1).mean()
```

## 19. Code and artifact map

| Purpose | Path |
|---|---|
| Predictor architecture | `binary_policy/predictor.py` |
| Grouped and duplicated collators | `binary_policy/dataset.py` |
| Visual feature batching | `binary_policy/multimodal.py` |
| Exact-set NLL and duplicated BCE | `binary_policy/losses.py` |
| Threshold and factorized top-k decoding | `binary_policy/decode.py` |
| Offline mask metrics | `binary_policy/evaluation.py` |
| Project-owned binary executor | `binary_policy/executor/` |
| Full10 training loop | `experiments/train_binary_polar_full10.py` |
| Full10 conditioning/decode evaluation | `experiments/evaluate_binary_polar_full10_conditioning.py` |
| Internal actual mask execution | `experiments/evaluate_binary_polar_full10_execution.py` |
| External 22,307-record execution | `experiments/evaluate_binary_polar_external.py` |
| Portable/reference binary executor | `eval/reference/shared_prefix_eval_20260812/code/dvr_qwen/` |
| Full visual feature extraction | `experiments/extract_binary_polar_full_visual_features.py` |
| Exact-set full10 config | `configs/binary_polar_full10_polar_matched_v1.yaml` |
| Duplicated-BCE full10 config | `configs/binary_polar_full10_polar_bce_v1.yaml` |
| Original POLAR release snapshot | `reference/polar/PoLar/` |
| Binary executor repair evidence | `reports/binary_polar_bp1_executor_repair.md` |
| Full10 exact-set results | `reports/binary_polar_full10_polar_matched_results.md` |
| Exact-set external evaluation | `reports/binary_polar_full10_external_eval.md` |
| Duplicated-BCE external evaluation | `reports/binary_polar_full10_bce_external_eval.md` |

## 20. One-sentence operational description

For each new image-question pair, a frozen Qwen3 question representation and,
optionally, the frozen Qwen2.5-VL pre-decoder visual rows are mapped by a
roughly 2.1M/3.0M-parameter POLAR-style network into 28 logits; logits at least
zero enable normal visual participation at the corresponding Qwen layer,
while negative logits remove visual rows from that layer, update text/control
only, carry vision unchanged, and preserve the resulting route-specific cache
for deterministic greedy answer generation.
