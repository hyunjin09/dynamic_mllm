# MCTS v2 Model Execution and Label Generation

## 1. Overview

MCTS v2 generates supervision labels by repeatedly running a frozen
Qwen2.5-VL model under different binary layer masks.

For a model with `L = 28` language layers, each route is

```text
m = (m_0, ..., m_27), where m_l is either 0 or 1.
```

- `m_l = 1`: run layer `l` with the original text and visual token sequence.
- `m_l = 0`: run layer `l` only on text/control tokens and carry the visual
  hidden states forward unchanged.

This is not full-layer skipping. Every language layer still processes the text
stream. The binary action controls whether visual tokens participate in that
layer.

Each candidate mask is evaluated through actual greedy generation. The
generated answer is scored with the benchmark-specific metric and converted to
a binary correctness reward for MCTS.

## 2. Relevant Code

```text
run_visual_mask_mcts_v2.py
    Main label-generation runner

visual_mask_mcts_v2.py
    Graph MCTS and binary mask search

runtime.py
    Model loading, input construction, generation, and scoring

modeling_dvr_qwen2_5_vl.py
    Wrapper around the original Hugging Face Qwen model

input_builder.py
    Reconstructs the original Qwen multimodal embeddings and position IDs

split_scatter.py
    Splits and recombines text and visual streams

binary_layer.py
    Implements visual-on and visual-off layer execution

binary_generate.py
    Route-conditioned prefill, KV caching, and greedy decoding

eval_metrics.py
    Benchmark-specific answer scoring
```

The MCTS package contains its own local `dvr_qwen` implementation. Because
`MCTS_v2/src` is inserted first into `sys.path`, this packaged implementation
is used instead of the repository-level `dvr_qwen` package.

## 3. Model Loading

The frozen backbone used for the 7B experiment is:

```text
Qwen/Qwen2.5-VL-7B-Instruct
snapshot: cc594898137f460bfe9f0759e9844b3ce807cfb5
```

The Hugging Face model is loaded as:

```python
Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
```

and wrapped by:

```python
DVRQwen2_5_VLForConditionalGeneration
```

The wrapper does not replace or train Qwen parameters. It exposes an
alternative route-conditioned execution path around the original pretrained
model.

Typical runtime settings are:

- frozen inference mode;
- BF16 weights;
- `device_map="auto"`;
- SDPA attention;
- local Hugging Face snapshot;
- deterministic greedy decoding;
- no sampling.

The processor is loaded from the same model snapshot.

## 4. Mandatory All-On Replay Gate

Before MCTS starts, the runner validates a saved replay gate. The gate requires
that the binary all-on route reproduces:

1. the current Hugging Face all-on generated token IDs;
2. the reference generated token IDs;
3. the reference prediction score;
4. available previously saved generated IDs.

The gate also verifies that the model snapshot path exactly matches the
snapshot used during replay validation. The required contract is:

```text
G(x, all_on_mask) == G_HF(x)
```

where `G` denotes deterministic generation. MCTS is aborted if this equivalence
check fails.

## 5. Multimodal Input Construction

For each sample, the processor constructs the normal Qwen chat prompt:

```text
<image>
user instruction
assistant generation marker
```

If the sample specifies `max_image_tokens`, it is converted to a pixel limit:

```text
max_pixels = max_image_tokens * 28 * 28
```

The model then constructs the same initial multimodal embedding sequence used
by ordinary Qwen2.5-VL:

1. Text tokens are embedded with Qwen's token embedding layer.
2. The image is processed by Qwen's vision encoder.
3. Image features replace the image-placeholder embeddings.
4. Qwen's multimodal 3D position IDs and RoPE deltas are computed.
5. The original full attention mask and token positions are preserved.

The resulting sequence is separated into:

```text
text_states
visual_states
text_position_ids
visual_position_ids
text_original_indices
visual_original_indices
```

`text_states` includes all non-visual control and prompt tokens. Visual tokens
are identified using multimodal token-type IDs. The split/scatter
implementation is designed to reconstruct the original full sequence exactly.

## 6. Input Reuse Across Candidate Masks

Image encoding and initial multimodal input construction are performed once per
sample:

```python
prepared = prepare_binary_dvrc_inputs(model, processor_inputs)
```

These immutable initial states are reused across all MCTS route evaluations for
that sample. This avoids rerunning the vision encoder for every candidate mask.

Each mask still performs a separate language-model prefill and greedy
generation because its hidden states and KV caches are route-dependent.

## 7. Binary Visual-On Operation

When `m_l = 1`, layer `l` follows the original Qwen computation. The current
text and visual states are scattered into their original token positions:

```python
full_states = scatter_to_full(text_states, visual_states, metadata)
```

The original Qwen decoder layer is then applied:

```python
outputs = layer(
    hidden_states=full_states,
    attention_mask=full_causal_mask,
    position_embeddings=full_position_embeddings,
    past_key_values=cache,
    use_cache=True,
)
```

The result is separated back into text and visual streams:

```python
next_text, next_visual = split_from_full(outputs[0], metadata)
```

Therefore, visual-on means:

- text tokens can use the visual-token K/V states allowed by the original
  causal sequence;
- visual tokens participate in self-attention;
- visual tokens pass through the layer FFN;
- text and visual hidden states are updated;
- the layer KV cache contains the route's full multimodal prefill states.

This path is intended to match the original Qwen decoder-layer behavior.

## 8. Binary Visual-Off Operation

When `m_l = 0`, visual tokens are excluded from that layer's active sequence.
The layer runs only on text/control states:

```python
outputs = layer(
    hidden_states=text_states,
    attention_mask=text_only_causal_mask,
    position_embeddings=text_position_embeddings,
    past_key_values=cache,
    use_cache=True,
)
```

The outputs are:

```python
next_text = outputs[0]
next_visual = visual_states
```

Thus, at a visual-off layer:

- the language layer itself is not skipped;
- text tokens still pass through MHSA and FFN;
- text tokens cannot attend to visual-token K/V states;
- visual tokens do not pass through MHSA;
- visual tokens do not pass through the FFN;
- visual hidden states remain unchanged;
- the prefill KV cache contains text/control states but no visual states.

The visual tokens are not permanently deleted. Their hidden states are carried
to subsequent layers and can be reintroduced at a later visual-on layer.

For example:

```text
Layer 10: visual-on
    text and visual states are updated

Layer 11: visual-off
    text is updated
    visual state remains equal to the Layer-10 visual output

Layer 12: visual-off
    text is updated again
    visual state remains unchanged

Layer 13: visual-on
    current text and preserved visual states are recombined
    the full multimodal layer is executed
```

The operation is best described as **layer-wise visual contextualization
suspension**, not visual-token pruning or full decoder-layer skipping.

## 9. Route-Conditioned Prefill

For a candidate mask, the prefill loop processes every language layer:

```python
for layer_idx, layer in enumerate(text_model.layers):
    if visual_on_mask[layer_idx]:
        text_states, visual_states = forward_visual_on_layer(...)
    else:
        text_states, visual_states = forward_text_only_layer(...)
```

Each layer produces a route-conditioned KV cache:

- visual-on layer: multimodal prefill cache;
- visual-off layer: text-only prefill cache.

After the final layer, the text hidden states pass through Qwen's final
normalization and LM head to produce the first generated-token logits.

## 10. Route-Conditioned Decoding

Generation uses deterministic argmax decoding:

```text
y_t = argmax_v logits_t[v]
```

Each generated token passes through every Qwen language layer. The binary mask
does not skip text computation during decoding. However, every layer reads its
own route-conditioned prefill cache:

- at a visual-on layer, the cache contains visual-token K/V states;
- at a visual-off layer, the cache does not contain visual-token K/V states.

Consequently, the mask continues to affect every generated token even though
only generated text tokens are processed during autoregressive decoding.

Generation stops when an EOS token is generated or the sample-specific
`max_new_tokens` limit is reached.

The result records:

- generated token IDs;
- decoded prediction;
- text and visual token counts;
- cache lengths;
- the binary route;
- the number of visual-on layers.

## 11. Benchmark Scoring and Binary Reward

Each generated prediction is scored with its benchmark-specific metric:

```python
score = score_prediction(
    metric_name,
    prediction,
    ground_truth_answer,
    all_answer_norms,
)
```

Correctness is determined by the sample-specific threshold:

```text
correct(x, m) = 1[score(G(x, m), ground_truth) >= correctness_threshold]
```

The MCTS reward is strictly binary:

```text
R(x, m) = correct(x, m), where R is either 0 or 1.
```

The raw task score is saved but is not directly used by UCB.

## 12. MCTS Search Space

The root mask is all visual-on:

```text
m_root = (1, ..., 1)
```

The all-off mask is also evaluated as an anchor:

```text
m_all_off = (0, ..., 0)
```

A graph state consists of:

1. a partially assigned binary mask;
2. a Boolean vector indicating which layer decisions are fixed.

At every expansion, MCTS jointly chooses:

```text
action = (layer_index, binary_visual_action)
```

where the layer can be any currently undecided layer and the action is either
visual-off or visual-on. There is no fixed early-to-late layer permutation.
MCTS decides both the layer location and the binary action.

Undecided layers are effectively visual-on in the partial state. During
rollout, all remaining undecided layers are completed randomly using the
configured visual-off probability.

## 13. UCB Selection

The child-selection score is:

```text
UCB(m) = mean_reward(m)
       + exploration_constant * sqrt(log(total_visits) / visits(m))
       - length_penalty * num_visual_on_layers(m) / num_layers
```

Typical settings are:

```text
num_simulations = 200
exploration_constant = 1.8
length_penalty = 3.0
random_selection_probability = 0.1
rollout_visual_off_probability = 0.5
```

With probability `0.1`, selection chooses a random child instead of the
maximum-UCB child. Repeated masks are cached within each sample, so an already
evaluated mask is not regenerated.

## 14. Successful Routes and Final Labels

MCTS does not stop after finding its first successful mask. After all
simulations, every evaluated mask with reward `1` is retained:

```text
successful_masks(x) = {m : R(x, m) = 1}
```

Successful masks are sorted by:

1. number of visual-on layers;
2. binary mask key.

The first route is stored as the minimum-budget successful route:

```text
best_mask(x) = argmin_{m in successful_masks(x)} num_visual_on_layers(m)
```

The output stores more than one best label:

- all-on route result;
- all-off route result;
- every unique evaluated mask;
- generated answer for every evaluated mask;
- raw benchmark score;
- thresholded correctness;
- binary reward;
- all successful route IDs;
- minimum-budget successful route;
- all MCTS simulations;
- expanded layer and action per simulation;
- graph nodes and visit statistics;
- transposition-table reuse information.

This supports later training with:

- one shortest route;
- multiple valid labels;
- pairwise preferences;
- route ranking;
- budget-aware objectives;
- success/failure classification.

## 15. Interpretation

MCTS v2 does not train Qwen and does not use a learned router during label
generation. Its complete procedure is:

```text
frozen Qwen backbone
+ candidate binary visual mask
+ actual route-conditioned generation
+ benchmark scoring
+ thresholded binary reward
+ MCTS exploration
```

A mask is considered valid only because the frozen model generated a correct
answer under that exact route. The labels are therefore empirical route
outcomes rather than attention-based heuristics, estimated rewards, or router
predictions.
