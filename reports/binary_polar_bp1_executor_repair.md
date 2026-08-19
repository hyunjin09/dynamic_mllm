# Binary POLAR BP-1 Executor-Equivalence Repair

Date: 2026-08-09

Decision: **PARTIAL REPAIR; BP-1 STILL FAILS; TRAINING BLOCKED**.

## Frozen scope

The repair used the existing non-reproducing ChartQA mixed-mask fixture
`chartqa:chartqa_train_14929682007417_cae26fbd9a`, the pinned Qwen2.5-VL-7B
revision, Transformers 5.3.0, BF16, and SDPA. The 16 BP-1 fixtures and native
logit tolerance of `0.005` were unchanged. No predictor training, answer
scoring, or held-out policy outcome was run.

## Incremental equivalence trace

The trace compared the native model, the supplied
`reference/binary_action_qwen` execution path, and the project-owned binary
executor.

### Inputs and token layout

Current and reference execution were exactly equal for:

- literal prompt and input token IDs;
- full input embeddings;
- the 39 text/control rows and 630 visual rows;
- text and visual original-sequence indices;
- full, text, and visual MRoPE position IDs;
- attention mask and tensor shapes.

Every corresponding maximum absolute difference was `0.0`.

### Layer execution

For both ALL-ON and the cached mixed mask, current and reference pre-layer and
post-layer hidden states were exactly equal at all 28 layers. The action at
every mixed-route layer also matched. Final normalized hidden states and logits
were exactly equal. Therefore, the cached mixed-route mismatch was not caused
by a divergence between the current port and the supplied reference.

### Earliest ALL-ON/native divergence

The native and binary inputs were equal before layer 0. The first difference
occurred inside layer-0 self-attention:

- native Qwen's SDPA dispatcher used its optimized causal path with
  `attention_mask=None`;
- the reference and current binary paths materialized an equivalent dense
  additive `[1,1,669,669]` causal mask;
- input-layer RMSNorm remained exact;
- self-attention differed by `0.03125` maximum absolute;
- post-attention RMSNorm differed by `0.0625`;
- MLP output differed by `0.03125`;
- layer output differed by `0.0625`.

This numerical kernel-path difference compounded through the dense stack,
explaining why greedy argmax tokens could match while the frozen logit-parity
gate failed.

## Root-cause repair

The binary prefill now detects a batch-one, unpadded ALL-ON route and dispatches
each full layer using the native maskless causal SDPA path and native
cache-position vector. Mixed routes retain the supplied reference's explicit
mask semantics. No threshold, fixture, OFF action, or mixed-route definition
changed.

A regression test verifies that the ALL-ON native path receives no explicit
attention mask and receives the full cache-position vector. Sixteen direct
assertion tests pass after the change.

## Unchanged 16-fixture BP-1 rerun

| Check | Result |
|---|---:|
| Split/scatter identity | 16/16 exact |
| ALL-ON native logits | 16/16 exact; maximum error `0.0` |
| ALL-ON native greedy IDs | 16/16 exact |
| OFF compact-text oracle | 16/16 exact |
| Visual bypass | 16/16 exact |
| Deterministic ALL-ON and arbitrary masks | 16/16 exact |
| ALL-ON cached generated IDs | 16/16 exact |
| ALL-OFF cached generated IDs | 14/16 |
| Cached best-mask IDs | 8/12 fixtures with a best mask |
| Prefill cache geometry | exact for every evaluated route |
| Complete fixture rows passing BP-1 | 11/16 |

The remaining failing fixtures are:

- `chartqa:chartqa_train_14929682007417_cae26fbd9a`: best mask;
- `docvqa:docvqa_train_ba41adcbcc64c5f1`: best mask;
- `docvqa:docvqa_train_07032c1101c6faca`: ALL-OFF and best mask;
- `docvqa:docvqa_train_7c4975fd69ec4ec5`: best mask;
- `textvqa:textvqa_train_260`: ALL-OFF.

## Remaining diagnosis

The non-FULL cached-output mismatch is directly confirmed, but its cause is
**unknown**. On the chosen mixed-mask fixture, current and reference final
prompt logits are exactly equal, yet their first fresh greedy token is `50170`
while the cached record stores `33548`. This rules out current-versus-reference
input, index, action, layer, residual, MLP, hidden-state, logit, and cache-length
differences for that fixture.

Possible label-generation hardware/kernel or unrecorded provenance differences
remain interpretations, not established causes. Altering mixed-route masks or
accepting cached labels despite the mismatch would change or bypass the frozen
gate and is not authorized.

## Gate decision

ALL-ON parity is repaired, but exact cached non-FULL reproduction still fails.
Under the unchanged plan, any cached-label mismatch blocks BP-2 and predictor
training. This result provides no evidence against the binary head, exact
valid-set NLL, or in favor of canonical POLAR segmentation.

## Evidence

- Equivalence trace:
  `outputs/binary_polar/preflight/executor_equivalence_trace_v1.json`
- Unchanged rerun:
  `outputs/binary_polar/preflight/executor_preflight_v2.json`
- Frozen fixtures:
  `outputs/binary_polar/preflight/executor_fixtures_v1.json`

