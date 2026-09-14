# Failed sanity attempt 01

- Contract: `5703b57d2ba24e10409a17088c88cb7c90caae1ec994e0c514bfb861b38ab4de`
- Base answer-start validation: 72/72 exact layer-27 top-1 matches.
- Shared-prefix validation: 9/13 exact matches; one mismatch occurred in
  each of GQA and ChartQA and two occurred in TextVQA.
- Direct observation: all failures occur only after reconstructing the prompt
  by appending answer-prefix tokens and recomputing a full multimodal prefill.
- Supported diagnosis: the appended full-prefill path is not an exact replay of
  the native cached greedy generation path and is therefore not a valid
  generation-position parity check.
- Mechanism: Qwen2.5-VL uses cached multimodal RoPE deltas during generation;
  the degree to which position handling versus cached-attention numerics causes
  each changed top-1 is not separately established and is not needed for the
  corrective action.
- Correction: replay the exact native `use_cache=True` greedy path through the
  stored common prefix, require that replayed prefix to match, and capture the
  layer state on the cached decoding call that predicts the first divergent
  token.

The full population run was not launched under this failed contract.
