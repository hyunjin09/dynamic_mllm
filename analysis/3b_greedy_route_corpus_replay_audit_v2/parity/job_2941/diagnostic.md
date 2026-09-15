Focused parity diagnostic for textvqa:textvqa_19417, Slurm job 2941.

Known failure reproduced: True. Instrumented prefixes match baseline: True.

First observed numerical divergence:
```json
{
  "step": 0,
  "stage": "prefill",
  "layer": 0,
  "field": "sdpa_out",
  "equal": false,
  "comparable": true,
  "max_abs": 0.0078125,
  "mean_abs": 2.0462378249566554e-05,
  "different_elements": 130322
}
```

Isolated SDPA convention check:
```json
{
  "same_qkv": true,
  "layer": 0,
  "step": 0,
  "native_convention_on_custom_qkv_vs_native": {
    "equal": true,
    "comparable": true,
    "max_abs": 0.0,
    "mean_abs": 0.0,
    "different_elements": 0
  },
  "native_convention_on_custom_qkv_vs_custom": {
    "equal": false,
    "comparable": true,
    "max_abs": 0.0078125,
    "mean_abs": 2.0462378249566554e-05,
    "different_elements": 130322
  },
  "scope": "isolated SDPA call, not an executor repair or full Gate-A pass"
}
```

Layer checksums/norms, Q/K/V/cache comparisons, position/RoPE/mask comparisons,
and both-step top10 logits are in the job directory. No raw hidden tensor
corpus was saved. Cache-position arguments absent from a layer call are
recorded as null; physical KV lengths and actual rotary positions are recorded.
Diagnosis requires review; no executor repair or full replay was performed.
